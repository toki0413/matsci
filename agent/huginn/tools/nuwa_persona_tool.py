"""Nuwa persona 工具 —— 给一个名字, 用 LLM 研究其思维框架并生成 perspective persona.

流程: LLM 研究 persona_name → 抠出 {name, description, when_to_use, system_prompt}
     → 转 Nuwa 格式 markdown → 调 PersonaManager.create_persona 落盘.

LLM 失败时直接报错, 绝不瞎编 persona 内容.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool
from huginn.types import ToolContext, ToolResult

logger = logging.getLogger(__name__)


# 让 LLM 研究 persona 并吐 JSON 的 system prompt
_RESEARCH_SYSTEM_PROMPT = """你是 persona 研究员。你的任务是研究给定的人物或角色, 提炼出可用于驱动 AI agent 的思维框架、表达风格和决策原则。

研究对象可能是:
- 真实历史人物 (如 feynman, marie-curie, linus-torvalds)
- 虚构或泛化角色 (如 "严厉的代码审查者", "苏格拉底式导师")
- 领域专家身份 (如 "电池领域专家")

研究要点:
1. name: persona 的简短标识 (英文, 用下划线或连字符, 如 feynman, marie_curie, linus_torvalds)
2. description: 一句话描述这个 persona 的核心定位
3. when_to_use: 3-5 个适用场景 (中文短句)
4. system_prompt: 完整的 system prompt, 让 AI 扮演这个 persona。要体现:
   - 这个人物的思维方式和决策原则
   - 典型的表达风格和语气
   - 关注什么、警惕什么、拒绝什么
   - 至少 200 字, 越具体越好, 不要空泛的形容词

如果研究对象是真实人物, system_prompt 要忠实于其公开形象和已知观点, 不要捏造不存在的轶事或观点。如果人物不存在或你不确定, 在 description 里说明并给出最合理的泛化解读, 不要硬编。

输出必须是严格的 JSON, 不要加 markdown 代码块标记, 不要加任何解释文字。JSON 格式:
{
  "name": "persona 标识",
  "description": "一句话定位",
  "when_to_use": ["场景1", "场景2", "场景3"],
  "system_prompt": "完整的 system prompt 文本"
}"""


class NuwaPersonaToolInput(BaseModel):
    persona_name: str = Field(
        ...,
        description=(
            "要创建的 persona 名字, 如 'feynman' / 'linus-torvalds' / 'marie-curie'。"
            "可以是真实人物、虚构角色或泛化身份。工具会用 LLM 研究其思维框架并生成 persona。"
        ),
    )
    overwrite: bool = Field(
        default=False,
        description="如果同名 persona 已存在, 是否覆盖。默认 False。",
    )


class NuwaPersonaTool(HuginnTool):
    """给一个名字, 用 LLM 研究其思维框架, 生成 Nuwa 格式 persona 并落盘."""

    name = "nuwa_persona_tool"
    category = "meta"
    description = (
        "Nuwa persona 生成器: 输入一个人物或角色名字 (如 feynman, linus-torvalds, "
        "marie-curie), 用 LLM 研究其思维框架、表达风格、决策原则, "
        "生成完整的 persona (含 system_prompt) 并写入 .huginn/personas/ 目录。"
        "LLM 失败时返回错误, 不瞎编 persona 内容。"
    )
    input_schema = NuwaPersonaToolInput
    # 要写 persona 文件, 不是只读
    read_only = False

    async def call(
        self, args: NuwaPersonaToolInput, context: ToolContext
    ) -> ToolResult:
        persona_name = args.persona_name.strip()
        if not persona_name:
            return ToolResult(
                data=None, success=False, error="persona_name 不能为空"
            )

        # 1. 拿 LLM 客户端
        try:
            model = self._get_model(context)
        except Exception as exc:
            return ToolResult(
                data=None,
                success=False,
                error=f"初始化 LLM 客户端失败: {exc}",
            )

        # 2. 调 LLM 研究 persona
        try:
            raw = await self._research_persona(persona_name, model)
        except Exception as exc:
            logger.warning("nuwa_persona_tool 研究 %s 失败: %s", persona_name, exc)
            return ToolResult(
                data=None,
                success=False,
                error=f"LLM 研究 persona '{persona_name}' 失败: {exc}",
            )

        # 3. 解析 JSON
        spec = self._parse_research_json(raw)
        if spec is None:
            return ToolResult(
                data=None,
                success=False,
                error=(
                    f"无法从 LLM 回复里解析出 persona JSON。"
                    f"原始回复前 500 字: {raw[:500]}"
                ),
            )

        # 4. 字段校验 + 兜底
        name = str(spec.get("name", "")).strip() or persona_name
        description = str(spec.get("description", "")).strip()
        when_to_use = spec.get("when_to_use", []) or []
        if isinstance(when_to_use, str):
            when_to_use = [when_to_use]
        when_to_use = [str(item).strip() for item in when_to_use if str(item).strip()]
        system_prompt = str(spec.get("system_prompt", "")).strip()

        if not system_prompt:
            return ToolResult(
                data=None,
                success=False,
                error="LLM 返回的 system_prompt 为空, 拒绝创建空 persona",
            )

        # 5. 落盘: 已存在且不覆盖就报错, 避免静默覆盖
        try:
            manager = self._get_manager(context)
        except Exception as exc:
            return ToolResult(
                data=None,
                success=False,
                error=f"初始化 PersonaManager 失败: {exc}",
            )

        # 名字规范化: persona 文件名用 name, 但 create_persona 内部会查重
        existing = manager.list()
        if name in existing and not args.overwrite:
            # 已存在的用户 persona 不覆盖; 内置 persona create_persona 会自己拒绝
            return ToolResult(
                data=None,
                success=False,
                error=(
                    f"Persona '{name}' 已存在。传 overwrite=True 覆盖, "
                    f"或换个名字。"
                ),
            )

        # overwrite 时先把旧文件删掉 (create_persona 不会覆盖已有文件内容,
        # 它直接 write_text 会覆盖文件, 但内存里的 persona 来自 _load,
        # 这里走 delete_persona 清理一下再 create 保证状态干净)
        if name in existing and args.overwrite:
            try:
                manager.delete_persona(name)
            except Exception:
                # 内置 persona 删不掉, 这种情况下 overwrite 无意义, 直接让 create 报错
                pass

        try:
            persona = manager.create_persona(
                name=name,
                description=description,
                system_prompt=system_prompt,
                when_to_use=when_to_use,
            )
        except Exception as exc:
            return ToolResult(
                data=None,
                success=False,
                error=f"落盘 persona 失败: {exc}",
            )

        preview = system_prompt[:200]
        if len(system_prompt) > 200:
            preview += "..."

        return ToolResult(
            data={
                "persona_name": persona.name,
                "description": description,
                "when_to_use": when_to_use,
                "system_prompt_preview": preview,
                "file_path": persona.source_path,
            },
            success=True,
        )

    # ------------------------------------------------------------------ helpers

    def _get_model(self, context: ToolContext) -> Any:
        """拿一个 LangChain chat model, 优先用 context.config."""
        from huginn.llm import get_model

        config = getattr(context, "config", None)
        return get_model(config=config, temperature=0.7, max_tokens=4000)

    def _get_manager(self, context: ToolContext) -> Any:
        """构造 PersonaManager, workspace 优先用 context.workspace."""
        from huginn.personas import PersonaManager

        workspace = getattr(context, "workspace", None) or None
        if workspace:
            return PersonaManager(workspace=workspace)
        return PersonaManager()

    async def _research_persona(self, persona_name: str, model: Any) -> str:
        """调一次 LLM 研究 persona, 返回原始回复文本."""
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content=_RESEARCH_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"请研究 persona: {persona_name}\n\n"
                    f"按格式输出 JSON, 包含 name / description / when_to_use / system_prompt。"
                )
            ),
        ]

        # 优先 ainvoke, 没有就退到同步 invoke + to_thread
        if hasattr(model, "ainvoke"):
            response = await model.ainvoke(messages)
        else:
            response = await asyncio.to_thread(model.invoke, messages)

        content = response.content if hasattr(response, "content") else str(response)
        # 个别 provider 返回 list[ContentBlock], 拼成纯文本
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        return content

    def _parse_research_json(self, content: str) -> dict[str, Any] | None:
        """从 LLM 回复里抠 JSON. 容忍前后多余文字和 ```json 代码块."""
        text = content.strip()
        # 先剥 markdown 代码块
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
            text = re.sub(r"\n?```\s*$", "", text)

        # 直接解析
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # 兜底: 找第一个 { 到最后一个 } 之间的内容
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    data = json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    return None
            else:
                return None

        if not isinstance(data, dict):
            return None
        return data

    def estimate_cost(self, args: NuwaPersonaToolInput) -> dict[str, float] | None:
        # 1 次 LLM 调用做研究, 估算为 0.01 小时
        return {"cpu_hours": 0.0, "gpu_hours": 0.0, "walltime_hours": 0.01}
