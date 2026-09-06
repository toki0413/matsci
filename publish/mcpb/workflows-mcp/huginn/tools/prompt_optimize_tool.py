"""提示词推荐/一键优化工具 (元工具, 面向用户).

背景: 用户希望把"我的目标"写成更结构化的提示词, 或把一段已写好的
prompt (用户 prompt / 系统提示词) 一键改写得更好。本项目原本只有:
  - PromptGuidanceHook (规则型关键词推荐工具, 不调 LLM)
  - PromptPatchStore / generate_patch (block 级替换/前置/后置, 供 harness 用)
面向用户的意图改写此前是不存在的。本工具补齐这一环, 且复用两者语义:
  - recommend: 输入用户目标/意图, 产出"提示词改进建议"(结构/要素/@context 引用/
    工具使用建议), 基于 guidance_hook 的关键词逻辑输出规则建议, 不强制调 LLM。
  - optimize : 输入一段提示词 + 意图/约束, 首选 LLM 重写/补全为更佳版本,
    支持 replace(prepend/append) 三种 mode (对齐 prompt_patch 的 block 语义),
    LLM 失败时优雅降级为规则模板结果, 绝不崩溃。

防护 (只读, 零副作用):
  - 这是纯文本改写, 不执行任何代码/命令。
  - 输出仅供用户采用, 不自动写回系统提示词 / patch store / 内存。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.core_types import ToolContext, ToolResult
from huginn.tools.base import HuginnTool

logger = logging.getLogger(__name__)

# 允许的 optimize mode, 语义对齐 prompt_patch 的 block 语义:
#   replace : 用新文本整体替换原提示词 (block 级 replace)
#   prepend : 改进段落拼到原提示词之前 (block 级 prepend)
#   append  : 改进段落拼到原提示词之后 (block 级 append)
OptimizeMode = Literal["replace", "prepend", "append"]


# ── recommend 通用建议素材 (规则模板, 中文注释说明每块的用途) ──────────

# 结构建议: 优秀提示词通常具备的骨架
_RECOMMEND_STRUCTURE: tuple[str, ...] = (
    "先给角色/场景设定, 再给目标, 然后是约束, 最后指定输出格式",
    "把大目标拆成可执行的小步骤, 每一步说明输入与期望输出",
    "能用 @context / @file 引用就不把对象内容全文贴进 prompt",
    "在结尾加一句自检: 输出是否满足单位、精度与安全性要求",
)

# 要素建议: 优秀提示词必备的信息要素
_RECOMMEND_ELEMENTS: tuple[str, ...] = (
    "目标: 用一句话说清最终要得到什么",
    "输入: 列出可用的数据、结构或上下文对象",
    "输出: 说明期望的格式、字段与单位",
    "约束: 写清误差范围、边界条件与造型禁忌",
    "工具: 是否需要调用工具来验证/计算/检索",
)

# @context 引用建议: 提示词里如何引用上下文对象
_RECOMMEND_CONTEXT_REFS: tuple[str, ...] = (
    "引用结构/数据: @context 里指定要用到的文件名或对象 id",
    "引用记忆: 用 @memory 或明确说明要检索的关键词, 帮助召回历史事实",
    "显式声明依赖: 需要哪个工具的输出作为本步骤输入, 就写清楚 `先调 X 再…`",
)

# 工具使用建议 (通用部分)
_RECOMMEND_TOOL_USAGE: tuple[str, ...] = (
    "验证/计算类问题: 明确说『请调用工具拿真实结果, 不要只用记忆作答』",
    "检索类问题: 指定走 rag / 文献检索, 而非凭空推断",
    "对比/分析类问题: 说清要对比的维度与期望的对比表格式",
)


# ── optimize 的规则降级素材 ────────────────────────────────────────────

# replace mode 的规则兜底: 在遇到不清楚的结构时按通用模板重排
_RULE_REWRITE_TEMPLATE = (
    "{prompt}\n\n"
    "补充说明:\n"
    "- 目标: 明确要得到的结果\n"
    "- 输入: 列出可用的数据/结构/@context 引用\n"
    "- 输出: 说明期望格式、单位与关键字段\n"
    "- 约束: 写清误差范围与边界条件\n"
    "- 工具: 需要验证/计算/检索时指定对应工具"
)

# prepend mode 的规则兜底: 一段加在开头的引导语
_RULE_PREPEND = (
    "请先明确目标、输入、输出与约束, 需要时引用 @context 和相关数据, "
    "需要验证/计算/检索时指定对应工具, 不要只用记忆作答。"
)

# append mode 的规则兜底: 一段加在结尾的核对清单
_RULE_APPEND = (
    "请复核: 目标是否清晰、输入/输出是否明确、约束(误差/单位/边界)是否写全、"
    "需要时是否用到了对应工具。"
)


# ── optimize 的 LLM 系统提示词 ────────────────────────────────────────

_OPTIMIZE_SYSTEM_PROMPT = (
    "你是提示词工程专家。你的任务是根据给定的 mode 把用户提示词改写或补全得"
    "更清晰、更结构化、信息更完整。\n"
    "要点:\n"
    "- 保留用户提示词的核心语义, 不要篡改事实。\n"
    "- 补充缺失的目标 / 输入 / 输出 / 约束 / 工具使用要素。\n"
    "- 需要外部数据时建议 @context 引用, 而不是把对象内容全文贴进来。\n"
    "- 只输出要求的内容, 不要加解释文字。\n"
    "mode 语义:\n"
    "- replace: 重写整段提示词, 输出改写后的完整正文。\n"
    "- prepend: 生成一小段用于加在原提示词【之前】的引导语, 只输出该段。\n"
    "- append : 生成一小段用于加在原提示词【之后】的核对清单, 只输出该段。"
)


class PromptOptimizeInput(BaseModel):
    """prompt_optimize 工具的输入 schema.

    - action='recommend': 只需要 objective, 产出提示词改进建议 (规则, 不调 LLM)。
    - action='optimize' : 需要 prompt + 意图/约束, 调 LLM 重写/补全。
    """

    action: Literal["recommend", "optimize"] = Field(
        default="recommend",
        description=(
            "recommend: 输入用户目标/意图, 产出提示词改进建议(结构/要素/@context/"
            "工具使用), 默认动作, 纯规则不调 LLM。"
            "optimize: 输入一段提示词+意图/约束, 调 LLM 重写/补全为更佳版本。"
        ),
    )
    objective: str = Field(
        default="",
        description=(
            "用户的目标/意图, 自然语言。recommend 动作下必填, "
            "例如 '帮我验证计算 Si 的带隙' / '调研高熵合金文献'。"
        ),
    )
    prompt: str = Field(
        default="",
        description=(
            "待优化的提示词文本 (用户 prompt 或系统提示词)。optimize 动作下必填。"
        ),
    )
    intent: str = Field(
        default="",
        description=(
            "optimize 时的意图/期望: 说明这段提示词想达成什么、目标用户或使用场景。"
        ),
    )
    constraints: str = Field(
        default="",
        description="optimize 时的约束: 误差范围、边界、输出格式、安全性等要求。",
    )
    mode: OptimizeMode = Field(
        default="replace",
        description=(
            "optimize 的改写方式, 对齐 prompt_patch 的 block 语义: "
            "replace(整体替换) / prepend(前置) / append(后置)。默认 replace。"
        ),
    )


class PromptOptimizeTool(HuginnTool):
    """元工具: 提示词推荐 / 一键优化。只做文本改写, 不执行, 不写回系统提示词。"""

    name = "prompt_optimize"
    category = "meta"
    description = (
        "提示词推荐/一键优化工具 (元工具, 面向用户, 只读文本处理). "
        "action='recommend': 输入用户目标/意图, 返回提示词改进建议(结构/要素/"
        "@context 引用/工具使用建议), 规则驱动不调 LLM. "
        "action='optimize': 输入一段提示词+意图/约束, 调 LLM 重写/补全为更佳版本, "
        "支持 replace/prepend/append 三种 mode, LLM 失败自动降级为规则模板. "
        "仅输出改写结果供用户采用, 不执行任何代码, 不自动写回系统提示词."
    )
    input_schema = PromptOptimizeInput
    # 纯文本改写, 无副作用, 可自动执行, 不破坏任何状态
    destructive = False
    read_only = True

    def is_read_only(self, args: PromptOptimizeInput) -> bool:
        return True

    async def _execute(self, args: Any, context: ToolContext) -> ToolResult:
        if isinstance(args, dict):
            args = PromptOptimizeInput(**args)
        if args.action == "recommend":
            return self._handle_recommend(args, context)
        return await self._handle_optimize(args, context)

    # ── recommend ──────────────────────────────────────────────────────

    def _handle_recommend(
        self, args: PromptOptimizeInput, context: ToolContext
    ) -> ToolResult:
        objective = (args.objective or "").strip()
        if not objective:
            return ToolResult(
                data=None,
                success=False,
                error="recommend 需要非空的 objective (用户目标/意图)",
            )
        data = self._rule_recommend(objective)
        data.setdefault("action", "recommend")
        return ToolResult(data=data, success=True)

    def _rule_recommend(self, objective: str) -> dict[str, Any]:
        """纯规则出建议。工具使用建议基于 guidance_hook 的关键词逻辑。"""
        structure = list(_RECOMMEND_STRUCTURE)
        elements = list(_RECOMMEND_ELEMENTS)
        context_refs = list(_RECOMMEND_CONTEXT_REFS)
        tool_usage = list(_RECOMMEND_TOOL_USAGE)
        recommended_tools: list[str] = []
        try:
            from huginn.hooks.prompt_guidance_hook import PromptGuidanceHook

            recommended_tools = PromptGuidanceHook()._recommend_tools(objective)
        except Exception:
            logger.debug("guidance_hook 关键词推荐失败, 忽略", exc_info=True)
        if recommended_tools:
            tool_usage.append(
                "根据目标命中的关键词, 建议调用: " + "、".join(recommended_tools)
            )
        return {
            "action": "recommend",
            "source": "rules",
            "objective": objective,
            "structure": structure,
            "elements": elements,
            "context_refs": context_refs,
            "tool_usage": tool_usage,
            "recommended_tools": recommended_tools,
            "summary": "基于关键词规则的提示词结构建议, 供用户直接采用。",
        }

    # ── optimize ───────────────────────────────────────────────────────

    async def _handle_optimize(
        self, args: PromptOptimizeInput, context: ToolContext
    ) -> ToolResult:
        prompt = (args.prompt or "").strip()
        if not prompt:
            return ToolResult(
                data=None,
                success=False,
                error="optimize 需要非空的 prompt (待优化的提示词)",
            )
        intent = (args.intent or "").strip()
        constraints = (args.constraints or "").strip()
        mode = args.mode if args.mode in ("replace", "prepend", "append") else "replace"

        rewritten: str | None = None
        try:
            model = self._get_model(context)
            rewritten = await self._llm_optimize(
                prompt, intent, constraints, mode, model
            )
        except Exception as exc:
            logger.warning("prompt_optimize LLM 调用失败, 降级为规则模板: %s", exc)
            rewritten = None

        if rewritten:
            optimized, applied, source, fallback = self._compose(
                prompt, rewritten, mode, source="llm"
            )
        else:
            optimized, applied, source, fallback = self._compose(
                prompt, "", mode, source="rules"
            )

        return ToolResult(
            data={
                "action": "optimize",
                "mode": mode,
                "optimized": optimized,
                "source": source,
                "fallback": fallback,
                "applied": applied,
                "original": prompt,
            },
            success=True,
        )

    def _compose(
        self, prompt: str, rewritten: str, mode: str, source: str
    ) -> tuple[str, str, str, bool]:
        """按 mode 组合最终文本。对齐 prompt_patch 的 replace/prepend/append。

        返回 (optimized, applied, source, fallback)。
        """
        if source == "llm":
            if mode == "replace":
                return rewritten, "llm_replace", "llm", False
            if mode == "prepend":
                return rewritten + "\n" + prompt, "llm_prepend", "llm", False
            return prompt + "\n" + rewritten, "llm_append", "llm", False
        # 规则降级
        if mode == "replace":
            return (
                _RULE_REWRITE_TEMPLATE.format(prompt=prompt),
                "rule_replace",
                "rules",
                True,
            )
        if mode == "prepend":
            return _RULE_PREPEND + "\n" + prompt, "rule_prepend", "rules", True
        return prompt + "\n" + _RULE_APPEND, "rule_append", "rules", True

    def _get_model(self, context: ToolContext) -> Any:
        """拿 LangChain chat model, 优先用 context.config, 温度压低增强确定性."""
        from huginn.llm import get_model

        config = getattr(context, "config", None)
        return get_model(config=config, temperature=0.1, max_tokens=1000)

    async def _llm_optimize(
        self,
        prompt: str,
        intent: str,
        constraints: str,
        mode: str,
        model: Any,
    ) -> str | None:
        """调一次 LLM 产出 rewrite/补全片段。失败返回 None."""
        from langchain_core.messages import HumanMessage, SystemMessage

        parts = [f"用户提示词:\n{prompt}"]
        if intent:
            parts.append(f"意图: {intent}")
        if constraints:
            parts.append(f"约束: {constraints}")
        body = "\n".join(parts)
        task = self._mode_instruction(mode)
        messages = [
            SystemMessage(content=_OPTIMIZE_SYSTEM_PROMPT),
            HumanMessage(content=f"{body}\n\n{mode} 指令: {task}"),
        ]
        if hasattr(model, "ainvoke"):
            response = await model.ainvoke(messages)
        else:
            response = await asyncio.to_thread(model.invoke, messages)
        content = response.content if hasattr(response, "content") else str(response)
        if not isinstance(content, str):
            content = str(content)
        content = content.strip()
        return content or None

    @staticmethod
    def _mode_instruction(mode: str) -> str:
        """按 mode 生成给 LLM 的指令, 语义对齐 prompt_patch."""
        if mode == "replace":
            return "重写整段提示词, 只输出改写后的完整正文。"
        if mode == "prepend":
            return "生成加在原提示词【之前】的引导语, 只输出该段。"
        return "生成加在原提示词【之后】的核对清单, 只输出该段。"

    def estimate_cost(self, args: PromptOptimizeInput) -> dict[str, float] | None:
        # recommend 纯规则零成本; optimize 至多一次 LLM 调用
        if getattr(args, "action", "recommend") == "recommend":
            return {"cpu_hours": 0.0, "gpu_hours": 0.0, "walltime_hours": 0.0001}
        return {"cpu_hours": 0.0, "gpu_hours": 0.0, "walltime_hours": 0.01}
