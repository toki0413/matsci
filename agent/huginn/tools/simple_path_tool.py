"""最简路径推荐工具 —— 按 Least Effort Path 决策树给任务排出推荐工具链.

跟 scenario_tool 不一样: scenario_tool 是把任务场景映射到预置 bundle,
simple_path_tool 是按 "常量查询 → 解析/经验 → 重型仿真 → 自定义代码"
四级决策树, 让 LLM 现场分析任务给出按优先级排序的工具链, 并标出哪些
重型工具被避开了. 用途:

  1. LLM 拿不准先调啥工具时, 调一次 simple_path_tool 拿到推荐路径
  2. 上层做任务规划时, 拿 recommended_path 当 skeleton
  3. heavy_tools_avoided 给 telemetry / 审计用, 看路径压缩效果

实现: 一次 LLM 调用做任务分析, 输出结构化 JSON, 失败时退到关键词兜底.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from pydantic import BaseModel, Field

from huginn.agents.tool_call_router import HEAVY_TOOLS, LIGHT_TOOLS
from huginn.tools.base import HuginnTool
from huginn.types import ToolContext, ToolResult

logger = logging.getLogger(__name__)


# Least Effort Path 决策树, 喂给 LLM 让它按这个分级推荐
_LEP_SYSTEM_PROMPT = """你是材料科研任务路径规划器. 按 "Least Effort Path" 决策树给用户任务排出推荐工具链.

决策树 (按优先级, 越靠前越优先):
1. 已知常量/查表 (元素性质/带隙/晶格常数/常见结构) → materials_database_tool / local_structure_db / rag_tool / web_search_tool
2. 解析解/经验公式 → symbolic_math_tool / numerical_tool (例: Murnaghan EOS 拟合 → numerical_tool.curve_fit)
3. 重型仿真 (DFT/MD/FEA/CFD) → vasp_tool / qe_tool / cp2k_tool / lammps_tool / abaqus_tool / comsol_tool / openfoam_tool
4. 自定义代码 → code_tool / bash_tool (只有现有工具组合做不到时才用)

判断要点:
- 能查到的常量不要算, 能解析算的不要仿真, 能用专用工具的不要写代码
- recommended_path 按调用顺序排, 每一步都要有必要性
- heavy_tools_avoided 列出 "本来可能用上但被更简路径避开" 的重型工具
- 如果任务确实需要重型仿真, recommended_path 里可以有重型工具, 但 rationale 要说清为什么 1&2 不够

输出严格 JSON, 不要 markdown 代码块, 不要解释文字:
{
  "recommended_path": ["tool_name_1", "tool_name_2", ...],
  "rationale": "一句话说明为什么是这条路径",
  "heavy_tools_avoided": ["vasp_tool", ...],
  "step_notes": {"tool_name_1": "这一步做啥", "tool_name_2": "这一步做啥"}
}"""


# 关键词兜底: LLM 不可用时按任务描述里的关键词粗匹配到一条路径
# 不求精准, 只求别返回空, 让上层有东西可用
_KEYWORD_FALLBACK_PATHS: list[tuple[list[str], dict[str, Any]]] = [
    (
        ["带隙", "band gap", "禁带", "晶格常数", "lattice constant", "元素性质"],
        {
            "recommended_path": ["materials_database_tool", "rag_tool"],
            "rationale": "常量查询, 先查数据库/RAG, 不需要计算",
            "heavy_tools_avoided": ["vasp_tool", "qe_tool", "cp2k_tool"],
            "step_notes": {
                "materials_database_tool": "查 MP/OQMD 拉带隙/晶格常数",
                "rag_tool": "数据库没有就查本地知识库",
            },
        },
    ),
    (
        ["eos", "状态方程", "equation of state", "murnaghan", "birch"],
        {
            "recommended_path": ["numerical_tool"],
            "rationale": "EOS 拟合是数值曲线拟合, 用 numerical_tool.curve_fit, 不用跑额外 DFT",
            "heavy_tools_avoided": ["vasp_tool"],
            "step_notes": {
                "numerical_tool": "对已有 E-V 数据做 Murnaghan/Birch-Murnaghan 拟合",
            },
        },
    ),
    (
        ["结构优化", "弛豫", "relax", "structure optimization"],
        {
            "recommended_path": ["structure_tool", "vasp_tool", "validate_tool"],
            "rationale": "结构优化需要 DFT 弛豫, 没有更简路径, 但要先 structure_tool 准备结构",
            "heavy_tools_avoided": [],
            "step_notes": {
                "structure_tool": "解析/准备输入结构",
                "vasp_tool": "ISIF=3 弛豫",
                "validate_tool": "校验受力/能量收敛",
            },
        },
    ),
    (
        ["弹性常数", "elastic constants", "c11", "c12", "c44", "弹性张量"],
        {
            "recommended_path": ["vasp_tool", "validate_tool"],
            "rationale": "弹性常数需要 DFT 应变-应力计算, vasp_tool 有 elastic_constants 动作",
            "heavy_tools_avoided": [],
            "step_notes": {
                "vasp_tool": "调 elastic_constants 动作算 C_ij",
                "validate_tool": "校验 Born 力学稳定性",
            },
        },
    ),
    (
        ["文献", "literature", "综述", "调研", "review"],
        {
            "recommended_path": ["web_search_tool", "rag_tool", "report_tool"],
            "rationale": "文献调研走搜索 + RAG, 不需要任何仿真",
            "heavy_tools_avoided": ["vasp_tool", "lammps_tool", "abaqus_tool"],
            "step_notes": {
                "web_search_tool": "搜网络资料",
                "rag_tool": "查本地知识库",
                "report_tool": "汇总成综述",
            },
        },
    ),
]


class SimplePathToolInput(BaseModel):
    task_description: str = Field(
        ...,
        description=(
            "用户任务的自然语言描述. 例如 '我要算硅的带隙' / "
            "'拟合 Murnaghan EOS' / '优化 Cu 结构'."
        ),
    )


class SimplePathTool(HuginnTool):
    """元工具: 按 Least Effort Path 决策树给任务推荐最简工具链."""

    name = "simple_path_tool"
    category = "meta"
    description = (
        "最简路径推荐器 (元工具). 输入任务描述, 按 '常量查询 → 解析/经验 → "
        "重型仿真 → 自定义代码' 四级决策树给出按优先级排序的推荐工具链、"
        "rationale 和被避开的重型工具列表. 用于让 LLM 在动手前先看清楚最简路径."
    )
    input_schema = SimplePathToolInput
    # 只调一次 LLM 做分析, 不写文件不跑计算
    read_only = True

    async def call(
        self, args: SimplePathToolInput, context: ToolContext
    ) -> ToolResult:
        task = (args.task_description or "").strip()
        if not task:
            return ToolResult(
                data=None,
                success=False,
                error="task_description 不能为空",
            )

        # 1. 优先用 LLM 按决策树分析
        path_data: dict[str, Any] | None = None
        match_source = ""
        try:
            model = self._get_model(context)
            path_data = await self._llm_plan(task, model)
            if path_data is not None:
                match_source = "llm"
        except Exception as exc:
            logger.warning("simple_path_tool LLM 分析失败, 退回关键词兜底: %s", exc)

        # 2. LLM 没给可用结果 → 关键词兜底
        if path_data is None:
            path_data = self._keyword_fallback(task)
            match_source = "keyword"

        if path_data is None:
            # 兜底都没匹配上, 给一个最保守的默认路径
            path_data = {
                "recommended_path": ["web_search_tool", "rag_tool"],
                "rationale": "无法识别任务类型, 默认先查资料再决定下一步",
                "heavy_tools_avoided": sorted(HEAVY_TOOLS),
                "step_notes": {},
            }
            match_source = "default"

        # 3. 规范化: 去掉推荐路径里不存在的工具名, 重型工具确实需要的保留
        path_data = self._sanitize(path_data)
        path_data["match_source"] = match_source
        # 附上决策树分级, 方便上层做审计
        path_data["decision_tree"] = {
            "level_1_lookup": sorted(LIGHT_TOOLS),
            "level_3_heavy": sorted(HEAVY_TOOLS),
        }
        return ToolResult(data=path_data, success=True)

    # ------------------------------------------------------------------ helpers

    def _get_model(self, context: ToolContext) -> Any:
        """拿一个 LangChain chat model, 低温保证路径规划确定性."""
        from huginn.llm import get_model

        config = getattr(context, "config", None)
        return get_model(config=config, temperature=0.1, max_tokens=600)

    async def _llm_plan(
        self, task: str, model: Any
    ) -> dict[str, Any] | None:
        """调 LLM 按决策树给推荐路径, 失败返回 None."""
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content=_LEP_SYSTEM_PROMPT),
            HumanMessage(content=f"用户任务: {task}"),
        ]

        if hasattr(model, "ainvoke"):
            response = await model.ainvoke(messages)
        else:
            response = await asyncio.to_thread(model.invoke, messages)

        content = response.content if hasattr(response, "content") else str(response)
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)

        return self._parse_plan_json(content)

    def _parse_plan_json(self, content: str) -> dict[str, Any] | None:
        """从 LLM 回复抠 JSON, 容忍前后多余文字和 ```json 代码块."""
        text = content.strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
            text = re.sub(r"\n?```\s*$", "", text)

        data: dict[str, Any] | None = None
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    data = json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    data = None

        if not isinstance(data, dict):
            return None
        # 必须有 recommended_path 且是 list
        path = data.get("recommended_path")
        if not isinstance(path, list) or not path:
            return None
        return data

    def _keyword_fallback(self, task: str) -> dict[str, Any] | None:
        """关键词兜底匹配, 命中第一个就返回."""
        lowered = task.lower()
        for keywords, payload in _KEYWORD_FALLBACK_PATHS:
            for kw in keywords:
                if any(ord(c) > 127 for c in kw):
                    if kw in task:
                        return json.loads(json.dumps(payload))  # deep copy
                else:
                    if kw in lowered:
                        return json.loads(json.dumps(payload))
        return None

    def _sanitize(self, data: dict[str, Any]) -> dict[str, Any]:
        """规范化 LLM 输出: 字段补齐 + 类型修正."""
        # recommended_path 必须是 list[str]
        path = data.get("recommended_path")
        if not isinstance(path, list):
            path = []
        path = [str(t) for t in path if t]
        data["recommended_path"] = path

        # rationale 补默认
        if not isinstance(data.get("rationale"), str) or not data["rationale"]:
            data["rationale"] = "路径由 simple_path_tool 推荐"

        # heavy_tools_avoided 补默认 + 去掉不在 HEAVY_TOOLS 里的
        avoided = data.get("heavy_tools_avoided")
        if not isinstance(avoided, list):
            avoided = []
        avoided = sorted({str(t) for t in avoided if str(t) in HEAVY_TOOLS})
        data["heavy_tools_avoided"] = avoided

        # step_notes 补默认
        if not isinstance(data.get("step_notes"), dict):
            data["step_notes"] = {}

        return data

    def estimate_cost(self, args: SimplePathToolInput) -> dict[str, float] | None:
        # 只调一次 LLM 做规划, 成本可忽略
        return {"cpu_hours": 0.0, "gpu_hours": 0.0, "walltime_hours": 0.001}
