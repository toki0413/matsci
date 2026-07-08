"""事件驱动的仿真管线 — 根据工具完成情况自动建议下一步.

当一个仿真工具成功完成并通过 hook 检查后, 管线根据预定义的科学逻辑
建议下一步操作. agent 不需要记住"该做什么了", 管线会推着它走.

典型流程:
  DFT 线: structure_tool → vasp_tool(relax) → vasp_tool(static) → vasp_tool(band/dos)
  MD  线: packing_tool → lammps_tool → compute_msd/compute_rdf

建议是信息性的, 不 block 工具调用. 最终决策权在 agent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from huginn.hooks import HookContext
from huginn.provenance.registry import ProvenanceEntry, ProvenanceRegistry

logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """仿真管线的阶段, 按科学逻辑排序."""

    STRUCTURE = "structure"    # 结构准备 (structure_tool, packing_tool)
    RELAX = "relax"            # 结构优化 (vasp_tool relax, qe_tool, cp2k_tool, lammps_tool)
    STATIC = "static"          # 静态计算 (vasp_tool static)
    PROPERTIES = "properties"  # 性质计算 (vasp_tool band/dos, characterization_tool)
    MECHANICAL = "mechanical"  # 力学性质 (mechanical_tool, 弹性常数/模量/硬度)
    MD = "md"                  # 分子动力学 (lammps_tool, gromacs_tool)
    ANALYSIS = "analysis"      # 数据分析 (任何产出最终结果的工具)


# ponytail: 把 DFT 和 MD 两条平行路线拍平到一条线性序号上算完成度.
# MD(4) 排在 PROPERTIES(3) 后面, 走 MD 线时会跳过 STATIC/PROPERTIES,
# 完成度直接从 20% 跳到 80%, 不精确但够用. 真要分线得改成 DAG, 没必要.
_STAGE_ORDER: dict[PipelineStage, int] = {
    PipelineStage.STRUCTURE: 0,
    PipelineStage.RELAX: 1,
    PipelineStage.STATIC: 2,
    PipelineStage.PROPERTIES: 3,
    PipelineStage.MECHANICAL: 3,  # 和 PROPERTIES 同级, 走力学线
    PipelineStage.MD: 4,
    PipelineStage.ANALYSIS: 5,
}

# 各阶段的前置阶段, prerequisite_met 检查用
_STAGE_PREREQUISITES: dict[PipelineStage, list[PipelineStage]] = {
    PipelineStage.STRUCTURE: [],
    PipelineStage.RELAX: [PipelineStage.STRUCTURE],
    PipelineStage.STATIC: [PipelineStage.RELAX],
    PipelineStage.PROPERTIES: [PipelineStage.STATIC, PipelineStage.RELAX],
    PipelineStage.MECHANICAL: [PipelineStage.RELAX],
    PipelineStage.MD: [PipelineStage.STRUCTURE],
    PipelineStage.ANALYSIS: [PipelineStage.MD, PipelineStage.PROPERTIES, PipelineStage.MECHANICAL],
}

# 管线认识的所有工具, 遍历注册表时用
_KNOWN_TOOLS = [
    "structure_tool", "packing_tool", "convergence_test_tool",
    "vasp_tool", "qe_tool", "cp2k_tool", "lammps_tool", "gromacs_tool",
    "mechanical_tool", "characterization_tool",
    "compute_msd", "compute_rdf",
]


@dataclass
class PipelineRule:
    """一条管线规则: 某工具(某 action)完成后可以做什么."""

    stage: PipelineStage
    tool_name: str
    action_matcher: str | None  # 匹配 tool_input 中的 action, None 表示不限
    next_stages: list[PipelineStage]
    next_tool_hints: list[str]
    description: str


# 预定义规则表
PIPELINE_RULES: list[PipelineRule] = [
    PipelineRule(
        stage=PipelineStage.STRUCTURE,
        tool_name="structure_tool",
        action_matcher=None,
        next_stages=[PipelineStage.RELAX],
        next_tool_hints=["vasp_tool", "qe_tool", "lammps_tool"],
        description="结构准备完成, 下一步进行结构优化",
    ),
    PipelineRule(
        stage=PipelineStage.STRUCTURE,
        tool_name="packing_tool",
        action_matcher=None,
        next_stages=[PipelineStage.MD],
        next_tool_hints=["lammps_tool", "gromacs_tool"],
        description="分子堆积完成, 下一步进行分子动力学模拟",
    ),
    PipelineRule(
        stage=PipelineStage.RELAX,
        tool_name="vasp_tool",
        action_matcher="relax",
        next_stages=[PipelineStage.STATIC, PipelineStage.PROPERTIES],
        next_tool_hints=["vasp_tool", "mechanical_tool"],
        description="结构优化完成, 下一步可做静态计算或力学性质计算",
    ),
    PipelineRule(
        stage=PipelineStage.RELAX,
        tool_name="qe_tool",
        action_matcher=None,
        next_stages=[PipelineStage.STATIC, PipelineStage.PROPERTIES],
        next_tool_hints=["qe_tool"],
        description="QE 结构优化完成, 下一步做静态计算或性质计算",
    ),
    PipelineRule(
        stage=PipelineStage.RELAX,
        tool_name="cp2k_tool",
        action_matcher=None,
        next_stages=[PipelineStage.STATIC, PipelineStage.PROPERTIES],
        next_tool_hints=["cp2k_tool"],
        description="CP2K 结构优化完成, 下一步做静态计算或性质计算",
    ),
    PipelineRule(
        stage=PipelineStage.RELAX,
        tool_name="lammps_tool",
        action_matcher="relax",
        next_stages=[PipelineStage.MD],
        next_tool_hints=["lammps_tool"],
        description="LAMMPS 结构优化完成, 下一步可做分子动力学模拟",
    ),
    PipelineRule(
        stage=PipelineStage.STATIC,
        tool_name="vasp_tool",
        action_matcher="static",
        next_stages=[PipelineStage.PROPERTIES],
        next_tool_hints=["vasp_tool"],
        description="静态计算完成, 下一步计算能带/态密度等性质",
    ),
    PipelineRule(
        stage=PipelineStage.PROPERTIES,
        tool_name="vasp_tool",
        action_matcher="band",
        next_stages=[PipelineStage.ANALYSIS],
        next_tool_hints=["characterization_tool"],
        description="能带计算完成, 可进入数据分析阶段",
    ),
    PipelineRule(
        stage=PipelineStage.PROPERTIES,
        tool_name="vasp_tool",
        action_matcher="dos",
        next_stages=[PipelineStage.ANALYSIS],
        next_tool_hints=["characterization_tool"],
        description="态密度计算完成, 可进入数据分析阶段",
    ),
    PipelineRule(
        stage=PipelineStage.PROPERTIES,
        tool_name="mechanical_tool",
        action_matcher=None,
        next_stages=[PipelineStage.ANALYSIS],
        next_tool_hints=["characterization_tool"],
        description="力学性质计算完成, 可进入数据分析阶段",
    ),
    PipelineRule(
        stage=PipelineStage.MD,
        tool_name="lammps_tool",
        action_matcher=None,
        next_stages=[PipelineStage.ANALYSIS],
        next_tool_hints=["compute_msd", "compute_rdf"],
        description="分子动力学完成, 下一步分析轨迹 (MSD/RDF)",
    ),
    PipelineRule(
        stage=PipelineStage.MD,
        tool_name="gromacs_tool",
        action_matcher=None,
        next_stages=[PipelineStage.ANALYSIS],
        next_tool_hints=["compute_msd", "compute_rdf"],
        description="分子动力学完成, 下一步分析轨迹 (MSD/RDF)",
    ),
    PipelineRule(
        stage=PipelineStage.STRUCTURE,
        tool_name="convergence_test_tool",
        action_matcher=None,
        next_stages=[PipelineStage.RELAX],
        next_tool_hints=["vasp_tool", "qe_tool"],
        description="收敛测试通过, 下一步用收敛参数做结构优化",
    ),
]


@dataclass
class PipelineSuggestion:
    """单条下一步建议."""

    stage: PipelineStage
    tool_hint: str
    description: str
    prerequisite_met: bool  # provenance 里有没有必需的输入文件
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value,
            "tool_hint": self.tool_hint,
            "description": self.description,
            "prerequisite_met": self.prerequisite_met,
            "reason": self.reason,
        }


# ── 工具函数 ──────────────────────────────────────────────────────


def _infer_stage(tool_name: str, tool_input: dict[str, Any]) -> PipelineStage | None:
    """从工具名和输入参数推断它属于哪个阶段."""
    action = str(tool_input.get("action", "")).lower()

    if tool_name in ("structure_tool", "packing_tool", "convergence_test_tool"):
        return PipelineStage.STRUCTURE

    if tool_name == "vasp_tool":
        if action == "relax":
            return PipelineStage.RELAX
        if action == "static":
            return PipelineStage.STATIC
        if action in ("band", "dos"):
            return PipelineStage.PROPERTIES
        return PipelineStage.RELAX  # 没写 action 默认当优化

    if tool_name in ("qe_tool", "cp2k_tool"):
        if action in ("static", "scf"):
            return PipelineStage.STATIC
        return PipelineStage.RELAX

    if tool_name == "lammps_tool":
        # ponytail: 只靠 action 字符串区分 relax/md, 没考虑 LAMMPS 的
        # minimize/cg/fire 等变体. 如果 action 没写或写错, 默认当 MD.
        if action in ("relax", "minimize", "min", "cg"):
            return PipelineStage.RELAX
        return PipelineStage.MD

    if tool_name == "gromacs_tool":
        return PipelineStage.MD

    if tool_name in ("mechanical_tool", "characterization_tool"):
        return PipelineStage.PROPERTIES

    if tool_name in ("compute_msd", "compute_rdf"):
        return PipelineStage.ANALYSIS

    return None


def _is_converged(tool_output: Any) -> bool:
    """检查 convergence_test_tool 的输出是否收敛."""
    if not isinstance(tool_output, dict):
        return False
    result = tool_output.get("result", tool_output)
    if not isinstance(result, dict):
        return False
    return result.get("converged") is True


# ── SimulationPipeline ────────────────────────────────────────────


class SimulationPipeline:
    """事件驱动的仿真管线, 持有 ProvenanceRegistry 引用, 根据工具完成情况建议下一步."""

    def __init__(self, registry: ProvenanceRegistry) -> None:
        self._registry = registry
        # 缓存最近一次 suggest_next 的结果, get_progress / to_context_block 用
        self._latest: list[PipelineSuggestion] = []

    # ---- 核心方法 ----

    def suggest_next(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_output: Any,
    ) -> list[PipelineSuggestion]:
        """根据当前工具和输出, 返回下一步建议."""
        action = str(tool_input.get("action", "")).lower()

        # 找所有匹配的规则
        matched: list[PipelineRule] = []
        for rule in PIPELINE_RULES:
            if rule.tool_name != tool_name:
                continue
            if rule.action_matcher is not None and rule.action_matcher.lower() != action:
                continue
            matched.append(rule)

        # 有特定 action 匹配时, 丢弃 action_matcher=None 的通用规则
        # 比如 lammps_tool(action=relax) 同时命中 relax 规则和 None 规则, 只留 relax
        specific = [r for r in matched if r.action_matcher is not None]
        if specific:
            matched = specific

        # convergence_test_tool 没收敛就不建议往下走
        if tool_name == "convergence_test_tool" and not _is_converged(tool_output):
            self._latest = []
            return []

        suggestions: list[PipelineSuggestion] = []
        for rule in matched:
            for stage in rule.next_stages:
                # stage 和 hints 数量对齐时取对应 hint, 否则拼一起
                if len(rule.next_tool_hints) == len(rule.next_stages):
                    hint = rule.next_tool_hints[rule.next_stages.index(stage)]
                else:
                    hint = ", ".join(rule.next_tool_hints) if rule.next_tool_hints else ""

                suggestions.append(PipelineSuggestion(
                    stage=stage,
                    tool_hint=hint,
                    description=rule.description,
                    prerequisite_met=self._check_prerequisite(stage),
                    reason=f"{tool_name}({action or 'default'}) 已完成",
                ))

        self._latest = suggestions
        return suggestions

    def get_current_stage(self) -> PipelineStage | None:
        """从 provenance 注册表推断当前处于哪个阶段 (取已完成的最高阶段)."""
        completed = self._get_completed_stages()
        if not completed:
            return None
        return max(completed, key=lambda s: _STAGE_ORDER.get(s, 0))

    def get_progress(self) -> dict[str, Any]:
        """返回管线进度: 当前阶段 / 已完成阶段 / 建议 / 完成度."""
        current = self.get_current_stage()
        completed = self._get_completed_stages()

        if current is None:
            pct = 0
        else:
            max_order = max(_STAGE_ORDER.values())
            pct = int(_STAGE_ORDER.get(current, 0) / max_order * 100)

        # 没有缓存的建议时, 从最近的注册表条目推一份
        suggestions = self._latest
        if not suggestions:
            entry = self._latest_entry()
            if entry is not None:
                suggestions = self.suggest_next(
                    entry.produced_by, entry.parameters, {}
                )

        return {
            "current_stage": current.value if current else None,
            "completed_stages": [
                s.value for s in sorted(completed, key=lambda st: _STAGE_ORDER.get(st, 0))
            ],
            "suggested_next": [s.to_dict() for s in suggestions],
            "completion_pct": pct,
        }

    def to_context_block(self) -> str:
        """生成可插入上下文的状态块, 告诉 agent 现在在哪个阶段, 下一步该做什么."""
        current = self.get_current_stage()
        if current is None:
            return ""

        progress = self.get_progress()
        lines = ["### Simulation pipeline status:"]
        lines.append(f"  Current stage: {current.value}")
        lines.append(f"  Completion: {progress['completion_pct']}%")
        if progress["completed_stages"]:
            lines.append(f"  Completed: {', '.join(progress['completed_stages'])}")

        suggestions = self._latest
        if not suggestions:
            entry = self._latest_entry()
            if entry is not None:
                suggestions = self.suggest_next(
                    entry.produced_by, entry.parameters, {}
                )

        if suggestions:
            lines.append("  Suggested next steps:")
            for s in suggestions:
                status = "ready" if s.prerequisite_met else "prerequisites missing"
                lines.append(
                    f"    - [{s.stage.value}] {s.tool_hint}: {s.description} ({status})"
                )
        else:
            lines.append("  No further suggestions — pipeline may be complete.")

        return "\n".join(lines)

    # ---- 内部方法 ----

    def _get_completed_stages(self) -> set[PipelineStage]:
        """扫描注册表, 返回所有已完成阶段."""
        stages: set[PipelineStage] = set()
        for tool in _KNOWN_TOOLS:
            for entry in self._registry.find_by_tool(tool):
                stage = _infer_stage(entry.produced_by, entry.parameters)
                if stage is not None:
                    stages.add(stage)
        return stages

    def _check_prerequisite(self, stage: PipelineStage) -> bool:
        """检查目标阶段的前置阶段是否已在注册表里."""
        prereqs = _STAGE_PREREQUISITES.get(stage, [])
        if not prereqs:
            return True
        completed = self._get_completed_stages()
        return any(p in completed for p in prereqs)

    def _latest_entry(self) -> ProvenanceEntry | None:
        """找注册表里时间戳最新的条目."""
        latest: ProvenanceEntry | None = None
        for tool in _KNOWN_TOOLS:
            for entry in self._registry.find_by_tool(tool):
                if latest is None or entry.produced_at > latest.produced_at:
                    latest = entry
        return latest


# ── 模块级单例 ────────────────────────────────────────────────────

_pipeline: SimulationPipeline | None = None


def get_pipeline() -> SimulationPipeline:
    """获取管线单例, 内部绑定 ProvenanceRegistry.shared()."""
    global _pipeline
    if _pipeline is None:
        _pipeline = SimulationPipeline(ProvenanceRegistry.shared())
    return _pipeline


# ── POST_TOOL_USE hook ────────────────────────────────────────────


async def pipeline_hook(ctx: HookContext) -> HookContext | None:
    """POST_TOOL_USE: 工具成功完成后建议下一步, 不 block, 纯信息性.

    建议存入 ctx.metadata["pipeline_suggestions"], 上层 adapter / agent 可读取.
    """
    try:
        # 工具失败时不建议
        if ctx.error is not None:
            return None
        result = ctx.result if isinstance(ctx.result, dict) else {}
        if result.get("error"):
            return None

        tool_input = ctx.args if isinstance(ctx.args, dict) else {}
        pipeline = get_pipeline()
        suggestions = pipeline.suggest_next(ctx.tool_name, tool_input, ctx.result)

        if suggestions:
            ctx.metadata["pipeline_suggestions"] = [s.to_dict() for s in suggestions]
    except Exception:
        # 建议失败不能影响主流程
        logger.debug("pipeline_hook failed (non-fatal)", exc_info=True)
    return None
