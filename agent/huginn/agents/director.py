"""Director / Pilot 双层 Agent 架构.

LingBot-World 2.0 的 Agentic Harness 启发: 把"决定做什么"和"执行怎么做"
拆成两个独立 Agent, 各自专注自己的领域.

- DirectorAgent (大脑): 读 pipeline 状态 + provenance + 假设图, 决定下一步实验
- PilotAgent (小脑): 接收 Directive, 调用工具执行, 返回结果

和 AutoloopEngine 的关系: Engine 的 run() 默认走内联的 6 阶段循环.
传入 use_director_pilot=True 时, perceive+hypothesize+plan 交给 DirectorAgent,
execute+validate 交给 PilotAgent, Engine 只负责编排和 learn/report.

两个 Agent 的分离让策略层可以:
1. 基于全局状态做长程规划 (不被工具调用细节分散注意力)
2. 主动提议下一步 (不等用户或压缩触发)
3. 并行规划多个步骤, Pilot 按序执行
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Directive:
    """Director 产出的结构化任务指令.

    比普通 plan dict 多了 hypothesis_id 和 expected_outcome,
    让 Pilot 知道"在验证什么假设"和"预期看到什么结果".
    """

    objective: str
    tool_hint: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    expected_outcome: str = ""
    hypothesis_id: str = ""
    # Director 可以一次规划多步, Pilot 按序执行
    follow_ups: list[dict[str, Any]] = field(default_factory=list)

    def to_plan_dict(self) -> dict[str, Any]:
        """转成 engine.run() 里 _plan 期望的 plan dict 格式."""
        return {
            "mode": self.tool_hint or "explore",
            "description": self.objective,
            "parameters": self.parameters,
            "rationale": self.rationale,
            "hypothesis_id": self.hypothesis_id,
            "expected_outcome": self.expected_outcome,
        }


@dataclass
class ExecutionResult:
    """Pilot 执行 Directive 后的返回值."""

    success: bool
    result: Any = None
    error: str = ""
    directive: Directive | None = None
    duration: float = 0.0
    # 关键产出文件, 写进 provenance
    output_files: list[str] = field(default_factory=list)
    key_properties: dict[str, Any] = field(default_factory=dict)


class DirectorAgent:
    """策略层 Agent — 读全局状态, 决定下一步做什么.

    LingBot-World 的 Director (VLM) 角色: 宏观语义推理 + 因果推断 + 时间规划.
    Huginn 的 Director 读 pipeline + provenance + hypothesis graph,
    综合 LLM 推理 + 规则匹配, 产出 Directive.

    不直接调工具, 只做"决定做什么"和"为什么做".
    """

    def __init__(self, model: Any = None) -> None:
        self._model = model
        # 缓存上一次的 directive, 让 pilot 知道上下文
        self._last_directive: Directive | None = None

    def propose(
        self,
        context: dict[str, Any] | None = None,
        hypothesis: str = "",
    ) -> Directive | None:
        """基于当前状态提议下一步实验.

        融合三个信息源:
        1. Pipeline 状态 (SimulationPipeline.suggest_next)
        2. Provenance 最近产出 (ProvenanceRegistry.recent)
        3. 假设图状态 (frontier / refuted)

        有 model 时走 LLM 推理, 没有时走规则匹配.
        """
        context = context or {}
        pipeline_hints = self._read_pipeline_state()
        prov_summary = self._read_provenance()
        hyp_state = self._read_hypothesis_graph()

        # 先试规则匹配: pipeline 有明确建议就直接用
        if pipeline_hints:
            hint = pipeline_hints[0]
            directive = Directive(
                objective=hint.get("description", "proceed with next pipeline step"),
                tool_hint=hint.get("tool_hint", ""),
                parameters=hint.get("parameters", {}),
                rationale=hint.get("reason", "pipeline suggested next step"),
                expected_outcome=hint.get("expected", ""),
            )
            self._last_directive = directive
            return directive

        # 有 LLM 时, 把三个信息源拼起来让 LLM 决策
        if self._model is not None:
            return self._llm_propose(context, hypothesis, prov_summary, hyp_state)

        # 没有规则匹配也没有 LLM, 用 provenance 最近产出推断
        if prov_summary:
            latest = prov_summary[0]
            directive = Directive(
                objective=f"Continue from {latest.get('produced_by', 'previous step')}",
                tool_hint=latest.get("tool_hint", ""),
                rationale="No pipeline suggestion, following provenance trail",
            )
            self._last_directive = directive
            return directive

        return None

    def _read_pipeline_state(self) -> list[dict[str, Any]]:
        """从 SimulationPipeline 读当前状态和建议."""
        try:
            from huginn.provenance.pipeline import get_pipeline
            pipeline = get_pipeline()
            suggestions = pipeline._latest
            if not suggestions:
                entry = pipeline._latest_entry()
                if entry is not None:
                    suggestions = pipeline.suggest_next(
                        entry.produced_by, entry.parameters, {}
                    )
            return [
                {
                    "tool_hint": s.tool_hint,
                    "description": s.description,
                    "reason": s.reason,
                    "stage": s.stage.value if hasattr(s.stage, "value") else str(s.stage),
                    "prerequisite_met": s.prerequisite_met,
                }
                for s in suggestions if s.prerequisite_met
            ]
        except Exception:
            logger.debug("pipeline state read failed", exc_info=True)
            return []

    def _read_provenance(self) -> list[dict[str, Any]]:
        """从 ProvenanceRegistry 读最近产出."""
        try:
            from huginn.provenance.registry import ProvenanceRegistry
            reg = ProvenanceRegistry.shared()
            entries = reg._entries[-5:]  # 最近 5 条
            return [e.to_dict() for e in entries]
        except Exception:
            return []

    def _read_hypothesis_graph(self) -> dict[str, Any]:
        """读假设图状态: 有几个未测、几个被反驳."""
        try:
            # engine 持有 hypothesis_graph, 这里不直接访问
            # 只读上次 directive 的 hypothesis_id 做上下文
            if self._last_directive and self._last_directive.hypothesis_id:
                return {"last_hypothesis": self._last_directive.hypothesis_id}
            return {}
        except Exception:
            return {}

    def _llm_propose(
        self,
        context: dict[str, Any],
        hypothesis: str,
        prov_summary: list[dict[str, Any]],
        hyp_state: dict[str, Any],
    ) -> Directive | None:
        """用 LLM 综合三个信息源生成 Directive."""
        try:
            parts = ["You are a materials science research strategist."]
            parts.append(f"Current context: {context.get('summary', 'N/A')}")
            if hypothesis:
                parts.append(f"Current hypothesis: {hypothesis}")
            if prov_summary:
                recent = "; ".join(
                    f"{e.get('produced_by', '?')}→{e.get('file_format', '?')}"
                    for e in prov_summary[:3]
                )
                parts.append(f"Recent outputs: {recent}")
            if hyp_state:
                parts.append(f"Hypothesis state: {hyp_state}")
            parts.append(
                "Propose the NEXT single experiment step. "
                "Format: objective|tool_hint|rationale|expected_outcome"
            )
            prompt = "\n".join(parts)
            response = self._model.invoke(prompt)
            text = response if isinstance(response, str) else str(response)
            # 简单解析, LLM 不按格式就退回规则匹配
            if "|" in text:
                fields = text.split("|", 3)
                directive = Directive(
                    objective=fields[0].strip(),
                    tool_hint=fields[1].strip() if len(fields) > 1 else "",
                    rationale=fields[2].strip() if len(fields) > 2 else "",
                    expected_outcome=fields[3].strip() if len(fields) > 3 else "",
                )
                self._last_directive = directive
                return directive
        except Exception:
            logger.debug("LLM propose failed", exc_info=True)
        return None


class PilotAgent:
    """执行层 Agent — 接收 Directive, 调用工具执行.

    LingBot-World 的 Pilot (DiT) 角色: 把语义决策落地成物理操作.
    Huginn 的 Pilot 拿到 Directive 后, 通过 ToolAdapter 调用对应工具,
    收集结果, 可选地注册到 provenance.
    """

    def __init__(self, tool_adapter: Any = None) -> None:
        self._tool_adapter = tool_adapter
        self._execution_count = 0

    async def execute(self, directive: Directive) -> ExecutionResult:
        """执行一个 Directive, 返回结构化结果."""
        t0 = time.time()
        self._execution_count += 1
        result = ExecutionResult(
            success=False,
            directive=directive,
            duration=0.0,
        )

        if self._tool_adapter is None:
            result.error = "no tool adapter configured"
            result.duration = time.time() - t0
            return result

        try:
            tool_name = directive.tool_hint or "explore"
            # 用 tool_adapter 调工具, 参数从 directive.parameters 取
            output = await self._tool_adapter.call(
                tool_name=tool_name,
                action=directive.parameters.get("action", "run"),
                **{k: v for k, v in directive.parameters.items() if k != "action"},
            )

            if isinstance(output, dict):
                result.success = not output.get("error")
                result.result = output.get("result", output)
                result.error = output.get("error", "")
                # 提取输出文件和关键属性
                result.output_files = output.get("output_files", [])
                if isinstance(result.output_files, dict):
                    result.output_files = list(result.output_files.values())
                result.key_properties = self._extract_key_props(output)
            else:
                result.success = True
                result.result = output
        except Exception as exc:
            result.error = str(exc)
            logger.warning("pilot execute failed: %s", exc, exc_info=True)

        result.duration = time.time() - t0
        return result

    def _extract_key_props(self, output: dict[str, Any]) -> dict[str, Any]:
        """从工具输出里提取关键科学属性 (能量/带隙/收敛状态等)."""
        props: dict[str, Any] = {}
        data = output.get("result", output)
        if not isinstance(data, dict):
            return props
        for key in ("energy", "bandgap", "converged", "spacegroup",
                     "forces", "stress", "total_energy", "fermi_energy"):
            if key in data:
                props[key] = data[key]
        return props


# ── Engine 集成 ──────────────────────────────────────────────────


def create_director_pilot(engine: Any) -> tuple[DirectorAgent, PilotAgent]:
    """从 AutoloopEngine 创建绑定的 Director/Pilot 对.

    Director 绑定 engine.model, Pilot 绑定 engine 的 tool_adapter.
    """
    model = getattr(engine, "model", None)
    # tool_adapter 从 agent 或 engine 上拿
    adapter = getattr(engine, "_tool_adapter", None)
    if adapter is None:
        # 尝试从 workflow_engine 上拿
        wf = getattr(engine, "workflow_engine", None)
        if wf is not None:
            adapter = getattr(wf, "tool_adapter", None)

    director = DirectorAgent(model=model)
    pilot = PilotAgent(tool_adapter=adapter)
    return director, pilot
