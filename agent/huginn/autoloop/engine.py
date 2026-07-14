"""Autoloop Engine — the main autonomous loop for Huginn.

Ties together exploration, coder, workflow, benchmark, and report
into a single closed-loop ecosystem:

    Perceive → Hypothesize → Plan → Execute → Validate → Learn → Report

Usage:
    engine = AutoloopEngine(workspace=Path("."))
    asyncio.run(engine.run(objective="Optimize C-S-H defect kinetics"))
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from huginn.autoloop.budget import IterationBudget, ProgressiveBudget
from huginn.autoloop.goal_scheduler import Goal, GoalScheduler
from huginn.autoloop.phase_gate import (
    PhaseGate,
    PhaseGateHook,
    _has_external_source as _validation_has_external_source,
    get_shared_phase_gate_state,
)
from huginn.bench.runner import BenchmarkRunner
from huginn.coder.loop import CoderRunner
from huginn.config import get_settings
from huginn.exploration.orchestrator import ExplorationOrchestrator
from huginn.exploration.strategies import ParetoPruningStrategy
from huginn.interaction.progress import ProgressTracker, get_progress_tracker
from huginn.kg.builder import ProjectKnowledgeGraph
from huginn.llm import get_model
from huginn.memory.manager import MemoryManager
from huginn.api.event import EventType, WorkflowStageEvent
from huginn.tools.report_tool import ReportTool
from huginn.tools.base import HuginnTool
from huginn.types import ToolContext, ToolResult
from huginn.workflows.engine import WorkflowEngine
from huginn.workflows.templates import get_template, standard_dft_workflow

# 跨源属性冲突检测用的正则; 提到模块级避免每次调用重编译
_PROP_RE = re.compile(
    r'([\w\s]{3,25}?)\s*[:=]\s*(-?\d+\.?\d*)\s*(eV(?:/\w+)?|GPa|THz|nm)',
    re.IGNORECASE,
)


# Autoloop 7-phase pipeline — single source of truth for phase names.
# ponytail: constants, not an enum — engine phases are imperative control
# flow labels, not a declarative state machine like ResearchPhase.
# If phases diverge enough to need transitions/validation, promote to Enum.
AUTOLOOP_PHASES = (
    "perceive",
    "hypothesize",
    "plan",
    "execute",
    "validate",
    "learn",
    "report",
)

# 7 阶段 → persona 分派表. None 表示该阶段不走 LLM persona 注入
# (比如 Execute 直接调 workflow, 不需要 persona 影响输出).
# Hypothesize 用 default, 真正的 persona 在 _hypothesize 里按研究类型动态选.
_MATH_SIGNALS = (
    "equation", "lagrangian", "pde", "hamiltonian", "derivative",
    "differential", "integral", "eigenvalue", "tensor", "manifold",
    "symmetry", "conservation", "variational", "continuum",
    "stress", "strain", "energy", "phonon", "band",
)
_PHASE_PERSONAS: dict[str, str | None] = {
    "perceive": "default",
    "hypothesize": None,  # 动态选 dft_expert / md_expert, 见 _hypothesize
    "plan": "default",
    "execute": None,  # 直接调 workflow / coder, 不走 LLM persona
    "validate": "reviewer",  # 关键: 校验阶段用 reviewer persona 做批判性审视
    "learn": "default",
    "report": "tutor",  # 教学风格输出
}
assert set(_PHASE_PERSONAS.keys()) == set(AUTOLOOP_PHASES), (
    "Phase persona keys must match AUTOLOOP_PHASES"
)

# result_data 里的 key -> 文献检索时用的性质名. _literature_comparison 遍历这个表.
_LIT_PROPERTY_MAP: dict[str, str] = {
    "energy": "total energy",
    "band_gap": "band gap",
    "volume": "volume",
    "bulk_modulus": "bulk modulus",
    "magnetization": "magnetization",
    "lattice_a": "lattice constant a",
    "lattice_b": "lattice constant b",
    "lattice_c": "lattice constant c",
}


def _extract_tests_passed(validation: Any) -> bool:
    """从 validation 结果里抽 tests_passed 布尔, 给 validate→learn 门用.

    validation 形状不固定 (dict / str / None), 抽不出明确失败就默认 True,
    避免门控把现有 happy path 误阻断. 只有明确说 fail / passed=False 才拦.
    """
    if isinstance(validation, dict):
        for key in ("tests_passed", "passed", "success", "ok"):
            if key in validation:
                return bool(validation[key])
        return True
    if isinstance(validation, str):
        low = validation.lower()
        if "fail" in low:
            return False
        return True
    # None 或其它: 没有明确失败信号, 放行
    return True


@dataclass
class LoopPhase:
    """A single phase in the autonomous loop."""

    name: str
    status: str = "pending"  # pending | running | completed | failed
    start_time: float | None = None
    end_time: float | None = None
    result: Any = None
    error: str | None = None


@dataclass
class AutoloopResult:
    """Result of a full autonomous loop iteration."""

    run_id: str
    objective: str
    phases: list[LoopPhase]
    success: bool
    report_path: str | None = None
    total_time_seconds: float = 0.0
    trajectory_path: str | None = None
    goal_achieved: bool | None = None
    goal_judgment: dict[str, Any] | None = None
    # 落盘的 provenance JSONL, run 结束后可回放整条 tool chain
    provenance_path: str | None = None


def objective_hash(objective: str) -> str:
    """Stable 8-char hash for an autoloop objective — used to dedup result snapshots.

    Same objective string → same hash → same snapshot file. If two objectives
    only differ by whitespace/casing they hash differently; that's fine, we'd
    rather over-store than silently reuse the wrong run.
    """
    return hashlib.md5(objective.encode("utf-8")).hexdigest()[:8]


def _snapshot_dir(workspace: str | Path) -> Path:
    return Path(workspace) / ".huginn" / "autoloop_results"


def save_autoloop_snapshot(
    result: AutoloopResult, workspace: str | Path
) -> Path | None:
    """Persist a compact JSON snapshot of an AutoloopResult under
    ``<workspace>/.huginn/autoloop_results/<objective_hash>.json``.

    Lets other components (DeliAutoResearch, future CLI subcommands) reuse a
    finished run without re-instantiating AutoloopEngine. Returns the snapshot
    path, or None on failure — callers treat None as "no snapshot, run normally".
    """
    try:
        snap_dir = _snapshot_dir(workspace)
        snap_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "objective": result.objective,
            "success": result.success,
            "goal_achieved": result.goal_achieved,
            "goal_judgment": result.goal_judgment,
            "report_path": result.report_path,
            "provenance_path": result.provenance_path,
            "trajectory_path": result.trajectory_path,
            "total_time_seconds": result.total_time_seconds,
            "phases_count": len(result.phases),
            "phases_summary": [
                {"name": p.name, "status": p.status} for p in result.phases
            ],
            "saved_at": time.time(),
        }
        path = snap_dir / f"{objective_hash(result.objective)}.json"
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path
    except Exception:
        logger.debug("failed to save autoloop snapshot", exc_info=True)
        return None


def load_autoloop_snapshot(
    workspace: str | Path, objective: str
) -> dict[str, Any] | None:
    """Read a previously saved snapshot for this objective.

    Returns None if the snapshot is missing or unreadable — callers fall back
    to a fresh engine.run() in that case.
    """
    path = _snapshot_dir(workspace) / f"{objective_hash(objective)}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.debug("failed to load autoloop snapshot: %s", path, exc_info=True)
        return None


class AutoloopEngine:
    """Main autonomous loop engine.

    Orchestrates perception, hypothesis generation, planning, execution,
    validation, learning, and reporting into a single cohesive loop.
    """

    def __init__(
        self,
        workspace: str | Path | None = None,
        goal_scheduler: GoalScheduler | None = None,
        verification_model: Any = None,
        memory_manager: MemoryManager | None = None,
    ):
        self.workspace = Path(workspace or ".").resolve()
        self.settings = get_settings()
        self.model = get_model(self.settings)
        # Moonshine 三槽: verification 用独立 LLM 验证假设, 避免确认偏差.
        # 默认 None 时退回 self.model, 保持向后兼容.
        self.verification_model = verification_model or self.model
        # 共享 MemoryManager: 由 agent/CLI 传入, 避免引擎私有实例和 agent 的
        # memory 隔离. 默认 None 时 new 一个, 保持向后兼容.
        self.memory = memory_manager or MemoryManager()
        self.kg = ProjectKnowledgeGraph(root=self.workspace)
        # 假设图: 跟踪 hypothesis 的 support/refute/derive 关系,
        # refute 时触发 RedTeam 审查 → 修正假设入队, 形成闭环
        from huginn.autoloop.hypothesis_loop import HypothesisGraph
        self.hypothesis_graph = HypothesisGraph()
        self.report_tool = ReportTool()

        # Sub-engines
        self.explorer = ExplorationOrchestrator(
            strategy=ParetoPruningStrategy(),
            max_parallel=3,
        )
        self.workflow_engine = WorkflowEngine(
            tool_registry=None,  # Will use default tool registry
        )
        self.coder = CoderRunner()

        self._should_stop = False
        self._iteration = 0
        # 连续验证失败计数: 给 _maybe_clarify 判断是否该问用户;
        # 超过 _max_consecutive_failures 时强制停止 autoloop, 避免无限重试坏方向.
        self._consecutive_failures = 0
        self._max_consecutive_failures = 5
        # refine 循环计数: 防止 refute→refine 无限循环
        self._refine_count = 0
        self._max_refines = 8
        # pivot 计数: refine 耗尽后换方向, 但 pivot 本身也要有上限 —
        # 否则 pivot→fail→refine→pivot→fail 无限循环, 烧 token 不出结果.
        self._pivot_count = 0
        self._max_pivots = 3
        # ClarificationManager 懒加载 — autoloop 期间在关键决策点提问用户
        self._clarification_mgr = None
        # Evolution engine 懒加载——只在 _learn 真正用到时初始化
        self._evolution = None
        # PersonaManager 懒加载 — 避免实例化时就扫描 .huginn/personas 目录
        self._persona_manager = None
        # 领域知识库 (first-principles seed docs) 懒加载 — 避免实例化时拉 ChromaDB
        self._kb = None
        # PerceptionLayer 懒加载 — 长生命周期, start() 后后台线程持续积累事件
        self._perception = None
        # Plan store: 持久化 plan 到 plans.json, 跨会话可恢复. 懒加载,
        # 跟 goal_scheduler 一套, 避免实例化时碰磁盘
        self._plan_store = None
        # 进度跟踪: 默认走进程级单例, 跟 WorkflowEngine 共享, 让 /tasks
        # 路由能汇总所有引擎的进度. 测试时可注入独立 tracker 隔离.
        self.progress_tracker: ProgressTracker | None = None
        # 投机执行 hint: on_turn_start 写入, _build_*_prompt 读出注入 LLM
        self._speculator_hint: str = ""
        # 视觉基元: _validate 从 tool 输出提取, _build_*_prompt 注入 LLM.
        # 跨迭代传递 — 上轮 tool 的数值指针下轮假设/计划能用到.
        self._last_visual_context: str = ""
        # JEPA 式预测: plan 阶段 LLM 预测预期结果, validate 阶段对比实际,
        # 预测误差 = surprise = intrinsic motivation 信号.
        # ponytail: 文本空间预测, 不是真正的嵌入空间 JEPA. 但原理一致 —
        # 执行前预测, 执行后对比, 误差驱动探索. 升级路径: 训练真正的编码器+预测器.
        self._current_prediction: str = ""
        self._last_surprise: float = 0.0
        # surprise 历史: 连续低 surprise = 心智模型已收敛, 可提前终止.
        # Chemputer 启发: Jaccard 稳定 = 反应完成; 这里 = 理解完成.
        # 每条存 (worst, cross_perturbation_std): std 高 = 测量噪声大, 需更严阈值.
        self._surprise_history: list[tuple[float, float]] = []
        # 上一轮执行结果, 给 _build_plan_prompt 的 pipeline suggest_next 用
        self._last_execution_result: dict | None = None
        # 阶段门 hook: 在 plan→execute / execute→validate / validate→learn
        # 三个转移点评估证据, 不足时阻断并把 feedback 拼进 _speculator_hint
        # 让下轮 prompt 带上"缺什么证据". R3 接入 red-team reviewer_fn:
        # 在 validate→learn 做 adversarial 审查, 有 high 发现则阻断.
        from huginn.autoloop.red_team import RedTeamReviewer
        from huginn.autoloop.phase_gate import MathEvidenceChecker

        self.phase_gate_hook = PhaseGateHook(
            reviewer_fn=RedTeamReviewer(
                model=self.model,
                # 跨模型审查: verification_model 默认 fallback 到 self.model,
                # 未配置 verification 槽时退化为同模型审查, 行为不变.
                critic_model=self.verification_model,
            ),
            math_checker=MathEvidenceChecker(graph=self.hypothesis_graph),
        )
        # 元认知层: 信息隔离 / 方法族注册 / 等价性审计 / 阻塞-重启协议.
        # 全部懒加载, 测试或不需要时不会拉起. _hypothesize / refine_failed 按需读.
        # ponytail: 不在 __init__ 实例化, 避免循环 import 和测试 mock 复杂度
        self._metacog_auditor = None
        self._metacog_block_registry = None
        self._metacog_method_registry = None
        self._metacog_last_audit = None  # 最近一次等价性审计结果, 给 learn 用
        # 过早收敛检测: agent 想提前返回时过一遍 effort floor, 未达标强制继续
        self._metacog_convergence_detector = None
        # 反完成审计: 综合 effort floor + 等价性陷阱 + 不完整性自白
        self._metacog_completion_auditor = None
        # Goal scheduler: 持久化目标到 $HUGINN_CACHE_DIR/goals.json.
        # engine.run(goal=...) 时每轮 learn 后查 completion, 满足则提前停.
        # None → 懒加载, 避免实例化时就碰磁盘 (测试隔离用).
        self._goal_scheduler = goal_scheduler
        # 侧边对话 channel: 轮空时 drain 待答问题. 默认走进程级单例,
        # 跟 HTTP /side 路由共享. None 时用 get_shared_side_channel() 懒拿.
        self._side_channel = None
        # 侧边对话开关: 测试或不需要侧边对话时关掉, 避免 idle 时碰 LLM.
        self._side_channel_enabled = True
        # 事件总线: 让外部插件能在阶段开始/结束/失败时挂钩.
        # 懒加载, 避免 import 时拉起 StarHandlerRegistry.
        self._event_bus = None
        # 阶段索引: 给 WorkflowStageEvent 用, 从 phase name 推算.
        self._phase_order = list(AUTOLOOP_PHASES)

    def _get_evolution(self):
        """懒加载 EvolutionEngine, 避免实例化时就拉起日志和规则文件。"""
        if self._evolution is None:
            from huginn.evolution.engine import EvolutionEngine
            from huginn.evolution.logger import ExecutionLogger

            self._evolution = EvolutionEngine(logger=ExecutionLogger())
        return self._evolution

    def _get_perception(self):
        """懒加载 PerceptionLayer — 长生命周期, start 后持续监听文件/日志事件.

        一旦创建就在后台持续运行, _perceive() 只取 snapshot 不再 start/stop.
        析构时由 GC 或显式 stop() 回收线程.
        """
        if self._perception is None:
            try:
                from huginn.perception import PerceptionLayer
                self._perception = PerceptionLayer(self.workspace)
                self._perception.start()
            except Exception:
                return None
        return self._perception

    def _get_persona_manager(self):
        """懒加载 PersonaManager, 实例化时才扫描 persona 文件."""
        if self._persona_manager is None:
            from huginn.personas import PersonaManager

            self._persona_manager = PersonaManager(workspace=self.workspace)
        return self._persona_manager

    def _get_kb(self):
        """懒加载领域知识库. ChromaDB 或 seed 文件不可用时返回 None,
        调用方需自行判空."""
        if self._kb is None:
            try:
                from huginn.knowledge.store import get_knowledge_base

                self._kb = get_knowledge_base(str(self.workspace))
            except Exception:
                return None
        return self._kb

    def _get_event_bus(self):
        """懒加载 EventBus. 没注册 handler 时返回 None, 调用方判空跳过."""
        if self._event_bus is not None:
            return self._event_bus
        try:
            from huginn.plugins.event_bus import EventBus

            self._event_bus = EventBus()
        except Exception:
            return None
        return self._event_bus

    async def _dispatch_stage_event(
        self,
        event_type: EventType,
        stage_name: str,
        duration_sec: float = 0.0,
        error: str | None = None,
    ) -> None:
        """向 EventBus 发一个 WorkflowStageEvent. 没总线或没 handler 时静默跳过."""
        bus = self._get_event_bus()
        if bus is None:
            return
        idx = self._phase_order.index(stage_name) + 1 if stage_name in self._phase_order else 0
        event = WorkflowStageEvent(
            type=event_type,
            workflow_name="autoloop",
            stage_name=stage_name,
            stage_index=idx,
            duration_sec=duration_sec,
            error=error,
        )
        try:
            result = await bus.dispatch(event)
            if result.executed == 0 and result.failed == 0:
                logger.debug(
                    "stage event %s.%s had no handlers",
                    event_type.name, stage_name,
                )
        except Exception:
            logger.warning("error in _dispatch_stage_event: bus.dispatch failed", exc_info=True)

    def _build_kb_text(self, query: str) -> str:
        """检索领域知识库, 把命中 chunk 拼成 prompt 上下文块. KB 没装、
        空、查询失败都返回空串, 不影响 loop. 用 query_with_dedup 去重,
        避免分块重叠导致的近似重复段落浪费 token."""
        if not query:
            return ""
        kb = self._get_kb()
        if kb is None:
            return ""
        try:
            if kb.count() == 0:
                return ""
            # 优先用带去重的检索
            if hasattr(kb, "query_with_dedup"):
                chunks = kb.query_with_dedup(query, top_k=5)
            else:
                chunks = kb.query(query, top_k=5)
            if not chunks:
                return ""
            lines = []
            for i, c in enumerate(chunks, 1):
                text = (c.get("text") or "").strip()
                if not text:
                    continue
                if len(text) > 800:
                    text = text[:800] + "…"
                lines.append(f"[{i}] {text}")
            if not lines:
                return ""
            body = "\n".join(lines)
            return (
                "### Domain Knowledge Context\n"
                "The following first-principles reference chunks may ground your "
                "hypothesis and plan. Cite source numbers when relevant.\n"
                f"{body}\n"
                "### End Domain Knowledge Context"
            )
        except Exception:
            return ""

    def _build_kg_text(self, query: str) -> str:
        """检索知识图谱, 把相关实体+关系拼成 prompt 上下文块.
        KG 没建、空、查询失败都返回空串. 这是 KG 读回闭环的关键 —
        _learn 写入的实体, _hypothesize/_plan 要能检索到."""
        if not query:
            return ""
        kg = getattr(self, "kg", None)
        if kg is None:
            return ""
        try:
            result = kg.query(query, depth=1, top_k=8)
            nodes = result.get("nodes") or []
            if not nodes:
                return ""
            lines = []
            for node in nodes[:8]:
                data = node.get("data", node)
                label = data.get("label", node.get("id", ""))
                etype = data.get("type", "")
                conf = data.get("confidence", 0)
                lines.append(f"- [{etype}] {label} (conf={conf:.2f})")
                # 把出边也带上
                for edge in (node.get("edges") or [])[:3]:
                    rel = edge.get("relation", "→")
                    dst = edge.get("dst_label", edge.get("dst", ""))
                    lines.append(f"  {rel} → {dst}")
            if not lines:
                return ""
            body = "\n".join(lines)
            # KG 缺口检测: 找 A-B 有边、B-C 有边、但 A-C 无边的三元组
            # 建议假设 "A 是否也和 C 有关?" — 这是 KG 主动驱动探索的关键.
            gap_hints = self._detect_kg_gaps(kg, nodes)
            gap_block = ""
            if gap_hints:
                gap_block = "\n\n### KG Gap Detection (potential research directions)\n" + "\n".join(gap_hints) + "\n"
            return (
                "### Knowledge Graph Context\n"
                "Previously discovered entities and relations from prior runs:\n"
                f"{body}\n"
                "### End Knowledge Graph Context"
                f"{gap_block}"
            )
        except Exception:
            return ""

    def _detect_kg_gaps(self, kg: Any, nodes: list[dict]) -> list[str]:
        """检测 KG 中的知识缺口: A-B 有边, B-C 有边, 但 A-C 无边.
        返回 "Consider whether {A} also relates to {C}" 格式的提示.
        用 NetworkX 的 common_neighbors, 零依赖."""
        try:
            graph = getattr(kg, "_graph", None)
            if graph is None or graph.number_of_nodes() < 3:
                return []
            import networkx as nx
            hints: list[str] = []
            # 只检查高置信度节点 (conf > 0.5)
            high_conf_nodes = []
            for nid, data in graph.nodes(data=True):
                if data.get("confidence", 0) > 0.5:
                    high_conf_nodes.append(nid)
            # 对每对高置信度节点, 检查是否有共同邻居但彼此无边
            checked = 0
            for i, a in enumerate(high_conf_nodes[:10]):
                for b in high_conf_nodes[i+1:10]:
                    if graph.has_edge(a, b) or graph.has_edge(b, a):
                        continue  # 已有边, 不是缺口
                    common = set(nx.common_neighbors(graph, a, b)) if graph.has_node(a) and graph.has_node(b) else set()
                    if common:
                        a_label = graph.nodes[a].get("label", a)[:40]
                        b_label = graph.nodes[b].get("label", b)[:40]
                        hints.append(f"- {a_label} and {b_label} share connections but no direct link — consider whether they relate")
                        checked += 1
                    if checked >= 3:
                        break
                if checked >= 3:
                    break
            return hints
        except Exception:
            return []

    def _build_memory_text(self, query: str) -> str:
        """检索长期记忆, 把跨会话的教训/发现拼成 prompt 上下文块.
        Memory 之前只写不读 — _learn 写入的迭代记录和失败教训,
        下轮 hypothesize/plan 完全看不到. 这个函数闭合了 memory 读回环.
        查询失败/空结果返回空串, 不影响 prompt."""
        if not query:
            return ""
        mem = getattr(self, "memory", None)
        if mem is None:
            return ""
        try:
            return mem.recall_for_prompt(query, max_entries=3)
        except Exception:
            return ""

    # 上下文预算: 防止 prompt block 累积超过 token 上限.
    # 优先级: body > math > kg > visual > kb > mem > hint > skill > composite > pipeline
    # 超预算时不是直接丢弃, 而是分层压缩: 先截断 → 再摘要 → 最后才删.
    # 视觉语言比文字语言更能压缩信息 — 一行 "[energies] peak=idx3, trend=↑"
    # 传达的信息等于 200 chars 的 JSON. 用压缩替代丢弃, 保留信息密度.
    _PROMPT_BUDGET = 12000  # chars, 约 3K tokens

    @staticmethod
    def _compress_block(name: str, text: str, level: int) -> str:
        """分层压缩: level 0=原样, 1=截断, 2=一行摘要, 3=删除."""
        if not text or level <= 0:
            return text if level <= 0 else ""
        if level == 1:
            # 截断到 300 字符, 保留开头
            if len(text) <= 300:
                return text
            return text[:300] + "..."
        if level == 2:
            # 压缩成一行摘要: 取关键信息
            lines = text.strip().split("\n")
            # KB/KG/mem: 只保留前 2 行 + "..."
            if len(lines) <= 2:
                return lines[0][:100] if lines else ""
            return lines[0][:100] + " | " + lines[1][:100] + " | ..."
        return ""

    def _scan_block_conflicts(self, blocks: list[tuple[str, str]]) -> str:
        """Lightweight cross-source conflict detection: same property, different values.

        Scans block text for 'property = value unit' patterns. When the same
        property appears in multiple blocks with different numeric values,
        returns a warning string. Uses regex only, no LLM calls.
        """
        # ponytail: 属性名前缀不一致 (如 "band gap" vs "the band gap") 会导致漏检,
        # 但对 <10 blocks 的 prompt 场景足够; 如需精确匹配可改用 NER 提取属性名
        prop_values: dict[str, dict[str, str]] = {}
        for name, text in blocks:
            if not text:
                continue
            for m in _PROP_RE.finditer(text):
                prop = m.group(1).strip().lower()
                val = m.group(2)
                unit = m.group(3).strip()
                key = f"{prop} ({unit})"
                prop_values.setdefault(key, {})
                if val not in prop_values[key]:
                    prop_values[key][val] = name
        conflicts = []
        for key, vals in prop_values.items():
            if len(vals) > 1:
                sources = ", ".join(f"{v} in [{s}]" for v, s in vals.items())
                conflicts.append(f"{key}: {sources}")
        if not conflicts:
            return ""
        return (
            "Cross-source conflicts detected (same property, different values):\n"
            + "\n".join(f"  - {c}" for c in conflicts[:5])
            + "\nVerify which value is correct before proceeding.\n"
        )

    def _trim_to_budget(self, blocks: list[tuple[str, str]]) -> str:
        """按优先级拼接 blocks, 超预算时分层压缩: 截断→摘要→删除."""
        # 跨源冲突检测: 扫描各 block 中的 property=value 对, 标注矛盾
        conflict_warn = self._scan_block_conflicts(blocks)
        if conflict_warn:
            blocks = [("conflict", conflict_warn)] + blocks

        kept = [(n, v) for n, v in blocks]
        total = sum(len(v) for _, v in kept)
        if total <= self._PROMPT_BUDGET:
            return "".join(v for _, v in kept)

        # Pass 1: 截断低优先级 block 到 300 字符
        for i in range(len(kept) - 1, -1, -1):
            if total <= self._PROMPT_BUDGET:
                break
            name, text = kept[i]
            if name == "body":  # body 永远不压缩
                continue
            compressed = self._compress_block(name, text, 1)
            total -= len(text) - len(compressed)
            kept[i] = (name, compressed)

        if total <= self._PROMPT_BUDGET:
            return "".join(v for _, v in kept)

        # Pass 2: 压缩成一行摘要
        for i in range(len(kept) - 1, -1, -1):
            if total <= self._PROMPT_BUDGET:
                break
            name, text = kept[i]
            if name == "body":
                continue
            compressed = self._compress_block(name, text, 2)
            total -= len(text) - len(compressed)
            kept[i] = (name, compressed)

        if total <= self._PROMPT_BUDGET:
            return "".join(v for _, v in kept)

        # Pass 3: 从最低优先级开始删除
        for i in range(len(kept) - 1, -1, -1):
            if total <= self._PROMPT_BUDGET:
                break
            name, text = kept[i]
            if name == "body":
                continue
            total -= len(text)
            kept[i] = (name, "")

        return "".join(v for _, v in kept)

    def _persona_system_prompt(self, persona_name: str | None) -> str:
        """取 persona 的 system prompt, 按层组装.

        层级 (SillyTavern 角色卡分层启发):
        1. permanent_core (或 system_prompt 向后兼容) — 身份/角色/安全约束
        2. adaptive_layer — 会话级风格/偏好 (由 StyleLearner/TasteProfile 填充)
        """
        if not persona_name:
            return ""
        try:
            persona = self._get_persona_manager().get(persona_name)
        except Exception:
            return ""
        # 优先用 permanent_core, 没设就退回 system_prompt (老 persona)
        core = persona.permanent_core or persona.system_prompt or ""
        adaptive = persona.adaptive_layer or ""
        if adaptive:
            return f"{core}\n\n--- Adaptive ---\n{adaptive}"
        return core

    # _phase_persona removed — per-call persona_name= in each phase method
    # is the active injection path. _PHASE_PERSONAS stays as documentation.

    # ──────────────────────────────────────────────────────────────
    # Phase-gate
    # ──────────────────────────────────────────────────────────────

    def _check_gate(
        self, from_phase: str, to_phase: str, evidence: dict[str, Any]
    ) -> bool:
        """评估阶段转移门. 通过/已 override 返回 True; 阻断时把 feedback
        拼进 _speculator_hint (下轮 prompt 用) 并返回 False, caller 应
        continue 到下一轮迭代, 不推进到 to_phase.

        共享状态写一条记录进 history, 让 phase_tool 能查到最新门决策.
        """
        state = get_shared_phase_gate_state()
        # override 优先: 已强制放行的转移直接记一条 approved, 不再评估
        if (from_phase, to_phase) in state.overrides:
            meta = state.override_meta.get((from_phase, to_phase), {})
            state.history.append(
                PhaseGate(
                    from_phase=from_phase,
                    to_phase=to_phase,
                    status="approved",
                    required_evidence=self.phase_gate_hook.config.required_for(
                        from_phase, to_phase
                    ),
                    feedback="override 放行",
                    reviewer=meta.get("actor", "user"),
                )
            )
            state.pending_transition = (from_phase, to_phase)
            # override 同时清除 pending_human_review (用户已决策)
            if state.pending_human_review == (from_phase, to_phase):
                state.pending_human_review = None
            return True

        gate = self.phase_gate_hook.evaluate(from_phase, to_phase, evidence)
        state.history.append(gate)
        state.pending_transition = (from_phase, to_phase)

        if gate.is_blocked:
            fb = gate.feedback or (
                f"阶段转移 {from_phase}→{to_phase} 被阻断: 缺 {gate.missing_evidence}"
            )
            self._speculator_hint = (
                (self._speculator_hint + "\n" + fb).strip() if self._speculator_hint else fb
            )
            logger.info(
                "gate blocked %s→%s: missing %s",
                from_phase, to_phase, gate.missing_evidence,
            )
            return False

        # ── Human-in-the-loop checkpoint (LangGraph interrupt_before 模式) ──
        # 硬性证据已通过, 但用户配置了该转移需要人工审查. 设 pending_human_review
        # 并返回 False 让 engine 停在当前 phase. UI 层读到 phase_checkpoint 事件后
        # 展示 evidence 给用户, 用户通过 phase_tool override 或 submit_evidence + resume.
        if state.needs_human_checkpoint(from_phase, to_phase):
            state.pending_human_review = (from_phase, to_phase)
            logger.info(
                "human checkpoint pending %s→%s: awaiting user review",
                from_phase, to_phase,
            )
            # 记一条 pending 状态, phase_tool 查得到
            state.history.append(PhaseGate(
                from_phase=from_phase,
                to_phase=to_phase,
                status="pending",
                required_evidence=self.phase_gate_hook.config.required_for(
                    from_phase, to_phase
                ),
                feedback="等待人工 checkpoint 审查. 用 phase_tool override 放行, "
                         "或 submit_evidence 补充后 resume.",
                reviewer="human_checkpoint",
            ))
            return False

        # 用户已审查完毕 (pending_human_review 被清除), 正常放行
        if state.pending_human_review == (from_phase, to_phase):
            state.pending_human_review = None
        return True

    # ──────────────────────────────────────────────────────────────
    # Progressive budget
    # ──────────────────────────────────────────────────────────────

    def _check_budget(self, iteration: int, plan: dict[str, Any]) -> bool:
        """检查 plan 的 mode 是否在当前迭代预算允许范围内.

        通过返回 True (含 budget 未启用 / 已降级放行 / mode 允许三种情况).
        不通过时把"用哪个 mode 代替"的提示拼进 _speculator_hint, 下轮 prompt
        能看到, 返回 False 让 caller continue 到下一轮迭代.

        每个档位有 max_calls 次拒绝额度, 用尽后整条预算降级为放行, 避免
        LLM 反复提同样的 mode 把循环卡死.
        """
        if self._budget is None or self._budget_degraded:
            return True
        tier = self._budget.for_iteration(iteration)
        mode = plan.get("mode")
        if tier.allows(mode):
            # 这轮通过了就清掉该档位的拒绝计数, 下次重新数
            self._budget_rejects.pop(tier.label, None)
            return True

        rejects = self._budget_rejects.get(tier.label, 0) + 1
        self._budget_rejects[tier.label] = rejects
        if tier.max_calls is not None and rejects > tier.max_calls:
            # 拒绝额度用尽, 降级放行剩下的所有 mode, 不再卡
            self._budget_degraded = True
            logger.info(
                "budget degraded at iter %d: %s reject cap %s hit, allowing all modes",
                iteration, tier.label, tier.max_calls,
            )
            return True

        allowed = ", ".join(tier.allowed_modes) if tier.allowed_modes else "any"
        fb = (
            f"迭代 {iteration} 预算档位 {tier.label}: mode={mode} 不被允许, "
            f"可用: {allowed}. 请改用允许的 mode 重新规划."
        )
        self._speculator_hint = (
            (self._speculator_hint + "\n" + fb).strip() if self._speculator_hint else fb
        )
        logger.info(
            "budget rejected mode=%s at iter %d (tier %s, reject %d/%s)",
            mode, iteration, tier.label, rejects, tier.max_calls,
        )
        return False

    async def _drain_side_questions(self) -> int:
        """轮空时把 pending 侧边问题答掉. 返回答了几个.

        拿 shared SideChannel 的 pending 快照, 逐条调 model.ainvoke 出答案,
        再 channel.respond() 写回. 单条失败不阻塞其他条, 也不抛异常 ——
        侧边对话是次要任务, 不能影响主 loop.
        """
        if not self._side_channel_enabled:
            return 0
        from huginn.side_conversation import get_shared_side_channel

        channel = self._side_channel or get_shared_side_channel()
        pending = channel.drain()
        if not pending:
            return 0
        from langchain_core.messages import HumanMessage, SystemMessage

        # Side questions are low-priority — use a cheap model when available
        side_model = self.model
        router = getattr(self, 'model_router', None) or getattr(getattr(self, 'agent', None), 'model_router', None)
        if router is not None:
            try:
                side_model = router.select("cheap", prefer_cheap=True) or self.model
            except Exception:
                pass

        answered = 0
        for sq in pending:
            try:
                messages = [
                    SystemMessage(
                        content=(
                            "You are answering a side question while the main "
                            "research loop is idle. Keep it concise and direct."
                        )
                    ),
                    HumanMessage(content=sq.question),
                ]
                response = await side_model.ainvoke(messages)
                answer = str(response.content).strip()
                if answer:
                    channel.respond(sq.id, answer)
                    answered += 1
                    logger.info("side answered %s: %s", sq.id, answer[:80])
            except Exception as exc:
                # 单条失败不影响其他, 也不影响主 loop
                logger.warning("side failed to answer %s", sq.id, exc_info=True)
        return answered

    def _get_clarification_manager(self):
        """懒加载 ClarificationManager. 不可用时返回 None, 调用方判空跳过."""
        if self._clarification_mgr is not None:
            return self._clarification_mgr
        try:
            from huginn.interaction.clarification import get_clarification_manager
            self._clarification_mgr = get_clarification_manager()
        except Exception:
            return None
        return self._clarification_mgr

    def _get_plan_store(self):
        """懒加载 PlanStore. 不可用时返回 None, 调用方判空走老的纯 dict 路径."""
        if self._plan_store is not None:
            return self._plan_store
        try:
            from huginn.autoloop.plan_store import PlanStore
            self._plan_store = PlanStore()
        except Exception:
            return None
        return self._plan_store

    def _get_refine_model(self):
        """获取 refine 用的 LLM model, 优先用验证模型 (便宜档).

        没有就用 None, hypothesis_graph.refine_failed 会走 findings 模板拼接.
        """
        # ponytail: 直接用已有的 verification_model, 不需要额外的 llm_config 模块
        return getattr(self, "verification_model", None) or None

    async def _maybe_clarify(
        self,
        checkpoint: str,
        phase_result: Any,
        thread_id: str = "autoloop",
    ) -> str | None:
        """在关键决策点检查是否需要向用户提问.

        checkpoint 取值:
        - "plan": 计划生成后, 高成本 mode (workflow/DFT) 时确认
        - "validation_fail": 验证失败后, 连续 3+ 次时问方向

        返回用户回答的字符串, 或 None (无需提问 / manager 不可用 / 超时走默认).

        非阻塞设计: 没有 async event loop 时直接返回 None, 不强制阻塞.
        autoloop 在 async 上下文里跑, 所以正常路径能拿到回答.
        """
        mgr = self._get_clarification_manager()
        if mgr is None:
            return None

        # 构建上下文
        if checkpoint == "plan":
            plan = phase_result or {}
            mode = plan.get("mode", "")
            # 只对高成本 mode 提问 (workflow=DFT/MD, 通常是几小时)
            expensive_modes = ("workflow", "dft", "md", "vasp", "lammps")
            if mode.lower() not in expensive_modes:
                return None

            ctx = {
                "thread_id": thread_id,
                "question_type": "cost_confirm",
                "phase": "plan",
                "summary": f"mode={mode}, desc={plan.get('description', '')[:200]}",
                "tool": mode,
                "cost_estimate_hours": 1.0,  # workflow 类至少 1h
            }
        elif checkpoint == "validation_fail":
            if self._consecutive_failures < 3:
                return None
            ctx = {
                "thread_id": thread_id,
                "question_type": "validation_fail",
                "phase": "validate",
                "summary": str(phase_result)[:300],
                "consecutive_failures": self._consecutive_failures,
            }
        else:
            return None

        if not mgr.should_ask_contextual(ctx.get("question_type", ""), ctx):
            return None

        # 生成提问
        question, options, default = mgr.generate_question(ctx, model=None)

        try:
            answer = await mgr.ask(
                thread_id=thread_id,
                question=question,
                options=options,
                context=ctx.get("summary", ""),
                default_answer=default,
                timeout=5,  # short timeout: if no human watching, proceed with default
                metadata={
                    "question_type": ctx.get("question_type", ""),
                    "checkpoint": checkpoint,
                    "iteration": self._iteration,
                },
            )
            logger.info("clarify %s: %s", checkpoint, answer[:80])
            return answer
        except Exception as exc:
            logger.warning("clarify %s failed", checkpoint, exc_info=True)
            return None

    # ──────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────

    async def run(
        self,
        objective: str,
        max_iterations: int = 20,
        progressive_budget: bool = True,
        goal: Goal | None = None,
        max_refines: int = 8,
    ) -> AutoloopResult:
        """Run the full autonomous loop for the given objective.

        max_iterations 默认 20 (W2 R1 从 5 提到 20), 给阶段门重试和 agentic
        search / goal scheduling 留迭代额度. progressive_budget=True 时按
        迭代数收紧允许的 plan mode (见 ProgressiveBudget.default), False
        则全程放行, 行为跟提预算之前一致.

        max_refines 控制 refute→refine 循环次数上限, 默认 8.
        超过后不再生成修正假设, 避免在错误方向上反复迭代.

        goal 不为空时, 每轮 learn 后用 GoalScheduler.check_completion 查
        success_criteria 是否满足, 满足则提前停循环并在 scheduler 里标记
        completed. 没传 goal 行为不变.
        """
        self._max_refines = max_refines
        self._refine_count = 0
        # 记下本轮上限给 effort floor 用 (check 方法在 self 上, 拿不到局部)
        self._max_iterations = max_iterations
        run_id, provenance_record, run_collector = self._prepare_run(
            objective, progressive_budget, goal
        )
        tracker = get_progress_tracker()
        total_steps = max_iterations * 6 + 1
        progress_task_id = f"autoloop:{run_id}"
        tracker.start_task(
            task_id=progress_task_id,
            description=f"autoloop: {objective[:80]}",
            total_steps=total_steps,
            stage_labels=list(AUTOLOOP_PHASES),
            engine_kind="autoloop",
            metadata={"run_id": run_id, "objective": objective[:200]},
        )
        completed_steps = 0
        phases: list[LoopPhase] = []
        # 记下 progress_task_id 供 _emit_campaign 关联 SSE 流
        self._progress_task_id = progress_task_id

        while self._iteration < max_iterations and not self._should_stop:
            self._iteration += 1
            # pivot 上限: 连续换方向 _max_pivots 次还没跑通, 说明 objective 本身
            # 有问题, 继续烧 token 没意义, 让循环自然退出.
            if self._pivot_count >= self._max_pivots:
                logger.warning(
                    "autoloop stopping: pivot budget exhausted (%d/%d)",
                    self._pivot_count, self._max_pivots,
                )
                break
            logger.info("autoloop iteration %d/%d: %s", self._iteration, max_iterations, objective)

            # goal persistence: increment iteration count on active goal
            try:
                from huginn.autoloop.goal_store import get_goal_store
                _gs = get_goal_store()
                _active_goal = _gs.get_active()
                if _active_goal:
                    _gs.increment_iteration(_active_goal.id)
            except Exception:
                logger.debug("goal_store.increment_iteration failed", exc_info=True)

            # 发布 campaign.iteration 事件
            self._emit_campaign("campaign.iteration", {
                "iteration": self._iteration,
                "max": max_iterations,
                "objective": objective[:200],
            })

            # truncate speculator hint to prevent unbounded growth across iterations
            # ponytail: keep last 2000 chars, earlier feedback is stale anyway
            if len(self._speculator_hint) > 2000:
                self._speculator_hint = self._speculator_hint[-2000:]

            # 1. Perceive — _perceive 里的 _perceive_legacy 会跑 git subprocess + rglob,
            # 阻塞事件循环 5-15s, 用 to_thread 丢到线程池
            phase = await self._run_phase_async(
                "perceive", lambda: asyncio.to_thread(self._perceive)
            )
            phases.append(phase)
            completed_steps += 1
            tracker.update(progress_task_id, current_step=completed_steps,
                           current_label=f"iter {self._iteration}: perceive ({phase.status})")
            if not phase.result:
                logger.info("no changes detected, waiting...")
                # 轮空时 drain 侧边对话: 有 pending 问题就顺手答掉, 不白等.
                await self._drain_side_questions()
                await asyncio.sleep(0.5)  # reduced from 2s for faster response
                continue

            context = phase.result

            # 1b. Blind spot pass — 在 hypothesize 之前扫描盲区
            # 借鉴 "Finding Your Unknowns" 文章: 事前发现 unknown unknowns
            # 只在第 1 轮和每隔 5 轮做 (不是每轮都需要, 避免 token 浪费)
            if self._iteration == 1 or self._iteration % 5 == 0:
                try:
                    blind_spots = await self._blind_spot_pass(context, self._objective)
                    if blind_spots:
                        context["blind_spots"] = blind_spots
                        logger.info("blind spot pass: %d potential unknowns found", len(blind_spots))
                except Exception:
                    logger.warning("blind spot pass failed", exc_info=True)

            # 2. Hypothesize
            phase = await self._run_phase_async("hypothesize", self._hypothesize, context)
            phases.append(phase)
            completed_steps += 1
            tracker.update(progress_task_id, current_step=completed_steps,
                           current_label=f"iter {self._iteration}: hypothesize ({phase.status})")
            hypothesis = phase.result
            if not hypothesis:
                # propagate error to speculator hint so next iteration knows why
                if phase.error:
                    self._speculator_hint += f"\n[failed: hypothesize] {phase.error}\n"
                logger.info("no hypothesis generated (%s), skipping", phase.error or "unknown")
                continue
            logger.info("hypothesis: %s", hypothesis)
            # 发布 campaign.hypothesis 事件
            self._emit_campaign("campaign.hypothesis", {
                "iteration": self._iteration,
                "hypothesis": str(hypothesis)[:300],
            })
            # 把假设记进 hypothesis graph, 方便后续 support/refute 追踪
            _current_hyp_id = None
            try:
                _current_hyp_id = self.hypothesis_graph.add_hypothesis(
                    statement=hypothesis,
                    rationale=context.get("summary", ""),
                )
            except Exception:
                logger.warning("error in run: hypothesis_graph.add_hypothesis failed", exc_info=True)
            # A: LUCID 必要条件闭环 — 把 LLM 自检的 necessary condition 加成派生节点
            self._attach_lucid_prereqs(_current_hyp_id)
            # B: 记当前假设 id 供 _plan_context_hint / _override_plan_mode 路由
            self._current_hyp_id_for_plan = _current_hyp_id

            # 拓扑维护: 检测搜索空间坍缩, 给下轮拼重定向 hint (advisory)
            self._metacog_check_topology_collapse()

            # 3. Plan
            phase = await self._run_phase_async("plan", self._plan, hypothesis, context)
            phases.append(phase)
            completed_steps += 1
            tracker.update(progress_task_id, current_step=completed_steps,
                           current_label=f"iter {self._iteration}: plan ({phase.status})")
            plan = phase.result
            if not plan:
                if phase.error:
                    self._speculator_hint += f"\n[failed: plan] {phase.error}\n"
                logger.info("no plan generated (%s), skipping", phase.error or "unknown")
                continue
            logger.info("plan: %s | %s", plan['mode'], plan['description'])

            # 高成本 plan 时问用户确认. 有 plan_id 说明 _plan 里已经走过
            # PlanStore 确认门了, 别重复问; 没 plan_id (PlanStore 不可用)
            # 才走老的 fire-and-forget 提问
            if not plan.get("plan_id"):
                await self._maybe_clarify("plan", plan)

            # 预算: 后期迭代限制昂贵 mode. 拒绝时把可用 mode 写进 hint,
            # continue 到下一轮让 LLM 改提 plan. 降级后全程放行.
            if not self._check_budget(self._iteration, plan):
                continue

            # gate: plan→execute — 必须有 mode + description 才放行
            if not self._check_gate(
                "plan", "execute",
                {"mode": plan.get("mode"), "description": plan.get("description")},
            ):
                continue

            # 4. Execute
            # JEPA: stash plan's prediction for validate to compare against actual
            self._current_prediction = plan.get("expected_prediction", "")
            phase = await self._run_phase_async("execute", self._execute, plan, context)
            phases.append(phase)
            completed_steps += 1
            tracker.update(progress_task_id, current_step=completed_steps,
                           current_label=f"iter {self._iteration}: execute ({phase.status})")
            execution_result = phase.result
            if phase.error:
                self._speculator_hint += f"\n[failed: execute] {phase.error}\n"
                logger.info("execution failed: %s", phase.error)
                continue
            logger.info("execution complete: %s", execution_result)

            # Git commit after execute — EurekAgent artifact engineering:
            # 每轮 execute 后提交, 让下轮 perceive 能 git diff 看到本轮变更,
            # 而不是看到从 run 开始累积的全部 diff.
            # subprocess.run / time.sleep 都是阻塞的, 在 async run() 里直接调
            # 会卡住事件循环 — 用 asyncio.to_thread 把整段丢到线程池.
            def _git_commit_after_execute():
                try:
                    import subprocess as _sp
                    import time as _time
                    _sp.run(["git", "add", "-A"], cwd=self.workspace,
                            capture_output=True, timeout=10)
                    _msg = f"[iter {self._iteration}] {plan.get('mode','?')}: {plan.get('description','')[:80]}"
                    for _attempt in range(3):
                        _r = _sp.run(["git", "commit", "-m", _msg], cwd=self.workspace,
                                     capture_output=True, timeout=10)
                        if _r.returncode == 0:
                            break
                        # index.lock 冲突等瞬时错误, 退避后重试
                        if _attempt < 2:
                            _time.sleep(1 * (_attempt + 1))
                except Exception:
                    pass  # no git repo or git unavailable — not our problem
            await asyncio.to_thread(_git_commit_after_execute)

            # Deviation log: 执行结果与 plan 预期不符时记录
            # 借鉴 "Finding Your Unknowns" 的 implementation-notes 技术
            self._log_deviation(plan, execution_result, context)

            # plan 跑完了, 标记 PlanStore 里的 plan 为 completed
            _plan_id = plan.get("plan_id") if isinstance(plan, dict) else None
            if _plan_id:
                try:
                    store = self._get_plan_store()
                    if store is not None:
                        store.complete_plan(_plan_id)
                except Exception:
                    logger.warning("error in run: store.complete_plan failed", exc_info=True)

            # gate: execute→validate — 必须有 mode (执行模式) 才放行
            if not self._check_gate(
                "execute", "validate",
                {"mode": plan.get("mode")},
            ):
                continue

            # 5. Validate
            phase = await self._run_phase_async("validate", self._validate, execution_result)
            phases.append(phase)
            completed_steps += 1
            tracker.update(progress_task_id, current_step=completed_steps,
                           current_label=f"iter {self._iteration}: validate ({phase.status})")
            validation = phase.result
            logger.info("validation: %s", validation)

            # 更新假设图: tests_passed → support, 否则 → refute → refine
            try:
                tests_passed = validation.get("tests_passed", False)
                if _current_hyp_id is not None:
                    if tests_passed:
                        # 循环A: 演绎路径 — tests_passed 即演绎证据, 标记 modality
                        # data_source: 标记数据来源, 供 dual_covered 检查独立性 (防 IPI)
                        self.hypothesis_graph.support(
                            _current_hyp_id,
                            evidence={
                                **validation,
                                "modality": "deductive",
                                "data_source": "symbolic_tool",
                            },
                        )
                        # 循环B: 割边节点触发 GP 数值验证 (跨模态独立路径)
                        # ponytail: GP 与符号演绎基底正交, 但同模型权重
                        # 是软独立. 跨模型/跨模态是升级路径.
                        # data_source 不同 → dual_covered 检查来源独立性
                        if self.hypothesis_graph.needs_dual_coverage(_current_hyp_id):
                            try:
                                gp_verdict = self._verify_via_gp(_current_hyp_id, validation)
                                if gp_verdict.get("agrees"):
                                    self.hypothesis_graph.support(
                                        _current_hyp_id,
                                        evidence={
                                            "modality": "numeric",
                                            "data_source": "gp_tool",
                                            "gp_fit": gp_verdict,
                                        },
                                    )
                            except Exception:
                                logger.debug(
                                    "GP verification failed for %s",
                                    _current_hyp_id, exc_info=True,
                                )
                    else:
                        # C: 失败类型区分 — 工具失败不 refute, 直接下轮重试
                        # RedTeam high severity findings 参与 classification
                        failure_type = self._classify_failure(
                            validation, redteam_cats=self._redteam_findings()
                        )
                        if failure_type == "tool_error":
                            logger.info(
                                "tool_error for %s, skipping refute (will retry)",
                                _current_hyp_id,
                            )
                            self._emit_campaign("campaign.retry", {
                                "iteration": self._iteration,
                                "hypothesis": str(_current_hyp_id),
                                "reason": "tool_error",
                            })
                        else:
                            if failure_type == "prompt_injection_suspect":
                                # ARGUS: 失败 + external_content 来源, 证据可能被注入.
                                # 走 refute 但标记 source_class, refine 时换路避开污染源.
                                logger.warning(
                                    "prompt_injection_suspect for %s: failure + "
                                    "external_content source, evidence may be tainted",
                                    _current_hyp_id,
                                )
                                self._emit_campaign("campaign.suspect", {
                                    "iteration": self._iteration,
                                    "hypothesis": str(_current_hyp_id),
                                    "reason": "prompt_injection_suspect",
                                })
                            self.hypothesis_graph.refute(
                                _current_hyp_id,
                                evidence={"errors": validation.get("errors", "tests failed"),
                                          "failure_type": failure_type},
                            )
                            # refine 闭环: refute 后生成修正假设, 下轮迭代处理
                            if self._refine_count < self._max_refines:
                                try:
                                    # 元认知: 把 block_registry + method_family 传入,
                                    # 让 refine 走阻塞-重启协议, 防止换名重启死路线.
                                    # method_family 从最近一次假设归类拿, 没有就 None (不阻塞).
                                    _metacog_family = None
                                    if self._last_hypothesis:
                                        _metacog_family = self._metacog_classify_family(
                                            self._last_hypothesis
                                        )
                                    new_hyp = self.hypothesis_graph.refine_failed(
                                        _current_hyp_id,
                                        evidence={"errors": validation.get("errors", "tests failed"),
                                                  "failure_type": failure_type},
                                        model=self._get_refine_model(),
                                        block_registry=self._get_metacog_block_registry(),
                                        method_family=_metacog_family,
                                    )
                                    self._refine_count += 1
                                    logger.info(
                                        "refine %d/%d (%s): %s → %s",
                                        self._refine_count, self._max_refines,
                                        failure_type,
                                        _current_hyp_id, new_hyp,
                                    )
                                    # 发布 campaign.refine 事件
                                    self._emit_campaign("campaign.refine", {
                                        "iteration": self._iteration,
                                        "refine_count": self._refine_count,
                                        "max_refines": self._max_refines,
                                        "failure_type": failure_type,
                                        "old_hypothesis": str(_current_hyp_id),
                                        "new_hypothesis": str(new_hyp)[:300] if new_hyp else "",
                                    })
                                except Exception:
                                    logger.warning("refine_failed failed", exc_info=True)
                            else:
                                # refine 次数耗尽 → 战略 pivot, 不再修参数, 换方向
                                try:
                                    _obj = self._objective if hasattr(self, "_objective") else ""
                                    new_hyp = self.hypothesis_graph.pivot(
                                        _current_hyp_id,
                                        evidence={"errors": validation.get("errors", "tests failed"),
                                                  "failure_type": failure_type},
                                        model=self._get_refine_model(),
                                        objective=_obj,
                                    )
                                    self._refine_count = 0  # reset: 新方向有新的 refine 预算
                                    self._pivot_count += 1
                                    logger.info(
                                        "PIVOT (%s): %s → %s (refine budget reset)",
                                        failure_type, _current_hyp_id, new_hyp,
                                    )
                                    self._emit_campaign("campaign.refine", {
                                        "iteration": self._iteration,
                                        "pivot": True,
                                        "failure_type": failure_type,
                                        "old_hypothesis": str(_current_hyp_id),
                                        "new_hypothesis": str(new_hyp)[:300] if new_hyp else "",
                                        "reason": "max_refines_reached",
                                    })
                                except Exception:
                                    logger.warning(
                                        "pivot failed for %s, giving up on this hypothesis",
                                        _current_hyp_id, exc_info=True,
                                    )
            except Exception:
                logger.warning("error in run: hypothesis_graph support/refute update failed", exc_info=True)

            # 连续失败计数: 通过则清零, 不通过则累加并在阈值时问用户
            # validate 阶段抛异常时 phase.result 是 None, _extract_tests_passed
            # 默认放行 True 会把"validate 挂了"误判成"测试通过"并清零失败计数,
            # 所以先看 phase.status — failed 直接视为没通过.
            if phase.status == "failed":
                _tests_ok = False
                validation = {"tests_passed": False, "error": phase.error or "validate phase failed"}
            else:
                _tests_ok = _extract_tests_passed(validation)
            if _tests_ok:
                self._consecutive_failures = 0
            else:
                self._consecutive_failures += 1
                # 连续 3+ 次失败, 问用户方向 (非阻塞, 超时走默认继续)
                await self._maybe_clarify("validation_fail", validation)
                # 连续失败超上限: objective 可能在死循环, 强制停.
                # _should_stop 让 while 条件自然退出, 不抛异常, 调用方拿到正常返回.
                if self._consecutive_failures >= self._max_consecutive_failures:
                    logger.warning(
                        "autoloop stopping: %d consecutive validation failures (max %d)",
                        self._consecutive_failures, self._max_consecutive_failures,
                    )
                    self._should_stop = True

            # gate: validate→learn — 要有 tests_passed 证据才放行.
            # tests 没过 = 没有"测试通过"的证据, 传空 dict 让门阻断,
            # 而不是传 tests_passed=False (hook 只查 key 存在性, False 会被当成有值)
            _gate_evidence: dict[str, Any] = (
                {"tests_passed": True} if _tests_ok else {}
            )
            # 透传数学证据 key (由 _validate 从 execution_result 收集):
            # conservation_law / dimensional_consistent / pde_classification /
            # sobol_top_features / constraint_check. math_checker 用 Dempster-
            # Shafer 合成这些 source, belief(pass) <= threshold 时阻断.
            for _mk in (
                "conservation_law",
                "dimensional_consistent",
                "pde_classification",
                "sobol_top_features",
                "constraint_check",
            ):
                if _mk in validation:
                    _gate_evidence[_mk] = validation[_mk]
            # 物理 oracle 透传: simulator tool 把 PhysicsAuditor.audit().to_dict()
            # 填到 execution_result["physics_audit"], 这里抽出来给 PhaseGate 做 oracle 否决.
            # ponytail: 只透传不解析. 升级: 按 finding severity 加权 + DS 合成.
            _pa_src = execution_result if isinstance(execution_result, dict) else {}
            if isinstance(_pa_src.get("physics_audit"), dict):
                _gate_evidence["physics_audit"] = _pa_src["physics_audit"]
            if not self._check_gate("validate", "learn", _gate_evidence):
                continue

            # 6. Learn
            phase = await self._run_phase_async("learn", self._learn, hypothesis, plan, validation)
            phases.append(phase)
            completed_steps += 1
            tracker.update(progress_task_id, current_step=completed_steps,
                           current_label=f"iter {self._iteration}: learn ({phase.status})")
            logger.info("learning complete")

            # Goal completion: success_criteria 全命中 → 提前停循环.
            # 没 goal 或 criteria 为空时 check_completion 返回 False, 不影响.
            if goal is not None and GoalScheduler.check_completion(goal, validation):
                # completion audit: goal 达标但探索不够 → 不停, 把缺口塞 hint 下轮看
                _blk, _why = self._metacog_check_completion()
                if _blk:
                    logger.info("metacog completion audit blocked goal-completion stop: %s", _why)
                    _why_msg = f"[completion audit] 不能停: {_why}"
                    self._speculator_hint = (
                        (self._speculator_hint + "\n" + _why_msg).strip()
                        if self._speculator_hint else _why_msg
                    )
                else:
                    logger.info("goal completed: %s", goal.objective)
                    goal.status = "completed"
                    if self._goal_scheduler is not None:
                        self._goal_scheduler.complete_goal(goal.id)
                    self._should_stop = True

            # GoalJudge 反馈: 每 3 轮或最后一轮做一次快速目标判定.
            # 没达成时把 gaps 拼进 _speculator_hint, 下轮 plan 会看到.
            if goal is not None and not self._should_stop:
                if self._iteration % 3 == 2 or self._iteration >= max_iterations - 1:
                    try:
                        from huginn.evaluation.goal_judge import GoalJudge
                        judge = GoalJudge(llm=None)  # rule-based in loop, LLM judge at exit
                        final_text = str(validation.get("summary") or
                                         validation.get("result_data") or
                                         execution_result.get("summary", ""))
                        gj = judge.judge(goal.objective, None, final_text)
                        if gj.get("achieved"):
                            # completion audit: 判定达标但探索不够 → 继续迭代
                            _blk, _why = self._metacog_check_completion()
                            if _blk:
                                logger.info("metacog completion audit blocked GoalJudge stop: %s", _why)
                                _why_msg = f"[completion audit] 不能停: {_why}"
                                self._speculator_hint = (
                                    (self._speculator_hint + "\n" + _why_msg).strip()
                                    if self._speculator_hint else _why_msg
                                )
                            else:
                                logger.info("goaljudge: achieved (score=%s)", gj['score'])
                                self._should_stop = True
                        elif gj.get("gaps"):
                            gap_hint = "; ".join(gj["gaps"][:3])
                            self._speculator_hint = (
                                (self._speculator_hint + "\n" + gap_hint).strip()
                                if self._speculator_hint else gap_hint
                            )
                            logger.info("goaljudge gaps: %s", gap_hint)
                    except Exception:
                        logger.warning("error in run: GoalJudge evaluation failed", exc_info=True)

            # JEPA intrinsic motivation: 高 surprise = 预测误差大 = 这个方向
            # agent 的心智模型不准, 值得继续探索. 把 surprise 信号注入
            # _speculator_hint, 下轮 hypothesize/plan 会看到并优先关注.
            # MPC 的 receding horizon: 每轮用预测误差调整下一步, 不是固定计划.
            surprise = getattr(self, "_last_surprise", 0.0)
            if surprise > 0.5:
                surprise_hint = (
                    f"High prediction surprise ({surprise:.2f}) last iteration: "
                    "the actual result differed significantly from what was predicted. "
                    "This area is poorly understood — consider exploring why."
                )
                self._speculator_hint = (
                    (self._speculator_hint + "\n" + surprise_hint).strip()
                    if self._speculator_hint else surprise_hint
                )
                logger.info("surprise: %.2f (high — exploring)", surprise)
            elif surprise > 0 and self._iteration > 1:
                logger.info("surprise: %.2f (low — model matches reality)", surprise)

            # Surprise-based early termination: 连续 3 轮低 surprise
            # 说明 agent 预测持续命中实际结果, 心智模型已收敛.
            # Chemputer 用 Jaccard 稳定判反应终点, 同理判理解终点.
            # 自适应阈值: cross-perturbation std 高 = 测量噪声大,
            # 需更低的 surprise 才能确信真正收敛 (避免噪声导致的假阳性).
            if len(self._surprise_history) >= 3:
                recent = self._surprise_history[-3:]
                recent_worsts = [w for w, _ in recent]
                avg_noise = sum(s for _, s in recent) / len(recent)
                # ponytail: 线性插值. noise=0 → threshold=0.20 (测量可信, 宽松);
                # noise≥0.3 → threshold=0.08 (测量噪声大, 严格). 下限 0.08 防永不终止.
                threshold = max(0.08, 0.20 - 0.4 * avg_noise)
                if all(w < threshold for w in recent_worsts):
                    # completion audit: surprise 收敛但探索不够 → 不停, 对抗快速收敛偏差
                    _blk, _why = self._metacog_check_completion()
                    if _blk:
                        logger.info("metacog completion audit blocked convergence stop: %s", _why)
                        _why_msg = f"[completion audit] 不能停: {_why}"
                        self._speculator_hint = (
                            (self._speculator_hint + "\n" + _why_msg).strip()
                            if self._speculator_hint else _why_msg
                        )
                    else:
                        logger.info(
                            "converged: surprise < %.2f (noise=%.2f) for 3 consecutive iterations",
                            threshold, avg_noise,
                        )
                        self._should_stop = True

        # 7. Report + finalize
        return await self._finalize_run(
            objective, phases, run_id, provenance_record,
            run_collector, tracker, progress_task_id, completed_steps,
        )

    def _emit_campaign(self, event_type: str, data: dict) -> None:
        """发布 campaign.* 事件到 EventBus + SSE 流, fire-and-forget.

        双通道: EventBus 给 audit/telemetry, SSE 给前端 IterationTimeline.
        之前只发 EventBus, 前端只能正则刮消息文本, retry/suspect/refine
        根本到不了前端.
        """
        try:
            from huginn.events.integration import _publish
            from huginn.utils.concurrency import track_task
            asyncio.get_running_loop()  # 检测在 event loop 里
            track_task(_publish(event_type, data, source="autoloop"), name="campaign-emit")
        except Exception:
            logger.debug("campaign EventBus emit failed", exc_info=True)
        # SSE 推送到 /tasks/stream 的 'campaign' event, 前端结构化消费
        try:
            from huginn.interaction.progress import get_progress_tracker
            get_progress_tracker().emit_campaign_event(
                getattr(self, "_progress_task_id", ""), event_type, data
            )
        except Exception:
            logger.debug("campaign SSE emit failed", exc_info=True)

    def _prepare_run(
        self, objective: str, progressive_budget: bool, goal: Goal | None
    ) -> tuple[str, Any, Any]:
        """Set up run state: provenance, telemetry, budget, speculator."""
        run_id = f"loop_{uuid.uuid4().hex[:8]}"
        self._run_start_time = time.time()
        self._objective = objective

        from huginn.provenance import ProvenanceLogger, ProvenanceRecord
        provenance_logger = ProvenanceLogger(
            self.workspace / ".huginn" / "provenance.jsonl"
        )
        provenance_record = ProvenanceRecord(
            run_id=run_id,
            objective=objective,
            timestamps={"start": datetime.now().isoformat()},
        )
        self._provenance_record = provenance_record
        self._provenance_logger = provenance_logger

        from huginn.telemetry import TelemetryCollector, set_telemetry_collector
        run_collector = TelemetryCollector()
        set_telemetry_collector(run_collector)

        self._iteration = 0
        self._should_stop = False
        self._consecutive_failures = 0

        if goal is not None and goal.status == "pending":
            goal.status = "active"
            if self._goal_scheduler is not None:
                self._goal_scheduler.update_goal(goal.id, status="active")

        self._budget = ProgressiveBudget.default() if progressive_budget else None
        self._budget_rejects: dict[str, int] = {}
        self._budget_degraded = False

        self._speculator_hint = ""
        self._last_visual_context = ""  # reset per run, stale data shapes mislead
        self._current_prediction = ""  # reset JEPA prediction buffer
        self._last_surprise = 0.0
        self._last_raw_hypothesis = ""  # 完整 LLM 输出, 含 LUCID review
        try:
            from huginn.agents.speculator import on_turn_start
            spec_result = on_turn_start(objective)
            self._speculator_hint = spec_result.get("hint", "")
            if spec_result.get("predictions"):
                logger.info("autoloop speculator: %s", self._speculator_hint)
        except Exception as exc:
            logger.warning("autoloop speculator skipped", exc_info=True)

        return run_id, provenance_record, run_collector

    async def _finalize_run(
        self, objective: str, phases: list[LoopPhase],
        run_id: str, provenance_record: Any,
        run_collector: Any, tracker: Any, progress_task_id: str,
        completed_steps: int,
    ) -> AutoloopResult:
        """Report, save trajectory, judge goal, write provenance + FAIR metadata."""
        total_time = time.time() - getattr(self, "_run_start_time", time.time())
        report_phase = await self._run_phase_async(
            "report", self._report, objective, phases, total_time
        )
        phases.append(report_phase)
        completed_steps += 1
        tracker.update(progress_task_id, current_step=completed_steps,
                       current_label=f"report ({report_phase.status})")

        if report_phase.status == "completed":
            tracker.complete(progress_task_id, result={"report_path": report_phase.result})
        else:
            tracker.fail(progress_task_id, f"report phase failed: {report_phase.error}")

        # session summary → long-term memory
        try:
            self.memory.promote_session_summary(tier="long")
        except Exception:
            logger.debug("session summary promotion failed", exc_info=True)

        # trajectory
        trajectory_path = None
        trajectory_data = None
        try:
            from huginn.telemetry import save_trajectory, load_trajectory
            traj_dir = self.workspace / ".huginn" / "trajectories"
            trajectory_path = traj_dir / f"{run_id}.json"
            save_trajectory(run_collector, trajectory_path, metadata={
                "run_id": run_id, "objective": objective[:200],
                "phases": [p.name for p in phases], "total_time": total_time,
            })
            trajectory_data = load_trajectory(trajectory_path)
        except Exception:
            trajectory_path = None

        # goal judgment
        goal_achieved = None
        goal_judgment = None
        try:
            from huginn.evaluation.goal_judge import GoalJudge
            final_output = str(report_phase.result or "")
            judge = GoalJudge(llm=self.verification_model or self.model)
            goal_judgment = judge.judge(
                objective=objective, trajectory=trajectory_data,
                final_output=final_output,
            )
            goal_achieved = goal_judgment.get("achieved")
        except Exception as e:
            logger.warning("autoloop goal judge skipped", exc_info=True)

        # provenance
        provenance_path = None
        try:
            provenance_record.timestamps["end"] = datetime.now().isoformat()
            self._provenance_logger.log(provenance_record)
            provenance_path = str(self._provenance_logger.path)
        except Exception:
            provenance_path = None

        # FAIR metadata
        try:
            from huginn.export.fair_metadata import generate_dataset_metadata, write_fair_jsonld
            run_results: dict[str, Any] = {}
            for ph in phases:
                if ph.result and isinstance(ph.result, dict):
                    run_results.update(ph.result)
            fair_metadata = generate_dataset_metadata(
                run_id=run_id, objective=objective, results=run_results,
                provenance={
                    "report_path": str(report_phase.result) if report_phase.result else None,
                    "trajectory_path": str(trajectory_path) if trajectory_path else None,
                    "provenance_path": provenance_path,
                    "start_time": provenance_record.timestamps.get("start"),
                    "end_time": provenance_record.timestamps.get("end"),
                },
            )
            jsonld_path = self.workspace / f"{run_id}_dataset.jsonld"
            write_fair_jsonld(fair_metadata, jsonld_path)
            logger.info("FAIR JSON-LD written to %s", jsonld_path)
        except Exception:
            logger.debug("FAIR metadata generation failed", exc_info=True)

        return AutoloopResult(
            run_id=run_id,
            objective=objective,
            phases=phases,
            success=all(p.status == "completed" for p in phases[-7:]),
            report_path=report_phase.result,
            total_time_seconds=total_time,
            trajectory_path=str(trajectory_path) if trajectory_path else None,
            goal_achieved=goal_achieved,
            goal_judgment=goal_judgment,
            provenance_path=provenance_path,
        )

    def stop(self) -> None:
        """Signal the loop to stop at the next safe point."""
        self._should_stop = True

    # ──────────────────────────────────────────────────────────────
    # Phase implementations
    # ──────────────────────────────────────────────────────────────

    def _perceive(self) -> dict[str, Any] | None:
        """Perceive the workspace using the multi-modal perception layer.

        The PerceptionLayer is now a long-lived member started in __init__,
        so background watchers and log tailers actually accumulate events
        between iterations. Previously we started+stopped it here, which
        killed the watcher threads before they could collect anything.
        """
        perception = self._get_perception()
        if perception is None:
            return self._perceive_legacy()
        try:
            snapshot = perception.get_snapshot()
        except Exception:
            return self._perceive_legacy()
        context = snapshot.to_context()
        if not snapshot.has_activity():
            return None
        # L3/L4: 语义对齐 + 认知整合, 把冲突和推荐动作塞进 context
        try:
            cog = perception.get_cognitive_state()
            if cog.conflicts:
                context["semantic_conflicts"] = [
                    {"sources": [c.source_a, c.source_b], "description": c.description}
                    for c in cog.conflicts
                ]
            if cog.recommended_actions:
                context["recommended_actions"] = cog.recommended_actions
            if cog.recommended_tools:
                context["recommended_tools"] = cog.recommended_tools
            if cog.simulation_converged is not None:
                context["simulation_converged"] = cog.simulation_converged
        except Exception:
            logger.debug("L3/L4 cognitive integration 失败", exc_info=True)
        return context

    def _perceive_legacy(self) -> dict[str, Any] | None:
        """Legacy perceive (fallback)."""
        changed_files = []
        git_diff = ""
        try:
            import subprocess
            result = subprocess.run(
                ["git", "status", "--short"],
                cwd=self.workspace, capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                changed_files = [line.strip() for line in result.stdout.strip().split("\n")]
                git_diff = subprocess.run(
                    ["git", "diff", "--stat"],
                    cwd=self.workspace, capture_output=True, text=True, timeout=10,
                ).stdout
        except Exception:
            logger.warning("error in _perceive_legacy: git diff collection failed", exc_info=True)
        error_patterns = []
        for log_file in self.workspace.rglob("*.log"):
            if log_file.stat().st_mtime > time.time() - 3600:
                try:
                    content = log_file.read_text(errors="ignore")
                    if "ERROR" in content or "FAIL" in content:
                        error_patterns.append(f"{log_file.name}: {content[:200]}")
                except Exception:
                    logger.warning("error in _perceive_legacy: log file error-pattern scan failed", exc_info=True)
        if not changed_files and not error_patterns:
            return None
        return {
            "changed_files": changed_files,
            "git_diff": git_diff,
            "error_patterns": error_patterns,
            "timestamp": datetime.now().isoformat(),
        }

    async def _hypothesize(self, context: dict[str, Any]) -> str | None:
        """Generate a hypothesis from perceived context."""
        # Use knowledge graph + LLM to generate hypothesis
        # symreg (async, up to 60s) and conjecture (sync, fast) are independent
        symreg_task = asyncio.create_task(self._symreg_hint(context))
        conjecture_hint = self._conjecture_hint(context)
        symreg_hint = await symreg_task
        prompt = self._build_hypothesis_prompt(context)
        if symreg_hint:
            prompt = f"{symreg_hint}\n{prompt}"
        if conjecture_hint:
            prompt = f"{conjecture_hint}\n{prompt}"
        # 按研究类型选 persona: MD 类用 md_expert, 默认走 dft_expert.
        # 这俩 persona 在 personas.py 内置, 直接取就行.
        persona_name = self._pick_hypothesis_persona(context)
        self._last_persona = persona_name  # 供 _learn 写入 memory/KG
        try:
            # True parallel sampling: main call + high-temp diversity call.
            # Both run concurrently via asyncio.gather — same wall-clock latency
            # as a single call, 2x tokens for genuine diversity insurance.
            # Main call's SELECTED: wins; diversity call is fallback.
            # ponytail: 2 calls not 3 — main provides quality, diversity provides
            # novelty; a third call adds cost without marginal diversity gain.
            hot_model = None
            try:
                hot_model = self.model.bind(temperature=1.0)
            except Exception:
                pass  # not all model wrappers support bind
            coros = [self._llm_chat(prompt, persona_name=persona_name, task="reasoning")]
            if hot_model is not None:
                coros.append(
                    self._llm_chat(prompt, persona_name=persona_name, model=hot_model, task="reasoning")
                )
            results = await asyncio.gather(*coros, return_exceptions=True)
            # Extract SELECTED: from results — main call first (priority)
            for raw in results:
                if isinstance(raw, Exception) or not raw:
                    continue
                raw = raw.strip()
                if "SELECTED:" in raw:
                    _after = raw.split("SELECTED:", 1)[1].strip()
                    _sel = _after.split("\n")[0].strip() if _after else ""
                    self._last_hypothesis = _sel or raw
                    self._last_raw_hypothesis = raw  # 保留 LUCID review 文本
                    self._metacog_audit_hypothesis(self._last_hypothesis, context)
                    return self._last_hypothesis
            # No SELECTED: found — fall back to first non-exception result
            for raw in results:
                if not isinstance(raw, Exception) and raw:
                    raw = raw.strip()
                    self._last_hypothesis = raw
                    self._last_raw_hypothesis = raw
                    self._metacog_audit_hypothesis(self._last_hypothesis, context)
                    return self._last_hypothesis
            return None
        except Exception:
            return None

    # ── 元认知层辅助 ────────────────────────────────────────────────
    # 懒加载 metacog 组件, 避免循环 import 和测试 mock 复杂度.
    # _hypothesize 拿到假设后调 _metacog_audit_hypothesis 做等价性审计 +
    # 方法族归类. 审计是 advisory (不阻断), 对齐用户 "math 结构 advisory" 偏好.

    def _get_metacog_auditor(self):
        if self._metacog_auditor is None:
            from huginn.metacog.equivalence_auditor import EquivalenceAuditor
            self._metacog_auditor = EquivalenceAuditor(model=self.model)
        return self._metacog_auditor

    def _get_metacog_block_registry(self):
        if self._metacog_block_registry is None:
            from huginn.metacog.block_registry import BlockRegistry
            self._metacog_block_registry = BlockRegistry(
                auditor=self._get_metacog_auditor()
            )
        return self._metacog_block_registry

    def _get_metacog_method_registry(self):
        if self._metacog_method_registry is None:
            from huginn.metacog.method_registry import MethodRegistry
            self._metacog_method_registry = MethodRegistry()
        return self._metacog_method_registry

    def _get_metacog_convergence_detector(self):
        if self._metacog_convergence_detector is None:
            from huginn.metacog.depth_search import PrematureConvergenceDetector
            self._metacog_convergence_detector = PrematureConvergenceDetector()
        return self._metacog_convergence_detector

    def _get_metacog_completion_auditor(self):
        if self._metacog_completion_auditor is None:
            from huginn.metacog.completion_auditor import CompletionAuditor
            self._metacog_completion_auditor = CompletionAuditor(
                convergence_detector=self._get_metacog_convergence_detector(),
                equivalence_auditor=self._get_metacog_auditor(),
            )
        return self._metacog_completion_auditor

    def _metacog_check_effort_floor(self) -> tuple[bool, str]:
        """[deprecated] 旧的最小努力下限检查, 保留向后兼容.

        新代码用 _metacog_check_completion — 它在 effort floor 之上加了
        等价性陷阱检查和显式不完整性自白. 这个方法只是 thin wrapper,
        保留是因为可能有外部调用方直接调它.
        """
        return self._metacog_check_completion()

    def _metacog_check_completion(self) -> tuple[bool, str]:
        """反完成审计: 综合检查是否过早收敛.

        替代旧的纯 effort floor, 调 CompletionAuditor 做四层检查:
        - 最小努力下限 (迭代/方法族/连通分量)
        - 等价性陷阱 (内部调 EquivalenceAuditor)
        - 显式不完整性自白 (从 _last_raw_hypothesis 提取 UNEXPLORED 块)
        - 对抗否决 (本 engine 暂不接 red_team, 留口子)

        advisory: 出错放行, 不阻断.
        """
        try:
            auditor = self._get_metacog_completion_auditor()
            families_explored = len([
                f for f in self._get_metacog_method_registry().all()
                if f.member_agent_ids
            ])
            live_components = self.hypothesis_graph.component_count()

            # 从最近一次 LLM 原始输出提取 UNEXPLORED: 块
            # ponytail: 字符串切片, 不上正则. 升级路径: 结构化 schema.
            unexplored = ""
            raw = getattr(self, '_last_raw_hypothesis', '') or ''
            if 'UNEXPLORED:' in raw:
                unexplored = raw.split('UNEXPLORED:', 1)[1].strip()
                # 截到下一个大写标记或结尾, 避免把后续块都吞进来
                for marker in ['\n\nHYPOTHESIS', '\n\nSELECTED', '\n\nRATIONALE']:
                    if marker in unexplored:
                        unexplored = unexplored.split(marker)[0].strip()
                        break

            checklist = auditor.audit(
                iteration=self._iteration,
                families_explored=families_explored,
                live_components=live_components,
                total_iterations=self._max_iterations if hasattr(self, '_max_iterations') else 10,
                candidate_finding=getattr(self, '_last_hypothesis', '') or '',
                original_problem=str(getattr(self, '_objective', '') or ''),
                unexplored_declaration=unexplored,
            )

            if not checklist.is_complete:
                return True, checklist.block_reason()
            return False, ""
        except Exception:
            logger.debug("metacog completion check failed", exc_info=True)
            return False, ""  # 出错不阻断, advisory

    def _metacog_check_topology_collapse(self) -> None:
        """坍缩检测: 连通分量数过低时, 强制从冷门族启动新探索.

        对应 prompt: "不要让一种方法占据主导...并发起新一轮".
        is_collapsed 时把重定向建议拼进 _speculator_hint, 下轮 hypothesize
        会看到. advisory: 不阻断, 出错放行.
        """
        try:
            from huginn.metacog.depth_search import DynamicComponentFloor
            # 动态下限: 早期 4, 中期 2, 后期 1. 实例化成本可忽略.
            floor = DynamicComponentFloor().current_floor(
                self._iteration, self._max_iterations
            )
            if self.hypothesis_graph.is_collapsed(min_components=floor):
                redirect = self._get_metacog_method_registry().suggest_redirect()
                hint = (
                    f"[topology] 搜索空间坍缩! "
                    f"连通分量 {self.hypothesis_graph.component_count()} < 下限 {floor}. "
                )
                if redirect:
                    hint += f"强制重定向到 {redirect.target_family}: {redirect.reason}"
                else:
                    hint += "建议启动新方法族探索"
                if self._speculator_hint:
                    self._speculator_hint = f"{self._speculator_hint}\n{hint}"
                else:
                    self._speculator_hint = hint
                logger.info("metacog: %s", hint)
        except Exception:
            logger.debug("metacog topology check failed", exc_info=True)

    def _metacog_component_representatives(self) -> list[str]:
        """每个连通分量的代表假设 id.

        代表参与根 agent 综合判断, 防止单分量靠节点数主导.
        对应 prompt: "根智能体应反复综合、挑战、重定向".
        出错返回空列表, 调用方按 advisory 处理.
        """
        try:
            components = self.hypothesis_graph.connected_components()
            reps = []
            for comp in components:
                rep = self.hypothesis_graph.component_representative(comp)
                if rep:
                    reps.append(rep)
            return reps
        except Exception:
            return []

    def _metacog_classify_family(self, hypothesis: str) -> str:
        """廉价关键词分类: 把假设归到方法族.

        用于 method_registry 收敛度监控 + block_registry 查阻塞路线.
        分类不准不致命 — 只影响监控, 不影响假设本身.
        """
        text = (hypothesis or "").lower()
        # ponytail: 关键词表, 不上 embedding. 升级路径: LLM 分类.
        rules = [
            ("ml-potential", ["mlp", "ml potential", "machine learning potential", "neural potential"]),
            ("symbolic-regression", ["symbolic", "symreg", "siprend", "解析式"]),
            ("gaussian-process", ["gp ", "gaussian process", "gpr", "核函数"]),
            ("calphad-thermo", ["calphad", "相图", "phase diagram", "thermodynamic"]),
            ("phase-field", ["phase field", "相场"]),
            ("bourbaki-structure", ["bourbaki", "格论", "lattice theory", "拓扑"]),
            ("extreme-argument", ["反例", "counterexample", "extreme", "极值"]),
            ("computational-check", ["benchmark", "计算验证", "computational check"]),
            ("dft-direct", ["dft", "第一性原理", "ab initio", "vasp", "qe", "cp2k"]),
        ]
        for family, keywords in rules:
            if any(kw in text for kw in keywords):
                return family
        return "dft-direct"  # 默认族

    def _metacog_audit_hypothesis(
        self, hypothesis: str, context: dict[str, Any]
    ) -> None:
        """假设生成后的等价性审计 + 方法族归类.

        advisory 不阻断: 即使判为 equivalent_renaming 也让假设通过,
        但记录到 _metacog_last_audit 给 learn 阶段参考. 这对齐用户
        'math 结构 advisory' 和 '先警告再 force proceed' 偏好.
        """
        if not hypothesis:
            return
        try:
            auditor = self._get_metacog_auditor()
            original_problem = str(context.get("summary", "")) or str(self._objective or "")
            verdict = auditor.audit(
                candidate_finding=hypothesis,
                original_problem=original_problem,
                reduction_chain="",  # _hypothesize 阶段还没有归约链
            )
            self._metacog_last_audit = verdict

            # 方法族归类 + 注册表更新
            family = self._metacog_classify_family(hypothesis)
            registry = self._get_metacog_method_registry()
            # 用 iteration + hypothesis 短哈希做 agent_id, 避免重复登记
            agent_id = f"hyp-{self._iteration}-{abs(hash(hypothesis)) % 10000}"
            registry.register_agent(family, agent_id)

            # 等价性审计发现换名归约 → 记日志, 不阻断
            if verdict.is_equivalent_renaming:
                logger.warning(
                    "metacog: 假设可能为换名归约 (trap=%s, target=%s): %s",
                    verdict.trap_category, verdict.reduction_target,
                    hypothesis[:100],
                )

            # 收敛度监控: 某族过热时记日志
            redirect = registry.suggest_redirect()
            if redirect is not None:
                logger.info(
                    "metacog: 方法族收敛度告警 — %s (建议下轮重定向到 %s)",
                    redirect.reason, redirect.target_family,
                )
        except Exception:
            logger.debug("metacog audit failed", exc_info=True)

    @staticmethod
    def _extract_lucid_prereqs(raw: str) -> dict[str, str]:
        """从 LLM 输出里解析 LUCID review 的三项必要条件.

        prompt 要求 LLM 在 SELECTED 后输出:
        - necessary condition: ...
        - hidden assumption: ...
        - falsifiable test: ...

        返回 {"necessary": ..., "hidden": ..., "falsifiable": ...}, 缺项为空串.
        LLM 格式不固定时降级到关键词模糊匹配."""
        if not raw:
            return {"necessary": "", "hidden": "", "falsifiable": ""}
        text = raw.lower()
        result = {"necessary": "", "hidden": "", "falsifiable": ""}
        # 关键词 + 后续行内容, 容忍中英文标点
        patterns = {
            "necessary": r"necessary[^:：]*[:：]\s*(.+)",
            "hidden": r"hidden\s*assumption[^:：]*[:：]\s*(.+)",
            "falsifiable": r"falsifiable\s*test[^:：]*[:：]\s*(.+)",
        }
        for key, pat in patterns.items():
            m = re.search(pat, text)
            if m:
                # 取到行尾或句号
                val = m.group(1).split("\n")[0].strip().rstrip(".。")
                result[key] = val[:300]  # 长度限制, 防异常输入
        return result

    @staticmethod
    def _classify_failure(
        validation: dict[str, Any],
        redteam_cats: list[str] | None = None,
    ) -> str:
        """根据 validation 证据分类失败类型, 决定走 retry/refine/pivot.

        返回值:
        - "tool_error": 工具崩溃/超时/连接失败 → 不 refute, 下轮重试同一假设
        - "prompt_injection_suspect": 失败 + 证据来源 external_content → 可能被注入, 单独标记
        - "param_error": 输入参数错 → refine (改参数)
        - "data_noise": 结果不确定/噪声大 → refine (重做或换方法)
        - "hypothesis_error": 结果与假设相反 → refine 或 pivot (假设本身错)

        优先级: tool_error (工具问题与假设无关) > prompt_injection_suspect
        (external_content + 失败) > RedTeam high severity findings
        > 关键词匹配 param/noise > 默认 hypothesis_error.

        ponytail: RedTeam findings 通过 redteam_cats 注入, 保持 staticmethod
        可测性. 映射: methodology_gap/hidden_assumption → param_error,
        confounder → data_noise, alternative_explanation → hypothesis_error.
        升级: 让 LLM 对 ambiguous 失败做语义分类 (当前纯关键词+规则).
        """
        errors = str(validation.get("errors", ""))
        result = str(validation.get("result", ""))
        text = (errors + " " + result).lower()
        # 工具失败: 超时/崩溃/连接/OOM — 不是假设错, 重试即可
        tool_markers = (
            "timeout", "timed out", "connection", "crash", "segfault",
            "oom", "out of memory", "exception", "subprocess",
            "slurm", "queue", "killed", "abort",
        )
        if any(m in text for m in tool_markers):
            return "tool_error"
        # ARGUS: 失败 + 证据来自 external_content → 可能 prompt injection.
        # 优先级低于 tool_error (技术故障与来源无关), 高于 RedTeam/关键词.
        # ponytail: 递归扫 validation 找 source_class=external_content.
        if _validation_has_external_source(validation):
            return "prompt_injection_suspect"
        # RedTeam high severity findings: 对抗性发现优先于关键词匹配
        # methodology_gap (方法论缺陷) / hidden_assumption (隐含前提缺失) → 改参数
        # confounder (混淆变量) → 数据噪声, 需重做排除混淆
        # alternative_explanation (替代解释) → 假设本身可能错
        if redteam_cats:
            _RT_MAP = {
                "methodology_gap": "param_error",
                "hidden_assumption": "param_error",
                "confounder": "data_noise",
                "alternative_explanation": "hypothesis_error",
            }
            for cat in redteam_cats:
                if cat in _RT_MAP:
                    return _RT_MAP[cat]
        # 参数错: 输入无效/类型错/值错
        param_markers = (
            "invalid", "argument", "parameter", "value error",
            "type error", "dimension", "shape mismatch", "key error",
        )
        if any(m in text for m in param_markers):
            return "param_error"
        # 数据噪声: 不确定/模糊/噪声大
        noise_markers = (
            "noise", "uncertain", "ambiguous", "inconclusive",
            "not converge", "did not converge", "no clear",
        )
        if any(m in text for m in noise_markers):
            return "data_noise"
        # 默认: 假设错 (结果与预期相反, 或无明确错误但测试失败)
        return "hypothesis_error"

    def _redteam_findings(self) -> list[str]:
        """拿最近一次 RedTeam 审查的 high severity findings category.

        C: _classify_failure 用这些 category 覆盖关键词分类.
        RedTeam reviewer 在 phase_gate_hook.reviewer_fn 上, _last_report
        存最近一次审查结果. 失败返回空列表, 不影响分类.
        """
        try:
            reviewer = getattr(self.phase_gate_hook, "reviewer_fn", None)
            report = getattr(reviewer, "_last_report", None)
            if not report:
                return []
            return [
                f.category for f in report.findings
                if f.severity == "high"
            ]
        except Exception:
            return []

    def _attach_lucid_prereqs(self, hyp_id: str) -> None:
        """把 LUCID review 的 necessary condition 加成派生假设节点, 进 frontier 队列.

        闭环: prompt 要求 LLM 自检必要条件, 但之前只取 SELECTED 行丢弃了 LUCID 文本.
        现在解析出来, 把 necessary condition 转成 hypothesis_graph 的派生节点,
        让 campaign 队列去验证它. 如果必要条件被 refute, 原假设也站不住.

        ponytail: 只加 necessary (最关键), hidden/falsifiable 记到 evidence.
        升级: necessary refute 时级联 refute parent (需改 refute 方法).
        """
        raw = getattr(self, "_last_raw_hypothesis", "")
        if not raw or not hyp_id:
            return
        prereqs = self._extract_lucid_prereqs(raw)
        necessary = prereqs["necessary"]
        if not necessary:
            return
        try:
            self.hypothesis_graph.add_hypothesis(
                statement=f"[必要条件] {necessary}",
                rationale=f"LUCID necessary condition for {hyp_id}",
                parent_id=hyp_id,
            )
        except Exception:
            logger.debug("attach lucid prereqs failed", exc_info=True)

    def _should_imaginate(self) -> bool:
        """是否触发想象力模式. 高 surprise 或连续 refine 时返回 True.

        MToM P4 (hybrid ST+TT): 心智模型预测错误时, 从 Theory Theory
        切到 Simulation Theory 重新建模. 这里就是那个切换信号.
        """
        return (
            getattr(self, "_last_surprise", 0.0) > 0.5
            or getattr(self, "_refine_count", 0) >= 2
        )

    def _recent_failed_hypotheses(self, limit: int = 3) -> list[str]:
        """从 hypothesis_graph 捞最近被 refuted 的假设, 给 forget_then_generate 用."""
        try:
            nodes = getattr(self.hypothesis_graph, "_nodes", {})
            failed = [
                n.statement for n in nodes.values()
                if n.status in ("refuted", "superseded")
            ]
            return failed[-limit:] if failed else []
        except Exception:
            return []

    def _conjecture_hint(self, context: dict[str, Any]) -> str:
        """跑 Moonshine 跨域猜想流水线, 返回注入 prompt 的 hint.

        从 context 提取源问题和领域, 调 ConjectureGenerator 生成跨域类比
        猜想. 失败返回空串, 不影响 hypothesize 主流程.

        想象力模式: _should_imaginate() 为 True 时改调 forget_then_generate,
        把已 refuted 的假设当 known_solutions 遗忘掉, 强制从第一性原理重来.
        """
        try:
            from huginn.autoloop.conjecture import get_conjecture_generator

            source_problem = context.get("goal") or context.get("observation") or ""
            if not source_problem or len(source_problem) < 10:
                return ""
            source_domain = context.get("domain") or "materials science"
            target_domain = context.get("target_domain") or "battery cathodes"

            gen = get_conjecture_generator()

            if self._should_imaginate():
                known = self._recent_failed_hypotheses()
                result = gen.forget_then_generate(
                    source_problem=str(source_problem)[:500],
                    source_domain=str(source_domain),
                    target_domain=str(target_domain),
                    known_solutions=known,
                    model=None,
                )
            else:
                result = gen.run(
                    source_problem=str(source_problem)[:500],
                    source_domain=str(source_domain),
                    target_domain=str(target_domain),
                    model=None,  # template mode, 不烧 token
                )
            conjecture = result.get("conjecture", {})
            statement = conjecture.get("statement", "")
            prediction = conjecture.get("prediction", "")
            if not statement:
                return ""
            # Prerequisite Inversion: 跨域类比不是直接用, 而是问"什么条件必须暗中获得满足"
            # 4 维反转防止结构错配 (Dream Layer v1.1 核心贡献)
            return (
                f"[Cross-domain analogy hint]\n"
                f"Conjecture: {statement}\n"
                f"Prediction: {prediction}\n"
                f"Before using this analogy, perform Prerequisite Inversion:\n"
                f"- Necessary: What condition MUST hold for this analogy to be valid?\n"
                f"- Boundary: In what parameter range does it break down?\n"
                f"- Hidden: What implicit assumption from the source domain may NOT hold here?\n"
                f"- Failure: If this analogy is wrong, what would the system look like instead?\n"
                f"(Template-based analogy — verify conditions before adopting.)"
            )
        except Exception:
            return ""

    async def _symreg_hint(self, context: dict[str, Any]) -> str:
        """从 observation_data 跑符号回归, 把最优解析表达式作为 hint 返回.

        PSE/PSRN 不可用、数据不全、搜索失败都返回空串, 不影响 hypothesize.
        time_limit 压到 60s 避免 hypothesize 阶段被符号回归卡死.

        A3: 同时查 KB 拿已知公式形式 (Arrhenius / Brillouin / Langmuir 等) 作为
        kb_candidate_forms 前缀, 跟 symreg 数据驱动候选一起注入 hypothesize prompt.
        KB 给先验形式, symreg 给数据拟合, 两边互补."""
        data = context.get("observation_data")
        if not isinstance(data, dict) or not data:
            return ""
        # A3: 先查 KB 拿已知公式形式, symreg 失败时 KB 候选仍能注入
        kb_forms = self._query_kb_known_forms(data)
        try:
            from huginn.tools.sci.symbolic_regression_tool import (
                SymbolicRegressionInput,
                SymbolicRegressionTool,
            )
            from huginn.types import ToolContext

            target = data.get("target_column") or data.get("target") or "y"
            if target not in data:
                return kb_forms
            tool = SymbolicRegressionTool()
            args = SymbolicRegressionInput(
                action="discover",
                data_json=data,
                target_column=target,
                time_limit=60,
                top_k=3,
            )
            ctx = ToolContext(
                session_id=f"symreg_{uuid.uuid4().hex[:8]}",
                workspace=str(self.workspace),
                config=self.settings,
            )
            vr = await tool.call(args, ctx)
            symreg_block = ""
            if vr.success and vr.data:
                cands = (
                    vr.data.get("candidates")
                    or vr.data.get("expressions")
                    or vr.data.get("pareto_front")
                    or []
                )
                if cands:
                    top = cands[0] if isinstance(cands, list) else cands
                    expr = top.get("expression") if isinstance(top, dict) else str(top)
                    if expr:
                        symreg_block = (
                            "### Data-driven candidate law (symbolic regression)\n"
                            f"Top recovered expression: {expr}\n"
                            "Use this as a data-driven candidate when forming the hypothesis.\n"
                            "### End candidate law"
                        )
            if symreg_block and kb_forms:
                return f"{kb_forms}\n{symreg_block}"
            return symreg_block or kb_forms
        except Exception:
            return kb_forms

    def _query_kb_known_forms(self, data: dict[str, Any]) -> str:
        """查 KB 拿已知公式形式 (Arrhenius / Brillouin / Langmuir 等) 作为
        symreg 先验. 失败/空都返回空串."""
        try:
            target = data.get("target_column") or data.get("target") or "y"
            feature_keys = [k for k in data.keys() if k != target and isinstance(data[k], list)]
            query = f"symbolic expression formula {target} {' '.join(feature_keys[:3])}"
            kb = self._get_kb()
            if kb is None or kb.count() == 0:
                return ""
            chunks = kb.query(query, top_k=2)
            if not chunks:
                return ""
            lines = []
            for i, c in enumerate(chunks, 1):
                text = (c.get("text") or "").strip()
                if text:
                    lines.append(f"[{i}] {text[:300]}")
            if not lines:
                return ""
            return (
                "### KB candidate forms (known first-principles expressions)\n"
                "The knowledge base suggests these known formula forms. Compare "
                "your data-driven candidate against them.\n"
                + "\n".join(lines)
                + "\n### End KB candidate forms"
            )
        except Exception:
            return ""

    def _pick_hypothesis_persona(self, context: dict[str, Any]) -> str:
        """根据 context + surprise + memory 选择 persona.
        高 surprise → 切换到 reviewer persona, 更批判地审视上轮意外结果.
        否则按内容走 DFT/MD 专家.

        深化: 查 memory 看 reviewer persona 历史效果, 如果上次 reviewer
        找到了问题(r_phys高), 倾向继续用 reviewer. 这是 persona→memory→persona
        闭环的关键一环."""
        # JEPA: 上轮预测误差大时, 用 reviewer persona 审视 —
        # 预测错了说明 agent 的心智模型不准, 需要更批判的视角.
        if getattr(self, "_last_surprise", 0.0) > 0.6:
            return "reviewer"

        # 查 memory: 上次 reviewer persona 效果如何?
        try:
            if self.memory:
                recall = self.memory.recall_for_prompt("reviewer persona autoloop", max_entries=3)
                if recall and "r_phys" in recall.lower():
                    # 如果 memory 里记录 reviewer 找到了问题, 倾向继续用
                    import re
                    scores = re.findall(r'r_phys[:\s]+([\d.]+)', recall)
                    if scores:
                        avg_score = sum(float(s) for s in scores) / len(scores)
                        if avg_score > 0.6:
                            return "reviewer"  # reviewer 历史效果好, 继续用
        except Exception:
            pass

        blob = json.dumps(context, ensure_ascii=False).lower()
        md_markers = ("md", "lammps", "molecular dynamics", "nvt", "npt", "md_steps")
        if any(m in blob for m in md_markers):
            return "md_expert"
        return "dft_expert"

    async def _plan(self, hypothesis: str, context: dict[str, Any]) -> dict[str, Any] | None:
        """Generate a plan from hypothesis and persist it to PlanStore.

        以前只返回一个临时 dict, turn 结束就丢了. 现在往 PlanStore 落一份,
        跨会话可恢复, 用户也能 confirm/reject. PlanStore 不可用时退回老行为.
        """
        prompt = self._build_plan_prompt(hypothesis, context)
        try:
            response = await self._llm_chat(prompt, persona_name="default", task="reasoning")
            plan = self._parse_plan(response)
        except Exception:
            return None

        if not plan:
            return None

        # B: 硬路由 — 根据上下文信号覆盖 LLM 的 mode 选择
        plan = self._override_plan_mode(plan)

        # 落 PlanStore: 创建 plan → cost 确认门 → confirm/reject
        plan_store = self._get_plan_store()
        if plan_store is None:
            return plan

        try:
            from huginn.autoloop.plan_store import PlanStep

            steps = [
                PlanStep(
                    id="step_0",
                    description=plan.get("description", ""),
                    tool=plan.get("mode", ""),
                )
            ]
            persisted = plan_store.create_plan(
                objective=hypothesis,
                steps=steps,
                auto_confirm=False,
                metadata={"mode": plan.get("mode", ""), "source": "autoloop"},
            )

            # cost 确认门: None = 不用问 (非高成本 mode / manager 不可用),
            # 直接放行; 显式拒绝才拦. bool 和字符串都兼容 (测试 mock 常传 bool)
            answer = await self._maybe_clarify("plan", plan)
            if answer is None:
                should_confirm = True
            elif isinstance(answer, bool):
                should_confirm = answer
            elif isinstance(answer, str):
                should_confirm = answer.lower().strip() not in (
                    "no", "n", "cancel", "reject", "decline", "stop", "abort",
                )
            else:
                should_confirm = bool(answer)

            if should_confirm:
                plan_store.confirm_plan(persisted.id)
                plan_store.mark_executing(persisted.id)
            else:
                plan_store.reject_plan(persisted.id, reason="user declined")
                return None

            plan["plan_id"] = persisted.id
        except Exception as e:
            logger.warning("plan store persistence failed: %s", e)
            # PlanStore 挂了不阻塞执行, 退回老的纯 dict 路径

        return plan

    async def _execute(self, plan: dict[str, Any], context: dict[str, Any]) -> Any:
        """Execute the plan using the appropriate sub-engine."""
        mode = plan.get("mode", "coder")
        description = plan.get("description", "")

        if mode == "coder":
            # Use CoderRunner to modify code
            result = await self._execute_coder(description, context)
        elif mode == "workflow":
            # Use WorkflowEngine to run computational pipeline
            result = await self._execute_workflow(description, context)
            # On failure, try applying a learned heuristic fix before giving up
            if isinstance(result, dict) and not result.get("success", True):
                result = await self._try_evolved_fix(mode, description, result) or result
        elif mode == "dynamic_workflow":
            # A5: agent 写的并行 subtask 脚本, orchestrator 并发跑
            result = await self._execute_dynamic_workflow(plan, context)
        elif mode == "explore":
            # Use ExplorationOrchestrator to search design space
            result = await self._execute_explore(description, context)
        elif mode == "skill":
            # Run a pre-built composite skill pipeline
            result = await self._execute_skill(plan, context)
        elif mode == "visual_inspect":
            # Path C: interactive visual inspection using existing visual tools
            result = await self._execute_visual_inspect(description, context)
        else:
            raise ValueError(f"Unknown plan mode: {mode}")

        # provenance: 记一次 tool call, mode 当工具名, plan 当输入参数
        self._record_provenance(mode, plan, result)
        # 缓存给 _build_plan_prompt 的 pipeline suggest_next 用
        self._last_execution_result = {
            '_tool_name': mode,
            '_tool_input': plan,
            'result': result if isinstance(result, dict) else {'value': str(result)[:500]},
        }
        return result

    def _record_provenance(
        self, tool_name: str, input_params: dict[str, Any], output: Any
    ) -> None:
        """往当前 run 的 provenance record 追加一次 tool-call 快照.

        run() 启动时建好 self._provenance_record; 没建 (比如单测里直接调
        _execute) 就跳过, 不强求调用方先 setup. provenance 是 best-effort,
        快照挂了不能把 execute 带挂.
        """
        record = getattr(self, "_provenance_record", None)
        if record is None:
            return
        try:
            from huginn.provenance import capture
            record.add_snapshot(capture(tool_name, input_params, output=output))
        except Exception:
            logger.warning("error in _record_provenance: capture snapshot failed", exc_info=True)

    async def _try_evolved_fix(
        self, tool_name: str, tool_input: dict[str, Any], error_result: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Check if the evolution engine has a learned fix for this error.

        This is the other half of the Learn→Execute loop: when a tool fails,
        we ask evolution if it's seen this error before and has a fix.
        Returns a patched result dict on hit, None on miss.
        """
        try:
            evolution = self._get_evolution()
            error_str = str(error_result.get("error", ""))
            fix = evolution.apply_heuristic_fix(tool_name, tool_input, error_str)
            if fix:
                patched_desc = fix.get("description", str(tool_input))
                return await self._execute_workflow(
                    patched_desc, {"_evolved_fix": True}
                )
        except Exception:
            logger.warning("error in _try_evolved_fix: apply_heuristic_fix failed", exc_info=True)
        return None

    async def _execute_dynamic_workflow(
        self, plan: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, Any]:
        """A5: 跑 agent 提交的并行工作流脚本.

        plan 里带 "script" 字段 (WorkflowScript.from_dict 的输入), 直接走
        WorkflowOrchestrator.run() 同步等完. 失败的 subtask 不炸整体,
        返回聚合结果让 validate/learn 阶段看.
        """
        from huginn.autoloop.dynamic_workflow import (
            WorkflowOrchestrator,
            WorkflowScript,
        )
        from huginn.types import ToolContext

        raw_script = plan.get("script") or {}
        if isinstance(raw_script, str):
            # agent 可能传 JSON 字符串
            import json
            try:
                raw_script = json.loads(raw_script)
            except json.JSONDecodeError:
                raw_script = {}
        script = WorkflowScript.from_dict(raw_script)
        if not script.subtasks:
            return {
                "mode": "dynamic_workflow",
                "success": False,
                "error": "脚本无有效 subtask",
            }
        orch = WorkflowOrchestrator(
            max_concurrent=script.max_concurrent,
        )
        ctx = ToolContext(
            session_id=f"dynwf_{script.id}",
            workspace=str(self.workspace),
            config=self.settings,
        )
        result = await orch.run(script, ctx)
        return {
            "mode": "dynamic_workflow",
            "success": result.success,
            "workflow_id": result.id,
            "n_total": result.n_total,
            "n_completed": result.n_completed,
            "n_failed": result.n_failed,
            "summary": result.summary(),
        }

    async def _validate(self, execution_result: Any) -> dict[str, Any]:
        """Validate execution results using benchmarks and constraints."""
        results = {
            "tests_passed": False,
            "constraints_satisfied": False,
            "benchmarks": {},
        }

        if isinstance(execution_result, dict):
            # Extract visual primitives from tool output — the deictic pointers
            # that let the next iteration's hypothesis/plan reason about data
            # shape without needing image input (Mirage + Visual Primitives).
            visual_hint = execution_result.get("_visual_hint")
            if visual_hint:
                results["visual_primitives"] = visual_hint
                self._last_visual_context = visual_hint

            # 比较性视觉原语: 把本轮结果和上轮做差分, 突出变化.
            # 峰值位移/新异常/趋势反转 — 这些是 agent 最关心的信号.
            prev_exec = getattr(self, '_last_execution_result', None)
            if prev_exec and isinstance(prev_exec.get('result'), dict):
                try:
                    from huginn.tools.visual_hook import extract_comparative_primitives
                    comp = extract_comparative_primitives(
                        prev_exec.get('result', {}), execution_result
                    )
                    if comp:
                        results["comparative_primitives"] = comp
                        # 也拼进 visual_context, 下轮 hypothesis 能看到
                        self._last_visual_context = (
                            f"{self._last_visual_context}\n{comp}".strip()
                            if self._last_visual_context else comp
                        )
                except Exception:
                    pass

            r_phys = execution_result.get("r_phys")
            if r_phys is None:
                result_type = execution_result.get("result_type")
                result_data = execution_result.get("result_data")
                if result_type and result_data:
                    try:
                        from huginn.tools.validate_tool import (
                            ValidateTool,
                            ValidateToolInput,
                        )

                        validator = ValidateTool()
                        tool_ctx = ToolContext(
                            session_id=f"validate_{uuid.uuid4().hex[:8]}",
                            workspace=str(self.workspace),
                            config=self.settings,
                        )
                        vr = await validator.call(
                            ValidateToolInput(
                                result_type=result_type,
                                result_data=result_data,
                            ),
                            tool_ctx,
                        )
                        if vr.success and vr.data:
                            r_phys = vr.data.get("r_phys")
                            results["physics_validation"] = vr.data
                    except Exception as e:
                        results["physics_validation_error"] = str(e)
            if r_phys is not None:
                results["r_phys"] = r_phys

        try:
            collapse = self._detect_thinking_collapse(execution_result)
            if collapse:
                results["thinking_collapse"] = collapse
        except Exception as e:
            results["thinking_collapse_error"] = str(e)

        # pytest, benchmark, math validation — sync, offload to thread
        py_test, bench_report, math_val = await asyncio.gather(
            self._run_pytest(),
            self._run_benchmark(),
            self._run_math_validation(execution_result),
            return_exceptions=True,
        )

        if isinstance(py_test, dict):
            results.update(py_test)
        elif isinstance(py_test, Exception):
            results["test_output"] = f"Test execution error: {py_test}"

        if isinstance(bench_report, dict):
            results["benchmarks"] = bench_report
        elif isinstance(bench_report, Exception):
            logger.warning("BenchmarkRunner failed", exc_info=True)

        if isinstance(math_val, dict):
            results["math_validation"] = math_val
        elif isinstance(math_val, Exception):
            results["math_validation_error"] = str(math_val)

        try:
            math_ev = await self._collect_math_evidence(
                execution_result, results.get("math_validation", {})
            )
            for _k, _v in math_ev.items():
                results[_k] = _v
        except Exception as e:
            results["math_evidence_error"] = str(e)

        # Conditional verification: run cheap generative_verify first,
        # only call expensive reviewer critique when score < 0.5.
        # ponytail: _generative_verify 依赖 results (被 math_evidence 修改过),
        # 不能和 _collect_math_evidence 并行. 但可以和 emergent_complexity/
        # literature_comparison 并行 — 它们在下面已经 gather 了.
        gen_verify = None
        try:
            gen_verify = await self._generative_verify(execution_result, results)
            if gen_verify:
                results["generative_verify"] = gen_verify
        except Exception as e:
            results["generative_verify_error"] = str(e)

        needs_review = gen_verify is None or gen_verify.get("score", 0.5) < 0.5
        if needs_review:
            try:
                reviewer_kb = self._build_kb_text(
                    query=self._summarize_for_kb(execution_result, results)
                )
                critique = await self._llm_chat(
                    self._build_reviewer_prompt(execution_result, results, reviewer_kb),
                    persona_name="reviewer",
                    model=self.verification_model,
                )
                if critique and critique.strip():
                    results["reviewer_critique"] = critique.strip()
            except Exception as e:
                results["reviewer_critique_error"] = str(e)

        # emergent complexity + literature + grader + eval — independent
        ec_task = asyncio.create_task(self._safe_emergent_complexity(execution_result, results))
        lit_task = asyncio.create_task(self._safe_literature_comparison(execution_result, results))
        await asyncio.gather(ec_task, lit_task)

        try:
            from huginn.validation.grader import default_registry
            reg = default_registry()
            merged: dict[str, Any] = {}
            if isinstance(execution_result, dict):
                merged.update(execution_result)
            merged.update(results)
            grader_list = reg.evaluate_all(merged)
            results["grader_scores"] = {
                gr.name: {
                    "score": gr.score,
                    "passed": gr.passed,
                    "message": gr.message,
                }
                for gr in grader_list
            }
            if grader_list:
                avg_score = sum(gr.score for gr in grader_list) / len(grader_list)
                results["grader_reward"] = round(avg_score, 4)
                try:
                    from huginn.events.integration import _publish
                    from huginn.utils.concurrency import track_task
                    asyncio.get_running_loop()
                    track_task(_publish("quality.check", {
                        "iteration": self._iteration,
                        "graders": results["grader_scores"],
                        "reward": results.get("grader_reward", 0),
                    }, source="autoloop"), name="quality-check-emit")
                except Exception:
                    logger.debug("quality.check emit failed", exc_info=True)
        except Exception as e:
            results["grader_error"] = str(e)

        try:
            from huginn.evaluation.matworld_bench import MatWorldBench
            bench = MatWorldBench()
            exec_data = execution_result if isinstance(execution_result, dict) else {}
            eval_scores: list[dict] = []
            for task in bench.tasks:
                if task.category in ("structure", "thermo", "electronic"):
                    try:
                        br = bench.evaluate(task.id, exec_data)
                        eval_scores.append({
                            "task_id": task.id,
                            "category": task.category,
                            "passed": br.passed,
                            "score": br.score,
                        })
                    except Exception:
                        pass
            if eval_scores:
                passed = sum(1 for e in eval_scores if e["passed"])
                results["eval_summary"] = {
                    "bench_passed": passed,
                    "bench_total": len(eval_scores),
                    "bench_pass_rate": round(passed / len(eval_scores), 4),
                    "details": eval_scores,
                }
        except Exception as e:
            logger.debug(f"[validate] eval bench failed: {e}")

        # JEPA 式预测误差: 对比 plan 阶段的预测 vs 实际结果.
        # 预测误差高 = surprise = 值得探索的方向 (intrinsic motivation).
        # 低误差 = agent 对这类任务已有良好心智模型.
        prediction = getattr(self, "_current_prediction", "")
        if prediction:
            actual_text = self._extract_text(execution_result)[:500]
            robust = self._compute_surprise_robust(prediction, actual_text)
            surprise = robust["worst"]
            results["prediction_error"] = {
                "predicted": prediction[:200],
                "actual": actual_text[:200],
                "surprise": round(surprise, 3),
                "surprise_mean": round(robust["mean"], 3),
                "surprise_worst": round(robust["worst"], 3),
                "surprise_std": round(robust["std"], 3),
            }
            self._last_surprise = surprise
            self._surprise_history.append((surprise, robust["std"]))

        self._last_validation = json.dumps(results, ensure_ascii=False, default=str)[:1000]
        # Store failure_mode for next hypothesis loop (Dream Layer: crash = discovery)
        _gv = results.get("generative_verify", {})
        if isinstance(_gv, dict):
            self._last_failure_mode = _gv.get("failure_mode", "")
        return results

    async def _run_pytest(self) -> dict[str, Any]:
        """Run pytest in workspace, return results dict."""
        import subprocess
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                ["python", "-m", "pytest", "-x", "-q", "--tb=line"],
                cwd=self.workspace,
                capture_output=True,
                text=True,
                timeout=60,
            )
            return {
                "tests_passed": result.returncode == 0,
                "test_output": result.stdout + result.stderr,
            }
        except Exception as e:
            return {"test_output": f"Test execution error: {e}"}

    async def _run_benchmark(self) -> dict[str, Any]:
        """Run BenchmarkRunner, return results dict."""
        try:
            from huginn.validation.benchmarks import BenchmarkRunner
            runner = BenchmarkRunner()
            report = await asyncio.to_thread(
                runner.run, categories=["math", "coding"]
            )
            return {
                "passed": report.passed,
                "failed": report.failed,
                "skipped": report.skipped,
            }
        except Exception:
            logger.warning("BenchmarkRunner failed", exc_info=True)
            return {}

    async def _safe_emergent_complexity(
        self, execution_result: Any, results: dict[str, Any]
    ) -> None:
        """Compute emergent complexity, mutate results in place."""
        try:
            from huginn.validation.emergent_complexity import compute_ec
            results["emergent_complexity"] = compute_ec(execution_result, results)
            ec_score = results["emergent_complexity"].get("ec_score", 0)
            if ec_score < 0.2 and self._iteration > 0:
                ec_hint = f"EC={ec_score:.2f}: low emergent complexity, try diverse tools or cross-domain reasoning"
                self._speculator_hint = (
                    (self._speculator_hint + "\n" + ec_hint).strip()
                    if self._speculator_hint else ec_hint
                )
        except Exception as e:
            results["emergent_complexity_error"] = str(e)

    async def _safe_literature_comparison(
        self, execution_result: Any, results: dict[str, Any]
    ) -> None:
        """Run literature comparison, mutate results in place."""
        try:
            lit_comp = await self._literature_comparison(execution_result)
            if lit_comp:
                results["literature_comparison"] = lit_comp
        except Exception as e:
            results["literature_comparison_error"] = str(e)

    async def _literature_comparison(self, execution_result: Any) -> dict[str, Any]:
        """Extract numeric results, look up literature benchmarks, run innovation
        signal detection. Best-effort — any failure just skips that property.

        Returns {property_key: InnovationSignal} for properties where literature
        data was found. Empty dict if nothing to compare.
        """
        if not isinstance(execution_result, dict):
            return {}

        result_data = (
            execution_result.get("result_data")
            or execution_result.get("parsed")
            or {}
        )
        if not isinstance(result_data, dict):
            return {}

        # system / formula: benchmark_lookup 必须知道查什么材料
        system = None
        for key in ("formula", "system", "material", "compound"):
            val = result_data.get(key) or execution_result.get(key)
            if val:
                system = str(val)
                break
        if not system:
            return {}

        # 抽数值: 扁平 key + 嵌套 lattice_params
        numerics: dict[str, float] = {}
        for key in _LIT_PROPERTY_MAP:
            val = result_data.get(key)
            if val is not None:
                try:
                    numerics[key] = float(val)
                except (TypeError, ValueError):
                    logger.warning("error in _literature_comparison: numeric property cast failed", exc_info=True)
        lattice = result_data.get("lattice_params") or {}
        if isinstance(lattice, dict):
            for param in ("a", "b", "c"):
                val = lattice.get(param)
                if val is not None:
                    try:
                        numerics[f"lattice_{param}"] = float(val)
                    except (TypeError, ValueError):
                        logger.warning("error in _literature_comparison: lattice param cast failed", exc_info=True)

        if not numerics:
            return {}

        try:
            from huginn.tools.literature import LiteratureInput, LiteratureTool
            from huginn.validation.innovation_signal import InnovationSignalDetector
        except ImportError:
            return {}

        tool = LiteratureTool()
        tool_ctx = ToolContext(
            session_id=f"litcmp_{uuid.uuid4().hex[:8]}",
            workspace=str(self.workspace),
            config=self.settings,
        )
        detector = InnovationSignalDetector()

        comparison: dict[str, Any] = {}
        for prop_key, agent_value in numerics.items():
            prop_name = _LIT_PROPERTY_MAP.get(prop_key, prop_key)
            try:
                res = await tool.call(
                    LiteratureInput(
                        action="benchmark_lookup",
                        system=system,
                        property=prop_name,
                        max_results=10,
                    ),
                    tool_ctx,
                )
                if not res.success or not res.data:
                    continue
                reported = res.data.get("reported_values") or []
                lit_values = [r["value"] for r in reported if "value" in r]
                if not lit_values:
                    continue
                signal = detector.detect(prop_key, agent_value, lit_values)
                comparison[prop_key] = signal
            except Exception:
                continue

        return comparison

    @staticmethod
    def _summarize_for_kb(execution_result: Any, results: dict[str, Any]) -> str:
        """把 execution_result + validation results 拍扁成短串当 KB query.
        给 reviewer 检索已知 first-principles 结论用, 失败无所谓."""
        try:
            parts: list[str] = []
            if isinstance(execution_result, dict):
                for k in ("result_type", "equations", "lagrangian", "summary"):
                    v = execution_result.get(k)
                    if v:
                        parts.append(str(v)[:120])
            for k in ("tests_passed", "constraints_satisfied"):
                v = results.get(k)
                if v is not None:
                    parts.append(f"{k}={v}")
            return " ".join(parts)[:400]
        except Exception:
            return ""

    # -- 思维坍塌检测 + 生成式验证 --

    def _detect_thinking_collapse(self, execution_result: Any) -> dict[str, Any] | None:
        """检查 LLM 输出是否陷入重复推理 / 发散 / 工具调用循环.

        三条规则, 纯文本分析不需要 LLM:
          1. 相同短语 (5 词 n-gram) 出现 3+ 次 → 重复推理路径
          2. 输出 > 200 词但 unique word ratio < 0.3 → 发散但不前进
          3. 相同工具 + 相同参数出现 2+ 次 → 工具循环

        检测到任一信号就返回 dict, 否则 None.
        """
        from collections import Counter

        text = self._extract_text(execution_result)
        if not text or len(text.strip()) < 20:
            return None

        signals: dict[str, Any] = {}

        # Rule 1: 重复短语 — 5 词 n-gram 出现 3+ 次
        words = text.lower().split()
        if len(words) >= 10:
            ngrams = [
                " ".join(words[i : i + 5])
                for i in range(len(words) - 4)
            ]
            counts = Counter(ngrams)
            repeated = [(p, c) for p, c in counts.items() if c >= 3]
            if repeated:
                repeated.sort(key=lambda x: -x[1])
                signals["repeated_phrases"] = repeated[:5]

        # Rule 2: 发散推理 — 长文本但词汇丰富度低
        if len(words) > 200:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:
                signals["divergent_reasoning"] = {
                    "word_count": len(words),
                    "unique_ratio": round(unique_ratio, 3),
                }

        # Rule 3: 工具调用循环 — 相同工具 + 相同参数 2+ 次
        if isinstance(execution_result, dict):
            loops = self._find_tool_call_loops(execution_result)
            if loops:
                signals["tool_call_loops"] = loops

        if not signals:
            return None

        # 严重度: 有重复短语或工具循环 = high, 只有发散 = medium
        has_loop = bool(signals.get("tool_call_loops"))
        has_repeat = bool(signals.get("repeated_phrases"))
        signals["severity"] = "high" if (has_loop or has_repeat) else "medium"
        return signals

    @staticmethod
    def _find_tool_call_loops(execution_result: dict) -> list[dict[str, Any]]:
        """从 execution_result 里找重复的工具调用 (同工具 + 同参数 2+ 次)."""
        import hashlib

        calls = (
            execution_result.get("tool_calls")
            or execution_result.get("steps")
            or execution_result.get("actions")
            or []
        )
        if not isinstance(calls, list):
            return []

        seen: dict[str, int] = {}
        for call in calls:
            if not isinstance(call, dict):
                continue
            name = call.get("tool") or call.get("name") or call.get("action") or ""
            params = call.get("input") or call.get("params") or call.get("args") or {}
            try:
                payload = name + json.dumps(params, sort_keys=True, default=str)
            except Exception:
                payload = name + str(params)
            key = hashlib.sha256(payload.encode()).hexdigest()[:12]
            seen[key] = seen.get(key, 0) + 1

        return [
            {"call_hash": k, "count": c}
            for k, c in seen.items()
            if c >= 2
        ]

    @staticmethod
    def _extract_text(execution_result: Any) -> str:
        """从 execution_result 里抽文本, 给坍塌检测做分析用."""
        if execution_result is None:
            return ""
        if isinstance(execution_result, str):
            return execution_result
        if not isinstance(execution_result, dict):
            return str(execution_result)

        parts: list[str] = []
        for key in ("summary", "description", "result_data", "output",
                     "error", "reasoning", "plan", "hypothesis"):
            v = execution_result.get(key)
            if v:
                parts.append(str(v))
        # 嵌套的 steps / tool_calls 里的文本也抽出来
        for key in ("steps", "tool_calls", "actions"):
            items = execution_result.get(key)
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        for sk in ("description", "output", "result", "error"):
                            sv = item.get(sk)
                            if sv:
                                parts.append(str(sv))
        return " ".join(parts)

    def _compute_surprise(self, prediction: str, actual: str) -> float:
        """JEPA 式预测误差: 预测文本 vs 实际文本的语义距离.

        ponytail: 用关键词 Jaccard 距离代替真正的嵌入余弦距离.
        纯文本操作, 零依赖, 零 LLM 调用. 对于"预测说了 energy, 实际也出了
        energy"这种常见场景已经够用. 升级路径: 用 sentence-transformers
        算 cosine distance, 或训练专门的 JEPA 编码器.
        """
        s = self._compute_surprise_robust(prediction, actual)
        return s["worst"]

    def _compute_surprise_robust(
        self, prediction: str, actual: str
    ) -> dict[str, float]:
        """分布鲁棒 surprise 估计.

        对 keyword 提取做多种扰动 (不同 stopword 集 / n-gram / 阈值),
        取 worst-case 作为决策依据. 这避免单一扰动下 surprise 被低估.

        返回 {mean, worst, std, point}:
        - point: 原始 Jaccard 距离 (兼容旧逻辑)
        - worst: 多扰动下的最大值 (决策用)
        - mean: 多扰动平均值 (趋势分析用)
        - std: 多扰动标准差 (置信度信号)
        """
        if not prediction or not actual:
            return {"mean": 0.0, "worst": 0.0, "std": 0.0, "point": 0.0}

        import statistics

        # 扰动 1: 标准停用词集
        stop1 = {"the", "a", "an", "is", "are", "was", "were", "be", "to", "of",
                 "in", "on", "at", "for", "and", "or", "not", "this", "that",
                 "it", "with", "from", "by", "as", "will", "can", "may"}
        # 扰动 2: 更激进的停用词集 (去掉更多常见词)
        stop2 = stop1 | {"energy", "result", "value", "system", "model",
                         "data", "using", "shown", "show", "also", "which",
                         "has", "have", "had", "been", "were", "more", "than"}
        # 扰动 3: 只保留长关键词 (>=5 chars)
        # 扰动 4: bigram Jaccard

        def keywords(text: str, stop: set[str], min_len: int = 3) -> set[str]:
            words = __import__("re").findall(r'[a-zA-Z_]\w{2,}', text.lower())
            return {w for w in words if w not in stop and len(w) >= min_len}

        def bigrams(text: str) -> set[str]:
            words = __import__("re").findall(r'[a-zA-Z_]\w{2,}', text.lower())
            return {f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1)}

        def jaccard(a: set, b: set) -> float:
            if not a and not b:
                return 0.0
            union = a | b
            if not union:
                return 0.0
            return 1.0 - len(a & b) / len(union)

        pred_kw1 = keywords(prediction, stop1)
        actual_kw1 = keywords(actual, stop1)
        pred_kw2 = keywords(prediction, stop2)
        actual_kw2 = keywords(actual, stop2)
        pred_kw3 = keywords(prediction, stop1, min_len=5)
        actual_kw3 = keywords(actual, stop1, min_len=5)
        pred_bg = bigrams(prediction)
        actual_bg = bigrams(actual)

        estimates = [
            jaccard(pred_kw1, actual_kw1),
            jaccard(pred_kw2, actual_kw2),
            jaccard(pred_kw3, actual_kw3),
            jaccard(pred_bg, actual_bg) if pred_bg or actual_bg else 0.0,
        ]

        return {
            "point": estimates[0],
            "mean": statistics.mean(estimates),
            "worst": max(estimates),
            "std": statistics.stdev(estimates) if len(estimates) > 1 else 0.0,
        }

    async def _generative_verify(
        self, execution_result: Any, results: dict[str, Any]
    ) -> dict[str, Any] | None:
        """用 verification_model 给 agent 输出打 0-1 质量分.

        分数 < 0.5 标记 needs_retry. verification_model 不可用时返回
        None, 让上层降级到规则检查 (thinking_collapse 等).
        """
        if self.verification_model is None:
            return None

        text = self._extract_text(execution_result)
        if not text or len(text.strip()) < 10:
            return None

        # 截断防止 prompt 爆炸
        snippet = text[:2000]
        collapse = results.get("thinking_collapse", {})
        collapse_hint = ""
        if collapse:
            collapse_hint = (
                f"\nNote: automated checks detected: {json.dumps(collapse, default=str)[:300]}"
            )

        # 注入历史记忆, 检查本次结果是否与历史迭代结果矛盾
        memory_hint = ""
        try:
            _mem_text = self._build_memory_text(query=snippet[:200])
            if _mem_text:
                memory_hint = (
                    f"\nPast iterations memory:\n{_mem_text}\n"
                    "Cross-check: does the current result contradict any historical finding above?\n"
                    "If yes, note the contradiction in 'reason'.\n"
                )
        except Exception:
            logger.debug("_build_memory_text failed — validate prompt missing cross-check", exc_info=True)

        prompt = (
            "You are a verification model. Score the quality of this agent output "
            "from 0.0 to 1.0.\n"
            "1.0 = well-reasoned, complete, correct.\n"
            "0.5 = acceptable but has issues.\n"
            "0.0 = poor, incorrect, or incomplete.\n"
            "Also check evidence chain (RCBench failure mode: evidence mismatch):\n"
            "- Does the conclusion have specific data/numbers supporting it?\n"
            "- Are claims grounded in the execution results, not assumed?\n"
            "- Is there a clear data→inference→conclusion link?\n"
            "Also describe failure mode (Dream Layer insight: how it crashes = new discovery):\n"
            "- If this hypothesis is WRONG, in what specific way would it fail?\n"
            "- What would the system look like if the opposite were true?\n"
            f"{collapse_hint}{memory_hint}\n\n"
            f"Agent output:\n{snippet}\n\n"
            "Respond with ONLY a JSON object: "
            '{"score": <float>, "evidence_score": <float 0-1>, '
            '"reason": "<brief>", "evidence_gap": "<what data is missing>", '
            '"failure_mode": "<how it would crash if wrong>"}'
        )

        resp = await self._llm_chat(prompt, model=self.verification_model)
        score, reason, evidence_score, evidence_gap, failure_mode = self._parse_verify_score(resp)

        return {
            "score": score,
            "reason": reason,
            "needs_retry": score < 0.5,
            "evidence_score": evidence_score,
            "evidence_gap": evidence_gap,
            "failure_mode": failure_mode,
        }

    @staticmethod
    def _parse_verify_score(resp: str) -> tuple[float, str, float, str, str]:
        """Parse score, reason, evidence_score, evidence_gap, failure_mode from LLM response."""
        import re

        if not resp:
            return 0.5, "empty response", 0.5, "", ""

        # try JSON first
        try:
            data = json.loads(resp.strip())
            score = float(data.get("score", 0.5))
            reason = str(data.get("reason", ""))
            ev_score = float(data.get("evidence_score", 0.5))
            ev_gap = str(data.get("evidence_gap", ""))
            fail_mode = str(data.get("failure_mode", ""))
            return score, reason, ev_score, ev_gap, fail_mode
        except (json.JSONDecodeError, ValueError):
            logger.warning("error in _parse_verify_score: JSON parse failed, falling back to regex", exc_info=True)

        # fallback: regex for first float
        m = re.search(r"([01]\.\d+|[01])\b", resp)
        if m:
            return float(m.group(1)), resp[:200], 0.5, "", ""

        return 0.5, resp[:200], 0.5, "", ""

    async def _run_math_validation(self, execution_result: Any) -> dict[str, Any]:
        """把执行结果里的数学结构抽出来, 用数学工具做形式化校验.

        三个独立子项, 互不影响:
          A. 守恒律 (BourbakiTool.check_conservation) — equations 非空时跑
          B. 变分原理 (LeanTool.constitutive/variational_principle) — lagrangian 非空时跑
          C. 自动微分 (AutoDiffTool.gradient) — function spec 齐全时跑

        工具懒加载, 任一缺失/报错只记 *_error, 不阻断其余子项与主 validate 流程.
        engine 没有自己的 tool_registry, 这里直接构造工具实例 (它们都是无状态轻量构造).
        """
        from huginn.types import ToolContext

        out: dict[str, Any] = {}
        if not isinstance(execution_result, dict):
            return out

        tool_ctx = ToolContext(
            session_id=f"mathval_{uuid.uuid4().hex[:8]}",
            workspace=str(self.workspace),
            config=self.settings,
        )

        equations = execution_result.get("equations") or ""
        lagrangian = execution_result.get("lagrangian") or ""
        coords = execution_result.get("coordinates") or []
        velocities = execution_result.get("velocities")
        domain = execution_result.get("conservation_domain") or "continuum_mechanics"
        if equations:
            try:
                from huginn.tools.bourbaki_tool import BourbakiTool

                tool = BourbakiTool()
                raw = await tool.call(
                    {
                        "task": "check_conservation",
                        "domain": domain,
                        "equations": equations,
                    },
                    tool_ctx,
                )
                # BourbakiTool.call 可能返回 dict 或 BourbakiResult; 统一成 dict
                if hasattr(raw, "model_dump"):
                    raw = raw.model_dump()
                out["conservation"] = {
                    "verified": raw.get("verified"),
                    "message": raw.get("message", ""),
                    "fallback": raw.get("fallback", False),
                    "method": "bourbaki",
                }
            except Exception as e:
                out["conservation_error"] = str(e)

        # A2: KB 交叉验证 — 把守恒律方程 + Lagrangian 关键词拿去查 KB, 命中的
        # first-principles 参考块作为 reference_principles 写回, 让下游 reviewer
        # 能对照已知结论. KB 不可用/空查询都不阻断, 只是不写该字段.
        kb_ref = self._query_kb_reference(equations, lagrangian)
        if kb_ref:
            out["reference_principles"] = kb_ref

        if lagrangian and coords:
            try:
                from huginn.tools.lean_tool import LeanTool, LeanToolInput

                tool = LeanTool()
                args = LeanToolInput(
                    action="constitutive",
                    sub_action="variational_principle",
                    lagrangian=lagrangian,
                    coordinates=list(coords),
                    velocities=velocities,
                )
                vr = await tool.call(args, tool_ctx)
                out["variational"] = {
                    "ok": bool(vr.success),
                    "data": vr.data,
                    "error": vr.error,
                    "method": "lean",
                }
            except Exception as e:
                out["variational_error"] = str(e)

        func_spec = execution_result.get("autodiff")
        if isinstance(func_spec, dict) and func_spec.get("function_type"):
            try:
                from huginn.tools.sci.autodiff_tool import (
                    AutoDiffInput,
                    AutoDiffTool,
                )

                tool = AutoDiffTool()
                args = AutoDiffInput(
                    action="gradient",
                    function_type=func_spec.get("function_type", "custom"),
                    function_params=func_spec.get("function_params", {}),
                    variables=func_spec.get("variables", {}),
                    target_variable=func_spec.get("target_variable"),
                )
                vr = await tool.call(args, tool_ctx)
                out["autodiff"] = {
                    "ok": bool(vr.success),
                    "data": vr.data,
                    "error": vr.error,
                }
            except Exception as e:
                out["autodiff_error"] = str(e)

        return out

    def _verify_via_gp(self, hyp_id: str, validation: dict) -> dict:
        """循环B: 用 GP 数值验证做独立路径. 与符号演绎 (循环A) 基底正交.

        升级: fit + leave-one-out 风格 predict, 检查后验均值与实验值
        的偏差是否在 ±2σ 内. 若有测试集 (X_test, y_test) 则用之, 否则
        在训练集上做 LOO-style 检查 (GP 在 fit 数据上 predict 时, sigma
        较小但仍有信号).

        ponytail: ±2σ 是 ~95% 置信区间. 拒绝域 = 5%. 升级路径:
        对假设的 testable_prediction 做数值区间解析, 用 KL(GP_posterior
        || hypothesis_interval) 代替 ±2σ 检查.
        """
        exec_res = getattr(self, "_last_execution_result", None)
        X = y = X_test = y_test = None
        for cand in (validation, exec_res if isinstance(exec_res, dict) else {}):
            X = cand.get("X") or cand.get("x_data") or cand.get("samples")
            y = cand.get("y") or cand.get("y_data") or cand.get("targets")
            X_test = cand.get("X_test") or cand.get("x_test")
            y_test = cand.get("y_test") or cand.get("y_true")
            if X is not None and y is not None:
                break
            X = y = None
        if X is None or y is None:
            return {"agrees": False, "reason": "no numeric data for GP fit"}

        try:
            import numpy as np
            from huginn.tools.sci.gp_tool import GPTool

            tool = GPTool()

            # 若有独立测试集, predict 在 X_test 上, 与 y_test 比对
            # 否则 fit 后 predict 在 X 上做自洽检查 (弱信号, sigma 小)
            pred_X = X_test if X_test is not None else X
            pred_y_ref = y_test if y_test is not None else y

            predict_res = tool.call({
                "action": "predict",
                "X": X, "y": y,
                "X_new": pred_X,
            })
            if not getattr(predict_res, "success", False):
                return {
                    "agrees": False,
                    "reason": "GP predict failed",
                    "error": getattr(predict_res, "error", ""),
                }

            data = getattr(predict_res, "data", None) or {}
            mu = np.asarray(data.get("mean", []), dtype=float)
            sigma = np.asarray(data.get("std", []), dtype=float)
            y_ref = np.asarray(pred_y_ref, dtype=float)

            # 后验一致检验: |y - mu| <= 2σ (95% CI)
            # sigma=0 时退化为 |y - mu| < eps (GP 完全过拟合)
            n = min(len(mu), len(y_ref))
            if n == 0:
                return {"agrees": True, "gp_fit": data,
                        "reason": "GP fit ok, no comparable points"}
            mu, sigma, y_ref = mu[:n], sigma[:n], y_ref[:n]
            eps = 1e-8
            deviation = np.abs(y_ref - mu)
            tolerance = np.maximum(2.0 * sigma, eps)
            agrees = bool(np.all(deviation <= tolerance))
            max_dev = float(np.max(deviation))
            max_tol = float(np.max(tolerance))
            return {
                "agrees": agrees,
                "gp_fit": data,
                "max_deviation": max_dev,
                "max_tolerance": max_tol,
                "n_points": n,
                "reason": (
                    f"posterior ±2σ check: max_dev={max_dev:.4g} "
                    f"vs tol={max_tol:.4g} over {n} points"
                ),
            }
        except Exception as e:
            return {"agrees": False, "reason": f"GP verify error: {e}"}

    async def _collect_math_evidence(
        self, execution_result: Any, math_validation: dict
    ) -> dict[str, Any]:
        """从 execution_result + math_validation 抽 5 个数学证据 key,
        供 PhaseGate 的 MathEvidenceChecker 做 Dempster-Shafer 合成.

        证据来源:
          1. conservation_law — 从 math_validation["conservation"] 透传
          2. dimensional_consistent — execution_result 带 equation 时跑
             symbolic_math_tool action=dimensional_analysis
          3. pde_classification — execution_result 带 pde_coefficients +
             expected_pde_class 时跑 symbolic_math_tool action=pde_classify
          4. sobol_top_features — execution_result 带 sobol_data +
             hypothesis_features 时跑 symbolic_regression_tool action=sobol_indices
          5. constraint_check — execution_result 带 expression + constraints
             时跑 symbolic_regression_tool action=constraint_check

        每项 best-effort: 数据不全/工具报错就跳过, 不写 key (math_checker 忽略缺失).
        """
        evidence: dict[str, Any] = {}
        if not isinstance(execution_result, dict):
            return evidence

        # 1. conservation_law — 从已有 math_validation 透传
        cons = math_validation.get("conservation")
        if isinstance(cons, dict) and "verified" in cons:
            evidence["conservation_law"] = {
                "verified": bool(cons["verified"]),
                "current": cons.get("message", ""),
                "symmetry": cons.get("method", ""),
            }

        from huginn.types import ToolContext

        tool_ctx = ToolContext(
            session_id=f"mathevid_{uuid.uuid4().hex[:8]}",
            workspace=str(self.workspace),
            config=self.settings,
        )

        # 2. dimensional_consistent — 跑量纲分析, 所有 quantity 都能解析 → True
        equation = (
            execution_result.get("equation")
            or execution_result.get("equations")
            or ""
        )
        if equation:
            try:
                from huginn.tools.symbolic_math.tool import (
                    SymbolicMathInput,
                    SymbolicMathTool,
                )

                tool = SymbolicMathTool()
                args = SymbolicMathInput(
                    action="dimensional_analysis",
                    expression=str(equation),
                    target="validate_expression",
                )
                vr = await tool.call(args, tool_ctx)
                if vr.success and vr.data:
                    quantities = vr.data.get("quantities", [])
                    has_error = any("error" in q for q in quantities)
                    evidence["dimensional_consistent"] = (
                        len(quantities) > 0 and not has_error
                    )
            except Exception:
                logger.warning("error in _collect_math_evidence: dimensional_analysis failed", exc_info=True)

        # 3. pde_classification — 跑 pde_classify, 比对 expected vs actual
        pde_coeffs = execution_result.get("pde_coefficients")
        expected_class = execution_result.get("expected_pde_class")
        if pde_coeffs and expected_class:
            try:
                from huginn.tools.symbolic_math.tool import (
                    SymbolicMathInput,
                    SymbolicMathTool,
                )

                tool = SymbolicMathTool()
                args = SymbolicMathInput(
                    action="pde_classify",
                    expression=str(pde_coeffs),
                )
                vr = await tool.call(args, tool_ctx)
                if vr.success and vr.data:
                    actual = vr.data.get("classification", "")
                    evidence["pde_classification"] = {
                        "consistent": actual.lower() == str(expected_class).lower(),
                        "expected": str(expected_class),
                        "actual": actual,
                    }
            except Exception:
                logger.warning("error in _collect_math_evidence: pde_classify failed", exc_info=True)

        # 4. sobol_top_features — 跑 sobol_indices, top features (S_i>0.1) 必须
        # 被 hypothesis_features 覆盖
        sobol_data = execution_result.get("sobol_data")
        hypothesis_features = execution_result.get("hypothesis_features")
        if sobol_data and hypothesis_features:
            try:
                from huginn.tools.sci.symbolic_regression_tool import (
                    SymbolicRegressionInput,
                    SymbolicRegressionTool,
                )

                tool = SymbolicRegressionTool()
                target_col = (
                    sobol_data.get("target", "y")
                    if isinstance(sobol_data, dict)
                    else "y"
                )
                args = SymbolicRegressionInput(
                    action="sobol_indices",
                    data_json=sobol_data,
                    target_column=target_col,
                    n_sobol_samples=512,
                )
                vr = await tool.call(args, tool_ctx)
                if vr.success and vr.data:
                    first_order = vr.data.get("first_order", {})
                    if first_order:
                        top = [f for f, s in first_order.items() if s > 0.1]
                        evidence["sobol_top_features"] = {
                            "hypothesis_covers_top": set(top).issubset(
                                set(hypothesis_features)
                            ),
                            "top_features": top,
                            "hypothesis_features": list(hypothesis_features),
                        }
            except Exception:
                logger.warning("error in _collect_math_evidence: sobol_indices failed", exc_info=True)

        # 5. constraint_check — 跑 constraint_check, 所有先验通过 → all_passed
        expr = execution_result.get("expression")
        constraints = execution_result.get("constraints")
        if expr and constraints:
            try:
                from huginn.tools.sci.symbolic_regression_tool import (
                    SymbolicRegressionInput,
                    SymbolicRegressionTool,
                )

                tool = SymbolicRegressionTool()
                args = SymbolicRegressionInput(
                    action="constraint_check",
                    probe_expression=str(expr),
                    constraints=constraints,
                )
                vr = await tool.call(args, tool_ctx)
                if vr.success and vr.data:
                    evidence["constraint_check"] = {
                        "all_passed": vr.data.get("all_passed", False),
                        "violations": vr.data.get("violations", []),
                    }
            except Exception:
                logger.warning("error in _collect_math_evidence: constraint_check failed", exc_info=True)

        return evidence

    def _query_kb_reference(self, equations: str, lagrangian: str) -> list[dict]:
        """查 KB 拿 first-principles 参考块. 把 equations + lagrangian 拼成
        query 串, 命中返回 [{text, source}], 失败/空都返回 []."""
        query = " ".join(filter(None, [equations, lagrangian])).strip()
        if not query:
            return []
        kb = self._get_kb()
        if kb is None:
            return []
        try:
            if kb.count() == 0:
                return []
            chunks = kb.query(f"conservation law variational {query}", top_k=2)
            return [
                {"text": (c.get("text") or "")[:300], "source": c.get("source", "")}
                for c in chunks
                if c.get("text")
            ]
        except Exception:
            return []

    @staticmethod
    def _build_reviewer_prompt(
        execution_result: Any,
        results: dict[str, Any],
        kb_text: str = "",
    ) -> str:
        """构造让 reviewer persona 点评执行结果的 prompt."""
        try:
            exec_blob = json.dumps(execution_result, ensure_ascii=False, default=str)[:1500]
        except Exception:
            exec_blob = str(execution_result)[:1500]
        try:
            res_blob = json.dumps(results, ensure_ascii=False, default=str)[:1500]
        except Exception:
            res_blob = str(results)[:1500]
        kb_section = f"\n{kb_text}\n" if kb_text else ""
        return (
            "Below is the execution result and validation summary from an "
            "autonomous materials-science research loop iteration.\n\n"
            f"Execution result:\n{exec_blob}\n\n"
            f"Validation summary:\n{res_blob}\n"
            f"{kb_section}"
            "As a critical peer reviewer, point out:\n"
            "1. Any methodological weakness or missing convergence check.\n"
            "2. Whether the result is reproducible and benchmarked.\n"
            "3. Whether the result aligns with the domain knowledge context above "
            "(if any), or contradicts known first-principles.\n"
            "4. Concrete next-step improvements.\n"
            "Be concise and direct."
        )

    async def _learn(self, hypothesis: str, plan: dict[str, Any], validation: dict[str, Any]) -> None:
        """Learn from iteration results — update memory, knowledge graph, evolution rules."""
        r_phys = validation.get("r_phys") if isinstance(validation, dict) else None

        # Log to memory
        self.memory.add_message(
            "system",
            {
                "iteration": self._iteration,
                "hypothesis": hypothesis,
                "plan": plan,
                "validation": validation,
                "r_phys": r_phys,
            },
        )

        # Long-term memory: 把关键迭代写入 long-term, 下次 RAG 能检索到
        # 包含 visual primitives 和 surprise 分数, 跨会话完整恢复上下文
        try:
            persona_name = getattr(self, "_last_persona", "unknown")
            mem_content = f"iter {self._iteration}: {hypothesis[:120]}"
            # Visual primitives 入 memory, 下次 recall_for_prompt 能检索到数据形状
            visual_ctx = validation.get("visual_primitives") if isinstance(validation, dict) else None
            if visual_ctx:
                mem_content += f"\nVisual: {visual_ctx[:200]}"
            # Surprise 入 memory, 下次能检索到"这类任务预测准不准"
            pred_err = validation.get("prediction_error", {}) if isinstance(validation, dict) else {}
            if pred_err:
                mem_content += f"\nSurprise: {pred_err.get('surprise', 0)} (worst: {pred_err.get('surprise_worst', pred_err.get('surprise', 0))}, std: {pred_err.get('surprise_std', 0)}) (predicted: {pred_err.get('predicted', '')[:80]})"
            # Persona 入 memory, 下次 _pick_hypothesis_persona 能查到历史效果
            mem_content += f"\nPersona: {persona_name}, r_phys: {r_phys}"
            # 结构化 tags: 供后续按 persona/r_phys/surprise 过滤检索
            _tags = [
                "autoloop",
                f"persona:{persona_name}",
                f"r_phys:{r_phys}" if r_phys is not None else "r_phys:none",
                f"surprise:{pred_err.get('surprise', 0):.2f}" if pred_err else "surprise:0",
            ]
            self.memory.remember(
                content=mem_content,
                category="autoloop_iteration",
                importance=0.6 if r_phys is None else min(0.9, float(r_phys)),
                tier="mid",
                tags=_tags,
            )
        except Exception:
            logger.warning("error in _learn: memory.remember iteration failed", exc_info=True)

        # 奖励回流: 把 R_phys 喂给 evolution engine, 驱动基于奖励的进化
        # 这是阶段4 单轨的核心闭环——物理校验分数真正影响 agent 后续行为
        if r_phys is not None:
            try:
                evolution = self._get_evolution()
                # 记录本次迭代的 reward, 供 evolve_from_rewards 消费
                evolution.logger.log_tool_call(
                    session_id=f"loop_{self._iteration}",
                    tool_name=plan.get("mode", "unknown"),
                    tool_input={"hypothesis": hypothesis, "plan": plan},
                    result=validation,
                    reward=r_phys,
                )
                reward_result = evolution.evolve_from_rewards()
                n_skills = len(reward_result["high_reward_skills"])
                n_patches = len(reward_result["low_reward_patches"])
                if n_skills or n_patches:
                    logger.info(
                        "reward evolution: +%d skills, +%d patches (R_phys=%.2f)",
                        n_skills, n_patches, r_phys,
                    )
            except Exception as e:
                logger.warning("reward evolution failed: %s", e)

        # KB 回写: 把本次实验结论存入知识库, 下次同类问题能从 KB 召回.
        # 不存原始数据 (太大), 只存 hypothesis + validation 摘要.
        # JEPA: 预测误差也写入, 下次同类任务能从 KB 检索到"这类任务
        # agent 的预测准不准", 帮助判断是否需要更多探索.
        try:
            kb = self._get_kb()
            if kb:
                pred_err = validation.get("prediction_error", {})
                surprise_line = ""
                if pred_err:
                    surprise_line = f"\nPrediction surprise: {pred_err.get('surprise', 0)}\nPredicted: {pred_err.get('predicted', '')[:100]}\nActual: {pred_err.get('actual', '')[:100]}"
                summary_text = (
                    f"Iteration {self._iteration}: {hypothesis[:200]}\n"
                    f"Mode: {plan.get('mode', 'unknown')}\n"
                    f"R_phys: {r_phys}\n"
                    f"Validation: {json.dumps(validation, default=str)[:500]}"
                    f"{surprise_line}"
                )
                kb.add_document(
                    filename=f"autoloop_iter_{self._iteration}.txt",
                    content=summary_text.encode("utf-8"),
                )
        except Exception:
            logger.warning("error in _learn: KB writeback failed", exc_info=True)

        # KB 自动清理: 每 10 轮迭代清理一次旧文档, 防止 KB 无限增长.
        # autoloop_iter_ 文档只保留最近 50 轮, 总文档上限 200.
        # 这解决了"每轮写入但永不删除"的内存泄漏问题.
        if self._iteration > 0 and self._iteration % 10 == 0:
            try:
                kb = self._get_kb()
                if kb and hasattr(kb, "cleanup_old_documents"):
                    deleted = kb.cleanup_old_documents(max_docs=200)
                    if deleted:
                        logger.info("KB cleanup: removed %d old documents", deleted)
            except Exception:
                pass

        # KG 回写: 把 hypothesis 作为 experiment 实体加入知识图,
        # 让 ProjectKnowledgeGraph 随实验增长而非只读展示.
        # 视觉基元 + surprise 都写入实体属性, 下次 KG 查询能检索到.
        try:
            kg_attrs: dict[str, Any] = {
                "iteration": self._iteration,
                "r_phys": r_phys,
            }
            visual_ctx = validation.get("visual_primitives") if isinstance(validation, dict) else None
            if visual_ctx:
                kg_attrs["visual_primitives"] = visual_ctx[:500]
            # JEPA: surprise 分数存入 KG, 下次查同类实验能看到"这类任务
            # agent 预测准不准", 帮助判断是否值得继续探索.
            pred_err = validation.get("prediction_error", {}) if isinstance(validation, dict) else {}
            if pred_err:
                kg_attrs["surprise"] = pred_err.get("surprise", 0)
                kg_attrs["predicted"] = pred_err.get("predicted", "")[:200]
            # Persona 入 KG: 以后可以查 "reviewer persona 的 experiments 平均 r_phys 是多少"
            kg_attrs["persona"] = getattr(self, "_last_persona", "unknown")
            exp_id = self.kg.add_entity(
                label=hypothesis[:80],
                entity_type="experiment",
                source="autoloop",
                confidence=float(r_phys) if r_phys is not None else 0.5,
                **kg_attrs,
            )
            # KG confidence 衰减: validation 失败时降低实验实体置信度.
            # 之前 confidence 只增不减, 被refute的假设在 KG 里永远高置信.
            tests_passed = validation.get("tests_passed") if isinstance(validation, dict) else False
            if not tests_passed and exp_id and hasattr(self.kg, '_graph'):
                try:
                    if exp_id in self.kg._graph:
                        old_conf = self.kg._graph.nodes[exp_id].get("confidence", 0.5)
                        self.kg._graph.nodes[exp_id]["confidence"] = old_conf * 0.7
                except Exception:
                    pass
            # Hyperedge: 把 hypothesis → plan_mode → validation 结果
            # 连成 n-ary 关系. 之前 add_hyperedge 是死代码, 现在接上.
            plan_id = self.kg.add_entity(
                label=f"plan_{plan.get('mode', 'unknown')}_iter{self._iteration}",
                entity_type="Method",
                source="autoloop",
            )
            result_label = "pass" if (validation.get("tests_passed") if isinstance(validation, dict) else False) else "fail"
            result_id = self.kg.add_entity(
                label=f"{result_label}_iter{self._iteration}",
                entity_type="Fact",
                source="autoloop",
                surprise=pred_err.get("surprise", 0) if pred_err else 0,
            )
            if exp_id and plan_id and result_id:
                self.kg.add_hyperedge(
                    [exp_id, plan_id, result_id],
                    relation="experiment_pipeline",
                    source="autoloop",
                    iteration=self._iteration,
                )
            self.kg.save()
        except Exception:
            logger.warning("error in _learn: KG add_entity failed", exc_info=True)

        # Benchmark 失败回写: 把验证失败写入 memory, 下次 _plan 能读到.
        if isinstance(validation, dict) and not validation.get("tests_passed", True):
            try:
                self.memory.remember(
                    content=(
                        f"Validation failure iter {self._iteration}: "
                        f"{json.dumps(validation, default=str)[:400]}"
                    ),
                    category="benchmark_failure",
                    tags=["autoloop", "validation"],
                    importance=0.7,
                    tier="mid",
                )
            except Exception:
                logger.warning("error in _learn: benchmark_failure memory writeback failed", exc_info=True)

        # Feynman learning: 高 surprise 或高奖励时, 让 agent 用通俗语言重新解释本轮发现.
        # 解释不出来的部分就是知识缺口, 写入 GoalStore 作为下轮子目标.
        # 触发条件: surprise > 0.5 (预测错误大) 或 r_phys > 0.7 (值得总结的成功)
        _should_feynman = False
        try:
            _surprise_val = 0.0
            if isinstance(validation, dict):
                _pe = validation.get("prediction_error", {})
                _surprise_val = _pe.get("surprise", 0) if isinstance(_pe, dict) else 0
            if _surprise_val > 0.5 or (r_phys is not None and r_phys > 0.7):
                _should_feynman = True
        except Exception:
            logger.debug("surprise detection failed — _feynman_learn trigger may silently skip", exc_info=True)

        if _should_feynman:
            try:
                await self._feynman_learn(hypothesis, plan, validation, r_phys, context)
            except Exception:
                logger.warning("error in _learn: feynman note generation failed", exc_info=True)

        # 把 plan 进度存进 long-term memory, 下次会话能接续
        _plan_id = plan.get("plan_id") if isinstance(plan, dict) else None
        if _plan_id:
            try:
                store = self._get_plan_store()
                if store is not None:
                    persisted = store.get_plan(_plan_id)
                    if persisted is not None:
                        self.memory.store_plan_progress(
                            plan_id=persisted.id,
                            objective=persisted.objective,
                            step_index=len(
                                [s for s in persisted.steps if s.status == "done"]
                            ),
                            status=persisted.status,
                            l1_coordinates=f"autoloop: {persisted.objective[:100]}",
                        )
            except Exception:
                logger.warning("error in _learn: store_plan_progress writeback failed", exc_info=True)

    async def _report(self, objective: str, phases: list[LoopPhase], total_time: float) -> str | None:
        """Generate a structured scientific research report.

        RCBench expects y=(π, o, r) where r is a research report with
        Introduction/Methods/Results/Discussion. We assemble execution data
        from self and let the LLM write a proper report instead of a loop summary.
        """
        report_data = {
            "objective": objective,
            "run_id": f"loop_{uuid.uuid4().hex[:8]}",
            "total_time_seconds": total_time,
            "phases": [
                {
                    "name": p.name,
                    "status": p.status,
                    "duration": (p.end_time or 0) - (p.start_time or 0) if p.start_time and p.end_time else 0,
                    "error": p.error,
                }
                for p in phases
            ],
        }

        # Collect scientific evidence from the engine instance for the report.
        # This is the (π, o) data RCBench expects: what ran, what came out.
        last_exec = getattr(self, "_last_execution_result", None)
        exec_summary = ""
        if last_exec and isinstance(last_exec, dict):
            _tool = last_exec.get("_tool_name", "unknown")
            _res = last_exec.get("result", last_exec)
            exec_summary = json.dumps(_res, ensure_ascii=False, default=str)[:1500]
            exec_summary = f"Tool: {_tool}\nResult: {exec_summary}"

        visual_ctx = getattr(self, "_last_visual_context", "")
        last_validation = getattr(self, "_last_validation", "")
        last_surprise = getattr(self, "_last_surprise", 0.0)
        last_hypothesis = getattr(self, "_last_hypothesis", "")

        kb_text = self._build_kb_text(query=objective)
        report_narrative = ""
        try:
            report_narrative = await self._llm_chat(
                self._build_science_report_prompt(
                    report_data, kb_text, exec_summary, visual_ctx,
                    last_validation, last_hypothesis, last_surprise,
                ),
                persona_name="tutor",
                task="summarize",
            )
            report_narrative = (report_narrative or "").strip()
        except Exception:
            report_narrative = ""

        report_path = self.workspace / f"huginn_autoloop_report_{report_data['run_id']}.md"
        report_content = self._render_report(report_data)
        if kb_text:
            report_content += "\n\n## Domain Knowledge References\n\n" + kb_text + "\n"
        if report_narrative:
            report_content += "\n\n## Research Report\n\n" + report_narrative + "\n"
        report_path.write_text(report_content, encoding="utf-8")

        return str(report_path)

    # ── Feynman learning ──────────────────────────────────────────

    _FEYNMAN_PROMPT = """You are studying your own research iteration using the Feynman Learning Method.
The core principle: if you can't explain it in simple terms, you don't truly understand it.

## Iteration Context
- Hypothesis: {hypothesis}
- Plan mode: {mode}
- R_phys (physical reward): {r_phys}
- Surprise: {surprise} (how much the actual result differed from prediction)
- Validation summary: {validation}
- Deviations from plan: {deviations}

## Your Task
Write TWO sections:

### Simple Explanation
Explain what happened in this iteration as if teaching a newcomer who has basic
materials science knowledge but no experience with computational tools.
Focus on: What was the physical question? What did the calculation reveal?
Why does the result make sense (or not)? Use analogies where helpful.
If there were deviations from the plan, explain WHY the path changed.

### Knowledge Gaps
List specific things you CANNOT confidently explain. Be honest — admitting
gaps is the point of this exercise. Mark each gap:
- [KU] for "known unknown" — you know you don't understand this
- [UU] for "unknown unknown" — you didn't even think about this until now
Examples:
- "[KU] I don't understand why the band gap changed non-monotonically with doping"
- "[UU] I never considered that GaN has two polymorphs until the result came back"

Output format (Markdown, no code blocks):
## Simple Explanation
...

## Knowledge Gaps
- [KU] gap 1
- [UU] gap 2
..."""

    async def _feynman_learn(
        self,
        hypothesis: str,
        plan: dict[str, Any],
        validation: dict[str, Any],
        r_phys: Any,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Feynman 学习法: 让 agent 用通俗语言解释本轮发现, 暴露知识缺口.

        生成的教学笔记存入蒸馏知识库 (feynman_note 类型, KB 检索优先).
        知识缺口写入 GoalStore 作为下轮子目标.
        """
        pred_err = validation.get("prediction_error", {}) if isinstance(validation, dict) else {}
        surprise_val = pred_err.get("surprise", 0) if isinstance(pred_err, dict) else 0

        # 收集 deviation log, 让 Feynman 解释也覆盖 "为什么偏移了计划"
        deviation_text = ""
        if context and context.get("_deviation_log"):
            deviations = context["_deviation_log"]
            deviation_text = "\n".join(
                f"- [{d.get('type', '?')}] {d.get('deviation', '')}"
                for d in deviations[-3:]  # 最近 3 条
            )

        prompt = self._FEYNMAN_PROMPT.format(
            hypothesis=hypothesis[:300],
            mode=plan.get("mode", "unknown") if isinstance(plan, dict) else "unknown",
            r_phys=r_phys,
            surprise=f"{surprise_val:.2f}",
            validation=json.dumps(validation, default=str)[:600],
            deviations=deviation_text or "(none)",
        )

        # 用 summarize task 路由到便宜模型 — Feynman note 不需要强推理
        response = await self._llm_chat(prompt, task="summarize")
        if not response or not response.strip():
            return

        # 解析 explanation 和 gaps
        text = response.strip()
        explanation = ""
        gaps: list[str] = []

        # 简单解析: ## Simple Explanation 和 ## Knowledge Gaps 两段
        parts = text.split("## Knowledge Gaps")
        explanation_part = parts[0].replace("## Simple Explanation", "", 1).strip()
        # gaps 带 [KU]/[UU] 分类, 传给 GoalStore
        gaps: list[tuple[str, str]] = []  # (text, unknown_type)
        if len(parts) > 1:
            for line in parts[1].strip().split("\n"):
                line = line.strip().lstrip("-").strip()
                if not line or len(line) <= 5:
                    continue
                # 解析 [KU] / [UU] 标记
                if line.startswith("[UU]"):
                    gaps.append((line[4:].strip(), "unknown_unknown"))
                elif line.startswith("[KU]"):
                    gaps.append((line[4:].strip(), "known_unknown"))
                else:
                    # 无标记时用启发式分类
                    gap_lower = line.lower()
                    is_uu = any(kw in gap_lower for kw in
                                ["didn't", "never", "wasn't aware", "didn't think",
                                 "hadn't", "overlooked", "完全没", "之前没", "没想到"])
                    gaps.append((line, "unknown_unknown" if is_uu else "known_unknown"))

        if not explanation_part:
            explanation_part = text[:500]

        explanation = explanation_part

        # 存入蒸馏知识库
        _feynman_conf = min(0.9, 0.5 + (r_phys or 0) * 0.3)
        try:
            from huginn.evolution.knowledge_distiller import KnowledgeDistiller
            distiller = KnowledgeDistiller()
            tags = ["feynman", "autoloop", f"iter_{self._iteration}"]
            if surprise_val > 0.5:
                tags.append("high_surprise")
            # gaps 现在是 list[tuple[str, str]], 转回 list[str] 给 distiller
            gap_texts = [g[0] for g in gaps]
            distiller.store_feynman_note(
                explanation=explanation,
                gaps=gap_texts,
                iteration=self._iteration,
                hypothesis=hypothesis,
                tags=tags,
                confidence=_feynman_conf,
            )
        except Exception:
            logger.warning("feynman note storage failed", exc_info=True)

        # 缺口写入 GoalStore, 分类为 known_unknown / unknown_unknown
        # known_unknown: "我知道我不懂X" → 直接当子目标, 下轮解决
        # unknown_unknown: "我之前完全没想到X" → 标记为需要更深的探索
        # 借鉴 "Finding Your Unknowns" 四象限框架
        if gaps:
            try:
                from huginn.autoloop.goal_store import get_goal_store
                _gs = get_goal_store()
                _active = _gs.get_active()
                if _active:
                    for gap_text, gap_type in gaps[:3]:  # 最多 3 个, 避免子目标爆炸
                        _gs.add_sub_goal(_active.id, f"[Feynman {gap_type}] {gap_text}")
                        _gs.add_unknown(_active.id, gap_text, unknown_type=gap_type)
            except Exception:
                pass

        # 同时把 feynman note 写入 KB, 下次检索能命中
        try:
            kb = self._get_kb()
            if kb:
                note_text = f"# Feynman Note (iter {self._iteration})\n\n{explanation}\n\n## Gaps\n"
                for g_text, g_type in gaps:
                    note_text += f"- [{g_type}] {g_text}\n"
                kb.add_text(
                    text=note_text,
                    filename=f"feynman_iter_{self._iteration}.txt",
                    metadata={"confidence": str(_feynman_conf)},
                )
        except Exception:
            logger.debug("feynman note save failed", exc_info=True)

    # ── Blind spot pass ───────────────────────────────────────────

    _BLIND_SPOT_PROMPT = """You are about to start a research task. Before diving in, do a blindspot pass.

## Task
Objective: {objective}

## Current Context
{context_summary}

## Your Job
Identify potential UNKNOWN UNKNOWNS — things that might go wrong, assumptions that might
be invalid, or aspects of the problem that haven't been considered yet.

Think about:
1. Physical assumptions: Are there structural/phase/electronic considerations being missed?
2. Computational pitfalls: Convergence, basis set, pseudopotential, k-grid issues?
3. Data gaps: Is there reference data missing? Are there known experimental values to compare against?
4. Methodology blind spots: Could the chosen method give qualitatively wrong results for this system?
5. Edge cases: Temperature, pressure, doping level boundaries?

Output up to 5 potential blind spots, one per line, prefixed with "BS:".
For each, also note the type: [structural], [computational], [data], [method], [edge_case].
Format: BS: [type] description

If you genuinely can't find any blind spots (unlikely), output: NONE"""

    async def _blind_spot_pass(
        self, context: dict[str, Any], objective: str
    ) -> list[dict[str, str]]:
        """Pre-implementation blind spot scan.

        借鉴 "Finding Your Unknowns" 的 Blind Spot Pass 技术:
        在开始工作前主动问 "我可能没想到什么?"
        发现的盲区写入 GoalStore.unknowns 供后续消解追踪.
        """
        # 压缩 context 到摘要, 避免太长
        ctx_parts: list[str] = []
        for k, v in context.items():
            if isinstance(v, str):
                ctx_parts.append(f"- {k}: {v[:150]}")
            elif isinstance(v, list) and v:
                ctx_parts.append(f"- {k}: {len(v)} items")
            elif isinstance(v, dict):
                ctx_parts.append(f"- {k}: {len(v)} keys")
        ctx_summary = "\n".join(ctx_parts[:10]) or "(minimal context)"

        prompt = self._BLIND_SPOT_PROMPT.format(
            objective=objective[:300],
            context_summary=ctx_summary,
        )

        response = await self._llm_chat(prompt, task="summarize")
        if not response or not response.strip():
            return []

        results: list[dict[str, str]] = []
        for line in response.strip().split("\n"):
            line = line.strip()
            if not line.startswith("BS:"):
                continue
            content = line[3:].strip()
            # 解析 [type] 前缀
            btype = "general"
            if content.startswith("["):
                end = content.find("]")
                if end > 0:
                    btype = content[1:end].strip()
                    content = content[end + 1:].strip()
            if content and content != "NONE":
                results.append({"type": btype, "text": content})
                # 写入 GoalStore
                try:
                    from huginn.autoloop.goal_store import get_goal_store
                    _gs = get_goal_store()
                    _active = _gs.get_active()
                    if _active:
                        _gs.add_unknown(
                            _active.id, content,
                            unknown_type="blind_spot",
                        )
                except Exception:
                    pass

        return results

    # ── Deviation log ─────────────────────────────────────────────

    def _log_deviation(
        self,
        plan: dict[str, Any],
        result: Any,
        context: dict[str, Any],
    ) -> None:
        """记录执行偏离计划的决策.

        借鉴 "Finding Your Unknowns" 的 implementation-notes.md 技术:
        agent 执行中发现需要换路径时, 记录 WHY — 不只记 WHAT (provenance 已做).

        触发条件:
        1. plan 有 expected_prediction 但 result 明显不符
        2. result 含 error/warning 字段
        3. _try_evolved_fix 被触发 (heuristic fix = 偏离了原 plan)

        存入 context['_deviation_log'] 供 _learn() 和 Feynman 使用.
        """
        deviations: list[dict[str, str]] = context.setdefault("_deviation_log", [])
        plan_mode = plan.get("mode", "unknown")
        plan_desc = plan.get("description", "")[:200]
        expected = plan.get("expected_prediction", "")

        # 检查 1: 有 error
        if isinstance(result, dict) and result.get("error"):
            deviations.append({
                "iteration": str(self._iteration),
                "type": "execution_error",
                "plan_mode": plan_mode,
                "plan_desc": plan_desc,
                "deviation": f"Execution failed: {str(result['error'])[:200]}",
                "expected": expected[:100] if expected else "(none)",
            })

        # 检查 2: evolved fix 被使用
        if isinstance(context, dict) and context.get("_evolved_fix"):
            deviations.append({
                "iteration": str(self._iteration),
                "type": "heuristic_fix",
                "plan_mode": plan_mode,
                "plan_desc": plan_desc,
                "deviation": "Applied evolved heuristic fix instead of following original plan",
                "expected": expected[:100] if expected else "(none)",
            })

        # 检查 3: result success=False
        if isinstance(result, dict) and result.get("success") is False:
            deviations.append({
                "iteration": str(self._iteration),
                "type": "plan_mismatch",
                "plan_mode": plan_mode,
                "plan_desc": plan_desc,
                "deviation": "Plan produced unsuccessful result, will need refinement",
                "expected": expected[:100] if expected else "(none)",
            })

    @staticmethod
    def _build_science_report_prompt(
        report_data: dict[str, Any],
        kb_text: str = "",
        exec_summary: str = "",
        visual_ctx: str = "",
        validation_summary: str = "",
        hypothesis: str = "",
        surprise: float = 0.0,
    ) -> str:
        """Build a prompt for generating a structured scientific research report.

        RCBench evaluates y=(π, o, r) where r must contain scientific findings,
        not just a loop status table. This prompt produces Introduction /
        Methods / Results / Discussion structure from the actual execution data.
        """
        try:
            phases_blob = json.dumps(report_data["phases"], ensure_ascii=False)[:800]
        except Exception:
            phases_blob = str(report_data.get("phases", ""))[:800]
        kb_section = f"\n## Domain Knowledge\n{kb_text}\n" if kb_text else ""
        exec_section = f"\n## Execution Data\n{exec_summary}\n" if exec_summary else ""
        visual_section = f"\n## Visual Primitives\n{visual_ctx}\n" if visual_ctx else ""
        val_section = f"\n## Validation\n{validation_summary}\n" if validation_summary else ""
        hyp_section = f"\n## Hypothesis Tested\n{hypothesis}\n" if hypothesis else ""

        return (
            "You are writing a structured scientific research report based on an "
            "autonomous research loop's execution data. This is NOT a loop summary — "
            "it must read like a research paper section.\n\n"
            f"Objective: {report_data['objective']}\n"
            f"Phases:\n{phases_blob}\n"
            f"Surprise score: {surprise:.2f} (0=predicted, 1=unexpected)"
            f"{hyp_section}{exec_section}{visual_section}{val_section}{kb_section}"
            "\nWrite the report with these sections (Markdown):\n"
            "## Introduction\n"
            "State the scientific question and why it matters. Reference domain knowledge above.\n\n"
            "## Methods\n"
            "Describe the computational approach: what tools were used, what parameters, "
            "what workflow. Be specific enough for reproducibility.\n\n"
            "## Results\n"
            "Report the key findings with specific numbers. If visual primitives are "
            "available, describe the trends/peaks/anomalies they indicate.\n\n"
            "## Discussion\n"
            "Interpret the results: Do they support the hypothesis? What was surprising "
            "(reference surprise score)? What are the limitations? "
            "What should the next experiment be?\n"
        )

    # ──────────────────────────────────────────────────────────────
    # Execution helpers
    # ──────────────────────────────────────────────────────────────

    async def _execute_coder(self, description: str, context: dict[str, Any]) -> dict[str, Any]:
        """Run the coder loop on the description, reusing self.coder."""
        task = f"""Task: {description}

Context:
- Changed files: {context.get('changed_files', [])}
- Git diff: {context.get('git_diff', '')[:500]}

Please modify the code to address this task."""
        try:
            # CoderRunner.run 是同步的, 丢线程里避免阻塞事件循环
            result = await asyncio.to_thread(self.coder.run, task)
            messages = result.get("messages", [])
            tool_calls = sum(
                1 for m in messages if getattr(m, "tool_calls", None)
            )
            return {
                "mode": "coder",
                "status": "completed",
                "final_answer": result.get("final_answer", ""),
                "tool_calls": tool_calls,
            }
        except Exception as e:
            logger.exception("coder execution failed")
            return {"mode": "coder", "status": "failed", "error": str(e)}

    # domain → 默认模板名; get_template 拿不到就 fallback standard_dft
    # ponytail: 硬编码映射表, 新模板加一行即可; 想自动发现就扫 WORKFLOW_TEMPLATES
    _DOMAIN_TEMPLATE_NAMES = {
        "cfd": "turbulent_flow",
        "fea": "structural_analysis",
        "qc": "wavefunction_analysis",
        "symbolic": "constitutive_derivation",
        "dft": "standard_dft",
    }

    def _classify_workflow_domain(self, description: str) -> str:
        """廉价关键词分类, 决定走哪个 workflow 模板."""
        text = description.lower()
        if any(k in text for k in ("cfd", "fluid", "fluent", "openfoam")):
            return "cfd"
        if any(k in text for k in ("fea", "stress", "mechanical", "abaqus", "ansys")):
            return "fea"
        if any(k in text for k in ("quantum", "qc", "chemistry", "gaussian", "orca")):
            return "qc"
        if any(k in text for k in ("symbolic", "regression", "拟合")):
            return "symbolic"
        return "dft"

    async def _execute_workflow(self, description: str, context: dict[str, Any]) -> dict[str, Any]:
        """Execute a workflow task, picking template by domain when possible."""
        try:
            domain = self._classify_workflow_domain(description)
            template_name = self._DOMAIN_TEMPLATE_NAMES.get(domain, "standard_dft")
            template_fn = get_template(template_name) or standard_dft_workflow

            # 找工作区里的输入文件; 只对 DFT/QC 用 structure_path
            structure_files = (
                list(self.workspace.rglob("*.cif"))
                + list(self.workspace.rglob("*.poscar"))
                + list(self.workspace.rglob("*.vasp"))
            )
            geometry_files = (
                list(self.workspace.rglob("*.stp"))
                + list(self.workspace.rglob("*.stl"))
                + list(self.workspace.rglob("*.msh"))
                + list(self.workspace.rglob("*.inp"))
            )
            xyz_files = (
                list(self.workspace.rglob("*.xyz"))
                + list(self.workspace.rglob("*.pdb"))
            )
            structure_path = str(structure_files[0]) if structure_files else "structure.cif"

            # 不同域模板参数不一样, 廉价 try 一组; 失败就 fallback DFT
            try:
                if domain == "cfd":
                    geo = str(geometry_files[0]) if geometry_files else "geometry.stp"
                    stages = template_fn(geometry_file=geo)
                elif domain == "fea":
                    geo = str(geometry_files[0]) if geometry_files else "geometry.inp"
                    stages = template_fn(geometry_file=geo)
                elif domain == "qc":
                    struct = str(xyz_files[0]) if xyz_files else structure_path
                    stages = template_fn(structure_file=struct)
                elif domain == "symbolic":
                    # symbolic 模板要 free_energy_expr, 没法从工作区推断, 拿 description 顶
                    stages = template_fn(free_energy_expr=description)
                else:
                    stages = template_fn(structure_path=structure_path, engine="vasp")
            except Exception as tmpl_err:
                logger.warning(
                    "workflow template %s (%s) failed: %s, fallback to standard_dft",
                    template_name, domain, tmpl_err,
                )
                stages = standard_dft_workflow(structure_path, engine="vasp")

            tool_context = ToolContext(
                session_id=f"workflow_{uuid.uuid4().hex[:8]}",
                workspace=str(self.workspace),
                config=self.settings,
            )
            result = await self.workflow_engine.execute(stages, tool_context)
            return {
                "mode": "workflow",
                "success": result.success,
                "stages": len(stages),
                "domain": domain,
            }
        except Exception as e:
            return {"mode": "workflow", "success": False, "error": str(e)}

    async def _execute_explore(self, description: str, context: dict[str, Any]) -> dict[str, Any]:
        """Execute an exploration task."""
        try:
            result = await self.explorer.explore(
                objective=description,
                initial_branches=[
                    {"name": "baseline", "hypothesis": f"Baseline for: {description}"}
                ],
                max_iterations=5,
            )
            return {
                "mode": "explore",
                "n_explored": result.n_branches_explored,
                "n_pruned": result.n_branches_pruned,
                "convergence": result.convergence_reason,
            }
        except Exception as e:
            return {"mode": "explore", "success": False, "error": str(e)}

    async def _execute_skill(self, plan: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """Run a pre-built composite skill pipeline."""
        try:
            from huginn.skills.registry import SkillRegistry
            from huginn.skills.base import DeclarativeSkillExecutor
            from huginn.skills.composite import _ensure_registered
            _ensure_registered()

            skill_name = plan.get("skill", "")
            skill = SkillRegistry.get(skill_name)
            if not skill:
                # Fuzzy match if exact name missing
                matches = SkillRegistry.search(skill_name or plan.get("description", ""))
                skill = matches[0] if matches else None
            if not skill:
                return {"mode": "skill", "success": False,
                        "error": f"no matching skill for '{skill_name}'"}

            # Reuse the same tool registry as the rest of the engine
            from huginn.tools.registry import ToolRegistry
            executor = DeclarativeSkillExecutor(ToolRegistry)
            result = await executor.execute(skill, {}, context)
            return {"mode": "skill", "skill": skill.name, **result}
        except Exception as e:
            return {"mode": "skill", "success": False, "error": str(e)}

    async def _execute_visual_inspect(
        self, description: str, context: dict[str, Any]
    ) -> dict[str, Any]:
        """Path C: 交互式视觉检查. 让 agent 主动调用视觉工具检查上一轮结果.

        这是 OpenThinkIMG 式的工具调用路径 — agent 在推理过程中主动选择
        "放大图表某区域"或"测量某数据点", 而不是被动接收预处理好的视觉基元.
        使用已有的 image_analysis_tool / visual_hook 基础设施, 不新建工具.

        description 解析: "zoom into band 3 near [500,800]" / "measure peak at [999,999]"
        坐标是 0-999 归一化的视觉原语坐标 (路径 B 格式).
        """
        import re

        result: dict[str, Any] = {
            "mode": "visual_inspect",
            "description": description,
            "actions": [],
        }

        # 获取上一轮的视觉基元和 base64 图片
        visual_ctx = getattr(self, "_last_visual_context", "")
        visual_base64 = getattr(self, "_visual_base64", "")

        if not visual_ctx and not visual_base64:
            return {
                **result,
                "success": False,
                "error": "No visual data from previous iteration to inspect",
            }

        # 解析 description 中的动作
        desc_lower = description.lower()

        # 动作 1: zoom — 放大某区域
        if "zoom" in desc_lower:
            # 提取坐标 [x1,y1,x2,y2] 或 [x,y]
            coords = re.findall(r'\[?(\d+)\s*,\s*(\d+)\]?', description)
            if len(coords) >= 2:
                x1, y1 = int(coords[0][0]), int(coords[0][1])
                x2, y2 = int(coords[1][0]), int(coords[1][1])
                # 把 0-999 坐标转成数据索引 (如果有上一轮的原始数据)
                action_result = {
                    "action": "zoom",
                    "region": [x1, y1, x2, y2],
                    "note": f"Zoomed into region [{x1},{y1}]-[{x2},{y2}]",
                }
                # 如果有 visual_base64, 调用 image_analysis_tool 做真正的区域分析
                if visual_base64:
                    try:
                        from huginn.tools.registry import ToolRegistry
                        img_tool = ToolRegistry.get("image_analysis_tool")
                        if img_tool:
                            # 裁剪 base64 图片到指定区域并分析
                            import base64 as b64
                            import io as _io
                            try:
                                from PIL import Image
                                img_data = b64.b64decode(visual_base64)
                                img = Image.open(_io.BytesIO(img_data))
                                w, h = img.size
                                # 0-999 → pixel coordinates
                                px1 = int(x1 / 999 * w)
                                py1 = int(y1 / 999 * h)
                                px2 = int(x2 / 999 * w)
                                py2 = int(y2 / 999 * h)
                                cropped = img.crop((px1, py1, px2, py2))
                                buf = _io.BytesIO()
                                cropped.save(buf, format="PNG")
                                cropped_b64 = b64.b64encode(buf.getvalue()).decode()
                                action_result["cropped_image"] = cropped_b64[:10000]  # limit size
                                action_result["crop_size"] = [px2 - px1, py2 - py1]
                            except ImportError:
                                action_result["note"] += " (PIL not available, coordinates only)"
                            except Exception as e:
                                action_result["note"] += f" (crop failed: {e})"
                    except Exception:
                        logger.debug("image crop action failed", exc_info=True)
                result["actions"].append(action_result)

        # 动作 2: measure — 测量某点或区域的数据值
        elif "measure" in desc_lower:
            coords = re.findall(r'\[?(\d+)\s*,\s*(\d+)\]?', description)
            if coords:
                x, y = int(coords[0][0]), int(coords[0][1])
                # 从 visual_ctx 中查找最接近的基元
                result["actions"].append({
                    "action": "measure",
                    "coordinate": [x, y],
                    "note": f"Measured at <point>[{x},{y}]</point>",
                    "visual_context_snippet": visual_ctx[:300] if visual_ctx else "",
                })

        # 动作 3: annotate — 标注结构特征
        elif "annotate" in desc_lower:
            result["actions"].append({
                "action": "annotate",
                "description": description,
                "note": "Annotation recorded for visual reasoning",
                "visual_context": visual_ctx[:500] if visual_ctx else "",
            })

        # 动作 4: compare — 比较两组数据
        elif "compare" in desc_lower:
            result["actions"].append({
                "action": "compare",
                "description": description,
                "visual_context": visual_ctx[:500] if visual_ctx else "",
                "note": "Comparison analysis requested",
            })

        # 默认: 记录检查请求
        else:
            result["actions"].append({
                "action": "inspect",
                "description": description,
                "visual_context": visual_ctx[:500] if visual_ctx else "",
            })

        # 生成新的视觉基元 (基于检查动作的输出)
        new_primitives = []
        for action in result["actions"]:
            if "note" in action:
                new_primitives.append(f"[{action['action']}] {action['note']}")
        result["visual_summary"] = "\n".join(new_primitives)
        result["success"] = True

        # 用 enrich_with_visual 给这次检查也生成视觉基元
        try:
            from huginn.tools.visual_hook import enrich_with_visual
            enriched = enrich_with_visual("visual_inspect", {"result": result})
            if "_visual_hint" in enriched:
                result["_visual_hint"] = enriched["_visual_hint"]
        except Exception:
            pass

        return result

    # ──────────────────────────────────────────────────────────────
    # LLM helpers
    # ──────────────────────────────────────────────────────────────

    async def _llm_chat(
        self,
        prompt: str,
        persona_name: str | None = None,
        model: Any = None,
        task: str | None = None,
    ) -> str:
        """Send a prompt to the LLM and return the response.

        persona_name 不为空时, 把对应 persona 的 system prompt 作为
        SystemMessage 插在最前, 实现"每阶段开始注入 persona system prompt".
        persona 找不到就退化为不注入, 行为跟改动前一致.

        model 不为空时用传入的模型 (用于三槽 verification), 否则用默认 self.model.

        task 不为空时, 优先从 model_router 路由 (team 模式):
        - "reasoning"/"science" → 强模型 (云端)
        - "summarize"/"format" → 便宜模型 (本地/小模型)
        - "verification" → 独立验证模型
        model 参数优先于 task — 显式指定的模型不被路由覆盖.
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        # Team 模式: task 路由优先, 但显式 model 不被覆盖
        if model is None and task is not None:
            router = getattr(self, 'model_router', None)
            if router is not None:
                try:
                    routed = router.select(task, prefer_cheap=(task in ("summarize", "format", "archival")))
                    if routed is not None:
                        model = routed
                except Exception:
                    logger.debug("model router select failed — using fallback model", exc_info=True)

        llm = model or self.model
        messages: list[Any] = []
        if persona_name:
            sys_prompt = self._persona_system_prompt(persona_name)
            if sys_prompt:
                sys_msg = SystemMessage(content=sys_prompt)
                # 静态 system prompt 跨调用不变, 给 Anthropic/Kimi 打 cache 标记
                _ident = f"{type(llm).__name__}{getattr(llm, 'model', '')}".lower()
                if any(k in _ident for k in ("anthropic", "claude", "kimi", "moonshot")):
                    sys_msg.additional_kwargs["cache_control"] = {"type": "ephemeral"}
                messages.append(sys_msg)
        messages.append(HumanMessage(content=prompt))
        response = await llm.ainvoke(messages)
        return str(response.content)

    def _build_hypothesis_prompt(self, context: dict[str, Any]) -> str:
        # 投机执行 hint: 基于历史预测的下一步意图, 注入给 LLM 参考
        # 预测只是 hint, LLM 可以无视, 不强制. 截断到 500 字符防止无界增长
        # — _speculator_hint 有 5 处 append, 不截断 20 轮后可能数 KB.
        hint_block = ""
        if self._speculator_hint:
            hint_block = (
                f"\nSpeculator hint (advisory, may be ignored): {self._speculator_hint[:500]}\n"
                "想返回时必须输出 UNEXPLORED: 块, 列出至少 3 个未探索的方向 "
                "(方法族/等价性陷阱/连通分量/缺口).\n"
            )
        # 三路检索共用一个 query, 避免重复序列化 context 3 次
        ctx_query = json.dumps(context, ensure_ascii=False)[:500]
        # 领域知识库检索: 命中 first-principles 参考块就拼进 prompt
        kb_block = self._build_kb_text(query=ctx_query)
        if kb_block:
            kb_block = f"\n{kb_block}\n"
        # 知识图谱检索: 把之前 run 发现的实体和关系拉回来, 避免
        # 重复发现已有结论, 也让假设能建立在已有发现上
        kg_block = self._build_kg_text(query=ctx_query)
        if kg_block:
            kg_block = f"\n{kg_block}\n"
        # 长期记忆检索: 跨会话的失败教训和发现. 之前只写不读,
        # 现在闭合 — _learn 写入的迭代记录下次能检索到.
        mem_block = self._build_memory_text(query=ctx_query)
        if mem_block:
            mem_block = f"\n{mem_block}\n"
        # 视觉基元: 上一轮 tool 输出的数值指针 (峰值/趋势/异常),
        # 给 LLM 具体坐标锚定推理 — Thinking with Visual Primitives 的
        # "point while it reasons" 原则, Mirage 效应的文本路径
        visual_block = getattr(self, '_last_visual_context', '')
        if visual_block:
            visual_block = f"\n### Visual Primitives (from last tool output)\n{visual_block}\n"
        # 数学深度引导: 提醒 agent 优先识别 PDE / 变分原理 / 微分几何结构,
        # 并用符号回归 + Sobol 灵敏度 + 物理约束先验 反复试探.
        # 条件化: 只在 context 含数学信号时注入, coder-only 任务不需要.
        # 节省 ~150 tokens × 2 calls/iter × 20 iters = 6K tokens/run.
        ctx_blob = json.dumps(context, ensure_ascii=False).lower()
        math_block = self._MATH_DEPTH_PROMPT_BLOCK if any(s in ctx_blob for s in _MATH_SIGNALS) else ""
        # MatterChat 启发: 把上轮 execution 结果摘要注入 hypothesis prompt,
        # 让假设建立在"上轮实际发生了什么"之上, 不只看 workspace 变化.
        # _last_execution_result 在 _execute 里写入, 之前只 _build_plan_prompt 用.
        exec_block = ""
        last_exec = getattr(self, '_last_execution_result', None)
        if last_exec and isinstance(last_exec, dict):
            _tool = last_exec.get('_tool_name', 'unknown')
            _res = last_exec.get('result', last_exec)
            _summary = json.dumps(_res, ensure_ascii=False, default=str)[:500]
            exec_block = f"\n### Last Execution Result ({_tool})\n{_summary}\n"
        # 想象力引导: 高 surprise 或连续 refine 时, 要求 LLM 跳出分析思维,
        # 考虑反事实假设. 基于 MToM P4 (hybrid ST+TT): 心智模型预测错误时
        # 切到仿真理论重新建模. 结构切换在数学结构族之间, 不是随机猜.
        imagination_block = ""
        if self._should_imaginate():
            imagination_block = self._IMAGINATION_PROMPT_BLOCK
        # Failure mode feedback (Dream Layer): 上轮 validate 描述的"如何崩溃"
        # 注入 hypothesis prompt, 让 agent 从崩溃模式中找新发现.
        # _last_failure_mode 在 _validate 里写入, 空字符串表示无上轮或未解析出.
        fail_block = ""
        _last_fail = getattr(self, "_last_failure_mode", "")
        if _last_fail:
            fail_block = f"\n### Previous Failure Mode\nIf the previous hypothesis is wrong, it would fail in this way:\n{_last_fail}\nConsider whether this failure mode points to a new hypothesis.\n"
        # Git log: EurekAgent artifact engineering — 让 agent 看到前几轮
        # 做了什么, 避免重复尝试已失败的方案. 只取 oneline 前 10 条.
        git_log_block = ""
        try:
            import subprocess as _sp
            _r = _sp.run(
                ["git", "log", "--oneline", "-10"],
                cwd=self.workspace, capture_output=True, text=True, timeout=10,
            )
            if _r.returncode == 0 and _r.stdout.strip():
                git_log_block = f"\n### Recent Experiments (git log)\n{_r.stdout.strip()}\n"
        except Exception:
            pass

        # 分量代表制: 多条独立探索路线时, 给 LLM 看各路线的代表假设,
        # 防止单分量靠节点数主导综合判断. 只在 >1 分量时注入, advisory.
        topology_block = ""
        try:
            reps = self._metacog_component_representatives()
            if len(reps) > 1:
                lines = []
                for rid in reps[:5]:  # ponytail: 截前 5 个, 防超大图撑爆 prompt
                    try:
                        stmt = self.hypothesis_graph.get(rid).statement
                    except Exception:
                        stmt = ""
                    lines.append(f"  - {rid}: {stmt[:120]}")
                topology_block = (
                    f"\n### Topology (advisory)\n"
                    f"当前有 {len(reps)} 条独立探索路线, 代表假设分别是:\n"
                    + "\n".join(lines) + "\n"
                    "综合判断时不要让某条路线靠节点数主导, 注意挑战和重定向.\n"
                )
        except Exception:
            pass

        # 按优先级拼接, 超预算自动裁剪低优先级 block
        return self._trim_to_budget([
            ("body", f"""You are an autonomous material science research agent.

Perceived context:
{json.dumps(context, indent=2, ensure_ascii=False)[:2000]}

Generate 3 divergent candidate hypotheses (different approaches, not variations).
For each, note one pro and one con in a single line.
Then select the best one — most testable and most novel — and state it after "SELECTED:".
Ground it in the domain knowledge context above when relevant.
Prefer hypotheses that can be expressed as governing PDEs, variational
principles, or conservation laws; identify the mathematical structure
before proposing numerical experiments.

Hypothesis:"""),
            ("git_log", git_log_block),
            ("fail", fail_block),
            ("imagination", imagination_block),
            ("exec", exec_block),
            ("math", math_block),
            ("kg", kg_block),
            ("visual", visual_block),
            ("kb", kb_block),
            ("mem", mem_block),
            ("topology", topology_block),
            ("hint", hint_block),
        ])

    _IMAGINATION_PROMPT_BLOCK = """
Imagination directive (speculative mode activated):
- Your prediction was significantly off, or your hypotheses keep getting refuted.
- Consider a counterfactual: what if the governing structure is different from what you assumed?
- Try shifting between mathematical structure families: PDE ↔ variational, continuum ↔ discrete, deterministic ↔ stochastic, linear ↔ nonlinear.
- This is NOT random guessing — the shift must be between mathematically valid structure families, grounded in the domain context.
- The conjecture hint above uses forget-then-generate: known failed approaches have been deliberately suppressed.

LUCID review (mandatory after generating hypothesis):
- You are allowed an absurd premise, but the reasoning must be rigorous.
- State ONE necessary condition: without it, your hypothesis definitely fails.
- State ONE hidden assumption from the source domain that may not hold here.
- State ONE falsifiable test: if result is X, hypothesis is refuted.
- If you cannot state these, the hypothesis is dream-only and must be discarded.
"""

    # 数学深度提示块: 在 hypothesis / plan prompt 里持续提醒 agent 用
    # 符号数学工具把"现象"翻译成"PDE / 变分 / 几何 / 灵敏度"语言.
    # 用户偏好: 物理、化学本质上是数学的一部分 — 这里把那条原则落进 prompt.
    _MATH_DEPTH_PROMPT_BLOCK = """
Math depth guidance (treat physics/chemistry as mathematics):
- Identify the governing PDE: use symbolic_math_tool action=pde_classify
  (A;B;C discriminant) to classify elliptic/parabolic/hyperbolic, then
  pde_separation or pde_characteristics for analytic structure.
- If the phenomenon extremizes a functional, derive Euler-Lagrange:
  symbolic_math_tool action=euler_lagrange or action=derive (alias).
  Check symmetries with action=noether to predict conserved currents.
- For curved manifolds (defects, interfaces, crystal plasticity), compute
  Christoffel/Ricci via action=diffgeo_metric or diffgeo_curvature.
- Before fitting data, run symbolic_regression_tool action=sobol_indices
  to rank feature importance, then discover expressions with
  action=discover and validate candidates with action=constraint_check
  (positivity / monotonicity / finiteness priors).
"""

    def _build_subgoal_block(self) -> str:
        """从 agent 或 self 上读 sub_goals, 注入到 prompt."""
        sgs = getattr(self, "_sub_goals", None) or []
        if not sgs:
            return ""
        lines = ["\n### Active Sub-goal Constraints (from /subgoal)"]
        for i, sg in enumerate(sgs, 1):
            lines.append(f"{i}. {sg}")
        lines.append("### End Sub-goal Constraints\n")
        return "\n".join(lines)

    def _build_plan_prompt(self, hypothesis: str, context: dict[str, Any]) -> str:
        # 同 hypothesize: 用 hypothesis 串检索 KB, 把参考块喂给 planner
        kb_block = self._build_kb_text(query=hypothesis)
        if kb_block:
            kb_block = f"\n{kb_block}\n"
        # KG 检索: 用 hypothesis 当 query, 看看已有实体里有没有相关的
        kg_block = self._build_kg_text(query=hypothesis)
        if kg_block:
            kg_block = f"\n{kg_block}\n"
        # 长期记忆检索 (同 hypothesize)
        mem_block = self._build_memory_text(query=hypothesis)
        if mem_block:
            mem_block = f"\n{mem_block}\n"
        # 视觉基元注入 (同 hypothesize)
        visual_block = getattr(self, '_last_visual_context', '')
        if visual_block:
            visual_block = f"\n### Visual Primitives (from last tool output)\n{visual_block}\n"
        # 条件化 math_block (同 hypothesize)
        hyp_blob = hypothesis.lower() + json.dumps(context, ensure_ascii=False).lower()[:500]
        math_block = self._MATH_DEPTH_PROMPT_BLOCK if any(s in hyp_blob for s in _MATH_SIGNALS) else ""

        # Inject learned skills + prompt patches from evolution engine.
        # This is the "use what you learned" half of the Learn→Plan loop.
        skill_hints = ""
        patch_hints = ""
        try:
            evolution = self._get_evolution()
            skills = evolution.get_relevant_skills(hypothesis)
            if skills:
                skill_lines = [f"  - {s.name}: {s.description}" for s in skills[:3]]
                skill_hints = "\nLearned skills (from past iterations):\n" + "\n".join(skill_lines) + "\n"
            patches = evolution.get_prompt_patches()
            if patches:
                patch_hints = "\nLearned patches:\n" + "\n".join(f"  - {p}" for p in patches[:3]) + "\n"
        except Exception:
            logger.warning("error in _build_plan_prompt: evolution skill/patch fetch failed", exc_info=True)

        # Inject matching composite skills — lets the LLM pick a pre-built
        # multi-tool pipeline instead of improvising from scratch.
        # 条件化: 只在 hypothesis 涉及仿真/计算/材料性质时注入, coder-only
        # 任务不需要 composite skill 列表. 节省 ~500 tokens.
        composite_block = ""
        hyp_lower = hypothesis.lower()
        _workflow_signals = ("workflow", "simulation", "band", "dos", "phonon",
                              "mechanical", "thermal", "optical", "dft", "vasp",
                              "lammps", "md ", "structure", "property", "energy",
                              "convergence", "optimize", "calc")
        if any(s in hyp_lower for s in _workflow_signals):
            try:
                from huginn.skills.registry import SkillRegistry
                from huginn.skills.composite import _ensure_registered
                _ensure_registered()
                matches = SkillRegistry.search(hypothesis)
                if not matches:
                    matches = SkillRegistry.get_all_definitions()
                if matches:
                    lines = [s.to_prompt() for s in matches[:4]]
                    composite_block = "\nAvailable composite skills (prefer these over manual workflow):\n" + "\n\n".join(lines) + "\n"
            except Exception:
                logger.debug("composite skill lookup failed", exc_info=True)

        # Pipeline 建议: 基于 provenance 规则推荐下一步工具.
        # 42 条领域规则, 零 LLM 调用. 让 plan 知道"这类任务通常下一步是 X".
        pipeline_block = ""
        try:
            from huginn.provenance.pipeline import SimulationPipeline
            pipeline = SimulationPipeline(self.kg.root if hasattr(self.kg, 'root') else None)
            # 用上一轮的 execution_result 触发 suggest_next
            last_result = getattr(self, '_last_execution_result', None)
            if last_result and isinstance(last_result, dict):
                tool_name = last_result.get('_tool_name', '')
                suggestions = pipeline.suggest_next(
                    tool_name=tool_name,
                    tool_input=last_result.get('_tool_input', {}),
                    tool_output=last_result.get('result', last_result),
                )
                if suggestions:
                    s_lines = [f"  - {s.tool_hint}: {s.description}" for s in suggestions[:3]]
                    pipeline_block = "\nPipeline suggestions (based on provenance):\n" + "\n".join(s_lines) + "\n"
        except Exception:
            pass  # pipeline 是 advisory, 失败不阻塞

        return self._trim_to_budget([
            ("body", f"""Given the hypothesis: "{hypothesis}"

Context:
{json.dumps(context, indent=2, ensure_ascii=False)[:1000]}

Choose ONE mode and describe the plan:
- coder: modify code/files to fix or improve something
- workflow: run a computational simulation pipeline
- explore: search a design space for optimal parameters
- skill: use a pre-built composite skill pipeline (band structure, mechanical properties, MD, etc.)
- visual_inspect: interactively inspect visual data (zoom into chart region, measure data points, annotate structure). Use this when you need to examine previous results more carefully before deciding next steps. Available actions: zoom, measure, annotate, compare.

Protocol completeness check (RCBench failure mode: experimental protocol mismatch):
Before finalizing, verify your plan covers all necessary steps:
- For DFT: structure optimization BEFORE property calculation? Convergence test (encut/kpoints)?
- For MD: equilibration BEFORE production run? Timestep appropriate for the system?
- For analysis: raw data processing BEFORE interpretation? Reference comparison?
- Are computational parameters appropriate for the target property (e.g. HSE06 for band gap, not PBE)?
- Cross-check against domain knowledge above: any known methodological requirements?
If a step is missing, add it to DESCRIPTION.

When the hypothesis involves a PDE / variational principle / curved
geometry, consider the symbolic_math_tool actions listed in the math
depth block above — but numerical solvers are equally valid.

Respond in this exact format:
MODE: <coder|workflow|explore|skill>
DESCRIPTION: <brief description of what to do>
SKILL: <composite skill name, only if MODE is skill>
PREDICTION: <what you expect the result to look like — be specific: "energy ~ -X eV", "converges in ~N steps", "band gap ~X eV". This prediction will be compared against actual results to measure surprise.>
"""),
            ("math", math_block),
            ("kg", kg_block),
            ("visual", visual_block),
            ("kb", kb_block),
            ("mem", mem_block),
            ("skill", skill_hints + patch_hints),
            ("composite", composite_block),
            ("pipeline", pipeline_block),
            ("subgoal", self._build_subgoal_block()),
            ("ctx_hint", self._plan_context_hint()),
        ])

    def _plan_context_hint(self) -> str:
        """B: 把上下文信号转成 plan prompt 提示文本 (软路由).

        让 LLM 知道当前图状态/失败次数/refine 次数, 倾向选验证型 mode.
        硬路由在 _override_plan_mode 里做.
        """
        hints = []
        # 割点节点需要双覆盖 → 倾向选能跑验证的 mode
        try:
            current_hyp = getattr(self, "_current_hyp_id_for_plan", None)
            if current_hyp and self.hypothesis_graph.needs_dual_coverage(current_hyp):
                hints.append(
                    "CRITICAL: 当前假设是图的关键割点, 需要双模态验证. "
                    "优先选 workflow/skill 跑符号验证, 不要只选 coder."
                )
        except Exception:
            pass
        # 连续失败 → 倾向换方向
        cf = getattr(self, "_consecutive_failures", 0)
        if cf >= 3:
            hints.append(
                f"WARNING: 已连续失败 {cf} 次. 考虑 explore 换参数空间, "
                "或换一个完全不同的方法论."
            )
        # refine 次数多 → 假设可能方向错
        rc = getattr(self, "_refine_count", 0)
        if rc >= 3:
            hints.append(
                f"NOTE: 已 refine {rc} 次. 如果再失败可能需要 pivot 换方向."
            )
        # surprise 高 → 预测误差大, 倾向 explore 重新假设
        surprise = getattr(self, "_last_surprise", 0.0)
        if surprise > 0.5:
            hints.append(
                f"NOTE: 预测误差大 (surprise={surprise:.2f}). "
                "预测与实际差异显著, 考虑 explore 重新假设或换方法论."
            )
        if not hints:
            return ""
        return "\n\nContext signals:\n" + "\n".join(f"- {h}" for h in hints) + "\n"

    def _override_plan_mode(self, plan: dict[str, Any]) -> dict[str, Any]:
        """B: 硬路由 — 在 LLM 选完 mode 后, 根据硬性规则覆盖.

        只在极端情况覆盖, 不破坏 LLM 的常规选择:
        - needs_dual_coverage=True → mode 不能是 coder (必须能跑验证)
        - consecutive_failures >= 5 或 surprise > 0.9 → 强制 explore

        覆盖记到 PhaseGateState.history 补审计缺口 (reviewer="auto_router"),
        plan["override_reason"] 留结构化标记供调用方读取.

        ponytail: 只覆盖极端情况, 常规让 LLM 决定.
        budget tier 已由 _check_budget 处理, 这里不重复.
        升级: campaign 队列状态 (queue 满则 workflow 批量验证).
        """
        current_mode = plan.get("mode", "coder")
        # 割点节点: 强制非 coder mode
        try:
            current_hyp = getattr(self, "_current_hyp_id_for_plan", None)
            if (current_hyp
                    and self.hypothesis_graph.needs_dual_coverage(current_hyp)
                    and current_mode == "coder"):
                plan["mode"] = "workflow"
                plan["override_reason"] = "cut_vertex_dual_coverage"
                plan["description"] = (
                    f"[auto-routed: 割点需双覆盖] {plan.get('description', '')}"
                )
                logger.info("override mode coder→workflow for cut vertex %s",
                            current_hyp)
                self._log_plan_override(
                    "cut_vertex_dual_coverage", f"割点 {current_hyp} 需双覆盖")
        except Exception:
            pass
        # 连败/surprise 强制 explore (合并条件, 共享覆盖路径)
        cf = getattr(self, "_consecutive_failures", 0)
        surprise = getattr(self, "_last_surprise", 0.0)
        explore_reasons: list[str] = []
        if cf >= 5:
            explore_reasons.append(f"连续失败{cf}次")
        if surprise > 0.9:
            explore_reasons.append(f"surprise={surprise:.2f}")
        if explore_reasons and plan["mode"] != "explore":
            reason = "+".join(explore_reasons)
            plan["mode"] = "explore"
            plan["override_reason"] = "force_explore"
            plan["description"] = (
                f"[auto-routed: {reason}] {plan.get('description', '')}"
            )
            logger.info("override mode →explore: %s", reason)
            self._log_plan_override("force_explore", reason)
        return plan

    def _log_plan_override(self, reason_code: str, reason_text: str) -> None:
        """把 mode 覆盖记到 PhaseGateState.history, 补审计缺口.

        _override_plan_mode 之前只 logger.info, PhaseGate.history 不知道发生过
        覆盖. 现在复用 history 通道, reviewer="auto_router" 标记来源.
        失败不阻塞 (测试/无 phase_gate_hook 场景).
        """
        try:
            from huginn.autoloop.phase_gate import (
                get_shared_phase_gate_state, PhaseGate,
            )
            state = get_shared_phase_gate_state()
            state.history.append(PhaseGate(
                from_phase="plan", to_phase="plan",
                status="approved",
                feedback=f"[auto-routed] {reason_code}: {reason_text}",
                reviewer="auto_router",
            ))
        except Exception:
            logger.debug("log plan override failed", exc_info=True)

    def _parse_plan(self, response: str) -> dict[str, Any]:
        """Parse LLM plan response."""
        mode = "coder"
        description = response.strip()
        skill_name = ""
        prediction = ""

        for line in response.split("\n"):
            if line.startswith("MODE:"):
                mode = line.replace("MODE:", "").strip().lower()
            elif line.startswith("DESCRIPTION:"):
                description = line.replace("DESCRIPTION:", "").strip()
            elif line.startswith("SKILL:"):
                skill_name = line.replace("SKILL:", "").strip()
            elif line.startswith("PREDICTION:"):
                prediction = line.replace("PREDICTION:", "").strip()

        plan = {"mode": mode, "description": description}
        if skill_name:
            plan["skill"] = skill_name
        if prediction:
            plan["expected_prediction"] = prediction
        return plan

    # ──────────────────────────────────────────────────────────────
    # Phase runner utilities
    # ──────────────────────────────────────────────────────────────

    def _run_phase(self, name: str, fn, *args) -> LoopPhase:
        """Run a synchronous phase function."""
        phase = LoopPhase(name=name)
        phase.start_time = time.time()
        phase.status = "running"
        # 同步路径: 如果当前在 event loop 里, fire-and-forget 发开始事件.
        # 不 await, 因为 _run_phase 本身是同步的.
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(
                self._dispatch_stage_event(
                    EventType.ON_WORKFLOW_STAGE_START, name
                )
            )
        except RuntimeError:
            logger.warning("error in _run_phase: stage-start event dispatch skipped (no running loop)", exc_info=True)
        # 包 telemetry span: 把 phase 级决策也记进轨迹, 回放时不止看 tool_call
        from huginn.telemetry import get_telemetry_collector
        span_cm = get_telemetry_collector().span(f"phase:{name}")
        try:
            with span_cm as phase_span:
                phase.result = fn(*args)
                phase.status = "completed"
                phase_span.metadata["status"] = "completed"
        except Exception as e:
            phase.status = "failed"
            phase.error = str(e)
            try:
                phase_span.metadata["status"] = "failed"
                phase_span.metadata["error"] = str(e)
            except Exception:
                logger.warning("error in _run_phase: span metadata update failed", exc_info=True)
        phase.end_time = time.time()
        # fire-and-forget 发结束/失败事件
        try:
            loop = asyncio.get_running_loop()
            done_type = (
                EventType.ON_WORKFLOW_STAGE_DONE
                if phase.status == "completed"
                else EventType.ON_WORKFLOW_FAILED
            )
            loop.create_task(
                self._dispatch_stage_event(
                    done_type,
                    name,
                    duration_sec=phase.end_time - (phase.start_time or 0),
                    error=phase.error,
                )
            )
        except RuntimeError:
            logger.warning("error in _run_phase: stage-done event dispatch skipped (no running loop)", exc_info=True)
        return phase

    async def _run_phase_async(self, name: str, fn, *args) -> LoopPhase:
        """Run an async phase function."""
        phase = LoopPhase(name=name)
        phase.start_time = time.time()
        phase.status = "running"
        await self._dispatch_stage_event(
            EventType.ON_WORKFLOW_STAGE_START, name
        )
        from huginn.telemetry import get_telemetry_collector
        span_cm = get_telemetry_collector().span(f"phase:{name}")
        try:
            with span_cm as phase_span:
                phase.result = await fn(*args)
                phase.status = "completed"
                phase_span.metadata["status"] = "completed"
        except Exception as e:
            phase.status = "failed"
            phase.error = str(e)
            try:
                phase_span.metadata["status"] = "failed"
                phase_span.metadata["error"] = str(e)
            except Exception:
                logger.warning("error in _run_phase_async: span metadata update failed", exc_info=True)
        phase.end_time = time.time()
        if phase.status == "completed":
            await self._dispatch_stage_event(
                EventType.ON_WORKFLOW_STAGE_DONE,
                name,
                duration_sec=phase.end_time - (phase.start_time or 0),
            )
        else:
            await self._dispatch_stage_event(
                EventType.ON_WORKFLOW_FAILED,
                name,
                duration_sec=phase.end_time - (phase.start_time or 0),
                error=phase.error,
            )
        return phase

    # ──────────────────────────────────────────────────────────────
    # Report rendering
    # ──────────────────────────────────────────────────────────────

    def _render_report(self, data: dict[str, Any]) -> str:
        """Render a markdown report."""
        lines = [
            f"# Huginn Autoloop Report",
            f"",
            f"**Objective:** {data['objective']}",
            f"**Run ID:** {data['run_id']}",
            f"**Total Time:** {data['total_time_seconds']:.1f}s",
            f"",
            f"## Phases",
            f"",
            f"| Phase | Status | Duration (s) | Error |",
            f"|-------|--------|--------------|-------|",
        ]
        for p in data["phases"]:
            lines.append(f"| {p['name']} | {p['status']} | {p['duration']:.1f} | {p['error'] or ''} |")
        lines.append("")
        lines.append("---")
        lines.append("Generated by Huginn Autoloop Engine")
        return "\n".join(lines)
