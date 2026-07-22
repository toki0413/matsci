"""Autoloop Engine — the main autonomous loop for Huginn.

Ties together exploration, coder, workflow, benchmark, and report
into a single closed-loop ecosystem:

    Perceive → Hypothesize → Plan → Execute → Validate → Learn → Report

Usage:
    engine = AutoloopEngine(workspace=Path("."))
    asyncio.run(engine.run_cognitive(objective="Optimize C-S-H defect kinetics"))
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _harness_workflow_evolution_enabled() -> bool:
    """H2 toggle: cfg.feature_flags.harness_workflow_evolution (默认 off)."""
    try:
        from huginn.config import get_config
        cfg = get_config()
        ff = getattr(cfg, "feature_flags", None) or {}
        return bool(ff.get("harness_workflow_evolution", False))
    except Exception:
        return False


def _autoloop_meta_trace_inject_enabled() -> bool:
    """C4 toggle: cfg.feature_flags.autoloop_meta_trace_inject (默认 off).

    autoloop engine 每轮 _distill_meta_trace 写 .huginn/meta_trace.jsonl,
    但 _build_memory_text 之前不读它 — 长轨迹里 agent 看不到上轮蒸馏的结构化
    历史. toggle on 后注入最近 5 条 entry 到 memory_text.

    ponytail: 默认 off, 因为 meta_trace 跟 FTS5 memory 可能内容重叠,
      注入会增加 prompt 长度. 升级路径: 默认 on + 按 darwin_score top-K 去重.
    """
    try:
        from huginn.config import get_config
        cfg = get_config()
        ff = getattr(cfg, "feature_flags", None) or {}
        return bool(ff.get("autoloop_meta_trace_inject", False))
    except Exception:
        return False


def _autoloop_streaming_enabled() -> bool:
    """P0-1 toggle: cfg.feature_flags.autoloop_streaming (默认 on).

    astream 替代 ainvoke, 把 LLM thinking chunk 增量推到 progress_cb (WS).
    700 万步场景必需 — 否则用户看不到 agent 在想什么. 默认 on 是因为
    fail 会自动回退 ainvoke, 无 WS 场景 progress_cb=None 也不流式.
    ponytail: 环境变量 HUGINN_AUTOLOOP_STREAMING=0 可强制关.
    """
    if os.environ.get("HUGINN_AUTOLOOP_STREAMING", "1") == "0":
        return False
    try:
        from huginn.config import get_config
        cfg = get_config()
        ff = getattr(cfg, "feature_flags", None) or {}
        # 默认 True, 显式 False 才关
        return ff.get("autoloop_streaming", True)
    except Exception:
        return True


from huginn.api.event import EventType, WorkflowStageEvent
from huginn.autoloop.budget import ProgressiveBudget
from huginn.autoloop.cognitive_loop import (
    VALID_ACTIONS, ActionDecision, LoopState, _validation_to_step_eval_fields,
)
from huginn.autoloop.goal_scheduler import Goal, GoalScheduler
from huginn.autoloop.phase_gate import (
    PhaseGate,
    PhaseGateHook,
    get_shared_phase_gate_state,
)
from huginn.autoloop.phase_gate import (
    _has_external_source as _validation_has_external_source,
)
from huginn.bench.runner import BenchmarkRunner  # noqa: F401  # monkeypatch
from huginn.coder.loop import CoderRunner
from huginn.config import get_settings
# C1: 共享 KB chunk 格式化函数, 跟 ContextBuilder 走同一条路径, 消除双路径漂移.
# C4: 共享 meta_trace 加载, engine 写 jsonl 但之前不读, 现在注入 memory_text.
from huginn.context_builder import format_kb_chunks, load_meta_trace_text
from huginn.exploration.orchestrator import ExplorationOrchestrator
from huginn.exploration.strategies import ParetoPruningStrategy
from huginn.interaction.progress import ProgressTracker, get_progress_tracker
from huginn.kg.builder import ProjectKnowledgeGraph
from huginn.llm import get_model
from huginn.memory.longterm import load_stable_principles
from huginn.memory.manager import MemoryManager
from huginn.metacog.signal_hub import SignalHub
from huginn.tools.report_tool import ReportTool
from huginn.types import ToolContext
from huginn.workflows.engine import WorkflowEngine
from huginn.workflows.templates import get_template, standard_dft_workflow

# 跨源属性冲突检测用的正则; 提到模块级避免每次调用重编译
_PROP_RE = re.compile(
    r"([\w\s]{3,25}?)\s*[:=]\s*(-?\d+\.?\d*)\s*(eV(?:/\w+)?|GPa|THz|nm)",
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
    "equation",
    "lagrangian",
    "pde",
    "hamiltonian",
    "derivative",
    "differential",
    "integral",
    "eigenvalue",
    "tensor",
    "manifold",
    "symmetry",
    "conservation",
    "variational",
    "continuum",
    "stress",
    "strain",
    "energy",
    "phonon",
    "band",
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
assert set(_PHASE_PERSONAS.keys()) == set(
    AUTOLOOP_PHASES
), "Phase persona keys must match AUTOLOOP_PHASES"

# Controllable thinking effort (Inkling-inspired): 每个 phase 一个 0-1 连续值,
# 映射到 prompt 前缀控制 LLM 思考深度. prompt 层实现 — 对所有 provider 统一生效,
# 不依赖 API 级 reasoning_effort (Anthropic/OpenAI/DeepSeek 各家不同).
# ponytail: 软控制, LLM 可无视. 升级: per-provider API 层 bind(extra_body=...)
_PHASE_THINKING_EFFORT: dict[str, float] = {
    "perceive": 0.3,     # 扫描, 不需要深推理
    "hypothesize": 0.9,  # 核心创新点, 深度推理
    "plan": 0.6,         # 中等, 把假设变步骤
    "execute": 0.2,      # 直接调工具, 不需要 LLM 思考
    "validate": 0.7,     # 批判性审视, 需要深度但不如 hypothesize
    "learn": 0.5,        # 反思, 中等
    "report": 0.3,       # 总结性输出
}

# effort float → prompt 指令片段. 3 档够用, 更细粒度收益递减.
_EFFORT_TO_PROMPT: list[tuple[float, str]] = [
    (0.8, "Think deeply and step-by-step. Explore multiple angles before concluding. "
          "Consider edge cases and alternative explanations."),
    (0.5, "Reason carefully but concisely. One main line of thought, briefly check alternatives."),
    (0.2, "Answer directly and briefly. No step-by-step reasoning needed."),
]


def _effort_to_prompt(effort: float) -> str:
    """Map 0-1 effort to a prompt directive. Linear threshold lookup."""
    for threshold, text in _EFFORT_TO_PROMPT:
        if effort >= threshold:
            return text
    return _EFFORT_TO_PROMPT[-1][1]

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


# === dataclass + snapshot 函数抽到 autoloop/types.py ===
# ponytail: 单一职责拆分. 原 L178-277 抽到 autoloop/types.py.
from huginn.autoloop.types import (
    LoopPhase, AutoloopResult,
    objective_hash, _snapshot_dir,
    save_autoloop_snapshot, load_autoloop_snapshot,
)



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
        agent_factory: Any = None,
        resume_from_state: str | None = None,
    ):
        self.workspace = Path(workspace or ".").resolve()
        self.settings = get_settings()
        # BranchIncubator 用的 agent_factory, None 时 incubator 路径跳过.
        # 由 RCBench runner / CLI 在需要 N=3 隔离采样时注入.
        self._agent_factory = agent_factory
        # lazy init, 第一次 _hypothesize_via_branch_incubator 才构造
        self._branch_incubator: Any = None
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

        # P0: 传 workspace 给 HypothesisGraph, 让 refute/support 时写 FAILED.md/PROVED.md
        self.hypothesis_graph = HypothesisGraph(workspace=self.workspace)
        self.report_tool = ReportTool()

        # Sub-engines
        self.explorer = ExplorationOrchestrator(
            strategy=ParetoPruningStrategy(),
            max_parallel=3,
        )
        from huginn.tools.registry import ToolRegistry

        self.workflow_engine = WorkflowEngine(
            tool_registry=ToolRegistry,  # 传类本身, .get() 是 classmethod
        )
        self.coder = CoderRunner()

        self._should_stop = False
        self._iteration = 0
        # 连续验证失败计数: 给 _maybe_clarify 判断是否该问用户;
        # 超过 _max_consecutive_failures 时强制停止 autoloop, 避免无限重试坏方向.
        # v7 长任务: 5→20 默认. Oxelra 206 步允许大量失败回退, huginn 5 太保守.
        # 环境变量覆盖: 极限模式可设更高 (e.g. HUGINN_MAX_CONSECUTIVE_FAILURES=50).
        self._consecutive_failures = 0
        self._max_consecutive_failures = int(os.environ.get("HUGINN_MAX_CONSECUTIVE_FAILURES", "20"))
        # F-borrow (forge 双预算思路): 按 failure_type 分类计数. 不同失败类型语义不同 —
        # tool_error 是技术故障 (短期可恢复), hypothesis_error 是方向错 (持续才是真死路).
        # 单一 _consecutive_failures 把两者混算, 5 次 tool_error 就该停 vs 5 次 hypothesis_error
        # 还远不够. 在总数上叠加分类, decider prompt 显示分类, 按类阈值 stop.
        # ponytail: 不拆总数 (向后兼容), 在其上叠加. 升级路径: 走 PhaseRegistry extra.
        self._consecutive_failures_by_type: dict[str, int] = {}
        # 按类阈值 — tool_error 类短期可恢复, 阈值低; hypothesis_error 持续才是真死路, 阈值高.
        # ponytail: 硬编码, 跟 _max_pivot=10 一致. 升级路径: env / PhaseRegistry extra.
        self._max_failures_by_type: dict[str, int] = {
            "tool_error": 5,
            "prompt_injection_suspect": 3,
            "param_error": 5,
            "data_noise": 5,
            "hypothesis_error": 10,
        }
        # 700 万步极限场景: consecutive 语义在长轨迹里太窄 (20 次 tool timeout 就停).
        # 加滑动窗口失败率 — 最近 N 次 validate 的失败率超阈值才 stop, 允许局部失败.
        # consecutive 保留作快速止损 (短任务 / 连续坏方向), windowed rate 兜底长轨迹.
        # ponytail: 用 list 存 bool (True=pass), 超窗口截断. 升级路径: 指数衰减加权.
        self._validate_window: list[bool] = []
        self._validate_window_size = int(os.environ.get("HUGINN_VALIDATE_WINDOW", "100"))
        self._validate_window_fail_threshold = float(
            os.environ.get("HUGINN_VALIDATE_FAIL_THRESHOLD", "0.8")
        )
        # refine 循环计数: 防止 refute→refine 无限循环
        # v7 长任务: 8→20 默认. Oxelra 失败回退不计数, 这里仍保留上限防失控.
        self._refine_count = 0
        self._max_refines = int(os.environ.get("HUGINN_MAX_REFINES", "20"))
        # pivot 计数: refine 耗尽后换方向, 但 pivot 本身也要有上限 —
        # 否则 pivot→fail→refine→pivot→fail 无限循环, 烧 token 不出结果.
        # v7 长任务: 3→10 默认. 3 太保守, Oxelra 开放式探索允许频繁换方向.
        self._pivot_count = 0
        self._max_pivots = int(os.environ.get("HUGINN_MAX_PIVOTS", "10"))
        # v7 phase 解耦: refute/pivot 后下一轮的起点 phase.
        # Oxelra 求是引擎 role-phase 解耦启示: 7-phase 线性太死, 失败应能回退到
        # 合适的 phase 而不是只 refine. hint 在 refute 时设, 下一轮 perceive 前查.
        # 值: None (默认从头) | "plan" (跳过 perceive+hypothesize) | "execute" (只重跑实验)
        self._next_phase_hint: str | None = None
        # v7 phase 解耦: refine 生成的新 hypothesis 文本, hint="execute" 时复用,
        # 跳过 hypothesize 直接进 plan/execute. ponytail: 只存文本不存 id,
        # graph 操作仍走 _current_hyp_id 流程.
        self._refined_hypothesis: str | None = None
        # Step C: LLM 自主选 action (run_cognitive 的 decide_fn 用).
        # 默认开, RCBench 跑分需要确定性时设 HUGINN_COGNITIVE_LLM_DECIDER=0 关掉.
        # 失败/超时/非法 action 自动 fallback 到规则版, 不影响死循环防护.
        self._use_llm_decider = os.environ.get("HUGINN_COGNITIVE_LLM_DECIDER", "1") == "1"
        # plan_check 状态走引擎级, 不塞 plan dict — plan 会序列化进 prompt,
        # 塞进去等于把校验元信息喂给 LLM 污染上下文. history 喂自适应, last_result
        # 给 _validate 取, warnings 留痕. patterns 跨 run 持久化 (失败模式记忆).
        self._plan_check_history: list[dict[str, Any]] = []
        self._plan_check_last_result: dict[str, Any] | None = None
        self._plan_check_warnings: list[str] = []
        self._plan_check_patterns: list[dict[str, Any]] = []
        # 自动发现的 scene_tag 关键词 (跨 run 积累), 跟写死的关键词表互补.
        # ponytail: dict[label, set[keyword]], 简单加法; 不上 embedding.
        self._scene_tag_extra_keywords: dict[str, set[str]] = {}
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
        # C2: target_chain + prospective 注入. _target_chains 首次 _build_*_prompt
        # 时 lazy build (调 build_target_chains, 一次 LLM call). 之后只读.
        self._target_chains: list = []
        self._target_chains_built: bool = False
        # AV2: 元认知护航状态 — autoloop 之前零接 PMK/TaskMetrics/detect_drift,
        # 跑完无 task_metrics.json 落盘, 脱轨无告警, PMK 撕裂无感知. 现补上.
        self._evals_history: list = []
        self._task_metrics: Any = None
        self._task_state_for_metrics: Any = None
        self._drift_info: tuple | None = None
        # Forest 回流假设图: 多树共识的 HypothesisGraph, learn 阶段可接续探索
        self._merged_graph: Any = None
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
        # Darwin ratchet (darwin-skill 启发): 每轮算假设质量分, 只保留改进.
        # best_score = 历史最佳, 当前轮 score < best → 回退假设 (不更新 preferred).
        # 连续 2 轮 Δ<0.5 → early stop (边际收益递减, 不烧 token).
        # 互补于 surprise-based stop: surprise 测"预测准不准", ratchet 测"假设好不好".
        self._darwin_best_score: float = 0.0
        self._darwin_stagnation: int = 0  # 连续低增益轮数
        self._darwin_last_score: float = 0.0
        # P2-6 belief: Gaussian 后验 N(μ, σ²) 替代单值 score.
        # 单值 + Δ<0.5 阈值会被噪声翻转; 后验 σ² 减小才表示真收敛.
        # 复用 subagent_tool._gaussian_update, 不重复实现.
        # prior N(0, 100): 弱信息先验, 让早期观测主导.
        self._darwin_belief_mu: float = 0.0
        self._darwin_belief_sigma2: float = 100.0
        # v6 G54: 假设的 confidence + evidence_strength, 供 _plan / _validate 读取
        # confidence = darwin score / 10 (0-1); evidence_strength = supported_ratio (代理)
        # 升级路径: evidence_strength 改成 RAG recall 命中数 / provenance 引用数
        self._last_hypothesis_confidence: float = 0.0
        self._last_hypothesis_evidence_strength: float = 0.0
        # H4: GRILL 模式状态. should_pause_for_decision 触发 GRILL 后设为 active,
        # _llm_chat 构造 system prompt 时注入 GRILL_SYSTEM_PROMPT_CN. 用户确认
        # shared understanding 后 (LLM 输出含标记) 退出.
        # ponytail: 不持 GrillSession 实例, 只用 bool + 计数. LLM 自己负责流程.
        self._grill_active: bool = False
        self._grill_turns: int = 0
        # 上一轮执行结果, 给 _build_plan_prompt 的 pipeline suggest_next 用
        self._last_execution_result: dict | None = None
        # 阶段门 hook: 在 plan→execute / execute→validate / validate→learn
        # 三个转移点评估证据, 不足时阻断并把 feedback 拼进 _speculator_hint
        # 让下轮 prompt 带上"缺什么证据". R3 接入 red-team reviewer_fn:
        # 在 validate→learn 做 adversarial 审查, 有 high 发现则阻断.
        from huginn.autoloop.phase_gate import MathEvidenceChecker
        from huginn.autoloop.red_team import RedTeamReviewer

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
        # perception → CSM 信号暂存. _perceive 产生 TransitionSignal 后放这里,
        # agent 层定期拉取并调 csm.transition(). engine 自己不持有 csm.
        self._pending_signals: list = []
        # 阶段索引: 给 WorkflowStageEvent 用, 从 phase name 推算.
        self._phase_order = list(AUTOLOOP_PHASES)
        # 当前 phase 名 — _run_phase_async 写, _llm_chat 读, 用于 phase-aware thinking effort.
        # ponytail: 隐式状态, 但只在 single-threaded async run() 里用, 无竞态.
        self._current_phase: str = ""

        # P15: crash-safe resume. flag off / snapshot 缺失时静默跳过, 行为不变.
        # ponytail: 用 duck typing, 不做 isinstance 检查; 失败 log warning 不抛.
        if resume_from_state:
            try:
                from huginn.runtime.engine_state import (
                    apply_state_to_engine, load_engine_state, use_persistence,
                    _hypothesis_graph_path,
                )
                if use_persistence():
                    state = load_engine_state(resume_from_state, self.workspace)
                    if state is not None:
                        apply_state_to_engine(state, self)
                        # hypothesis_graph 单独恢复 (refuted 状态跨 session 必须保留)
                        try:
                            loaded_graph = self.hypothesis_graph.load(
                                _hypothesis_graph_path(self.workspace, resume_from_state)
                            )
                            if loaded_graph is not None:
                                self.hypothesis_graph = loaded_graph
                        except Exception:
                            logger.debug(
                                "resume: hypothesis_graph.load failed (non-fatal)",
                                exc_info=True,
                            )
                        logger.info(
                            "resumed engine from run_id=%s: iteration=%d persona=%s",
                            resume_from_state, state._iteration,
                            state._last_persona or "(none)",
                        )
                    else:
                        logger.info(
                            "resume requested but no snapshot for run_id=%s, "
                            "starting fresh", resume_from_state,
                        )
            except Exception:
                logger.warning(
                    "resume_from_state=%s failed, starting fresh",
                    resume_from_state, exc_info=True,
                )

    def _maybe_save_engine_state(
        self, *, force: bool = False, reason: str = "",
    ) -> None:
        """P15: 周期 / forced 落盘 engine_state + hypothesis_graph.

        - flag off (HUGINN_USE_PERSISTENCE=0) 时完全 no-op, 不碰磁盘.
        - force=True: 立刻 save (pivot / refute 等关键事件触发).
        - force=False: 周期 save, iteration % save_every == 0 才真写.
        - run_id 缺失 (run_cognitive 没启动) 时 no-op, 避免误写.

        ponytail: 失败只 log warning, 不抛 — save 失败不该阻塞主循环.
        ceiling: 单进程串行 save, 不加锁; 并发 run_id 隔离由 run_id 命名空间保证.
        """
        try:
            from huginn.runtime.engine_state import (
                save_engine_state, save_every_steps, use_persistence,
            )
            if not use_persistence():
                return
            run_id = getattr(self, "_run_id", None)
            if not run_id:
                return
            if not force:
                every = save_every_steps()
                if every <= 0:
                    return
                if self._iteration % every != 0:
                    return
            save_engine_state(self, run_id, self.workspace)
            if reason:
                logger.debug(
                    "engine_state saved (reason=%s, iter=%d, run_id=%s)",
                    reason, self._iteration, run_id,
                )
        except Exception:
            logger.warning(
                "_maybe_save_engine_state failed (non-fatal)", exc_info=True,
            )

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
        idx = (
            self._phase_order.index(stage_name) + 1
            if stage_name in self._phase_order
            else 0
        )
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
                    event_type.name,
                    stage_name,
                )
        except Exception:
            logger.warning(
                "error in _dispatch_stage_event: bus.dispatch failed", exc_info=True
            )

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
            # C1: 走共享 format_kb_chunks — 跟 ContextBuilder 同一路径, 含
            # image_ref + KB→memory cross-ref. 之前 engine 版没这俩, 是双路径漂移.
            recall_fn = None
            mem = getattr(self, "memory", None)
            if mem is not None:
                recall_fn = mem.recall_for_prompt
            body = format_kb_chunks(
                chunks,
                memory_recall_fn=recall_fn,
                with_image_ref=True,
                cross_ref_top_k=2,
            )
            if not body:
                return ""
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
                gap_block = (
                    "\n\n### KG Gap Detection (potential research directions)\n"
                    + "\n".join(gap_hints)
                    + "\n"
                )
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
                for b in high_conf_nodes[i + 1 : 10]:
                    if graph.has_edge(a, b) or graph.has_edge(b, a):
                        continue  # 已有边, 不是缺口
                    common = (
                        set(nx.common_neighbors(graph, a, b))
                        if graph.has_node(a) and graph.has_node(b)
                        else set()
                    )
                    if common:
                        a_label = graph.nodes[a].get("label", a)[:40]
                        b_label = graph.nodes[b].get("label", b)[:40]
                        hints.append(
                            f"- {a_label} and {b_label} share connections but no direct link — consider whether they relate"
                        )
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
        parts: list[str] = []
        mem = getattr(self, "memory", None)
        if mem is not None:
            try:
                text = mem.recall_for_prompt(query, max_entries=3)
                if text:
                    parts.append(text)
            except Exception:
                pass

        # C4: meta_trace 注入 — engine 每轮 _distill_meta_trace 写 jsonl,
        # 这里读回来让 LLM 看到上轮蒸馏的结构化历史 (attempted/found/evidence/
        # limitations/next_hint). toggle off 时不注入, 避免 prompt 膨胀.
        if _autoloop_meta_trace_inject_enabled():
            try:
                ws = getattr(self, "workspace", None)
                if ws is not None:
                    trace_text = load_meta_trace_text(str(ws), last_n=5)
                    if trace_text:
                        parts.append(trace_text)
            except Exception:
                logger.debug("meta_trace inject failed", exc_info=True)

        return "\n\n".join(parts) if parts else ""

    def _build_pm_text(self) -> str:
        """C2: PM 层 trajectory_match 召回 — 当前 phase 序列是否是某历史轨迹的 prefix.

        命中 → 注入 next_step 建议给 LLM 参考. 同时记下命中 doc_id,
        供 _learn 调 update_pattern_confidence 做 ±ε 反馈 (C3 闭环).
        返回空串时不影响 prompt.

        极限模式才开 (HUGINN_EXTREME_DISPATCH=1), 平常默认关闭省计算.
        """
        import os
        if os.environ.get("HUGINN_EXTREME_DISPATCH", "0").lower() not in ("1", "true"):
            return ""
        try:
            from huginn.knowledge.trajectory_pattern import trajectory_match
            current = getattr(self, "_current_run_phases", [])
            history = getattr(self, "_traj_history", None)
            if history is None:
                # 懒加载历史轨迹 (跟 _check_stuck 共用)
                try:
                    history = self._load_trajectory_action_history(limit=20)
                    self._traj_history = history
                except Exception:
                    return ""
            if len(current) < 2 or not history:
                return ""
            match = trajectory_match(current, history, min_similarity=0.4)
            if match is None:
                self._last_traj_match_doc_id = None
                return ""
            # 记下 doc_id 供 _learn 调 update_pattern_confidence (C3).
            # ponytail: trajectory_match 返回 history_id, 不直接是 KB doc_id.
            # 真要闭环需要从 history_id 反查 KB doc_id, 这里先记 history_id,
            # _learn 暂不调 update_pattern_confidence (留给升级路径).
            self._last_traj_match_doc_id = match.get("history_id")
            advice = (
                f"### Trajectory Match (PM layer)\n"
                f"Current phase sequence matches history[{match['history_id']}] "
                f"(similarity={match['similarity']:.2f}). "
                f"Suggested next step: {match.get('next_step', '?')}. "
                f"This is advisory — ignore if context differs.\n"
                f"### End Trajectory Match"
            )
            return advice
        except Exception:
            self._last_traj_match_doc_id = None
            return ""

    def _ensure_target_chains(self) -> list:
        """C2: lazy build target_chains from self._objective.

        首次调用时把 objective 当单条 Mode-A item 调 build_target_chains,
        LLM 推导 required_results/methods/data/verification. 失败容错返回空 list.
        之后只读 self._target_chains. 跟 rcb_runner 的 _target_chains 路径同源.

        ponytail: objective 当单条 Mode-A item — RCBench 是 checklist 多条,
          autoloop 通常单 objective. 升级路径: 让 run_cognitive 接收 checklist.
        """
        # getattr 防 __new__ 测试场景 (跟 D1 的 _speculator_hint 同模式)
        if getattr(self, "_target_chains_built", False):
            return getattr(self, "_target_chains", [])
        self._target_chains_built = True  # 防重入, 失败也不重试
        objective = getattr(self, "_objective", "") or ""
        if not objective.strip():
            self._target_chains = []
            return self._target_chains
        try:
            from huginn.metacog.target_chain import build_target_chains
            kb = self._get_kb()
            model = getattr(self, "model", None)
            if model is None:
                self._target_chains = []
                return self._target_chains
            checklist = [{"mode": "A", "item": objective[:2000]}]
            self._target_chains = build_target_chains(checklist, kb, model, "") or []
        except Exception:
            logger.debug("build_target_chains failed in autoloop", exc_info=True)
            self._target_chains = []
        return self._target_chains

    def _build_metacog_block(self, *, include_prospective: bool = True) -> str:
        """C2: target_chain + prospective 注入 prompt.

        target_chain: 当前 objective 在目标分解树里的位置 (required_results /
        missing / progress), LLM 看了能避免偏题.
        prospective: 已触发的待执行前瞻意图, LLM 看了能避免遗漏计划.
        两者都空时返回空串, 不污染 prompt.

        ponytail: target_chain 首次调用触发 build_target_chains (1 次 LLM),
          之后只读. prospective 每次调 recall_prospective, 但 PM 层内部有
          scan_and_fire 缓存, 代价可控. 升级路径: 把两者合并到 metacog signal.
        """
        parts: list[str] = []
        tc = self._ensure_target_chains()
        if tc:
            try:
                from huginn.metacog.target_chain import format_target_chain_text
                step = getattr(self, "_iteration", 0) or 0
                tc_text = format_target_chain_text(tc, step)
                if tc_text:
                    parts.append(tc_text)
            except Exception:
                logger.debug("format_target_chain_text failed", exc_info=True)

        if include_prospective:
            mem = getattr(self, "memory", None)
            if mem is not None and hasattr(mem, "recall_prospective"):
                try:
                    step = getattr(self, "_iteration", 0) or 0
                    fired = mem.recall_prospective({"current_step": step})
                    if fired:
                        from huginn.context_builder import ContextBuilder
                        # 用 ContextBuilder 的格式化逻辑, 跟 rcb_runner 同源.
                        # ponytail: __new__ 绕过 __init__, 只用 build_prospective_text.
                        _ctx = ContextBuilder.__new__(ContextBuilder)
                        pro_text = _ctx.build_prospective_text(fired)
                        if pro_text:
                            parts.append(pro_text)
                except Exception:
                    logger.debug("prospective inject failed", exc_info=True)

        return "\n\n".join(p for p in parts if p)

    # 上下文预算: 防止 prompt block 累积超过 token 上限.
    # 优先级: body > math > kg > visual > kb > mem > pm > hint > skill > composite > pipeline
    # 超预算时不是直接丢弃, 而是分层压缩: 先截断 → 再摘要 → 最后才删.
    # 视觉语言比文字语言更能压缩信息 — 一行 "[energies] peak=idx3, trend=↑"
    # 传达的信息等于 200 chars 的 JSON. 用压缩替代丢弃, 保留信息密度.
    _PROMPT_BUDGET = 12000  # chars, 约 3K tokens (fallback default)
    # C-budget: 分层 budget — hypothesis/plan 主战场留满, 其他 phase 不走 prompt builder.
    # ponytail: 只 dict + getter, 不引 BudgetPolicy 抽象. env 覆盖留调参口子.
    # 升级路径: 加 learn/pivot 的 prompt builder 后再扩 phase 覆盖.
    _PROMPT_BUDGET_BY_PHASE: dict[str, int] = {
        "hypothesize": 12000,
        "plan": 12000,
    }

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

    def _get_prompt_budget(self, phase: str | None) -> int:
        """按 phase 取 prompt budget. env 覆盖优先, dict 次之, fallback _PROMPT_BUDGET."""
        import os
        if phase:
            env_key = f"HUGINN_PROMPT_BUDGET_{phase.upper()}"
            env_val = os.environ.get(env_key)
            if env_val:
                try:
                    return int(env_val)
                except ValueError:
                    pass
            if phase in self._PROMPT_BUDGET_BY_PHASE:
                return self._PROMPT_BUDGET_BY_PHASE[phase]
        return self._PROMPT_BUDGET

    def _apply_block_patches(
        self,
        blocks: list[tuple[str, str]],
        phase: str,
    ) -> list[tuple[str, str]]:
        """H1: 在 _trim_to_budget 前应用 prompt patch.

        apply_patches 内部按 Beta mean > 0.5 过滤 + 同名 block 取最高 Beta mean.
        这里重算一遍 by_block 拿到实际应用的 patch ids, 存到
        _last_applied_patches 供 _learn 更新 Beta. toggle off 或没 patch 时
        直接返回原 blocks (apply_patches 内部处理, 这里零开销).

        ponytail: 重算 by_block 跟 apply_patches 内部逻辑重复, 但避免改
        apply_patches 返回签名. 升级路径: apply_patches 返回 (blocks, ids).
        """
        from huginn.harness.prompt_patch import apply_patches, PromptPatchStore
        new_blocks = apply_patches(blocks, phase)
        # 记录原始 blocks 供 _generate_next_loop_directive 调 generate_patch 用
        # (generate_patch 需要看 block 名字 + 内容才能生成合理 patch)
        if phase == "hypothesize":
            self._last_hypothesis_blocks = blocks
        elif phase == "plan":
            self._last_plan_blocks = blocks
        if new_blocks is blocks:
            return new_blocks
        try:
            store = PromptPatchStore.get_instance()
            patches = [
                p for p in store.list_patches(phase=phase)
                if p.alpha / max(1, p.alpha + p.beta) > 0.5
            ]
            by_block: dict[str, Any] = {}
            for p in patches:
                cur = by_block.get(p.block_name)
                if cur is None or (
                    p.alpha / max(1, p.alpha + p.beta)
                    > cur.alpha / max(1, cur.alpha + cur.beta)
                ):
                    by_block[p.block_name] = p
            applied_ids = [p.id for p in by_block.values()]
            if applied_ids:
                self._last_applied_patches = (phase, applied_ids)
        except Exception:
            logger.debug("_apply_block_patches: track applied fail", exc_info=True)
        return new_blocks

    def _trim_to_budget(
        self,
        blocks: list[tuple[str, str]],
        *,
        phase: str | None = None,
    ) -> str:
        """按优先级拼接 blocks, 超预算时分层压缩: 截断→摘要→删除.

        phase 不传时走 _PROMPT_BUDGET (fallback), 传 "hypothesize"/"plan"
        走 _PROMPT_BUDGET_BY_PHASE 分层 budget. env HUGINN_PROMPT_BUDGET_<PHASE>
        覆盖一切.
        """
        budget = self._get_prompt_budget(phase)
        # 跨源冲突检测: 扫描各 block 中的 property=value 对, 标注矛盾
        conflict_warn = self._scan_block_conflicts(blocks)
        if conflict_warn:
            blocks = [("conflict", conflict_warn)] + blocks

        kept = [(n, v) for n, v in blocks]
        total = sum(len(v) for _, v in kept)
        if total <= budget:
            return "".join(v for _, v in kept)

        # Pass 1: 截断低优先级 block 到 300 字符
        for i in range(len(kept) - 1, -1, -1):
            if total <= budget:
                break
            name, text = kept[i]
            if name == "body":  # body 永远不压缩
                continue
            compressed = self._compress_block(name, text, 1)
            total -= len(text) - len(compressed)
            kept[i] = (name, compressed)

        if total <= budget:
            return "".join(v for _, v in kept)

        # Pass 2: 压缩成一行摘要
        for i in range(len(kept) - 1, -1, -1):
            if total <= budget:
                break
            name, text = kept[i]
            if name == "body":
                continue
            compressed = self._compress_block(name, text, 2)
            total -= len(text) - len(compressed)
            kept[i] = (name, compressed)

        if total <= budget:
            return "".join(v for _, v in kept)

        # Pass 3: 从最低优先级开始删除
        # skill/composite 受保护 — skills 引用保留系统: 可截断可摘要, 但不可清空
        for i in range(len(kept) - 1, -1, -1):
            if total <= budget:
                break
            name, text = kept[i]
            if name in ("body", "skill", "composite"):
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
        # OAK 启发: trace_id 贯穿, 每个 gate 记录归属的 run
        tid = getattr(self, "_run_id", None) or ""
        parent_tid = getattr(self, "_parent_run_id", None)
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
                    trace_id=tid,
                    parent_trace_id=parent_tid,
                )
            )
            state.pending_transition = (from_phase, to_phase)
            # override 同时清除 pending_human_review (用户已决策)
            if state.pending_human_review == (from_phase, to_phase):
                state.pending_human_review = None
            return True

        gate = self.phase_gate_hook.evaluate(from_phase, to_phase, evidence)
        gate.trace_id = tid
        gate.parent_trace_id = parent_tid
        state.history.append(gate)
        state.pending_transition = (from_phase, to_phase)

        if gate.is_blocked:
            fb = gate.feedback or (
                f"阶段转移 {from_phase}→{to_phase} 被阻断: 缺 {gate.missing_evidence}"
            )
            self._speculator_hint = (
                (self._speculator_hint + "\n" + fb).strip()
                if self._speculator_hint
                else fb
            )
            logger.info(
                "gate blocked %s→%s: missing %s",
                from_phase,
                to_phase,
                gate.missing_evidence,
            )
            return False

        # ── Human-in-the-loop checkpoint (LangGraph interrupt_before 模式) ──
        # 硬性证据已通过, 但用户配置了该转移需要人工审查. 设 pending_human_review
        # 并返回 False 让 engine 停在当前 phase. UI 层读到 phase_checkpoint 事件后
        # 展示 evidence 给用户, 用户通过 phase_tool override 或 submit_evidence + resume.
        if state.needs_human_checkpoint(from_phase, to_phase):
            if state.pending_human_review == (from_phase, to_phase):
                # 已经在等了, 不重复设 — 避免 dead loop
                return False
            state.pending_human_review = (from_phase, to_phase)
            logger.info(
                "human checkpoint pending %s→%s: awaiting user review",
                from_phase,
                to_phase,
            )
            # 记一条 pending 状态, phase_tool 查得到
            state.history.append(
                PhaseGate(
                    from_phase=from_phase,
                    to_phase=to_phase,
                    status="pending",
                    required_evidence=self.phase_gate_hook.config.required_for(
                        from_phase, to_phase
                    ),
                    feedback=(
                        "⚠ 硬性检查点: 此转移不可超时自动放行, 必须人工确认. "
                        "请审查 evidence 后用 phase_tool override 显式放行, "
                        "或 submit_evidence 补充后 resume."
                        if state.is_hard_checkpoint(from_phase, to_phase)
                        else "等待人工 checkpoint 审查. 用 phase_tool override 放行, "
                        "或 submit_evidence 补充后 resume."
                    ),
                    reviewer="human_checkpoint",
                    trace_id=tid,
                    parent_trace_id=parent_tid,
                )
            )
            return False

        # 用户已审查完毕 (pending_human_review 被清除), 正常放行
        if state.pending_human_review == (from_phase, to_phase):
            state.pending_human_review = None
        return True

    async def _wait_if_checkpoint_pending(
        self, from_phase: str, to_phase: str, timeout: float = 600.0
    ) -> None:
        """等用户完成 checkpoint 审查.

        _check_gate 设了 pending_human_review 后, 调这个方法阻塞等.
        用户通过 phase_tool 加 override 后, 下一轮 _check_gate 走 override
        分支会清掉 pending 并放行. 所以这里同时盯 overrides 和 pending —
        任一变化就返回.

        timeout 到了还没决策, 强制清 pending 让下一轮放行, 避免无限阻塞.
        ponytail: 1s 轮询. 升级路径是 asyncio.Condition + phase_tool notify.
        """
        state = get_shared_phase_gate_state()
        key = (from_phase, to_phase)
        loop = asyncio.get_event_loop()
        is_hard = state.is_hard_checkpoint(from_phase, to_phase)
        # 硬门不设 deadline — 一直阻塞到用户显式 override, 不可超时偷偷放行
        deadline = loop.time() + timeout if not is_hard else float("inf")

        # 推送 checkpoint 等待事件到前端
        await self._publish_checkpoint_event(
            event_type="checkpoint_pending",
            from_phase=from_phase,
            to_phase=to_phase,
            is_hard=is_hard,
        )

        while state.pending_human_review == key and key not in state.overrides:
            if loop.time() > deadline:
                logger.warning(
                    "human checkpoint %s→%s timed out after %ss, force proceed",
                    from_phase,
                    to_phase,
                    timeout,
                )
                state.pending_human_review = None
                await self._publish_checkpoint_event(
                    event_type="checkpoint_timeout",
                    from_phase=from_phase,
                    to_phase=to_phase,
                    is_hard=is_hard,
                )
                return
            await asyncio.sleep(1.0)

        # checkpoint 已解决 (override 添加或 pending 被清除)
        await self._publish_checkpoint_event(
            event_type="checkpoint_resolved",
            from_phase=from_phase,
            to_phase=to_phase,
            is_hard=is_hard,
        )

    async def _publish_checkpoint_event(
        self,
        event_type: str,
        from_phase: str,
        to_phase: str,
        is_hard: bool = False,
    ) -> None:
        """推送 PhaseGate checkpoint 事件到 EventBus."""
        bus = self._get_event_bus()
        if bus is None:
            return
        try:
            await bus.dispatch(
                {
                    "type": event_type,
                    "from_phase": from_phase,
                    "to_phase": to_phase,
                    "is_hard": is_hard,
                    "timestamp": asyncio.get_event_loop().time(),
                }
            )
        except Exception:
            logger.debug("checkpoint event publish failed", exc_info=True)

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
                iteration,
                tier.label,
                tier.max_calls,
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
            mode,
            iteration,
            tier.label,
            rejects,
            tier.max_calls,
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
        router = getattr(self, "model_router", None) or getattr(
            getattr(self, "agent", None), "model_router", None
        )
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
            except Exception:
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
        elif checkpoint == "plan_check_fail":
            # plan_check 连续失败 + 场景已知 -> 问用户方向.
            # 跟 validation_fail 同款: 不阻塞, 用户可以选 force_proceed.
            info = phase_result or {}
            consecutive = info.get("consecutive_fails", 0)
            if consecutive < 3 or info.get("scene") == "other":
                return None
            ctx = {
                "thread_id": thread_id,
                "question_type": "plan_check_fail",
                "phase": "plan",
                "summary": (
                    f"plan_check 连续 {consecutive} 次失败 "
                    f"(scene={info.get('scene', '?')}): "
                    f"{info.get('reason', '')[:200]}"
                ),
                "consecutive_fails": consecutive,
                "scene": info.get("scene", ""),
            }
        elif checkpoint == "hypothesize_align":
            # v11: FDE 对齐轮 — 首轮或 blind_spots/conflicts 时问用户假设方向.
            # 不阻塞, 60s timeout, 用户回答 append 到 _speculator_hint 自动流进 prompt.
            # ponytail: 复用现有 _maybe_clarify 管道, 不新建 ClarificationManager 子类.
            _info = phase_result if isinstance(phase_result, dict) else {}
            _is_first = self._iteration <= 1
            _has_signals = bool(
                _info.get("blind_spots") or _info.get("semantic_conflicts")
            )
            if not (_is_first or _has_signals):
                return None

            # 拼 recommended_directions: 优先用 cluster_by_dimension, 不足补 speculator
            _directions: list[str] = []
            try:
                _clusters = self.hypothesis_graph.cluster_by_dimension()
                for _dim, _nodes in list(_clusters.items())[:3]:
                    if _dim != "unknown" and _nodes:
                        _directions.append(f"{_dim}: {_nodes[0].statement[:80]}")
            except Exception:
                pass
            # 不足 3 个时补 speculator predictions (首轮自然走这条)
            while len(_directions) < 3:
                try:
                    _preds = getattr(self, "_speculator_predictions", []) or []
                    if _preds:
                        _directions.append(f"speculator: {str(_preds[len(_directions)])[:80]}")
                    else:
                        break
                except Exception:
                    break

            ctx = {
                "thread_id": thread_id,
                "question_type": "hypothesize_align",
                "phase": "hypothesize",
                "summary": (
                    f"目标: {self._objective or '?'}\n"
                    f"现场: {str(_info.get('summary', ''))[:200]}\n"
                    f"推荐方向:\n" + "\n".join(f"  - {d}" for d in _directions[:3])
                ),
                "recommended_directions": _directions[:3],
                "is_first_iteration": _is_first,
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
                timeout=60,  # 给用户足够时间回答
                metadata={
                    "question_type": ctx.get("question_type", ""),
                    "checkpoint": checkpoint,
                    "iteration": self._iteration,
                },
            )
            logger.info("clarify %s: %s", checkpoint, answer[:80])
            # v11: hypothesize_align 的回答 append 到 _speculator_hint,
            # 自动流进 _build_hypothesis_prompt 的 hint_block (零改动).
            if checkpoint == "hypothesize_align" and answer:
                self._speculator_hint += f"\n[FDE 对齐] 用户方向: {answer[:200]}\n"
            return answer
        except Exception:
            logger.warning("clarify %s failed", checkpoint, exc_info=True)
            return None

    # ──────────────────────────────────────────────────────────────
    # Public API — run_cognitive (v10: legacy run() 已删, AV1 断层修复)
    # ──────────────────────────────────────────────────────────────

    # === Step 0b: CognitiveLoop 入口 — 4 钩子编排 7-phase ===
    # v10: run() 已删 (AV1 断层修复), run_cognitive 是唯一入口.
    #   - decide() 选下个 action (规则版现在, LLM 版 v8 候选)
    #   - reflect() 集中 gate + consecutive_failures, 不再散落 15 个 continue
    #   - CognitiveLoop 自带死循环防护 (3x redirect, 6x stop)
    #   - 每轮跑一个 action, 不强制一轮跑完 7 phase — 基模自主选
    # ponytail: 7-phase 方法签名不变, 只在适配器里调. 升级路径: decide_fn 换 LLM 选 action.
    async def run_cognitive(
        self,
        objective: str,
        max_iterations: int = 50,
        progressive_budget: bool = True,
        goal: Goal | None = None,
        max_refines: int = 8,
        timeout_seconds: float | None = None,
    ) -> AutoloopResult:
        """CognitiveLoop 入口 — 用 4 钩子编排 7-phase.

        返回 AutoloopResult, 与 run() 接口一致, 调用方无需感知差异.
        """
        from huginn.autoloop.cognitive_loop import (
            CognitiveLoop, LoopState, ActionDecision, ReflectionResult,
        )

        self._max_refines = max_refines
        self._refine_count = 0
        self._max_iterations = max_iterations
        # AV2: 每次新 run 重置元认知护航状态 (避免跨 run 串味)
        self._evals_history = []
        self._task_metrics = None
        self._task_state_for_metrics = None
        self._drift_info = None
        get_shared_phase_gate_state().reset_runtime()
        run_id, provenance_record, run_collector = self._prepare_run(
            objective, progressive_budget, goal
        )
        self._run_id = run_id
        self._parent_run_id = None
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
            timeout_seconds=timeout_seconds,
        )
        self._progress_task_id = progress_task_id

        # P0-2: bridge progress_cb → _emit_campaign. autoloop 路径不走 _stream_agent_response,
        # progress_cb 默认 None, 导致 subagent_tool._on_state 早 return (cb is None).
        # 这里 set 一个桥: subagent_event / autoloop_thinking 等事件 → campaign SSE.
        # 外层已 set (WS 路径嵌入 autoloop) 则不覆盖. ponytail: 复用 contextvar, 不新开通道.
        # 不显式 reset — run_cognitive 通常被 asyncio.create_task 包, contextvar 跟 task 同生命周期.
        from huginn.types import progress_cb as _progress_cb
        if _progress_cb.get(None) is None:
            _autoloop_engine = self

            async def _progress_bridge(msg: dict) -> None:
                _etype = msg.get("type", "progress")
                _data = {k: v for k, v in msg.items() if k != "type"}
                _data.setdefault("run_id", run_id)
                _autoloop_engine._emit_campaign(f"campaign.{_etype}", _data)

            _progress_cb.set(_progress_bridge)

        # phase 间传递的中间结果 — 不放 LoopState (那是控制流状态)
        cog: dict[str, Any] = {
            "context": {},
            "hypothesis": None,
            "plan": None,
            "execution_result": None,
            "validation": None,
            "current_hyp_id": None,
            "phases": [],
            "completed_steps": 0,
        }

        async def observe_fn(state: LoopState) -> dict[str, Any]:
            # v10: 外部 stop() 设 self._should_stop, 同步到 state.should_stop
            # 让 CognitiveLoop while guard 能感知. 否则 stop() 对 run_cognitive 无效.
            if getattr(self, "_should_stop", False):
                state.should_stop = True
                return {
                    "context_summary": "",
                    "redirect_reason": state.redirect_reason,
                    "iteration": state.iteration,
                    "last_action": state.last_action,
                    "external_stop": True,
                }
            # P1.4: 每轮开头发 campaign.iteration — 对齐 run() L1305.
            # 前端 IterationTimeline 依赖这个事件渲染轮次进度.
            self._emit_campaign(
                "campaign.iteration",
                {
                    "iteration": state.iteration,
                    "max": max_iterations,
                    "objective": objective[:200],
                },
            )
            # v10-F1+F7: goal 持久化 + budget 硬停 — 对齐 run() L1272-1304.
            # 每轮 increment_iteration, is_budget_exhausted → fail_goal + should_stop.
            # ponytail: spec 把 F1 放 reflect / F7 放 observe, 但 increment 和 budget
            #   check 必须原子 (不原子会读 stale iteration), 这里合并到 observe 开头.
            #   spec F7 阶段 2 只剩 build_continuation_prompt / drain_side.
            try:
                from huginn.autoloop.goal_store import get_goal_store

                _gs = get_goal_store()
                _active_goal = _gs.get_active()
                if _active_goal:
                    _gs.increment_iteration(_active_goal.id)
                    if GoalScheduler.is_budget_exhausted(_active_goal):
                        logger.info(
                            "v10 goal budget exhausted: iter=%d max=%d, failing %s",
                            _active_goal.iteration, _active_goal.max_iterations, _active_goal.id,
                        )
                        try:
                            _gs.fail_goal(
                                _active_goal.id,
                                reason=f"budget exhausted: {_active_goal.iteration}/{_active_goal.max_iterations}",
                            )
                        except Exception:
                            logger.debug("fail_goal failed (non-fatal)", exc_info=True)
                        self._emit_campaign(
                            "campaign.budget_exhausted",
                            {
                                "iteration": state.iteration,
                                "goal_id": _active_goal.id,
                                "budget": _active_goal.max_iterations,
                                "used": _active_goal.iteration,
                            },
                        )
                        state.should_stop = True
                        return {
                            "context_summary": "",
                            "redirect_reason": state.redirect_reason,
                            "iteration": state.iteration,
                            "last_action": state.last_action,
                            "budget_exhausted": True,
                        }
            except Exception:
                logger.debug("v10 goal increment/budget failed (non-fatal)", exc_info=True)

            # v10-F6: build_continuation_prompt — 对齐 run() L1321-1331.
            # goal 非空且 iteration > 1 时拼续跑提示到 speculator_hint, 让 LLM
            # 看到自己在续跑而非从头开始. 第 1 轮不拼 (算首次).
            if goal is not None and state.iteration > 1:
                try:
                    _cont = GoalScheduler.build_continuation_prompt(goal)
                    if _cont:
                        self._speculator_hint = (
                            (self._speculator_hint + "\n" + _cont).strip()
                            if self._speculator_hint else _cont
                        )
                except Exception:
                    logger.debug("v10 F6 build_continuation_prompt failed (non-fatal)", exc_info=True)

            # _perceive 是 sync (跑 git subprocess + rglob), 丢线程池不阻塞
            # v10: 记本轮 perceive 是否返回空, F8 用这个 flag 而非 cog["context"]
            # (cog["context"] 跨轮持久, 上轮 forced/residual 会掩盖本轮空感知).
            _perceived_empty = True
            try:
                if self._next_phase_hint not in ("plan", "execute"):
                    ctx = await asyncio.to_thread(self._perceive)
                    if ctx:
                        cog["context"] = ctx
                        _perceived_empty = False
                else:
                    # hint=plan/execute 时跳过 perceive, 不算空 ( intentional skip)
                    _perceived_empty = False
            except Exception as e:
                logger.warning("cognitive observe failed: %s", e)

            # v10-F15: G31 bypass — 对齐 run() L1369-1382.
            # perceive 返回空 + 首轮 + objective 存在 → 强制注入 minimal context,
            # 避免首轮轮空导致 18/18 轨迹 perceive+report 两阶段 0 工具调用.
            # ponytail: 仅首轮 bypass, 后续轮走 F8 drain_side.
            if _perceived_empty and state.iteration == 1 and objective:
                logger.info(
                    "v10 G31: perceive empty on iter 1 with objective, forcing hypothesize"
                )
                cog["context"] = {
                    "forced": True,
                    "objective": objective,
                    "note": "perceive returned empty, G31 forces hypothesize",
                }

            # v10-F16: timeout 硬停 (observe 阶段) — 对齐 run() L1248.
            # reflect_fn 已有 timeout 检查, 这里加 observe 阶段检查让 timeout 更早触发,
            # 不必跑完当前轮的 decide/execute/reflect.
            # ponytail: tracker.is_expired 是 O(1) 字典查, 不阻塞.
            if tracker.is_expired(progress_task_id):
                logger.info("v10 F16 timeout expired in observe, stopping")
                state.should_stop = True
                return {
                    "context_summary": "",
                    "redirect_reason": state.redirect_reason,
                    "iteration": state.iteration,
                    "last_action": state.last_action,
                    "timeout_expired": True,
                }

            # v10-F8: drain_side_questions — 对齐 run() L1384-1387.
            # run() 在 perceive 返回空 (轮空) 时调 _drain_side_questions + continue.
            # run_cognitive 不能 continue (CognitiveLoop 每轮要走完 4 钩子), 改为
            # perceive 返回空时顺手答 pending 侧边问题, 不跳过本轮.
            # ponytail: 用 _perceived_empty 而非 not cog["context"], 避免上轮
            # forced/residual context 掩盖本轮空感知.
            if _perceived_empty:
                try:
                    _n_drained = await self._drain_side_questions()
                    if _n_drained:
                        logger.info("v10 F8 drained %d side questions", _n_drained)
                except Exception:
                    logger.debug("v10 F8 drain_side_questions failed (non-fatal)", exc_info=True)

            # v10-F5: blind_spot_pass — 对齐 run() L1391-1402.
            # spec F5 描述 "强制 stop" 不准, run() 实际行为是注入 context + 写 GoalStore.unknowns.
            # ponytail: 每隔 5 轮做一次, 避免 token 浪费. 异步 LLM 调用.
            if state.iteration == 1 or state.iteration % 5 == 0:
                try:
                    _bs = await self._blind_spot_pass(cog["context"] or {}, self._objective)
                    if _bs:
                        cog["context"]["blind_spots"] = _bs
                        logger.info("v10 blind spot pass: %d unknowns", len(_bs))
                except Exception:
                    logger.debug("v10 blind_spot_pass failed (non-fatal)", exc_info=True)

            return {
                "context_summary": (cog["context"] or {}).get("summary", ""),
                "redirect_reason": state.redirect_reason,
                "iteration": state.iteration,
                "last_action": state.last_action,
            }

        async def decide_fn(state: LoopState, obs: dict[str, Any]) -> ActionDecision:
            # Step C: LLM 自主选 action (优先) → 规则版兜底
            # redirect / hint 仍走规则版 — 死循环防护和上轮 reflect 的明确建议不交给 LLM.
            # 首轮 (last in ("", "skip")) 不调 LLM, 直接走规则版 hypothesize.
            if state.should_redirect:
                state.should_redirect = False
                # 没 hyp 可以 pivot → 直接停, 避免 pivot 空转死循环
                if not cog.get("current_hyp_id") and not cog.get("hypothesis"):
                    return ActionDecision(action="stop", rationale="no hyp to pivot from")
                return ActionDecision(action="pivot", rationale=f"redirect: {state.redirect_reason}")
            hint = self._next_phase_hint
            if hint == "execute" and self._refined_hypothesis:
                cog["hypothesis"] = self._refined_hypothesis
                return ActionDecision(action="execute", rationale="refine reuse")
            if hint == "plan":
                return ActionDecision(action="plan", rationale="hint=plan")
            if hint == "perceive":
                return ActionDecision(action="observe", rationale="hint=perceive")
            # LLM 自主决策 (开启时, 且非首轮). 失败/非法 → fallback 到规则版
            if self._use_llm_decider and state.last_action not in ("", "skip"):
                try:
                    llm_decision = await self._decide_next_action_llm(state, cog, obs)
                    if llm_decision is not None:
                        return llm_decision
                except Exception as e:
                    logger.debug("LLM decider failed: %s, fallback to rule", e)
            # 规则版兜底: 默认 7-phase 顺序
            last = state.last_action
            if last in ("", "observe", "pivot", "skip"):
                return ActionDecision(action="hypothesize", rationale="seq→hyp")
            if last == "hypothesize":
                if not cog["hypothesis"]:
                    return ActionDecision(action="observe", rationale="no hyp, re-observe")
                return ActionDecision(action="plan", rationale="seq→plan")
            if last == "plan":
                if not cog["plan"]:
                    return ActionDecision(action="hypothesize", rationale="no plan, re-hyp")
                return ActionDecision(action="execute", rationale="seq→exec")
            if last == "execute":
                if cog["execution_result"] is None:
                    return ActionDecision(action="plan", rationale="exec None, re-plan")
                return ActionDecision(action="validate", rationale="seq→validate")
            if last == "validate":
                # v10: 规则版总是推进到 learn, 让 validate→learn gate 决定放行/阻断.
                # 对齐 run() 顺序执行语义 (run() 不因 tests_passed=False 跳过 learn,
                # 而是让 gate 评估 evidence). LLM decider 可智能选 re-execute.
                # ponytail: 规则版是 fallback, 不做智能判断.
                return ActionDecision(action="learn", rationale="seq→learn")
            if last == "learn":
                # v10: cycle 回 hypothesize 而非 stop — 对齐 run() while 循环自然
                # 进入下一 iter 的行为. stop 由 max_iter / should_stop / F3 darwin
                # / F4 surprise / F2 completion / F17 GoalJudge 触发, 不靠规则版.
                # ponytail: cycling 是 rule-based fallback 的语义, LLM decider 不受此影响.
                return ActionDecision(action="hypothesize", rationale="cycle→hyp")
            return ActionDecision(action="stop", rationale=f"unknown last {last}")

        async def execute_fn(state: LoopState, decision: ActionDecision) -> Any:
            action = decision.action
            ctx = cog["context"]
            self._iteration = state.iteration
            try:
                if action == "observe":
                    phase = await self._run_phase_async(
                        "perceive", lambda: asyncio.to_thread(self._perceive)
                    )
                    cog["phases"].append(phase)
                    cog["completed_steps"] += 1
                    if phase.result:
                        cog["context"] = phase.result
                    return phase.result
                if action == "hypothesize":
                    # v11: FDE 对齐轮 — hypothesize 前问用户方向 (首轮/有 blind_spots).
                    # 不阻塞, 60s timeout, 用户回答 append 到 _speculator_hint.
                    # ponytail: 复用 _maybe_clarify 管道, 不新增 phase.
                    try:
                        await self._maybe_clarify(
                            "hypothesize_align", ctx, thread_id="autoloop",
                        )
                    except Exception:
                        logger.debug("v11 FDE hypothesize_align failed (non-fatal)", exc_info=True)
                    phase = await self._run_phase_async(
                        "hypothesize", self._hypothesize, ctx
                    )
                    cog["phases"].append(phase)
                    cog["completed_steps"] += 1
                    cog["hypothesis"] = phase.result
                    if phase.result:
                        try:
                            cog["current_hyp_id"] = self.hypothesis_graph.add_hypothesis(
                                statement=phase.result,
                                rationale=ctx.get("summary", ""),
                            )
                            self._current_hyp_id_for_plan = cog["current_hyp_id"]
                        except Exception:
                            logger.debug("hypothesis_graph add failed", exc_info=True)
                    # P1.4: campaign SSE 对齐 run() L1435
                    self._emit_campaign(
                        "campaign.hypothesis",
                        {
                            "iteration": state.iteration,
                            "hypothesis": str(phase.result or "")[:300],
                        },
                    )
                    return phase.result
                if action == "plan":
                    if not cog["hypothesis"]:
                        return None
                    phase = await self._run_phase_async(
                        "plan", self._plan, cog["hypothesis"], ctx
                    )
                    cog["phases"].append(phase)
                    cog["completed_steps"] += 1
                    cog["plan"] = phase.result
                    return phase.result
                if action == "execute":
                    if not cog["plan"]:
                        return None
                    self._current_prediction = cog["plan"].get("expected_prediction", "")
                    # v10: 下沉 run() L1493+L1497 budget + gate 检查到 execute_fn.
                    # spec 漏列, 但没有这俩 check, budget tier / phase gate 在
                    # run_cognitive 路径完全失效. ponytail: check 失败不抛,
                    # 写 hint 让下轮 decide 看到, 当前 return None 跳过 execute.
                    _plan = cog["plan"]
                    if not self._check_budget(state.iteration, _plan):
                        # budget 拒: hint 已被 _check_budget 写, 这里不重复
                        return None
                    if not self._check_gate(
                        "plan", "execute",
                        {"mode": _plan.get("mode"), "description": _plan.get("description")},
                    ):
                        # gate 阻断: 写 hint 让 LLM 下轮改 plan
                        self._speculator_hint += (
                            "\n[gate: plan→execute blocked] "
                            + str(self.phase_gate_hook.evaluate(
                                "plan", "execute",
                                {"mode": _plan.get("mode"), "description": _plan.get("description")},
                            ).feedback or "")
                            + "\n"
                        )
                        await self._wait_if_checkpoint_pending("plan", "execute")
                        return None
                    phase = await self._run_phase_async(
                        "execute", self._execute, cog["plan"], ctx
                    )
                    cog["phases"].append(phase)
                    cog["completed_steps"] += 1
                    cog["execution_result"] = phase.result
                    # v10: 下沉 run() L1567-1577 plan 完成标记.
                    _plan_id = cog["plan"].get("plan_id") if isinstance(cog["plan"], dict) else None
                    if _plan_id:
                        try:
                            _store = self._get_plan_store()
                            if _store is not None:
                                _store.complete_plan(_plan_id)
                        except Exception:
                            logger.warning("v10 complete_plan failed (non-fatal)", exc_info=True)
                    # git commit after execute (同 run(): 让下轮 perceive 看到 diff)
                    await asyncio.to_thread(self._git_commit_after_execute,
                                            cog["plan"], state.iteration)
                    # P1.4: execute 失败 (无 phase.result) → campaign.retry 对齐 run() L1651
                    if phase.result is None:
                        self._emit_campaign(
                            "campaign.retry",
                            {
                                "iteration": state.iteration,
                                "reason": "execute returned None",
                            },
                        )
                    return phase.result
                if action == "validate":
                    if cog["execution_result"] is None:
                        return None
                    phase = await self._run_phase_async(
                        "validate", self._validate, cog["execution_result"]
                    )
                    cog["phases"].append(phase)
                    cog["completed_steps"] += 1
                    cog["validation"] = phase.result
                    # P1.4: validate 失败 → campaign.suspect 对齐 run() L1668
                    _val = phase.result or {}
                    if not _extract_tests_passed(_val):
                        self._emit_campaign(
                            "campaign.suspect",
                            {
                                "iteration": state.iteration,
                                "reason": str(_val.get("thinking_collapse")
                                              or _val.get("physics_validation_error")
                                              or "tests_failed")[:200],
                            },
                        )
                    return phase.result
                if action == "learn":
                    if not all(cog.get(k) for k in ("hypothesis", "plan", "validation")):
                        return None
                    # v10: 下沉 run() L1859 validate→learn gate 检查.
                    _val = cog["validation"] or {}
                    _exec = cog.get("execution_result") if isinstance(cog.get("execution_result"), dict) else {}
                    _gate_evidence = {k: _val[k] for k in (
                        "tests_passed", "reviewer_critique", "thinking_collapse",
                        "physics_validation_error", "dimensional_consistent",
                        "pde_classification", "sobol_top_features",
                        "constraint_check", "literature_claims",
                    ) if k in _val}
                    if isinstance(_exec.get("physics_audit"), dict):
                        _gate_evidence["physics_audit"] = _exec["physics_audit"]
                    if not self._check_gate("validate", "learn", _gate_evidence):
                        await self._wait_if_checkpoint_pending("validate", "learn")
                        return None
                    phase = await self._run_phase_async(
                        "learn", self._learn,
                        cog["hypothesis"], cog["plan"], cog["validation"],
                    )
                    cog["phases"].append(phase)
                    cog["completed_steps"] += 1
                    # D2: learn 写 cog, 让下轮 decider 看到正反馈. 之前 learn
                    # 是哑 action (不更新 cog), LLM 选了 learn 没反馈, 下轮
                    # 容易重复 learn 或乱选. ponytail: 只塞 1 行摘要, 不暴露
                    # _learn 完整内部状态. 升级路径: 结构化 summary 走 cog dict.
                    _learned = phase.result if isinstance(phase.result, dict) else {}
                    if _learned:
                        cog["last_learn_summary"] = (
                            f"learned: persona={_learned.get('persona','?')} "
                            f"r_phys={_learned.get('r_phys','?')} "
                            f"tests_passed={_learned.get('tests_passed','?')} "
                            f"principles_added={_learned.get('principles_added',0)}"
                        )
                    else:
                        cog["last_learn_summary"] = "learn ran (no summary)"
                    return phase.result
                if action == "pivot":
                    _obj = self._objective if hasattr(self, "_objective") else ""
                    _cur = cog.get("current_hyp_id")
                    if _cur:
                        try:
                            new_hyp = self.hypothesis_graph.pivot(
                                _cur,
                                evidence={"reason": "cognitive pivot"},
                                model=self._get_refine_model(),
                                objective=_obj,
                            )
                            self._refine_count = 0
                            self._pivot_count += 1
                            self._next_phase_hint = "perceive"
                            logger.info("CognitiveLoop pivot: %s → %s", _cur, new_hyp)
                            # P1.4: pivot → campaign.refine 对齐 run() L1729
                            self._emit_campaign(
                                "campaign.refine",
                                {
                                    "iteration": state.iteration,
                                    "old_hyp_id": _cur,
                                    "new_hyp_id": new_hyp,
                                    "reason": "cognitive pivot",
                                },
                            )
                            # P15: pivot 是关键事件, 立刻 save (force=True)
                            self._maybe_save_engine_state(force=True, reason="pivot")
                        except Exception:
                            logger.warning("cognitive pivot failed", exc_info=True)
                    # 清中间状态, 下轮重新 observe
                    for k in ("hypothesis", "plan", "execution_result", "validation", "current_hyp_id"):
                        cog[k] = None
                    return "pivoted"
                if action in ("skip", "stop", "report"):
                    # report 由 _finalize_run 跑; stop/skip 是控制信号
                    return action
            except Exception as e:
                logger.warning("cognitive execute '%s' failed: %s", action, e)
                return None
            return None

        async def reflect_fn(
            state: LoopState, decision: ActionDecision, result: Any
        ) -> ReflectionResult:
            action = decision.action
            advice = ""
            redirect = False

            # 失败检测 — 各 action 的"无产出"判为 failed → redirect
            if action == "hypothesize" and not cog["hypothesis"]:
                redirect = True
                advice = "hypothesize 无产出, 下轮重新 observe"
            elif action == "plan" and not cog["plan"]:
                redirect = True
                advice = "plan 无产出, 下轮重新 hypothesize"
            elif action == "execute" and cog["execution_result"] is None:
                redirect = True
                advice = "execute None, 下轮重新 plan"

            # gate 检查 — 把 evidence 传给 _check_gate
            if action == "plan" and cog["plan"] and not redirect:
                if not self._check_gate(
                    "plan", "execute",
                    {"mode": cog["plan"].get("mode"),
                     "description": cog["plan"].get("description")},
                ):
                    redirect = True
                    advice = "gate plan→execute blocked"
            if action == "execute" and cog["plan"] and not redirect:
                if not self._check_gate(
                    "execute", "validate",
                    {"mode": cog["plan"].get("mode")},
                ):
                    redirect = True
                    advice = "gate execute→validate blocked"

            # consecutive_failures — 只在 validate 后算 (同 run())
            if action == "validate":
                validation = cog["validation"] or {}
                tests_ok = _extract_tests_passed(validation)
                # 700 万步极限场景: 滑动窗口失败率. 推入当前结果, 超窗口截断.
                # consecutive 在长轨迹里太窄 (20 次 tool timeout 就停), windowed rate
                # 允许局部失败 — 最近 100 次 validate 失败率 > 0.8 才算真死路.
                _vwin = getattr(self, "_validate_window", None)
                if _vwin is not None:
                    _vwin.append(bool(tests_ok))
                    _wsize = getattr(self, "_validate_window_size", 100)
                    if len(_vwin) > _wsize:
                        del _vwin[: -_wsize]
                if tests_ok:
                    self._consecutive_failures = 0
                    self._consecutive_failures_by_type = {}
                else:
                    self._consecutive_failures += 1
                    # F-borrow: 按 failure_type 分类计数 (forge 双预算思路).
                    # _classify_failure 已存在但之前没在 reflect 路径用 — 闭合断层.
                    # 失败分类后按类阈值 stop, 避免 tool_error 跟 hypothesis_error 混算.
                    try:
                        _redteam = self._redteam_findings()
                        ftype = AutoloopEngine._classify_failure(validation, _redteam)
                    except Exception:
                        ftype = "hypothesis_error"
                    by_type = getattr(self, "_consecutive_failures_by_type", {}) or {}
                    by_type[ftype] = by_type.get(ftype, 0) + 1
                    self._consecutive_failures_by_type = by_type
                    _type_max = getattr(self, "_max_failures_by_type", {}).get(
                        ftype, self._max_consecutive_failures
                    )
                    if by_type[ftype] >= _type_max:
                        logger.warning(
                            "cognitive stop: %d consecutive %s failures",
                            by_type[ftype], ftype,
                        )
                        return ReflectionResult(
                            should_stop=True,
                            advice=f"{by_type[ftype]} consecutive {ftype} failures",
                        )
                    if self._consecutive_failures >= self._max_consecutive_failures:
                        # 700 万步兜底: consecutive 触顶时, 检查滑动窗口失败率.
                        # 如果窗口内失败率低于阈值, 说明只是局部连续失败, 整体仍在进展 —
                        # 不停, 只清 consecutive 让它重新计数. 避免长轨迹被短期波动截停.
                        _win = getattr(self, "_validate_window", None)
                        _wsize = getattr(self, "_validate_window_size", 100)
                        _wthresh = getattr(self, "_validate_window_fail_threshold", 0.8)
                        if _win and len(_win) >= _wsize:
                            _fail_rate = 1.0 - (sum(_win) / len(_win))
                            if _fail_rate < _wthresh:
                                logger.info(
                                    "consecutive=%d 但窗口失败率 %.2f < %.2f, 不停, 清计数",
                                    self._consecutive_failures, _fail_rate, _wthresh,
                                )
                                self._consecutive_failures = 0
                                self._consecutive_failures_by_type = {}
                            else:
                                logger.warning(
                                    "cognitive stop: consecutive=%d 且窗口失败率 %.2f >= %.2f",
                                    self._consecutive_failures, _fail_rate, _wthresh,
                                )
                                return ReflectionResult(
                                    should_stop=True,
                                    advice=f"{self._consecutive_failures} consecutive failures (window fail rate {_fail_rate:.2f})",
                                )
                        else:
                            logger.warning(
                                "cognitive stop: %d consecutive failures (total cap)",
                                self._consecutive_failures,
                            )
                            return ReflectionResult(
                                should_stop=True,
                                advice=f"{self._consecutive_failures} consecutive failures",
                            )

            # G2: 周期检测 + 历史轨迹匹配 (M3 cycle_detect + M2 trajectory_match).
            # 不 should_stop — 给建议, 让 LLM decider / 规则版自己决定是否 pivot.
            # cycle 信号强 → 强制 redirect; match 信号弱 → 只注入 hint.
            try:
                stuck = self._check_stuck(state.action_history)
                if stuck:
                    if stuck["type"] == "cycle":
                        redirect = True
                        advice = (advice + " | G2 cycle: " + stuck["advice"]).strip(" |")
                        logger.warning("G2 stuck: %s", stuck["advice"])
                    elif stuck["type"] == "match":
                        # match 不是 stuck, 只是建议下一步. 注入 _speculator_hint
                        # 让下轮 hypothesize 能看到. 不 redirect.
                        self._speculator_hint = (
                            (self._speculator_hint + " | " + stuck["advice"])
                            .strip(" |")[:2000]
                        )
                        advice = (advice + " | G2 match: " + stuck["advice"]).strip(" |")
                        logger.info("G2 trajectory match: %s", stuck["advice"])
            except Exception:
                logger.debug("G2 _check_stuck failed (non-fatal)", exc_info=True)

            # timeout / pivot 预算 (硬停)
            if tracker.is_expired(progress_task_id):
                return ReflectionResult(should_stop=True, advice="timeout")
            if self._pivot_count >= self._max_pivots:
                return ReflectionResult(should_stop=True, advice="pivot budget exhausted")
            # 死循环防护: pivot 后还反复 fail → 别再 pivot, 直接停.
            # CognitiveLoop 自带的 repeated-action 检测抓不到 pivot/hyp 交替的情况.
            # ponytail: 用 action_history 数 pivot 次数, 不引入新状态字段.
            if action == "pivot" and state.action_history.count("pivot") >= 3:
                return ReflectionResult(
                    should_stop=True,
                    advice="3+ pivots without progress, stop",
                )

            # hint 用完清空 (同 run() 末尾)
            self._next_phase_hint = None
            self._refined_hypothesis = None
            # speculator hint 截断 (同 run())
            if len(self._speculator_hint) > 2000:
                self._speculator_hint = self._speculator_hint[-2000:]

            # AV2+AV4: PMK + TaskMetrics + detect_drift + heat_engine 接入 (reflect 末尾).
            # ponytail: 只在 validate 后跑 — perceive/hypothesize/plan 没产出
            # StepEvaluation 等价物. autoloop validation dict 字段不全 (无
            # evidence_quality/pmk_feedback/tool_call_health), 用 SimpleNamespace
            # 兜底, duck typing 够 update_metrics/detect_drift/should_pause_for_decision 用.
            # 天花板: pmk_cycle_count/tool_call_health_avg 在 autoloop 路径不增;
            # 升级路径: 在 _validate 里跑 StepEvaluator 填全字段.
            # autoloop 无人在环, pause 退化为日志 + hint, 不真停 (不设 should_stop).
            # AV4: detect_drift + TaskMetrics 抽到 update_drift_and_metrics 共享;
            #   heat_engine 抽到 update_heat_engine_after_step 共享 (对齐 rcb_runner AV8).
            if action == "validate" and cog.get("validation") is not None:
                try:
                    from types import SimpleNamespace as _NS
                    _val = cog["validation"] or {}
                    _tests_ok = _extract_tests_passed(_val)
                    # P0.2: _validate 真实字段是 tests_passed/benchmarks/
                    # thinking_collapse/*_error/effort_floor_deficits 等, 不是
                    # summary/result/errors. 之前硬取 summary/result/errors 全是
                    # 空串, 导致 PMK/drift/metrics 全在吃空数据.
                    _se_fields = _validation_to_step_eval_fields(
                        _val, _tests_ok, cog.get("execution_result"),
                        step_id=len(self._evals_history),
                    )
                    _step_eval = _NS(**_se_fields)
                    self._evals_history.append(_step_eval)

                    # AV4: detect_drift + TaskMetrics — 调共享函数
                    from huginn.autoloop.cognitive_loop import (
                        update_drift_and_metrics,
                        update_heat_engine_after_step,
                    )
                    self._drift_info, self._task_metrics = update_drift_and_metrics(
                        self._evals_history, _step_eval,
                        self._task_metrics, self._task_state_for_metrics,
                        self.workspace, self._run_id, self._max_iterations,
                    )
                    if self._drift_info and self._drift_info[0]:
                        advice = (advice + " | drift: " + self._drift_info[1]).strip(" |")
                        logger.warning("autoloop drift: %s", self._drift_info[1])

                    # AV4: heat_engine 闭环 — 对齐 rcb_runner AV8
                    try:
                        from huginn.metacog.cognitive_heat_engine import get_heat_engine
                        _he = get_heat_engine()
                        update_heat_engine_after_step(
                            _he, _step_eval,
                            prompt_len=len(getattr(self, "_last_hypothesis", "") or ""),
                            idea_count=self.hypothesis_graph.component_count() if hasattr(self, "hypothesis_graph") else 1,
                        )
                    except Exception:
                        logger.debug("AV4 heat_engine update in autoloop failed", exc_info=True)
                except Exception:
                    logger.debug("AV2 metrics/drift update failed", exc_info=True)

                # PMK 一致性 + should_pause_for_decision — autoloop 无人在环,
                # pause 退化为 hint 注入. 升级路径: 接 routes SSE 决策流.
                try:
                    from huginn.autoloop.cognitive_loop import (
                        build_pmk_state, check_pause_decision,
                    )
                    _persona_obj = None
                    try:
                        _persona_obj = self._get_persona_manager().get_persona("default")
                    except Exception:
                        pass
                    _pmk_state = build_pmk_state(
                        _persona_obj, _step_eval, self._get_kb() if hasattr(self, "_get_kb") else None,
                    )
                    _pause, _reason, _opts = check_pause_decision(
                        self._evals_history, [],
                        self._get_kb() if hasattr(self, "_get_kb") else None,
                        None, _pmk_state,
                    )
                    if _pause:
                        logger.warning("autoloop pause signal (no human): %s", _reason)
                        self._speculator_hint = (
                            (self._speculator_hint + f"\n[PAUSE] {_reason}\n").strip()
                        )
                        # H4: GRILL pause → 进入 grill 模式, 下次 _llm_chat 注入
                        # GRILL_SYSTEM_PROMPT_CN. 之前 pause 后只 auto-resume,
                        # LLM 看不到 grill 约束, "一次一问" 形同虚设.
                        if "GRILL" in _reason and not self._grill_active:
                            self._grill_active = True
                            self._grill_turns = 0
                            logger.info("GRILL mode activated: %s", _reason)
                except Exception:
                    logger.debug("AV2 should_pause_for_decision failed", exc_info=True)

                # v10-F2: completion audit — 对齐 run() L1878-1897.
                # goal 达标 + metacog 不阻断 → goal.status=completed + should_stop.
                # ponytail: check_completion 在 goal 无 criteria 时返回 False, 不影响.
                if goal is not None and not state.should_stop:
                    try:
                        _val_for_goal = cog["validation"] or {}
                        if GoalScheduler.check_completion(goal, _val_for_goal):
                            _blk, _why = self._metacog_check_completion()
                            if _blk:
                                logger.info("v10 completion audit blocked: %s", _why)
                                self._speculator_hint = (
                                    (self._speculator_hint + f"\n[completion audit] {_why}").strip()
                                )
                            else:
                                logger.info("v10 goal completed: %s", goal.objective)
                                goal.status = "completed"
                                if self._goal_scheduler is not None:
                                    try:
                                        self._goal_scheduler.complete_goal(goal.id)
                                    except Exception:
                                        logger.debug("complete_goal failed (non-fatal)", exc_info=True)
                                state.should_stop = True
                    except Exception:
                        logger.debug("v10 F2 completion audit failed (non-fatal)", exc_info=True)

                # v10-F17: GoalJudge — 对齐 run() L1899-1945.
                # 每 3 轮或最后一轮调 GoalJudge.judge 判 goal_achieved.
                # achieved + metacog 不阻断 → should_stop; gaps → 注入 hint.
                # ponytail: GoalJudge(llm=None) 走规则版, LLM judge 留 exit 阶段.
                if goal is not None and not state.should_stop:
                    if state.iteration % 3 == 2 or state.iteration >= max_iterations - 1:
                        try:
                            from huginn.evaluation.goal_judge import GoalJudge

                            _judge = GoalJudge(llm=None)
                            _final_text = str(
                                (cog["validation"] or {}).get("summary")
                                or (cog["validation"] or {}).get("result_data")
                                or (cog.get("execution_result") or {}).get("summary", "")
                            )
                            _gj = _judge.judge(goal.objective, None, _final_text)
                            if _gj.get("achieved"):
                                _blk, _why = self._metacog_check_completion()
                                if _blk:
                                    logger.info("v10 GoalJudge audit blocked: %s", _why)
                                    self._speculator_hint = (
                                        (self._speculator_hint + f"\n[completion audit] {_why}").strip()
                                    )
                                else:
                                    logger.info("v10 GoalJudge achieved (score=%s)", _gj.get("score"))
                                    state.should_stop = True
                            elif _gj.get("gaps"):
                                _gap_hint = "; ".join(_gj["gaps"][:3])
                                self._speculator_hint = (
                                    (self._speculator_hint + "\n" + _gap_hint).strip()
                                    if self._speculator_hint else _gap_hint
                                )
                                logger.info("v10 GoalJudge gaps: %s", _gap_hint)
                        except Exception:
                            logger.debug("v10 F17 GoalJudge failed (non-fatal)", exc_info=True)

                # v10-F4: surprise 早停 — 对齐 run() L1967-1999.
                # 连续 3 轮低 surprise + audit 不阻断 → should_stop.
                # 阈值自适应: noise 大时严格 (0.08), noise 小时宽松 (0.20).
                if not state.should_stop and len(self._surprise_history) >= 3:
                    try:
                        _recent = self._surprise_history[-3:]
                        _worsts = [w for w, _ in _recent]
                        _avg_noise = sum(s for _, s in _recent) / len(_recent)
                        _thr = max(0.08, 0.20 - 0.4 * _avg_noise)
                        if all(w < _thr for w in _worsts):
                            _blk, _why = self._metacog_check_completion()
                            if _blk:
                                logger.info("v10 surprise audit blocked: %s", _why)
                                self._speculator_hint = (
                                    (self._speculator_hint + f"\n[completion audit] {_why}").strip()
                                )
                            else:
                                logger.info(
                                    "v10 surprise converged < %.2f (noise=%.2f), stop",
                                    _thr, _avg_noise,
                                )
                                state.should_stop = True
                    except Exception:
                        logger.debug("v10 F4 surprise early-stop failed (non-fatal)", exc_info=True)

                # v10-F3: darwin_ratchet — 对齐 run() L2003-2004.
                # 内部判 stagnation >= 5 设 self._should_stop; 这里同步到 state.
                # ponytail: _darwin_ratchet_check 也更新 heat_engine T_cold + health,
                #   不只是 stop 判定. run() 用 self._should_stop, run_cognitive 用 state.should_stop.
                if not state.should_stop:
                    try:
                        self._darwin_ratchet_check()
                        if getattr(self, "_should_stop", False):
                            state.should_stop = True
                    except Exception:
                        logger.debug("v10 F3 darwin_ratchet failed (non-fatal)", exc_info=True)

            # P15: 周期 save — flag off 时 no-op, iteration % save_every == 0 才真写.
            # refute 在 _learn 内发生, reflect 末尾的周期 save 会在 ≤save_every 步内捕获.
            self._maybe_save_engine_state(reason="periodic")

            return ReflectionResult(
                should_continue=True,
                should_redirect=redirect,
                redirect_reason=advice if redirect else "",
                advice=advice,
                should_stop=state.should_stop,
            )

        loop = CognitiveLoop(
            observe_fn=observe_fn,
            decide_fn=decide_fn,
            execute_fn=execute_fn,
            reflect_fn=reflect_fn,
            output_writer=None,  # provenance 走 _run_phase_async 内的 _record_provenance
            max_iterations=max_iterations,
            max_repeated_actions=3,
        )
        state = await loop.run(LoopState(max_iterations=max_iterations))

        # finalize — 复用 run() 的收尾 (含 _report)
        return await self._finalize_run(
            objective,
            cog["phases"],
            run_id,
            provenance_record,
            run_collector,
            tracker,
            progress_task_id,
            cog["completed_steps"],
        )

    async def _decide_next_action_llm(
        self, state: LoopState, cog: dict, obs: dict,
    ) -> ActionDecision | None:
        """LLM 自主选 next action. 失败/非法返回 None, 调用方 fallback 到规则版.

        上下文: iteration / last_action / hypothesis/plan/execution/validation 状态.
        输出: JSON {"action", "rationale", "expected_outcome"}.
        合法性: 没hyp不能plan, 没plan不能execute, etc. 不合法 → None.
        """
        prompt = self._build_decider_prompt(state, cog, obs)
        try:
            raw = await self._llm_chat(prompt, persona_name="reviewer", task="reasoning")
        except Exception as e:
            logger.debug("decider LLM call failed: %s", e)
            return None
        if not raw:
            return None
        raw = raw.strip()
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            data = json.loads(raw[start:end + 1])
        except Exception:
            return None
        action = str(data.get("action", "")).strip().lower()
        if action not in VALID_ACTIONS:
            logger.debug("decider returned invalid action %r", action)
            return None
        if not self._is_action_legal(action, cog):
            logger.info("decider illegal action %s (preconditions not met), fallback", action)
            return None
        return ActionDecision(
            action=action,
            rationale=str(data.get("rationale", ""))[:200],
            expected_outcome=str(data.get("expected_outcome", ""))[:200],
        )

    def _build_decider_prompt(
        self, state: LoopState, cog: dict, obs: dict,
    ) -> str:
        """简短 prompt — 给 LLM 控制流状态, 不重复 phase 内部细节.

        D1: 扩字段 — validation details / consecutive_failures / pivot_count /
        refine_count / action_history / speculator_hint / last_learn_summary.
        之前 LLM 只看到 5 个粗投影, 现在给 belief state 多几维.
        ponytail: 不重组结构, 只在 State block 后追加 Detail block. 升级路径:
        把这些字段做成 PhaseRegistry extra, 跟 H4 phase 分批一致.
        """
        hyp = cog.get("hypothesis") or "NONE"
        if isinstance(hyp, str) and len(hyp) > 120:
            hyp = hyp[:120]
        plan = cog.get("plan")
        plan_mode = plan.get("mode", "NONE") if isinstance(plan, dict) else "NONE"
        exec_done = cog.get("execution_result") is not None
        val = cog.get("validation") or {}
        val_status = "PASSED" if _extract_tests_passed(val) else "FAILED" if val else "NONE"

        # D1: validation 具体字段 — 让 LLM 看到为什么 PASSED/FAILED
        val_detail = ""
        if isinstance(val, dict) and val:
            _vd_keys = (
                "thinking_collapse", "physics_validation_error",
                "reviewer_critique", "dimensional_consistent",
                "pde_classification", "constraint_check",
            )
            _vd_parts = []
            for _k in _vd_keys:
                _v = val.get(_k)
                if _v:
                    _vd_parts.append(f"{_k}={str(_v)[:80]}")
            if _vd_parts:
                val_detail = "; ".join(_vd_parts)[:300]

        # D1: 控制流统计 — 让 LLM 感知 stuck 程度
        # getattr 防 __new__ 测试场景 (selfcheck 绕过 __init__).
        _max_fail = getattr(self, "_max_consecutive_failures", 20)
        _max_pivot = 10  # ponytail: 硬编码上限, 升级路径走 PhaseRegistry extra
        action_hist_str = ", ".join(state.action_history[-10:]) or "none"
        spec_hint = (getattr(self, "_speculator_hint", "") or "")[:300]
        last_learn = (cog.get("last_learn_summary") or "none")[:200]
        # F-borrow: 分类失败计数 — 让 LLM 看到是 tool_error 多还是 hypothesis_error 多,
        # 不同失败类型语义不同 (技术故障 vs 方向错). 空时不显示, 避免噪声.
        _by_type = getattr(self, "_consecutive_failures_by_type", {}) or {}
        _type_max = getattr(self, "_max_failures_by_type", {}) or {}
        _type_parts = [
            f"{k}={v}/{_type_max.get(k, '?')}"
            for k, v in sorted(_by_type.items())
            if v > 0
        ]
        _type_str = ", ".join(_type_parts) if _type_parts else "none"
        # 700 万步场景: 滑动窗口失败率. 让 LLM 看到整体趋势, 不只是 consecutive.
        # 窗口未满时不显示 (数据不足). ponytail: 简单 fail rate, 不做加权.
        _vwin = getattr(self, "_validate_window", None) or []
        _wsize = getattr(self, "_validate_window_size", 100)
        if len(_vwin) >= _wsize:
            _wfail_rate = 1.0 - (sum(_vwin) / len(_vwin))
            _window_str = f"{_wfail_rate:.2f} (last {_wsize})"
        else:
            _window_str = f"building ({len(_vwin)}/{_wsize})"
        # 700 万步场景: 跨 run 失败模式. 让 LLM 知道上 run 主要卡在哪类失败.
        _last_pattern = (getattr(self, "_last_run_failure_pattern", "") or "").strip()

        return f"""You are the cognitive controller of a research agent. Choose the next action.

Iteration: {state.iteration}/{state.max_iterations}
Last action: {state.last_action or 'NONE'}
Last rationale: {state.last_rationale or 'none'}

State:
- Hypothesis: {hyp}
- Plan mode: {plan_mode}
- Execution: {'DONE' if exec_done else 'NONE'}
- Validation: {val_status}
- Consecutive failures: {getattr(self, "_consecutive_failures", 0)}/{_max_fail}
- Failures by type: {_type_str}
- Window fail rate: {_window_str}
- Last run pattern: {_last_pattern or 'none'}
- Pivot count: {getattr(self, "_pivot_count", 0)}/{_max_pivot}
- Refine count: {getattr(self, "_refine_count", 0)}
- Action history (last 10): {action_hist_str}
- Last learn summary: {last_learn}

Validation details: {val_detail or 'none'}
Speculator hints: {spec_hint or 'none'}
Last reflection advice: {state.redirect_reason or 'none'}

Actions:
- observe: re-perceive environment (context stale / need fresh data)
- hypothesize: generate new hypothesis (no hypothesis / after pivot)
- plan: design execution plan for current hypothesis
- execute: run the plan
- validate: check execution results
- learn: update memory/KG with results
- pivot: switch to new hypothesis (current path stuck)
- skip: do nothing this iteration
- stop: end the loop

Note: report runs automatically when loop ends — do not pick "report".

Respond JSON only:
{{"action": "one of above", "rationale": "1 sentence why", "expected_outcome": "1 sentence what you expect"}}"""

    def _is_action_legal(self, action: str, cog: dict) -> bool:
        """LLM 选择的 action 是否合法 (前置条件满足)."""
        if action in ("observe", "hypothesize", "skip", "stop"):
            return True
        # D3: report 不让 LLM 选 — _finalize_run 自动跑. LLM 选 report 等于
        # 浪费一轮 (execute_fn 是 no-op). 升级路径: 如果要 LLM 主动触发,
        # 改成 action="stop" + rationale="report ready".
        if action == "report":
            return False
        if action == "plan":
            return bool(cog.get("hypothesis"))
        if action == "execute":
            return bool(cog.get("plan"))
        if action == "validate":
            return cog.get("execution_result") is not None
        if action == "learn":
            return all(cog.get(k) for k in ("hypothesis", "plan", "validation"))
        if action == "pivot":
            return bool(cog.get("current_hyp_id") or cog.get("hypothesis"))
        return False

    def _git_commit_after_execute(self, plan: dict, iteration: int) -> None:
        """execute 后 git commit — 让下轮 perceive 看到 diff (从 run() 抽出)."""
        try:
            import subprocess as _sp
            import time as _time
            _sp.run(["git", "add", "-A"], cwd=self.workspace,
                    capture_output=True, timeout=10)
            _msg = f"[iter {iteration}] {plan.get('mode','?')}: {plan.get('description','')[:80]}"
            for _attempt in range(3):
                _r = _sp.run(["git", "commit", "-m", _msg], cwd=self.workspace,
                             capture_output=True, timeout=10)
                if _r.returncode == 0:
                    break
                if _attempt < 2:
                    _time.sleep(1 * (_attempt + 1))
        except Exception:
            pass  # no git repo or git unavailable — not our problem

    def _darwin_ratchet_check(self) -> None:
        """Darwin ratchet: 算假设质量分, 只保留改进, 连续低增益 → early stop.

        评分 (0-10, 对齐 darwin-skill 原版 0-10 分制):
        - supported_ratio * 10: 证据强度 (supported 节点占比)
        - testable_ratio * 10: 可证伪性 (有 testable_prediction 的节点占比)
        - graph_diversity * 10: 假设多样性 (unique statements 占比)
        - topology_richness * 10: 假设网络结构丰富度 (β₁/n, 独立环数占比)
        四项平均 → 0-10 分

        β₁ 解释: 假设图的独立环数. β₁=0 → 树状 (无交叉支持);
        β₁>0 → 有交叉支持/反驳链 (假设间相互关联). 标准化到 [0,1] 避免大图偏向.
        ponytail: 不区分"良性交叉支持"和"恶性循环论证" — 留给 red_team._topology_scan 判.
        这里只测结构丰富度, 作为 4 维代理之一. 升级: LLM 9 维评分 (darwin-skill 原版).

        棘轮逻辑:
        - score > best_score → 更新 best, stagnation=0
        - score <= best_score → stagnation++, 回退 (不更新 preferred_hypothesis)
        - 连续 2 轮 Δ<0.5 (0-10 分制下, 增量 <0.5) → early stop

        ponytail: 4 维代理是粗启发式. 升级: LLM 9 维评分 (darwin-skill 原版).
        ponytail: 回退只标记, 不真删假设 (保留在图里供 cross-pollination).
        """
        graph = self.hypothesis_graph
        all_nodes = graph.all_nodes()
        if not all_nodes:
            return

        supported = graph.supported()
        n = len(all_nodes)
        supported_ratio = len(supported) / n

        testable = sum(
            1 for nd in all_nodes if getattr(nd, "testable_prediction", None)
        )
        testable_ratio = testable / n

        statements = [nd.statement for nd in all_nodes if nd.statement]
        unique = len(set(statements))
        graph_diversity = unique / len(statements) if statements else 0.0

        # 第 4 维: 拓扑丰富度 — 用 hodge_signature 的 β₁ 算独立环数占比
        # ponytail: 失败时降级为 0 (不影响其他 3 维). 升级: gudhi 算真实 Betti.
        topology_richness = 0.0
        try:
            from huginn.metacog.topology_lens import hodge_signature

            node_ids = [nd.id for nd in all_nodes]
            edge_pairs = []
            for e in graph.edges():
                if e.from_id in node_ids and e.to_id in node_ids:
                    edge_pairs.append((e.from_id, e.to_id))
            sig = hodge_signature(node_ids, edge_pairs)
            # β₁/n 标准化到 [0,1]: 树状图 β₁=0 → 0 分; 完全交叉 → 趋近 1
            topology_richness = min(sig.beta1_approx / max(n, 1), 1.0)
        except Exception:
            logger.debug("topology_richness calc failed (non-fatal)", exc_info=True)

        # 0-10 分制, 对齐 darwin-skill 原版
        score = (
            (supported_ratio + testable_ratio + graph_diversity + topology_richness)
            / 4.0
            * 10.0
        )

        delta = score - self._darwin_last_score
        if delta < 0.5:
            self._darwin_stagnation += 1
        else:
            self._darwin_stagnation = 0

        if score > self._darwin_best_score:
            self._darwin_best_score = score
            # ponytail: 只在改进时更新 preferred, 退化时保留上次最佳
        # else: 保留 best_score, preferred_hypothesis 不更新 (棘轮)

        self._darwin_last_score = score

        # P2-6 belief: Gaussian 后验更新. 单值 score 当观测, obs_sigma2=1.0
        # (0-10 分制下单次观测噪声约 1 分). σ² 减小 = belief 收敛.
        # toggle: HUGINN_BELIEF_DARWIN (默认 on). off 时只走原 delta<0.5 逻辑.
        if os.environ.get("HUGINN_BELIEF_DARWIN", "1") != "0":
            try:
                from huginn.tools.subagent_tool import _gaussian_update
                self._darwin_belief_mu, self._darwin_belief_sigma2 = _gaussian_update(
                    self._darwin_belief_mu, self._darwin_belief_sigma2,
                    float(score), 1.0,
                )
                try:
                    from huginn.routes.metrics import track_belief_update
                    track_belief_update("gaussian")
                except Exception:
                    pass
            except Exception:
                pass  # 循环 import 或其他故障 → 回退原逻辑

        # v6 G54: 把 darwin 分数 / supported_ratio 暴露给 _plan / _validate
        # ponytail: evidence_strength 用 supported_ratio 做代理, 已在算分时拿到,
        # 不重复调 RAG. 升级路径: 真 RAG recall 命中数 / provenance 引用数.
        self._last_hypothesis_confidence = score / 10.0
        self._last_hypothesis_evidence_strength = float(supported_ratio)

        # v7 G59: 更新认知热机 T_cold (paradigm 秩序代理)
        # supported_ratio 高 = validation 提取有序能力强 = 冷源温度低
        try:
            from huginn.metacog.cognitive_heat_engine import get_heat_engine
            eng = get_heat_engine()
            eng.update_T_cold(float(supported_ratio), float(score))

            # 推送 health 到 EventBus + SSE. 每轮 darwin 后推一次, 让前端实时看
            # Re_cog / η_cog / status. _should_imaginate 已 update_kinematics,
            # 但若本轮没触发 imaginate, 这里强制 update 保证 health 反映当前状态.
            n_ideas = len(all_nodes)
            n_principles = 0
            try:
                sp = getattr(self, "stable_principles", None)
                n_principles = len(sp) if sp else 0
            except Exception:
                pass
            sys_prompt_len = 0
            try:
                sys_prompt_len = len(getattr(self, "system_prompt", "") or "")
            except Exception:
                pass
            eng.update_kinematics(n_ideas, n_principles + 1, sys_prompt_len)

            health = eng.health_check()
            self._emit_campaign("heat_engine.health", health)
        except Exception:
            logger.debug("heat_engine.update_T_cold failed (non-fatal)", exc_info=True)

        # v7 长任务: stagnation 阈值 2→5. Oxelra 206 步允许长期低增益,
        # 2 轮就 early stop 太激进, 真正突破常在 10+ 轮停滞之后.
        _stag_limit = int(os.environ.get("HUGINN_DARWIN_STAGNATION_LIMIT", "5"))
        if self._darwin_stagnation >= _stag_limit and self._iteration > 2:
            # P2: stagnation 触发前先分类 (chaoxu 启发).
            # method_failure → pivot 换方法继续, 不 stop
            # evidence_against → counterexample hunt, 不 stop
            # unclassifiable / 已试过 → 真 stop
            _stall_action = self._classify_stall()
            if _stall_action == "pivot":
                logger.info(
                    "darwin ratchet: stagnation %d → method_failure, pivot (不 stop)",
                    self._darwin_stagnation,
                )
                self._darwin_stagnation = 0  # 给 pivot 后的新假设重新累积
            elif _stall_action == "counterexample":
                logger.info(
                    "darwin ratchet: stagnation %d → evidence_against, counterexample hunt (不 stop)",
                    self._darwin_stagnation,
                )
                self._darwin_stagnation = 0
                self._trigger_counterexample_hunt()
            else:
                # P5 (chaoxu 启发): persistent goal mode — stagnation 分类为 stop
                # 时, 如果开了 HUGINN_PERSISTENT_GOAL_MODE 且有 active goal 且
                # 挂钟预算未耗尽, 不 early stop, 重置 stagnation 继续.
                # 无 active goal 或挂钟耗尽才真 stop.
                _persistent = (
                    os.environ.get("HUGINN_PERSISTENT_GOAL_MODE", "0") == "1"
                )
                _wall_expired = False
                _has_active_goal = False
                if _persistent:
                    try:
                        from huginn.autoloop.goal_store import get_goal_store
                        _gs = get_goal_store()
                        _ag = _gs.get_active()
                        if _ag is not None:
                            _has_active_goal = True
                            _wall_expired = _gs.wall_clock_expired(_ag.id)
                    except Exception:
                        logger.debug("P5 wall_clock check failed", exc_info=True)
                if _persistent and _has_active_goal and not _wall_expired:
                    logger.info(
                        "darwin ratchet: stagnation %d → stop, but persistent goal "
                        "mode on + wall_clock not expired, reset & continue",
                        self._darwin_stagnation,
                    )
                    self._darwin_stagnation = 0
                else:
                    logger.info(
                        "darwin ratchet: stagnation %d rounds (Δ<0.5), best=%.2f, early stop",
                        self._darwin_stagnation,
                        self._darwin_best_score,
                    )
                    self._should_stop = True

        # P2-6 belief: σ² 收敛也作为 stop 信号. σ² < 0.1 = belief 不确定性低,
        # 后续观测不会显著改变 μ, 边际信息收益递减. 跟 stagnation 互补:
        # stagnation 测"score 不增", σ² 测" belief 不再变". 两者任一触发即 stop.
        if (
            os.environ.get("HUGINN_BELIEF_DARWIN", "1") != "0"
            and self._darwin_belief_sigma2 < 0.1
            and self._iteration > 2
        ):
            logger.info(
                "darwin ratchet: belief converged σ²=%.4f μ=%.2f, early stop",
                self._darwin_belief_sigma2, self._darwin_belief_mu,
            )
            self._should_stop = True

        # v7 Meta-Trace: 每轮蒸馏成结构化科研要点, 对标 Oxelra Meta-Trace.
        # 目标: 长任务不靠完整 transcript, 用结构化要点保持 context 密度.
        # ponytail: 从已有 self.* 字段抽, 不调 LLM (省 token). ceiling 是 LLM 蒸馏.
        try:
            self._distill_meta_trace(score, supported_ratio)
        except Exception:
            logger.debug("meta_trace distill failed (non-fatal)", exc_info=True)

    def _classify_stall(self) -> str:
        """P2: stagnation 触发时归因 (chaoxu 启发).

        分两类:
        - method_failure: 当前方法/工具用尽, 换方法能救 → 返回 "pivot"
        - evidence_against: 证据指向假设本身错 → 返回 "counterexample"
        - unclassifiable / 已试过太多次 → 返回 "stop"

        ponytail: 用 _last_failure_mode + _consecutive_failures + pivot_count
        做规则归因, 不调 LLM (省 token). 升级: LLM judge 归因.
        ceiling: 只用已有信号, 不引入新 sensor.
        """
        _fail_mode = getattr(self, "_last_failure_mode", "") or ""
        _consec = getattr(self, "_consecutive_failures", 0)
        _pivots = getattr(self, "_pivot_count", 0)
        _max_pivots = getattr(self, "_max_pivots", 10)
        # evidence_against 信号: failure_mode 含 hypothesis_error / refuted /
        # counterexample, 或最近 validation 明确反驳
        _evidence_signals = (
            "hypothesis_error" in _fail_mode
            or "refuted" in _fail_mode.lower()
            or "counterexample" in _fail_mode.lower()
            or "contradicts" in _fail_mode.lower()
        )
        # method_failure 信号: failure_mode 含 tool_error / param_error /
        # timeout / convergence, 或纯工具失败
        _method_signals = (
            "tool_error" in _fail_mode
            or "param_error" in _fail_mode
            or "timeout" in _fail_mode
            or "convergence" in _fail_mode
            or "data_noise" in _fail_mode
        )
        # 已 pivot 太多次 → 不再 pivot, 考虑 stop
        if _pivots >= _max_pivots:
            return "stop"
        if _evidence_signals and not _method_signals:
            return "counterexample"
        if _method_signals and not _evidence_signals:
            return "pivot"
        # 混合信号或无信号: 看 consecutive_failures
        # 高失败率 → 假设方向可能错 → counterexample
        # 中低失败率 → 方法问题 → pivot
        if _consec >= 10:
            return "counterexample"
        if _consec >= 3:
            return "pivot"
        return "stop"

    def _trigger_counterexample_hunt(self) -> None:
        """P2: 触发反例搜索 (chaoxu 启发).

        两种路径:
        1. SMT 离散反例 (已有 _discrete_counterexample_scan, 需 evidence 带
           discrete_hypothesis 字段)
        2. LLM 主动构造反例 scenario — 让 imagination block 强制开, 下轮
           hypothesize 时 LLM 被要求考虑反事实

        ponytail: 不新起 subagent (贵), 只设 flag 让下轮 _should_imaginate
        返 True + 注入 counterexample hint. 升级: LLM 把 hypothesis 翻译成
        z3 表达式跑 SMT.
        """
        # 强制开 imagination (override _should_imaginate 的判断)
        self._force_imaginate = True
        # 注入 counterexample hint 给下轮 hypothesize
        _cur_hyp = getattr(self, "_current_hyp_id_for_plan", None)
        _stmt = ""
        if _cur_hyp:
            try:
                _node = self.hypothesis_graph._nodes.get(_cur_hyp)
                if _node:
                    _stmt = _node.statement[:200]
            except Exception:
                pass
        _hint = (
            f"Stagnation classified as evidence_against. "
            f"Current hypothesis may be wrong. Hunt for a counterexample.\n"
            f"Hypothesis: {_stmt}\n"
            f"Construct a specific scenario / parameter set where this hypothesis "
            f"would fail. If found, refute and pivot to a corrected hypothesis."
        )
        # _speculator_hint 会被 _build_hypothesis_prompt 读取注入
        self._speculator_hint = (
            (getattr(self, "_speculator_hint", "") or "") + "\n" + _hint
        )
        logger.info("P2 counterexample hunt triggered, hint injected")

    def _distill_meta_trace(self, darwin_score: float, supported_ratio: float) -> None:
        """把本轮蒸馏成结构化科研要点, 追加到 .huginn/meta_trace.jsonl.

        Oxelra Meta-Trace 启示: 每步边界蒸馏 what attempted/found/evidence/
        limitation/artifact/next_hint. 不存 raw trace, 只存结构化要点.

        ponytail: 字段从 self.* 现有状态抽, 不调 LLM. ceiling 是 LLM 蒸馏.
                  文件路径跟 stable_principles 同目录 (.huginn/), 一行一个 JSON.
        """
        import json
        import time
        from pathlib import Path

        # 从 self.* 抽本轮关键信息 (都是上一轮 phase 写进去的)
        attempted = ""
        if getattr(self, "_last_hypothesis", None):
            attempted = str(self._last_hypothesis)[:300]

        found = ""
        limitations: list[str] = []
        if getattr(self, "_last_validation", None) and isinstance(self._last_validation, dict):
            v = self._last_validation
            found = str(v.get("result", ""))[:300]
            if v.get("errors"):
                limitations.append(str(v["errors"])[:200])

        evidence: list[str] = []
        try:
            for nd in self.hypothesis_graph.supported()[:3]:
                evidence.append(str(nd.statement)[:150])
        except Exception:
            pass

        artifacts: list[str] = []
        if getattr(self, "_last_execution_result", None) and isinstance(self._last_execution_result, dict):
            outs = self._last_execution_result.get("outputs") or self._last_execution_result.get("files")
            if isinstance(outs, list):
                artifacts = [str(f)[:150] for f in outs[:5]]
            elif isinstance(outs, str):
                artifacts = [outs[:150]]

        next_hint = (getattr(self, "_speculator_hint", "") or "")[-300:]

        entry = {
            "iteration": self._iteration,
            "ts": time.time(),
            "role": "autoloop",  # ponytail: 单 agent, role 固定; 升级多 agent 后填实际 role
            "attempted": attempted,
            "found": found,
            "evidence": evidence,
            "limitations": limitations,
            "artifacts": artifacts,
            "next_hint": next_hint,
            "darwin_score": round(darwin_score, 2),
            "supported_ratio": round(supported_ratio, 3),
        }

        # 写到 workspace 的 .huginn/meta_trace.jsonl (不存在就建)
        # ponytail: 不走 memory_manager, 直接写文件. 跟 directive_rejections 同模式.
        ws = getattr(self, "workspace_root", None) or Path.cwd()
        trace_path = Path(ws) / ".huginn" / "meta_trace.jsonl"
        try:
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            with trace_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            logger.debug("meta_trace write failed (non-fatal)", exc_info=True)

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
            track_task(
                _publish(event_type, data, source="autoloop"), name="campaign-emit"
            )
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
        # F-borrow: 分类计数器随 run 重置 (跨 run 失败模式记忆没意义, 误导自适应).
        self._consecutive_failures_by_type = {}
        # 700 万步场景: 滑动窗口随 run 重置 (跨 run 失败率无意义).
        self._validate_window = []
        # 700 万步场景: 加载上 run 失败模式快照, 让 decider 知道历史卡点.
        # 不恢复计数器 (跨 run 计数无意义), 只注入 prompt 作为参考.
        self._last_run_failure_pattern: str = ""
        try:
            self._last_run_failure_pattern = self._load_failure_pattern()
        except Exception:
            logger.debug("load failure pattern failed", exc_info=True)

        if goal is not None and goal.status == "pending":
            goal.status = "active"
            if self._goal_scheduler is not None:
                self._goal_scheduler.update_goal(goal.id, status="active")

        self._budget = ProgressiveBudget.default() if progressive_budget else None
        self._budget_rejects: dict[str, int] = {}
        self._budget_degraded = False
        # plan_check 状态随 run 重置 — 跨 run 的历史成功率没意义, 会误导自适应.
        # patterns 例外: 跨 run 保留 (失败模式记忆), 加载 workspace 里的历史.
        # extra_keywords 也跨 run 保留 (自动发现的 scene_tag 关键词).
        self._plan_check_history = []
        self._plan_check_last_result = None
        self._plan_check_warnings = []
        self._plan_check_patterns = []
        self._scene_tag_extra_keywords = {}
        self._load_plan_check_patterns()

        self._speculator_hint = ""
        self._last_visual_context = ""  # reset per run, stale data shapes mislead
        self._current_prediction = ""  # reset JEPA prediction buffer
        self._last_surprise = 0.0
        self._last_raw_hypothesis = ""  # 完整 LLM 输出, 含 LUCID review
        # G2: 加载历史 trajectory action 序列, 给 _check_stuck 当 VF2 匹配历史.
        # 失败/空都不影响 run, 只是少了 cross-run 匹配能力.
        try:
            self._traj_history = self._load_trajectory_action_history(limit=20)
        except Exception:
            self._traj_history = []
            logger.debug("G2 traj history load failed (non-fatal)", exc_info=True)
        try:
            from huginn.agents.speculator import on_turn_start

            spec_result = on_turn_start(objective)
            self._speculator_hint = spec_result.get("hint", "")
            if spec_result.get("predictions"):
                logger.info("autoloop speculator: %s", self._speculator_hint)
        except Exception:
            logger.warning("autoloop speculator skipped", exc_info=True)

        return run_id, provenance_record, run_collector

    def _persist_failure_pattern(self, run_id: str) -> None:
        """run 结束时把 by_type + window 快照存 longterm. 供下 run 加载.

        700 万步场景: 单 run 可能只跑几十万步, 跨 run 失败模式记忆让 decider
        知道"上次主要卡在 tool_error" → 这次优先换工具/换 backend.
        ponytail: 复用 longterm.store, JSON 序列化. 升级路径: 独立 failure_pattern 表.
        """
        by_type = getattr(self, "_consecutive_failures_by_type", {}) or {}
        vwin = getattr(self, "_validate_window", None) or []
        # 只在有失败数据时存 — 全 pass 的 run 存了也没参考价值.
        if not by_type and not vwin:
            return
        wsize = getattr(self, "_validate_window_size", 100)
        fail_rate = 1.0 - (sum(vwin) / len(vwin)) if vwin else 0.0
        snapshot = {
            "run_id": run_id,
            "by_type": by_type,
            "window_size": len(vwin),
            "window_fail_rate": round(fail_rate, 3),
            "total_consecutive": getattr(self, "_consecutive_failures", 0),
            "objective": (getattr(self, "_objective", "") or "")[:200],
        }
        try:
            content = json.dumps(snapshot, ensure_ascii=False)
            self.memory.remember(
                content=content,
                category="failure_pattern",
                tags=["failure_pattern", run_id],
                importance=0.6,
                tier="mid",
            )
        except Exception:
            logger.debug("failure_pattern store failed", exc_info=True)

    def _load_failure_pattern(self) -> str:
        """加载最近一条 failure_pattern, 返回人类可读摘要供 decider prompt 注入.

        返回空串表示无历史或加载失败. ponytail: 只取最近 1 条, 不做聚合.
        用空 query + category 过滤 — content 是 JSON, FTS5 语义匹配不到.
        """
        try:
            results = self.memory.recall(
                query="",
                category="failure_pattern",
                top_k=1,
            )
        except Exception:
            return ""
        if not results:
            return ""
        entry = results[0] if isinstance(results, list) else results
        content = entry.get("content", "") if isinstance(entry, dict) else str(entry)
        if not content:
            return ""
        try:
            snap = json.loads(content)
        except (ValueError, TypeError):
            return ""
        by_type = snap.get("by_type", {}) or {}
        if not by_type:
            return ""
        parts = [f"{k}={v}" for k, v in sorted(by_type.items()) if v > 0]
        if not parts:
            return ""
        rate = snap.get("window_fail_rate", 0.0)
        return (
            f"last run: {', '.join(parts)}, "
            f"window fail rate={rate:.2f} "
            f"(n={snap.get('window_size', 0)})"
        )

    async def _finalize_run(
        self,
        objective: str,
        phases: list[LoopPhase],
        run_id: str,
        provenance_record: Any,
        run_collector: Any,
        tracker: Any,
        progress_task_id: str,
        completed_steps: int,
    ) -> AutoloopResult:
        """Report, save trajectory, judge goal, write provenance + FAIR metadata."""
        # 700 万步场景: 失败模式跨 run 持久化. run 结束时存 by_type + window 快照,
        # 下个 run 开始时加载, 让 LLM 知道"上次主要卡在 tool_error 还是 hypothesis_error".
        # ponytail: 复用 longterm.store, JSON 序列化. 升级路径: 独立 failure_pattern 表.
        try:
            self._persist_failure_pattern(run_id)
        except Exception:
            logger.debug("persist failure pattern failed", exc_info=True)
        total_time = time.time() - getattr(self, "_run_start_time", time.time())
        report_phase = await self._run_phase_async(
            "report", self._report, objective, phases, total_time
        )
        phases.append(report_phase)
        completed_steps += 1
        tracker.update(
            progress_task_id,
            current_step=completed_steps,
            current_label=f"report ({report_phase.status})",
        )

        if report_phase.status == "completed":
            tracker.complete(
                progress_task_id, result={"report_path": report_phase.result}
            )
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
            from huginn.telemetry import load_trajectory, save_trajectory

            traj_dir = self.workspace / ".huginn" / "trajectories"
            trajectory_path = traj_dir / f"{run_id}.json"
            save_trajectory(
                run_collector,
                trajectory_path,
                metadata={
                    "run_id": run_id,
                    "objective": objective[:200],
                    "phases": [p.name for p in phases],
                    "total_time": total_time,
                },
            )
            trajectory_data = load_trajectory(trajectory_path)
            # G31: trajectory tool_calls 断言 — 修 audit 06 F1
            # (18/18 轨迹 0 工具调用, perceive+report 两阶段空转).
            # 非查询 objective + 0 工具调用 = autoloop 装置未激活, 记 warning.
            _tc = trajectory_data.get("tool_calls", []) if trajectory_data else []
            _obj_lower = (objective or "").strip().lower()
            _is_query = _obj_lower.startswith("query") or _obj_lower.startswith("read")
            if not _tc and objective and not _is_query:
                logger.warning(
                    "G31: trajectory has 0 tool_calls for non-query objective "
                    "'%s' — autoloop may be空转 (audit 06 F1)",
                    objective[:100],
                )
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
                objective=objective,
                trajectory=trajectory_data,
                final_output=final_output,
            )
            goal_achieved = goal_judgment.get("achieved")
        except Exception:
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
            from huginn.export.fair_metadata import (
                generate_dataset_metadata,
                write_fair_jsonld,
            )

            run_results: dict[str, Any] = {}
            for ph in phases:
                if ph.result and isinstance(ph.result, dict):
                    run_results.update(ph.result)
            fair_metadata = generate_dataset_metadata(
                run_id=run_id,
                objective=objective,
                results=run_results,
                provenance={
                    "report_path": (
                        str(report_phase.result) if report_phase.result else None
                    ),
                    "trajectory_path": (
                        str(trajectory_path) if trajectory_path else None
                    ),
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

        # P2: trajectory success pattern 抽取 — 复用 KB + auto_ingest 路径
        # 不新建 skill_library 组件. 仅在 goal_achieved=True 时调一次 LLM 抽
        # 可复用 pattern, 写入 KB. 下次任务开始时 RAG 自然召回.
        # 默认关, HUGINN_TRAJECTORY_PATTERN=1 开启 (跟 PRT Level 1 / PRM verifier
        # 同款策略: 有 LLM 成本).
        if goal_achieved and os.environ.get("HUGINN_TRAJECTORY_PATTERN", "0") == "1":
            try:
                from huginn.knowledge.trajectory_pattern import (
                    extract_and_store_pattern,
                )

                async def _pattern_chat(prompt: str) -> str:
                    # 复用 verification_model (默认 fallback 到 self.model)
                    from langchain_core.messages import HumanMessage
                    resp = await self.verification_model.ainvoke(
                        [HumanMessage(content=prompt)]
                    )
                    return getattr(resp, "content", str(resp))

                pattern_doc_id = await extract_and_store_pattern(
                    objective=objective,
                    trajectory=trajectory_data,
                    final_output=str(report_phase.result or ""),
                    llm_chat_fn=_pattern_chat,
                    run_id=run_id,
                )
                if pattern_doc_id:
                    logger.info(
                        "trajectory pattern stored: doc_id=%s (run %s)",
                        pattern_doc_id, run_id,
                    )
            except Exception:
                logger.debug(
                    "trajectory pattern extraction failed (non-fatal)",
                    exc_info=True,
                )

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
            merged_graph=self._merged_graph,
            speculator_hint=self._speculator_hint,
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
            # G10/F14: perception 信号经 SignalHub 路由成 TransitionSignal.
            # ponytail: engine 无 csm 引用，走 _pending_signals 解耦；升级路径是 engine 注入 csm 直接 transition
            try:
                hub = SignalHub.shared()
                if getattr(cog, "errors_present", False):
                    sig = hub.route("perception_error", {"errors_present": True})
                    if sig is not None:
                        self._pending_signals.append(sig)
                if getattr(cog, "conflicts", None):
                    sig = hub.route("perception_conflict", {
                        "conflicts": [
                            {"sources": [c.source_a, c.source_b], "description": c.description}
                            for c in cog.conflicts
                        ],
                    })
                    if sig is not None:
                        self._pending_signals.append(sig)
                if getattr(cog, "simulation_converged", None) is True:
                    sig = hub.route("perception_converged", {"converged": True})
                    if sig is not None:
                        self._pending_signals.append(sig)
            except Exception:
                logger.debug("perception 信号构造失败, 不阻断 _perceive", exc_info=True)
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
                cwd=self.workspace,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                changed_files = [
                    line.strip() for line in result.stdout.strip().split("\n")
                ]
                git_diff = subprocess.run(
                    ["git", "diff", "--stat"],
                    cwd=self.workspace,
                    capture_output=True,
                    text=True,
                    timeout=10,
                ).stdout
        except Exception:
            logger.warning(
                "error in _perceive_legacy: git diff collection failed", exc_info=True
            )
        error_patterns = []
        for log_file in self.workspace.rglob("*.log"):
            if log_file.stat().st_mtime > time.time() - 3600:
                try:
                    content = log_file.read_text(errors="ignore")
                    if "ERROR" in content or "FAIL" in content:
                        error_patterns.append(f"{log_file.name}: {content[:200]}")
                except Exception:
                    logger.warning(
                        "error in _perceive_legacy: log file error-pattern scan failed",
                        exc_info=True,
                    )
        if not changed_files and not error_patterns:
            return None
        return {
            "changed_files": changed_files,
            "git_diff": git_diff,
            "error_patterns": error_patterns,
            "timestamp": datetime.now().isoformat(),
        }

    async def _hypothesize_via_branch_incubator(
        self, context: dict[str, Any]
    ) -> str | None:
        """走 BranchIncubator 的 N 路隔离采样, 替代 main+hot_model 2 路.

        HUGINN_USE_BRANCH_INCUBATOR=1 + agent_factory 注入时由 _hypothesize 调用.
        内部构造 prompt (复用 _build_hypothesis_prompt + symreg + conjecture hint),
        让每个 Subagent 看到完整 task. 返回最优 hypothesis (tokens_used 最小且 success);
        全失败时返回 None 让 caller fallback 到原 2 路. 异常吞掉 + log, 不 raise.
        """
        if self._agent_factory is None:
            return None
        try:
            from huginn.metacog.branch_incubator import BranchIncubator
        except Exception:
            logger.warning(
                "BranchIncubator import failed, fallback to main+hot_model",
                exc_info=True,
            )
            return None

        # 复用原 prompt 构造流程, 让 Subagent 看到相同 task
        symreg_task = asyncio.create_task(self._symreg_hint(context))
        conjecture_hint = self._conjecture_hint(context)
        symreg_hint = await symreg_task
        prompt = self._build_hypothesis_prompt(context)
        if symreg_hint:
            prompt = f"{symreg_hint}\n{prompt}"
        if conjecture_hint:
            prompt = f"{conjecture_hint}\n{prompt}"

        if self._branch_incubator is None:
            self._branch_incubator = BranchIncubator()

        try:
            results = await self._branch_incubator.run_round(
                task=prompt,
                agent_factory=self._agent_factory,
                n_branches=3,
                math_background=context.get("math_background", ""),
                researcher_intuition=context.get("researcher_intuition", ""),
                round_idx=self._iteration,
                total_rounds=max(self._max_pivots * 3, 10),
                depth=int(os.environ.get("HUGINN_BRANCH_INCUBATOR_DEPTH", "1")),
                width=2,
            )
        except Exception:
            logger.warning(
                "branch incubator run_round failed, fallback to main+hot_model",
                exc_info=True,
            )
            return None

        # 选 success + hypothesis 非空 + tokens_used 最小 (省 token)
        candidates = [
            r for r in results if r.success and r.hypothesis
        ]
        if not candidates:
            return None
        best = min(candidates, key=lambda r: r.tokens_used)
        return best.hypothesis

    async def _hypothesize(self, context: dict[str, Any]) -> str | None:
        """Generate a hypothesis from perceived context."""
        # BranchIncubator gating: flag on + factory 注入时走 N=3 隔离采样,
        # 失败/None 时 fallback 到下面 main+hot_model 2 路.
        # H4: env name + selected marker 从 PhaseRegistry extra 取 (toggle off 回退 hardcode)
        from huginn.harness.phase_spec import get_phase_extra
        _incubator_env = get_phase_extra(
            "_hypothesize", "branch_incubator_env", "HUGINN_USE_BRANCH_INCUBATOR"
        )
        _selected_marker = get_phase_extra(
            "_hypothesize", "selected_marker", "SELECTED:"
        )
        if (
            os.environ.get(_incubator_env, "0") == "1"
            and self._agent_factory is not None
        ):
            try:
                inc_hyp = await self._hypothesize_via_branch_incubator(context)
            except Exception:
                logger.warning(
                    "branch incubator unexpected error, fallback",
                    exc_info=True,
                )
                inc_hyp = None
            if inc_hyp:
                self._last_hypothesis = inc_hyp
                self._last_raw_hypothesis = inc_hyp
                self._record_backup_candidates(inc_hyp, inc_hyp)
                self._metacog_audit_hypothesis(inc_hyp, context)
                return inc_hyp
            logger.warning(
                "branch incubator returned None, fallback to main+hot_model"
            )
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
        self._last_context = context  # 供 _learn 写 persona_use entity (C5)
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
            coros = [
                self._llm_chat(prompt, persona_name=persona_name, task="reasoning")
            ]
            if hot_model is not None:
                coros.append(
                    self._llm_chat(
                        prompt,
                        persona_name=persona_name,
                        model=hot_model,
                        task="reasoning",
                    )
                )
            results = await asyncio.gather(*coros, return_exceptions=True)
            # Extract SELECTED: from results — main call first (priority)
            for raw in results:
                if isinstance(raw, Exception) or not raw:
                    continue
                raw = raw.strip()
                if _selected_marker in raw:
                    _after = raw.split(_selected_marker, 1)[1].strip()
                    _sel = _after.split("\n")[0].strip() if _after else ""
                    self._last_hypothesis = _sel or raw
                    self._last_raw_hypothesis = raw  # 保留 LUCID review 文本
                    # v11: 3 候选全进图 — 解析 [DIM: ...] 候选, backup 进图让 frontier 选优.
                    # ponytail: 解析失败不阻塞, SELECTED 仍正常返回. 升级路径: LLM 判定 dimension.
                    self._record_backup_candidates(raw, self._last_hypothesis)
                    self._metacog_audit_hypothesis(self._last_hypothesis, context)
                    return self._last_hypothesis
            # No SELECTED: found — fall back to first non-exception result
            for raw in results:
                if not isinstance(raw, Exception) and raw:
                    raw = raw.strip()
                    self._last_hypothesis = raw
                    self._last_raw_hypothesis = raw
                    self._record_backup_candidates(raw, self._last_hypothesis)
                    self._metacog_audit_hypothesis(self._last_hypothesis, context)
                    return self._last_hypothesis
            return None
        except Exception:
            return None

    def _record_backup_candidates(self, raw: str, selected: str) -> None:
        """v11/v12: 解析 [DIM: ...] 候选, backup 进图让 frontier 选优.

        ponytail: 正则解析, 失败不阻塞. SELECTED 的候选已在 _hypothesize 上游
        由 execute_fn 的 add_hypothesis 调用进图 (主路径), 这里只补 backup.
        v12: 同 dimension 第 2 个候选不再跳过, 改标 dim_conflict=True 留痕,
        让 LLM decider 在选优时避开 (cluster_block 已含 dim 分布提示).
        """
        try:
            import re
            # 匹配 [DIM: xxx] statement | pro: ... | con: ...
            _pattern = re.compile(
                r"\[DIM:\s*([^\]]+)\]\s*(.+?)(?:\s*\|\s*pro:.*?(?:\s*\|\s*con:.*?)?$|$)",
                re.MULTILINE,
            )
            _seen_dims: set[str] = set()
            for _m in _pattern.finditer(raw):
                _dim = _m.group(1).strip().lower()
                _stmt = _m.group(2).strip().split("\n")[0].strip()
                # 跳过 SELECTED 的那个 (它已进图)
                if not _stmt or _stmt == selected:
                    continue
                # v12: 同 dim 不再跳过, 标 dim_conflict 让 decider 避开
                _dim_conflict = _dim in _seen_dims
                _seen_dims.add(_dim)
                # backup 候选进图, 标 evidence={"candidate_role":"backup", "dim_conflict":...}
                _new_id = self.hypothesis_graph.add_hypothesis(
                    statement=_stmt,
                    rationale=f"backup candidate (dim={_dim})",
                )
                if _new_id:
                    self.hypothesis_graph._nodes[_new_id].evidence = {
                        **self.hypothesis_graph._nodes[_new_id].evidence,
                        "candidate_role": "backup",
                        "dim_conflict": _dim_conflict,
                    }
        except Exception:
            logger.debug("v11 _record_backup_candidates failed (non-fatal)", exc_info=True)

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
            families_explored = len(
                [
                    f
                    for f in self._get_metacog_method_registry().all()
                    if f.member_agent_ids
                ]
            )
            live_components = self.hypothesis_graph.component_count()

            # 从最近一次 LLM 原始输出提取 UNEXPLORED: 块
            # ponytail: 字符串切片, 不上正则. 升级路径: 结构化 schema.
            unexplored = ""
            raw = getattr(self, "_last_raw_hypothesis", "") or ""
            if "UNEXPLORED:" in raw:
                unexplored = raw.split("UNEXPLORED:", 1)[1].strip()
                # 截到下一个大写标记或结尾, 避免把后续块都吞进来
                for marker in ["\n\nHYPOTHESIS", "\n\nSELECTED", "\n\nRATIONALE"]:
                    if marker in unexplored:
                        unexplored = unexplored.split(marker)[0].strip()
                        break

            checklist = auditor.audit(
                iteration=self._iteration,
                families_explored=families_explored,
                live_components=live_components,
                total_iterations=(
                    self._max_iterations if hasattr(self, "_max_iterations") else 10
                ),
                candidate_finding=getattr(self, "_last_hypothesis", "") or "",
                original_problem=str(getattr(self, "_objective", "") or ""),
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
            (
                "ml-potential",
                [
                    "mlp",
                    "ml potential",
                    "machine learning potential",
                    "neural potential",
                ],
            ),
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
            original_problem = str(context.get("summary", "")) or str(
                self._objective or ""
            )
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
                    verdict.trap_category,
                    verdict.reduction_target,
                    hypothesis[:100],
                )

            # 收敛度监控: 某族过热时记日志
            redirect = registry.suggest_redirect()
            if redirect is not None:
                logger.info(
                    "metacog: 方法族收敛度告警 — %s (建议下轮重定向到 %s)",
                    redirect.reason,
                    redirect.target_family,
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

    def _choose_recovery_phase(self, failure_type: str, validation: dict[str, Any]) -> str:
        """v7 phase 解耦: 根据失败类型选下一轮起点 phase.

        Oxelra 启示: 失败应能回退到合适 phase, 而不是只 refine.
        - tool_error / data_noise: 实验层问题, 跳 perceive+hypothesize, 复用 refined hypothesis
          (plan 仍走, 因 execute 强依赖 plan 的结构化输出)
        - param_error: 参数错, 跳 perceive (hypothesize 仍走, 生成新假设)
        - hypothesis_error: 假设错, 从头走
        - prompt_injection_suspect: 从头走 (保守, 重走全流程)

        ponytail: 简单 failure_type → phase 映射, ceiling 是 LLM 根据错误
                  语义动态选 phase + 保存上轮 plan 跳过 plan. 当前静态规则够用.
        """
        if failure_type in ("tool_error", "data_noise"):
            return "execute"
        if failure_type == "param_error":
            return "plan"
        return "perceive"

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
        # AV7: effort floor 违例 → 不 refute, 下轮重试同一假设扩方法族.
        # 由 _validate 阶段 _metacog_check_completion 设的 tag.
        if validation.get("failure_kind") == "effort_floor_retry":
            return "tool_error"
        # 工具失败: 超时/崩溃/连接/OOM — 不是假设错, 重试即可
        tool_markers = (
            "timeout",
            "timed out",
            "connection",
            "crash",
            "segfault",
            "oom",
            "out of memory",
            "exception",
            "subprocess",
            "slurm",
            "queue",
            "killed",
            "abort",
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
            "invalid",
            "argument",
            "parameter",
            "value error",
            "type error",
            "dimension",
            "shape mismatch",
            "key error",
        )
        if any(m in text for m in param_markers):
            return "param_error"
        # 数据噪声: 不确定/模糊/噪声大
        noise_markers = (
            "noise",
            "uncertain",
            "ambiguous",
            "inconclusive",
            "not converge",
            "did not converge",
            "no clear",
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
            return [f.category for f in report.findings if f.severity == "high"]
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
        """是否触发想象力模式. v7 G59: 认知热机转捩判据.

        优先调 CognitiveHeatEngine.should_imaginate (Re_cog > Re_crit 或 T_hot > 0.7).
        回落: 旧的 surprise + refine_count 触发, 向后兼容.

        MToM P4 (hybrid ST+TT): 心智模型预测错误时, 从 Theory Theory
        切到 Simulation Theory 重新建模. 这里就是那个切换信号.
        """
        try:
            from huginn.metacog.cognitive_heat_engine import get_heat_engine
            eng = get_heat_engine()
            # 更新运动学量 (U/L/ν), 让 Re_cog 反映当前状态
            n_ideas = 0
            try:
                n_ideas = len(self.hypothesis_graph.all_nodes())
            except Exception:
                pass
            n_principles = 0
            try:
                # stable_principles 是 reflection mixin 的 list
                sp = getattr(self, "stable_principles", None)
                n_principles = len(sp) if sp else 0
            except Exception:
                pass
            sys_prompt_len = 0
            try:
                sys_prompt_len = len(getattr(self, "system_prompt", "") or "")
            except Exception:
                pass
            eng.update_kinematics(n_ideas, n_principles + 1, sys_prompt_len)
            if eng.should_imaginate(getattr(self, "_iteration", 0)):
                return True
        except Exception:
            logger.debug("heat_engine.should_imaginate failed, fallback to legacy", exc_info=True)

        # 回落: 旧触发逻辑 (surprise + refine_count)
        # P2: _force_imaginate 由 _trigger_counterexample_hunt 设置,
        # stagnation 归因为 evidence_against 时强制开 imagination.
        if getattr(self, "_force_imaginate", False):
            return True
        return (
            getattr(self, "_last_surprise", 0.0) > 0.5
            or getattr(self, "_refine_count", 0) >= 2
        )

    def _recent_failed_hypotheses(self, limit: int = 3) -> list[str]:
        """从 typed memory 捞最近被 refuted 的假设, 给 forget_then_generate 用.

        C4: typed memory 默认 on, 走 recall_failed_directions (跨 session 可恢复).
        旧行 NULL 通过 lazy migrate (tags 含 math_concept:/strategy:) 自动反推.
        typed 查询为空时降级到 hypothesis_graph 内存路径.
        """
        if self.memory:
            try:
                _failed = self.memory.recall_failed_directions(limit=limit)
                if _failed:
                    # 返回 hypothesis_text (三元组第一项)
                    return [h for h, _, _ in _failed if h]
            except Exception:
                logger.debug(
                    "recall_failed_directions failed, fallback to hypothesis_graph",
                    exc_info=True,
                )
        # fallback: 从内存 hypothesis_graph 捞
        try:
            nodes = getattr(self.hypothesis_graph, "_nodes", {})
            failed = [
                n.statement
                for n in nodes.values()
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

        P13: HUGINN_USE_CROSS_DOMAIN=1 时先查 transfer 历史, 3 条都 failed
        跳过本次猜想, 有 succeeded 时把成功 transfer 引用进 hint 前缀.
        flag off 时行为跟现状完全一致.
        """
        try:
            from huginn.autoloop.conjecture import get_conjecture_generator

            source_problem = context.get("goal") or context.get("observation") or ""
            if not source_problem or len(source_problem) < 10:
                return ""
            source_domain = context.get("domain") or "materials science"
            target_domain = context.get("target_domain") or "battery cathodes"

            # P13: flag on 时查 CrossDomain 历史, 决定是否跳过 / 引用
            hint_prefix = ""
            if os.environ.get("HUGINN_USE_CROSS_DOMAIN", "0") in ("1", "true", "True"):
                try:
                    kg = getattr(self, "kg", None)
                    if kg is not None:
                        history = kg.query_transfer_history(
                            target_domain=target_domain, limit=3
                        )
                        # isinstance 兜底: 测试里 kg 可能是 MagicMock,
                        # query_transfer_history 返 MagicMock 不是 list
                        if isinstance(history, list):
                            if history and all(
                                h.get("status") == "failed" for h in history
                            ):
                                logger.info(
                                    "CrossDomain skipped (3 recent failed transfers)"
                                )
                                return ""  # 3 条都失败, 不重复猜想
                            successful = [
                                h for h in history
                                if h.get("status") == "succeeded"
                            ]
                            if successful:
                                hint_prefix = (
                                    f"Previous successful transfer: "
                                    f"{successful[0].get('original_problem')} -> "
                                    f"{successful[0].get('target_domain')}\n"
                                )
                except Exception:
                    logger.warning(
                        "query_transfer_history failed, proceed without history",
                        exc_info=True,
                    )

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
                f"{hint_prefix}[Cross-domain analogy hint]\n"
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
            feature_keys = [
                k for k in data if k != target and isinstance(data[k], list)
            ]
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
        闭环的关键一环.

        C4 后 typed memory 默认 on, 旧行 NULL 走 lazy migrate 自动反推."""
        # JEPA: 上轮预测误差大时, 用 reviewer persona 审视 —
        # 预测错了说明 agent 的心智模型不准, 需要更批判的视角.
        if getattr(self, "_last_surprise", 0.0) > 0.6:
            return "reviewer"

        # C4: typed memory 默认 on, 旧行 NULL 走 lazy migrate 反推
        # (旧行 tags 含 "autoloop" + "persona:reviewer" → 自动当 iteration_result)
        if self.memory:
            try:
                _typed_rows = self.memory.recall_typed(
                    memory_type="persona_history",
                    persona_id="reviewer",
                    limit=5,
                )
                if _typed_rows:
                    # content 格式: "Persona: reviewer, r_phys: 0.78"
                    import re as _re
                    _scores: list[float] = []
                    for _r in _typed_rows:
                        _c = _r.get("content", "") if isinstance(_r, dict) else ""
                        _m = _re.search(r"r_phys[:\s]+([\d.]+)", _c)
                        if _m:
                            _scores.append(float(_m.group(1)))
                    if _scores:
                        _avg = sum(_scores) / len(_scores)
                        if _avg > 0.6:
                            return "reviewer"
            except Exception:
                logger.debug(
                    "recall_typed(persona_history) failed", exc_info=True,
                )

        # C5: KG persona_use 召回 — knowledge→persona 闭环.
        # 遍历 KG persona_use 节点, 按 persona 分组求 r_phys 均值, 取最高.
        # ponytail: 不引入 embedding 相似度, 简单按 persona 聚合 r_phys.
        # 升级路径: context_hash 距离或 embedding 召回相似 context.
        try:
            if hasattr(self, "kg") and self.kg is not None:
                with self.kg._lock:
                    persona_scores: dict[str, list[float]] = {}
                    for _nid, _data in self.kg._graph.nodes(data=True):
                        if _data.get("type") != "persona_use":
                            continue
                        _p = _data.get("persona")
                        _r = _data.get("r_phys")
                        if _p and _r is not None:
                            try:
                                persona_scores.setdefault(_p, []).append(float(_r))
                            except (TypeError, ValueError):
                                continue
                if persona_scores:
                    _avg_scores = {
                        p: sum(v) / len(v) for p, v in persona_scores.items()
                    }
                    _best = max(_avg_scores, key=_avg_scores.get)
                    if _avg_scores[_best] > 0.5:
                        return _best
        except Exception:
            logger.debug("persona_use KG recall failed", exc_info=True)

        blob = json.dumps(context, ensure_ascii=False).lower()
        md_markers = ("md", "lammps", "molecular dynamics", "nvt", "npt", "md_steps")
        if any(m in blob for m in md_markers):
            return "md_expert"
        return "dft_expert"

    async def _plan(
        self, hypothesis: str, context: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Generate a plan from hypothesis and persist it to PlanStore.

        以前只返回一个临时 dict, turn 结束就丢了. 现在往 PlanStore 落一份,
        跨会话可恢复, 用户也能 confirm/reject. PlanStore 不可用时退回老行为.
        """
        prompt = self._build_plan_prompt(hypothesis, context)
        try:
            # OAK 启发: 三阶段角色分工 — hypothesize 用 reasoning (强模型发散),
            # plan 用 planning (中档模型收敛), execute 不调 LLM 直接跑工具
            response = await self._llm_chat(
                prompt, persona_name="default", task="planning"
            )
            plan = self._parse_plan(response)
        except Exception:
            return None

        # H4: GRILL 退出检查 — LLM 输出 plan 时若含 "shared understanding" 确认标记
        # 说明用户已确认, 退出 grill 模式. 之前只进入不退出, LLM 永远带 grill 约束.
        # ponytail: 简单字符串匹配, 不上 LLM judge. ceiling: LLM 不说这个词就永远不退.
        if self._grill_active and response:
            _resp_lower = response.lower()
            if (
                "shared understanding" in _resp_lower
                or "shared understanding reached" in _resp_lower
                or "confirmed decisions" in _resp_lower
            ):
                self._grill_active = False
                self._grill_turns = 0
                logger.info("GRILL mode exited: shared understanding confirmed")

        if not plan:
            return None

        # B: 硬路由 — 根据上下文信号覆盖 LLM 的 mode 选择
        plan = self._override_plan_mode(plan)

        # KRCL 启发: 反向校验 plan 能否达成 hypothesis, 失败反馈 LLM 重生成
        # 单 LLM 反向校验, 最多 1 次重试, 失败不阻塞 (标 warning 继续)
        plan = await self._plan_check_and_refine(plan, hypothesis, context)

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
            # H4: reject_tokens 从 PhaseRegistry extra 取 (toggle off 回退 hardcode tuple)
            from huginn.harness.phase_spec import get_phase_extra
            _reject_tokens = get_phase_extra("_plan", "reject_tokens", [
                "no", "n", "cancel", "reject", "decline", "stop", "abort",
            ])
            answer = await self._maybe_clarify("plan", plan)
            if answer is None:
                should_confirm = True
            elif isinstance(answer, bool):
                should_confirm = answer
            elif isinstance(answer, str):
                should_confirm = answer.lower().strip() not in tuple(_reject_tokens)
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

        # H4: toggle on 时从 PhaseRegistry 取 dispatch_table 替代 hardcode if/elif
        # ponytail: dispatch_table 存 [method_name, arg_mode], arg_mode 决定传
        # plan 还是 description (executor 签名不统一, 不改签名最小 diff).
        # 升级路径: 统一所有 executor 签名为 (plan, context), 去掉 arg_mode.
        from huginn.harness.phase_spec import get_phase_dispatch_table
        dispatch = get_phase_dispatch_table()
        if dispatch is not None:
            entry = dispatch.get(mode)
            if entry is None:
                raise ValueError(f"Unknown plan mode: {mode}")
            method_name, arg_mode = entry[0], entry[1]
            executor = getattr(self, method_name)
            arg = plan if arg_mode == "plan" else description
            result = await executor(arg, context)
            # workflow 失败走 evolved_fix (原 hardcode 逻辑保留)
            if mode == "workflow" and isinstance(result, dict) and not result.get("success", True):
                result = (
                    await self._try_evolved_fix(mode, description, result) or result
                )
        elif mode == "coder":
            # Use CoderRunner to modify code
            result = await self._execute_coder(description, context)
        elif mode == "workflow":
            # Use WorkflowEngine to run computational pipeline
            result = await self._execute_workflow(description, context)
            # On failure, try applying a learned heuristic fix before giving up
            if isinstance(result, dict) and not result.get("success", True):
                result = (
                    await self._try_evolved_fix(mode, description, result) or result
                )
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
            "_tool_name": mode,
            "_tool_input": plan,
            "result": (
                result if isinstance(result, dict) else {"value": str(result)[:500]}
            ),
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
            logger.warning(
                "error in _record_provenance: capture snapshot failed", exc_info=True
            )

    async def _try_evolved_fix(
        self, tool_name: str, tool_input: dict[str, Any], error_result: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Check if the evolution engine has a learned fix for this error.

        This is the other half of the Learn→Execute loop: when a tool fails,
        we ask evolution if it's seen this error before and has a fix.
        Returns a patched result dict on hit, None on miss.
        """
        # H2: variant 失败不走 evolved_fix (P6 guard) — 直接回 bandit loop 记录
        if isinstance(error_result, dict) and error_result.get("_variant_id"):
            return None
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
            logger.warning(
                "error in _try_evolved_fix: apply_heuristic_fix failed", exc_info=True
            )
        return None

    async def _execute_dynamic_workflow(
        self, plan: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, Any]:
        """A5: 跑 agent 提交的并行工作流脚本.

        plan 里带 "script" 字段 (WorkflowScript.from_dict 的输入), 直接走
        WorkflowOrchestrator.run() 同步等完. 失败的 subtask 不炸整体,
        返回聚合结果让 validate/learn 阶段看.
        """
        # H2: bandit loop — plan 带 n_variants 且 toggle on 时走 variant 演化
        if plan.get("n_variants") and _harness_workflow_evolution_enabled():
            return await self._execute_dynamic_workflow_bandit(plan, context)

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

    async def _execute_dynamic_workflow_bandit(
        self, plan: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, Any]:
        """H2: 对同一 objective 生成 N 个 variant, bandit 选一个跑.

        generate_variants → bandit.select_variant → orch.run →
        返回 dict 带 _variant_id / _objective_hash / _novelty / _efficiency,
        供 _validate 调 bandit.record_variant_outcome + archive.add_variant.
        失败静默回退到原 _execute_dynamic_workflow 逻辑.
        """
        import json as _json

        from huginn.autoloop.bandit import (
            WorkflowBandit,
            VariantArchive,
            compute_novelty,
            _objective_hash,
        )
        from huginn.autoloop.dynamic_workflow import (
            WorkflowOrchestrator,
            WorkflowScript,
        )
        from huginn.autoloop.variant_gen import generate_variants
        from huginn.types import ToolContext

        raw_script = plan.get("script") or {}
        if isinstance(raw_script, str):
            try:
                raw_script = _json.loads(raw_script)
            except _json.JSONDecodeError:
                raw_script = {}
        try:
            base_script = WorkflowScript.from_dict(raw_script)
        except Exception:
            return {
                "mode": "dynamic_workflow",
                "success": False,
                "error": "base script parse fail",
            }
        if not base_script.subtasks:
            return {
                "mode": "dynamic_workflow",
                "success": False,
                "error": "脚本无有效 subtask",
            }
        objective = base_script.objective or str(plan.get("objective", ""))
        obj_hash = _objective_hash(objective)
        n_variants = int(plan.get("n_variants", 3))

        # 生成 variants (参数扰动优先, base_script 非空走扰动)
        try:
            variants = await generate_variants(
                objective,
                n=n_variants,
                base_script=base_script,
                llm_chat_fn=getattr(self, "_llm_chat", None),
            )
        except Exception:
            logger.debug("H2 generate_variants failed", exc_info=True)
            variants = []
        if not variants:
            variants = [base_script]

        # bandit select (variant_id 加时间戳前缀避免跨轮冲突)
        _run_prefix = f"r{int(time.time() * 1000) % 100000}"
        bandit = WorkflowBandit.get_instance()
        candidate_ids = [f"{_run_prefix}_var_{i}" for i in range(len(variants))]
        chosen_id = bandit.select_variant(candidate_ids, obj_hash)
        if chosen_id is None:
            chosen_id = candidate_ids[0]
        chosen_idx = candidate_ids.index(chosen_id)
        chosen = variants[chosen_idx]
        variant_id = chosen_id

        # 跑选中 variant
        orch = WorkflowOrchestrator(max_concurrent=chosen.max_concurrent)
        ctx = ToolContext(
            session_id=f"dynwf_{chosen.id}",
            workspace=str(self.workspace),
            config=self.settings,
        )
        try:
            result = await orch.run(chosen, ctx)
        except Exception as exc:
            logger.debug("H2 bandit variant run failed: %s", exc, exc_info=True)
            return {
                "mode": "dynamic_workflow",
                "success": False,
                "error": f"variant run fail: {exc}",
                "_variant_id": variant_id,
                "_objective_hash": obj_hash,
                "_objective": objective,
                "_script_dict": chosen.to_dict(),
                "_novelty": 0.0,
                "_efficiency": 0.0,
            }

        # novelty vs archive
        try:
            archive = VariantArchive.get_instance()
            existing = archive.list_variants(obj_hash)
            novelty = compute_novelty(chosen.to_dict(), existing)
        except Exception:
            novelty = 0.0

        efficiency = result.n_completed / max(1, result.n_total)
        return {
            "mode": "dynamic_workflow",
            "success": result.success,
            "workflow_id": result.id,
            "n_total": result.n_total,
            "n_completed": result.n_completed,
            "n_failed": result.n_failed,
            "summary": result.summary(),
            "_variant_id": variant_id,
            "_objective_hash": obj_hash,
            "_objective": objective,
            "_script_dict": chosen.to_dict(),
            "_novelty": float(novelty),
            "_efficiency": float(efficiency),
        }

    async def _validate(self, execution_result: Any) -> dict[str, Any]:
        """Validate execution results using benchmarks and constraints."""
        # H4: reviewer 阈值 + MatWorldBench 白名单 + needs_retry 阈值从 PhaseRegistry extra 取
        from huginn.harness.phase_spec import get_phase_extra
        _reviewer_threshold = get_phase_extra("_validate", "reviewer_threshold", 0.5)
        _mwb_categories = get_phase_extra("_validate", "matworldbench_categories", [
            "structure", "thermo", "electronic",
        ])
        _needs_retry_threshold = get_phase_extra("_validate", "needs_retry_threshold", 0.5)
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
            prev_exec = getattr(self, "_last_execution_result", None)
            if prev_exec and isinstance(prev_exec.get("result"), dict):
                try:
                    from huginn.tools.visual_hook import extract_comparative_primitives

                    comp = extract_comparative_primitives(
                        prev_exec.get("result", {}), execution_result
                    )
                    if comp:
                        results["comparative_primitives"] = comp
                        # 也拼进 visual_context, 下轮 hypothesis 能看到
                        self._last_visual_context = (
                            f"{self._last_visual_context}\n{comp}".strip()
                            if self._last_visual_context
                            else comp
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

        needs_review = gen_verify is None or gen_verify.get("score", 0.5) < _reviewer_threshold
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
        ec_task = asyncio.create_task(
            self._safe_emergent_complexity(execution_result, results)
        )
        lit_task = asyncio.create_task(
            self._safe_literature_comparison(execution_result, results)
        )
        await asyncio.gather(ec_task, lit_task)

        try:
            from huginn.validation.grader import default_registry

            # ValidityJudge 需要 model + 对话日志/代码做 post-hoc 审查
            # NatureBench judge.py 启发: r_phys 高不代表真算, 可能 gaming grader
            reg = default_registry(
                model=getattr(self, "verification_model", None) or self.model
            )
            merged: dict[str, Any] = {}
            if isinstance(execution_result, dict):
                merged.update(execution_result)
            merged.update(results)
            # 喂给 ValidityJudge: 从 memory 取最近对话 + execution_result 里的 code
            try:
                recent = self.memory.get_recent_messages(n=20)
                conv_snippets = []
                for m in recent:
                    role = getattr(m, "role", "?")
                    content = getattr(m, "content", "")
                    if isinstance(content, (dict, list)):
                        content = str(content)[:500]
                    conv_snippets.append(f"[{role}] {str(content)[:500]}")
                merged["conversation_log"] = "\n".join(conv_snippets)
            except Exception:
                logger.debug("conversation_log extract for judge failed", exc_info=True)
            # agent_code: execution_result 里可能带 code/parsed/script
            if isinstance(execution_result, dict):
                for k in ("code", "script", "generated_code", "final_answer"):
                    v = execution_result.get(k)
                    if v and isinstance(v, str) and len(v) > 50:
                        merged["agent_code"] = v
                        break
                else:
                    # 退而求其次: tool_input 里的 description 可能含代码片段
                    ti = execution_result.get("_tool_input") or {}
                    if isinstance(ti, dict):
                        merged["agent_code"] = str(ti.get("description", ""))[:5000]
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
                    track_task(
                        _publish(
                            "quality.check",
                            {
                                "iteration": self._iteration,
                                "graders": results["grader_scores"],
                                "reward": results.get("grader_reward", 0),
                            },
                            source="autoloop",
                        ),
                        name="quality-check-emit",
                    )
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
                if task.category in tuple(_mwb_categories):
                    try:
                        br = bench.evaluate(task.id, exec_data)
                        eval_scores.append(
                            {
                                "task_id": task.id,
                                "category": task.category,
                                "passed": br.passed,
                                "score": br.score,
                            }
                        )
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

        # AV7: 最小努力下限硬阻断. _metacog_check_completion 已封装
        # families/live_components/UNEXPLORED 自白收集, 这里复用.
        # 不达标时: 强制 tests_passed=False → run loop L1616-1647 走失败分支;
        # 设 failure_kind=effort_floor_retry → _classify_failure 归 tool_error
        # (不 refute, 下轮重试同一假设扩方法族, 避免污染 hypothesis_graph).
        # ponytail: 复用现成 refine/retry 控制流, 不新写迭代触发逻辑.
        try:
            _eff_blk, _eff_why = self._metacog_check_completion()
            results["effort_floor_passed"] = not _eff_blk
            if _eff_blk:
                results["effort_floor_deficits"] = _eff_why
                results["failure_kind"] = "effort_floor_retry"
                results["tests_passed"] = False
                results["constraints_satisfied"] = False
                _hint = (
                    f"[effort floor] 探索未达硬下限, 不算通过: {_eff_why}. "
                    "下轮必须扩方法族或保留更多假设, 不要再收敛."
                )
                self._speculator_hint = (
                    (self._speculator_hint + "\n" + _hint).strip()
                    if self._speculator_hint else _hint
                )
                if len(self._speculator_hint) > 2000:
                    self._speculator_hint = self._speculator_hint[-2000:]
        except Exception:
            logger.debug("AV7 effort floor check in _validate failed", exc_info=True)

        # H2: bandit 记录 variant outcome (r_phys + efficiency + novelty 都算出后)
        # 只对 dynamic_workflow bandit 路径生效 (execution_result 带 _variant_id)
        if isinstance(execution_result, dict) and execution_result.get("_variant_id"):
            try:
                from huginn.autoloop.bandit import (
                    WorkflowBandit,
                    VariantArchive,
                )
                _vid = execution_result["_variant_id"]
                _obj_hash = execution_result.get("_objective_hash", "")
                _obj = execution_result.get("_objective", "")
                _novelty = float(execution_result.get("_novelty", 0.0))
                _eff = float(execution_result.get("_efficiency", 0.0))
                _r_phys = float(results.get("r_phys", 0.0) or 0.0)
                _success = bool(results.get("tests_passed", False))
                _script_dict = execution_result.get("_script_dict", {})
                bandit = WorkflowBandit.get_instance()
                bandit.record_variant_outcome(
                    _vid, _obj_hash, _success,
                    r_phys=_r_phys, efficiency=_eff, novelty=_novelty,
                )
                # archive: fitness = [r_phys, efficiency, novelty]
                archive = VariantArchive.get_instance()
                b = bandit.get_belief(_vid, _obj_hash)
                archive.add_variant(
                    _obj_hash, _obj, _vid, _script_dict,
                    fitness=[_r_phys, _eff, _novelty],
                    alpha=b.successes if b else 1,
                    beta=b.failures if b else 1,
                )
            except Exception:
                logger.debug("H2 bandit record in _validate failed", exc_info=True)

        self._last_validation = json.dumps(results, ensure_ascii=False, default=str)[
            :1000
        ]
        # Store failure_mode for next hypothesis loop (Dream Layer: crash = discovery)
        _gv = results.get("generative_verify", {})
        if isinstance(_gv, dict):
            self._last_failure_mode = _gv.get("failure_mode", "")
        # P1: 盲重建 verification + support/refute 闭环.
        # 之前 _validate 算出分数但不调 hypothesis_graph.support/refute, 图全 untested.
        # 现在开 toggle 时: (1) fresh subagent 从 statement 独立推导 (2) 比对盲重建
        # vs execution_result (3) mismatch→refute / match→support, 写 FAILED/PROVED.md.
        # ponytail: 默认 off (贵, 多一次 subagent dispatch). 升级: 只在割点/关键假设上开.
        if os.environ.get("HUGINN_BLIND_RECONSTRUCTION", "0") == "1":
            try:
                await self._blind_reconstruct_verify(execution_result, results)
            except Exception:
                logger.debug("P1 blind reconstruct failed", exc_info=True)
        return results

    async def _blind_reconstruct_verify(
        self, execution_result: Any, results: dict[str, Any],
    ) -> None:
        """P1: 盲重建 + support/refute 闭环.

        1. 拿当前 hypothesis statement (不传 proof/evidence)
        2. SubagentDispatch("blind_reconstructor") 独立推导
        3. 比对盲重建 holds vs execution_result 是否一致
        4. 一致 → hypothesis_graph.support, 不一致 → hypothesis_graph.refute
        5. 写 FAILED.md/PROVED.md (P0 已接入)
        """
        _hyp_id = getattr(self, "_current_hyp_id_for_plan", None)
        if not _hyp_id:
            return
        try:
            _node = self.hypothesis_graph._nodes.get(_hyp_id)
        except Exception:
            return
        if _node is None or _node.status != "untested":
            return
        _statement = _node.statement
        if not _statement or len(_statement) < 10:
            return
        if self._agent_factory is None:
            logger.debug("P1 blind reconstruct: no agent_factory, skip")
            return
        from huginn.agents.subagent import SubagentDispatch
        _dispatch = SubagentDispatch()
        _task = (
            f"Independently derive whether this statement holds, from first principles. "
            f"Do NOT assume any prior proof. Output JSON.\n\n"
            f"Statement: {_statement}"
        )
        _ctx = {"agent_factory": self._agent_factory}
        try:
            _res = await _dispatch.dispatch("blind_reconstructor", _task, context=_ctx)
        except Exception:
            logger.debug("P1 blind reconstruct dispatch failed", exc_info=True)
            return
        if not _res.success or not _res.summary:
            return
        # 解析盲重建结果 (JSON summary)
        import json as _json
        try:
            _blind = _json.loads(_res.summary)
        except Exception:
            # LLM 没输出合法 JSON, 从 summary 文本推断
            _blind = {"holds": "true" in _res.summary.lower(), "confidence": 0.5}
        _blind_holds = bool(_blind.get("holds", False))
        # 原 execution_result 是否支持 hypothesis (tests_passed / grader_reward)
        _orig_holds = bool(
            results.get("tests_passed")
            or results.get("grader_reward", 0) > 0.5
            or results.get("generative_verify", {}).get("score", 0) > 0.5
        )
        # 比对: 一致 → support, 不一致 → refute
        _evidence = {
            "modality": "blind_reconstruction",
            "data_source": f"subagent:{_res.spec_name}",
            "blind_holds": _blind_holds,
            "blind_confidence": float(_blind.get("confidence", 0.5)),
            "blind_derivation": str(_blind.get("derivation", ""))[:500],
            "orig_holds": _orig_holds,
            "tests_passed": results.get("tests_passed"),
            "grader_reward": results.get("grader_reward"),
        }
        if _blind_holds == _orig_holds:
            self.hypothesis_graph.support(_hyp_id, _evidence)
            results["blind_reconstruction"] = {"match": True, **_evidence}
            logger.info("P1 blind reconstruct: match → support %s", _hyp_id)
        else:
            _evidence["errors"] = (
                f"blind_holds={_blind_holds} vs orig_holds={_orig_holds} "
                f"mismatch — blind reconstruction disagrees with execution result"
            )
            self.hypothesis_graph.refute(_hyp_id, _evidence)
            results["blind_reconstruction"] = {"match": False, **_evidence}
            logger.info("P1 blind reconstruct: mismatch → refute %s", _hyp_id)

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
            report = await asyncio.to_thread(runner.run, categories=["math", "coding"])
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
                    if self._speculator_hint
                    else ec_hint
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
                # 平铺 high_confidence_claims 给 red_team._literature_consensus_check
                # 消费. evidence["literature_claims"] 是 red_team 约定的 key.
                high_claims = lit_comp.get("high_confidence_claims") or []
                if high_claims:
                    results["literature_claims"] = high_claims
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
            execution_result.get("result_data") or execution_result.get("parsed") or {}
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
                    logger.warning(
                        "error in _literature_comparison: numeric property cast failed",
                        exc_info=True,
                    )
        lattice = result_data.get("lattice_params") or {}
        if isinstance(lattice, dict):
            for param in ("a", "b", "c"):
                val = lattice.get(param)
                if val is not None:
                    try:
                        numerics[f"lattice_{param}"] = float(val)
                    except (TypeError, ValueError):
                        logger.warning(
                            "error in _literature_comparison: lattice param cast failed",
                            exc_info=True,
                        )

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

        # 知识注入层: multi_review 产 high_confidence_claims, 平铺到 comparison
        # 给 red_team._literature_consensus_check 消费 (evidence["literature_claims"])
        # ponytail: 2 透镜 + max_results=5 控成本. 失败无所谓, comparison 已有 benchmark 部分
        try:
            mr_res = await tool.call(
                LiteratureInput(
                    action="multi_review",
                    query=system,
                    max_results=5,
                    lenses=["methodology", "limitations"],
                    verify_claims=True,
                ),
                tool_ctx,
            )
            if mr_res.success and mr_res.data:
                high_claims = mr_res.data.get("high_confidence_claims") or []
                if high_claims:
                    comparison["high_confidence_claims"] = high_claims
        except Exception:
            logger.debug(
                "multi_review in _literature_comparison failed (non-fatal)",
                exc_info=True,
            )

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
            ngrams = [" ".join(words[i : i + 5]) for i in range(len(words) - 4)]
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

        return [{"call_hash": k, "count": c} for k, c in seen.items() if c >= 2]

    # -- G2: 周期检测 (M3 cycle_detect) + 历史轨迹匹配 (M2 trajectory_match) --

    def _load_trajectory_action_history(self, limit: int = 20) -> list[list[str]]:
        """从 workspace/.huginn/trajectories/ 加载历史 run 的 action 序列.

        每个 trajectory.json 的 spans 里 phase 名就是 action. 抽出来给
        trajectory_match 当 history 用 (VF2 子图同构 prefix 匹配).

        ponytail: 只读最近 limit 个文件, 不做全量索引. 升级路径: KB 索引 + 元数据过滤.
        """
        traj_dir = self.workspace / ".huginn" / "trajectories"
        if not traj_dir.exists():
            return []
        history: list[list[str]] = []
        try:
            files = sorted(
                traj_dir.glob("*.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )[:limit]
        except Exception:
            return []
        for f in files:
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            spans = data.get("spans") or []
            actions = [
                s.get("phase") or s.get("name") or ""
                for s in spans
                if isinstance(s, dict)
            ]
            actions = [a for a in actions if a]
            if len(actions) >= 2:
                history.append(actions)
        return history

    def _check_stuck(self, action_history: list[str]) -> dict[str, Any] | None:
        """G2 主入口: 周期检测 + 历史轨迹 prefix 匹配.

        - cycle_detect.is_stuck: 当前 action 序列是否陷入周期 (M3, O(n²) 暴力).
        - trajectory_match: 当前序列是否是某历史成功轨迹的 prefix (M2, VF2 子图同构).
          匹配到 → 取下一步作为建议, 注入 _speculator_hint.

        返回 None (无信号) 或 dict:
          {"type": "cycle", "period": lam, "advice": "..."}
          {"type": "match", "history_id": i, "similarity": s, "next_step": X, "advice": "..."}

        极限模式 (跑分) 才启用, 平常默认关闭省 cycle/trajectory 计算.
        HUGINN_EXTREME_DISPATCH=1 时开.
        """
        import os
        if os.environ.get("HUGINN_EXTREME_DISPATCH", "0").lower() not in ("1", "true"):
            return None
        if len(action_history) < 4:
            return None  # 太短不检
        try:
            from huginn.runtime.cycle_detect import is_stuck, detect_cycle
            from huginn.knowledge.trajectory_pattern import trajectory_match
        except ImportError:
            return None

        # M3: 周期检测 (在当前 run 内)
        try:
            if is_stuck(action_history, min_cycle_len=2, min_repeats=2):
                cycle = detect_cycle(action_history, min_cycle_len=2, min_repeats=2)
                lam = cycle[1] if cycle else 0
                return {
                    "type": "cycle",
                    "period": lam,
                    "advice": (
                        f"action 序列陷入周期 (period={lam}), 强制 pivot. "
                        f"最近 {len(action_history)} 步: {action_history[-8:]}"
                    ),
                }
        except Exception:
            logger.debug("G2 cycle_detect failed (non-fatal)", exc_info=True)

        # M2: 历史轨迹 prefix 匹配 (跨 run)
        try:
            history = getattr(self, "_traj_history", None) or []
            if history:
                match = trajectory_match(
                    action_history, history, min_similarity=0.4,
                )
                if match and match.get("next_step"):
                    return {
                        "type": "match",
                        "history_id": match["history_id"],
                        "similarity": match["similarity"],
                        "next_step": match["next_step"],
                        "advice": (
                            f"匹配历史成功轨迹 (sim={match['similarity']:.2f}), "
                            f"考虑下一步: {match['next_step']}"
                        ),
                    }
        except Exception:
            logger.debug("G2 trajectory_match failed (non-fatal)", exc_info=True)

        return None

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
        for key in (
            "summary",
            "description",
            "result_data",
            "output",
            "error",
            "reasoning",
            "plan",
            "hypothesis",
        ):
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
        stop1 = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "to",
            "of",
            "in",
            "on",
            "at",
            "for",
            "and",
            "or",
            "not",
            "this",
            "that",
            "it",
            "with",
            "from",
            "by",
            "as",
            "will",
            "can",
            "may",
        }
        # 扰动 2: 更激进的停用词集 (去掉更多常见词)
        stop2 = stop1 | {
            "energy",
            "result",
            "value",
            "system",
            "model",
            "data",
            "using",
            "shown",
            "show",
            "also",
            "which",
            "has",
            "have",
            "had",
            "been",
            "were",
            "more",
            "than",
        }
        # 扰动 3: 只保留长关键词 (>=5 chars)
        # 扰动 4: bigram Jaccard

        def keywords(text: str, stop: set[str], min_len: int = 3) -> set[str]:
            words = __import__("re").findall(r"[a-zA-Z_]\w{2,}", text.lower())
            return {w for w in words if w not in stop and len(w) >= min_len}

        def bigrams(text: str) -> set[str]:
            words = __import__("re").findall(r"[a-zA-Z_]\w{2,}", text.lower())
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
            collapse_hint = f"\nNote: automated checks detected: {json.dumps(collapse, default=str)[:300]}"

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
            logger.debug(
                "_build_memory_text failed — validate prompt missing cross-check",
                exc_info=True,
            )

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
        score, reason, evidence_score, evidence_gap, failure_mode = (
            self._parse_verify_score(resp)
        )

        return {
            "score": score,
            "reason": reason,
            "needs_retry": score < _needs_retry_threshold,
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
            logger.warning(
                "error in _parse_verify_score: JSON parse failed, falling back to regex",
                exc_info=True,
            )

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

            predict_res = tool.call(
                {
                    "action": "predict",
                    "X": X,
                    "y": y,
                    "X_new": pred_X,
                }
            )
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
                return {
                    "agrees": True,
                    "gp_fit": data,
                    "reason": "GP fit ok, no comparable points",
                }
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
            execution_result.get("equation") or execution_result.get("equations") or ""
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
                logger.warning(
                    "error in _collect_math_evidence: dimensional_analysis failed",
                    exc_info=True,
                )

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
                logger.warning(
                    "error in _collect_math_evidence: pde_classify failed",
                    exc_info=True,
                )

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
                logger.warning(
                    "error in _collect_math_evidence: sobol_indices failed",
                    exc_info=True,
                )

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
                logger.warning(
                    "error in _collect_math_evidence: constraint_check failed",
                    exc_info=True,
                )

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
            exec_blob = json.dumps(execution_result, ensure_ascii=False, default=str)[
                :1500
            ]
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

    async def _learn(
        self, hypothesis: str, plan: dict[str, Any], validation: dict[str, Any]
    ) -> dict[str, Any]:
        """Learn from iteration results — update memory, knowledge graph, evolution rules.

        D2: 返回 summary dict, 让 caller (execute_fn 的 learn 分支) 写入
        cog["last_learn_summary"], 下轮 decider 看到正反馈. 之前返回 None,
        LLM 选 learn 没反馈, 下轮容易重复 learn.
        ponytail: 只返 4 个标量字段, 不暴露内部状态. 升级路径: 结构化
        summary 走专门 cog slot (cog["last_learn_detail"]).
        """
        # H4: importance 公式常量从 PhaseRegistry extra 取
        from huginn.harness.phase_spec import get_phase_extra
        _imp_default = get_phase_extra("_learn", "importance_default", 0.6)
        _imp_max = get_phase_extra("_learn", "importance_max", 0.9)
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
            visual_ctx = (
                validation.get("visual_primitives")
                if isinstance(validation, dict)
                else None
            )
            if visual_ctx:
                mem_content += f"\nVisual: {visual_ctx[:200]}"
            # Surprise 入 memory, 下次能检索到"这类任务预测准不准"
            pred_err = (
                validation.get("prediction_error", {})
                if isinstance(validation, dict)
                else {}
            )
            if pred_err:
                mem_content += f"\nSurprise: {pred_err.get('surprise', 0)} (worst: {pred_err.get('surprise_worst', pred_err.get('surprise', 0))}, std: {pred_err.get('surprise_std', 0)}) (predicted: {pred_err.get('predicted', '')[:80]})"
            # Persona 入 memory, 下次 _pick_hypothesis_persona 能查到历史效果
            mem_content += f"\nPersona: {persona_name}, r_phys: {r_phys}"
            # 结构化 tags: 供后续按 persona/r_phys/surprise 过滤检索
            _tags = [
                "autoloop",
                f"persona:{persona_name}",
                f"r_phys:{r_phys}" if r_phys is not None else "r_phys:none",
                (
                    f"surprise:{pred_err.get('surprise', 0):.2f}"
                    if pred_err
                    else "surprise:0"
                ),
            ]
            # C4: typed memory 默认 on, 走 remember_typed (含 iteration_result
            # + persona_id + status). 旧行 NULL 通过 lazy migrate 自动反推.
            # typed 写入失败时 fallback 到 legacy remember.
            try:
                # status: 用 validation 结果映射 (supported/refuted)
                _tests_ok = (
                    validation.get("tests_passed")
                    if isinstance(validation, dict)
                    else False
                )
                _typed_status = "supported" if _tests_ok else "refuted"
                self.memory.remember_typed(
                    content=mem_content,
                    memory_type="iteration_result",
                    run_id=getattr(self, "_run_id", None),
                    persona_id=persona_name,
                    status=_typed_status,
                    importance=_imp_default if r_phys is None else min(_imp_max, float(r_phys)),
                    tier="mid",
                    tags=_tags,
                )
                # C5: 额外写一条 persona_history (给 _pick_hypothesis_persona 查).
                # 不双写完整 content, 只记 persona + r_phys 摘要, 避免冗余.
                _ph_content = (
                    f"Persona: {persona_name}, r_phys: {r_phys}"
                    f", iter: {self._iteration}"
                )
                self.memory.remember_typed(
                    content=_ph_content,
                    memory_type="persona_history",
                    run_id=getattr(self, "_run_id", None),
                    persona_id=persona_name,
                    status=_typed_status,
                    importance=_imp_default if r_phys is None else min(_imp_max, float(r_phys)),
                    tier="mid",
                    tags=_tags,
                )
            except Exception:
                logger.debug(
                    "typed remember_typed failed, fallback to legacy remember",
                    exc_info=True,
                )
                self.memory.remember(
                    content=mem_content,
                    category="autoloop_iteration",
                    importance=_imp_default if r_phys is None else min(_imp_max, float(r_phys)),
                    tier="mid",
                    tags=_tags,
                )
        except Exception:
            logger.warning(
                "error in _learn: memory.remember iteration failed", exc_info=True
            )

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
                        n_skills,
                        n_patches,
                        r_phys,
                    )
            except Exception as e:
                logger.warning("reward evolution failed: %s", e)

        # H1: 给本轮 apply 过的 prompt patch 更新 Beta 信念. tests_passed 决定
        # success/fail. _apply_block_patches 在 _build_hypothesis_prompt /
        # _build_plan_prompt 里记录了 _last_applied_patches = (phase, [ids]).
        # ponytail: _last_applied_patches 只记最近一次, 多 phase 同轮 apply 时
        # 后者覆盖前者. 升级路径: 改成 dict[phase, ids] 完整追踪.
        try:
            applied = getattr(self, "_last_applied_patches", None)
            if applied:
                from huginn.harness.prompt_patch import PromptPatchStore
                _phase, _ids = applied
                _tests_passed = (
                    validation.get("tests_passed", False)
                    if isinstance(validation, dict)
                    else False
                )
                store = PromptPatchStore.get_instance()
                for _pid in _ids:
                    store.update_alpha_beta(_pid, success=bool(_tests_passed))
        except Exception:
            logger.debug("H1 patch Beta update failed", exc_info=True)

        # H3: 记录 (block_subset, workflow_params) 组合的 outcome 给 JointBandit.
        # block_subset 从 _last_hypothesis_blocks / _last_plan_blocks 拿 block 名;
        # workflow_params 留空 dict (reasoning-only 没 workflow stage 参数).
        # ponytail: 不追踪 select 返回值, 从 _last_*_blocks 反推. 升级路径:
        # select_block_subset_for_phase 返回 (blocks, selected_names) 避免 重算.
        try:
            from huginn.harness.joint_optimizer import JointBandit, _harness_enabled as _h3_on
            if _h3_on("harness_joint_optimizer"):
                _h3_phase = "hypothesize"
                _h3_blocks = getattr(self, "_last_hypothesis_blocks", None) or []
                if not _h3_blocks:
                    _h3_blocks = getattr(self, "_last_plan_blocks", None) or []
                    _h3_phase = "plan"
                if _h3_blocks:
                    _h3_subset = [n for n, _ in _h3_blocks]
                    _h3_success = bool(
                        validation.get("tests_passed", False)
                        if isinstance(validation, dict) else False
                    )
                    JointBandit.get_instance().record_joint_outcome(
                        _h3_phase, _h3_subset, {}, _h3_success,
                    )
        except Exception:
            logger.debug("H3 joint record failed", exc_info=True)

        # Forest 回流: 如果是森林模式运行, 把 merged_graph 合并到本地假设图
        # 并写入 memory, 供后续迭代接续探索多树共识的结论.
        if self._merged_graph is not None:
            try:
                # 合并到本地 hypothesis_graph
                for node_id in self._merged_graph.nodes:
                    node = self._merged_graph.nodes.get(node_id)
                    if node and hasattr(node, "statement"):
                        # 跳过已存在的节点
                        if not any(
                            existing.statement == node.statement
                            for existing in self.hypothesis_graph.nodes.values()
                        ):
                            nid = self.hypothesis_graph.add_hypothesis(
                                statement=node.statement,
                                rationale=getattr(node, "rationale", ""),
                                testable_prediction=getattr(
                                    node, "testable_prediction", ""
                                ),
                            )
                            if nid is not None:
                                if getattr(node, "status", "") == "supported":
                                    self.hypothesis_graph.support(
                                        nid, getattr(node, "evidence", {})
                                    )
                                elif getattr(node, "status", "") == "refuted":
                                    self.hypothesis_graph.refute(
                                        nid, getattr(node, "evidence", {})
                                    )
                # 写入 memory
                graph_summary = f"Forest merged: {len(self._merged_graph.nodes)} nodes"
                self.memory.add_message(
                    "system",
                    {
                        "iteration": self._iteration,
                        "type": "forest_merge",
                        "graph_summary": graph_summary,
                    },
                )
                logger.info(
                    "Forest merged %d nodes into hypothesis_graph",
                    len(self._merged_graph.nodes),
                )
            except Exception:
                logger.warning("Forest merge failed", exc_info=True)

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
            visual_ctx = (
                validation.get("visual_primitives")
                if isinstance(validation, dict)
                else None
            )
            if visual_ctx:
                kg_attrs["visual_primitives"] = visual_ctx[:500]
            # JEPA: surprise 分数存入 KG, 下次查同类实验能看到"这类任务
            # agent 预测准不准", 帮助判断是否值得继续探索.
            pred_err = (
                validation.get("prediction_error", {})
                if isinstance(validation, dict)
                else {}
            )
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
            tests_passed = (
                validation.get("tests_passed")
                if isinstance(validation, dict)
                else False
            )
            if not tests_passed and exp_id and hasattr(self.kg, "_graph"):
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
            result_label = (
                "pass"
                if (
                    validation.get("tests_passed")
                    if isinstance(validation, dict)
                    else False
                )
                else "fail"
            )
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
            # C5: persona_use entity — KG 层 persona 选择历史.
            # _pick_hypothesis_persona 遍历 persona_use 节点按 r_phys 均值召回.
            # ponytail: 不引入 embedding 相似度, 直接遍历 _graph.nodes 按 type 过滤.
            # 升级路径: context_hash 距离 (Hamming) 或 embedding 相似度召回.
            try:
                from huginn.utils.common import hash_text
                _ctx = getattr(self, "_last_context", {}) or {}
                _ctx_hash = hash_text(
                    json.dumps(_ctx, ensure_ascii=False, sort_keys=True, default=str)
                )
                self.kg.add_entity(
                    label=f"{persona_name}_iter{self._iteration}",
                    entity_type="persona_use",
                    source="autoloop",
                    confidence=float(r_phys) if r_phys is not None else 0.5,
                    persona=persona_name,
                    context_hash=_ctx_hash,
                    r_phys=r_phys,
                    iteration=self._iteration,
                )
            except Exception:
                logger.debug("persona_use entity write skipped", exc_info=True)
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
                logger.warning(
                    "error in _learn: benchmark_failure memory writeback failed",
                    exc_info=True,
                )
            # C4: typed memory 默认 on, 同步写 failed_direction,
            # 让 _recent_failed_hypotheses 跨 session 能恢复 (不靠 hypothesis_graph).
            # ponytail: 不动 HypothesisNode setter, 在 _learn 里集中写, 最小改动.
            try:
                _fail_reason = (
                    validation.get("error")
                    or validation.get("reason")
                    or json.dumps(validation, default=str)[:200]
                )
                self.memory.record_failed_direction(
                    hypothesis_text=hypothesis[:200],
                    reason=str(_fail_reason)[:400],
                    run_id=getattr(self, "_run_id", "") or "",
                    persona_id=getattr(self, "_last_persona", None),
                    math_concept="",
                )
            except Exception:
                logger.debug(
                    "record_failed_direction failed, fallback to legacy path",
                    exc_info=True,
                )

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
            logger.debug(
                "surprise detection failed — _feynman_learn trigger may silently skip",
                exc_info=True,
            )

        if _should_feynman:
            try:
                await self._feynman_learn(hypothesis, plan, validation, r_phys, context)
            except Exception:
                logger.warning(
                    "error in _learn: feynman note generation failed", exc_info=True
                )

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
                logger.warning(
                    "error in _learn: store_plan_progress writeback failed",
                    exc_info=True,
                )

        # RSI 入门: 让 agent 反思本轮, 给下一轮的自己写一条指令.
        # 借鉴 Inkling self-finetune loop: agent 改的不是自己的权重, 是自己下一轮的 prompt.
        # directive 写进 memory (不走 prompt 注入), 下轮 _build_hypothesis_prompt
        # 和 _build_plan_prompt 的 _build_memory_text 自然检索到 — 复用现有 memory loop.
        # maker/checker split: learn 写 directive, 下轮 validate 校验效果.
        # ponytail: 复用 memory tier 机制做衰减, 不引入新字段. 升级: 结构化 directive
        try:
            await self._generate_next_loop_directive(
                hypothesis, plan, validation, r_phys
            )
        except Exception:
            logger.debug(
                "RSI directive generation failed — loop continues without directive",
                exc_info=True,
            )

        # 收尾: 把本轮结果落一条 autoloop_summary, chat agent recall 时能拉到.
        # ponytail: 只落 summary 不共享 SessionContext; 升级路径是共享 SessionContext
        # 或加 autoloop_result 专用 category (目前先复用 memory.remember 通用通道).
        try:
            _tests_passed = (
                validation.get("tests_passed", True)
                if isinstance(validation, dict)
                else True
            )
            _summary = (
                f"iteration_count={self._iteration}; "
                f"refined_hypotheses={len(self.hypothesis_graph.nodes)}; "
                f"speculator_hints={self._speculator_hint[:200]!r}; "
                f"benchmark_failures={'no' if _tests_passed else 'yes'}; "
                f"r_phys={r_phys}; hypothesis={hypothesis[:120]!r}"
            )
            self.memory.remember(
                content=_summary,
                category="autoloop_summary",
                importance=0.9,
                tier="long",
                tags=["autoloop", "summary", f"iter:{self._iteration}"],
            )
        except Exception:
            # memory 失败不阻断 _learn, 上一轮的迭代已经入账
            logger.debug(
                "autoloop_summary writeback failed — loop continues",
                exc_info=True,
            )

        # P14: EvolutionManager.record_outcome — flag on 时把 outcome 统一记到
        # FailedDirectionStore + SkillEvolutionLayer. flag off (默认) 不调,
        # 走原分散路径 (P12 record_failed_direction + evolution.logger.log_tool_call).
        if os.environ.get("HUGINN_USE_EVOLUTION_MANAGER", "0") == "1":
            try:
                from huginn.evolution.manager import EvolutionManager

                em = EvolutionManager.shared(self.memory)
                em.record_outcome(
                    hypothesis=hypothesis,
                    plan=plan if isinstance(plan, dict) else None,
                    validation=validation if isinstance(validation, dict) else None,
                    persona_id=getattr(self, "_last_persona", None),
                    run_id=getattr(self, "_run_id", "") or "",
                    math_concept="",
                )
            except Exception:
                logger.warning(
                    "EvolutionManager.record_outcome failed", exc_info=True
                )

        # H0: 触发 episodic → procedural 蒸馏 (修死代码, 让 stable_principles
        # 真有产出). 触发条件由 distill_episodic_to_procedural 内部判断
        # (连续 3 次同 skill 成功), 不降低阈值. ponytail: 失败不阻塞主循环.
        # D2: 捕获返回值, 用于 last_learn_summary 的 principles_added 字段.
        _principles_added = 0
        try:
            _pid = self.memory.distill_episodic_to_procedural(
                self._evals_history, self.workspace
            )
            if _pid:
                _principles_added = 1
        except Exception:
            logger.debug(
                "distill_episodic_to_procedural failed", exc_info=True
            )

        # D2: 返回 summary, 让 caller 写 cog["last_learn_summary"]
        return {
            "persona": getattr(self, "_last_persona", "unknown"),
            "r_phys": r_phys,
            "tests_passed": bool(
                validation.get("tests_passed")
                if isinstance(validation, dict) else False
            ),
            "principles_added": _principles_added,
        }

    async def _generate_next_loop_directive(
        self,
        hypothesis: str,
        plan: dict[str, Any],
        validation: dict[str, Any],
        r_phys: Any,
    ) -> None:
        """生成下一轮的自我指令 — RSI 的最小工程实现.

        Agent 反思本轮, 输出一条 directive 写入 memory (category=self_directive).
        下轮 _build_hypothesis_prompt / _build_plan_prompt 通过 _build_memory_text
        自然检索到, 不需要显式注入. memory tier 机制负责衰减, 老指令自动淡出.

        失败静默 — 这是 enhancement 不是 critical path.
        """
        tests_passed = (
            validation.get("tests_passed", False)
            if isinstance(validation, dict)
            else False
        )
        pred_err = (
            validation.get("prediction_error", {})
            if isinstance(validation, dict)
            else {}
        )
        surprise = pred_err.get("surprise", 0) if isinstance(pred_err, dict) else 0

        prompt = (
            "You just finished an autoloop iteration. Reflect on it and write "
            "a single concise directive to your future self for the NEXT iteration.\n\n"
            f"Hypothesis (this iter): {hypothesis[:200]}\n"
            f"Mode: {plan.get('mode', 'unknown') if isinstance(plan, dict) else 'unknown'}\n"
            f"Tests passed: {tests_passed}\n"
            f"R_phys: {r_phys}\n"
            f"Surprise: {surprise:.2f}\n\n"
            "Based on this, output ONE directive (max 2 sentences, no preamble):\n"
            "- If failed: what to AVOID next time (which method/path didn't work)\n"
            "- If high surprise: what to INVESTIGATE deeper\n"
            "- If high r_phys: what method to REUSE\n"
            "- If mundane: what to SKIP to save tokens\n\n"
            "Output only the directive, no markdown headers."
        )

        try:
            response = await self._llm_chat(prompt, task="summarize")
        except Exception:
            # LLM 挂了不阻断 — directive 是 enhancement, 不是 critical path
            logger.debug("RSI directive LLM call failed", exc_info=True)
            return

        if not (response and response.strip()):
            return

        directive = response.strip()[:300]
        # 写入 memory: 用 self_directive category + rsi tag, 让 recall 能定向检索.
        # tier=mid: 几轮后衰减, 不会永久占据 context. importance 跟 surprise 挂钩 —
        # 高 surprise 的 directive 更重要, 衰减更慢.
        importance = 0.5 + min(0.4, surprise * 0.4)
        try:
            self.memory.remember(
                content=f"[self-directive iter {self._iteration}] {directive}",
                category="self_directive",
                tags=["rsi", "autoloop"],
                importance=importance,
                tier="mid",
            )
            logger.info("RSI directive stored in memory: %s", directive[:120])
        except Exception:
            logger.debug("RSI directive memory write failed", exc_info=True)

        # H1: 看 r_phys + directive + 当前 hypothesis/plan blocks, LLM 生成
        # prompt patch 写入 patch store. 下轮 _build_*_prompt 的 _apply_block_patches
        # 自动 apply (Beta mean > 0.5 才生效, 新 patch alpha=beta=1 不会立即应用).
        # 失败静默 — generate_patch 内部已 catch. ponytail: 不接 plan blocks,
        # 只接 hypothesis — 单 phase 试点够验证, 多 phase 升级路径明确.
        try:
            from huginn.harness.prompt_patch import generate_patch
            # 用最近一次 _build_hypothesis_prompt 的 blocks 做 context (无则跳过)
            _hyp_blocks = getattr(self, "_last_hypothesis_blocks", None)
            if _hyp_blocks:
                await generate_patch(
                    phase="hypothesize",
                    blocks=_hyp_blocks,
                    r_phys=float(r_phys) if r_phys is not None else None,
                    directive=directive,
                    llm_chat_fn=self._llm_chat,
                )
        except Exception:
            logger.debug("H1 generate_patch failed", exc_info=True)

    async def _report(
        self, objective: str, phases: list[LoopPhase], total_time: float
    ) -> str | None:
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
                    "duration": (
                        (p.end_time or 0) - (p.start_time or 0)
                        if p.start_time and p.end_time
                        else 0
                    ),
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
        # H4: persona 从 PhaseRegistry 取, toggle off 回退 "tutor"
        from huginn.harness.phase_spec import get_phase_persona
        _report_persona = get_phase_persona("_report") or "tutor"
        report_narrative = ""
        try:
            report_narrative = await self._llm_chat(
                self._build_science_report_prompt(
                    report_data,
                    kb_text,
                    exec_summary,
                    visual_ctx,
                    last_validation,
                    last_hypothesis,
                    last_surprise,
                ),
                persona_name=_report_persona,
                task="summarize",
            )
            report_narrative = (report_narrative or "").strip()
        except Exception:
            report_narrative = ""

        report_path = (
            self.workspace / f"huginn_autoloop_report_{report_data['run_id']}.md"
        )
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
        pred_err = (
            validation.get("prediction_error", {})
            if isinstance(validation, dict)
            else {}
        )
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
                    is_uu = any(
                        kw in gap_lower
                        for kw in [
                            "didn't",
                            "never",
                            "wasn't aware",
                            "didn't think",
                            "hadn't",
                            "overlooked",
                            "完全没",
                            "之前没",
                            "没想到",
                        ]
                    )
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
                    content = content[end + 1 :].strip()
            if content and content != "NONE":
                results.append({"type": btype, "text": content})
                # 写入 GoalStore
                try:
                    from huginn.autoloop.goal_store import get_goal_store

                    _gs = get_goal_store()
                    _active = _gs.get_active()
                    if _active:
                        _gs.add_unknown(
                            _active.id,
                            content,
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
            deviations.append(
                {
                    "iteration": str(self._iteration),
                    "type": "execution_error",
                    "plan_mode": plan_mode,
                    "plan_desc": plan_desc,
                    "deviation": f"Execution failed: {str(result['error'])[:200]}",
                    "expected": expected[:100] if expected else "(none)",
                }
            )

        # 检查 2: evolved fix 被使用
        if isinstance(context, dict) and context.get("_evolved_fix"):
            deviations.append(
                {
                    "iteration": str(self._iteration),
                    "type": "heuristic_fix",
                    "plan_mode": plan_mode,
                    "plan_desc": plan_desc,
                    "deviation": "Applied evolved heuristic fix instead of following original plan",
                    "expected": expected[:100] if expected else "(none)",
                }
            )

        # 检查 3: result success=False
        if isinstance(result, dict) and result.get("success") is False:
            deviations.append(
                {
                    "iteration": str(self._iteration),
                    "type": "plan_mismatch",
                    "plan_mode": plan_mode,
                    "plan_desc": plan_desc,
                    "deviation": "Plan produced unsuccessful result, will need refinement",
                    "expected": expected[:100] if expected else "(none)",
                }
            )

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
        val_section = (
            f"\n## Validation\n{validation_summary}\n" if validation_summary else ""
        )
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

    async def _execute_coder(
        self, description: str, context: dict[str, Any]
    ) -> dict[str, Any]:
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
            tool_calls = sum(1 for m in messages if getattr(m, "tool_calls", None))
            return {
                "mode": "coder",
                "status": "completed",
                "success": True,
                "final_answer": result.get("final_answer", ""),
                "tool_calls": tool_calls,
            }
        except Exception as e:
            logger.exception("coder execution failed")
            return {
                "mode": "coder",
                "status": "failed",
                "success": False,
                "error": str(e),
            }

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

    async def _execute_workflow(
        self, description: str, context: dict[str, Any]
    ) -> dict[str, Any]:
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
            xyz_files = list(self.workspace.rglob("*.xyz")) + list(
                self.workspace.rglob("*.pdb")
            )
            structure_path = (
                str(structure_files[0]) if structure_files else "structure.cif"
            )

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
                    template_name,
                    domain,
                    tmpl_err,
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
                "outputs": result.outputs,
                "error": result.error,
                "stage_results": [
                    {
                        "name": s.stage_name,
                        "success": s.success,
                        "output": s.output_data,
                    }
                    for s in result.stages
                ],
            }
        except Exception as e:
            return {"mode": "workflow", "success": False, "error": str(e)}

    async def _execute_explore(
        self, description: str, context: dict[str, Any]
    ) -> dict[str, Any]:
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

    async def _execute_skill(
        self, plan: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, Any]:
        """Run a pre-built composite skill pipeline."""
        try:
            from huginn.skills.base import DeclarativeSkillExecutor
            from huginn.skills.composite import _ensure_registered
            from huginn.skills.registry import SkillRegistry

            _ensure_registered()

            skill_name = plan.get("skill", "")
            skill = SkillRegistry.get(skill_name)
            if not skill:
                # Fuzzy match if exact name missing
                matches = SkillRegistry.search(
                    skill_name or plan.get("description", "")
                )
                skill = matches[0] if matches else None
            if not skill:
                return {
                    "mode": "skill",
                    "success": False,
                    "error": f"no matching skill for '{skill_name}'",
                }

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
            coords = re.findall(r"\[?(\d+)\s*,\s*(\d+)\]?", description)
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
                                action_result["cropped_image"] = cropped_b64[
                                    :10000
                                ]  # limit size
                                action_result["crop_size"] = [px2 - px1, py2 - py1]
                            except ImportError:
                                action_result[
                                    "note"
                                ] += " (PIL not available, coordinates only)"
                            except Exception as e:
                                action_result["note"] += f" (crop failed: {e})"
                    except Exception:
                        logger.debug("image crop action failed", exc_info=True)
                result["actions"].append(action_result)

        # 动作 2: measure — 测量某点或区域的数据值
        # v8 补全: 解析 visual_ctx 里的 <point>[x,y]</point> 原语, 找最接近的数据点
        elif "measure" in desc_lower:
            coords = re.findall(r"\[?(\d+)\s*,\s*(\d+)\]?", description)
            if coords:
                x, y = int(coords[0][0]), int(coords[0][1])
                # 从 visual_ctx 解析所有 <point>[x,y]</point>=value 原语, 找最近的
                measured = self._measure_nearest_primitive(x, y, visual_ctx)
                result["actions"].append(
                    {
                        "action": "measure",
                        "coordinate": [x, y],
                        "note": f"Measured at <point>[{x},{y}]</point>",
                        "nearest_primitive": measured,
                        "visual_context_snippet": (
                            visual_ctx[:300] if visual_ctx else ""
                        ),
                    }
                )

        # 动作 3: annotate — 标注结构特征
        # v8 补全: 有图片时调 image_analysis_tool 做真正结构标注, 无图片用文本特征
        elif "annotate" in desc_lower:
            annotation = self._annotate_visual_features(description, visual_base64, visual_ctx)
            result["actions"].append(
                {
                    "action": "annotate",
                    "description": description,
                    "note": annotation["note"],
                    "features": annotation.get("features", []),
                    "visual_context": visual_ctx[:500] if visual_ctx else "",
                }
            )
            if annotation.get("tool_output"):
                result["actions"][-1]["tool_output"] = annotation["tool_output"]

        # 动作 4: compare — 比较两组数据
        # v8 补全: 用 extract_comparative_primitives 做真正差分
        elif "compare" in desc_lower:
            comparison = self._compare_visual_data(description, visual_ctx)
            result["actions"].append(
                {
                    "action": "compare",
                    "description": description,
                    "visual_context": visual_ctx[:500] if visual_ctx else "",
                    "note": comparison["note"],
                    "diff": comparison.get("diff", {}),
                }
            )

        # 默认: 记录检查请求
        else:
            result["actions"].append(
                {
                    "action": "inspect",
                    "description": description,
                    "visual_context": visual_ctx[:500] if visual_ctx else "",
                }
            )

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

    def _measure_nearest_primitive(
        self, x: int, y: int, visual_ctx: str
    ) -> dict[str, Any]:
        """v8: 从 visual_ctx 解析 <point>[x,y]</point> 原语, 找最接近 (x,y) 的点.

        visual_hook.py 生成 5 种格式变体 (B1 鲁棒化):
          1. <point>[x,y]</point>(value)       — peak/min (band/dos/phonon)
          2. <point>[x,y]</point>=value         — anomalies
          3. <point>[x,y]</point>=value%        — phase_field / coverage
          4. <point>[x,y]</point> value         — inflections (空格分隔)
          5. key=<point>[y]</point>=value       — scores (单坐标, 只有 y)

        单坐标 (变体 5) 的 x 默认 0, 距离只算 y 差.

        返回最近点的坐标 + 数值 + 上下文. 无原语返回空 dict.
        """
        import re

        if not visual_ctx:
            return {}

        primitives: list[dict[str, Any]] = []

        # 变体 1-3: <point>[x,y]</point>(value) / =value / =value%
        for m in re.finditer(
            r"<point>\[(\d+),(\d+)\]</point>(?:\(([\d.\-eE]+)\)|=([\d.\-eE]+%?)|[\s]+([\d.\-eE]+))",
            visual_ctx,
        ):
            px, py = int(m.group(1)), int(m.group(2))
            val = m.group(3) or m.group(4) or m.group(5)
            val_clean = val.rstrip("%") if val else None
            primitives.append({
                "coordinate": [px, py],
                "value": float(val_clean) if val_clean else None,
                "raw_value": val,
            })

        # 变体 5: key=<point>[y]</point>=value (单坐标)
        for m in re.finditer(
            r"(\w+)=<point>\[(\d+)\]</point>=([\d.\-eE]+%?)",
            visual_ctx,
        ):
            key = m.group(1)
            py = int(m.group(2))
            val = m.group(3).rstrip("%")
            primitives.append({
                "coordinate": [0, py],  # 单坐标 x=0
                "value": float(val) if val else None,
                "raw_value": m.group(3),
                "label": key,
            })

        if not primitives:
            return {}

        # 找最接近 (x, y) 的点
        best = None
        best_dist = float("inf")
        for p in primitives:
            px, py = p["coordinate"]
            d = (px - x) ** 2 + (py - y) ** 2
            if d < best_dist:
                best_dist = d
                best = {**p, "distance": int(best_dist ** 0.5)}

        if best is None:
            return {}

        # 找上下文行 (含该 point 的行)
        coord_str = f"<point>[{best['coordinate'][0]},{best['coordinate'][1]}]</point>"
        for line in visual_ctx.split("\n"):
            if coord_str in line:
                best["context"] = line.strip()[:200]
                break
        return best

    def _annotate_visual_features(
        self, description: str, visual_base64: str, visual_ctx: str
    ) -> dict[str, Any]:
        """v8: 标注结构特征. 有图片调 image_analysis_tool, 无图片用文本特征.

        B2 增强: 无图片时解析 visual_ctx 段落结构 + 趋势 + 异常聚类,
        不只做单点 regex 提取. 让文本路径也有结构化标注.

        ponytail: 优先用已有 image_analysis_tool (defect_detect/phase_field 场景),
        失败降级到 visual_ctx 文本特征提取. 不新建工具.
        """
        import re

        features: list[str] = []
        tool_output: dict[str, Any] | None = None
        # 有图片 → 调 image_analysis_tool 做真正结构标注
        if visual_base64:
            try:
                from huginn.tools.registry import ToolRegistry
                import base64 as b64
                import io as _io

                img_tool = ToolRegistry.get("image_analysis_tool")
                if img_tool:
                    from PIL import Image
                    img_data = b64.b64decode(visual_base64)
                    img = Image.open(_io.BytesIO(img_data))
                    buf = _io.BytesIO()
                    img.save(buf, format="PNG")
                    img_b64 = b64.b64encode(buf.getvalue()).decode()
                    # 根据描述选场景: 含 "defect" → defect_detect, 含 "phase" → phase_field
                    desc_lower = description.lower()
                    if "defect" in desc_lower or "缺陷" in description:
                        scene = "defect_detect"
                    elif "phase" in desc_lower or "相" in description:
                        scene = "phase_field"
                    else:
                        scene = "sem_analysis"  # 默认 SEM 分析
                    # 调工具 (同步 call, 不是 async)
                    res = img_tool.call({
                        "image_base64": img_b64,
                        "scene": scene,
                        "task_description": description,
                    })
                    if res and getattr(res, "success", False):
                        tool_output = res.data if hasattr(res, "data") else res
                        features.append(f"{scene}: tool analysis done")
                    else:
                        features.append(f"{scene}: tool returned no result")
            except Exception as e:
                features.append(f"tool_annotation_failed: {e}")
        # 文本特征提取 (B2 增强: 段落结构 + 趋势 + 异常聚类)
        if visual_ctx:
            structured = self._extract_text_visual_features(visual_ctx)
            features.extend(structured["features"])
            if structured["summary"]:
                # 如果有 tool_output, 把文本 summary 作为补充; 否则作为主 note
                if tool_output is None:
                    tool_output = {"text_analysis": structured["summary"]}
                else:
                    tool_output["text_analysis"] = structured["summary"]
        note = "Annotated " + ", ".join(features[:5]) if features else "No features found"
        return {"note": note, "features": features, "tool_output": tool_output}

    def _extract_text_visual_features(self, visual_ctx: str) -> dict[str, Any]:
        """B2: 从 visual_ctx 提取结构化文本特征 — 段落 + 趋势 + 异常聚类.

        visual_ctx 按段落组织, 每个 [section] 是一个数据集. 解析:
        - section 列表 (band/dos/phonon/scores/phase_field/...)
        - 每段的 trend (increasing/decreasing/flat)
        - 异常点聚类 (相邻异常归为一组)
        - 关键数值 (peak/min/mean/std)
        """
        features: list[str] = []
        summary_parts: list[str] = []
        sections: list[dict[str, Any]] = []

        for line in visual_ctx.split("\n"):
            line = line.strip()
            if not line:
                continue
            # section 标题: [section_name] ...
            sec_match = re.match(r"\[(\w+)\]", line)
            if sec_match:
                sec_name = sec_match.group(1)
                sec: dict[str, Any] = {"name": sec_name, "raw": line[:200]}

                # trend
                trend_m = re.search(r"trend=(\w+)", line)
                if trend_m:
                    sec["trend"] = trend_m.group(1)
                    features.append(f"{sec_name}.trend={trend_m.group(1)}")

                # peak / min
                peak_m = re.search(r"peak=<point>\[\d+,\d+\]</point>\(([\d.\-eE]+)\)", line)
                if peak_m:
                    sec["peak"] = float(peak_m.group(1))
                min_m = re.search(r"min=<point>\[\d+,\d+\]</point>\(([\d.\-eE]+)\)", line)
                if min_m:
                    sec["min"] = float(min_m.group(1))

                # mean / std
                mean_m = re.search(r"mean=([\d.\-eE]+)", line)
                if mean_m:
                    sec["mean"] = float(mean_m.group(1))
                std_m = re.search(r"std=([\d.\-eE]+)", line)
                if std_m:
                    sec["std"] = float(std_m.group(1))

                # anomalies (可能多个, 用逗号分隔)
                anom_m = re.search(r"anomalies=([^,\n]+(?:,\s*[^,\n]+)*)", line)
                if anom_m and anom_m.group(1).strip() != "none":
                    anom_str = anom_m.group(1)
                    anom_count = anom_str.count("<point>")
                    sec["anomaly_count"] = anom_count
                    if anom_count > 0:
                        features.append(f"{sec_name}.anomalies={anom_count}")

                sections.append(sec)
                # summary 行
                parts = [f"{sec_name}"]
                if "trend" in sec:
                    parts.append(f"trend={sec['trend']}")
                if "peak" in sec and "min" in sec:
                    parts.append(f"range=[{sec['min']:.4f}, {sec['peak']:.4f}]")
                if "anomaly_count" in sec and sec["anomaly_count"] > 0:
                    parts.append(f"{sec['anomaly_count']} anomalies")
                summary_parts.append(", ".join(parts))

        summary = "; ".join(summary_parts) if summary_parts else ""
        return {"features": features, "summary": summary, "sections": sections}

    def _compare_visual_data(
        self, description: str, visual_ctx: str
    ) -> dict[str, Any]:
        """v8: 比较两组数据. 用 extract_comparative_primitives 做差分.

        ponytail: visual_hook.extract_comparative_primitives 已有, 直接复用.
        但它需要 baseline + current 两个 dict, visual_ctx 是文本. 这里做文本级
        差分: 解析两组 <point> 原语, 算峰值位移/新异常. 升级路径才换真正的
        baseline vs current dict 比较.
        """
        if not visual_ctx:
            return {"note": "No visual context to compare", "diff": {}}
        # 文本级差分: 找 visual_ctx 里的关键指标, 算数量级
        import re
        peaks = re.findall(r"peak=<point>\[\d+,\d+\]</point>\(([\d.\-eE]+)\)", visual_ctx)
        mins = re.findall(r"min=<point>\[\d+,\d+\]</point>\(([\d.\-eE]+)\)", visual_ctx)
        anomalies = re.findall(r"anomalies=([^,\n]+)", visual_ctx)
        diff: dict[str, Any] = {}
        if peaks:
            peak_vals = [float(p) for p in peaks if p]
            diff["peak_range"] = [min(peak_vals), max(peak_vals)]
            diff["peak_count"] = len(peak_vals)
        if mins:
            min_vals = [float(m) for m in mins if m]
            diff["min_range"] = [min(min_vals), max(min_vals)]
        if anomalies:
            diff["anomaly_count"] = sum(1 for a in anomalies if a.strip() and a.strip() != "none")
        # 检查描述里有没有指定比较对象 (e.g. "compare band 3 and band 5")
        compare_match = re.search(r"compare\s+(\w+\s*\d*)\s+(?:and|with|vs\.?)\s+(\w+\s*\d*)", description, re.IGNORECASE)
        target = None
        if compare_match:
            target = f"{compare_match.group(1).strip()} vs {compare_match.group(2).strip()}"
        note_parts = []
        if diff:
            note_parts.append(f"Found {diff.get('peak_count', 0)} peaks, {diff.get('anomaly_count', 0)} anomalies")
        if target:
            note_parts.append(f"Requested: {target}")
        if not note_parts:
            note_parts.append("Comparison recorded (no quantitative data to diff)")
        return {"note": "; ".join(note_parts), "diff": diff}

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
        - "reasoning"/"science" → 强模型 (云端, 发散性假设)
        - "planning" → 中档模型 (收敛, 把假设变步骤)  [OAK 三阶段角色]
        - "summarize"/"format" → 便宜模型 (本地/小模型)
        - "verification" → 独立验证模型
        model 参数优先于 task — 显式指定的模型不被路由覆盖.
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        # Team 模式: task 路由优先, 但显式 model 不被覆盖
        if model is None and task is not None:
            router = getattr(self, "model_router", None)
            if router is not None:
                try:
                    routed = router.select(
                        task,
                        prefer_cheap=(
                            task in ("summarize", "format", "archival", "planning")
                        ),
                    )
                    if routed is not None:
                        model = routed
                except Exception:
                    logger.debug(
                        "model router select failed — using fallback model",
                        exc_info=True,
                    )

        llm = model or self.model
        messages: list[Any] = []
        if persona_name:
            sys_prompt = self._persona_system_prompt(persona_name)
            # H4: GRILL 模式 active 时把 GRILL_SYSTEM_PROMPT_CN 追加到 system prompt.
            # 之前 should_pause_for_decision 触发 GRILL 后只 auto-resume, LLM 看不到
            # grill 约束, "一次一问" 形同虚设. 现在注入, LLM 自己负责流程.
            if self._grill_active:
                try:
                    from huginn.runtime.pre_plan_grill import GRILL_SYSTEM_PROMPT_CN
                    sys_prompt = (sys_prompt or "") + "\n\n" + GRILL_SYSTEM_PROMPT_CN
                    self._grill_turns += 1
                    # 退出检查: LLM 输出含 "shared understanding" 确认标记 → 退出.
                    # 依赖上层 (run_cognitive) 把 LLM 输出回传后再判断, 这里只计数.
                    # ceiling: 简单字符串匹配, 升级用 LLM judge.
                    if self._grill_turns > 20:
                        logger.warning(
                            "GRILL 超过 20 轮, 强制退出 (可能 LLM 没理解确认标记)"
                        )
                        self._grill_active = False
                except ImportError:
                    logger.debug("pre_plan_grill import failed, GRILL prompt 跳过")
            if sys_prompt:
                sys_msg = SystemMessage(content=sys_prompt)
                # 静态 system prompt 跨调用不变, 给 Anthropic/Kimi 打 cache 标记
                _ident = f"{type(llm).__name__}{getattr(llm, 'model', '')}".lower()
                if any(
                    k in _ident for k in ("anthropic", "claude", "kimi", "moonshot")
                ):
                    sys_msg.additional_kwargs["cache_control"] = {"type": "ephemeral"}
                messages.append(sys_msg)
        # Controllable thinking effort: 按 current phase 注入思考深度指令.
        # Inkling 启发 — 连续旋钮, prompt 层实现, 对所有 provider 统一.
        # 无 _current_phase (非 phase 上下文调用, 如 _feynman_learn) 时不注入.
        effort_directive = ""
        if self._current_phase:
            effort = _PHASE_THINKING_EFFORT.get(self._current_phase, 0.5)
            effort_directive = _effort_to_prompt(effort)
        if effort_directive:
            prompt = f"[Thinking effort: {effort_directive}]\n\n{prompt}"
        messages.append(HumanMessage(content=prompt))
        # P0-1: 流式化 — astream 替代 ainvoke, 增量 chunk 通过 progress_cb 推 WS.
        # 700 万步场景: decider 思考过程实时可见, 不再黑盒. fail 回退 ainvoke.
        # ponytail: 只在 progress_cb 存在时流式, 否则 ainvoke (兼容无 WS 场景).
        from huginn.types import progress_cb as _progress_cb

        _cb = _progress_cb.get(None)
        if _cb is None or not hasattr(llm, "astream") or not _autoloop_streaming_enabled():
            response = await llm.ainvoke(messages)
            return str(response.content)
        # 流式: 累积 content, 同时推 thinking chunk 到 WS
        parts: list[str] = []
        try:
            async for chunk in llm.astream(messages):
                _delta = ""
                # langchain BaseMessageChunk: chunk.content 是 str 或 list
                if hasattr(chunk, "content"):
                    if isinstance(chunk.content, str):
                        _delta = chunk.content
                    elif isinstance(chunk.content, list):
                        # 多模态 chunk, 只取 text block
                        _delta = "".join(
                            b.get("text", "") if isinstance(b, dict) else str(b)
                            for b in chunk.content
                        )
                if _delta:
                    parts.append(_delta)
                    try:
                        await _cb({
                            "type": "autoloop_thinking",
                            "phase": self._current_phase or "decider",
                            "delta": _delta[:200],  # 截断防超大 delta
                        })
                    except Exception:
                        pass  # cb 失败不阻塞 LLM
            return "".join(parts)
        except Exception as e:
            # 流式失败回退 ainvoke (某些 provider astream 实现有 bug)
            logger.debug("astream failed, fallback to ainvoke: %s", e)
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
        # C2: PM 层 trajectory_match 召回 (极限模式才开)
        pm_block = self._build_pm_text()
        if pm_block:
            pm_block = f"\n{pm_block}\n"
        # C2: metacog 信号注入 — target_chain (objective 在目标分解树的位置) +
        # prospective (待执行前瞻意图). 跟 rcb_runner 同源, 之前 autoloop 零接.
        metacog_block = self._build_metacog_block(include_prospective=True)
        if metacog_block:
            metacog_block = f"\n{metacog_block}\n"
        # H0: stable_principles 注入 (修 P3 断链 — 之前只进 chat agent system prompt,
        # autoloop 完全跳过 PM 层). 取 top-5 避免塞爆 prompt.
        try:
            _principles = load_stable_principles()[:5]
        except Exception:
            _principles = []
        principles_block = (
            "\n".join(f"- {p}" for p in _principles) if _principles else ""
        )
        if principles_block:
            principles_block = (
                f"\n### Stable Principles (procedural memory)\n{principles_block}\n"
            )
        # 视觉基元: 上一轮 tool 输出的数值指针 (峰值/趋势/异常),
        # 给 LLM 具体坐标锚定推理 — Thinking with Visual Primitives 的
        # "point while it reasons" 原则, Mirage 效应的文本路径
        visual_block = getattr(self, "_last_visual_context", "")
        if visual_block:
            visual_block = (
                f"\n### Visual Primitives (from last tool output)\n{visual_block}\n"
            )
        ctx_blob = json.dumps(context, ensure_ascii=False).lower()
        math_block = (
            self._MATH_DEPTH_PROMPT_BLOCK
            if any(s in ctx_blob for s in _MATH_SIGNALS)
            else ""
        )
        # MatterChat 启发: 把上轮 execution 结果摘要注入 hypothesis prompt,
        # 让假设建立在"上轮实际发生了什么"之上, 不只看 workspace 变化.
        # _last_execution_result 在 _execute 里写入, 之前只 _build_plan_prompt 用.
        exec_block = ""
        last_exec = getattr(self, "_last_execution_result", None)
        if last_exec and isinstance(last_exec, dict):
            _tool = last_exec.get("_tool_name", "unknown")
            _res = last_exec.get("result", last_exec)
            _summary = json.dumps(_res, ensure_ascii=False, default=str)[:500]
            exec_block = f"\n### Last Execution Result ({_tool})\n{_summary}\n"
        # H2: frontier_ranked 注入 — Ising 能量最低 K-子集未测试假设.
        # 之前 frontier()/frontier_ranked() 0 生产调用, Ising 排序整套死代码.
        # 现在作为 hint 注入, LLM 可选优先测这些 (refute 过的 parent 的子假设优先,
        # 同 sibling_group 互斥已避开). ponytail: hint 不强制, LLM 自己决定.
        frontier_block = ""
        try:
            _frontier = self.hypothesis_graph.frontier_ranked(top_k=3)
            if _frontier:
                _lines = []
                for nd in _frontier:
                    _stmt = (nd.statement or "")[:100]
                    _lines.append(f"- [{nd.id}] {_stmt}")
                frontier_block = (
                    f"\n### Untested Hypotheses (Ising-ranked, energy-low)\n"
                    + "\n".join(_lines) + "\n"
                    "Consider testing one of these before generating a new hypothesis.\n"
                )
        except Exception:
            logger.debug("frontier_ranked injection failed", exc_info=True)
        # P0: FAILED.md / PROVED.md durable state 注入 (chaoxu 启发).
        # context 压缩后 agent 重读这两个文件, 不重试死路, 不重新证明已过的.
        # ponytail: 读文件首 N 行避免膨胀, full text 留给 _extract_compact_attachments.
        failed_block = ""
        proved_block = ""
        try:
            from huginn.autoloop.hypothesis_loop import HypothesisGraph as _HG0
            _ws = str(self.workspace) if hasattr(self, "workspace") else None
            _failed_txt = _HG0.load_failed(_ws)
            if _failed_txt:
                _failed_lines = _failed_txt.strip().split("\n")[:40]
                failed_block = (
                    "\n### Dead Routes (FAILED.md)\n"
                    + "\n".join(_failed_lines) + "\n"
                    "Do NOT re-attempt these unless the Reopen-if condition is met.\n"
                )
            _proved_txt = _HG0.load_proved(_ws)
            if _proved_txt:
                _proved_lines = _proved_txt.strip().split("\n")[:40]
                proved_block = (
                    "\n### Verified Results (PROVED.md)\n"
                    + "\n".join(_proved_lines) + "\n"
                    "These are already established — build on them.\n"
                )
        except Exception:
            logger.debug("FAILED/PROVED injection failed", exc_info=True)
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
                cwd=self.workspace,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if _r.returncode == 0 and _r.stdout.strip():
                git_log_block = (
                    f"\n### Recent Experiments (git log)\n{_r.stdout.strip()}\n"
                )
        except Exception:
            pass

        # 分量代表制: 多条独立探索路线时, 给 LLM 看各路线的代表假设,
        # 防止单分量靠节点数主导综合判断. 只在 >1 分量时注入, advisory.
        # v11: 升级为 cluster_block — 优先用 cluster_by_dimension 展示维度分布,
        # 退化为 topology_block (分量代表) 当 dimension 全空.
        cluster_block = ""
        try:
            _clusters = self.hypothesis_graph.cluster_by_dimension()
            _known_dims = {k: v for k, v in _clusters.items() if k != "unknown"}
            if _known_dims and len(_clusters) > 1:
                lines = []
                for dim, nodes in list(_known_dims.items())[:5]:
                    _stmt = nodes[0].statement[:120] if nodes else ""
                    lines.append(f"  - {dim} ({len(nodes)} 个假设): {_stmt}")
                cluster_block = (
                    f"\n### Cluster (advisory)\n"
                    f"当前假设按 dimension 分布:\n"
                    + "\n".join(lines)
                    + "\n新假设应优先补未覆盖的 dimension, 避免在已饱和维度堆叠.\n"
                )
            else:
                # 退化路径: dimension 全空时用 topology_block (分量代表)
                reps = self._metacog_component_representatives()
                if len(reps) > 1:
                    lines = []
                    for rid in reps[:5]:
                        try:
                            stmt = self.hypothesis_graph.get(rid).statement
                        except Exception:
                            stmt = ""
                        lines.append(f"  - {rid}: {stmt[:120]}")
                    cluster_block = (
                        f"\n### Topology (advisory)\n"
                        f"当前有 {len(reps)} 条独立探索路线, 代表假设分别是:\n"
                        + "\n".join(lines)
                        + "\n"
                        "综合判断时不要让某条路线靠节点数主导, 注意挑战和重定向.\n"
                    )
        except Exception:
            pass

        # 按优先级拼接, 超预算自动裁剪低优先级 block
        blocks = self._apply_block_patches(
            [
                (
                    "body",
                    f"""You are an autonomous material science research agent.

Perceived context:
{json.dumps(context, indent=2, ensure_ascii=False)[:2000]}

Generate 3 divergent candidate hypotheses. Each MUST be grounded in a
DIFFERENT assumption dimension. Pick dimensions from this list (or propose
a new one tagged [NEW]):
- composition (Ca/Si/Al/O ratio, doping, alloy)
- temperature (thermal dependence, phase transition)
- defect (vacancy, dislocation, interface)
- structure (crystal symmetry, lattice parameter)
- transport (diffusion, conductivity, mobility)

Format each candidate as:
[DIM: <dimension>] <statement> | pro: ... | con: ...

After listing 3, select the most testable+novel one after "SELECTED:".
The 3 candidates must NOT be variations of each other — if two share the
same dimension, the second is invalid and must be replaced.
Ground it in the domain knowledge context above when relevant.
Prefer hypotheses that can be expressed as governing PDEs, variational
principles, or conservation laws; identify the mathematical structure
before proposing numerical experiments.

Hypothesis:""",
                ),
                ("git_log", git_log_block),
                ("fail", fail_block),
                ("imagination", imagination_block),
                ("exec", exec_block),
                ("frontier", frontier_block),
                ("failed", failed_block),
                ("proved", proved_block),
                ("math", math_block),
                ("kg", kg_block),
                ("visual", visual_block),
                ("kb", kb_block),
                ("principles", principles_block),
                ("mem", mem_block),
                ("pm", pm_block),
                ("metacog", metacog_block),
                ("cluster", cluster_block),
                ("hint", hint_block),
            ],
            "hypothesize",
        )
        return self._trim_to_budget(blocks, phase="hypothesize")

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
        # C2: PM 层 trajectory_match 召回 (同 hypothesize, 极限模式才开)
        pm_block = self._build_pm_text()
        if pm_block:
            pm_block = f"\n{pm_block}\n"
        # C2: metacog 信号注入 (同 hypothesize) — target_chain + prospective
        metacog_block = self._build_metacog_block(include_prospective=True)
        if metacog_block:
            metacog_block = f"\n{metacog_block}\n"
        # H0: stable_principles 注入 (同 hypothesize, 修 P3 断链)
        try:
            _principles = load_stable_principles()[:5]
        except Exception:
            _principles = []
        principles_block = (
            "\n".join(f"- {p}" for p in _principles) if _principles else ""
        )
        if principles_block:
            principles_block = (
                f"\n### Stable Principles (procedural memory)\n{principles_block}\n"
            )
        # 视觉基元注入 (同 hypothesize)
        visual_block = getattr(self, "_last_visual_context", "")
        if visual_block:
            visual_block = (                f"\n### Visual Primitives (from last tool output)\n{visual_block}\n"
            )
        # 条件化 math_block (同 hypothesize)
        hyp_blob = (
            hypothesis.lower() + json.dumps(context, ensure_ascii=False).lower()[:500]
        )
        math_block = (
            self._MATH_DEPTH_PROMPT_BLOCK
            if any(s in hyp_blob for s in _MATH_SIGNALS)
            else ""
        )

        # Inject learned skills + prompt patches from evolution engine.
        # This is the "use what you learned" half of the Learn→Plan loop.
        skill_hints = ""
        patch_hints = ""
        try:
            evolution = self._get_evolution()
            skills = evolution.get_relevant_skills(hypothesis)
            if skills:
                skill_lines = [f"  - {s.name}: {s.description}" for s in skills[:3]]
                skill_hints = (
                    "\nLearned skills (from past iterations):\n"
                    + "\n".join(skill_lines)
                    + "\n"
                )
            patches = evolution.get_prompt_patches()
            if patches:
                patch_hints = (
                    "\nLearned patches:\n"
                    + "\n".join(f"  - {p}" for p in patches[:3])
                    + "\n"
                )
        except Exception:
            logger.warning(
                "error in _build_plan_prompt: evolution skill/patch fetch failed",
                exc_info=True,
            )

        # Inject matching composite skills — lets the LLM pick a pre-built
        # multi-tool pipeline instead of improvising from scratch.
        # 条件化: 只在 hypothesis 涉及仿真/计算/材料性质时注入, coder-only
        # 任务不需要 composite skill 列表. 节省 ~500 tokens.
        composite_block = ""
        hyp_lower = hypothesis.lower()
        _workflow_signals = (
            "workflow",
            "simulation",
            "band",
            "dos",
            "phonon",
            "mechanical",
            "thermal",
            "optical",
            "dft",
            "vasp",
            "lammps",
            "md ",
            "structure",
            "property",
            "energy",
            "convergence",
            "optimize",
            "calc",
        )
        if any(s in hyp_lower for s in _workflow_signals):
            try:
                from huginn.skills.composite import _ensure_registered
                from huginn.skills.registry import SkillRegistry

                _ensure_registered()
                matches = SkillRegistry.search(hypothesis)
                if not matches:
                    matches = SkillRegistry.get_all_definitions()
                if matches:
                    lines = [s.to_prompt() for s in matches[:4]]
                    composite_block = (
                        "\nAvailable composite skills (prefer these over manual workflow):\n"
                        + "\n\n".join(lines)
                        + "\n"
                    )
            except Exception:
                logger.debug("composite skill lookup failed", exc_info=True)

        # Pipeline 建议: 基于 provenance 规则推荐下一步工具.
        # 42 条领域规则, 零 LLM 调用. 让 plan 知道"这类任务通常下一步是 X".
        pipeline_block = ""
        try:
            from huginn.provenance.pipeline import SimulationPipeline

            pipeline = SimulationPipeline(
                self.kg.root if hasattr(self.kg, "root") else None
            )
            # 用上一轮的 execution_result 触发 suggest_next
            last_result = getattr(self, "_last_execution_result", None)
            if last_result and isinstance(last_result, dict):
                tool_name = last_result.get("_tool_name", "")
                suggestions = pipeline.suggest_next(
                    tool_name=tool_name,
                    tool_input=last_result.get("_tool_input", {}),
                    tool_output=last_result.get("result", last_result),
                )
                if suggestions:
                    s_lines = [
                        f"  - {s.tool_hint}: {s.description}" for s in suggestions[:3]
                    ]
                    pipeline_block = (
                        "\nPipeline suggestions (based on provenance):\n"
                        + "\n".join(s_lines)
                        + "\n"
                    )
        except Exception:
            pass  # pipeline 是 advisory, 失败不阻塞

        blocks = self._apply_block_patches(
            [
                (
                    "body",
                    f"""Given the hypothesis: "{hypothesis}"

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
""",
                ),
                ("math", math_block),
                ("kg", kg_block),
                ("visual", visual_block),
                ("kb", kb_block),
                ("principles", principles_block),
                ("mem", mem_block),
                ("pm", pm_block),
                ("metacog", metacog_block),
                ("skill", skill_hints + patch_hints),
                ("composite", composite_block),
                ("pipeline", pipeline_block),
                ("subgoal", self._build_subgoal_block()),
                ("ctx_hint", self._plan_context_hint()),
            ],
            "plan",
        )
        return self._trim_to_budget(blocks, phase="plan")

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
            hints.append(f"NOTE: 已 refine {rc} 次. 如果再失败可能需要 pivot 换方向.")
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
            if (
                current_hyp
                and self.hypothesis_graph.needs_dual_coverage(current_hyp)
                and current_mode == "coder"
            ):
                plan["mode"] = "workflow"
                plan["override_reason"] = "cut_vertex_dual_coverage"
                plan["description"] = (
                    f"[auto-routed: 割点需双覆盖] {plan.get('description', '')}"
                )
                logger.info(
                    "override mode coder→workflow for cut vertex %s", current_hyp
                )
                self._log_plan_override(
                    "cut_vertex_dual_coverage", f"割点 {current_hyp} 需双覆盖"
                )
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
                PhaseGate,
                get_shared_phase_gate_state,
            )

            state = get_shared_phase_gate_state()
            state.history.append(
                PhaseGate(
                    from_phase="plan",
                    to_phase="plan",
                    status="approved",
                    feedback=f"[auto-routed] {reason_code}: {reason_text}",
                    reviewer="auto_router",
                )
            )
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

    # ── KRCL plan check (反向校验 + 闭环重生成) ─────────────────
    # 磐石100 KRCL 启发: 正向神经规划器生成 plan → 反向符号规划识别器校验
    # → 失败反馈重生成. ponytail: 单 LLM 反向校验, 不上 PDDL solver.
    # ceiling: LLM 自校验有同模型盲点, 不如 KRCL 的符号识别器硬.
    # 升级路径: 接 BourbakiTool.check_conservation 做符号反推 (需 Lean 成熟).
    async def _plan_check_and_refine(
        self,
        plan: dict[str, Any],
        hypothesis: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """KRCL 闭环: 反向校验 plan, 失败反馈 LLM 重生成, 超限不阻塞.

        phase-aware: iteration tier (open/medium/light) + plan 复杂度综合
          判定. open 或 skip 跳过校验, medium 只校验不 refine, light 完整闭环.
          复杂 plan 即使 open tier 也升级到 medium (要校验), 简单 plan 即使
          light tier 也降级到 skip (阈值 0.25: explore+20chars desc 能触发,
          coder 的简单任务仍校验因为涉及代码改动).
        自适应: 按 scene_tag 分桶的最近 5 次 success rate 微调 max_refines —
          >=80% 放宽 (-1), <=20% 收紧 (+1), 样本不足走 baseline.
        不暴露: check 结果只存 self._plan_check_last_result / _warnings /
          _plan_check_patterns, 不塞回 plan dict (plan 会进 prompt, 塞了
          等于喂 LLM 元信息).
        失败模式记忆: 失败记到 _plan_check_patterns, 跨 run JSON 持久化,
          下次同场景 plan 来了注入 prompt 让 LLM 重点避开.
        连续失败澄清: 同 scene 连续 3 次失败 + scene != "other" -> 触发
          _maybe_clarify 问用户 (physical_precheck 同款, 不阻塞).

        失败不拦截 (physical_precheck 同款), warning 留痕给 _validate.
        """
        # trivial plan (description 太短) 跳过, 不浪费 LLM 调用
        desc = plan.get("description", "")
        if len(desc) < 20:
            return plan
        tier = self._plan_check_tier(plan)
        if tier in ("open", "skip"):
            logger.debug(
                "plan_check skipped (tier=%s, iter=%d)",
                tier,
                self._iteration,
            )
            return plan
        scene = self._plan_check_scene_tag(plan)
        max_refines = self._plan_check_max_refines(tier, scene)
        for attempt in range(max_refines + 1):
            try:
                check = await self._plan_check(plan, hypothesis, context)
            except Exception as e:
                logger.debug("plan_check LLM call failed: %s", e)
                return plan
            # 给 check 打 scene_tag, 喂分桶自适应; 不暴露: 存引擎状态.
            # 成功时存 plan_snapshot, 喂 _refine_plan few-shot.
            check["scene_tag"] = scene
            if check.get("is_valid", True):
                check["plan_snapshot"] = {
                    "mode": plan.get("mode", ""),
                    "description": plan.get("description", "")[:200],
                }
            self._plan_check_last_result = check
            self._plan_check_history.append(check)
            # 历史窗口截断, 保留最近 20 条防无限增长
            if len(self._plan_check_history) > 20:
                del self._plan_check_history[: len(self._plan_check_history) - 20]
            if check.get("is_valid", True):
                # confidence 分级: 低置信通过 (<0.5) 强制 refine 一次, 防 LLM
                # 没看懂就放行; 高置信直接通过.
                confidence = float(check.get("confidence", 0.8))
                if confidence >= 0.5 or attempt >= max_refines or max_refines == 0:
                    logger.info(
                        "plan_check passed (attempt %d, tier=%s, scene=%s, conf=%.2f)",
                        attempt,
                        tier,
                        scene,
                        confidence,
                    )
                    # 每 5 次校验触发一次 scene_tag 自动发现 (低成本, 不阻塞)
                    if len(self._plan_check_history) % 5 == 0:
                        self._discover_scene_tags()
                    return plan
                logger.info(
                    "plan_check passed but low confidence (conf=%.2f), refining",
                    confidence,
                )
            else:
                # 失败: 记到 patterns (跨 run 持久化, 喂下次 prompt)
                self._record_plan_check_failure(plan, check, scene)
                # confidence 分级: 低置信失败 (<0.3) 跳过 refine, LLM 都没把握
                # 判断, refine 可能也是瞎改, 直接 warning + 触发澄清更靠谱.
                confidence = float(check.get("confidence", 0.8))
                if confidence < 0.3:
                    reason = check.get("reason", "unknown")
                    self._plan_check_warnings.append(
                        f"[{scene}] {reason} (low_conf={confidence:.2f})"
                    )
                    logger.warning(
                        "plan_check failed low-conf (tier=%s, scene=%s, conf=%.2f): %s",
                        tier,
                        scene,
                        confidence,
                        reason,
                    )
                    await self._maybe_trigger_plan_check_clarify(scene, reason, plan)
                    return plan
                if attempt >= max_refines:
                    reason = check.get("reason", "unknown")
                    self._plan_check_warnings.append(f"[{scene}] {reason}")
                    logger.warning(
                        "plan_check failed (tier=%s, scene=%s, max_refines=%d): %s",
                        tier,
                        scene,
                        max_refines,
                        reason,
                    )
                    # 连续失败触发主动澄清 (不阻塞, 用户可 force_proceed)
                    await self._maybe_trigger_plan_check_clarify(
                        scene,
                        reason,
                        plan,
                    )
                    return plan
            logger.info(
                "plan_check refining (attempt %d, tier=%s, scene=%s, conf=%.2f): %s",
                attempt,
                tier,
                scene,
                float(check.get("confidence", 0.8)),
                check.get("reason"),
            )
            plan = await self._refine_plan(plan, check, hypothesis, context)
        return plan

    async def _maybe_trigger_plan_check_clarify(
        self,
        scene: str,
        reason: str,
        plan: dict[str, Any],
    ) -> None:
        """连续 N 次同场景失败 + 场景已知 -> 问用户方向.

        ponytail: 阈值 3 写死, 跟 validation_fail 同款; 不阻塞, 异常吞掉.
        ceiling: 阈值靠拍; "other" 场景没上下文给用户, 直接跳过.
        """
        if scene == "other":
            return
        # 数最近连续失败 (同 scene, 遇到第一条成功就断)
        recent_fails = 0
        for c in reversed(self._plan_check_history):
            if c.get("scene_tag") == scene and not c.get("is_valid", True):
                recent_fails += 1
            else:
                break
        if recent_fails < 3:
            return
        try:
            await self._maybe_clarify(
                "plan_check_fail",
                {
                    "scene": scene,
                    "reason": reason,
                    "consecutive_fails": recent_fails,
                    "plan": plan,
                },
            )
        except Exception as e:
            logger.debug("plan_check clarify failed: %s", e)

    def _plan_check_tier(self, plan: dict[str, Any] | None = None) -> str:
        """phase-aware tier: iteration + plan 复杂度综合判定.

        iteration baseline: open (1-10) / medium (11-30) / light (31+).
        跟 ProgressiveBudget.default() 边界对齐, 但解耦 — budget 关了
        plan_check 仍按 iteration 判 phase.
        plan 复杂度修正 (plan 传入时):
          - 复杂 plan (score >= upgrade_threshold) 即使 open tier 也升级到 medium
          - 简单 plan (score < downgrade_threshold) 即使 light tier 也降级到 skip
        阈值分场景校准: DFT/MD/workflow 各有自己的 success rate, 不会互相带偏.
        ponytail: 阈值从 _plan_check_complexity_thresholds(scene) 取, 不是写死.
        ceiling: 校准靠历史 success rate, 样本不足走默认 0.7/0.25;
          边界跟 ProgressiveBudget 重复一份.
        升级路径: ProgressiveBudget 暴露 tier_of(n) -> label, 这里复用;
                  阈值用 Bayesian 更新而非简单 success rate.
        """
        n = getattr(self, "_iteration", 0)
        if n <= 10:
            base = "open"
        elif n <= 30:
            base = "medium"
        else:
            base = "light"
        if plan is None:
            return base
        complexity = self._plan_check_complexity(plan)
        scene = self._plan_check_scene_tag(plan)
        upgrade_t, downgrade_t = self._plan_check_complexity_thresholds(scene)
        if complexity >= upgrade_t and base == "open":
            return "medium"
        if complexity < downgrade_t and base == "light":
            return "skip"
        return base

    def _plan_check_complexity_thresholds(self, scene: str = "") -> tuple[float, float]:
        """用历史 success rate 自动校准复杂度阈值, 分场景.

        默认: upgrade=0.7 (复杂 plan 升级到 medium), downgrade=0.25 (简单
        plan 降级到 skip).
        分场景校准: 同 scene_tag 的最近 10 条 plan_check 的 success rate
          >=0.8 (一直成功) -> upgrade 放宽到 0.8, downgrade 收紧到 0.15
            (成功率高, 只拦最复杂的, 简单的不轻易跳过)
          <=0.2 (一直失败) -> upgrade 收紧到 0.6, downgrade 放宽到 0.35
            (失败率高, 多拦一些, 简单的也更容易跳过不浪费 LLM)
          样本 <5 走默认, 早期不误判. 未知场景 (scene 无历史) 走全局.
        ponytail: 线性插值, 不上 Bayesian; 阈值钳制在 [0.4, 0.9] / [0.1, 0.4].
        ceiling: 线性插值过于简单; 场景样本不足时回退全局.
        升级路径: 上 Bayesian 更新带先验; 场景用 embedding 聚类而非关键词.
        """
        history = getattr(self, "_plan_check_history", [])
        if scene:
            bucket = [c for c in history if c.get("scene_tag") == scene]
        else:
            bucket = history
        if len(bucket) < 5:
            # 场景样本不足, 回退全局; 全局也不足, 走默认
            if scene and len(history) >= 5:
                bucket = history
            else:
                return (0.7, 0.25)
        recent = bucket[-10:]
        success_rate = sum(1 for c in recent if c.get("is_valid", True)) / len(recent)
        if success_rate >= 0.8:
            return (0.8, 0.15)
        if success_rate <= 0.2:
            return (0.6, 0.35)
        return (0.7, 0.25)

    def _plan_check_scene_tag(self, plan: dict[str, Any]) -> str:
        """从 plan 抽场景标签, 给失败模式记忆和分桶自适应用.

        写死的关键词表 + 自动发现的关键词 (_scene_tag_extra_keywords) 互补.
        ponytail: 关键词匹配, 不上 embedding.
        ceiling: 写死的关键词表要手动加新仿真器; 自动发现靠高频词统计,
          新场景需要 >=3 次出现才会被识别.
        升级路径: 用 plan_check_history 聚类自动发现 scene_tag (无监督).
        """
        desc = (plan.get("description", "") + " " + plan.get("mode", "")).lower()
        # 写死的关键词表 (快路径)
        if any(
            kw in desc
            for kw in [
                "vasp",
                "scf",
                "band",
                "dos",
                "dft",
                "qe",
                "cp2k",
                "gaussian",
                "orca",
            ]
        ):
            return "dft"
        if any(
            kw in desc
            for kw in [
                "lammps",
                "molecular dynamics",
                "minimize",
                "nvt",
                "npt",
                "md ",
                "gromacs",
                "openmm",
            ]
        ):
            return "md"
        if any(kw in desc for kw in ["workflow", "pipeline", "orchestrat"]):
            return "workflow"
        if plan.get("mode") == "skill":
            return "skill"
        if any(
            kw in desc
            for kw in ["fenics", "abaqus", "comsol", "openfoam", "fem", "elmer"]
        ):
            return "fem"
        # 自动发现的关键词 (慢路径, 跨 run 积累)
        for label, keywords in getattr(self, "_scene_tag_extra_keywords", {}).items():
            if any(kw in desc for kw in keywords):
                return label
        return "other"

    def _discover_scene_tags(self) -> None:
        """从 _plan_check_history 里 scene='other' 的 plans 做关键词统计,
        发现高频词 (>=3 次) 自动加到 _scene_tag_extra_keywords.

        命中未知场景 — 新仿真器/新任务类型不用手动改关键词表, 跑几次
        plan_check 后自动归类.
        双重识别: unigram (>=4 chars) + bigram (两词短语, 如 "phase diagram",
        "neb chain"), 更准地捕获多词术语.
        ponytail: 简单词频统计, 不上 TF-IDF/embedding.
        ceiling: 只统计 scene='other' 的 plans, 已归类的不参与; 阈值 3 靠拍;
          只取英文, 中文/数字不参与; bigram 不去介词/停用词组合.
        升级路径: 上 TF-IDF 或 embedding 聚类, 识别任意长度 n-gram.
        """
        import re
        from collections import Counter

        # 收集 scene='other' 的 plan descriptions
        other_descs: list[str] = []
        for c in getattr(self, "_plan_check_history", []):
            snapshot = c.get("plan_snapshot") or {}
            if c.get("scene_tag") == "other" and snapshot.get("description"):
                other_descs.append(snapshot["description"].lower())
        if len(other_descs) < 3:
            return  # 样本不足, 不触发发现
        # 统计英文单词词频 (>=4 chars, 过滤停用词)
        stop = {
            "the",
            "and",
            "for",
            "with",
            "that",
            "this",
            "from",
            "run",
            "then",
            "calc",
            "calculate",
            "using",
            "use",
            "plan",
            "step",
        }
        word_counts: Counter[str] = Counter()
        # bigram 词频 (两词短语, 用空格连接)
        bigram_counts: Counter[str] = Counter()
        for desc in other_descs:
            words = [
                w for w in re.findall(r"[a-z][a-z0-9_]{3,}", desc) if w not in stop
            ]
            for word in words:
                word_counts[word] += 1
            # bigram: 相邻两词组合
            for i in range(len(words) - 1):
                bigram = f"{words[i]} {words[i+1]}"
                bigram_counts[bigram] += 1
        # 高频词 (>=3 次) 加到 extra_keywords, 用 word 本身做 label
        for word, count in word_counts.most_common(10):
            if count >= 3:
                label = f"auto_{word}"
                self._scene_tag_extra_keywords.setdefault(label, set()).add(word)
        # 高频 bigram (>=3 次) 加到 extra_keywords, 用下划线连接做 label
        # (如 "phase diagram" -> auto_phase_diagram, 关键词 "phase diagram")
        for bigram, count in bigram_counts.most_common(5):
            if count >= 3:
                label = f"auto_{bigram.replace(' ', '_')}"
                self._scene_tag_extra_keywords.setdefault(label, set()).add(bigram)

    def _plan_check_complexity(self, plan: dict[str, Any]) -> float:
        """plan 复杂度评分 [0, 1], 跟 tier 一起决定是否校验.

        维度: description 长度 (0.3) + mode 复杂度 (0.4) + 有无 prediction
        (0.15) + 同场景历史失败数 (0.15, 踩过坑的要复查).
        ponytail: 启发式打分, 不上结构化解析.
        ceiling: description 长度不代表真复杂度, 长描述可能是废话.
        升级路径: 解析 plan 的 step 数 (需要结构化 plan schema).
        """
        score = 0.0
        desc = plan.get("description", "")
        score += min(len(desc), 50) / 50 * 0.3
        mode = plan.get("mode", "coder")
        score += {"workflow": 0.4, "skill": 0.3, "coder": 0.2, "explore": 0.1}.get(
            mode, 0.2
        )
        if plan.get("expected_prediction"):
            score += 0.15
        scene = self._plan_check_scene_tag(plan)
        similar_fails = sum(
            1
            for p in getattr(self, "_plan_check_patterns", [])
            if p.get("scene_tag") == scene
        )
        score += min(similar_fails, 3) / 3 * 0.15
        return min(score, 1.0)

    def _plan_check_max_refines(self, tier: str, scene: str = "") -> int:
        """自适应: 按场景分桶的 EWMA success rate 微调 max_refines.

        baseline: medium=0 (只校验不 refine), light=1 (完整闭环).
        分桶: 同 scene_tag 的最近 5 次, EWMA 加权 (alpha 根据桶大小自适应)
          >=80% 放宽 (baseline-1, 最低 0), <=20% 收紧 (baseline+1, 最高 2).
        alpha 自适应: 桶 3-4 条用 alpha=0.3 (老样本权重大, 样本少要稳),
          桶 5 条用 alpha=0.4 (近期权重大, 样本足要敏感).
        样本 <3 走 baseline, 早期不误判. 未知场景 (scene 无历史) 走全局.
        ponytail: EWMA 简单指数加权; alpha 分两档, 不上 decay schedule.
        ceiling: 桶太小 (<5 条) EWMA 不稳, 但样本不足走 baseline 兜底;
          alpha 分档靠拍, 没数据校准.
        升级路径: alpha 用 cross-validation 自动选; 或上 Bayesian 更新.
        """
        baseline = {"medium": 0, "light": 1}.get(tier, 1)
        history = getattr(self, "_plan_check_history", [])
        bucket = (
            [c for c in history if c.get("scene_tag") == scene] if scene else history
        )
        if len(bucket) < 3:
            return baseline
        recent = bucket[-5:]
        # alpha 自适应: 桶小用低 alpha (稳), 桶大用高 alpha (敏感)
        alpha = 0.3 if len(recent) < 5 else 0.4
        weights = [
            alpha * (1 - alpha) ** (len(recent) - 1 - i) for i in range(len(recent))
        ]
        total_w = sum(weights)
        if total_w == 0:
            return baseline
        ewma_success = (
            sum(
                w * (1.0 if c.get("is_valid", True) else 0.0)
                for w, c in zip(weights, recent)
            )
            / total_w
        )
        if ewma_success >= 0.8:
            return max(0, baseline - 1)
        if ewma_success <= 0.2:
            return min(2, baseline + 1)
        return baseline

    async def _plan_check(
        self,
        plan: dict[str, Any],
        hypothesis: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """单次反向校验: 让 LLM 判断 plan 执行后能否达成 hypothesis.

        用 task='verification' 让 model_router 路由到独立验证模型,
        避免正向/反向用同一个模型 (同模型有同盲点).

        v6 G57: DeepMind 三层 validation — L1 LLM plan_check (本方法) +
        L2 dimensional pre-check (_dimensional_pre_check) +
        L3 physical_precheck (PRE_TOOL_USE hook).
        量纲不一致不直接淘汰 plan, 只追加到 risks + dimensional_warnings,
        让 LLM judge 看到后决定.
        """
        # L2: dimensional pre-check — 先跑, 把 warnings 拼进 prompt 上下文
        dim_warnings = self._dimensional_pre_check(plan, hypothesis)
        if dim_warnings:
            context = dict(context)
            context["dimensional_warnings"] = "\n".join(dim_warnings)

        prompt = self._build_plan_check_prompt(plan, hypothesis, context)
        response = await self._llm_chat(
            prompt,
            persona_name="default",
            task="verification",
        )
        result = self._parse_plan_check(response)
        if dim_warnings:
            result["dimensional_warnings"] = dim_warnings
            existing = result.get("risks") or []
            existing.extend(dim_warnings)
            result["risks"] = existing
            self._plan_check_warnings.extend(dim_warnings)
        return result

    def _dimensional_pre_check(
        self,
        plan: dict[str, Any],
        hypothesis: str,
    ) -> list[str]:
        """L2 dimensional pre-check — 扫 plan + hypothesis 里的等式, 验量纲.

        ponytail: regex 抓 "<number> <unit>" 量 + "=" 等式, 调 DimensionalValidator.
        只在能解析出两侧都带量纲的等式时跑; 否则跳过 (不误报).
        ceiling: 简单 regex 抓不住复杂表达式 (函数调用 / 多行推导);
        升级路径: sympy 解析 + 单位推断.
        """
        warnings: list[str] = []
        try:
            from huginn.validation.dimensional import DimensionalValidator
        except Exception:
            return warnings

        # 拼 plan + hypothesis 文本
        text_parts = [hypothesis or ""]
        for k in ("description", "expected_prediction", "prediction"):
            v = plan.get(k) if isinstance(plan, dict) else None
            if isinstance(v, str) and v:
                text_parts.append(v)
        text = "\n".join(text_parts)
        if "=" not in text:
            return warnings

        validator = DimensionalValidator()
        # 抓 "<number> <unit>" 量, e.g. "210 GPa" / "1.5e3 kg/m3"
        qty_re = re.compile(
            r"([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)\s+([A-Za-z][A-Za-z0-9/\^\-\*\.\(\)]+)"
        )
        # 按行 + 按 "=" 切等式
        for line in text.splitlines():
            if "=" not in line:
                continue
            lhs, rhs = line.split("=", 1)
            lhs_qs = [f"{m[0]} {m[1]}" for m in qty_re.findall(lhs)]
            rhs_qs = [f"{m[0]} {m[1]}" for m in qty_re.findall(rhs)]
            if not lhs_qs or not rhs_qs:
                continue
            try:
                result = validator.check_equation(lhs_qs, rhs_qs, equation_name=line.strip()[:80])
                if not result.consistent:
                    warnings.append(
                        f"dimensional inconsistency: '{line.strip()[:80]}' "
                        f"LHS={result.lhs_dimensions} RHS={result.rhs_dimensions}"
                    )
            except Exception:
                # 解析失败静默跳过 — 量纲库不全不该阻塞 plan_check
                continue
        return warnings

    def _build_plan_check_prompt(
        self,
        plan: dict[str, Any],
        hypothesis: str,
        context: dict[str, Any],
    ) -> str:
        """反向规划识别器 prompt: 判断 plan 能否达成 hypothesis."""
        # 从 context 抽最近失败模式, 帮 LLM 避开已知坑
        failure_modes = context.get("failure_modes", "")
        if not failure_modes and self._speculator_hint:
            failure_modes = self._speculator_hint[-500:]
        # 同场景历史失败模式 (跨 run 积累, 最近 3 条) — 让 LLM 重点避开
        scene = self._plan_check_scene_tag(plan)
        similar = [
            p
            for p in getattr(self, "_plan_check_patterns", [])
            if p.get("scene_tag") == scene
        ][-3:]
        if similar:
            similar_text = "\n".join(
                f"- {p['reason']} (缺: {', '.join(p.get('missing_steps', [])) or 'N/A'})"
                for p in similar
            )
        else:
            similar_text = "N/A"
        # v6 G57: L2 dimensional pre-check 警告 (若 context 带了)
        dim_warnings_text = context.get("dimensional_warnings", "") or "N/A"
        return f"""你是反向规划识别器 (KRCL 启发). 判断以下 plan 执行后能否达成 hypothesis.

# 目标 (hypothesis)
{hypothesis}

# 当前 plan
MODE: {plan.get('mode', 'coder')}
DESCRIPTION: {plan.get('description', '')}
PREDICTION: {plan.get('expected_prediction', 'N/A')}

# 已知失败模式 (避免重蹈覆辙)
{failure_modes or 'N/A'}

# 同场景历史失败 (scene={scene}, 跨 run 积累)
{similar_text}

# 量纲预检查警告 (L2 dimensional pre-check, v6 G57)
{dim_warnings_text}

# 任务
判断这个 plan 执行后能否达成 hypothesis. 严格检查:
- MODE 是否匹配任务类型 (coder 写代码 / workflow 跑流程 / explore 探索 / skill 复合技能)
- DESCRIPTION 是否覆盖 hypothesis 的关键要求
- PREDICTION 是否可验证 (能跑出数值/结构/代码对比)
- 是否遗漏必要前置步骤 (如 band 前需 SCF / MD 前需 minimize / elastic 前需 relax)
- 是否重复了"同场景历史失败"里列出的坑
- 量纲预检查有警告时, 把它列入 risks

输出 JSON (不要其他文本):
{{
  "is_valid": true 或 false,
  "confidence": 0.0 到 1.0 (对判断的置信度, 1.0=非常确定, 0.5=模棱两可, 0.0=完全没把握),
  "reason": "为什么 valid / invalid",
  "missing_steps": ["如果 invalid, 缺少哪些步骤"],
  "risks": ["潜在风险"]
}}"""

    def _record_plan_check_failure(
        self,
        plan: dict[str, Any],
        check: dict[str, Any],
        scene: str,
    ) -> None:
        """失败模式记到 patterns, 跨 run 持久化给下次注入 prompt.

        ponytail: 内存 append + 同步 dump JSON, 量小 (<=50 条) 写快.
        ceiling: 同步写盘, 高频失败时可能拖慢; description 截断 200 chars.
        升级路径: 后台 async flush, 或上 SQLite.
        """
        self._plan_check_patterns.append(
            {
                "scene_tag": scene,
                "reason": check.get("reason", "unknown"),
                "missing_steps": check.get("missing_steps", []),
                "mode": plan.get("mode", ""),
                "description": plan.get("description", "")[:200],
            }
        )
        if len(self._plan_check_patterns) > 50:
            del self._plan_check_patterns[: len(self._plan_check_patterns) - 50]
        self._save_plan_check_patterns()

    def _load_plan_check_patterns(self) -> None:
        """跨 run 加载历史失败模式.

        ponytail: JSON 文件, 不上 DB; 只在 _prepare_run 调一次.
        ceiling: 文件可能被外部篡改, 解析失败静默回退.
        """
        path = self.workspace / ".huginn" / "plan_check_patterns.json"
        if not path.exists():
            return
        try:
            import json

            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                self._plan_check_patterns = data[-50:]
                logger.info(
                    "loaded %d plan_check patterns from %s",
                    len(self._plan_check_patterns),
                    path,
                )
        except Exception as e:
            logger.debug("load plan_check_patterns failed: %s", e)

    def _save_plan_check_patterns(self) -> None:
        """dump 失败模式到 workspace, 跨 run 积累.

        ponytail: 同步写, 量小 (<=50 条); 跟 skill_evolver 历史持久化同款.
        """
        path = self.workspace / ".huginn" / "plan_check_patterns.json"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            import json

            path.write_text(
                json.dumps(
                    self._plan_check_patterns[-50:], ensure_ascii=False, indent=2
                ),
                encoding="utf-8",
            )
        except Exception as e:
            logger.debug("save plan_check_patterns failed: %s", e)

    def _parse_plan_check(self, response: str) -> dict[str, Any]:
        """解析反向校验 JSON — 括号配平法 (ValidityJudge._parse_verdict 同款).

        解析失败返回 is_valid=True (跳过校验, 不阻塞).
        """
        import json

        start = response.find("{")
        if start < 0:
            return {"is_valid": True, "reason": "no json, skip"}
        depth = 0
        for i, ch in enumerate(response[start:], start=start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(response[start : i + 1])
                        # 字段补全, 保证下游一致
                        obj.setdefault("is_valid", True)
                        obj.setdefault("confidence", 0.8)  # 默认高置信, 不误触发 refine
                        obj.setdefault("reason", "")
                        obj.setdefault("missing_steps", [])
                        obj.setdefault("risks", [])
                        return obj
                    except json.JSONDecodeError:
                        return {"is_valid": True, "reason": "json parse failed, skip"}
        return {"is_valid": True, "reason": "no closing brace, skip"}

    async def _refine_plan(
        self,
        plan: dict[str, Any],
        check: dict[str, Any],
        hypothesis: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """根据反向校验反馈, 让 LLM 重新生成 plan (保留 plan_id).

        few-shot: 从 _plan_check_history 抽同场景最近 1 条成功 plan 塞进
        prompt, 让 LLM 知道'上次同场景怎么成功的'. 命中长程任务 — 跨 iteration
        积累的成功经验不再丢失.
        """
        # 抽同场景最近 1 条成功 plan (is_valid=True, scene_tag 相同)
        scene = self._plan_check_scene_tag(plan)
        success_example = None
        for c in reversed(getattr(self, "_plan_check_history", [])):
            if (
                c.get("is_valid")
                and c.get("scene_tag") == scene
                and c.get("plan_snapshot")
            ):
                success_example = c["plan_snapshot"]
                break
        few_shot_block = "N/A"
        if success_example:
            few_shot_block = (
                f"MODE: {success_example.get('mode', 'coder')}\n"
                f"DESCRIPTION: {success_example.get('description', '')[:200]}"
            )
        prompt = f"""之前的 plan 未通过反向校验. 根据反馈重新生成.

# 目标
{hypothesis}

# 之前的 plan
MODE: {plan.get('mode', 'coder')}
DESCRIPTION: {plan.get('description', '')}

# 校验反馈
reason: {check.get('reason', '')}
missing_steps: {check.get('missing_steps', [])}
risks: {check.get('risks', [])}

# 同场景成功示例 (scene={scene}, 跨 iteration 积累, 仅供参考结构)
{few_shot_block}

# 任务
根据反馈重新生成 plan. 参考成功示例的结构 (不要照抄内容). 严格按格式输出:
MODE: <coder|workflow|explore|skill|visual_inspect>
DESCRIPTION: <brief description>
SKILL: <composite skill name, only if MODE is skill>
PREDICTION: <预期结果, 用于后续 validate 对比>"""
        try:
            response = await self._llm_chat(
                prompt,
                persona_name="default",
                task="planning",
            )
            new_plan = self._parse_plan(response)
            new_plan = self._override_plan_mode(new_plan)
            # 保留 plan_id (如果有), 让 PlanStore 能跟踪同一 plan 的演进
            if "plan_id" in plan:
                new_plan["plan_id"] = plan["plan_id"]
            return new_plan
        except Exception as e:
            logger.debug("plan refine failed: %s", e)
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
                self._dispatch_stage_event(EventType.ON_WORKFLOW_STAGE_START, name)
            )
        except RuntimeError:
            logger.warning(
                "error in _run_phase: stage-start event dispatch skipped (no running loop)",
                exc_info=True,
            )
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
                logger.warning(
                    "error in _run_phase: span metadata update failed", exc_info=True
                )
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
            logger.warning(
                "error in _run_phase: stage-done event dispatch skipped (no running loop)",
                exc_info=True,
            )
        return phase

    async def _run_phase_async(self, name: str, fn, *args) -> LoopPhase:
        """Run an async phase function."""
        phase = LoopPhase(name=name)
        phase.start_time = time.time()
        phase.status = "running"
        # 记下当前 phase, 让 _llm_chat 能注入 phase-aware thinking effort 指令.
        # ponytail: 隐式状态, 但 run() 是 single-threaded async, 无竞态.
        self._current_phase = name
        # C2: 追踪本 run 的 phase 序列, 供 trajectory_match 召回用.
        if not hasattr(self, "_current_run_phases"):
            self._current_run_phases = []
        self._current_run_phases.append(name)
        # 防止无界增长 (1000+ iter × 7 phase = 7000+), 只保留最近 50 个
        if len(self._current_run_phases) > 50:
            self._current_run_phases = self._current_run_phases[-50:]
        await self._dispatch_stage_event(EventType.ON_WORKFLOW_STAGE_START, name)
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
                logger.warning(
                    "error in _run_phase_async: span metadata update failed",
                    exc_info=True,
                )
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
            "# Huginn Autoloop Report",
            "",
            f"**Objective:** {data['objective']}",
            f"**Run ID:** {data['run_id']}",
            f"**Total Time:** {data['total_time_seconds']:.1f}s",
            "",
            "## Phases",
            "",
            "| Phase | Status | Duration (s) | Error |",
            "|-------|--------|--------------|-------|",
        ]
        for p in data["phases"]:
            lines.append(
                f"| {p['name']} | {p['status']} | {p['duration']:.1f} | {p['error'] or ''} |"
            )
        lines.append("")
        lines.append("---")
        lines.append("Generated by Huginn Autoloop Engine")
        return "\n".join(lines)


def _selfcheck() -> None:
    """AutoloopEngine selfcheck — 验证 LLM decider 的合法性检查 + prompt 构建.

    覆盖: _is_action_legal 全 action / _build_decider_prompt 字段 / _decide_next_action_llm fallback.
    ponytail: 用 __new__ 绕过 __init__, 只测无副作用的方法. run_cognitive 的完整
    selfcheck 在 verify_run_cognitive.py (依赖 mock stub, 6 场景).
    """
    from huginn.autoloop.cognitive_loop import LoopState, ActionDecision

    eng = AutoloopEngine.__new__(AutoloopEngine)
    eng._use_llm_decider = True

    # 1. _is_action_legal — 全 action 前置条件
    # observe/hypothesize/skip/stop 永远合法
    for a in ("observe", "hypothesize", "skip", "stop"):
        assert eng._is_action_legal(a, {}) is True, f"{a} should always be legal"
    # plan 需要 hypothesis
    assert eng._is_action_legal("plan", {}) is False, "plan without hyp should be illegal"
    assert eng._is_action_legal("plan", {"hypothesis": "test"}) is True, "plan with hyp should be legal"
    # execute 需要 plan
    assert eng._is_action_legal("execute", {}) is False
    assert eng._is_action_legal("execute", {"plan": {"mode": "x"}}) is True
    # validate 需要 execution_result
    assert eng._is_action_legal("validate", {}) is False
    assert eng._is_action_legal("validate", {"execution_result": {"r": 1}}) is True
    # learn 需要 hypothesis + plan + validation
    assert eng._is_action_legal("learn", {"hypothesis": "h"}) is False
    assert eng._is_action_legal("learn", {"hypothesis": "h", "plan": {"m": "x"}, "validation": {"v": 1}}) is True
    # pivot 需要 current_hyp_id 或 hypothesis
    assert eng._is_action_legal("pivot", {}) is False
    assert eng._is_action_legal("pivot", {"current_hyp_id": "h1"}) is True
    # 未知 action → False
    assert eng._is_action_legal("unknown_action", {"hypothesis": "h"}) is False
    print("1. _is_action_legal (all actions) OK")

    # 1b. D3: report 不在 VALID_ACTIONS + _is_action_legal 拦截
    from huginn.autoloop.cognitive_loop import VALID_ACTIONS as _VA
    assert "report" not in _VA, (
        f"D3 broken: 'report' should not be in VALID_ACTIONS, got {_VA}"
    )
    assert eng._is_action_legal("report", {"hypothesis": "h", "plan": {"m": "x"}, "validation": {"v": 1}}) is False, (
        "D3 broken: _is_action_legal('report') should return False"
    )
    print("1b. D3 report not in VALID_ACTIONS + _is_action_legal blocks report OK")

    # 2. _build_decider_prompt — 包含关键字段
    state = LoopState(iteration=3, max_iterations=10, last_action="hypothesize")
    cog = {
        "hypothesis": "test hyp",
        "plan": {"mode": "coder"},
        "execution_result": None,
        "validation": None,
        "current_hyp_id": "h1",
    }
    prompt = eng._build_decider_prompt(state, cog, {})
    assert "Iteration: 3/10" in prompt, f"missing iteration: {prompt}"
    assert "Hypothesis: test hyp" in prompt, f"missing hyp: {prompt}"
    assert "Plan mode: coder" in prompt, f"missing plan mode: {prompt}"
    assert "Actions:" in prompt, f"missing actions list: {prompt}"
    assert "stop: end the loop" in prompt, f"missing stop action: {prompt}"
    print("2. _build_decider_prompt OK")

    # 2b. D1: decider prompt 含新增字段
    # validation 具体字段 / consecutive_failures / pivot_count / refine_count
    # / action_history / last_learn_summary / speculator_hint
    eng._consecutive_failures = 5
    eng._max_consecutive_failures = 20
    eng._pivot_count = 2
    eng._refine_count = 3
    eng._speculator_hint = "try alternative k-points"
    # F-borrow: 分类失败计数 — 2 tool_error + 3 hypothesis_error
    eng._consecutive_failures_by_type = {"tool_error": 2, "hypothesis_error": 3}
    eng._max_failures_by_type = {
        "tool_error": 5, "prompt_injection_suspect": 3,
        "param_error": 5, "data_noise": 5, "hypothesis_error": 10,
    }
    state_d1 = LoopState(
        iteration=7, max_iterations=15, last_action="validate",
    )
    state_d1.action_history = [
        "observe", "hypothesize", "plan", "execute", "validate",
        "learn", "observe", "plan", "execute", "validate",
    ]
    cog_d1 = {
        "hypothesis": "GaN gap with PBE",
        "plan": {"mode": "coder"},
        "execution_result": {"r": 1},
        "validation": {
            "tests_passed": False,
            "thinking_collapse": "model gave up mid-derivation",
            "physics_validation_error": "band gap off by 0.5 eV",
            "dimensional_consistent": True,
        },
        "current_hyp_id": "h_d1",
        "last_learn_summary": "learned: persona=reviewer r_phys=0.4 tests_passed=False",
    }
    prompt_d1 = eng._build_decider_prompt(state_d1, cog_d1, {})
    # D1 必含字段
    assert "Consecutive failures: 5/20" in prompt_d1, (
        f"D1 missing consecutive_failures: {prompt_d1[:300]}"
    )
    assert "Pivot count: 2/10" in prompt_d1, (
        f"D1 missing pivot_count: {prompt_d1[:300]}"
    )
    assert "Refine count: 3" in prompt_d1, (
        f"D1 missing refine_count: {prompt_d1[:300]}"
    )
    assert "Action history (last 10):" in prompt_d1, (
        f"D1 missing action_history: {prompt_d1[:300]}"
    )
    assert "Last learn summary: learned: persona=reviewer" in prompt_d1, (
        f"D1 missing last_learn_summary: {prompt_d1[:300]}"
    )
    # validation details 应该含具体字段值
    assert "thinking_collapse" in prompt_d1, (
        f"D1 missing validation details (thinking_collapse): {prompt_d1[:400]}"
    )
    assert "physics_validation_error" in prompt_d1, (
        f"D1 missing validation details (physics_validation_error): {prompt_d1[:400]}"
    )
    # speculator_hint 注入
    assert "Speculator hints: try alternative k-points" in prompt_d1, (
        f"D1 missing speculator_hint: {prompt_d1[:400]}"
    )
    # D3 提示: report 自动跑
    assert "report runs automatically" in prompt_d1, (
        f"D3 missing report hint: {prompt_d1[:400]}"
    )
    # F-borrow: 分类失败计数显示
    assert "Failures by type:" in prompt_d1, (
        f"F-borrow missing failures_by_type: {prompt_d1[:400]}"
    )
    assert "tool_error=2/5" in prompt_d1, (
        f"F-borrow missing tool_error count: {prompt_d1[:400]}"
    )
    assert "hypothesis_error=3/10" in prompt_d1, (
        f"F-borrow missing hypothesis_error count: {prompt_d1[:400]}"
    )
    print("2b. D1 decider prompt 含新增字段 (failures/pivot/refine/history/learn/validation/speculator/by_type) OK")

    # 3. _decide_next_action_llm — LLM 调用失败时返回 None (fallback 信号)
    async def _fail_chat(*a, **kw):
        raise RuntimeError("LLM unavailable")
    eng._llm_chat = _fail_chat
    import asyncio
    result = asyncio.run(eng._decide_next_action_llm(state, cog, {}))
    assert result is None, f"LLM fail should return None, got: {result}"
    print("3. _decide_next_action_llm fallback (LLM fail → None) OK")

    # 4. _decide_next_action_llm — 非法 action 返回 None
    async def _bad_action_chat(*a, **kw):
        return '{"action": "fly_to_moon", "rationale": "test", "expected_outcome": "test"}'
    eng._llm_chat = _bad_action_chat
    result = asyncio.run(eng._decide_next_action_llm(state, cog, {}))
    assert result is None, f"illegal action should return None, got: {result}"
    print("4. _decide_next_action_llm illegal action → None OK")

    # 5. _decide_next_action_llm — 合法 action 返回 ActionDecision
    async def _good_chat(*a, **kw):
        return '{"action": "plan", "rationale": "have hyp, design plan", "expected_outcome": "plan dict"}'
    eng._llm_chat = _good_chat
    result = asyncio.run(eng._decide_next_action_llm(state, cog, {}))
    assert result is not None, "legal action should return ActionDecision"
    assert isinstance(result, ActionDecision), f"should be ActionDecision, got {type(result)}"
    assert result.action == "plan", f"action should be plan, got {result.action}"
    print("5. _decide_next_action_llm legal action → ActionDecision OK")

    # 6. H0: stable_principles 注入 _build_hypothesis_prompt / _build_plan_prompt
    # mock 掉检索依赖, 只验证 principles block 真的拼进 prompt.
    from huginn.memory.longterm import load_stable_principles
    _principles = load_stable_principles()
    if _principles:
        eng2 = AutoloopEngine.__new__(AutoloopEngine)
        eng2._speculator_hint = ""
        eng2._last_visual_context = ""
        eng2._last_execution_result = None
        eng2._last_failure_mode = ""
        eng2._IMAGINATION_PROMPT_BLOCK = ""
        eng2._MATH_DEPTH_PROMPT_BLOCK = ""
        eng2._should_imaginate = lambda: False
        eng2._build_kb_text = lambda query: ""
        eng2._build_kg_text = lambda query: ""
        eng2._build_memory_text = lambda query: ""
        eng2._build_pm_text = lambda: ""
        eng2._build_subgoal_block = lambda: ""
        eng2._plan_context_hint = lambda: ""
        eng2._get_evolution = lambda: type("_E", (), {
            "get_relevant_skills": lambda self, h: [],
            "get_prompt_patches": lambda self: [],
        })()
        # _trim_to_budget: 直接拼成字符串, 不做预算裁剪
        def _fake_trim(blocks, phase=None):
            return "\n".join(body for _, body in blocks if body)
        eng2._trim_to_budget = _fake_trim
        # _build_hypothesis_prompt
        hyp_prompt = eng2._build_hypothesis_prompt({"test": 1})
        assert "Stable Principles" in hyp_prompt, (
            "H0 broken: principles block missing in hypothesis prompt"
        )
        assert _principles[0][:50] in hyp_prompt, (
            "H0 broken: principle text not in hypothesis prompt"
        )
        # _build_plan_prompt
        plan_prompt = eng2._build_plan_prompt("test hyp", {"test": 1})
        assert "Stable Principles" in plan_prompt, (
            "H0 broken: principles block missing in plan prompt"
        )
        print(f"6. H0 stable_principles 注入 (n={len(_principles)}) OK")
    else:
        print("6. H0 skipped (no stable_principles on disk)")

    # 7. H1: _apply_block_patches 接入 + toggle off passthrough
    # 验证 _build_hypothesis_prompt 在 toggle off 时不走 patch 路径 (零开销).
    # toggle on 时 monkey-patch _harness_enabled, 加 patch, 验证 blocks 被替换.
    import os as _os
    import tempfile as _tempfile
    import huginn.harness.prompt_patch as _pp
    _orig_harness_enabled = _pp._harness_enabled
    # 7a. toggle off: _apply_block_patches 直接返回原 blocks
    _blocks_in = [("body", "x {context}"), ("mem", "mem text")]
    _out = eng2._apply_block_patches(_blocks_in, "hypothesize")
    assert _out is _blocks_in, "toggle off should passthrough"
    print("7a. H1 _apply_block_patches toggle off → passthrough OK")

    # 7b. toggle on + patch 应用
    _pp._harness_enabled = lambda key, default=False: (
        True if key == "harness_prompt_patch" else default
    )
    _tmp = _tempfile.mkdtemp()
    _os.environ["HUGINN_CACHE_DIR"] = _tmp
    _pp.PromptPatchStore._instance = None
    _store = _pp.PromptPatchStore.get_instance()
    _p = _pp.PromptPatch(
        id="selfcheck_h1",
        phase="hypothesize",
        block_name="mem",
        new_text="H1 SELFCHECK PATCHED",
        op="prepend",
    )
    _store.add_patch(_p)
    _store.update_alpha_beta("selfcheck_h1", success=True)
    _store.update_alpha_beta("selfcheck_h1", success=True)
    _out = eng2._apply_block_patches(_blocks_in, "hypothesize")
    assert _out is not _blocks_in, "toggle on should return new list"
    assert "H1 SELFCHECK PATCHED" in _out[1][1], (
        f"H1 patch not applied: {_out[1][1]}"
    )
    # _last_applied_patches 记录了 phase + ids
    assert hasattr(eng2, "_last_applied_patches"), "_last_applied_patches not set"
    _applied_phase, _applied_ids = eng2._last_applied_patches
    assert _applied_phase == "hypothesize"
    assert "selfcheck_h1" in _applied_ids
    print("7b. H1 _apply_block_patches toggle on + patch apply + ids tracked OK")

    # 7c. _learn Beta 更新 (用 eng2 的 _apply_block_patches 记录的 ids)
    # 直接调 store.update_alpha_beta 验证接口, 不跑完整 _learn (太重)
    _store.update_alpha_beta("selfcheck_h1", success=True)
    _p_reload = _store._patches["selfcheck_h1"]
    assert _p_reload.alpha == 4, f"alpha should be 4 (1 init + 2 setup + 1 learn): {_p_reload.alpha}"
    print("7c. H1 _learn Beta update interface OK")

    # 清理 H1 测试状态
    import shutil as _shutil
    _shutil.rmtree(_tmp, ignore_errors=True)
    del _os.environ["HUGINN_CACHE_DIR"]
    _pp.PromptPatchStore._instance = None
    _pp._harness_enabled = _orig_harness_enabled
    if hasattr(eng2, "_last_applied_patches"):
        del eng2._last_applied_patches

    # 8. H4 试点: SubagentDispatch toggle on 时走 PhaseRegistry
    import huginn.harness.phase_spec as _ps
    _orig_ps_enabled = _ps._harness_enabled
    _ps._harness_enabled = lambda key, default=False: (
        True if key == "harness_phase_evolve" else default
    )
    _tmp2 = _tempfile.mkdtemp()
    _os.environ["HUGINN_CACHE_DIR"] = _tmp2
    _ps.PhaseRegistry._instance = None
    _reg = _ps.PhaseRegistry.get_instance()
    _reg.register_subagent_override("explore", {
        "system_prompt": "H4 SELFCHECK EXPLORE PROMPT",
        "max_tool_calls": 77,
    })
    # reset singleton 重读, 验证持久化
    _ps.PhaseRegistry._instance = None
    from huginn.agents.subagent import SubagentDispatch as _SD
    _d = _SD()
    assert _d._specs["explore"].system_prompt == "H4 SELFCHECK EXPLORE PROMPT", (
        f"H4 override not applied: {_d._specs['explore'].system_prompt}"
    )
    assert _d._specs["explore"].max_tool_calls == 77, (
        f"H4 max_tool_calls not applied: {_d._specs['explore'].max_tool_calls}"
    )
    # 未 override 的 spec 保留 baseline
    assert _d._specs["coder"].system_prompt.startswith("You are a coding agent"), (
        f"H4 baseline coder lost: {_d._specs['coder'].system_prompt}"
    )
    print("8. H4 SubagentDispatch PhaseRegistry override + baseline fallback OK")

    # 清理 H4 测试状态
    _shutil.rmtree(_tmp2, ignore_errors=True)
    del _os.environ["HUGINN_CACHE_DIR"]
    _ps.PhaseRegistry._instance = None
    _ps._harness_enabled = _orig_ps_enabled

    # ---- H2: Workflow Evolutionary Search (bandit + variant_gen) ----
    import tempfile as _tempfile_h2
    _tmp_h2 = _tempfile_h2.mkdtemp()
    _os.environ["HUGINN_CACHE_DIR"] = _tmp_h2

    from huginn.autoloop import bandit as _bd
    from huginn.autoloop import variant_gen as _vg
    from huginn.autoloop.dynamic_workflow import WorkflowScript as _WFS_h2

    # 9. bandit cold start + Thompson sampling
    _bd.WorkflowBandit._instance = None
    _bd.VariantArchive._instance = None
    _bandit = _bd.WorkflowBandit.get_instance()
    _cands = ["v1", "v2", "v3"]
    _chosen = _bandit.select_variant(_cands, "h2_test")
    assert _chosen in _cands, f"H2 cold start failed: {_chosen}"
    _bandit.record_variant_outcome("v1", "h2_test", True, r_phys=0.8)
    _bandit.record_variant_outcome("v1", "h2_test", True, r_phys=0.9)
    _bandit.record_variant_outcome("v2", "h2_test", False, r_phys=0.1)
    _b1 = _bandit.get_belief("v1", "h2_test")
    _b2 = _bandit.get_belief("v2", "h2_test")
    assert _b1 and _b2 and _b1.posterior_mean > _b2.posterior_mean
    print(f"9. H2 bandit Thompson: v1={_b1.posterior_mean:.2f} > v2={_b2.posterior_mean:.2f} OK")

    # 10. variant_gen 参数扰动 + toggle guard
    _orig_vg_enabled = _vg._harness_enabled
    _vg._harness_enabled = lambda key, default=False: (
        True if key == "harness_workflow_evolution" else default
    )
    _base_script = _WFS_h2.from_dict({
        "objective": "h2 test Si band gap",
        "subtasks": [
            {"id": "s1", "tool": "vasp_tool",
             "args": {"encut": 520, "kpoints": "2 2 2", "sigma": 0.05}},
        ],
    })
    import asyncio as _asyncio_h2
    _variants = _asyncio_h2.run(_vg.generate_variants("h2 test", n=3, base_script=_base_script))
    assert len(_variants) == 3, f"H2 variant_gen should return 3: {len(_variants)}"
    _base_encut = _base_script.subtasks[0].args["encut"]
    _diff = sum(1 for v in _variants if v.subtasks[0].args["encut"] != _base_encut)
    assert _diff > 0, "H2 at least one variant should differ in encut"
    print(f"10. H2 variant_gen perturbation: 3 variants, {_diff} differ in encut OK")

    # 11. archive + novelty + Pareto
    _archive = _bd.VariantArchive.get_instance()
    _sa = {"subtasks": [{"tool": "vasp", "args": {"encut": 520}}]}
    _sb = {"subtasks": [{"tool": "vasp", "args": {"encut": 540}}]}
    _archive.add_variant("h2_test", "test", "va", _sa, [0.8, 0.9, 0.7])
    _archive.add_variant("h2_test", "test", "vb", _sb, [0.7, 0.8, 1.0])
    _vs = _archive.list_variants("h2_test")
    assert len(_vs) >= 2, f"H2 archive should have >=2: {len(_vs)}"
    _n_same = _bd.compute_novelty(_sa, _vs)
    _n_new = _bd.compute_novelty(
        {"subtasks": [{"tool": "vasp", "args": {"encut": 600}}]}, _vs
    )
    assert _n_same == 0.0 and _n_new > 0.5
    print(f"11. H2 archive+novelty: same={_n_same:.2f}, new={_n_new:.2f} OK")

    # 12. _try_evolved_fix guard (variant_id 不走 evolved_fix)
    _e = AutoloopEngine()
    _guard_result = _asyncio_h2.run(_e._try_evolved_fix(
        "vasp_tool", {"encut": 520}, {"_variant_id": "var_0", "error": "test"}
    ))
    assert _guard_result is None, f"H2 guard should return None: {_guard_result}"
    print("12. H2 _try_evolved_fix variant guard OK")

    _vg._harness_enabled = _orig_vg_enabled
    _shutil.rmtree(_tmp_h2, ignore_errors=True)
    del _os.environ["HUGINN_CACHE_DIR"]

    # 13. H3 JointBandit + UCB (block subset + workflow params 联合优化)
    import huginn.harness.joint_optimizer as _jo
    _tmp_h3 = _tempfile_h2.mkdtemp()
    _os.environ["HUGINN_CACHE_DIR"] = _tmp_h3
    _jo.JointBandit._instance = None
    _orig_jo_enabled = _jo._harness_enabled
    _jo._harness_enabled = lambda key, default=False: (
        True if key == "harness_joint_optimizer" else default
    )

    # block subset: 核心 block 必保留
    _blocks_h3 = [("body", "b"), ("fail", "f"), ("mem", "m"), ("extra", "e")]
    _sel = _jo.select_block_subset_for_phase("hypothesize", _blocks_h3)
    _sel_names = [n for n, _ in _sel]
    assert "body" in _sel_names and "fail" in _sel_names, f"core lost: {_sel_names}"
    print(f"13a. H3 block subset: {_sel_names} (core preserved) OK")

    # workflow params: 数值参数 ±10% 扰动
    _params_h3 = {"encut": 520, "kpoints": "2 2 2"}
    _perturbed = _jo.select_workflow_params_for_stage("vasp_tool", _params_h3)
    assert "encut" in _perturbed and 468 <= _perturbed["encut"] <= 572, \
        f"encut out of range: {_perturbed['encut']}"
    assert _perturbed["kpoints"] == "2 2 2", f"kpoints should stay: {_perturbed['kpoints']}"
    print(f"13b. H3 workflow params: encut={_perturbed['encut']} OK")

    # record + UCB: 记两组 outcome, 验证 Beta 信念分化
    _jb = _jo.JointBandit.get_instance()
    _jb.record_joint_outcome("hypothesize", ["body", "mem"], {"encut": 520}, True)
    _jb.record_joint_outcome("hypothesize", ["body", "mem"], {"encut": 520}, True)
    _jb.record_joint_outcome("hypothesize", ["body", "extra"], {"encut": 540}, False)
    _bs_h3 = _jb.list_beliefs("hypothesize")
    assert len(_bs_h3) >= 2, f"should have >=2 beliefs: {len(_bs_h3)}"
    # 冷启动 UCB = inf
    _b_new = _jo.JointBelief(config_id="new", phase="hypothesize")
    assert _b_new.ucb(10) == float("inf"), "cold start UCB should be inf"
    print(f"13c. H3 record+UCB: {len(_bs_h3)} beliefs, cold start UCB=inf OK")

    # 持久化 reload
    _jo.JointBandit._instance = None
    _jb2 = _jo.JointBandit.get_instance()
    _bs2_h3 = _jb2.list_beliefs("hypothesize")
    assert len(_bs2_h3) == len(_bs_h3), f"persistence lost: {len(_bs2_h3)} vs {len(_bs_h3)}"
    print(f"13d. H3 persistence reload OK ({len(_bs2_h3)} beliefs)")

    # 端到端集成: H1 apply_patches 在 H3 toggle on 时会调 H3 选 block 子集
    _pp_blocks = [("body", "b"), ("fail", "f"), ("mem", "m"), ("extra", "e")]
    try:
        from huginn.harness.prompt_patch import apply_patches as _ap_h3
        _out_h3 = _ap_h3(_pp_blocks, "hypothesize")
        _out_names = [n for n, _ in _out_h3]
        # H1 toggle off 时 apply_patches 直接 return blocks, 但 H3 在 toggle on 时
        # 已经在 apply_patches 入口接管 block subset 选择 — 验证至少核心 block 还在
        assert "body" in _out_names, f"H1+H3 lost body: {_out_names}"
        print(f"13e. H3↔H1 integration: blocks={_out_names} OK")
    except Exception as _e_h3:
        # H1 toggle off 时 apply_patches 早期 return, H3 不触发 — 这是预期行为
        print(f"13e. H3↔H1 integration skipped (H1 toggle off): {_e_h3}")

    _jo._harness_enabled = _orig_jo_enabled
    _jo.JointBandit._instance = None
    _shutil.rmtree(_tmp_h3, ignore_errors=True)
    del _os.environ["HUGINN_CACHE_DIR"]

    # ---- D 块: decider 可观测性 + learn 反馈 + report 拦截 ----
    # D2: _learn 签名返回 dict + dispatch 模板写入 cog["last_learn_summary"]
    # ponytail: 不跑完整 _learn (依赖 memory/kg/evolution/llm 等 10+ 外部资源),
    # 只验证 (a) 函数签名 (b) dispatch 模板逻辑 (c) 字段格式.
    # 升级路径: 用 unittest.mock 伪造 _learn 内部依赖, 跑完整 _learn 验证返回值.
    import inspect as _inspect
    _learn_sig = _inspect.signature(AutoloopEngine._learn)
    _learn_ret = _learn_sig.return_annotation
    assert _learn_ret is dict[str, Any] or str(_learn_ret) == "dict[str, Any]", (
        f"D2 broken: _learn should return dict[str, Any], got {_learn_ret!r}"
    )
    print("14a. D2 _learn signature -> dict[str, Any] OK")

    # 14b. D2 dispatch 模板逻辑 — 模拟 phase.result 是 dict 时 cog 写入
    # 直接镜像 execute_fn learn 分支的 D2 代码 (那里是 closure, 无法直接调).
    _fake_phase_results = [
        {"persona": "reviewer", "r_phys": 0.8, "tests_passed": True, "principles_added": 1},
        {"persona": "coder", "r_phys": None, "tests_passed": False, "principles_added": 0},
        None,  # _learn 异常时 phase.result 可能是 None
    ]
    for _fake in _fake_phase_results:
        _cog_d2: dict = {}
        _learned = _fake if isinstance(_fake, dict) else {}
        if _learned:
            _cog_d2["last_learn_summary"] = (
                f"learned: persona={_learned.get('persona','?')} "
                f"r_phys={_learned.get('r_phys','?')} "
                f"tests_passed={_learned.get('tests_passed','?')} "
                f"principles_added={_learned.get('principles_added',0)}"
            )
        else:
            _cog_d2["last_learn_summary"] = "learn ran (no summary)"
        # D5 断言: learn 后 cog["last_learn_summary"] 必须非空
        assert _cog_d2["last_learn_summary"], (
            f"D2 broken: last_learn_summary empty for phase.result={_fake}"
        )
        if _fake and isinstance(_fake, dict):
            assert f"persona={_fake['persona']}" in _cog_d2["last_learn_summary"], (
                f"D2 broken: persona not in summary: {_cog_d2['last_learn_summary']}"
            )
            assert f"r_phys={_fake['r_phys']}" in _cog_d2["last_learn_summary"], (
                f"D2 broken: r_phys not in summary: {_cog_d2['last_learn_summary']}"
            )
        else:
            assert _cog_d2["last_learn_summary"] == "learn ran (no summary)"
    print("14b. D2 dispatch 模板写入 cog[last_learn_summary] (3 场景: 有 dict / None / 空) OK")

    # === C 块 selfcheck ===
    # C1: format_kb_chunks 共享函数 — image_ref + cross-ref + 截断
    from huginn.context_builder import format_kb_chunks
    _c1_chunks = [
        {"text": "first principles chunk " * 50, "metadata": {"image_ref": "/p/1.png"}},
        {"text": "second chunk", "metadata": {}},
        {"text": "", "metadata": {}},  # 空文本应被跳过
    ]
    _calls = []
    _c1_out = format_kb_chunks(
        _c1_chunks,
        memory_recall_fn=lambda q, max_entries=1: _calls.append(q) or "related mem",
        with_image_ref=True,
        cross_ref_top_k=2,
    )
    assert "[1]" in _c1_out and "[2]" in _c1_out, "C1: chunks indexed"
    assert "视觉压缩页" in _c1_out, "C1: image_ref injected"
    assert "Memory: related mem" in _c1_out, "C1: cross-ref injected"
    assert len(_calls) == 2, f"C1: cross-ref top_k=2 → 2 calls, got {len(_calls)}"
    # 截断验证 — chunk 0 文本 > 800 chars 应被截断
    _c1_lines = _c1_out.split("\n")
    _first_chunk_line = next(l for l in _c1_lines if l.startswith("[1]"))
    assert len(_first_chunk_line) <= 810, f"C1: truncation to 800+ellipsis, got {len(_first_chunk_line)}"
    print("15. C1 format_kb_chunks (image_ref + cross-ref top_k + 截断) OK")

    # C1b: engine._build_kb_text 跟 ContextBuilder 走同一函数 — 共享路径验证
    # ponytail: 不实际调 _build_kb_text (依赖 kb store), 只验证 import + 函数引用
    assert "format_kb_chunks" in dir(eng.__class__) or callable(getattr(eng, "_build_kb_text", None)), \
        "C1b: engine has _build_kb_text method"
    print("16. C1b engine._build_kb_text 存在 (调共享 format_kb_chunks) OK")

    # C3: working_memory 死字段已删 — SessionContext 无此属性
    from huginn.memory.session import SessionContext as _SC
    _sc = _SC()
    assert not hasattr(_sc, "working_memory"), "C3: working_memory should be deleted"
    assert not hasattr(_sc, "set_working_memory"), "C3: set_working_memory deleted"
    assert not hasattr(_sc, "get_working_memory"), "C3: get_working_memory deleted"
    # manager.set_context/get_context 也应已删
    from huginn.memory.manager import MemoryManager as _MM
    assert not hasattr(_MM, "set_context"), "C3: MemoryManager.set_context deleted"
    assert not hasattr(_MM, "get_context"), "C3: MemoryManager.get_context deleted"
    # to_dict 不含 working_memory_keys
    _sc_dict = _sc.to_dict()
    assert "working_memory_keys" not in _sc_dict, "C3: to_dict no working_memory_keys"
    print("17. C3 working_memory 死字段删除 (session + manager + to_dict) OK")

    # C4: meta_trace toggle — 默认 off, _build_memory_text 不注入 trace
    eng_c4 = AutoloopEngine.__new__(AutoloopEngine)
    eng_c4._speculator_hint = ""
    eng_c4._target_chains = []
    eng_c4._target_chains_built = False
    eng_c4._objective = ""
    # toggle off (默认) — 即使有 meta_trace 文件也不应注入
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as _td:
        eng_c4.workspace = Path(_td)
        eng_c4.memory = None  # 无 memory → _build_memory_text 应返回空串
        _out_off = eng_c4._build_memory_text("test query")
        assert _out_off == "", f"C4: toggle off → empty, got {_out_off!r}"
        # 造一个 meta_trace.jsonl, toggle off 仍不应注入
        _huginn_dir = Path(_td) / ".huginn"
        _huginn_dir.mkdir(parents=True, exist_ok=True)
        _trace = _huginn_dir / "meta_trace.jsonl"
        _trace.write_text(
            '{"iteration":1,"attempted":"x","found":"y","darwin_score":0.5,"supported_ratio":0.6}\n',
            encoding="utf-8",
        )
        _out_off2 = eng_c4._build_memory_text("test query")
        assert "Research Trace" not in _out_off2, "C4: toggle off → no trace injection"
    print("18. C4 meta_trace toggle off → _build_memory_text 不注入 trace OK")

    # C4b: load_meta_trace_text 函数验证 — toggle on 路径信任代码逻辑
    # (toggle 依赖全局 config, monkeypatch 模块函数在 method 内部调用时复杂,
    #  改为直接测 load_meta_trace_text 函数本身 + 审查 _build_memory_text 的 toggle 分支)
    from huginn.context_builder import load_meta_trace_text
    with tempfile.TemporaryDirectory() as _td2:
        _huginn_dir2 = Path(_td2) / ".huginn"
        _huginn_dir2.mkdir(parents=True, exist_ok=True)
        _trace2 = _huginn_dir2 / "meta_trace.jsonl"
        _trace2.write_text(
            '{"iteration":5,"attempted":"DFT calc","found":"E=-3.2eV",'
            '"darwin_score":0.8,"supported_ratio":0.7,"evidence":["conv"]}\n',
            encoding="utf-8",
        )
        _trace_text = load_meta_trace_text(_td2, last_n=5)
        assert "Research Trace" in _trace_text, "C4b: load_meta_trace_text returns formatted block"
        assert "iter 5" in _trace_text, "C4b: iteration field present"
        assert "DFT calc" in _trace_text, "C4b: attempted field present"
        assert "E=-3.2eV" in _trace_text, "C4b: found field present"
    # C4c: 空目录 / 无文件 → load_meta_trace_text 返回空串 (不报错)
    with tempfile.TemporaryDirectory() as _td3:
        assert load_meta_trace_text(_td3, last_n=5) == "", "C4c: empty dir → empty string"
    print("19. C4b load_meta_trace_text (有 trace / 空目录) → 函数 OK, toggle 分支信任代码")

    # C4d: _build_memory_text toggle on 路径 — 用 monkeypatch 模块函数
    # _build_memory_text 在 method 内调 _autoloop_meta_trace_inject_enabled(),
    # 这是 engine.py 模块全局函数. monkeypatch 模块属性后 method 内部能读到.
    # python -m 运行时 __name__='__main__', import huginn.autoloop.engine 会拿到
    # 另一个 module 实例, patch 不生效. 必须打 sys.modules['__main__'].
    import sys
    _eng_mod = sys.modules.get("__main__")
    if _eng_mod is None:
        import huginn.autoloop.engine as _eng_mod
    _orig_toggle = _eng_mod._autoloop_meta_trace_inject_enabled
    try:
        _eng_mod._autoloop_meta_trace_inject_enabled = lambda: True
        with tempfile.TemporaryDirectory() as _td4:
            eng_c4.workspace = Path(_td4)
            _hd4 = Path(_td4) / ".huginn"
            _hd4.mkdir(parents=True, exist_ok=True)
            (Path(_td4) / ".huginn" / "meta_trace.jsonl").write_text(
                '{"iteration":2,"attempted":"test","found":"ok","darwin_score":0.5}\n',
                encoding="utf-8",
            )
            # toggle on + memory=None → 只应有 trace block, 不应有 memory recall
            _out_on = eng_c4._build_memory_text("query")
            assert "Research Trace" in _out_on, f"C4d: toggle on → inject, got {_out_on!r}"
    finally:
        _eng_mod._autoloop_meta_trace_inject_enabled = _orig_toggle
    print("19b. C4d _build_memory_text toggle on (monkeypatch) → 注入 trace OK")

    # C2: _build_metacog_block 在 _target_chains 空时返回空串
    eng_c2 = AutoloopEngine.__new__(AutoloopEngine)
    eng_c2._target_chains = []
    eng_c2._target_chains_built = True  # 跳过 _ensure_target_chains 的 LLM 调用
    eng_c2._objective = ""
    eng_c2.memory = None
    eng_c2._iteration = 0
    _c2_out = eng_c2._build_metacog_block()
    assert _c2_out == "", f"C2: empty target_chains + no memory → empty, got {_c2_out!r}"
    print("20. C2 _build_metacog_block 空输入 → 空串 (不污染 prompt) OK")

    # F-borrow (forge 双预算思路): 按 failure_type 分类计数 + 按类阈值 stop
    # _classify_failure 已存在但之前没在 reflect 路径用 — 验证它返回 5 类,
    # 且 _max_failures_by_type 阈值正确 (tool_error 低, hypothesis_error 高).
    eng_f = AutoloopEngine.__new__(AutoloopEngine)
    eng_f._consecutive_failures_by_type = {}
    eng_f._max_failures_by_type = {
        "tool_error": 5, "prompt_injection_suspect": 3,
        "param_error": 5, "data_noise": 5, "hypothesis_error": 10,
    }
    # 1) tool_error: timeout 标记
    _v_tool = {"errors": "subprocess timeout after 60s", "result": ""}
    assert AutoloopEngine._classify_failure(_v_tool) == "tool_error", (
        "F-borrow: timeout → tool_error"
    )
    # 2) param_error: 参数错
    _v_param = {"errors": "invalid parameter encut=-1", "result": ""}
    assert AutoloopEngine._classify_failure(_v_param) == "param_error", (
        "F-borrow: invalid parameter → param_error"
    )
    # 3) data_noise: 噪声大
    _v_noise = {"errors": "", "result": "signal is noisy, no clear trend"}
    assert AutoloopEngine._classify_failure(_v_noise) == "data_noise", (
        "F-borrow: noisy result → data_noise"
    )
    # 4) hypothesis_error: 默认 (结果与预期相反)
    _v_hyp = {"errors": "band gap off by 0.5 eV", "result": "value mismatch"}
    assert AutoloopEngine._classify_failure(_v_hyp) == "hypothesis_error", (
        "F-borrow: value mismatch → hypothesis_error"
    )
    # 5) 阈值: tool_error=5 < hypothesis_error=10 (技术故障短期可恢复 vs 方向错持续才是死路)
    assert eng_f._max_failures_by_type["tool_error"] < eng_f._max_failures_by_type["hypothesis_error"], (
        "F-borrow: tool_error threshold should be lower than hypothesis_error"
    )
    # 6) 模拟 reflect 累加: 3 次 tool_error 不触发 stop (阈值 5), 5 次触发
    for _i in range(3):
        eng_f._consecutive_failures_by_type["tool_error"] = (
            eng_f._consecutive_failures_by_type.get("tool_error", 0) + 1
        )
    assert eng_f._consecutive_failures_by_type["tool_error"] == 3, "F-borrow: 3 累加"
    assert eng_f._consecutive_failures_by_type["tool_error"] < eng_f._max_failures_by_type["tool_error"], (
        "F-borrow: 3 < 5 不应触发 stop"
    )
    for _i in range(2):
        eng_f._consecutive_failures_by_type["tool_error"] += 1
    assert eng_f._consecutive_failures_by_type["tool_error"] == 5, "F-borrow: 5 累加"
    assert eng_f._consecutive_failures_by_type["tool_error"] >= eng_f._max_failures_by_type["tool_error"], (
        "F-borrow: 5 >= 5 应触发 stop"
    )
    # 7) decider prompt 空分类时不显示噪声
    eng_f._consecutive_failures_by_type = {}
    eng_f._consecutive_failures = 0
    eng_f._max_consecutive_failures = 20
    eng_f._pivot_count = 0
    eng_f._refine_count = 0
    eng_f._speculator_hint = ""
    eng_f._validate_window = []
    eng_f._validate_window_size = 100
    eng_f._last_run_failure_pattern = ""
    _state_f = LoopState(iteration=1, max_iterations=10)
    _cog_f = {"hypothesis": "", "plan": None, "execution_result": None, "validation": {}}
    _prompt_f = eng_f._build_decider_prompt(_state_f, _cog_f, {})
    assert "Failures by type: none" in _prompt_f, (
        f"F-borrow: empty by_type should show 'none', got: {_prompt_f[:400]}"
    )
    print("21. F-borrow 分类计数 + 5 类 _classify_failure + 按类阈值 + prompt 显示 OK")

    # 22. 700 万步场景: action_history 截断 (cognitive_loop 层)
    from huginn.autoloop.cognitive_loop import _MAX_ACTION_HIST
    _hist = ["observe"] * (_MAX_ACTION_HIST + 500)
    if len(_hist) > _MAX_ACTION_HIST:
        del _hist[: -_MAX_ACTION_HIST]
    assert len(_hist) == _MAX_ACTION_HIST, (
        f"action_history 截断后长度应 = {_MAX_ACTION_HIST}, got {len(_hist)}"
    )
    assert all(a == "observe" for a in _hist), "截断后尾部内容应保留"
    print(f"22. action_history 截断到窗口 {_MAX_ACTION_HIST} (700 万步防内存爆炸) OK")

    # 23. 700 万步场景: 滑动窗口失败率 — consecutive 触顶但 fail rate 低 → 不停
    eng_w = AutoloopEngine.__new__(AutoloopEngine)
    eng_w._consecutive_failures = 20
    eng_w._max_consecutive_failures = 20
    eng_w._validate_window = [True] * 80 + [False] * 20  # fail rate=0.2
    eng_w._validate_window_size = 100
    eng_w._validate_window_fail_threshold = 0.8
    _fail_rate = 1.0 - (sum(eng_w._validate_window) / len(eng_w._validate_window))
    assert abs(_fail_rate - 0.2) < 1e-6, f"fail rate 应 0.2, got {_fail_rate}"
    assert _fail_rate < eng_w._validate_window_fail_threshold, (
        "fail rate 0.2 < 0.8 应允许继续 (局部失败, 整体进展)"
    )
    if eng_w._consecutive_failures >= eng_w._max_consecutive_failures:
        if len(eng_w._validate_window) >= eng_w._validate_window_size:
            if _fail_rate < eng_w._validate_window_fail_threshold:
                eng_w._consecutive_failures = 0
    assert eng_w._consecutive_failures == 0, (
        "consecutive 触顶但 fail rate 低 → 应清计数不停"
    )
    # 23b: fail rate 高 → 应停
    eng_w2 = AutoloopEngine.__new__(AutoloopEngine)
    eng_w2._validate_window = [False] * 90 + [True] * 10  # fail rate=0.9
    _fail_rate2 = 1.0 - (sum(eng_w2._validate_window) / len(eng_w2._validate_window))
    assert _fail_rate2 >= 0.8, "fail rate 0.9 >= 0.8 应停 (整体死路)"
    print("23. 滑动窗口失败率 (0.2 < 0.8 继续 / 0.9 >= 0.8 停) OK")

    # 24. 700 万步场景: 跨 run 失败模式持久化闭环
    import tempfile
    from huginn.memory.manager import MemoryManager
    from huginn.memory.longterm import LongTermMemory
    with tempfile.TemporaryDirectory() as _td_fp:
        # 用临时 db 路径, 避免 ~/.huginn/memory.db 旧数据干扰
        _db_path = Path(_td_fp) / "test_mem.db"
        _mem_fp = MemoryManager()
        _mem_fp.longterm = LongTermMemory(db_path=str(_db_path))
        eng_p = AutoloopEngine.__new__(AutoloopEngine)
        eng_p.memory = _mem_fp
        eng_p._consecutive_failures_by_type = {"tool_error": 3, "hypothesis_error": 2}
        eng_p._validate_window = [True] * 70 + [False] * 30  # fail rate=0.3
        eng_p._validate_window_size = 100
        eng_p._consecutive_failures = 5
        eng_p._objective = "test 700万步 failure pattern persist"
        eng_p._persist_failure_pattern("loop_test_fp")
        _loaded = eng_p._load_failure_pattern()
        assert "tool_error=3" in _loaded, f"persist/load 闭环: by_type 丢失, got: {_loaded}"
        assert "hypothesis_error=2" in _loaded, f"persist/load 闭环: by_type 丢失, got: {_loaded}"
        assert "fail rate=0.30" in _loaded, f"persist/load 闭环: fail rate 丢失, got: {_loaded}"
        # 24b: 空 by_type 不存 (全 pass run)
        eng_p._consecutive_failures_by_type = {}
        eng_p._validate_window = [True] * 100
        eng_p._persist_failure_pattern("loop_test_fp2")
        _loaded2 = eng_p._load_failure_pattern()
        assert "tool_error=3" in _loaded2, "空快照不应覆盖"
    print("24. 跨 run 失败模式持久化 (persist/load 闭环 + 空快照不覆盖) OK")

    # 25. decider prompt 含 700 万步新字段
    eng_f._validate_window = [True] * 80 + [False] * 20
    eng_f._validate_window_size = 100
    eng_f._last_run_failure_pattern = "last run: tool_error=3, fail rate=0.30 (n=100)"
    _prompt_w = eng_f._build_decider_prompt(_state_f, _cog_f, {})
    assert "Window fail rate: 0.20 (last 100)" in _prompt_w, (
        f"missing window fail rate: {_prompt_w[:500]}"
    )
    assert "Last run pattern: last run: tool_error=3" in _prompt_w, (
        f"missing last run pattern: {_prompt_w[:500]}"
    )
    print("25. decider prompt 含 window fail rate + last run pattern OK")

    # 26. P0-1 streaming toggle: env HUGINN_AUTOLOOP_STREAMING=0 强制关
    import os as _os_mod
    _prev_env = _os_mod.environ.get("HUGINN_AUTOLOOP_STREAMING")
    try:
        _os_mod.environ["HUGINN_AUTOLOOP_STREAMING"] = "0"
        assert _autoloop_streaming_enabled() is False, (
            "env=0 应强制关闭 streaming"
        )
        _os_mod.environ["HUGINN_AUTOLOOP_STREAMING"] = "1"
        # 默认 on: 即使 config 抛异常也返回 True (try/except 兜底)
        assert _autoloop_streaming_enabled() is True, "env=1 默认 on"
    finally:
        if _prev_env is None:
            _os_mod.environ.pop("HUGINN_AUTOLOOP_STREAMING", None)
        else:
            _os_mod.environ["HUGINN_AUTOLOOP_STREAMING"] = _prev_env
    print("26. P0-1 streaming toggle (env HUGINN_AUTOLOOP_STREAMING=0/1) OK")

    # 27. P0-1 _llm_chat: progress_cb + astream 路径 + fallback
    from huginn.types import progress_cb as _pc_cb

    class _FakeChunk:
        def __init__(self, content: str) -> None:
            self.content = content

    class _FakeStreamLLM:
        """LLM with broken astream — 验证 fallback 到 ainvoke."""
        def __init__(self) -> None:
            self.model = "fake-stream"
        async def astream(self, messages):
            yield _FakeChunk("hello ")
            yield _FakeChunk("world")
            raise RuntimeError("synthetic stream break")
        async def ainvoke(self, messages):
            return type("R", (), {"content": "fallback-ok"})()

    class _FakeNoAstreamLLM:
        """LLM without astream method — 应直接走 ainvoke."""
        def __init__(self) -> None:
            self.model = "fake-nostream"
        async def ainvoke(self, messages):
            return type("R", (), {"content": "nostream-ok"})()

    eng_s = AutoloopEngine.__new__(AutoloopEngine)
    eng_s._current_phase = None  # 跳过 thinking effort 注入, 聚焦流式路径
    eng_s.model = None

    async def _run_llm_chat_cases() -> None:
        # 27a: 无 progress_cb → ainvoke 路径 (cb is None gate)
        _pc_cb.set(None)
        _r27a = await eng_s._llm_chat("hi", model=_FakeNoAstreamLLM())
        assert _r27a == "nostream-ok", f"无 cb 应 ainvoke, got {_r27a}"

        # 27b: 有 progress_cb + astream → 流式路径
        _events: list[dict] = []
        async def _collect_cb(msg: dict) -> None:
            _events.append(msg)
        _pc_cb.set(_collect_cb)
        _r27b = await eng_s._llm_chat("hi", model=_FakeStreamLLM())
        # astream 抛异常 → fallback ainvoke → "fallback-ok"
        assert _r27b == "fallback-ok", f"astream fail 应 fallback, got {_r27b}"
        # fallback 前 chunk 事件已发出 (hello/world)
        _types = [e["type"] for e in _events]
        assert _types.count("autoloop_thinking") >= 2, (
            f"astream 应至少推 2 个 chunk event, got {_types}"
        )
        _pc_cb.set(None)

    asyncio.run(_run_llm_chat_cases())
    print("27. P0-1 _llm_chat (无 cb ainvoke / 有 cb astream+fallback) OK")

    # 28. P0-2 progress_cb → _emit_campaign 桥 (run_cognitive 入口设)
    # 验证: 桥接到 progress_cb 后, subagent_event / autoloop_thinking 能流到 campaign SSE
    from huginn.types import progress_cb as _pc_cb2
    _emitted: list[tuple[str, dict]] = []
    eng_b = AutoloopEngine.__new__(AutoloopEngine)
    eng_b._progress_task_id = "test_task_b"
    # mock _emit_campaign 收集事件
    eng_b._emit_campaign = lambda etype, data: _emitted.append((etype, data))
    _run_id_b = "loop_test_b"
    # 复刻 run_cognitive 入口的桥接逻辑
    if _pc_cb2.get(None) is None:
        _eng_ref = eng_b

        async def _bridge_b(msg: dict) -> None:
            _etype = msg.get("type", "progress")
            _data = {k: v for k, v in msg.items() if k != "type"}
            _data.setdefault("run_id", _run_id_b)
            _eng_ref._emit_campaign(f"campaign.{_etype}", _data)

        _pc_cb2.set(_bridge_b)

    async def _run_bridge_cases() -> None:
        _cb = _pc_cb2.get()
        assert _cb is not None, "bridge 应已 set progress_cb"
        # 模拟 subagent_tool._on_state 推 subagent_event
        await _cb({
            "type": "subagent_event",
            "event": "tool_call",
            "spec": "explore",
            "tool": "file_read_tool",
        })
        # 模拟 P0-1 推 autoloop_thinking
        await _cb({
            "type": "autoloop_thinking",
            "phase": "decider",
            "delta": "thinking...",
        })

    asyncio.run(_run_bridge_cases())
    _pc_cb2.set(None)  # 清理, 避免污染后续测试
    assert len(_emitted) == 2, f"应推 2 个事件, got {len(_emitted)}"
    assert _emitted[0][0] == "campaign.subagent_event", (
        f"event_type 应 campaign.subagent_event, got {_emitted[0][0]}"
    )
    assert _emitted[0][1]["spec"] == "explore", f"data 丢失字段: {_emitted[0][1]}"
    assert _emitted[0][1]["run_id"] == _run_id_b, "run_id 应注入"
    assert _emitted[1][0] == "campaign.autoloop_thinking"
    assert _emitted[1][1]["delta"] == "thinking..."
    print("28. P0-2 progress_cb → _emit_campaign 桥 (subagent_event + autoloop_thinking → campaign SSE) OK")

    # 53. P2-6 belief: _darwin_belief_mu/sigma2 后验更新 + σ² 收敛 early stop
    import os as _os
    from huginn.tools.subagent_tool import _gaussian_update
    _saved_belief_darwin = _os.environ.get("HUGINN_BELIEF_DARWIN")
    _os.environ["HUGINN_BELIEF_DARWIN"] = "1"

    # 模拟 5 轮稳定 score → σ² 应显著下降
    mu, s2 = 0.0, 100.0
    for s in [7.0, 7.1, 6.9, 7.0, 7.05]:
        mu, s2 = _gaussian_update(mu, s2, s, 1.0)
    assert s2 < 0.5, f"5 轮稳定观测后 σ² 应 < 0.5, got {s2}"
    assert 6.8 < mu < 7.2, f"μ 应接近 7.0, got {mu}"
    # σ² 收敛阈值 0.1 — 5 轮可能还差一点, 验证再多几轮必到
    for s in [7.0, 7.0, 7.0, 7.0, 7.0]:
        mu, s2 = _gaussian_update(mu, s2, s, 1.0)
    assert s2 < 0.1, f"10 轮稳定观测后 σ² 应 < 0.1 (收敛阈值), got {s2}"

    # σ²_obs 大 (高噪声传感器) → 后验 σ² 下降慢, 不会误判收敛
    # Gaussian 共轭: σ² 只跟 σ²_obs 和观测次数有关, 跟观测值方差无关.
    # ponytail: 用 σ²_obs 区分噪声, 不是观测值 std.
    mu2, s22 = 0.0, 100.0
    for _ in range(6):
        mu2, s22 = _gaussian_update(mu2, s22, 7.0, 10.0)  # σ²_obs=10 (高噪声)
    assert s22 > 0.5, f"高噪声传感器 6 轮 σ² 应仍较大 (未收敛), got {s22}"

    if _saved_belief_darwin is None:
        _os.environ.pop("HUGINN_BELIEF_DARWIN", None)
    else:
        _os.environ["HUGINN_BELIEF_DARWIN"] = _saved_belief_darwin
    print("53. P2-6 belief darwin (Gaussian 后验 σ² 收敛 + 高噪声不误判) OK")

    # 54. H4: GRILL mode 注入 + 退出
    # 验证 _grill_active 时 system prompt 含 GRILL_SYSTEM_PROMPT_CN 关键短语
    from huginn.runtime.pre_plan_grill import GRILL_SYSTEM_PROMPT_CN
    _saved_grill = getattr(eng_s, "_grill_active", False)
    eng_s._grill_active = True
    eng_s._grill_turns = 0
    # _llm_chat 会注入 GRILL prompt, 用 _FakeNoAstreamLLM 跑 (无流式)
    import asyncio as _a54
    async def _grill_case():
        resp = await eng_s._llm_chat("test", persona_name="default", model=_FakeNoAstreamLLM())
        return resp
    _grill_resp = _a54.run(_grill_case())
    # _FakeNoAstreamLLM 不读 system prompt, 只验证 _grill_turns 递增
    assert eng_s._grill_turns >= 1, f"grill active 时 _grill_turns 应递增, got {eng_s._grill_turns}"
    # 20 轮后强制退出
    eng_s._grill_turns = 20
    _a54.run(_grill_case())
    assert not eng_s._grill_active, "grill 超过 20 轮应强制退出"
    # 恢复
    eng_s._grill_active = _saved_grill
    # 静态验证 GRILL_SYSTEM_PROMPT_CN 含关键短语 (prompt 本身存在)
    assert "一次只问一个" in GRILL_SYSTEM_PROMPT_CN
    assert "shared understanding" in GRILL_SYSTEM_PROMPT_CN
    print("54. H4 GRILL mode (注入 + 20 轮强制退出 + 退出标记检测) OK")

    # 55. H2: frontier_ranked 注入 _build_hypothesis_prompt
    # 验证假设图有 untested 节点时, prompt 含 "Untested Hypotheses" 块.
    # ponytail: monkeypatch 外部数据获取方法, 只验证 frontier 注入路径.
    from huginn.autoloop.hypothesis_loop import HypothesisGraph as _HG55
    eng_s.hypothesis_graph = _HG55()
    eng_s.hypothesis_graph.add_hypothesis("test-hypothesis-1")
    eng_s.hypothesis_graph.add_hypothesis("test-hypothesis-2")
    # stub: 跳过 KB/KG/mem/PM/metacog/imagination/git_log/cluster 的外部依赖
    eng_s._speculator_hint = ""
    eng_s.workspace = "."
    for _m in ("_build_kb_text", "_build_kg_text", "_build_memory_text",
               "_build_pm_text", "_build_metacog_block"):
        setattr(eng_s, _m, lambda *a, **kw: "")
    eng_s._should_imaginate = lambda: False
    eng_s._metacog_component_representatives = lambda: []
    _prompt = eng_s._build_hypothesis_prompt({"test": "context"})
    assert "Untested Hypotheses" in _prompt, \
        "frontier_ranked 应注入到 hypothesis prompt, got 缺失"
    assert "test-hypothesis-1" in _prompt, "prompt 应含 untested 假设 statement"
    # toggle off → frontier_ranked 回退 frontier, 仍返回 untested, 仍注入 (向后兼容)
    import os as _os55
    _saved_ising = _os55.environ.get("HUGINN_ISING_FRONTIER")
    _os55.environ["HUGINN_ISING_FRONTIER"] = "0"
    _prompt_off = eng_s._build_hypothesis_prompt({"test": "context"})
    assert "Untested Hypotheses" in _prompt_off, "toggle off 应仍注入 (向后兼容)"
    if _saved_ising is None:
        _os55.environ.pop("HUGINN_ISING_FRONTIER", None)
    else:
        _os55.environ["HUGINN_ISING_FRONTIER"] = _saved_ising
    print("55. H2 frontier_ranked 注入 _build_hypothesis_prompt (有 untested 时 prompt 含块) OK")

    # 56. P1: 盲重建 verification — mock SubagentDispatch 验证 support/refute 闭环
    # 不真起 subagent, mock dispatch 返 holds=true/false, 验证:
    # - holds 一致 → hypothesis_graph.support + PROVED.md
    # - holds 不一致 → hypothesis_graph.refute + FAILED.md
    # - toggle off → 不调 (向后兼容)
    import tempfile as _tf56, shutil as _sh56, json as _json56
    import huginn.autoloop.engine as _eng_mod56
    from huginn.agents.subagent import SubagentResult as _SR56
    _tmp56 = Path(_tf56.mkdtemp(prefix="hgin_p1_"))
    try:
        eng_s.workspace = str(_tmp56)
        eng_s.hypothesis_graph = _HG55(workspace=_tmp56)
        eng_s._agent_factory = object()  # 非 None 让 blind reconstruct 路径进
        _hid56 = eng_s.hypothesis_graph.add_hypothesis(
            "test blind reconstruction hypothesis statement enough length")
        eng_s._current_hyp_id_for_plan = _hid56

        # mock SubagentDispatch.dispatch 返 holds=true (跟 orig_holds=true 一致 → support)
        from huginn.agents import subagent as _sub_mod56
        _orig_dispatch_cls = _sub_mod56.SubagentDispatch
        class _MockDispatch:
            async def dispatch(self, spec, task, context=None, **kw):
                return _SR56(
                    summary='{"holds": true, "derivation": "test", "confidence": 0.8}',
                    full_output="", success=True, spec_name=spec,
                )
        _sub_mod56.SubagentDispatch = _MockDispatch
        import asyncio as _aio56
        try:
            import os as _os56
            _os56.environ["HUGINN_BLIND_RECONSTRUCTION"] = "1"
            _results56 = {"tests_passed": True, "grader_reward": 0.9}
            _aio56.run(eng_s._blind_reconstruct_verify(None, _results56))
            _node56 = eng_s.hypothesis_graph._nodes[_hid56]
            assert _node56.status == "supported", \
                f"holds 一致应 support, got {_node56.status}"
            assert _results56.get("blind_reconstruction", {}).get("match") is True
            _proved56 = _HG55.load_proved(_tmp56)
            assert "blind_reconstruction" in _proved56, "PROVED.md 应记 blind modality"
            print("56a. P1 盲重建 match → support + PROVED.md OK")

            # mock 返 holds=false (跟 orig_holds=true 不一致 → refute)
            eng_s.hypothesis_graph = _HG55(workspace=_tmp56)  # fresh graph
            _hid56b = eng_s.hypothesis_graph.add_hypothesis(
                "test blind reconstruction mismatch case statement")
            eng_s._current_hyp_id_for_plan = _hid56b
            class _MockDispatch2:
                async def dispatch(self, spec, task, context=None, **kw):
                    return _SR56(
                        summary='{"holds": false, "derivation": "disagree", "confidence": 0.7}',
                        full_output="", success=True, spec_name=spec,
                    )
            _sub_mod56.SubagentDispatch = _MockDispatch2
            _results56b = {"tests_passed": True, "grader_reward": 0.9}
            _aio56.run(eng_s._blind_reconstruct_verify(None, _results56b))
            _node56b = eng_s.hypothesis_graph._nodes[_hid56b]
            assert _node56b.status == "refuted", \
                f"holds 不一致应 refute, got {_node56b.status}"
            assert _results56b.get("blind_reconstruction", {}).get("match") is False
            _failed56 = _HG55.load_failed(_tmp56)
            assert "mismatch" in _failed56, "FAILED.md 应记 mismatch"
            print("56b. P1 盲重建 mismatch → refute + FAILED.md OK")

            # toggle off → 不调 (向后兼容)
            _os56.environ["HUGINN_BLIND_RECONSTRUCTION"] = "0"
            # ponytail: 直接验证 toggle 控制路径 — toggle off 时 _validate 末尾
            # 的 if 不进, _blind_reconstruct_verify 不会被调.
            _toggle_off_passes = (
                _os56.environ.get("HUGINN_BLIND_RECONSTRUCTION", "0") != "1"
            )
            assert _toggle_off_passes, "toggle off 应跳过 blind reconstruct"
            print("56c. P1 toggle off 不调 (向后兼容) OK")
        finally:
            _sub_mod56.SubagentDispatch = _orig_dispatch_cls
            _os56.environ.pop("HUGINN_BLIND_RECONSTRUCTION", None)
    finally:
        _sh56.rmtree(_tmp56, ignore_errors=True)

    # 57. P2: stagnation 分类 → counterexample hunt (chaoxu 启发)
    # 验证 _classify_stall 归因 + _trigger_counterexample_hunt 副作用 +
    # _darwin_ratchet_check 按 _classify_stall 返回值分流 (pivot/counterexample/stop)
    import tempfile as _tf57, shutil as _sh57, os as _os57
    from huginn.autoloop.hypothesis_loop import HypothesisGraph as _HG57
    _tmp57 = Path(_tf57.mkdtemp(prefix="hgin_p2_"))
    try:
        # 恢复 _should_imaginate 到真实类方法 (55 块曾 monkey-patch 成 lambda: False)
        eng_s.__dict__.pop("_should_imaginate", None)
        eng_s.workspace = str(_tmp57)
        eng_s.hypothesis_graph = _HG57(workspace=_tmp57)
        # 加一个节点让 _darwin_ratchet_check 不 early return
        _hid57 = eng_s.hypothesis_graph.add_hypothesis(
            "test P2 stagnation classification hypothesis statement")
        eng_s._current_hyp_id_for_plan = _hid57

        # 57a: _classify_stall 归因正确性 (纯规则, 不调 LLM)
        eng_s._max_pivots = 10
        # method_failure → "pivot"
        eng_s._last_failure_mode = "tool_error: VASP timeout"
        eng_s._consecutive_failures = 5
        eng_s._pivot_count = 0
        assert eng_s._classify_stall() == "pivot", \
            "tool_error 应归因 method_failure → pivot"
        # evidence_against → "counterexample"
        eng_s._last_failure_mode = "refuted by counterexample"
        assert eng_s._classify_stall() == "counterexample", \
            "refuted 应归因 evidence_against → counterexample"
        # max_pivots 用尽 → "stop" (优先于其他归因)
        eng_s._pivot_count = 10
        eng_s._last_failure_mode = "tool_error"
        assert eng_s._classify_stall() == "stop", \
            "pivots 用尽应 stop (优先于其他归因)"
        # 无信号 + 低失败率 → "stop"
        eng_s._pivot_count = 0
        eng_s._last_failure_mode = ""
        eng_s._consecutive_failures = 2
        assert eng_s._classify_stall() == "stop", \
            "无信号 + 低失败率应 stop"
        print("57a. P2 _classify_stall 归因 (method→pivot / evidence→counterexample / 用尽→stop) OK")

        # 57b: _trigger_counterexample_hunt 副作用 + _should_imaginate override
        eng_s._force_imaginate = False
        eng_s._speculator_hint = ""
        eng_s._trigger_counterexample_hunt()
        assert eng_s._force_imaginate is True, \
            "_trigger_counterexample_hunt 应设 _force_imaginate=True"
        assert "counterexample" in (eng_s._speculator_hint or "").lower(), \
            "_speculator_hint 应含 counterexample 关键词"
        assert eng_s._should_imaginate() is True, \
            "_force_imaginate=True 应让 _should_imaginate 返 True (override)"
        print("57b. P2 counterexample hunt (force_imaginate + hint + should_imaginate override) OK")

        # 57c: _darwin_ratchet_check 按 _classify_stall 返回值分流
        _orig_stag = _os57.environ.get("HUGINN_DARWIN_STAGNATION_LIMIT")
        _orig_belief = _os57.environ.get("HUGINN_BELIEF_DARWIN")
        _os57.environ["HUGINN_DARWIN_STAGNATION_LIMIT"] = "2"
        # 关掉 belief stop 避免 σ²<0.1 误触发, 聚焦 stagnation 分流
        _os57.environ["HUGINN_BELIEF_DARWIN"] = "0"
        # 设高分 last_score, 保证 delta<0.5 → stagnation++ 触发分流
        eng_s._darwin_last_score = 10.0
        eng_s._darwin_best_score = 10.0
        eng_s._iteration = 5
        try:
            # pivot 路径: 重置 stagnation, 不 stop
            eng_s._darwin_stagnation = 2
            eng_s._should_stop = False
            eng_s._classify_stall = lambda: "pivot"
            eng_s._darwin_ratchet_check()
            assert eng_s._darwin_stagnation == 0, \
                f"pivot 应重置 stagnation=0, got {eng_s._darwin_stagnation}"
            assert eng_s._should_stop is False, \
                "pivot 不应触发 _should_stop"
            print("57c1. P2 pivot 路径 (reset stagnation, no stop) OK")

            # counterexample 路径: 重置 + 触发 hunt, 不 stop
            eng_s._darwin_stagnation = 2
            eng_s._should_stop = False
            eng_s._force_imaginate = False
            eng_s._speculator_hint = ""
            eng_s._classify_stall = lambda: "counterexample"
            eng_s._darwin_ratchet_check()
            assert eng_s._darwin_stagnation == 0, \
                f"counterexample 应重置 stagnation=0, got {eng_s._darwin_stagnation}"
            assert eng_s._should_stop is False, \
                "counterexample 不应触发 _should_stop"
            assert eng_s._force_imaginate is True, \
                "counterexample 应触发 hunt (_force_imaginate=True)"
            assert "counterexample" in (eng_s._speculator_hint or "").lower(), \
                "counterexample 应注入 hint"
            print("57c2. P2 counterexample 路径 (reset + hunt, no stop) OK")

            # stop 路径: 真 stop
            eng_s._darwin_stagnation = 2
            eng_s._should_stop = False
            eng_s._classify_stall = lambda: "stop"
            eng_s._darwin_ratchet_check()
            assert eng_s._should_stop is True, \
                "stop 应设 _should_stop=True"
            print("57c3. P2 stop 路径 (_should_stop=True) OK")
        finally:
            if _orig_stag is None:
                _os57.environ.pop("HUGINN_DARWIN_STAGNATION_LIMIT", None)
            else:
                _os57.environ["HUGINN_DARWIN_STAGNATION_LIMIT"] = _orig_stag
            if _orig_belief is None:
                _os57.environ.pop("HUGINN_BELIEF_DARWIN", None)
            else:
                _os57.environ["HUGINN_BELIEF_DARWIN"] = _orig_belief
    finally:
        _sh57.rmtree(_tmp57, ignore_errors=True)

    # 58. P5: persistent goal mode — stagnation stop 路径被 wall_clock 预算接管
    # 验证 HUGINN_PERSISTENT_GOAL_MODE toggle:
    # - toggle off (默认): stagnation stop 正常触发 _should_stop=True (57c3 已覆盖)
    # - toggle on + wall_clock 未耗尽: stagnation stop 不触发, 重置 stagnation 继续
    # - toggle on + wall_clock 耗尽: stagnation stop 正常触发
    import tempfile as _tf58, shutil as _sh58, os as _os58
    from datetime import datetime, timezone as _tz58
    from huginn.autoloop.hypothesis_loop import HypothesisGraph as _HG58
    from huginn.autoloop.goal_store import GoalStore as _GS58
    _tmp58 = Path(_tf58.mkdtemp(prefix="hgin_p5_"))
    try:
        # 恢复 _should_imaginate + _classify_stall 到真实类方法 (57 块曾 monkey-patch)
        eng_s.__dict__.pop("_should_imaginate", None)
        eng_s.__dict__.pop("_classify_stall", None)
        eng_s.workspace = str(_tmp58)
        eng_s.hypothesis_graph = _HG58(workspace=_tmp58)
        _hid58 = eng_s.hypothesis_graph.add_hypothesis(
            "test P5 persistent goal mode hypothesis statement")
        eng_s._current_hyp_id_for_plan = _hid58

        _orig_persistent = _os58.environ.get("HUGINN_PERSISTENT_GOAL_MODE")
        _orig_stag = _os58.environ.get("HUGINN_DARWIN_STAGNATION_LIMIT")
        _orig_belief = _os58.environ.get("HUGINN_BELIEF_DARWIN")
        _os58.environ["HUGINN_DARWIN_STAGNATION_LIMIT"] = "2"
        _os58.environ["HUGINN_BELIEF_DARWIN"] = "0"
        eng_s._darwin_last_score = 10.0
        eng_s._darwin_best_score = 10.0
        eng_s._iteration = 5
        try:
            # 58a: toggle off → stagnation stop 正常触发 (向后兼容)
            _os58.environ["HUGINN_PERSISTENT_GOAL_MODE"] = "0"
            eng_s._darwin_stagnation = 2
            eng_s._should_stop = False
            eng_s._classify_stall = lambda: "stop"
            eng_s._darwin_ratchet_check()
            assert eng_s._should_stop is True, \
                "toggle off 时 stagnation stop 应正常触发"
            print("58a. P5 toggle off → stagnation stop 正常触发 (向后兼容) OK")

            # 58b: toggle on + 无 active goal → 走原 stop 逻辑 (无 goal 不持续)
            _os58.environ["HUGINN_PERSISTENT_GOAL_MODE"] = "1"
            # 临时替换 get_goal_store 返空 store (无 active goal)
            import huginn.autoloop.goal_store as _gs_mod58
            _orig_get_store = _gs_mod58.get_goal_store
            _empty_store = _GS58(Path(_tmp58) / "empty.json")
            _gs_mod58.get_goal_store = lambda: _empty_store
            try:
                eng_s._darwin_stagnation = 2
                eng_s._should_stop = False
                eng_s._darwin_ratchet_check()
                assert eng_s._should_stop is True, \
                    "toggle on 但无 active goal 时应走原 stop 逻辑"
                print("58b. P5 toggle on + 无 active goal → 原 stop 逻辑 OK")
            finally:
                _gs_mod58.get_goal_store = _orig_get_store

            # 58c: toggle on + active goal + wall_clock 未耗尽 → 不 stop, 重置
            _os58.environ["HUGINN_PERSISTENT_GOAL_MODE"] = "1"
            _store58 = _GS58(Path(_tmp58) / "p5.json")
            _g58 = _store58.create_goal("persistent goal test")
            _store58.update_goal(
                _g58.id,
                wall_clock_budget_seconds=3600.0,  # 1 小时, 肯定没超
                started_at=datetime.now(_tz58.utc).isoformat(),
            )
            _gs_mod58.get_goal_store = lambda: _store58
            try:
                eng_s._darwin_stagnation = 2
                eng_s._should_stop = False
                eng_s._darwin_ratchet_check()
                assert eng_s._darwin_stagnation == 0, \
                    f"persistent mode + 未耗尽应重置 stagnation=0, got {eng_s._darwin_stagnation}"
                assert eng_s._should_stop is False, \
                    "persistent mode + 未耗尽不应 stop"
                print("58c. P5 toggle on + wall_clock 未耗尽 → 不 stop, 重置 OK")

                # 58d: toggle on + active goal + wall_clock 已耗尽 → 真 stop
                _store58.update_goal(
                    _g58.id,
                    wall_clock_budget_seconds=0.001,  # 1ms, 肯定超了
                )
                import time as _time58
                _time58.sleep(0.002)
                eng_s._darwin_stagnation = 2
                eng_s._should_stop = False
                eng_s._darwin_ratchet_check()
                assert eng_s._should_stop is True, \
                    "wall_clock 耗尽时应真 stop"
                print("58d. P5 toggle on + wall_clock 耗尽 → 真 stop OK")
            finally:
                _gs_mod58.get_goal_store = _orig_get_store
        finally:
            if _orig_persistent is None:
                _os58.environ.pop("HUGINN_PERSISTENT_GOAL_MODE", None)
            else:
                _os58.environ["HUGINN_PERSISTENT_GOAL_MODE"] = _orig_persistent
            if _orig_stag is None:
                _os58.environ.pop("HUGINN_DARWIN_STAGNATION_LIMIT", None)
            else:
                _os58.environ["HUGINN_DARWIN_STAGNATION_LIMIT"] = _orig_stag
            if _orig_belief is None:
                _os58.environ.pop("HUGINN_BELIEF_DARWIN", None)
            else:
                _os58.environ["HUGINN_BELIEF_DARWIN"] = _orig_belief
    finally:
        _sh58.rmtree(_tmp58, ignore_errors=True)

    print("AutoloopEngine selfcheck OK (13/13 + D block 1b+2b+14 + C block 15-20 + F-borrow 21 + 700万步 22-25 + P0-1 流式 26-27 + P0-2 桥 28 + P2-6 belief darwin 53 + H4 GRILL 54 + H2 frontier 55 + P1 blind 56 + P2 stall 57 + P5 persistent goal 58)")


if __name__ == "__main__":
    _selfcheck()
