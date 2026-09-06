"""Autoloop Engine — the main autonomous loop for Huginn.

Ties together exploration, coder, workflow, benchmark, and report
into a single closed-loop ecosystem:

    Perceive → Hypothesize → Plan → Execute → Validate → Learn → Report

Usage:
    engine = AutoloopEngine(workspace=Path("."))
    asyncio.run(engine.run_cognitive(objective="Optimize C-S-H defect kinetics"))
"""

from __future__ import annotations

import logging
import os
import random
import re
from pathlib import Path
from typing import Any

from huginn.autoloop.budget import TokenBudget

logger = logging.getLogger(__name__)


def _feature_flag(name: str, default: bool) -> bool:
    """读取 cfg.feature_flags.<name>, 缺省返回 default (契约收敛单一读取路径).

    原三个 flag 探测函数 (_harness_workflow_evolution_enabled /
    _autoloop_meta_trace_inject_enabled / _autoloop_streaming_enabled) 各自重复
    get_config→feature_flags→get 逻辑 (审计 R3 知识重复). 统一收拢到此处, 异常
    时 best-effort 返回 default, 与原实现逐字段一致.
    """
    try:
        from huginn.config import get_config
        cfg = get_config()
        ff = getattr(cfg, "feature_flags", None) or {}
        return bool(ff.get(name, default))
    except Exception:
        logger.debug("best-effort op failed", exc_info=True)
        return default


def _harness_workflow_evolution_enabled() -> bool:
    """H2 toggle: cfg.feature_flags.harness_workflow_evolution (默认 off)."""
    return _feature_flag("harness_workflow_evolution", False)


#: 分类失败阈值默认值 — 历史硬编码语义, 收敛到此作为单一缺省依据.
_DEFAULT_MAX_FAILURES_BY_TYPE: dict[str, int] = {
    "tool_error": 5,
    "prompt_injection_suspect": 3,
    "param_error": 5,
    "data_noise": 5,
    "hypothesis_error": 10,
}


def _config_failures_by_type() -> dict[str, int]:
    """读取分类失败阈值, 默认从 env_defaults 注册表取 (HUGINN_MAX_FAILURES_BY_TYPE, JSON).

    支持按执行者定制: 设 ``HUGINN_MAX_FAILURES_BY_TYPE='{"tool_error":8,"hypothesis_error":15}'``
    即覆盖对应键, 未覆盖键回退默认. 解析失败/缺省 → 全默认 (兼容旧硬编码行为).
    """
    import json as _json

    default = dict(_DEFAULT_MAX_FAILURES_BY_TYPE)
    raw = os.environ.get("HUGINN_MAX_FAILURES_BY_TYPE", "")
    if not raw:
        return default
    try:
        parsed = _json.loads(raw)
    except ValueError:
        return default
    if isinstance(parsed, dict):
        for k, v in parsed.items():
            try:
                int_v = int(float(v))
            except (TypeError, ValueError):
                continue
            if int_v > 0:
                default[str(k)] = int_v
    return default


def _autoloop_meta_trace_inject_enabled() -> bool:
    """C4 toggle: cfg.feature_flags.autoloop_meta_trace_inject (默认 off).

    autoloop engine 每轮 _distill_meta_trace 写 .huginn/meta_trace.jsonl,
    但 _build_memory_text 之前不读它 — 长轨迹里 agent 看不到上轮蒸馏的结构化
    历史. toggle on 后注入最近 5 条 entry 到 memory_text.

    ponytail: 默认 off, 因为 meta_trace 跟 FTS5 memory 可能内容重叠,
      注入会增加 prompt 长度. 升级路径: 默认 on + 按 darwin_score top-K 去重.
    """
    return _feature_flag("autoloop_meta_trace_inject", False)


def _autoloop_streaming_enabled() -> bool:
    """P0-1 toggle: cfg.feature_flags.autoloop_streaming (默认 on).

    astream 替代 ainvoke, 把 LLM thinking chunk 增量推到 progress_cb (WS).
    700 万步场景必需 — 否则用户看不到 agent 在想什么. 默认 on 是因为
    fail 会自动回退 ainvoke, 无 WS 场景 progress_cb=None 也不流式.
    ponytail: 环境变量 HUGINN_AUTOLOOP_STREAMING=0 可强制关.
    """
    if os.environ.get("HUGINN_AUTOLOOP_STREAMING", "1") == "0":
        return False
    return _feature_flag("autoloop_streaming", True)


from huginn.autoloop.cognitive_loop import CognitiveLoopMixin  # noqa: E402
from huginn.autoloop.engine_act import EngineActMixin  # noqa: E402
from huginn.autoloop.engine_control import EngineControlMixin  # noqa: E402
from huginn.autoloop.engine_observe import EngineObserveMixin  # noqa: E402

# P3 slim-down: 5 个 engine_* mixin (perceive/observe/act/reflect/control) 把
# AutoloopEngine 的方法族拆到独立模块. mixin 通过 self 访问 engine 状态,
# 对 engine.py 模块级符号用方法内 lazy import 避免 circular.
from huginn.autoloop.engine_perceive import EnginePerceiveMixin  # noqa: E402
from huginn.autoloop.engine_reflect import EngineReflectMixin  # noqa: E402
from huginn.autoloop.goal_scheduler import GoalScheduler  # noqa: E402
from huginn.autoloop.hypothesis_loop import HypothesisMixin  # noqa: E402
from huginn.autoloop.math_validation import MathValidationMixin  # noqa: E402
from huginn.autoloop.phase_gate import (  # noqa: E402
    PhaseGateHook,
)
from huginn.autoloop.plan_check import PlanCheckMixin  # noqa: E402
from huginn.autoloop.visual_inspect import VisualInspectMixin  # noqa: E402
from huginn.bench.runner import BenchmarkRunner  # noqa: F401, E402  # monkeypatch
from huginn.coder.loop import CoderRunner  # noqa: E402
from huginn.config import get_settings  # noqa: E402

# C1: 共享 KB chunk 格式化函数, 跟 ContextBuilder 走同一条路径, 消除双路径漂移.
# C4: 共享 meta_trace 加载, engine 写 jsonl 但之前不读, 现在注入 memory_text.
from huginn.exploration.orchestrator import ExplorationOrchestrator  # noqa: E402
from huginn.exploration.strategies import ParetoPruningStrategy  # noqa: E402
from huginn.interaction.progress import ProgressTracker  # noqa: E402
from huginn.kg.builder import ProjectKnowledgeGraph  # noqa: E402
from huginn.llm import get_model  # noqa: E402
from huginn.memory.manager import MemoryManager  # noqa: E402
from huginn.tools.report_tool import ReportTool  # noqa: E402
from huginn.utils.runtime import HUGINN_DIR_NAME  # noqa: E402
from huginn.workflows.engine import WorkflowEngine  # noqa: E402

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

    实现已下沉到 cognitive_loop.py (P3 slim-down, 跟 4 个调用点一起).
    这里保留 re-export 兼容 engine_selfcheck 的 `from ...engine import _extract_tests_passed`.
    """
    from huginn.autoloop.cognitive_loop import _extract_tests_passed as _impl
    return _impl(validation)


# === dataclass + snapshot 函数抽到 autoloop/types.py ===
# ponytail: 单一职责拆分. 原 L178-277 抽到 autoloop/types.py.



class AutoloopEngine(
    EnginePerceiveMixin,
    EngineObserveMixin,
    EngineActMixin,
    EngineReflectMixin,
    EngineControlMixin,
    PlanCheckMixin,
    MathValidationMixin,
    VisualInspectMixin,
    CognitiveLoopMixin,
    HypothesisMixin,
):
    """Main autonomous loop engine.

    Orchestrates perception, hypothesis generation, planning, execution,
    validation, learning, and reporting into a single cohesive loop.

    Method 分组 (P3 slim-down): perceive/observe/act/reflect/control 方法族
    已拆到 engine_*.py mixin 模块, 本文件保留 __init__ + 对齐数据 + 懒加载访问器.
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
        # P1#1: 注入 LLM model provider, 供 hypothesis_semantic 语义判定 (维度/方法族/
        # 失败类型) 延迟取用. flag 默认关, 无副作用; 开启后无 model 时优雅降级.
        try:
            from huginn.autoloop.hypothesis_semantic import set_model_provider

            set_model_provider(lambda: self.model)
        except Exception:
            logger.debug("hypothesis_semantic model provider inject skipped", exc_info=True)
        # H5-a: 从 config 的 ModelManager 挂 model_router, 让 _llm_chat 的
        # task 路由真正生效 (之前 getattr(self,"model_router",None) 恒 None,
        # "reasoning/summarize/verification" 分类路由全是死代码).
        # 单模型配置时 build_agent_kwargs 返回 model_router=None, 行为不变.
        self.model_router = None
        try:
            from huginn.config import get_config

            _cfg = get_config()
            _r = _cfg.build_agent_kwargs().get("model_router")
            if _r is not None:
                self.model_router = _r
        except Exception:
            logger.debug("model_router build skipped (non-fatal)", exc_info=True)
        # Moonshine 三槽: verification 用独立 LLM 验证假设, 避免确认偏差.
        # 显式注入的 verification_model 优先; 未注入时默认走 select_model
        # ("verification" 路由会选注册的独立验证模型, 无则退回 self.model),
        # 保持单模型下行为与改动前一致.
        self.verification_model = verification_model or self.select_model(
            "verification"
        )
        # 共享 MemoryManager: 由 agent/CLI 传入, 避免引擎私有实例和 agent 的
        # memory 隔离. 默认 None 时 new 一个, 保持向后兼容.
        self.memory = memory_manager or MemoryManager()
        self.kg = ProjectKnowledgeGraph(root=self.workspace)
        # 假设图: 跟踪 hypothesis 的 support/refute/derive 关系,
        # refute 时触发 RedTeam 审查 → 修正假设入队, 形成闭环
        from huginn.autoloop.hypothesis_loop import HypothesisGraph

        # P0: 传 workspace 给 HypothesisGraph, 让 refute/support 时写 FAILED.md/PROVED.md
        self.hypothesis_graph = HypothesisGraph(workspace=self.workspace)
        # E2-1: 通用假设流形 — 懒加载, 让所有 benchmark (不只 RCB) 都能用
        # 后验引导 hint. 存 _hypo_obs 跨 iter 保留上轮观测, hint 用上轮值.
        self._hypo_manifold: Any = None
        self._hypo_obs_for_hint: list = []
        # 高阶网络挂载: 把 stable_principles / evolution_rules 作为高维单纯形
        # 挂到 graph 的 _simplicials, 让知识蒸馏的约束结构进入同调可见性.
        # 失败非致命 (知识库空/损坏时图照常工作).
        try:
            self.hypothesis_graph.mount_knowledge()
        except Exception:
            logger.debug("mount_knowledge failed (non-fatal)", exc_info=True)
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

        self._init_failure_state()
        self._init_lazy_backends(goal_scheduler)

        # P15: crash-safe resume. flag off / snapshot 缺失时静默跳过, 行为不变.
        # ponytail: 用 duck typing, 不做 isinstance 检查; 失败 log warning 不抛.
        # Task 2.2: 未显式传 resume_from_state 时, 默认尝试加载最新 checkpoint
        # (按 mtime). 加载失败不 fatal, warn 后从零开始.
        self._init_resume_state(resume_from_state)
        self._init_runtime_ledger()

        # Step 8: latent space 对齐数据集. None = 还没用过 (懒加载, 跟 cognitive_map 同范式).
        # 首次 _collect_alignment_pair 触发时从 <workspace>/.huginn/alignment_dataset.json 加载.
        # ponytail: 全进程一份, JSON 落盘. ceiling: 多进程不共享, 升级路径接 SQLite.
        self._alignment_dataset: Any = None

    # ── 状态初始化分片 (b4 slim: 纯 self.xxx 默认值块外移, 行为等价) ──

    def _init_failure_state(self) -> None:
        """失败跟踪 / 窗口 / refine / pivot 状态 (原 __init__ 主体, 纯默认值)."""
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
        # 默认值集中在 env_defaults 注册表 (HUGINN_MAX_FAILURES_BY_TYPE, JSON), 可按执行者校准.
        # 兼容旧行为: 注册表缺省 {tool_error:5, ...}, 与历史硬编码一致.
        self._max_failures_by_type: dict[str, int] = _config_failures_by_type()
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

    def _init_lazy_backends(self, goal_scheduler: GoalScheduler | None = None) -> None:
        """懒加载词汇表 + phase_gate_hook 装配 (原 __init__ 主体, 行为等价)."""
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
        # 桥 E: 最近一轮 apply_heuristic_fix 命中的 rule_id, 进 _snapshot 供 episodic replay.
        self._last_rule_hit_id: str = ""
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
        self._metacog_last_topology = None  # 最近一次同调/拓扑审计 (sheaf H¹/Betti/Hodge)
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
        # H3: autoloop 事件溯源 — 每次 phase 切换 append autoloop_phase_change 事件
        # 到 workspace 级事件日志 (<workspace>/.huginn/events/autoloop.jsonl).
        # read_runtime_state() 从事件投影读 (可重放/可恢复), 而非只看可变属性.
        # ponytail: best-effort — 日志打不开时静默降级回内存属性, 不阻塞主循环.
        self._event_log: Any = None
        self._event_log_failed: bool = False
        # P0 Task 4: 自我效能统计周期更新计数器. _learn 末尾自增, 每 10 次触发
        # _compute_self_model + 缓存写入. toggle HUGINN_SELF_MODEL 默认 off,
        # off 时不触发 (向后兼容). ponytail: 复用 longterm typed memory 存储.
        self._experiment_count_since_self_model_update = 0

    def _init_resume_state(self, resume_from_state: str | None) -> None:
        """crash-safe resume + WakeStore 自动唤醒 flag (原 __init__ 主体, 行为等价)."""
        # P15: crash-safe resume. flag off / snapshot 缺失时静默跳过, 行为不变.
        # ponytail: 用 duck typing, 不做 isinstance 检查; 失败 log warning 不抛.
        # Task 2.2: 未显式传 resume_from_state 时, 默认尝试加载最新 checkpoint
        # (按 mtime). 加载失败不 fatal, warn 后从零开始.
        _resume_id = resume_from_state
        if not _resume_id:
            try:
                from huginn.runtime.engine_state import latest_run_id
                _resume_id = latest_run_id(self.workspace)
            except Exception:
                logger.debug("best-effort op failed", exc_info=True)
                _resume_id = None
        if _resume_id:
            try:
                from huginn.runtime.engine_state import (
                    _hypothesis_graph_path,
                    apply_state_to_engine,
                    load_engine_state,
                    use_persistence,
                )
                if use_persistence():
                    state = load_engine_state(_resume_id, self.workspace)
                    if state is not None:
                        apply_state_to_engine(state, self)
                        # H3: 事件投影优先 — 若 workspace 有 autoloop 事件日志,
                        # 用投影恢复当前 phase (而非只信任可变快照的 phase 字段).
                        # best-effort: 投影读失败时沿用快照值, 不阻塞.
                        try:
                            projected = self.read_runtime_state()
                            if projected.get("phase"):
                                self._current_phase = projected["phase"]
                                logger.info(
                                    "restored autoloop phase from event projection: %s",
                                    projected["phase"],
                                )
                        except Exception:
                            logger.debug(
                                "autoloop event projection restore failed (non-fatal)",
                                exc_info=True,
                            )
                        # hypothesis_graph 单独恢复 (refuted 状态跨 session 必须保留)
                        try:
                            loaded_graph = self.hypothesis_graph.load(
                                _hypothesis_graph_path(self.workspace, _resume_id)
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
                            _resume_id, state._iteration,
                            state._last_persona or "(none)",
                        )
                    else:
                        logger.info(
                            "resume requested but no snapshot for run_id=%s, "
                            "starting fresh", _resume_id,
                        )
            except Exception:
                logger.warning(
                    "resume_from_state=%s failed, starting fresh",
                    _resume_id, exc_info=True,
                )

        # P1-4 / P2-6 自动触发: resume_from_state 恢复完后, 检测 WakeStore
        # 有无 pending wakes → 有则 spawn WakeScheduler. runner 调用方负责
        # 真正 resume session (autoloop 自身是长跑循环, 不需要 P1-4 resume;
        # 这个 hook 给 Unattended 场景用: 长任务被 sleep_for 挂起后, scheduler
        # 到点唤醒继续 run_cognitive).
        # ponytail: 不在 __init__ spawn (可能还没 event loop), 延迟到 run_cognitive
        # 第一次 await 时 lazy start. 升级路径: 由调用方 (CLI/routes) 显式管理.
        self._wake_scheduler: Any = None
        self._auto_wake_enabled = os.environ.get(
            "HUGINN_AUTO_WAKE", "1"
        ) == "1"

    def _init_runtime_ledger(self) -> None:
        """MCMC 状态 + token/cost 预算 (原 __init__ 主体, 纯默认值)."""
        # MCMC 状态 (单链) — 长程采样断点续跑, _maybe_save_engine_state 周期落盘
        # rcb_runner 路径不经过 AutoloopEngine, 那边自己持 holder 对象
        self._mcmc_current: str | None = None
        self._mcmc_rng = random.Random(
            int(os.environ.get("HUGINN_MCMC_SEED", "42")))
        self._mcmc_rng_state: tuple | None = None
        self._mcmc_accept_count = 0
        self._mcmc_step_count = 0
        self._mcmc_chains: dict[int, dict] = {}
        # P2-7: token/cost 硬刹车预算. 每次 LLM 调用后 update, 超硬上限抛 BudgetExhausted.
        # 默认 10M tokens / $50, 长任务/极限模式用 HUGINN_TOKEN_BUDGET / HUGINN_COST_BUDGET 覆盖.
        self._token_budget: TokenBudget = TokenBudget()

    # ── H5-a: 模型选择 ────────────────────────────────────────────
    # 统一模型选择入口. 多模型配置 (config.models 非空) 时走 model_router
    # 按 task 分流 (verification→独立模型 / summarize→便宜模型);
    # 单模型时 router 为 None, 回退 self.model, 行为与改动前一致.
    # 供 4 个 mixin (act/reflect/hypothesis/cognitive) 共享, 避免各写一份
    # "select_model or self.model" 的 fallback.

    def select_model(self, task: str = "agent", band: str | None = None) -> Any:
        router = getattr(self, "model_router", None)
        if router is not None:
            # H6: 双吸引子 band 路由. 显式 band 优先 (调用方已量化到稳定带);
            # 否则用分类器把 task/阶段量化到稳定带 (spec/react), 避开 mixed
            # 相变陷阱. 未标注 band 的模型走通用回退, 不破坏旧 task 路由.
            try:
                if band is not None or router.list_bands() and any(
                    router.list_bands().values()
                ):
                    target = band if band is not None else ""
                    if target:
                        _m = router.select_band(target)
                    else:
                        from huginn.models.router import classify_band
                        _m = router.select_band(classify_band(task))
                    if _m is not None:
                        return _m
            except Exception:
                logger.debug(
                    f"model_router band select({task!r},{band!r}) failed — "
                    "falling back to task routing",
                    exc_info=True,
                )
            try:
                _m = router.select(task)
                if _m is not None:
                    return _m
            except Exception:
                logger.debug(
                    f"model_router.select({task!r}) failed — using fallback",
                    exc_info=True,
                )
        return self.model

    # ── H3: autoloop 事件溯源 ──────────────────────────────────────
    # phase 切换写进 workspace 级事件日志, read_runtime_state() 从投影读,
    # 替代"只看可变快照". 全部 best-effort: 日志不可用时静默回退内存属性.

    def _event_log_path(self) -> Path:
        """autoloop 事件日志落盘路径 — <workspace>/.huginn/events/autoloop.jsonl."""
        return self.workspace / HUGINN_DIR_NAME / "events" / "autoloop.jsonl"

    def _get_event_log(self):
        """懒加载 workspace 级 SessionEventLog; 失败返回 None (不重试)."""
        if self._event_log is not None:
            return self._event_log
        if self._event_log_failed:
            return None
        try:
            from huginn.events.session_log import SessionEventLog

            path = self._event_log_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            self._event_log = SessionEventLog(
                session_id=f"autoloop:{self.workspace.name or 'ws'}",
                path=path,
                load=True,
            )
        except Exception:
            logger.debug("autoloop event log open failed (non-fatal)", exc_info=True)
            self._event_log_failed = True
        return self._event_log

    def _record_autoloop_phase(self, phase: str, status: str, iteration: int) -> None:
        """Append an ``autoloop_phase_change`` event (best-effort, non-fatal)."""
        log = self._get_event_log()
        if log is None:
            return
        try:
            from huginn.events.session_log import EVENT_AUTOLOOP_PHASE

            log.append(
                EVENT_AUTOLOOP_PHASE,
                {
                    "phase": phase,
                    "status": status,
                    "iteration": int(iteration),
                },
            )
        except Exception:
            logger.debug(
                "autoloop phase event append failed (non-fatal)", exc_info=True,
            )

    def read_runtime_state(self) -> dict[str, Any]:
        """H3: 从事件投影读引擎运行时状态 (phase/status/iteration).

        事件日志存在时, 用 ``AutoloopStateProjection`` 折叠事件路径得到读模型 —
        可重放/可恢复, 与日志强一致. 日志缺失或为空时回退到内存属性
        (``_current_phase`` / ``_iteration``), 保持向后兼容.
        """
        log = self._get_event_log()
        if log is not None:
            try:
                from huginn.events.projection import (
                    AutoloopStateProjection,
                    ProjectionEngine,
                )

                engine = ProjectionEngine()
                engine.register(AutoloopStateProjection())
                state = engine.build(log, "autoloop")
                if state.get("phase") or state.get("status"):
                    return state
            except Exception:
                logger.debug(
                    "autoloop runtime state read failed (non-fatal)", exc_info=True,
                )
        return {
            "phase": getattr(self, "_current_phase", "") or "",
            "status": "running",
            "iteration": getattr(self, "_iteration", 0) or 0,
            "phase_seq": -1,
        }

    def _publish_progress(self) -> None:
        """H3: 把投影读模型 (phase/status/iteration) 推到 ProgressTracker.

        UI 进度通道 (/tasks + /tasks/stream SSE) 现在消费的是事件投影派生值,
        而非纯可变快照 — 与事件日志强一致, 且携带 ``phase_seq`` 供前端与
        事件流 ``after=<seq>`` 游标对齐. best-effort: tracker 不可用静默跳过.
        """
        state = self.read_runtime_state()
        phase = state.get("phase") or ""
        status = state.get("status") or ""
        tracker = getattr(self, "progress_tracker", None)
        if tracker is None:
            try:
                from huginn.interaction.progress import get_progress_tracker

                tracker = get_progress_tracker()
            except Exception:
                tracker = None
        task_id = getattr(self, "_progress_task_id", None)
        if tracker is None or not task_id:
            return
        try:
            tracker.update(
                task_id,
                current_label=f"{phase} ({status})" if phase else status or "…",
                metadata={
                    "phase": phase,
                    "phase_status": status,
                    "iteration": state.get("iteration", 0),
                    "phase_seq": state.get("phase_seq", -1),
                    "source": "event_projection",
                },
            )
        except Exception:
            logger.debug(
                "autoloop progress publish failed (non-fatal)", exc_info=True,
            )


    def _alignment_dataset_path(self) -> Path:
        """Step 8: AlignmentDataset 落盘路径 — <workspace>/.huginn/alignment_dataset.json."""
        return self.workspace / HUGINN_DIR_NAME / "alignment_dataset.json"



    def _get_alignment_dataset(self):
        """Step 8: 懒加载 AlignmentDataset, 首次调用时从磁盘 load (如果有).

        返回 None 表示未启用 (没磁盘文件就 new 一个空集, 不主动写盘).
        跟 cognitive_map 同范式: 不上 env flag, 是否真收集由 _collect_alignment_pair
        里"能拿到 structure + haptic 才 add"的自然条件决定.
        """
        if self._alignment_dataset is not None:
            return self._alignment_dataset
        from huginn.metacog.alignment_dataset import AlignmentDataset
        path = self._alignment_dataset_path()
        if path.exists():
            try:
                self._alignment_dataset = AlignmentDataset.load(path)
            except Exception:
                logger.warning(
                    "alignment_dataset load failed, starting fresh (non-fatal)",
                    exc_info=True,
                )
                self._alignment_dataset = AlignmentDataset()
        else:
            self._alignment_dataset = AlignmentDataset()
        return self._alignment_dataset



    def _save_alignment_dataset(self) -> None:
        """Step 8: 把 AlignmentDataset 落盘. 跟 engine_state 同节奏调.

        ponytail: 失败只 log warning, 不抛 — dataset save 失败不该阻塞主循环.
        数据集从未被触达 (None) 时直接跳过, 不创建空文件.
        """
        ds = getattr(self, "_alignment_dataset", None)
        if ds is None:
            return
        try:
            path = self._alignment_dataset_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            ds.save(path)
        except Exception:
            logger.warning(
                "alignment_dataset save failed (non-fatal)", exc_info=True,
            )



    def _get_active_cognitive_map(self):
        """Step 8: 拿当前活跃的 StructureCognitiveMap, 没有返 None.

        ponytail: 取 structure_cognitive_map_tool._MAPS 最后一个 (最近创建的).
        ceiling: 多 map 时不知道哪个对应当前 tool, 取最近的是启发式.
        升级路径: tool 调用时把 map_id 写进 result, 这里按 id 取.
        """
        try:
            from huginn.tools import structure_cognitive_map_tool as _cm
            if not _cm._MAPS:
                return None
            return list(_cm._MAPS.values())[-1]
        except Exception:
            logger.debug("best-effort op failed", exc_info=True)
            return None



    def _extract_haptic_layer(self, execution_result: Any):
        """Step 8: 从 execution_result 抽力学字段, 构 HapticPropertyLayer.

        没力学字段返 None. 只看顶层 + result_data/parsed 嵌套一层 — 不递归,
        不爬 stage_results. ponytail: 处理最常见格式 (elastic_tensor / bulk_modulus /
        phonon_freqs / surface_energy / thermal), 其他格式让各 tool 自己 add.
        ceiling: stage_results[*].output 里的力学字段抓不到, 升级路径递归扫.
        """
        if not isinstance(execution_result, dict):
            return None
        data = execution_result
        # _validate / _literature_comparison 用的同款 fallback: result_data 优先, parsed 兜底
        nested = data.get("result_data")
        if isinstance(nested, dict):
            data = nested
        else:
            nested = data.get("parsed")
            if isinstance(nested, dict):
                data = nested

        # 力学字段 — 跟 validate_tool._run_elastic_validation 对齐
        elastic_raw = data.get("elastic_tensor") or data.get("elastic_constants") or data.get("C")
        bulk = data.get("bulk_modulus")
        phonon = data.get("phonon_freqs") or data.get("phonon_frequencies")
        surface = data.get("surface_energy")
        thermal = data.get("thermal")

        if not any(v is not None for v in (
            elastic_raw, bulk, phonon, surface, thermal,
        )):
            return None

        from huginn.metacog.haptic_property_layer import HapticPropertyLayer

        elastic = None
        if elastic_raw is not None:
            try:
                import numpy as np

                from huginn.mechanics import ElasticTensor
                C = np.array(elastic_raw, dtype=float)
                if C.shape == (6, 6):
                    elastic = ElasticTensor(C=C)
            except Exception:
                logger.debug(
                    "elastic_tensor parse failed, skipping elastic field",
                    exc_info=True,
                )

        phonon_arr = None
        if phonon is not None:
            try:
                import numpy as np
                phonon_arr = np.asarray(phonon, dtype=float)
            except Exception:
                logger.debug("phonon array parse skipped", exc_info=True)

        return HapticPropertyLayer(
            elastic=elastic,
            phonon_freqs=phonon_arr,
            surface_energy=float(surface) if surface is not None else None,
            thermal=thermal if isinstance(thermal, dict) else None,
        )



    def _collect_alignment_pair(
        self, execution_result: Any, tool_name: str | None = None,
    ) -> None:
        """Step 8: tool 算出力学属性时, 自动收 (structure_vec, haptic_vec) 对.

        触发条件: execution_result 含力学字段 + 当前有活跃 cognitive map.
        两者缺一就跳过 — 没结构没法对齐, 没力学没必要对齐.

        失败只 log warning, 不报错. 收集是 best-effort, 不能阻塞主循环.
        ponytail: 不改 tool 返回格式, 不引新依赖. 升级路径: 各 tool 自己 add.
        """
        try:
            haptic = self._extract_haptic_layer(execution_result)
            if haptic is None:
                return
            structure = self._get_active_cognitive_map()
            if structure is None:
                return
            from huginn.metacog.haptic_descriptor import HapticDescriptor
            from huginn.metacog.structure_descriptor import StructureDescriptor
            svec = StructureDescriptor().encode(structure)
            hvec = HapticDescriptor().encode(haptic)
            ds = self._get_alignment_dataset()
            ds.add(
                svec, hvec, "structure", "haptic",
                metadata={
                    "tool": tool_name or "unknown",
                    "source": getattr(haptic, "source", "DFT"),
                    "iteration": getattr(self, "_iteration", 0),
                },
            )
            logger.debug(
                "alignment pair collected: tool=%s iter=%d total=%d",
                tool_name, getattr(self, "_iteration", 0), ds.count(),
            )
        except Exception:
            logger.warning(
                "alignment pair collection failed (non-fatal)", exc_info=True,
            )



    def _get_evolution(self):
        """懒加载 EvolutionEngine, 避免实例化时就拉起日志和规则文件。"""
        if self._evolution is None:
            from huginn.evolution.engine import EvolutionEngine
            from huginn.evolution.logger import ExecutionLogger

            self._evolution = EvolutionEngine(logger=ExecutionLogger())
        return self._evolution
