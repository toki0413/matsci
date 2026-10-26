"""RCB harness 入口: 读 workspace/INSTRUCTIONS.md, 跑 huginn agent, 输出到 stdout.

ResearchClawBench 的 TaskRunner 通过 subprocess 跑 agent_cmd, 捕获 stdout.
本脚本作为 huginn 的 RCB adapter:
  python huginn/cli/rcb_runner.py --workspace <workspace>

agent 在 workspace 里工作 (cwd=workspace), 用 code_tool/bash_tool 读写文件,
最终产出 report/report.md. RCB 的 INSTRUCTIONS.md 模板已经很详细, system
prompt 只需简短研究导向.

ponytail: 不重复 RCB prompt 已有的内容, 不加交互式渲染, 纯文本输出.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from huginn.utils.runtime import HUGINN_DIR_NAME, get_runtime_home

logger = logging.getLogger(__name__)

# onnxruntime C++ 层 EP Error 走 fd 2 (cerr), env var + set_default_logger_severity
# 都拦不住 (run_task.py 用 stderr=STDOUT 合并到 _agent_output.jsonl, 472 行/96s 淹死
# agent 实际输出). 直接 dup2 把 fd 2 重定向到 devnull, 所有继承的子进程也跟着静音.
# 副作用: huginn json_logging 的 StreamHandler(sys.stderr) 也进 devnull, RCBench
# 只关心 agent stdout, 内部诊断日志不重要.
# 升级路径: dup2 到 ws/.huginn/stderr.log 保留诊断, 但当前 devnull 是最小 diff.
# ponytail: 临时把 stderr 重定向到文件而非 devnull, 之前 devnull 吞了
# Traceback 导致 Step 2 崩溃无法定位. onnxruntime 刷屏仍进文件不淹 agent stdout.
# 升级路径: 确认崩溃点后, 若 onnxruntime 已不再刷屏, 可回 devnull.
try:
    _stderr_log = get_runtime_home() / "rcb_stderr.log"
    _stderr_log.parent.mkdir(parents=True, exist_ok=True)
    _stderr_fd = os.open(str(_stderr_log), os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    os.dup2(_stderr_fd, 2)
    os.close(_stderr_fd)
except OSError:
    logger.debug("best-effort op failed", exc_info=True)

# 全局 agent 引用: asyncio.wait_for 取消 run() 时, rcb_huginn.py 入口能拿到
# agent 实例再调一次 reflection, 确保 evolution 记录最后的 tool 失败.
# ponytail: 不改 run() 签名, 用模块级变量. ceiling: 多任务并发跑会互相覆盖.
_last_agent_for_reflection: Any = None

# 在 import huginn 之前关掉秒级限流 — RCB 任务的 prompt 长 + 工具多,
# 默认 5000 tokens/s 会在第一轮就超限. RCB 是离线评测, 不需要限流.
# P1-A7: 7 个评测关键 patch 改强制赋值, 防外部 env 让 patch 失效.
os.environ["HUGINN_RATE_LIMIT_ENABLED"] = "0"
os.environ.setdefault("HUGINN_RATE_LIMIT_TOKENS_PER_TURN", "500000")
# 允许本地沙箱执行 code_tool/bash_tool — RCB subprocess 没有 docker, 用本地 python
os.environ["HUGINN_ALLOW_LOCAL_BASH"] = "1"
# HUGINN_CACHE_DIR 可能被设成空串, 导致 LongTermMemory 用相对路径 memory.db,
# 在 RCB workspace cwd 下 sqlite WAL 创建失败. 强制用绝对路径.
if not os.environ.get("HUGINN_CACHE_DIR"):
    os.environ["HUGINN_CACHE_DIR"] = str(get_runtime_home())
# Benchmark 场景用 CSM 子集: 3-step 映射 S1/S4/S6+S7, 不再全 skip (Task 18, R8 减法修正).
# ponytail: S7 自修改仍走 (Task 2), 只跳过 compaction — 见 reflection.py L245.
os.environ["HUGINN_CSM_SUBSET_MODE"] = "1"
# Bandit Q-table 持久化路径: 跨 RCB task 累积 bandit 状态.
os.environ.setdefault(
    "HUGINN_BANDIT_Q_PATH",
    str(get_runtime_home() / "rcb_cross_task" / "bandit_q.json"),
)
# Benchmark Mode hint: agent core.py 读此 env var 注入 system prompt.
# RCB 路径必须严格按 paper 方法实现, judge 会按 paper 方法打分.
os.environ.setdefault(
    "HUGINN_BENCHMARK_MODE_PROMPT",
    "BENCHMARK MODE: This is a benchmark task scored against the reference paper. "
    "Implement the EXACT methodology from the paper (e.g., VAE+GPR, not RF+fingerprint "
    "substitution). If compute budget prevents full implementation, implement as much "
    "of the paper's pipeline as possible AND write a 'Negative Results' section in "
    "report.md comparing your metrics to the paper's reported metrics (e.g., 'our "
    "MAE 49.93K vs paper's LOOCV MAE 13K, gap explained by...'). Substituting the core "
    "method without justification scores 0.",
)
# Sandbox 路径阻塞: rcb_runner 不强制阻塞额外路径, 由用户 env var 覆盖.
os.environ.setdefault("HUGINN_SANDBOX_BLOCKED_PATHS", "")
# RCB 场景 compaction 保留前 2 条 root (task + Step 1 checklist) — 修同伦断裂 (σ₂)
os.environ.setdefault("HUGINN_KEEP_ROOT_N", "2")
# F3: σ₂ 半修补全 — 位置切片保不到 Step 1 checklist prompt (在 msgs[2:4]),
# 改用内容 marker 标 root. 同时保 FCM winner_plan (F4) 和 compass 不被 compaction drop.
# ponytail: marker 选 ## 标题, 足够独特不会误匹配普通正文.
os.environ.setdefault(
    "HUGINN_ROOT_MARKERS",
    "## Methodology Checklist;## Selected Execution Plan;## Report Coverage Compass;## Intuitive Gamer",
)
# RCB 场景跳过 Rust sandbox — 它在 RDKit+sklearn GPR 场景静默崩溃返回空 stderr.
# P0-7: 真实开关是 HUGINN_USE_RUST_SANDBOX (bash_tool 只读它); 旧写的
# HUGINN_NO_RUST_SANDBOX 全仓无人读 = 死配置, 已改为强写真实开关为 0.
os.environ["HUGINN_USE_RUST_SANDBOX"] = "0"
# RCB 跑分关掉 LLM decider — 跑分需要确定性, run_cognitive 的规则版 decide_fn 已够用.
# 生产路径 (deli_research/cli/routes) 不设这个变量, 默认开 LLM decider.
os.environ["HUGINN_COGNITIVE_LLM_DECIDER"] = "0"
# RCB 场景关熔断器 — file_read_tool 误触发 circuit_open 阻止 agent 读文件 (σ₇)
os.environ["HUGINN_HEALTH_MONITOR"] = "0"
# RCB 场景关循环检测 — agent 反复跑 code_tool 是正常行为, 误判为 loop (σ₈).
# streaming.py 读 FeatureFlags, 唯一真实开关是 HUGINN_FEATURE_LOOP_DETECTOR;
# 旧的 HUGINN_SKIP_LOOP_DETECTOR 全仓无人读, 已删除 (契约收敛).
os.environ["HUGINN_FEATURE_LOOP_DETECTOR"] = "false"

# === 从拆分模块 re-export (rcb_runner 作为向后兼容聚合入口) ===
import contextlib  # noqa: E402

from huginn.cli.rcb.audit import (  # noqa: E402
    _lint_report_markers,
    _step2_outputs_gate,
    _step2_substitution_audit,
)
from huginn.cli.rcb.fallback import (  # noqa: E402,F401
    build_retry_budget,
    generate_fallback_figures,
    step2_5_report_fallback,
)
from huginn.cli.rcb.prompt_builders import (  # noqa: F401,E402  re-export for backward compat
    _legacy_build_step2_prompt,
)
from huginn.cli.rcb_critique import (  # noqa: E402
    adversarial_critique,
)
from huginn.cli.rcb_fork_merge import (  # noqa: E402
    _FCM_PERSPECTIVES,
    _extract_sci_numbers,
    _reproduction_gate,
    anneal_fork_count,
    fork_critique_merge,
    judge_fork_reports,
)
from huginn.utils.common import now_iso  # noqa: E402

# Backward-compatible aliases (self-checks + internal callers use _-prefixed names)
_build_retry_budget = build_retry_budget
_generate_fallback_figures = generate_fallback_figures
_step2_5_report_fallback = step2_5_report_fallback

from huginn.agent_config import resolve_tool_budget  # noqa: E402,I001
from huginn.cli.rcb.audit import (  # noqa: E402,F401
    _METRIC_RE,
    _derive_gap_type,
    _infer_beta_1_simple,
    _recompute_report_metrics,
    _should_retry_execute,
    _write_directive_rejection,
)
from huginn.cli.rcb.self_checks import (  # noqa: F401,E402  re-export for backward compat
    self_check_a2,
    self_check_a3,
    self_check_a4,
    self_check_v14_all,
    self_check_v14_comprehensive,
    self_check_v14_p234,
    self_check_v14_task1,
    self_check_v14_task2,
    self_check_v14_task3,
    self_check_v14_task4,
    self_check_v14_task6,
    self_check_v14_task8,
    self_check_v15_task3,
    self_check_v15_task4,
)
from huginn.cli.rcb.smoke import rcb_smoke_test as _rcb_smoke_test  # noqa: E402,F401
from huginn.cli.rcb_audit import (  # noqa: E402,F401  backward-compat re-export
    _checklist_item_parser,
    _ChecklistItem,
    _derivation_chain_audit,
    _llm_coverage_audit,
    _rcb_effort_floor,
    _report_coverage_compass,
    _time_slot_index,
)
from huginn.cli.rcb_cognition import (  # noqa: E402,F401  backward-compat re-export
    _append_observations_log,
    _collect_observations,
    _compute_v15_fields,
    _init_hypothesis_manifold,
    _record_abduction,
    _trigger_anomaly_hypothesis,
    _write_cognitive_evidence,
)
from huginn.cli.rcb_mcmc import (  # noqa: E402,F401  backward-compat re-export
    _run_mcmc_mode,
)
from huginn.cli.rcb_step2 import (  # noqa: E402,F401  backward-compat re-export
    _RCBStep2Ctx,
    _step2_execute,
)
from huginn.cli.rcb_step3 import (  # noqa: E402,F401  backward-compat re-export
    _step3_adversarial,
)
from huginn.cli.rcb_utils import (  # noqa: E402,F401  backward-compat re-export
    _METRIC_WHITELIST,
    _MODEL_VERSION,
    _NUMERIC_PAIR_RE,
    _cross_task_store,
    _detect_file_rewrite_stagnation,
    _detect_gpu_safe,
    _extract_numeric_targets,
    _infer_domain,
    _infer_task_id_from_workspace,
    _load_manifold,
    _make_simplex_id,
    _save_manifold,
)
from huginn.context_manager import get_context_window  # noqa: E402,I001  (full module import)


# === run() 阶段化拆分 =========================================================
# 原 run 是 1090 行 / nest 7 的单体 (全仓最深). 按「环境装配 → agent 装配 →
# Step 1 发现 → Step 2/3 执行验证 → 收尾」切段, run 只做参数校验 + 串接.
# 每段 ≤50 行 / nest ≤4, 语义逐行保持等价.


@dataclass
class _RcbAgentBundle:
    """agent 装配产物 — HuginnAgent + 周边单例 + 两个闭包."""

    agent: Any
    step2_filter: set[str]
    mem_mgr: Any
    kg: Any
    kb: Any
    persona: Any
    persona_name: str
    thread_id: str
    stream_chat: Any
    csm_advance: Any


@dataclass
class _RcbDiscovery:
    """Step 1 / 1.2 / 1.5 / 1.7 的产物 — 供 Step 2 / Step 3 消费."""

    checklist: str
    scan_text: str
    fcm: dict
    target_chains: list
    trace_path: Path | None


def _disable_restricted_python_validation() -> None:
    """按 env 关闭 L1 restricted_python AST 校验."""
    # L1 restricted_python AST 校验: 默认开 (HUGINN_RESTRICTED_PYTHON=1).
    # 之前启动时 monkey-patch 禁用 AST 校验, 让 7 层沙箱只剩 3 层生效.
    # ponytail: 失败时降级到 L4 subprocess 沙箱, 不让 RCB 启动崩.
    restricted = os.environ.get("HUGINN_RESTRICTED_PYTHON", "1")
    if restricted == "0":
        try:
            import huginn.security.restricted_python as _rp

            _rp.validate_code = lambda code: None  # type: ignore[method-assign]
        except ImportError:
            logger.debug("best-effort op failed", exc_info=True)


def _redirect_ml_cache_env(ws: Path) -> None:
    """把 torch/HF 缓存路径重定向到 workspace 内."""
    # ML 缓存路径重定向到 workspace, 避免 TRAE 沙箱拦截 ~/.cache 写入.
    # torch.save('xxx.pt') 在 C:\tmp\ 会被沙箱拦, 重定向 TORCH_HOME 解决.
    ml_cache = ws / ".huginn_cache" / "ml_cache"
    ml_cache.mkdir(parents=True, exist_ok=True)
    for env_key in (
        "TORCH_HOME",
        "HF_HOME",
        "XDG_CACHE_HOME",
        "TRANSFORMERS_CACHE",
        "HF_HUB_CACHE",
    ):
        os.environ.setdefault(env_key, str(ml_cache))


def _configure_gpu_ort_noise() -> None:
    """GPU 策略 + onnxruntime 日志降噪 + paddle 屏蔽."""
    # GPU 策略: 沿用入口 _detect_gpu_safe() 的判定, 这里不重复检测.
    if os.environ.get("HUGINN_TORCH_DEVICE") != "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["ORT_DISABLE_TENSORRT_EP"] = (
        "1"  # 直接禁 TensorRT EP, 不让它 try→fail→log
    )
    os.environ["ORT_DISABLE_CUDA_EP"] = "1"  # 同上, 禁 CUDA EP
    os.environ.setdefault("ORT_LOGGING_LEVEL", "3")  # 抑制 Warning 及以下
    # 再 monkey-patch onnxruntime logger severity, 兜底 env var 不生效的 EP Error 输出.
    # 必须在 import huginn 之前调, 否则 onnxruntime 已被第三方 lib 触发 import.
    try:
        import onnxruntime as _ort_init

        _ort_init.set_default_logger_severity(3)  # 3=Error, 4=Fatal
    except (ImportError, Exception):
        logger.debug("best-effort op failed", exc_info=True)
    sys.modules.setdefault("paddle", None)
    sys.modules.setdefault("paddlepaddle", None)


def _prepare_runtime_environment(ws: Path) -> Path:
    """RCB 进程级环境装配 — 沙箱校验 / 缓存重定向 / GPU 与 ORT 降噪.

    返回 workspace 内的 RCB 缓存目录 (snapshot 与 ML 缓存共用).
    """
    _disable_restricted_python_validation()
    _redirect_ml_cache_env(ws)
    _configure_gpu_ort_noise()

    # RCB subprocess 跑时主 memory.db 可能被 IDE/桌面端锁定 (sqlite WAL),
    # 改用 workspace 下的独立缓存目录. RCB 是无状态离线评测, 不需要跨任务记忆.
    rcb_cache = ws / ".huginn_cache"
    rcb_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HUGINN_CACHE_DIR"] = str(rcb_cache)
    return rcb_cache


def _load_resume_iter(task_id: str, ws: Path) -> int:
    """Task 3: Checkpoint resume — 找到就续跑, audit chain 校验失败则从头开始."""
    try:
        from huginn.runtime.checkpoint import load_checkpoint, resume_from_checkpoint

        resumed = load_checkpoint(task_id, ws)
        if resumed is not None:
            print(f"[Resume] found checkpoint at step {resumed.step_id}", flush=True)
            resume_iter = max(0, resume_from_checkpoint(resumed, ws) - 1)
            print(f"[Resume] continuing from iter {resume_iter}", flush=True)
            return resume_iter
    except Exception as _e:
        print(f"[Resume] failed, starting fresh: {_e}", flush=True)
    return 0


def _apply_extreme_mode(cfg: Any) -> Any:
    """v6 极限模式: 解除一切限制, 性能优先 (更高思考强度 + 更长任务轨迹)."""
    from huginn.config import HuginnConfig

    os.environ.setdefault("HUGINN_EXTREME_DISPATCH", "1")
    os.environ.setdefault("HUGINN_USE_COGNITIVE_MAP", "1")
    # P0-B: flag-gated 系统统一 — extreme 模式统一开启 mental_imagery /
    # KG / cognitive_map 三套 flag-gated 机制, 让 RCB 跑分中都能触发.
    os.environ.setdefault("HUGINN_USE_MENTAL_IMAGERY", "1")
    os.environ.setdefault("HUGINN_USE_KNOWLEDGE_GRAPH", "1")
    # ponytail: 让 _check_stuck VF2 cycle 和 StructureCognitiveMap 15 个 transform 真触发
    os.environ.setdefault("HUGINN_THINKING", "high")
    # LoopDetector 在 RCB 全程保持关闭: 模块顶部已强制写入
    # HUGINN_FEATURE_LOOP_DETECTOR="false" (agent 反复跑 code_tool 属正常行为, 误判为
    # loop σ₈). 不引入会误杀长程 run 的控制流.
    # v7 长任务: extreme 模式同时放宽 autoloop stop 阈值, 允许 200+ 步轨迹.
    os.environ.setdefault("HUGINN_MAX_CONSECUTIVE_FAILURES", "50")
    os.environ.setdefault("HUGINN_MAX_REFINES", "50")
    os.environ.setdefault("HUGINN_MAX_PIVOTS", "20")
    os.environ.setdefault("HUGINN_DARWIN_STAGNATION_LIMIT", "15")
    # P4 Task 26.5: extreme 模式启用 persistent goal mode + 放开 wall_clock 预算到天级.
    # 700 万 tool call 长程愿景的挂钟抓手: 7M × 100ms ≈ 8 天, 需天级 wall_clock.
    os.environ.setdefault("HUGINN_PERSISTENT_GOAL_MODE", "1")
    os.environ.setdefault("HUGINN_RCB_TIMEOUT", "86400")
    # P3-2: 删除 HUGINN_RCB_SWARM setdefault — dispatch_parallel 实际只走
    # asyncio.gather, 从不查这个 flag. HUGINN_SWARM_DISTRIBUTED 保留 (跨进程队列).
    os.environ.setdefault("HUGINN_SWARM_DISTRIBUTED", "0")
    # P2-2 / Round 8: 兑现"完整实现但全仓无开启路径"的两个 router.
    # 用 FeatureFlags 的 runtime 覆盖 (最高优先级, 与构造时序无关): extreme = 强制开.
    from huginn.feature_flags import FeatureFlags as _FeatureFlags

    _FeatureFlags.shared().enable("context_router")
    _FeatureFlags.shared().enable("task_tool_router")
    cfg = HuginnConfig.from_env()  # 重读 env 拿 thinking
    print(
        "[EXTREME MODE] thinking=high, max_tool_calls=1200, context_budget=model-window, autoloop thresholds 50/50/20/15, persistent_goal=on, wall_clock=86400s",
        flush=True,
    )
    return cfg


def _create_persistent_goal(ws: Path) -> tuple[Any, str | None]:
    """P5: persistent goal mode — 创建 active goal with wall_clock_budget.

    返回 (goal_store, goal_id); 未开启或失败返回 (None, None).
    """
    if os.environ.get("HUGINN_PERSISTENT_GOAL_MODE", "0") != "1":
        return None, None
    try:
        from huginn.autoloop.goal_store import get_goal_store

        timeout_s = float(os.environ.get("HUGINN_RCB_TIMEOUT", "7200"))
        goal_store = get_goal_store()
        goal = goal_store.create_goal(f"RCB {ws.name}")
        goal_store.update_goal(
            goal.id,
            wall_clock_budget_seconds=timeout_s,
            started_at=now_iso(),
        )
        print(
            f"[P5] persistent goal: budget={timeout_s}s, goal_id={goal.id[:8]}",
            flush=True,
        )
        return goal_store, goal.id
    except Exception as _e:
        print(f"[P5] goal creation warning: {_e}", flush=True)
        return None, None


def _wall_clock_expired(goal_store: Any, goal_id: str | None) -> bool:
    """P5: 查 goal 挂钟预算是否耗尽. 无 goal 或未开 P5 返 False."""
    if goal_id is None or goal_store is None:
        return False
    try:
        return goal_store.wall_clock_expired(goal_id)
    except Exception:
        logger.debug("best-effort op failed", exc_info=True)
        return False


def _resolve_model(cfg: Any) -> tuple[Any, str] | None:
    """解析默认模型; 无可用模型返回 None (调用方打印错误并返 1)."""
    from huginn.models.registry import ModelRegistry

    registry = ModelRegistry.from_config(cfg)
    alias = registry.default_alias()
    if alias:
        model = registry.resolve(alias)
    elif cfg.provider and cfg.provider != "default":
        model = registry.resolve(f"{cfg.provider}/{cfg.model or 'auto'}")
    else:
        return None
    # RCB harness 从 stdout 检测 model 名 (run_task._detect_model)
    model_name = (
        getattr(model, "name", None) or getattr(model, "model_id", None) or str(model)
    )
    print(f"model: {model_name}", flush=True)
    return model, model_name


def _system_prompt_intro(ws: Path) -> str:
    """system prompt 开场: workspace 路径 + 工具操作事实."""
    return (
        f"You are an autonomous scientific research agent. "
        f"Your workspace is: {ws}\n"
        f"Current working directory IS the workspace. All relative paths "
        f"(data/, related_work/, code/, outputs/, report/) resolve from here.\n"
        f"Follow INSTRUCTIONS.md as your primary guide — it defines the task.\n"
        f"Prefer real implementations over shortcuts; document failures honestly.\n\n"
        "## Tool facts (sandbox constraints, not priorities)\n"
        "- code_tool: run Python for ALL code execution. This is your PRIMARY tool.\n"
        "  Sandbox BLOCKS open() and os inside code_tool — CANNOT write files via code_tool.\n"
        "- bash_tool: pip install, run scripts. On errors, stderr IS in error_detail field.\n"
        "- execute: runs shell commands (Windows PowerShell). Use for quick one-liners.\n"
        "- file_write_tool: CREATE or OVERWRITE text files (report.md, code/*.py). "
        "Pass FULL content each time.\n"
        "- matplotlib.savefig() WORKS (library code, not AST-scanned) — use it for figures.\n"
        "- code_tool security scanner may false-positive on eval() in torch/numpy — "
        "if so, write script via file_write_tool and run with bash_tool.\n"
        "- file_read_tool/glob/grep: explore data/ and related_work/.\n"
        "- web_search_tool: verify constants, methods, or edge cases.\n"
        "- NOTE: In Step 1 (read-only phase), code_tool/bash_tool/file_write_tool "
        "are NOT available. Use file_read_tool/glob/grep/web_search_tool only. "
        "They unlock in Step 2.\n\n"
    )


def _system_prompt_rules() -> str:
    """system prompt 操作规则 + 分阶段协议."""
    return (
        "## Operating rules\n"
        "- Every response before task completion MUST include a tool call. "
        "Text-only response = task termination.\n"
        "- Push through errors: debug, install missing packages, try alternatives.\n"
        "- Write report/report.md EARLY, then OVERWRITE as you add results.\n\n"
        "## PHASED PROTOCOL (MANDATORY — agent repeatedly fails by over-engineering)\n"
        "Phase 1 (calls 1-10): Explore data, read instructions, basic EDA. NO modeling yet.\n"
        # A3: 退役 MODEL COMPLEXITY CEILING — 路线图 P1-A3 / N5.
        # 旧 "Phase 2: NO deep learning yet" 与 RCB 核心方法 (GNN/VAE/Transformer)
        # 正面冲突, 训练 agent "先交卷后补做". 数学/物理任务该用 DL 就用, 由
        # Step-2 substitution audit 兜底 (实现痕迹缺失即回退执行).
        "Phase 2 (calls 11-20): Fit ONE model matching the paper's methodology + 2-3 figures. "
        "Use whatever model class the paper specifies (GNN/VAE/DL/GP all OK) — "
        "complexity上限由 paper 决定, 不由 harness 预设.\n"
        "Phase 3 (call 20 MANDATORY): WRITE report/report.md NOW with file_write_tool. "
        "Use what you have — incomplete results are fine, you will update later. "
        "If you reach call 25 with no report/report.md, STOP all analysis and "
        "write report.md skeleton (Abstract + Method + whatever Results you have).\n"
        "Phase 4 (calls 26-60): Iterate — add models, new figures, then UPDATE report.md.\n\n"
    )


def _system_prompt_deliverables() -> str:
    """system prompt 交付纪律段."""
    return (
        "## DELIVERABLE DISCIPLINE\n"
        "- Re-read INSTRUCTIONS before writing report. List ALL required quantities.\n"
        "- Each missing quantity = 0 for that criterion. Do them ALL.\n"
        "- Quantitative results REQUIRE numeric values with units, not methodology description.\n"
        "- EXECUTED vs EXPECTED markers (P1-B4): EVERY numeric value in report.md MUST be "
        "tagged with one of: [EXECUTED] (you ran the code and got this number from a real "
        "output file), [EXPECTED] (the paper says this should be the value, you did NOT "
        "verify it), [NOT EXECUTED] (you couldn't run this part). Example: 'R²=0.79 "
        "[EXECUTED] (from outputs/r2.json)' vs 'training loss ≈ 0.3 [EXPECTED] (paper "
        "Table 2)'. A numeric value with NO marker = treated as fabricated. The grader "
        "lints report.md for untagged numbers and discounts them.\n"
        "- BASELINE COMPARISON (P1-A6): for every quantitative metric you report, include "
        "a side-by-side comparison to the paper's baseline value (from your Step 1 checklist) "
        "in a `## Baseline Comparison` table at the end of report.md. Columns: Metric / "
        "Your Value / Paper Baseline / Δ / Status (match/better/worse). If your value is "
        "WORSE than the paper baseline by >10%, add a one-line root-cause note. If you "
        "can't match the baseline after 3 attempts, mark it 'below baseline — fallback to "
        "paper reproduction' and ship the closest reproducible result. Submitting a metric "
        "with no baseline column = treated as unvalidated.\n"
        "- DL tasks (AlphaFold3/FuXi/Janus): CANNOT install/train in sandbox. "
        "Check data/ for pre-computed model outputs (.nc/.pt) → EVALUATE not TRAIN. "
        "If training needed, use simpler model to reproduce qualitative trend. "
        "NEVER spend >5 calls trying to install a large model.\n"
        "- Before report.md, list ALL required figures. Generate EACH with matplotlib "
        "→ report/images/. Verify with glob before claiming complete. "
        "Name descriptively: figure3_triangle.png. Reference as images/figure_name.png.\n\n"
    )


def _system_prompt_architecture() -> str:
    """system prompt 架构保真段."""
    return (
        "## ARCHITECTURE FIDELITY (physics-driven, not word-association)\n"
        "- Before writing model code, extract the PHYSICS of the target property "
        "from related_work/*. What symmetry/interaction/topology defines it?\n"
        "- For EACH architectural decision, justify how it captures THAT physics. "
        "Reject 'self-supervised → SimCLR' style word-association.\n"
        "- Write code/architecture_spec.md: PHYSICS / DECISION / LINK / GROUNDING per decision.\n"
        "- If code implements generic GCN/GAT with no physics-specific structure, redo it.\n\n"
    )


def _system_prompt_windows() -> str:
    """system prompt Windows shell 段."""
    return (
        "## WINDOWS SHELL (sandbox runs on Windows)\n"
        "- bash_tool runs on Windows PowerShell — NO head/tail/cat/grep/find/sed/awk.\n"
        "- To peek at a file: use file_read_tool, or python: "
        "`bash_tool python -c \"print(open('f.csv').read()[:1000])\"`.\n"
        "- To count lines: `bash_tool python -c \"print(sum(1 for _ in open('f.csv')))\"`.\n"
        "- To grep: use grep_tool, or python: "
        "`bash_tool python -c \"[print(l,end='') for l in open('f') if 'pat' in l]\"`.\n"
        "- Avoid shell pipes/heredocs. Write scripts to code/*.py via file_write_tool, run with bash_tool.\n\n"
    )


def _system_prompt_extension() -> str:
    """system prompt 可选扩展提案段."""
    return (
        "## EXTENSION PROPOSAL (OPTIONAL, advisory, does NOT affect baseline)\n"
        "- MAY append `## EXTENSION PROPOSAL`: Hypothesis / Why Novel / "
        "Prediction / Grounding. Omitting does NOT lower baseline score.\n"
    )


def _build_system_prompt(ws: Path) -> str:
    """system prompt: workspace 路径 + 工具操作事实. 让 INSTRUCTIONS.md 做 task gradient."""
    # ponytail: 删 CRITICAL override 层 (σ₄) — control loop 不替 LLM 决策"什么重要".
    return (
        _system_prompt_intro(ws)
        + _system_prompt_rules()
        + _system_prompt_deliverables()
        + _system_prompt_architecture()
        + _system_prompt_windows()
        + _system_prompt_extension()
    )


def _init_memory_manager(ws: Path, model: Any) -> Any | None:
    """Task 12: Memory 接线 — 默认跨任务共享 db, 让 self_model/curiosity 积累."""
    mem_mgr = None
    try:
        from huginn.memory.factory import build_memory_manager

        if os.environ.get("HUGINN_RCB_CROSS_TASK", "1") == "1":
            # 路径优先读 HUGINN_RCB_CROSS_TASK_DIR (rcb_huginn 入口设到 workspace 内,
            # 避开 TRAE 沙箱拦截 ~/.huginn/ 写入). 没设则回退 ~/.huginn/rcb_cross_task.
            mem_dir = Path(
                os.environ.get(
                    "HUGINN_RCB_CROSS_TASK_DIR",
                    str(get_runtime_home() / "rcb_cross_task"),
                )
            )
            mem_dir.mkdir(parents=True, exist_ok=True)
            print(f"[Memory] cross-task shared: {mem_dir}", flush=True)
        else:
            mem_dir = ws / HUGINN_DIR_NAME / "memory"
        # 共享构造路径: 与 server 用同一 build_memory_manager, 消除重复装配.
        mem_mgr = build_memory_manager(memory_dir=mem_dir, llm=model)
    except Exception as _e:
        print(f"[Memory] init warning: {_e}", flush=True)
    return mem_mgr


def _ingest_kg_entities(kg: Any, ws: Path) -> int:
    """P0-B: 摄入 workspace 关键资源作为 entity (file entity + mentions, 不做 LLM 抽取).

    ponytail: 只加 file entity, 升级路径是 KG.add_entity 走 LLM 抽 paper 里的
    method/dataset/baseline.
    """
    n_ingested = 0
    if (ws / "INSTRUCTIONS.md").exists():
        kg.add_entity(
            "INSTRUCTIONS.md",
            "task_spec",
            source="rcb_runner",
            confidence=0.95,
            path=str(ws / "INSTRUCTIONS.md"),
        )
        n_ingested += 1
    rw_dir = ws / "related_work"
    if rw_dir.exists():
        for rw_file in rw_dir.glob("*"):
            if rw_file.is_file() and rw_file.suffix in (".md", ".txt", ".pdf"):
                kg.add_entity(
                    rw_file.name,
                    "reference_paper",
                    source="rcb_runner",
                    confidence=0.8,
                    path=str(rw_file),
                )
                n_ingested += 1
    return n_ingested


def _init_knowledge_graph(ws: Path) -> Any | None:
    """25.1: project KG 实例. 落 ws/.huginn/, 与 ContextBuilder.build_kg_text 同路径.

    失败降级 None — kg 是可选增强, 失败不阻塞主流程.
    """
    kg = None
    try:
        from huginn.kg.graph import ProjectKnowledgeGraph

        kg = ProjectKnowledgeGraph(ws / HUGINN_DIR_NAME)
        # P0-B: KG 初始化后摄入 workspace 关键资源作为 entity.
        if os.environ.get("HUGINN_USE_KNOWLEDGE_GRAPH", "0") == "1":
            try:
                n_ingested = _ingest_kg_entities(kg, ws)
                kg.save()
                print(
                    f"[KG] initialized: {n_ingested} entities ingested "
                    f"(HUGINN_USE_KNOWLEDGE_GRAPH=1)",
                    flush=True,
                )
            except Exception as _kie:
                print(f"[KG] ingest warning: {_kie}", flush=True)
        else:
            print(
                "[KG] initialized (HUGINN_USE_KNOWLEDGE_GRAPH=0, no ingest)", flush=True
            )
    except Exception as _e:
        print(f"[KG] init warning: {_e}", flush=True)
    return kg


def _init_persona(ws: Path) -> tuple[Any | None, str]:
    """Task 13: Persona 接线 — 从 ws.name 推断领域, 选对应 built-in persona."""
    persona_name = "default"
    persona = None
    try:
        from huginn.personas import PersonaManager

        persona_mgr = PersonaManager(workspace=str(ws))
        ws_name_lower = ws.name.lower()
        if any(k in ws_name_lower for k in ("astronom", "cosmo", "galaxy", "star")):
            persona_name = "reviewer"
        elif any(k in ws_name_lower for k in ("material", "dft", "vasp", "crystal")):
            persona_name = "dft_expert"
        elif any(k in ws_name_lower for k in ("md", "lammps", "molecular")):
            persona_name = "md_expert"
        persona = persona_mgr.get(persona_name)
    except Exception as _e:
        print(f"[Persona] init warning: {_e}", flush=True)
    return persona, persona_name


def _rcb_tool_filter() -> list[str]:
    """RCB 任务的工具白名单."""
    # file_write_tool 写文本文件 (report.md, code/*.py);
    # code_tool 的 sandbox 禁 open(), 只能跑分析/画图 (savefig 库代码不受限).
    # 不给 file_edit_tool: 它要求文件已存在, agent 误用 edit 写新文件会失败.
    return [
        "code_tool",
        "bash_tool",
        "file_read_tool",
        "file_write_tool",
        "glob",
        "grep",
        "web_search_tool",
        "self_observe",
        # G27: 数学工具解除 filter 屏蔽 — repro 数量级错误 (χ=1.0 vs 0.004) 的根因之一
        # 是四个外部适配器 tool_filter 把数学工具整体摘除 (audit 13 F1).
        # P1-B2: 补 unit_tool — Material/Physics 量纲一致性检查的关键工具
        "symbolic_math_tool",
        "lean_tool",
        "validate_tool",
        "unit_tool",
        # 视觉/子智能体工具 — 画图 + CV 验证 + 并行.
        # CUDA_VISIBLE_DEVICES="" 在 rcb_huginn.py 入口已设, 避免 cudnn 栈损坏.
        "plot_tool",  # Arial 20pt+ 加粗画图 (user rule)
        "image_analysis_tool",  # 反向 CV 分析自己生成的 PNG, 闭环视觉验证
        "vision_describe",  # 分层视觉描述 (OCR/CV), image criterion 评分闭环
        "subagent_tool",  # Layer 3: explore/coder/analyst 并行
    ]


def _create_agent(
    *,
    model: Any,
    system_prompt: str,
    mem_mgr: Any,
    persona_name: str,
    ws: Path,
    cfg: Any,
    ctx_budget: int,
    max_calls: int,
    max_per_tool: int,
) -> Any:
    """构造 HuginnAgent + 从 registry 注册工具."""
    from huginn.agent import HuginnAgent

    agent = HuginnAgent(
        model=model,
        system_prompt=system_prompt,
        memory_manager=mem_mgr,
        persona_name=persona_name,
        workspace=ws,
        max_tool_output_tokens=cfg.max_tool_output_tokens,
        context_budget_tokens=ctx_budget,
        max_tool_calls=max_calls,
        max_tool_calls_per_tool=max_per_tool,
        tool_filter=_rcb_tool_filter(),
        # RCB 是无人工 subprocess, 所有工具自动 approve
        auto_approve=True,
    )
    agent.register_tools_from_registry()
    return agent


def _apply_extreme_tool_filter(agent: Any, step2_filter: set[str]) -> None:
    """extreme 模式: 全量开放注册表工具 (黑名单减法)."""
    # RepoLaw 硬底线: extreme (RCBench) 模式强制注入 _DEFAULT_SANDBOX_PATH_RULES,
    # agent 不能改 INSTRUCTIONS.md / score.py / rubric.json 等关键文件.
    agent._permission_config.sandbox_mode = True
    # P0-A: extreme 模式全量开放注册表工具 — 之前手工枚举 14 个工具,
    # 28 个 sci/ + sim/ + design/ + causal/ 全被旁路 (注册表 145 工具仅暴露 29).
    # 根因: 白名单是"加法"思路, 应该是"减法" (黑名单 + 全量开放).
    from huginn.tools.registry import ToolRegistry as _TR  # noqa: N814

    blocked = {
        t.strip()
        for t in os.environ.get(
            "HUGINN_RCB_BLOCKED_TOOLS",
            # 默认空集 — extreme 模式信任 agent, 全开放.
            "",
        ).split(",")
        if t.strip()
    }
    all_registered = set(_TR.list_tools())
    step2_filter.update(all_registered - blocked)
    # blocked list 也作用于 hardcode 列表 — 否则 subagent_tool hardcode 进
    # _step2_filter, env var block 不掉, agent 嵌套调 subagent 死等拒绝的 execute.
    step2_filter -= blocked
    print(
        f"[P0-A] extreme 白名单全量开放: "
        f"{len(all_registered)} registered, "
        f"{len(blocked)} blocked, "
        f"{len(step2_filter)} exposed",
        flush=True,
    )


def _configure_step_filters(agent: Any, extreme: bool) -> set[str]:
    """v16.1: 阶段化工具白名单 — Step 1 只读, 返回 Step 2 完整白名单.

    根因: agent 在 Step 1 拿到完整工具集 + 任务描述, 直接做完任务写 report.md,
    Step 2 的 derivation audit / effort floor 全部失效 (gate 装在被跳过的房间里).
    """
    step2_filter = set(agent.tool_filter or [])
    if extreme:
        _apply_extreme_tool_filter(agent, step2_filter)
    # ponytail: Step 1 只需读和搜, lean_tool/symbolic_math_tool 是执行类工具会触发越权
    #   (agent 拿到 lean_tool 就开始写证明引擎, 拿到 symbolic_math_tool 就开始算).
    step1_filter = {
        "file_read_tool",
        "glob",
        "grep",
        "web_search_tool",
    }
    agent.tool_filter = step1_filter
    # refresh 而非 register: register 只刷 langchain_tools list, 不重建已编译的 graph.
    # 第一次 agent.chat 会 build_graph, 之后改 tool_filter 必须置 _agent_graph=None
    # 否则 Step 2 解锁后 graph 还是 Step 1 的只读工具集.
    agent.refresh_tools_from_registry()
    return step2_filter


def _resolve_image_path(msg: str, ws: Path) -> str | None:
    """扫 msg 里的图片路径 — 命中且存在就透传给 VisionRouter."""
    try:
        from huginn.vision.router import _IMAGE_PATH_RE

        match = _IMAGE_PATH_RE.search(msg or "")
        if not match:
            return None
        candidate = match.group(0)
        # 相对路径补成 workspace 绝对路径, VisionRouter 需要能 open
        path = Path(candidate)
        if not path.is_absolute():
            path = (ws / candidate).resolve()
        return str(path) if path.exists() else None
    except Exception:
        logger.debug(
            "best-effort op failed", exc_info=True
        )  # 视觉接入是增强, 失败不阻塞文本路径
        return None


def _last_ai_content(msgs: list) -> Any | None:
    """从消息列表倒序找最后一条 AIMessage, 返回其 content (无则 None)."""
    from langchain_core.messages import AIMessage

    # ponytail: 倒序扫描, ceiling: 累积所有 AIMessage content (多轮 tool 调用中间文本).
    for message in reversed(msgs):
        if isinstance(message, AIMessage):
            return getattr(message, "content", "")
    return None


async def _finalize_chat_gen(agent: Any, chat_gen: Any) -> None:
    """显式 aclose chat generator + 补跑反思闭环."""
    # async for 被 CancelledError 中断时不自动 aclose, generator 悬挂等 GC,
    # streaming.py 的 [FINALLY-REFLECT] 不跑 -> 反射链路断在第一层.
    if chat_gen is not None:
        with contextlib.suppress(Exception):
            await chat_gen.aclose()
    # 反思闭环: chat generator 被 asyncio.wait_for 取消时, streaming.py 的 finally
    # 块不保证执行 (async generator aclose 依赖 GC). 在这里显式调 reflection,
    # 确保 tool 失败被 evolution 记录.
    n_trs = (
        len(agent._session_state.tool_results_this_turn)
        if hasattr(agent, "_session_state")
        else -1
    )
    print(
        f"[RCB-FINALLY] entering, tool_results_this_turn={n_trs}, "
        f"has_reflection={hasattr(agent, '_run_post_turn_reflection')}",
        file=sys.stderr,
        flush=True,
    )
    try:
        if hasattr(agent, "_run_post_turn_reflection"):
            agent._run_post_turn_reflection()
            print("[RCB-FINALLY] reflection done", file=sys.stderr, flush=True)
    except Exception as _re:
        print(f"[RCB-FINALLY] reflection error: {_re}", file=sys.stderr, flush=True)


async def _stream_chat_once(
    agent: Any,
    ws: Path,
    default_thread_id: str,
    msg: str,
    step_label: str,
    tid: str | None = None,
    fresh_history: bool = False,
    extra_budget: int | None = None,
) -> str:
    """跑一轮 agent.chat, 流式打印 AIMessage, 返回最后的 AI 文本."""
    ai_text = ""
    chat_gen = None
    budget_override = _build_retry_budget(extra_budget)
    try:
        image_path = _resolve_image_path(msg, ws)
        chat_gen = agent.chat(
            msg,
            thread_id=tid or default_thread_id,
            image_path=image_path,
            include_history=not fresh_history,
            budget_override=budget_override,
        )
        async for chunk in chat_gen:
            content = _last_ai_content(chunk.get("messages", []))
            if not content:
                continue
            print(content, flush=True)
            ai_text = content
    except Exception as e:
        # 打 stdout 不打 stderr — _AsyncTee 只重定向 stdout, stderr 不进日志,
        # Step 1 三次空响应的异常就被吞了, 看起来像"LLM 没响应".
        import traceback as _tb

        print(f"ERROR [{step_label}]: {e}\n{_tb.format_exc()}", flush=True)
    finally:
        await _finalize_chat_gen(agent, chat_gen)
    return ai_text


def _make_stream_chat(agent: Any, ws: Path, thread_id: str) -> Any:
    """构造 _stream_chat 闭包 (绑定 agent / ws / thread_id).

    跑一轮 agent.chat, 流式打印 AIMessage, 返回最后的 AI 文本.

    tid: TFM 分叉用独立 thread 隔离 graph 内态 (历史从 ConversationTree 重建,
    thread 只影响 checkpoint 内态, 换 tid 无历史损失).

    fresh_history: True 时不拉 ConversationTree 历史. Step 3 retry 用 — fresh
    thread 的 prompt 已结构化注入所有必要 state, 不需要 Step 2 的 1M+ tokens
    历史累积. 之前换 tid 没解决超限, 因为 ConversationTree 是 agent 实例属性,
    thread_id 切换不影响它, history 照样被拉回来.

    extra_budget: A4 — Step 3 verdict≠pass 时追加专用工具调用预算. 传 50 →
    临时覆盖 max_tool_calls=50 + recursion_limit=250. 不影响 agent 全局配置,
    仅本次 chat 生效 (streaming.py chat() 已原生支持 budget_override).
    """

    async def _stream_chat(
        msg: str,
        step_label: str,
        tid: str | None = None,
        fresh_history: bool = False,
        extra_budget: int | None = None,
    ) -> str:
        return await _stream_chat_once(
            agent, ws, thread_id, msg, step_label, tid, fresh_history, extra_budget
        )

    return _stream_chat


def _csm_advance(agent: Any, signal_type: str, ctx: dict) -> None:
    """RCB step 开始时手动推 CSM 状态. advisory: 不允许就 no-op."""
    csm = getattr(agent, "_csm", None)
    if csm is None:
        return
    from huginn.cognitive_engine import TransitionSignal

    try:
        csm.transition(TransitionSignal(signal_type, ctx))
    except Exception:
        logger.debug("RCB CSM transition failed", exc_info=True)


def _ingest_workspace_kb(ws: Path) -> Any | None:
    """Step 0: KB 摄入任务数据 (Task 13).

    把 related_work/ + INSTRUCTIONS.md + data/ 灌进 workspace KB, 给后续
    target_chain 反推和 StepEvaluator 提供先验. 失败只 warn, 不阻塞主流程.
    """
    kb = None
    try:
        from huginn.knowledge.store import get_knowledge_base

        kb = get_knowledge_base(str(ws))
        for rw_file in (ws / "related_work").glob("*"):
            if rw_file.is_file() and rw_file.suffix in (".md", ".txt", ".pdf"):
                kb.add_document(rw_file.name, rw_file.read_bytes())
        if (ws / "INSTRUCTIONS.md").exists():
            kb.add_text(
                (ws / "INSTRUCTIONS.md").read_text(encoding="utf-8"),
                filename="INSTRUCTIONS.md",
            )
        data_dir = ws / "data"
        if data_dir.exists():
            for data_file in data_dir.glob("*"):
                if data_file.is_file() and data_file.suffix in (
                    ".csv",
                    ".json",
                    ".txt",
                    ".md",
                ):
                    kb.add_text(
                        data_file.read_text(encoding="utf-8", errors="ignore"),
                        filename=f"data/{data_file.name}",
                    )
        print("[Step 0] KB ingest done", flush=True)
    except Exception as _e:
        print(f"[Step 0] KB ingest warning: {_e}", flush=True)
    return kb


def _resolve_budgets(cfg: Any, model_name: str, extreme: bool) -> tuple[int, int, int]:
    """解析 (max_calls, max_per_tool, ctx_budget) — 极限模式放宽并跟随模型窗口."""
    # C3: 预算扩容 — 默认 150→400, 极限 300→600.
    # 走统一 resolve_tool_budget, 但传 audit 拟合出的经验拐点做 base,
    # 不并入 mode 表的 research=300 (那会退化掉分). 显式配置更高则 max 向上取.
    mode = "extreme" if extreme else "research"
    max_calls, max_per_tool = resolve_tool_budget(
        mode,
        base=(1200, 300) if extreme else (400, 50),
    )
    # 上下文预算对齐模型窗口, 不写死 200K. 模型 1M (deepseek-v4) 时用 1M,
    # 写死 200K 会在长任务里过早触发压缩, 截断本该继续探索的轨迹.
    ctx_budget = (
        get_context_window(model_name) if extreme else cfg.context_budget_tokens
    )
    return max_calls, max_per_tool, ctx_budget


def _make_rcb_bundle(
    *,
    agent: Any,
    step2_filter: set[str],
    mem_mgr: Any,
    kg: Any,
    ws: Path,
    persona: Any,
    persona_name: str,
) -> _RcbAgentBundle:
    """组装 _RcbAgentBundle — 绑定 kb / stream_chat 闭包 / csm 推进入口."""
    thread_id = f"rcb_{ws.name}"
    return _RcbAgentBundle(
        agent=agent,
        step2_filter=step2_filter,
        mem_mgr=mem_mgr,
        kg=kg,
        kb=_ingest_workspace_kb(ws),
        persona=persona,
        persona_name=persona_name,
        thread_id=thread_id,
        stream_chat=_make_stream_chat(agent, ws, thread_id),
        csm_advance=partial(_csm_advance, agent),
    )


def _setup_agent_bundle(
    ws: Path, cfg: Any, model: Any, model_name: str, *, extreme: bool, rcb_cache: Path
) -> _RcbAgentBundle:
    """装配 agent 及其周边单例 (system prompt / 工具 / memory / KG / persona)."""
    global _last_agent_for_reflection

    from huginn.snapshot import file_snapshot as _fs
    from huginn.tools import register_all_tools

    # snapshot 默认用 ~/.huginn/snapshots, RCB subprocess 跑时该目录可能被
    # IDE/桌面端锁定 (PermissionError). 重定向到 workspace 下的独立目录.
    _fs._SNAPSHOT_ROOT = rcb_cache / "snapshots"
    system_prompt = _build_system_prompt(ws)
    # 先注册工具到 ToolRegistry, 再让 agent 从 registry 拉取
    register_all_tools()

    # C3: 预算扩容 + 上下文窗口对齐模型 (见 _resolve_budgets).
    max_calls, max_per_tool, ctx_budget = _resolve_budgets(cfg, model_name, extreme)

    mem_mgr = _init_memory_manager(ws, model)
    kg = _init_knowledge_graph(ws)
    persona, persona_name = _init_persona(ws)
    agent = _create_agent(
        model=model,
        system_prompt=system_prompt,
        mem_mgr=mem_mgr,
        persona_name=persona_name,
        ws=ws,
        cfg=cfg,
        ctx_budget=ctx_budget,
        max_calls=max_calls,
        max_per_tool=max_per_tool,
    )
    step2_filter = _configure_step_filters(agent, extreme)

    # 暴露 agent 给 rcb_huginn.py 入口, 超时取消后能再调一次 reflection.
    _last_agent_for_reflection = agent

    return _make_rcb_bundle(
        agent=agent,
        step2_filter=step2_filter,
        mem_mgr=mem_mgr,
        kg=kg,
        ws=ws,
        persona=persona,
        persona_name=persona_name,
    )


def _build_step1_prompt(task_prompt: str) -> str:
    """Step 1 prompt: 论文方法论提取 checklist (含 [EXACT]/[VARIANT] 标注要求)."""
    step1_prompt = (
        f"Read the task instructions below AND explore related_work/ directory for reference papers.\n"
        f"PDF reading: file_read_tool supports line_offset/n_lines for .pdf — read the paper in "
        f"PAGES (offset through the file) to get the FULL Methods section, do NOT rely on web "
        f"fetching. The reference papers are local in related_work/.\n"
        f"Extract a METHODOLOGY CHECKLIST from the paper:\n"
        f"1. Core method components (model architecture, training protocol, key algorithms).\n"
        f"   For EACH component, label it [EXACT] (must reproduce as-specified) or [VARIANT]\n"
        f"   (justified deviation with reason). Default to [EXACT]. The label forces honesty\n"
        f"   about substitutions — Step 3 will audit them.\n"
        f"   For EACH [EXACT] component, you MUST also report:\n"
        f"     data_input_available: <read first 5 lines of relevant data/ file, report its header/columns>\n"
        f"     expected_output: <what the paper's method produces (plot type, numeric table, etc.)>\n"
        f"   If data/ already contains the method's INPUT (best-fit table, cached posterior, extracted\n"
        f"   spectrum), relabel as [EXACT-CACHED] — reproduction means LOAD+PLOT/ANALYZE, not re-run\n"
        f"   the full pipeline. Example: header says 'best-fit and 1σ' → [EXACT-CACHED], use\n"
        f"   numpy.random.normal + GetDist, NOT cobaya+CLASS from scratch.\n"
        f"2. Key quantitative metrics with the paper's BASELINE VALUES (e.g. 'R²=0.79, MAE=48K').\n"
        f"   These are the targets your results will be compared against in Step 3.\n"
        f"3. Critical implementation details that must be reproduced\n\n"
        f"Output the checklist as a numbered list. Be SPECIFIC (e.g. 'CGCNNConv with gating "
        f"and residual connections', not just 'GNN'). This checklist will guide your implementation "
        f"and will be used in Step 3's substitution audit and sanity check.\n\n"
        f"Task instructions:\n{task_prompt}"
    )
    return step1_prompt


async def _extract_checklist(stream_chat: Any, task_prompt: str) -> str:
    """跑 Step 1 提取 checklist; 空响应两级兜底, 避免 checklist=0 让 Step 3 被 skip.

    ponytail: 只对 step1 重试, 其他 step 空响应让 agent 自己在后续 iter 补.
    ceiling: 指数退避 + 更多次重试.
    """
    step1_prompt = _build_step1_prompt(task_prompt)
    checklist = await stream_chat(step1_prompt, "step1")
    if not checklist.strip():
        print("[step1] 空响应, 重试一次...", flush=True)
        checklist = await stream_chat(step1_prompt, "step1_retry")
    # 二级 fallback: LLM 连续两次空响应 (卡在 tool call 不出文本),
    # 用无 tool 的直接 prompt 强制输出 checklist. fresh_history=True 避免 tool
    # call 历史干扰. ceiling: 独立 thread + 禁用 tool_filter.
    if not checklist.strip():
        print("[step1] 仍为空, 强制直接输出 checklist (无 tool)...", flush=True)
        force_prompt = (
            "Based on the task instructions below, output a METHODOLOGY CHECKLIST "
            "as a numbered list. Do NOT call any tools. Just output the checklist text directly.\n\n"
            "For each component, label [EXACT], [VARIANT], or [EXACT-CACHED].\n"
            "Include key quantitative metrics with baseline values.\n\n"
            f"Task instructions:\n{task_prompt}"
        )
        checklist = await stream_chat(force_prompt, "step1_force", fresh_history=True)
    print(f"\n[checklist extracted: {len(checklist)} chars]\n", flush=True)
    return checklist


def _store_checklist(checklist: str, ws: Path) -> None:
    """G29: checklist 永驻 system_prompt — 写 stable_principles, 不进 compaction 范围."""
    if not (checklist and checklist.strip()):
        return
    try:
        from huginn.memory import store_stable_principle

        # 截断到 2000 字符防 persona 膨胀, 完整 checklist 在 ws/checklist.md
        store_stable_principle(
            f"[METHODOLOGY CHECKLIST]\n{checklist[:2000]}",
            source="rcb_step1_checklist",
        )
        # 同时写到 ws/checklist.md 让 agent 能 file_read_tool 读完整版
        (ws / "checklist.md").write_text(checklist, encoding="utf-8")
        print(
            "[G29: checklist stored as stable_principle + ws/checklist.md]", flush=True
        )
    except Exception as e:
        print(f"[G29: checklist store skipped: {e}]", flush=True)


def _build_target_chains(
    checklist: Any, kb: Any, model: Any, ws: Path, trace_task_id: str
) -> list:
    """Step 1.2: 目标链反推 (G62, Task 14).

    把 checklist 每条 Mode-A 目标反推成 required_results/methods/data/verification 链.
    ponytail: 天花板是单条粗粒度 item — LLM 拿到整段 checklist 做分解, 不会逐条对齐;
    升级路径是先用结构化 prompt 让 Step 1 直接输出 list[dict] (mode/item).
    """
    chains: list = []
    try:
        from huginn.metacog.target_chain import build_target_chains

        checklist_items = (
            checklist
            if isinstance(checklist, list)
            else [{"mode": "A", "item": (checklist or "")[:2000]}]
        )
        task_ctx = (
            (ws / "INSTRUCTIONS.md").read_text(encoding="utf-8")[:2000]
            if (ws / "INSTRUCTIONS.md").exists()
            else ""
        )
        # build_target_chains 是同步函数, 旧版 await list 报 "object list can't
        # be used in 'await' expression". 去掉 await.
        chains = build_target_chains(checklist_items, kb, model, task_ctx)
        _append_target_chain_trace(ws, len(checklist_items), len(chains), trace_task_id)
        print(f"[Step 1.2] built {len(chains)} target chains", flush=True)
    except Exception as _e:
        print(f"[Step 1.2] target chain warning: {_e}", flush=True)
    return chains


def _append_target_chain_trace(
    ws: Path, n_items: int, n_chains: int, trace_task_id: str
) -> None:
    """Step 1.2 的 meta_trace entry (写失败只 warn, 不影响 target chain)."""
    entry = {
        "iteration": 0,
        "ts": time.time(),
        "role": "target_chain",
        "attempted": f"build_target_chains for {n_items} item(s)",
        "found": f"{n_chains} chains built",
        "evidence": [],
        "limitations": [],
        "artifacts": [],
        "next_hint": "step1.5 structure scan",
        "darwin_score": 0.0,
        "supported_ratio": 0.0,
        # v14 Task 1: target_chain build 不在三族主映射里 → legacy.
        "simplex_id": _make_simplex_id(trace_task_id, 0, "target_chain"),
        "cochain_type": "legacy",
        "domain": _infer_domain(trace_task_id),
        "task_id": trace_task_id,
        "model_version": _MODEL_VERSION,
    }
    trace_path = ws / HUGINN_DIR_NAME / "meta_trace.jsonl"
    try:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with trace_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    except Exception as _e:
        print(f"[Step 1.2] trace write skipped: {_e}", flush=True)


def _build_scan_prompt(checklist: str) -> str:
    """Step 1.5 prompt: fast flat scan + 数学直觉结构识别."""
    scan_prompt = (
        "FAST FLAT SCAN with mathematical structure identification.\n"
        "Goal: identify structure + invariants for each checklist item, not just difficulty.\n\n"
        "For EACH checklist item, output a block:\n"
        "  [item N] structure: <type>\n"
        "    invariant: <one-line statement, or 'none'>\n"
        "    verifiable_via: <method, or 'none'>\n"
        "    anchor: <first-principles reference, or 'exploratory'>\n\n"
        "structure types (enum, pick one):\n"
        "  empirical | symmetry | asymptotic | dimensional | topological | probabilistic | algebraic\n"
        "  - empirical: pure data/observation, no known mathematical structure\n"
        "  - symmetry: invariant under transformation group (rotation, gauge, etc)\n"
        "  - asymptotic: limit behavior (t->inf, x->0) constrains the answer\n"
        "  - dimensional: Buckingham Pi / dimensional homogeneity must hold\n"
        "  - topological: invariant under continuous deformation (winding number, etc)\n"
        "  - probabilistic: distributional constraint (normalization, Bayes consistency, posterior contractibility)\n"
        "  - algebraic: equation/identity must hold exactly (eigenvalue eq, etc)\n\n"
        "verifiable_via (enum, pick one):\n"
        "  none | dimensional | asymptotic_limit | exact_formula | conservation_law | symmetry_argument | topological_invariant\n"
        "  - 'none' only if you genuinely cannot specify a verification method\n"
        "  - must correspond to the structure type (e.g. structure=dimensional → verifiable_via=dimensional)\n\n"
        "anchor:\n"
        "  - cite first-principles reference (e.g. 'black hole thermodynamics', 'Noether theorem', 'Bayes rule')\n"
        "  - 'exploratory' is valid — accept that structure may be uncertain at this stage\n\n"
        "Guidance (calibrated — avoid both over-claiming and under-claiming):\n"
        "- If the item involves a physical quantity with units → structure=dimensional, verifiable_via=dimensional.\n"
        "- If the item involves a Bayesian posterior / probability distribution → structure=probabilistic, verifiable_via=exact_formula (Bayes rule) or asymptotic_limit (posterior concentration).\n"
        "- If the item involves a conservation law (energy, charge, angular momentum) → structure=symmetry, verifiable_via=conservation_law.\n"
        "- If the item involves an exact equation (eigenvalue, polynomial root) → structure=algebraic, verifiable_via=exact_formula.\n"
        "- If the item involves a limit behavior (large-N, t→∞, x→0) → structure=asymptotic, verifiable_via=asymptotic_limit.\n"
        "- Only use empirical/none when the item is genuinely pure observation with no mathematical constraint.\n"
        "- Do NOT fabricate invariants you cannot verify — but DO identify invariants that genuinely apply.\n\n"
        "Constraints:\n"
        "- 1 tool call MAX (file_read or code_tool for quick check). Prefer 0.\n"
        "- Do NOT execute analysis. Do NOT write report.md.\n\n"
        "After all items, output a STRATEGY line:\n"
        "  STRATEGY: <one-line plan — order items by verifiable_via priority:\n"
        "    hard_check (dimensional/exact_formula/conservation_law) first to bank structural wins,\n"
        "    then soft_check (asymptotic/symmetry/topological), then empirical/none last>\n"
        f"\nChecklist:\n{checklist[:4000]}"
    )
    return scan_prompt


async def _run_structure_scan(model: Any, checklist: str) -> str:
    """Step 1.5: 单次 LLM 调用做结构扫描, 绕过 agent.chat 的 ReAct loop.

    原因: ReAct agent 拿到 prompt 后会直接调 tool 执行, 不给文本规划输出.
    Step 1.5 要的是纯文本 structure scan, 不允许 tool call.
    ponytail: 不另建 agent 实例 (省 memory), 直接调 model.invoke.
    """
    scan_text = ""
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        scan_msgs = [
            SystemMessage(
                content=(
                    "You are a mathematical structure scanner. Output ONLY text, "
                    "no tool calls. Identify the mathematical structure of each "
                    "checklist item and the invariant it must satisfy."
                )
            ),
            HumanMessage(content=_build_scan_prompt(checklist)),
        ]
        scan_resp = await asyncio.to_thread(model.invoke, scan_msgs)
        scan_text = (
            scan_resp.content if hasattr(scan_resp, "content") else str(scan_resp)
        )
        print(scan_text, flush=True)
    except Exception as _e:
        print(f"[Step 1.5 LLM call failed: {_e}]", flush=True)
        scan_text = ""
    print(f"\n[structure scan done: {len(scan_text)} chars]\n", flush=True)
    return scan_text


def _write_intuitive_gamer_trace(
    ws: Path, scan_text: str, trace_task_id: str
) -> Path | None:
    """写 role="intuitive_gamer" 的 Meta-Trace entry, 返回 trace 路径 (供 FCM 复用)."""
    entry = {
        "iteration": 0,
        "ts": time.time(),
        "role": "intuitive_gamer",
        "attempted": "fast flat scan with mathematical structure identification",
        "found": (scan_text or "")[:500],
        "evidence": [],
        "limitations": [
            "single-sample, no k-sampling (v8 upgrade)",
            "structure labels not schema-validated (v8: pydantic + Lean)",
        ],
        "artifacts": [],
        "next_hint": "execute hard_check items first to bank structural wins",
        "darwin_score": 0.0,
        "supported_ratio": 0.0,
        # v14 Task 1: intuitive_gamer = harmonic (imagination-driven, 拓扑探测).
        "simplex_id": _make_simplex_id(trace_task_id, 0, "intuitive_gamer"),
        "cochain_type": "harmonic",
        "domain": _infer_domain(trace_task_id),
        "task_id": trace_task_id,
        "model_version": _MODEL_VERSION,
    }
    trace_path = ws / HUGINN_DIR_NAME / "meta_trace.jsonl"
    try:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with trace_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print("[intuitive_gamer + math structure trace entry written]", flush=True)
        return trace_path
    except Exception as _e:
        print(f"[intuitive_gamer trace skipped: {_e}]", flush=True)
        return None


async def _run_plan_fork_merge(checklist: Any, scan_text: str, model: Any) -> dict:
    """Step 1.7: fork-critique-merge — k 路视角采样执行方案, 独立评审选优.

    成本: k+1 次 cheap LLM 调用 (无 tool, 纯文本).
    """
    fcm = await fork_critique_merge(checklist, scan_text, model, k=3)
    if fcm["winner_plan"]:
        print(
            f"[fcm: winner={fcm['winner_perspective']} scores={fcm.get('scores')}]\n",
            flush=True,
        )
        print(fcm["winner_plan"], flush=True)
    else:
        print("[fcm: all forks failed, Step 2 runs plan-free]", flush=True)
    return fcm


def _write_fcm_trace(fcm: dict, trace_path: Path | None, trace_task_id: str) -> None:
    """Step 1.7 的 meta_trace entry (复用 Step 1.5 的 trace 路径)."""
    if trace_path is None:
        print("[fcm trace skipped: no trace path]", flush=True)
        return
    entry = {
        "iteration": 0,
        "ts": time.time(),
        "role": "fork_critique_merge",
        "attempted": f"{len(_FCM_PERSPECTIVES)}-perspective plan sampling + independent critique",
        "found": f"winner={fcm['winner_perspective']} scores={fcm.get('scores')}",
        "evidence": [fcm["winner_plan"][:300]] if fcm["winner_plan"] else [],
        "limitations": [
            "plan-level critique only; trajectory-level fork deferred (v8 fcm-2)"
        ],
        "artifacts": [],
        "next_hint": "execute winner plan; merge insights as fallback options",
        "darwin_score": 0.0,
        "supported_ratio": 0.0,
        # v14 Task 1: FCM winner = gradient (task-driven plan 选路).
        "simplex_id": _make_simplex_id(trace_task_id, 0, "fork_critique_merge"),
        "cochain_type": "gradient",
        "domain": _infer_domain(trace_task_id),
        "task_id": trace_task_id,
        "model_version": _MODEL_VERSION,
    }
    try:
        with trace_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as _e:
        print(f"[fcm trace skipped: {_e}]", flush=True)


def _clear_step1_violations(ws: Path) -> None:
    """v16.1 越权检测兜底 — Step 1 应只输出 checklist 文本, 不该有产物.

    即使白名单漏 (web_search side effect / 1.5 误调 code_tool), 清空保证
    Step 2 gate 仍跑.
    """
    violations: list[str] = []
    report_md = ws / "report" / "report.md"
    if report_md.exists() and report_md.stat().st_size > 0:
        violations.append(f"report/report.md ({report_md.stat().st_size}b)")
    outputs_dir = ws / "outputs"
    extra = [
        f
        for f in (outputs_dir.glob("*") if outputs_dir.exists() else [])
        if f.is_file()
    ]
    if extra:
        violations.append(f"outputs/ ({len(extra)} files)")
    if not violations:
        return
    print(
        f"[v16.1 越权检测] Step 1 生成 {violations}, 清空强制 Step 2 重做", flush=True
    )
    try:
        report_md.unlink(missing_ok=True)
        for f in extra:
            f.unlink(missing_ok=True)
    except OSError as _e:
        print(f"[v16.1 清空失败: {_e}]", flush=True)


async def _run_discovery_phase(
    ws: Path,
    kb: Any,
    model: Any,
    prompt: str,
    bundle: _RcbAgentBundle,
    trace_task_id: str,
) -> _RcbDiscovery:
    """Step 1 / 1.2 / 1.5 / 1.7 — 方法论提取 + 目标链 + 结构扫描 + plan 评审."""
    stream_chat = bundle.stream_chat
    # 3 步认知循环: 用同一 thread_id 保持上下文连续, Step 2 能看到 Step 1 的 checklist.
    print("\n=== Step 1: Methodology Extraction ===\n", flush=True)
    bundle.csm_advance(
        "user_goal", {"goal": "understand problem and extract methodology"}
    )
    checklist = await _extract_checklist(stream_chat, prompt)
    _store_checklist(checklist, ws)
    target_chains = _build_target_chains(checklist, kb, model, ws, trace_task_id)

    print("\n=== Step 1.5: Intuitive Gamer + math structure scan ===\n", flush=True)
    scan_text = await _run_structure_scan(model, checklist)
    trace_path = _write_intuitive_gamer_trace(ws, scan_text, trace_task_id)

    print("\n=== Step 1.7: Plan Fork-Critique-Merge ===\n", flush=True)
    fcm = await _run_plan_fork_merge(checklist, scan_text, model)
    _write_fcm_trace(fcm, trace_path, trace_task_id)

    _clear_step1_violations(ws)
    # v16.1: 解锁 Step 2 完整工具集 — Step 1 已结束, gate 现在能拦得住.
    # refresh 触发 graph 重建, 否则 Step 2 用的还是 Step 1 的只读工具集.
    bundle.agent.tool_filter = bundle.step2_filter if bundle.step2_filter else None
    bundle.agent.refresh_tools_from_registry()
    return _RcbDiscovery(
        checklist=checklist,
        scan_text=scan_text,
        fcm=fcm,
        target_chains=target_chains,
        trace_path=trace_path,
    )


async def _run_substitution_audit(
    ws: Path, checklist: str, evals_history: list, stream_chat: Any
) -> None:
    """A3: silent substitution 结构性拦截 — [EXACT] 组件 ↔ 实现痕迹机械比对.

    缺失即回退执行. 不调 LLM 做语义判断, 纯正则 + 文件扫描.
    """
    try:
        audit = await _step2_substitution_audit(
            ws, checklist, evals_history, stream_chat
        )
        if audit["unresolved"]:
            print(
                f"[A3 audit] unresolved silent substitutions: "
                f"{[it['component'] for it in audit['unresolved']]}",
                flush=True,
            )
        elif audit["remediated"]:
            print(f"[A3 audit] remediated: {audit['remediated']}", flush=True)
        # 落盘供 Step 3 / 评分器引用
        try:
            (ws / HUGINN_DIR_NAME / "step2_audit.json").write_text(
                json.dumps(audit, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
        except OSError as _e:
            print(f"[A3 audit] write report failed: {_e}", flush=True)
    except Exception as _e:
        import traceback as _tb

        print(f"[A3 audit] crashed: {_e}\n{_tb.format_exc()}", flush=True)


async def _run_outputs_gate(ws: Path, stream_chat: Any) -> None:
    """A2: 产物级门控 — outputs/ 无真实 metrics 文件时禁止虚写 Results."""
    try:
        gate = await _step2_outputs_gate(ws, stream_chat)
        if gate["blocker"]:
            print(
                "[A2 gate] BLOCKER: outputs/ 无真实 metrics 文件, "
                "Step 3 应降权 Results claim",
                flush=True,
            )
        elif gate["remediated"]:
            print(
                f"[A2 gate] remediated: {len(gate['metrics_files'])} metrics files",
                flush=True,
            )
        try:
            (ws / HUGINN_DIR_NAME / "step2_outputs_gate.json").write_text(
                json.dumps(gate, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
        except OSError as _e:
            print(f"[A2 gate] write report failed: {_e}", flush=True)
    except Exception as _e:
        import traceback as _tb

        print(f"[A2 gate] crashed: {_e}\n{_tb.format_exc()}", flush=True)


async def _run_step3(
    ws: Path,
    model: Any,
    agent: Any,
    checklist: str,
    evals_history: list,
    stream_chat: Any,
    csm_advance: Any,
    *,
    persona: Any,
    kb: Any,
    mem_mgr: Any,
    task_id: str,
    persona_name: str,
) -> str:
    """Step 3 整体兜底 — 崩溃时返 "step3_crashed", 不阻塞磁盘上的 deliverables.

    retry/finalize 内部 try/except 只防单次调用崩溃, 但整个 _step3_adversarial
    仍可能因 OOM/系统杀进程等不可恢复异常退出. deliverables 已在磁盘上 (Step 2
    产出), Step 3 失败不应阻塞评分 — 评分读 report.md/figures.
    """
    try:
        return await _step3_adversarial(
            ws,
            model,
            agent,
            checklist,
            evals_history,
            stream_chat,
            csm_advance,
            persona=persona,
            kb=kb,
            mem_mgr=mem_mgr,
            cross_task_store=_cross_task_store,
            task_id=task_id,
            persona_name=persona_name,
        )
    except Exception as _e:
        import traceback as _tb

        print(
            f"[step3] _step3_adversarial crashed: {_e}\n{_tb.format_exc()}", flush=True
        )
        return "step3_crashed"


async def _run_verify_phase(
    *,
    ws: Path,
    model: Any,
    bundle: _RcbAgentBundle,
    checklist: str,
    evals_history: list,
    task_id: str,
) -> str:
    """Step 2 后置审计 (A3/A2) + Step 2.5 报告兜底 + Step 3 自验证."""
    await _run_substitution_audit(ws, checklist, evals_history, bundle.stream_chat)
    await _run_outputs_gate(ws, bundle.stream_chat)
    # Step 2.5 + Step 3 抽到模块级函数 — 闭包 _stream_chat / csm 作参数传入.
    await _step2_5_report_fallback(ws, bundle.stream_chat)
    return await _run_step3(
        ws,
        model,
        bundle.agent,
        checklist,
        evals_history,
        bundle.stream_chat,
        bundle.csm_advance,
        persona=bundle.persona,
        kb=bundle.kb,
        mem_mgr=bundle.mem_mgr,
        task_id=task_id,
        persona_name=bundle.persona_name,
    )


def _warn_cross_task_append(err: Exception) -> None:
    logger.warning("cross_task append failed: %s", err)


def _read_trace_entries(ws: Path, on_error: Any = None) -> list:
    """读 ws/.huginn/meta_trace.jsonl 全部可解析 entry (on_error 收 parse 异常)."""
    trace_path = ws / HUGINN_DIR_NAME / "meta_trace.jsonl"
    entries: list = []
    if not trace_path.exists():
        return entries
    with trace_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except Exception as _e:
                if on_error is not None:
                    on_error(_e)
                else:
                    logger.debug("trace entry parse failed", exc_info=True)
    return entries


def _flush_cross_task_store(ws: Path) -> None:
    """v14 Task 14: 跨 task Meta-Trace 累积, 供后续同 domain task 作 prior 查询.

    ponytail: HUGINN_CACHE_DIR 在 run() 入口被改到 ws/.huginn_cache, 所以
    CrossTaskStore() 默认落在 workspace-local — 跨 task 累积目前实际只在同
    workspace resume 场景生效. 升级路径: 显式传 db_path.
    """
    try:
        from huginn.metacog.cross_task_store import CrossTaskStore

        store = CrossTaskStore()
        entries = _read_trace_entries(ws, on_error=_warn_cross_task_append)
        for entry in entries:
            try:
                store.append(entry)
            except Exception as _e:
                _warn_cross_task_append(_e)
    except Exception as _e:
        logger.warning("cross_task_store init failed: %s", _e)


def _resolve_task_score(final_verdict: str) -> float | None:
    """v14 Task 18: 失败 trace 进训练池 — task score < 20 时标 negative sample.

    ponytail: agent 跑时拿不到 RCBench 最终评分, 用两层 fallback:
      1) env RCB_TASK_SCORE (RCBench 父进程注入子进程)
      2) critique verdict=fix_needed 视为 score<20 (spec 允许的代理)
    天花板: critique=pass 但评分 <20 的 case 漏掉; fix_needed 但评分 >=20 的
    case 误判. 升级路径: RCBench 显式传 score.
    """
    raw_score = os.environ.get("RCB_TASK_SCORE")
    if raw_score:
        try:
            return float(raw_score)
        except ValueError:
            logger.debug("best-effort op failed", exc_info=True)
    if final_verdict == "fix_needed":
        return 15.0  # 代理值: fix_needed 视为 <20
    return None


def _export_coevolution(ws: Path, trace_task_id: str, final_verdict: str) -> None:
    """v14 Task 17: 训练数据导出 — env HUGINN_COEVOLUTION=1 时导出 SFT/DPO 数据.

    ponytail: 失败不阻塞主流程, 只 log warning.
    """
    if os.environ.get("HUGINN_COEVOLUTION", "0").lower() not in ("1", "true", "yes"):
        return
    try:
        from huginn.training.darwin_exporter import DarwinRewardExporter

        exporter = DarwinRewardExporter()
        entries = _read_trace_entries(ws)
        n_sft = exporter.export_sft(entries, task_id=trace_task_id)
        n_dpo = exporter.export_dpo(entries, task_id=trace_task_id)
        task_score = _resolve_task_score(final_verdict)
        if task_score is not None and task_score < 20:
            n_failure = exporter.export_failure_trace(
                entries, trace_task_id, task_score
            )
            logger.info(
                "coevolution failure trace: n_failure=%d, task_score=%s",
                n_failure,
                task_score,
            )
        logger.info("coevolution export: n_sft=%d, n_dpo=%d", n_sft, n_dpo)
    except Exception as _e:
        logger.warning("coevolution export failed: %s", _e)


def _flush_bandit_episode() -> None:
    """bandit MDP: run 结束 flush 最后一条 episode 轨迹, 避免永不回传."""
    # ponytail: 失败静默, bandit 内部 catch.
    try:
        from huginn.agent.bandit_controller import EffortBandit

        EffortBandit.get_instance().end_episode()
    except Exception:
        logger.debug("bandit end_episode flush skipped", exc_info=True)


async def _run_pipeline_setup(
    ws: Path,
    cfg: Any,
    model: Any,
    model_name: str,
    prompt: str,
    task_id: str,
    *,
    extreme: bool,
    rcb_cache: Path,
) -> dict[str, Any]:
    """装配 agent bundle + 跑 Step 1 发现阶段."""
    # v14 Task 1: trace 里存短 task_id (剥时间戳后缀).
    trace_task_id = _infer_task_id_from_workspace(task_id)
    # P5: persistent goal mode — 主循环每轮查 wall_clock_expired.
    goal_store, goal_id = _create_persistent_goal(ws)
    bundle = _setup_agent_bundle(
        ws,
        cfg,
        model,
        model_name,
        extreme=extreme,
        rcb_cache=rcb_cache,
    )
    discovery = await _run_discovery_phase(
        ws,
        bundle.kb,
        model,
        prompt,
        bundle,
        trace_task_id,
    )
    return {
        "trace_task_id": trace_task_id,
        "goal_store": goal_store,
        "goal_id": goal_id,
        "bundle": bundle,
        "discovery": discovery,
    }


def _build_step2_ctx(
    *,
    ws: Path,
    model: Any,
    bundle: _RcbAgentBundle,
    discovery: Any,
    task_id: str,
    resume_from_iter: int,
    extreme: bool,
    instructions: Path,
    goal_store: Any,
    goal_id: Any,
    run_start_iso: str,
) -> _RCBStep2Ctx:
    """组装 Step 2 执行上下文 (字段顺序与原构造保持一致)."""
    return _RCBStep2Ctx(
        ws=ws,
        model=model,
        agent=bundle.agent,
        kb=bundle.kb,
        mem_mgr=bundle.mem_mgr,
        persona=bundle.persona,
        kg=bundle.kg,
        thread_id=bundle.thread_id,
        task_id=task_id,
        resume_from_iter=resume_from_iter,
        extreme=extreme,
        checklist=discovery.checklist,
        scan_text=discovery.scan_text,
        fcm=discovery.fcm,
        target_chains=discovery.target_chains,
        instructions=instructions,
        stream_chat_fn=bundle.stream_chat,
        rcb_csm_advance_fn=bundle.csm_advance,
        wall_expired_fn=partial(_wall_clock_expired, goal_store, goal_id),
        persona_name=bundle.persona_name,
        run_start_iso=run_start_iso,
    )


def _finalize_rcb_run(ws: Path, trace_task_id: str, final_verdict: Any) -> None:
    """收尾: 跨任务 store / coevolution 导出 / bandit episode flush."""
    _flush_cross_task_store(ws)
    _export_coevolution(ws, trace_task_id, final_verdict)
    _flush_bandit_episode()


async def _run_execution_and_verify(
    *,
    ws: Path,
    model: Any,
    task_id: str,
    instructions: Path,
    resume_from_iter: int,
    extreme: bool,
    setup: dict[str, Any],
) -> int:
    """Step 2 执行 → Step 2.5/3 验证 → 跨任务收尾."""
    from datetime import datetime

    bundle = setup["bundle"]
    discovery = setup["discovery"]
    step2_ctx = _build_step2_ctx(
        ws=ws,
        model=model,
        bundle=bundle,
        discovery=discovery,
        task_id=task_id,
        resume_from_iter=resume_from_iter,
        extreme=extreme,
        instructions=instructions,
        goal_store=setup["goal_store"],
        goal_id=setup["goal_id"],
        run_start_iso=datetime.now().isoformat(),
    )
    evals_history = await _step2_execute(step2_ctx)  # 供 Step 3 用

    final_verdict = await _run_verify_phase(
        ws=ws,
        model=model,
        bundle=bundle,
        checklist=discovery.checklist,
        evals_history=evals_history,
        task_id=task_id,
    )
    _finalize_rcb_run(ws, setup["trace_task_id"], final_verdict)
    return 0


async def _run_rcb_pipeline(
    *,
    ws: Path,
    cfg: Any,
    model: Any,
    model_name: str,
    prompt: str,
    instructions: Path,
    task_id: str,
    resume_from_iter: int,
    extreme: bool,
    rcb_cache: Path,
) -> int:
    """3 步认知循环主链路: 装配 → Step 1 发现 → Step 2 执行 → Step 2.5/3 验证 → 收尾."""
    setup = await _run_pipeline_setup(
        ws,
        cfg,
        model,
        model_name,
        prompt,
        task_id,
        extreme=extreme,
        rcb_cache=rcb_cache,
    )
    return await _run_execution_and_verify(
        ws=ws,
        model=model,
        task_id=task_id,
        instructions=instructions,
        resume_from_iter=resume_from_iter,
        extreme=extreme,
        setup=setup,
    )


def _prepare_rcb_run_kwargs(ws: Path, extreme: bool) -> int | dict[str, Any]:
    """RCB 主循环前置: INSTRUCTIONS 校验 + 环境装配 + 配置/模型解析."""
    instructions = ws / "INSTRUCTIONS.md"
    if not instructions.exists():
        print(f"ERROR: {instructions} not found", file=sys.stderr)
        return 1

    from huginn.config import HuginnConfig

    rcb_cache = _prepare_runtime_environment(ws)
    task_id = ws.name
    resume_from_iter = _load_resume_iter(task_id, ws)
    prompt = instructions.read_text(encoding="utf-8")

    cfg = HuginnConfig.from_env()
    # v6 极限模式: 解除一切限制, 性能优先 (不改默认值, 只在 --extreme 时 override).
    if extreme:
        cfg = _apply_extreme_mode(cfg)

    resolved = _resolve_model(cfg)
    if resolved is None:
        print("ERROR: no model configured", file=sys.stderr)
        return 1
    model, model_name = resolved

    return {
        "ws": ws,
        "cfg": cfg,
        "model": model,
        "model_name": model_name,
        "prompt": prompt,
        "instructions": instructions,
        "task_id": task_id,
        "resume_from_iter": resume_from_iter,
        "extreme": extreme,
        "rcb_cache": rcb_cache,
    }


async def run(
    workspace: str,
    extreme: bool = False,
    *,
    mcmc_mode: str | None = None,
    mcmc_steps: int = 7_000_000,
    mcmc_chains: int = 4,
    mcmc_checkpoint_interval: int = 10_000,
    mcmc_se3: bool = False,
    mcmc_se3_angle_sigma: float = 30.0,
    mcmc_haptic: bool = False,
    mcmc_haptic_temperature: float = 1.0,
    mcmc_alignment: bool = False,
    mcmc_alignment_temperature: float = 1.0,
) -> int:
    """RCB 入口: 参数校验 + 环境装配后交给 _run_rcb_pipeline 跑 3 步循环."""
    ws = Path(workspace).resolve()
    # Task 4.1: --mcmc-mode 走独立 MCMC 路径, 不跑 RCB agent 主循环.
    # 不传时 mcmc_mode=None, 下面走原 RCB 逻辑 100% 不变.
    if mcmc_mode is not None:
        return await _run_mcmc_mode(
            ws=ws,
            task_id=ws.name,
            mode=mcmc_mode,
            n_steps=mcmc_steps,
            n_chains=mcmc_chains,
            checkpoint_interval=mcmc_checkpoint_interval,
            se3_enabled=mcmc_se3,
            se3_angle_sigma=mcmc_se3_angle_sigma,
            haptic_enabled=mcmc_haptic,
            haptic_temperature=mcmc_haptic_temperature,
            alignment_enabled=mcmc_alignment,
            alignment_temperature=mcmc_alignment_temperature,
        )
    pipeline_kwargs = _prepare_rcb_run_kwargs(ws, extreme)
    if isinstance(pipeline_kwargs, int):
        return pipeline_kwargs
    return await _run_rcb_pipeline(**pipeline_kwargs)


def _add_mcmc_chain_args(parser: argparse.ArgumentParser) -> None:
    """注册 MCMC 链级参数 (mode / steps / chains / checkpoint 间隔)."""
    parser.add_argument(
        "--mcmc-mode",
        choices=["single", "multi"],
        default=os.environ.get("HUGINN_MCMC_MODE"),
        help="MCMC 模式: single=单链, multi=多链+R-hat. 不传走原 RCB 路径",
    )
    parser.add_argument(
        "--mcmc-steps",
        type=int,
        default=int(os.environ.get("HUGINN_MCMC_STEPS", "7000000")),
        help="MCMC 总步数 (multi 模式下每链 steps//chains)",
    )
    parser.add_argument(
        "--mcmc-chains",
        type=int,
        default=int(os.environ.get("HUGINN_MCMC_CHAINS", "4")),
        help="多链数 (仅 multi 模式)",
    )
    parser.add_argument(
        "--mcmc-checkpoint-interval",
        type=int,
        default=int(os.environ.get("HUGINN_MCMC_CHECKPOINT_INTERVAL", "10000")),
        help="checkpoint 落盘间隔 (步)",
    )


def _add_mcmc_guidance_args(parser: argparse.ArgumentParser) -> None:
    """注册 MCMC 引导参数 (SE3 / 触觉 / 对齐)."""
    parser.add_argument(
        "--mcmc-se3",
        action="store_true",
        default=os.environ.get("HUGINN_MCMC_SE3", "0") == "1",
        help="SE(3) 群作用引导 MCMC proposal (需 cognitive_map, 无则退化到 fisher)",
    )
    parser.add_argument(
        "--mcmc-se3-angle-sigma",
        type=float,
        default=float(os.environ.get("HUGINN_MCMC_SE3_ANGLE_SIGMA", "30.0")),
        help="SE(3) proposal 旋转角度高斯标准差 (度, 默认 30)",
    )
    parser.add_argument(
        "--mcmc-haptic",
        action="store_true",
        default=os.environ.get("HUGINN_MCMC_HAPTIC", "0") == "1",
        help="触觉 (近场力学) 引导 MCMC proposal (需 haptic_layers.json, 无则退化到 fisher)",
    )
    parser.add_argument(
        "--mcmc-haptic-temperature",
        type=float,
        default=float(os.environ.get("HUGINN_MCMC_HAPTIC_TEMPERATURE", "1.0")),
        help="触觉引导 proposal softmax 温度 (tau, 默认 1.0, 大=趋随机, 小=趋贪心)",
    )
    parser.add_argument(
        "--mcmc-alignment",
        action="store_true",
        default=os.environ.get("HUGINN_MCMC_ALIGNMENT", "0") == "1",
        help="对齐引导 proposal: 用 GP 预测 structure→haptic 引导 MCMC (需 alignment_dataset.json, 无则退化)",
    )
    parser.add_argument(
        "--mcmc-alignment-temperature",
        type=float,
        default=float(os.environ.get("HUGINN_MCMC_ALIGNMENT_TEMPERATURE", "1.0")),
        help="对齐引导 proposal softmax 温度 (tau, 默认 1.0)",
    )


def _add_mcmc_args(parser: argparse.ArgumentParser) -> None:
    """注册 MCMC 多模式入口参数 (env var 回退)."""
    # Task 4.1: MCMC 多模式入口 — 不传 --mcmc-mode 走原 RCB 路径, 100% 不变.
    # env var 回退: CLI 未传时读 HUGINN_MCMC_*
    _add_mcmc_chain_args(parser)
    _add_mcmc_guidance_args(parser)


def _build_arg_parser() -> argparse.ArgumentParser:
    """构造 CLI 参数表 — RCB workspace + 极限模式 + MCMC 多模式入口."""
    parser = argparse.ArgumentParser(description="Huginn RCB runner")
    parser.add_argument("--workspace", required=True, help="RCB workspace path")
    parser.add_argument(
        "--extreme",
        action="store_true",
        help="v6 极限模式: thinking=high, max_tool_calls=1200, context_budget=model-window",
    )
    _add_mcmc_args(parser)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()

    # P1-A7: 启动前 30s 环境冒烟 — RDKit+sklearn GP + torch 是 RCB 最常见的
    # 工具链失败点. 失败即 fail-fast 打印修复清单, 不让 agent 跑半小时后才发现环境烂.
    if os.environ.get("HUGINN_SKIP_SMOKE", "0") != "1":
        _rcb_smoke_test()

    rc = asyncio.run(
        run(
            args.workspace,
            extreme=args.extreme,
            mcmc_mode=args.mcmc_mode,
            mcmc_steps=args.mcmc_steps,
            mcmc_chains=args.mcmc_chains,
            mcmc_checkpoint_interval=args.mcmc_checkpoint_interval,
            mcmc_se3=args.mcmc_se3,
            mcmc_se3_angle_sigma=args.mcmc_se3_angle_sigma,
            mcmc_haptic=args.mcmc_haptic,
            mcmc_haptic_temperature=args.mcmc_haptic_temperature,
            mcmc_alignment=args.mcmc_alignment,
            mcmc_alignment_temperature=args.mcmc_alignment_temperature,
        )
    )
    sys.exit(rc)


if __name__ == "__main__":
    if "--self-check-a3" in sys.argv:
        # A3: silent substitution 拦截 self-check.
        # 不依赖 RCB workspace, 纯函数验证机械比对逻辑.
        self_check_a3()
        sys.exit(0)
    if "--self-check-a2" in sys.argv:
        # A2: outputs/ 产物门控 self-check.
        # 不依赖 RCB workspace, 纯函数验证 metrics 文件判定.
        self_check_a2()
        sys.exit(0)
    if "--self-check-a4" in sys.argv:
        # A4: Step-3 retry 专用预算构造 self-check.
        # 不依赖 RCB workspace, 纯函数验证 BudgetSpec 构造.
        self_check_a4()
        sys.exit(0)
    if "--self-check-a7" in sys.argv:
        # A7: RCB 环境冒烟 + 7 个评测 setdefault 改强制赋值 self-check.
        # 验证 _rcb_smoke_test 函数定义 + 7 个关键 env 强制赋值生效 (防外部 env 覆盖).
        assert callable(_rcb_smoke_test), "_rcb_smoke_test not defined"
        _expected = {
            "HUGINN_RATE_LIMIT_ENABLED": "0",
            "HUGINN_ALLOW_LOCAL_BASH": "1",
            "HUGINN_CSM_SUBSET_MODE": "1",
            "HUGINN_USE_RUST_SANDBOX": "0",
            "HUGINN_COGNITIVE_LLM_DECIDER": "0",
            "HUGINN_HEALTH_MONITOR": "0",
            "HUGINN_FEATURE_LOOP_DETECTOR": "false",
        }
        for k, v in _expected.items():
            got = os.environ.get(k)
            assert got == v, f"env {k}={got!r}, expected {v!r} (强制赋值失效?)"
        print("[CHECK A7.1] _rcb_smoke_test defined OK")
        print("[CHECK A7.2] 7 env vars force-assigned OK")
        print("[CHECK A7] ALL ASSERTS PASSED")
        sys.exit(0)
    if "--self-check-b3" in sys.argv:
        # B3: critique 数值重算 self-check.
        # 不依赖 RCB workspace, 纯源码字符串验证 (step3_prompt 在函数内部, 拼接复杂).
        # ponytail: 直接 grep 自身源码验证注入. ceiling: 不验证 agent 是否真调 validate_tool.
        #   升级路径: 跑真实 RCB 任务, 抓 transcript 看 step3 是否调 validate_tool.
        _src = Path(__file__).read_text(encoding="utf-8")
        # 1. step3_prompt 的 A 段含 RECOMPUTE 硬规则 (替代文本对照)
        assert "RECOMPUTE each claim" in _src, (
            "missing RECOMPUTE rule in step3_prompt A"
        )
        assert "P1-B3 hard rule" in _src, "missing P1-B3 hard rule marker"
        # 2. 三个重算工具都列出 (validate_tool / symbolic_math_tool / code_tool)
        for _t in ("validate_tool", "symbolic_math_tool", "code_tool"):
            # step3_prompt A 段范围内必须出现
            _idx_a = _src.find("## A. Sanity Check")
            _idx_b = _src.find("## B. Substitution Audit")
            assert _idx_a > 0 and _idx_b > _idx_a, "A/B section markers not found"
            _a_section = _src[_idx_a:_idx_b]
            assert _t in _a_section, f"tool {_t} not mentioned in A section"
        # 3. 替代了原 "Compare each to the paper's baseline value" 文本对照
        # ponytail: 只在 A 段范围内检查, 避免本 self-check 注释里的字符串误判.
        _old_pattern = "Compare each to the paper's baseline value"
        assert _old_pattern not in _a_section, (
            "old text-comparison pattern still in A section, B3 replacement incomplete"
        )
        # 4. table 含 Claimed / Recomputed / Tool used / Match? 四列
        _idx_a = _src.find("## A. Sanity Check")
        _idx_b = _src.find("## B. Substitution Audit")
        _a_section = _src[_idx_a:_idx_b]
        for _col in ("Claimed", "Recomputed", "Tool used", "Match?"):
            assert _col in _a_section, f"missing table column: {_col}"
        print("[CHECK B3.1] RECOMPUTE rule injected in step3_prompt A")
        print("[CHECK B3.2] 3 recompute tools listed (validate/symbolic/code)")
        print("[CHECK B3.3] old text-comparison pattern removed")
        print("[CHECK B3.4] recompute table has 4 cols (Claimed/Recomputed/Tool/Match)")
        print("[CHECK B3] ALL ASSERTS PASSED")
        sys.exit(0)
    if "--self-check-b4" in sys.argv:
        # B4: report.md 数值标记 lint self-check.
        # 不依赖 RCB workspace, 用临时文件验证 _lint_report_markers 逻辑.
        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as f:
            f.write("# Report\n")
            f.write("R²=0.79 [EXECUTED] (from outputs/r2.json)\n")
            f.write("Training loss ≈ 0.3 [EXPECTED] (paper Table 2)\n")
            f.write("Accuracy: 0.85 (no marker)\n")
            f.write("In 2024 we ran 100 epochs\n")  # 年份应被排除
            f.write("F1=0.62 [NOT EXECUTED]\n")
            tmp_path = Path(f.name)
        try:
            r = _lint_report_markers(tmp_path)
            assert r["total_numbers"] >= 4, (
                f"应至少 4 个数值 (排除年份), got {r['total_numbers']}"
            )
            assert r["untagged"] >= 1, f"应至少 1 个未标记 (0.85), got {r['untagged']}"
            assert r["marker_counts"]["[EXECUTED]"] == 1, r["marker_counts"]
            assert r["marker_counts"]["[EXPECTED]"] == 1, r["marker_counts"]
            assert r["marker_counts"]["[NOT EXECUTED]"] == 1, r["marker_counts"]
            # 未标记样本应含 0.85
            assert any("0.85" in s for s in r["untagged_samples"]), r[
                "untagged_samples"
            ]
            print(
                f"[CHECK B4.1] total={r['total_numbers']} tagged={r['tagged']} untagged={r['untagged']}"
            )
            print(f"[CHECK B4.2] marker_counts={r['marker_counts']}")
            print(f"[CHECK B4.3] untagged_samples={r['untagged_samples']}")
            print("[CHECK B4] ALL ASSERTS PASSED")
        finally:
            tmp_path.unlink(missing_ok=True)
        sys.exit(0)
    if "--self-check-v14-all" in sys.argv:
        # v14 Phase 1-4 全综合验收: Phase 1 (Task 1-10) + Phase 2/3/4 (Task 11-19).
        # 不依赖 RCB workspace. RCBench 实测留给用户手动跑 (见各 phase NOTE).
        self_check_v14_all()
        sys.exit(0)
    if "--self-check-v14-p234" in sys.argv:
        # v14 Phase 2/3/4 综合验收: 顺序跑 Task 11-19 所有 self-check.
        # 不依赖 RCB workspace. 调各模块 __main__ self-check.
        self_check_v14_p234()
        sys.exit(0)
    if "--self-check-v14" in sys.argv:
        # v14 Phase 1 综合验收: 顺序跑 Task 1-10 所有 self-check.
        # 不依赖 RCB workspace. RCBench 实测留给用户手动跑 (见末尾 NOTE).
        self_check_v14_comprehensive()
        sys.exit(0)
    if "--self-check-v14-task4" in sys.argv:
        # v14 Task 4: Betti 数计算 (β_0 / β_1) self-check.
        # 不依赖 RCB workspace, 纯函数验证. 跑通后 sys.exit(0).
        self_check_v14_task4()
        sys.exit(0)
    if "--self-check-v14-task6" in sys.argv:
        # v14 Task 6: HintCoordinator 接入 rcb_runner 后产物 self-check.
        # 不依赖 RCB workspace, 纯函数验证 HintCoordinator.coordinate + legacy 路径.
        # HUGINN_HINT_COORDINATOR=0 也能跑 (legacy 路径不要求 hint ≤1500, 但函数本身仍验证).
        self_check_v14_task6()
        sys.exit(0)
    if "--self-check-v14-task1" in sys.argv:
        # v14 Task 1: Meta-Trace schema + 向后兼容 self-check.
        # 不依赖 RCB workspace, 纯函数验证. 跑通后 sys.exit(0).
        self_check_v14_task1()
        sys.exit(0)
    if "--self-check-v14-task2" in sys.argv:
        # v14 Task 2: darwin_score 真实计算 self-check.
        # 不依赖 RCB workspace, 纯函数验证. 跑通后 sys.exit(0).
        self_check_v14_task2()
        sys.exit(0)
    if "--self-check-v14-task3" in sys.argv:
        # v14 Task 3: supported_ratio 跨轮语义重叠 self-check.
        # 不依赖 RCB workspace, 纯函数验证 TF-IDF cosine + supported_ratio 计算.
        self_check_v14_task3()
        sys.exit(0)
    if "--self-check-v14-task8" in sys.argv:
        # v14 Task 8: Step3→Step2 回退执行 self-check.
        # 转调 self_check_v14_task8() 函数, 跑通后 sys.exit(0).
        self_check_v14_task8()
        sys.exit(0)
    if "--self-check-v15-task3" in sys.argv:
        # v15 Phase 2 Task 3: HypothesisManifold 接入 rcb_runner self-check.
        # 不依赖 RCB workspace, 纯函数验证 + tempdir. 跑通后 sys.exit(0).
        self_check_v15_task3()
        sys.exit(0)
    if "--self-check-v15-task4" in sys.argv:
        # v15 Phase 2 Task 4: HintCoordinator posterior-guided hint self-check.
        # 不依赖 RCB workspace, mock manifold + obs 验证 hint 构造/注入/boost/降级.
        self_check_v15_task4()
        sys.exit(0)
    if "--self-check" in sys.argv:
        # Task 3 self-check: meta mode 早期拒绝 (不调 LLM)
        # ponytail: 命中查重直接返回, llm_client=None 也能跑, 验证 ponytail 优化没退化.
        # 用 asyncio.run 包裹因 adversarial_critique 是 async (object mode 调用点 L434 依赖)
        rejections = [
            "always use Tanimoto kernel for GP",
            "add CRITICAL: never use RBF",
        ]
        proposal = "always use Tanimoto kernel for GP regression"
        result = asyncio.run(
            adversarial_critique(
                mode="meta",
                proposal=proposal,
                recent_rejections=rejections,
                system_prompt_summary="",
                llm_client=None,
            )
        )
        assert result["verdict"] == "reject", f"expected reject, got {result}"
        assert result.get("early_reject") is True, "should be early_reject"
        print("Task 3 self-check PASS")

        # FCM self-check: fake model 验证 winner 选择 / 越界保护 / 空候选退化
        class _FakeResp:
            def __init__(self, text):
                self.content = text

        class _FakeModel:
            """前 3 次 (fork) 返回方案, 第 4 次 (critique) 返回评审 JSON."""

            def __init__(self, critique_json):
                self.calls = 0
                self._cj = critique_json

            def invoke(self, msgs):
                self.calls += 1
                if self.calls <= 3:
                    return _FakeResp(f"plan from call {self.calls}")
                return _FakeResp(self._cj)

        # 1. winner=2 → 选第二个候选, merge_insights 透传
        m = _FakeModel(
            '{"scores": [3, 9, 5], "winner": 2, '
            '"merge_insights": ["use robust split"], "fatal_flaws": {}}'
        )
        r = asyncio.run(fork_critique_merge("checklist text", "scan text", m, k=3))
        assert r["winner_plan"] == "plan from call 2", r
        assert r["merge_insights"] == ["use robust split"], r
        assert r["scores"] == [3.0, 9.0, 5.0], r

        # 2. winner 越界 → 保护性回落第一个候选
        m = _FakeModel('{"scores": [1, 2, 3], "winner": 99}')
        r = asyncio.run(fork_critique_merge("c", "s", m, k=3))
        assert r["winner_plan"] == "plan from call 1", r

        # 3. critique JSON 坏掉 → fallback 第一个候选, 带 error 字段
        m = _FakeModel("not json at all")
        r = asyncio.run(fork_critique_merge("c", "s", m, k=3))
        assert r["winner_plan"] == "plan from call 1" and "error" in r, r

        # 4. k=1 → 单候选直接过, 不调 critique (calls==1)
        m = _FakeModel('{"winner": 1}')
        r = asyncio.run(fork_critique_merge("c", "s", m, k=1))
        assert r["winner_plan"] == "plan from call 1" and m.calls == 1, r
        print("FCM self-check PASS (4 cases)")

        # TFM self-check: 退火边界 + judge 评审
        assert anneal_fork_count(1.0, 3) == 3
        assert anneal_fork_count(0.7, 3) == 3
        assert anneal_fork_count(0.5, 3) == 2
        assert anneal_fork_count(0.4, 3) == 2
        assert anneal_fork_count(0.2, 3) == 1
        assert anneal_fork_count(0.9, 1) == 1

        class _FakeJudge:
            def __init__(self, text):
                self._t = text

            def invoke(self, msgs):
                return _FakeResp(self._t)

        # 1. 正常 winner 选择 + merge_notes 透传
        r = asyncio.run(
            judge_fork_reports(
                {"fast": "report A", "robust": "report B"},
                "checklist",
                _FakeJudge(
                    '{"scores": {"fast": 5, "robust": 8}, "winner": "robust", '
                    '"merge_notes": ["use CV split"]}'
                ),
            )
        )
        assert r["winner"] == "robust" and r["merge_notes"] == ["use CV split"], r
        # 2. LLM 编了不存在的 fork 名 → fallback 第一份非空
        r = asyncio.run(
            judge_fork_reports(
                {"fast": "A", "exact": "B"}, "c", _FakeJudge('{"winner": "nope"}')
            )
        )
        assert r["winner"] == "fast", r
        # 3. 空报告过滤 → 单候选直接过, 不调 LLM
        r = asyncio.run(
            judge_fork_reports({"fast": "", "robust": "B"}, "c", _FakeJudge("bad"))
        )
        assert r["winner"] == "robust", r
        # 4. 全空 → winner None
        r = asyncio.run(
            judge_fork_reports({"fast": " ", "robust": ""}, "c", _FakeJudge("bad"))
        )
        assert r["winner"] is None, r
        print("TFM self-check PASS (anneal 6 + judge 4)")

        # 复现门禁 self-check: 编数字的 fork 被门禁淘汰, 真实数字的免 LLM 直接胜
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            import numpy as np

            ad = Path(td)
            np.save(ad / "stats.npy", np.array([2.6e-20, 1.4e-13, 0.95]))
            # 三种 sci 写法都要抓到
            assert len(_extract_sci_numbers("μ < 2.6e-20 eV, 1.4×10^-13, 3×10⁻⁵")) == 3
            ok, note = _reproduction_gate("limits 2.6e-20 eV and 1.4e-13 eV", ad)
            assert ok and "2/2" in note, note
            ok, note = _reproduction_gate("we found 9.9e-55 and 8.8e-44", ad)
            assert not ok and "FAIL" in note, note
            # 无承重数字 → 不可验证, 放行不误杀
            ok, _ = _reproduction_gate("qualitative discussion only", ad)
            assert ok
            # judge 集成: fabricator 淘汰后单候选, 不调 LLM
            r = asyncio.run(
                judge_fork_reports(
                    {
                        "fast": "we found 9.9e-55 and 8.8e-44",
                        "robust": "limits 2.6e-20 eV and 1.4e-13 eV",
                    },
                    "c",
                    _FakeJudge("bad"),
                    artifact_dirs={"fast": ad, "robust": ad},
                )
            )
            assert r["winner"] == "robust" and r["gate"]["fast"].startswith("FAIL"), r
        print("reproduction gate self-check PASS (4 cases)")

        # v14 Task 7: Step3→Step2 回退触发条件
        # case 1: 触发回退 (fix_needed + β_1>0 + numeric/exact gap)
        assert _should_retry_execute(
            verdict="fix_needed", beta_1=1, gap_type="numeric_recompute"
        )
        assert _should_retry_execute(
            verdict="fix_needed", beta_1=2, gap_type="exact_component_missing"
        )
        # case 1b: verdict=fail 也触发回退 (fail + 具体 gap = 可修问题, 放弃=0分)
        assert _should_retry_execute(
            verdict="fail", beta_1=1, gap_type="numeric_recompute"
        )
        assert _should_retry_execute(
            verdict="fail", beta_1=1, gap_type="exact_component_missing"
        )
        # case 2: verdict=pass 不回退
        assert not _should_retry_execute(
            verdict="pass", beta_1=1, gap_type="numeric_recompute"
        )
        # case 3: β_1=0 不回退 (拓扑不许可, 无循环回退路径)
        assert not _should_retry_execute(
            verdict="fix_needed", beta_1=0, gap_type="numeric_recompute"
        )
        # case 4: text_description 不回退 (文字补完在 Step 3 内即可, 不必重跑 execute)
        assert not _should_retry_execute(
            verdict="fix_needed", beta_1=1, gap_type="text_description"
        )
        # case 5: gap_type=none 不回退
        assert not _should_retry_execute(
            verdict="fix_needed", beta_1=1, gap_type="none"
        )
        # case 6: verdict=reject 也不回退 (reject 走 finalize, 不走 retry)
        assert not _should_retry_execute(
            verdict="reject", beta_1=1, gap_type="numeric_recompute"
        )
        # case 7: fail + text_description 不回退 (文字问题不必重跑 execute)
        assert not _should_retry_execute(
            verdict="fail", beta_1=1, gap_type="text_description"
        )
        print("[CHECK v14 Task 7] Step3→Step2 retry trigger OK (8 cases)")

        # v14 Task 7 SubTask 7.1: CritiqueResult.gap_type 字段 + 默认值
        # 验证 dataclass 默认 gap_type="none", 模板路径不显式传 gap_type 时也是 none
        from huginn.cli.rcb_critique import CritiqueResult as _CR  # noqa: N814

        _cr_default = _CR(verdict="accept")
        assert _cr_default.gap_type == "none", (
            f"expected none, got {_cr_default.gap_type}"
        )
        # 显式构造每种 gap_type 都能正常存取
        for _gt in (
            "numeric_recompute",
            "exact_component_missing",
            "text_description",
            "none",
        ):
            _cr = _CR(verdict="fix_needed", gap_type=_gt)
            assert _cr.gap_type == _gt, f"expected {_gt}, got {_cr.gap_type}"
        print("[CHECK v14 Task 7] CritiqueResult.gap_type field OK")
        sys.exit(0)
    if "--self-check-c7" in sys.argv:
        # C7: 正向验证 — 升温替代了 break, advisory 替代了终止.
        _src = Path(__file__).read_text(encoding="utf-8")
        assert "iters, heating]" in _src, "C7 FAIL: stagnation 未改成 heating"
        assert "advisory, not stopping" in _src, (
            "C7 FAIL: score_history 未改成 advisory"
        )
        assert "_t_hot = min(1.0, _t_hot + 0.5)" in _src, "C7 FAIL: 升温逻辑丢失"
        print(
            "[C7] stagnation/score_history early-stop removed, heating retained — PASS"
        )
        sys.exit(0)
    main()
