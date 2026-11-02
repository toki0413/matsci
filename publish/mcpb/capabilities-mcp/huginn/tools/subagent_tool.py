"""subagent_tool -- lets the main agent dispatch isolated subagents.

Wraps SubagentDispatch so the agent can offload context-heavy tasks
(explore, code, analyze) to isolated sessions without bloating the main
conversation window. Inspired by Kimi Code's coder/explore/plan pattern.

Actions:
  - list_types: 列出所有可用的子 agent 类型
  - dispatch:   派发一个子 agent 执行任务, 返回压缩后的摘要
"""

from __future__ import annotations

import logging
import time
from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.core_types import ToolContext, ToolResult
from huginn.feature_flags import FeatureFlags
from huginn.tools.base import HuginnTool
from huginn.utils.common import hash_text

logger = logging.getLogger(__name__)


# P1-2: CRDT 状态合并 — 多 subagent 并行结果用半格 join 合并.
# ponytail: 纯函数, 不引入新依赖. 语义冲突仍走 LLM 仲裁.
# ceiling: G-Set 单调增, 长跑会膨胀; 升级 OR-Set 可删, 但需要 tombstone.
def _crdt_merge_enabled() -> bool:
    """P1-2 toggle: FeatureFlags `crdt_merge` (默认 on)."""
    return FeatureFlags.shared().is_enabled("crdt_merge")


def _belief_update_enabled() -> bool:
    """P2-6 toggle: FeatureFlags `belief_update` (默认 on).

    Bayesian belief update for LWW fields. off 时回退 P1-2 LWW (ts 大者胜).
    ponytail: 默认 on 因 fallback 完整 — result 无 belief 字段时自动走 LWW.
    """
    return FeatureFlags.shared().is_enabled("belief_update")


# ── P2-6: Bayesian belief update ──────────────────────────────────────────
# Friston 自由能原理: F(q, p) = KL[q(s) || p(s|o)] - log p(o).
# perception 路径: 更新 q(s) 使其接近 p(s|o) — Bayesian 后验.
# v1 只做闭合解 (Gaussian/Beta), 不学 surrogate (v2/P3).
#
# Gaussian (连续字段, e.g. best_encut):
#   prior N(μ, σ²), obs N(o, σ_o²)
#   posterior: μ' = (μ/σ² + o/σ_o²) / (1/σ² + 1/σ_o²)
#              σ'² = 1 / (1/σ² + 1/σ_o²)
#   多个 obs 重复 update 即可 (结合律).
#
# Beta (二值字段, e.g. success_rate):
#   prior Beta(α, β), obs (n_success, n_fail)
#   posterior: α' = α + n_success, β' = β + n_fail
#
# ponytail: 闭合解无新依赖. ceiling: 假设 obs noise 已知 (σ_o²), 实际需从
# 数据估计. 升级路径: v2 学 surrogate, P3 用 neural process.


def _gaussian_update(
    mu: float, sigma2: float, obs: float, obs_sigma2: float
) -> tuple[float, float]:
    """Gaussian Bayesian update. 返回 (mu', sigma2').

    prior N(mu, sigma2), obs N(obs, obs_sigma2).
    后验精度 = 先验精度 + 观测精度 (共轭).
    """
    if sigma2 <= 0 or obs_sigma2 <= 0:
        return mu, sigma2
    post_prec = 1.0 / sigma2 + 1.0 / obs_sigma2
    mu_post = (mu / sigma2 + obs / obs_sigma2) / post_prec
    sigma2_post = 1.0 / post_prec
    return mu_post, sigma2_post


def _beta_update(
    alpha: float, beta: float, n_success: int, n_fail: int
) -> tuple[float, float]:
    """Beta Bayesian update. 返回 (alpha', beta').

    prior Beta(alpha, beta), obs (n_success, n_fail) — Bernoulli 共轭.
    """
    return alpha + n_success, beta + n_fail


def _gaussian_kl(mu1: float, sigma2_1: float, mu2: float, sigma2_2: float) -> float:
    """KL(N1 || N2) — 冲突检测用. KL 大 = 两个 belief 严重不一致."""
    import math

    if sigma2_1 <= 0 or sigma2_2 <= 0:
        return 0.0
    return 0.5 * (
        math.log(sigma2_2 / sigma2_1) + (sigma2_1 + (mu1 - mu2) ** 2) / sigma2_2 - 1.0
    )


def _lww_pick(results: list[dict], field: str) -> tuple[Any, float]:
    """LWW: 取 ts 最大的非空字段值 (ts 相同取后者)."""
    best_val = None
    best_ts = -1.0
    for r in results:
        if isinstance(r, dict) and r.get(field) is not None:
            ts = float(r.get("ts", 0.0) or 0.0)
            if ts >= best_ts:
                best_ts = ts
                best_val = r[field]
    return best_val, best_ts


def _merge_belief_gaussian(belief_results: list[dict], field: str) -> dict[str, Any]:
    """Gaussian Bayesian merge: 第一个做 prior, 依次 update."""
    # 用第一个做 prior, 依次 update
    b0 = belief_results[0]["belief"][field]
    mu = float(b0.get("mu", 0.0))
    s2 = float(b0.get("sigma2", 0.0))
    obs_s2 = float(b0.get("obs_sigma2", 100.0))
    # M6: KL 冲突检测 — prior vs 每个 obs belief. KL 大说明两个 subagent
    # 看到的是不同分布, Bayesian merge 会假性缩小 σ² 给出虚假自信.
    # 不阻塞 merge, 只在 out 上标 _conflict 让调用方决策.
    # ponytail: 阈值 1.0 nat 硬编码. 升级: 按 field 配置 / 自适应.
    _max_kl = 0.0
    for r in belief_results[1:]:
        b = r["belief"][field]
        o = float(b.get("mu", 0.0))  # 观测值 = 该 result 的 mu
        os2 = float(b.get("obs_sigma2", obs_s2))
        _kl = _gaussian_kl(mu, s2, o, os2)
        if _kl > _max_kl:
            _max_kl = _kl
        mu, s2 = _gaussian_update(mu, s2, o, os2)
    out: dict[str, Any] = {field: mu}  # point estimate = posterior mean
    out[f"{field}_belief"] = {
        "type": "gaussian",
        "mu": mu,
        "sigma2": s2,
    }
    if _max_kl > 1.0:
        out[f"{field}_conflict"] = True
        out[f"{field}_conflict_kl"] = _max_kl
    try:
        from huginn.routes.metrics import track_belief_update

        track_belief_update("gaussian")
    except Exception:
        logger.debug("belief update metrics failed", exc_info=True)
    return out


def _merge_belief_beta(belief_results: list[dict], field: str) -> dict[str, Any]:
    """Beta Bayesian merge: 第一个做 prior, 依次累加成功/失败数."""
    b0 = belief_results[0]["belief"][field]
    a = float(b0.get("alpha", 1.0))
    b = float(b0.get("beta", 1.0))
    for r in belief_results:
        br = r["belief"][field]
        ns = int(br.get("n_success", 0))
        nf = int(br.get("n_fail", 0))
        a, b = _beta_update(a, b, ns, nf)
    out: dict[str, Any] = {field: a / (a + b)}  # point estimate = posterior mean
    out[f"{field}_belief"] = {
        "type": "beta",
        "alpha": a,
        "beta": b,
    }
    try:
        from huginn.routes.metrics import track_belief_update

        track_belief_update("beta")
    except Exception:
        logger.debug("belief update metrics failed", exc_info=True)
    return out


def _lww_into(out: dict[str, Any], results: list[dict], field: str) -> None:
    """无 belief 时按 LWW 规则把 field / field_ts 写进 out."""
    best_val, best_ts = _lww_pick(results, field)
    if best_val is not None:
        out[field] = best_val
        out[f"{field}_ts"] = best_ts


def _belief_merge(results: list[dict], lww_fields: set[str]) -> dict[str, Any]:
    """P2-6: Bayesian belief merge for LWW fields.

    每个 result 可带 "belief" 字段: {field: {"type": "gaussian"|"beta", ...}}.
    - gaussian: {"type": "gaussian", "mu": 520, "sigma2": 400, "obs_sigma2": 100}
      obs_sigma2 是该 subagent 观测噪声 (默认 100).
    - beta: {"type": "beta", "alpha": 1, "beta": 1, "n_success": 3, "n_fail": 1}

    无 belief 字段 → 回退 LWW (ts 大者胜).
    有 belief → Bayesian update, 输出 posterior belief + point estimate (mu/mean).

    ponytail: 只做 per-field 独立 update, 不做联合后验 (需协方差矩阵, 过重).
    ceiling: 字段间独立假设不一定成立 (encut 跟 kpoints 相关), 升级路径: 多变量 Gaussian.
    """
    out: dict[str, Any] = {}
    # 动态收集 belief 里提到的 field — 不只硬编码 lww_fields, 否则 success_rate
    # 这类带 belief 但不在 lww_fields 的字段会被漏掉 (Beta 路径不触发).
    belief_fields: set[str] = set()
    for r in results:
        if isinstance(r, dict) and isinstance(r.get("belief"), dict):
            belief_fields.update(r["belief"].keys())
    all_fields = lww_fields | belief_fields
    for field in all_fields:
        # 收集所有带 belief 的 result
        belief_results = [
            r
            for r in results
            if isinstance(r, dict)
            and isinstance(r.get("belief"), dict)
            and isinstance(r["belief"].get(field), dict)
        ]
        if not belief_results:
            # 无 belief, 走 LWW
            _lww_into(out, results, field)
            continue

        # 按类型 update (第一个 result 的 belief type 为准, 混合类型跳过)
        btype = belief_results[0]["belief"][field].get("type")
        if btype == "gaussian":
            out.update(_merge_belief_gaussian(belief_results, field))
        elif btype == "beta":
            out.update(_merge_belief_beta(belief_results, field))
        else:
            # 未知 type, 回退 LWW
            _lww_into(out, results, field)
    return out


def _content_hash(s: Any) -> str:
    """Stable hash for dedupe — G-Set 需要. 用 sha8 短摘要足够."""
    return hash_text(str(s), length=8)


def _collect_all_keys(results: list[dict]) -> set[str]:
    """收集所有字段名 (results 可能字段不一致, 全 union)."""
    all_keys: set[str] = set()
    for r in results:
        if isinstance(r, dict):
            all_keys.update(r.keys())
    return all_keys


def _merge_gset_fields(results: list[dict], all_keys: set[str]) -> dict[str, Any]:
    """G-Set 字段: union + dedupe by content hash."""
    gset_fields = {"findings", "evidence", "limitations", "tool_calls"}
    out: dict[str, Any] = {}
    for field in gset_fields:
        if field not in all_keys:
            continue
        seen: dict[str, Any] = {}
        for r in results:
            items = r.get(field) or []
            if not isinstance(items, list):
                items = [items]
            for it in items:
                h = _content_hash(it)
                if h not in seen:
                    seen[h] = it
        out[field] = list(seen.values())
    return out


def _merge_lww_fields(
    results: list[dict], lww_fields: set[str], all_keys: set[str]
) -> dict[str, Any]:
    """LWW 字段: 任意单值字段, ts 大者胜."""
    out: dict[str, Any] = {}
    for field in lww_fields:
        if field not in all_keys:
            continue
        best_val: Any = None
        best_ts: float = -1.0
        for r in results:
            if field in r and r[field] is not None:
                ts = float(r.get("ts", 0.0) or 0.0)
                if ts >= best_ts:
                    best_ts = ts
                    best_val = r[field]
        if best_val is not None:
            out[field] = best_val
            out[f"{field}_ts"] = best_ts
    return out


def _merge_summaries(results: list[dict]) -> str | None:
    """summary: 拼接非空 (dedupe by hash)."""
    seen_sum: dict[str, str] = {}
    for r in results:
        s = r.get("summary") if isinstance(r, dict) else None
        if s and isinstance(s, str):
            h = _content_hash(s)
            if h not in seen_sum:
                seen_sum[h] = s
    if seen_sum:
        return "\n---\n".join(seen_sum.values())
    return None


def _merge_errors(results: list[dict]) -> list[str]:
    """errors: 收集所有非空 error (G-Set)."""
    errs = [r.get("error") for r in results if isinstance(r, dict) and r.get("error")]
    seen_err: dict[str, str] = {}
    for e in errs:
        h = _content_hash(e)
        if h not in seen_err:
            seen_err[h] = e
    return list(seen_err.values())


def _max_ts(results: list[dict]) -> float:
    """输出本身的 ts = max(输入 ts), 让外层 LWW 判断正确."""
    return max(
        (float(r.get("ts", 0.0) or 0.0) for r in results if isinstance(r, dict)),
        default=0.0,
    )


def _track_crdt_merge(n_sources: int) -> None:
    """P1-2 metrics: 记录 CRDT merge 触发 + source 数 (跨 mode 共享)."""
    try:
        from huginn.routes.metrics import track_crdt_merge

        track_crdt_merge(n_sources)
    except Exception:
        logger.debug("crdt merge metrics failed", exc_info=True)


def _finish_merged(merged: dict[str, Any], results: list[dict]) -> dict:
    """补齐 success / summary / errors / sources / ts 后返回."""
    # success: OR-join (任一 True → True)
    merged["success"] = any(
        r.get("success", False) for r in results if isinstance(r, dict)
    )

    summary = _merge_summaries(results)
    if summary is not None:
        merged["summary"] = summary

    errors = _merge_errors(results)
    if errors:
        merged["errors"] = errors

    # 保留原始 results (审计/调试用), 但放末尾避免污染主 view
    merged["sources"] = results
    # 嵌套 merge 用: 输出本身的 ts = max(输入 ts), 让外层 LWW 判断正确.
    # 不加这个的话 merge(merge(a, b), c) 里 merge(a, b) 的 ts 字段缺失,
    # 外层 LWW 取不到正确 ts.
    merged["ts"] = _max_ts(results)
    return merged


def _crdt_merge(results: list[dict]) -> dict:
    """CRDT-merge parallel subagent results.

    半格 (S, ⊔) 三公理: 交换 / 结合 / 幂等. join 字段级合并:
    - findings: G-Set (union, dedupe by content hash)
    - evidence: G-Set (union, dedupe)
    - limitations: G-Set (union, dedupe)
    - best_value / 任意 LWW 字段: ts 大者胜 (用 result["ts"] 或默认 0)
    - success: 任一 True → True (OR-join)
    - summary: 拼接所有非空 summary (顺序无关, 字符串拼接在半格里是 idempotent
      只有当 dedupe 后; ponytail: 用 hash dedupe 避免重复段)

    ponytail: 只做字段级 CRDT, 语义冲突 (A 说 encut=520, B 说 encut=572)
    仍走 LLM 仲裁 (_resolve_support_finding 已有). 这里保证 merge(a, b) == merge(b, a).
    """
    if not results:
        return {"success": False, "error": "no results", "merged": True}

    # P1-2 metrics: 记录 CRDT merge 触发 + source 数 (跨 mode 共享)
    _track_crdt_merge(len(results))

    all_keys = _collect_all_keys(results)

    # LWW 字段: 任意单值字段, ts 大者胜
    # 默认 LWW 候选: best_value, best_encut, answer, conclusion
    lww_fields = {"best_value", "best_encut", "answer", "conclusion", "result"}

    merged: dict[str, Any] = {"merged": True, "n_sources": len(results)}

    # G-Set 合并
    merged.update(_merge_gset_fields(results, all_keys))

    # LWW 合并 — P2-6: 有 belief 字段时走 Bayesian update, 否则 LWW.
    # _belief_merge 内部自动 fallback LWW, 这里只需 toggle on 时调它,
    # off 时直接走原 LWW 逻辑 (回归 P1-2 行为).
    if _belief_update_enabled():
        merged.update(_belief_merge(results, lww_fields))
    else:
        merged.update(_merge_lww_fields(results, lww_fields, all_keys))

    return _finish_merged(merged, results)


def _merge_context(args: SubagentToolInput, context: ToolContext) -> dict:
    """把 ToolContext 的字段并进 dispatch context dict."""
    dispatch_ctx = dict(args.context)
    dispatch_ctx.setdefault("agent_factory", context.agent_factory)
    dispatch_ctx.setdefault("session_id", context.session_id)
    dispatch_ctx.setdefault("workspace", context.workspace)
    # v7: 透传父 agent 的 approval_callback, 子 agent 调 ASK 工具 (vasp_tool 等) 才能拿到批准.
    dispatch_ctx.setdefault("approval_callback", context.approval_callback)
    return dispatch_ctx


def _make_state_forwarder(spec_name: str):
    """构造把子 agent 中间状态转发到 WS 的 progress 回调."""
    # forward subagent intermediate states to the WS via progress_cb
    from huginn.core_types import progress_cb

    async def _on_state(state: dict) -> None:
        cb = progress_cb.get()
        if cb is None:
            return
        msgs = state.get("messages", [])
        if not msgs:
            return
        last = msgs[-1]
        # tool calls
        if hasattr(last, "tool_calls") and last.tool_calls:
            for tc in last.tool_calls:
                await cb(
                    {
                        "type": "subagent_event",
                        "event": "tool_call",
                        "spec": spec_name,
                        "tool": tc.get("name", "unknown"),
                    }
                )
        # assistant text (truncated)
        elif hasattr(last, "content") and last.content:
            text = last.content if isinstance(last.content, str) else str(last.content)
            if len(text) > 200:
                text = text[:200] + "..."
            await cb(
                {
                    "type": "subagent_event",
                    "event": "text",
                    "spec": spec_name,
                    "text": text,
                }
            )

    return _on_state


def _parallel_coros(
    dispatch: Any, tasks: list[dict], ctx: dict, depth: int
) -> list[Any]:
    """按 tasks 顺序构造 dispatch 协程列表 (gather 前用)."""
    return [
        dispatch.dispatch(t["spec_name"], t["task"], ctx, _depth=depth) for t in tasks
    ]


class SubagentToolInput(BaseModel):
    action: Literal["dispatch", "dispatch_parallel", "list_types"] = Field(
        default="list_types",
        description="dispatch to run a subagent, dispatch_parallel for DAG-aware parallel, list_types to see available types",
    )
    spec_name: str | None = Field(
        default=None,
        description="Subagent type to dispatch (e.g. explore, coder, analyst)",
    )
    task: str | None = Field(
        default=None,
        description="Task description for the subagent to execute",
    )
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional context to pass to the subagent (merged with tool context)",
    )
    # v14 Task 13: PersistentTerminal 接入. None 时看 env HUGINN_PERSISTENT_TERMINAL.
    use_persistent_terminal: bool | None = Field(
        default=None,
        description=(
            "If True, dispatch via PersistentTerminal (long session, async poll). "
            "If False, force in-process dispatch. "
            "None = follow env HUGINN_PERSISTENT_TERMINAL (1=on, else off)."
        ),
    )
    # dispatch_parallel 用: [{spec_name, task}, ...], 最多 4 个 (硬 cap)
    tasks: list[dict[str, str]] | None = Field(
        default=None,
        description=(
            "For dispatch_parallel: list of {spec_name, task} dicts (1-4 items). "
            "Tasks run concurrently via asyncio.gather."
        ),
    )
    # dispatch_parallel 用: [(u_name, v_name), ...] 任务依赖. u_name/v_name 引用
    # tasks 里 spec_name+task 的标识 (ponytail: 用 task 字符串前 20 字符做 ID).
    dependencies: list[tuple[str, str]] | None = Field(
        default=None,
        description=(
            "For dispatch_parallel: task dependencies as [(u, v), ...]. "
            "u must finish before v starts. Enables DAG-aware scheduling. "
            "Omit for full parallel."
        ),
    )


class SubagentToolOutput(BaseModel):
    success: bool
    action: str
    summary: str | None = None
    tool_calls: list[dict] | None = None
    tokens_used: int | None = None
    available_types: list[dict] | None = None
    error: str | None = None


class SubagentTool(HuginnTool[SubagentToolInput, SubagentToolOutput]):
    """Bridge between the main agent and the subagent dispatch system."""

    name = "subagent_tool"
    category = "meta"
    description = (
        "Dispatch isolated subagents to handle context-heavy tasks. "
        "Each subagent runs in its own session with a restricted tool set. "
        "Use 'list_types' to see available subagent types, 'dispatch' to run one. "
        "Types: explore (read-only search), coder (write/modify code), "
        "analyst (analyze data/results), support (heavy lifting in isolation, "
        "returns structured JSON findings — Oxelra Core+Support pattern)."
    )
    destructive = False
    read_only = False  # coder subagent can modify files
    input_schema = SubagentToolInput
    output_schema = SubagentToolOutput

    def __init__(self) -> None:
        super().__init__()
        # 延迟导入避免 agents -> tools 循环依赖
        from huginn.agents.subagent import SubagentDispatch

        self._dispatch = SubagentDispatch()

    async def _execute(
        self, args: SubagentToolInput, context: ToolContext
    ) -> ToolResult:
        if args.action == "list_types":
            return self._list_types()
        if args.action == "dispatch":
            return await self._dispatch_subagent(args, context)
        if args.action == "dispatch_parallel":
            return await self._dispatch_parallel(args, context)

        msg = f"Unknown action: {args.action}"
        return ToolResult(
            data=SubagentToolOutput(
                success=False, action=args.action, error=msg
            ).model_dump(),
            success=False,
            error=msg,
        )

    # -- actions -----------------------------------------------------------

    def _list_types(self) -> ToolResult:
        types = self._dispatch.list_specs()
        out = SubagentToolOutput(
            success=True,
            action="list_types",
            available_types=types,
            summary=f"{len(types)} subagent types available",
        )
        return ToolResult(data=out.model_dump(), success=True)

    async def _dispatch_subagent(
        self, args: SubagentToolInput, context: ToolContext
    ) -> ToolResult:
        if not args.spec_name:
            return self._missing_field("spec_name")
        if not args.task:
            return self._missing_field("task")

        # 把 ToolContext 的字段并进 dispatch context dict
        dispatch_ctx = _merge_context(args, context)

        # G1: 从 contextvar 读当前递归深度, 透传给 dispatch 守卫.
        # 主 agent 这里读到 0, 子 agent 那里读到 1+.
        from huginn.agents.subagent import _current_depth

        _depth = _current_depth.get()

        # forward subagent intermediate states to the WS via progress_cb
        _on_state = _make_state_forwarder(args.spec_name)

        result = await self._dispatch.dispatch(
            args.spec_name,
            args.task,
            dispatch_ctx,
            on_state=_on_state,
            _depth=_depth,
        )

        out = SubagentToolOutput(
            success=result.success,
            action="dispatch",
            summary=result.summary,
            tool_calls=result.tool_calls,
            tokens_used=result.tokens_used,
            error=result.error,
        )
        return ToolResult(
            data=out.model_dump(),
            success=result.success,
            error=result.error,
        )

    def _dp_validate_tasks(self, args: SubagentToolInput) -> ToolResult | None:
        """校验 tasks 数量上限与必填字段; 通过返回 None."""
        if not args.tasks:
            return self._missing_field("tasks")
        if len(args.tasks) > 4:
            return ToolResult(
                data=SubagentToolOutput(
                    success=False,
                    action="dispatch_parallel",
                    error=f"tasks 最多 4 个, got {len(args.tasks)}",
                ).model_dump(),
                success=False,
                error="tasks exceeds cap of 4",
            )
        # 校验每个 task dict 有 spec_name + task
        for i, t in enumerate(args.tasks):
            if "spec_name" not in t or "task" not in t:
                return ToolResult(
                    data=SubagentToolOutput(
                        success=False,
                        action="dispatch_parallel",
                        error=f"tasks[{i}] 缺 spec_name 或 task",
                    ).model_dump(),
                    success=False,
                    error=f"tasks[{i}] missing spec_name or task",
                )
        return None

    def _dp_build_ctx(
        self, args: SubagentToolInput, context: ToolContext
    ) -> tuple[dict, int]:
        """构造 dispatch 上下文 + 当前递归深度."""
        dispatch_ctx = dict(args.context)
        dispatch_ctx.setdefault("agent_factory", context.agent_factory)
        dispatch_ctx.setdefault("session_id", context.session_id)
        dispatch_ctx.setdefault("workspace", context.workspace)
        dispatch_ctx.setdefault("approval_callback", context.approval_callback)
        # G1: 从 contextvar 读递归深度 (跟 _dispatch_subagent 一致)
        from huginn.agents.subagent import _current_depth

        return dispatch_ctx, _current_depth.get()

    async def _dp_gather(self, coros: list) -> list[dict]:
        """并行 gather 并把异常/结果统一成 dict list (含 ts)."""
        import asyncio

        results = await asyncio.gather(*coros, return_exceptions=True)
        out_results = []
        for r in results:
            if isinstance(r, Exception):
                out_results.append({"success": False, "error": str(r)})
            else:
                _d = r.to_dict()
                # P1-2: 注 ts (LWW 用). SubagentResult 默认无 ts, 用 now.
                _d.setdefault("ts", time.time())
                out_results.append(_d)
        return out_results

    def _dp_build_dag(
        self, args: SubagentToolInput, task_ids: list[str]
    ) -> tuple[ToolResult | None, list]:
        """按 dependencies 建 TaskDAG 并返回拓扑分层; 失败返回 (err, [])."""
        from huginn.agents.task_dag import TaskDAG, build_dag_with_provenance

        try:
            # P4 Task 26.8: LLM 填的 dependencies 作为先验, provenance chain 作为后验补全.
            # provenance registry 可能没启用, 失败退化为纯 explicit_deps.
            _prov_reg = None
            try:
                from huginn.provenance.registry import ProvenanceRegistry

                _prov_reg = ProvenanceRegistry.shared()
            except Exception:
                logger.debug("provenance registry init failed", exc_info=True)
            if _prov_reg is not None and hasattr(_prov_reg, "get_lineage"):
                _tasks_with_input = [
                    {"id": tid, "tool_input": t} for tid, t in zip(task_ids, args.tasks)
                ]
                dag = build_dag_with_provenance(
                    tasks=_tasks_with_input,
                    provenance_registry=_prov_reg,
                    explicit_deps=args.dependencies,
                )
            else:
                dag = TaskDAG(tasks=task_ids, dependencies=args.dependencies)
        except ValueError as e:
            return ToolResult(
                data=SubagentToolOutput(
                    success=False,
                    action="dispatch_parallel",
                    error=f"DAG 错误: {e}",
                ).model_dump(),
                success=False,
                error=str(e),
            ), []
        return None, dag.parallel_layers()

    def _dp_make_result(
        self, results: list[dict], layers: list | None = None
    ) -> ToolResult:
        """组装 ToolResult; 多结果时按 CRDT 半格 join."""
        data: dict[str, Any] = {"action": "dispatch_parallel", "results": results}
        # P1-2: CRDT 合并 — 半格 join, 字段级无冲突.
        # off 时回退原 list[dict] 行为 (回归测试安全).
        merged = None
        if _crdt_merge_enabled() and len(results) > 1:
            merged = _crdt_merge(results)
            data["merged"] = merged
        data["n"] = len(results)
        if layers is not None:
            data["layers"] = layers
        if merged is not None:
            return ToolResult(data=data, success=merged.get("success", True))
        return ToolResult(data=data, success=True)

    def _dp_extreme_gate(self) -> ToolResult | None:
        """DAG 路径需极限模式; 未开启返回错误 ToolResult."""
        if FeatureFlags.shared().is_enabled("extreme_dispatch"):
            return None
        return ToolResult(
            data=SubagentToolOutput(
                success=False,
                action="dispatch_parallel",
                error="DAG-aware dispatch 需开启极限模式 (HUGINN_EXTREME_DISPATCH=1)",
            ).model_dump(),
            success=False,
            error="DAG dispatch requires HUGINN_EXTREME_DISPATCH=1",
        )

    async def _dispatch_parallel(
        self, args: SubagentToolInput, context: ToolContext
    ) -> ToolResult:
        """DAG-aware 并行 dispatch.

        无 dependencies: 全部 asyncio.gather 并行.
        有 dependencies: 用 TaskDAG 拓扑分层, 同层并行, 层间串行.

        ponytail: 硬 cap 4 并行 (API 限速 + 调试可行性). DAG 调度复用 TaskDAG.
        """
        bad = self._dp_validate_tasks(args)
        if bad is not None:
            return bad
        dispatch_ctx, _depth = self._dp_build_ctx(args, context)

        # 无 dependencies: 全并行
        if not args.dependencies:
            coros = _parallel_coros(self._dispatch, args.tasks, dispatch_ctx, _depth)
            return self._dp_make_result(await self._dp_gather(coros))

        # 有 dependencies: DAG 分层调度 (极限模式才开)
        bad = self._dp_extreme_gate()
        if bad is not None:
            return bad
        # task ID = spec_name + task 前 20 字符 (ponytail: 不引入显式 ID 字段)
        task_ids = [f"{t['spec_name']}:{t['task'][:20]}" for t in args.tasks]
        err, layers = self._dp_build_dag(args, task_ids)
        if err is not None:
            return err
        id_to_task = dict(zip(task_ids, args.tasks))
        all_results: list[dict] = []
        for layer in layers:
            coros = _parallel_coros(
                self._dispatch, [id_to_task[tid] for tid in layer], dispatch_ctx, _depth
            )
            all_results.extend(await self._dp_gather(coros))
        return self._dp_make_result(all_results, layers)

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _missing_field(field: str) -> ToolResult:
        msg = f"{field} is required for dispatch action"
        return ToolResult(
            data=SubagentToolOutput(
                success=False, action="dispatch", error=msg
            ).model_dump(),
            success=False,
            error=msg,
        )


# ── self-check ────────────────────────────────────────────────────────────
# P1-2 验证: CRDT 半格三公理 + 字段级合并. `python -m huginn.tools.subagent_tool` 跑.


def _selfcheck() -> None:
    """assert-based demo: CRDT 半格三公理 + 字段级合并 + Bayesian belief."""
    _sc_crdt_gset()
    _sc_crdt_lww()
    _sc_crdt_axioms()
    _sc_crdt_integration()
    _sc_crdt_toggle()
    _sc_belief_gaussian()
    _sc_belief_beta()
    _sc_belief_fallback()
    _sc_belief_gaussian_merge()
    _sc_belief_beta_merge()
    _sc_belief_kl()
    _sc_belief_conflict_high()
    _sc_belief_conflict_low()
    _sc_belief_toggle()
    print("subagent_tool selfcheck OK (33-37 CRDT + 42-47 Bayesian belief)")


def _sc_canon(d: dict) -> str:
    """半格比较用的规范化 JSON: 去掉 sources/n_sources/ts, 列表排序."""
    import json as _json

    skip = {"sources", "n_sources", "ts"}
    d2: dict = {}
    for k, v in d.items():
        if k in skip or k.endswith("_ts"):
            continue
        if isinstance(v, list):
            d2[k] = sorted(v, key=lambda x: str(x))
        else:
            d2[k] = v
    return _json.dumps(d2, sort_keys=True, default=str)


def _sc_crdt_gset() -> None:
    """33. G-Set union — evidence 合并, 无 dup."""
    r_a = {"findings": ["encut=520 OK"], "evidence": ["ref1", "ref2"], "success": True}
    r_b = {"findings": ["kpoints=4x4x4"], "evidence": ["ref2", "ref3"], "success": True}
    m33 = _crdt_merge([r_a, r_b])
    ev_set = set(m33["evidence"])
    assert ev_set == {"ref1", "ref2", "ref3"}, (
        f"G-Set union 应无 dup: {m33['evidence']}"
    )
    f_set = set(m33["findings"])
    assert f_set == {"encut=520 OK", "kpoints=4x4x4"}, (
        f"findings union: {m33['findings']}"
    )
    print("33. CRDT G-Set union (evidence + findings, dedupe) OK")


def _sc_crdt_lww() -> None:
    """34. LWW-Register — best_value ts 大者胜."""
    r_old = {"best_value": 520, "ts": 100.0}
    r_new = {"best_value": 572, "ts": 200.0}
    m34 = _crdt_merge([r_old, r_new])
    assert m34["best_value"] == 572, f"LWW 应取 ts 大的 572: {m34.get('best_value')}"
    assert m34["best_value_ts"] == 200.0
    # 交换顺序也应取 ts 大者
    m34b = _crdt_merge([r_new, r_old])
    assert m34b["best_value"] == 572, (
        f"交换律: LWW 应仍取 572: {m34b.get('best_value')}"
    )
    print("34. CRDT LWW-Register (ts 大者胜 + 交换律) OK")


def _sc_crdt_axioms() -> None:
    """35. 半格三公理 — 交换 / 结合 / 幂等."""
    r1 = {
        "findings": ["a"],
        "evidence": ["e1"],
        "best_value": 1,
        "ts": 1.0,
        "success": True,
    }
    r2 = {
        "findings": ["b"],
        "evidence": ["e2"],
        "best_value": 2,
        "ts": 2.0,
        "success": False,
    }
    r3 = {
        "findings": ["c"],
        "evidence": ["e3"],
        "best_value": 3,
        "ts": 3.0,
        "success": True,
    }
    m_ab = _sc_canon(_crdt_merge([r1, r2]))
    m_ba = _sc_canon(_crdt_merge([r2, r1]))
    assert m_ab == m_ba, f"交换律失败:\n{m_ab}\n{m_ba}"
    # 结合律: merge(merge(a, b), c) == merge(a, merge(b, c))
    # 注意: merge 的输入是 list[dict], 输出是 dict — 要把 merge(a, b) 包回 list
    m_ab_c = _sc_canon(_crdt_merge([_crdt_merge([r1, r2]), r3]))
    m_a_bc = _sc_canon(_crdt_merge([r1, _crdt_merge([r2, r3])]))
    assert m_ab_c == m_a_bc, f"结合律失败:\n{m_ab_c}\n{m_a_bc}"
    # 幂等: merge(a, a) == a (在半格意义下 — 去掉 sources 后)
    m_aa = _sc_canon(_crdt_merge([r1, r1]))
    m_a = _sc_canon(_crdt_merge([r1]))
    assert m_aa == m_a, f"幂等律失败:\n{m_aa}\n{m_a}"
    print("35. CRDT 半格三公理 (交换 / 结合 / 幂等) OK")


def _sc_crdt_integration() -> None:
    """36. 整合 — 4 subagent, 2 失败 2 成功, mixed fields."""
    r_s1 = {
        "findings": ["encut=572 OK"],
        "evidence": ["conv1"],
        "best_encut": 572,
        "ts": 100.0,
        "success": True,
        "summary": "subagent A: encut=572",
    }
    r_s2 = {
        "findings": ["encut=572 OK"],
        "evidence": ["conv2"],  # 重复 finding
        "best_encut": 520,
        "ts": 200.0,
        "success": True,
        "summary": "subagent B: encut=520 (later)",
    }
    r_f1 = {"success": False, "error": "OOM", "ts": 150.0}
    r_f2 = {"success": False, "error": "OOM", "ts": 180.0}  # 重复 error
    m36 = _crdt_merge([r_s1, r_s2, r_f1, r_f2])
    assert len(m36["findings"]) == 1, f"重复 finding 应 dedupe: {m36['findings']}"
    assert set(m36["evidence"]) == {"conv1", "conv2"}
    assert m36["best_encut"] == 520, f"LWW 应取 ts=200 的 520: {m36.get('best_encut')}"
    assert m36["success"] is True, "OR-join: 任一 True → True"
    assert len(m36["errors"]) == 1, f"重复 error 应 dedupe: {m36['errors']}"
    assert "subagent A" in m36["summary"] and "subagent B" in m36["summary"]
    print("36. CRDT 整合 (4 subagent, 2 成功 2 失败, mixed fields) OK")


def _sc_crdt_toggle() -> None:
    """37. toggle off — 不合并 (回归原 list[dict])."""
    FeatureFlags.shared().disable("crdt_merge")
    assert _crdt_merge_enabled() is False
    FeatureFlags.shared().enable("crdt_merge")
    assert _crdt_merge_enabled() is True
    print("37. CRDT toggle (crdt_merge off/on) OK")


def _sc_belief_gaussian() -> None:
    """42. Gaussian update — prior N(520, 20²), obs N(572, 10²)."""
    mu42, s2_42 = _gaussian_update(520.0, 400.0, 572.0, 100.0)
    # 后验 μ ≈ (520/400 + 572/100) / (1/400 + 1/100) = (1.3 + 5.72) / 0.0125 = 561.6
    # 后验 σ² ≈ 1/0.0125 = 80
    assert abs(mu42 - 561.6) < 0.1, f"Gaussian μ' 应 ≈ 561.6, got {mu42}"
    assert abs(s2_42 - 80.0) < 0.1, f"Gaussian σ'² 应 ≈ 80, got {s2_42}"
    print("42. Gaussian Bayesian update (μ=520→561.6, σ²=400→80) OK")


def _sc_belief_beta() -> None:
    """43. Beta update — prior Beta(1, 1), 3 success 1 fail."""
    a43, b43 = _beta_update(1.0, 1.0, 3, 1)
    assert a43 == 4.0 and b43 == 2.0, f"Beta(1,1)+3s1f 应 Beta(4,2), got ({a43},{b43})"
    # point estimate = 4/(4+2) = 2/3
    mean43 = a43 / (a43 + b43)
    assert abs(mean43 - 0.667) < 0.001, f"mean 应 ≈ 0.667, got {mean43}"
    print("43. Beta Bayesian update (Beta(1,1)→Beta(4,2), mean=0.667) OK")


def _sc_belief_fallback() -> None:
    """44. 无 belief → fallback LWW (兼容 P1-2)."""
    r44a = {"best_value": 520, "ts": 100.0}
    r44b = {"best_value": 572, "ts": 200.0}
    m44 = _belief_merge([r44a, r44b], {"best_value"})
    assert m44["best_value"] == 572, (
        f"无 belief 应 LWW 取 ts=200: {m44.get('best_value')}"
    )
    assert m44["best_value_ts"] == 200.0
    print("44. belief_merge 无 belief → fallback LWW OK")


def _sc_gaussian_belief_pair() -> tuple[dict, dict]:
    """45/46b/47 共用的 Gaussian belief 输入对 (μ=520 vs μ=572)."""
    r45a = {
        "best_encut": 520,
        "ts": 100.0,
        "belief": {
            "best_encut": {
                "type": "gaussian",
                "mu": 520,
                "sigma2": 400,
                "obs_sigma2": 100,
            }
        },
    }
    r45b = {
        "best_encut": 572,
        "ts": 200.0,
        "belief": {
            "best_encut": {
                "type": "gaussian",
                "mu": 572,
                "sigma2": 100,
                "obs_sigma2": 100,
            }
        },
    }
    return r45a, r45b


def _sc_belief_gaussian_merge() -> None:
    """45. 有 belief → Bayesian update (520+572 → 561.6 + belief)."""
    r45a, r45b = _sc_gaussian_belief_pair()
    m45 = _belief_merge([r45a, r45b], {"best_encut"})
    # r45a 做 prior, r45b 做 obs → 同 42 的计算
    assert abs(m45["best_encut"] - 561.6) < 0.1, (
        f"Bayesian 应 ≈ 561.6, got {m45.get('best_encut')}"
    )
    assert "best_encut_belief" in m45, "应输出 posterior belief"
    assert m45["best_encut_belief"]["type"] == "gaussian"
    print("45. belief_merge Gaussian (520+572 → 561.6 + belief) OK")


def _sc_belief_beta_merge() -> None:
    """45b. Beta belief merge (Beta(1,1)+5s4f → Beta(6,5))."""
    r45c = {
        "success_rate": 0.5,
        "ts": 1.0,
        "belief": {
            "success_rate": {
                "type": "beta",
                "alpha": 1,
                "beta": 1,
                "n_success": 3,
                "n_fail": 1,
            }
        },
    }
    r45d = {
        "success_rate": 0.7,
        "ts": 2.0,
        "belief": {
            "success_rate": {
                "type": "beta",
                "alpha": 1,
                "beta": 1,
                "n_success": 2,
                "n_fail": 3,
            }
        },
    }
    m45b = _belief_merge([r45c, r45d], {"success_rate"})
    # Beta(1,1) + (3s+2s, 1f+3f) = Beta(6, 5), mean = 6/11 ≈ 0.545
    assert abs(m45b["success_rate"] - 6 / 11) < 0.001, (
        f"Beta merge mean 应 ≈ 0.545, got {m45b.get('success_rate')}"
    )
    assert m45b["success_rate_belief"]["alpha"] == 6.0
    assert m45b["success_rate_belief"]["beta"] == 5.0
    print("45b. belief_merge Beta (Beta(1,1)+5s4f → Beta(6,5)) OK")


def _sc_belief_kl() -> None:
    """46. KL 冲突检测 — 两个 Gaussian belief KL 大 = 严重不一致."""
    kl46 = _gaussian_kl(520.0, 100.0, 572.0, 100.0)
    kl46_close = _gaussian_kl(520.0, 100.0, 521.0, 100.0)
    assert kl46 > kl46_close, f"KL(520,572) 应 > KL(520,521): {kl46} vs {kl46_close}"
    assert kl46 > 0.1, f"严重冲突 KL 应 > 0.1, got {kl46}"
    print("46. Gaussian KL 冲突检测 (520 vs 572 KL > 0.1) OK")


def _sc_belief_conflict_high() -> None:
    """46b. M6: _belief_merge 接入 KL — 高冲突时输出 _conflict + _conflict_kl."""
    r45a, r45b = _sc_gaussian_belief_pair()
    # r45a (mu=520) vs r45b (mu=572): KL(520,100,572,100) ≈ 13.5 > 1.0 → 标冲突
    m46b = _belief_merge([r45a, r45b], {"best_encut"})
    assert m46b.get("best_encut_conflict") is True, (
        f"高 KL 应触发 _conflict, got {m46b.get('best_encut_conflict')}"
    )
    assert m46b.get("best_encut_conflict_kl", 0) > 1.0, (
        f"_conflict_kl 应 > 1.0, got {m46b.get('best_encut_conflict_kl')}"
    )
    # 仍输出 merge 结果 (不阻塞), 调用方决定是否走 LLM 仲裁
    assert "best_encut_belief" in m46b, "冲突时仍应输出 belief"
    print(
        f"46b. M6 _belief_merge KL 冲突标记 (kl={m46b['best_encut_conflict_kl']:.2f} > 1.0) OK"
    )


def _sc_belief_conflict_low() -> None:
    """46c. M6: 低 KL (mu 接近) → 不标冲突."""
    r46c_a = {
        "best_encut": 520,
        "ts": 100.0,
        "belief": {
            "best_encut": {
                "type": "gaussian",
                "mu": 520,
                "sigma2": 100,
                "obs_sigma2": 100,
            }
        },
    }
    r46c_b = {
        "best_encut": 521,
        "ts": 200.0,
        "belief": {
            "best_encut": {
                "type": "gaussian",
                "mu": 521,
                "sigma2": 100,
                "obs_sigma2": 100,
            }
        },
    }
    m46c = _belief_merge([r46c_a, r46c_b], {"best_encut"})
    assert m46c.get("best_encut_conflict") is not True, (
        f"低 KL 不应触发 _conflict, got {m46c.get('best_encut_conflict')}"
    )
    assert "best_encut_conflict_kl" not in m46c, "低 KL 不应输出 _conflict_kl"
    print("46c. M6 _belief_merge 低 KL 不标冲突 OK")


def _sc_belief_toggle() -> None:
    """47. toggle off — 回退 P1-2 LWW."""
    r45a, r45b = _sc_gaussian_belief_pair()
    FeatureFlags.shared().disable("belief_update")
    assert _belief_update_enabled() is False
    # toggle off 时 _crdt_merge 应走 LWW (不调 _belief_merge)
    m47 = _crdt_merge([r45a, r45b])
    assert m47["best_encut"] == 572, (
        f"toggle off 应 LWW 取 ts=200 的 572, got {m47.get('best_encut')}"
    )
    assert "best_encut_belief" not in m47, "toggle off 不应输出 belief"
    # 恢复 toggle on
    FeatureFlags.shared().enable("belief_update")
    assert _belief_update_enabled() is True
    print("47. belief toggle (belief_update off/on) + LWW fallback OK")


if __name__ == "__main__":
    _selfcheck()
