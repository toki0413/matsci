"""subagent_tool -- lets the main agent dispatch isolated subagents.

Wraps SubagentDispatch so the agent can offload context-heavy tasks
(explore, code, analyze) to isolated sessions without bloating the main
conversation window. Inspired by Kimi Code's coder/explore/plan pattern.

Actions:
  - list_types: 列出所有可用的子 agent 类型
  - dispatch:   派发一个子 agent 执行任务, 返回压缩后的摘要
"""

from __future__ import annotations

import hashlib
import os
import time
from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool
from huginn.types import ToolContext, ToolResult


# P1-2: CRDT 状态合并 — 多 subagent 并行结果用半格 join 合并.
# ponytail: 纯函数, 不引入新依赖. 语义冲突仍走 LLM 仲裁.
# ceiling: G-Set 单调增, 长跑会膨胀; 升级 OR-Set 可删, 但需要 tombstone.
def _crdt_merge_enabled() -> bool:
    """P1-2 toggle: env HUGINN_CRDT_MERGE (默认 on)."""
    return os.environ.get("HUGINN_CRDT_MERGE", "1") != "0"


def _belief_update_enabled() -> bool:
    """P2-6 toggle: env HUGINN_BELIEF_UPDATE (默认 on).

    Bayesian belief update for LWW fields. off 时回退 P1-2 LWW (ts 大者胜).
    ponytail: 默认 on 因 fallback 完整 — result 无 belief 字段时自动走 LWW.
    """
    return os.environ.get("HUGINN_BELIEF_UPDATE", "1") != "0"


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


def _gaussian_kl(
    mu1: float, sigma2_1: float, mu2: float, sigma2_2: float
) -> float:
    """KL(N1 || N2) — 冲突检测用. KL 大 = 两个 belief 严重不一致."""
    import math
    if sigma2_1 <= 0 or sigma2_2 <= 0:
        return 0.0
    return (
        0.5 * (
            math.log(sigma2_2 / sigma2_1)
            + (sigma2_1 + (mu1 - mu2) ** 2) / sigma2_2
            - 1.0
        )
    )


def _belief_merge(
    results: list[dict], lww_fields: set[str]
) -> dict[str, Any]:
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
    for field in lww_fields:
        # 收集所有带 belief 的 result
        belief_results = [
            r for r in results
            if isinstance(r, dict)
            and isinstance(r.get("belief"), dict)
            and isinstance(r["belief"].get(field), dict)
        ]
        if not belief_results:
            # 无 belief, 走 LWW
            best_val = None
            best_ts = -1.0
            for r in results:
                if isinstance(r, dict) and r.get(field) is not None:
                    ts = float(r.get("ts", 0.0) or 0.0)
                    if ts >= best_ts:
                        best_ts = ts
                        best_val = r[field]
            if best_val is not None:
                out[field] = best_val
                out[f"{field}_ts"] = best_ts
            continue

        # 按类型 update (第一个 result 的 belief type 为准, 混合类型跳过)
        btype = belief_results[0]["belief"][field].get("type")
        if btype == "gaussian":
            # 用第一个做 prior, 依次 update
            b0 = belief_results[0]["belief"][field]
            mu = float(b0.get("mu", 0.0))
            s2 = float(b0.get("sigma2", 0.0))
            obs_s2 = float(b0.get("obs_sigma2", 100.0))
            for r in belief_results[1:]:
                b = r["belief"][field]
                o = float(b.get("mu", 0.0))  # 观测值 = 该 result 的 mu
                os2 = float(b.get("obs_sigma2", obs_s2))
                mu, s2 = _gaussian_update(mu, s2, o, os2)
            out[field] = mu  # point estimate = posterior mean
            out[f"{field}_belief"] = {
                "type": "gaussian", "mu": mu, "sigma2": s2,
            }
        elif btype == "beta":
            b0 = belief_results[0]["belief"][field]
            a = float(b0.get("alpha", 1.0))
            b = float(b0.get("beta", 1.0))
            for r in belief_results:
                br = r["belief"][field]
                ns = int(br.get("n_success", 0))
                nf = int(br.get("n_fail", 0))
                a, b = _beta_update(a, b, ns, nf)
            out[field] = a / (a + b)  # point estimate = posterior mean
            out[f"{field}_belief"] = {
                "type": "beta", "alpha": a, "beta": b,
            }
        else:
            # 未知 type, 回退 LWW
            best_val = None
            best_ts = -1.0
            for r in results:
                if isinstance(r, dict) and r.get(field) is not None:
                    ts = float(r.get("ts", 0.0) or 0.0)
                    if ts >= best_ts:
                        best_ts = ts
                        best_val = r[field]
            if best_val is not None:
                out[field] = best_val
                out[f"{field}_ts"] = best_ts
    return out


def _content_hash(s: Any) -> str:
    """Stable hash for dedupe — G-Set 需要. 用 sha8 短摘要足够."""
    raw = str(s).encode("utf-8", errors="ignore")
    return hashlib.sha256(raw).hexdigest()[:8]


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

    # 收集所有字段名 (results 可能字段不一致, 全 union)
    all_keys: set[str] = set()
    for r in results:
        if isinstance(r, dict):
            all_keys.update(r.keys())

    # G-Set 字段: union + dedupe by content hash
    gset_fields = {"findings", "evidence", "limitations", "tool_calls"}
    # LWW 字段: 任意单值字段, ts 大者胜
    # 默认 LWW 候选: best_value, best_encut, answer, conclusion
    lww_fields = {"best_value", "best_encut", "answer", "conclusion", "result"}

    merged: dict[str, Any] = {"merged": True, "n_sources": len(results)}

    # G-Set 合并
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
        merged[field] = list(seen.values())

    # LWW 合并 — P2-6: 有 belief 字段时走 Bayesian update, 否则 LWW.
    # _belief_merge 内部自动 fallback LWW, 这里只需 toggle on 时调它,
    # off 时直接走原 LWW 逻辑 (回归 P1-2 行为).
    if _belief_update_enabled():
        belief_out = _belief_merge(results, lww_fields)
        merged.update(belief_out)
    else:
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
                merged[field] = best_val
                merged[f"{field}_ts"] = best_ts

    # success: OR-join (任一 True → True)
    merged["success"] = any(
        r.get("success", False) for r in results if isinstance(r, dict)
    )

    # summary: 拼接非空 (dedupe by hash)
    seen_sum: dict[str, str] = {}
    for r in results:
        s = r.get("summary") if isinstance(r, dict) else None
        if s and isinstance(s, str):
            h = _content_hash(s)
            if h not in seen_sum:
                seen_sum[h] = s
    if seen_sum:
        merged["summary"] = "\n---\n".join(seen_sum.values())

    # errors: 收集所有非空 error (G-Set)
    errs = [
        r.get("error") for r in results
        if isinstance(r, dict) and r.get("error")
    ]
    if errs:
        seen_err: dict[str, str] = {}
        for e in errs:
            h = _content_hash(e)
            if h not in seen_err:
                seen_err[h] = e
        merged["errors"] = list(seen_err.values())

    # 保留原始 results (审计/调试用), 但放末尾避免污染主 view
    merged["sources"] = results
    # 嵌套 merge 用: 输出本身的 ts = max(输入 ts), 让外层 LWW 判断正确.
    # 不加这个的话 merge(merge(a, b), c) 里 merge(a, b) 的 ts 字段缺失,
    # 外层 LWW 取不到正确 ts.
    merged["ts"] = max(
        (float(r.get("ts", 0.0) or 0.0) for r in results if isinstance(r, dict)),
        default=0.0,
    )
    return merged


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
        dispatch_ctx = dict(args.context)
        dispatch_ctx.setdefault("agent_factory", context.agent_factory)
        dispatch_ctx.setdefault("session_id", context.session_id)
        dispatch_ctx.setdefault("workspace", context.workspace)
        # v7: 透传父 agent 的 approval_callback, 子 agent 调 ASK 工具 (vasp_tool 等) 才能拿到批准.
        dispatch_ctx.setdefault("approval_callback", context.approval_callback)

        # G1: 从 contextvar 读当前递归深度, 透传给 dispatch 守卫.
        # 主 agent 这里读到 0, 子 agent 那里读到 1+.
        from huginn.agents.subagent import _current_depth
        _depth = _current_depth.get()

        # forward subagent intermediate states to the WS via progress_cb
        from huginn.types import progress_cb

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
                    await cb({
                        "type": "subagent_event",
                        "event": "tool_call",
                        "spec": args.spec_name,
                        "tool": tc.get("name", "unknown"),
                    })
            # assistant text (truncated)
            elif hasattr(last, "content") and last.content:
                text = last.content if isinstance(last.content, str) else str(last.content)
                if len(text) > 200:
                    text = text[:200] + "..."
                await cb({
                    "type": "subagent_event",
                    "event": "text",
                    "spec": args.spec_name,
                    "text": text,
                })

        result = await self._dispatch.dispatch(
            args.spec_name, args.task, dispatch_ctx,
            on_state=_on_state, _depth=_depth,
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

    async def _dispatch_parallel(
        self, args: SubagentToolInput, context: ToolContext
    ) -> ToolResult:
        """DAG-aware 并行 dispatch.

        无 dependencies: 全部 asyncio.gather 并行.
        有 dependencies: 用 TaskDAG 拓扑分层, 同层并行, 层间串行.

        ponytail: 硬 cap 4 并行 (API 限速 + 调试可行性). DAG 调度复用 TaskDAG.
        """
        import asyncio

        if not args.tasks:
            return self._missing_field("tasks")
        if len(args.tasks) > 4:
            return ToolResult(
                data=SubagentToolOutput(
                    success=False, action="dispatch_parallel",
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
                        success=False, action="dispatch_parallel",
                        error=f"tasks[{i}] 缺 spec_name 或 task",
                    ).model_dump(),
                    success=False,
                    error=f"tasks[{i}] missing spec_name or task",
                )

        dispatch_ctx = dict(args.context)
        dispatch_ctx.setdefault("agent_factory", context.agent_factory)
        dispatch_ctx.setdefault("session_id", context.session_id)
        dispatch_ctx.setdefault("workspace", context.workspace)
        dispatch_ctx.setdefault("approval_callback", context.approval_callback)
        # G1: 从 contextvar 读递归深度 (跟 _dispatch_subagent 一致)
        from huginn.agents.subagent import _current_depth
        _depth = _current_depth.get()

        # 无 dependencies: 全并行
        if not args.dependencies:
            coros = [
                self._dispatch.dispatch(
                    t["spec_name"], t["task"], dispatch_ctx, _depth=_depth,
                )
                for t in args.tasks
            ]
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
            # P1-2: CRDT 合并 — 半格 join, 字段级无冲突.
            # off 时回退原 list[dict] 行为 (回归测试安全).
            if _crdt_merge_enabled() and len(out_results) > 1:
                merged = _crdt_merge(out_results)
                return ToolResult(
                    data={"action": "dispatch_parallel", "results": out_results, "merged": merged, "n": len(out_results)},
                    success=merged.get("success", True),
                )
            return ToolResult(
                data={"action": "dispatch_parallel", "results": out_results, "n": len(out_results)},
                success=True,
            )

        # 有 dependencies: DAG 分层调度 (极限模式才开)
        import os
        if os.environ.get("HUGINN_EXTREME_DISPATCH", "0").lower() not in ("1", "true"):
            return ToolResult(
                data=SubagentToolOutput(
                    success=False, action="dispatch_parallel",
                    error="DAG-aware dispatch 需开启极限模式 (HUGINN_EXTREME_DISPATCH=1)",
                ).model_dump(),
                success=False,
                error="DAG dispatch requires HUGINN_EXTREME_DISPATCH=1",
            )
        from huginn.agents.task_dag import TaskDAG
        # task ID = spec_name + task 前 20 字符 (ponytail: 不引入显式 ID 字段)
        task_ids = [f"{t['spec_name']}:{t['task'][:20]}" for t in args.tasks]
        try:
            dag = TaskDAG(tasks=task_ids, dependencies=args.dependencies)
        except ValueError as e:
            return ToolResult(
                data=SubagentToolOutput(
                    success=False, action="dispatch_parallel", error=f"DAG 错误: {e}",
                ).model_dump(),
                success=False,
                error=str(e),
            )
        layers = dag.parallel_layers()
        id_to_task = dict(zip(task_ids, args.tasks))
        all_results: list[dict] = []
        for layer in layers:
            coros = [
                self._dispatch.dispatch(
                    id_to_task[tid]["spec_name"],
                    id_to_task[tid]["task"],
                    dispatch_ctx, _depth=_depth,
                )
                for tid in layer
            ]
            layer_results = await asyncio.gather(*coros, return_exceptions=True)
            for r in layer_results:
                if isinstance(r, Exception):
                    all_results.append({"success": False, "error": str(r)})
                else:
                    _d = r.to_dict()
                    _d.setdefault("ts", time.time())
                    all_results.append(_d)
        # P1-2: DAG 路径同样 CRDT 合并
        if _crdt_merge_enabled() and len(all_results) > 1:
            merged = _crdt_merge(all_results)
            return ToolResult(
                data={"action": "dispatch_parallel", "results": all_results, "merged": merged, "n": len(all_results), "layers": layers},
                success=merged.get("success", True),
            )
        return ToolResult(
            data={"action": "dispatch_parallel", "results": all_results, "n": len(all_results), "layers": layers},
            success=True,
        )

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
    import json as _json

    # 33. G-Set union — evidence 合并, 无 dup
    r_a = {"findings": ["encut=520 OK"], "evidence": ["ref1", "ref2"], "success": True}
    r_b = {"findings": ["kpoints=4x4x4"], "evidence": ["ref2", "ref3"], "success": True}
    m33 = _crdt_merge([r_a, r_b])
    ev_set = set(m33["evidence"])
    assert ev_set == {"ref1", "ref2", "ref3"}, f"G-Set union 应无 dup: {m33['evidence']}"
    f_set = set(m33["findings"])
    assert f_set == {"encut=520 OK", "kpoints=4x4x4"}, f"findings union: {m33['findings']}"
    print("33. CRDT G-Set union (evidence + findings, dedupe) OK")

    # 34. LWW-Register — best_value ts 大者胜
    r_old = {"best_value": 520, "ts": 100.0}
    r_new = {"best_value": 572, "ts": 200.0}
    m34 = _crdt_merge([r_old, r_new])
    assert m34["best_value"] == 572, f"LWW 应取 ts 大的 572: {m34.get('best_value')}"
    assert m34["best_value_ts"] == 200.0
    # 交换顺序也应取 ts 大者
    m34b = _crdt_merge([r_new, r_old])
    assert m34b["best_value"] == 572, f"交换律: LWW 应仍取 572: {m34b.get('best_value')}"
    print("34. CRDT LWW-Register (ts 大者胜 + 交换律) OK")

    # 35. 半格三公理 — 交换 / 结合 / 幂等
    import random
    random.seed(35)
    r1 = {"findings": ["a"], "evidence": ["e1"], "best_value": 1, "ts": 1.0, "success": True}
    r2 = {"findings": ["b"], "evidence": ["e2"], "best_value": 2, "ts": 2.0, "success": False}
    r3 = {"findings": ["c"], "evidence": ["e3"], "best_value": 3, "ts": 3.0, "success": True}
    # 交换律: merge(a, b) == merge(b, a) — 比较 JSON 字符串 (顺序无关)
    def _canon(d: dict) -> str:
        # 去掉 sources (含原始 list, 顺序敏感) + 元信息字段 (n_sources / ts 是
        # 审计/计数, 不是半格 join 的状态值), 只看合并字段.
        # G-Set 字段在半格语义里是 set, list 顺序无关 — 排序后再比.
        _skip = {"sources", "n_sources", "ts"}
        d2: dict = {}
        for k, v in d.items():
            if k in _skip or k.endswith("_ts"):
                continue
            if isinstance(v, list):
                d2[k] = sorted(v, key=lambda x: str(x))
            else:
                d2[k] = v
        return _json.dumps(d2, sort_keys=True, default=str)
    m_ab = _canon(_crdt_merge([r1, r2]))
    m_ba = _canon(_crdt_merge([r2, r1]))
    assert m_ab == m_ba, f"交换律失败:\n{m_ab}\n{m_ba}"
    # 结合律: merge(merge(a, b), c) == merge(a, merge(b, c))
    # 注意: merge 的输入是 list[dict], 输出是 dict — 要把 merge(a, b) 包回 list
    m_ab_c = _canon(_crdt_merge([_crdt_merge([r1, r2]), r3]))
    m_a_bc = _canon(_crdt_merge([r1, _crdt_merge([r2, r3])]))
    assert m_ab_c == m_a_bc, f"结合律失败:\n{m_ab_c}\n{m_a_bc}"
    # 幂等: merge(a, a) == a (在半格意义下 — 去掉 sources 后)
    m_aa = _canon(_crdt_merge([r1, r1]))
    m_a = _canon(_crdt_merge([r1]))
    assert m_aa == m_a, f"幂等律失败:\n{m_aa}\n{m_a}"
    print("35. CRDT 半格三公理 (交换 / 结合 / 幂等) OK")

    # 36. 整合 — 4 subagent, 2 失败 2 成功, mixed fields
    r_s1 = {
        "findings": ["encut=572 OK"], "evidence": ["conv1"],
        "best_encut": 572, "ts": 100.0, "success": True,
        "summary": "subagent A: encut=572",
    }
    r_s2 = {
        "findings": ["encut=572 OK"], "evidence": ["conv2"],  # 重复 finding
        "best_encut": 520, "ts": 200.0, "success": True,
        "summary": "subagent B: encut=520 (later)",
    }
    r_f1 = {"success": False, "error": "OOM", "ts": 150.0}
    r_f2 = {"success": False, "error": "OOM", "ts": 180.0}  # 重复 error
    m36 = _crdt_merge([r_s1, r_s2, r_f1, r_f2])
    # findings dedupe
    assert len(m36["findings"]) == 1, f"重复 finding 应 dedupe: {m36['findings']}"
    # evidence union
    assert set(m36["evidence"]) == {"conv1", "conv2"}
    # LWW: best_encut ts=200 胜 (520)
    assert m36["best_encut"] == 520, f"LWW 应取 ts=200 的 520: {m36.get('best_encut')}"
    # success OR-join
    assert m36["success"] is True, "OR-join: 任一 True → True"
    # errors dedupe
    assert len(m36["errors"]) == 1, f"重复 error 应 dedupe: {m36['errors']}"
    # summary 拼接 (dedupe)
    assert "subagent A" in m36["summary"] and "subagent B" in m36["summary"]
    print("36. CRDT 整合 (4 subagent, 2 成功 2 失败, mixed fields) OK")

    # 37. toggle off — 不合并 (回归原 list[dict])
    os.environ["HUGINN_CRDT_MERGE"] = "0"
    assert _crdt_merge_enabled() is False
    os.environ["HUGINN_CRDT_MERGE"] = "1"
    assert _crdt_merge_enabled() is True
    print("37. CRDT toggle (HUGINN_CRDT_MERGE=0/1) OK")

    # ── P2-6: Bayesian belief update ──────────────────────────────
    # 42. Gaussian update — prior N(520, 20²), obs N(572, 10²)
    mu42, s2_42 = _gaussian_update(520.0, 400.0, 572.0, 100.0)
    # 后验 μ ≈ (520/400 + 572/100) / (1/400 + 1/100) = (1.3 + 5.72) / 0.0125 = 561.6
    # 后验 σ² ≈ 1/0.0125 = 80
    assert abs(mu42 - 561.6) < 0.1, f"Gaussian μ' 应 ≈ 561.6, got {mu42}"
    assert abs(s2_42 - 80.0) < 0.1, f"Gaussian σ'² 应 ≈ 80, got {s2_42}"
    print("42. Gaussian Bayesian update (μ=520→561.6, σ²=400→80) OK")

    # 43. Beta update — prior Beta(1, 1), 3 success 1 fail
    a43, b43 = _beta_update(1.0, 1.0, 3, 1)
    assert a43 == 4.0 and b43 == 2.0, f"Beta(1,1)+3s1f 应 Beta(4,2), got ({a43},{b43})"
    # point estimate = 4/(4+2) = 2/3
    mean43 = a43 / (a43 + b43)
    assert abs(mean43 - 0.667) < 0.001, f"mean 应 ≈ 0.667, got {mean43}"
    print("43. Beta Bayesian update (Beta(1,1)→Beta(4,2), mean=0.667) OK")

    # 44. 无 belief → fallback LWW (兼容 P1-2)
    r44a = {"best_value": 520, "ts": 100.0}
    r44b = {"best_value": 572, "ts": 200.0}
    m44 = _belief_merge([r44a, r44b], {"best_value"})
    assert m44["best_value"] == 572, f"无 belief 应 LWW 取 ts=200: {m44.get('best_value')}"
    assert m44["best_value_ts"] == 200.0
    print("44. belief_merge 无 belief → fallback LWW OK")

    # 45. 有 belief → Bayesian update
    r45a = {
        "best_encut": 520, "ts": 100.0,
        "belief": {"best_encut": {"type": "gaussian", "mu": 520, "sigma2": 400, "obs_sigma2": 100}},
    }
    r45b = {
        "best_encut": 572, "ts": 200.0,
        "belief": {"best_encut": {"type": "gaussian", "mu": 572, "sigma2": 100, "obs_sigma2": 100}},
    }
    m45 = _belief_merge([r45a, r45b], {"best_encut"})
    # r45a 做 prior, r45b 做 obs → 同 42 的计算
    assert abs(m45["best_encut"] - 561.6) < 0.1, (
        f"Bayesian 应 ≈ 561.6, got {m45.get('best_encut')}"
    )
    assert "best_encut_belief" in m45, "应输出 posterior belief"
    assert m45["best_encut_belief"]["type"] == "gaussian"
    print("45. belief_merge Gaussian (520+572 → 561.6 + belief) OK")

    # 45b. Beta belief merge
    r45c = {
        "success_rate": 0.5, "ts": 1.0,
        "belief": {"success_rate": {"type": "beta", "alpha": 1, "beta": 1, "n_success": 3, "n_fail": 1}},
    }
    r45d = {
        "success_rate": 0.7, "ts": 2.0,
        "belief": {"success_rate": {"type": "beta", "alpha": 1, "beta": 1, "n_success": 2, "n_fail": 3}},
    }
    m45b = _belief_merge([r45c, r45d], {"success_rate"})
    # Beta(1,1) + (3s+2s, 1f+3f) = Beta(6, 5), mean = 6/11 ≈ 0.545
    assert abs(m45b["success_rate"] - 6/11) < 0.001, (
        f"Beta merge mean 应 ≈ 0.545, got {m45b.get('success_rate')}"
    )
    assert m45b["success_rate_belief"]["alpha"] == 6.0
    assert m45b["success_rate_belief"]["beta"] == 5.0
    print("45b. belief_merge Beta (Beta(1,1)+5s4f → Beta(6,5)) OK")

    # 46. KL 冲突检测 — 两个 Gaussian belief KL 大 = 严重不一致
    kl46 = _gaussian_kl(520.0, 100.0, 572.0, 100.0)
    kl46_close = _gaussian_kl(520.0, 100.0, 521.0, 100.0)
    assert kl46 > kl46_close, f"KL(520,572) 应 > KL(520,521): {kl46} vs {kl46_close}"
    assert kl46 > 0.1, f"严重冲突 KL 应 > 0.1, got {kl46}"
    print("46. Gaussian KL 冲突检测 (520 vs 572 KL > 0.1) OK")

    # 47. toggle off — 回退 P1-2 LWW
    os.environ["HUGINN_BELIEF_UPDATE"] = "0"
    assert _belief_update_enabled() is False
    # toggle off 时 _crdt_merge 应走 LWW (不调 _belief_merge)
    m47 = _crdt_merge([r45a, r45b])
    assert m47["best_encut"] == 572, (
        f"toggle off 应 LWW 取 ts=200 的 572, got {m47.get('best_encut')}"
    )
    assert "best_encut_belief" not in m47, "toggle off 不应输出 belief"
    # 恢复 toggle on
    os.environ["HUGINN_BELIEF_UPDATE"] = "1"
    assert _belief_update_enabled() is True
    print("47. belief toggle (HUGINN_BELIEF_UPDATE=0/1) + LWW fallback OK")

    print("subagent_tool selfcheck OK (33-37 CRDT + 42-47 Bayesian belief)")


if __name__ == "__main__":
    _selfcheck()
