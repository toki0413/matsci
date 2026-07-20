"""CodeAct loop — LLM outputs executable Python code as the action space.

Dual-track with the default tool_call mode:
- tool_call (default): LLM emits JSON function calls, langgraph runs them.
- code_act: LLM emits ```python blocks; we exec them in-process with all
  registered tools injected as namespace functions.

Research: CodeAct paper (Wang et al., ICML 2024, arXiv:2402.01030) reports
+20% success / -30% steps on M3ToolEval vs JSON function calling.

Safety:
- restricted_python.validate_code rejects os/subprocess/__import__/eval/etc.
- HPC / bash / code_tool are NOT injected (no side effects, no recursion).
- Each code source is audit-logged.
- 3 consecutive code exceptions -> degrade to tool_call (chat() handles).
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import time
from datetime import datetime
from typing import Any, AsyncIterator, Awaitable, Callable

from huginn.security.restricted_python import RestrictedPythonError, validate_code
from huginn.tools.registry import ToolRegistry
from huginn.types import ToolContext
from huginn.utils.async_bridge import run_async

logger = logging.getLogger(__name__)


# Tools we never inject into the CodeAct namespace.
# - hpc_client / bash_tool / shell_tool / container_exec: external side effects
#   that bypass the audit trail CodeAct sets up. Keep them on the tool_call
#   track where langgraph + callbacks already trace them.
# - code_tool: would let LLM spawn nested sandboxes from inside code_act,
#   recursion footgun.
_BLOCKED_TOOLS = frozenset(
    {"hpc_client", "bash_tool", "shell_tool", "container_exec", "code_tool"}
)

# Hard ceiling on turns per CodeAct run. The paper shows median 6-8 steps on
# M³ToolEval; 15 leaves headroom for exploration without runaway cost.
_MAX_TURNS = 15
_DEGRADE_AFTER_ERRORS = 3

_CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)

# ── Human-in-the-loop approval gate ─────────────────────────────────────
#
# 借鉴 Cline 三档审批: 只读自动放行 / 写入确认 / 破坏性强制确认.
# 避免确认疲劳的关键是 default-allow 纯计算代码, 只在真有副作用时拦.
#
# ApprovalFn 签名: async (code, risk_level, reason) -> ApprovalDecision
#   - risk_level: "low" | "medium" | "high"
#   - 返回 ("approve", None) 放行, ("approve_always", reason) 本会话同类自动放行,
#     ("deny", reason) 拒绝并把 reason 喂回 LLM, ("edit", new_code) 替换代码.
#
# 不传 approval_fn 时: low 自动放行, medium/high yield approval_request 事件
# 并暂停 (调用方负责 resume). 这是 LangGraph interrupt 模式的轻量版.
RiskLevel = str  # "low" | "medium" | "high"
ApprovalDecision = tuple[str, str | None]  # (action, payload)
ApprovalFn = Callable[[str, RiskLevel, str], Awaitable[ApprovalDecision]]

# 代码里暗示副作用的关键词. 命中任一就升到 high risk.
# ponytail: 关键词扫描会有误报 (如变量名含 "write"), 但 exec 前拦一道比
# 事后审计强. 升级: AST 分析精确识别 Call 节点的函数名.
_SIDE_EFFECT_PATTERNS = re.compile(
    r"\b(?:savefig|to_csv|to_excel|to_json|to_sql|\.write\(|open\(|"
    r"mkdir|rmdir|remove|unlink|system|popen|subprocess|"
    r"requests\.|urllib|httpx|aiohttp)\b"
)
# 调用 destructive 工具的代码块也是 high risk
_DESTRUCTIVE_TOOL_CALLS = re.compile(
    r"\b(?:delete_|remove_|drop_|submit_|run_job|execute_hpc|deploy_)\w*\s*\("
)


def _assess_risk(code: str, tools: dict[str, Any]) -> tuple[RiskLevel, str]:
    """三档风险评级 (Cline 模式).

    low: 纯计算, 只调 print/math/json/numpy 等只读工具, 无副作用关键词
    medium: 调用 search/rag/literature 等只读外部工具 (有网络/IO 但不改状态)
    high: 代码含 savefig/open/write 等副作用, 或调用 destructive 工具
    """
    if _SIDE_EFFECT_PATTERNS.search(code) or _DESTRUCTIVE_TOOL_CALLS.search(code):
        return ("high", "代码含副作用关键词或 destructive 工具调用")
    # 扫代码里调用的工具名, 看是否触发 destructive
    called = set()
    for name in tools:
        if name in _BLOCKED_TOOLS:
            continue
        if re.search(rf"\b{name}\s*\(", code):
            called.add(name)
    for name in called:
        tool = tools.get(name)
        if tool is None:
            continue
        if getattr(tool, "destructive", False):
            return ("high", f"调用 destructive 工具 {name}")
    if called:
        return ("medium", f"调用工具: {sorted(called)}")
    return ("low", "纯计算")


# 本会话内 "approve_always" 的风险级别白名单, 避免确认疲劳.
# ponytail: 模块级 dict, 进程内共享. 升级: 按 session_id 隔离.
_auto_approved: dict[str, set[str]] = {}


def _mark_auto_approved(session_id: str, risk_level: RiskLevel) -> None:
    _auto_approved.setdefault(session_id, set()).add(risk_level)


def _is_auto_approved(session_id: str, risk_level: RiskLevel) -> bool:
    return risk_level in _auto_approved.get(session_id, set())


def reset_auto_approvals(session_id: str | None = None) -> None:
    """测试用: 清空 approve_always 白名单."""
    if session_id is None:
        _auto_approved.clear()
    else:
        _auto_approved.pop(session_id, None)


# ── Trust score (HRI: trust calibration, Lee & See 2004) ──
# trust = f(performance, process, purpose). 这里用 approval 历史近似:
# deny → -0.10, edit → -0.05, approve_always → +0.05, approve → +0.02
# trust < 0.3: 即使 approve_always 也强制 ASK (微管理倾向)
# trust > 0.7: medium 风险自动放行 (减少打扰)
_trust_scores: dict[str, float] = {}
_approval_history: dict[str, list[dict]] = {}

_TRUST_DELTA = {
    "deny": -0.10,
    "edit": -0.05,
    "approve_always": 0.05,
    "approve": 0.02,
}


def _get_trust(session_id: str) -> float:
    return _trust_scores.get(session_id, 0.5)


def _record_approval(session_id: str, action: str, risk: str, code_preview: str = "") -> float:
    """记录一次 approval 决策, 更新 trust_score, 返回新值."""
    delta = _TRUST_DELTA.get(action, 0.0)
    new_score = max(0.0, min(1.0, _get_trust(session_id) + delta))
    _trust_scores[session_id] = new_score
    hist = _approval_history.setdefault(session_id, [])
    hist.append({"action": action, "risk": risk, "ts": time.time(), "code": code_preview[:200]})
    if len(hist) > 50:
        del hist[: len(hist) - 50]
    return new_score


def _should_force_ask(session_id: str) -> bool:
    """trust < 0.3 时强制 ASK, 即使 approve_always 也不放行."""
    return _get_trust(session_id) < 0.3


def _should_auto_medium(session_id: str, risk: str) -> bool:
    """trust > 0.7 时 medium 风险自动放行."""
    return risk == "medium" and _get_trust(session_id) > 0.7


# ── Approval frequency budget (HRI: alert fatigue avoidance) ──
# 每次进 approval gate decrement, budget <= 3 时自动升级 (避免确认疲劳)
# 但 trust < 0.3 时不升级 (微管理优先于疲劳)
_approval_budgets: dict[str, int] = {}
_BUDGET_INITIAL = 10
_BUDGET_ESCALATION_THRESHOLD = 3


def _get_budget(session_id: str) -> int:
    return _approval_budgets.get(session_id, _BUDGET_INITIAL)


def _decrement_budget(session_id: str) -> int:
    curr = _get_budget(session_id)
    if curr > 0:
        curr -= 1
        _approval_budgets[session_id] = curr
    return curr


# ── SUGGEST mode (HRI: Levels of Automation Level 4-6) ──
# agent 输出代码, 前端展示为可编辑, 用户 Ctrl+Enter 才执行.
# 强制所有 risk 级别都走 approval 流程, 用 suggest_code 事件让前端展示编辑器.
_suggest_modes: dict[str, bool] = {}


def _is_suggest_mode(session_id: str) -> bool:
    return _suggest_modes.get(session_id, False)


def set_suggest_mode(session_id: str, enabled: bool) -> None:
    _suggest_modes[session_id] = enabled


# SUGGEST 恢复机制: WS handler 调 resume_suggest() 唤醒被阻塞的 code_act_turn.
# ponytail: 模块级 dict 做 agent 注册表, 进程内共享. 升级: 按 user_id 隔离 + TTL.
_active_agents: dict[str, Any] = {}


def _register_agent(session_id: str, agent: Any) -> None:
    _active_agents[session_id] = agent


def _unregister_agent(session_id: str) -> None:
    _active_agents.pop(session_id, None)


def resume_suggest(session_id: str, action: str, edited_code: str = "") -> bool:
    """WS handler 调这个来唤醒被 SUGGEST 阻塞的 agent."""
    agent = _active_agents.get(session_id)
    if agent is None:
        return False
    agent._approval_decision = (action, edited_code or None)
    event = getattr(agent, "_approval_resume", None)
    if event is None:
        return False
    event.set()
    return True


# ── Dynamic risk threshold (HRI: trust-adaptive risk classification) ──
# threshold > 0.7: lenient — medium 降级为 low (auto-approve)
# threshold < 0.3: strict — medium 升级为 high (force ask)
# threshold = f(trust, error_streak): trust 高 + 无错误 → 宽容; 反之 → 严格
_risk_thresholds: dict[str, float] = {}


def _compute_risk_threshold(session_id: str, error_streak: int = 0) -> float:
    trust = _get_trust(session_id)
    # error_streak 拉低阈值: 每次错误 -0.05, 最多 -0.2
    error_penalty = min(error_streak * 0.05, 0.2)
    threshold = max(0.1, min(0.9, trust - error_penalty))
    _risk_thresholds[session_id] = threshold
    return threshold


def _apply_dynamic_threshold(risk: str, threshold: float) -> tuple[str, str]:
    """根据动态阈值调整 risk 级别. 返回 (new_risk, reason)."""
    if risk == "medium":
        if threshold > 0.7:
            return ("low", f"lenient threshold={threshold:.2f}, medium→low")
        if threshold < 0.3:
            return ("high", f"strict threshold={threshold:.2f}, medium→high")
    return (risk, "")


def _extract_python_blocks(text: str) -> list[str]:
    """Pull ```python ...``` blocks out of an LLM response. Falls back to
    treating the whole text as code if it has no fences but looks like Python
    (starts with a keyword / identifier)."""
    blocks = _CODE_BLOCK_RE.findall(text)
    if blocks:
        return [b.strip() for b in blocks]
    stripped = text.strip()
    # heuristic: bare code with no fences — accept only if it parses
    if stripped and not stripped.startswith(("#", "```")):
        try:
            compile(stripped, "<code_act>", "exec")
            return [stripped]
        except SyntaxError:
            return []
    return []


def _tool_signature(name: str, tool: Any) -> str:
    """One-line signature for the system prompt."""
    desc = (tool.description or "").splitlines()[0] if tool.description else ""
    schema = tool.input_schema
    if schema is None:
        return f"{name}()  # {desc}"
    fields = getattr(schema, "model_fields", None) or {}
    parts = [fname for fname in fields if fname != "action"]
    inner = ", ".join(parts)
    return f"{name}({inner})  # {desc}"


def _build_system_prompt(tools: dict[str, Any]) -> str:
    sigs = "\n".join(f"- {name}: {_tool_signature(name, t)}" for name, t in tools.items())
    prompt = f"""You are Huginn, a materials science agent running in CodeAct mode.

In this mode you express every action as a Python code block. The block is
executed in-process; tools below are available as plain Python functions.

Available tools:
{sigs}

Rules:
1. Output ONE ```python block per turn. It is exec'd in-process immediately.
2. Tool calls return their `data` payload on success, or "ERROR: <msg>" on failure.
3. Use print() to surface intermediate results — printed text is fed back to you.
4. End with a normal text answer (no code block) when you have the answer.
5. Imports of os, sys, subprocess, socket are blocked. Do not attempt them.
6. Stay within the working directory. No network, no fork/exec.

Remember: one code block per turn, then stop and wait for the execution result."""
    # AtomWorld benchmark functions are injected when HUGINN_USE_ATOMWORLD=1.
    # Surface them in the prompt so the agent knows they exist without dir().
    if os.environ.get("HUGINN_USE_ATOMWORLD", "0") == "1":
        prompt += (
            "\n\nAtomWorld benchmark tools (plain Python functions in this "
            "namespace): atomworld_evaluate(target_cif, generated_output), "
            "atomworld_apply_action(input_cif, action_name, **params), "
            "atomworld_list_actions(). You can also `from atomworld import "
            "evaluate` for the raw upstream API."
        )
    return prompt


def _build_namespace(
    agent: Any,
    tools: dict[str, Any],
    context: ToolContext,
    stdout_buf: io.StringIO,
) -> dict[str, Any]:
    """Assemble the globals dict for exec(). Tools become sync wrappers."""
    namespace: dict[str, Any] = {
        "__name__": "code_act",
        "_stdout_buf": stdout_buf,
        "print": lambda *a, **kw: stdout_buf.write(
            " ".join(str(x) for x in a) + (kw.get("end") or "\n")
        ),
        "json": json,
    }

    # Optional scientific stack — only if the user has them installed.
    for mod_name in ("math", "statistics", "numpy", "pandas", "sympy"):
        try:
            namespace[mod_name] = __import__(mod_name)
        except ImportError:
            pass

    # AtomWorld benchmark — opt-in via HUGINN_USE_ATOMWORLD=1 (mirrors
    # BranchIncubator gating). atomworld_tool already no-ops when the
    # atomworld package isn't installed, so we just log and skip.
    if os.environ.get("HUGINN_USE_ATOMWORLD", "0") == "1":
        try:
            from huginn.tools import atomworld_tool as _aw
            if _aw.is_available():
                namespace["atomworld_evaluate"] = _aw.evaluate
                namespace["atomworld_apply_action"] = _aw.apply_action
                namespace["atomworld_list_actions"] = _aw.list_actions
            else:
                logger.warning(
                    "atomworld_tool: atomworld package not installed, "
                    "skipping CodeAct registration"
                )
        except ImportError:
            logger.warning(
                "atomworld_tool: module import failed, skipping CodeAct registration"
            )

    # Tool wrappers — sync facade over async tool.call via run_async bridge.
    for name, tool in tools.items():
        if name in _BLOCKED_TOOLS:
            continue
        if not tool.active or not tool.is_available():
            continue
        namespace[name] = _make_tool_wrapper(tool, context, name)

    return namespace


def _make_tool_wrapper(tool: Any, context: ToolContext, name: str) -> Any:
    """Wrap an async HuginnTool as a sync callable for the exec namespace."""

    def _call(**kwargs: Any) -> Any:
        if tool.input_schema is not None:
            try:
                args = tool.input_schema(**kwargs)
            except Exception as exc:
                return f"ERROR: invalid args for {name}: {exc}"
        else:
            args = kwargs
        try:
            result = run_async(tool.call(args, context))
        except Exception as exc:
            return f"ERROR: {name} raised: {exc}"
        if not result.success:
            return f"ERROR: {result.error}"
        return result.data

    _call.__name__ = name
    _call.__doc__ = tool.description or ""
    return _call


def _audit_code(agent: Any, code: str, error: str | None) -> None:
    """Best-effort audit log of every code source we exec."""
    audit_logger = getattr(getattr(agent, "session_state", None), "audit_logger", None)
    if audit_logger is None:
        try:
            ctx = getattr(agent, "_session_state", None)
            audit_logger = getattr(ctx, "audit_logger", None)
        except Exception:
            return
    if audit_logger is None:
        return
    try:
        audit_logger.log(
            event_type="code_act_exec",
            actor="agent",
            action="code_act",
            details={
                "success": error is None,
                "timestamp": datetime.now().isoformat(),
            },
            input_data=code,
            output_data=error,
        )
    except Exception:
        logger.debug("code_act audit log failed", exc_info=True)


async def run_code_act_turn(
    agent: Any,
    message: str,
    thread_id: str = "default",
) -> AsyncIterator[dict[str, Any]]:
    """One CodeAct conversation turn. Yields stream events.

    Event types:
      - {type: "token", content}: streamed LLM token (best-effort, model-dependent)
      - {type: "assistant_text", content}: full LLM response text for this turn
      - {type: "code_executed", code, stdout, error}: result of exec'ing a block
      - {type: "final", content}: terminal answer, loop ends
      - {type: "code_act_degraded"}: 3 consecutive errors, caller should fall back

    The loop terminates when the LLM stops emitting code blocks, or after
    _MAX_TURNS iterations, or on degradation.
    """
    from langchain_core.messages import (
        AIMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
    )

    # Collect eligible tools up-front so the system prompt lists a stable set.
    tools: dict[str, Any] = {}
    for tool_name in ToolRegistry.list_tools():
        tool = ToolRegistry.get(tool_name)
        if tool is None:
            continue
        tools[tool_name] = tool

    workspace = getattr(getattr(agent, "session_state", None), "workspace", None) or "."
    context = ToolContext(
        session_id=f"code_act:{thread_id}",
        workspace=str(workspace),
        audit_logger=getattr(getattr(agent, "session_state", None), "audit_logger", None),
    )

    system_prompt = _build_system_prompt(tools)
    messages: list[Any] = [SystemMessage(content=system_prompt), HumanMessage(content=message)]

    model = agent.select_model("agent") if hasattr(agent, "select_model") else agent.model

    # 注册 agent 供 SUGGEST mode WS handler 唤醒
    import asyncio
    if not hasattr(agent, "_approval_resume"):
        agent._approval_resume = asyncio.Event()
    _register_agent(context.session_id, agent)

    error_streak = 0
    for turn in range(_MAX_TURNS):
        try:
            resp = await model.ainvoke(messages)
        except Exception as exc:
            _unregister_agent(context.session_id)
            yield {"type": "final", "content": f"[CodeAct] model call failed: {exc}"}
            return

        content = str(resp.content) if not isinstance(resp.content, list) else "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in resp.content
        )
        messages.append(AIMessage(content=content))
        yield {"type": "assistant_text", "content": content, "turn": turn}

        code_blocks = _extract_python_blocks(content)
        if not code_blocks:
            # No code → terminal answer
            _unregister_agent(context.session_id)
            yield {"type": "final", "content": content}
            return

        # Exec each block in order; feed results back as a single ToolMessage.
        combined_feedback: list[str] = []
        for code in code_blocks:
            # ── Human-in-the-loop approval gate (Cline 三档模式) ──
            risk, risk_reason = _assess_risk(code, tools)
            approval_fn: ApprovalFn | None = getattr(agent, "approval_fn", None)
            session_id = context.session_id

            # ── Dynamic risk threshold (HRI #5: trust-adaptive classification) ──
            threshold = _compute_risk_threshold(session_id, error_streak)
            new_risk, adj_reason = _apply_dynamic_threshold(risk, threshold)
            threshold_evt: dict[str, Any] = {
                "type": "risk_threshold",
                "threshold": round(threshold, 2),
                "risk": new_risk,
            }
            if new_risk != risk:
                threshold_evt["original_risk"] = risk
                threshold_evt["adjusted_risk"] = new_risk
                threshold_evt["reason"] = adj_reason
                risk = new_risk
                if adj_reason:
                    risk_reason = f"{risk_reason} | {adj_reason}" if risk_reason else adj_reason
            yield threshold_evt

            # ── SUGGEST mode (HRI #4: LoA Level 4-6, 强制所有代码先经用户编辑) ──
            # SUGGEST 批准后设 suggest_skip=True, 跳过下面的 trust/budget gate
            suggest_skip = False
            if _is_suggest_mode(session_id):
                resume_event = agent._approval_resume
                resume_event.clear()
                yield {
                    "type": "suggest_code",
                    "code": code,
                    "risk": risk,
                    "reason": risk_reason,
                    "turn": turn,
                }
                await resume_event.wait()
                resume_decision = getattr(agent, "_approval_decision", None)
                agent._approval_decision = None
                _nt = _record_approval(
                    session_id,
                    resume_decision[0] if resume_decision else "deny",
                    risk,
                    code[:200],
                )
                yield {"type": "trust_update", "trust": _nt, "action": resume_decision[0] if resume_decision else "deny", "risk": risk}
                if resume_decision is None or resume_decision[0] == "deny":
                    err = f"用户拒绝 (suggest, risk={risk}): {resume_decision[1] if resume_decision else 'no decision'}"
                    combined_feedback.append(f"```python\n{code}\n```\n--- DENIED ---\n{err}")
                    error_streak += 1
                    continue
                action, payload = resume_decision
                if action == "approve_always":
                    _mark_auto_approved(session_id, risk)
                if action == "edit" and payload:
                    code = payload
                suggest_skip = True

            # ── Trust + Budget adaptive threshold ──
            trust = _get_trust(session_id)
            budget = _get_budget(session_id)
            # budget 不足 + 信任度不低 → 自动升级 (避免确认疲劳)
            budget_escalated = (
                not suggest_skip
                and risk != "low"
                and budget <= _BUDGET_ESCALATION_THRESHOLD
                and not _should_force_ask(session_id)
                and not _is_auto_approved(session_id, risk)
            )
            if budget_escalated:
                _mark_auto_approved(session_id, risk)

            skip_approval = suggest_skip or (
                risk == "low"
                or _should_auto_medium(session_id, risk)
                or budget_escalated
                or (_is_auto_approved(session_id, risk) and not _should_force_ask(session_id))
            )

            if budget_escalated:
                yield {
                    "type": "budget_escalation",
                    "remaining": budget,
                    "risk": risk,
                }

            if not skip_approval:
                new_budget = _decrement_budget(session_id)
                yield {"type": "budget_update", "remaining": new_budget}
                if approval_fn is not None:
                    # 调用方注入了同步/异步审批回调, 直接等结果
                    action, payload = await approval_fn(code, risk, risk_reason)
                    if action == "deny":
                        err = f"用户拒绝执行 (risk={risk}): {payload or risk_reason}"
                        _audit_code(agent, code, err)
                        combined_feedback.append(
                            f"```python\n{code}\n```\n--- DENIED ---\n{err}"
                        )
                        error_streak += 1
                        if error_streak >= _DEGRADE_AFTER_ERRORS:
                            _unregister_agent(context.session_id)
                            yield {
                                "type": "code_act_degraded",
                                "reason": f"{error_streak} consecutive errors/denials",
                                "last_error": err,
                            }
                            return
                        continue
                    if action == "approve_always":
                        _mark_auto_approved(session_id, risk)
                    if action == "edit" and payload:
                        code = payload  # 用户改了代码, 用新版执行
                else:
                    # 没有回调 → yield approval_request, 调用方 resume 后继续
                    # ponytail: 简化版 LangGraph interrupt. 调用方拿到事件后
                    # 决定 approve/deny/edit, 通过 asyncio.Event 或类似机制 resume.
                    # 这里用 yield + 等待 agent._approval_resume event 的模式.
                    resume_event = getattr(agent, "_approval_resume", None)
                    resume_decision: ApprovalDecision | None = None
                    yield {
                        "type": "approval_request",
                        "code": code,
                        "risk": risk,
                        "reason": risk_reason,
                        "turn": turn,
                    }
                    if resume_event is not None:
                        # 调用方 set event 后把决策放进 agent._approval_decision
                        await resume_event.wait()
                        resume_decision = getattr(agent, "_approval_decision", None)
                        resume_event.clear()
                    if resume_decision is None:
                        # 无 resume 机制 → 保守拒绝
                        err = f"approval required (risk={risk}) but no resume mechanism"
                        combined_feedback.append(
                            f"```python\n{code}\n```\n--- DENIED ---\n{err}"
                        )
                        error_streak += 1
                        continue
                    action, payload = resume_decision
                    _nt = _record_approval(session_id, action, risk, code[:200])
                    yield {"type": "trust_update", "trust": _nt, "action": action, "risk": risk}
                    if action == "deny":
                        err = f"用户拒绝 (risk={risk}): {payload or risk_reason}"
                        combined_feedback.append(
                            f"```python\n{code}\n```\n--- DENIED ---\n{err}"
                        )
                        error_streak += 1
                        continue
                    if action == "approve_always":
                        _mark_auto_approved(session_id, risk)
                    if action == "edit" and payload:
                        code = payload

            stdout_buf = io.StringIO()
            namespace = _build_namespace(agent, tools, context, stdout_buf)
            error: str | None = None
            try:
                validate_code(code)
                # ponytail: exec in restricted namespace. validate_code already
                # rejected forbidden imports/builtins; we additionally strip
                # __builtins__ to a safe subset. Ceiling: in-process exec shares
                # the interpreter — a sufficiently clever payload could still
                # escape via attribute traversal. Upgrade path: Docker sandbox
                # with the same namespace, or E2B for hard isolation.
                safe_builtins = {
                    k: v
                    for k, v in __builtins__.items()
                    if k not in ("__import__", "exec", "eval", "compile", "open", "globals", "locals")
                } if isinstance(__builtins__, dict) else dict(__builtins__)
                safe_builtins["__import__"] = _safe_import
                namespace["__builtins__"] = safe_builtins

                exec(compile(code, "<code_act>", "exec"), namespace)
            except RestrictedPythonError as exc:
                error = f"RestrictedPython: {exc}"
            except Exception as exc:  # noqa: BLE001 — exec surface is unbounded
                error = f"{type(exc).__name__}: {exc}"

            stdout = stdout_buf.getvalue()
            _audit_code(agent, code, error)
            yield {
                "type": "code_executed",
                "code": code,
                "stdout": stdout,
                "error": error,
                "turn": turn,
                "risk": risk,
            }

            if error:
                error_streak += 1
                combined_feedback.append(
                    f"```python\n{code}\n```\n--- stdout ---\n{stdout}\n--- error ---\n{error}"
                )
            else:
                error_streak = 0
                combined_feedback.append(
                    f"```python\n{code}\n```\n--- stdout ---\n{stdout}"
                )

            if error_streak >= _DEGRADE_AFTER_ERRORS:
                _unregister_agent(context.session_id)
                yield {
                    "type": "code_act_degraded",
                    "reason": f"{error_streak} consecutive code errors",
                    "last_error": error,
                }
                return

        messages.append(
            ToolMessage(
                content="\n\n".join(combined_feedback),
                name="code_act_executor",
                tool_call_id=f"code_act_{turn}",
            )
        )

    # Hit the turn ceiling — emit what we have as final.
    _unregister_agent(context.session_id)
    yield {
        "type": "final",
        "content": f"[CodeAct] reached max turns ({_MAX_TURNS}). Last assistant message above.",
    }


# A tiny import whitelist for the exec namespace. Anything not here raises
# ImportError inside the exec'd code, which surfaces as a normal code error
# (counted toward the degrade threshold).
_ALLOWED_IMPORTS = frozenset(
    {
        "math",
        "statistics",
        "json",
        "re",
        "numpy",
        "pandas",
        "sympy",
        "scipy",
        "matplotlib",
        "ase",
        "pymatgen",
    }
)


def _safe_import(name: str, *args: Any, **kwargs: Any) -> Any:
    """Replacement for __import__ inside the exec namespace."""
    # AtomWorld is opt-in via HUGINN_USE_ATOMWORLD=1 — keeps flag-off behavior
    # identical even if the package happens to be installed.
    if name == "atomworld":
        if os.environ.get("HUGINN_USE_ATOMWORLD", "0") != "1":
            raise ImportError(
                "import of 'atomworld' requires HUGINN_USE_ATOMWORLD=1"
            )
        return __import__(name, *args, **kwargs)
    if name not in _ALLOWED_IMPORTS:
        raise ImportError(
            f"import of {name!r} is not allowed in CodeAct mode; "
            f"allowed: {sorted(_ALLOWED_IMPORTS)}"
        )
    return __import__(name, *args, **kwargs)
