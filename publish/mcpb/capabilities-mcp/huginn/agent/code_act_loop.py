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

import contextlib
import io
import json
import logging
import os
import re
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from huginn.core_types import ToolContext
from huginn.feature_flags import FeatureFlags
from huginn.security.code_act_sandbox import (
    _BLOCKED_TOOLS,
    make_safe_builtins,
)
from huginn.security.restricted_python import RestrictedPythonError, validate_code
from huginn.tools.registry import ToolRegistry
from huginn.utils.async_bridge import run_async

logger = logging.getLogger(__name__)


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


def _scan_called_tools(code: str, tools: dict[str, Any]) -> list[str]:
    """扫代码里调用的工具名, 返回 sorted list. _assess_risk 和 standing rule 共用."""
    called: set[str] = set()
    for name in tools:
        if name in _BLOCKED_TOOLS:
            continue
        if re.search(rf"\b{name}\s*\(", code):
            called.add(name)
    return sorted(called)


def _assess_risk(code: str, tools: dict[str, Any]) -> tuple[RiskLevel, str, list[str]]:
    """三档风险评级 (Cline 模式). 返回 (risk, reason, called_tools).

    called_tools 给 standing rule 用; high risk 返回空 list (destructive 不记 standing).
    low: 纯计算, 只调 print/math/json/numpy 等只读工具, 无副作用关键词
    medium: 调用 search/rag/literature 等只读外部工具 (有网络/IO 但不改状态)
    high: 代码含 savefig/open/write 等副作用, 或调用 destructive 工具
    """
    if _SIDE_EFFECT_PATTERNS.search(code) or _DESTRUCTIVE_TOOL_CALLS.search(code):
        return ("high", "代码含副作用关键词或 destructive 工具调用", [])
    called = _scan_called_tools(code, tools)
    for name in called:
        tool = tools.get(name)
        if tool is None:
            continue
        if getattr(tool, "destructive", False):
            return ("high", f"调用 destructive 工具 {name}", [])
    if called:
        return ("medium", f"调用工具: {sorted(called)}", called)
    return ("low", "纯计算", [])


# 本会话内 "approve_always" 的风险级别白名单, 避免确认疲劳.
# ponytail: 模块级 dict, 进程内共享. 升级: 按 session_id 隔离.
_auto_approved: dict[str, set[str]] = {}


def _mark_auto_approved(session_id: str, risk_level: RiskLevel) -> None:
    _auto_approved.setdefault(session_id, set()).add(risk_level)


def _is_auto_approved(session_id: str, risk_level: RiskLevel) -> bool:
    return risk_level in _auto_approved.get(session_id, set())


# P1-3: Standing rules by tool combination (derived from OpenWorker permissions.py:62-80).
# 比 risk_level 精细 (区分 "允许 file_write" vs "允许 bash"), 比 (tool, target) 简单
# (CodeAct 场景下 args 从代码解析太重). key = frozenset(called_tools).
# ponytail: 升级路径 — tool_call 模式接 (tool, target) 精细维度.
_standing_rules: dict[str, set[frozenset[str]]] = {}


def _mark_standing_rule(session_id: str, called_tools: list[str]) -> None:
    """approve_always 时记录当前工具组合, 下次同组合自动放行."""
    if not called_tools:
        return
    _standing_rules.setdefault(session_id, set()).add(frozenset(called_tools))


def _is_standing_rule(session_id: str, called_tools: list[str]) -> bool:
    """检查当前工具组合是否在 standing rules 里."""
    if not called_tools:
        return False
    key = frozenset(called_tools)
    return key in _standing_rules.get(session_id, set())


def reset_standing_rules(session_id: str | None = None) -> None:
    """测试用: 清空 standing rules."""
    if session_id is None:
        _standing_rules.clear()
    else:
        _standing_rules.pop(session_id, None)


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


def _record_approval(
    session_id: str, action: str, risk: str, code_preview: str = ""
) -> float:
    """记录一次 approval 决策, 更新 trust_score, 返回新值."""
    delta = _TRUST_DELTA.get(action, 0.0)
    new_score = max(0.0, min(1.0, _get_trust(session_id) + delta))
    _trust_scores[session_id] = new_score
    hist = _approval_history.setdefault(session_id, [])
    hist.append(
        {"action": action, "risk": risk, "ts": time.time(), "code": code_preview[:200]}
    )
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
    sigs = "\n".join(
        f"- {name}: {_tool_signature(name, t)}" for name, t in tools.items()
    )
    prompt = f"""You are Huginn, a research agent running in CodeAct mode.

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
    prompt += _atomworld_prompt_suffix()
    # Structure Cognitive Map — explicit 3D coords + adjacency + SE(3) equivariant
    # queries, opt-in via HUGINN_USE_COGNITIVE_MAP=1. Returns map_id handles so
    # the agent doesn't dump large structures into the conversation directly.
    prompt += _cognitive_map_prompt_suffix()
    return prompt


def _atomworld_prompt_suffix() -> str:
    if not FeatureFlags.shared().is_enabled("use_atomworld"):
        return ""
    return (
        "\n\nAtomWorld benchmark tools (plain Python functions in this "
        "namespace): atomworld_evaluate(target_cif, generated_output), "
        "atomworld_apply_action(input_cif, action_name, **params), "
        "atomworld_list_actions(). You can also `from atomworld import "
        "evaluate` for the raw upstream API."
    )


def _cognitive_map_prompt_suffix() -> str:
    if not FeatureFlags.shared().is_enabled("use_cognitive_map"):
        return ""
    return (
        "\n\nStructure Cognitive Map tools (3D spatial reasoning, plain "
        "Python functions): cognitive_map_from_cif(cif_str) -> map_id, "
        "cognitive_map_query(map_id, query_type, **params) -> dict "
        "(query_type in distance/angle/neighbors/subgraph/after_rotation/"
        "after_translation), cognitive_map_transform(map_id, op, **params) "
        "-> new_map_id (op in rotate/translate/supercell/remove_atom/"
        "add_atom/swap_atoms), cognitive_map_to_text(map_id, max_atoms=50) "
        "-> str. SE(3) equivariant: rotations preserve distances/angles. "
        "cognitive_map_se3_act(map_id, euler_angles=[rx,ry,rz], "
        "translation=[tx,ty,tz], order='XYZ', degrees=True) -> dict "
        "{text, coords, species, n_atoms} applies SE(3) group action "
        "simultaneously to text <point3d> and 3D coords (semidirect "
        "product (M×V)⋊SE(3) engineering entry)."
    )


def _install_standard_modules(namespace: dict[str, Any]) -> None:
    """Optional scientific stack — only if the user has them installed."""
    for mod_name in ("math", "statistics", "numpy", "pandas", "sympy"):
        with contextlib.suppress(ImportError):
            namespace[mod_name] = __import__(mod_name)


def _install_atomworld(namespace: dict[str, Any]) -> None:
    """AtomWorld benchmark — opt-in via HUGINN_USE_ATOMWORLD=1 (mirrors
    BranchIncubator gating). atomworld_tool already no-ops when the
    atomworld package isn't installed, so we just log and skip."""
    if FeatureFlags.shared().is_enabled("use_atomworld"):
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


def _install_cognitive_map(namespace: dict[str, Any]) -> None:
    """Structure Cognitive Map — opt-in via HUGINN_USE_COGNITIVE_MAP=1.
    SE(3) equivariant 3D spatial queries for rotation-skill recovery."""
    if FeatureFlags.shared().is_enabled("use_cognitive_map"):
        try:
            from huginn.tools import structure_cognitive_map_tool as _cm

            if _cm.is_available():
                namespace["cognitive_map_from_cif"] = _cm.cognitive_map_from_cif
                namespace["cognitive_map_query"] = _cm.cognitive_map_query
                namespace["cognitive_map_transform"] = _cm.cognitive_map_transform
                namespace["cognitive_map_to_text"] = _cm.cognitive_map_to_text
                namespace["cognitive_map_se3_act"] = _cm.cognitive_map_se3_act
            else:
                logger.warning(
                    "structure_cognitive_map_tool: pymatgen/scipy not installed, "
                    "skipping CodeAct registration"
                )
        except ImportError:
            logger.warning(
                "structure_cognitive_map_tool: module import failed, "
                "skipping CodeAct registration"
            )


def _install_tool_wrappers(
    namespace: dict[str, Any], tools: dict[str, Any], context: ToolContext
) -> None:
    """Tool wrappers — sync facade over async tool.call via run_async bridge."""
    for name, tool in tools.items():
        if name in _BLOCKED_TOOLS:
            continue
        if not tool.active or not tool.is_available():
            continue
        namespace[name] = _make_tool_wrapper(tool, context, name)


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

    _install_standard_modules(namespace)
    _install_atomworld(namespace)
    _install_cognitive_map(namespace)
    _install_tool_wrappers(namespace, tools, context)

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
        except Exception:  # 防御: 审计状态装配失败则放弃本轮
            logger.debug("best-effort op failed", exc_info=True)
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
    except Exception:  # 非致命: code_act 审计上报失败则忽略
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
    session = _init_code_act_session(agent, message, thread_id)
    error_streak = 0
    for turn in range(_MAX_TURNS):
        result = _TurnResult(error_streak=error_streak)
        async for event in _run_one_code_act_turn(
            session, agent, turn, error_streak, result
        ):
            yield event
        error_streak = result.error_streak
        if result.terminal_content is not None:
            _unregister_agent(session.context.session_id)
            yield {"type": "final", "content": result.terminal_content}
            return
        if result.degraded_reason:
            _unregister_agent(session.context.session_id)
            yield {
                "type": "code_act_degraded",
                "reason": result.degraded_reason,
                "last_error": result.last_error,
            }
            return

    # Hit the turn ceiling — emit what we have as final.
    _unregister_agent(session.context.session_id)
    yield {
        "type": "final",
        "content": f"[CodeAct] reached max turns ({_MAX_TURNS}). Last assistant message above.",
    }


@dataclass
class _CodeActSession:
    """一轮 CodeAct 会话的稳定上下文 (tools / workspace / messages / model)."""

    tools: dict[str, Any]
    context: ToolContext
    messages: list[Any]
    model: Any


@dataclass
class _BlockState:
    """单个 code block 在审批→执行链路上的可变产物."""

    code: str
    risk: str = "low"
    risk_reason: str = ""
    called: Any = None
    stdout: str = ""
    error: str | None = None
    feedback: str = ""
    denied: bool = False
    degraded_reason: str = ""
    suggest_skip: bool = False
    decision: ApprovalDecision | None = None


@dataclass
class _TurnResult:
    """单轮结束时的去向: terminal content / degrade / 累计 error streak."""

    error_streak: int = 0
    terminal_content: str | None = None
    degraded_reason: str = ""
    last_error: str | None = None


def _init_code_act_session(agent: Any, message: str, thread_id: str) -> _CodeActSession:
    """注册 agent + 装配工具面 / ToolContext / 初始 messages / model."""
    import asyncio

    from langchain_core.messages import HumanMessage, SystemMessage

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
        audit_logger=getattr(
            getattr(agent, "session_state", None), "audit_logger", None
        ),
    )

    system_prompt = _build_system_prompt(tools)
    messages: list[Any] = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=message),
    ]

    model = (
        agent.select_model("agent") if hasattr(agent, "select_model") else agent.model
    )

    # 注册 agent 供 SUGGEST mode WS handler 唤醒
    if not hasattr(agent, "_approval_resume"):
        agent._approval_resume = asyncio.Event()
    _register_agent(context.session_id, agent)
    return _CodeActSession(
        tools=tools,
        context=context,
        messages=messages,
        model=model,
    )


def _extract_response_text(resp: Any) -> str:
    if not isinstance(resp.content, list):
        return str(resp.content)
    return "".join(
        part.get("text", "") if isinstance(part, dict) else str(part)
        for part in resp.content
    )


async def _run_one_code_act_turn(
    session: _CodeActSession,
    agent: Any,
    turn: int,
    error_streak: int,
    result: _TurnResult,
) -> AsyncIterator[dict[str, Any]]:
    """模型调用 → 文本事件 → 逐个 code block 审批/执行 → 回填 ToolMessage."""
    from langchain_core.messages import AIMessage, ToolMessage

    try:
        resp = await session.model.ainvoke(session.messages)
    except Exception as exc:
        result.terminal_content = f"[CodeAct] model call failed: {exc}"
        return

    content = _extract_response_text(resp)
    session.messages.append(AIMessage(content=content))
    yield {"type": "assistant_text", "content": content, "turn": turn}

    code_blocks = _extract_python_blocks(content)
    if not code_blocks:
        # No code → terminal answer
        result.terminal_content = content
        return

    # Exec each block in order; feed results back as a single ToolMessage.
    combined_feedback: list[str] = []
    async for event in _run_code_blocks(
        session,
        agent,
        code_blocks,
        turn,
        error_streak,
        result,
        combined_feedback,
    ):
        yield event

    if result.terminal_content is not None or result.degraded_reason:
        return

    session.messages.append(
        ToolMessage(
            content="\n\n".join(combined_feedback),
            name="code_act_executor",
            tool_call_id=f"code_act_{turn}",
        )
    )


async def _run_code_blocks(
    session: _CodeActSession,
    agent: Any,
    code_blocks: list[str],
    turn: int,
    error_streak: int,
    result: _TurnResult,
    combined_feedback: list[str],
) -> AsyncIterator[dict[str, Any]]:
    """顺序处理本轮所有 code block, 累计 error streak, 必要时标记 degrade."""
    for code in code_blocks:
        state = _BlockState(code=code)
        async for event in _process_code_block(
            session, agent, state, turn, error_streak
        ):
            yield event

        if state.denied:
            error_streak += 1
            combined_feedback.append(state.feedback)
            if state.degraded_reason:
                result.degraded_reason = state.degraded_reason
                result.last_error = state.error
                result.error_streak = error_streak
                return
            continue

        if state.error:
            error_streak += 1
        else:
            error_streak = 0
        combined_feedback.append(state.feedback)

        if error_streak >= _DEGRADE_AFTER_ERRORS:
            result.degraded_reason = f"{error_streak} consecutive code errors"
            result.last_error = state.error
            result.error_streak = error_streak
            return

    result.error_streak = error_streak


async def _process_code_block(
    session: _CodeActSession,
    agent: Any,
    state: _BlockState,
    turn: int,
    error_streak: int,
) -> AsyncIterator[dict[str, Any]]:
    """单个 code block 的审批门禁 + 执行 (结果写回 state)."""
    yield _assess_block_risk(session, state, error_streak)

    # ── SUGGEST mode (HRI #4: LoA Level 4-6, 强制所有代码先经用户编辑) ──
    # SUGGEST 批准后设 suggest_skip=True, 跳过下面的 trust/budget gate
    async for event in _run_suggest_mode(session, agent, state, turn):
        yield event
    if state.denied:
        return

    # ── Trust + Budget adaptive threshold ──
    skip_approval, budget_escalated, budget = _evaluate_skip_approval(session, state)

    # standing rule 命中时 yield 事件 (借鉴 OpenWorker Decision.rule 的可观测性)
    session_id = session.context.session_id
    if (
        not state.suggest_skip
        and state.called
        and _is_standing_rule(session_id, state.called)
    ):
        yield {"type": "standing_rule_hit", "tools": state.called}

    if budget_escalated:
        yield {
            "type": "budget_escalation",
            "remaining": budget,
            "risk": state.risk,
        }

    if not skip_approval:
        async for event in _run_approval_gate(
            session, agent, state, turn, error_streak
        ):
            yield event
        if state.denied:
            return

    async for event in _execute_code_block_events(session, agent, state, turn):
        yield event


def _assess_block_risk(
    session: _CodeActSession,
    state: _BlockState,
    error_streak: int,
) -> dict[str, Any]:
    """静态风险 + 动态阈值调整; 返回 risk_threshold 事件并写回 state."""
    risk, risk_reason, called = _assess_risk(state.code, session.tools)

    # ── Dynamic risk threshold (HRI #5: trust-adaptive classification) ──
    threshold = _compute_risk_threshold(session.context.session_id, error_streak)
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
    state.risk = risk
    state.risk_reason = risk_reason
    state.called = called
    return threshold_evt


async def _run_suggest_mode(
    session: _CodeActSession,
    agent: Any,
    state: _BlockState,
    turn: int,
) -> AsyncIterator[dict[str, Any]]:
    """SUGGEST 模式: 强制用户编辑/确认后才放行, 批准后置 suggest_skip=True."""
    session_id = session.context.session_id
    if not _is_suggest_mode(session_id):
        return

    resume_event = agent._approval_resume
    resume_event.clear()
    yield {
        "type": "suggest_code",
        "code": state.code,
        "risk": state.risk,
        "reason": state.risk_reason,
        "turn": turn,
    }
    await resume_event.wait()
    resume_decision = getattr(agent, "_approval_decision", None)
    agent._approval_decision = None
    action = resume_decision[0] if resume_decision else "deny"
    _nt = _record_approval(session_id, action, state.risk, state.code[:200])
    yield {
        "type": "trust_update",
        "trust": _nt,
        "action": action,
        "risk": state.risk,
    }
    if resume_decision is None or action == "deny":
        err = (
            f"用户拒绝 (suggest, risk={state.risk}): "
            f"{resume_decision[1] if resume_decision else 'no decision'}"
        )
        state.denied = True
        state.error = err
        state.feedback = f"```python\n{state.code}\n```\n--- DENIED ---\n{err}"
        return
    _, payload = resume_decision
    _apply_approval_action(session_id, state, action, payload)
    state.suggest_skip = True


def _evaluate_skip_approval(
    session: _CodeActSession,
    state: _BlockState,
) -> tuple[bool, bool, int]:
    """信任度/预算自适应判定; 返回 (skip_approval, budget_escalated, budget)."""
    session_id = session.context.session_id
    _get_trust(session_id)
    budget = _get_budget(session_id)
    # budget 不足 + 信任度不低 → 自动升级 (避免确认疲劳)
    budget_escalated = (
        not state.suggest_skip
        and state.risk != "low"
        and budget <= _BUDGET_ESCALATION_THRESHOLD
        and not _should_force_ask(session_id)
        and not _is_auto_approved(session_id, state.risk)
    )
    if budget_escalated:
        _mark_auto_approved(session_id, state.risk)

    skip_approval = state.suggest_skip or (
        state.risk == "low"
        or _should_auto_medium(session_id, state.risk)
        or budget_escalated
        or (
            _is_auto_approved(session_id, state.risk)
            and not _should_force_ask(session_id)
        )
        or _is_standing_rule(session_id, state.called)
    )
    return skip_approval, budget_escalated, budget


async def _run_approval_gate(
    session: _CodeActSession,
    agent: Any,
    state: _BlockState,
    turn: int,
    error_streak: int,
) -> AsyncIterator[dict[str, Any]]:
    """未 skip 时的审批门: 回调直调 / inbox 跨会话队列 / resume event."""
    session_id = session.context.session_id
    new_budget = _decrement_budget(session_id)
    yield {"type": "budget_update", "remaining": new_budget}

    approval_fn: ApprovalFn | None = getattr(agent, "approval_fn", None)
    if approval_fn is not None:
        # 调用方注入了同步/异步审批回调, 直接等结果
        await _approval_via_callback(
            agent, session_id, state, approval_fn, error_streak
        )
        return

    async for event in _approval_via_inbox(session, agent, state, turn):
        yield event

    if state.decision is None:
        # 无 resume 机制 → 保守拒绝
        err = f"approval required (risk={state.risk}) but no resume mechanism"
        state.denied = True
        state.error = err
        state.feedback = f"```python\n{state.code}\n```\n--- DENIED ---\n{err}"
        return

    action, payload = state.decision
    _nt = _record_approval(session_id, action, state.risk, state.code[:200])
    yield {"type": "trust_update", "trust": _nt, "action": action, "risk": state.risk}
    if action == "deny":
        err = f"用户拒绝 (risk={state.risk}): {payload or state.risk_reason}"
        state.denied = True
        state.error = err
        state.feedback = f"```python\n{state.code}\n```\n--- DENIED ---\n{err}"
        return
    _apply_approval_action(session_id, state, action, payload)


async def _approval_via_callback(
    agent: Any,
    session_id: str,
    state: _BlockState,
    approval_fn: ApprovalFn,
    error_streak: int,
) -> None:
    action, payload = await approval_fn(state.code, state.risk, state.risk_reason)
    if action == "deny":
        err = f"用户拒绝执行 (risk={state.risk}): {payload or state.risk_reason}"
        _audit_code(agent, state.code, err)
        state.denied = True
        state.error = err
        state.feedback = f"```python\n{state.code}\n```\n--- DENIED ---\n{err}"
        if error_streak + 1 >= _DEGRADE_AFTER_ERRORS:
            state.degraded_reason = f"{error_streak + 1} consecutive errors/denials"
        return
    _apply_approval_action(session_id, state, action, payload)


async def _approval_via_inbox(
    session: _CodeActSession,
    agent: Any,
    state: _BlockState,
    turn: int,
) -> AsyncIterator[dict[str, Any]]:
    """无回调时: 优先 unattended inbox 队列, 其次 resume event."""
    # ponytail: 简化版 LangGraph interrupt. 调用方拿到事件后
    # 决定 approve/deny/edit, 通过 asyncio.Event 或类似机制 resume.
    # 这里用 yield + 等待 agent._approval_resume event 的模式.
    resume_event = getattr(agent, "_approval_resume", None)

    # P2-5 自动 fallback: 无 resume_event (调用方没接 approval 信号)
    # 时, 自动转 inbox 跨会话队列. attended session (有 WS handler)
    # 会设 _approval_resume, unattended 不会 → 自动走 inbox.
    if resume_event is None and os.environ.get("HUGINN_APPROVAL_MODE", "") != "disable":
        try:
            # 用 code 的 hash 做 tool_call_id (CodeAct 没有原生
            # tool_call_id), 让同 code 重试时复用 inbox item
            import hashlib as _hl

            from huginn.interaction.inbox import (
                get_inbox_store,
                inbox_approval_fn,
            )

            tc_id = f"codeact_{_hl.sha256(state.code.encode()).hexdigest()[:16]}"
            inbox_fn = inbox_approval_fn(
                get_inbox_store(),
                session.context.session_id,
                tool_call_id=tc_id,
            )
            yield {
                "type": "approval_request",
                "code": state.code,
                "risk": state.risk,
                "reason": state.risk_reason,
                "turn": turn,
                "routed_to": "inbox",
            }
            action, payload = await inbox_fn(state.code, state.risk, state.risk_reason)
            state.decision = (action, payload)
        except Exception:  # 防御: inbox 失效则保守拒绝
            logger.debug("inbox fallback failed, conservative deny", exc_info=True)
            state.decision = None

    if state.decision is None and resume_event is not None:
        async for event in _approval_via_resume(agent, state, turn, resume_event):
            yield event


async def _approval_via_resume(
    agent: Any,
    state: _BlockState,
    turn: int,
    resume_event: Any,
) -> AsyncIterator[dict[str, Any]]:
    """attended session: yield approval_request 后等调用方 set event."""
    yield {
        "type": "approval_request",
        "code": state.code,
        "risk": state.risk,
        "reason": state.risk_reason,
        "turn": turn,
    }
    # 调用方 set event 后把决策放进 agent._approval_decision
    await resume_event.wait()
    state.decision = getattr(agent, "_approval_decision", None)
    resume_event.clear()


def _apply_approval_action(
    session_id: str,
    state: _BlockState,
    action: str,
    payload: str | None,
) -> None:
    """approve_always / edit 的共享副作用."""
    if action == "approve_always":
        _mark_auto_approved(session_id, state.risk)
        _mark_standing_rule(session_id, state.called)
    if action == "edit" and payload:
        state.code = payload


def _execute_code_block(
    session: _CodeActSession,
    agent: Any,
    code: str,
) -> tuple[str, str | None]:
    """在受限命名空间内 exec 单块代码, 返回 (stdout, error)."""
    stdout_buf = io.StringIO()
    namespace = _build_namespace(agent, session.tools, session.context, stdout_buf)
    error: str | None = None
    try:
        validate_code(code)
        # ponytail: exec in restricted namespace. validate_code already
        # rejected forbidden imports/builtins; make_safe_builtins (from
        # code_act_sandbox) strips dangerous builtins + swaps __import__.
        # Ceiling: in-process exec shares the interpreter — a sufficiently
        # clever payload could still escape via attribute traversal.
        # Upgrade path: Docker sandbox with the same namespace, or E2B.
        namespace["__builtins__"] = make_safe_builtins()
        # P4-6: exec 用 wrapper 加 tracemalloc 内存峰值监控
        # 超阈值抛 MemoryError 而不是让进程 OOM 被杀. 阈值从 env 读, 默认 2GB.
        # ponytail: tracemalloc 有 ~10% overhead, 但比 OOM 安全.
        from huginn.security.code_act_sandbox import (
            exec_with_mem_cap as _exec_cap,
        )

        _mem_cap_bytes = int(os.environ.get("HUGINN_CODEACT_MEM_CAP", "2147483648"))

        _exec_cap(code, namespace, _mem_cap_bytes)
    except RestrictedPythonError as exc:
        error = f"RestrictedPython: {exc}"
    except Exception as exc:  # noqa: BLE001 — exec surface is unbounded
        error = f"{type(exc).__name__}: {exc}"
    return stdout_buf.getvalue(), error


async def _execute_code_block_events(
    session: _CodeActSession,
    agent: Any,
    state: _BlockState,
    turn: int,
) -> AsyncIterator[dict[str, Any]]:
    """执行并 yield code_executed 事件, 同时构造回喂给模型的 feedback."""
    stdout, error = _execute_code_block(session, agent, state.code)
    state.stdout = stdout
    state.error = error
    _audit_code(agent, state.code, error)
    yield {
        "type": "code_executed",
        "code": state.code,
        "stdout": stdout,
        "error": error,
        "turn": turn,
        "risk": state.risk,
    }
    if error:
        state.feedback = f"```python\n{state.code}\n```\n--- stdout ---\n{stdout}\n--- error ---\n{error}"
    else:
        state.feedback = f"```python\n{state.code}\n```\n--- stdout ---\n{stdout}"
