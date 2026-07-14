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
import re
from datetime import datetime
from typing import Any, AsyncIterator

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
# M3ToolEval; 15 leaves headroom for exploration without runaway cost.
_MAX_TURNS = 15
_DEGRADE_AFTER_ERRORS = 3

_CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


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
    return f"""You are Huginn, a materials science agent running in CodeAct mode.

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

    error_streak = 0
    for turn in range(_MAX_TURNS):
        try:
            resp = await model.ainvoke(messages)
        except Exception as exc:
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
            yield {"type": "final", "content": content}
            return

        # Exec each block in order; feed results back as a single ToolMessage.
        combined_feedback: list[str] = []
        for code in code_blocks:
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
    if name not in _ALLOWED_IMPORTS:
        raise ImportError(
            f"import of {name!r} is not allowed in CodeAct mode; "
            f"allowed: {sorted(_ALLOWED_IMPORTS)}"
        )
    return __import__(name, *args, **kwargs)
