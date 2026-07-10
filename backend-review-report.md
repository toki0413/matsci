# Huginn Backend Code Review Report

**Review Date**: 2026-07-10  
**Reviewer**: python-backend-reviewer skill  
**Codebase**: `C:\Users\wanzh\Desktop\matsci-agent\agent\huginn\`  
**Scope**: agent.py, routes/ws.py, routes/agents.py, routes/provenance.py, routes/event_stream.py, prompts.py, schemas.py, server.py, context_builder.py, tools/, skills/

---

## Executive Summary

The Huginn backend is a large-scale computational materials science agent platform (~3145 lines in agent.py alone, 116 tool files, 50+ route handlers). The architecture is feature-rich but suffers from several systemic issues:

- **365 bare `except Exception:` clauses** across 100+ files, silently swallowing errors
- **agent.py is a 3145-line god class** with excessive constructor parameters (~40 kwargs)
- **Module-level mutable global state** shared across concurrent WebSocket sessions without per-connection isolation
- **Unsanitized `eval()`** in numerical optimization tool
- **54 re-export shim files** adding import indirection without functional value
- **Deprecated `asyncio.get_event_loop()`** usage in WebSocket handlers
- **REST routes returning 200 OK with error bodies** instead of proper HTTP status codes

The codebase shows strong domain expertise (materials science, simulation workflows) but would benefit significantly from error handling discipline, state isolation, and splitting the god class.

---

## CRITICAL Issues

### C1. Unsanitized `eval()` in numerical optimization tool

**File**: `tools/sci/numerical_tool.py:749, 763`  
**Severity**: CRITICAL  
**Category**: Security - Code Injection

```python
# Line 749
objective_expr = eval(obj_line, eval_globals, eval_locals)

# Line 763
c_expr = eval(cstr, eval_globals, eval_locals)
```

User-provided objective and constraint strings are passed directly to Python's `eval()`. The `eval_globals` dict includes `cp` (cvxpy), `sum`, `norm`, `abs` etc., but `eval()` allows arbitrary expression evaluation including attribute access chains (`().__class__.__bases__[0].__subclasses__()`) that can escape the intended sandbox. An LLM-generated or user-crafted string like `__import__('os').system('rm -rf /')` would execute.

The codebase already has a `huginn.security.safe_eval` module with restricted expression evaluation. This tool should use `safe_eval` instead of raw `eval()`.

**Recommendation**: Replace `eval()` with `safe_eval()` from `huginn.security.safe_eval`, or use `ast.literal_eval` + a restricted expression parser.

---

### C2. `shell=True` with curl pipe to sh in Lean installation

**File**: `bourbaki_env.py:123-125`  
**Severity**: CRITICAL  
**Category**: Security - Command Injection

```python
result = subprocess.run(
    "curl --proto '=https' --tlsv1.2 -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y",
    shell=True, capture_output=True, text=True, timeout=300,
)
```

This pipes a remote script directly into `sh`. If the URL is compromised, DNS is spoofed, or TLS is intercepted, arbitrary code executes on the host. The `shell=True` is required for the pipe, but the pattern itself is dangerous. The `scheduler.py:124` comment acknowledges this: "shell=True has command injection risk on Windows."

**Recommendation**: Download the script to a temp file first, inspect its hash against a known checksum, then execute with `subprocess.run(["sh", script_path, "-y"])` without `shell=True`.

---

### C3. Module-level mutable global state shared across concurrent requests

**File**: `server_core.py:54-58`, `routes/ws.py:34, 43-44`  
**Severity**: CRITICAL  
**Category**: Concurrency - Shared State Mutation

```python
# server_core.py
_context: ServerContext | None = None
_checkpoints: dict[str, tuple[Path, dict[str, str]]] = {}
_threads: dict[str, dict[str, Any]] = {}

# routes/ws.py
_pending_tasks: set[asyncio.Task] = set()
_pending_plans: dict[str, asyncio.Future] = {}
_pending_approvals: dict[str, asyncio.Future] = {}
```

These module-level dicts are shared across all concurrent WebSocket connections. While `_state_lock` (RLock) protects `_checkpoints` and `_threads`, the `_pending_plans` and `_pending_approvals` dicts in `ws.py` have **no synchronization**. Two concurrent connections could collide on `plan_id` keys (generated as `uuid.uuid4().hex[:8]` - only 8 hex chars = ~4 billion space, birthday paradox at ~65K entries).

More critically, `_pending_plans` is used by the `send_plan_and_wait()` function (line 150) which is a module-level function, but plan contexts are stored per-connection in `_pending_plan_contexts` (line 603). The split between module-level and per-connection state is inconsistent and error-prone.

**Recommendation**: Move `_pending_plans` and `_pending_approvals` into per-connection state (already done for `_pending_plan_contexts` at line 603). Apply the same pattern to the other two dicts.

---

## HIGH Issues

### H1. 365 bare `except Exception:` clauses silently swallowing errors

**Files**: 100+ files across the entire codebase  
**Severity**: HIGH  
**Category**: Error Handling - Error Swallowing

Quantitative breakdown of the worst offenders:

| File | Count |
|------|-------|
| `agent.py` | 45 |
| `autoloop/engine.py` | 49 |
| `execution/physics_auditor.py` | 10 |
| `server_core.py` | 10 |
| `execution/kernel_session.py` | 9 |
| `autoloop/conjecture.py` | 7 |
| `hooks/anomaly_llm_hook.py` | 7 |
| `knowledge/auto_ingest.py` | 8 |
| `memory/manager.py` | 8 |
| `perception/pdf_parser.py` | 6 |
| `perception/visual_encoder.py` | 6 |
| `context_builder.py` | 6 |
| `mcp_client.py` | 6 |
| `knowledge/smart_ingest.py` | 6 |
| `knowledge/store.py` | 6 |

The dominant pattern is:

```python
try:
    from huginn.some_module import some_function
    some_function()
except Exception:
    logger.debug("... failed", exc_info=True)
```

This silently degrades functionality without alerting operators. In many cases (e.g., `agent.py:703`, `agent.py:784`, `agent.py:799`), model router failures are caught and logged at WARNING but execution continues with a potentially broken state.

Additionally, `routes/ws.py:1031` has an anti-pattern: `except (Exception,):` - a tuple with a single element, functionally identical to `except Exception:` but syntactically confusing.

**Recommendation**: 
1. Replace blanket `except Exception:` with specific exception types where the failure mode is known.
2. For optional imports, use `except ImportError:` specifically (not `Exception`).
3. For truly unexpected errors, at minimum log at `ERROR` level, not `DEBUG`.

---

### H2. agent.py is a 3145-line god class with ~40 constructor parameters

**File**: `agent.py` (3145 lines)  
**Severity**: HIGH  
**Category**: Over-complexity - God Class

The `HuginnAgent` class has:
- **~40 constructor kwargs** (lines 260-304), many using `_UNSET_SENTINEL` sentinel pattern
- **`__init__` + `_init_from_config`** together span ~400 lines (lines 260-675) just for initialization
- The `chat()` method spans ~770 lines (lines 2234-3001), exceeding the 300-line hard limit by 2.5x
- The `_process_stream_state` method is ~100 lines with 5 levels of nesting
- The `_effective_system_prompt` method is ~80 lines with 4 try/except blocks

While some delegation has been done (to `ContextBuilder`, `ToolAdapter`, `PromptCacheBuilder`), the agent still directly manages:
- Phase state machine
- Cognitive state machine
- Conversation tree
- Memory decay
- Telemetry
- Privacy guard
- Style learner
- Evolution engine
- Pet event bus
- Interrupt manager
- Thought loop detector
- Tool call budget/router/loop detector
- Session state

**Recommendation**: 
1. Extract `chat()` into a `ChatOrchestrator` class that takes the agent as a dependency.
2. Move phase/cognitive state machine management to a dedicated `WorkflowManager`.
3. Move the retry/fallback logic (lines 2588-2750) to `llm_retry.py` where it belongs.

---

### H3. routes/agents.py returns HTTP 200 with error bodies instead of proper status codes

**File**: `routes/agents.py` (20+ occurrences)  
**Severity**: HIGH  
**Category**: API Design - Error Handling

Every endpoint in `routes/agents.py` follows this anti-pattern:

```python
@router.get("/personas/{name}")
async def get_persona(name: str) -> dict[str, Any]:
    try:
        ...
        return {"success": True, "name": p.name, ...}
    except Exception as e:
        return {"success": False, "error": str(e)}  # Returns HTTP 200!
```

This means:
1. Clients cannot distinguish success from failure via HTTP status codes
2. `str(e)` leaks internal exception messages to the API consumer (information disclosure)
3. API clients must parse JSON to check `"success"` instead of checking status codes
4. No structured error codes for programmatic handling

The `chat_with_agent` endpoint (line 68) at least validates input with `ChatRequest`, but still returns 200 on timeout (line 209) and 200 on unexpected exceptions (line 225).

**Recommendation**: 
1. Raise `HTTPException(status_code=404, detail=...)` for not-found errors.
2. Return 500 for server errors, 422 for validation, 404 for not-found.
3. Use the `huginn_error_response` helper from `huginn.errors` (already used in `server.py:135`).

---

### H4. routes/ws.py WebSocket handler is 1371 lines with deeply nested logic

**File**: `routes/ws.py` (1371 lines)  
**Severity**: HIGH  
**Category**: Over-complexity - Function Length

The `agent_websocket` function (line 566) is a single async function spanning ~800 lines. It contains:
- Authentication (lines 577-584)
- Heartbeat setup (lines 605-629)
- Pet event forwarding (lines 631-662)
- Main message loop with 6 message type handlers (lines 664-1371):
  - `user_input` handler: ~400 lines with plan mode, team mode, research mode, RAG, persona routing
  - `explore_start` handler: ~80 lines
  - `approval_response` handler: ~10 lines
  - `plan_confirm` handler: ~60 lines
  - `clarification_response` handler: ~20 lines
  - `set_auto_approve` handler: ~10 lines

The `user_input` branch alone has 5 levels of nesting and interleaves routing decisions (`@agent`, `/team`, `/plan`, `/research`) with inline business logic.

**Recommendation**: 
1. Extract each message type handler into its own async function.
2. Extract the mode detection logic (plan/team/research) into a `MessageRouter` class.
3. Move the `_stream_agent_response` helper (already extracted at line 243) and apply the same pattern to the rest.

---

### H5. Duplicate function definitions across modules

**Files**: Multiple  
**Severity**: HIGH  
**Category**: Code Duplication

**5a. `_wants_dict()` duplicated:**
- `tools/adapter.py:72` 
- `skills/base.py:20`

Both implementations are nearly identical:
```python
# tools/adapter.py
def _wants_dict(tool: HuginnTool) -> bool:
    try:
        hints = typing.get_type_hints(tool.call)
    except Exception:
        hints = {}
    ...

# skills/base.py
def _wants_dict(tool: Any) -> bool:
    try:
        hints = typing.get_type_hints(tool.call)
    except Exception:
        return False
    ...
```

**5b. `_extract_usage()` duplicated:**
- `agent.py:175` (method on `RateLimitMiddleware`)
- `security/rate_limiter.py:378` (module-level function)

The agent method just delegates:
```python
def _extract_usage(self, result: Any) -> tuple[int, int]:
    from huginn.security.rate_limiter import _extract_usage as _extract
    return _extract(result)
```
This thin wrapper adds an import inside a method (called on every LLM invocation) without adding value.

**5c. Repeated `import json as _json` pattern:**
Found in 6 locations:
- `agent.py:1730`
- `routes/config.py:234, 459`
- `routes/ws.py:1027, 1291`
- `tools/code_tool.py:230`

Each re-imports `json` with a local alias inside a function body, despite `json` already being imported at the module level (or trivially importable). This suggests copy-paste evolution.

**Recommendation**: 
1. Move `_wants_dict` to `huginn.utils.typing` and import from there.
2. Remove the `RateLimitMiddleware._extract_usage` wrapper and call `_extract_usage` directly.
3. Use the top-level `json` import everywhere; remove `import json as _json`.

---

### H6. routes/provenance.py lacks input validation and bounds checking

**File**: `routes/provenance.py`  
**Severity**: HIGH  
**Category**: Security - Input Validation

```python
@router.get("/recent")
async def recent(n: int = 20):  # No upper bound - n=999999 returns everything

@router.get("/search")
async def search(q: str):  # No length limit on query string

@router.get("/lineage")
async def lineage(path: str, depth: int = 5):  # path is arbitrary, depth unbounded

@router.delete("/cleanup")
async def cleanup(days: int = 30):  # days could be negative or enormous
```

- `n` has no upper bound - a client could request `n=1000000`, causing OOM or excessive DB load
- `q` has no length constraint - could be used for ReDoS if the search uses regex
- `path` is passed directly to `get_lineage()` - potential path traversal if the registry uses it in file operations
- `depth` has no upper bound - could cause deep recursion
- `days` accepts negative values

Unlike `routes/schemas.py` which defines proper Pydantic validators with `max_length` and `pattern` constraints, the provenance endpoints use raw FastAPI query parameters with no validation.

**Recommendation**: 
1. Add `Query(max=500)` to `n` parameter.
2. Add `Query(max_length=500)` to `q` and `path`.
3. Add `Query(ge=1, le=50)` to `depth`.
4. Add `Query(ge=0, le=3650)` to `days`.
5. Sanitize `path` to reject `..` traversal sequences.

---

### H7. Deprecated `asyncio.get_event_loop()` usage

**File**: `routes/ws.py:148, 206, 610, 617, 667`  
**Severity**: HIGH  
**Category**: Compatibility - Deprecated API

```python
# ws.py:148
loop = asyncio.get_event_loop()
future = loop.create_future()

# ws.py:610
last_recv: dict[str, float] = {"t": asyncio.get_event_loop().time()}
```

`asyncio.get_event_loop()` is deprecated since Python 3.10 and emits a `DeprecationWarning`. In Python 3.12+, calling it without a running event loop raises `RuntimeError`. Inside async code (which all these call sites are), `asyncio.get_running_loop()` should be used instead.

Also found in `tools/wetlab_rpc_tool.py:588, 603`.

**Recommendation**: Replace all `asyncio.get_event_loop()` with `asyncio.get_running_loop()` in async contexts, or `asyncio.get_event_loop_policy().get_event_loop()` for the sync fallback case.

---

## MEDIUM Issues

### M1. 54 re-export shim files in tools/ creating import indirection

**Files**: `tools/vasp_tool.py`, `tools/comsol_tool.py`, `tools/abaqus_tool.py`, etc. (54 files)  
**Severity**: MEDIUM  
**Category**: Dead Code / Import Issues

54 of 116 tool files are shims that just re-export from subpackages:

```python
# tools/vasp_tool.py
"""shim: file moved to huginn.tools.sim.vasp_tool."""
from huginn.tools.sim.vasp_tool import VaspTool, VaspToolInput, VaspToolOutput
__all__ = ["VaspTool", "VaspToolInput", "VaspToolOutput"]
```

While this maintains backward compatibility, it:
1. Doubles the effective file count
2. Adds import latency (each shim is a separate module load)
3. Creates confusion about where code actually lives
4. Makes refactoring harder (two import paths to update)

**Recommendation**: If no external code imports from the old paths, remove the shims. If backward compatibility is needed, add a deprecation warning and a timeline for removal.

---

### M2. ToolRegistry uses class-level mutable state as a singleton

**File**: `tools/registry.py:20-21`  
**Severity**: MEDIUM  
**Category**: Design - Global Mutable State

```python
class ToolRegistry:
    _tools: dict[str, HuginnTool] = {}
    _schemas_cache: list[dict] | None = None
```

Class-level mutable dicts are shared across all instances, all tests, and all concurrent requests. The `clear()` method (line 123) wipes all tools globally, which is dangerous if called during runtime. The cache invalidation (`_schemas_cache = None`) happens on every `register()` call, but the cache is only rebuilt on `get_all_schemas()`, creating a window where stale schemas could be served.

**Recommendation**: If a singleton is truly needed, use a proper singleton pattern with `__new__` or a module-level instance. Otherwise, pass `ToolRegistry` instances as dependencies.

---

### M3. Inconsistent error response format across routes

**Files**: `routes/agents.py`, `routes/provenance.py`, `routes/event_stream.py`  
**Severity**: MEDIUM  
**Category**: API Design - Consistency

Three different error response formats coexist:

```python
# routes/agents.py - returns 200 with error
return {"error": str(e)}  # No "success" key
return {"success": False, "error": str(e)}  # With "success" key

# routes/provenance.py - returns 200 with error
return {"success": False, "error": str(exc)}

# server.py - proper error response via huginn_error_response
return JSONResponse(status_code=500, content=huginn_error_response(...))
```

The `server.py` has a proper global exception handler (line 184) and HTTPException handler (line 209) that use `huginn_error_response()`, but individual route handlers bypass this by catching exceptions themselves and returning dicts.

**Recommendation**: Remove per-route try/except blocks and let the global exception handler catch errors. For expected errors (not found, validation), raise `HTTPException` with appropriate status codes.

---

### M4. `server.py` sys.modules wrapper hack for backward compatibility

**File**: `server.py:244-272`  
**Severity**: MEDIUM  
**Category**: Code Smell - Fragile Hack

```python
class _ServerModule(types.ModuleType):
    """Module wrapper delegating shared-state attrs to server_core / lifespan."""
    
    def __getattr__(self, name: str) -> Any:
        if name in _DELEGATED_SC:
            return getattr(_sc, name)
        ...
    def __setattr__(self, name: str, value: Any) -> None:
        if name in _DELEGATED_SC:
            setattr(_sc, name, value)
            return
        ...

_wrapper = _ServerModule(__name__)
sys.modules[__name__] = _wrapper
```

This replaces the actual server module with a wrapper that delegates attribute access to `server_core` and `lifespan`. It exists because tests do `import huginn.server as m; m._context = ctx` and expect it to propagate to `server_core._context`.

This is fragile because:
1. It changes the module type at import time
2. IDE introspection and static analysis tools can't follow the delegation
3. Any new shared state requires updating `_DELEGATED_SC` / `_DELEGATED_LF`
4. The `__dict__.update()` at line 265 copies everything non-dunder, which could shadow delegated attributes

**Recommendation**: Fix the tests to use `import huginn.server_core as m; m._context = ctx` directly. Remove the wrapper.

---

### M5. `context_builder.py` has repeated import-then-insert pattern

**File**: `context_builder.py:420-453`  
**Severity**: MEDIUM  
**Category**: Code Duplication

The `build_input_messages` method has 6 nearly identical blocks:

```python
emotion_text = self.build_emotion_text(message)
if emotion_text:
    from langchain_core.messages import SystemMessage
    messages.insert(-1, SystemMessage(content=emotion_text, id="ctx_emotion"))

plan_text = self.build_plan_text(session_state)
if plan_text:
    from langchain_core.messages import SystemMessage
    messages.insert(-1, SystemMessage(content=plan_text, id="ctx_plan"))

# ... 4 more identical blocks for cognitive, tool_hint, evolution, continuity
```

Each block:
1. Calls a `build_*_text` method
2. Checks if the result is truthy
3. Re-imports `SystemMessage` (already imported at top of file via TYPE_CHECKING)
4. Inserts at position -1

**Recommendation**: Extract a helper:
```python
def _inject_context(messages, text, ctx_id):
    if text:
        messages.insert(-1, SystemMessage(content=text, id=ctx_id))
```
And loop over `(text, ctx_id)` pairs. Move the `SystemMessage` import to the top of the file.

---

### M6. Plan mode JSON parsing with triple-nested try/except fallback

**File**: `routes/ws.py:1029-1067`  
**Severity**: MEDIUM  
**Category**: Over-complexity - Nested Error Handling

```python
try:
    plan_data = _json.loads(plan_text)
except (Exception,):  # Note: tuple with single element
    try:
        match = re.search(r"\{[\s\S]*\}", plan_text)
        if match:
            plan_data = _json.loads(match.group())
        else:
            plan_data = {  # Fallback 1
                "steps": [{"name": "Execute task", ...}],
                ...
            }
    except Exception:
        plan_data = {  # Fallback 2 - identical to Fallback 1
            "steps": [{"name": "Execute task", ...}],
            ...
        }
```

The two fallback blocks (lines 1043-1054 and 1056-1067) are **identical**. The `except (Exception,):` syntax (line 1031) is a confusing way to write `except Exception:`.

**Recommendation**: 
1. Flatten to a single fallback.
2. Use `except json.JSONDecodeError:` specifically.
3. Extract the fallback plan dict to a helper function.

---

### M7. `numerical_tool.py` eval uses restricted but incomplete sandbox

**File**: `tools/sci/numerical_tool.py:730-763`  
**Severity**: MEDIUM  
**Category**: Security - Incomplete Sandboxing

Even though the `eval_globals` dict limits available names, Python's `eval()` allows:
- Attribute access: `eval_globals["cp"].__class__.__module__.__loader__`
- String formatting that can access `__builtins__`
- List comprehension variable leakage

The `eval_locals` is empty, which means `eval()` will inject `__builtins__` into it on first call, potentially providing access to `__import__`.

**Recommendation**: Add `"__builtins__": {}` to `eval_globals` to prevent builtins injection. Better yet, use `ast.parse` + `ast.NodeVisitor` to validate the expression tree before evaluation.

---

## LOW Issues

### L1. Repeated `"error": str(e)` pattern across route handlers

**Files**: `routes/agents.py` (20+ occurrences), `routes/provenance.py` (6 occurrences)  
**Severity**: LOW  
**Category**: Code Duplication

Every route handler repeats:
```python
except Exception as e:
    logger.error("unexpected error", exc_info=True)
    return {"success": False, "error": str(e)}
```

The error message is always `"unexpected error"` regardless of what actually happened.

**Recommendation**: Create a decorator `@handle_errors` that wraps route handlers and centralizes error handling.

---

### L2. `prompts.py` contains an extremely long single string

**File**: `prompts.py`  
**Severity**: LOW  
**Category**: Maintainability

The `HUGINN_SYSTEM_PROMPT` and `EXPLORATION_PROMPT` are multi-hundred-line string literals embedded directly in Python source. Changes to the prompt require modifying a Python file and redeploying. The tool descriptions section (lines 69-99) duplicates information that also exists in each tool's `.md` description file.

**Recommendation**: Move prompts to external `.md` or `.yaml` files, similar to how `HuginnTool._load_description()` already loads tool descriptions from `.md` files.

---

### L3. `routes/event_stream.py` has no authentication or connection limit

**File**: `routes/event_stream.py`  
**Severity**: LOW  
**Category**: Security - Resource Exhaustion

```python
@router.get("/events/stream")
async def event_stream() -> StreamingResponse:
    bus = EventBus.shared()
    return StreamingResponse(bus.sse_stream(), ...)
```

While the app-level `require_api_key` dependency (server.py:71) applies to all routes, the SSE endpoint creates an unbounded streaming connection. A client that opens many connections without closing them could exhaust server resources (file descriptors, memory for per-consumer queues).

**Recommendation**: Add a connection counter and limit concurrent SSE connections per client.

---

### L4. `skills/base.py` `_resolve_value` uses `ast.literal_eval` on user-controlled input

**File**: `skills/base.py:291-294`  
**Severity**: LOW  
**Category**: Security - Input Validation

```python
try:
    return ast.literal_eval(mapping)
except (ValueError, SyntaxError):
    return mapping
```

`ast.literal_eval` is safer than `eval` but can still cause stack overflow on deeply nested input (e.g., `[[[[[[...` with thousands of levels). The function is called on `input_mapping` values from skill definitions, which may come from imported skill files.

**Recommendation**: Add a length limit on the input string before calling `literal_eval`.

---

### L5. Unused `import traceback` in routes/agents.py

**File**: `routes/agents.py:8`  
**Severity**: LOW  
**Category**: Dead Code

```python
import traceback
```

This import is never used in the file. The error handling uses `logger.error(..., exc_info=True)` which handles traceback formatting internally.

**Recommendation**: Remove the unused import.

---

### L6. `server.py` rate limiter uses `global` keyword in async middleware

**File**: `server.py:148-152`  
**Severity**: LOW  
**Category**: Performance - Global Variable

```python
global _request_counter
_request_counter += 1
if _request_counter >= _BUCKET_SWEEP_INTERVAL:
    _request_counter = 0
    _sweep_empty_buckets()
```

The `global` variable increment is not thread-safe. FastAPI's async middleware runs in a single-threaded event loop, so this is currently safe, but if the app ever uses thread-based middleware or multiple workers, this counter will race.

**Recommendation**: Use `itertools.count()` or a `threading.Counter` if multi-worker safety is needed. For now, add a comment noting the single-thread assumption.

---

### L7. `context_builder.py` `build_evolution_rules` reads JSON with bare `json.load`

**File**: `context_builder.py:356-357`  
**Severity**: LOW  
**Category**: Error Handling

```python
with rules_path.open("r", encoding="utf-8") as f:
    rules = json.load(f)
```

If the `evolution_rules.json` file is corrupted or partially written (common during concurrent writes), `json.load` raises `JSONDecodeError`. The outer `except Exception: return ""` catches it, but silently, making it hard to diagnose why evolution rules stopped appearing in context.

**Recommendation**: Log the specific error: `except json.JSONDecodeError: logger.warning("evolution_rules.json is corrupted", exc_info=True)`.

---

## Summary by Severity

| Severity | Count | Key Themes |
|----------|-------|------------|
| CRITICAL | 3 | Code injection (eval, shell=True), shared mutable state |
| HIGH | 7 | Error swallowing (365 instances), god class, API design, duplication |
| MEDIUM | 7 | Import shims (54 files), singleton state, inconsistent error formats |
| LOW | 7 | Dead code, prompt maintenance, resource limits |

## Top 5 Recommended Actions (Priority Order)

1. **Replace `eval()` in `numerical_tool.py` with `safe_eval`** - Immediate security risk
2. **Fix `shell=True` in `bourbaki_env.py`** - Remove curl-pipe-to-sh pattern
3. **Isolate WebSocket per-connection state** - Move `_pending_plans`/`_pending_approvals` to per-connection scope
4. **Split `agent.py` chat() method** - Extract the 770-line method into a `ChatOrchestrator`
5. **Standardize route error handling** - Remove per-route try/except, let the global handler work; return proper HTTP status codes