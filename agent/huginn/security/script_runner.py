"""Safe script execution engine for live scripts."""

from __future__ import annotations

import asyncio
import io
import sys
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from huginn.security.code_act_sandbox import _ALLOWED_IMPORTS


@dataclass
class ScriptResult:
    """Result of a script execution."""

    success: bool
    stdout: str = ""
    stderr: str = ""
    result_value: Any = None
    execution_time_ms: float = 0
    error: str | None = None


# Allowed built-in functions (whitelist approach)
_SAFE_BUILTINS = {
    "abs",
    "all",
    "any",
    "bin",
    "bool",
    "chr",
    "dict",
    "divmod",
    "enumerate",
    "filter",
    "float",
    "format",
    "frozenset",
    "hex",
    "int",
    "isinstance",
    "issubclass",
    "iter",
    "len",
    "list",
    "map",
    "max",
    "min",
    "next",
    "oct",
    "ord",
    "pow",
    "print",
    "range",
    "repr",
    "reversed",
    "round",
    "set",
    "slice",
    "sorted",
    "str",
    "sum",
    "tuple",
    "type",
    "zip",
    "True",
    "False",
    "None",
}

# 白名单 import (迁移自 CodeAct, G22): 只允许这些模块.
# 任何不在白名单的 import 直接拒绝, 比黑名单更安全.
# 权威定义在 security/code_act_sandbox.py (见文件顶部 import), 此处不再留第二份副本.

# Blocked submodules — 白名单内的包也可能有危险子模块
# (e.g. numpy.ctypeslib 暴露 ctypes 能力), 单独拦截
_BLOCKED_SUBMODULES = {
    "numpy.ctypeslib",
    "numpy.testing",
    "scipy.weave",
}


def _captured_runner(
    code: Any,
    sandbox_globals: dict[str, Any],
    stdout_capture: io.StringIO,
    stderr_capture: io.StringIO,
) -> Callable[[], None]:
    """构造在 exec 期间替换 sys.stdout/stderr 的运行函数 (结束必恢复)."""

    def _run() -> None:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = stdout_capture
        sys.stderr = stderr_capture
        try:
            exec(code, sandbox_globals)
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    return _run


def _success_result(
    sandbox_globals: dict[str, Any],
    stdout_capture: io.StringIO,
    stderr_capture: io.StringIO,
    max_output: int,
    elapsed: float,
) -> ScriptResult:
    """构造成功结果 (取捕获输出与脚本内 ``result`` 变量)."""
    result_value = sandbox_globals.get("result")
    return ScriptResult(
        success=True,
        stdout=stdout_capture.getvalue()[:max_output],
        stderr=stderr_capture.getvalue()[:max_output],
        result_value=str(result_value) if result_value is not None else None,
        execution_time_ms=elapsed,
    )


class ScriptRunner:
    """Execute Python scripts in a restricted environment."""

    def __init__(self, timeout: float = 30.0, max_output: int = 100_000):
        self.timeout = timeout
        self.max_output = max_output

    def _build_safe_globals(self) -> dict[str, Any]:
        """Build restricted global namespace."""
        safe_builtins_dict: dict[str, Any] = {}
        import builtins

        for name in _SAFE_BUILTINS:
            if hasattr(builtins, name):
                safe_builtins_dict[name] = getattr(builtins, name)

        # Add safe math
        import math

        for name in dir(math):
            if not name.startswith("_"):
                safe_builtins_dict[name] = getattr(math, name)

        # Block __import__ for dangerous modules
        real_import = builtins.__import__

        def safe_import(name: str, *args: Any, **kwargs: Any) -> Any:
            base = name.split(".")[0]
            if base not in _ALLOWED_IMPORTS:
                raise ImportError(
                    f"Import of '{name}' is not allowed (whitelist: math/statistics/json/re/numpy/pandas/sympy/scipy/matplotlib/ase/pymatgen)"
                )
            # 白名单内的包也可能有危险子模块, 单独拦截
            if name in _BLOCKED_SUBMODULES:
                raise ImportError(f"Import of '{name}' is not allowed in live scripts")
            return real_import(name, *args, **kwargs)

        safe_builtins_dict["__import__"] = safe_import
        safe_builtins_dict["__builtins__"] = safe_builtins_dict

        return {
            "__builtins__": safe_builtins_dict,
            "__name__": "__live_script__",
        }

    @staticmethod
    def _resolve_loop() -> asyncio.AbstractEventLoop:
        """取当前事件循环; 无运行中循环时新建并设为当前循环."""
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop

    async def _exec_in_loop(
        self,
        code: Any,
        sandbox_globals: dict[str, Any],
        stdout_capture: io.StringIO,
        stderr_capture: io.StringIO,
    ) -> None:
        """在线程池执行已编译脚本, 受 ``self.timeout`` 约束."""
        loop = ScriptRunner._resolve_loop()
        _run = _captured_runner(code, sandbox_globals, stdout_capture, stderr_capture)
        await asyncio.wait_for(
            loop.run_in_executor(None, _run),
            timeout=self.timeout,
        )

    async def execute(
        self, script: str, variables: dict[str, Any] | None = None
    ) -> ScriptResult:
        """Execute a Python script in a sandboxed environment."""
        import time

        start = time.perf_counter()

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        sandbox_globals = self._build_safe_globals()
        if variables:
            sandbox_globals.update(variables)

        try:
            # Compile and exec with timeout
            code = compile(script, "<live_script>", "exec")
            await ScriptRunner._exec_in_loop(
                self, code, sandbox_globals, stdout_capture, stderr_capture
            )

            return _success_result(
                sandbox_globals,
                stdout_capture,
                stderr_capture,
                self.max_output,
                (time.perf_counter() - start) * 1000,
            )

        except TimeoutError:
            return ScriptResult(
                success=False,
                error=f"Script execution timed out ({self.timeout}s)",
                stdout=stdout_capture.getvalue()[: self.max_output],
                execution_time_ms=self.timeout * 1000,
            )
        except SyntaxError as e:
            return ScriptResult(
                success=False,
                error=f"Syntax error: {e}",
                execution_time_ms=(time.perf_counter() - start) * 1000,
            )
        except Exception as e:
            return ScriptResult(
                success=False,
                error=f"{type(e).__name__}: {e}",
                stderr=traceback.format_exc()[: self.max_output],
                execution_time_ms=(time.perf_counter() - start) * 1000,
            )
