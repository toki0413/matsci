"""Safe script execution engine for live scripts."""

from __future__ import annotations

import asyncio
import io
import sys
import traceback
from dataclasses import dataclass, field
from typing import Any


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

# Blocked modules (security)
_BLOCKED_IMPORTS = {
    "os",
    "sys",
    "subprocess",
    "shutil",
    "pathlib",
    "socket",
    "ctypes",
    "multiprocessing",
    "threading",
    "signal",
    "importlib",
    "code",
    "codeop",
    "compileall",
    "py_compile",
    "pickle",
    "shelve",
    "marshal",
}

# Blocked submodules — these can be imported indirectly through allowed
# parent packages (e.g. numpy.ctypeslib) and provide dangerous capabilities.
_BLOCKED_SUBMODULES = {
    "numpy.ctypeslib",
    "numpy.testing",
    "scipy.weave",
}


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
            if base in _BLOCKED_IMPORTS:
                raise ImportError(
                    f"Import of '{name}' is not allowed in live scripts"
                )
            # Block dangerous submodules of otherwise-allowed packages
            if name in _BLOCKED_SUBMODULES:
                raise ImportError(
                    f"Import of '{name}' is not allowed in live scripts"
                )
            mod = real_import(name, *args, **kwargs)
            # Strip dangerous attributes from numpy after import
            if base == "numpy" and hasattr(mod, "ctypeslib"):
                try:
                    delattr(mod, "ctypeslib")
                except AttributeError:
                    pass
            return mod

        safe_builtins_dict["__import__"] = safe_import
        safe_builtins_dict["__builtins__"] = safe_builtins_dict

        return {
            "__builtins__": safe_builtins_dict,
            "__name__": "__live_script__",
        }

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

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            def _run() -> None:
                old_stdout, old_stderr = sys.stdout, sys.stderr
                sys.stdout = stdout_capture
                sys.stderr = stderr_capture
                try:
                    exec(code, sandbox_globals)
                finally:
                    sys.stdout = old_stdout
                    sys.stderr = old_stderr

            await asyncio.wait_for(
                loop.run_in_executor(None, _run),
                timeout=self.timeout,
            )

            elapsed = (time.perf_counter() - start) * 1000
            stdout_text = stdout_capture.getvalue()[: self.max_output]
            stderr_text = stderr_capture.getvalue()[: self.max_output]

            # Check for a 'result' variable in the script
            result_value = sandbox_globals.get("result", None)

            return ScriptResult(
                success=True,
                stdout=stdout_text,
                stderr=stderr_text,
                result_value=(
                    str(result_value) if result_value is not None else None
                ),
                execution_time_ms=elapsed,
            )

        except asyncio.TimeoutError:
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
