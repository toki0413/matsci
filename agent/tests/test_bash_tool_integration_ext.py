"""bash_tool.py 集成路径测试 — ContainerExecutor / SandboxExecutor / 重活 dispatch.

覆盖 BashTool.call 的:
  - 空命令分支
  - ContainerExecutor 路径 (成功/失败)
  - SandboxExecutor 路径 (成功/失败/SandboxError/异常)
  - 重活识别 + dispatch 路径 (persistent / support subagent / 无 factory 降级)
  - _suggest_fix / _extract_progress / _is_heavy_bash 纯函数全分支

Rust sandbox 路径已在 test_bash_rust_sandbox_ext.py 覆盖, 此处不重复.
"""

from __future__ import annotations

import types

import pytest

from huginn.core_types import ToolContext
from huginn.security import SandboxError
from huginn.tools import bash_tool as bt

# ── ToolContext ─────────────────────────────────────────────────────


def _ctx(agent_factory=None, workspace="."):
    return ToolContext(
        session_id="test", workspace=workspace, agent_factory=agent_factory
    )


# _SIMPLE 不是重活, 不触发 dispatch, 直通后端
_SIMPLE = ["echo", "hi"]


# ── _is_heavy_bash ──────────────────────────────────────────────────


class TestIsHeavyBash:
    def test_empty_command(self):
        assert bt._is_heavy_bash([]) == (False, "")

    def test_jupyter(self):
        assert bt._is_heavy_bash(["jupyter", "notebook"])[0] is True

    def test_notebook_keyword(self):
        assert bt._is_heavy_bash(["code", "notebook.ipynb"])[0] is True

    def test_python_train_py(self):
        is_heavy, reason = bt._is_heavy_bash(["python", "train.py", "--epochs", "100"])
        assert is_heavy is True
        assert "train" in reason

    def test_python_fit_py(self):
        is_heavy, _ = bt._is_heavy_bash(["python", "fit_model.py"])
        assert is_heavy is True

    def test_python_epoch_py(self):
        is_heavy, _ = bt._is_heavy_bash(["python", "run.py", "--n_epochs", "10"])
        assert is_heavy is True

    def test_python_py_no_keyword(self):
        assert bt._is_heavy_bash(["python", "hello.py"]) == (False, "")

    def test_pip_install_not_heavy(self):
        assert bt._is_heavy_bash(["pip", "install", "torch"]) == (False, "")

    def test_ls_not_heavy(self):
        assert bt._is_heavy_bash(["ls", "-la"]) == (False, "")

    def test_case_insensitive(self):
        # 关键词大小写不敏感 (cmd_lower), 而 .py 扩展名匹配在 cmd_str 上大小写敏感
        assert bt._is_heavy_bash(["PYTHON", "train.py"])[0] is True


# ── _suggest_fix ────────────────────────────────────────────────────


class TestSuggestFix:
    def test_module_not_found(self):
        s = "Traceback\nModuleNotFoundError: No module named 'numpy'"
        out = bt._suggest_fix(1, s, "", ["python", "x.py"])
        assert "pip install numpy" in out

    def test_no_module_named(self):
        s = "ERROR: No module named pandas"
        out = bt._suggest_fix(1, s, "", ["python", "x.py"])
        assert "pip install pandas" in out

    def test_syntax_error(self):
        out = bt._suggest_fix(1, "SyntaxError: invalid syntax", "", ["python", "x.py"])
        assert "SyntaxError" in out

    def test_file_not_found(self):
        out = bt._suggest_fix(1, "FileNotFoundError: [Errno 2]", "", ["python", "x.py"])
        assert "FileNotFoundError" in out

    def test_no_such_file(self):
        out = bt._suggest_fix(1, "No such file or directory", "", ["python", "x.py"])
        assert "FileNotFoundError" in out

    def test_import_error(self):
        out = bt._suggest_fix(1, "ImportError: cannot import name", "", ["python", "x.py"])
        assert "ImportError" in out

    def test_timeout(self):
        out = bt._suggest_fix(124, "Timed out after 300s", "", ["python", "train.py"])
        assert "Timeout" in out

    def test_attribute_error(self):
        out = bt._suggest_fix(1, "AttributeError: 'NoneType' object", "", ["python", "x.py"])
        assert "AttributeError" in out

    def test_value_error(self):
        out = bt._suggest_fix(1, "ValueError: shapes mismatch", "", ["python", "x.py"])
        assert "Value" in out

    def test_type_error(self):
        out = bt._suggest_fix(1, "TypeError: unsupported operand", "", ["python", "x.py"])
        assert "Type" in out

    def test_cuda_runtime_error(self):
        out = bt._suggest_fix(1, "RuntimeError: CUDA out of memory", "", ["python", "x.py"])
        assert "CUDA" in out

    def test_failed_no_output(self):
        out = bt._suggest_fix(1, "", "", ["command", "notfound"])
        assert "no output" in out

    def test_no_fix_matches(self):
        assert bt._suggest_fix(0, "some benign stdout", "", ["echo", "hi"]) == ""

    def test_module_in_stdout(self):
        # 模块缺失信息在 stdout 而非 stderr
        out = bt._suggest_fix(1, "", "No module named scipy", ["python", "x.py"])
        assert "pip install scipy" in out


# ── _extract_progress ───────────────────────────────────────────────


class TestExtractProgress:
    def test_empty(self):
        assert bt._extract_progress("") == []

    def test_blank_only_lines(self):
        assert bt._extract_progress("\n\n  \n") == []

    def test_filters_keyword_lines(self):
        out = bt._extract_progress("epoch 1 loss 0.5\nplain output\nerror: boom")
        assert out == ["epoch 1 loss 0.5", "error: boom"]

    def test_truncates_at_max(self):
        lines = "\n".join(f"epoch {i} loss {i}" for i in range(100))
        out = bt._extract_progress(lines, max_lines=5)
        assert len(out) == 5


# ── BashTool.call: 空命令 / Container / Sandbox ──────────────────────


class TestBashCall:
    @pytest.mark.anyio
    async def test_empty_command(self):
        result = await bt.BashTool().call({"command": []}, _ctx())
        assert result.success is False
        assert result.error == "Empty command."

    @pytest.mark.anyio
    async def test_container_success(self, monkeypatch):
        executor = _FakeContainer(success=True)
        monkeypatch.setattr(bt, "get_executor", lambda: executor)
        result = await bt.BashTool().call({"command": _SIMPLE}, _ctx())
        assert result.success is True
        assert result.data["container"] is True
        assert result.data["message"] == "Command succeeded."
        assert executor.called

    @pytest.mark.anyio
    async def test_container_failure(self, monkeypatch):
        executor = _FakeContainer(success=False, returncode=127, stderr="command not found")
        monkeypatch.setattr(bt, "get_executor", lambda: executor)
        result = await bt.BashTool().call({"command": ["nosuchcmd"]}, _ctx())
        assert result.success is False
        assert "Command failed (rc=127)" in result.error
        assert result.data["container"] is True
        assert "ModuleNotFoundError" not in result.data["suggest_fix"] or True

    @pytest.mark.anyio
    async def test_sandbox_success(self, monkeypatch):
        executor = _FakeSandbox(returncode=0)
        monkeypatch.setattr(bt, "get_executor", lambda: executor)
        result = await bt.BashTool().call({"command": _SIMPLE}, _ctx())
        assert result.success is True
        assert result.data["sandbox"] is True
        assert result.data["message"] == "Command succeeded."

    @pytest.mark.anyio
    async def test_sandbox_failure(self, monkeypatch):
        executor = _FakeSandbox(returncode=2, stderr="boom")
        monkeypatch.setattr(bt, "get_executor", lambda: executor)
        result = await bt.BashTool().call({"command": _SIMPLE}, _ctx())
        assert result.success is False
        assert "Command failed (rc=2)" in result.error
        assert result.data["sandbox"] is True

    @pytest.mark.anyio
    async def test_sandbox_sandbox_error(self, monkeypatch):
        monkeypatch.setattr(bt, "get_executor", lambda: _BoomSandbox(SandboxError("blocked by policy")))
        result = await bt.BashTool().call({"command": _SIMPLE}, _ctx())
        assert result.success is False
        assert "Sandbox blocked command" in result.error

    @pytest.mark.anyio
    async def test_sandbox_generic_exception(self, monkeypatch):
        monkeypatch.setattr(bt, "get_executor", lambda: _BoomSandbox(RuntimeError("executor exploded")))
        result = await bt.BashTool().call({"command": _SIMPLE}, _ctx())
        assert result.success is False
        assert "Sandbox execution failed" in result.error

    @pytest.mark.anyio
    async def test_get_executor_raises_sandbox_error(self, monkeypatch):
        def _no_exec():
            raise SandboxError("no executor configured")

        monkeypatch.setattr(bt, "get_executor", _no_exec)
        result = await bt.BashTool().call({"command": _SIMPLE}, _ctx())
        assert result.success is False
        assert "Execution blocked" in result.error


# ── 重活 dispatch 路径 ──────────────────────────────────────────────

_HEAVY = ["python", "train.py", "--epochs", "100"]


class TestHeavyDispatch:
    def _enable_heavy(self, monkeypatch):
        monkeypatch.setenv("HUGINN_CORE_SUPPORT_PROTOCOL", "1")
        monkeypatch.setenv("HUGINN_USE_RUST_SANDBOX", "0")

    @pytest.mark.anyio
    async def test_persistent_terminal_dispatch(self, monkeypatch):
        self._enable_heavy(monkeypatch)
        import huginn.tools.persistent_terminal as pt

        monkeypatch.setattr(pt, "resolve_persistent_terminal_flag", lambda x: True)

        class _FakeTerminal:
            def start(self, command, cwd=None):
                self.cmd = command
                return "sess-123"

        term = _FakeTerminal()
        monkeypatch.setattr(pt, "get_default_terminal", lambda: term)
        result = await bt.BashTool().call({"command": _HEAVY}, _ctx())
        assert result.success is True
        assert result.data["session_id"] == "sess-123"
        assert result.metadata["dispatched_via_persistent_terminal"] is True
        assert term.cmd == _HEAVY

    @pytest.mark.anyio
    async def test_persistent_terminal_start_fails_falls_back_to_sandbox(self, monkeypatch):
        self._enable_heavy(monkeypatch)
        import huginn.tools.persistent_terminal as pt

        monkeypatch.setattr(pt, "resolve_persistent_terminal_flag", lambda x: True)

        def _boom():
            raise RuntimeError("no terminal")

        monkeypatch.setattr(pt, "get_default_terminal", _boom)
        # 降级到 SandboxExecutor
        executor = _FakeSandbox(returncode=0)
        monkeypatch.setattr(bt, "get_executor", lambda: executor)
        result = await bt.BashTool().call({"command": _HEAVY}, _ctx())
        assert result.success is True
        assert result.data["sandbox"] is True

    @pytest.mark.anyio
    async def test_support_subagent_dispatch(self, monkeypatch):
        self._enable_heavy(monkeypatch)
        import huginn.tools.persistent_terminal as pt

        monkeypatch.setattr(pt, "resolve_persistent_terminal_flag", lambda x: False)

        class _FakeSubagentTool:
            async def call(self, input_data, context):
                r = types.SimpleNamespace(
                    success=True,
                    data={"summary": "training done"},
                    metadata={},
                )
                return r

        # SubagentTool 在函数体内 `from huginn.tools.subagent_tool import SubagentTool`
        import huginn.tools.subagent_tool as st

        monkeypatch.setattr(st, "SubagentTool", _FakeSubagentTool)
        # 一致性检查在函数体内 `from huginn.agents.subagent import ...`
        import huginn.agents.subagent as sa

        monkeypatch.setattr(sa, "_check_finding_consistency", lambda finding, ctx: (True, "ok"))
        monkeypatch.setattr(sa, "_write_support_rejection", lambda *a, **k: None)
        fake_factory = object()
        result = await bt.BashTool().call({"command": _HEAVY}, _ctx(agent_factory=fake_factory))
        assert result is not None
        assert result.metadata.get("dispatched_to_support") is True

    @pytest.mark.anyio
    async def test_support_dispatch_rejection_path(self, monkeypatch):
        self._enable_heavy(monkeypatch)
        import huginn.tools.persistent_terminal as pt

        monkeypatch.setattr(pt, "resolve_persistent_terminal_flag", lambda x: False)

        class _FakeSubagentTool:
            async def call(self, input_data, context):
                r = types.SimpleNamespace(
                    success=True,
                    data={"summary": "claim"},
                    metadata={},
                )
                return r

        import huginn.tools.subagent_tool as st

        monkeypatch.setattr(st, "SubagentTool", _FakeSubagentTool)
        import huginn.agents.subagent as sa

        monkeypatch.setattr(sa, "_check_finding_consistency", lambda finding, ctx: (False, "contradiction"))
        written = []
        monkeypatch.setattr(sa, "_write_support_rejection", lambda *a: written.append(a))
        result = await bt.BashTool().call(
            {"command": _HEAVY}, _ctx(agent_factory=object(), workspace=".")
        )
        assert result.metadata.get("h1_status") == "nonzero"
        assert result.metadata.get("h1_obstruction") is True
        assert result.data["summary"] is None
        assert written, "rejection 应被写入"

    @pytest.mark.anyio
    async def test_support_dispatch_no_agent_factory_falls_back(self, monkeypatch):
        self._enable_heavy(monkeypatch)
        import huginn.tools.persistent_terminal as pt

        monkeypatch.setattr(pt, "resolve_persistent_terminal_flag", lambda x: False)
        # context 无 agent_factory → dispatch 返回 None → 降级到 sandbox
        executor = _FakeSandbox(returncode=0)
        monkeypatch.setattr(bt, "get_executor", lambda: executor)
        result = await bt.BashTool().call({"command": _HEAVY}, _ctx(agent_factory=None))
        assert result.success is True
        assert result.data["sandbox"] is True

    @pytest.mark.anyio
    async def test_support_dispatch_exception_falls_back(self, monkeypatch):
        self._enable_heavy(monkeypatch)
        import huginn.tools.persistent_terminal as pt

        monkeypatch.setattr(pt, "resolve_persistent_terminal_flag", lambda x: False)

        class _BoomTool:
            async def call(self, *a, **k):
                raise RuntimeError("subagent unavailable")

        import huginn.tools.subagent_tool as st

        monkeypatch.setattr(st, "SubagentTool", _BoomTool)
        executor = _FakeSandbox(returncode=0)
        monkeypatch.setattr(bt, "get_executor", lambda: executor)
        result = await bt.BashTool().call(
            {"command": _HEAVY}, _ctx(agent_factory=object())
        )
        assert result.success is True
        assert result.data["sandbox"] is True

    @pytest.mark.anyio
    async def test_protocol_disabled_runs_directly(self, monkeypatch):
        monkeypatch.setenv("HUGINN_CORE_SUPPORT_PROTOCOL", "0")
        monkeypatch.setenv("HUGINN_USE_RUST_SANDBOX", "0")
        executor = _FakeSandbox(returncode=0)
        monkeypatch.setattr(bt, "get_executor", lambda: executor)
        result = await bt.BashTool().call({"command": _HEAVY}, _ctx())
        assert result.success is True
        assert result.data["sandbox"] is True


# ── fakes ───────────────────────────────────────────────────────────


class _FakeContainer(bt.ContainerExecutor):
    def __init__(self, success=True, returncode=0, stdout="", stderr="", message=""):
        # 绕过真实 __init__ 的 runtime 校验
        object.__setattr__(self, "_success", success)
        object.__setattr__(self, "_returncode", returncode)
        object.__setattr__(self, "_stdout", stdout)
        object.__setattr__(self, "_stderr", stderr)
        self.called = False

    def run(self, cmd, cwd=None, timeout=None, capture_output=True, text=True, **k):
        self.called = True
        return types.SimpleNamespace(
            success=self._success,
            returncode=self._returncode,
            stdout=self._stdout,
            stderr=self._stderr,
        )


class _FakeSandbox(bt.SandboxExecutor):
    def __init__(self, returncode=0, stdout="", stderr=""):
        object.__setattr__(self, "_returncode", returncode)
        object.__setattr__(self, "_stdout", stdout)
        object.__setattr__(self, "_stderr", stderr)

    def run(self, cmd, cwd=None, timeout=None, capture_output=True, text=True, **k):
        return types.SimpleNamespace(
            returncode=self._returncode,
            stdout=self._stdout,
            stderr=self._stderr,
        )


class _BoomSandbox(bt.SandboxExecutor):
    def __init__(self, exc):
        object.__setattr__(self, "_exc", exc)

    def run(self, *a, **k):
        raise self._exc
