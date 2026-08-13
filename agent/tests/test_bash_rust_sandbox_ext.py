"""bash_tool.py 的 Rust sandbox 桥接测试 — 覆盖 `huginn_ext.sandbox` 路径.

huginn_ext 未编译安装时, `BashTool.call` 的 Rust sandbox 分支默认走不到
(还要 HUGINN_USE_RUST_SANDBOX=1 才启用). 这里通过 sys.modules 注入 fake
huginn_ext.sandbox 覆盖: 成功 / 失败 / import 缺失回退 / 默认关闭.
"""

from __future__ import annotations

import sys
import types

import pytest

from huginn.core_types import ToolContext
from huginn.security import SandboxError
from huginn.tools import bash_tool as bt


# ── 注入 fake huginn_ext.sandbox ─────────────────────────────────────────

def _install_rust_sandbox(monkeypatch, run_sandboxed=None, raise_import=False):
    """把 fake huginn_ext.sandbox 装进 sys.modules.

    - raise_import=True: 让 `from huginn_ext.sandbox import run_sandboxed`
      抛 ImportError (父模块占位 None), 触发回退到配置后端.
    - 否则: 注入带 run_sandboxed 的 fake 模块.
    """
    if raise_import:
        monkeypatch.setitem(sys.modules, "huginn_ext", None)
        sys.modules.pop("huginn_ext.sandbox", None)
        return

    sandbox = types.ModuleType("huginn_ext.sandbox")
    sandbox.run_sandboxed = run_sandboxed
    ext = types.ModuleType("huginn_ext")
    ext.sandbox = sandbox
    monkeypatch.setitem(sys.modules, "huginn_ext", ext)
    monkeypatch.setitem(sys.modules, "huginn_ext.sandbox", sandbox)


def _enable_rust(monkeypatch):
    monkeypatch.setenv("HUGINN_USE_RUST_SANDBOX", "1")


def _ctx():
    return ToolContext(session_id="test", workspace=".")


def _ok_result():
    return {
        "success": True,
        "returncode": 0,
        "stdout": "hello from rust\n",
        "stderr": "",
        "message": "Command succeeded.",
        "timed_out": False,
    }


# 让简单命令直达 Rust sandbox 分支 (不是重活, 不 dispatch 给 Support subagent)
_SIMPLE = ["echo", "hi"]


# ── Rust sandbox 成功 / 失败 / 降级 ──────────────────────────────────────

@pytest.mark.anyio
async def test_rust_sandbox_success(monkeypatch):
    _enable_rust(monkeypatch)
    _install_rust_sandbox(monkeypatch, run_sandboxed=lambda **k: _ok_result())
    result = await bt.BashTool().call({"command": _SIMPLE}, _ctx())
    assert result.success is True
    assert result.error is None
    assert result.data["stdout"] == "hello from rust\n"
    assert result.data["returncode"] == 0
    assert result.data["timed_out"] is False


@pytest.mark.anyio
async def test_rust_sandbox_failure_with_stderr(monkeypatch):
    _enable_rust(monkeypatch)

    def _fail(**k):
        return {
            "success": False,
            "returncode": 1,
            "stdout": "",
            "stderr": "compilation error",
            "message": "Command failed.",
            "timed_out": False,
        }

    _install_rust_sandbox(monkeypatch, run_sandboxed=_fail)
    result = await bt.BashTool().call({"command": _SIMPLE}, _ctx())
    assert result.success is False
    assert result.error == "compilation error"
    assert result.data["returncode"] == 1
    # suggest_fix 是 str 字段 (对 echo 这类命令可能为空)
    assert isinstance(result.data["suggest_fix"], str)


@pytest.mark.anyio
async def test_rust_sandbox_failure_no_stderr_uses_message(monkeypatch):
    _enable_rust(monkeypatch)

    def _fail(**k):
        return {
            "success": False,
            "returncode": 2,
            "stdout": "",
            "stderr": "",
            "message": "sandbox blew up",
            "timed_out": False,
        }

    _install_rust_sandbox(monkeypatch, run_sandboxed=_fail)
    result = await bt.BashTool().call({"command": _SIMPLE}, _ctx())
    assert result.success is False
    assert result.error == "sandbox blew up"


@pytest.mark.anyio
async def test_rust_sandbox_runs_under_workdir(monkeypatch):
    _enable_rust(monkeypatch)
    captured = {}

    def _spy(**k):
        captured.update(k)
        return _ok_result()

    _install_rust_sandbox(monkeypatch, run_sandboxed=_spy)
    result = await bt.BashTool().call(
        {"command": _SIMPLE, "working_dir": "."}, _ctx()
    )
    assert result.success is True
    # 命令被拆成 command + args 传给 Rust runner
    assert captured["command"] == "echo"
    assert captured["args"] == ["hi"]
    assert captured["timeout"] is not None
    assert isinstance(captured["allowed_base_dirs"], list)


@pytest.mark.anyio
async def test_rust_sandbox_import_missing_falls_back(monkeypatch):
    """huginn_ext.sandbox 缺失时回退到配置后端 (get_executor)."""
    _enable_rust(monkeypatch)
    _install_rust_sandbox(monkeypatch, raise_import=True)

    # 没有可用 executor → SandboxError 表明确实走了配置后端而非 Rust
    def _no_executor():
        raise SandboxError("no executor")

    monkeypatch.setattr(bt, "get_executor", _no_executor)
    result = await bt.BashTool().call({"command": _SIMPLE}, _ctx())
    assert result.success is False
    assert "Execution blocked" in result.error


@pytest.mark.anyio
async def test_rust_sandbox_raises_falls_back(monkeypatch):
    """run_sandboxed 本身抛异常 → 回退到配置后端, 不向上炸."""
    _enable_rust(monkeypatch)

    def _boom(**k):
        raise RuntimeError("rust sandbox crash")

    _install_rust_sandbox(monkeypatch, run_sandboxed=_boom)

    def _no_executor():
        raise SandboxError("no executor")

    monkeypatch.setattr(bt, "get_executor", _no_executor)
    result = await bt.BashTool().call({"command": _SIMPLE}, _ctx())
    assert result.success is False
    assert "Execution blocked" in result.error


@pytest.mark.anyio
async def test_rust_sandbox_disabled_by_default(monkeypatch):
    """HUGINN_USE_RUST_SANDBOX 未设 → 不跑 Rust sandbox, 直接走配置后端."""
    monkeypatch.delenv("HUGINN_USE_RUST_SANDBOX", raising=False)
    called = {"n": 0}

    def _spy(**k):
        called["n"] += 1
        return _ok_result()

    _install_rust_sandbox(monkeypatch, run_sandboxed=_spy)

    def _no_executor():
        raise SandboxError("no executor")

    monkeypatch.setattr(bt, "get_executor", _no_executor)
    result = await bt.BashTool().call({"command": _SIMPLE}, _ctx())
    assert called["n"] == 0  # Rust 路径没被触发
    assert result.success is False
    assert "Execution blocked" in result.error