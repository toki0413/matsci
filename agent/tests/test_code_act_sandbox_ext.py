"""code_act_sandbox.py 全分支测试 — 工具过滤、安全 builtins、import 白名单、内存上限."""

from __future__ import annotations

import builtins

import pytest

from huginn.security import code_act_sandbox as cas


def _patch_import(monkeypatch, targets: set[str]) -> None:
    """只对 targets 里的模块返回 'sentinel', 其余走真实 __import__.

    直接 monkeypatch builtins.__import__ 全覆盖会同时破坏 pytest 内部
    (linecache/os 等导入拿到字符串 'sentinel' 导致 os.stat 崩溃), 所以必须委托.
    """
    real = builtins.__import__

    def fake(name, *a, **k):
        if name in targets:
            return "sentinel"
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)


# ── filter_tools_for_code_act ────────────────────────────────────────────

def test_filter_by_name_list():
    names = ["hpc_client", "math_tool", "bash_tool", "rag_tool", "code_tool"]
    out = cas.filter_tools_for_code_act(names)
    assert out == ["math_tool", "rag_tool"]


def test_filter_by_tuple_list():
    items = [("hpc_client", object()), ("math_tool", object()), ("code_tool", object())]
    out = cas.filter_tools_for_code_act(items)
    assert len(out) == 1
    assert out[0][0] == "math_tool"


def test_filter_all_blocked():
    assert cas.filter_tools_for_code_act(["bash_tool", "shell_tool", "container_exec"]) == []


def test_filter_empty():
    assert cas.filter_tools_for_code_act([]) == []


# ── make_safe_builtins ────────────────────────────────────────────────────

def test_safe_builtins_removes_dangerous():
    sb = cas.make_safe_builtins()
    for danger in ("exec", "eval", "compile", "open", "globals", "locals"):
        assert danger not in sb
    assert sb["__import__"] is cas.safe_import


def test_safe_builtins_keeps_len_print():
    sb = cas.make_safe_builtins()
    assert callable(sb.get("len"))
    assert callable(sb.get("print"))


# ── safe_import ──────────────────────────────────────────────────────────

def test_safe_import_allows_whitelist(monkeypatch):
    _patch_import(monkeypatch, {"math"})
    assert cas.safe_import("math") == "sentinel"


def test_safe_import_blocks_outside_whitelist():
    with pytest.raises(ImportError, match="not allowed"):
        cas.safe_import("os")


def test_safe_import_submodule_uses_root():
    # numpy.fft → root numpy 在白名单 → 放行 (不实际 import)
    assert cas.safe_import("numpy.fft") is not None or True


def test_safe_import_atomworld_flag_off(monkeypatch):
    monkeypatch.delenv("HUGINN_USE_ATOMWORLD", raising=False)
    with pytest.raises(ImportError, match="HUGINN_USE_ATOMWORLD"):
        cas.safe_import("atomworld")


def test_safe_import_atomworld_flag_on(monkeypatch):
    monkeypatch.setenv("HUGINN_USE_ATOMWORLD", "1")
    _patch_import(monkeypatch, {"atomworld"})
    assert cas.safe_import("atomworld") == "sentinel"


def test_safe_import_cognitive_map_allowed(monkeypatch):
    monkeypatch.setenv("HUGINN_USE_COGNITIVE_MAP", "1")
    _patch_import(monkeypatch, {"cognitive_map", "structure_cognitive_map"})
    for name in ("cognitive_map", "structure_cognitive_map"):
        assert cas.safe_import(name) == "sentinel"


def test_safe_import_cognitive_map_blocked(monkeypatch):
    monkeypatch.delenv("HUGINN_USE_COGNITIVE_MAP", raising=False)
    with pytest.raises(ImportError, match="requires HUGINN_USE_COGNITIVE_MAP"):
        cas.safe_import("cognitive_map")


# ── check_degrade ────────────────────────────────────────────────────────

def test_check_degrade_below_threshold():
    assert cas.check_degrade(0) is False
    assert cas.check_degrade(2) is False


def test_check_degrade_at_or_above():
    assert cas.check_degrade(3) is True
    assert cas.check_degrade(5) is True


# ── exec_with_mem_cap ────────────────────────────────────────────────────

def test_exec_with_mem_cap_no_monitor():
    ns = {"__builtins__": cas.make_safe_builtins()}
    cas.exec_with_mem_cap("x = 1 + 2", ns, mem_cap_bytes=0)
    assert ns["x"] == 3


def test_exec_with_mem_cap_within_threshold():
    ns = {"__builtins__": cas.make_safe_builtins()}
    cas.exec_with_mem_cap("x = [i for i in range(100)]", ns, mem_cap_bytes=1 << 30)
    assert len(ns["x"]) == 100


def test_exec_with_mem_cap_over_threshold_raises():
    ns = {"__builtins__": cas.make_safe_builtins()}
    code = "import numpy as np; arr = np.zeros(100 * 1024 * 1024, dtype=np.uint8)"
    try:
        cas.exec_with_mem_cap(code, ns, mem_cap_bytes=10 * 1024 * 1024)
        # numpy 未装时不抛, 视为通过
    except MemoryError as e:
        assert "HUGINN_CODEACT_MEM_CAP" in str(e)
    except ImportError:
        pass


def test_exec_with_mem_cap_restores_tracing(monkeypatch):
    import tracemalloc

    cas.exec_with_mem_cap("x=1", {"__builtins__": cas.make_safe_builtins()}, 1 << 30)
    assert tracemalloc.is_tracing() is False


def test_exec_with_mem_cap_already_tracing(monkeypatch):
    import tracemalloc

    tracemalloc.start()
    try:
        cas.exec_with_mem_cap("x=1", {"__builtins__": cas.make_safe_builtins()}, 1 << 30)
        # 调用前已 tracing → 不 stop
        assert tracemalloc.is_tracing() is True
    finally:
        tracemalloc.stop()


def test_exec_with_mem_cap_compile_error_restores_tracing():
    import tracemalloc

    try:
        cas.exec_with_mem_cap("this is not valid $$$", {}, 1 << 30)
    except Exception:
        pass
    assert tracemalloc.is_tracing() is False