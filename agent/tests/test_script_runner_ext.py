"""script_runner.py 全分支测试 — 安全 globals、白名单 import、变量注入、超时、异常."""

from __future__ import annotations

import asyncio

import pytest

from huginn.security.script_runner import ScriptRunner, ScriptResult, _ALLOWED_IMPORTS


def _run(script, variables=None, timeout=30.0):
    async def _a():
        return await ScriptRunner(timeout=timeout).execute(script, variables)

    return asyncio.run(_a())


@pytest.fixture
def runner():
    return ScriptRunner()


# ── _build_safe_globals ──────────────────────────────────────────────────

def test_safe_globals_has_whitelist_builtins(runner):
    sb = runner._build_safe_globals()
    builtins_ = sb["__builtins__"]
    assert builtins_["__import__"] is not None
    for name in ("len", "print", "sum", "range", "map"):
        assert callable(builtins_.get(name))


def test_safe_globals_includes_math(runner):
    sb = runner._build_safe_globals()
    builtins_ = sb["__builtins__"]
    assert callable(builtins_.get("sqrt"))
    assert callable(builtins_.get("sin"))


# ── execute 成功路径 ─────────────────────────────────────────────────────

def test_execute_success_result_variable(runner):
    r = _run("result = 1 + 2")
    assert r.success is True
    assert r.result_value == "3"


def test_execute_success_no_result(runner):
    r = _run("x = 5")
    assert r.success is True
    assert r.result_value is None


def test_execute_captures_stdout(runner):
    r = _run("print('hello')")
    assert r.success is True
    assert "hello" in r.stdout


def test_execute_variables_injected(runner):
    r = _run("result = base * 2", {"base": 21})
    assert r.result_value == "42"


def test_execute_math_import_allowed(runner):
    r = _run("import math; result = math.sqrt(16)")
    assert r.success is True
    assert r.result_value == "4.0"


def test_execute_whitelist_module(runner):
    r = _run("import json; result = json.dumps([1,2])")
    assert r.success is True
    assert r.result_value == "[1, 2]"


# ── 白名单拒绝 ───────────────────────────────────────────────────────────

def test_execute_blocks_dangerous_import(runner):
    r = _run("import os; result = 1")
    assert r.success is False
    assert "not allowed" in r.error


def test_execute_blocks_blocked_submodule(runner):
    r = _run("import numpy.ctypeslib; result = 1")
    # numpy 可能没装, 但白名单拦截在真实 import 前, 应报 not allowed
    assert r.success is False
    assert "not allowed" in r.error


# ── 异常路径 ─────────────────────────────────────────────────────────────

def test_execute_syntax_error(runner):
    r = _run("this is not valid python !!!")
    assert r.success is False
    assert "Syntax error" in r.error


def test_execute_runtime_error(runner):
    r = _run("x = 1 / 0")
    assert r.success is False
    assert "ZeroDivisionError" in r.error


def test_execute_timeout(runner):
    r = _run("import time; time.sleep(5)", timeout=0.1)
    assert r.success is False
    assert "timed out" in r.error


# ── 输出截断 ─────────────────────────────────────────────────────────────

def test_execute_truncates_stdout():
    r = _run("print('x' * 5000)", timeout=30.0)
    # max_output 默认 100000, 5000 字符不截断 → 验证有内容
    assert r.success is True
    assert len(r.stdout) <= 10_000


def test_execute_truncates_to_max_output():
    async def _a():
        return await ScriptRunner(max_output=50).execute("print('x'*200)")

    r = asyncio.run(_a())
    assert r.success is True
    assert len(r.stdout) <= 50


# ── ScriptResult dataclass ───────────────────────────────────────────────

def test_script_result_defaults():
    r = ScriptResult(success=True)
    assert r.stdout == ""
    assert r.stderr == ""
    assert r.result_value is None
    assert r.error is None