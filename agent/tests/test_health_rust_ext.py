"""`/health/rust` 端点的 Rust 扩展可用性检查测试.

覆盖可用/不可用两个分支, 通过 sys.modules 注入 fake huginn_ext.
"""

from __future__ import annotations

import sys
import types

import pytest

from huginn.routes.health import health_rust


def _install_huginn_ext(monkeypatch, names=(), raise_import=False):
    if raise_import:
        monkeypatch.setitem(sys.modules, "huginn_ext", None)
        return
    ext = types.ModuleType("huginn_ext")
    for n in names:
        setattr(ext, n, object())
    monkeypatch.setitem(sys.modules, "huginn_ext", ext)


@pytest.mark.anyio
async def test_health_rust_available(monkeypatch):
    _install_huginn_ext(monkeypatch, names=["top_k", "tail_lines", "sandbox"])
    res = await health_rust()
    assert res["available"] is True
    assert res["module"] == "huginn_ext"
    assert set(res["functions"]) == {"top_k", "tail_lines", "sandbox"}


@pytest.mark.anyio
async def test_health_rust_not_available(monkeypatch):
    _install_huginn_ext(monkeypatch, raise_import=True)
    res = await health_rust()
    assert res["available"] is False
    assert res["module"] == "huginn_ext"
    assert res["functions"] == []
    assert "error" in res