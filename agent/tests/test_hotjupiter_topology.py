"""高阶拓扑环流诊断单测 (hotjupiter_topology.py).

验证:
  1. 纯方位角环流(常数 zonal, = 超自转谐和)被谐和分量**精确**捕获.
  2. 静止流场分解无病态 (全 0, frac=0).
  3. 分解输出窗口健康 (harmonic/divergent/curl frac + ortho_* 自我证伪键).
  4. topological_circulation 对任意暴露 H/U/V/dx/dy 的 solver 输出字典诊断.

确定性, 纯 numpy + scipy, 零网络/零 LLM、零积分 (用合成场/假 solver).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))
from hotjupiter_topology import hodge_decompose, topological_circulation  # noqa: E402


def _grid():
    ny, nx, dx, dy = 24, 48, 0.4, 0.4
    xx = np.arange(nx)[None, :] * dx
    yy = np.arange(ny)[:, None] * dy - ny * dy / 2
    return ny, nx, dx, dy, xx, yy


def test_pure_loop_is_harmonic_exact():
    """超自转 = 常数方位角环流 → 谐和分量应精确捕获 (frac=1, 正交≈0, 散/旋≈0)."""
    ny, nx, dx, dy, xx, yy = _grid()
    d = hodge_decompose(np.full((ny, nx), 2.0), np.zeros((ny, nx)), dx, dy)
    assert abs(d["harmonic_frac"] - 1.0) < 1e-6
    assert d["divergent_frac"] < 1e-6 and d["curl_frac"] < 1e-6
    assert d["ortho_grad_harm"] < 1e-6 and d["ortho_curl_harm"] < 1e-6
    assert d["harmonic_div_residual"] < 1e-6 and d["harmonic_curl_residual"] < 1e-6


def test_stopped_flow_is_well_defined():
    ny, nx, dx, dy, xx, yy = _grid()
    d = hodge_decompose(np.zeros((ny, nx)), np.zeros((ny, nx)), dx, dy)
    assert d["harmonic_amp"] == 0.0 and d["harmonic_frac"] == 0.0


def test_decomposition_is_bounded_and_self_checked():
    """混合流场: 输出窗口完整、值有限有界(不爆数值); 真谐和性用 div/curl 自检(如实)."""
    ny, nx, dx, dy, xx, yy = _grid()
    u = np.sin(xx) * np.cos(yy) + 1.5
    v = -np.cos(xx) * np.sin(yy)
    d = hodge_decompose(u, v, dx, dy)
    for k in ["harmonic_amp", "harmonic_frac", "divergent_frac", "curl_frac",
              "ortho_grad_curl", "ortho_grad_harm", "ortho_curl_harm",
              "harmonic_div_residual", "harmonic_curl_residual"]:
        assert k in d and np.isfinite(d[k]), (k, d.get(k))
    # 有界: 无爆数值 (早期 pin 版本曾到 ~1e13)
    for k in ["harmonic_frac", "divergent_frac", "curl_frac"]:
        assert 0.0 <= d[k] < 10.0, (k, d[k])
    for k in ["ortho_grad_curl", "ortho_grad_harm", "ortho_curl_harm"]:
        assert 0.0 <= d[k] <= 1.0, (k, d[k])


def test_topological_circulation_from_solver():
    """对暴露 H/U/V/dx/dy 的 solver 输出字典诊断 (含 note 说明)."""
    ny, nx, dx, dy, xx, yy = _grid()

    class FakeSolver:
        H = np.full((ny, nx), 1.0)
        U = np.full((ny, nx), 2.0)   # u = U/H = 2 → 纯谐和
        V = np.zeros((ny, nx))
        dx = 0.4
        dy = 0.4

    d = topological_circulation(FakeSolver)
    assert "harmonic_frac" in d and "note" in d
    assert abs(d["harmonic_frac"] - 1.0) < 1e-6, d

    # 完整性: components 被剥离, 不泄漏大数组
    assert "components" not in d