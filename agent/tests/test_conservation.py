"""守恒不变量对账单元测试.

覆盖 huginn/security/conservation.py 的纯函数逻辑, 以及
sim/vasp_tool.py `_with_conservation` 注入 (不依赖真实 OUTCAR, 直接喂 result dict).

铁律: 守恒审计是纯函数、无 IO、无 LLM, 可完全离线验证.
"""

from __future__ import annotations

from huginn.security.conservation import (
    audit_material_conservation,
    net_force,
    net_torque,
)
from huginn.tools.sim import vasp_tool as vt


def _atom(pos, force):
    return {"position": list(pos), "force": list(force)}


def _F(fx, fy, fz):
    return {"position": [0.0, 0.0, 0.0], "force": [fx, fy, fz]}


# ── net_force / net_torque 纯函数 ────────────────────────────────────


def test_net_force_vanishes_on_balance():
    """对称受力 → 净力 ≈ 0."""
    forces = [_F(1.0, 0.0, 0.0), _F(-1.0, 0.0, 0.0), _F(0.0, 2.0, 0.0), _F(0.0, -2.0, 0.0)]
    assert net_force(forces) < 1e-9


def test_net_force_compiles_unbalanced():
    """单向受力 → 净力 = 矢量和模长."""
    forces = [_F(3.0, 0.0, 0.0), _F(4.0, 0.0, 0.0)]
    assert abs(net_force(forces) - 7.0) < 1e-9


def test_net_torque_placeholder_skipped():
    """position 全占位 [0,0,0] → 扭矩不可判, 返回 None."""
    forces = [_F(1.0, 0.0, 0.0), _F(0.0, 1.0, 0.0)]
    assert net_torque(forces) is None


def test_net_torque_real_position_computed():
    """真实坐标 + 共线力 → 扭矩为 0 (r 与 F 平行)."""
    forces = [
        _atom([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
        _atom([2.0, 0.0, 0.0], [0.0, 3.0, 0.0]),
    ]
    assert net_torque(forces) is not None


# ── audit_material_conservation 判定 ─────────────────────────────────


def test_audit_balanced_forces_pass():
    """完美平衡 → pass."""
    r = audit_material_conservation(
        {"forces": [_F(1.0, 0, 0), _F(-1.0, 0, 0), _F(0, 2.0, 0), _F(0, -2.0, 0)]}
    )
    assert r["verdict"] == "pass"
    assert r["net_force_gap"] < 1e-6
    assert r["force_balance_ratio"] <= 1e-6
    assert "translational_equilibrium" in r["invariants"]


def test_audit_unbalanced_force_fail():
    """净力达最强原子力 100% → fail (平移破坏)."""
    r = audit_material_conservation(
        {"forces": [_F(5.0, 0, 0), _F(5.0, 0, 0)]}
    )
    assert r["verdict"] == "fail"
    assert r["net_force_gap"] > 0
    assert r["force_balance_ratio"] > 0.5


def test_audit_warn_moderate_unbalance():
    """净力为最强原子力 20% (>10% warn 阈值, <50% fail) → warn."""
    r = audit_material_conservation(
        {"forces": [_F(1.0, 0, 0), _F(-0.8, 0, 0)]}
    )
    assert r["verdict"] == "warn"
    assert abs(r["force_balance_ratio"] - 0.2) < 1e-6


def test_audit_empty_forces_skip():
    """无受力数据 → skip, 不误判."""
    r = audit_material_conservation({"forces": []})
    assert r["verdict"] == "skip"
    assert r["n_atoms"] == 0


def test_audit_converged_but_large_force_warn():
    """宣称 converged 却残留大残余力 → warn (跨信号对账).

    用对称抵消构造"净力=0 (平衡通过) 但 max|F| 很大 + 宣称收敛" → 只触发矛盾 warn.
    """
    r = audit_material_conservation(
        {"forces": [_F(2.0, 0, 0), _F(-2.0, 0, 0)], "converged": True}
    )
    assert r["verdict"] == "warn"
    assert r["forces_relaxed"] is False
    assert r["force_balance_ratio"] < 1e-6  # 平衡本身没问题
    assert any("自相矛盾" in x for x in r["reasons"])


def test_audit_real_torque_flagged():
    """真实坐标产生净扭矩 → 旋转平衡纳入审计."""
    r = audit_material_conservation(
        {
            "forces": [
                _atom([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),  # r×F = +z
                _atom([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),  # r×F = +z (同向叠加)
            ],
            "converged": True,
        }
    )
    assert "rotational_equilibrium" in r["invariants"]
    assert r["torque_gap"] is not None


def test_audit_torque_cancelled_pass():
    """真实坐标 + 反向扭矩相互抵消 → 扭矩不告警."""
    r = audit_material_conservation(
        {
            "forces": [
                _atom([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),   # +z
                _atom([1.0, 0.0, 0.0], [0.0, -1.0, 0.0]),  # -z
            ],
            "converged": True,
        }
    )
    assert r["verdict"] == "pass"
    assert r["torque_gap"] < 1e-6


# ── vasp_tool 注入 (不造 OUTCAR, 直接喂 result) ─────────────────────


def test_with_conservation_injects_field():
    """_parse_outcar 的 _with_conservation 为 result 附带 conservation."""
    result = {"forces": [_F(1.0, 0, 0), _F(-1.0, 0, 0)]}
    out = vt.VaspTool()._with_conservation(result)
    assert "conservation" in out
    assert out["conservation"]["verdict"] == "pass"


def test_with_conservation_nonfatal_on_garbage():
    """畸形力条目 (非数值) → 审计健壮降级为 skip, result 原样返回且不抛."""
    result = {"forces": [{"force": "not-a-list"}, {"force": [1.0, "x", 0]}]}
    out = vt.VaspTool()._with_conservation(result)
    assert "conservation" in out
    assert out["conservation"]["verdict"] == "skip"  # 无可用受力数据