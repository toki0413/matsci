"""守恒不变量对账单元测试.

覆盖 huginn/security/conservation.py 的纯函数逻辑, 以及
sim/vasp_tool.py `_with_conservation` 注入 (不依赖真实 OUTCAR, 直接喂 result dict).

铁律: 守恒审计是纯函数、无 IO、无 LLM, 可完全离线验证.
"""

from __future__ import annotations

import types

import pytest

from huginn.security.conservation import (
    audit_material_conservation,
    net_force,
    net_torque,
)
from huginn.tools.sim import vasp_tool as vt


def _atom(pos, force):
    return {"position": list(pos), "force": list(force)}


def _force_vec(fx, fy, fz):
    return {"position": [0.0, 0.0, 0.0], "force": [fx, fy, fz]}


# ── net_force / net_torque 纯函数 ────────────────────────────────────


def test_net_force_vanishes_on_balance():
    """对称受力 → 净力 ≈ 0."""
    forces = [_force_vec(1.0, 0.0, 0.0), _force_vec(-1.0, 0.0, 0.0), _force_vec(0.0, 2.0, 0.0), _force_vec(0.0, -2.0, 0.0)]
    assert net_force(forces) < 1e-9


def test_net_force_compiles_unbalanced():
    """单向受力 → 净力 = 矢量和模长."""
    forces = [_force_vec(3.0, 0.0, 0.0), _force_vec(4.0, 0.0, 0.0)]
    assert abs(net_force(forces) - 7.0) < 1e-9


def test_net_torque_placeholder_skipped():
    """position 全占位 [0,0,0] → 扭矩不可判, 返回 None."""
    forces = [_force_vec(1.0, 0.0, 0.0), _force_vec(0.0, 1.0, 0.0)]
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
        {"forces": [_force_vec(1.0, 0, 0), _force_vec(-1.0, 0, 0), _force_vec(0, 2.0, 0), _force_vec(0, -2.0, 0)]}
    )
    assert r["verdict"] == "pass"
    assert r["net_force_gap"] < 1e-6
    assert r["force_balance_ratio"] <= 1e-6
    assert "translational_equilibrium" in r["invariants"]


def test_audit_unbalanced_force_fail():
    """净力达最强原子力 100% → fail (平移破坏)."""
    r = audit_material_conservation(
        {"forces": [_force_vec(5.0, 0, 0), _force_vec(5.0, 0, 0)]}
    )
    assert r["verdict"] == "fail"
    assert r["net_force_gap"] > 0
    assert r["force_balance_ratio"] > 0.5


def test_audit_warn_moderate_unbalance():
    """净力为最强原子力 20% (>10% warn 阈值, <50% fail) → warn."""
    r = audit_material_conservation(
        {"forces": [_force_vec(1.0, 0, 0), _force_vec(-0.8, 0, 0)]}
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
        {"forces": [_force_vec(2.0, 0, 0), _force_vec(-2.0, 0, 0)], "converged": True}
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
    result = {"forces": [_force_vec(1.0, 0, 0), _force_vec(-1.0, 0, 0)]}
    out = vt.VaspTool()._with_conservation(result)
    assert "conservation" in out
    assert out["conservation"]["verdict"] == "pass"


def test_with_conservation_nonfatal_on_garbage():
    """畸形力条目 (非数值) → 审计健壮降级为 skip, result 原样返回且不抛."""
    result = {"forces": [{"force": "not-a-list"}, {"force": [1.0, "x", 0]}]}
    out = vt.VaspTool()._with_conservation(result)
    assert "conservation" in out
    assert out["conservation"]["verdict"] == "skip"  # 无可用受力数据


# ── 端到端冒烟: mock 走完整 call() 真实解析路径, 守恒不破坏主流程 ─────


def _balanced_outcar(tmp_path) -> list[str]:
    """构造一份平移/旋转都平衡的 OUTCAR 文本.

    4 个原子成对称配置: 受力成对抵消, 位置绕中心对称, 故
    ΣF≈0 且 Σ(r×F)≈0, 守恒审计应判 pass. 含 relax 收敛标记.
    """
    return [
        "INCAR: POTIM = 0.02",
        "free  energy   TOTEN  =       -152.3 eV",
        "reached required accuracy - stopping structural energy minimisation",
        "volume of cell :      42.875",
        "TOTAL-FORCE (eV/Angst)",
        # 径向配置: 每原子受力与位置坐标平行 (r×F=0), 且受力成对抵消 (ΣF=0)
        #          position                      force
        "  1.000  0.000  0.000    1.000  0.000  0.000",
        " -1.000  0.000  0.000   -1.000  0.000  0.000",
        "  0.000  1.000  0.000    0.000  1.000  0.000",
        "  0.000 -1.000  0.000    0.000 -1.000  0.000",
        "",
        "    total drift:    0.000000    0.000000    0.000000",
    ]


def _install_fake_sandbox(monkeypatch, outcar_lines):
    """装一个假 SandboxExecutor: run 返回 success + 写 OUTCAR 到 cwd."""

    class _Result:
        success = True
        returncode = 0
        stdout = ""
        stderr = ""
        command = ["vasp_std"]
        dry_run = False
        blocked = False
        block_reason = None
        timed_out = False
        error_kind = "none"

    class _FakeSandbox:
        def run(self, cmd, cwd=None, timeout=None, queue=None, walltime=None):
            outcar = __import__("pathlib").Path(cwd) / "OUTCAR"
            outcar.write_text("\n".join(outcar_lines), encoding="utf-8")
            return _Result()

    monkeypatch.setattr(vt, "_HAS_HUGINN_EXT", False)
    return _FakeSandbox()


@pytest.mark.anyio
async def test_call_consumes_conservation_end_to_end(monkeypatch, tmp_path):
    """走真实 call() → _run_vasp → _parse_outcar 全链路.

    核心断言: 守恒审计被注入 `parsed.conservation`, 且平移/旋转平衡判 pass;
    主流程 success 不被守恒破坏 (守恒失败也绝不阻塞/抛错).
    """
    work = tmp_path / "run"
    work.mkdir()
    (work / "INCAR").write_text("System = Si\n", encoding="utf-8")
    (work / "POSCAR").write_text("Si\n1.0\n1 0 0\n0 1 0\n0 0 1\n2\ncart\n0 0 0\n0.5 0.5 0.5\n", encoding="utf-8")

    fake = _install_fake_sandbox(monkeypatch, _balanced_outcar(tmp_path))
    tool = vt.VaspTool(vasp_executable="vasp_std", sandbox=fake)

    res = await tool.call(
        vt.VaspToolInput(action="relax", working_dir=str(work)),
        context=types.SimpleNamespace(workspace=str(tmp_path)),
    )

    # 主流程成功, 守恒注入没把计算搞挂
    assert res.success is True
    parsed = res.data["parsed"]
    assert "conservation" in parsed
    assert parsed["conservation"]["verdict"] == "pass"
    assert parsed["converged"] is True


@pytest.mark.anyio
async def test_call_conservation_nonfatal_when_unbalanced(monkeypatch, tmp_path):
    """守恒判 fail 也只进审计结果, 绝不改主流程 success 判定."""
    work = tmp_path / "run2"
    work.mkdir()
    (work / "INCAR").write_text("System = Fe\n", encoding="utf-8")
    (work / "POSCAR").write_text("Fe\n1.0\n1 0 0\n0 1 0\n0 0 1\n1\ncart\n0 0 0\n", encoding="utf-8")

    unbalanced = _balanced_outcar(tmp_path)
    # 第 4 个原子 (索引 8) 受力改成带 Z 分量, 净力不再为零 → 平移破坏
    unbalanced[8] = "  0.000 -1.000  0.000    0.000 -1.000  2.000"

    fake = _install_fake_sandbox(monkeypatch, unbalanced)
    tool = vt.VaspTool(vasp_executable="vasp_std", sandbox=fake)

    res = await tool.call(
        vt.VaspToolInput(action="relax", working_dir=str(work)),
        context=types.SimpleNamespace(workspace=str(tmp_path)),
    )

    # 守恒判 fail 但主流程依然 success, 审计独立于成功判定
    assert res.success is True
    parsed = res.data["parsed"]
    assert "conservation" in parsed
    assert parsed["conservation"]["verdict"] == "fail"
