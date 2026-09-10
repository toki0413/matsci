"""量子临界第三域泛化验证测试.

验证: 1) 第三域已登记进 DOMAIN_PROFILES(冷启动守卫一行接入); 2) 物理内核真实且数值
正确(Landau 应变调控/临界指数普适类/相图/材料谱系); 3) 结果契约{objectives,summary,
success}可落地门禁.

目的: 证明同一套 agent 机制可驱动与固体力学零族的陌生域(标称"泛化能力"的观测证据),
而非只对 fracture/rigidity 有效.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

# 直接加载模块文件, 绕开 huginn/__init__ 重依赖 —— 保持纯函数测试轻量.
_ROOT = Path(__file__).resolve().parents[1]  # agent/
_EXDIR = Path(__file__).resolve().parents[2] / "examples"

_spec_cg = importlib.util.spec_from_file_location("cg", _ROOT / "huginn/research/coldstart_guards.py")
cg = importlib.util.module_from_spec(_spec_cg); _spec_cg.loader.exec_module(cg)

_spec_qc = importlib.util.spec_from_file_location("qc", _EXDIR / "shusheng_quantum_critical.py")
qc = importlib.util.module_from_spec(_spec_qc); _spec_qc.loader.exec_module(qc)


def test_quantum_critical_registered_in_guards():
    """第三域已登记 DOMAIN_PROFILES → 冷启动守卫能编译出声明."""
    assert "quantum_critical" in cg.DOMAIN_PROFILES
    g = cg.compile_domain_guards("quantum_critical")
    assert g["domain"] == "quantum_critical"
    assert g["deps_check"].get("numpy") is True
    assert g["deps_check"].get("scipy") is True
    assert cg.verify_domain_ready(g)["ready"] is True
    assert "probe_qc_landau" in g["probes"]


def test_landau_strain_bidirectional_control():
    """应变双向调控铁电: 负应变增强序, 正应变抑制序(真实物理)."""
    r = qc.exp_qc_landau(T=300.0)
    assert r["success"] is True
    eq = {x["strain"]: x["eta*"] for x in r["summary"]["eq_states"]}
    # 负应变 → 铁电序非零; 正应变 → 抑制
    assert eq[-0.6] != 0.0, f"负应变应增强铁电序, got {eq}"
    assert eq[0.6] == 0.0, f"正应变应抑制铁电序, got {eq}"
    assert r["summary"]["eta_spread"] > 0
    assert r["objectives"]["strain_eta_spread"] > 0


def test_critical_beta_matches_mean_field_universality():
    """临界指数 β≈1/2(平均场普适类), 各材料稳定, 应变不移动普适类."""
    for mat in ("BaTiO3", "SrTiO3", "PbTiO3", "KNbO3"):
        r = qc.exp_qc_critical(mat)
        assert r["success"] is True
        assert abs(r["summary"]["beta_fit"] - 0.5) < 0.05, (mat, r)
        assert r["summary"]["universality"] == "mean_field", (mat, r)
        # 应变不改变普适类(物理: 平均场对对称破缺普适类稳定)
        assert abs(r["summary"]["beta_spread_across_strain"]) < 0.2
    assert qc.exp_qc_critical("BaTiO3")["objectives"]["beta"] > 0


def test_phase_topology_and_materials():
    """相图拓扑 + 材料谱系判别真实返回."""
    p = qc.exp_qc_phase()
    assert p["success"] is True and p["summary"]["topology"]
    m = qc.exp_qc_materials()
    assert m["success"] is True
    assert m["summary"]["top_tunable"] in {x[0] for x in qc.PEROVSKITE}
    assert m["objectives"]["tunability_span"] > 0


def test_scan_routes_all_dims():
    """扫描维路由真实(不崩批): material/strain/T 各自分支返回."""
    for dim in ("material", "strain", "T", "kind", "unknown"):
        r = qc.exp_scan({"dim": dim})
        assert r["success"] is True, dim
        assert isinstance(r["objectives"], dict)
        assert isinstance(r["summary"], dict)


def test_result_contract_grounding_ready():
    """结果契约可落地门禁: 所有单值 objective 可在 trace 中溯源(非配置参数)."""
    for f in (qc.exp_qc_landau, qc.exp_qc_critical, qc.exp_qc_phase, qc.exp_qc_materials):
        r = f()
        # objective 必须只有一个纯数值键(可被 _first_scalar 溯源到结果)
        assert len(r["objectives"]) == 1, f
        k = next(iter(r["objectives"]))
        assert isinstance(r["objectives"][k], (int, float))
        assert not isinstance(r["objectives"][k], bool)