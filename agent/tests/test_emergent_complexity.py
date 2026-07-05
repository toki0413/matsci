"""涌现复杂度 (EC) 指标测试."""
from __future__ import annotations

from huginn.validation.emergent_complexity import compute_ec


def test_ec_empty_input():
    """空输入 -> 全 0."""
    r = compute_ec({})
    assert r["ec_score"] == 0.0
    assert r["ec_tool_diversity"] == 0.0


def test_ec_tool_diversity():
    """多工具调用 -> tool_diversity > 0."""
    r = compute_ec({
        "tool_calls": [
            {"tool": "vasp_tool"},
            {"tool": "xrd_tool"},
            {"tool": "thermo_tool"},
        ],
        "summary": "computed band structure and XRD pattern",
    })
    assert r["ec_tool_diversity"] > 0.0


def test_ec_cross_domain():
    """跨领域文本 -> cross_domain > 0."""
    r = compute_ec({
        "summary": "The crystal lattice energy and electronic band gap "
                   "affect the elastic modulus and catalytic adsorption.",
    })
    # 命中结构/热力学/电子/力学/催化 5 个域
    assert r["ec_domains_hit"] >= 3
    assert r["ec_cross_domain"] > 0.0


def test_ec_single_domain():
    """单领域文本 -> cross_domain = 0 (命中 <2 个域)."""
    r = compute_ec({
        "summary": "The crystal lattice symmetry determines diffraction.",
    })
    assert r["ec_cross_domain"] == 0.0


def test_ec_score_range():
    """EC 总分在 [0, 1] 区间."""
    r = compute_ec({
        "tool_calls": [{"tool": "a"}, {"tool": "b"}],
        "summary": "crystal lattice band gap elastic modulus catalytic "
                   "adsorption novel discovery unprecedented",
    })
    assert 0.0 <= r["ec_score"] <= 1.0


def test_ec_geometric_mean_zero_dim():
    """任一维度为 0 不一定让总分归零 (几何平均只取非零维度)."""
    # 没有工具调用但文本丰富
    r = compute_ec({
        "summary": "crystal lattice energy band gap elastic modulus "
                   "catalytic adsorption unprecedented discovery",
    })
    # tool_diversity=0, 但其他维度 >0, EC 仍 >0
    assert r["ec_tool_diversity"] == 0.0
    assert r["ec_score"] > 0.0


def test_ec_string_input():
    """字符串输入不崩溃."""
    r = compute_ec("crystal lattice band gap elastic modulus")
    assert r["ec_score"] >= 0.0


def test_ec_high_complexity():
    """高复杂度场景: 多工具 + 多域 + 丰富文本."""
    r = compute_ec({
        "tool_calls": [
            {"tool": "vasp_tool"},
            {"tool": "xrd_tool"},
            {"tool": "thermo_tool"},
            {"tool": "descriptor_tool"},
        ],
        "summary": (
            "We computed the crystal lattice parameters and electronic band "
            "structure. The phase energy reveals thermodynamic stability. "
            "The elastic modulus shows mechanical strength. Catalytic "
            "adsorption sites were identified. This unprecedented discovery "
            "connects structure-property relationships across domains."
        ),
    })
    assert r["ec_tool_diversity"] > 0.5
    assert r["ec_cross_domain"] > 0.5
    assert r["ec_score"] > 0.3
