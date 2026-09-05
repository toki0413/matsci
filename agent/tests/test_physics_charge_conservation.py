"""电荷守恒验证器单测 — HarnessDev 结论②补盲区.

覆盖: 守恒恒等通过/失败、物理界超、缺字段 advisory、接入 DFT 全流程.
"""

from __future__ import annotations

from huginn.validation.physics import PhysicsValidator


def test_conservation_passes_when_electrons_balance_nuclear():
    """charge + electrons == nuclear_charge 且 |charge| <= electrons → 通过."""
    check = PhysicsValidator()._check_charge_conservation(
        {"charge": 0.0, "total_electrons": 10.0, "nuclear_charge": 10.0}
    )
    assert check.name == "charge_conservation"
    assert check.passed is True
    assert "OK" in check.message


def test_conservation_fails_when_charge_wrong():
    """电荷不守恒 → fail (净电荷应等于核电荷−电子数)."""
    check = PhysicsValidator()._check_charge_conservation(
        {"charge": 0.0, "total_electrons": 10.0, "nuclear_charge": 12.0}
    )
    assert check.passed is False


def test_unphysical_net_charge_exceeding_electrons():
    """|net charge| > total_electrons → fail (物理界)."""
    check = PhysicsValidator()._check_charge_conservation(
        {"charge": 12.0, "total_electrons": 10.0}
    )
    assert check.passed is False
    assert "unphysical" in check.message


def test_missing_charge_is_advisory_pass():
    """缺 charge 字段 → advisory 通过, 不误报."""
    check = PhysicsValidator()._check_charge_conservation({"energy": -1.0})
    assert check.passed is True
    assert "not available" in check.message


def test_partial_data_treated_ok():
    """有 charge 但缺 nuclear_charge → 不判成败, treated as OK."""
    check = PhysicsValidator()._check_charge_conservation(
        {"charge": 0.0, "total_electrons": 10.0}
    )
    assert check.passed is True


def test_charge_conservation_in_dft_pipeline():
    """接入 validate_dft_result 全流程."""
    checks = PhysicsValidator().validate_dft_result(
        {"charge": 0.0, "total_electrons": 8.0, "nuclear_charge": 8.0}
    )
    names = [c.name for c in checks]
    assert "charge_conservation" in names
    cc = next(c for c in checks if c.name == "charge_conservation")
    assert cc.passed is True
