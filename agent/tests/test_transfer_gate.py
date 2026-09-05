"""P2 转用门测试 —『同源异流』先验固化为跨工具复用开关.

覆盖三条转用规则 (⊂ is_subcase_of / ≅ isomorphic_to / ≢ independent_of) 与
默认失效安全 (None = 隔离), 以及它挂到 ToolSpec / registry 后的端到端判定.
"""

from __future__ import annotations

from huginn.security.tool_registry import borrow_verdict
from huginn.security.transfer_gate import (
    INDEPENDENT,
    ISOMORPHIC,
    SUBTYPE,
    TransferRelation,
    resolve_transfer,
)


def test_subtype_allows_machinery_not_parameters():
    """⊂ 包含: 子情形可借父求解骨架, 但参数须重标定, 不可整段搬."""
    rel = TransferRelation(kind=SUBTYPE, with_tool="van_der_waals")
    v = resolve_transfer(rel)
    assert v.allow_machinery is True
    assert v.allow_parameters is False


def test_isomorphic_allows_machinery_only():
    """≅ MSR: 只借算法骨架, 参数因现象而异不转移."""
    rel = TransferRelation(kind=ISOMORPHIC, with_tool="equilibrium_field_theory")
    v = resolve_transfer(rel)
    assert v.allow_machinery is True
    assert v.allow_parameters is False


def test_independent_blocks_both():
    """≢ 物理不等价: 普适类不同 → 禁用跨域求解与参数 (失效安全门)."""
    rel = TransferRelation(kind=INDEPENDENT, with_tool="kramers_escape")
    v = resolve_transfer(rel)
    assert v.allow_machinery is False
    assert v.allow_parameters is False


def test_none_relation_failsafe_blocked():
    """未声明关系 → 按隔离失效安全: 全部禁止借用."""
    v = resolve_transfer(None)
    assert v.allow_machinery is False
    assert v.allow_parameters is False


def test_unknown_kind_failsafe_blocked():
    """非法关系值 → 失效安全: 禁转, 不误放行."""
    v = resolve_transfer(TransferRelation(kind="garbage", with_tool="x"))
    assert v.allow_machinery is False
    assert v.allow_parameters is False


def test_registry_subtype_integration_ideal_gas_borrows_vdw():
    """端到端: ideal_gas 声明 ⊂ van_der_waals → 借 vdW 骨架放行, 参数禁借."""
    v = borrow_verdict("van_der_waals", "ideal_gas")
    assert v.allow_machinery is True
    assert v.allow_parameters is False


def test_registry_unrelated_tools_blocked():
    """无声明关系 (如 ideal_gas vs oscillator) → 隔离失效安全: 禁转."""
    v = borrow_verdict("oscillator", "ideal_gas")
    assert v.allow_machinery is False
    assert v.allow_parameters is False


def test_registry_unknown_tool_blocked():
    """目标工具不存在 → 失效安全: 禁转."""
    v = borrow_verdict("does_not_exist", "ideal_gas")
    assert v.allow_machinery is False
    assert v.allow_parameters is False
