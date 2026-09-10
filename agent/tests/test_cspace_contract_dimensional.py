"""S1+S2·C-Space 契约门禁 & 量纲代数.

S1: ``cspace_bridge.contract_gate`` 把 C-Space 的"状态在场"从文本溯源升级为
    "可证伪且法律一致" —— 状态 Being 的 objectives 越出域科学契约有效域即拒进场.
S2: ``external_validator.validate_declared_units`` 把 scientific_contract.unit 从
    字符串标注升级为**可注册的量纲**(经 dimensional_validator.UnitRegistry 求维度签名).
全部非学习, 走既有 corroborate 钩子不动 C-Space 内核.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

from huginn.research.cspace import CSpace
from huginn.research import cspace_bridge as cb
from huginn.research.coldstart_guards import compile_domain_guards
from huginn.research.external_validator import (
    validate_declared_units, validate_derived_dimensions, resolve_unit_dimension,
)

_QUANT = compile_domain_guards("ecology_dynamics")["scientific_contract"]["quantities"]


# ═══════════════ S1: C-Space 契约门禁 ═══════════════

def _state_being(cs: CSpace, beid: str, objectives: dict):
    cs.beings[beid] = cb.DeliberationBeing(
        id=beid, kind="state", falsifiable=False,
        payload={"objectives": objectives},
    )


def test_cspace_state_gate_promotes_only_contract_valid():
    """状态 Being 落进声明的有效域 → confirmed 进场; 越出 → rejected, 不进场."""
    cs = CSpace()
    _state_being(cs, "good", {"period_est": 17.15, "x_star": 0.6})
    _state_being(cs, "bad", {"period_est": -1.0, "x_star": 0.6})   # period_est 越出 min=0
    gate = cb.contract_gate(_QUANT)
    assert cb.promote_to_at_hand(cs, "good", corroborate=gate)["promoted"] is True
    r_bad = cb.promote_to_at_hand(cs, "bad", corroborate=gate)
    assert r_bad["promoted"] is False and r_bad["state"] == "rejected"
    # 唯一可引用集合只有契约内的 good
    assert [b.id for b in cb.confirmed_at_hand(cs)] == ["good"]
    assert cb.governance(cs)["confirmed"] == 1
    assert cb.governance(cs)["rejected"] == 1


def test_cspace_state_gate_without_objectives_is_not_confirmed():
    """无 objectives 键的状态 Being 不能被契约门禁确认(它不是可对账的状态数值)."""
    cs = CSpace()
    # payload 里没有 objectives 键 → contract_gate 读不到数值, 判不进场
    cs.beings["s"] = cb.DeliberationBeing(id="s", kind="state", falsifiable=False,
                                          payload={"note": "no numeric state"})
    gate = cb.contract_gate(_QUANT)
    assert cb.promote_to_at_hand(cs, "s", corroborate=gate)["promoted"] is False


# ═══════════════ S2: 量纲代数(unit → 维度签名) ═══════════════

def test_declared_units_resolve_to_dimensions():
    """ecology 声明的 unit 都是合法量纲: time→T, count→dimensionless, 且都 valid."""
    res = validate_declared_units(_QUANT)
    assert res["period_est"]["unit"] == "time"
    assert res["period_est"]["dimension_signature"] == "T1"
    assert res["period_est"]["valid"] is True
    assert res["x_star"]["dimension_signature"] == "dimensionless"
    assert res["x_star"]["valid"] is True
    assert all(v["valid"] for v in res.values())


def test_garbage_unit_is_dimensionally_invalid():
    """量纲引擎能咬出拼错的/未知的 unit —— unit 不再是裸字符串标注."""
    bad = {"speed": {"unit": "secods"}}   # 拼错
    res = validate_declared_units(bad)
    assert res["speed"]["valid"] is False
    assert res["speed"]["dimension_signature"] is None
    # 复合量纲能解析: velocity = m/s
    assert resolve_unit_dimension("velocity") == "L1·T-1"


# ═══════════════ S3: 跨量量纲恒等式(velocity=distance/time) ═══════════════

_PHOTOS = {
    "distance": {"unit": "length"},
    "time_elapsed": {"unit": "time"},
    "velocity": {"unit": "velocity",
                 "derived_from": ["distance"], "derived_denom": ["time_elapsed"]},
    "acceleration": {"unit": "acceleration",
                     "derived_from": ["velocity"], "derived_denom": ["time_elapsed"]},
}


def test_cross_quantity_dimensional_identity_consistent():
    """派生量单位与 constituents 的量纲恒等(v=d/t, a=v/t)成立 → 全 ok."""
    res = validate_derived_dimensions(_PHOTOS)
    assert res["velocity"]["ok"] is True
    assert res["velocity"]["declared"] == "L1·T-1"
    assert res["velocity"]["expected"] == "L1·T-1"
    assert res["acceleration"]["ok"] is True
    assert res["acceleration"]["expected"] == "L1·T-2"


def test_cross_quantity_dimensional_identity_catches_contract_typo():
    """契约自身声明不自洽(velocity 单位误写为 time) → 量纲恒等式咬住. """
    bad = dict(_PHOTOS)
    bad["velocity"] = {"unit": "time",
                       "derived_from": ["distance"], "derived_denom": ["time_elapsed"]}
    res = validate_derived_dimensions(bad)
    assert res["velocity"]["ok"] is False
    assert res["velocity"]["declared"] == "T1"          # 被误写成 time
    assert res["velocity"]["expected"] == "L1·T-1"      # 但理论上必须是 distance/time


# ═══════════════ S4: law_model 卡片 ↔ C-Space 契约(自证法律一致) ═══════════════

class _EcoLaw():
    domain = "ecology_dynamics"

    def predict(self, *a):
        return type("P", (), {"as_dict": lambda self: {"x_star": 0.6, "period": 17.15}})()
    def law(self): return "dx/dt=a·x−b·x·y; dy/dt=c·x·y−d·y"
    def seed(self, *a): return None


def test_world_model_card_carries_domain_contract():
    """LawModel 治理卡片携带其域的科学契约(量纲/有效域)."""
    from huginn.research.law_model import world_model_card
    card = world_model_card(_EcoLaw())
    assert card["domain"] == "ecology_dynamics"
    assert card["contract"]["quantities"]["period_est"]["unit"] == "time"
    # 未登记域 → 契约空, 不阻断
    class NoDomain:
        domain = "no_such_domain"
        def predict(self, *a): return None
        def law(self): return ""
        def seed(self, *a): return None
    assert world_model_card(NoDomain())["contract"] == {}


def test_cspace_state_from_lawmodel_carries_contract_and_gates_itself():
    """C-Space.add_state 从 LawModel 灌状态 → Being 自带契约; contract_gate 自证:
    契约内 confirmed, 越域 self-rejected —— 不传显式 quantities 也有效."""
    cs = CSpace()

    def mk(obj):
        b = cs.add_state("st", _EcoLaw(), state=obj)
        b.payload["objectives"] = obj
        b.falsifiable = False
        return b

    good = mk({"period_est": 17.15, "x_star": 0.6})
    assert cb.promote_to_at_hand(cs, "st", corroborate=cb.contract_gate())["promoted"] is True
    good.falsifiable = False

    bad = mk({"period_est": -1.0, "x_star": 0.6})
    r = cb.promote_to_at_hand(cs, "st", corroborate=cb.contract_gate())
    assert r["promoted"] is False and bad.payload["scientific_contract"]["quantities"]["period_est"]["unit"] == "time"