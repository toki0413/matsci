"""phase_recompute 相位门 + 增量边重算 测试.

锚定 Ghidra 启发落地的两个原语:
  1. 相位级全序门: 缺省逐层自增(层义不变/默认零行为变化), 自定义映射合法.
  2. 增量边重算: 下游可达闭包确定性; 重算只落"上游相位之后"子集; 无下游→空.
红线: 纯函数、只算范围、不伪造证据、默认零行为变化.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

# 直接加载 phase_recompute 模块文件, 不经过 huginn/__init__ —— 避免被包级重依赖
# (langchain/cryptography) 绑架, 让纯函数测试保持轻量与确定性.
_spec = importlib.util.spec_from_file_location(
    "phase_recompute",
    Path(__file__).resolve().parents[1] / "huginn/research/phase_recompute.py",
)
phase_recompute = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(phase_recompute)

assign_phases = phase_recompute.assign_phases
phase_total_order = phase_recompute.phase_total_order
downstream_closure = phase_recompute.downstream_closure
recompute_plan = phase_recompute.recompute_plan


# ── 1. 相位级全序门 ──────────────────────────────────────

def test_default_phase_preserves_layer_identity():
    """缺省相位逐层自增 → 层义不变 (默认零行为变化)."""
    layers = [["A"], ["B", "C"], ["D"], ["E"]]
    ph = assign_phases(layers)
    assert ph == {"A": 0, "B": 1, "C": 1, "D": 2, "E": 3}
    assert phase_total_order(ph) == [0, 1, 2, 3]


def test_custom_phase_total_order():
    """自定义相位映射合法, 相位全序去重升序."""
    layers = [["A"], ["B", "C"], ["D"], ["E"]]
    ph = assign_phases(layers, {0: 1, 1: 2, 2: 4, 3: 4})
    assert ph["A"] == 1 and ph["B"] == 2 and ph["D"] == 4
    assert phase_total_order(ph) == [1, 2, 4]


# ── 2. 增量边重算 ────────────────────────────────────────

def test_downstream_closure():
    deps = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D"), ("D", "E")]
    assert downstream_closure(deps, ["B"]) == {"D", "E"}   # 传递下游, 不含 C(不分叉)
    assert downstream_closure(deps, ["D"]) == {"E"}
    assert downstream_closure(deps, ["A"]) == {"B", "C", "D", "E"}  # 不含自身


def test_recompute_only_after_upstream_phase():
    """重算仅落在上游相位之后的下游子集: 不回溯已结算基础层, 同层不误伤."""
    deps = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D"), ("D", "E")]
    layers = [["A"], ["B", "C"], ["D"], ["E"]]
    rp = recompute_plan(deps, layers, ["C"])
    assert rp["affected_set"] == ["D", "E"]
    assert rp["recompute"] == ["D", "E"]
    rp2 = recompute_plan(deps, layers, ["B"])
    assert rp2["recompute"] == ["D", "E"]  # B 同层 C 不回溯
    assert rp2["note"]


def test_recompute_no_downstream_is_empty():
    """无下游的变更 → 空重算 (不误伤)."""
    deps = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D"), ("D", "E")]
    layers = [["A"], ["B", "C"], ["D"], ["E"]]
    rp = recompute_plan(deps, layers, ["E"])
    assert rp["affected_set"] == [] and rp["recompute"] == []


def test_recompute_deterministic():
    """纯函数确定性: 同输入两次结果一致 (可证伪)."""
    deps = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D"), ("D", "E")]
    layers = [["A"], ["B", "C"], ["D"], ["E"]]
    assert recompute_plan(deps, layers, ["A"]) == recompute_plan(deps, layers, ["A"])


def test_closure_does_not_fabricate():
    """受影响范围只由 DAG 边推出, 不引用任何实验结果/不做质量判断."""
    deps = [("A", "B")]                       # 只有一条边
    layers = [["A"], ["B"], ["C"]]            # C 与 A/B 无依赖
    rp = recompute_plan(deps, layers, ["A"])
    assert rp["affected_set"] == ["B"]
    assert "C" not in rp["affected_set"]       # 无依赖即不受影响
    assert "C" not in rp["recompute"]