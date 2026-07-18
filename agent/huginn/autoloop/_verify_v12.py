"""v12 验证脚本 — AlphaEvolve crossover + 同 dim 标记 + constraint hard-veto.

3 项独立改动, 每项有独立测试. 全套通过 = v12 验收.
ponytail: 跟 _verify_v11 同范式, assert + print, 无 pytest 依赖.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace as _NS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def test_v12_p0_crossover_method() -> None:
    """P0: crossover 方法存在, 无 model 返回 None."""
    from huginn.autoloop.hypothesis_loop import HypothesisGraph
    import inspect

    assert hasattr(HypothesisGraph, "crossover"), "缺 crossover 方法"
    sig = inspect.signature(HypothesisGraph.crossover)
    assert list(sig.parameters)[:4] == ["self", "parent_a_id", "parent_b_id", "model"], \
        f"crossover 签名不对: {sig}"

    # 无 model (None) → 返回 None
    g = HypothesisGraph()
    a = g.add_hypothesis("假设 A: Ca/Si ratio affects strength")
    b = g.add_hypothesis("假设 B: temperature affects hydration rate")
    # _is_real_model(None) 会 False, 但 None 没 _mock_name 属性... 用 mock object
    class _Mock:
        _mock_name = "mock"
    assert g.crossover(a, b, _Mock()) is None, "mock model 应返回 None"
    print("v12 P0 crossover method: PASS")


def test_v12_p0_crossover_with_fake_model() -> None:
    """P0: crossover 用 fake model 产生 child, 标 crossover_parents + candidate_role."""
    from huginn.autoloop.hypothesis_loop import HypothesisGraph

    class _FakeModel:
        def invoke(self, prompt):
            return _NS(content="组合假设: Ca/Si ratio 在高温下影响 strength via hydration rate")

    g = HypothesisGraph()
    a = g.add_hypothesis("Ca/Si ratio affects strength", rationale="化学计量控制")
    b = g.add_hypothesis("temperature affects hydration rate", rationale="热动力学")

    child_id = g.crossover(a, b, _FakeModel(), objective="understand C-S-H")
    assert child_id is not None, "crossover 应产生 child"
    child = g.get(child_id)
    assert child.evidence.get("crossover_parents") == [a, b], \
        f"child evidence 缺 crossover_parents: {child.evidence}"
    assert child.evidence.get("candidate_role") == "crossover", \
        f"child evidence 缺 candidate_role=crossover: {child.evidence}"
    assert child.parent_id == a, "child 应通过 derive 边连 parent_a"

    # child 同质 (跟 a 或 b 同 statement) → 返回 None
    class _SameAsA:
        def invoke(self, prompt):
            return _NS(content=g.get(a).statement)
    assert g.crossover(a, b, _SameAsA()) is None, "同质 child 应返回 None"
    print("v12 P0 crossover with fake model: PASS")


def test_v12_p0_pivot_triggers_crossover() -> None:
    """P0: pivot 末尾自动调 crossover, sibling_group 有 3 候选 (主+备+crossover)."""
    from huginn.autoloop.hypothesis_loop import HypothesisGraph

    class _FakeModel:
        """主候选 / 备候选 / crossover child 三次 invoke 返回不同内容."""
        def __init__(self):
            self.calls = 0
        def invoke(self, prompt):
            self.calls += 1
            if self.calls == 1:
                return _NS(content="主候选: defect density drives diffusion")
            elif self.calls == 2:
                return _NS(content="备候选: lattice strain modulates transport")
            else:  # crossover
                return _NS(content="crossover: defect density + lattice strain 共同决定扩散")
        def bind(self, temperature=1.0):
            return self  # 测试用, 真实 model bind 返回新 model

    g = HypothesisGraph()
    failed = g.add_hypothesis("原假设: X controls Y")
    g.refute(failed, {"errors": "实验失败"})

    new_id = g.pivot(failed, {"errors": "fail"}, model=_FakeModel(), objective="test")
    assert new_id is not None, "pivot 应返回主候选 id"

    # sibling_group 有 3 候选 (主 + 备 + crossover)
    main_node = g.get(new_id)
    assert main_node.sibling_group_id is not None, "主候选应有 sibling_group_id"
    siblings = g.siblings(new_id)
    assert len(siblings) == 2, f"sibling_group 应有 2 个兄弟 (备+crossover), 实际 {len(siblings)}"

    # crossover child 标 candidate_role=crossover
    _roles = [s.evidence.get("candidate_role") for s in siblings]
    assert "backup" in _roles, f"缺 backup 候选: {_roles}"
    assert "crossover" in _roles, f"缺 crossover 候选: {_roles}"
    print("v12 P0 pivot triggers crossover: PASS")


def test_v12_p1a_dim_conflict_marking() -> None:
    """P1-a: _record_backup_candidates 同 dim 第 2 候选标 dim_conflict=True."""
    from huginn.autoloop.engine import AutoloopEngine
    import inspect

    src = inspect.getsource(AutoloopEngine._record_backup_candidates)
    assert "dim_conflict" in src, "_record_backup_candidates 缺 dim_conflict 标记"
    assert "_dim_conflict = _dim in _seen_dims" in src, "缺 dim_conflict 判定逻辑"

    # 模拟解析: 2 个同 dim 候选, 第 2 个应标 dim_conflict=True
    # 直接调 _record_backup_candidates 需要完整 engine, 用单元测试方式
    raw = (
        "[DIM: composition] Ca/Si ratio affects strength | pro: ... | con: ...\n"
        "[DIM: composition] Al2O3 doping also affects strength | pro: ... | con: ...\n"
        "[DIM: temperature] thermal annealing changes phase | pro: ... | con: ...\n"
    )

    # 构造最小 engine mock, 只需要 hypothesis_graph
    class _MinimalEngine:
        pass

    from huginn.autoloop.hypothesis_loop import HypothesisGraph
    eng = _MinimalEngine()
    eng.hypothesis_graph = HypothesisGraph()

    # selected 是空字符串, 让所有候选都进图
    AutoloopEngine._record_backup_candidates(eng, raw, "")

    nodes = eng.hypothesis_graph.all_nodes()
    assert len(nodes) == 3, f"应有 3 个 backup 候选进图, 实际 {len(nodes)}"

    # 找 composition dim 的 2 个候选
    _comp_nodes = [n for n in nodes if n.dimension == "composition"]
    assert len(_comp_nodes) == 2, f"composition dim 应有 2 候选, 实际 {len(_comp_nodes)}"

    # 第 2 个 composition 候选标 dim_conflict=True
    _conflict_count = sum(1 for n in _comp_nodes if n.evidence.get("dim_conflict"))
    assert _conflict_count == 1, \
        f"composition dim 应有 1 个 dim_conflict=True, 实际 {_conflict_count}"

    # temperature dim 的候选 dim_conflict=False
    _temp_nodes = [n for n in nodes if n.dimension == "temperature"]
    assert len(_temp_nodes) == 1
    assert _temp_nodes[0].evidence.get("dim_conflict") is False, \
        "首个 temperature 候选 dim_conflict 应为 False"
    print("v12 P1-a dim_conflict marking: PASS")


def test_v12_p1b_constraint_hard_veto() -> None:
    """P1-b: MathEvidenceChecker 对 constraint_check.violations 非空 hard-veto."""
    from huginn.autoloop.phase_gate import MathEvidenceChecker
    c = MathEvidenceChecker()

    # violations 非空 list → hard-veto
    passed, feedback, details = c({
        "constraint_check": {"violations": ["E > 0 required, got -1.2"], "all_passed": False}
    })
    assert passed is False, "constraint violations 非空应 hard-veto"
    assert "hard-veto" in feedback, f"feedback 缺 hard-veto: {feedback}"
    assert "constraint" in feedback, f"feedback 缺 constraint: {feedback}"
    assert details.get("hard_veto") == "constraint_violations", \
        f"details 缺 hard_veto=constraint_violations: {details}"

    # violations 空列表 → 不触发 hard-veto, 走原 DS (no masses → pass)
    passed, _, details = c({"constraint_check": {"violations": [], "all_passed": True}})
    assert passed is True, "空 violations 不应 hard-veto"
    assert "hard_veto" not in details, "空 violations 不应标 hard_veto"

    # violations 缺失 → 不触发
    passed, _, _ = c({"constraint_check": {"all_passed": True}})
    assert passed is True, "violations 缺失不应 hard-veto"

    # constraint_check 不是 dict → 不触发
    passed, _, _ = c({"constraint_check": True})
    assert passed is True, "constraint_check 非 dict 不应 hard-veto"

    # 空 evidence → 不触发
    passed, _, _ = c({})
    assert passed is True, "空 evidence 不应 hard-veto"

    # violations 是字符串 → truthy 触发
    passed, _, _ = c({"constraint_check": {"violations": "mass conservation violated"}})
    assert passed is False, "violations 非空字符串应 hard-veto"
    print("v12 P1-b constraint hard-veto: PASS")


def main() -> None:
    tests = [
        test_v12_p0_crossover_method,
        test_v12_p0_crossover_with_fake_model,
        test_v12_p0_pivot_triggers_crossover,
        test_v12_p1a_dim_conflict_marking,
        test_v12_p1b_constraint_hard_veto,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            failed += 1
            import traceback
            traceback.print_exc()
            print(f"FAIL: {t.__name__}: {e}", file=sys.stderr)
    print(f"\nv12 验证: {passed}/{len(tests)} passed, {failed} failed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
