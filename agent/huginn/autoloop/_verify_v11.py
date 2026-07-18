"""v11 验证脚本 — N-best pivot + Hard-veto + 假设聚焦 + FDE 对齐.

4 项独立改动并行进 v11, 每项有独立测试. 全套通过 = v11 验收.
ponytail: 不新建测试框架, 用 assert + print, 跟 _verify_av/_verify_f 同范式.
"""
from __future__ import annotations

import sys
from pathlib import Path

# 让 huginn 可 import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def test_v11_p0a_dimension_field() -> None:
    """P0-a: HypothesisNode 有 dimension 字段, add_hypothesis 自动抽 dimension."""
    from huginn.autoloop.hypothesis_loop import (
        HypothesisGraph, HypothesisNode, _extract_dimension,
    )
    # 字段存在
    import dataclasses
    fields = {f.name for f in dataclasses.fields(HypothesisNode)}
    assert "dimension" in fields, "HypothesisNode 缺 dimension 字段"
    assert "sibling_group_id" in fields, "HypothesisNode 缺 sibling_group_id 字段"

    # 关键词命中 (ponytail: 测试字符串需避免子串误伤, 如 "concentration" 含 "ratio")
    assert _extract_dimension("Ca/Si ratio affects diffusion") == "composition"
    assert _extract_dimension("温度依赖性") == "temperature"
    assert _extract_dimension("vacancy density") == "defect"
    assert _extract_dimension("crystal symmetry") == "structure"
    assert _extract_dimension("diffusion coefficient") == "transport"
    assert _extract_dimension("random statement") == ""

    # add_hypothesis 自动抽 dimension
    g = HypothesisGraph()
    hid = g.add_hypothesis("Ca/Si ratio affects diffusion")
    assert g.get(hid).dimension == "composition", "add_hypothesis 未自动抽 dimension"

    # to_dict / from_dict 往返
    d = g.get(hid).to_dict()
    assert "dimension" in d and "sibling_group_id" in d
    n2 = HypothesisNode.from_dict(d)
    assert n2.dimension == "composition"
    assert n2.sibling_group_id is None
    print("v11 P0-a dimension field: PASS")


def test_v11_p0a_cluster_by_dimension() -> None:
    """P0-a: cluster_by_dimension 方法存在, 正确分组."""
    from huginn.autoloop.hypothesis_loop import HypothesisGraph
    g = HypothesisGraph()
    g.add_hypothesis("Ca/Si ratio affects diffusion")  # composition
    g.add_hypothesis("temperature dependence of X")     # temperature
    g.add_hypothesis("Al doping effect")                # composition (doping)
    g.add_hypothesis("random statement")                # unknown
    clusters = g.cluster_by_dimension()
    assert "composition" in clusters and len(clusters["composition"]) == 2
    assert "temperature" in clusters and len(clusters["temperature"]) == 1
    assert "unknown" in clusters and len(clusters["unknown"]) == 1
    print("v11 P0-a cluster_by_dimension: PASS")


def test_v11_p0a_siblings() -> None:
    """P0-a: siblings 方法存在, 无 sibling_group_id 返回空."""
    from huginn.autoloop.hypothesis_loop import HypothesisGraph
    g = HypothesisGraph()
    hid = g.add_hypothesis("Ca/Si ratio affects diffusion")
    # 无 sibling_group_id
    assert g.siblings(hid) == []
    # 手动设 sibling_group_id
    g.get(hid).sibling_group_id = "sg_test"
    hid2 = g.add_hypothesis("another hypothesis")
    g.get(hid2).sibling_group_id = "sg_test"
    sibs = g.siblings(hid)
    assert len(sibs) == 1 and sibs[0].id == hid2
    print("v11 P0-a siblings: PASS")


def test_v11_p0b_pivot_n_best() -> None:
    """P0-b: pivot 接受 n_best 参数, 无 model 时单候选 (template)."""
    from huginn.autoloop.hypothesis_loop import HypothesisGraph
    import inspect
    sig = inspect.signature(HypothesisGraph.pivot)
    assert "n_best" in sig.parameters, "pivot 缺 n_best 参数"
    assert sig.parameters["n_best"].default == 2, "n_best 默认值不是 2"

    # 无 model 时只产生 1 候选 (template_pivot), siblings 为空
    g = HypothesisGraph()
    hid = g.add_hypothesis("Ca/Si ratio affects diffusion")
    g.refute(hid, {"errors": "test failed"})
    new_id = g.pivot(hid, {"errors": "fail"}, model=None)
    assert new_id is not None
    assert g.siblings(new_id) == [], "无 model 时不应有 backup 候选"
    print("v11 P0-b pivot n_best: PASS")


def test_v11_p1a_hard_veto() -> None:
    """P1-a: MathEvidenceChecker hard-veto 短路."""
    from huginn.autoloop.phase_gate import MathEvidenceChecker
    c = MathEvidenceChecker()

    # conservation_law.verified == False → hard-veto (feedback 用空格 "conservation law")
    passed, feedback, details = c({"conservation_law": {"verified": False}})
    assert passed is False, "conservation_law violated 应 hard-veto"
    assert "hard-veto" in feedback, "feedback 应含 hard-veto 标记"
    assert "conservation" in feedback, f"feedback 应含 conservation: {feedback}"
    assert details.get("hard_veto") == "conservation_law", f"details 应含 hard_veto=conservation_law: {details}"

    # dimensional_consistent == False → hard-veto
    passed, feedback, _ = c({"dimensional_consistent": False})
    assert passed is False, "dimensional_inconsistent 应 hard-veto"
    assert "hard-veto" in feedback
    assert "dimensional" in feedback

    # conservation_law.verified == True (但无其他 masses) → 走原 DS, no masses → pass
    passed, _, _ = c({"conservation_law": {"verified": True}})
    assert passed is True, "verified=True 不应 hard-veto"

    # verified == None (缺失) → 不触发 hard-veto, 走原 DS.
    # DS 把 bool(None)=False 当 fail, passed=False (非 hard-veto 路径)
    passed, _, details = c({"conservation_law": {"verified": None}})
    assert passed is False, "verified=None 走 DS, bool(None)=False → fail"
    assert "hard_veto" not in details, "verified=None 不应 hard-veto"

    # 空 evidence → 不触发 hard-veto
    passed, _, _ = c({})
    assert passed is True, "空 evidence 不应 hard-veto"
    print("v11 P1-a hard-veto: PASS")


def test_v11_p1b_fde_checkpoint() -> None:
    """P1-b: _maybe_clarify 接受 hypothesize_align checkpoint."""
    import inspect
    from huginn.autoloop.engine import AutoloopEngine
    # _maybe_clarify 存在, checkpoint 是字符串参数
    sig = inspect.signature(AutoloopEngine._maybe_clarify)
    assert "checkpoint" in sig.parameters

    # 读源码确认 hypothesize_align 分支存在
    src = inspect.getsource(AutoloopEngine._maybe_clarify)
    assert 'checkpoint == "hypothesize_align"' in src, "缺 hypothesize_align 分支"
    assert "cluster_by_dimension" in src, "hypothesize_align 未用 cluster_by_dimension"
    assert "_speculator_hint" in src, "hypothesize_align 未 append 到 _speculator_hint"

    # execute_fn 的 hypothesize 分支调 _maybe_clarify
    src_engine = inspect.getsource(AutoloopEngine)
    assert 'self._maybe_clarify(' in src_engine
    assert '"hypothesize_align"' in src_engine
    print("v11 P1-b FDE checkpoint: PASS")


def test_v11_p0a_hypothesis_prompt_dimension_constraint() -> None:
    """P0-a: _build_hypothesis_prompt 含维度约束 block."""
    import inspect
    from huginn.autoloop.engine import AutoloopEngine
    src = inspect.getsource(AutoloopEngine._build_hypothesis_prompt)
    assert "[DIM:" in src, "prompt 缺 [DIM:] 标签要求"
    assert "composition" in src and "temperature" in src
    assert "defect" in src and "structure" in src and "transport" in src
    assert "cluster_block" in src, "topology_block 未升级为 cluster_block"
    print("v11 P0-a hypothesis prompt dimension constraint: PASS")


def test_v11_p0a_record_backup_candidates() -> None:
    """P0-a: _record_backup_candidates 方法存在, 解析 [DIM:] 候选."""
    import inspect
    from huginn.autoloop.engine import AutoloopEngine
    assert hasattr(AutoloopEngine, "_record_backup_candidates"), "缺 _record_backup_candidates"
    src = inspect.getsource(AutoloopEngine._record_backup_candidates)
    assert "[DIM:" in src or r"\[DIM:" in src, "未解析 [DIM:] 标签"
    assert "add_hypothesis" in src, "未调 add_hypothesis"
    print("v11 P0-a _record_backup_candidates: PASS")


def main() -> None:
    tests = [
        test_v11_p0a_dimension_field,
        test_v11_p0a_cluster_by_dimension,
        test_v11_p0a_siblings,
        test_v11_p0b_pivot_n_best,
        test_v11_p1a_hard_veto,
        test_v11_p1b_fde_checkpoint,
        test_v11_p0a_hypothesis_prompt_dimension_constraint,
        test_v11_p0a_record_backup_candidates,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"FAIL: {t.__name__}: {e}", file=sys.stderr)
    print(f"\nv11 验证: {passed}/{len(tests)} passed, {failed} failed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
