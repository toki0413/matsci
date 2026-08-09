"""huginn.metacog 子包的 smoke import 测试.

覆盖 huginn/metacog/ 下所有 .py 模块 (除 __init__.py):
  - 每个模块一个 test_import_<module_name>, 验证可正常 import.
  - 对有明确 class/function 的模块, 额外补 1-2 个轻量实例化/调用测试
    (dataclass 构造、常量访问、空输入调用等).

依赖说明 (本测试环境 = pytest 用的 Python 3.14):
  - numpy / pydantic 已安装, 绝大多数模块直接可 import.
  - 仅 alignment 依赖 sklearn (未安装), 用 pytest.importorskip("sklearn") 跳过.
  - 部分模块 (latent_space 的 LatentSpace / haptic 的 ElasticTensor 等) 是
    抽象基类或需重型输入, 只做 import + 公开 API 存在性检查, 不强行实例化.

只读不写源码, 不修改 huginn/ 下任何文件.
"""

from __future__ import annotations

import pytest


# ── 1. alignment (sklearn) ─────────────────────────────────────


def test_import_alignment():
    # alignment 直接 import sklearn, 缺失时跳过
    pytest.importorskip("sklearn")
    import huginn.metacog.alignment  # noqa: F401


# ── 2. alignment_dataset ───────────────────────────────────────


def test_import_alignment_dataset():
    import huginn.metacog.alignment_dataset  # noqa: F401


def test_alignment_dataset_construct():
    from huginn.metacog.alignment_dataset import AlignmentDataset

    ds = AlignmentDataset()  # 无参默认构造
    assert ds is not None


# ── 3. blind_spot_mapper ───────────────────────────────────────


def test_import_blind_spot_mapper():
    import huginn.metacog.blind_spot_mapper  # noqa: F401


def test_blind_spot_construct():
    # BlindSpot 是 dataclass, 能用必填字段正常构造
    from huginn.metacog.blind_spot_mapper import BlindSpot

    bs = BlindSpot(skill="vasp_tool", why_blind="no spin-polarization check")
    assert bs.skill == "vasp_tool"
    assert bs.priority == "medium"  # 默认值


# ── 4. block_registry ──────────────────────────────────────────


def test_import_block_registry():
    import huginn.metacog.block_registry  # noqa: F401


def test_block_registry_construct():
    from huginn.metacog.block_registry import BlockRegistry, BlockedRoute

    route = BlockedRoute(
        route_id="r1", method_family="gp", block_reason="circular"
    )
    assert route.route_id == "r1"
    assert route.status == "blocked"
    reg = BlockRegistry()
    assert isinstance(reg, BlockRegistry)


# ── 5. branch_incubator ────────────────────────────────────────


def test_import_branch_incubator():
    import huginn.metacog.branch_incubator  # noqa: F401


def test_branch_result_construct():
    from huginn.metacog.branch_incubator import BranchResult

    res = BranchResult(family_id="gp", agent_id="a1", hypothesis="h")
    assert res.family_id == "gp"
    assert res.success is True  # 默认值


# ── 6. category_functor ────────────────────────────────────────


def test_import_category_functor():
    import huginn.metacog.category_functor  # noqa: F401


def test_category_functor_construct():
    from huginn.metacog.category_functor import Category, Morphism

    m = Morphism(src="a", dst="b", relation="maps_to")
    assert m.src == "a"
    cat = Category(name="cat", objects=["a", "b"], morphisms=[m])
    assert cat.name == "cat"
    assert cat.objects == ["a", "b"]


# ── 7. cognitive_heat_engine ───────────────────────────────────


def test_import_cognitive_heat_engine():
    import huginn.metacog.cognitive_heat_engine  # noqa: F401


def test_heat_engine_defaults():
    from huginn.metacog.cognitive_heat_engine import (
        CognitiveHeatEngine,
        get_heat_engine,
    )

    eng = CognitiveHeatEngine()
    assert eng.T_hot == 0.0  # 默认冷启动
    shared = get_heat_engine()
    assert isinstance(shared, CognitiveHeatEngine)


# ── 8. completion_auditor ──────────────────────────────────────


def test_import_completion_auditor():
    import huginn.metacog.completion_auditor  # noqa: F401


def test_completion_checklist_and_parse():
    from huginn.metacog.completion_auditor import (
        CompletionChecklist,
        parse_unexplored_declaration,
    )

    cl = CompletionChecklist()
    assert cl.effort_floor_passed is False
    n = parse_unexplored_declaration("there are 3 unexplored directions")
    assert isinstance(n, int)


# ── 9. completion_gate ─────────────────────────────────────────


def test_import_completion_gate():
    import huginn.metacog.completion_gate  # noqa: F401


def test_completion_gate_dataclasses():
    from huginn.metacog.completion_gate import GateContext, GateDecision

    ctx = GateContext(
        iteration=1, max_iterations=10, families_explored=2, live_components=2
    )
    assert ctx.iteration == 1
    dec = GateDecision(status="continue")
    assert dec.status == "continue"
    assert dec.should_stop is False


# ── 10. composite_token_experiment ─────────────────────────────


def test_import_composite_token_experiment():
    import huginn.metacog.composite_token_experiment  # noqa: F401


def test_composite_token_construct():
    # CompositeToken 需要 numpy ndarray 坐标
    np = pytest.importorskip("numpy")
    from huginn.metacog.composite_token_experiment import CompositeToken

    tok = CompositeToken(text="A", coords=np.array([[0.0, 0.0, 0.0]]))
    assert tok.text == "A"


# ── 11. context_isolation ──────────────────────────────────────


def test_import_context_isolation():
    import huginn.metacog.context_isolation  # noqa: F401


def test_context_isolation_basic():
    from huginn.metacog.context_isolation import ContextBundle, isolate

    bundle = ContextBundle(
        task_definition="predict gap",
        current_preferred_hypothesis="GPR",
    )
    assert bundle.is_exploratory is False
    # exploration 角色看不到 current_preferred_hypothesis (隔离生效)
    exp_ctx = isolate(bundle, "exploration")
    assert "current_preferred_hypothesis" not in exp_ctx
    assert "task_definition" in exp_ctx


# ── 12. cross_domain_pipeline ──────────────────────────────────


def test_import_cross_domain_pipeline():
    import huginn.metacog.cross_domain_pipeline  # noqa: F401


def test_cross_domain_reframe_no_model():
    # 无 model 时返回 None 而不抛异常 (优雅降级)
    from huginn.metacog.cross_domain_pipeline import cross_domain_reframe

    res = cross_domain_reframe("some problem", model=None)
    assert res is None


# ── 13. cross_task_store ───────────────────────────────────────


def test_import_cross_task_store():
    import huginn.metacog.cross_task_store  # noqa: F401


def test_cross_task_store_construct(tmp_path):
    from huginn.metacog.cross_task_store import CrossTaskStore

    store = CrossTaskStore(db_path=tmp_path / "cts.db")
    assert store is not None


# ── 14. depth_search ───────────────────────────────────────────


def test_import_depth_search():
    import huginn.metacog.depth_search  # noqa: F401


def test_min_effort_floor_defaults():
    from huginn.metacog.depth_search import (
        MinEffortFloor,
        PrematureConvergenceDetector,
    )

    floor = MinEffortFloor()
    assert floor.min_iterations == 3
    assert floor.min_method_families == 3
    det = PrematureConvergenceDetector()
    assert det is not None


# ── 15. encounter_space ────────────────────────────────────────


def test_import_encounter_space():
    import huginn.metacog.encounter_space  # noqa: F401


def test_encounter_space_construct():
    from huginn.metacog.encounter_space import EncounterSpace

    # EncounterSpace 可无参构造 (LatentSpace 是抽象基类, 不在此实例化)
    space = EncounterSpace()
    assert space is not None


# ── 16. episodic_index ─────────────────────────────────────────


def test_import_episodic_index():
    import huginn.metacog.episodic_index  # noqa: F401


def test_episodic_index_store_construct(tmp_path):
    from huginn.metacog.episodic_index import EpisodicIndexStore

    store = EpisodicIndexStore(workspace=tmp_path, task_id="t1")
    assert store is not None


# ── 17. episodic_replay ────────────────────────────────────────


def test_import_episodic_replay():
    import huginn.metacog.episodic_replay  # noqa: F401


def test_episodic_replay_construct(tmp_path):
    from huginn.metacog.episodic_index import EpisodicIndexStore
    from huginn.metacog.episodic_replay import EpisodicReplay

    replay = EpisodicReplay(
        index_store=EpisodicIndexStore(workspace=tmp_path, task_id="t1")
    )
    assert replay is not None


# ── 18. equivalence_auditor ────────────────────────────────────


def test_import_equivalence_auditor():
    import huginn.metacog.equivalence_auditor  # noqa: F401


def test_equivalence_auditor_basic():
    from huginn.metacog.equivalence_auditor import (
        EquivalenceAuditor,
        EquivalenceVerdict,
    )

    auditor = EquivalenceAuditor()  # 无 model 也能构造
    assert auditor is not None
    verdict = EquivalenceVerdict(verdict="novel")
    assert verdict.verdict == "novel"


# ── 19. failure_modes ──────────────────────────────────────────


def test_import_failure_modes():
    import huginn.metacog.failure_modes  # noqa: F401


def test_failure_modes_registry():
    from huginn.metacog.failure_modes import (
        DEFAULT_REGISTRY,
        FailureMode,
        FailureModeRegistry,
    )

    mode = FailureMode(
        id="custom", category="data", description="d", severity="warn"
    )
    assert mode.id == "custom"
    reg = FailureModeRegistry()
    assert isinstance(reg.all(), list) and len(reg.all()) > 0
    assert reg.by_id("data-leakage-structural") is not None
    assert DEFAULT_REGISTRY is not None


# ── 20. haptic_descriptor ──────────────────────────────────────


def test_import_haptic_descriptor():
    import huginn.metacog.haptic_descriptor  # noqa: F401


def test_haptic_descriptor_construct():
    from huginn.metacog.haptic_descriptor import HapticDescriptor, HapticPropertyLayer

    # 全部字段可选, 无参构造
    assert HapticDescriptor() is not None
    layer = HapticPropertyLayer()
    assert layer.source == "DFT"  # 默认来源
    assert layer.confidence == 1.0


# ── 21. haptic_property_layer ──────────────────────────────────


def test_import_haptic_property_layer():
    import huginn.metacog.haptic_property_layer  # noqa: F401


def test_haptic_property_layer_defaults():
    from huginn.metacog.haptic_property_layer import HapticPropertyLayer

    layer = HapticPropertyLayer()
    assert layer.source == "DFT"
    assert layer.elastic is None  # 默认无张量


# ── 22. hypothesis_manifold ────────────────────────────────────


def test_import_hypothesis_manifold():
    import huginn.metacog.hypothesis_manifold  # noqa: F401


def test_hypothesis_manifold_basic():
    from huginn.metacog.hypothesis_manifold import (
        Hypothesis,
        HypothesisManifold,
        Observation,
    )

    h = Hypothesis(h_id="h1", description="d")
    assert h.h_id == "h1"
    obs = Observation(name="energy", value=-1.0)
    assert obs.name == "energy"
    man = HypothesisManifold()
    assert man is not None


def test_mcmc_multi_chain_processpool_parallel():
    """升级路径 A: ProcessPool 真并行多链 + gate 回退 asyncio.

    验证: 默认走 ProcessPool; HUGINN_MCMC_PARALLEL=0 回退 asyncio;
    两条路径结果一致性 + R̂ 可计算.
    """
    import asyncio
    import math
    import os

    from huginn.metacog.hypothesis_manifold import (
        Hypothesis,
        HypothesisManifold,
        Observation,
    )

    obs = [Observation(name="accuracy", value=0.92, sigma=0.1)]

    def build():
        m = HypothesisManifold()
        for hid, desc, scale, n in (
            ("h_a", "Hypothesis A", 1.0, 2),
            ("h_b", "Hypothesis B", 0.5, 3),
            ("h_c", "Hypothesis C", 0.0, 1),
        ):
            try:
                m.add(Hypothesis(hid, desc, predictions={"accuracy": 0.92 * scale}, n_params=n))
            except ValueError:
                pass
        return m

    async def run(parallel):
        os.environ["HUGINN_MCMC_PARALLEL"] = parallel
        return await build().mcmc_multi_chain(
            obs, n_chains=4, n_steps_per_chain=20000,
            checkpoint_interval=0, anneal=True, t_high=10.0,
            global_proposal_prob=0.3,
        )

    pp = asyncio.run(run("1"))
    al = asyncio.run(run("0"))

    # 两条路径都应跑出 4 条链
    assert len(pp["chains"]) == 4, f"ProcessPool chains: {len(pp['chains'])}"
    assert len(al["chains"]) == 4, f"asyncio chains: {len(al['chains'])}"
    # R̂ 可计算且收敛 (尖锐后验 + 全局混合 + 退火)
    assert not math.isnan(pp["r_hat"]), f"ProcessPool r_hat nan: {pp['r_hat']}"
    assert pp["r_hat"] < 1.1, f"ProcessPool not converged: {pp['r_hat']}"
    # 一致性: 相同种子 + 相同逻辑, 接受率高度接近
    for a, b in zip(sorted(pp["accept_rates"]), sorted(al["accept_rates"])):
        assert abs(a - b) < 0.02, f"path mismatch: pp={pp['accept_rates']} asyncio={al['accept_rates']}"


# ── 23. imagination ────────────────────────────────────────────


def test_import_imagination():
    import huginn.metacog.imagination  # noqa: F401


def test_imagination_helpers():
    from huginn.metacog.hypothesis_manifold import Hypothesis
    from huginn.metacog.imagination import (
        check_falsifiability,
        detect_stagnation,
        imagine,
    )

    h = Hypothesis(h_id="h", description="d", predictions={"y": 1.0})
    # 无 model 时 imagine 优雅返回 None, 不抛
    assert imagine(h, "algebraic", model=None) is None
    assert isinstance(check_falsifiability(h, model=None), bool)
    assert isinstance(detect_stagnation([], N=3), bool)


# ── 24. latent_space ───────────────────────────────────────────


def test_import_latent_space():
    import huginn.metacog.latent_space  # noqa: F401


def test_latent_space_class_exported():
    import inspect

    from huginn.metacog import latent_space as lsmod

    # LatentSpace 是抽象基类 (有 abstract encode), 仅校验导出, 不实例化
    assert inspect.isclass(lsmod.LatentSpace)


# ── 25. llm_likelihood ─────────────────────────────────────────


def test_import_llm_likelihood():
    import huginn.metacog.llm_likelihood  # noqa: F401


def test_llm_likelihood_flag():
    from huginn.metacog.llm_likelihood import is_llm_likelihood_enabled

    assert isinstance(is_llm_likelihood_enabled(), bool)


# ── 26. mental_imagery ─────────────────────────────────────────


def test_import_mental_imagery():
    import huginn.metacog.mental_imagery  # noqa: F401


def test_mental_imagery_sketch_returns_bytes():
    from huginn.metacog.mental_imagery import sketch

    # sketch 在无渲染后端时优雅返回空 bytes, 不抛
    out = sketch("a box")
    assert isinstance(out, (bytes, bytearray))


# ── 27. method_registry ────────────────────────────────────────


def test_import_method_registry():
    import huginn.metacog.method_registry  # noqa: F401


def test_method_registry_basic():
    from huginn.metacog.method_registry import MethodFamily, MethodRegistry

    fam = MethodFamily(id="gp", essence="gaussian process")
    assert fam.id == "gp"
    reg = MethodRegistry()
    assert isinstance(reg.all(), list)
    assert reg.total_agents() == 0


# ── 28. persistence_landscape ──────────────────────────────────


def test_import_persistence_landscape():
    import huginn.metacog.persistence_landscape  # noqa: F401


def test_persistence_landscape_gudhi_flag():
    from huginn.metacog.persistence_landscape import is_gudhi_available

    # gudhi 是可选拓扑计算后端, 缺失时返回 False, 不抛
    assert isinstance(is_gudhi_available(), bool)


# ── 29. physical_structure ─────────────────────────────────────


def test_import_physical_structure():
    import huginn.metacog.physical_structure  # noqa: F401


def test_physical_structure_basic():
    from huginn.metacog.physical_structure import (
        PhysicalStructure,
        classify_implementation_gap,
    )

    ps = PhysicalStructure(relation_type="catalytic_geometry", relation_expr="x")
    assert ps.relation_type == "catalytic_geometry"
    gap = classify_implementation_gap(ps, {})
    assert isinstance(gap, str)


# ── 30. reflector ──────────────────────────────────────────────


def test_import_reflector():
    import huginn.metacog.reflector  # noqa: F401


def test_reflector_reflect_empty():
    from huginn.metacog.reflector import reflect
    from huginn.metacog.step_evaluator import ToolCallHealth

    actions = reflect(ToolCallHealth(), last_step_evaluations=None)
    assert isinstance(actions, list)


# ── 31. selection_bias ─────────────────────────────────────────


def test_import_selection_bias():
    import huginn.metacog.selection_bias  # noqa: F401


def test_selection_bias_basic():
    from huginn.metacog.selection_bias import (
        SelectionBiasVerdict,
        detect_selection_bias,
    )

    v = SelectionBiasVerdict()
    assert v.biased is False
    res = detect_selection_bias([])
    assert hasattr(res, "biased")


# ── 32. self_model ─────────────────────────────────────────────


def test_import_self_model():
    import huginn.metacog.self_model  # noqa: F401


def test_self_model_basic(tmp_path):
    from huginn.metacog.self_model import SelfModel, SkillRecord

    rec = SkillRecord(skill="vasp_tool", tier="core")
    assert rec.skill == "vasp_tool"
    model = SelfModel(
        task_local_path=tmp_path / "sm.json",
        cross_task_path=tmp_path / "sm_ct.db",
    )
    assert model is not None


# ── 33. sheaf_cohomology ───────────────────────────────────────


def test_import_sheaf_cohomology():
    import huginn.metacog.sheaf_cohomology  # noqa: F401


def test_sheaf_stalk_construct():
    from huginn.metacog.sheaf_cohomology import Stalk

    stalk = Stalk(source_id="find-1", claims={"energy": -1.0})
    assert stalk.source_id == "find-1"
    assert stalk.claims == {"energy": -1.0}


# ── 34. signal_hub ─────────────────────────────────────────────


def test_import_signal_hub():
    import huginn.metacog.signal_hub  # noqa: F401


def test_signal_hub_construct():
    from huginn.metacog.signal_hub import SignalHub

    hub = SignalHub()
    assert hub is not None


# ── 35. simplicial_homology ────────────────────────────────────


def test_import_simplicial_homology():
    import huginn.metacog.simplicial_homology  # noqa: F401


def test_simplicial_homology_gudhi_flag():
    from huginn.metacog.simplicial_homology import is_gudhi_available

    assert isinstance(is_gudhi_available(), bool)


# ── 36. step_evaluator ─────────────────────────────────────────


def test_import_step_evaluator():
    import huginn.metacog.step_evaluator  # noqa: F401


def test_step_evaluator_basic():
    from huginn.metacog.step_evaluator import (
        MeasurementUncertainty,
        ToolCallHealth,
        should_continue,
    )

    health = ToolCallHealth()
    assert health.success_rate == 1.0
    unc = MeasurementUncertainty()
    assert unc.propagated is False
    decision, reason = should_continue([], window=3)
    assert isinstance(decision, bool)
    assert isinstance(reason, str)


# ── 37. structure_cognitive_map ────────────────────────────────


def test_import_structure_cognitive_map():
    import huginn.metacog.structure_cognitive_map  # noqa: F401


def test_structure_cognitive_map_bond():
    from huginn.metacog.structure_cognitive_map import Bond, CoordinationShell

    b = Bond(i=0, j=1, length=1.5, bond_type="A-A")
    assert b.bond_type == "A-A"
    shell = CoordinationShell(
        center=0, neighbors=[1, 2], neighbor_distances=[1.5, 1.6], geometry="tet"
    )
    assert shell.center == 0


# ── 38. structure_descriptor ───────────────────────────────────


def test_import_structure_descriptor():
    import huginn.metacog.structure_descriptor  # noqa: F401


def test_structure_descriptor_construct():
    from huginn.metacog.structure_descriptor import StructureDescriptor

    sd = StructureDescriptor()  # 无参默认构造
    assert sd is not None


# ── 39. target_chain ───────────────────────────────────────────


def test_import_target_chain():
    import huginn.metacog.target_chain  # noqa: F401


def test_target_chain_basic():
    from huginn.metacog.target_chain import TargetChain, detect_drift

    tc = TargetChain(
        target_id="t1",
        target="goal",
        required_results=[],
        required_methods=[],
        required_data=[],
        verification="check",
    )
    assert tc.target_id == "t1"
    assert tc.progress == 0.0
    drifted, msg = detect_drift([], window=3)
    assert isinstance(drifted, bool)


# ── 40. three_cabin_reflector ──────────────────────────────────


def test_import_three_cabin_reflector():
    import huginn.metacog.three_cabin_reflector  # noqa: F401


def test_three_cabin_run_minimal(tmp_path):
    from huginn.metacog.three_cabin_reflector import run_three_cabin

    res = run_three_cabin({"goal": "g"}, [], 0, tmp_path)
    assert isinstance(res, tuple)


# ── 41. topology_lens ──────────────────────────────────────────


def test_import_topology_lens():
    import huginn.metacog.topology_lens  # noqa: F401


def test_topology_lens_basic():
    from huginn.metacog.topology_lens import (
        ClosureCheck,
        classify_system,
        hodge_signature,
        needs_downward_closure,
    )

    cc = ClosureCheck(needs_closure=True, reason="hypergraph")
    assert cc.needs_closure is True
    nc = needs_downward_closure([{"a", "b"}])
    assert hasattr(nc, "needs_closure")
    sig = hodge_signature(["a", "b"], [("a", "b")])
    assert sig.n_vertices == 2
    fv = classify_system([{"a", "b"}])
    assert hasattr(fv, "family")


# ── 42. topology_protocol ──────────────────────────────────────


def test_import_topology_protocol():
    import huginn.metacog.topology_protocol  # noqa: F401


def test_topology_protocol_public_api():
    # Protocol 类 + 入口函数都能从模块拿到
    from huginn.metacog import topology_protocol as tp

    assert hasattr(tp, "use_topology")
    assert hasattr(tp, "neighborhood_of")
    assert hasattr(tp, "BoxPrimitivesView")
    assert hasattr(tp, "HippocampusView")


# ── 43. trace_topology ─────────────────────────────────────────


def test_import_trace_topology():
    import huginn.metacog.trace_topology  # noqa: F401


def test_trace_topology_compute_betti_empty():
    from huginn.metacog.trace_topology import compute_betti

    betti = compute_betti([])
    assert isinstance(betti, tuple)
    assert len(betti) == 2


# ── 44. unified_complex ────────────────────────────────────────


def test_import_unified_complex():
    import huginn.metacog.unified_complex  # noqa: F401


def test_unified_complex_vertex():
    from huginn.metacog.unified_complex import Vertex

    v = Vertex(vertex_id="v1", source="kg", content="c")
    assert v.vertex_id == "v1"
    assert v.darwin_score == 0.0


# ── 45. visual_hippocampus ─────────────────────────────────────


def test_import_visual_hippocampus():
    import huginn.metacog.visual_hippocampus  # noqa: F401


def test_visual_hippocampus_recall_empty():
    from huginn.metacog.visual_hippocampus import recall

    res = recall([], query="anything", top_k=3)
    assert isinstance(res, list)
