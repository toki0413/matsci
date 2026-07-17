"""E2: 数学深度接入 PhaseGate 证据评估 — Dempster-Shafer 合成 + MathEvidenceChecker.

覆盖:
- DempsterShaferCombiner: 单源/多源/完全冲突/归一化
- MathEvidenceChecker: 空证据/全通过/全失败/混合/阈值
- PhaseGateHook 集成: 硬证据缺失优先阻断/math 阻断/math 通过/math_checker 异常降级
- 仅 validate→learn 触发 math_checker (其他转移不触发)
"""

from __future__ import annotations

import pytest

from huginn.autoloop.phase_gate import (
    DempsterShaferCombiner,
    MathEvidenceChecker,
    PhaseGateConfig,
    PhaseGateHook,
)


class TestDempsterShaferCombiner:
    def test_empty_returns_total_uncertainty(self):
        m_pass, m_fail, m_unc = DempsterShaferCombiner.combine([])
        assert m_pass == 0.0
        assert m_fail == 0.0
        assert m_unc == 1.0

    def test_single_source_passthrough(self):
        m = (0.7, 0.2, 0.1)
        result = DempsterShaferCombiner.combine([m])
        assert result == pytest.approx(m, abs=1e-9)

    def test_two_consistent_sources_reinforce(self):
        # 两个都偏向 pass 的源合并 → belief_pass 应当更高
        m1 = (0.6, 0.1, 0.3)
        m2 = (0.7, 0.05, 0.25)
        m_pass, m_fail, m_unc = DempsterShaferCombiner.combine([m1, m2])
        assert m_pass > 0.7  # 合并后比单源都高
        assert m_fail < 0.1
        assert m_pass + m_fail + m_unc == pytest.approx(1.0, abs=1e-9)

    def test_two_conflicting_sources_reduce_belief(self):
        # 一个 pass 一个 fail → 合并后两边 belief 都被削弱, 不确定增加
        m1 = (0.8, 0.05, 0.15)
        m2 = (0.05, 0.8, 0.15)
        m_pass, m_fail, m_unc = DempsterShaferCombiner.combine([m1, m2])
        # 强冲突 → 两边都不会太高
        assert m_pass < 0.5
        assert m_fail < 0.5

    def test_total_conflict_returns_all_fail(self):
        # m_pass=1, m_fail=1 → K=1, 完全冲突
        m1 = (1.0, 0.0, 0.0)
        m2 = (0.0, 1.0, 0.0)
        m_pass, m_fail, m_unc = DempsterShaferCombiner.combine([m1, m2])
        assert m_pass == 0.0
        assert m_fail == 1.0
        assert m_unc == 0.0

    def test_normalization_holds(self):
        # 任意组合, 三元组之和必须 = 1
        sources = [(0.6, 0.1, 0.3), (0.55, 0.1, 0.35), (0.7, 0.05, 0.25)]
        m_pass, m_fail, m_unc = DempsterShaferCombiner.combine(sources)
        assert m_pass + m_fail + m_unc == pytest.approx(1.0, abs=1e-9)

    def test_three_sources_consistent(self):
        # 三个都 pass → 合并后 belief_pass 很高
        sources = [(0.6, 0.1, 0.3), (0.65, 0.05, 0.3), (0.7, 0.05, 0.25)]
        m_pass, m_fail, m_unc = DempsterShaferCombiner.combine(sources)
        assert m_pass > 0.85
        assert m_fail < 0.05


class TestMathEvidenceChecker:
    def test_no_math_evidence_passes(self):
        # 无任何数学证据 key — 不阻断
        checker = MathEvidenceChecker()
        passed, feedback, details = checker({"tests_passed": True})
        assert passed is True
        assert "no math evidence" in feedback
        assert details["sources"] == []

    def test_all_pass_evidence_passes_with_high_belief(self):
        checker = MathEvidenceChecker()
        evidence = {
            "conservation_law": {"verified": True},
            "dimensional_consistent": True,
            "pde_classification": {"consistent": True},
            "sobol_top_features": {"hypothesis_covers_top": True},
            "constraint_check": {"all_passed": True},
        }
        passed, feedback, details = checker(evidence)
        assert passed is True
        assert details["belief_pass"] > 0.8
        assert details["n_sources"] == 5

    def test_all_fail_evidence_blocks(self):
        checker = MathEvidenceChecker()
        evidence = {
            "conservation_law": {"verified": False},
            "dimensional_consistent": False,
            "pde_classification": {"consistent": False},
            "sobol_top_features": {"hypothesis_covers_top": False},
            "constraint_check": {"all_passed": False},
        }
        passed, feedback, details = checker(evidence)
        assert passed is False
        assert details["belief_fail"] > 0.8
        assert "insufficient" in feedback

    def test_mixed_evidence_uses_combination(self):
        # 3 通过 + 2 失败 → 合并后 belief_pass 应当在中间
        checker = MathEvidenceChecker()
        evidence = {
            "conservation_law": {"verified": True},
            "dimensional_consistent": True,
            "pde_classification": {"consistent": True},
            "sobol_top_features": {"hypothesis_covers_top": False},
            "constraint_check": {"all_passed": False},
        }
        passed, feedback, details = checker(evidence)
        # 3 通过 vs 2 失败 → belief_pass > 0.5 (通过)
        assert details["belief_pass"] > 0.4
        assert details["n_sources"] == 5

    def test_threshold_respected(self):
        # 高阈值 → 同样的证据可能从 pass 变 block
        evidence = {
            "dimensional_consistent": True,
            "pde_classification": {"consistent": True},
        }
        low_threshold = MathEvidenceChecker(threshold=0.1)
        high_threshold = MathEvidenceChecker(threshold=0.95)
        passed_low, _, _ = low_threshold(evidence)
        passed_high, _, _ = high_threshold(evidence)
        assert passed_low is True
        assert passed_high is False

    def test_bool_value_extracted(self):
        # dimensional_consistent 直接是 bool
        checker = MathEvidenceChecker()
        passed, _, details = checker({"dimensional_consistent": True})
        assert passed is True
        assert "dimensional_consistent" in details["sources"]

    def test_dict_value_extracted(self):
        # conservation_law 是 dict, 抽 verified 字段
        checker = MathEvidenceChecker()
        passed, _, details = checker({"conservation_law": {"verified": False}})
        assert passed is False
        assert details["belief_fail"] > 0.5


class TestPhaseGateHookWithMathChecker:
    def setup_method(self):
        self.math_checker = MathEvidenceChecker()

    def test_hard_evidence_missing_blocks_before_math(self):
        # 硬证据 (tests_passed) 缺失 → 直接 block, 不跑 math_checker
        # R6 advisory: 默认 advisory 不阻断, 测 block 路径要显式开 human_checkpoint
        hook = PhaseGateHook(
            math_checker=self.math_checker,
            human_checkpoint_phases={("validate", "learn")},
        )
        gate = hook.evaluate("validate", "learn", {})
        assert gate.is_blocked
        assert "tests_passed" in gate.missing_evidence
        assert gate.reviewer != "math_checker"  # 不是 math 阻断的

    def test_math_evidence_blocks(self):
        # 硬证据齐全但数学证据全失败 → math_checker 阻断
        hook = PhaseGateHook(math_checker=self.math_checker)
        evidence = {
            "tests_passed": True,
            "conservation_law": {"verified": False},
            "dimensional_consistent": False,
        }
        gate = hook.evaluate("validate", "learn", evidence)
        assert gate.is_blocked
        assert gate.reviewer == "math_checker"
        assert "Math evidence" in gate.feedback

    def test_math_evidence_passes(self):
        # 硬证据 + 数学证据都通过 → approved
        hook = PhaseGateHook(math_checker=self.math_checker)
        evidence = {
            "tests_passed": True,
            "conservation_law": {"verified": True},
            "dimensional_consistent": True,
        }
        gate = hook.evaluate("validate", "learn", evidence)
        assert gate.status == "approved"
        assert not gate.is_blocked

    def test_no_math_keys_still_passes(self):
        # 硬证据齐全, 无数学证据 key → math_checker 跳过, approved
        hook = PhaseGateHook(math_checker=self.math_checker)
        gate = hook.evaluate("validate", "learn", {"tests_passed": True})
        assert gate.status == "approved"

    def test_math_checker_only_runs_on_validate_to_learn(self):
        # 其他转移 (plan→execute) 不应触发 math_checker
        # 即使数学证据全失败, plan→execute 也不应被 math 阻断
        hook = PhaseGateHook(math_checker=self.math_checker)
        evidence = {
            "mode": "workflow",
            "description": "run DFT",
            "conservation_law": {"verified": False},
            "dimensional_consistent": False,
        }
        gate = hook.evaluate("plan", "execute", evidence)
        assert gate.status == "approved"
        assert gate.reviewer != "math_checker"

    def test_math_checker_exception_degrades_gracefully(self):
        # math_checker 抛异常 → 不阻断, 降级放行
        def broken_checker(evidence):
            raise RuntimeError("math checker crashed")

        hook = PhaseGateHook(math_checker=broken_checker)
        gate = hook.evaluate("validate", "learn", {"tests_passed": True})
        assert gate.status == "approved"  # 降级放行

    def test_math_blocks_before_reviewer(self):
        # math_checker 阻断时不应调 reviewer (reviewer 是更重的 LLM 调用)
        reviewer_called = []

        def tracking_reviewer(from_phase, to_phase, evidence):
            reviewer_called.append((from_phase, to_phase))
            return True, "ok"

        hook = PhaseGateHook(
            math_checker=self.math_checker,
            reviewer_fn=tracking_reviewer,
        )
        evidence = {
            "tests_passed": True,
            "conservation_law": {"verified": False},
            "dimensional_consistent": False,
        }
        gate = hook.evaluate("validate", "learn", evidence)
        assert gate.is_blocked
        assert gate.reviewer == "math_checker"
        assert reviewer_called == []  # reviewer 没被调用


class TestPhaseGateConfigExtensions:
    def test_config_still_works_without_math_checker(self):
        # 不传 math_checker 时, 行为与原来一致
        hook = PhaseGateHook()
        gate = hook.evaluate("validate", "learn", {"tests_passed": True})
        assert gate.status == "approved"

    def test_config_can_add_new_requirements(self):
        # 现有 add_requirement 仍然工作
        # R6 advisory: 默认不阻断, 测 block 路径要显式开 human_checkpoint
        config = PhaseGateConfig()
        config.add_requirement("validate", "learn", ["tests_passed", "reviewer_critique"])
        hook = PhaseGateHook(
            config=config,
            math_checker=MathEvidenceChecker(),
            human_checkpoint_phases={("validate", "learn")},
        )
        gate = hook.evaluate("validate", "learn", {"tests_passed": True})
        assert gate.is_blocked
        assert "reviewer_critique" in gate.missing_evidence
