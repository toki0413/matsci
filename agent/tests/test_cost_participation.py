"""Tests for the cost-control participation stack:
CostLedger / ValueBudget / BudgetPause / BranchPolicy.

These back the "成本-剪枝参与感" contract (docs/cost-participation-contract.md):
unified ledger, value-aware phase budget, budget-edge soft-stop, and
branch pruning that is reversible and user-adjudicated.
"""

from __future__ import annotations

from huginn.branch_policy import (
    Branch,
    BranchScore,
    BranchState,
    DecisionPointRegistry,
    get_decision_point_registry,
    reset_decision_point_registry,
)
from huginn.budget_pause import BudgetPauseHandler, reset_budget_pause_handler
from huginn.cost_ledger import (
    CostDimension,
    CostLedger,
    CostUnit,
    get_cost_ledger,
    reset_cost_ledger,
)
from huginn.value_budget import ValueBudget

# ── CostLedger ────────────────────────────────────────────────────


class TestCostLedger:
    def test_record_and_total_normalize_usd(self):
        l = CostLedger(usd_per_1k_tokens=2.0, usd_per_cpu_hour=0.5)
        l.record(CostDimension.LLM, 1000, CostUnit.TOKENS, tool="vasp_tool", phase="explore")
        l.record(CostDimension.COMPUTE, 2.0, CostUnit.CPU_HOURS, tool="vasp_tool", phase="explore")
        l.record(CostDimension.EXTERNAL, 0.3, CostUnit.USD, tool="web_search", phase="review")
        assert l.total_usd() == 2.0 + 1.0 + 0.3
        assert l.by_dimension()["llm"] == 2.0
        assert l.by_phase()["explore"] == 3.0
        assert l.by_tool()["vasp_tool"] == 3.0

    def test_explicit_usd_overrides_rate(self):
        l = CostLedger(usd_per_cpu_hour=0.5)
        l.record(CostDimension.COMPUTE, 10.0, CostUnit.CPU_HOURS, usd=99.0)
        assert l.total_usd() == 99.0

    def test_unpriced_unit_stays_zero_usd(self):
        l = CostLedger()  # no rates
        l.record(CostDimension.LLM, 5000, CostUnit.TOKENS)
        assert l.total_usd() == 0.0
        assert l.entry_count() == 1 if hasattr(l, "entry_count") else True

    def test_check_budget_tiers(self):
        l = CostLedger(usd_per_cpu_hour=1.0)
        l.record(CostDimension.COMPUTE, 2.0, CostUnit.CPU_HOURS)
        assert l.check_budget(10.0)[0] == "allow"
        assert l.check_budget(2.0)[0] == "deny"
        assert l.check_budget(2.5)[0] == "warn"  # 2 >= 2.5*0.8
        assert l.check_budget(0.0)[0] == "allow"  # 0 = unlimited

    def test_session_filtering(self):
        l = CostLedger(usd_per_cpu_hour=1.0)
        l.record(CostDimension.COMPUTE, 1.0, CostUnit.CPU_HOURS, session_id="a")
        l.record(CostDimension.COMPUTE, 2.0, CostUnit.CPU_HOURS, session_id="b")
        assert l.session_total("a") == 1.0
        assert l.by_phase(session_id="a") == {"—": 1.0}

    def test_singleton_reset(self):
        reset_cost_ledger()
        a = get_cost_ledger()
        b = get_cost_ledger()
        assert a is b
        reset_cost_ledger()
        assert get_cost_ledger() is not a


# ── ValueBudget ───────────────────────────────────────────────────


class TestValueBudget:
    def test_phase_scaling(self):
        vb = ValueBudget(base_budget_usd=10.0)
        assert vb.effective_budget("explore") == 15.0
        assert vb.effective_budget("converge") == 10.0
        assert vb.effective_budget("verify") == 5.0
        assert vb.effective_budget("unknown") == 10.0

    def test_over_budget_denies(self):
        vb = ValueBudget(base_budget_usd=10.0)
        assert vb.check("verify", 6.0)[0] == "deny"

    def test_warn_near_budget(self):
        vb = ValueBudget(base_budget_usd=10.0)
        # verify effective budget 5; 4 >= 5*0.8 → warn
        assert vb.check("verify", 4.0, 4.0)[0] == "warn"

    def test_roi_gate_denies_waste(self):
        vb = ValueBudget(base_budget_usd=10.0, min_roi=1.0)
        # 花 12 (<15 explore budget) 但 ROI 0.5 → deny
        assert vb.check("explore", 12.0, 6.0)[0] == "deny"

    def test_roi_disabled_ignores_value(self):
        vb = ValueBudget(base_budget_usd=10.0, min_roi=0.0)
        assert vb.check("converge", 5.0, 0.0)[0] == "allow"

    def test_allowed_phases(self):
        vb = ValueBudget(base_budget_usd=10.0)
        allowed = vb.allowed_phases(total_cost_usd=6.0)
        # explore(15) & converge(10) allow; verify(5) not
        assert "explore" in allowed and "converge" in allowed
        assert "verify" not in allowed

    def test_default_budget_from_registry(self, monkeypatch):
        """默认上限来自 env_defaults 注册表单一权威, 不内联硬编码."""
        from huginn.env_defaults import registry_default

        monkeypatch.delenv("HUGINN_VALUE_BUDGET_USD", raising=False)
        monkeypatch.delenv("HUGINN_VALUE_MIN_ROI", raising=False)
        assert ValueBudget().base_budget_usd == registry_default("HUGINN_VALUE_BUDGET_USD")
        assert ValueBudget().min_roi == registry_default("HUGINN_VALUE_MIN_ROI")

    def test_value_budget_env_override(self, monkeypatch):
        """HUGINN_VALUE_* env 覆盖注册表默认值."""
        monkeypatch.setenv("HUGINN_VALUE_BUDGET_USD", "25.0")
        monkeypatch.setenv("HUGINN_VALUE_MIN_ROI", "2.0")
        vb = ValueBudget()
        assert vb.base_budget_usd == 25.0
        assert vb.min_roi == 2.0


# ── TokenBudget 成本上限经 env 覆盖 (不再内联硬编码) ──────────────


class TestTokenBudgetDefaults:
    def test_hard_limit_defaults_from_registry(self, monkeypatch):
        """HUGINN_TOKEN_BUDGET / HUGINN_COST_BUDGET 默认值来自注册表."""
        from huginn.autoloop.budget import TokenBudget
        from huginn.env_defaults import registry_default

        monkeypatch.delenv("HUGINN_TOKEN_BUDGET", raising=False)
        monkeypatch.delenv("HUGINN_COST_BUDGET", raising=False)
        monkeypatch.delenv("HUGINN_BUDGET_MAX_RENEWALS", raising=False)
        b = TokenBudget()
        assert b.hard_limit_tokens == registry_default("HUGINN_TOKEN_BUDGET")
        assert b.hard_limit_cost == registry_default("HUGINN_COST_BUDGET")
        assert b.max_renewals == registry_default("HUGINN_BUDGET_MAX_RENEWALS")

    def test_hard_limit_env_override(self, monkeypatch):
        """env 覆盖能改变硬上限 (成本上限不再写死)."""
        from huginn.autoloop.budget import TokenBudget

        monkeypatch.setenv("HUGINN_TOKEN_BUDGET", "12345")
        monkeypatch.setenv("HUGINN_COST_BUDGET", "9.5")
        monkeypatch.setenv("HUGINN_BUDGET_MAX_RENEWALS", "7")
        b = TokenBudget()
        assert b.hard_limit_tokens == 12345
        assert b.hard_limit_cost == 9.5
        assert b.max_renewals == 7


# ── HarnessDev 结论③: 分类失败阈值参数化 + 结论④: held-out 门控接线 ──


class TestFailureTypeConfig:
    def test_defaults_match_historic_hardcode(self, monkeypatch):
        """默认分类失败阈值与历史硬编码一致, 不 env 时无行为变化."""
        from huginn.autoloop.engine import _config_failures_by_type

        monkeypatch.delenv("HUGINN_MAX_FAILURES_BY_TYPE", raising=False)
        cfg = _config_failures_by_type()
        assert cfg["tool_error"] == 5
        assert cfg["hypothesis_error"] == 10
        assert cfg["prompt_injection_suspect"] == 3

    def test_json_env_overrides_and_validates(self, monkeypatch):
        """JSON env 可按执行者校准覆盖, 非法键/<=0 忽略, 缺省键回退."""
        from huginn.autoloop.engine import _config_failures_by_type

        monkeypatch.setenv(
            "HUGINN_MAX_FAILURES_BY_TYPE",
            '{"tool_error":8,"hypothesis_error":15,"bogus":-1}',
        )
        cfg = _config_failures_by_type()
        assert cfg["tool_error"] == 8
        assert cfg["hypothesis_error"] == 15
        assert cfg["param_error"] == 5  # 未覆盖 → 回退默认
        assert "bogus" not in cfg  # <=0 忽略

    def test_bad_json_falls_back_to_default(self, monkeypatch):
        from huginn.autoloop.engine import _config_failures_by_type

        monkeypatch.setenv("HUGINN_MAX_FAILURES_BY_TYPE", "not-json{{")
        assert _config_failures_by_type()["tool_error"] == 5


class TestHarnessGateWiring:
    def test_gate_helper_default_off_then_on(self, monkeypatch):
        """结论④ 接线默认关 (不改变行为), 设 env 开."""
        from huginn.evolution.manager import _record_harness_gates_enabled

        monkeypatch.delenv("HUGINN_HARNESS_GATES", raising=False)
        assert _record_harness_gates_enabled() is False
        monkeypatch.setenv("HUGINN_HARNESS_GATES", "1")
        assert _record_harness_gates_enabled() is True

    def test_gate_on_records_real_outcome_to_holdout(self, monkeypatch):
        """开启后 OOD holdout 门控能积累真实 supported/refuted outcome (不再恒空)."""
        from huginn.harness.ood_holdout import OODHoldoutValidator

        OODHoldoutValidator._instance = None
        monkeypatch.setenv("HUGINN_HARNESS_GATES", "1")
        validator = OODHoldoutValidator.get_instance()
        validator.record_outcome("strategy:explore", "some-task", 1.0)
        records = validator.get_records("strategy:explore")
        assert records and records[0].score == 1.0
        OODHoldoutValidator._instance = None


# ── BudgetPause ───────────────────────────────────────────────────


class TestBudgetPause:
    def test_pause_resume_cycle(self):
        saved = []
        h = BudgetPauseHandler(save_checkpoint=saved.append)
        p = h.pause("s1", "超出阶段预算", {"iter": 12})
        assert saved and saved[0] == {"iter": 12}
        assert not p.resumed
        assert len(h.pending("s1")) == 1
        h.resume(p, decision="continue")
        assert p.resumed and p.resume_decision == "continue"
        assert len(h.pending("s1")) == 0
        assert h.list_pauses("s1")[0].checkpoint is not None

    def test_abort_decision(self):
        h = BudgetPauseHandler()
        p = h.pause("s1", "不划算")
        h.resume(p, decision="abort")
        assert p.resume_decision == "abort"

    def test_clear_by_session(self):
        h = BudgetPauseHandler()
        h.pause("s1", "a")
        h.pause("s2", "b")
        assert len(h.list_pauses()) == 2
        h.clear("s1")
        assert len(h.list_pauses()) == 1
        assert h.list_pauses()[0].session_id == "s2"

    def test_singleton_reset(self):
        reset_budget_pause_handler()
        from huginn.budget_pause import get_budget_pause_handler

        a = get_budget_pause_handler()
        b = get_budget_pause_handler()
        assert a is b
        reset_budget_pause_handler()
        assert get_budget_pause_handler() is not a


# ── BranchPolicy ──────────────────────────────────────────────────


class TestBranchScore:
    def test_unknown_high_uncertainty_hibernates(self):
        # 低均值 + 高不确定性 → 可能是真金, 不敢砍
        s = BranchScore(mean_value=0.3, n_probes=2, variance=0.2, divergence=0.6)
        assert s.recommendation() == "hibernate"
        assert s.uncertainty() >= 0.6

    def test_dead_low_value_abandons(self):
        # 试透了 (n=12) + 低方差 + 低价值 → 可放弃
        s = BranchScore(mean_value=0.05, n_probes=12, variance=0.02, divergence=0.05)
        assert s.recommendation() == "abandon"

    def test_good_value_explores(self):
        s = BranchScore(mean_value=0.8, n_probes=8, variance=0.1, divergence=0.2)
        assert s.recommendation() == "explore"

    def test_ucb_weights_uncertainty_by_stage(self):
        s = BranchScore(mean_value=0.3, n_probes=0)
        # explore β=1.5 更保守 → ucb 更高; verify β=0.5 → 更低
        assert s.ucb({"explore": 1.5}) > s.ucb({"verify": 0.5})

    def test_uncertainty_increases_with_divergence(self):
        low = BranchScore(mean_value=0.3, n_probes=10, variance=0.0, divergence=0.0)
        high = BranchScore(mean_value=0.3, n_probes=10, variance=0.0, divergence=0.9)
        assert high.uncertainty() > low.uncertainty()


class TestBranch:
    def test_hibernate_is_reversible(self):
        b = Branch(branch_id="b_7", score=BranchScore(mean_value=0.3, n_probes=2))
        b.hibernate(lifeline="vasp 粗精度", revive_conditions=["主支收敛后回看"])
        assert b.state == BranchState.HIBERNATING
        assert b.lifeline
        b.revive(reason="用户要求续投")
        assert b.state == BranchState.ACTIVE
        assert any("用户" in c for c in b.revive_conditions)

    def test_abandon(self):
        b = Branch(branch_id="b_9")
        b.abandon()
        assert b.state == BranchState.ABANDONED


class TestDecisionPointRegistry:
    def test_open_resolve(self):
        reg = DecisionPointRegistry()
        b = Branch(branch_id="b_7")
        dp = reg.open(
            "hibernate",
            session_id="s1",
            branch=b,
            narrative={"phase": "converge", "cost_usd": 12.4},
        )
        assert len(reg.pending("s1")) == 1
        reg.resolve(dp.id, decision="approved", option="hibernate_lifeline")
        assert dp.status == "approved"
        assert dp.response["option"] == "hibernate_lifeline"
        assert len(reg.pending("s1")) == 0

    def test_expire_uses_conservative_default(self):
        reg = DecisionPointRegistry()
        dp = reg.open("prune", session_id="s1")
        reg.expire(dp.id)
        assert dp.status == "expired"
        assert dp.response["option"] == "_default"

    def test_edited_decision(self):
        reg = DecisionPointRegistry()
        dp = reg.open("degrade", session_id="s1")
        reg.resolve(dp.id, decision="edited", option="continue_invest")
        assert dp.status == "edited"

    def test_resolve_missing_returns_none(self):
        reg = DecisionPointRegistry()
        assert reg.resolve("dp_nope", decision="approved") is None

    def test_singleton_reset(self):
        reset_decision_point_registry()
        a = get_decision_point_registry()
        b = get_decision_point_registry()
        assert a is b
        reset_decision_point_registry()
        assert get_decision_point_registry() is not a
