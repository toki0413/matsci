"""A: LUCID 必要条件闭环自检.

验证 _extract_lucid_prereqs 能从 LLM 输出里解析 necessary condition,
_attach_lucid_prereqs 能把它加成 hypothesis_graph 的派生节点.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from huginn.autoloop.engine import AutoloopEngine


class TestExtractLucidPrereqs:
    """解析器: 从 LLM 原始输出里提取 necessary / hidden / falsifiable."""

    def test_extract_all_three(self):
        raw = """
SELECTED: 掺杂增加导致带隙减小

LUCID review:
- Necessary condition: 材料必须是直接带隙半导体
- Hidden assumption: 掺杂不改变能带极值位置
- Falsifiable test: 带隙随掺杂增加而增大则假设被反驳
"""
        r = AutoloopEngine._extract_lucid_prereqs(raw)
        assert "直接带隙半导体" in r["necessary"]
        assert "能带极值" in r["hidden"]
        assert "带隙随掺杂增加而增大" in r["falsifiable"]

    def test_extract_handles_english(self):
        raw = """
SELECTED: Doping increases bandgap

LUCID review:
- Necessary condition: The material must be a direct semiconductor
- Hidden assumption: Doping does not shift the band edge
- Falsifiable test: If bandgap decreases with doping, hypothesis is refuted
"""
        r = AutoloopEngine._extract_lucid_prereqs(raw)
        assert "direct semiconductor" in r["necessary"]
        assert "shift the band edge" in r["hidden"]
        assert "bandgap decreases" in r["falsifiable"]

    def test_extract_missing_keys_return_empty(self):
        raw = "SELECTED: just a hypothesis"
        r = AutoloopEngine._extract_lucid_prereqs(raw)
        assert r == {"necessary": "", "hidden": "", "falsifiable": ""}

    def test_extract_empty_input(self):
        r = AutoloopEngine._extract_lucid_prereqs("")
        assert r == {"necessary": "", "hidden": "", "falsifiable": ""}

    def test_extract_truncates_long_values(self):
        long_val = "x" * 500
        raw = f"SELECTED: H\n- Necessary condition: {long_val}"
        r = AutoloopEngine._extract_lucid_prereqs(raw)
        assert len(r["necessary"]) <= 300


class TestAttachLucidPrereqs:
    """_attach_lucid_prereqs: 把必要条件加成 hypothesis_graph 派生节点."""

    def _make_engine_with_graph(self):
        """轻量构造: 只装 hypothesis_graph, 不跑真实 init."""
        engine = object.__new__(AutoloopEngine)
        from huginn.autoloop.hypothesis_loop import HypothesisGraph
        engine.hypothesis_graph = HypothesisGraph()
        return engine

    def test_attach_creates_derived_node(self):
        engine = self._make_engine_with_graph()
        engine._last_raw_hypothesis = (
            "SELECTED: 掺杂增加导致带隙减小\n"
            "- Necessary condition: 材料必须是直接带隙半导体"
        )
        h = engine.hypothesis_graph.add_hypothesis("掺杂增加导致带隙减小")
        engine._attach_lucid_prereqs(h)

        # 应该有 2 个节点: 原假设 + 派生必要条件
        assert len(engine.hypothesis_graph.all_nodes()) == 2
        derived = [n for n in engine.hypothesis_graph.all_nodes()
                   if n.id != h][0]
        assert "必要条件" in derived.statement
        assert "直接带隙半导体" in derived.statement
        assert derived.parent_id == h
        assert derived.status == "untested"  # 进 frontier 队列

    def test_attach_no_necessary_does_nothing(self):
        engine = self._make_engine_with_graph()
        engine._last_raw_hypothesis = "SELECTED: just a hypothesis"
        h = engine.hypothesis_graph.add_hypothesis("H")
        engine._attach_lucid_prereqs(h)
        assert len(engine.hypothesis_graph.all_nodes()) == 1

    def test_attach_empty_raw_does_nothing(self):
        engine = self._make_engine_with_graph()
        engine._last_raw_hypothesis = ""
        h = engine.hypothesis_graph.add_hypothesis("H")
        engine._attach_lucid_prereqs(h)
        assert len(engine.hypothesis_graph.all_nodes()) == 1

    def test_attach_none_hyp_id_does_nothing(self):
        engine = self._make_engine_with_graph()
        engine._last_raw_hypothesis = (
            "SELECTED: H\n- Necessary condition: N"
        )
        # hyp_id 为 None 时不应崩
        engine._attach_lucid_prereqs(None)
        assert len(engine.hypothesis_graph.all_nodes()) == 0

    def test_attach_failure_does_not_propagate(self):
        """hypothesis_graph 出错时, attach 应静默失败不影响主流程."""
        engine = self._make_engine_with_graph()
        engine._last_raw_hypothesis = (
            "SELECTED: H\n- Necessary condition: N"
        )
        engine.hypothesis_graph = MagicMock()
        engine.hypothesis_graph.add_hypothesis.side_effect = RuntimeError("boom")
        # 不应抛
        engine._attach_lucid_prereqs("h_xxx")


# ── C: 失败类型区分 ─────────────────────────────────────────────────────────


class TestClassifyFailure:
    """_classify_failure: 根据 validation 证据分类失败类型.

    分类决定走 retry (tool_error) 还是 refine/pivot (其他).
    """

    def test_timeout_classified_as_tool_error(self):
        v = {"errors": "VASP subprocess timeout after 3600s"}
        assert AutoloopEngine._classify_failure(v) == "tool_error"

    def test_crash_classified_as_tool_error(self):
        v = {"errors": "LAMMPS segfault, core dumped"}
        assert AutoloopEngine._classify_failure(v) == "tool_error"

    def test_oom_classified_as_tool_error(self):
        v = {"errors": "out of memory (OOM) killed by slurm"}
        assert AutoloopEngine._classify_failure(v) == "tool_error"

    def test_connection_error_classified_as_tool_error(self):
        v = {"errors": "connection refused to remote executor"}
        assert AutoloopEngine._classify_failure(v) == "tool_error"

    def test_invalid_argument_classified_as_param_error(self):
        v = {"errors": "invalid argument: POSCAR path is None"}
        assert AutoloopEngine._classify_failure(v) == "param_error"

    def test_shape_mismatch_classified_as_param_error(self):
        v = {"errors": "shape mismatch: expected (3,3) got (4,4)"}
        assert AutoloopEngine._classify_failure(v) == "param_error"

    def test_value_error_classified_as_param_error(self):
        v = {"errors": "value error: negative concentration"}
        assert AutoloopEngine._classify_failure(v) == "param_error"

    def test_no_convergence_classified_as_data_noise(self):
        v = {"errors": "SCF did not converge in 100 steps"}
        assert AutoloopEngine._classify_failure(v) == "data_noise"

    def test_uncertain_result_classified_as_data_noise(self):
        v = {"result": "uncertain, signal within noise band"}
        assert AutoloopEngine._classify_failure(v) == "data_noise"

    def test_opposite_result_classified_as_hypothesis_error(self):
        v = {"result": "bandgap increased, opposite to prediction"}
        assert AutoloopEngine._classify_failure(v) == "hypothesis_error"

    def test_tests_failed_no_errors_classified_as_hypothesis_error(self):
        v = {"tests_passed": False}
        assert AutoloopEngine._classify_failure(v) == "hypothesis_error"

    def test_empty_validation_classified_as_hypothesis_error(self):
        assert AutoloopEngine._classify_failure({}) == "hypothesis_error"

    def test_tool_error_takes_priority_over_param(self):
        """timeout + invalid argument 同时出现 → tool_error 优先."""
        v = {"errors": "timeout after retry, invalid parameter ENCUT"}
        assert AutoloopEngine._classify_failure(v) == "tool_error"


# ── B: mode 切换上下文路由 ──────────────────────────────────────────────────


class TestPlanContextRouting:
    """_plan_context_hint (软路由) + _override_plan_mode (硬路由)."""

    def _make_engine_with_graph(self):
        from huginn.autoloop.hypothesis_loop import HypothesisGraph
        engine = object.__new__(AutoloopEngine)
        engine.hypothesis_graph = HypothesisGraph()
        engine._consecutive_failures = 0
        engine._refine_count = 0
        engine._current_hyp_id_for_plan = None
        engine._last_surprise = 0.0
        return engine

    def test_context_hint_empty_when_no_signals(self):
        engine = self._make_engine_with_graph()
        assert engine._plan_context_hint() == ""

    def test_context_hint_warns_on_consecutive_failures(self):
        engine = self._make_engine_with_graph()
        engine._consecutive_failures = 3
        hint = engine._plan_context_hint()
        assert "连续失败 3 次" in hint
        assert "explore" in hint.lower()

    def test_context_hint_notes_high_refine_count(self):
        engine = self._make_engine_with_graph()
        engine._refine_count = 3
        hint = engine._plan_context_hint()
        assert "refine 3 次" in hint

    def test_context_hint_for_cut_vertex(self):
        """割点节点 → hint 提示需要双模态验证."""
        engine = self._make_engine_with_graph()
        h1 = engine.hypothesis_graph.add_hypothesis("H1")
        h2 = engine.hypothesis_graph.add_hypothesis("H2", parent_id=h1)
        h3 = engine.hypothesis_graph.add_hypothesis("H3", parent_id=h2)
        # h2 是割点
        engine._current_hyp_id_for_plan = h2
        hint = engine._plan_context_hint()
        assert "割点" in hint or "CRITICAL" in hint
        assert "双模态" in hint or "workflow" in hint

    def test_override_no_change_when_no_signals(self):
        engine = self._make_engine_with_graph()
        plan = {"mode": "coder", "description": "fix bug"}
        result = engine._override_plan_mode(plan)
        assert result["mode"] == "coder"

    def test_override_coder_to_workflow_for_cut_vertex(self):
        """割点节点 + LLM 选 coder → 硬路由覆盖成 workflow."""
        engine = self._make_engine_with_graph()
        h1 = engine.hypothesis_graph.add_hypothesis("H1")
        h2 = engine.hypothesis_graph.add_hypothesis("H2", parent_id=h1)
        h3 = engine.hypothesis_graph.add_hypothesis("H3", parent_id=h2)
        engine._current_hyp_id_for_plan = h2  # h2 是割点
        plan = {"mode": "coder", "description": "edit config"}
        result = engine._override_plan_mode(plan)
        assert result["mode"] == "workflow"
        assert "auto-routed" in result["description"]

    def test_override_keeps_workflow_for_cut_vertex(self):
        """割点节点 + LLM 选 workflow → 不覆盖."""
        engine = self._make_engine_with_graph()
        h1 = engine.hypothesis_graph.add_hypothesis("H1")
        h2 = engine.hypothesis_graph.add_hypothesis("H2", parent_id=h1)
        h3 = engine.hypothesis_graph.add_hypothesis("H3", parent_id=h2)
        engine._current_hyp_id_for_plan = h2
        plan = {"mode": "workflow", "description": "run DFT"}
        result = engine._override_plan_mode(plan)
        assert result["mode"] == "workflow"
        assert "auto-routed" not in result["description"]

    def test_override_to_explore_after_5_failures(self):
        """连续失败 5 次 → 强制 explore."""
        engine = self._make_engine_with_graph()
        engine._consecutive_failures = 5
        plan = {"mode": "coder", "description": "fix bug"}
        result = engine._override_plan_mode(plan)
        assert result["mode"] == "explore"
        assert "auto-routed" in result["description"]

    def test_override_keeps_explore_if_already_explore(self):
        engine = self._make_engine_with_graph()
        engine._consecutive_failures = 5
        plan = {"mode": "explore", "description": "search space"}
        result = engine._override_plan_mode(plan)
        assert result["mode"] == "explore"
        assert "auto-routed" not in result["description"]

    def test_override_does_not_crash_on_missing_hyp_id(self):
        """_current_hyp_id_for_plan 为 None 时不崩."""
        engine = self._make_engine_with_graph()
        plan = {"mode": "coder", "description": "x"}
        result = engine._override_plan_mode(plan)
        assert result["mode"] == "coder"

    # ── B ceiling: surprise 信号 ──────────────────────────────────────

    def test_context_hint_warns_on_high_surprise(self):
        """surprise > 0.5 → 软提示预测误差大."""
        engine = self._make_engine_with_graph()
        engine._last_surprise = 0.6
        hint = engine._plan_context_hint()
        assert "surprise" in hint.lower()
        assert "explore" in hint.lower() or "重新假设" in hint

    def test_context_hint_silent_on_low_surprise(self):
        """surprise <= 0.5 → 不提示."""
        engine = self._make_engine_with_graph()
        engine._last_surprise = 0.3
        assert engine._plan_context_hint() == ""

    def test_override_explore_on_extreme_surprise(self):
        """surprise > 0.9 → 硬路由强制 explore."""
        engine = self._make_engine_with_graph()
        engine._last_surprise = 0.95
        plan = {"mode": "coder", "description": "fix bug"}
        result = engine._override_plan_mode(plan)
        assert result["mode"] == "explore"
        assert "surprise" in result["description"].lower()

    def test_override_keeps_explore_when_surprise_high_but_already_explore(self):
        """surprise > 0.9 + 已经是 explore → 不重复覆盖."""
        engine = self._make_engine_with_graph()
        engine._last_surprise = 0.95
        plan = {"mode": "explore", "description": "search"}
        result = engine._override_plan_mode(plan)
        assert result["mode"] == "explore"
        assert "auto-routed" not in result["description"]

    def test_override_no_change_on_moderate_surprise(self):
        """surprise 在 0.5-0.9 之间 → 软提示但不硬覆盖."""
        engine = self._make_engine_with_graph()
        engine._last_surprise = 0.7
        plan = {"mode": "coder", "description": "fix bug"}
        result = engine._override_plan_mode(plan)
        assert result["mode"] == "coder"  # 不硬覆盖


# ── C ceiling: RedTeam findings 参与失败分类 ─────────────────────────────────


class TestRedteamClassification:
    """_classify_failure 接收 redteam_cats + _redteam_findings helper."""

    def test_redteam_methodology_gap_overrides_to_param_error(self):
        """methodology_gap → param_error."""
        v = {"errors": "tests failed"}
        cats = ["methodology_gap"]
        assert AutoloopEngine._classify_failure(v, redteam_cats=cats) == "param_error"

    def test_redteam_hidden_assumption_overrides_to_param_error(self):
        """hidden_assumption → param_error (边界条件缺失类似参数错)."""
        v = {"errors": "tests failed"}
        cats = ["hidden_assumption"]
        assert AutoloopEngine._classify_failure(v, redteam_cats=cats) == "param_error"

    def test_redteam_confounder_overrides_to_data_noise(self):
        """confounder → data_noise (混淆变量需重做排除)."""
        v = {"errors": "tests failed"}
        cats = ["confounder"]
        assert AutoloopEngine._classify_failure(v, redteam_cats=cats) == "data_noise"

    def test_redteam_alternative_explanation_overrides_to_hypothesis_error(self):
        """alternative_explanation → hypothesis_error (假设本身可能错)."""
        v = {"errors": "tests failed"}
        cats = ["alternative_explanation"]
        assert (
            AutoloopEngine._classify_failure(v, redteam_cats=cats)
            == "hypothesis_error"
        )

    def test_tool_error_still_prioritized_over_redteam(self):
        """timeout + RedTeam finding → tool_error 优先 (工具问题与假设无关)."""
        v = {"errors": "timeout after 60s"}
        cats = ["alternative_explanation"]
        assert (
            AutoloopEngine._classify_failure(v, redteam_cats=cats) == "tool_error"
        )

    def test_no_redteam_cats_falls_back_to_keywords(self):
        """redteam_cats=None → 走原有关键词匹配 (向后兼容)."""
        v = {"errors": "invalid argument: path is None"}
        assert AutoloopEngine._classify_failure(v) == "param_error"
        assert (
            AutoloopEngine._classify_failure(v, redteam_cats=None) == "param_error"
        )

    def test_empty_redteam_cats_falls_back_to_keywords(self):
        """redteam_cats=[] → 走关键词匹配."""
        v = {"errors": "shape mismatch"}
        assert (
            AutoloopEngine._classify_failure(v, redteam_cats=[]) == "param_error"
        )

    def test_unknown_redteam_category_ignored(self):
        """redteam_cats 含未知 category → 忽略, 走关键词匹配."""
        v = {"errors": "invalid argument"}
        cats = ["unknown_category", "methodology_gap"]
        # 第一个未知被跳过, 第二个 methodology_gap 命中 → param_error
        assert (
            AutoloopEngine._classify_failure(v, redteam_cats=cats) == "param_error"
        )

    def test_redteam_takes_priority_over_keyword_match(self):
        """RedTeam finding 优先于关键词匹配 (无 tool_error 时)."""
        v = {"errors": "invalid argument"}  # 关键词会匹配 param_error
        cats = ["confounder"]  # RedTeam 说 confounder → data_noise
        assert (
            AutoloopEngine._classify_failure(v, redteam_cats=cats) == "data_noise"
        )

    # ── _redteam_findings helper ─────────────────────────────────────

    def _make_engine_with_phase_gate(self, report=None):
        """构造带 phase_gate_hook 的 engine, reviewer_fn._last_report 可设."""
        engine = object.__new__(AutoloopEngine)

        class _FakeReviewer:
            pass

        reviewer = _FakeReviewer()
        if report is not None:
            reviewer._last_report = report
        # phase_gate_hook.reviewer_fn → reviewer
        engine.phase_gate_hook = type("_FG", (), {"reviewer_fn": reviewer})()
        return engine

    def test_redteam_findings_reads_last_report(self):
        """_redteam_findings 从 _last_report 拿 high severity category."""
        from huginn.autoloop.red_team import RedTeamReport, RedTeamFinding

        report = RedTeamReport(
            transition=("validate", "learn"),
            findings=[
                RedTeamFinding("methodology_gap", "gap", "high"),
                RedTeamFinding("confounder", "c", "low"),  # 低严重度过滤掉
                RedTeamFinding("alternative_explanation", "alt", "high"),
            ],
        )
        engine = self._make_engine_with_phase_gate(report=report)
        cats = engine._redteam_findings()
        assert "methodology_gap" in cats
        assert "alternative_explanation" in cats
        assert "confounder" not in cats  # low severity 被过滤
        assert len(cats) == 2

    def test_redteam_findings_empty_when_no_report(self):
        """_last_report 为 None → 返回空列表."""
        engine = self._make_engine_with_phase_gate(report=None)
        # reviewer 没有 _last_report 属性 → getattr 返回 None
        assert engine._redteam_findings() == []

    def test_redteam_findings_empty_when_no_reviewer(self):
        """phase_gate_hook.reviewer_fn 为 None → 返回空列表."""
        engine = object.__new__(AutoloopEngine)
        engine.phase_gate_hook = type("_FG", (), {"reviewer_fn": None})()
        assert engine._redteam_findings() == []

    def test_redteam_findings_empty_when_no_phase_gate(self):
        """phase_gate_hook 属性不存在 → 返回空列表 (不崩)."""
        engine = object.__new__(AutoloopEngine)
        # 不设 phase_gate_hook
        assert engine._redteam_findings() == []


# ── 机制3: _log_plan_override / _check_gate reviewer 审计 ──────────────────


class TestPlanOverrideAudit:
    """_override_plan_mode 的覆盖要写 PhaseGateState.history 补审计."""

    def _make_engine_with_graph(self):
        from huginn.autoloop.hypothesis_loop import HypothesisGraph
        engine = object.__new__(AutoloopEngine)
        engine.hypothesis_graph = HypothesisGraph()
        engine._consecutive_failures = 0
        engine._refine_count = 0
        engine._current_hyp_id_for_plan = None
        engine._last_surprise = 0.0
        return engine

    def test_log_plan_override_writes_history_entry(self):
        """force_explore 触发后, PhaseGateState.history 多一条 reviewer=auto_router."""
        from huginn.autoloop.phase_gate import (
            PhaseGateState, set_shared_phase_gate_state,
            get_shared_phase_gate_state,
        )
        set_shared_phase_gate_state(PhaseGateState())
        try:
            engine = self._make_engine_with_graph()
            engine._consecutive_failures = 5
            plan = {"mode": "coder", "description": "fix bug"}
            engine._override_plan_mode(plan)
            state = get_shared_phase_gate_state()
            auto_entries = [
                h for h in state.history
                if h.reviewer == "auto_router"
            ]
            assert len(auto_entries) == 1
            assert "force_explore" in auto_entries[0].feedback
            assert "连续失败5次" in auto_entries[0].feedback
        finally:
            set_shared_phase_gate_state(None)

    def test_log_plan_override_no_duplicate_for_cut_vertex(self):
        """割点单条覆盖也写 history, 不重复."""
        from huginn.autoloop.phase_gate import (
            PhaseGateState, set_shared_phase_gate_state,
            get_shared_phase_gate_state,
        )
        set_shared_phase_gate_state(PhaseGateState())
        try:
            engine = self._make_engine_with_graph()
            h1 = engine.hypothesis_graph.add_hypothesis("H1")
            h2 = engine.hypothesis_graph.add_hypothesis("H2", parent_id=h1)
            h3 = engine.hypothesis_graph.add_hypothesis("H3", parent_id=h2)
            engine._current_hyp_id_for_plan = h2
            plan = {"mode": "coder", "description": "edit"}
            engine._override_plan_mode(plan)
            state = get_shared_phase_gate_state()
            auto_entries = [
                h for h in state.history
                if h.reviewer == "auto_router"
            ]
            assert len(auto_entries) == 1
            assert "cut_vertex" in auto_entries[0].feedback
        finally:
            set_shared_phase_gate_state(None)

    def test_log_plan_override_silent_when_no_state(self):
        """shared state 为 None (未初始化) 时 _log_plan_override 不崩, 静默."""
        from huginn.autoloop.phase_gate import (
            set_shared_phase_gate_state,
        )
        set_shared_phase_gate_state(None)
        try:
            engine = self._make_engine_with_graph()
            # 走 _log_plan_override 内部 import 路径, get_shared 返回 None 时
            # state.history.append 会 AttributeError → 走 except
            engine._log_plan_override("test_code", "test reason")
            # 不抛异常即通过
        finally:
            set_shared_phase_gate_state(None)

    def test_log_plan_override_combines_reasons(self):
        """连败 + surprise 同时满足 → 单条 history, reason 合并."""
        from huginn.autoloop.phase_gate import (
            PhaseGateState, set_shared_phase_gate_state,
            get_shared_phase_gate_state,
        )
        set_shared_phase_gate_state(PhaseGateState())
        try:
            engine = self._make_engine_with_graph()
            engine._consecutive_failures = 6
            engine._last_surprise = 0.95
            plan = {"mode": "coder", "description": "fix bug"}
            engine._override_plan_mode(plan)
            state = get_shared_phase_gate_state()
            auto_entries = [
                h for h in state.history
                if h.reviewer == "auto_router"
            ]
            assert len(auto_entries) == 1
            fb = auto_entries[0].feedback
            # 两个 reason 都该出现在 feedback 里
            assert "连续失败6次" in fb
            assert "surprise=0.95" in fb
        finally:
            set_shared_phase_gate_state(None)


class TestCheckGateReviewer:
    """_check_gate 走 override 分支时, history 条目带 reviewer 字段."""

    def test_check_gate_override_records_actor(self):
        """override 放行 → PhaseGate.history 写一条 reviewer 来自 override_meta."""
        from huginn.autoloop.engine import AutoloopEngine
        from huginn.autoloop.phase_gate import (
            PhaseGateState, PhaseGateHook, set_shared_phase_gate_state,
            get_shared_phase_gate_state,
        )
        state = PhaseGateState()
        set_shared_phase_gate_state(state)
        try:
            # 加 override + meta
            key = ("plan", "execute")
            state.overrides.add(key)
            state.override_meta[key] = {
                "ts": "2026-07-13T00:00:00Z",
                "actor": "alice",
                "reason": "manual_override",
            }
            # 构造 engine: phase_gate_hook 是必需的
            engine = object.__new__(AutoloopEngine)
            engine.phase_gate_hook = PhaseGateHook()
            ok = engine._check_gate("plan", "execute", {"mode": "coder", "description": "x"})
            assert ok is True
            last = state.last_gate()
            assert last.status == "approved"
            assert last.reviewer == "alice"
            assert "override 放行" in last.feedback
        finally:
            set_shared_phase_gate_state(None)

    def test_check_gate_override_falls_back_to_user_when_no_meta(self):
        """override 在 set 里但 override_meta 缺 → reviewer 默认 'user'."""
        from huginn.autoloop.engine import AutoloopEngine
        from huginn.autoloop.phase_gate import (
            PhaseGateState, PhaseGateHook, set_shared_phase_gate_state,
            get_shared_phase_gate_state,
        )
        state = PhaseGateState()
        set_shared_phase_gate_state(state)
        try:
            state.overrides.add(("plan", "execute"))
            # 不写 override_meta — 模拟老调用路径
            engine = object.__new__(AutoloopEngine)
            engine.phase_gate_hook = PhaseGateHook()
            ok = engine._check_gate("plan", "execute", {"mode": "coder", "description": "x"})
            assert ok is True
            last = state.last_gate()
            assert last.reviewer == "user"
        finally:
            set_shared_phase_gate_state(None)


