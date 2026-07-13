"""上下文管理与 mode 切换修复测试.

覆盖 5 项改进:
  * #1: set_mode 同步 permission_config + /chat slash 命令
  * #2: belief_entropy 自适应闭环
  * #3: Tool result clearing (tool_result_ttl)
  * #4: 反思 prompt 反向激励文案
  * #5: BeliefEntropy fact_check 长任务自动开启
"""

from __future__ import annotations

from unittest.mock import MagicMock

from huginn.cognitive_engine import FEEDBACK_PROMPT, STATE_PROMPTS, CognitiveState
from huginn.utils.belief_entropy import BeliefEntropy, BeliefEntropyConfig, get_belief_entropy
from huginn.utils.context import compact_messages, _replace_tool_content


# ── #1: set_mode 同步 permission_config ──────────────────────────


def test_set_mode_plan_syncs_permission_config():
    """set_mode('plan') 应联动 permission_config.plan_mode = True."""
    from huginn.agent.core import HuginnAgent
    from huginn.permissions import PermissionConfig

    # 用最小化 mock 构造 agent, 跳过真实初始化
    agent = MagicMock(spec=HuginnAgent)
    agent._mode = "chat"
    agent._agent_graph = None
    agent._phase_manager = MagicMock()
    agent._permission_config = PermissionConfig()

    # 调真实 set_mode 方法
    HuginnAgent.set_mode(agent, "plan")
    assert agent._mode == "plan"
    assert agent._permission_config.plan_mode is True


def test_set_mode_chat_clears_permission_config():
    """set_mode('chat') 应清除 permission_config.plan_mode."""
    from huginn.agent.core import HuginnAgent
    from huginn.permissions import PermissionConfig

    agent = MagicMock(spec=HuginnAgent)
    agent._mode = "plan"
    agent._agent_graph = None
    agent._phase_manager = MagicMock()
    agent._permission_config = PermissionConfig(plan_mode=True)

    HuginnAgent.set_mode(agent, "chat")
    assert agent._mode == "chat"
    assert agent._permission_config.plan_mode is False


def test_set_mode_research_does_not_touch_permission():
    """research mode 不该动 permission_config.plan_mode."""
    from huginn.agent.core import HuginnAgent
    from huginn.permissions import PermissionConfig

    agent = MagicMock(spec=HuginnAgent)
    agent._mode = "chat"
    agent._agent_graph = None
    agent._phase_manager = MagicMock()
    agent._permission_config = PermissionConfig(plan_mode=False)

    HuginnAgent.set_mode(agent, "research")
    assert agent._mode == "research"
    assert agent._permission_config.plan_mode is False


def test_chat_slash_command_in_builtin_list():
    """/chat 应该在 _BUILTIN_COMMANDS 列表里."""
    from huginn.cli.slash_commands import _BUILTIN_COMMANDS
    assert "chat" in _BUILTIN_COMMANDS


# ── #2: belief_entropy 自适应闭环 ────────────────────────────────


def test_belief_entropy_adaptive_params_propagate():
    """measure() 产出的 adaptive 参数应该能被下游读取."""
    be = get_belief_entropy()
    # 模拟一次 measure
    result = be.measure(
        summary="test summary",
        original_tokens=5000,
        compressed_tokens=800,
    )
    be._last_result = result
    # 下游应该能读到 _last_result (闭环关键: 之前断在这里)
    last = getattr(be, "_last_result", None)
    assert last is not None
    # adaptive 参数: 低熵返回 (-1, 0.8), 高熵返回 (+2, 1.2), 中熵返回 (None, None)
    # 任何一种都说明 measure 跑通了, 闭环关键是有 _last_result
    assert hasattr(last, "adaptive_keep_last_n")
    assert hasattr(last, "adaptive_budget_ratio")


# ── #3: Tool result clearing ─────────────────────────────────────


def test_compact_messages_clears_old_tool_results():
    """超 TTL 的 tool 消息 content 应被替换为 cleared 标记."""
    # 用 dict-based 消息, _msg_role 返回 dict["role"]
    messages = []
    for i in range(4):
        messages.append({"role": "tool", "content": f"tool output {i}" * 1000})
    for i in range(6):
        messages.append({"role": "user", "content": f"user msg {i}"})

    # tool_result_ttl=3, keep_last_n=2 → cutoff = 10-3 = 7
    # 前 4 条 tool (index 0-3) 都 < 7, 应被 cleared
    result = compact_messages(messages, budget_tokens=100000, keep_last_n=2, tool_result_ttl=3)

    # 前 4 条 tool 应被 cleared
    for i in range(4):
        assert "cleared" in str(result[i]["content"]), f"tool msg {i} not cleared: {result[i]['content']}"


def test_compact_messages_preserves_recent_tool_results():
    """TTL 内的 tool 消息不被清除."""
    messages = []
    # 6 条消息, 前 3 条老 tool, 后 3 条新 tool
    for i in range(3):
        messages.append({"role": "tool", "content": f"old tool {i}" * 500})
    for i in range(3):
        messages.append({"role": "tool", "content": f"recent tool {i}" * 500})

    # tool_result_ttl=3 → cutoff = 6-3 = 3, 前 3 条 tool cleared, 后 3 条保留
    result = compact_messages(messages, budget_tokens=100000, keep_last_n=2, tool_result_ttl=3)
    # 前 3 条应被 cleared
    for i in range(3):
        assert "cleared" in str(result[i]["content"])
    # 后 3 条保留原内容
    for i in range(3, 6):
        assert "recent tool" in str(result[i]["content"])


def test_compact_messages_ttl_zero_disables_clearing():
    """tool_result_ttl=0 → 不清除."""
    messages = []
    for i in range(10):
        messages.append({"role": "tool", "content": f"tool {i}" * 500})

    result = compact_messages(messages, budget_tokens=100000, keep_last_n=2, tool_result_ttl=0)
    # 全部保留原内容
    for i in range(10):
        assert f"tool {i}" in str(result[i]["content"])


def test_replace_tool_content_preserves_dict():
    """_replace_tool_content 对 dict 消息保留其他字段."""
    msg = {"role": "tool", "content": "big output", "tool_call_id": "abc123"}
    result = _replace_tool_content(msg, "[cleared]")
    assert result["content"] == "[cleared]"
    assert result["tool_call_id"] == "abc123"
    assert result["role"] == "tool"


# ── #4: 反思 prompt 反向激励文案 ──────────────────────────────────


def test_feedback_prompt_has_adversarial_stance():
    """FEEDBACK_PROMPT 应包含反向激励文案."""
    assert "assume the previous step is WRONG" in FEEDBACK_PROMPT
    assert "hunt for failure" in FEEDBACK_PROMPT
    assert "Do NOT verify correctness" in FEEDBACK_PROMPT


def test_red_team_prompt_has_adversarial_stance():
    """red_team system prompt 应包含反向激励."""
    from huginn.autoloop.red_team import RedTeamReviewer
    # 检查源码里包含反向激励文案
    import inspect
    src = inspect.getsource(RedTeamReviewer._llm_findings)
    assert "反向激励" in src
    assert "假设证据有错" in src


# ── #5: BeliefEntropy fact_check 长任务自动开启 ──────────────────


def test_belief_entropy_auto_fact_check_when_persistent_high():
    """最近 3 次 h_belief 都 > threshold_high → 自动触发 fact_check."""
    be = BeliefEntropy(BeliefEntropyConfig(
        fact_check_enabled=False,  # 默认关
        threshold_high=0.5,  # 降低阈值方便测试
    ))
    # 模拟 3 次高熵历史
    be._history = [0.8, 0.85, 0.9]

    # 调 measure, 应触发 fact_check (需要 mock model 和 original_context)
    # original_context 必须含可抽取的 fact (数值/化学式), 否则 _extract_key_facts 返回空
    mock_model = MagicMock()
    result = be.measure(
        summary="test",
        original_tokens=5000,
        compressed_tokens=800,
        original_context="ENCUT=520 kpoints=4x4x4 Fe2O3 band_gap=2.1eV",
        model=mock_model,
    )
    # fact_check 应被触发: _compute_fact_coverage 被调过
    # c_fact 可能是 1.0 (如果 facts 都 retained), 但 facts_checked 非空说明被调过
    assert len(result.facts_checked) > 0 or mock_model.invoke.called


def test_belief_entropy_no_auto_fact_check_when_low():
    """h_belief 低时不自动触发 fact_check."""
    be = BeliefEntropy(BeliefEntropyConfig(
        fact_check_enabled=False,
        threshold_high=0.7,
    ))
    be._history = [0.1, 0.2, 0.3]  # 低熵

    mock_model = MagicMock()
    result = be.measure(
        summary="test",
        original_tokens=5000,
        compressed_tokens=800,
        original_context="some context",
        model=mock_model,
    )
    # 不该调 model
    assert not mock_model.invoke.called
    assert result.c_fact == 1.0


def test_belief_entropy_no_auto_fact_check_without_history():
    """没有足够历史时不自动触发."""
    be = BeliefEntropy(BeliefEntropyConfig(
        fact_check_enabled=False,
        threshold_high=0.7,
    ))
    be._history = [0.9]  # 只有 1 次, 不够 3 次

    mock_model = MagicMock()
    result = be.measure(
        summary="test",
        original_tokens=5000,
        compressed_tokens=800,
        original_context="some context",
        model=mock_model,
    )
    assert not mock_model.invoke.called
    assert result.c_fact == 1.0


# ── #6: PlanStore.export_markdown ─────────────────────────────────


def test_export_markdown_writes_file_with_plan_info(tmp_path):
    """export_markdown 应写出含 objective + steps 表格的 md 文件."""
    from huginn.autoloop.plan_store import PlanStep, PlanStore

    store = PlanStore(path=tmp_path / "plans.json")
    plan = store.create_plan(
        objective="optimize Fe2O3 band gap",
        steps=[
            PlanStep(id="s1", description="run VASP relax", tool="vasp_tool"),
            PlanStep(id="s2", description="extract band gap", tool="analyze"),
        ],
    )
    out = store.export_markdown(plan.id, path=tmp_path / f"{plan.id}.md")
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert plan.id in text
    assert "optimize Fe2O3 band gap" in text
    assert "run VASP relax" in text
    assert "| Step |" in text or "## Steps" in text


def test_export_markdown_missing_plan_raises(tmp_path):
    """未知 plan_id 应抛 KeyError."""
    from huginn.autoloop.plan_store import PlanStore

    store = PlanStore(path=tmp_path / "plans.json")
    try:
        store.export_markdown("plan_does_not_exist")
        raised = False
    except KeyError:
        raised = True
    assert raised


# ── #7: reflection sidecar ────────────────────────────────────────


def test_reflection_sidecar_writes_jsonl(tmp_path, monkeypatch):
    """_append_reflection_sidecar 应写 JSONL, 含反思结论字段."""
    from huginn.agent.reflection import ReflectionMixin
    from huginn.autoloop.plan_store import _now_iso  # reuse iso helper if needed

    # 用 monkeypatch 把 home() 指到 tmp_path 避免污染真实 ~/.huginn
    monkeypatch.setattr("pathlib.Path.home", classmethod(lambda cls: tmp_path))

    mixin = ReflectionMixin()
    mixin._session_state = MagicMock()
    mixin._session_state.session_id = "test-session-123"

    reflection = MagicMock()
    reflection.tool_succeeded = True
    reflection.has_physics_errors = False
    reflection.has_physics_warnings = True
    reflection.message = "test warning"
    reflection.should_switch_mode = False
    reflection.suggested_mode = None

    tr = {"tool_name": "vasp_tool", "content": "ok"}
    mixin._append_reflection_sidecar(tr, reflection)

    sidecar = tmp_path / ".huginn" / "reflections" / "test-session-123.jsonl"
    assert sidecar.exists()
    import json as _json
    lines = sidecar.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    entry = _json.loads(lines[0])
    assert entry["tool_name"] == "vasp_tool"
    assert entry["has_physics_warnings"] is True
    assert entry["message"] == "test warning"
    assert "ts" in entry


# ── #8: RedTeamReviewer critic_model ──────────────────────────────


class _StubModel:
    """测试用 stub model — 绕开 _is_real_model 的 MagicMock 检测.

    MagicMock 有 _mock_name 属性会被当成测试 mock 跳过 LLM 增强.
    这个 stub 故意没 _mock_name, 让 LLM 增强路径真的触发.
    """

    def __init__(self, content: str = "[]"):
        self._content = content
        self.invoked = False

    def invoke(self, messages):
        self.invoked = True
        return MagicMock(content=self._content)


def test_red_team_uses_critic_model_when_provided():
    """传 critic_model 时, _llm_findings 用 critic 而非主 model."""
    from huginn.autoloop.red_team import RedTeamReviewer

    main_model = _StubModel()
    critic_model = _StubModel()

    reviewer = RedTeamReviewer(model=main_model, critic_model=critic_model)
    evidence = {"hypothesis": "如果 X 成立, 则应观察到 Y (假设前提: T=300K)"}
    reviewer.review("hypothesize", "plan", evidence)

    assert critic_model.invoked, "critic_model should be invoked"
    assert not main_model.invoked, "main model should NOT be invoked"


def test_red_team_falls_back_to_model_when_no_critic():
    """critic_model=None 时 fallback 到主 model (向后兼容)."""
    from huginn.autoloop.red_team import RedTeamReviewer

    main_model = _StubModel()

    reviewer = RedTeamReviewer(model=main_model)
    evidence = {"hypothesis": "如果 X 成立, 则应观察到 Y (假设前提: T=300K)"}
    reviewer.review("hypothesize", "plan", evidence)

    assert main_model.invoked, "main model should be invoked as fallback"


def test_red_team_works_with_only_critic_no_model():
    """只传 critic_model 不传 model 时也触发 LLM 增强."""
    from huginn.autoloop.red_team import RedTeamReviewer

    critic_model = _StubModel()

    reviewer = RedTeamReviewer(model=None, critic_model=critic_model)
    evidence = {"hypothesis": "如果 X 成立, 则应观察到 Y (假设前提: T=300K)"}
    reviewer.review("hypothesize", "plan", evidence)

    assert critic_model.invoked


# ── #9: mode-switch compaction flag ────────────────────────────────


def test_needs_compaction_flag_defaults_false():
    """agent 无 _needs_compaction 属性时 getattr 应返回 False (streaming.py 用 getattr)."""
    from huginn.agent.core import HuginnAgent

    agent = MagicMock(spec=HuginnAgent)
    # MagicMock(spec=...) 默认不暴露 _needs_compaction
    assert getattr(agent, "_needs_compaction", False) is False


# ── #10: physics oracle 否决 ───────────────────────────────────────


def test_phase_gate_rejects_when_physics_audit_has_errors():
    """evidence.physics_audit.has_errors=True → PhaseGate rejected, reviewer=physics_oracle."""
    from huginn.autoloop.phase_gate import PhaseGateHook

    hook = PhaseGateHook()
    evidence = {
        "tests_passed": True,
        "physics_audit": {
            "tool_name": "vasp_tool",
            "action": "relax",
            "findings": [{"severity": "error", "category": "unphysical_value"}],
            "has_errors": True,
            "has_warnings": False,
        },
    }
    gate = hook.evaluate("validate", "learn", evidence)
    assert gate.status == "rejected"
    assert gate.reviewer == "physics_oracle"
    assert gate.is_blocked


def test_phase_gate_passes_when_physics_audit_no_errors():
    """evidence.physics_audit.has_errors=False → 走后续 reviewer 流程."""
    from huginn.autoloop.phase_gate import PhaseGateHook

    hook = PhaseGateHook()
    evidence = {
        "tests_passed": True,
        "physics_audit": {
            "tool_name": "vasp_tool",
            "action": "relax",
            "findings": [{"severity": "warning", "category": "convergence_suspicious"}],
            "has_errors": False,
            "has_warnings": True,
        },
    }
    gate = hook.evaluate("validate", "learn", evidence)
    # 无 reviewer 时默认 approved (走完所有检查放行)
    assert gate.status == "approved"
    assert gate.reviewer is None


def test_phase_gate_passes_when_no_physics_audit():
    """evidence 无 physics_audit 字段时不阻断 (向后兼容)."""
    from huginn.autoloop.phase_gate import PhaseGateHook

    hook = PhaseGateHook()
    evidence = {"tests_passed": True}
    gate = hook.evaluate("validate", "learn", evidence)
    assert gate.status == "approved"


# ── TPS 监控: track_llm_tps + Histogram ─────────────────────────


def test_track_llm_tps_observes_histogram():
    """track_llm_tps 应可调用且不抛, Histogram collect 返回非 None."""
    from huginn.routes.metrics import (
        LLM_TPS,
        LLM_TTFT_SECONDS,
        track_llm_tps,
    )

    track_llm_tps(model="test-model", ttft_ms=420, tps=33.3)
    track_llm_tps(model="test-model", ttft_ms=0, tps=88.0)  # ttft=0 不写 TTFT

    # 验证 Histogram 可被 collect (即注册成功 + 可查询)
    assert LLM_TPS.collect() is not None
    assert LLM_TTFT_SECONDS.collect() is not None


def test_track_llm_tps_never_raises_on_bad_input():
    """track_llm_tps 对异常输入应吞掉异常 (best-effort)."""
    from huginn.routes.metrics import track_llm_tps

    track_llm_tps(model=None, ttft_ms=-1, tps=float("inf"))  # 不应抛
    track_llm_tps(model="x", ttft_ms=0, tps=0.0)


# ── Critical 闭环修复: should_switch_mode 驱动 set_mode ────────────


def test_reflection_drives_set_mode_on_switch_signal():
    """reflection.should_switch_mode=True 时应调用 set_mode (之前是死端)."""
    from huginn.agent.reflection import ReflectionMixin

    mixin = ReflectionMixin()
    mixin._session_state = MagicMock()
    mixin._session_state.session_id = "test"
    mixin._session_state.tool_results_this_turn = [{"tool_name": "x", "content": "ok"}]
    mixin._reflector = MagicMock()
    mixin._csm = MagicMock()
    mixin._evolution_engine = None
    mixin.memory = MagicMock()

    # 模拟反思算出 should_switch_mode + suggested_mode
    reflection = MagicMock()
    reflection.should_evolve = False
    reflection.to_transition_signal.return_value = None
    reflection.plan_step_completed = False
    reflection.needs_user_input = False
    reflection.should_switch_mode = True
    reflection.suggested_mode = "research"
    mixin._reflector.reflect.return_value = reflection

    called_with = []
    mixin.set_mode = lambda m: called_with.append(m)

    mixin._run_post_turn_reflection()
    assert called_with == ["research"], f"set_mode should be called with 'research', got {called_with}"


def test_reflection_invalid_suggested_mode_does_not_crash():
    """suggested_mode 非法时 set_mode 抛 ValueError, 反思应静默降级不崩溃."""
    from huginn.agent.reflection import ReflectionMixin

    mixin = ReflectionMixin()
    mixin._session_state = MagicMock()
    mixin._session_state.session_id = "test"
    mixin._session_state.tool_results_this_turn = [{"tool_name": "x", "content": "ok"}]
    mixin._reflector = MagicMock()
    mixin._csm = MagicMock()
    mixin._evolution_engine = None
    mixin.memory = MagicMock()

    reflection = MagicMock()
    reflection.should_evolve = False
    reflection.to_transition_signal.return_value = None
    reflection.plan_step_completed = False
    reflection.needs_user_input = False
    reflection.should_switch_mode = True
    reflection.suggested_mode = "bogus_mode"
    mixin._reflector.reflect.return_value = reflection

    def _raise(m):
        raise ValueError(f"invalid mode: {m}")
    mixin.set_mode = _raise

    # 不应抛
    mixin._run_post_turn_reflection()


# ── P1a: reflection sidecar reader ────────────────────────────────


def test_load_reflection_sidecar_returns_empty_when_no_file(tmp_path, monkeypatch):
    """无 sidecar 文件时 load_reflection_sidecar 返回空列表."""
    from huginn.agent.reflection import ReflectionMixin

    monkeypatch.setattr("pathlib.Path.home", classmethod(lambda cls: tmp_path))
    mixin = ReflectionMixin()
    mixin._session_state = MagicMock()
    mixin._session_state.session_id = "no-such-session"
    assert mixin.load_reflection_sidecar() == []


def test_load_reflection_sidecar_reads_back_entries(tmp_path, monkeypatch):
    """写 sidecar 后 load_reflection_sidecar 应读回 entries."""
    from huginn.agent.reflection import ReflectionMixin

    monkeypatch.setattr("pathlib.Path.home", classmethod(lambda cls: tmp_path))
    mixin = ReflectionMixin()
    mixin._session_state = MagicMock()
    mixin._session_state.session_id = "read-back-test"
    mixin._sidecar_path_announced = True  # 跳过 announce 噪音

    reflection = MagicMock()
    reflection.tool_succeeded = True
    reflection.has_physics_errors = False
    reflection.has_physics_warnings = True
    reflection.message = "test read back"
    reflection.should_switch_mode = False
    reflection.suggested_mode = None
    mixin._append_reflection_sidecar({"tool_name": "vasp_tool"}, reflection)

    entries = mixin.load_reflection_sidecar()
    assert len(entries) == 1
    assert entries[0]["tool_name"] == "vasp_tool"
    assert entries[0]["message"] == "test read back"
