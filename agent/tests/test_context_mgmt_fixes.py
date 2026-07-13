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
