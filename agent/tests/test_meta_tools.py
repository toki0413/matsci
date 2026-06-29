"""Tests for meta 类 7 个工具 — 行为测试, 不依赖 LLM.

覆盖: Orchestrate / Skill / Remember+Recall / Scenario / SimplePath /
Personalization / ConfigWizard
风格参考 test_design_tools.py: async def + await tool.call + 断言 result.success/data.
单例 (StyleLearner) 用 autouse fixture 重置, 防止跨测试污染.
LLM 路径用 monkeypatch 实例方法 _get_model 触发异常, 走关键词兜底分支.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from huginn.personalization import StyleLearner, set_shared_style_learner
from huginn.skills.registry import SkillRegistry
from huginn.tools.config_wizard_tool import ConfigWizardTool
from huginn.tools.memory_tool import RecallInput, RecallTool, RememberInput, RememberTool
from huginn.tools.orchestrate_tool import OrchestrateInput, OrchestrateTool
from huginn.tools.personalization_tool import PersonalizationInput, PersonalizationTool
from huginn.tools.scenario_tool import ScenarioTool, ScenarioToolInput
from huginn.tools.simple_path_tool import SimplePathTool, SimplePathToolInput
from huginn.tools.skill_tool import SkillTool, SkillToolInput


# ── 单例重置 fixture ─────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_meta_singletons():
    """每个测试前注入干净的 StyleLearner, 防止 profile 跨测试污染."""
    set_shared_style_learner(StyleLearner(":memory:"))
    yield
    set_shared_style_learner(None)


# ── 通用 fake 工具 ───────────────────────────────────────────────────


class FakeAgentFactory:
    """假 agent 工厂, list_profiles 返回固定列表."""

    def __init__(self, profile_ids=("lead", "dft"), max_concurrent=2):
        self._profiles = [SimpleNamespace(id=pid, name=pid) for pid in profile_ids]
        self.config = SimpleNamespace(max_concurrent_subagents=max_concurrent)

    def list_profiles(self):
        return self._profiles


class FakeExecutor:
    """假 skill executor, execute() 按预设返回. 记录调用参数."""

    def __init__(self, result=None, raises=None):
        self._result = result if result is not None else {"success": True, "data": {}}
        self._raises = raises
        self.calls: list[dict] = []

    async def execute(self, skill, params, exec_context):
        if self._raises:
            raise self._raises
        self.calls.append(
            {"skill": skill, "params": params, "exec_context": exec_context}
        )
        return self._result


class FakeMemoryManager:
    """假 memory manager, remember/recall 记录调用参数 + 预设返回."""

    def __init__(self, remember_mid="mid-1", recall_results=None, raises=None):
        self._mid = remember_mid
        self._results = recall_results or [{"content": "old fact"}]
        self._raises = raises
        self.remember_calls: list[dict] = []
        self.recall_calls: list[dict] = []

    def remember(self, content, category, tags, importance, tier):
        if self._raises:
            raise self._raises
        self.remember_calls.append(
            {
                "content": content,
                "category": category,
                "tags": tags,
                "importance": importance,
                "tier": tier,
            }
        )
        return self._mid

    def recall(self, query, category, tier, top_k):
        if self._raises:
            raise self._raises
        self.recall_calls.append(
            {
                "query": query,
                "category": category,
                "tier": tier,
                "top_k": top_k,
            }
        )
        return self._results


def _make_fake_orch_cls(result=None, raises=None):
    """造一个假的 Orchestrator 类, run() 返回预设结果或抛异常."""

    class _FakeOrch:
        def __init__(self, factory, memory_manager, max_concurrent):
            self.factory = factory
            self.memory_manager = memory_manager
            self.max_concurrent = max_concurrent
            self.run_calls: list[str] = []

        async def run(self, objective):
            self.run_calls.append(objective)
            if raises:
                raise raises
            return result

    return _FakeOrch


# ════════════════════════════════════════════════════════════════════
# OrchestrateTool
# ════════════════════════════════════════════════════════════════════


async def test_orchestrate_no_agent_factory_returns_error():
    """context.agent_factory=None 时短路返回 success=False."""
    tool = OrchestrateTool()
    args = OrchestrateInput(objective="测试目标")
    ctx = SimpleNamespace(
        session_id="t", workspace=".", agent_factory=None, memory_manager=None
    )
    result = await tool.call(args, context=ctx)
    assert result.success is False
    assert "factory not available" in result.error.lower()


async def test_orchestrate_unknown_agent_id_returns_error():
    """agent_ids 含未知 profile → 返回 success=False 列出可用 profile."""
    tool = OrchestrateTool()
    factory = FakeAgentFactory(profile_ids=("lead", "dft"))
    args = OrchestrateInput(objective="测试", agent_ids=["ghost", "lead"])
    ctx = SimpleNamespace(
        session_id="t", workspace=".", agent_factory=factory, memory_manager=None
    )
    result = await tool.call(args, context=ctx)
    assert result.success is False
    assert "ghost" in result.error
    assert "lead" in result.error  # 可用 profile 列表里要有 lead


async def test_orchestrate_run_success(monkeypatch):
    """正常路径: monkeypatch Orchestrator, run() 返回成功结果."""
    fake_result = SimpleNamespace(
        success=True,
        summary="all done",
        outputs={"lead": "did it"},
        objective="obj",
        error=None,
        plan=SimpleNamespace(
            tasks=[
                SimpleNamespace(
                    task_id="t1", agent_id="lead", status="done", prompt="go"
                )
            ]
        ),
    )
    monkeypatch.setattr(
        "huginn.agents.orchestrator.Orchestrator",
        _make_fake_orch_cls(result=fake_result),
    )
    tool = OrchestrateTool()
    factory = FakeAgentFactory()
    args = OrchestrateInput(objective="跑个 DFT")
    ctx = SimpleNamespace(
        session_id="t", workspace=".", agent_factory=factory, memory_manager=None
    )
    result = await tool.call(args, context=ctx)
    assert result.success is True
    assert result.data["summary"] == "all done"
    assert result.data["outputs"] == {"lead": "did it"}
    assert result.data["plan"][0]["task_id"] == "t1"
    assert result.data["plan"][0]["agent_id"] == "lead"


async def test_orchestrate_exception_swallowed(monkeypatch):
    """Orchestrator.run 抛异常时被 try/except 吞掉, 返回 success=False."""
    monkeypatch.setattr(
        "huginn.agents.orchestrator.Orchestrator",
        _make_fake_orch_cls(raises=RuntimeError("boom")),
    )
    tool = OrchestrateTool()
    factory = FakeAgentFactory()
    args = OrchestrateInput(objective="会失败的")
    ctx = SimpleNamespace(
        session_id="t", workspace=".", agent_factory=factory, memory_manager=None
    )
    result = await tool.call(args, context=ctx)
    assert result.success is False
    assert "boom" in result.error


# ════════════════════════════════════════════════════════════════════
# SkillTool
# ════════════════════════════════════════════════════════════════════


async def test_skill_list_returns_summary():
    """action=list 返回 success=True + available_skills 列表."""
    tool = SkillTool()
    args = SkillToolInput(action="list")
    ctx = SimpleNamespace(session_id="t")
    result = await tool.call(args, context=ctx)
    assert result.success is True
    assert isinstance(result.data["available_skills"], list)
    assert result.data["result"]["count"] == len(result.data["available_skills"])


async def test_skill_describe_missing_name_returns_error():
    """action=describe 但没传 skill_name → 返回 success=False."""
    tool = SkillTool()
    args = SkillToolInput(action="describe", skill_name=None)
    ctx = SimpleNamespace(session_id="t")
    result = await tool.call(args, context=ctx)
    assert result.success is False
    assert "skill_name" in result.error


async def test_skill_execute_unknown_skill_returns_error():
    """action=execute 但 skill_name 在 registry 里找不到 → 返回 success=False."""
    tool = SkillTool()
    args = SkillToolInput(action="execute", skill_name="bogus_skill_xyz")
    ctx = SimpleNamespace(session_id="t")
    result = await tool.call(args, context=ctx)
    assert result.success is False
    assert "bogus_skill_xyz" in result.error


async def test_skill_execute_passes_through_result(monkeypatch):
    """action=execute 命中已知 skill, executor 返回 success=True → 透传."""
    fake_skill = SimpleNamespace(
        name="fake_skill",
        description="fake",
        category="test",
        tags=[],
        parameters=[],
        required_tools=[],
        steps=[],
    )
    monkeypatch.setattr(
        SkillRegistry, "get", classmethod(lambda cls, name: fake_skill)
    )
    executor = FakeExecutor(result={"success": True, "data": {"x": 1}})
    tool = SkillTool(skill_executor=executor)
    args = SkillToolInput(
        action="execute", skill_name="fake_skill", parameters={"a": 1}
    )
    ctx = SimpleNamespace(session_id="sess-1")
    result = await tool.call(args, context=ctx)
    assert result.success is True
    assert result.data["result"]["data"] == {"x": 1}
    # executor 应该收到 parameters 和 session_id
    assert executor.calls[0]["params"] == {"a": 1}
    assert executor.calls[0]["exec_context"]["session_id"] == "sess-1"


# ════════════════════════════════════════════════════════════════════
# RememberTool / RecallTool
# ════════════════════════════════════════════════════════════════════


async def test_remember_no_manager_returns_error():
    """context.memory_manager=None → 返回 success=False."""
    tool = RememberTool()
    args = RememberInput(content="a fact")
    ctx = SimpleNamespace(session_id="t", memory_manager=None)
    result = await tool.call(args, context=ctx)
    assert result.success is False
    assert "memory manager" in result.error.lower()


async def test_remember_passes_args_returns_mid():
    """remember 把所有参数透传给 memory_manager, 返回 mid."""
    mgr = FakeMemoryManager(remember_mid="mid-42")
    tool = RememberTool()
    args = RememberInput(
        content="C-S-H 凝胶密度约 2.6 g/cm3",
        category="fact",
        tags=["cement", "density"],
        importance=0.8,
        tier="long",
    )
    ctx = SimpleNamespace(session_id="t", memory_manager=mgr)
    result = await tool.call(args, context=ctx)
    assert result.success is True
    assert result.data["memory_id"] == "mid-42"
    assert mgr.remember_calls[0]["content"] == "C-S-H 凝胶密度约 2.6 g/cm3"
    assert mgr.remember_calls[0]["tier"] == "long"
    assert mgr.remember_calls[0]["importance"] == 0.8


async def test_recall_passes_top_k():
    """recall 把 top_k 透传, 返回 memory_manager 的结果."""
    mgr = FakeMemoryManager(recall_results=[{"content": "old"}])
    tool = RecallTool()
    args = RecallInput(query="C-S-H", top_k=10, tier="long")
    ctx = SimpleNamespace(session_id="t", memory_manager=mgr)
    result = await tool.call(args, context=ctx)
    assert result.success is True
    assert result.data["results"] == [{"content": "old"}]
    assert mgr.recall_calls[0]["top_k"] == 10
    assert mgr.recall_calls[0]["tier"] == "long"


async def test_remember_exception_swallowed():
    """memory_manager.remember 抛异常被 try/except 吞掉, 返回 success=False."""
    mgr = FakeMemoryManager(raises=RuntimeError("db locked"))
    tool = RememberTool()
    args = RememberInput(content="x")
    ctx = SimpleNamespace(session_id="t", memory_manager=mgr)
    result = await tool.call(args, context=ctx)
    assert result.success is False
    assert "db locked" in result.error


# ════════════════════════════════════════════════════════════════════
# ScenarioTool
# ════════════════════════════════════════════════════════════════════


async def test_scenario_empty_returns_error():
    """scenario="" 时短路返回 success=False."""
    tool = ScenarioTool()
    args = ScenarioToolInput(action="match", scenario="")
    ctx = SimpleNamespace(session_id="t")
    result = await tool.call(args, context=ctx)
    assert result.success is False
    assert "不能为空" in result.error


async def test_scenario_llm_exception_falls_back_to_keywords(monkeypatch):
    """LLM 路径抛异常 → 关键词兜底, 命中已知场景."""
    tool = ScenarioTool()

    def boom(ctx):
        raise RuntimeError("no llm available")

    monkeypatch.setattr(tool, "_get_model", boom)
    args = ScenarioToolInput(action="match", scenario="我要优化 Si 的晶体结构")
    ctx = SimpleNamespace(session_id="t")
    result = await tool.call(args, context=ctx)
    assert result.success is True
    assert result.data["scenario_type"] == "dft_structure_optimization"
    assert result.data["match_info"]["source"] == "keyword"


async def test_scenario_keyword_fallback_hits_known_scenario():
    """无 LLM 调用时关键词兜底直接命中."""
    tool = ScenarioTool()

    def boom(ctx):
        raise RuntimeError("no llm")

    # 直接 monkeypatch 实例方法, 走关键词兜底分支
    tool._get_model = boom  # type: ignore[method-assign]
    args = ScenarioToolInput(action="match", scenario="帮我调研高熵合金文献")
    ctx = SimpleNamespace(session_id="t")
    result = await tool.call(args, context=ctx)
    assert result.success is True
    assert result.data["scenario_type"] == "literature_review"
    assert "web_search_tool" in result.data["recommended_tools"]


async def test_scenario_all_fail_lists_available_scenarios(monkeypatch):
    """LLM 失败 + 关键词兜底也不命中 → 返回 success=False 列出可用场景."""
    tool = ScenarioTool()

    def boom(ctx):
        raise RuntimeError("no llm")

    monkeypatch.setattr(tool, "_get_model", boom)
    args = ScenarioToolInput(action="match", scenario="完全无法识别的乱码xyz123")
    ctx = SimpleNamespace(session_id="t")
    result = await tool.call(args, context=ctx)
    assert result.success is False
    # error 里要列出可用场景, 方便 LLM 重试
    assert "dft_structure_optimization" in result.error
    assert "literature_review" in result.error


# ════════════════════════════════════════════════════════════════════
# SimplePathTool
# ════════════════════════════════════════════════════════════════════


async def test_path_empty_task_returns_error():
    """task_description="" → 短路返回 success=False."""
    tool = SimplePathTool()
    args = SimplePathToolInput(task_description="")
    ctx = SimpleNamespace(session_id="t")
    result = await tool.call(args, context=ctx)
    assert result.success is False
    assert "不能为空" in result.error


async def test_path_llm_exception_falls_back_to_keywords(monkeypatch):
    """LLM 抛异常 → 关键词兜底, match_source='keyword'."""
    tool = SimplePathTool()

    def boom(ctx):
        raise RuntimeError("no llm")

    monkeypatch.setattr(tool, "_get_model", boom)
    args = SimplePathToolInput(task_description="拟合 Murnaghan EOS")
    ctx = SimpleNamespace(session_id="t")
    result = await tool.call(args, context=ctx)
    assert result.success is True
    assert result.data["match_source"] == "keyword"
    assert "numerical_tool" in result.data["recommended_path"]


async def test_path_keyword_fallback_band_gap():
    """关键词 'band gap' 命中常量查询路径, 推荐 materials_database_tool."""
    tool = SimplePathTool()

    def boom(ctx):
        raise RuntimeError("no llm")

    tool._get_model = boom  # type: ignore[method-assign]
    args = SimplePathToolInput(task_description="I want the band gap of Silicon")
    ctx = SimpleNamespace(session_id="t")
    result = await tool.call(args, context=ctx)
    assert result.success is True
    assert result.data["match_source"] == "keyword"
    assert "materials_database_tool" in result.data["recommended_path"]
    # 带隙查询避开了 vasp/qe/cp2k
    assert "vasp_tool" in result.data["heavy_tools_avoided"]


async def test_path_default_path_when_no_match():
    """LLM 失败 + 关键词兜底都不命中 → 走默认路径, match_source='default'."""
    tool = SimplePathTool()

    def boom(ctx):
        raise RuntimeError("no llm")

    tool._get_model = boom  # type: ignore[method-assign]
    args = SimplePathToolInput(task_description="完全无法识别的随机文本xyz999")
    ctx = SimpleNamespace(session_id="t")
    result = await tool.call(args, context=ctx)
    assert result.success is True
    assert result.data["match_source"] == "default"
    assert "web_search_tool" in result.data["recommended_path"]
    # 默认路径要避开所有重型工具
    assert "vasp_tool" in result.data["heavy_tools_avoided"]


# ════════════════════════════════════════════════════════════════════
# PersonalizationTool
# ════════════════════════════════════════════════════════════════════


async def test_personalization_get_profile_returns_data():
    """get_profile 返回 success=True + profile dict 含 vocabulary_level."""
    tool = PersonalizationTool()
    args = PersonalizationInput(action="get_profile")
    result = await tool.call(args, context=None)
    assert result.success is True
    assert "vocabulary_level" in result.data
    assert "formality" in result.data
    assert result.data["sample_count"] == 0  # 全新 learner, 没观察过


async def test_personalization_set_preference_missing_dimension_returns_error():
    """set_preference 没传 dimension → 返回 success=False."""
    tool = PersonalizationTool()
    args = PersonalizationInput(action="set_preference", dimension=None, value="zh")
    result = await tool.call(args, context=None)
    assert result.success is False
    assert "dimension" in result.error


async def test_personalization_set_preference_invalid_dimension_returns_false():
    """set_preference 传不存在的 dimension → StyleLearner 返回 False, 工具报错."""
    tool = PersonalizationTool()
    args = PersonalizationInput(
        action="set_preference", dimension="bogus_dim", value="x"
    )
    result = await tool.call(args, context=None)
    assert result.success is False
    assert "无效维度" in result.error


async def test_personalization_unknown_action_returns_error():
    """未知 action → 返回 success=False 列出支持的 action."""
    tool = PersonalizationTool()
    # PersonalizationInput.action 是 str, 不限制取值, 所以能传 "bogus"
    args = PersonalizationInput(action="bogus_action")
    result = await tool.call(args, context=None)
    assert result.success is False
    assert "未知 action" in result.error
    assert "get_profile" in result.error  # error 里列出支持的 action


# ════════════════════════════════════════════════════════════════════
# ConfigWizardTool
# ════════════════════════════════════════════════════════════════════


async def test_config_list_providers_returns_19():
    """list_providers 返回 19 个 provider (含 anthropic/openai/ollama 等)."""
    tool = ConfigWizardTool()
    result = await tool.call({"action": "list_providers"}, context=None)
    assert result.success is True
    assert result.data["count"] == 19
    provider_names = [p["provider"] for p in result.data["providers"]]
    assert "anthropic" in provider_names
    assert "ollama" in provider_names
    assert "deepseek" in provider_names


async def test_config_recommend_provider_keyword_match():
    """requirement='本地部署' 命中关键词 → 推荐 ollama."""
    tool = ConfigWizardTool()
    result = await tool.call(
        {"action": "recommend_provider", "requirement": "我要本地部署, 不要 key"},
        context=None,
    )
    assert result.success is True
    assert result.data["recommended_provider"] == "ollama"
    assert "ollama" in result.data["config_example"]["provider"]


async def test_config_validate_config_missing_field(monkeypatch):
    """validate_config: model 缺 api_key → 返回 valid=False + error 级 issue."""
    fake_cfg = SimpleNamespace(
        models=[
            SimpleNamespace(
                alias="m1",
                provider="openai",
                model="gpt-4",
                api_key=None,
                base_url=None,
                enabled=True,
            )
        ],
        agents=[],
    )
    monkeypatch.setattr(
        "huginn.tools.config_wizard_tool._load_config", lambda path: fake_cfg
    )
    # resolve_provider_key 可能受 env 影响, 直接 mock 掉返回 None
    monkeypatch.setattr(
        "huginn.tools.config_wizard_tool.resolve_provider_key", lambda p, k: None
    )
    tool = ConfigWizardTool()
    result = await tool.call({"action": "validate_config"}, context=None)
    assert result.success is True  # 工具调用本身成功
    assert result.data["valid"] is False  # 但配置无效
    error_issues = [i for i in result.data["issues"] if i["severity"] == "error"]
    assert len(error_issues) >= 1
    assert any("api_key" in i["message"] for i in error_issues)


async def test_config_toggle_feature_unknown_returns_error():
    """toggle_feature 传未知 feature → 返回 success=False."""
    tool = ConfigWizardTool()
    result = await tool.call(
        {"action": "toggle_feature", "feature": "bogus_feature_xyz", "enabled": True},
        context=None,
    )
    assert result.success is False
    assert "未知 feature" in result.error


async def test_config_set_privacy_missing_level_returns_error():
    """set_privacy 没传 level → _set_privacy('') → 返回 success=False."""
    tool = ConfigWizardTool()
    result = await tool.call({"action": "set_privacy"}, context=None)
    assert result.success is False
    assert "未知隐私级别" in result.error


async def test_config_migrate_from_env_skips_when_no_env_keys(monkeypatch):
    """migrate_from_env: _PROVIDER_KEY_ENV 为空 → 找不到任何 key, migrated_count=0."""
    monkeypatch.setattr(
        "huginn.tools.config_wizard_tool._PROVIDER_KEY_ENV", {}
    )
    tool = ConfigWizardTool()
    result = await tool.call(
        {"action": "migrate_from_env", "config_path": "nonexistent_for_test.toml"},
        context=None,
    )
    assert result.success is True
    assert result.data["migrated_count"] == 0
    assert "未在环境变量里找到" in result.data["message"]
