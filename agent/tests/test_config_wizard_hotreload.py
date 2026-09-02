"""ConfigWizardTool 热生效衔接 — 对话改配置写盘后触发 agent 热重建(best-effort).

回归: config_wizard_tool 原先只用裸 ``cfg.save``, 不像 routes/_persist_config 那样
清缓存 + 重建 agent, 导致"对话配好模型 → 同一会话直接用"需要重启. 本测试验证
新增 ``_persist_config`` 在保存后真的触发热重建路径, 且无 server 上下文时不崩.
"""

from __future__ import annotations

from types import SimpleNamespace

from huginn.tools.config_wizard_tool import ConfigWizardTool


async def test_setup_local_model_persists_and_triggers_hot_reload(
    tmp_path, monkeypatch
):
    cfg_path = tmp_path / "cfg.toml"

    async def _fake_connectivity(self, config):  # noqa: ANN001
        del self, config
        return {"success": True, "latency_ms": 1, "error": None, "model_response": "hi"}

    monkeypatch.setattr(
        ConfigWizardTool, "_test_local_connectivity", _fake_connectivity
    )

    # 记录"热重建"是否真实触发: fake ctx 的 4 个 agent 字段应被置 None.
    fake_ctx = SimpleNamespace(
        agent="x", agent_factory="x", planner_agent="x", orchestrator="x"
    )
    monkeypatch.setattr("huginn.server_core.get_context", lambda: fake_ctx)
    monkeypatch.setattr("huginn.pet.configure_pet", lambda *a, **k: None)

    tool = ConfigWizardTool()
    result = await tool.call(
        {
            "action": "setup_local_model",
            "model_type": "ollama",
            "model_name": "qwen3:8b",
            "host": "localhost",
            "port": 11434,
            "config_path": str(cfg_path),
        },
        context=None,
    )
    assert result.success is True
    assert cfg_path.exists()
    assert "qwen3_8b" in cfg_path.read_text()
    # 热重建路径真实被触发: 4 个字段被置 None (对齐 routes/_persist_config).
    assert fake_ctx.agent is None
    assert fake_ctx.agent_factory is None
    assert fake_ctx.planner_agent is None
    assert fake_ctx.orchestrator is None


async def test_persist_config_no_server_context_still_writes(tmp_path, monkeypatch):
    """无 server 上下文时可被调用方触发 get_context 抛错 — 保存仍完成, 不崩."""
    cfg_path = tmp_path / "cfg2.toml"
    monkeypatch.setattr(
        "huginn.server_core.get_context",
        lambda: (_ for _ in ()).throw(RuntimeError("no server")),
    )
    tool = ConfigWizardTool()
    from huginn.config import HuginnConfig

    cfg = HuginnConfig.from_env()
    # 存一个不落任何网络依赖的配置项, 仅验证写盘 + best-effort 静默.
    tool._persist_config(cfg, cfg_path)  # noqa: SLF001
    assert cfg_path.exists()
    assert cfg_path.read_text().strip()
