"""eco_tool 生态管理工具的单元测试.

全部用 mock, 不真实联网、不真实连接 MCP server:
- skill_install 走本地文件/目录 + mock 拉取 (``_fetch_url_text``).
- plugin_enable/disable 用 fake loader 替换 ``_get_plugin_loader``.
- mcp_connect 用 fake manager / fake config / fake 注册方法替换, 避免 import ``mcp``
  包 (本环境未安装该 pip 包, 任何真实路径都不应触发 ``import mcp``).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from huginn.skills.registry import SkillRegistry
from huginn.tools.eco_tool import EcoTool

# 一个合法的 SKILL.md: 机械式解析不透传任何 steps, 仅作为元数据注册.
_VALID_SKILL = """---
name: eco_demo_skill
description: eco tool test skill
---
# Demo
"""


@pytest.fixture
def tool() -> EcoTool:
    return EcoTool()


@pytest.fixture
def skills_dir(tmp_path: Path) -> Path:
    return tmp_path / "skills"


def _clear_registered_skill(name: str) -> None:
    SkillRegistry._skills.pop(name, None)


# ── skill_install ─────────────────────────────────────────────────


async def test_skill_install_local_file(tool, skills_dir, monkeypatch):
    src = Path(skills_dir.parent) / "SKILL.md"
    src.write_text(_VALID_SKILL, encoding="utf-8")
    monkeypatch.setattr(tool, "_skills_install_dir", lambda: skills_dir)

    result = await tool.call(
        {"action": "skill_install", "local_path": str(src)}, context=None
    )

    assert result.success is True
    data = result.data
    assert data["action"] == "skill_install"
    assert data["result"]["count"] == 1
    assert data["result"]["installed"][0]["name"] == "eco_demo_skill"
    # 落盘: <skills_dir>/eco_demo_skill/SKILL.md
    assert (skills_dir / "eco_demo_skill" / "SKILL.md").exists()
    # 注册进 SkillRegistry
    assert SkillRegistry.get("eco_demo_skill") is not None
    _clear_registered_skill("eco_demo_skill")


async def test_skill_install_local_dir(tool, skills_dir):
    src = skills_dir.parent / "my_skill"
    src.mkdir(parents=True)
    (src / "SKILL.md").write_text(_VALID_SKILL, encoding="utf-8")
    (src / "scripts").mkdir()
    (src / "scripts" / "run.py").write_text("print('hi')\n", encoding="utf-8")
    tool._skills_install_dir = lambda: skills_dir

    result = await tool.call(
        {"action": "skill_install", "local_path": str(src)}, context=None
    )

    assert result.success is True
    # 目录整体复制, 保留脚本
    assert (skills_dir / "my_skill" / "scripts" / "run.py").exists()
    assert SkillRegistry.get("eco_demo_skill") is not None
    _clear_registered_skill("eco_demo_skill")


async def test_skill_install_url_via_mock_fetch(tool, skills_dir, monkeypatch):
    # mock 网络拉取, 不真实联网; 验证 url 分支的落盘+注册.
    monkeypatch.setattr(tool, "_skills_install_dir", lambda: skills_dir)
    monkeypatch.setattr(tool, "_fetch_url_text", lambda url: _VALID_SKILL)

    result = await tool.call(
        {"action": "skill_install", "url": "github://acme/skills/eco_demo_skill"},
        context=None,
    )

    assert result.success is True
    assert result.data["result"]["count"] == 1
    assert SkillRegistry.get("eco_demo_skill") is not None
    _clear_registered_skill("eco_demo_skill")


async def test_skill_install_url_network_failure(tool, skills_dir, monkeypatch):
    monkeypatch.setattr(tool, "_skills_install_dir", lambda: skills_dir)

    def _boom(_url: str) -> str:
        raise ValueError("connection refused")

    monkeypatch.setattr(tool, "_fetch_url_text", _boom)
    result = await tool.call(
        {"action": "skill_install", "url": "https://example.com/SKILL.md"},
        context=None,
    )

    assert result.success is False
    assert "connection refused" in result.error


async def test_skill_install_needs_source(tool):
    result = await tool.call({"action": "skill_install"}, context=None)
    assert result.success is False
    assert "local_path 或 url" in result.error


# ── plugin_enable / plugin_disable ────────────────────────────────


async def test_plugin_enable_and_disable(tool, tmp_path):
    """验证 eco_tool 会调用 fake loader 的 load_one/unload, 且隔离 PluginLoader 依赖."""
    from collections import namedtuple

    Meta = namedtuple("Meta", "name version")
    plugin_dir = tmp_path / "demo_plugin"
    plugin_dir.mkdir()
    (plugin_dir / "metadata.yaml").write_text(
        "name: demo_plugin\nversion: 1.0.0\n", encoding="utf-8"
    )
    calls: list[tuple[str, str]] = []

    class FakeLoader:
        plugins_dir = str(tmp_path)

        def load_one(self, d):
            calls.append(("load_one", str(d)))
            return Meta("demo_plugin", "1.0.0")

        def unload(self, name):
            calls.append(("unload", name))
            return True

        def list_loaded(self):
            return ["demo_plugin"]

    fake = FakeLoader()
    tool._get_plugin_loader = lambda: fake

    enable = await tool.call(
        {"action": "plugin_enable", "name": "demo_plugin"}, context=None
    )
    assert enable.success is True
    assert enable.data["result"]["plugin"] == "demo_plugin"
    assert calls[0][0] == "load_one"

    disable = await tool.call(
        {"action": "plugin_disable", "name": "demo_plugin"}, context=None
    )
    assert disable.success is True
    assert calls[-1] == ("unload", "demo_plugin")


async def test_plugin_enable_real_dir_loads(tool, tmp_path):
    """给一个含 metadata.yaml 的目录, 验证 eco_tool 走 loader.load_one."""
    plugin_dir = tmp_path / "demo_plugin"
    plugin_dir.mkdir()
    (plugin_dir / "metadata.yaml").write_text(
        "name: demo_plugin\nversion: 1.0.0\n", encoding="utf-8"
    )
    # 复用真实 PluginLoader, 但 main.py 缺失会抛错 → 用 fake loader 证明调用链.
    fake = SimpleNamespace(
        plugins_dir=str(tmp_path),
        load_one=lambda d: SimpleNamespace(name="demo_plugin", version="1.0.0"),
        list_loaded=lambda: ["demo_plugin"],
    )
    tool._get_plugin_loader = lambda: fake

    result = await tool.call(
        {"action": "plugin_enable", "name": "demo_plugin"}, context=None
    )
    assert result.success is True
    assert result.data["result"]["plugin"] == "demo_plugin"


async def test_plugin_disable_calls_unload(tool):
    fake = SimpleNamespace(
        plugins_dir=".",
        unload=lambda name: True,
        list_loaded=lambda: [],
    )
    tool._get_plugin_loader = lambda: fake

    result = await tool.call(
        {"action": "plugin_disable", "name": "demo_plugin"}, context=None
    )
    assert result.success is True
    assert result.data["result"] == {"plugin": "demo_plugin"}


async def test_plugin_disable_not_loaded(tool):
    fake = SimpleNamespace(unload=lambda name: False)
    tool._get_plugin_loader = lambda: fake

    result = await tool.call({"action": "plugin_disable", "name": "nope"}, context=None)
    assert result.success is False
    assert "无需卸载" in result.error


# ── mcp_connect ───────────────────────────────────────────────────


class _FakeConfig:
    def __init__(self, name, transport, command, args, env, url):
        self.name = name
        self.transport = transport
        self.command = command
        self.args = args
        self.env = env
        self.url = url


class _FakeManager:
    """fake MCPClientManager: 记录 connect 调用, 不做真实握手."""

    def __init__(self):
        self.connected = []  # [(config, origin)]

    async def connect(self, config, origin=None):
        self.connected.append((config, origin))


async def test_mcp_connect_stdio(tool, monkeypatch):
    mgr = _FakeManager()

    def _fake_resolve():
        return mgr

    def _fake_build(name, transport, command, args, env, url):
        return _FakeConfig(
            name=name, transport=transport, command=command, args=args, env=env, url=url
        )

    async def _fake_register(mgr_, server_name):
        return [SimpleNamespace(name="demo_tool", description="from demo mcp")]

    monkeypatch.setattr(tool, "_resolve_mcp_manager", _fake_resolve)
    monkeypatch.setattr(tool, "_build_mcp_config", _fake_build)
    monkeypatch.setattr(tool, "_register_mcp_tools_and_refresh", _fake_register)

    result = await tool.call(
        {
            "action": "mcp_connect",
            "name": "demo-server",
            "transport": "stdio",
            "command": "python",
            "args": ["server.py"],
            "env": {"KEY": "V"},
        },
        context=None,
    )

    assert result.success is True
    assert len(mgr.connected) == 1
    config, origin = mgr.connected[0]
    assert config.name == "demo-server"
    assert config.command == "python"
    assert config.args == ["server.py"]
    assert config.env == {"KEY": "V"}
    assert origin == "eco_tool"
    assert result.data["result"]["tools"] == [
        {"name": "demo_tool", "description": "from demo mcp"}
    ]


async def test_mcp_connect_sse(tool, monkeypatch):
    mgr = _FakeManager()

    def _fake_resolve():
        return mgr

    def _fake_build(name, transport, command, args, env, url):
        return _FakeConfig(
            name=name, transport=transport, command=command, args=args, env=env, url=url
        )

    async def _fake_register(mgr_, server_name):
        return []

    monkeypatch.setattr(tool, "_resolve_mcp_manager", _fake_resolve)
    monkeypatch.setattr(tool, "_build_mcp_config", _fake_build)
    monkeypatch.setattr(tool, "_register_mcp_tools_and_refresh", _fake_register)

    result = await tool.call(
        {
            "action": "mcp_connect",
            "name": "sse-server",
            "transport": "sse",
            "url": "https://example.com/mcp",
        },
        context=None,
    )

    assert result.success is True
    assert mgr.connected[0][0].transport == "sse"
    assert mgr.connected[0][0].url == "https://example.com/mcp"


async def test_mcp_connect_requires_name(tool, monkeypatch):
    monkeypatch.setattr(tool, "_resolve_mcp_manager", _FakeManager)
    result = await tool.call(
        {"action": "mcp_connect", "transport": "stdio"}, context=None
    )
    assert result.success is False
    assert "需要 name" in result.error


# ── URL 解析 (纯逻辑, 不联网) ─────────────────────────────────────


def test_github_to_raw():
    assert EcoTool._github_to_raw("github://a/b") == (
        "https://raw.githubusercontent.com/a/b/HEAD/SKILL.md"
    )
    assert EcoTool._github_to_raw("github://a/b/path/skill") == (
        "https://raw.githubusercontent.com/a/b/HEAD/path/skill/SKILL.md"
    )


def test_to_http_url_rejects_unknown_scheme():
    with pytest.raises(ValueError):
        EcoTool._to_http_url("ftp://x/y")
