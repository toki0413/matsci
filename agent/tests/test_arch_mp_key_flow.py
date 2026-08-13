"""制度化闭环：Materials Project API key 前端配置 → 后端 → mat-db MCP env 链路。

用户在设置面板填 MP_API_KEY → 保存 → /config → HuginnConfig.mp_api_key →
mat-db MCP 子进程 env。本门禁保证这条链路不退化:
  1. `/config` 的 legacy 映射会把 mp_api_key 刷成 MP_API_KEY 环境变量;
  2. HuginnConfig.from_env() 能读回 MP_API_KEY -> mp_api_key;
  3. mcp_manager.set_server_env 能把 key 刷进 mat-db 的注册配置(供重连用)。
"""

from __future__ import annotations

import os

from huginn.config import HuginnConfig
from huginn.mcp_client import mcp_manager
from huginn.routes.config import _apply_legacy_params_to_env


def test_mp_key_maps_into_env_and_from_env():
    os.environ.pop("MP_API_KEY", None)
    try:
        _apply_legacy_params_to_env({"mp_api_key": "sk-test-123"})
        assert os.environ.get("MP_API_KEY") == "sk-test-123"
        assert HuginnConfig.from_env().mp_api_key == "sk-test-123"
    finally:
        os.environ.pop("MP_API_KEY", None)


def test_mp_key_clear_removes_env():
    os.environ.pop("MP_API_KEY", None)
    try:
        _apply_legacy_params_to_env({"mp_api_key": ""})
        assert "MP_API_KEY" not in os.environ
        assert HuginnConfig.from_env().mp_api_key is None
    finally:
        os.environ.pop("MP_API_KEY", None)


def test_set_server_env_updates_matdb_config():
    mcp_manager.register_server("mat-db", {"command": "python", "args": ["x.py"]})
    try:
        mcp_manager.set_server_env("mat-db", {"MP_API_KEY": "sk-test-123"})
        assert mcp_manager._registry["mat-db"]["env"]["MP_API_KEY"] == "sk-test-123"
    finally:
        mcp_manager.remove_server("mat-db")
        mcp_manager._configs.pop("mat-db", None)
