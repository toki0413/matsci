"""安全默认值守护 (结构完整性修复 B 轨 T5–T7).

三类默认拒绝的可执行探针:
  1. 本地 bash 默认拒绝; 仅 ``HUGINN_ALLOW_LOCAL_BASH=1`` **且** Landlock 可用,
     或运维显式 ``HUGINN_UNSAFE_ENV=1`` 接受降级时才放行.
  2. 审批默认 ``workspace`` 档: 工作区内的写放行, 工作区外的写 DENY.
  3. 子进程环境剥离密钥: 经沙箱执行的命令读不到 ``*_API_KEY`` / ``*_TOKEN`` 等.
"""

from __future__ import annotations

import pytest

from huginn.core_types import PermissionMode
from huginn.permissions import (
    PermissionChecker,
    PermissionConfig,
    approval_mode,
    autoloop_auto_approve,
)
from huginn.security import SandboxError, get_executor
from huginn.security.child_env import sanitize_child_env
from huginn.security.execution import allow_local_bash, unsafe_env_enabled
from huginn.security.sandbox import SandboxConfig, SandboxExecutor


def _no_landlock(monkeypatch) -> None:
    monkeypatch.setattr("huginn.security.execution.landlock_available", lambda: False)


def _with_landlock(monkeypatch) -> None:
    monkeypatch.setattr("huginn.security.execution.landlock_available", lambda: True)


# ── 1. 本地 bash 默认拒绝 ────────────────────────────────────────
class TestLocalBashDefaultDeny:
    def test_no_backend_by_default(self, monkeypatch):
        monkeypatch.setenv("HUGINN_CONTAINER_RUNTIME", "none")
        monkeypatch.delenv("HUGINN_ALLOW_LOCAL_BASH", raising=False)
        monkeypatch.delenv("HUGINN_UNSAFE_ENV", raising=False)
        _no_landlock(monkeypatch)
        with pytest.raises(SandboxError, match="No execution backend available"):
            get_executor()

    def test_allow_flag_alone_is_not_enough_without_confinement(self, monkeypatch):
        monkeypatch.setenv("HUGINN_CONTAINER_RUNTIME", "none")
        monkeypatch.setenv("HUGINN_ALLOW_LOCAL_BASH", "1")
        monkeypatch.delenv("HUGINN_UNSAFE_ENV", raising=False)
        _no_landlock(monkeypatch)
        assert allow_local_bash() is False
        with pytest.raises(SandboxError, match="No execution backend available"):
            get_executor()

    def test_allow_flag_with_landlock(self, monkeypatch):
        monkeypatch.setenv("HUGINN_CONTAINER_RUNTIME", "none")
        monkeypatch.setenv("HUGINN_ALLOW_LOCAL_BASH", "1")
        monkeypatch.delenv("HUGINN_UNSAFE_ENV", raising=False)
        _with_landlock(monkeypatch)
        assert allow_local_bash() is True
        assert isinstance(get_executor(), SandboxExecutor)

    def test_allow_flag_with_explicit_unsafe(self, monkeypatch):
        monkeypatch.setenv("HUGINN_CONTAINER_RUNTIME", "none")
        monkeypatch.setenv("HUGINN_ALLOW_LOCAL_BASH", "1")
        monkeypatch.setenv("HUGINN_UNSAFE_ENV", "1")
        _no_landlock(monkeypatch)
        assert unsafe_env_enabled() is True
        assert isinstance(get_executor(), SandboxExecutor)


# ── 2. 审批三档, 默认 workspace ──────────────────────────────────
class TestApprovalMode:
    def test_defaults_to_workspace(self, monkeypatch):
        monkeypatch.delenv("HUGINN_APPROVE_MODE", raising=False)
        monkeypatch.delenv("HUGINN_AUTOLOOP_AUTO_APPROVE", raising=False)
        assert approval_mode() == "workspace"
        assert autoloop_auto_approve() is True

    @pytest.mark.parametrize("value", ["all", "deny", "workspace"])
    def test_explicit_value_is_honoured(self, monkeypatch, value):
        monkeypatch.delenv("HUGINN_AUTOLOOP_AUTO_APPROVE", raising=False)
        monkeypatch.setenv("HUGINN_APPROVE_MODE", value)
        assert approval_mode() == value
        assert autoloop_auto_approve() is (value != "deny")

    def test_invalid_value_falls_back_to_workspace(self, monkeypatch):
        monkeypatch.delenv("HUGINN_AUTOLOOP_AUTO_APPROVE", raising=False)
        monkeypatch.setenv("HUGINN_APPROVE_MODE", "bogus")
        assert approval_mode() == "workspace"

    @pytest.mark.parametrize("legacy,expected", [("1", "all"), ("0", "deny")])
    def test_legacy_bool_maps(self, monkeypatch, legacy, expected):
        monkeypatch.delenv("HUGINN_APPROVE_MODE", raising=False)
        monkeypatch.setenv("HUGINN_AUTOLOOP_AUTO_APPROVE", legacy)
        assert approval_mode() == expected


# ── 2b. workspace 档边界 ─────────────────────────────────────────
class TestWorkspaceBoundary:
    async def test_write_inside_workspace_is_allowed(self, tmp_path):
        root = tmp_path / "ws"
        root.mkdir()
        checker = PermissionChecker(
            PermissionConfig(auto_approve_all=True, workspace_root=root)
        )
        result = await checker.check(
            "file_write_tool", args={"file_path": str(root / "out.txt")}
        )
        assert result.mode == PermissionMode.AUTO

    async def test_write_outside_workspace_is_denied(self, tmp_path):
        root = tmp_path / "ws"
        root.mkdir()
        checker = PermissionChecker(
            PermissionConfig(auto_approve_all=True, workspace_root=root)
        )
        result = await checker.check(
            "file_write_tool", args={"file_path": str(tmp_path / "etc.txt")}
        )
        assert result.mode == PermissionMode.DENY
        assert "workspace:deny" in result.matched_rules

    async def test_workspace_rule_does_not_block_read_only_tools(self, tmp_path):
        root = tmp_path / "ws"
        root.mkdir()
        checker = PermissionChecker(
            PermissionConfig(auto_approve_all=True, workspace_root=root)
        )
        result = await checker.check(
            "file_read_tool", args={"file_path": str(tmp_path / "etc.txt")}
        )
        assert result.mode == PermissionMode.AUTO

    async def test_no_workspace_root_means_no_boundary(self, tmp_path):
        checker = PermissionChecker(PermissionConfig(auto_approve_all=True))
        result = await checker.check(
            "file_write_tool", args={"file_path": str(tmp_path / "etc.txt")}
        )
        assert result.mode == PermissionMode.AUTO


# ── 3. 子进程环境剥离密钥 ────────────────────────────────────────
class TestChildEnvSanitised:
    def test_sanitize_strips_secret_shaped_names(self):
        env = {
            "PATH": "/bin",
            "SAFE_VAR": "ok",
            "INTERNLM_API_KEY": "sk-x",
            "GH_TOKEN": "t",
            "DB_PASSWORD": "p",
            "AWS_SECRET_ACCESS_KEY": "s",
            "HUGINN_API_KEY": "k",
            "MY_CREDENTIALS": "c",
        }
        clean = sanitize_child_env(env)
        assert clean == {"PATH": "/bin", "SAFE_VAR": "ok"}

    def test_defaults_to_parent_env(self, monkeypatch):
        monkeypatch.setenv("INTERNLM_API_KEY", "sk-parent")
        monkeypatch.setenv("SAFE_VAR", "ok")
        clean = sanitize_child_env()
        assert "INTERNLM_API_KEY" not in clean
        assert clean["SAFE_VAR"] == "ok"

    def test_child_process_cannot_read_secret(self, tmp_path, monkeypatch):
        """真跑白名单 ``env`` 命令: 密钥不出现在子进程输出里."""
        monkeypatch.setenv("INTERNLM_API_KEY", "sk-leak-canary-123")
        monkeypatch.setenv("HUGINN_CHILD_ENV_CANARY", "visible")
        executor = SandboxExecutor(
            SandboxConfig(allowed_work_dirs={tmp_path.resolve()})
        )
        result = executor.run(["env"], cwd=tmp_path)
        assert result.success, result.stderr
        assert "sk-leak-canary-123" not in result.stdout
        assert "HUGINN_CHILD_ENV_CANARY=visible" in result.stdout
