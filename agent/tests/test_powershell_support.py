"""PowerShell 执行支持的安全回归测试.

覆盖:
  (a) 默认 posix 白名单不含 pwsh/powershell
  (b) HUGINN_ALLOW_POWERSHELL=1 (或 allow_powershell 参数) 打开后白名单含 pwsh
  (c) 含 -enc / -EncodedCommand 的 powershell/pwsh 命令被 command_filter 拒绝
  (d) 非法 -enc 路径不进入白名单 (policy 层 deny)

Run with: pytest tests/test_powershell_support.py -q -p no:randomly
"""

from __future__ import annotations

import os

import pytest


def _clear_pwsh_env() -> None:
    """清除 PowerShell 开关环境变量, 保证用平台默认值."""
    os.environ.pop("HUGINN_ALLOW_POWERSHELL", None)


def _assume_posix() -> None:
    """本套验证以 posix 环境为前提 (CI/沙箱通常为 Linux)."""
    if os.name == "nt":
        pytest.skip("posix-only assertion (win32 默认允许 PowerShell)")


def test_default_posix_whitelist_excludes_pwsh() -> None:
    _assume_posix()
    _clear_pwsh_env()
    from huginn.security.sandbox import SandboxConfig

    cfg = SandboxConfig()
    # 默认 posix 白名单不应含 pwsh/powershell, 不破坏既有安全假设.
    assert "pwsh" not in cfg.allowed_executables
    assert "powershell" not in cfg.allowed_executables


def test_env_switch_enabled_adds_pwsh() -> None:
    os.environ["HUGINN_ALLOW_POWERSHELL"] = "1"
    try:
        from huginn.security.sandbox import SandboxConfig

        cfg = SandboxConfig()
        # 开关打开后 pwsh/powershell 应进入白名单.
        assert cfg.allow_powershell is True
        assert "pwsh" in cfg.allowed_executables
        assert "powershell" in cfg.allowed_executables
    finally:
        os.environ.pop("HUGINN_ALLOW_POWERSHELL", None)


def test_allow_powershell_param_controls_whitelist() -> None:
    from huginn.security.sandbox import SandboxExecutor

    # allow_powershell=True 显式打开 → 白名单含 pwsh.
    se_on = SandboxExecutor(allow_powershell=True)
    assert "pwsh" in se_on.config.allowed_executables
    assert "powershell" in se_on.config.allowed_executables

    # allow_powershell=False 显式关闭 → 白名单不含 pwsh (即使默认打开亦回收).
    se_off = SandboxExecutor(allow_powershell=False)
    assert "pwsh" not in se_off.config.allowed_executables
    assert "powershell" not in se_off.config.allowed_executables


@pytest.mark.parametrize(
    "cmd",
    [
        ["powershell", "-enc", "SQBFAFgAKA..."],
        ["powershell", "-EncodedCommand", "SQBFAFgAKA..."],
        ["pwsh", "-enc", "SQBFAFgAKA..."],
        ["pwsh", "-EncodedCommand", "SQBFAFgAKA..."],
        ["pwsh.exe", "-EncodedCommand", "SQBFAFgAKA..."],
    ],
)
def test_encoded_command_rejected_by_command_filter(cmd) -> None:
    from huginn.security.command_filter import check_command_safety

    # EncodedCommand / -enc / pwsh / .exe 变体应一律视为危险模式被拦下,
    # 防止用 -EncodedCommand 绕过 -enc 正则 (参数注入安全护栏).
    assert check_command_safety(cmd).is_safe is False


@pytest.mark.parametrize(
    "cmd",
    [
        ["powershell", "-enc", "AAAA"],
        ["powershell", "-EncodedCommand", "AAAA"],
        ["pwsh", "-enc", "AAAA"],
        ["pwsh", "-EncodedCommand", "AAAA"],
    ],
)
def test_encoded_command_not_whitelisted(cmd) -> None:
    from huginn.security.policy_engine import evaluate_command_hook

    # 即使 pwsh/powershell 被允许执行, 携带 -enc/-EncodedCommand 的"非法路径"
    # 也不能进入可执行集合 — 策略层 deny 优先于白名单放行.
    decision = evaluate_command_hook(cmd)
    assert decision.action == "deny", decision.reason
