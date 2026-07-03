"""ComputerBooter 抽象层与命令安全过滤器的测试。

覆盖:
- check_command_safety() 全部危险模式 + 安全命令放行
- LocalBooter 的 shell / python 执行、危险命令拦截、超时
- SSHBooter 各组件 (用 fake HPCClient 替换真连)
- create_booter() 工厂的 local / ssh 模式
- ExecResult 数据类
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from huginn.hpc.client import HPCConfig
from huginn.security.booter import (
    ComputerBooter,
    ExecResult,
    LocalBooter,
    LocalPython,
    LocalShell,
    SSHBooter,
    SSHFileSystem,
    SSHPython,
    SSHShell,
    create_booter,
)
from huginn.security.command_filter import (
    CommandFilterResult,
    check_command_safety,
)
from huginn.security.sandbox import SandboxConfig, SandboxExecutor


# ── 命令安全过滤器 ─────────────────────────────────────────────


class TestCommandFilter:
    """check_command_safety 覆盖所有危险模式 + 安全命令放行。"""

    @pytest.mark.parametrize(
        "cmd",
        [
            "rm -rf /",
            "rm -rf ~",
            "rm -rf *",
            "rm -fr /",
            "mkfs.ext4 /dev/sda1",
            "dd if=/dev/zero of=/dev/sda",
            "shutdown now",
            "reboot",
            "sudo ls",
            ":(){ :|:& };",
            "kill -9 1",
            "killall python",
            "echo x > /dev/sda",
            "chmod -R 777 /",
            "chown -R user /",
        ],
    )
    def test_blocked_commands(self, cmd: str) -> None:
        result = check_command_safety(cmd)
        assert result.is_safe is False
        assert result.matched_pattern is not None

    @pytest.mark.parametrize(
        "cmd",
        [
            "ls -la",
            "python script.py",
            "echo hello",
            "vasp_std",
            "cat /etc/hostname",
            "python -c print(42)",
        ],
    )
    def test_safe_commands(self, cmd: str) -> None:
        result = check_command_safety(cmd)
        assert result.is_safe is True
        assert result.matched_pattern is None

    def test_list_and_string_input_equivalent(self) -> None:
        # list 和 str 两种输入都要能正确匹配
        assert check_command_safety(["rm", "-rf", "/"]).is_safe is False
        assert check_command_safety("rm -rf /").is_safe is False
        assert check_command_safety(["python", "script.py"]).is_safe is True
        assert check_command_safety("python script.py").is_safe is True

    def test_matched_pattern_returned(self) -> None:
        result = check_command_safety("rm -rf /")
        assert result.is_safe is False
        assert result.matched_pattern == r"rm\s+-rf\s+/"

    def test_case_insensitive(self) -> None:
        # 大写也要拦得住 (匹配前会转小写)
        assert check_command_safety("RM -RF /").is_safe is False
        assert check_command_safety("SUDO ls").is_safe is False

    def test_result_defaults(self) -> None:
        r = CommandFilterResult(is_safe=True)
        assert r.matched_pattern is None


# ── ExecResult ─────────────────────────────────────────────────


class TestExecResult:
    def test_defaults(self) -> None:
        r = ExecResult(success=True)
        assert r.success is True
        assert r.stdout == ""
        assert r.stderr == ""
        assert r.exit_code == 0
        assert r.latency_ms == 0.0
        assert r.error is None

    def test_custom_values(self) -> None:
        r = ExecResult(
            success=False,
            stdout="out",
            stderr="err",
            exit_code=2,
            latency_ms=12.5,
            error="boom",
        )
        assert r.success is False
        assert r.exit_code == 2
        assert r.latency_ms == 12.5
        assert r.error == "boom"


# ── LocalBooter ────────────────────────────────────────────────


class TestLocalBooter:
    def test_capabilities(self) -> None:
        booter = LocalBooter()
        caps = booter.capabilities()
        assert "shell" in caps
        assert "python" in caps
        # 本地后端不带文件系统抽象
        assert "filesystem" not in caps

    async def test_boot_and_shutdown_noop(self) -> None:
        booter = LocalBooter()
        # 本地后端的 boot/shutdown 是空操作, 调了不报错就行
        await booter.boot()
        await booter.shutdown()

    async def test_shell_exec_safe(self) -> None:
        booter = LocalBooter()
        result = await booter.shell.exec(["python", "-c", "print(42)"])
        assert result.success is True
        assert "42" in result.stdout
        assert result.exit_code == 0

    async def test_python_exec(self) -> None:
        booter = LocalBooter()
        result = await booter.python.exec("print('hello from python')")
        assert result.success is True
        assert "hello from python" in result.stdout

    async def test_shell_blocks_dangerous(self) -> None:
        booter = LocalBooter()
        result = await booter.shell.exec(["rm", "-rf", "/"])
        assert result.success is False
        assert result.error is not None
        assert "Blocked" in result.error

    async def test_shell_blocks_dangerous_in_whitelisted_exe(self) -> None:
        # python 在白名单里, 但参数里夹了 rm -rf / —— 验证命令过滤的纵深防御
        # 能挡住白名单程序被喂危险参数的情况
        booter = LocalBooter()
        code = 'import os; os.system("rm -rf /")'
        result = await booter.shell.exec(["python", "-c", code])
        assert result.success is False
        assert result.error is not None
        assert "Blocked" in result.error

    async def test_shell_timeout(self) -> None:
        booter = LocalBooter()
        # 沙箱内部会捕获 subprocess.TimeoutExpired, 返回 success=False / returncode=-1
        result = await booter.shell.exec(
            ["python", "-c", "import time; time.sleep(5)"], timeout=0.5
        )
        assert result.success is False
        assert result.exit_code == -1

    async def test_shell_exec_with_injected_dry_run_sandbox(self) -> None:
        # LocalBooter 接受外部注入的沙箱; 用 dry_run 验证注入链路
        cfg = SandboxConfig(
            allowed_executables={"python", "python3"}, dry_run=True
        )
        booter = LocalBooter(SandboxExecutor(cfg))
        result = await booter.shell.exec(["python", "-c", "print(42)"])
        assert result.success is True
        assert "dry-run" in result.stdout


# ── fake HPCClient ─────────────────────────────────────────────


class _FakeHPCClient:
    """最小化的 HPCClient 替身, 只实现 booter 用到的方法。

    _exec 返回 (stdout, stderr, exit_code), 和真实 HPCClient 保持一致。
    """

    def __init__(self, exec_result: tuple[str, str, int] = ("", "", 0)) -> None:
        self.exec_calls: list[str] = []
        self.uploads: list[tuple[str, str]] = []
        self.downloads: list[tuple[str, str]] = []
        self.disconnect_called = False
        self._exec_result = exec_result

    def connect(self, timeout: int = 10) -> None:
        pass

    def disconnect(self) -> None:
        self.disconnect_called = True

    def _exec(self, command: str) -> tuple[str, str, int]:
        self.exec_calls.append(command)
        return self._exec_result

    def upload_file(self, local_path: str, remote_path: str) -> None:
        self.uploads.append((local_path, remote_path))

    def download_file(self, remote_path: str, local_path: str) -> None:
        self.downloads.append((remote_path, local_path))


def _ssh_config() -> HPCConfig:
    return HPCConfig(host="hpc.example.com", username="user")


# ── SSHShell ───────────────────────────────────────────────────


class TestSSHShell:
    async def test_exec_success(self) -> None:
        shell = SSHShell(_ssh_config())
        fake = _FakeHPCClient(("hello\n", "", 0))
        shell._client = fake
        result = await shell.exec(["echo", "hello"])
        assert result.success is True
        assert "hello" in result.stdout
        assert result.exit_code == 0

    async def test_exec_failure_exit_code(self) -> None:
        shell = SSHShell(_ssh_config())
        fake = _FakeHPCClient(("", "nope", 2))
        shell._client = fake
        result = await shell.exec(["ls", "/nope"])
        assert result.success is False
        assert result.exit_code == 2
        assert result.stderr == "nope"

    async def test_exec_prepends_cwd(self) -> None:
        shell = SSHShell(_ssh_config())
        fake = _FakeHPCClient(("ok\n", "", 0))
        shell._client = fake
        await shell.exec(["ls"], cwd="/data")
        # 远端拿到的命令应带 cd 前缀
        assert fake.exec_calls == ["cd /data && ls"]

    async def test_exec_blocks_dangerous(self) -> None:
        shell = SSHShell(_ssh_config())
        fake = _FakeHPCClient(("", "", 0))
        shell._client = fake
        result = await shell.exec(["rm", "-rf", "/"])
        assert result.success is False
        assert "Blocked" in (result.error or "")
        # 危险命令不会真的发到远端
        assert fake.exec_calls == []


# ── SSHPython ──────────────────────────────────────────────────


class TestSSHPython:
    async def test_exec(self) -> None:
        py = SSHPython(_ssh_config())
        fake = _FakeHPCClient(("42\n", "", 0))
        py._shell._client = fake
        result = await py.exec("print(42)")
        assert result.success is True
        assert "42" in result.stdout
        # 远端应是 python3 -c '...' 形式
        assert fake.exec_calls[0].startswith("python3 -c ")

    async def test_exec_blocks_dangerous_code(self) -> None:
        py = SSHPython(_ssh_config())
        fake = _FakeHPCClient(("", "", 0))
        py._shell._client = fake
        # 代码里夹带 rm -rf / 应被命令过滤器拦下
        result = await py.exec('import os; os.system("rm -rf /")')
        assert result.success is False
        assert fake.exec_calls == []


# ── SSHFileSystem ──────────────────────────────────────────────


class TestSSHFileSystem:
    async def test_upload_and_download(self, tmp_path) -> None:
        fs = SSHFileSystem(_ssh_config())
        fake = _FakeHPCClient(("", "", 0))
        fs._client = fake

        local = tmp_path / "local.txt"
        local.write_text("data", encoding="utf-8")

        assert await fs.upload(str(local), "/remote/local.txt") is True
        assert fake.uploads == [(str(local), "/remote/local.txt")]

        dest = tmp_path / "downloaded.txt"
        assert await fs.download("/remote/file.txt", str(dest)) is True
        assert fake.downloads == [("/remote/file.txt", str(dest))]

    async def test_list_files_success(self) -> None:
        fs = SSHFileSystem(_ssh_config())
        fake = _FakeHPCClient(("file1.txt\nfile2.txt\n", "", 0))
        fs._client = fake
        files = await fs.list_files("/remote/dir")
        assert files == ["file1.txt", "file2.txt"]

    async def test_list_files_failure_returns_empty(self) -> None:
        fs = SSHFileSystem(_ssh_config())
        fake = _FakeHPCClient(("", "no such dir", 1))
        fs._client = fake
        files = await fs.list_files("/remote/dir")
        assert files == []


# ── SSHBooter 整体 ─────────────────────────────────────────────


class TestSSHBooter:
    def test_capabilities_include_filesystem(self) -> None:
        booter = SSHBooter(_ssh_config())
        caps = booter.capabilities()
        assert "shell" in caps
        assert "python" in caps
        assert "filesystem" in caps

    async def test_boot_uses_existing_client(self) -> None:
        # boot() 会触发 _ensure_client, 预先注入 fake 避免真连
        booter = SSHBooter(_ssh_config())
        fake = _FakeHPCClient(("", "", 0))
        booter.shell._client = fake
        await booter.boot()  # 不报错即通过

    async def test_shutdown_disconnects_clients(self) -> None:
        booter = SSHBooter(_ssh_config())
        fake = _FakeHPCClient(("", "", 0))
        booter.shell._client = fake
        booter.python._shell._client = fake
        booter.fs._client = fake
        await booter.shutdown()
        assert fake.disconnect_called is True

    async def test_shutdown_is_safe_without_clients(self) -> None:
        # 没连过就 shutdown 也不应抛异常
        booter = SSHBooter(_ssh_config())
        await booter.shutdown()


# ── create_booter 工厂 ─────────────────────────────────────────


class TestCreateBooter:
    def test_local(self) -> None:
        booter = create_booter("local")
        assert isinstance(booter, LocalBooter)
        assert isinstance(booter, ComputerBooter)
        assert "shell" in booter.capabilities()

    def test_ssh_with_hpc_config(self) -> None:
        cfg = _ssh_config()
        booter = create_booter("ssh", hpc_config=cfg)
        assert isinstance(booter, SSHBooter)
        assert "filesystem" in booter.capabilities()

    def test_ssh_with_credential_id(self, monkeypatch) -> None:
        # 不真连凭据库, 用 mock 替换 get_credential_store
        fake_store = MagicMock()
        fake_store.to_hpc_config.return_value = _ssh_config()
        monkeypatch.setattr(
            "huginn.security.credential_store.get_credential_store",
            lambda: fake_store,
        )
        booter = create_booter("ssh", credential_id="abc123")
        assert isinstance(booter, SSHBooter)
        fake_store.to_hpc_config.assert_called_once_with("abc123")

    def test_ssh_without_args_raises(self) -> None:
        with pytest.raises(ValueError):
            create_booter("ssh")

    def test_unknown_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="未知后端"):
            create_booter("kubernetes")
