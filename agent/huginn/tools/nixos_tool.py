"""NixOS 运维工具 —— agent 在 NixOS 宿主上执行系统运维的入口。

对齐参考 `dsh-nixos-shell` (DeepSeek Harness NixOS 插件) 的语义:

  - 只读诊断 (capabilities / system-status / generations / journal /
    audit-store-paths) 一律经 `huginn-nixos-cli` 执行, 无 sudo。
  - 变更性操作 `rebuild` 一律走 `huginn-rebuild switch` —— 它把 nixos-rebuild
    收进 detached 瞬态 systemd 单元 (独立 cgroup), 断连≠取消, 显式取消走
    `rebuild-cancel`。绝不让 agent 直接 sudo bash -c 跑 nixos-rebuild
    (激活阶段会重启 dsh + stop sudo.socket, 同步跑会在激活中途被整树杀掉)。

NixOS 宿主门禁: 本工具只在 NixOS 上可用 (`is_available()` 检查 /etc/NIXOS),
非 NixOS 会对 LLM 隐藏 (对齐 nixos-gate 的"非 NixOS 拒绝"语义)。
"""

from __future__ import annotations

import asyncio
import shlex
import shutil
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.core_types import ToolContext, ToolResult, ValidationResult
from huginn.tools.base import HuginnTool

_READ_ONLY_ACTIONS = frozenset(
    {
        "capabilities",
        "system-status",
        "generations",
        "journal",
        "audit-store-paths",
        "rebuild-status",
    }
)

# 只读诊断命令 (NixOS 模块安装, 见 nixos/modules/huginn-nixos-cli.nix)
_NIXOS_CLI = "huginn-nixos-cli"
# detached 重建分发助手 (见 nixos/modules/huginn-rebuild-dispatch.nix)
_REBUILD = "huginn-rebuild"


class NixosInput(BaseModel):
    action: Literal[
        "capabilities",
        "system-status",
        "generations",
        "journal",
        "audit-store-paths",
        "rebuild",
        "rebuild-status",
        "rebuild-cancel",
    ] = Field(
        ...,
        description="capabilities: modern 命令对照; system-status: 系统运行态+失败单元; "
        "generations: 系统代际; journal: 单元日志尾; audit-store-paths: 硬编码 store 路径审计; "
        "rebuild: 变更性触发 nixos-rebuild (detached+root); rebuild-status: 查重建结果; "
        "rebuild-cancel: 显式取消某次重建.",
    )
    unit: str | None = Field(
        default=None,
        description="journal / rebuild-cancel 需要: systemd 单元名 (journal 支持 glob, 尾 @ 匹配模板所有实例).",
    )
    lines: int = Field(
        default=50,
        description="journal 行数 (默认 50, 上限 500).",
    )
    limit: int = Field(
        default=20,
        description="generations 返回的最大代际数 (默认 20, 上限 200).",
    )


class NixOSTool(HuginnTool):
    """NixOS 系统运维: 只读诊断 + detached 安全重建路由。仅 NixOS 宿主可用。"""

    name = "nixos_tool"
    category = "core"

    description = (
        "NixOS system operations. Read-only diagnostics (capabilities, system-status, "
        "generations, journal, audit-store-paths) run without sudo via huginn-nixos-cli. "
        "Mutating operations (rebuild) are detached into a transient systemd unit via "
        "huginn-rebuild (survives the mid-activation dsh restart), cancelled explicitly. "
        "Only available on a NixOS host."
    )
    input_schema = NixosInput

    def is_available(self) -> bool:
        """仅在 NixOS 宿主上对 LLM 可见 (对齐 nixos-gate 宿主门禁)."""
        return _is_nixos()

    def is_read_only(self, args: NixosInput) -> bool:
        a = getattr(args, "action", None)
        return a in _READ_ONLY_ACTIONS

    async def validate_input(
        self, args: NixosInput, context: ToolContext | None = None
    ) -> ValidationResult:
        a = args.action
        if a == "journal" and not (args.unit or "").strip():
            return ValidationResult(result=False, message="journal requires a unit.")
        if a == "rebuild-cancel" and not (args.unit or "").strip():
            return ValidationResult(result=False, message="rebuild-cancel requires a unit.")
        if a == "journal" and not (1 <= args.lines <= 500):
            return ValidationResult(result=False, message="lines must be 1..500.")
        if a == "generations" and not (1 <= args.limit <= 200):
            return ValidationResult(result=False, message="limit must be 1..200.")
        return ValidationResult(result=True)

    async def _execute(self, args: NixosInput, context: ToolContext) -> ToolResult:
        # 双重门禁: is_available 已挡, 这里兜底拒绝, 不静默.
        if not _is_nixos():
            return ToolResult(
                success=False,
                data=None,
                error="nixos_tool 要求 NixOS 宿主 (/etc/NIXOS 缺失). 非 NixOS 系统请勿调用.",
            )

        argv = self._build_argv(args)
        if argv is None:
            return ToolResult(
                success=False,
                data=None,
                error=f"unknown action or missing required argument: {args.action}",
            )

        missing = [b for b in frozenset(argv) if b in (_NIXOS_CLI, _REBUILD) and shutil.which(b) is None]
        if missing:
            return ToolResult(
                success=False,
                data=None,
                error=(
                    f"缺少 NixOS 助手命令: {', '.join(missing)}. "
                    "请启用 NixOS flake 的 services.huginn-agent.nixosTools = true "
                    "(它会把 huginn-nixos-cli / huginn-rebuild 挂进服务 PATH)."
                ),
            )

        # 变更性 rebuild 走 detached 分发, 返回即交接 (结果须经 rebuild-status/journal 查).
        code, stdout, stderr, timed_out = await _run(argv, timeout=90.0)
        if timed_out:
            success = False
            data = {"action": args.action, "timedOut": True}
        elif code == 0:
            success = True
            data = {"action": args.action, "output": stdout.strip()}
        else:
            success = False
            data = {
                "action": args.action,
                "exitCode": code,
                "stdout": stdout.strip(),
                "stderr": stderr.strip(),
            }
        return ToolResult(data=data, success=success)

    def _build_argv(self, args: NixosInput) -> list[str] | None:
        a = args.action
        if a == "capabilities":
            return [_NIXOS_CLI, "capabilities"]
        if a == "system-status":
            return [_NIXOS_CLI, "system-status"]
        if a == "generations":
            return [_NIXOS_CLI, "generations", shlex.quote(str(args.limit))]
        if a == "journal":
            if not (args.unit or "").strip():
                return None
            return [_NIXOS_CLI, "journal", args.unit.strip(), shlex.quote(str(args.lines))]
        if a == "audit-store-paths":
            return [_NIXOS_CLI, "audit-store-paths"]
        if a == "rebuild":
            return [_REBUILD, "switch"]
        if a == "rebuild-status":
            return [_REBUILD, "status"]
        if a == "rebuild-cancel":
            if not (args.unit or "").strip():
                return None
            return [_REBUILD, "cancel", args.unit.strip()]
        return None


def _is_nixos() -> bool:
    try:
        if Path("/etc/NIXOS").exists() or Path("/etc/NIXOS").is_file():
            return True
    except Exception:
        pass
    try:
        text = Path("/etc/os-release").read_text(encoding="utf-8", errors="replace")
        return any(line.strip() == "ID=nixos" for line in text.splitlines())
    except Exception:
        return False


async def _run(argv: list[str], timeout: float) -> tuple[int, str, str, bool]:
    """执行子进程, 返回 (exitcode, stdout, stderr, timed_out). stdout/stderr 截断限流."""
    _MAX_BYTES = 4 * 1024 * 1024

    def _clip(data: bytes) -> str:
        return data[:_MAX_BYTES].decode("utf-8", errors="replace")

    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        code = proc.returncode if proc.returncode is not None else -1
        return code, _clip(out), _clip(err), False
    except asyncio.TimeoutError:
        proc.kill()
        try:
            out, err = await proc.communicate()
        except Exception:
            out, err = b"", b""
        return -1, _clip(out), _clip(err), True