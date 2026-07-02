"""Permission system — inspired by Claude Code's utils/permissions/.

Three-level permission model:
- AUTO: read-only / safe tools execute without confirmation
- ASK: potentially expensive / destructive tools require confirmation
- DENY: explicitly blocked tools cannot be executed
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from huginn.types import PermissionMode, PermissionResult

# 危险命令模式 — 命中任意一条就强制 ASK, 即使 auto_approve_all=True 也要人工确认.
# 主要为了防止 yolo 模式下误删仓库或者把系统搞坏.
DANGEROUS_PATTERNS: list[str] = [
    r"rm\s+-rf?\s+/",           # rm -rf /
    r"rm\s+-rf?\s+\*",          # rm -rf *
    r"rm\s+-rf?\s+~",           # rm -rf ~
    r"git\s+push\s+.*--force",  # git push --force
    r"git\s+push\s+.*-f\b",     # git push -f
    r"git\s+reset\s+--hard",    # git reset --hard
    r"git\s+clean\s+-fd",       # git clean -fd
    r"chmod\s+-R\s+777",        # chmod -R 777
    r"dd\s+if=.*of=/dev/",      # dd 写设备
    r"mkfs\.",                  # 格式化
    r":\(\)\{.*\|.*&\};",       # fork bomb
    r">\s*/dev/sda",            # 直接写设备
    r"curl\s+.*\|\s*sh",        # curl | sh
    r"wget\s+.*\|\s*sh",        # wget | sh
    r"shutdown\s",              # 关机
    r"reboot\s",                # 重启
]

# Default permission rules for material science tools
# Note: science_* tools (science-skills bridge) are auto-approved via wildcard
# prefix matching in PermissionConfig.get_mode() — no entries needed here.
DEFAULT_PERMISSION_RULES: dict[str, PermissionMode] = {
    # Read-only / safe tools
    "structure_tool": PermissionMode.AUTO,
    "extract_tool": PermissionMode.AUTO,
    "diff_tool": PermissionMode.AUTO,
    "database_tool": PermissionMode.AUTO,
    "materials_database_tool": PermissionMode.AUTO,
    "experimental_data_tool": PermissionMode.AUTO,
    "descriptor_tool": PermissionMode.AUTO,
    "structural_analytical_tool": PermissionMode.AUTO,
    "specialty_analysis_tool": PermissionMode.AUTO,
    "fem_tool": PermissionMode.AUTO,
    "validate_tool": PermissionMode.AUTO,
    "visualize_tool": PermissionMode.AUTO,
    "web_search_tool": PermissionMode.AUTO,
    # agentic_search_tool: 只读多跳检索, 无副作用, 放行
    "agentic_search_tool": PermissionMode.AUTO,
    # onboarding_tool: 只读写本地 taste_profile.json, 无副作用, 放行
    "onboarding_tool": PermissionMode.AUTO,
    # phase_tool: 读门状态/补证据/请求评审 无副作用, 放行; override 内部过 ASK
    "phase_tool": PermissionMode.AUTO,
    # 短期补强: 只读分析类
    "gap_analysis_tool": PermissionMode.AUTO,
    "doe_tool": PermissionMode.AUTO,
    "debugger_tool": PermissionMode.AUTO,
    # 中期补强: 状态管理 + 参数微调 (不破坏数据)
    "design_plan_tool": PermissionMode.AUTO,
    "nudge_tool": PermissionMode.AUTO,
    # 长期补强: 原子渲染只读, generative_design 测试只走 html 模式
    "design_atom_tool": PermissionMode.AUTO,
    "generative_design_tool": PermissionMode.AUTO,
    # CV 扩展: 图像分析只读, 7 个 action 全部本地计算无副作用
    "image_analysis_tool": PermissionMode.AUTO,
    # CV 扩展: 输出设计只读, 只生成图片文件不修改输入
    "image_design_tool": PermissionMode.AUTO,
    # Medium risk — ask for confirmation
    "vasp_tool": PermissionMode.ASK,
    "lammps_tool": PermissionMode.ASK,
    "comsol_tool": PermissionMode.ASK,
    "qe_tool": PermissionMode.ASK,
    "cp2k_tool": PermissionMode.ASK,
    "openfoam_tool": PermissionMode.ASK,
    "packing_tool": PermissionMode.ASK,
    "abaqus_tool": PermissionMode.ASK,
    "fenics_tool": PermissionMode.ASK,
    "elmer_tool": PermissionMode.ASK,
    "code_tool": PermissionMode.ASK,
    "gromacs_tool": PermissionMode.ASK,
    "job_tool": PermissionMode.ASK,
    "batch_tool": PermissionMode.ASK,
    "container_tool": PermissionMode.ASK,
    # Destructive — always ask
    "notebook_edit_tool": PermissionMode.ASK,
    "git_commit_tool": PermissionMode.ASK,
    # Dangerous — deny by default
    "file_delete_tool": PermissionMode.DENY,
    "system_shell_tool": PermissionMode.DENY,
    # Coder tools
    "file_read_tool": PermissionMode.AUTO,
    "git_tool": PermissionMode.AUTO,
    # github_tool: 读动作在 tool 内部跳过权限检查直接执行, 写动作在 call() 里过权限
    "github_tool": PermissionMode.ASK,
    "file_write_tool": PermissionMode.ASK,
    "file_edit_tool": PermissionMode.ASK,
    "bash_tool": PermissionMode.ASK,
}


@dataclass
class PermissionConfig:
    """User-configurable permission settings."""

    rules: dict[str, PermissionMode] = field(
        default_factory=lambda: DEFAULT_PERMISSION_RULES.copy()
    )
    auto_approve_all: bool = False  # For CI/automation mode
    # plan mode: 把所有写工具降级成 ASK, 只读工具保持 AUTO
    plan_mode: bool = False

    def get_mode(self, tool_name: str) -> PermissionMode:
        # 先按 rules / 通配规则算出"原始"模式
        if tool_name in self.rules:
            mode = self.rules[tool_name]
        elif tool_name.startswith("science_"):
            mode = PermissionMode.AUTO
        else:
            mode = PermissionMode.ASK

        # plan mode 优先级最高: 只读工具(AUTO)放行, DENY 继续拦, 其它一律 ASK
        # 即使 auto_approve_all=True, 写工具在 plan mode 下也必须人工确认
        if self.plan_mode:
            if mode == PermissionMode.AUTO:
                return PermissionMode.AUTO
            if mode == PermissionMode.DENY:
                return PermissionMode.DENY
            return PermissionMode.ASK

        if self.auto_approve_all:
            return PermissionMode.AUTO
        return mode

    def set_mode(self, tool_name: str, mode: PermissionMode) -> None:
        self.rules[tool_name] = mode


class PermissionChecker:
    """Checks permissions before tool execution."""

    def __init__(self, config: PermissionConfig | None = None):
        self.config = config or PermissionConfig()

    def _check_dangerous(self, tool_name: str, args: dict | None = None) -> tuple[bool, str | None]:
        """检查工具参数是否命中危险模式.

        返回 (is_dangerous, matched_pattern). matched_pattern 是命中的正则字符串,
        方便上游在 reason 里告诉用户到底触发了哪条规则.
        """
        if not args:
            return False, None

        # bash_tool: 把 command 拼回字符串后做正则匹配
        if tool_name == "bash_tool":
            cmd = args.get("command", [])
            if isinstance(cmd, list):
                cmd_str = " ".join(str(c) for c in cmd)
            else:
                cmd_str = str(cmd)
            for pattern in DANGEROUS_PATTERNS:
                if re.search(pattern, cmd_str, re.IGNORECASE):
                    return True, pattern

        # file_delete_tool: 删除操作一律视为危险, 不看参数
        if tool_name == "file_delete_tool":
            return True, "file_delete_tool"

        return False, None

    async def check(
        self,
        tool_name: str,
        is_read_only: bool = False,
        is_destructive: bool = False,
        cost_estimate: dict[str, float] | None = None,
        args: dict | None = None,
    ) -> PermissionResult:
        # 危险命令检测优先级最高 — 即使 auto_approve_all=True 也要拦下来要求确认.
        # 这条检查必须放在 get_mode() 之前, 不然 yolo 模式会直接放行.
        is_dangerous, matched = self._check_dangerous(tool_name, args)
        if is_dangerous:
            reason = (
                f"Tool '{tool_name}' matches dangerous pattern '{matched}' — "
                "requires explicit approval even in auto-approve mode"
            )
            return PermissionResult(mode=PermissionMode.ASK, reason=reason)

        mode = self.config.get_mode(tool_name)

        if mode == PermissionMode.DENY:
            return PermissionResult(
                mode=PermissionMode.DENY,
                reason=f"Tool '{tool_name}' is explicitly blocked by permission policy",
            )

        if mode == PermissionMode.AUTO:
            return PermissionResult(mode=PermissionMode.AUTO)

        # ASK mode — build a reason string
        reasons = []
        if is_destructive:
            reasons.append("this operation is destructive")
        if cost_estimate:
            cpu = cost_estimate.get("cpu_hours", 0)
            if cpu > 1:
                reasons.append(f"estimated cost: {cpu:.1f} CPU hours")

        reason = f"Tool '{tool_name}' requires approval"
        if reasons:
            reason += f" ({', '.join(reasons)})"

        return PermissionResult(mode=PermissionMode.ASK, reason=reason)
