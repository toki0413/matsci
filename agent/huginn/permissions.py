"""Permission system — inspired by Claude Code's utils/permissions/.

Three-level permission model:
- AUTO: read-only / safe tools execute without confirmation
- ASK: potentially expensive / destructive tools require confirmation
- DENY: explicitly blocked tools cannot be executed

Path-level declarative rules (inspired by Deep Agents) allow overriding
the tool-level mode based on file path glob patterns, e.g. deny *.env
or ask before writing to data/.
"""

from __future__ import annotations

import fnmatch
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from huginn.core_types import PermissionMode, PermissionResult, RiskLevel

# 危险命令模式 — 统一使用 command_filter 的 _BLOCKED_PATTERNS 作为 single source of truth.
# permissions.py 补充 git 相关模式 (command_filter 不覆盖 git).
# ponytail: 以前两处各维护一份重复且不一致的列表, 现在合并.
try:
    from huginn.security.command_filter import _BLOCKED_PATTERNS as _CF_PATTERNS
except ImportError:
    _CF_PATTERNS = []

DANGEROUS_PATTERNS: list[str] = list(_CF_PATTERNS) + [
    r"git\s+push\s+.*--force",  # git push --force
    r"git\s+push\s+.*-f\b",     # git push -f
    r"git\s+reset\s+--hard",    # git reset --hard
    r"git\s+clean\s+-fd",       # git clean -fd
]

# Read-only tool names — SSE chat 路径用来区分“可 auto_approve”和“需 ASK”.
# ponytail: frozenset 常量, O(1) 查找, 不引新依赖.
READ_ONLY_TOOLS: frozenset[str] = frozenset({
    "read_file", "list_dir", "grep", "glob", "search",
    "codebase_search", "search_codebase", "ls", "find",
    "cat", "head", "tail",
})

# Write / exec tool names — 显式标记, SSE 路径下不 auto_approve.
# 未列在 READ_ONLY_TOOLS 里的工具默认就返回 False, 这个集合留着做可观测性 / 未来显式校验.
WRITE_EXEC_TOOLS: frozenset[str] = frozenset({
    "file_write", "file_edit", "vasp_tool", "lammps_tool",
    "code_act", "bash", "shell", "subprocess",
    "git_commit", "git_push",
})


def is_read_only_tool(tool_name: str) -> bool:
    """Return True if the tool is a read-only operation safe to auto-approve.

    SSE chat 路径用这个函数收窄 auto_approve 范围: 只读工具直接放行,
    写/执行工具仍走 ASK 确认. 未知名默认 False — 按非只读处理.
    """
    return tool_name in READ_ONLY_TOOLS

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
    # Dangerous — deny by default
    "file_delete_tool": PermissionMode.DENY,
    "system_shell_tool": PermissionMode.DENY,
    # Coder tools
    "file_read_tool": PermissionMode.AUTO,
    "glob": PermissionMode.AUTO,
    "grep": PermissionMode.AUTO,
    "eval_tool": PermissionMode.AUTO,
    "git_tool": PermissionMode.AUTO,
    # github_tool: 读动作在 tool 内部跳过权限检查直接执行, 写动作在 call() 里过权限
    "github_tool": PermissionMode.ASK,
    "file_write_tool": PermissionMode.ASK,
    "file_edit_tool": PermissionMode.ASK,
    "bash_tool": PermissionMode.ASK,
}


# Sandbox 硬底线路径规则 -- agent 不能改任务描述/评分器/恢复状态.
# 参考 CodeWhale RepoLaw: Full Access 也不能越过, 只覆盖 file-write 不覆盖
# shell 重定向 (shell 重定向由 bash_tool command_pattern 规则覆盖).
# 规则只能收紧不能放宽, sandbox_mode=True 时强制注入, env var 只能追加不能移除默认项.
_DEFAULT_SANDBOX_PATH_RULES: list[tuple[str, PermissionMode]] = [
    ("INSTRUCTIONS.md", PermissionMode.DENY),
    ("score.py", PermissionMode.DENY),
    ("evaluation/*.py", PermissionMode.DENY),
    ("rubric.json", PermissionMode.DENY),
    (".huginn/checkpoints*", PermissionMode.DENY),
    (".huginn/engine_state*.json", PermissionMode.DENY),
]


@dataclass
class PermissionConfig:
    """User-configurable permission settings."""

    rules: dict[str, PermissionMode] = field(
        default_factory=lambda: DEFAULT_PERMISSION_RULES.copy()
    )
    auto_approve_all: bool = False  # For CI/automation mode
    # plan mode: 把所有写工具降级成 ASK, 只读工具保持 AUTO
    plan_mode: bool = False
    # path-level overrides: [(glob_pattern, mode), ...]
    # matched against file_path from tool args, first match wins
    path_rules: list[tuple[str, PermissionMode]] = field(default_factory=list)
    # Sandbox 模式: 强制注入 _DEFAULT_SANDBOX_PATH_RULES 硬底线 (跟 path_rules 取并集).
    # 非 sandbox 入口不设, path_rules 仍默认空, 行为不变.
    sandbox_mode: bool = False
    # ── 细粒度新增 (M5) ──
    # 成本预算 (CPU 小时): 超过该预算的工具自动升 ASK. None=不限制 (默认).
    cost_budget_hours: float | None = None
    # 信任自适应: 开启后按 trust score 浮动 medium 风险 (trust 高放行 / 低强制 ASK).
    # 默认 False 保持向后兼容.
    trust_adaptive: bool = False

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


# ── 细粒度推断与信任存储 (M5) ─────────────────────────────────────
# 与 code_act_loop.py 的 risk/trust 算法对齐, 但在此独立实现轻量版,
# 避免 tool_call 路径因导入 code_act_loop (依赖 ToolRegistry 全量注册) 引入重依赖.
# trust 是 session 维度, 进程内共享; 升级路径: 持久化到 config/store.

_RISK_BY_MODE = {
    PermissionMode.AUTO: RiskLevel.NONE,
    PermissionMode.ASK: RiskLevel.MEDIUM,
    PermissionMode.DENY: RiskLevel.CRITICAL,
}


def _infer_risk(mode: PermissionMode) -> RiskLevel:
    """从工具基础 mode 推断风险等级."""
    return _RISK_BY_MODE.get(mode, RiskLevel.MEDIUM)


_trust_scores: dict[str, float] = {}
_TRUST_DELTA = {"approve": 0.02, "deny": -0.10}


def _get_trust(session_id: str) -> float:
    """当前会话信任分, 默认 0.5 (中性)."""
    return _trust_scores.get(session_id, 0.5)


def _record_approval(session_id: str, action: str) -> float:
    """记录一次审批决策, 更新信任分, 返回新值."""
    delta = _TRUST_DELTA.get(action, 0.0)
    new_score = max(0.0, min(1.0, _get_trust(session_id) + delta))
    _trust_scores[session_id] = new_score
    return new_score


def reset_trust(session_id: str | None = None) -> None:
    """测试辅助: 清空信任分 (全部或单会话)."""
    if session_id is None:
        _trust_scores.clear()
    else:
        _trust_scores.pop(session_id, None)


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

    def _check_path_rules(self, args: dict | None, tool_name: str = "") -> PermissionMode | None:
        """Match file path from tool args against path_rules (first match wins).

        Looks for common path field names in args: file_path, path, working_dir.
        sandbox_mode=True 时强制注入 _DEFAULT_SANDBOX_PATH_RULES (硬底线, 只能收紧不能放宽),
        并读 env var HUGINN_SANDBOX_BLOCKED_PATHS 追加用户路径 (跟默认项取并集).

        path_rules 支持两种 tuple 形态:
          - (glob_pattern, mode)             : 旧格式, 对所有工具生效
          - (tool_name, glob_pattern, mode)  : 新格式, 工具×路径矩阵 (per-tool)
        """
        if not args:
            return None
        if self.config.sandbox_mode:
            # 默认硬底线在前, 用户 path_rules 在后 -- first match wins, 默认项优先.
            effective_rules = list(_DEFAULT_SANDBOX_PATH_RULES) + list(self.config.path_rules)
            # env var HUGINN_SANDBOX_BLOCKED_PATHS: 逗号分隔路径, 只能追加 DENY.
            for _p in os.environ.get("HUGINN_SANDBOX_BLOCKED_PATHS", "").split(","):
                _p = _p.strip()
                if _p:
                    effective_rules.append((_p, PermissionMode.DENY))
            # 自检: 默认项是硬底线, 不能被移除. 用户只能追加, 不能放宽.
            _default_paths = {p for p, _ in _DEFAULT_SANDBOX_PATH_RULES}
            _effective_paths = {
                r[1] if len(r) == 3 else r[0] for r in effective_rules
            }
            assert _default_paths.issubset(_effective_paths), \
                "Sandbox default path rules removed -- defaults are a hard floor"
        else:
            effective_rules = self.config.path_rules
        if not effective_rules:
            return None
        # extract path from common field names
        path_str = args.get("file_path") or args.get("path") or ""
        if not path_str:
            return None
        # match basename + full path so both "*.env" and "secrets/*" work
        basename = Path(str(path_str)).name
        for rule in effective_rules:
            if len(rule) == 3:
                # 工具×路径矩阵: 仅当工具名匹配 (或 tool_name 为空=通用) 才命中
                rule_tool, pattern, mode = rule
                if tool_name and rule_tool and tool_name != rule_tool:
                    continue
            else:
                pattern, mode = rule
            if fnmatch.fnmatch(path_str, pattern) or fnmatch.fnmatch(basename, pattern):
                return mode
        return None

    async def check(
        self,
        tool_name: str,
        is_read_only: bool = False,
        is_destructive: bool = False,
        cost_estimate: dict[str, float] | None = None,
        args: dict | None = None,
        session_id: str = "default",
    ) -> PermissionResult:
        """多阶段细粒度判定 (M5).

        判定维度依次叠加, 每命中一个维度记入 matched_rules 供可观测:
          1. 危险命令 (最高优先级, 即使 auto_approve_all 也拦)
          2. 路径规则 (工具×路径矩阵覆盖或指定工具)
          3. 工具基础规则 → 推算出风险等级 (risk_level)
          4. 成本分级: 超过 cost_budget_hours 的放行工具升 ASK
          5. 信任自适应 (trust_adaptive 时): 低信任强制 ASK / 高信任放行 medium
        """
        matched: list[str] = []

        # 阶段1: 危险命令 — 必须放在 get_mode() 之前, 否则 yolo 会直接放行.
        is_dangerous, matched_pat = self._check_dangerous(tool_name, args)
        if is_dangerous:
            matched.append(f"dangerous:{matched_pat}")
            reason = (
                f"Tool '{tool_name}' matches dangerous pattern '{matched_pat}' — "
                "requires explicit approval even in auto-approve mode"
            )
            return PermissionResult(
                mode=PermissionMode.ASK,
                reason=reason,
                risk_level=RiskLevel.CRITICAL,
                matched_rules=matched,
            )

        # 阶段2: 路径规则 (工具×路径矩阵 / 通用路径)
        path_mode = self._check_path_rules(args, tool_name)
        if path_mode is not None:
            if path_mode == PermissionMode.DENY:
                return PermissionResult(
                    mode=PermissionMode.DENY,
                    reason="Path blocked by path-level rule",
                    risk_level=RiskLevel.CRITICAL,
                    matched_rules=["path:deny"],
                )
            if path_mode == PermissionMode.ASK:
                return PermissionResult(
                    mode=PermissionMode.ASK,
                    reason="Path requires approval by path-level rule",
                    risk_level=RiskLevel.MEDIUM,
                    matched_rules=["path:ask"],
                )
            # AUTO — fall through (path 命中但放行)
            matched.append("path:auto")

        # 阶段3: 工具基础规则 → mode + 风险推断
        mode = self.config.get_mode(tool_name)
        risk = _infer_risk(mode)
        matched.append(f"tool:{tool_name}:{mode.value}")

        # 阶段4: 成本分级 — 放行工具超预算 → 升 ASK, 风险升 HIGH
        cost_hours = self._extract_cost(cost_estimate)
        budget = self.config.cost_budget_hours
        if cost_hours is not None and budget is not None and cost_hours > budget:
            matched.append(f"cost:{cost_hours:.1f}>{budget}")
            if mode == PermissionMode.AUTO:
                mode = PermissionMode.ASK
            if risk not in (RiskLevel.HIGH, RiskLevel.CRITICAL):
                risk = RiskLevel.HIGH

        # 阶段5: 信任自适应 — trust 高放行 medium 以下, 低强制 ASK (high/critical 不浮动)
        if self.config.trust_adaptive:
            trust = _get_trust(session_id)
            if mode == PermissionMode.AUTO and risk in (RiskLevel.MEDIUM, RiskLevel.HIGH) and trust < 0.3:
                matched.append(f"trust:{trust:.2f}<0.3")
                mode = PermissionMode.ASK
            elif mode == PermissionMode.ASK and risk in (RiskLevel.LOW, RiskLevel.MEDIUM) and trust > 0.7:
                matched.append(f"trust:{trust:.2f}>0.7")
                mode = PermissionMode.AUTO

        if mode == PermissionMode.DENY:
            return PermissionResult(
                mode=PermissionMode.DENY,
                reason=f"Tool '{tool_name}' is explicitly blocked by permission policy",
                risk_level=RiskLevel.CRITICAL,
                cost_hours=cost_hours,
                matched_rules=matched,
            )

        if mode == PermissionMode.AUTO:
            return PermissionResult(
                mode=PermissionMode.AUTO,
                risk_level=risk,
                cost_hours=cost_hours,
                matched_rules=matched,
            )

        # ASK mode — build a reason string
        reasons = []
        if is_destructive:
            reasons.append("this operation is destructive")
        if cost_hours is not None and cost_hours > 1:
            reasons.append(f"estimated cost: {cost_hours:.1f} CPU hours")

        reason = f"Tool '{tool_name}' requires approval"
        if reasons:
            reason += f" ({', '.join(reasons)})"

        return PermissionResult(
            mode=PermissionMode.ASK,
            reason=reason,
            risk_level=risk,
            cost_hours=cost_hours,
            matched_rules=matched,
        )

    @staticmethod
    def _extract_cost(cost_estimate: dict[str, float] | None) -> float | None:
        """从 cost_estimate 提取 cpu_hours, 无则 None."""
        if not cost_estimate:
            return None
        cpu = cost_estimate.get("cpu_hours", 0)
        return float(cpu) if cpu else None


# ── P4-2: Standing Rules — (tool, target) 维度常驻授权 ──────────
#
# OpenWorker 的 Standing Rules 按 tool→target 授权, e.g. "允许 file_write_tool
# 写 /tmp/*". approve_always 后记录, 下次同 tool+target 自动放行, 不再调
# approval_callback.
#
# CodeAct 路径 (code_act_loop.py) 已有简化版 (frozenset(called_tools) 组合维度),
# 这里给 tool_call 路径用 (tool, target) 精细维度. 两者独立, 不强行统一 —
# CodeAct 场景 args 从代码解析太重, 组合维度够用; tool_call 场景 args 明确,
# 可走 target 维度.
#
# target 提取: 从 args 的 file_path/path/working_dir 字段取, 无 target 时用 "*".
# target 做 fnmatch 匹配, e.g. grant("file_write_tool", "/tmp/*") 后,
# is_granted("file_write_tool", "/tmp/foo.txt") 返回 True.
#
# ponytail: 进程级单例, 不持久化 (重启清空). 升级路径: 持久化到 config.


class StandingRulesStore:
    """(tool, target) 维度常驻授权. 线程安全."""

    def __init__(self) -> None:
        import threading
        self._lock = threading.Lock()
        # session_id → {(tool_name, target_pattern)}
        self._rules: dict[str, set[tuple[str, str]]] = {}

    def grant(
        self, session_id: str, tool_name: str, target_pattern: str = "*"
    ) -> None:
        """记录一条 standing rule. target_pattern 支持 fnmatch."""
        with self._lock:
            self._rules.setdefault(session_id, set()).add((tool_name, target_pattern))

    def is_granted(
        self, session_id: str, tool_name: str, target: str = "*"
    ) -> bool:
        """检查是否命中 standing rule. target 做 fnmatch 匹配."""
        with self._lock:
            rules = self._rules.get(session_id, set())
        for r_tool, r_pattern in rules:
            if r_tool != tool_name:
                continue
            if r_pattern == "*" or fnmatch.fnmatch(target, r_pattern):
                return True
        return False

    def reset(self, session_id: str | None = None) -> None:
        """清空. 传 session_id 只清该 session."""
        with self._lock:
            if session_id is None:
                self._rules.clear()
            else:
                self._rules.pop(session_id, None)

    def list_rules(self, session_id: str | None = None) -> list[dict]:
        """列出 standing rules (可观测性用)."""
        with self._lock:
            if session_id is None:
                return [
                    {"session_id": sid, "tool": t, "target": p}
                    for sid, rules in self._rules.items()
                    for t, p in rules
                ]
            return [
                {"session_id": session_id, "tool": t, "target": p}
                for t, p in self._rules.get(session_id, set())
            ]


_singleton: StandingRulesStore | None = None
_singleton_lock = None


def _get_lock():
    global _singleton_lock
    if _singleton_lock is None:
        import threading
        _singleton_lock = threading.Lock()
    return _singleton_lock


def get_standing_rules_store() -> StandingRulesStore:
    """进程级单例."""
    global _singleton
    if _singleton is None:
        with _get_lock():
            if _singleton is None:
                _singleton = StandingRulesStore()
    return _singleton


def reset_standing_rules_store() -> None:
    """测试用: 清空单例."""
    global _singleton
    with _get_lock():
        _singleton = None


def extract_target_from_args(args: dict | None) -> str:
    """从 tool args 提取 target path. 无 target 返回 "*"."""
    if not args:
        return "*"
    for key in ("file_path", "path", "working_dir", "output_path", "target"):
        val = args.get(key)
        if val and isinstance(val, str):
            return val
    return "*"
