"""Anti-Hacking ① 的输入侧: 从"本轮实际改动文件面"算 authorized_ratio.

背景: `claim_reward.strict_scope_reward` 需要一个 `authorized_ratio`, 但此前
全仓没有任何地方计算它 —— 三件套 (`anti_hacking_reward` / `strict_scope_reward`
/ `efficiency_discount` / `idle_turn_penalty`) 是**定义了但零调用者**的死代码。
本模块补上缺失的那个输入, 让 strict-scope 真正能咬住"越界改动".

     authorized_ratio = 落在授权范围内的改动文件数 / 实际改动文件数

授权范围**复用既有的权限面, 不新造一套** (single source of truth 在
`huginn/permissions.py`):
  - `_DEFAULT_SANDBOX_PATH_RULES`  沙箱硬底线 (score.py / evaluation/*.py /
    rubric.json / .huginn/checkpoints* ... 一律 DENY, 只能收紧不能放宽)
  - `PermissionConfig.path_rules`   用户/task 追加的 (glob, mode) 或
    (tool, glob, mode)

口径 S1 (合规口径, 本模块): 改动路径命中 DENY 规则即判越界。这直接对应 DSec
论文 §6.5 里 AppArmor 挡的那类"去翻评分产物/日志找残留答案"—— 改 `score.py`
正是 reward hacking 最典型的签名。

口径 S2 (意图口径, 未实现): 授权集合 = 本轮 plan 声明的目标 globs。需要 plan
增加文件级字段; 目前 `PlanStep` 只有 description/tool/parameters (见
`huginn/autoloop/plan_store.py`), 无从表达, 故留作升级路径。

零回归: 授权面为空 (未开 sandbox_mode 且无 path_rules) → `source="unavailable"`,
ratio=1.0, 调用方据此 no-op; 无改动文件 → ratio=1.0。纯标准库/幂等/可单测。
"""
from __future__ import annotations

import fnmatch
import logging
import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 授权面不可用时的哨兵: 调用方看到它必须原样放行 r_phys, 不得改动.
_SOURCE_UNAVAILABLE = "unavailable"
_SOURCE_PERMISSIONS = "permissions.path_rules"

# git porcelain 行首状态码: " M f" / "?? f" / "MM f" / "R  a -> b".
# 只剥前导状态标记, 不碰裸路径 (裸路径首字符不在类里, 不会误伤).
_GIT_STATUS_RE = re.compile(r"^[\s?MADRCU!]{1,3}\s+")


def _normalize(entry: Any) -> str:
    """把一条改动记录归一成相对路径.

    兼容两种来源 (都是仓库里真实存在的):
      - perception file_events: 裸路径 "src/a.py"
      - legacy `git status --short`: 带状态码 " M src/a.py" / "?? new.py" /
        rename "R  old.py -> new.py" (取目标路径)
    """
    s = str(entry or "").strip()
    if " -> " in s:
        s = s.split(" -> ")[-1].strip()
    s = _GIT_STATUS_RE.sub("", s).strip()
    return s.replace("\\", "/")


def _effective_rules(
    path_rules: Sequence[Any] | None,
    *,
    sandbox_mode: bool,
) -> list[tuple[Any, ...]]:
    """取生效的规则集. 惰性 import permissions 以保证单一真源.

    取不到 (重依赖未装等) 返回空列表 → 调用方判 unavailable → no-op.
    """
    rules: list[tuple[Any, ...]] = []
    try:
        from huginn.permissions import _DEFAULT_SANDBOX_PATH_RULES

        if sandbox_mode:
            rules.extend(_DEFAULT_SANDBOX_PATH_RULES)
    except Exception:  # noqa: BLE001 — 授权面取不到就不判, 绝不误杀
        logger.debug("scope_authority: permissions import failed", exc_info=True)
        if not path_rules:
            return []
    if path_rules:
        rules.extend(path_rules)
    if sandbox_mode:
        for _p in os.environ.get("HUGINN_SANDBOX_BLOCKED_PATHS", "").split(","):
            _p = _p.strip()
            if _p:
                rules.append((_p, "deny"))
    return rules


def _mode_of(path: str, rules: Sequence[tuple[Any, ...]]) -> str | None:
    """first match wins, 与 permissions._check_path_rules 同语义.

    支持 (glob, mode) 与 (tool, glob, mode) 两种形态。文件面拿不到工具名,
    故 per-tool 规则 (rule_tool 非空) 无法归因 —— 保守跳过, 不据此判越界。
    全路径与 basename 都匹配, 让 "*.env" 和 "secrets/*" 都能生效。
    """
    basename = Path(path).name
    for rule in rules:
        if len(rule) == 3:
            rule_tool, pattern, mode = rule
            if rule_tool:
                continue  # 无工具名可归因, 跳过
        else:
            pattern, mode = rule
        if fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(basename, pattern):
            return str(getattr(mode, "value", mode)).lower()
    return None


def compute_authorized_ratio(
    changed_files: Sequence[Any],
    *,
    path_rules: Sequence[Any] | None = None,
    sandbox_mode: bool = False,
) -> dict[str, Any]:
    """算 strict-scope 需要的 authorized_ratio.

    Returns: {
        "authorized_ratio": float,   # 授权面不可用时恒为 1.0 (no-op)
        "violations": list[str],     # 命中 DENY 的改动路径
        "total": int,                # 归一后去重的改动文件数
        "in_scope": int,
        "source": str,               # "unavailable" 时必须原样放行
        "rules_n": int,
    }
    """
    paths = [p for p in (_normalize(e) for e in (changed_files or [])) if p]
    seen: dict[str, None] = {}
    for p in paths:
        seen.setdefault(p, None)
    uniq = list(seen)

    rules = _effective_rules(path_rules, sandbox_mode=sandbox_mode)
    if not rules:
        return {
            "authorized_ratio": 1.0,
            "violations": [],
            "total": len(uniq),
            "in_scope": len(uniq),
            "source": _SOURCE_UNAVAILABLE,
            "rules_n": 0,
        }
    if not uniq:
        return {
            "authorized_ratio": 1.0,
            "violations": [],
            "total": 0,
            "in_scope": 0,
            "source": _SOURCE_PERMISSIONS,
            "rules_n": len(rules),
        }

    violations = [p for p in uniq if _mode_of(p, rules) == "deny"]
    in_scope = len(uniq) - len(violations)
    return {
        "authorized_ratio": in_scope / len(uniq),
        "violations": violations,
        "total": len(uniq),
        "in_scope": in_scope,
        "source": _SOURCE_PERMISSIONS,
        "rules_n": len(rules),
    }


__all__ = ["compute_authorized_ratio"]
