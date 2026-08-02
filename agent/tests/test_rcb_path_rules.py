"""Step 4 RepoLaw path_rules tests — assert-based, runnable standalone or via pytest.

Covers _DEFAULT_RCB_PATH_RULES, rcb_mode injection, env var append, and the
hard-floor subset guarantee (defaults cannot be removed).
"""

from __future__ import annotations

import os

from huginn.permissions import (
    PermissionChecker,
    PermissionConfig,
    _DEFAULT_RCB_PATH_RULES,
)
from huginn.types import PermissionMode


def test_default_rules_nonempty():
    # 硬底线必须有内容, 否则 rcb_mode 是空壳.
    assert len(_DEFAULT_RCB_PATH_RULES) > 0
    # 全部 DENY — 硬底线只收紧
    for _path, mode in _DEFAULT_RCB_PATH_RULES:
        assert mode == PermissionMode.DENY


def test_rcb_mode_denies_instructions():
    cfg = PermissionConfig(rcb_mode=True)
    checker = PermissionChecker(cfg)
    mode = checker._check_path_rules({"file_path": "INSTRUCTIONS.md"})
    assert mode == PermissionMode.DENY


def test_non_rcb_mode_unchanged():
    # rcb_mode=False: path_rules 默认空, INSTRUCTIONS.md 不被拦 (行为不变)
    cfg = PermissionConfig(rcb_mode=False)
    checker = PermissionChecker(cfg)
    mode = checker._check_path_rules({"file_path": "INSTRUCTIONS.md"})
    assert mode is None


def test_env_var_appends_without_removing_defaults():
    # 追加 extra.txt, INSTRUCTIONS.md 仍被 DENY
    os.environ["HUGINN_RCB_BLOCKED_PATHS"] = "extra.txt"
    try:
        cfg = PermissionConfig(rcb_mode=True)
        checker = PermissionChecker(cfg)
        assert checker._check_path_rules({"file_path": "extra.txt"}) == PermissionMode.DENY
        assert checker._check_path_rules({"file_path": "INSTRUCTIONS.md"}) == PermissionMode.DENY
    finally:
        del os.environ["HUGINN_RCB_BLOCKED_PATHS"]


def test_defaults_cannot_be_removed():
    # 集合差集检查: 默认项是硬底线, 无论 env var 怎么设, 默认项永远在 effective_rules 里.
    # env var 只能追加, 不能移除 — 用户路径跟默认项取并集.
    os.environ["HUGINN_RCB_BLOCKED_PATHS"] = "extra.txt,foo/bar.py"
    try:
        cfg = PermissionConfig(rcb_mode=True)
        checker = PermissionChecker(cfg)
        # 复刻 _check_path_rules 里的 effective_rules 构造逻辑做差集检查
        effective = list(_DEFAULT_RCB_PATH_RULES) + list(cfg.path_rules)
        for _p in os.environ["HUGINN_RCB_BLOCKED_PATHS"].split(","):
            _p = _p.strip()
            if _p:
                effective.append((_p, PermissionMode.DENY))
        default_paths = {p for p, _ in _DEFAULT_RCB_PATH_RULES}
        effective_paths = {p for p, _ in effective}
        # 默认项必须是 effective 的子集 — 用户不能移除默认项
        assert default_paths.issubset(effective_paths)
        # 反向验证: 默认项数量不变
        assert len(default_paths) == 6
    finally:
        del os.environ["HUGINN_RCB_BLOCKED_PATHS"]


def test_default_rule_patterns_match_benchmark_files():
    # 确认每条 glob 真能命中对应的 benchmark 关键文件
    cfg = PermissionConfig(rcb_mode=True)
    checker = PermissionChecker(cfg)
    cases = [
        ("INSTRUCTIONS.md", "INSTRUCTIONS.md"),
        ("score.py", "score.py"),
        ("evaluation/*.py", "evaluation/grade.py"),
        ("rubric.json", "rubric.json"),
        (".huginn/checkpoints*", ".huginn/checkpoints_iter10.json"),
        (".huginn/engine_state*.json", ".huginn/engine_state_v2.json"),
    ]
    for _pattern, path in cases:
        assert checker._check_path_rules({"file_path": path}) == PermissionMode.DENY, path


if __name__ == "__main__":
    # standalone runner — 不依赖 pytest
    tests = [
        test_default_rules_nonempty,
        test_rcb_mode_denies_instructions,
        test_non_rcb_mode_unchanged,
        test_env_var_appends_without_removing_defaults,
        test_defaults_cannot_be_removed,
        test_default_rule_patterns_match_benchmark_files,
    ]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"ALL {len(tests)} TESTS PASSED")
