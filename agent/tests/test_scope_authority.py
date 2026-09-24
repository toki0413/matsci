"""Anti-Hacking strict-scope 接线验收.

把"越界改动 → 整轨奖励清零"从死代码接成真闭环, 锁住三条属性:
  1. 授权面 = 既有权限面 (沙箱硬底线 + path_rules), 不新造一套;
  2. 归一化兼容两种真实来源 (perception 裸路径 / git status --short 带状态码);
  3. 零回归: flag off / 授权面不可用 / 无改动文件 → r_phys 原样不动。

两口径: ① 合规口径 (改动命中 DENY 规则, `compute_authorized_ratio`);
② 意图口径 (改动偏离 plan 声明集, `compute_intent_ratio`)。各自独立开关。
"""
from __future__ import annotations

import pytest

from huginn.validation.scope_authority import (
    compute_authorized_ratio,
    compute_intent_ratio,
)

# ──────────────────── compute_authorized_ratio ────────────────────


def test_no_rules_is_unavailable_and_noop() -> None:
    """未给授权面 → source=unavailable + ratio 1.0, 调用方据此放行."""
    r = compute_authorized_ratio(["score.py"])
    assert r["source"] == "unavailable", r
    assert r["authorized_ratio"] == 1.0, r
    assert r["violations"] == [], r


def test_no_changed_files_is_noop() -> None:
    r = compute_authorized_ratio([], sandbox_mode=True)
    assert r["authorized_ratio"] == 1.0, r
    assert r["total"] == 0, r
    assert r["violations"] == [], r


def test_in_scope_ratio_is_one() -> None:
    r = compute_authorized_ratio(["src/a.py", "docs/b.md"], sandbox_mode=True)
    assert r["source"] == "permissions.path_rules", r
    assert r["authorized_ratio"] == 1.0, r
    assert r["violations"] == [], r


def test_hard_floor_violation_zeroes_ratio() -> None:
    """改评分产物 (score.py) 是 reward hacking 的典型签名 → 判越界."""
    r = compute_authorized_ratio(["score.py"], sandbox_mode=True)
    assert r["authorized_ratio"] == 0.0, r
    assert r["violations"] == ["score.py"], r


def test_partial_ratio() -> None:
    r = compute_authorized_ratio(["src/a.py", "evaluation/run.py"], sandbox_mode=True)
    assert r["authorized_ratio"] == 0.5, r
    assert r["violations"] == ["evaluation/run.py"], r


def test_git_porcelain_lines_are_normalized() -> None:
    """legacy 来源是 `git status --short` 的整行, 必须剥掉状态码再判."""
    assert compute_authorized_ratio([" M score.py"], sandbox_mode=True)["violations"] == [
        "score.py"
    ]
    assert compute_authorized_ratio(["?? score.py"], sandbox_mode=True)["violations"] == [
        "score.py"
    ]
    # rename 取目标路径
    r = compute_authorized_ratio(["R  old.py -> score.py"], sandbox_mode=True)
    assert r["violations"] == ["score.py"], r
    # 干净改动不被状态码误伤
    assert compute_authorized_ratio([" M src/a.py"], sandbox_mode=True)[
        "authorized_ratio"
    ] == 1.0


def test_duplicates_counted_once() -> None:
    r = compute_authorized_ratio(["score.py", "score.py"], sandbox_mode=True)
    assert r["total"] == 1, r
    assert r["authorized_ratio"] == 0.0, r


def test_extra_path_rules_deny() -> None:
    """用户/task 追加的 (glob, mode) 也生效 (授权面取并集)."""
    r = compute_authorized_ratio(
        ["secrets/token.txt"], path_rules=[("secrets/*", "deny")]
    )
    assert r["authorized_ratio"] == 0.0, r
    assert r["violations"] == ["secrets/token.txt"], r


# ──────────────────── compute_intent_ratio (意图口径 S2) ────────────────────


def test_intent_no_globs_is_unavailable_and_noop() -> None:
    """plan 未声明目标 → source=unavailable + ratio 1.0, 调用方据此放行."""
    r = compute_intent_ratio(["src/a.py"], intent_globs=[])
    assert r["source"] == "unavailable", r
    assert r["authorized_ratio"] == 1.0, r
    assert r["violations"] == [], r


def test_intent_exact_path_in_scope() -> None:
    r = compute_intent_ratio(["src/a.py"], intent_globs=["src/a.py"])
    assert r["source"] == "intent.target_files", r
    assert r["authorized_ratio"] == 1.0, r
    assert r["violations"] == [], r


def test_intent_off_scope_change_zeroes() -> None:
    """plan 说改 A, 实际改了 B → 偏离, 判越界."""
    r = compute_intent_ratio(["src/b.py"], intent_globs=["src/a.py"])
    assert r["authorized_ratio"] == 0.0, r
    assert r["violations"] == ["src/b.py"], r


def test_intent_glob_and_dir_prefix() -> None:
    # 通配 glob
    r = compute_intent_ratio(["src/x/a.py", "src/y/b.py"], intent_globs=["src/**/*.py"])
    assert r["authorized_ratio"] == 1.0, r
    # 目录声明 (含子目录)
    r = compute_intent_ratio(["src/x/a.py"], intent_globs=["src/"])
    assert r["authorized_ratio"] == 1.0, r
    r = compute_intent_ratio(["src/x/a.py"], intent_globs=["src"])
    assert r["authorized_ratio"] == 1.0, r
    # 目录前缀不误配同前缀兄弟目录
    r = compute_intent_ratio(["srcfoo/a.py"], intent_globs=["src"])
    assert r["authorized_ratio"] == 0.0, r


def test_intent_bare_pattern_matches_basename() -> None:
    """裸模式 (无 "/") 允许 basename 匹配, 让 "*.py" 生效."""
    r = compute_intent_ratio(["deep/dir/a.py"], intent_globs=["*.py"])
    assert r["authorized_ratio"] == 1.0, r
    r = compute_intent_ratio(["deep/dir/a.txt"], intent_globs=["*.py"])
    assert r["authorized_ratio"] == 0.0, r


def test_intent_exact_path_not_relaxed_to_basename() -> None:
    """精确路径 "src/a.py" 不得放宽成"任意目录下的 a.py"."""
    r = compute_intent_ratio(["other/a.py"], intent_globs=["src/a.py"])
    assert r["authorized_ratio"] == 0.0, r


def test_intent_partial_ratio_and_normalization() -> None:
    """混合: 一半在声明集内; git 状态码先归一."""
    r = compute_intent_ratio(
        [" M src/a.py", "?? src/b.py"], intent_globs=["src/a.py"]
    )
    assert r["authorized_ratio"] == 0.5, r
    assert r["violations"] == ["src/b.py"], r


def test_intent_no_changed_files_is_noop() -> None:
    r = compute_intent_ratio([], intent_globs=["src/a.py"])
    assert r["authorized_ratio"] == 1.0, r
    assert r["total"] == 0, r


# ──────────────────── 折叠点 (_learn 的 r_phys) ────────────────────


@pytest.fixture()
def flag():
    """开关两个 anti-hacking flag, 用例结束复位."""
    from huginn.feature_flags import FeatureFlags

    ff = FeatureFlags.shared()
    yield ff
    ff.reset("anti_hacking_reward")
    ff.reset("intent_scope_reward")


def _engine_with(files: list[str], target_files: list[str] | None = None):
    """最小 stub: _apply_strict_scope 挂在 EngineReflect 上, 改动面/意图集经
    __getattr__ 从 engine 转发 (与运行时 self._engine_reflector 的取值路径一致)."""
    from huginn.autoloop.engine_reflect import EngineReflect

    class _StubEngine:
        pass

    eng = _StubEngine()
    eng._last_execution_files = files
    if target_files is not None:
        eng._current_plan_target_files = target_files
    return EngineReflect(eng)


def test_fold_flag_off_is_zero_regression(flag) -> None:
    flag.disable("anti_hacking_reward")
    flag.disable("intent_scope_reward")
    eng = _engine_with(["score.py"])
    assert eng._apply_strict_scope(0.9) == 0.9


def test_fold_flag_on_violation_zeroes(flag) -> None:
    flag.enable("anti_hacking_reward")
    flag.disable("intent_scope_reward")
    eng = _engine_with(["src/a.py", "score.py"])
    assert eng._apply_strict_scope(0.9) == 0.0


def test_fold_flag_on_clean_keeps_r_phys(flag) -> None:
    flag.enable("anti_hacking_reward")
    flag.disable("intent_scope_reward")
    eng = _engine_with(["src/a.py"])
    assert eng._apply_strict_scope(0.9) == 0.9


def test_fold_flag_on_no_changed_files_keeps_r_phys(flag) -> None:
    flag.enable("anti_hacking_reward")
    flag.disable("intent_scope_reward")
    eng = _engine_with([])
    assert eng._apply_strict_scope(0.9) == 0.9


def test_fold_none_r_phys_passthrough(flag) -> None:
    flag.enable("anti_hacking_reward")
    flag.disable("intent_scope_reward")
    eng = _engine_with(["score.py"])
    assert eng._apply_strict_scope(None) is None


# ── 意图口径 (S2) 折叠 ──


def test_fold_intent_off_is_zero_regression(flag) -> None:
    """意图 flag 关时, 即便偏离声明集也不折 (与 S1 独立)."""
    flag.disable("anti_hacking_reward")
    flag.disable("intent_scope_reward")
    eng = _engine_with(["src/b.py"], target_files=["src/a.py"])
    assert eng._apply_strict_scope(0.9) == 0.9


def test_fold_intent_on_deviation_zeroes(flag) -> None:
    flag.disable("anti_hacking_reward")
    flag.enable("intent_scope_reward")
    eng = _engine_with(["src/b.py"], target_files=["src/a.py"])
    assert eng._apply_strict_scope(0.9) == 0.0


def test_fold_intent_on_in_scope_keeps_r_phys(flag) -> None:
    flag.disable("anti_hacking_reward")
    flag.enable("intent_scope_reward")
    eng = _engine_with(["src/a.py"], target_files=["src/a.py"])
    assert eng._apply_strict_scope(0.9) == 0.9


def test_fold_intent_on_no_declared_globs_is_noop(flag) -> None:
    """plan 未声明 FILES: → 意图口径 no-op (即便 flag 开)."""
    flag.disable("anti_hacking_reward")
    flag.enable("intent_scope_reward")
    eng = _engine_with(["src/b.py"], target_files=[])
    assert eng._apply_strict_scope(0.9) == 0.9


def test_fold_intent_only_does_not_need_compliance_flag(flag) -> None:
    """两个口径完全解耦: 只开 S2 也能独立咬住偏离."""
    flag.disable("anti_hacking_reward")
    flag.enable("intent_scope_reward")
    eng = _engine_with(["evaluation/run.py", "src/a.py"], target_files=["src/"])
    assert eng._apply_strict_scope(0.9) == 0.0
