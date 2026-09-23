"""Anti-Hacking ① strict-scope 接线验收.

把"越界改动 → 整轨奖励清零"从死代码接成真闭环, 锁住三条属性:
  1. 授权面 = 既有权限面 (沙箱硬底线 + path_rules), 不新造一套;
  2. 归一化兼容两种真实来源 (perception 裸路径 / git status --short 带状态码);
  3. 零回归: flag off / 授权面不可用 / 无改动文件 → r_phys 原样不动。

诚实边界: 口径是"合规口径"(改动是否命中 DENY 规则), 不是"意图口径"
(是否落在本轮 plan 声明的目标内) —— 后者需要 plan 增加文件级字段, 未实现。
"""
from __future__ import annotations

import pytest

from huginn.validation.scope_authority import compute_authorized_ratio

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


# ──────────────────── 折叠点 (_learn 的 r_phys) ────────────────────


@pytest.fixture()
def flag():
    """开关 anti_hacking_reward, 用例结束复位."""
    from huginn.feature_flags import FeatureFlags

    ff = FeatureFlags.shared()
    yield ff
    ff.reset("anti_hacking_reward")


def _engine_with(files: list[str]):
    """最小 stub: _apply_strict_scope 挂在 EngineReflect 上, 改动面经 __getattr__
    从 engine 转发 (与运行时 self._engine_reflector 的取值路径一致)."""
    from huginn.autoloop.engine_reflect import EngineReflect

    class _StubEngine:
        pass

    eng = _StubEngine()
    eng._last_execution_files = files
    return EngineReflect(eng)


def test_fold_flag_off_is_zero_regression(flag) -> None:
    flag.disable("anti_hacking_reward")
    eng = _engine_with(["score.py"])
    assert eng._apply_strict_scope(0.9) == 0.9


def test_fold_flag_on_violation_zeroes(flag) -> None:
    flag.enable("anti_hacking_reward")
    eng = _engine_with(["src/a.py", "score.py"])
    assert eng._apply_strict_scope(0.9) == 0.0


def test_fold_flag_on_clean_keeps_r_phys(flag) -> None:
    flag.enable("anti_hacking_reward")
    eng = _engine_with(["src/a.py"])
    assert eng._apply_strict_scope(0.9) == 0.9


def test_fold_flag_on_no_changed_files_keeps_r_phys(flag) -> None:
    flag.enable("anti_hacking_reward")
    eng = _engine_with([])
    assert eng._apply_strict_scope(0.9) == 0.9


def test_fold_none_r_phys_passthrough(flag) -> None:
    flag.enable("anti_hacking_reward")
    eng = _engine_with(["score.py"])
    assert eng._apply_strict_scope(None) is None
