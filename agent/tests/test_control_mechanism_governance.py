"""控制面治理测试: 机制登记 + 只减不增的预算棘轮 + 硬终止白名单.

见 :mod:`huginn.autoloop.control_mechanisms` 的模块 docstring. 本测试把
``control_surface_audit.md`` §3 的"控制面预算"从散文变成可强制的四条不变量:

1. 代码里发射的每个 trace name 必须已登记 (拦截静默新增机制);
2. 每条登记必须仍有发射点 (拦截死机制);
3. 机制总数 ``<= CONTROL_MECHANISM_BUDGET`` (只减不增);
4. ``effect == "stop"`` 只允许出现在 ``STOP_ALLOWLIST`` (科学出口 + 资源熔断).

新增机制而没登记 → 测试红; 要收敛控制面 → 删机制并下调 ``CONTROL_MECHANISM_BUDGET``.
"""

from __future__ import annotations

from huginn.autoloop import control_mechanisms as cm


def test_every_emitted_mechanism_is_declared():
    """不变量 1: 代码里发射的每个 trace name 必须已登记."""
    emitted = set(cm.scan_emitted_names())
    missing = sorted(emitted - set(cm.MECHANISMS))
    assert not missing, (
        "以下控制面 trace name 在代码里被发射但未在 "
        "huginn/autoloop/control_mechanisms.py 的 MECHANISMS 登记. 新增机制须先"
        "过 control_surface_audit.md §3 两问, 再登记: " + ", ".join(missing)
    )


def test_no_dead_mechanism_declarations():
    """不变量 2: 每条登记必须仍有发射点 (死机制必须回收)."""
    emitted = set(cm.scan_emitted_names())
    dead = sorted(set(cm.MECHANISMS) - emitted)
    assert not dead, (
        "以下机制已登记但代码里无发射点 (死机制). 删掉登记与残留代码, 并下调 "
        "CONTROL_MECHANISM_BUDGET: " + ", ".join(dead)
    )


def test_control_mechanism_budget_ratchet():
    """不变量 3: 机制总数只减不增 (棘轮)."""
    total = len(cm.MECHANISMS)
    assert total <= cm.CONTROL_MECHANISM_BUDGET, (
        f"控制面机制数 {total} 超过预算 {cm.CONTROL_MECHANISM_BUDGET}. "
        "新增机制前先删旧机制; 若确要上调预算, 必须写明理由并同步更新 "
        "control_surface_audit.md."
    )


def test_registry_entries_wellformed():
    """登记表自检: 类别/效果合法, 名字与键一致 (无坏声明)."""
    for name, m in cm.MECHANISMS.items():
        assert isinstance(name, str) and name, "机制名必须是非空字符串"
        assert m.category in cm.CATEGORIES, f"{name}: 未知类别 {m.category!r}"
        assert m.effect in cm.EFFECTS, f"{name}: 未知效果 {m.effect!r}"
        assert m.note.strip(), f"{name}: 说明不能为空"


def test_only_allowlisted_mechanisms_can_stop():
    """不变量 4: 能结束 run 的机制只允许 STOP_ALLOWLIST (科学出口 + 资源熔断).

    这把 §3 的"科学硬终止只留挂钟与目标达成, 外加一类资源熔断"从散文变成约束:
    任何其他机制若被标成 ``effect="stop"`` 即测试红, 迫使作者回答 §3 两问.
    """
    stoppers = sorted(n for n, m in cm.MECHANISMS.items() if m.effect == "stop")
    assert set(stoppers) <= cm.STOP_ALLOWLIST, (
        "以下机制声明 effect='stop' 但不在 STOP_ALLOWLIST (控制面预算 §3 规定"
        "科学硬终止只留挂钟/目标达成, 外加 resource fuse failure_budget): "
        + ", ".join(sorted(set(stoppers) - cm.STOP_ALLOWLIST))
    )