"""Test the budget gui approval path wired to Inbox.

Covers:
- on_tokens_used_async: gui approve -> renewed, gui deny -> abort, off -> continue
- engine._build_budget_human_decide maps Inbox answers (approve/deny) to bool
- _maybe_run_budget_approval renews on approval and raises BudgetExhausted on deny

No real Inbox / LLM / network; _await_human_decision_via_inbox is stubbed.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from huginn.autoloop.budget import TokenBudget
from huginn.budget_pause import BudgetApprovalController


async def _approve(question: str, detail: str) -> bool:
    return True


async def _deny(question: str, detail: str) -> bool:
    return False


def test_on_tokens_used_async_approve_renews():
    b = TokenBudget()
    b.hard_limit_tokens = 1000
    b._recompute_soft()
    b.current_tokens = 900  # over soft=800
    ctl = BudgetApprovalController(b, mode="gui")
    d = asyncio.run(ctl.on_tokens_used_async(_approve))
    assert d == "renewed"
    assert b.renewals_left() == b.max_renewals - 1


def test_on_tokens_used_async_deny_aborts_without_spending():
    b = TokenBudget()
    b.hard_limit_tokens = 1000
    b._recompute_soft()
    b.current_tokens = 900
    d = asyncio.run(BudgetApprovalController(b, mode="gui").on_tokens_used_async(_deny))
    assert d == "abort"
    assert b.renewals_left() == b.max_renewals


def test_on_tokens_used_async_off_is_noop():
    b = TokenBudget()
    b.hard_limit_tokens = 1000
    b._recompute_soft()
    b.current_tokens = 900
    d = asyncio.run(BudgetApprovalController(b, mode="off").on_tokens_used_async(_approve))
    assert d == "continue"
    assert b.renewals_left() == b.max_renewals


class _StubEngine(SimpleNamespace):
    """Minimal stand-in for AutoloopEngine helper methods."""

    _iteration = 3
    _await_human_decision_via_inbox = AsyncMock()
    # 真实 engine 由 EngineControlMixin._build_budget_human_decide 挂载;
    # 这里为此方法提供同样签名的替身, 供 _maybe_run 调用.
    _build_budget_human_decide = None


def _attach_budget_helpers(engine: _StubEngine) -> None:
    from huginn.autoloop.engine import AutoloopEngine

    engine._build_budget_human_decide = AutoloopEngine._build_budget_human_decide.__get__(
        engine
    )


def _human_decide_result(engine: _StubEngine) -> Callable[[str, str], Awaitable[bool]]:
    from huginn.autoloop.engine import AutoloopEngine

    return AutoloopEngine._build_budget_human_decide(engine)


def test_build_human_decide_maps_approve_answer():
    engine = _StubEngine()
    engine._await_human_decision_via_inbox.return_value = "approve: 批准续投"
    decide = _human_decide_result(engine)
    assert asyncio.run(decide("q", "d")) is True


def test_build_human_decide_maps_deny_answer():
    engine = _StubEngine()
    engine._await_human_decision_via_inbox.return_value = "deny: 停止"
    decide = _human_decide_result(engine)
    assert asyncio.run(decide("q", "d")) is False


def test_build_human_decide_unavailable_falls_back_to_false():
    engine = _StubEngine()
    engine._await_human_decision_via_inbox.return_value = None
    decide = _human_decide_result(engine)
    assert asyncio.run(decide("q", "d")) is False


def test_maybe_run_budget_approval_renews_on_gui_approve():
    import huginn.autoloop.engine_control as ec

    engine = _StubEngine()
    _attach_budget_helpers(engine)
    engine._token_budget = TokenBudget()
    engine._token_budget.hard_limit_tokens = 1000
    engine._token_budget._recompute_soft()
    engine._token_budget.current_tokens = 900
    engine._await_human_decision_via_inbox = AsyncMock(return_value="approve: 批准续投")
    engine._maybe_save_engine_state = lambda **kw: None

    with patch.dict("os.environ", {"HUGINN_BUDGET_APPROVAL": "gui"}):
        asyncio.run(ec.EngineControlMixin._maybe_run_budget_approval(engine))
    assert engine._token_budget.renewals_left() == engine._token_budget.max_renewals - 1


def test_maybe_run_budget_approval_aborts_on_gui_deny():
    import pytest

    import huginn.autoloop.engine_control as ec
    from huginn.autoloop.budget import BudgetExhausted

    engine = _StubEngine()
    _attach_budget_helpers(engine)
    engine._token_budget = TokenBudget()
    engine._token_budget.hard_limit_tokens = 1000
    engine._token_budget._recompute_soft()
    engine._token_budget.current_tokens = 900
    engine._await_human_decision_via_inbox = AsyncMock(return_value="deny: 停止")
    engine._maybe_save_engine_state = lambda **kw: None

    with patch.dict("os.environ", {"HUGINN_BUDGET_APPROVAL": "gui"}), pytest.raises(
        BudgetExhausted
    ):
        asyncio.run(ec.EngineControlMixin._maybe_run_budget_approval(engine))
