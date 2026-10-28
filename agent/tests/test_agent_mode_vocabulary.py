"""A3 门禁: agent 顶层 mode 词表收敛到单一来源 (huginn/agent_modes.py).

此前六处各写一份 mode 字面量, 增删 mode 时必漏改. 本测试锁住"派生"关系:
各处词表必须是 agent_modes 的子集/等集, 且运行期行为 (set_mode 校验 /
session 快照恢复) 只认同一值域.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from huginn.agent import HuginnAgent
from huginn.agent.prompt_builder import _MODE_INSTRUCTIONS
from huginn.agent.session import SessionMixin
from huginn.agent_modes import (
    ALL_MODES,
    CRITIQUE_MODES,
    LONG_HORIZON_MODES,
    PROMPT_MODES,
    SETTABLE_MODES,
)
from huginn.metacog.critique import _VALID_MODES


def test_all_modes_is_the_superset():
    """ALL_MODES 本身无重复, 且各子集都落在全集内."""
    assert len(ALL_MODES) == len(set(ALL_MODES))
    for subset in (SETTABLE_MODES, LONG_HORIZON_MODES, PROMPT_MODES):
        assert set(subset) <= set(ALL_MODES)
    assert set(ALL_MODES) >= CRITIQUE_MODES


def test_prompt_segments_match_prompt_modes():
    """prompt_builder 的 mode 段键集 == PROMPT_MODES."""
    assert set(_MODE_INSTRUCTIONS) == set(PROMPT_MODES)


def test_derived_vocabularies_match_source():
    """core / critique 的词表必须是 agent_modes 的同一对象等集."""
    assert HuginnAgent._LONG_HORIZON_MODES == LONG_HORIZON_MODES
    assert _VALID_MODES == CRITIQUE_MODES


@pytest.mark.parametrize("mode", ["code", "fusion"])
def test_set_mode_rejects_non_settable_modes(mode):
    """有 prompt 段但不可通过 set_mode 切换的 mode 必须抛 ValueError.

    校验发生在任何 self 属性访问之前, 故 object() 足以验证值域.
    """
    with pytest.raises(ValueError):
        HuginnAgent.set_mode(object(), mode)


def test_set_mode_accepts_all_settable_modes(monkeypatch):
    """SETTABLE_MODES 内每个值都能通过校验 (用 stub 承接实际切换)."""
    # 长程模式切换会读写该 env; 交给 monkeypatch 负责测试结束还原.
    monkeypatch.setenv("HUGINN_EXTREME_DISPATCH", "")
    for mode in SETTABLE_MODES:
        stub = SimpleNamespace(
            _mode="chat",
            _agent_graph=object(),
            _phase_manager=MagicMock(),
            _permission_config=SimpleNamespace(plan_mode=False),
            _LONG_HORIZON_MODES=HuginnAgent._LONG_HORIZON_MODES,
        )
        HuginnAgent.set_mode(stub, mode)
        assert stub._mode == mode


def test_session_restore_uses_settable_modes():
    """快照恢复白名单 == SETTABLE_MODES: 非可设置 mode 不得覆盖当前状态."""
    for mode in ALL_MODES:
        stub = SimpleNamespace(_mode="chat")
        SessionMixin._restore_snapshot_mode(stub, {"mode": mode})
        if mode in SETTABLE_MODES:
            assert stub._mode == mode
        else:
            assert stub._mode == "chat"
