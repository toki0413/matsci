"""Tests for compaction policy plugin (Everything is a Plugin, 形态 B)."""

from __future__ import annotations

import pytest

from huginn.plugins.compaction_policy import (
    CompactionPolicy,
    clear_policies,
    never_trim_block_types,
    protected_roles,
    register_compaction_policy,
    root_content_markers,
    unregister_compaction_policy,
)
from huginn.utils.context import _msg_content, compact_messages


@pytest.fixture(autouse=True)
def _reset_policies():
    yield
    clear_policies()


class TestAggregation:
    def test_defaults_match_builtin_constants(self):
        assert protected_roles() == frozenset({"system"})
        assert never_trim_block_types() == frozenset(
            {"thinking", "redacted_thinking"}
        )
        assert root_content_markers() == ()

    def test_register_extends_union(self):
        register_compaction_policy(
            "ext",
            CompactionPolicy(
                protected_roles=frozenset({"memory"}),
                never_trim_block_types=frozenset({"cache"}),
                root_content_markers=("## EXTRA_ROOT",),
            ),
        )
        assert "memory" in protected_roles()
        assert "system" in protected_roles()  # 默认仍在
        assert "cache" in never_trim_block_types()
        assert "thinking" in never_trim_block_types()  # 默认仍在
        assert "## EXTRA_ROOT" in root_content_markers()

    def test_unregister_removes_contribution(self):
        register_compaction_policy(
            "t", CompactionPolicy(protected_roles=frozenset({"memory"}))
        )
        assert "memory" in protected_roles()
        unregister_compaction_policy("t")
        assert "memory" not in protected_roles()


class TestMarkersIntegration:
    def test_policy_marker_protects_message_from_drop(self):
        register_compaction_policy(
            "extra", CompactionPolicy(root_content_markers=("## EXTRA_ROOT",))
        )
        msgs = [
            {"role": "user", "content": "## EXTRA_ROOT\ntask instruction"},
            {"role": "assistant", "content": "x" * 5000},
        ]
        out = compact_messages(msgs, budget_tokens=10, keep_last_n=0)
        assert any("EXTRA_ROOT" in _msg_content(m) for m in out)

    def test_without_policy_marker_message_is_dropped(self):
        msgs = [
            {"role": "user", "content": "## EXTRA_ROOT\ntask instruction"},
            {"role": "assistant", "content": "x" * 5000},
        ]
        out = compact_messages(msgs, budget_tokens=10, keep_last_n=0)
        assert not any("EXTRA_ROOT" in _msg_content(m) for m in out)
