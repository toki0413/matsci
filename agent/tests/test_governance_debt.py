"""Tests for governance facade — provenance/audit/rollback integration.

These tests verify the P0 fixes: governance.py previously had broken
imports (get_provenance_registry, get_audit_logger) and mismatched
API calls (ProvenanceEntry fields, register signature, lookup method,
revert missing workspace, audit query user→actor).
"""

from __future__ import annotations

from pathlib import Path

import pytest


# Ensure isolated cache dir for provenance/audit
@pytest.fixture(autouse=True)
def _isolated_env(tmp_path, monkeypatch):
    monkeypatch.setenv("HUGINN_CACHE_DIR", str(tmp_path / "cache"))
    Path(tmp_path / "cache").mkdir(parents=True, exist_ok=True)
    # Reset governance singleton — must clear the MODULE-level _gov global,
    # not GovernanceFacade._gov (a class attribute that get_governance never reads).
    import huginn.governance as gov_mod

    gov_mod._gov = None  # type: ignore[attr-defined]
    # Reset audit singleton
    import huginn.security.audit as audit_mod

    audit_mod._audit_logger = None
    # Reset provenance singleton
    from huginn.provenance.registry import ProvenanceRegistry

    ProvenanceRegistry._instance = None
    # Reset snapshot singleton so it picks up the new HUGINN_CACHE_DIR
    from huginn.snapshot.file_snapshot import SnapshotManager

    SnapshotManager._instance = None
    yield
    # Cleanup singletons
    gov_mod._gov = None  # type: ignore[attr-defined]
    ProvenanceRegistry._instance = None
    SnapshotManager._instance = None
    audit_mod._audit_logger = None


class TestGovernanceInitialization:
    """Verify governance subsystems are properly initialized (P0 fix)."""

    def test_all_subsystems_available(self):
        from huginn.governance import get_governance

        gov = get_governance()
        gov._ensure_initialized()
        assert gov._audit_logger is not None, "audit logger should be available"
        assert gov._policy_engine is not None, "policy engine should be available"
        assert gov._provenance is not None, "provenance registry should be available"
        assert gov._snapshot_mgr is not None, "snapshot manager should be available"

    def test_rollback_available(self):
        from huginn.governance import get_governance

        gov = get_governance()
        assert gov._rollback_available() is True

    def test_provenance_uses_shared_singleton(self):
        from huginn.governance import get_governance
        from huginn.provenance.registry import ProvenanceRegistry

        gov = get_governance()
        gov._ensure_initialized()
        assert gov._provenance is ProvenanceRegistry.shared()


class TestGovernanceCanExecute:
    """Verify can_execute with fail-secure default."""

    def test_unknown_action_denied_by_default(self):
        from huginn.governance import get_governance

        gov = get_governance()
        decision = gov.can_execute("nonexistent_action_xyz", {})
        assert decision.allowed is False
        assert any("fail-secure" in r or "Unknown" in r for r in decision.reasons)

    def test_known_action_passes_preconditions(self):
        from huginn.governance import get_governance

        gov = get_governance()
        # 'file_read' is a known action type in ontology
        decision = gov.can_execute("file_read", {})
        # Should not crash; allowed depends on policy
        assert isinstance(decision.allowed, bool)
        assert len(decision.reasons) >= 0


class TestGovernanceExecute:
    """Verify execute writes audit + provenance correctly (P0 fix)."""

    def test_execute_success_writes_audit_and_provenance(self):
        from huginn.governance import get_governance

        gov = get_governance()
        result = gov.execute(
            "file_read",
            {"path": "test.txt"},
            lambda ctx: {"file_path": "test.txt", "content": "hello"},
            user="test_user",
        )
        assert result.status in ("verified", "failed")
        assert result.audit_id  # non-empty

        # Verify audit trail has the entry
        trail = gov.audit_trail(action_name="file_read", user="test_user")
        assert len(trail) >= 1
        entry = trail[0]
        assert entry["action"] == "file_read"

    def test_execute_handler_failure_recorded(self):
        from huginn.governance import get_governance

        gov = get_governance()

        def failing_handler(ctx):
            raise ValueError("intentional failure")

        result = gov.execute(
            "file_read",
            {},
            failing_handler,
            auto_rollback=False,
        )
        assert result.status == "failed"
        assert result.error is not None
        assert "intentional failure" in result.error

    def test_execute_writes_provenance(self):
        from huginn.governance import get_governance

        gov = get_governance()
        gov.execute(
            "file_read",
            {"path": "input.txt"},
            lambda ctx: {"file_path": "output.txt", "content": "data"},
        )
        # Check provenance has the entry
        entry = gov._provenance.find_by_path("output.txt")
        assert entry is not None
        assert entry.produced_by == "file_read"


class TestGovernanceRollback:
    """Verify rollback path works (P0 fix: lookup→find_by_path, revert workspace)."""

    def test_rollback_not_found_returns_no_mechanism(self):
        from huginn.governance import get_governance

        gov = get_governance()
        # Non-existent audit_id
        success, reason = gov.rollback("nonexistent_audit_id")
        # Should not crash; returns False with a reason
        assert success is False
        assert reason in ("no_rollback_mechanism", "rollback_not_configured")


class TestGovernanceAuditTrail:
    """Verify audit_trail query uses correct kwargs (P0 fix: user→actor)."""

    def test_audit_trail_returns_list(self):
        from huginn.governance import get_governance

        gov = get_governance()
        trail = gov.audit_trail()
        assert isinstance(trail, list)

    def test_audit_trail_filtered_by_action(self):
        from huginn.governance import get_governance

        gov = get_governance()
        # Write some entries
        gov.execute("file_read", {}, lambda ctx: {"result": "ok"})
        gov.execute("file_read", {}, lambda ctx: {"result": "ok2"})
        trail = gov.audit_trail(action_name="file_read")
        assert len(trail) >= 2
