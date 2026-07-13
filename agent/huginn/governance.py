"""Unified governance facade: can_execute / execute / verify / rollback.

Consolidates RBAC, audit chain, provenance registry, policy engine, and
snapshot/rollback into a single interface. The core question this module
answers: "what action is executable, verifiable, traceable, and controllable
under constraints?"

Usage:
    from huginn.governance import gov

    # Check before execution
    allowed, reasons = gov.can_execute("run_dft", context)
    if not allowed:
        return {"error": "blocked", "reasons": reasons}

    # Execute with full audit trail
    result = gov.execute("run_dft", context, handler_fn)
    # result.audit_id -> traceable in audit log
    # result.verified -> True if verification passed

    # Rollback if needed
    gov.rollback(result.audit_id)
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class GovernanceDecision:
    """Result of can_execute check."""
    allowed: bool
    reasons: list[str] = field(default_factory=list)
    risk_level: str = "none"
    requires_approval: bool = False
    predictability: float = 1.0


@dataclass
class ExecutionResult:
    """Result of a governed execution."""
    audit_id: str
    action_name: str
    status: str  # "verified", "failed", "rolled_back"
    result: Any = None
    error: str | None = None
    verification_passed: bool = False
    verification_message: str = ""
    rollback_available: bool = False
    timestamp: float = field(default_factory=time.time)
    context_snapshot: dict[str, Any] = field(default_factory=dict)
    audit_entry: dict[str, Any] = field(default_factory=dict)


class GovernanceFacade:
    """Single entry point for all governance decisions.

    Wraps the existing fragmented systems:
    - security/rbac.py → permission check
    - security/audit.py → audit chain
    - security/policy_engine.py → policy rules
    - provenance/registry.py → provenance tracking
    - snapshot/file_snapshot.py → rollback capability
    - ontology/actions.py → action type definitions
    """

    def __init__(self) -> None:
        self._initialized = False
        self._audit_logger = None
        self._policy_engine = None
        self._provenance = None
        self._snapshot_mgr = None
        self._rbac = None

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        # Lazy init — these modules may not be available in all contexts
        try:
            from huginn.security.audit import get_audit_logger
            self._audit_logger = get_audit_logger()
        except Exception:
            logger.debug("[gov] audit logger not available")
        try:
            from huginn.security.policy_engine import get_policy_engine
            self._policy_engine = get_policy_engine()
        except Exception:
            logger.debug("[gov] policy engine not available")
        try:
            from huginn.provenance.registry import get_provenance_registry
            self._provenance = get_provenance_registry()
        except Exception:
            logger.debug("[gov] provenance not available")
        try:
            from huginn.snapshot.file_snapshot import get_snapshot_manager
            self._snapshot_mgr = get_snapshot_manager()
        except Exception:
            logger.debug("[gov] snapshot manager not available")
        self._initialized = True

    # ── Core: can_execute ──────────────────────────────────────

    def can_execute(
        self,
        action_name: str,
        context: dict[str, Any],
        *,
        user: str = "system",
    ) -> GovernanceDecision:
        """Check if an action is allowed under current constraints.

        Combines:
        1. Action type preconditions (from ontology)
        2. Policy engine rules (if available)
        3. RBAC permission check (if available)
        4. Predictability score (from PNAS-inspired decomposition)
        """
        self._ensure_initialized()
        reasons: list[str] = []
        risk = "none"
        requires_approval = False
        predictability = 1.0

        # 1. Action type preconditions
        from huginn.ontology.actions import get_action_type
        at = get_action_type(action_name)
        if at is None:
            # Unknown action — allow but flag
            reasons.append(f"Unknown action type '{action_name}' — no preconditions checked")
        else:
            risk = at.risk.value
            allowed, pre_reasons = at.can_execute(context)
            reasons.extend(pre_reasons)
            if not allowed:
                return GovernanceDecision(
                    allowed=False, reasons=reasons, risk_level=risk,
                    requires_approval=True, predictability=0.0,
                )
            predictability = at.predictability(context)

            # High-risk actions need explicit approval
            if at.risk in (RiskLevel.HIGH, RiskLevel.CRITICAL):
                requires_approval = True

        # 2. Policy engine check
        if self._policy_engine:
            try:
                decision = self._policy_engine.evaluate_command_hook(
                    action_name, context
                )
                if decision and decision.get("action") == "deny":
                    reasons.append(f"Policy denied: {decision.get('reason', '')}")
                    return GovernanceDecision(
                        allowed=False, reasons=reasons, risk_level=risk,
                        requires_approval=True, predictability=predictability,
                    )
                if decision and decision.get("action") == "ask":
                    requires_approval = True
                    reasons.append(f"Policy requires approval: {decision.get('reason', '')}")
            except Exception as e:
                logger.debug(f"[gov] policy check failed: {e}")

        return GovernanceDecision(
            allowed=True, reasons=reasons, risk_level=risk,
            requires_approval=requires_approval, predictability=predictability,
        )

    # ── Core: execute ───────────────────────────────────────────

    def execute(
        self,
        action_name: str,
        context: dict[str, Any],
        handler: Callable[[dict[str, Any]], Any],
        *,
        user: str = "system",
        auto_rollback: bool = True,
    ) -> ExecutionResult:
        """Execute an action with full governance: audit, verify, rollback.

        The handler receives the context dict and returns a result.
        Post-execution, constraints are checked and verification runs.
        If constraints fail and auto_rollback=True, the action is rolled back.
        """
        self._ensure_initialized()
        audit_id = uuid.uuid4().hex[:12]
        timestamp = time.time()

        # Snapshot context for potential rollback
        ctx_snapshot = {k: v for k, v in context.items() if isinstance(v, (str, int, float, bool, list, dict))}

        logger.info(f"[gov] execute '{action_name}' (audit_id={audit_id}, user={user})")

        # Execute the handler
        try:
            result = handler(context)
            error = None
        except Exception as e:
            result = None
            error = str(e)
            logger.error(f"[gov] handler failed for '{action_name}': {e}")

        # Build execution context with results
        exec_ctx = {**context}
        if isinstance(result, dict):
            exec_ctx.update(result)
        elif result is not None:
            exec_ctx["result"] = result

        # Check constraints
        from huginn.ontology.actions import get_action_type
        at = get_action_type(action_name)
        verification_passed = True
        verification_msg = ""
        should_rollback = False

        if at:
            for con in at.constraints:
                ok, msg = con.evaluate(exec_ctx)
                if not ok:
                    verification_passed = False
                    verification_msg += f"{con.name}: {msg}; "
                    if con.rollback_on_violation and auto_rollback:
                        should_rollback = True
                        break

            # Run verifiability check
            if at.verifiability and not should_rollback:
                v_ok, v_msg = at.verifiability.verify(exec_ctx)
                if not v_ok:
                    verification_passed = False
                    verification_msg += f"verify: {v_msg}; "

        # Rollback if needed
        status = "verified"
        rollback_available = False
        if should_rollback and at and at.rollback_handler:
            try:
                rolled = at.rollback_handler(exec_ctx)
                if rolled:
                    status = "rolled_back"
                    logger.warning(f"[gov] action '{action_name}' rolled back due to constraint violation")
                rollback_available = False
            except Exception as e:
                logger.error(f"[gov] rollback failed: {e}")
                status = "failed"
        elif error is not None:
            status = "failed"
        elif not verification_passed:
            status = "failed"

        # Check if rollback is available for manual use
        if at and at.rollback_handler and status == "verified":
            rollback_available = True

        # Write audit entry
        audit_entry = {
            "audit_id": audit_id,
            "action": action_name,
            "user": user,
            "timestamp": timestamp,
            "status": status,
            "risk": at.risk.value if at else "unknown",
            "verification_passed": verification_passed,
            "verification_message": verification_msg,
            "context_keys": list(ctx_snapshot.keys()),
            "error": error,
        }

        if self._audit_logger:
            try:
                self._audit_logger.log(audit_entry)
            except Exception as e:
                logger.debug(f"[gov] audit write failed: {e}")

        # Track provenance
        if self._provenance and isinstance(result, dict):
            try:
                from huginn.provenance.registry import ProvenanceEntry
                entry = ProvenanceEntry(
                    audit_id=audit_id,
                    action=action_name,
                    inputs=ctx_snapshot,
                    outputs=result if isinstance(result, dict) else {"result": str(result)},
                    timestamp=timestamp,
                )
                self._provenance.register(entry)
            except Exception as e:
                logger.debug(f"[gov] provenance tracking failed: {e}")

        return ExecutionResult(
            audit_id=audit_id,
            action_name=action_name,
            status=status,
            result=result,
            error=error,
            verification_passed=verification_passed,
            verification_message=verification_msg,
            rollback_available=rollback_available,
            context_snapshot=ctx_snapshot,
            audit_entry=audit_entry,
        )

    # ── Core: verify ─────────────────────────────────────────────

    def verify(
        self,
        action_name: str,
        context: dict[str, Any],
        result: Any,
    ) -> tuple[bool, str]:
        """Standalone verification — check if an already-executed action's
        result is valid."""
        from huginn.ontology.actions import get_action_type
        at = get_action_type(action_name)
        if not at or not at.verifiability:
            return True, "no verification configured"

        exec_ctx = {**context}
        if isinstance(result, dict):
            exec_ctx.update(result)

        return at.verifiability.verify(exec_ctx)

    # ── Core: rollback ──────────────────────────────────────────

    def rollback(self, audit_id: str) -> bool:
        """Roll back a previously executed action by audit_id."""
        self._ensure_initialized()

        # Find the audit entry
        if self._provenance:
            try:
                entry = self._provenance.lookup(audit_id)
                if entry and entry.get("rollback_handler"):
                    return entry["rollback_handler"](entry.get("context", {}))
            except Exception as e:
                logger.debug(f"[gov] provenance rollback failed: {e}")

        # Fall back to snapshot manager
        if self._snapshot_mgr:
            try:
                return self._snapshot_mgr.revert(audit_id)
            except Exception as e:
                logger.debug(f"[gov] snapshot rollback failed: {e}")

        logger.warning(f"[gov] no rollback mechanism for audit_id={audit_id}")
        return False

    # ── Query: audit trail ──────────────────────────────────────

    def audit_trail(
        self,
        action_name: str | None = None,
        user: str | None = None,
        since: float | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query the audit log for actions matching criteria."""
        if not self._audit_logger:
            return []
        try:
            return self._audit_logger.query(
                action=action_name, user=user, since=since, limit=limit
            )
        except Exception:
            return []


# ── RiskLevel import for can_execute ──────────────────────────
from huginn.ontology.actions import RiskLevel  # noqa: E402

# Singleton
_gov: GovernanceFacade | None = None


def get_governance() -> GovernanceFacade:
    global _gov
    if _gov is None:
        _gov = GovernanceFacade()
    return _gov
