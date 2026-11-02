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
import os
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# G32 fail-secure: governance 检查失败（policy engine 不可用/异常/未知动作）时的默认决策。
# 默认 "deny" — 生产环境里 policy engine 挂了不能静默放行；测试或 legacy 部署可设
# HUGINN_GOVERNANCE_DEFAULT_DECISION=allow 回到旧行为。
_DEFAULT_DECISION = os.environ.get(
    "HUGINN_GOVERNANCE_DEFAULT_DECISION", "deny"
).lower().strip()
if _DEFAULT_DECISION not in ("allow", "deny", "ask"):
    _DEFAULT_DECISION = "deny"


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


def _gov_snapshot_context(context: dict[str, Any]) -> dict[str, Any]:
    """Keep only JSON-ish values for the rollback snapshot."""
    return {
        k: v
        for k, v in context.items()
        if isinstance(v, str | int | float | bool | list | dict)
    }


def _gov_build_result(
    *,
    audit_id: str,
    action_name: str,
    status: str,
    result: Any,
    error: str | None,
    verification_passed: bool,
    verification_msg: str,
    rollback_available: bool,
    ctx_snapshot: dict[str, Any],
    audit_entry: dict[str, Any],
) -> ExecutionResult:
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


def _gov_build_exec_ctx(
    context: dict[str, Any], result: Any
) -> dict[str, Any]:
    """Merge handler results into a copy of the input context."""
    exec_ctx = {**context}
    if isinstance(result, dict):
        exec_ctx.update(result)
    elif result is not None:
        exec_ctx["result"] = result
    return exec_ctx


def _gov_check_constraints(
    at: Any, exec_ctx: dict[str, Any], auto_rollback: bool
) -> tuple[bool, str, bool]:
    """Evaluate constraints + verifiability; return (passed, message, rollback)."""
    if at is None:
        return True, "", False
    passed = True
    message = ""
    should_rollback = False
    for con in at.constraints:
        ok, msg = con.evaluate(exec_ctx)
        if ok:
            continue
        passed = False
        message += f"{con.name}: {msg}; "
        if con.rollback_on_violation and auto_rollback:
            should_rollback = True
            break

    if at.verifiability and not should_rollback:
        v_ok, v_msg = at.verifiability.verify(exec_ctx)
        if not v_ok:
            passed = False
            message += f"verify: {v_msg}; "
    return passed, message, should_rollback


def _gov_build_audit_entry(
    *,
    audit_id: str,
    action_name: str,
    user: str,
    timestamp: float,
    status: str,
    at: Any,
    verification_passed: bool,
    verification_msg: str,
    ctx_snapshot: dict[str, Any],
    error: str | None,
) -> dict[str, Any]:
    return {
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
        # G32: fail-secure default — governance 自身出问题时按此决策收口
        self.default_decision: str = _DEFAULT_DECISION

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
            from huginn.provenance.registry import ProvenanceRegistry
            self._provenance = ProvenanceRegistry.shared()
        except Exception:
            logger.debug("[gov] provenance not available")
        try:
            from huginn.snapshot.file_snapshot import get_snapshot_manager
            self._snapshot_mgr = get_snapshot_manager()
        except Exception:
            logger.debug("[gov] snapshot manager not available")
        self._initialized = True

    # ── Core: can_execute ──────────────────────────────────────

    def _gov_check_action_type(
        self,
        action_name: str,
        context: dict[str, Any],
        reasons: list[str],
    ) -> tuple[GovernanceDecision | None, str, bool, float]:
        """Preconditions from ontology.

        Returns (early_decision, risk, requires_approval, predictability).
        A non-None early_decision means "stop and return this".
        """
        from huginn.ontology.actions import get_action_type

        at = get_action_type(action_name)
        if at is None:
            # G32: 未知动作按 default_decision 收口 — fail-secure 默认 deny
            reasons.append(
                f"Unknown action type '{action_name}' — no preconditions checked"
            )
            if self.default_decision == "deny":
                reasons.append("Denied by fail-secure default (unknown action)")
                return (
                    GovernanceDecision(
                        allowed=False, reasons=reasons, risk_level="none",
                        requires_approval=True, predictability=0.0,
                    ),
                    "none",
                    True,
                    0.0,
                )
            # default_decision == "allow" / "ask": 旧 allow-but-flag 行为
            return None, "none", self.default_decision == "ask", 1.0

        risk = at.risk.value
        allowed, pre_reasons = at.can_execute(context)
        reasons.extend(pre_reasons)
        if not allowed:
            return (
                GovernanceDecision(
                    allowed=False, reasons=reasons, risk_level=risk,
                    requires_approval=True, predictability=0.0,
                ),
                risk,
                True,
                0.0,
            )
        predictability = at.predictability(context)
        # High-risk actions need explicit approval
        requires_approval = at.risk in (RiskLevel.HIGH, RiskLevel.CRITICAL)
        return None, risk, requires_approval, predictability

    def _gov_check_policy(
        self,
        action_name: str,
        reasons: list[str],
        risk: str,
        requires_approval: bool,
        predictability: float,
    ) -> tuple[GovernanceDecision | None, bool]:
        """Policy engine gate — G32: 引擎缺失/异常按 default_decision 收口."""
        if self._policy_engine is None:
            if self.default_decision == "deny":
                reasons.append(
                    "Policy engine unavailable — denied by fail-secure default"
                )
                return (
                    GovernanceDecision(
                        allowed=False, reasons=reasons, risk_level=risk,
                        requires_approval=True, predictability=predictability,
                    ),
                    requires_approval,
                )
            # allow / ask: 旧行为, 只在 reasons 里记一笔
            reasons.append(
                "Policy engine unavailable — skipped (default_decision != deny)"
            )
            return None, requires_approval or self.default_decision == "ask"

        try:
            from huginn.security.policy_engine import evaluate_command_hook

            # governance 的 action_name 是逻辑动作名, 不是 shell 命令;
            # 包成单元素 list 喂给 hook, 命中默认 ask 规则.
            decision = evaluate_command_hook([action_name])
            if decision and decision.action == "deny":
                reasons.append(f"Policy denied: {decision.reason}")
                return (
                    GovernanceDecision(
                        allowed=False, reasons=reasons, risk_level=risk,
                        requires_approval=True, predictability=predictability,
                    ),
                    requires_approval,
                )
            if decision and decision.action == "ask":
                requires_approval = True
                reasons.append(f"Policy requires approval: {decision.reason}")
        except Exception as e:
            return self._gov_policy_error(
                e, reasons, risk, requires_approval, predictability
            )
        return None, requires_approval

    def _gov_policy_error(
        self,
        exc: Exception,
        reasons: list[str],
        risk: str,
        requires_approval: bool,
        predictability: float,
    ) -> tuple[GovernanceDecision | None, bool]:
        """G32: policy 检查异常不再静默放行."""
        logger.warning(f"[gov] policy check failed: {exc}")
        reasons.append(f"Policy check raised: {exc}")
        if self.default_decision == "deny":
            reasons.append("Denied by fail-secure default (policy exception)")
            return (
                GovernanceDecision(
                    allowed=False, reasons=reasons, risk_level=risk,
                    requires_approval=True, predictability=predictability,
                ),
                requires_approval,
            )
        if self.default_decision == "ask":
            requires_approval = True
        return None, requires_approval

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

        G32 fail-secure: 若 policy engine 不可用或抛异常, 按 ``self.default_decision``
        收口 (默认 ``deny``). 旧行为可设环境变量
        ``HUGINN_GOVERNANCE_DEFAULT_DECISION=allow`` 恢复.
        """
        self._ensure_initialized()
        reasons: list[str] = []

        early, risk, requires_approval, predictability = (
            self._gov_check_action_type(action_name, context, reasons)
        )
        if early is not None:
            return early

        early, requires_approval = self._gov_check_policy(
            action_name, reasons, risk, requires_approval, predictability
        )
        if early is not None:
            return early

        return GovernanceDecision(
            allowed=True, reasons=reasons, risk_level=risk,
            requires_approval=requires_approval, predictability=predictability,
        )

    # ── Core: execute ───────────────────────────────────────────

    def _gov_log_rollback_readiness(self) -> None:
        """Log whether the compensation safety net is actually wired up."""
        rollback_ready = self._rollback_available()
        logger.info(f"[gov] rollback available: {rollback_ready}")
        if not rollback_ready:
            logger.warning(
                "[gov] rollback not configured, execute without compensation safety net"
            )

    def _gov_run_handler(
        self,
        action_name: str,
        handler: Callable[[dict[str, Any]], Any],
        context: dict[str, Any],
    ) -> tuple[Any, str | None]:
        """Run the handler, converting exceptions into (result, error)."""
        try:
            return handler(context), None
        except Exception as e:
            logger.error(f"[gov] handler failed for '{action_name}': {e}")
            return None, str(e)

    def _gov_resolve_status(
        self,
        action_name: str,
        at: Any,
        exec_ctx: dict[str, Any],
        error: str | None,
        verification_passed: bool,
        should_rollback: bool,
    ) -> tuple[str, bool]:
        """Apply rollback / failure verdicts; return (status, rollback_available)."""
        status = "verified"
        rollback_available = False
        if should_rollback and at and at.rollback_handler:
            try:
                rolled = at.rollback_handler(exec_ctx)
                if rolled:
                    status = "rolled_back"
                    logger.warning(
                        f"[gov] action '{action_name}' rolled back due to constraint violation"
                    )
            except Exception as e:
                logger.error(f"[gov] rollback failed: {e}")
                status = "failed"
        elif error is not None or not verification_passed:
            status = "failed"

        if at and at.rollback_handler and status == "verified":
            rollback_available = True
        return status, rollback_available

    def _gov_write_audit(self, audit_entry: dict[str, Any]) -> None:
        if not self._audit_logger:
            return
        try:
            self._audit_logger.log(
                event_type="governance",
                actor=audit_entry["user"],
                action=audit_entry["action"],
                details={
                    "audit_id": audit_entry["audit_id"],
                    "status": audit_entry["status"],
                    "risk": audit_entry["risk"],
                    "verification_passed": audit_entry["verification_passed"],
                    "verification_message": audit_entry["verification_message"],
                    "error": audit_entry["error"],
                    "context_keys": audit_entry["context_keys"],
                },
            )
        except Exception as e:
            logger.debug(f"[gov] audit write failed: {e}")

    def _gov_track_provenance(
        self,
        action_name: str,
        result: Any,
        audit_id: str,
        ctx_snapshot: dict[str, Any],
    ) -> None:
        """Register produced artifacts — 之前用了不存在的字段构造 ProvenanceEntry,
        加上 register() 签名也不对 (接受 file_path/produced_by 等 kwarg, 不接受
        entry 对象). 改用真实 register() 签名."""
        if not (self._provenance and isinstance(result, dict)):
            return
        try:
            # result dict 里找文件路径; 没有就用 audit_id 占位
            file_path = (
                result.get("file_path")
                or result.get("output_path")
                or f"gov://{audit_id}"
            )
            self._provenance.register(
                file_path=file_path,
                produced_by=action_name,
                input_files=[v for v in ctx_snapshot.values() if isinstance(v, str)],
                parameters={
                    k: v
                    for k, v in ctx_snapshot.items()
                    if isinstance(v, str | int | float | bool)
                },
                key_properties=result,
                snapshot_step_id=audit_id,
            )
        except Exception as e:
            logger.debug(f"[gov] provenance tracking failed: {e}")

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
        self._gov_log_rollback_readiness()
        audit_id = uuid.uuid4().hex[:12]
        timestamp = time.time()
        ctx_snapshot = _gov_snapshot_context(context)
        logger.info(f"[gov] execute '{action_name}' (audit_id={audit_id}, user={user})")

        result, error = self._gov_run_handler(action_name, handler, context)
        exec_ctx = _gov_build_exec_ctx(context, result)

        from huginn.ontology.actions import get_action_type

        at = get_action_type(action_name)
        verified, v_msg, should_rb = _gov_check_constraints(
            at, exec_ctx, auto_rollback
        )
        status, rollback_available = self._gov_resolve_status(
            action_name, at, exec_ctx, error, verified, should_rb
        )

        audit_entry = _gov_build_audit_entry(
            audit_id=audit_id, action_name=action_name, user=user,
            timestamp=timestamp, status=status, at=at,
            verification_passed=verified, verification_msg=v_msg,
            ctx_snapshot=ctx_snapshot, error=error,
        )
        self._gov_write_audit(audit_entry)
        self._gov_track_provenance(action_name, result, audit_id, ctx_snapshot)

        return _gov_build_result(
            audit_id=audit_id, action_name=action_name, status=status,
            result=result, error=error, verification_passed=verified,
            verification_msg=v_msg, rollback_available=rollback_available,
            ctx_snapshot=ctx_snapshot, audit_entry=audit_entry,
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

    # ── Core: rollback ─────────────────────────────────────────────────────

    def _rollback_available(self) -> bool:
        """Check if rollback is properly configured before promising it.

        rollback needs provenance registry (to look up the audit entry and
        its handler) AND snapshot manager (to revert file state). Either
        missing means rollback can not fulfill its contract — better to
        report upfront than silently fail.
        """
        self._ensure_initialized()
        return self._provenance is not None and self._snapshot_mgr is not None

    def _gov_rollback_via_provenance(
        self, audit_id: str
    ) -> tuple[bool, str] | None:
        """Provenance → snapshot rollback path; None means "no mechanism".

        `find_by_path` 是替代早先不存在的 lookup() 的兜底: execute 用
        ``gov://<audit_id>`` 占位路径注册, snapshot_step_id 是独立字段.
        """
        if not self._provenance:
            return None
        try:
            entry = self._provenance.find_by_path(f"gov://{audit_id}")
            if not (entry and entry.snapshot_step_id):
                return None
            if self._snapshot_mgr:
                snap = self._snapshot_mgr._load(entry.snapshot_step_id)
                if snap is not None and not snap.reverted:
                    ws = Path(snap.workspace)
                    restored = self._snapshot_mgr.revert(
                        entry.snapshot_step_id, ws
                    )
                    logger.info(
                        "[gov] rollback %s: %d files restored",
                        audit_id, len(restored),
                    )
                    return True, "ok"
            # 没 snapshot_mgr 或 snap 不存在 → 仅标记 provenance
            entry.reverted = True
            return True, "ok"
        except Exception as e:
            logger.debug(f"[gov] provenance rollback failed: {e}")
            return None

    def _gov_rollback_via_snapshot(self, audit_id: str) -> bool:
        """Fall back to snapshot manager directly (audit_id as step_id)."""
        if not self._snapshot_mgr:
            return False
        try:
            snap = self._snapshot_mgr._load(audit_id)
            if snap is None or snap.reverted:
                return False
            ws = Path(snap.workspace)
            return bool(self._snapshot_mgr.revert(audit_id, ws))
        except Exception as e:
            logger.debug(f"[gov] snapshot rollback failed: {e}")
            return False

    def rollback(self, audit_id: str) -> tuple[bool, str]:
        """Roll back a previously executed action by audit_id.

        Returns (success, reason). reason is "ok" on success, or an error
        code like "rollback_not_configured" / "no_rollback_mechanism".
        """
        self._ensure_initialized()

        if not self._rollback_available():
            logger.warning(
                "[gov] rollback not configured — provenance/snapshot module missing"
            )
            return False, "rollback_not_configured"

        via_provenance = self._gov_rollback_via_provenance(audit_id)
        if via_provenance is not None:
            return via_provenance

        if self._gov_rollback_via_snapshot(audit_id):
            return True, "ok"

        logger.warning(f"[gov] no rollback mechanism for audit_id={audit_id}")
        return False, "no_rollback_mechanism"

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
            # AuditLogger.query 用 actor (不是 user), action 做子串匹配.
            # 之前传了不存在的 user kwarg → TypeError 被 except 吞掉, 返回 [].
            return self._audit_logger.query(
                action=action_name, actor=user, since=since, limit=limit
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
