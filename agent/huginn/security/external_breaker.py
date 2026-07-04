"""External API circuit breaker -- protects HTTP calls to third-party services.

The existing CircuitBreaker in huginn.agents.circuit_breaker only covers tool
calls (keyed by tool_name). External HTTP calls -- LLM providers, Semantic
Scholar, OpenAlex, Materials Project, etc. -- have no protection at all. If a
service goes down, the agent keeps hammering it and every request times out,
dragging the whole reasoning loop down.

This module applies the same classic three-state pattern (closed / open /
half_open) but keyed by service name instead of tool name. The states:

    closed    Normal traffic. Count consecutive failures.
    open      Failure threshold reached -- reject immediately, wait for cooldown.
    half_open Cooldown expired -- let one trial request through. Success closes
              the circuit; failure re-opens it.

In-memory only, resets on restart (a stale open circuit shouldn't block a
fresh process). Thread-safe via an RLock, same as the tool breaker.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Any, Generator

# Services we know about up front. Pre-registering them means they show up
# in list_all() / dashboards even before the first call, and lets us apply
# per-service config overrides at construction time.
_KNOWN_SERVICES = (
    "llm_provider",
    "semantic_scholar",
    "openalex",
    "materials_project",
    "zenodo",
    "crossref",
    "datacite",
    "nomad",
)


class CircuitOpenError(Exception):
    """Raised when the breaker is open for a given service.

    Callers should catch this and degrade gracefully (return empty results,
    fall back to a different source, etc.) rather than letting it crash the
    agent.
    """

    def __init__(self, service: str, retry_after: float = 0.0) -> None:
        self.service = service
        self.retry_after = retry_after
        msg = f"circuit open for '{service}'"
        if retry_after:
            msg += f", retry after {retry_after:.1f}s"
        super().__init__(msg)


class _ServiceState:
    """Per-service breaker state. Mirrors _BreakerState in circuit_breaker.py."""

    __slots__ = (
        "state",
        "consecutive_failures",
        "last_failure_time",
        "last_error",
        "half_open_trials",
    )

    def __init__(self) -> None:
        self.state: str = "closed"
        self.consecutive_failures: int = 0
        self.last_failure_time: float | None = None
        self.last_error: str = ""
        self.half_open_trials: int = 0


class ExternalCircuitBreaker:
    """Circuit breaker for external HTTP services, keyed by service name.

    Each service gets an independent three-state machine. Per-service config
    overrides can be applied via configure(); services without explicit config
    fall back to the defaults passed at construction time.

    Usage::

        breaker = get_external_breaker()
        if not breaker.can_call("semantic_scholar"):
            return []
        try:
            data = http_get(url)
            breaker.record_success("semantic_scholar")
        except Exception as exc:
            breaker.record_failure("semantic_scholar", str(exc))
            raise

    Or equivalently via the circuit_guard context manager::

        try:
            with circuit_guard("semantic_scholar"):
                data = http_get(url)
        except CircuitOpenError:
            return []
    """

    _singleton_lock = threading.Lock()
    _singleton: ExternalCircuitBreaker | None = None

    def __init__(
        self,
        failure_threshold: int = 5,
        cooldown_seconds: float = 60.0,
        half_open_max: int = 1,
    ) -> None:
        self._default_threshold = failure_threshold
        self._default_cooldown = cooldown_seconds
        self._default_half_open_max = half_open_max
        self._lock = threading.RLock()
        self._states: dict[str, _ServiceState] = {}
        # Per-service config overrides. Keys: failure_threshold, cooldown_seconds,
        # half_open_max. Missing keys fall back to defaults.
        self._config: dict[str, dict[str, Any]] = {}
        # Touch each known service so it shows up in list_all() right away.
        for svc in _KNOWN_SERVICES:
            self._get_state(svc)

    @classmethod
    def shared(cls) -> ExternalCircuitBreaker:
        """Process-level singleton. One breaker for all external calls."""
        with cls._singleton_lock:
            if cls._singleton is None:
                cls._singleton = cls()
            return cls._singleton

    def configure(
        self,
        service: str,
        failure_threshold: int | None = None,
        cooldown_seconds: float | None = None,
        half_open_max: int | None = None,
    ) -> None:
        """Override breaker config for a specific service.

        Only the kwargs you pass are overridden; the rest keep their previous
        (or default) values. Call this during startup if you know e.g. S2 is
        extra flaky and needs a shorter threshold.
        """
        with self._lock:
            cfg = self._config.setdefault(service, {})
            if failure_threshold is not None:
                cfg["failure_threshold"] = failure_threshold
            if cooldown_seconds is not None:
                cfg["cooldown_seconds"] = cooldown_seconds
            if half_open_max is not None:
                cfg["half_open_max"] = half_open_max

    # ---- internal helpers ----

    def _get_state(self, service: str) -> _ServiceState:
        st = self._states.get(service)
        if st is None:
            st = _ServiceState()
            self._states[service] = st
        return st

    def _threshold(self, service: str) -> int:
        return self._config.get(service, {}).get(
            "failure_threshold", self._default_threshold
        )

    def _cooldown(self, service: str) -> float:
        return self._config.get(service, {}).get(
            "cooldown_seconds", self._default_cooldown
        )

    def _half_open_max(self, service: str) -> int:
        return self._config.get(service, {}).get(
            "half_open_max", self._default_half_open_max
        )

    def _refresh(self, st: _ServiceState, service: str) -> None:
        """If the cooldown has elapsed, flip open -> half_open for a trial."""
        if st.state == "open" and st.last_failure_time is not None:
            if time.time() - st.last_failure_time >= self._cooldown(service):
                st.state = "half_open"
                st.half_open_trials = 0

    # ---- public API ----

    def can_call(self, service: str) -> bool:
        """Whether a call to this service should be allowed right now.

        closed always passes. half_open passes up to half_open_max trial
        requests. open is blocked until the cooldown expires.
        """
        with self._lock:
            st = self._get_state(service)
            self._refresh(st, service)
            if st.state == "closed":
                return True
            if st.state == "half_open":
                if st.half_open_trials < self._half_open_max(service):
                    st.half_open_trials += 1
                    return True
                return False
            return False

    def record_success(self, service: str) -> None:
        """Mark a call as successful -- reset failure count and close circuit."""
        with self._lock:
            st = self._get_state(service)
            st.consecutive_failures = 0
            st.half_open_trials = 0
            st.state = "closed"

    def record_failure(self, service: str, error: str = "") -> None:
        """Mark a call as failed. Accumulate until threshold, then open.

        A failure during half_open immediately re-opens the circuit and
        restarts the cooldown timer.
        """
        with self._lock:
            st = self._get_state(service)
            st.consecutive_failures += 1
            st.last_failure_time = time.time()
            if error:
                st.last_error = error

            if st.state == "half_open":
                st.state = "open"
                st.half_open_trials = 0
                return

            if st.consecutive_failures >= self._threshold(service):
                st.state = "open"

    def get_state(self, service: str) -> str:
        """Return closed / open / half_open. Triggers cooldown refresh."""
        with self._lock:
            st = self._get_state(service)
            self._refresh(st, service)
            return st.state

    def get_stats(self, service: str) -> dict[str, Any]:
        """Detailed breaker info for one service, including remaining cooldown."""
        with self._lock:
            st = self._get_state(service)
            self._refresh(st, service)
            retry_after = 0.0
            if st.state == "open" and st.last_failure_time is not None:
                remaining = self._cooldown(service) - (
                    time.time() - st.last_failure_time
                )
                retry_after = max(0.0, remaining)
            return {
                "service": service,
                "state": st.state,
                "consecutive_failures": st.consecutive_failures,
                "failure_threshold": self._threshold(service),
                "last_failure_time": st.last_failure_time,
                "last_error": st.last_error,
                "half_open_trials": st.half_open_trials,
                "retry_after": round(retry_after, 2),
            }

    def list_all(self) -> list[dict[str, Any]]:
        """Snapshot of all tracked services. Handy for dashboards / debugging."""
        with self._lock:
            return [self.get_stats(svc) for svc in self._states]

    def force_open(self, service: str, reason: str = "") -> None:
        """Manually trip the breaker for a service.

        Useful when an external monitor detects a service is down before
        enough failures accumulate organically.
        """
        with self._lock:
            st = self._get_state(service)
            st.state = "open"
            st.last_failure_time = time.time()
            st.half_open_trials = 0
            if reason:
                st.last_error = reason

    def reset(self, service: str | None = None) -> None:
        """Reset the breaker. Pass a service name to reset just one;
        pass None to reset everything."""
        with self._lock:
            if service is None:
                self._states.clear()
                # Re-seed known services
                for svc in _KNOWN_SERVICES:
                    self._get_state(svc)
            else:
                self._states.pop(service, None)


def get_external_breaker() -> ExternalCircuitBreaker:
    """Get the process-wide singleton ExternalCircuitBreaker."""
    return ExternalCircuitBreaker.shared()


@contextmanager
def circuit_guard(service: str) -> Generator[None, None, None]:
    """Context manager that wraps an external HTTP call with breaker logic.

    Raises CircuitOpenError immediately if the circuit is open. On clean
    exit, records success. On any exception, records failure and re-raises.

    Usage::

        try:
            with circuit_guard("semantic_scholar"):
                data = await _http_get_json(url)
        except CircuitOpenError:
            logger.info("S2 circuit open, skipping")
            return []
        except Exception as exc:
            logger.warning("S2 request failed: %s", exc)
            return []
    """
    breaker = get_external_breaker()
    if not breaker.can_call(service):
        stats = breaker.get_stats(service)
        raise CircuitOpenError(service, stats.get("retry_after", 0.0))
    try:
        yield
        breaker.record_success(service)
    except Exception as exc:
        breaker.record_failure(service, str(exc))
        raise


__all__ = [
    "CircuitOpenError",
    "ExternalCircuitBreaker",
    "circuit_guard",
    "get_external_breaker",
]
