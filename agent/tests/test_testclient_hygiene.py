"""Regression guard: forbid the TestClient anti-pattern that froze CI.

History: tests created ``client = TestClient(app)`` at module scope or inside
functions without ever closing it. Each unclosed client leaked an anyio portal
thread (freezing xdist workers) and, for the full app, never shut down the
lifespan (memory accumulated until OOM). These failures were intermittent and
"fixed-then-broken" until the suite was refactored onto shared, context-managed
clients (see conftest.py's ``app_client`` fixture).

This test scans every ``*.py`` under tests/ and fails immediately if anyone
reintroduces an unbound ``VAR = TestClient(...)``. Only the context-manager form
is allowed:

    with TestClient(app) as client:
        ...

Running as part of the normal pytest run means CI enforces this guard on every
push — prevention is automated, not reliant on code review remembering it.
"""

from __future__ import annotations

import re
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent

# Matches `VAR = TestClient(` at any indentation. The allowed `with TestClient(...)`
# form never matches: after `with ` the token is followed by `TestClient(`, not `=`.
_UNCLOSED = re.compile(r"^\s*\w+\s*=\s*TestClient\(")


def _python_files() -> list[Path]:
    return sorted(_TESTS_DIR.rglob("*.py"))


def test_only_context_managed_testclient():
    offenders: list[str] = []
    for path in _python_files():
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
        ):
            if _UNCLOSED.match(line):
                offenders.append(
                    f"{path.relative_to(_TESTS_DIR)}:{lineno}:" f" {line.strip()}"
                )

    assert not offenders, (
        "TestClient must be used as a context manager (`with TestClient(...) as c:`)\n"
        "so the portal thread is closed and the app lifespan shuts down. Found\n"
        "unclosed instances (each leaks a thread and, for the full app, memory):\n"
        + "\n".join(offenders)
    )
