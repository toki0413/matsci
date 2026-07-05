"""HTTP-level concurrency tests for the Huginn FastAPI server.

test_concurrent_e2e.py drives the HuginnAgent singleton directly; this
module stresses the server surface itself -- many simultaneous clients
hitting read endpoints, the credential store, and the SQLite long-term
memory. Run with::

    python -m pytest tests/test_concurrency.py -v --tb=short --no-cov
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
from httpx import ASGITransport

from huginn.memory.longterm import LongTermMemory
from huginn.security import credential_store as cs_mod
from huginn.security.credential_store import (
    CredentialStore,
    ServiceCredentialStore,
)
from huginn.server import app

# conftest sets HUGINN_API_KEY=test-key and HUGINN_DEV_MODE=1, so auth is
# bypassed anyway -- but we send the header regardless so the test stays
# correct the day dev mode is turned off in CI.
_HEADERS = {"X-HUGINN-API-KEY": "test-key"}


@pytest.fixture(autouse=True)
def _no_rate_limit():
    """Disable the per-IP HTTP rate limiter for this module.

    The suite fires 100+ requests inside a single 60s window; the default
    120 req/min ceiling would start returning 429s partway through and
    flake the later tests. server.py swaps sys.modules for a wrapper, so
    patching huginn.server._RATE_LIMIT only touches the wrapper's copy --
    we patch the global the middleware closure actually reads from.
    """
    from huginn.server import rate_limit_middleware

    mw_globals = rate_limit_middleware.__globals__
    saved = mw_globals.get("_RATE_LIMIT")
    mw_globals["_RATE_LIMIT"] = 0
    try:
        yield
    finally:
        if saved is not None:
            mw_globals["_RATE_LIMIT"] = saved


@pytest.fixture
def isolated_stores(tmp_path, monkeypatch):
    """Point the credentials route at tmp_path-backed stores.

    Keeps the CRUD test hermetic -- no writes to the shared ~/.huginn cred
    DB and no leakage between tests. Mirrors the isolation pattern in
    test_credential_store.py.
    """
    from cryptography.fernet import Fernet

    from huginn.routes import credentials as cred_route

    fernet = Fernet(Fernet.generate_key())
    llm_store = CredentialStore(tmp_path / "cred.sqlite", fernet=fernet)
    svc_store = ServiceCredentialStore(tmp_path / "svc_creds.json", fernet=fernet)

    # The route module holds its own imported binding to these getters, so
    # patch that reference -- not just the security module's.
    monkeypatch.setattr(cred_route, "get_credential_store", lambda: llm_store)
    monkeypatch.setattr(cred_route, "get_service_credential_store", lambda: svc_store)
    monkeypatch.setattr(cs_mod, "get_credential_store", lambda: llm_store)
    monkeypatch.setattr(cs_mod, "get_service_credential_store", lambda: svc_store)
    return llm_store


@pytest.mark.asyncio
async def test_concurrent_health_checks():
    """50 simultaneous liveness probes must all return 200.

    /health/live is the cheap path -- it must never fail under load since
    Kubernetes uses it to decide whether to kill the pod.
    """
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", headers=_HEADERS
    ) as client:
        responses = await asyncio.gather(
            *[client.get("/health/live") for _ in range(50)]
        )

    assert len(responses) == 50
    statuses = [r.status_code for r in responses]
    assert all(s == 200 for s in statuses), f"non-200 health probes: {statuses}"
    assert all(r.json()["status"] == "alive" for r in responses)


@pytest.mark.asyncio
async def test_concurrent_tool_listing():
    """20 concurrent GET /tools must all succeed and agree.

    The ToolRegistry is a process-wide singleton populated at import
    time, so every response must be byte-identical -- a mismatch would
    point at shared mutable state being mutated mid-read.
    """
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", headers=_HEADERS
    ) as client:
        responses = await asyncio.gather(
            *[client.get("/tools") for _ in range(20)]
        )

    assert all(r.status_code == 200 for r in responses)
    bodies = [r.json() for r in responses]
    assert all(b == bodies[0] for b in bodies), "tool listings diverged across reads"


@pytest.mark.asyncio
async def test_concurrent_skill_listing():
    """20 concurrent GET /skills must all return 200 and the same list."""
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", headers=_HEADERS
    ) as client:
        responses = await asyncio.gather(
            *[client.get("/skills") for _ in range(20)]
        )

    assert all(r.status_code == 200 for r in responses)
    bodies = [r.json() for r in responses]
    assert all(b == bodies[0] for b in bodies), "skill listings diverged across reads"


@pytest.mark.asyncio
async def test_concurrent_credential_crud(isolated_stores):
    """10 concurrent POST /credentials with distinct names must all land,
    then a single GET /credentials must list every one of them.

    Hits the real /credentials endpoint (the kind/name-keyed SQLite store),
    not the service-keyed variant, so we can create many entries with
    arbitrary names in parallel and exercise Fernet + SQLite write contention.
    """
    transport = ASGITransport(app=app)
    names = [f"conc-svc-{i}" for i in range(10)]

    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", headers=_HEADERS
    ) as client:
        payloads = [
            {
                "kind": "llm",
                "name": name,
                "metadata": {"provider": "openai", "model": "gpt-3.5-turbo"},
                "secret": f"key-{i}",
            }
            for i, name in enumerate(names)
        ]
        creates = await asyncio.gather(
            *[client.post("/credentials", json=p) for p in payloads]
        )

        assert all(r.status_code == 200 for r in creates)
        for r in creates:
            assert r.json()["success"] is True, f"create failed: {r.text}"

        # Read back -- all 10 must be retrievable from one listing.
        listing = await client.get("/credentials")
        assert listing.status_code == 200
        listed_names = {c["name"] for c in listing.json()["credentials"]}
        missing = set(names) - listed_names
        assert not missing, f"creds lost after concurrent writes: {missing}"


@pytest.mark.asyncio
async def test_concurrent_sqlite_writes(tmp_path):
    """20 concurrent LongTermMemory.store() calls against one tmp db must
    all persist. SQLite WAL is supposed to serialize the writers without
    any 'database is locked' failures -- this pins that guarantee.
    """
    mem = LongTermMemory(str(tmp_path / "concurrent.db"))

    contents = [f"memory-{i}" for i in range(20)]
    # store() is blocking sqlite; fan the calls out across worker threads
    # so they genuinely contend on the same db file.
    await asyncio.gather(
        *[
            asyncio.to_thread(mem.store, content=c, category="test")
            for c in contents
        ]
    )

    rows = mem.list_all(alive_only=False, limit=200)
    stored = {row["content"] for row in rows}
    assert len(rows) == 20, f"expected 20 rows, got {len(rows)}: {stored}"
    missing = set(contents) - stored
    assert not missing, f"memories lost under concurrent writes: {missing}"


@pytest.mark.asyncio
async def test_concurrent_mixed_endpoints():
    """30 concurrent requests spread across /health/live, /tools, /skills,
    and /diagnostics must all return 200. Mixes read-only and aggregation
    paths to catch shared-state issues that single-endpoint bursts miss.
    """
    paths = ["/health/live", "/tools", "/skills", "/diagnostics"]

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", headers=_HEADERS
    ) as client:
        responses = await asyncio.gather(
            *[client.get(paths[i % len(paths)]) for i in range(30)]
        )

    statuses = [r.status_code for r in responses]
    assert all(s == 200 for s in statuses), f"non-200 in mixed burst: {statuses}"
