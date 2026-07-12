"""Tests for HPC job management: RemoteJobRecord credential_id, REST endpoints,
and the background JobMonitor.

Covers:
  * RemoteJobRecord.credential_id field — default, round-trip, backward compat
  * GET /hpc/jobs — list tracked jobs
  * GET /hpc/jobs/{local_id} — single job details
  * POST /hpc/jobs/{local_id}/cancel — cancel via scheduler
  * POST /hpc/jobs/{local_id}/refresh — poll scheduler for latest status
  * POST /hpc/submit — creates a RemoteJobRecord in the store
  * JobMonitor — polls PENDING jobs, skips COMPLETED ones
"""

from __future__ import annotations

# ── Python 3.10 compat ──────────────────────────────────────────────────
# StrEnum and datetime.UTC both landed in 3.11.  Several modules in the
# routes package use them unconditionally, so we shim both before any
# huginn imports.  We also bypass huginn/routes/__init__.py entirely —
# it drags in the full server stack (agents, persona, etc.) which has
# even more 3.11-isms.  Instead we register a lightweight stub package
# and let Python import hpc.py directly from it.
import datetime
import enum
import sys
import types
from pathlib import Path

if sys.version_info < (3, 11):
    if not hasattr(enum, "StrEnum"):

        class _StrEnumShim(str, enum.Enum):
            pass

        enum.StrEnum = _StrEnumShim
    if not hasattr(datetime, "UTC"):
        datetime.UTC = datetime.timezone.utc

# Stub out huginn.routes so __init__.py never runs.  We only need hpc.py.
# We also have to set it as an attribute on the huginn package itself —
# mock.patch resolves targets via getattr(huginn, "routes") and won't
# find it through sys.modules alone.
import huginn  # noqa: E402 — safe, __init__.py is just a version string

_routes_dir = str(Path(__file__).resolve().parent.parent / "huginn" / "routes")
if "huginn.routes" not in sys.modules:
    _stub = types.ModuleType("huginn.routes")
    _stub.__path__ = [_routes_dir]
    sys.modules["huginn.routes"] = _stub
    huginn.routes = _stub

# ── Real imports ────────────────────────────────────────────────────────

import time
from unittest.mock import MagicMock, patch

import pytest

from huginn.execution.remote_job_store import RemoteJobRecord, RemoteJobStore
from huginn.hpc.client import HPCConfig, JobStatus


# ── Test app setup ──────────────────────────────────────────────────────
#
# The stub package above lets us import hpc.py in isolation.  We build a
# minimal FastAPI app with just the hpc router for endpoint tests.

@pytest.fixture
def hpc_app():
    """Minimal FastAPI app exposing only the hpc router."""
    from fastapi import FastAPI
    from huginn.routes.hpc import router

    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client(hpc_app):
    from fastapi.testclient import TestClient
    return TestClient(hpc_app)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Point HUGINN_WORKSPACE at a temp dir and return the path."""
    monkeypatch.setenv("HUGINN_WORKSPACE", str(tmp_path))
    return tmp_path


def _make_record(**overrides):
    """Quick helper to build a RemoteJobRecord with sane defaults."""
    defaults = dict(
        local_id="job-1",
        scheduler_id="12345",
        command=["python", "train.py"],
        cwd="/scratch/job-1",
        credential_id="cred-abc",
        status="PENDING",
        submitted_at=time.time(),
    )
    defaults.update(overrides)
    return RemoteJobRecord(**defaults)


# ════════════════════════════════════════════════════════════════════════
#  RemoteJobRecord — credential_id field
# ════════════════════════════════════════════════════════════════════════


class TestRemoteJobRecordCredentialId:
    def test_remote_job_record_has_credential_id(self):
        """The field exists and defaults to None when not provided."""
        record = RemoteJobRecord(
            local_id="r1",
            scheduler_id="100",
            command=["echo", "hi"],
            cwd="/tmp",
        )
        assert record.credential_id is None

    def test_record_round_trip_with_credential_id(self):
        """to_dict / from_dict preserves the credential_id value."""
        record = _make_record(credential_id="cred-xyz")
        dumped = record.to_dict()
        assert dumped["credential_id"] == "cred-xyz"

        restored = RemoteJobRecord.from_dict(dumped)
        assert restored.credential_id == "cred-xyz"
        # Make sure nothing else got mangled
        assert restored.local_id == record.local_id
        assert restored.scheduler_id == record.scheduler_id

    def test_record_backward_compatible(self):
        """Old saved records (no credential_id key) load fine with None."""
        old_data = {
            "local_id": "legacy-1",
            "scheduler_id": "999",
            "command": ["hostname"],
            "cwd": "/old/path",
            "status": "COMPLETED",
            "exit_code": 0,
            # deliberately no credential_id
        }
        record = RemoteJobRecord.from_dict(old_data)
        assert record.credential_id is None
        assert record.local_id == "legacy-1"
        assert record.status == "COMPLETED"


# ════════════════════════════════════════════════════════════════════════
#  GET /hpc/jobs — list tracked jobs
# ════════════════════════════════════════════════════════════════════════


class TestListJobsEndpoint:
    def test_list_jobs_endpoint(self, client, workspace):
        """GET /hpc/jobs returns all records from the store."""
        store = RemoteJobStore(workspace=workspace)
        store.add_or_update(_make_record(local_id="a", scheduler_id="1"))
        store.add_or_update(_make_record(local_id="b", scheduler_id="2", status="COMPLETED"))

        resp = client.get("/hpc/jobs")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["count"] == 2
        ids = {j["local_id"] for j in data["jobs"]}
        assert ids == {"a", "b"}

    def test_list_jobs_filter_by_credential(self, client, workspace):
        """The credential_id query param filters the result set."""
        store = RemoteJobStore(workspace=workspace)
        store.add_or_update(_make_record(local_id="a", credential_id="cred-1"))
        store.add_or_update(_make_record(local_id="b", credential_id="cred-2"))

        resp = client.get("/hpc/jobs", params={"credential_id": "cred-1"})
        data = resp.json()
        assert data["count"] == 1
        assert data["jobs"][0]["local_id"] == "a"

    def test_list_jobs_empty(self, client, workspace):
        """An empty store returns count 0."""
        resp = client.get("/hpc/jobs")
        data = resp.json()
        assert data["success"] is True
        assert data["count"] == 0


# ════════════════════════════════════════════════════════════════════════
#  GET /hpc/jobs/{local_id} — single job details
# ════════════════════════════════════════════════════════════════════════


class TestGetJobEndpoint:
    def test_get_job_endpoint(self, client, workspace):
        """GET /hpc/jobs/{id} returns the full record."""
        store = RemoteJobStore(workspace=workspace)
        store.add_or_update(_make_record(local_id="find-me", scheduler_id="42"))

        resp = client.get("/hpc/jobs/find-me")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["job"]["local_id"] == "find-me"
        assert data["job"]["scheduler_id"] == "42"

    def test_get_job_not_found(self, client, workspace):
        """Unknown local_id returns success=False."""
        resp = client.get("/hpc/jobs/does-not-exist")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is False
        assert "not found" in data["error"]


# ════════════════════════════════════════════════════════════════════════
#  POST /hpc/jobs/{local_id}/cancel
# ════════════════════════════════════════════════════════════════════════


class TestCancelJobEndpoint:
    def test_cancel_job_endpoint(self, client, workspace):
        """Cancel marks the record as CANCELLED and runs scancel on the host."""
        store = RemoteJobStore(workspace=workspace)
        store.add_or_update(
            _make_record(local_id="kill-me", scheduler_id="777", status="RUNNING")
        )

        fake_cfg = HPCConfig(host="hpc.test", username="tester", scheduler="slurm")

        with patch(
            "huginn.routes.hpc._resolve_hpc_config", return_value=(fake_cfg, None)
        ), patch("huginn.routes.hpc.HPCClient") as mock_cls:
            mock_client = MagicMock()
            mock_client._exec.return_value = ("", "", 0)
            mock_cls.return_value.__enter__.return_value = mock_client

            resp = client.post("/hpc/jobs/kill-me/cancel", json={})

        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["job"]["status"] == "CANCELLED"
        assert data["job"]["completed_at"] is not None

        # scancel should have been called with the scheduler id
        mock_client._exec.assert_called_once()
        assert "scancel" in mock_client._exec.call_args[0][0]
        assert "777" in mock_client._exec.call_args[0][0]

    def test_cancel_job_not_found(self, client, workspace):
        """Cancelling a non-existent job returns an error."""
        resp = client.post("/hpc/jobs/ghost/cancel", json={})
        data = resp.json()
        assert data["success"] is False
        assert "not found" in data["error"]


# ════════════════════════════════════════════════════════════════════════
#  POST /hpc/jobs/{local_id}/refresh
# ════════════════════════════════════════════════════════════════════════


class TestRefreshJobEndpoint:
    def test_refresh_job_endpoint(self, client, workspace):
        """Refresh polls the scheduler and updates the stored record."""
        store = RemoteJobStore(workspace=workspace)
        store.add_or_update(
            _make_record(local_id="refresh-me", scheduler_id="555", status="PENDING")
        )

        fake_cfg = HPCConfig(host="hpc.test", username="tester")
        fake_status = JobStatus(
            job_id="555", state="RUNNING", exit_code=None, message="ok"
        )

        with patch(
            "huginn.routes.hpc._resolve_hpc_config", return_value=(fake_cfg, None)
        ), patch("huginn.routes.hpc.HPCClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.poll_status.return_value = fake_status
            mock_cls.return_value.__enter__.return_value = mock_client

            resp = client.post("/hpc/jobs/refresh-me/refresh", json={})

        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["job"]["status"] == "RUNNING"
        assert data["job"]["message"] == "ok"

        # The store on disk should reflect the update
        updated = RemoteJobStore(workspace=workspace).get("refresh-me")
        assert updated.status == "RUNNING"

    def test_refresh_sets_completed_at_on_terminal(self, client, workspace):
        """When the scheduler reports a terminal state, completed_at is set."""
        store = RemoteJobStore(workspace=workspace)
        store.add_or_update(
            _make_record(local_id="finishing", scheduler_id="888", status="RUNNING")
        )

        fake_cfg = HPCConfig(host="hpc.test", username="tester")
        fake_status = JobStatus(job_id="888", state="COMPLETED", exit_code=0)

        with patch(
            "huginn.routes.hpc._resolve_hpc_config", return_value=(fake_cfg, None)
        ), patch("huginn.routes.hpc.HPCClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.poll_status.return_value = fake_status
            mock_cls.return_value.__enter__.return_value = mock_client

            resp = client.post("/hpc/jobs/finishing/refresh", json={})

        data = resp.json()
        assert data["success"] is True
        assert data["job"]["status"] == "COMPLETED"
        assert data["job"]["completed_at"] is not None


# ════════════════════════════════════════════════════════════════════════
#  POST /hpc/submit — saves to RemoteJobStore
# ════════════════════════════════════════════════════════════════════════


class TestSubmitSavesToStore:
    def test_submit_saves_to_store(self, client, workspace):
        """Submit creates a RemoteJobRecord with credential_id and local_id."""
        fake_cfg = HPCConfig(
            host="hpc.test", username="tester", scheduler="slurm",
            remote_work_dir="/scratch/jobs",
        )

        with patch(
            "huginn.routes.hpc._resolve_hpc_config", return_value=(fake_cfg, None)
        ), patch("huginn.routes.hpc.HPCClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.generate_job_script.return_value = "#!/bin/bash\necho hi"
            mock_client.submit_job.return_value = "4242"
            mock_cls.return_value.__enter__.return_value = mock_client

            resp = client.post("/hpc/submit", json={
                "command": "python train.py",
                "credential_id": "cred-submit",
                "queue": "gpu",
            })

        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["job_id"] == "4242"
        assert "local_id" in data

        # Verify the record landed in the store
        record = RemoteJobStore(workspace=workspace).get(data["local_id"])
        assert record is not None
        assert record.scheduler_id == "4242"
        assert record.credential_id == "cred-submit"
        assert record.queue == "gpu"
        assert record.status == "PENDING"
        assert record.command == ["python", "train.py"]
        assert record.cwd == "/scratch/jobs"


# ════════════════════════════════════════════════════════════════════════
#  JobMonitor — background polling
# ════════════════════════════════════════════════════════════════════════


class TestJobMonitor:
    def test_monitor_polls_non_terminal(self, tmp_path):
        """PENDING jobs get refreshed; COMPLETED jobs are skipped."""
        from huginn.hpc.monitor import JobMonitor

        store = RemoteJobStore(workspace=tmp_path)
        store.add_or_update(_make_record(
            local_id="active", scheduler_id="100", status="PENDING",
            credential_id="cred-active", submitted_at=time.time(),
        ))
        store.add_or_update(_make_record(
            local_id="done", scheduler_id="101", status="COMPLETED",
            credential_id="cred-done", submitted_at=time.time(),
        ))

        fake_cfg = HPCConfig(host="hpc.test", username="tester")
        fake_status = JobStatus(job_id="100", state="RUNNING")

        monitor = JobMonitor(workspace=tmp_path)

        with patch(
            "huginn.routes.hpc._resolve_hpc_config",
            return_value=(fake_cfg, None),
        ), patch("huginn.hpc.connection_pool.get_pool") as mock_get_pool:
            mock_client = MagicMock()
            mock_client.poll_status.return_value = fake_status
            mock_pool = MagicMock()
            mock_pool.borrow.return_value.__enter__.return_value = mock_client
            mock_get_pool.return_value = mock_pool

            monitor._poll_cycle()

        # The PENDING job should now be RUNNING
        active = store.get("active")
        assert active.status == "RUNNING"

        # The COMPLETED job should be untouched
        done = store.get("done")
        assert done.status == "COMPLETED"

        # poll_status should have been called exactly once (for "active" only)
        mock_client.poll_status.assert_called_once_with("100")

    def test_monitor_skips_job_without_credential(self, tmp_path):
        """A job with no credential_id is silently skipped — can't reconnect."""
        from huginn.hpc.monitor import JobMonitor

        store = RemoteJobStore(workspace=tmp_path)
        store.add_or_update(_make_record(
            local_id="no-cred", scheduler_id="200", status="PENDING",
            credential_id=None, submitted_at=time.time(),
        ))

        monitor = JobMonitor(workspace=tmp_path)

        with patch("huginn.hpc.connection_pool.get_pool") as mock_get_pool:
            monitor._poll_cycle()

            # get_pool should never have been called
            mock_get_pool.assert_not_called()

    def test_monitor_start_stop(self, tmp_path):
        """The daemon thread starts and stops cleanly."""
        from huginn.hpc.monitor import JobMonitor

        monitor = JobMonitor(workspace=tmp_path)
        monitor.start()
        assert monitor._thread is not None
        assert monitor._thread.is_alive()

        monitor.stop(timeout=2.0)
        assert not monitor._thread.is_alive()
