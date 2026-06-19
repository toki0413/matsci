"""Tests for RemoteJobStore persistence."""

from __future__ import annotations

from pathlib import Path

from huginn.execution.remote_job_store import RemoteJobRecord, RemoteJobStore


class TestRemoteJobStore:
    def test_add_and_get(self, tmp_path: Path) -> None:
        store = RemoteJobStore(path=tmp_path / "jobs.json")
        record = RemoteJobRecord(
            local_id="abc",
            scheduler_id="123",
            command=["vasp_std"],
            cwd=str(tmp_path),
            queue="gpu",
            status="PENDING",
            submitted_at=1.0,
        )
        store.add_or_update(record)

        loaded = store.get("abc")
        assert loaded is not None
        assert loaded.scheduler_id == "123"
        assert loaded.queue == "gpu"

    def test_persists_across_instances(self, tmp_path: Path) -> None:
        path = tmp_path / "jobs.json"
        store1 = RemoteJobStore(path=path)
        store1.add_or_update(
            RemoteJobRecord(
                local_id="j1",
                scheduler_id="s1",
                command=["a"],
                cwd=str(tmp_path),
                submitted_at=2.0,
            )
        )

        store2 = RemoteJobStore(path=path)
        jobs = store2.list_jobs()
        assert [j.local_id for j in jobs] == ["j1"]

    def test_list_orders_newest_first(self, tmp_path: Path) -> None:
        store = RemoteJobStore(path=tmp_path / "jobs.json")
        for i, ts in enumerate([1.0, 3.0, 2.0]):
            store.add_or_update(
                RemoteJobRecord(
                    local_id=f"j{i}",
                    scheduler_id=f"s{i}",
                    command=["cmd"],
                    cwd=str(tmp_path),
                    submitted_at=ts,
                )
            )
        assert [j.local_id for j in store.list_jobs()] == ["j1", "j2", "j0"]

    def test_update_overwrites(self, tmp_path: Path) -> None:
        store = RemoteJobStore(path=tmp_path / "jobs.json")
        store.add_or_update(
            RemoteJobRecord(
                local_id="x",
                scheduler_id="s",
                command=["cmd"],
                cwd=str(tmp_path),
                submitted_at=1.0,
                status="PENDING",
            )
        )
        store.add_or_update(
            RemoteJobRecord(
                local_id="x",
                scheduler_id="s",
                command=["cmd"],
                cwd=str(tmp_path),
                submitted_at=1.0,
                status="COMPLETED",
            )
        )
        assert store.get("x").status == "COMPLETED"
        assert len(store.list_jobs()) == 1

    def test_remove(self, tmp_path: Path) -> None:
        store = RemoteJobStore(path=tmp_path / "jobs.json")
        store.add_or_update(
            RemoteJobRecord(
                local_id="r1",
                scheduler_id="s",
                command=["cmd"],
                cwd=str(tmp_path),
                submitted_at=1.0,
            )
        )
        assert store.remove("r1") is True
        assert store.get("r1") is None
        assert store.remove("missing") is False

    def test_auto_cap_keeps_non_terminal(self, tmp_path: Path) -> None:
        store = RemoteJobStore(path=tmp_path / "jobs.json", max_records=3)
        for i in range(3):
            store.add_or_update(
                RemoteJobRecord(
                    local_id=f"old{i}",
                    scheduler_id="s",
                    command=["cmd"],
                    cwd=str(tmp_path),
                    status="COMPLETED",
                    submitted_at=float(i),
                )
            )
        store.add_or_update(
            RemoteJobRecord(
                local_id="running",
                scheduler_id="s",
                command=["cmd"],
                cwd=str(tmp_path),
                status="RUNNING",
                submitted_at=10.0,
            )
        )
        store.add_or_update(
            RemoteJobRecord(
                local_id="newest",
                scheduler_id="s",
                command=["cmd"],
                cwd=str(tmp_path),
                status="COMPLETED",
                submitted_at=20.0,
            )
        )
        jobs = store.list_jobs()
        ids = {j.local_id for j in jobs}
        assert "running" in ids
        assert "newest" in ids
        assert len(jobs) <= 3

    def test_prune(self, tmp_path: Path) -> None:
        store = RemoteJobStore(path=tmp_path / "jobs.json")
        for i in range(5):
            store.add_or_update(
                RemoteJobRecord(
                    local_id=f"j{i}",
                    scheduler_id="s",
                    command=["cmd"],
                    cwd=str(tmp_path),
                    status="COMPLETED",
                    submitted_at=float(i),
                )
            )
        removed = store.prune(max_records=2)
        assert removed == 3
        assert len(store.list_jobs()) == 2
