"""Tests for the lightweight file-based scheduler."""

from __future__ import annotations

from datetime import datetime

import pytest

from huginn.scheduler import ScheduleManager

croniter = pytest.importorskip("croniter", reason="croniter not installed")


class TestScheduleManager:
    def test_add_and_list(self, tmp_path):
        manager = ScheduleManager(tmp_path)
        job_id = manager.add("0 9 * * *", "echo hello")
        jobs = manager.list()
        assert len(jobs) == 1
        assert jobs[0].id == job_id
        assert jobs[0].command == "echo hello"

    def test_invalid_cron(self, tmp_path):
        manager = ScheduleManager(tmp_path)
        with pytest.raises(ValueError):
            manager.add("invalid", "echo hello")

    def test_remove(self, tmp_path):
        manager = ScheduleManager(tmp_path)
        job_id = manager.add("0 9 * * *", "echo hello")
        assert manager.remove(job_id) is True
        assert manager.list() == []
        assert manager.remove("missing") is False

    def test_enable_disable(self, tmp_path):
        manager = ScheduleManager(tmp_path)
        job_id = manager.add("0 9 * * *", "echo hello")
        assert manager.enable(job_id, False) is True
        assert manager.list()[0].enabled is False

    def test_due_jobs(self, tmp_path):
        manager = ScheduleManager(tmp_path)
        job_id = manager.add("* * * * *", "echo hello")
        due = manager.due_jobs(now=datetime(2030, 1, 1, 12, 0, 0))
        assert len(due) == 1
        assert due[0].id == job_id

    def test_disabled_not_due(self, tmp_path):
        manager = ScheduleManager(tmp_path)
        job_id = manager.add("* * * * *", "echo hello")
        manager.enable(job_id, False)
        due = manager.due_jobs(now=datetime(2030, 1, 1, 12, 0, 0))
        assert due == []

    def test_run_due_executes_command(self, tmp_path):
        manager = ScheduleManager(tmp_path)
        manager.add("* * * * *", "echo hello")
        results = manager.run_due(now=datetime(2030, 1, 1, 12, 0, 0))
        assert len(results) == 1
        assert results[0]["success"] is True
        assert "hello" in results[0]["stdout"]
