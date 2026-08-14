"""CumulativeAuditor 全分支测试 — 快照、阈值检查、历史记录、OSError 降级."""

from __future__ import annotations

from pathlib import Path

from huginn.security.cumulative_audit import (
    CumulativeAuditConfig,
    CumulativeAuditor,
)


def _write(tmp_path: Path, name: str, size: int = 4) -> Path:
    p = tmp_path / name
    p.write_bytes(b"x" * size)
    return p


def _snap(n=0, bytes_=0, extensions=None, files=None):
    return {
        "n_files": n,
        "total_bytes": bytes_,
        "extensions": extensions or {},
        "files": files or [],
    }


# ── snapshot ─────────────────────────────────────────────────────────────

def test_snapshot_empty_dir(tmp_path):
    d = tmp_path / "outputs"
    d.mkdir()
    snap = CumulativeAuditor().snapshot(d)
    assert snap["n_files"] == 0
    assert snap["total_bytes"] == 0
    assert snap["extensions"] == {}
    assert snap["files"] == []


def test_snapshot_missing_dir(tmp_path):
    snap = CumulativeAuditor().snapshot(tmp_path / "nope")
    assert snap["n_files"] == 0


def test_snapshot_counts_files_and_bytes(tmp_path):
    _write(tmp_path, "a.py", 10)
    _write(tmp_path, "b.json", 20)
    snap = CumulativeAuditor().snapshot(tmp_path)
    assert snap["n_files"] == 2
    assert snap["total_bytes"] == 30
    assert snap["extensions"] == {".py": 1, ".json": 1}


def test_snapshot_recursive_rglob(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    _write(sub, "c.txt", 5)
    snap = CumulativeAuditor().snapshot(tmp_path)
    assert snap["n_files"] == 1
    assert "sub/c.txt" in snap["files"]


def test_snapshot_skips_dirs(tmp_path):
    (tmp_path / "adir").mkdir()
    snap = CumulativeAuditor().snapshot(tmp_path)
    assert snap["n_files"] == 0


def test_snapshot_oserror_stat_returns_zero(tmp_path, monkeypatch):
    from pathlib import Path

    target = _write(tmp_path, "a.py", 10)
    real_stat = Path.stat

    # 只对目标文件抛 OSError, 不影响 is_file()/rglob/pytest tmpdir 清理
    def selective_stat(self, *args, **kwargs):
        if self == target:
            raise OSError("stat failed")
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", selective_stat)
    snap = CumulativeAuditor().snapshot(tmp_path)
    assert snap["total_bytes"] == 0
    assert snap["n_files"] == 1


# ── check_threshold ──────────────────────────────────────────────────────

def test_check_passes_small_snapshot():
    ok, reason = CumulativeAuditor().check_threshold(_snap(n=1, bytes_=10))
    assert ok is True
    assert reason == ""


def test_check_blocked_extension_wins():
    a = CumulativeAuditor()
    ok, reason = a.check_threshold(_snap(n=1, extensions={".tex": 1}))
    assert ok is False
    assert "blocked extensions present" in reason


def test_check_file_count_default():
    a = CumulativeAuditor(CumulativeAuditConfig(max_total_files=3))
    ok, reason = a.check_threshold(_snap(n=4))
    assert ok is False
    assert "file count 4 > 3" in reason
    assert "default" in reason


def test_check_file_count_repro_uses_repro_limit():
    a = CumulativeAuditor(CumulativeAuditConfig(max_repro_files=2, max_total_files=50))
    ok, reason = a.check_threshold(_snap(n=3), task_type="repro")
    assert ok is False
    assert "repro" in reason


def test_check_repro_at_limit_passes():
    a = CumulativeAuditor(CumulativeAuditConfig(max_repro_files=2))
    ok, _ = a.check_threshold(_snap(n=2), task_type="repro")
    assert ok is True


def test_check_byte_limit():
    a = CumulativeAuditor(CumulativeAuditConfig(max_total_bytes=100))
    ok, reason = a.check_threshold(_snap(n=1, bytes_=101))
    assert ok is False
    assert "total bytes 101 > 100" in reason


# ── audit_step / history ─────────────────────────────────────────────────

def test_audit_step_records_history(tmp_path):
    _write(tmp_path, "a.py")
    a = CumulativeAuditor()
    rec = a.audit_step(tmp_path)
    assert rec["passed"] is True
    assert rec["blocked"] is False
    assert rec["reason"] is None
    assert rec["task_type"] == "default"
    assert rec["snapshot"]["n_files"] == 1
    assert len(a.history()) == 1


def test_audit_step_blocked_records_reason(tmp_path):
    _write(tmp_path, "paper.tex")
    a = CumulativeAuditor()
    rec = a.audit_step(tmp_path)
    assert rec["passed"] is False
    assert rec["blocked"] is True
    assert "blocked extensions" in rec["reason"]


def test_history_is_shallow_copy(tmp_path):
    _write(tmp_path, "a.py")
    a = CumulativeAuditor()
    a.audit_step(tmp_path)
    h = a.history()
    h.clear()
    assert len(a.history()) == 1
