"""Cumulative trajectory audit for multi-step tasks.

AuditLogger (audit.py) is append-only per-event jsonl — it can't tell you when
a sequence of individually-harmless steps has piled up into a paper-reproduction
deliverable. This module does exactly that: snapshot the outputs/ dir, compare
against thresholds, block when the cumulative trajectory crosses a line.

Deliberately decoupled from AuditLogger: independent state, independent API,
no shared mutation. Callers wire them together if they want both.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class CumulativeAuditConfig:
    """Thresholds for cumulative outputs.

    max_repro_files only applies when task_type="repro"; everything else is
    task-agnostic. blocked_extensions are paper-deliverable indicators —
    seeing any of them in outputs/ is itself a block, regardless of count.
    """

    max_total_files: int = 50
    max_total_bytes: int = 100 * 1024 * 1024  # 100MB
    max_repro_files: int = 20  # repro 任务特有上限
    blocked_extensions: tuple = (".tex", ".pdf", ".docx")  # 论文复现包指标


class CumulativeAuditor:
    """Threshold-based cumulative trajectory auditor.

    Usage::

        auditor = CumulativeAuditor()
        result = auditor.audit_step(Path("outputs"), task_type="repro")
        if result["blocked"]:
            raise RuntimeError(f"cumulative limit: {result['reason']}")
    """

    def __init__(self, config: CumulativeAuditConfig | None = None):
        self.config = config or CumulativeAuditConfig()
        self._snapshots: list[dict] = []

    def snapshot(self, outputs_dir: Path) -> dict:
        """Scan outputs/ and return current state.

        ponytail: 全量扫描, 不做增量 diff. 简单可靠, outputs/ 通常不大;
        真正成规模时再上 mtime/inotify 增量.
        """
        outputs_dir = Path(outputs_dir)
        if not outputs_dir.exists():
            return {
                "n_files": 0,
                "total_bytes": 0,
                "extensions": {},
                "files": [],
            }

        files: list[str] = []
        total_bytes = 0
        extensions: dict[str, int] = {}

        for p in sorted(outputs_dir.rglob("*")):
            if not p.is_file():
                continue
            rel = str(p.relative_to(outputs_dir)).replace("\\", "/")
            files.append(rel)
            try:
                size = p.stat().st_size
            except OSError:
                size = 0
            total_bytes += size
            ext = p.suffix.lower()
            extensions[ext] = extensions.get(ext, 0) + 1

        return {
            "n_files": len(files),
            "total_bytes": total_bytes,
            "extensions": extensions,
            "files": files,
        }

    def check_threshold(self, snapshot: dict, task_type: str = "default") -> tuple[bool, str]:
        """Check snapshot against thresholds. Returns (passed, reason).

        task_type="repro" uses max_repro_files; otherwise max_total_files.
        blocked_extensions trigger a block regardless of task_type.
        """
        # 论文复现包扩展名优先判定 — 出现即阻断
        hit_blocked = [
            ext for ext in self.config.blocked_extensions
            if snapshot["extensions"].get(ext, 0) > 0
        ]
        if hit_blocked:
            return False, f"blocked extensions present: {hit_blocked}"

        # 文件数上限按任务类型切换
        max_files = (
            self.config.max_repro_files
            if task_type == "repro"
            else self.config.max_total_files
        )
        if snapshot["n_files"] > max_files:
            return False, f"file count {snapshot['n_files']} > {max_files} (task_type={task_type})"

        if snapshot["total_bytes"] > self.config.max_total_bytes:
            return False, (
                f"total bytes {snapshot['total_bytes']} > "
                f"{self.config.max_total_bytes}"
            )

        return True, ""

    def audit_step(self, outputs_dir: Path, task_type: str = "default") -> dict:
        """Audit one step: snapshot + threshold check. Records into history."""
        snap = self.snapshot(outputs_dir)
        passed, reason = self.check_threshold(snap, task_type=task_type)
        record = {
            "snapshot": snap,
            "passed": passed,
            "reason": reason if not passed else None,
            "blocked": not passed,
            "task_type": task_type,
        }
        self._snapshots.append(record)
        return record

    def history(self) -> list[dict]:
        """Return all snapshot history (shallow copy so callers can't mutate)."""
        return list(self._snapshots)


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "outputs"
        d.mkdir()
        # 创建一些文件
        (d / "a.py").write_text("print('hi')")
        (d / "b.json").write_text("{}")
        auditor = CumulativeAuditor()
        result = auditor.audit_step(d, task_type="default")
        assert result["passed"], f"small outputs should pass: {result}"
        # 测试超阈值
        cfg = CumulativeAuditConfig(max_total_files=1)
        auditor2 = CumulativeAuditor(cfg)
        result2 = auditor2.audit_step(d, task_type="default")
        assert not result2["passed"], f"2 files > 1 threshold should block: {result2}"
        # 测试 repro 任务 + 论文复现包扩展名
        (d / "paper.tex").write_text("\\documentclass{article}")
        cfg3 = CumulativeAuditConfig(max_repro_files=2)
        auditor3 = CumulativeAuditor(cfg3)
        result3 = auditor3.audit_step(d, task_type="repro")
        assert not result3["passed"], f"3 files > 2 repro threshold should block: {result3}"
    print("cumulative_audit self-check OK")
