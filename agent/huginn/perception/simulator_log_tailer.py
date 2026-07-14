"""Simulator log tailer — continuously tail simulation output logs.

Monitors VASP, LAMMPS, CP2K, and other material science simulation
output files for convergence, errors, and progress in real-time.

Usage:
    tailer = SimulatorLogTailer()
    tailer.watch("vasp.log")
    for update in tailer.updates():
        print(update.status, update.progress)
"""
from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass
class SimulationUpdate:
    """A parsed update from a simulation log."""
    source: str  # log file path
    simulator: str  # vasp, lammps, cp2k, etc.
    status: str  # running, converged, failed, error
    progress: float | None = None  # percent if available
    iteration: int | None = None  # SCF/MD step number
    energy: float | None = None  # last energy value
    message: str = ""  # last significant line


class SimulatorLogTailer:
    """Tail simulation logs and extract structured state."""

    PATTERNS = {
        "vasp": {
            "converged": re.compile(r"(?i)reached required accuracy\s*-\s*stopping structural energy minimisation"),
            "scf_iter": re.compile(r"N\s+([\d.]+)\s+([\d.]+)"),  # N, energy
            "step": re.compile(r"LOOP\+:\s*cpu time\s+[\d.]+:\s*real time\s+([\d.]+)"),
            "error": re.compile(r"(?i)(error|fatal|abort| segmentation fault)"),
        },
        "lammps": {
            "converged": re.compile(r"(?i)All done"),
            "step": re.compile(r"Step\s+([\d]+)"),
            "thermo": re.compile(r"([\d]+)\s+([\-\d.eE]+)"),  # step, energy
            "error": re.compile(r"(?i)(error|cannot|invalid|segmentation)"),
        },
        "cp2k": {
            "converged": re.compile(r"(?i)SCF run converged"),
            "scf_iter": re.compile(r"Iteration\s+(\d+)\s+.*energy\s*=\s*([\-\d.eE]+)"),
            "error": re.compile(r"(?i)(error|fatal|abort|failed)"),
        },
        "generic": {
            "converged": re.compile(r"(?i)(converged|complete|finished|success)"),
            "progress": re.compile(r"(\d{1,3})\s*%"),
            "error": re.compile(r"(?i)(error|fatal|abort|failed|exception)"),
        },
    }

    def __init__(self) -> None:
        self._watches: dict[str, dict] = {}  # path -> {file, position, sim_type}
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

    def watch(self, path: str | Path, simulator_type: str | None = None) -> None:
        """Start watching a log file."""
        path = str(Path(path).resolve())
        sim_type = simulator_type or self._detect_type(path)
        try:
            f = open(path, "r", encoding="utf-8", errors="replace")
            f.seek(0, 2)  # Seek to end
        except Exception:
            return
        with self._lock:
            self._watches[path] = {
                "file": f,
                "position": f.tell(),
                "type": sim_type,
            }

    def unwatch(self, path: str | Path) -> None:
        """Stop watching a log file."""
        path = str(Path(path).resolve())
        with self._lock:
            if path in self._watches:
                self._watches[path]["file"].close()
                del self._watches[path]

    def start(self) -> None:
        """Start background polling thread."""
        self._running = True
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop background polling."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        # 关闭所有 watch 的文件句柄, 否则 fd 泄漏到进程退出
        with self._lock:
            for info in self._watches.values():
                try:
                    info["file"].close()
                except Exception:
                    pass
            self._watches.clear()

    def updates(self) -> Iterator[SimulationUpdate]:
        """Yield all new updates from watched files (non-blocking)."""
        with self._lock:
            watches = list(self._watches.items())
        for path, info in watches:
            f = info["file"]
            f.seek(info["position"])
            new_lines = f.readlines()
            if new_lines:
                info["position"] = f.tell()
                for line in new_lines:
                    update = self._parse_line(info["type"], path, line)
                    if update:
                        yield update

    def _poll(self) -> None:
        """Background polling loop."""
        while self._running:
            for _ in self.updates():
                pass  # Just update positions
            time.sleep(1.0)

    def _detect_type(self, path: str) -> str:
        """Detect simulator type from filename."""
        lower = path.lower()
        if "vasp" in lower or "outcar" in lower:
            return "vasp"
        if "lammps" in lower or "log.lammps" in lower:
            return "lammps"
        if "cp2k" in lower:
            return "cp2k"
        return "generic"

    def _parse_line(self, sim_type: str, path: str, line: str) -> SimulationUpdate | None:
        """Parse a single log line into an update."""
        patterns = self.PATTERNS.get(sim_type, self.PATTERNS["generic"])

        # Check for errors first
        if patterns.get("error") and patterns["error"].search(line):
            return SimulationUpdate(
                source=path, simulator=sim_type, status="error",
                message=line.strip(),
            )

        # Check for convergence
        if patterns.get("converged") and patterns["converged"].search(line):
            return SimulationUpdate(
                source=path, simulator=sim_type, status="converged",
                message=line.strip(),
            )

        # Extract progress/iteration
        progress = None
        iteration = None
        energy = None

        if patterns.get("progress"):
            m = patterns["progress"].search(line)
            if m:
                progress = float(m.group(1))

        if patterns.get("scf_iter"):
            m = patterns["scf_iter"].search(line)
            if m:
                iteration = int(float(m.group(1)))
                if len(m.groups()) > 1:
                    try:
                        energy = float(m.group(2))
                    except ValueError:
                        pass

        if patterns.get("step"):
            m = patterns["step"].search(line)
            if m:
                iteration = int(m.group(1))

        if progress is not None or iteration is not None:
            return SimulationUpdate(
                source=path, simulator=sim_type, status="running",
                progress=progress, iteration=iteration, energy=energy,
                message=line.strip(),
            )

        return None
