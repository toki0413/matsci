"""执行覆盖度量: 默认按模块采样调用并落 ``trace.json``, ``HUGINN_TRACE=0`` 关闭.

「设计覆盖率」不等于「执行覆盖率」. 静态 census 标出 KEEP 模块, 但一次真实
运行到底跑了哪些、哪些 KEEP 模块零执行, 只有实际采样才能回答 —— 不产生
trace 时不得声称已度量.

实现: ``sys.setprofile`` 只在 ``call`` 事件上按**文件**计数 (不记录参数, 开销
可控), 再把文件归属到 ``huginn/`` 下的顶层模块 (与 census 的 ``module`` 列对齐).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from types import FrameType

from huginn.env_access import env_bool

TRACE_FILENAME = "trace.json"
CENSUS_FILENAME = "huginn_module_census.csv"
HUGINN_ROOT = Path(__file__).resolve().parent


def trace_enabled() -> bool:
    """开关: 默认开; ``HUGINN_TRACE=0`` 显式关闭 (与 env_schema 的 true 一致)."""
    return env_bool("HUGINN_TRACE", True)


def module_of(filename: str, root: Path = HUGINN_ROOT) -> str | None:
    """把帧的文件路径归到顶层模块名; 不在 ``huginn/`` 下返回 None."""
    try:
        rel = Path(filename).resolve().relative_to(root.resolve())
    except (ValueError, OSError):
        return None
    parts = rel.parts
    if not parts:
        return None
    if len(parts) == 1:
        stem = Path(parts[0]).stem
        return stem if stem not in ("__init__",) else None
    return parts[0]


class TraceSampler:
    """``sys.setprofile`` 采样器; 作为上下文管理器或显式 start/stop."""

    def __init__(self, root: Path = HUGINN_ROOT) -> None:
        self.root = root
        self.counts: dict[str, int] = {}
        self._prev_profile = None
        self._active = False
        # 文件路径 → 模块名 的缓存: profile 钩子每次 call 都跑, 不能反复 resolve.
        self._module_cache: dict[str, str | None] = {}
        # 重入闸: 钩子内部自己的调用也会触发 profile, 不拦住会自我递归.
        self._busy = False

    def _module(self, filename: str) -> str | None:
        if filename not in self._module_cache:
            self._module_cache[filename] = module_of(filename, self.root)
        return self._module_cache[filename]

    def _profile(self, frame: FrameType, event: str, arg: object) -> None:
        if event != "call" or self._busy:
            return
        self._busy = True
        try:
            key = self._module(frame.f_code.co_filename)
            if key is not None:
                self.counts[key] = self.counts.get(key, 0) + 1
        finally:
            self._busy = False

    def start(self) -> None:
        if self._active:
            return
        self._prev_profile = sys.getprofile()
        sys.setprofile(self._profile)
        self._active = True

    def stop(self) -> None:
        if not self._active:
            return
        sys.setprofile(self._prev_profile)
        self._active = False

    def __enter__(self) -> TraceSampler:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def snapshot(self, census_path: Path | None = None) -> dict:
        payload: dict = {
            "counts": dict(sorted(self.counts.items())),
            "total_calls": sum(self.counts.values()),
            "sampled_by": "sys.setprofile",
            "huginn_root": str(self.root),
        }
        if census_path is not None:
            rows = load_census(census_path)
            keep_zero = keep_but_zero(self.counts, rows)
            payload["census"] = {
                "path": str(census_path),
                "keep_total": len(keep_modules(rows)),
                "keep_zero": keep_zero,
            }
        return payload

    def write(self, workspace: Path | str, census_path: Path | None = None) -> Path:
        out = Path(workspace) / TRACE_FILENAME
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(self.snapshot(census_path), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return out


def load_census(path: Path | str) -> list[dict[str, str]]:
    """读 census CSV (module / verdict 两列是必须的)."""
    with Path(path).open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def keep_modules(rows: list[dict[str, str]]) -> set[str]:
    """census 中 verdict 以 ``KEEP`` 开头的模块集合."""
    return {
        str(row.get("module", "")).strip()
        for row in rows
        if str(row.get("verdict", "")).strip().upper().startswith("KEEP")
    }


def keep_but_zero(counts: dict[str, int], rows: list[dict[str, str]]) -> list[str]:
    """「KEEP 但零执行」模块清单 (可被逐条审计)."""
    return sorted(m for m in keep_modules(rows) if counts.get(m, 0) == 0)


def default_census_path() -> Path | None:
    """默认 census 位置: 仓内 research/ 或相邻工作区 research/."""
    candidates = [
        HUGINN_ROOT.parent / "research" / CENSUS_FILENAME,
        HUGINN_ROOT.parent.parent / "research" / CENSUS_FILENAME,
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None
