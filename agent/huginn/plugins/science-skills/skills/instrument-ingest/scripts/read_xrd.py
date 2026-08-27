#!/usr/bin/env python3
"""Parse instrument export files (XRD/EC) into a structured measurement summary.

File-exchange semi-loop: Huginn has no live OPC-UA/MQTT driver, so the bridge
between "instrument" and "computation" is a file. This skill reads whatever the
instrument exports (.xy / .csv / bare 2-column text), extracts peaks, prepares
a compact JSON summary, and flags whether a "new phase" family may be present
for the downstream materials_database cross-check. Pure numpy — no heavy deps.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np


# Supported instrument export formats and how to read them.
# Each entry: (regex for header line) -> column meaning.
# We stay deliberately narrow: a delimiter-separated numeric table where one
# column is angle/energy and another is intensity/counts.
def _looks_like_two_col(raw: str) -> bool:
    """Crude sniff: at least one line of two or more whitespace/comma numbers."""
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "//", ";")):
            continue
        tokens = re.split(r"[\s,;]+", line)
        if len(tokens) >= 2 and all(_is_num(t) for t in tokens[:2]):
            return True
    return False


def _is_num(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def _read_table(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a 2-column table, skipping comment lines. Returns (x, y)."""
    xs: list[float] = []
    ys: list[float] = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith(("#", "//", ";")):
                continue
            tokens = re.split(r"[\s,;]+", line)
            if len(tokens) < 2 or not _is_num(tokens[0]) or not _is_num(tokens[1]):
                continue
            xs.append(float(tokens[0]))
            ys.append(float(tokens[1]))
    if len(xs) < 3:
        raise ValueError(f"{path.name}: 数据点不足(<3)，无法可靠提峰。")
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def find_peaks(x: np.ndarray, y: np.ndarray, *, prominence: float | None = None) -> list[dict[str, Any]]:
    """Simple peak detection: local maxima above a relative height threshold.

    Avoids scipy dependency; uses a moving-window local-max scan. Good enough
    to flag candidate reflections, not a full Rietveld.
    """
    if prominence is None:
        prominence = 0.08 * float(np.max(y)) if float(np.max(y)) > 0 else 1.0
    peaks: list[dict[str, Any]] = []
    half = max(1, len(y) // 20)
    for i in range(half, len(y) - half):
        window = y[i - half : i + half + 1]
        if y[i] == float(np.max(window)) and y[i] >= prominence and y[i] > y[i - 1]:
            peaks.append(
                {
                    "position": float(x[i]),
                    "intensity": float(y[i]),
                    # rough FWHM on the raw points around the peak
                    "approx_fwhm": _approx_fwhm(x, y, i),
                }
            )
    # free up adjacent duplicates: keep the highest within neighboring indices
    dedup: list[dict[str, Any]] = []
    for p in peaks:
        if dedup and abs(p["position"] - dedup[-1]["position"]) < (x[1] - x[0]) * 2:
            if p["intensity"] > dedup[-1]["intensity"]:
                dedup[-1] = p
        else:
            dedup.append(p)
    return dedup


def _approx_fwhm(x: np.ndarray, y: np.ndarray, i: int) -> float:
    half_max = y[i] / 2.0
    lo = i
    while lo > 0 and y[lo] > half_max:
        lo -= 1
    hi = i
    while hi < len(y) - 1 and y[hi] > half_max:
        hi += 1
    return float(x[hi] - x[lo])


def parse_measurement(path: Path) -> dict[str, Any]:
    """Parse one instrument file into a species-summary dict."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    if not _looks_like_two_col(raw):
        raise ValueError(f"{path.name}: 无法识别为两列数值表(角度/强度)。")
    x, y = _read_table(path)
    peaks = find_peaks(x, y)
    ymax = float(np.max(y)) if len(y) else 0.0
    return {
        "source": "instrument_export",
        "file": path.name,
        "instrument": "xrd",
        "points": int(len(x)),
        "x_min": float(np.min(x)),
        "x_max": float(np.max(x)),
        "intensity_max": ymax,
        "peak_count": len(peaks),
        "peaks": peaks,
        # heuristic: "可能含新相" 的提示, 供下游 materials_database 交叉验证.
        "new_phase_hint": len(peaks) >= 3,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Parse an XRD/EC instrument export into a structured summary JSON."
    )
    ap.add_argument("--query", help="仪器导出文件路径 (.xy / .csv / 两列 txt)")
    ap.add_argument("--output", help="写入 JSON 的路径; 缺省打到 stdout")
    args = ap.parse_args(argv)
    if not args.query:
        print("ERROR: 需要提供 --query 文件路径", file=sys.stderr)
        return 2
    try:
        summary = parse_measurement(Path(args.query))
    except (ValueError, FileNotFoundError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    payload = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
