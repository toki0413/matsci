#!/usr/bin/env python3
"""Deterministic smoke test for instrument-ingest (synthetic data, no real files)."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from read_xrd import find_peaks, parse_measurement


def test_parse_and_json() -> None:
    """Two Gaussian-like peaks on a flat background parse into a stable summary."""
    x = [float(i) / 10.0 for i in range(30, 401)]  # 2theta 3.0 .. 40.0 step 0.1
    y = [0.0] * len(x)
    # two peaks ~ 20 and 25 deg
    for j in range(len(x)):
        g = 0.0
        g += 100.0 * 2.71828 ** (-0.5 * ((x[j] - 20.0) / 0.12) ** 2)
        g += 60.0 * 2.71828 ** (-0.5 * ((x[j] - 25.0) / 0.15) ** 2)
        y[j] = g
    lines = [f"{x[j]:.6f} {y[j]:.6f}" for j in range(len(x))]
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "sample.xy"
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        summary = parse_measurement(p)
        assert summary["instrument"] == "xrd"
        assert summary["points"] == len(x)
        assert summary["peak_count"] == 2
        assert summary["new_phase_hint"] is False  # <3 peaks
        positions = sorted(pe["position"] for pe in summary["peaks"])
        assert 19.9 < positions[0] < 20.1
        assert 24.9 < positions[1] < 25.1
        json.dumps(summary, ensure_ascii=False)  # must be serializable


def test_comment_skipping() -> None:
    """Comment lines and blank lines are ignored."""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "c.csv"
        p.write_text(
            "# header\n\n// more\n,,,\n1.0 10.0\n2.0 12.0\n3.0 11.0\n4.0 9.0\n",
            encoding="utf-8",
        )
        summary = parse_measurement(p)
        assert summary["points"] == 4
        assert summary["peak_count"] == 1


def test_peaks_work_on_array() -> None:
    """find_peaks returns dicts with the expected keys."""
    # two-blob shape, peaks at 5 and 15
    xs = list(range(0, 200))
    ys = []
    for i in xs:
        ys.append(20.0 * 2.71828 ** (-0.5 * ((i - 50) / 5.0) ** 2) + \
                  15.0 * 2.71828 ** (-0.5 * ((i - 150) / 6.0) ** 2))
    peaks = find_peaks([float(v) for v in xs], [float(v) for v in ys])
    assert len(peaks) >= 2
    assert all(set(pe) == {"position", "intensity", "approx_fwhm"} for pe in peaks)


if __name__ == "__main__":
    test_parse_and_json()
    test_comment_skipping()
    test_peaks_work_on_array()
    print("instrument-ingest smoke test OK")