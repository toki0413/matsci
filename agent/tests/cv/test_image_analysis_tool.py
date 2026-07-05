"""End-to-end tests for ImageAnalysisTool across all 8 actions.

Each test feeds a tiny synthetic image (built by conftest fixtures) straight
into the tool instance - no HTTP, no real microscopy data, no GPU. Actions that
lean on scipy skip themselves when scipy isn't around.
"""

from __future__ import annotations

import asyncio

import pytest

from huginn.tools.image_analysis.tool import ImageAnalysisTool


def _run(tool: ImageAnalysisTool, args: dict, context) -> object:
    """Drive the async tool.call() to completion from a sync test."""
    return asyncio.run(tool.call(args, context))


# ── SEM morphology ──────────────────────────────────────────────────────────


def test_sem_analysis(cv_tool, tool_context, generate_synthetic_sem_image):
    res = _run(cv_tool, {
        "image_path": generate_synthetic_sem_image,
        "action": "sem_analysis",
        "parameters": {"pixel_size_nm": 1.0},
    }, tool_context)

    assert res.success, res.error
    m = res.data["measurements"]
    # contrast / roughness stats are the core SEM outputs
    for key in ("contrast_mean", "contrast_std", "surface_roughness_rms", "edge_density"):
        assert key in m, f"missing {key}"
        assert isinstance(m[key], (int, float))
    assert m["image_shape"] == [128, 128]


# ── TEM lattice (FFT + d-spacing) ───────────────────────────────────────────


def test_tem_lattice(cv_tool, tool_context, generate_synthetic_tem_image):
    pytest.importorskip("scipy.fft")
    res = _run(cv_tool, {
        "image_path": generate_synthetic_tem_image,
        "action": "tem_lattice",
        "parameters": {"pixel_size_nm": 1.0},
    }, tool_context)

    assert res.success, res.error
    m = res.data["measurements"]
    # the synthetic sine lattice should land at least one FFT peak
    assert m["n_peaks"] >= 1
    d_spacings = m["d_spacings"]
    assert d_spacings, "expected at least one d-spacing entry"
    assert d_spacings[0]["d_nm"] > 0.0


# ── EDS element mapping ─────────────────────────────────────────────────────


def test_eds_mapping(cv_tool, tool_context, generate_synthetic_eds_image):
    colors = {"Fe": "red", "Cu": "green", "Co": "blue", "Au": "yellow"}
    res = _run(cv_tool, {
        "image_path": generate_synthetic_eds_image,
        "action": "eds_mapping",
        "parameters": {"element_colors": colors, "color_tolerance": 30.0},
    }, tool_context)

    assert res.success, res.error
    elements = res.data["measurements"]["elements"]
    assert set(elements) == set(colors)
    for name, stats in elements.items():
        assert stats["coverage_fraction"] > 0.0, f"{name} has zero coverage"


# ── particle size statistics ────────────────────────────────────────────────


def test_particle_stats(cv_tool, tool_context, generate_synthetic_particle_image):
    pytest.importorskip("scipy.ndimage")
    res = _run(cv_tool, {
        "image_path": generate_synthetic_particle_image,
        "action": "particle_stats",
        "parameters": {"pixel_size_nm": 1.0, "min_area_px": 5},
    }, tool_context)

    assert res.success, res.error
    m = res.data["measurements"]
    assert m["n_particles"] >= 1
    # D10 <= D50 <= D90 by definition
    assert m["d10_nm"] <= m["d50_nm"] <= m["d90_nm"]


# ── defect detection ────────────────────────────────────────────────────────


def test_defect_detect(cv_tool, tool_context, generate_synthetic_defect_image):
    pytest.importorskip("scipy.ndimage")
    res = _run(cv_tool, {
        "image_path": generate_synthetic_defect_image,
        "action": "defect_detect",
        "parameters": {
            "defect_type": "pore",
            "sensitivity": 0.5,
            "min_defect_area_px": 5,
        },
    }, tool_context)

    assert res.success, res.error
    m = res.data["measurements"]
    assert m["defect_type"] == "pore"
    assert m["n_defects"] >= 1, "expected at least one detected pore"
    assert len(m["defects"]) == m["n_defects"]


# ── plot data extraction ───────────────────────────────────────────────────


def test_plot_extract(cv_tool, tool_context, generate_synthetic_chart_image):
    chart = generate_synthetic_chart_image
    res = _run(cv_tool, {
        "image_path": chart["path"],
        "action": "plot_extract",
        "parameters": chart["params"],
    }, tool_context)

    assert res.success, res.error
    m = res.data["measurements"]
    assert m["n_points"] > 0, "no data points extracted from the chart"
    assert len(res.data["points"]) == m["n_points"]

    xs = [p[0] for p in res.data["points"]]
    params = chart["params"]
    assert min(xs) >= params["x_min"] - 1e-6
    assert max(xs) <= params["x_max"] + 1e-6


# ── phase-field post-processing ────────────────────────────────────────────


def test_phase_field(cv_tool, tool_context, generate_synthetic_sem_image):
    res = _run(cv_tool, {
        "image_path": generate_synthetic_sem_image,
        "action": "phase_field",
        "parameters": {"n_phases": 3, "pixel_size_nm": 1.0},
    }, tool_context)

    assert res.success, res.error
    m = res.data["measurements"]
    assert m["n_phases"] == 3
    vol = m["volume_fractions"]
    assert len(vol) == 3
    assert abs(sum(vol.values()) - 1.0) < 1e-6
    assert 0.0 <= m["interface_pixel_fraction"] <= 1.0


# ── chart-to-table (DePlot) graceful degradation ───────────────────────────


def test_deplot_chart_unavailable(cv_tool, tool_context, generate_synthetic_chart_image):
    res = _run(cv_tool, {
        "image_path": generate_synthetic_chart_image["path"],
        "action": "deplot_chart",
        "parameters": {},
    }, tool_context)

    assert res.success, res.error
    d = res.data
    # No torch / transformers here -> the action must degrade, not crash.
    if d.get("available") is True:
        pytest.skip("DePlot backend present; unavailable-path test not applicable")
    assert d["available"] is False
    assert d["missing_dependencies"]
