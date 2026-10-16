"""L1 端到端 smoke 单测 —— weightless: 用注入的假 spec, 不依赖 numpy/rdkit."""

from __future__ import annotations

from huginn.capabilities import smoke_specs as ss


def test_run_smoke_passes():
    specs = (ss.SmokeSpec(name="ok", capability="m", runner=lambda: 42, expect=lambda v: v == 42),)
    res = ss.run_smoke(specs)
    assert [r.status for r in res] == ["passed"]


def test_run_smoke_fails_on_exception():
    def boom():
        raise ValueError("x")

    specs = (ss.SmokeSpec(name="boom", capability="m", runner=boom),)
    res = ss.run_smoke(specs)
    assert res[0].status == "failed"
    assert "ValueError" in res[0].detail


def test_run_smoke_fails_on_expect_reject():
    specs = (ss.SmokeSpec(name="bad", capability="m", runner=lambda: 1, expect=lambda v: v == 999),)
    res = ss.run_smoke(specs)
    assert res[0].status == "failed"


def test_run_smoke_skips_when_requires_missing():
    specs = (
        ss.SmokeSpec(
            name="need",
            capability="m",
            requires=("definitely_absent_pkg_xyz",),
            runner=lambda: 1,
        ),
    )
    res = ss.run_smoke(specs)
    assert res[0].status == "skipped"
    assert "definitely_absent_pkg_xyz" in res[0].detail


def test_smoke_names_tuple_matches_specs():
    """台账探针读 SMOKE_SPEC_NAMES; 必须与 SMOKE_SPECS 一致 (单一事实来源)."""
    assert tuple(s.name for s in ss.SMOKE_SPECS) == ss.SMOKE_SPEC_NAMES


def test_smoke_results_ordered_deterministically():
    specs = (
        ss.SmokeSpec(name="z", capability="m", runner=lambda: 1),
        ss.SmokeSpec(name="a", capability="m", runner=lambda: 1),
    )
    assert [r.name for r in ss.run_smoke(specs)] == ["a", "z"]
