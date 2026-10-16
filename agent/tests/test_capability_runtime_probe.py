"""L0 依赖探针单测 —— 全部 weightless: 只用合成临时文件, 不碰重依赖."""

from __future__ import annotations

import pathlib

from huginn.capabilities import runtime_probe as rp


def test_classify_precedence():
    """分类优先级: 缺依赖 > 不可实例化 > 隐藏 > 可运行."""
    assert rp.classify(missing=["numpy"]) == rp.STATUS_MISSING
    assert rp.classify(missing=[], constructed=False) == rp.STATUS_NOT_INSTANTIABLE
    assert rp.classify(missing=[], visible=False) == rp.STATUS_HIDDEN
    assert rp.classify(missing=[], visible=True) == rp.STATUS_RUNNABLE
    assert rp.classify(missing=[], imported=False) == rp.STATUS_MISSING


def test_probe_detects_missing_dep_on_synthetic_module(tmp_path: pathlib.Path):
    mod = tmp_path / "fake_heavy.py"
    mod.write_text("import definitely_absent_pkg_xyz\n", encoding="utf-8")
    results = rp.probe_runtime([rp.ToolSpec(module="fake_heavy", cls="FakeHeavy", path=mod)])
    assert results[0].status == rp.STATUS_MISSING
    assert results[0].missing_deps == ["definitely_absent_pkg_xyz"]
    assert results[0].install_hint == "pip install definitely_absent_pkg_xyz"


def test_probe_marks_stdlib_only_module_runnable(tmp_path: pathlib.Path):
    mod = tmp_path / "fake_ok.py"
    mod.write_text("import json\nimport os\n", encoding="utf-8")
    results = rp.probe_runtime([rp.ToolSpec(module="fake_ok", cls="FakeOk", path=mod)])
    assert results[0].status == rp.STATUS_RUNNABLE
    assert results[0].missing_deps == []


def test_probe_never_raises_on_missing_path():
    """路径不存在时不得抛 —— 收敛为 missing_deps(无法验证)."""
    results = rp.probe_runtime(
        [rp.ToolSpec(module="nope", cls="Nope", path=pathlib.Path("/no/such/file.py"))]
    )
    assert len(results) == 1
    assert results[0].status == rp.STATUS_MISSING


def test_probe_orders_deterministically():
    """相同输入 → 输出逐字一致 (确定性)."""
    a = rp.probe_runtime([rp.ToolSpec(module="m", cls="B"), rp.ToolSpec(module="m", cls="A")])
    b = rp.probe_runtime([rp.ToolSpec(module="m", cls="A"), rp.ToolSpec(module="m", cls="B")])
    assert [r.name for r in a] == [r.name for r in b] == ["A", "B"]


def test_summary_line_all_ok():
    rep = rp.build_report([rp.ProbeResult(name="A", kind="tool", status=rp.STATUS_RUNNABLE)])
    assert rp.summary_line(rep) == rp.ALL_OK_LINE


def test_summary_line_reports_degraded_with_hint():
    rep = rp.build_report(
        [
            rp.ProbeResult(
                name="RDKitTool",
                kind="tool",
                status=rp.STATUS_MISSING,
                missing_deps=["rdkit"],
                install_hint="pip install rdkit",
            )
        ]
    )
    line = rp.summary_line(rep)
    assert "degraded: 1" in line
    assert "rdkit" in line
    assert "pip install rdkit" in line
