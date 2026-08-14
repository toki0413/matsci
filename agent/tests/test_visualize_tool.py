"""Tests for the visualize tool."""

from __future__ import annotations

import asyncio
import json
import sys
import types
from pathlib import Path

import pytest

from huginn.core_types import ToolContext
from huginn.tools.registry import ToolRegistry
from huginn.tools.visualize_tool import VisualizeTool, VisualizeToolInput


@pytest.fixture(autouse=True)
def _isolate_registry():
    """隔离 ToolRegistry: 测试自用工具不影响全局注册表."""
    before = ToolRegistry.snapshot()
    yield
    ToolRegistry.restore(before)


class TestVisualizeTool:
    def test_benchmark_plot(self, tmp_path):
        tool = VisualizeTool()
        report_path = tmp_path / "benchmark.json"
        report_path.write_text(
            json.dumps(
                {
                    "results": [
                        {
                            "task_id": "t1",
                            "category": "dft",
                            "passed": True,
                            "exec_time_seconds": 1.5,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        output_path = tmp_path / "bench.png"

        result = asyncio.run(
            tool.call(
                tool.input_schema(
                    action="benchmark",
                    report_path=str(report_path),
                    output_path=str(output_path),
                    plot_type="bar",
                ),
                ToolContext(session_id="test", workspace=str(tmp_path)),
            )
        )

        assert result.success is True
        assert output_path.exists()
        assert result.data["exists"] is True

    def test_evolution_plot(self, tmp_path):
        tool = VisualizeTool()
        report_path = tmp_path / "evolution.json"
        report_path.write_text(
            json.dumps(
                {
                    "failure_rules": [{"confidence": 0.8}],
                    "success_skills": [],
                    "prompt_patches": [],
                }
            ),
            encoding="utf-8",
        )
        output_path = tmp_path / "evolution.png"

        result = asyncio.run(
            tool.call(
                tool.input_schema(
                    action="evolution",
                    report_path=str(report_path),
                    output_path=str(output_path),
                    plot_type="summary",
                ),
                ToolContext(session_id="test", workspace=str(tmp_path)),
            )
        )

        assert result.success is True
        assert output_path.exists()

    def test_exploration_plot(self, tmp_path):
        tool = VisualizeTool()
        report_path = tmp_path / "exploration.json"
        report_path.write_text(
            json.dumps(
                {
                    "pareto_front": [{"objectives": {"energy": -1.0, "distance": 2.0}}],
                    "best_branch": {"objectives": {"energy": -1.0, "distance": 2.0}},
                }
            ),
            encoding="utf-8",
        )
        output_path = tmp_path / "exploration.png"

        result = asyncio.run(
            tool.call(
                tool.input_schema(
                    action="exploration",
                    report_path=str(report_path),
                    output_path=str(output_path),
                    plot_type="2d",
                ),
                ToolContext(session_id="test", workspace=str(tmp_path)),
            )
        )

        assert result.success is True
        assert output_path.exists()

    def test_missing_report_fails(self, tmp_path):
        tool = VisualizeTool()
        output_path = tmp_path / "missing.png"

        result = asyncio.run(
            tool.call(
                tool.input_schema(
                    action="benchmark",
                    report_path=str(tmp_path / "nope.json"),
                    output_path=str(output_path),
                ),
                ToolContext(session_id="test", workspace=str(tmp_path)),
            )
        )

        assert result.success is False
        assert "Failed to load report" in (result.error or "")

    def test_report_data_input(self, tmp_path):
        tool = VisualizeTool()
        output_path = tmp_path / "from_data.png"

        result = asyncio.run(
            tool.call(
                tool.input_schema(
                    action="benchmark",
                    report_data={"results": []},
                    output_path=str(output_path),
                    plot_type="pie",
                ),
                ToolContext(session_id="test", workspace=str(tmp_path)),
            )
        )

        assert result.success is False
        assert "No results" in (result.error or "")

    def test_benchmark_plot_has_visual_gate(self, tmp_path):
        """批次 C 后半: 成功生成图后附加 visual_gate 字段."""
        tool = VisualizeTool()
        report_path = tmp_path / "benchmark.json"
        report_path.write_text(
            json.dumps(
                {
                    "results": [
                        {
                            "task_id": "t1",
                            "category": "dft",
                            "passed": True,
                            "exec_time_seconds": 1.5,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        output_path = tmp_path / "bench_gate.png"

        result = asyncio.run(
            tool.call(
                tool.input_schema(
                    action="benchmark",
                    report_path=str(report_path),
                    output_path=str(output_path),
                    plot_type="bar",
                ),
                ToolContext(session_id="test", workspace=str(tmp_path)),
            )
        )

        assert result.success is True
        assert "visual_gate" in result.data
        gate = result.data["visual_gate"]
        assert "action" in gate
        assert "image_path" in gate

    def test_figure_ir_numeric_extraction(self):
        """_figure_ir_numeric: 摊平 series data 为 {label: float|list}."""
        tool = VisualizeTool()
        ir = {
            "series": [
                {"label": "best_fitness", "data": [0.1, 0.2, 0.3]},
                {"label": "single", "data": [5.0]},
                {"label": "empty", "data": []},
                {"label": "nonnum", "data": ["a", "b"]},
            ]
        }
        out = tool._figure_ir_numeric(ir)
        assert out["best_fitness"] == [0.1, 0.2, 0.3]
        assert out["single"] == 5.0
        assert "empty" not in out
        assert "nonnum" not in out

    def test_figure_ir_numeric_malformed(self):
        tool = VisualizeTool()
        assert tool._figure_ir_numeric({}) == {}
        assert tool._figure_ir_numeric({"series": "nope"}) == {}
        assert tool._figure_ir_numeric(None) == {}


# ── 集成路径补测 (原 test_visualize_tool_ext.py) ─────────────────────────––

class _FakeVisualize:
    """Stand-in for huginn.visualize — records calls, returns fake paths."""

    def __init__(self):
        self.calls = []

    def _r(self, out):
        self.calls.append(out)
        p = Path(out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("fake-image", encoding="utf-8")
        return str(p)

    def plot_band_structure(self, bands, kpath, fermi, output_path):
        return self._r(Path(output_path))

    def plot_dos(self, dos, energy, fermi, output_path):
        return self._r(Path(output_path))

    def plot_phonon_dispersion(self, branches, kpath, output_path):
        return self._r(Path(output_path))

    def plot_structure_3d(self, structure, output_path):
        return self._r(Path(output_path))

    def plot_benchmark_report(self, report, output_path=None, plot_type=None):
        return self._r(Path(output_path))

    def plot_evolution_report(self, report, output_path=None, plot_type=None):
        return self._r(Path(output_path))

    def plot_exploration_result(self, report, output_path=None, plot_type=None):
        return self._r(Path(output_path))


class _FakeFigureIR:
    """Stand-in for huginn.vision.figure_ir — keeps series so we can assert
    on the series the tool constructed."""

    def to_ir(self, **kw):
        return kw

    def ir_to_structured(self, ir):
        series = ir.get("series", [])
        return {
            "chart_type": ir.get("chart_type"),
            "title": ir.get("title"),
            "axes": ir.get("axes", {}),
            "series": series,
            "n_series": len(series),
            "series_labels": [s.get("label") for s in series if isinstance(s, dict)],
        }


def _install_visualize(monkeypatch):
    fake = _FakeVisualize()
    mod = types.ModuleType("huginn.visualize")
    for name in (
        "plot_band_structure", "plot_dos", "plot_phonon_dispersion",
        "plot_structure_3d", "plot_benchmark_report", "plot_evolution_report",
        "plot_exploration_result",
    ):
        setattr(mod, name, getattr(fake, name))
    mod._fake = fake
    monkeypatch.setitem(sys.modules, "huginn.visualize", mod)
    import huginn
    monkeypatch.setattr(huginn, "visualize", mod, raising=False)
    return mod


def _install_figure_ir(monkeypatch):
    fake = _FakeFigureIR()
    pkg = types.ModuleType("huginn.vision")
    pkg.figure_ir = fake
    ir_mod = types.ModuleType("huginn.vision.figure_ir")
    ir_mod.to_ir = fake.to_ir
    ir_mod.ir_to_structured = fake.ir_to_structured
    monkeypatch.setitem(sys.modules, "huginn.vision", pkg)
    import huginn.vision as vision_pkg
    monkeypatch.setattr(vision_pkg, "figure_ir", ir_mod, raising=False)
    return fake


def _install_gate(monkeypatch, decision=None):
    gate_mod = types.ModuleType("huginn.tools.visualize_gate")
    gate_mod.run_figure_gate = lambda *a, **k: decision or {"action": "pass", "image_path": str(a[0])}
    monkeypatch.setitem(sys.modules, "huginn.tools.visualize_gate", gate_mod)
    import huginn.tools.visualize_gate as gate_pkg
    monkeypatch.setattr(gate_pkg, "run_figure_gate", gate_mod.run_figure_gate, raising=False)
    return gate_mod


def _tool():
    return VisualizeTool()


def _acall(tool, args, ctx=None):
    return asyncio.run(tool.call(args, ctx))


def test_is_read_only():
    t = _tool()
    assert t.read_only is True
    assert t.is_read_only(VisualizeToolInput(action="benchmark", output_path="x.png")) is True


def test_input_schema_field_defaults():
    inp = VisualizeToolInput(action="dos", output_path="o.png")
    assert inp.plot_type == "auto"
    assert inp.fermi == 0.0


def test_call_band_structure(monkeypatch, tmp_path):
    _install_visualize(monkeypatch)
    _install_figure_ir(monkeypatch)
    _install_gate(monkeypatch)
    out = tmp_path / "band.png"
    res = _acall(_tool(), VisualizeToolInput(
        action="band_structure", output_path=str(out),
        bands_data=[{"kpoints": [], "energies": []}], kpath=["G", "X"], fermi=0.5,
    ))
    assert res.success is True
    assert res.data["exists"] is True
    assert res.data["figure_ir"]["chart_type"] == "line"
    assert [s["label"] for s in res.data["figure_ir"]["series"]] == ["band_0"]


def test_call_dos(monkeypatch, tmp_path):
    _install_visualize(monkeypatch)
    _install_figure_ir(monkeypatch)
    _install_gate(monkeypatch)
    out = tmp_path / "dos.png"
    res = _acall(_tool(), VisualizeToolInput(
        action="dos", output_path=str(out),
        dos_data={"total": [1.0, 2.0], "orbital_s": [0.1]}, energy=[-1.0, 0.0, 1.0],
    ))
    assert res.success is True
    assert [s["label"] for s in res.data["figure_ir"]["series"]] == ["total", "orbital_s"]


def test_call_phonon(monkeypatch, tmp_path):
    _install_visualize(monkeypatch)
    _install_figure_ir(monkeypatch)
    _install_gate(monkeypatch)
    out = tmp_path / "ph.png"
    res = _acall(_tool(), VisualizeToolInput(
        action="phonon", output_path=str(out),
        branches=[{"qpoints": [], "frequencies": []}], kpath=["G", "M"],
    ))
    assert res.success is True
    assert [s["label"] for s in res.data["figure_ir"]["series"]] == ["branch_0"]


def test_call_structure_3d(monkeypatch, tmp_path):
    _install_visualize(monkeypatch)
    _install_figure_ir(monkeypatch)
    _install_gate(monkeypatch)
    out = tmp_path / "struct.png"
    res = _acall(_tool(), VisualizeToolInput(
        action="structure_3d", output_path=str(out),
        structure={"lattice": [], "species": [], "coords": []},
    ))
    assert res.success is True
    assert res.data["figure_ir"]["chart_type"] == "scatter"
    assert res.data["figure_ir"]["series"] == []


def test_call_materials_plotter_exception(monkeypatch, tmp_path):
    mod = _install_visualize(monkeypatch)
    _install_figure_ir(monkeypatch)
    _install_gate(monkeypatch)

    def boom(bands, kpath, fermi, output_path):
        raise RuntimeError("no band code")

    mod.plot_band_structure = boom
    res = _acall(_tool(), VisualizeToolInput(
        action="band_structure", output_path=str(tmp_path / "b.png"),
        bands_data=[{"kpoints": [], "energies": []}],
    ))
    assert res.success is False
    assert "Visualization failed" in res.error


def test_call_materials_gate_error_swallowed(monkeypatch, tmp_path):
    _install_visualize(monkeypatch)
    _install_figure_ir(monkeypatch)
    gate_mod = types.ModuleType("huginn.tools.visualize_gate")

    def boom(*a, **k):
        raise RuntimeError("gate boom")

    gate_mod.run_figure_gate = boom
    monkeypatch.setitem(sys.modules, "huginn.tools.visualize_gate", gate_mod)
    res = _acall(_tool(), VisualizeToolInput(
        action="dos", output_path=str(tmp_path / "d.png"),
        dos_data={"total": [1.0]}, energy=[0.0],
    ))
    assert res.success is True
    assert res.data["visual_gate"]["action"] == "error"


def test_call_benchmark_success(monkeypatch, tmp_path):
    _install_visualize(monkeypatch)
    _install_figure_ir(monkeypatch)
    _install_gate(monkeypatch)
    res = _acall(_tool(), VisualizeToolInput(
        action="benchmark", output_path=str(tmp_path / "b.png"),
        report_data={"scores": {"dft": 0.9, "md": 0.7}}, plot_type="bar",
    ))
    assert res.success is True
    ir = res.data["figure_ir"]
    assert ir["chart_type"] == "bar"
    assert {"label": "dft", "data": [0.9]} in ir["series"]


def test_call_evolution_success(monkeypatch, tmp_path):
    _install_visualize(monkeypatch)
    _install_figure_ir(monkeypatch)
    _install_gate(monkeypatch)
    res = _acall(_tool(), VisualizeToolInput(
        action="evolution", output_path=str(tmp_path / "e.png"),
        report_data={"timeline": [{"best_fitness": 0.1}, {"best_fitness": 0.2}]},
    ))
    assert res.success is True
    ir = res.data["figure_ir"]
    assert ir["chart_type"] == "line"
    assert ir["series"] == [{"label": "best_fitness", "data": [0.1, 0.2]}]


def test_call_exploration_success(monkeypatch, tmp_path):
    _install_visualize(monkeypatch)
    _install_figure_ir(monkeypatch)
    _install_gate(monkeypatch)
    res = _acall(_tool(), VisualizeToolInput(
        action="exploration", output_path=str(tmp_path / "x.png"),
        report_data={"candidates": [{"objective": 1.0}, {"objective": 2.0}]},
    ))
    assert res.success is True
    ir = res.data["figure_ir"]
    assert ir["chart_type"] == "scatter"
    assert ir["series"] == [{"label": "candidates", "data": [1.0, 2.0]}]


def test_call_report_plotter_exception(monkeypatch, tmp_path):
    mod = _install_visualize(monkeypatch)
    _install_figure_ir(monkeypatch)
    _install_gate(monkeypatch)

    def boom(report, output_path=None, plot_type=None):
        raise RuntimeError("plot failed")

    mod.plot_benchmark_report = boom
    res = _acall(_tool(), VisualizeToolInput(
        action="benchmark", output_path=str(tmp_path / "b.png"),
        report_data={"scores": {"a": 1.0}},
    ))
    assert res.success is False
    assert "Visualization failed" in res.error


def test_call_load_report_failure(monkeypatch, tmp_path):
    _install_visualize(monkeypatch)
    _install_figure_ir(monkeypatch)
    _install_gate(monkeypatch)
    res = _acall(_tool(), VisualizeToolInput(
        action="benchmark", output_path=str(tmp_path / "b.png"),
        report_path=str(tmp_path / "nope.json"),
    ))
    assert res.success is False
    assert "Failed to load report" in res.error


def test_run_gate_success(monkeypatch, tmp_path):
    _install_gate(monkeypatch, decision={"action": "pass", "image_path": "x.png"})
    t = _tool()
    assert t._run_gate(tmp_path / "x.png")["action"] == "pass"


def test_run_gate_exception_swallowed(monkeypatch, tmp_path):
    gate_mod = types.ModuleType("huginn.tools.visualize_gate")

    def boom(*a, **k):
        raise RuntimeError("gate boom")

    gate_mod.run_figure_gate = boom
    monkeypatch.setitem(sys.modules, "huginn.tools.visualize_gate", gate_mod)
    t = _tool()
    gate = t._run_gate(tmp_path / "x.png", expected={"a": 1.0})
    assert gate["action"] == "error"
    assert "gate failed" in gate["error"]


def test_figure_ir_numeric_variants():
    t = _tool()
    multi = {"label": "multi", "data": [1, 2.0, 3]}
    single = {"label": "single", "data": [7.0]}
    no_label = {"data": [1.0]}
    empty = {"label": "empty", "data": []}
    nonnum = {"label": "nonnum", "data": ["x", "y"]}
    mixed = {"label": "mixed", "data": [1, "x", 2]}

    out = t._figure_ir_numeric({"series": [multi, single]})
    assert out["multi"] == [1.0, 2.0, 3.0]
    assert out["single"] == 7.0

    out2 = t._figure_ir_numeric({"series": ["junk", multi]})
    assert out2 == {"multi": [1.0, 2.0, 3.0]}

    out3 = t._figure_ir_numeric({"series": [no_label]})
    assert out3 == {}

    out4 = t._figure_ir_numeric({"series": [empty]})
    assert out4 == {}

    out5 = t._figure_ir_numeric({"series": [nonnum]})
    assert out5 == {}

    out6 = t._figure_ir_numeric({"series": [mixed]})
    assert out6 == {"mixed": [1.0, 2.0]}

    out7 = t._figure_ir_numeric({"series": "not-a-list"})
    assert out7 == {}


def test_load_report_data():
    t = _tool()
    assert t._load_report(VisualizeToolInput(action="benchmark", output_path="x", report_data={"a": 1})) == {"a": 1}


def test_load_report_path(tmp_path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"x": 1}), encoding="utf-8")
    t = _tool()
    assert t._load_report(VisualizeToolInput(action="benchmark", output_path="x", report_path=str(p))) == {"x": 1}


def test_load_report_neither():
    t = _tool()
    with pytest.raises(ValueError):
        t._load_report(VisualizeToolInput(action="benchmark", output_path="x"))


def test_build_figure_ir_band(monkeypatch):
    _install_figure_ir(monkeypatch)
    t = _tool()
    ir = t._build_figure_ir(VisualizeToolInput(
        action="band_structure", output_path="x", bands_data=[{}, {}], kpath=["G", "X"], fermi=1.0,
    ))
    assert ir["chart_type"] == "line"
    assert [s["label"] for s in ir["series"]] == ["band_0", "band_1"]


def test_build_figure_ir_dos(monkeypatch):
    _install_figure_ir(monkeypatch)
    t = _tool()
    ir = t._build_figure_ir(VisualizeToolInput(
        action="dos", output_path="x", dos_data={"total": [], "orbital_p": []},
    ))
    assert ir["chart_type"] == "line"
    assert [s["label"] for s in ir["series"]] == ["total", "orbital_p"]


def test_build_figure_ir_phonon(monkeypatch):
    _install_figure_ir(monkeypatch)
    t = _tool()
    ir = t._build_figure_ir(VisualizeToolInput(action="phonon", output_path="x", branches=[{}]))
    assert ir["chart_type"] == "line"
    assert [s["label"] for s in ir["series"]] == ["branch_0"]


def test_build_figure_ir_structure_3d(monkeypatch):
    _install_figure_ir(monkeypatch)
    t = _tool()
    ir = t._build_figure_ir(VisualizeToolInput(action="structure_3d", output_path="x", structure={}))
    assert ir["chart_type"] == "scatter"
    assert ir["series"] == []


def _install_figure_ir_boom(monkeypatch):
    """figure_ir 模块存在但调用抛异常 → 触发 except 分支."""
    pkg = types.ModuleType("huginn.vision")

    def boom(*a, **k):
        raise RuntimeError("figure_ir broken")

    pkg.figure_ir = types.SimpleNamespace(to_ir=boom, ir_to_structured=boom)
    monkeypatch.setitem(sys.modules, "huginn.vision", pkg)
    import huginn.vision as vision_pkg
    monkeypatch.setattr(vision_pkg, "figure_ir", pkg.figure_ir, raising=False)


def test_build_figure_ir_import_error(monkeypatch):
    _install_figure_ir_boom(monkeypatch)
    t = _tool()
    ir = t._build_figure_ir(VisualizeToolInput(action="dos", output_path="x", dos_data={"t": []}))
    assert "error" in ir


def test_build_report_ir_benchmark_scores_dict(monkeypatch):
    _install_figure_ir(monkeypatch)
    t = _tool()
    ir = t._build_report_figure_ir("benchmark", {"scores": {"a": 1.0, "b": 2.0}})
    assert ir["chart_type"] == "bar"
    assert {"label": "a", "data": [1.0]} in ir["series"]


def test_build_report_ir_benchmark_scores_list(monkeypatch):
    _install_figure_ir(monkeypatch)
    t = _tool()
    ir = t._build_report_figure_ir("benchmark", {"metrics": [1.0, 2.0, 3.0]})
    assert ir["chart_type"] == "bar"
    assert {"label": "item_0", "data": [1.0]} in ir["series"]


def test_build_report_ir_benchmark_scan_fallback(monkeypatch):
    _install_figure_ir(monkeypatch)
    t = _tool()
    ir = t._build_report_figure_ir("benchmark", {"deep": {"nested": {"val": 5.0}}})
    assert ir["chart_type"] == "bar"
    assert {"label": "deep.nested.val", "data": [5.0]} in ir["series"]


def test_build_report_ir_evolution_timeline_dict(monkeypatch):
    _install_figure_ir(monkeypatch)
    t = _tool()
    ir = t._build_report_figure_ir("evolution", {
        "timeline": [{"best_fitness": 0.1}, {"fitness": 0.2}, {"score": 0.3}, {"other": 1}],
    })
    assert ir["series"] == [{"label": "best_fitness", "data": [0.1, 0.2, 0.3]}]


def test_build_report_ir_evolution_scalar_list(monkeypatch):
    _install_figure_ir(monkeypatch)
    t = _tool()
    ir = t._build_report_figure_ir("evolution", {"generations": [1.0, 2.0, 3.0]})
    assert ir["series"] == [{"label": "best_fitness", "data": [1.0, 2.0, 3.0]}]


def test_build_report_ir_evolution_scan_fallback(monkeypatch):
    _install_figure_ir(monkeypatch)
    t = _tool()
    ir = t._build_report_figure_ir("evolution", {"gens": [[0.5, 0.6], [0.7, 0.8]]})
    assert ir["series"] and ir["series"][0]["data"] == [0.5, 0.6]


def test_build_report_ir_exploration_candidates(monkeypatch):
    _install_figure_ir(monkeypatch)
    t = _tool()
    ir = t._build_report_figure_ir("exploration", {
        "candidates": [{"objective": 1.0}, {"score": 2.0}, {"value": 3.0}, {"x": "skip"}],
    })
    assert ir["series"] == [{"label": "candidates", "data": [1.0, 2.0, 3.0]}]


def test_build_report_ir_exploration_scan_fallback(monkeypatch):
    _install_figure_ir(monkeypatch)
    t = _tool()
    ir = t._build_report_figure_ir("exploration", {"pts": [0.1, 0.2, 0.3]})
    assert ir["series"] == [{"label": "values", "data": [0.1, 0.2, 0.3]}]


def test_build_report_ir_import_error(monkeypatch):
    _install_figure_ir_boom(monkeypatch)
    t = _tool()
    ir = t._build_report_figure_ir("benchmark", {"scores": {"a": 1.0}})
    assert "error" in ir


def test_scan_numeric_fields():
    t = _tool()
    found = t._scan_numeric_fields({"a": 1, "b": 2.5, "c": True, "d": "x", "e": {"f": 3}})
    assert ("a", 1.0) in found
    assert ("b", 2.5) in found
    assert ("e.f", 3.0) in found
    assert all(k != "c" for k, _ in found)  # bool 被过滤


def test_scan_numeric_fields_depth_limit():
    t = _tool()
    nested = {"l0": {"l1": {"l2": {"l3": {"l4": {"l5": {"l6": {"l7": {"l8": {"x": 1}}}}}}}}}}
    found = t._scan_numeric_fields(nested)
    assert found == []  # 深度 >8 时返回空


def test_scan_numeric_fields_max_items():
    t = _tool()
    found = t._scan_numeric_fields({f"k{i}": i for i in range(20)}, max_items=5)
    assert len(found) == 5


def test_scan_numeric_list_best():
    t = _tool()
    assert t._scan_numeric_list({"a": [1.0, 2.0], "b": [1.0, 2.0, 3.0]}) == [1.0, 2.0, 3.0]


def test_scan_numeric_list_nested():
    t = _tool()
    assert t._scan_numeric_list({"a": [[1.0, 2.0], [3.0]]}) == [1.0, 2.0]


def test_scan_numeric_list_filters_bool():
    t = _tool()
    assert t._scan_numeric_list({"a": [1.0, True, 2.0]}) == [1.0, 2.0]


def test_scan_numeric_list_depth_limit():
    t = _tool()
    deep = {"a": [[[[[[1.0]]]]]]}
    assert t._scan_numeric_list(deep) == []


def test_scan_numeric_list_scalar_input():
    t = _tool()
    assert t._scan_numeric_list(5.0) == []
    assert t._scan_numeric_list("x") == []


def test_scan_numeric_list_max_items_in_nested():
    """max_items 在嵌套 dict/list 递归中已达上限 → 提前 return."""
    t = _tool()
    data = {"a": {"b": {"c": [1.0, 2.0, 3.0, 4.0, 5.0]}}}
    assert t._scan_numeric_list(data, max_items=2) == [1.0, 2.0]


def test_extract_timeline_values():
    t = _tool()
    tl = [
        {"best_fitness": 0.1},
        {"fitness": 0.2},
        {"score": 0.3},
        {"other": 1},
        "skip-str",
        0.5,
        None,
    ]
    assert t._extract_timeline_values(tl) == [0.1, 0.2, 0.3, 0.5]


def test_extract_candidate_values():
    t = _tool()
    cands = [
        {"objective": 1.0},
        {"score": 2.0},
        {"value": 3.0},
        {"other": 1},
        "skip",
        4.0,
    ]
    assert t._extract_candidate_values(cands) == [1.0, 2.0, 3.0, 4.0]
