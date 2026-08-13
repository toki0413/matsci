"""Extended tests for report_tool.py — 覆盖 ReportGenerator / ReportComparator /
ReportTool 的 generate/compare/export/scan_directory/compile_pdf 全部分支.

纯 Python 模块, 无 numpy/scipy 依赖; 用 `-p tests._fakenp_plugin` 跑覆盖率避免
coverage 与 numpy C 扩展冲突 (conftest registry fixture 会拉起 numpy 链).
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from huginn.tools.report_tool import (
    ReportComparator,
    ReportGenerator,
    ReportSection,
    ReportTool,
    ReportToolInput,
)

# ── ReportGenerator ──────────────────────────────────────────────────────────

class TestReportGenerator:
    def test_add_section_and_out(self):
        gen = ReportGenerator(style="full", fmt="markdown")
        gen.add_section("A", "content-a", 2)
        gen.add_section("B", "content-b", 1)
        assert isinstance(gen.sections[0], ReportSection)
        # 按添加顺序
        assert [s.title for s in gen.sections] == ["A", "B"]
        # generate 内会按 order 排序
        gen.generate({})
        assert gen.sections[0].title == "B"

    def test_generate_markdown(self):
        gen = ReportGenerator(style="brief", fmt="markdown")
        out = gen.generate(self._sample_data())
        assert "# Computational Report: Brief Format" in out
        assert "## Methods" in out

    def test_generate_latex(self):
        gen = ReportGenerator(style="full", fmt="latex")
        out = gen.generate(self._sample_data())
        assert "\\documentclass{article}" in out
        assert "\\end{document}" in out

    def test_generate_json(self):
        gen = ReportGenerator(style="full", fmt="json")
        out = gen.generate(self._sample_data())
        parsed = json.loads(out)
        assert parsed["style"] == "full"
        assert any(s["title"] == "Methods" for s in parsed["sections"])

    def test_generate_html(self):
        gen = ReportGenerator(style="full", fmt="html")
        out = gen.generate(self._sample_data())
        assert "<!DOCTYPE html>" in out
        assert "<h1>Computational Report</h1>" in out

    def test_generate_unknown_format_falls_back_markdown(self):
        gen = ReportGenerator(style="full", fmt="unknown_fmt")
        out = gen.generate(self._sample_data())
        assert "# Computational Report:" in out

    def test_render_methods_brief(self):
        gen = ReportGenerator(style="brief", fmt="markdown")
        out = gen._render_methods({"method": "DFT", "functional": "PBE", "encut": "520"})
        assert "**Method**: DFT" in out
        assert "520" in out

    def test_render_methods_full(self):
        gen = ReportGenerator(style="full", fmt="markdown")
        out = gen._render_methods(
            {"software": "VASP", "functional": "PBE", "encut": "520",
             "kpoints": "5x5x5", "pseudopotentials": "PAW", "smearing": "Methfessel",
             "ediffg": "0.01"}
        )
        assert "VASP" in out
        assert "5x5x5" in out
        assert "PAW" in out

    def test_render_structure_with_change(self):
        gen = ReportGenerator(style="full", fmt="markdown")
        out = gen._render_structure(
            {"formula": "Si", "spacegroup": "Fd-3m",
             "initial_a": 5.4, "final_a": 5.43}
        )
        assert "Si" in out
        assert "+0.56%" in out  # (5.43-5.4)/5.4*100 = 0.5556

    def test_render_structure_non_numeric_no_change(self):
        gen = ReportGenerator(style="full", fmt="markdown")
        out = gen._render_structure({"initial_a": "N/A", "final_a": "5.4"})
        assert "| a | N/A | 5.4 | N/A |" in out

    def test_render_convergence(self):
        gen = ReportGenerator(style="full", fmt="markdown")
        out = gen._render_convergence(
            {"energy": -10.5, "n_iterations": 5, "n_electronic": 12}
        )
        assert "- Final energy: -10.5 eV" in out
        assert "- Total ionic steps: 5" in out
        assert "- Average electronic steps: 12" in out

    def test_render_convergence_empty(self):
        gen = ReportGenerator(style="full", fmt="markdown")
        out = gen._render_convergence({})
        assert "## Convergence History" in out

    def test_render_results_nested_and_scalar(self):
        gen = ReportGenerator(style="full", fmt="markdown")
        out = gen._render_results(
            {"band_gap": 1.1, "elastic": {"c11": 165, "c12": 64}}
        )
        assert "- **band_gap**: 1.1" in out
        assert "- c11: 165" in out

    def test_render_validation(self):
        gen = ReportGenerator(style="full", fmt="markdown")
        out = gen._render_validation(
            {"checks": [
                {"passed": True, "name": "converged", "message": "ok"},
                {"passed": False, "name": "gamma", "message": "bad"},
            ]}
        )
        assert "✅ converged: ok" in out
        assert "❌ gamma: bad" in out

    def test_render_literature(self):
        gen = ReportGenerator(style="full", fmt="markdown")
        out = gen._render_literature(
            {"comparisons": [
                {"property": "Eg", "calculated": 1.1, "reference": 1.2, "source": "exp"}
            ]}
        )
        assert "Eg" in out
        assert "calculated = 1.1" in out

    def test_render_resources(self):
        gen = ReportGenerator(style="full", fmt="markdown")
        out = gen._render_resources(
            {"cpu_hours": 12, "walltime_hours": 3, "memory_gb": 8, "cores": 16}
        )
        assert "- **CPU time**: 12 hours" in out
        assert "- **Cores**: 16" in out

    def test_render_reproducibility(self):
        gen = ReportGenerator(style="full", fmt="markdown")
        out = gen._render_reproducibility(
            {"software_version": "1.0", "input_hash": "abc",
             "random_seed": 42, "input_files": ["POSCAR", "INCAR"]}
        )
        assert "- **Software version**: 1.0" in out
        assert "- `POSCAR`" in out
        assert "- `INCAR`" in out

    def test_build_sections_conditional(self):
        gen = ReportGenerator(style="full", fmt="markdown")
        gen._build_sections(self._sample_data())
        titles = [s.title for s in gen.sections]
        assert "Validation" in titles
        assert "Literature Comparison" in titles

    def test_build_sections_no_optional(self):
        gen = ReportGenerator(style="full", fmt="markdown")
        gen._build_sections({"methods": {}, "structure": {}})
        titles = [s.title for s in gen.sections]
        assert "Validation" not in titles
        assert "Literature Comparison" not in titles

    @staticmethod
    def _sample_data():
        return {
            "methods": {
                "software": "VASP", "functional": "PBE", "encut": "520",
                "kpoints": "5x5x5", "pseudopotentials": "PAW", "smearing": "Methfessel",
                "ediffg": "0.01",
            },
            "structure": {
                "formula": "Si", "spacegroup": "Fd-3m",
                "initial_a": 5.4, "final_a": 5.43,
            },
            "convergence": {"energy": -10.5, "n_iterations": 5, "n_electronic": 12},
            "results": {"band_gap": 1.1, "elastic": {"c11": 165}},
            "validation": {"checks": [{"passed": True, "name": "c", "message": "ok"}]},
            "literature_comparison": {"comparisons": [{"property": "Eg"}]},
            "resources": {"cpu_hours": 12, "memory_gb": 8},
            "software_version": "1.0", "input_hash": "abc", "random_seed": 42,
            "input_files": ["POSCAR"],
        }


# ── ReportComparator ─────────────────────────────────────────────────────────

class TestReportComparator:
    def test_generate_markdown(self):
        comp = ReportComparator(names=["run_A", "run_B"], style="full", fmt="markdown")
        datasets = self._two_datasets()
        out = comp.generate(datasets)
        assert "# Comparison Report: run_A vs run_B" in out
        assert "## Methods Comparison" in out
        assert "## Results Comparison" in out
        assert "## Convergence Comparison" in out
        assert "## Resource Comparison" in out

    def test_generate_json_header_empty(self):
        comp = ReportComparator(names=["run_A"], style="full", fmt="json")
        assert comp._header() == ""

    def test_header_latex(self):
        comp = ReportComparator(names=["run_A"], style="full", fmt="latex")
        assert "\\section*{" in comp._header()

    def test_methods_comparison(self):
        comp = ReportComparator(names=["run_A", "run_B"], style="full", fmt="markdown")
        out = comp._methods_comparison(self._two_datasets())
        assert "| Parameter | run_A | run_B |" in out
        assert "| method | DFT | DFT |" in out

    def test_results_comparison_union_keys(self):
        comp = ReportComparator(names=["run_A", "run_B"], style="full", fmt="markdown")
        out = comp._results_comparison(self._two_datasets())
        assert "| band_gap |" in out
        assert "| N/A |" in out  # run_B 缺 band_gap 显示 N/A

    def test_convergence_comparison(self):
        comp = ReportComparator(names=["run_A", "run_B"], style="full", fmt="markdown")
        out = comp._convergence_comparison(self._two_datasets())
        assert "| energy |" in out
        assert "| n_iterations |" in out

    def test_resources_comparison(self):
        comp = ReportComparator(names=["run_A", "run_B"], style="full", fmt="markdown")
        out = comp._resources_comparison(self._two_datasets())
        assert "| cpu_hours |" in out
        assert "| cores |" in out

    @staticmethod
    def _two_datasets():
        return {
            "run_A": {
                "methods": {"method": "DFT", "functional": "PBE", "encut": "520"},
                "results": {"band_gap": 1.1, "eg": 1.2},
                "convergence": {"energy": -10.5, "n_iterations": 5, "converged": True},
                "resources": {"cpu_hours": 12, "cores": 8},
            },
            "run_B": {
                "methods": {"method": "DFT", "functional": "PBE", "encut": "400"},
                "results": {"eg": 1.3, "extra": 0.5},
                "convergence": {"energy": -9.8, "n_iterations": 8, "converged": False},
                "resources": {"cpu_hours": 20, "cores": 16},
            },
        }


# ── ReportTool ───────────────────────────────────────────────────────────────

class TestReportTool:
    @pytest.mark.asyncio
    async def test_is_read_only_generate_compare(self):
        tool = ReportTool()
        assert tool.is_read_only(ReportToolInput(action="generate")) is True
        assert tool.is_read_only(ReportToolInput(action="compare")) is True
        assert tool.is_read_only(ReportToolInput(action="export")) is False
        assert tool.is_read_only(ReportToolInput(action="compile_pdf")) is False
        assert tool.read_only is False

    @pytest.mark.asyncio
    async def test_generate_no_results(self):
        tool = ReportTool()
        result = await tool.call(ReportToolInput(action="generate"), context=None)
        assert result.success
        assert "## Methods" in result.data["report"]

    @pytest.mark.asyncio
    async def test_generate_with_output_path(self, tmp_path):
        tool = ReportTool()
        out = tmp_path / "sub" / "report.md"
        result = await tool.call(
            ReportToolInput(
                action="generate",
                workflow_results={"methods": {"encut": "520"}},
                output_path=str(out),
            ),
            context=None,
        )
        assert result.success
        assert result.data["saved_to"] == str(out)
        assert out.exists()
        assert "520" in out.read_text(encoding="utf-8")

    @pytest.mark.asyncio
    async def test_generate_with_calculation_dir(self, tmp_path):
        tool = ReportTool()
        calc = tmp_path / "calc"
        calc.mkdir()
        (calc / "params.incar").write_text("ENCUT = 520\n", encoding="utf-8")
        (calc / "OUTCAR").write_text(
            "ENCUT =      520.0 eV\nsome other line\n", encoding="utf-8"
        )
        (calc / "job.log").write_text("log", encoding="utf-8")
        result = await tool.call(
            ReportToolInput(action="generate", calculation_dir=str(calc)),
            context=None,
        )
        assert result.success
        data_report = result.data["report"]
        # input_files 含 params.incar (小写后缀). output_files 被 _scan_directory
        # 收集但 ReportGenerator._render_reproducibility 只渲染 input_files.
        assert "params.incar" in data_report
        # ENCUT 从 OUTCAR 提取
        assert "520" in data_report
        # 直接验证 _scan_directory 的 output_files 收集 (OUTCAR 无后缀, 走
        # name.upper()=="OUTCAR" 单独解析, 不进 output_files)
        scanned = tool._scan_directory(str(calc), {})
        assert "job.log" in scanned["output_files"]
        assert scanned["methods"]["encut"] == "520.0"

    @pytest.mark.asyncio
    async def test_export_action(self, tmp_path):
        tool = ReportTool()
        out = tmp_path / "out" / "report.latex"
        result = await tool.call(
            ReportToolInput(
                action="export",
                workflow_results={"methods": {"encut": "400"}},
                format="latex",
                output_path=str(out),
            ),
            context=None,
        )
        assert result.success
        assert out.exists()
        assert "\\documentclass" in out.read_text(encoding="utf-8")

    @pytest.mark.asyncio
    async def test_compare_requires_datasets(self):
        tool = ReportTool()
        result = await tool.call(
            ReportToolInput(action="compare", workflow_results={}), context=None
        )
        assert not result.success
        assert "comparison_datasets required" in result.error

    @pytest.mark.asyncio
    async def test_compare_extract_from_workflow_results(self):
        tool = ReportTool()
        result = await tool.call(
            ReportToolInput(
                action="compare",
                workflow_results={
                    "comparison_datasets": {"A": {"methods": {}}, "B": {"methods": {}}}
                },
            ),
            context=None,
        )
        assert result.success
        assert "Comparison Report: A vs B" in result.data["report"]

    @pytest.mark.asyncio
    async def test_compare_requires_two_datasets(self):
        tool = ReportTool()
        result = await tool.call(
            ReportToolInput(
                action="compare",
                comparison_datasets={"A": {"methods": {}}},
            ),
            context=None,
        )
        assert not result.success
        assert "At least 2 datasets" in result.error

    @pytest.mark.asyncio
    async def test_compare_with_output_path(self, tmp_path):
        tool = ReportTool()
        out = tmp_path / "cmp.md"
        result = await tool.call(
            ReportToolInput(
                action="compare",
                comparison_datasets={"A": {"results": {}}, "B": {"results": {}}},
                output_path=str(out),
            ),
            context=None,
        )
        assert result.success
        assert out.exists()

    @pytest.mark.asyncio
    async def test_unknown_action(self):
        tool = ReportTool()
        result = await tool.call(ReportToolInput(action="generate"), context=None)
        # 直接构造非法 action 走 unknown 分支
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ReportToolInput(action="bogus")

    @pytest.mark.asyncio
    async def test_estimate_cost(self):
        tool = ReportTool()
        assert tool.estimate_cost(ReportToolInput(action="generate"))[
            "walltime_hours"
        ] == 0.01

    # ── compile_pdf 补充分支 (同 test_report_compile, 补 timeout/no-pdf) ──

    @pytest.mark.asyncio
    async def test_compile_pdf_no_tex_source(self):
        tool = ReportTool()
        result = await tool.call(
            ReportToolInput(action="compile_pdf", engine="pdflatex"), context=None
        )
        assert not result.success
        assert "requires tex_source" in result.error

    @pytest.mark.asyncio
    async def test_compile_pdf_timeout(self):
        import subprocess

        tool = ReportTool()
        args = ReportToolInput(
            action="compile_pdf",
            tex_source=r"\documentclass{article}\begin{document}Hi\end{document}",
            engine="pdflatex",
        )

        def fake_run(cmd, cwd, capture_output, text, timeout):
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout)

        with patch("shutil.which", return_value="/usr/bin/pdflatex"), patch(
            "subprocess.run", side_effect=fake_run
        ):
            result = await tool.call(args, context=None)
        assert not result.success
        assert "timed out" in result.error

    @pytest.mark.asyncio
    async def test_compile_pdf_no_pdf_generated(self, tmp_path):
        """两次编译成功但无 .pdf → success=False + 'PDF not generated'."""
        tool = ReportTool()
        args = ReportToolInput(
            action="compile_pdf",
            tex_source=r"\documentclass{article}\begin{document}Hi\end{document}",
            engine="pdflatex",
        )

        def fake_run(cmd, cwd, capture_output, text, timeout):
            mock_proc = MagicMock()
            mock_proc.returncode = 0
            mock_proc.stdout = "ok"
            mock_proc.stderr = ""
            return mock_proc

        with patch("shutil.which", return_value="/usr/bin/pdflatex"), patch(
            "subprocess.run", side_effect=fake_run
        ):
            result = await tool.call(args, context=None)
        assert not result.success
        assert "PDF not generated" in result.error

    @pytest.mark.asyncio
    async def test_scan_directory_outcar_without_encut(self, tmp_path):
        tool = ReportTool()
        calc = tmp_path / "calc2"
        calc.mkdir()
        (calc / "OUTCAR").write_text("no encut here\n", encoding="utf-8")
        data = tool._scan_directory(str(calc), {})
        # OUTCAR 无后缀不被 output_files 收集, 但整段 OUTCAR 解析逻辑仍执行
        assert "methods" not in data

    @pytest.mark.asyncio
    async def test_scan_directory_incar_extraction(self, tmp_path):
        tool = ReportTool()
        calc = tmp_path / "calc3"
        calc.mkdir()
        (calc / "OUTCAR").write_text(
            "ENCUT =      520.0 eV\n", encoding="utf-8"
        )
        data = tool._scan_directory(str(calc), {})
        assert data["methods"]["encut"] == "520.0"

    @pytest.mark.asyncio
    async def test_scan_directory_outcar_edge_branches(self, tmp_path):
        """覆盖 OUTCAR 解析的边界分支:
        - 目录里 OUTCAR 不是第一个文件 (585->584 继续循环)
        - OUTCAR 含 ENCUT 但 ENCUT 行无 '=' (589->588)
        - split('=') 后只有 1 段 (591->595 len<2)
        - 前 100 行无 'ENCUT ... =' 命中 (588->596)
        """
        tool = ReportTool()
        calc = tmp_path / "calc4"
        calc.mkdir()
        (calc / "a.incar").write_text("ENCUT = 400\n", encoding="utf-8")
        (calc / "OUTCAR").write_text("ENCUT 520\n", encoding="utf-8")
        data = tool._scan_directory(str(calc), {})
        # OUTCAR 无 '=' → 不写入 methods
        assert "methods" not in data

        calc5 = tmp_path / "calc5"
        calc5.mkdir()
        (calc5 / "OUTCAR").write_text(
            "some header\nENCUT\nmore header\n", encoding="utf-8"
        )
        data5 = tool._scan_directory(str(calc5), {})
        # ENCUT 存在但无 '=' 行 → 无 encut
        assert "methods" not in data5
