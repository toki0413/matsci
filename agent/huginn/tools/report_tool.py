"""Report generation tool — automatically produce computational reports.

Generates publication-quality reports from workflow results, including:
- Methods section (DFT functional, pseudopotentials, k-points, cutoff)
- Structural information (initial/final structures, symmetry changes)
- Convergence history (energy, force, electronic steps)
- Physical properties (band gap, DOS, phonon spectrum, elastic constants)
- Comparison with literature/databases
- Resource consumption and reproducibility info

Output formats: Markdown (default), LaTeX, JSON.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool, ResearchPhase, ToolProfile
from huginn.types import ToolContext, ToolResult


class ReportToolInput(BaseModel):
    action: Literal["generate", "compare", "export", "compile_pdf"] = Field(
        default="generate"
    )
    workflow_results: dict[str, Any] | None = Field(
        default=None, description="Results from a workflow execution"
    )
    calculation_dir: str | None = Field(
        default=None, description="Directory containing calculation outputs"
    )
    style: Literal["brief", "full", "paper_methods"] = Field(default="full")
    format: Literal["markdown", "latex", "json", "html"] = Field(default="markdown")
    output_path: str | None = Field(
        default=None, description="Where to save the report"
    )
    include_files: list[str] = Field(
        default_factory=list, description="Additional files to include"
    )
    comparison_datasets: dict[str, dict[str, Any]] | None = Field(
        default=None,
        description="Named datasets for comparison, e.g. {'run_A': {...}, 'run_B': {...}}",
    )
    # compile_pdf 专用
    tex_source: str | None = Field(
        default=None, description="完整 LaTeX 源码 (compile_pdf 时必填)"
    )
    engine: Literal["pdflatex", "xelatex", "lualatex"] = Field(
        default="pdflatex", description="TeX 编译引擎"
    )


@dataclass
class ReportSection:
    title: str
    content: str
    order: int = 0


class ReportGenerator:
    """Generate computational reports from materials science workflows."""

    def __init__(self, style: str = "full", fmt: str = "markdown"):
        self.style = style
        self.format = fmt
        self.sections: list[ReportSection] = []

    def add_section(self, title: str, content: str, order: int = 0) -> None:
        self.sections.append(ReportSection(title, content, order))

    def generate(self, data: dict[str, Any]) -> str:
        """Generate report from structured data."""
        self._build_sections(data)
        self.sections.sort(key=lambda s: s.order)

        if self.format == "markdown":
            return self._to_markdown()
        elif self.format == "latex":
            return self._to_latex()
        elif self.format == "json":
            return self._to_json()
        elif self.format == "html":
            return self._to_html()
        else:
            return self._to_markdown()

    def _build_sections(self, data: dict[str, Any]) -> None:
        """Build report sections from data."""
        # Methods
        methods = data.get("methods", {})
        self.add_section("Methods", self._render_methods(methods), 1)

        # Structure
        structure = data.get("structure", {})
        self.add_section("Structure", self._render_structure(structure), 2)

        # Convergence
        convergence = data.get("convergence", {})
        self.add_section("Convergence", self._render_convergence(convergence), 3)

        # Results
        results = data.get("results", {})
        self.add_section("Results", self._render_results(results), 4)

        # Validation
        validation = data.get("validation", {})
        if validation:
            self.add_section("Validation", self._render_validation(validation), 5)

        # Literature comparison
        literature = data.get("literature_comparison", {})
        if literature:
            self.add_section(
                "Literature Comparison", self._render_literature(literature), 6
            )

        # Resources
        resources = data.get("resources", {})
        self.add_section(
            "Computational Resources", self._render_resources(resources), 7
        )

        # Reproducibility
        self.add_section("Reproducibility", self._render_reproducibility(data), 8)

    def _render_methods(self, methods: dict[str, Any]) -> str:
        if self.style == "brief":
            lines = [
                f"- **Method**: {methods.get('method', 'Not specified')}",
                f"- **Functional**: {methods.get('functional', 'Not specified')}",
                f"- **Energy cutoff**: {methods.get('encut', 'Not specified')} eV",
            ]
            return "\n".join(lines)

        lines = [
            "## Computational Methods",
            "",
            f"All calculations were performed using {methods.get('software', 'DFT software')}",
            f"with the {methods.get('functional', 'exchange-correlation functional')} functional.",
            "",
            "### Parameters",
            "",
            "| Parameter | Value |",
            "|-----------|-------|",
            f"| Plane-wave cutoff | {methods.get('encut', 'N/A')} eV |",
            f"| K-point mesh | {methods.get('kpoints', 'N/A')} |",
            f"| Pseudopotentials | {methods.get('pseudopotentials', 'N/A')} |",
            f"| Smearing | {methods.get('smearing', 'N/A')} |",
            f"| Force tolerance | {methods.get('ediffg', 'N/A')} eV/Å |",
            "",
        ]
        return "\n".join(lines)

    def _render_structure(self, structure: dict[str, Any]) -> str:
        lines = [
            "## Structural Information",
            "",
            f"**Formula**: {structure.get('formula', 'N/A')}",
            f"**Space group**: {structure.get('spacegroup', 'N/A')}",
            "",
            "### Lattice parameters",
            "",
            "| Parameter | Initial | Final | Change |",
            "|-----------|---------|-------|--------|",
        ]
        for param in ["a", "b", "c", "alpha", "beta", "gamma"]:
            init = structure.get(f"initial_{param}", "N/A")
            final = structure.get(f"final_{param}", "N/A")
            change = "N/A"
            if isinstance(init, (int, float)) and isinstance(final, (int, float)):
                change = f"{((final - init) / init * 100):+.2f}%"
            lines.append(f"| {param} | {init} | {final} | {change} |")
        lines.append("")
        return "\n".join(lines)

    def _render_convergence(self, convergence: dict[str, Any]) -> str:
        lines = ["## Convergence History", ""]
        if "energy" in convergence:
            lines.append(f"- Final energy: {convergence['energy']} eV")
        if "n_iterations" in convergence:
            lines.append(f"- Total ionic steps: {convergence['n_iterations']}")
        if "n_electronic" in convergence:
            lines.append(f"- Average electronic steps: {convergence['n_electronic']}")
        lines.append("")
        return "\n".join(lines)

    def _render_results(self, results: dict[str, Any]) -> str:
        lines = ["## Physical Properties", ""]
        for key, value in results.items():
            if isinstance(value, dict):
                lines.append(f"### {key}")
                for k, v in value.items():
                    lines.append(f"- {k}: {v}")
                lines.append("")
            else:
                lines.append(f"- **{key}**: {value}")
        lines.append("")
        return "\n".join(lines)

    def _render_validation(self, validation: dict[str, Any]) -> str:
        lines = ["## Validation", ""]
        checks = validation.get("checks", [])
        for check in checks:
            status = "✅" if check.get("passed") else "❌"
            lines.append(
                f"{status} {check.get('name', 'Unknown')}: {check.get('message', '')}"
            )
        lines.append("")
        return "\n".join(lines)

    def _render_literature(self, literature: dict[str, Any]) -> str:
        lines = ["## Literature Comparison", ""]
        comparisons = literature.get("comparisons", [])
        for comp in comparisons:
            lines.append(
                f"- {comp.get('property', 'N/A')}: "
                f"calculated = {comp.get('calculated', 'N/A')}, "
                f"reference = {comp.get('reference', 'N/A')} "
                f"({comp.get('source', 'N/A')})"
            )
        lines.append("")
        return "\n".join(lines)

    def _render_resources(self, resources: dict[str, Any]) -> str:
        lines = [
            "## Computational Resources",
            "",
            f"- **CPU time**: {resources.get('cpu_hours', 'N/A')} hours",
            f"- **Wall time**: {resources.get('walltime_hours', 'N/A')} hours",
            f"- **Memory**: {resources.get('memory_gb', 'N/A')} GB",
            f"- **Cores**: {resources.get('cores', 'N/A')}",
            "",
        ]
        return "\n".join(lines)

    def _render_reproducibility(self, data: dict[str, Any]) -> str:
        lines = [
            "## Reproducibility Information",
            "",
            f"- **Report generated**: {datetime.now().isoformat()}",
            f"- **Software version**: {data.get('software_version', 'N/A')}",
            f"- **Input hash**: {data.get('input_hash', 'N/A')}",
            f"- **Random seed**: {data.get('random_seed', 'N/A')}",
            "",
            "### Input files",
            "",
        ]
        for fname in data.get("input_files", []):
            lines.append(f"- `{fname}`")
        lines.append("")
        return "\n".join(lines)

    def _to_markdown(self) -> str:
        lines = [f"# Computational Report: {self.style.title()} Format", ""]
        for section in self.sections:
            lines.append(f"## {section.title}")
            lines.append("")
            lines.append(section.content)
            lines.append("")
        return "\n".join(lines)

    def _to_latex(self) -> str:
        lines = [
            "\\documentclass{article}",
            "\\usepackage{booktabs}",
            "\\begin{document}",
            "\\section*{Computational Report}",
        ]
        for section in self.sections:
            lines.append(f"\\subsection*{{{section.title}}}")
            # Very rough markdown-to-latex conversion
            content = section.content.replace("**", "\\textbf{").replace(
                "##", "\\subsection*{"
            )
            lines.append(content)
        lines.append("\\end{document}")
        return "\n".join(lines)

    def _to_json(self) -> str:
        data = {
            "style": self.style,
            "generated_at": datetime.now().isoformat(),
            "sections": [
                {"title": s.title, "content": s.content} for s in self.sections
            ],
        }
        return json.dumps(data, indent=2, ensure_ascii=False)

    def _to_html(self) -> str:
        lines = [
            "<!DOCTYPE html>",
            "<html><head><meta charset='utf-8'><title>Report</title></head><body>",
            "<h1>Computational Report</h1>",
        ]
        for section in self.sections:
            lines.append(f"<h2>{section.title}</h2>")
            # Very rough conversion
            content = section.content.replace("\n", "<br>")
            lines.append(f"<p>{content}</p>")
        lines.append("</body></html>")
        return "\n".join(lines)


class ReportComparator:
    """Generate side-by-side comparison reports from multiple named datasets."""

    def __init__(self, names: list[str], style: str = "full", fmt: str = "markdown"):
        self.names = names
        self.style = style
        self.format = fmt

    def generate(self, datasets: dict[str, dict[str, Any]]) -> str:
        sections: list[str] = []
        sections.append(self._header())
        sections.append(self._methods_comparison(datasets))
        sections.append(self._results_comparison(datasets))
        sections.append(self._convergence_comparison(datasets))
        sections.append(self._resources_comparison(datasets))
        return "\n\n".join(s for s in sections if s)

    def _header(self) -> str:
        if self.format == "markdown":
            return f"# Comparison Report: {' vs '.join(self.names)}"
        elif self.format == "json":
            return ""
        return f"\\section*{{Comparison: {' vs '.join(self.names)}}}"

    def _methods_comparison(self, datasets: dict[str, dict]) -> str:
        lines = ["## Methods Comparison", ""]
        lines.append("| Parameter | " + " | ".join(self.names) + " |")
        lines.append("|-----------|" + "|".join(["-------"] * len(self.names)) + "|")
        param_keys = ["method", "functional", "encut", "kpoints", "pseudopotentials", "smearing"]
        for key in param_keys:
            vals = []
            for name in self.names:
                methods = datasets[name].get("methods", {})
                vals.append(str(methods.get(key, "N/A")))
            lines.append(f"| {key} | " + " | ".join(vals) + " |")
        return "\n".join(lines)

    def _results_comparison(self, datasets: dict[str, dict]) -> str:
        lines = ["## Results Comparison", ""]
        lines.append("| Property | " + " | ".join(self.names) + " |")
        lines.append("|----------|" + "|".join(["-------"] * len(self.names)) + "|")
        all_keys: set[str] = set()
        for ds in datasets.values():
            all_keys.update(ds.get("results", {}).keys())
        for key in sorted(all_keys):
            vals = []
            for name in self.names:
                val = datasets[name].get("results", {}).get(key, "N/A")
                vals.append(str(val))
            lines.append(f"| {key} | " + " | ".join(vals) + " |")
        return "\n".join(lines)

    def _convergence_comparison(self, datasets: dict[str, dict]) -> str:
        lines = ["## Convergence Comparison", ""]
        conv_keys = ["energy", "n_iterations", "n_electronic", "converged"]
        lines.append("| Metric | " + " | ".join(self.names) + " |")
        lines.append("|--------|" + "|".join(["-------"] * len(self.names)) + "|")
        for key in conv_keys:
            vals = []
            for name in self.names:
                val = datasets[name].get("convergence", {}).get(key, "N/A")
                vals.append(str(val))
            lines.append(f"| {key} | " + " | ".join(vals) + " |")
        return "\n".join(lines)

    def _resources_comparison(self, datasets: dict[str, dict]) -> str:
        lines = ["## Resource Comparison", ""]
        res_keys = ["cpu_hours", "walltime_hours", "memory_gb", "cores"]
        lines.append("| Resource | " + " | ".join(self.names) + " |")
        lines.append("|----------|" + "|".join(["-------"] * len(self.names)) + "|")
        for key in res_keys:
            vals = []
            for name in self.names:
                val = datasets[name].get("resources", {}).get(key, "N/A")
                vals.append(str(val))
            lines.append(f"| {key} | " + " | ".join(vals) + " |")
        return "\n".join(lines)


class ReportTool(HuginnTool):
    """Generate computational reports from simulation results."""

    name = "report_tool"
    category = "core"
    profile = ToolProfile(phases=frozenset({ResearchPhase.REPORTING}))
    description = (
        "Automatically generate computational reports (Markdown/LaTeX/HTML/JSON) "
        "from DFT/MD simulation results, including methods, structures, convergence, "
        "physical properties, and reproducibility information."
    )
    input_schema = ReportToolInput

    def is_read_only(self, args: ReportToolInput) -> bool:
        return args.action in ["generate", "compare"]

    async def call(self, args: ReportToolInput, context: ToolContext) -> ToolResult:
        if args.action == "generate":
            return self._generate(args)
        elif args.action == "compare":
            return self._compare(args)
        elif args.action == "export":
            return self._export(args)
        elif args.action == "compile_pdf":
            return self._do_compile_pdf(args)
        return ToolResult(
            data=None, success=False, error=f"Unknown action: {args.action}"
        )

    def _do_compile_pdf(self, args: ReportToolInput) -> ToolResult:
        """把 tex_source 编译成 PDF, 返回 base64 + log.

        engine 不存在时返回 success=False + 安装提示, 不抛异常.
        """
        import base64
        import shutil
        import subprocess
        import tempfile

        if not args.tex_source:
            return ToolResult(
                data=None,
                success=False,
                error="compile_pdf requires tex_source.",
            )

        engine = args.engine
        if shutil.which(engine) is None:
            return ToolResult(
                data=None,
                success=False,
                error=(
                    f"LaTeX engine '{engine}' not found on PATH. "
                    f"Install TeX Live or MiKTeX and ensure {engine} is available."
                ),
            )

        with tempfile.TemporaryDirectory() as tmp:
            tex_file = Path(tmp) / "report.tex"
            tex_file.write_text(args.tex_source, encoding="utf-8")

            errors: list[str] = []
            log_text = ""
            for pass_num in (1, 2):
                try:
                    proc = subprocess.run(
                        [
                            engine,
                            "-interaction=nonstopmode",
                            "-halt-on-error",
                            str(tex_file),
                        ],
                        cwd=tmp,
                        capture_output=True,
                        text=True,
                        timeout=120,
                    )
                except subprocess.TimeoutExpired:
                    return ToolResult(
                        data=None,
                        success=False,
                        error=f"{engine} timed out on pass {pass_num}.",
                    )
                if proc.returncode != 0:
                    errors.append(
                        f"Pass {pass_num} failed (exit {proc.returncode}): "
                        f"{proc.stderr[:500] if proc.stderr else proc.stdout[:500]}"
                    )
                    break

            log_file = Path(tmp) / "report.log"
            if log_file.exists():
                log_text = log_file.read_text(errors="ignore")[:5000]

            pdf_file = Path(tmp) / "report.pdf"
            if not pdf_file.exists() or errors:
                return ToolResult(
                    data={
                        "engine": engine,
                        "success": False,
                        "log": log_text,
                        "errors": errors,
                    },
                    success=False,
                    error="; ".join(errors) if errors else "PDF not generated.",
                )

            pdf_bytes = pdf_file.read_bytes()
            pdf_b64 = base64.b64encode(pdf_bytes).decode("ascii")
            return ToolResult(
                data={
                    "pdf_base64": pdf_b64,
                    "pdf_size_bytes": len(pdf_bytes),
                    "log": log_text,
                    "engine": engine,
                    "success": True,
                    "errors": [],
                    "message": f"PDF compiled with {engine}, {len(pdf_bytes)} bytes.",
                },
                success=True,
            )

    def _generate(self, args: ReportToolInput) -> ToolResult:
        data = args.workflow_results or {}
        if args.calculation_dir:
            data = self._scan_directory(args.calculation_dir, data)

        generator = ReportGenerator(style=args.style, fmt=args.format)
        report = generator.generate(data)

        if args.output_path:
            path = Path(args.output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(report, encoding="utf-8")
            return ToolResult(
                data={"report": report[:500], "saved_to": str(path)},
                success=True,
            )

        return ToolResult(data={"report": report}, success=True)

    def _compare(self, args: ReportToolInput) -> ToolResult:
        datasets = args.comparison_datasets or {}
        if not datasets and args.workflow_results:
            # Try to extract comparison_datasets from workflow_results
            datasets = args.workflow_results.get("comparison_datasets", {})
        if not datasets:
            return ToolResult(
                data=None,
                success=False,
                error="comparison_datasets required for compare action (dict of named result sets)",
            )
        if len(datasets) < 2:
            return ToolResult(
                data=None,
                success=False,
                error="At least 2 datasets required for comparison",
            )

        names = list(datasets.keys())
        gen = ReportComparator(names, style=args.style, fmt=args.format)
        report = gen.generate(datasets)

        if args.output_path:
            path = Path(args.output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(report, encoding="utf-8")
            return ToolResult(
                data={"report_preview": report[:500], "saved_to": str(path)},
                success=True,
            )
        return ToolResult(data={"report": report}, success=True)

    def _export(self, args: ReportToolInput) -> ToolResult:
        return self._generate(args)

    def _scan_directory(
        self, calc_dir: str, existing: dict[str, Any]
    ) -> dict[str, Any]:
        """Scan a calculation directory for common output files."""
        path = Path(calc_dir)
        data = dict(existing)

        # Look for common files
        files = list(path.glob("*"))
        data["input_files"] = [
            f.name
            for f in files
            if f.suffix in {".incar", ".poscar", ".kpoints", ".lammps"}
        ]
        data["output_files"] = [
            f.name for f in files if f.suffix in {".outcar", ".oszicar", ".xml", ".log"}
        ]

        # Try to extract basic info from OUTCAR-like files
        for f in files:
            if f.name.upper() == "OUTCAR":
                text = f.read_text(errors="ignore")
                if "ENCUT" in text:
                    for line in text.split("\n")[:100]:
                        if "ENCUT" in line and "=" in line:
                            parts = line.split("=")
                            if len(parts) >= 2:
                                data.setdefault("methods", {})["encut"] = (
                                    parts[-1].strip().split()[0]
                                )
                            break
                break

        return data

    def estimate_cost(self, args: ReportToolInput) -> dict[str, float] | None:
        return {"cpu_hours": 0.0, "gpu_hours": 0.0, "walltime_hours": 0.01}
