#!/usr/bin/env python3
"""
Computational Chemistry Diagnosis Tool.

Uses the Sobko knowledge base to diagnose common errors in
quantum chemistry and molecular dynamics calculations.

This tool provides troubleshooting guidance based on authoritative
computational chemistry sources (Sobko database).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool
from huginn.types import ToolContext, ToolResult


class DiagnoseInput(BaseModel):
    error_message: str = Field(
        ..., description="Error message or symptom from the calculation"
    )
    software: str | None = Field(
        default=None,
        description="Software being used (e.g., Gaussian, ORCA, VASP, LAMMPS)",
    )
    calculation_type: str | None = Field(
        default=None, description="Type of calculation (e.g., DFT, MD, TDDFT)"
    )
    context: str | None = Field(
        default=None, description="Additional context about the calculation setup"
    )


class DiagnoseTool(HuginnTool):
    """Diagnose computational chemistry errors using domain knowledge."""

    name = "diagnose_tool"
    description = (
        "Diagnose errors in quantum chemistry or MD calculations using "
        "authoritative computational chemistry knowledge. Provide the error message, "
        "software name, and calculation type for targeted troubleshooting."
    )
    input_schema = DiagnoseInput

    def __init__(self, knowledge_base_path: str | None = None):
        super().__init__()
        self._kb: dict[str, Any] = {}
        self._load_kb(knowledge_base_path)

    def _load_kb(self, path: str | None) -> None:
        """Load troubleshooting knowledge base."""
        if path and Path(path).exists():
            kb_path = Path(path)
        else:
            # Try to find Sobko troubleshooting data relative to repo root
            repo_root = Path(__file__).resolve().parent.parent.parent.parent
            kb_path = (
                repo_root
                / "Sobko_MCP_project"
                / "advanced_optimization"
                / "troubleshooting_by_software.json"
            )

        if kb_path.exists():
            with kb_path.open("r", encoding="utf-8") as f:
                self._kb = json.load(f)

    def is_read_only(self, args: DiagnoseInput) -> bool:
        return True

    async def call(self, args: DiagnoseInput, context: ToolContext) -> ToolResult:
        error_lower = args.error_message.lower()
        software = (args.software or "").lower()

        findings: list[dict[str, Any]] = []

        # Search in software-specific troubleshooting entries
        if software and software in {k.lower() for k in self._kb}:
            for sw_name, entries in self._kb.items():
                if sw_name.lower() == software:
                    for entry in entries:
                        entry_text = entry.get("text", "").lower()
                        # Score by keyword overlap
                        score = sum(
                            1
                            for word in error_lower.split()
                            if len(word) > 3 and word in entry_text
                        )
                        if score > 0:
                            findings.append(
                                {
                                    "source": sw_name,
                                    "title": entry.get("title", ""),
                                    "text": entry.get("text", ""),
                                    "score": score,
                                    "relevance": "high" if score >= 3 else "medium",
                                }
                            )

        # Also search across all software if no specific match or few results
        if len(findings) < 3:
            for sw_name, entries in self._kb.items():
                if software and sw_name.lower() == software:
                    continue  # Already searched
                for entry in entries:
                    entry_text = entry.get("text", "").lower()
                    score = sum(
                        1
                        for word in error_lower.split()
                        if len(word) > 3 and word in entry_text
                    )
                    if score >= 2:
                        findings.append(
                            {
                                "source": sw_name,
                                "title": entry.get("title", ""),
                                "text": entry.get("text", ""),
                                "score": score,
                                "relevance": "medium",
                            }
                        )

        # Sort by score
        findings.sort(key=lambda x: -x["score"])
        findings = findings[:10]

        # Build response
        if not findings:
            return ToolResult(
                data={
                    "query": args.error_message,
                    "software": args.software,
                    "findings": [],
                    "general_advice": self._general_advice(args),
                },
                success=True,
            )

        return ToolResult(
            data={
                "query": args.error_message,
                "software": args.software,
                "calculation_type": args.calculation_type,
                "findings": findings,
                "general_advice": self._general_advice(args),
                "recommended_next_steps": self._next_steps(args, findings),
            },
            success=True,
        )

    def _general_advice(self, args: DiagnoseInput) -> list[str]:
        """Provide general troubleshooting advice based on calculation type."""
        advice = []
        calc = (args.calculation_type or "").lower()
        sw = (args.software or "").lower()

        if "scf" in calc or "dft" in calc:
            advice.extend(
                [
                    "Check initial guess quality — read converged wavefunction from previous calculation if available",
                    "Try different convergence algorithms (DIIS → RCA → EDIIS in Gaussian; DIIS → SOSCF in ORCA)",
                    "Verify that your geometry is chemically reasonable (no overlapping atoms)",
                    "For metallic systems, ensure proper smearing and k-point density",
                ]
            )

        if "md" in calc or "molecular dynamics" in calc:
            advice.extend(
                [
                    "Check energy conservation in NVE ensemble (drift should be < 1% over 1 ns)",
                    "Verify force field parameters — especially for non-standard residues or ions",
                    "Inspect trajectory for atom clashes or unusual bond breaking",
                    "Ensure proper thermostat/barostat coupling constants (not too aggressive)",
                ]
            )

        if "td" in calc or "excited" in calc:
            advice.extend(
                [
                    "Increase number of roots if target state is higher in energy",
                    "Check for root flipping — the state ordering may change along a coordinate",
                    "For charge transfer states, range-separated functionals (CAM-B3LYP, ωB97X) are often necessary",
                    "Verify that oscillator strengths are physically reasonable",
                ]
            )

        # FEA / Solid Mechanics
        if "fea" in calc or "structural" in calc or "solid" in calc:
            advice.extend(
                [
                    "Check mesh quality — aspect ratio, skewness, and Jacobian distortion",
                    "Verify boundary conditions are statically admissible (no over/under-constraint)",
                    "Check material property units — GPa vs Pa vs MPa is a common source of error",
                    "For contact problems, ensure proper master/slave pairing and penalty stiffness",
                ]
            )

        if "cpfem" in calc or "crystal plasticity" in calc:
            advice.extend(
                [
                    "Verify crystal orientation input format (Euler angles, Rodrigues, or quaternion)",
                    "Check that slip system definitions match the crystal structure (fcc, bcc, hcp)",
                    "Ensure mesh resolves grain boundaries adequately",
                    "For spectral solvers (DAMASK), check FFT grid resolution",
                ]
            )

        # CFD / Fluid Mechanics
        if "cfd" in calc or "fluid" in calc:
            advice.extend(
                [
                    "Check mesh quality — orthogonality, skewness, and y+ values",
                    "Verify boundary condition consistency (mass flux balance)",
                    "For transient simulations, ensure CFL condition is satisfied",
                    "Check that turbulence model is appropriate for the flow regime",
                ]
            )

        if "gaussian" in sw:
            advice.append(
                "Check Gaussian output for 'Error termination' lines — they usually indicate the specific problem"
            )

        if "orca" in sw:
            advice.append(
                "Check ORCA output for 'ABORTING THE RUN' and preceding warnings"
            )

        if "vasp" in sw:
            advice.extend(
                [
                    "Check for 'ZBRENT' or 'EDDDAV' errors — usually SCF convergence issues",
                    "Verify ENCUT ≥ max(ENMAX) across all species",
                    "For magnetic systems, ensure correct MAGMOM initialization",
                ]
            )

        if "abaqus" in sw:
            advice.extend(
                [
                    "Check .dat and .msg files for detailed error diagnostics",
                    "'Too many attempts' → reduce increment size or improve initial guess",
                    "'Negative eigenvalues' → check for buckling or material instability",
                    "'Excessive distortion' → check element quality or material properties",
                ]
            )

        if "ansys" in sw:
            advice.extend(
                [
                    "Check .err and .out files for specific error codes",
                    "'Solution diverges' → check contact settings, element distortion, or material units",
                    "Verify license availability for required features",
                ]
            )

        if "openfoam" in sw:
            advice.extend(
                [
                    "Check log.simpleFoam (or corresponding solver log) for residual history",
                    "'Floating point exception' → check boundary conditions, initial fields, or mesh quality",
                    "'Continuity error' → check pressure-velocity coupling or boundary flux consistency",
                    "Run checkMesh to identify mesh quality issues",
                ]
            )

        if "fluent" in sw:
            advice.extend(
                [
                    "Check console output for divergence warnings",
                    "Reduce under-relaxation factors if solution diverges",
                    "Verify mesh quality metrics (skewness < 0.85, orthogonal quality > 0.15)",
                ]
            )

        if not advice:
            advice = [
                "Check input file syntax carefully — most errors are simple typos",
                "Verify that all required files (POTCAR, basis sets, topology) are present and compatible",
                "Try running on a minimal test system to isolate the problem",
                "Consult the software manual for the specific error message",
            ]

        return advice

    def _next_steps(self, args: DiagnoseInput, findings: list[dict]) -> list[str]:
        """Suggest next steps based on findings."""
        steps = []
        sw = (args.software or "").lower()

        # If findings mention specific keywords, suggest actions
        all_text = " ".join(f["text"] for f in findings).lower()

        if "converg" in all_text:
            steps.append("Adjust convergence criteria or switch algorithms")
        if "basis" in all_text or "diffuse" in all_text:
            steps.append(
                "Review basis set choice — consider adding diffuse functions for anions/Rydberg states"
            )
        if "smear" in all_text:
            steps.append(
                "Check smearing parameters — SIGMA too large can cause unphysical occupancies"
            )
        if "grid" in all_text:
            steps.append(
                "Increase integration grid quality (fine or ultrafine in Gaussian)"
            )
        if "memory" in all_text or "disk" in all_text:
            steps.append(
                "Increase memory/disk allocation or reduce system size for testing"
            )

        # FEA-specific next steps
        if "abaqus" in sw or "ansys" in sw or "comsol" in sw:
            if "mesh" in all_text or "distortion" in all_text or "element" in all_text:
                steps.append("Remesh with finer resolution or better element type")
            if "contact" in all_text:
                steps.append("Review contact pair definitions and penalty stiffness")
            if "convergence" in all_text and (
                "newton" in all_text or "equilibrium" in all_text
            ):
                steps.append("Reduce load increment or enable line search")
            if "material" in all_text:
                steps.append("Verify material property units and consistency")

        # CFD-specific next steps
        if "openfoam" in sw or "fluent" in sw:
            if "mesh" in all_text or "quality" in all_text or "skewness" in all_text:
                steps.append("Improve mesh quality or add inflation layers")
            if "boundary" in all_text or "flux" in all_text:
                steps.append(
                    "Check boundary condition types and values for consistency"
                )
            if "divergence" in all_text or "residual" in all_text:
                steps.append("Reduce under-relaxation factors or improve initial guess")
            if "courant" in all_text or "cfl" in all_text:
                steps.append("Reduce time step size to satisfy CFL condition")
            if "turbulence" in all_text or "yplus" in all_text:
                steps.append(
                    "Check y+ distribution and refine near-wall mesh if needed"
                )

        if not steps:
            steps.append("Implement the most relevant fix from the findings above")
            steps.append("Run a minimal test case to verify the fix works")

        return steps
