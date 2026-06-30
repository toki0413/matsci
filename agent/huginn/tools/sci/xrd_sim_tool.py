"""XRD pattern simulation tool — calculate diffractograms from crystal structures.

Wraps pymatgen's XRDCalculator to give the agent forward-simulation of powder
XRD patterns from a Structure, plus experimental pattern parsing and comparison.
Complementary to characterization_tool (which only processes experimental data).
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool, ResearchPhase, ToolProfile
from huginn.types import ToolContext, ToolResult, ValidationResult


class XrdSimToolInput(BaseModel):
    action: Literal["simulate_xrd", "parse_pattern", "compare_patterns", "index_peaks"] = Field(
        ..., description="XRD action to perform."
    )
    file_path: str | None = Field(
        default=None,
        description="Path to structure file (simulate_xrd, index_peaks) or XRD data CSV (parse_pattern).",
    )
    structure_str: str | None = Field(
        default=None,
        description="Inline CIF/POSCAR string (alternative to file_path for simulate_xrd).",
    )
    wavelength: float = Field(
        default=1.5406,
        description="X-ray wavelength in Angstroms (1.5406 = Cu Kα).",
    )
    two_theta_min: float = Field(default=10.0, description="Minimum 2θ in degrees.")
    two_theta_max: float = Field(default=90.0, description="Maximum 2θ in degrees.")
    tolerance: float = Field(
        default=0.5,
        description="Peak matching tolerance in degrees 2θ (compare_patterns).",
    )
    simulated_peaks: list[dict] | None = Field(
        default=None,
        description="Pre-computed simulated peaks [{two_theta, intensity, hkl}] for compare_patterns.",
    )
    experimental_file: str | None = Field(
        default=None,
        description="Path to experimental XRD CSV for compare_patterns.",
    )
    peaks: list[float] | None = Field(
        default=None,
        description="Observed peak 2θ positions for index_peaks.",
    )


class XrdSimTool(HuginnTool):
    """Simulate powder XRD patterns from crystal structures and compare with experiments."""

    name = "xrd_sim_tool"
    category = "sci"
    profile = ToolProfile(
        cost_tier="light",
        phases=frozenset({ResearchPhase.VALIDATION, ResearchPhase.REPORTING}),
        light_alternatives=("characterization_tool",),
    )
    description = (
        "Simulate powder XRD patterns from crystal structures (pymatgen XRDCalculator), "
        "parse experimental XRD data files, compare simulated vs experimental patterns, "
        "and index peaks to Miller indices."
    )
    input_schema = XrdSimToolInput

    def is_read_only(self, args: XrdSimToolInput) -> bool:
        return True

    async def validate_input(
        self, args: XrdSimToolInput, context: ToolContext | None = None
    ) -> ValidationResult:
        if args.action == "simulate_xrd" and not args.file_path and not args.structure_str:
            return ValidationResult(
                result=False,
                message="simulate_xrd requires file_path or structure_str.",
            )
        if args.action == "parse_pattern" and not args.file_path:
            return ValidationResult(
                result=False,
                message="parse_pattern requires file_path (CSV data file).",
            )
        if args.action == "compare_patterns":
            if not args.simulated_peaks or (not args.experimental_file and not args.peaks):
                return ValidationResult(
                    result=False,
                    message="compare_patterns requires simulated_peaks and (experimental_file or peaks).",
                )
        if args.action == "index_peaks":
            if not args.peaks or (not args.file_path and not args.structure_str):
                return ValidationResult(
                    result=False,
                    message="index_peaks requires peaks and (file_path or structure_str).",
                )
        return ValidationResult(result=True)

    async def call(
        self, args: dict[str, Any], context: ToolContext | None = None
    ) -> ToolResult:
        input_data = XrdSimToolInput(**args)

        if input_data.action == "simulate_xrd":
            return self._simulate_xrd(input_data)
        elif input_data.action == "parse_pattern":
            return self._parse_pattern(input_data)
        elif input_data.action == "compare_patterns":
            return self._compare_patterns(input_data)
        elif input_data.action == "index_peaks":
            return self._index_peaks(input_data)
        return ToolResult(data=None, success=False, error=f"Unknown action: {input_data.action}")

    def _load_structure(self, input_data: XrdSimToolInput):
        """Load pymatgen Structure from file or string."""
        try:
            from pymatgen.core import Structure
        except ImportError:
            return None, ToolResult(
                data=None,
                success=False,
                error="pymatgen is required. Install with: pip install pymatgen",
            )
        try:
            if input_data.structure_str:
                from io import StringIO
                return Structure.from_file(StringIO(input_data.structure_str)), None
            return Structure.from_file(input_data.file_path), None
        except Exception as e:
            return None, ToolResult(
                data=None,
                success=False,
                error=f"Failed to load structure: {e}",
            )

    def _simulate_xrd(self, input_data: XrdSimToolInput) -> ToolResult:
        try:
            from pymatgen.analysis.diffraction.xrd import XRDCalculator
        except ImportError:
            return ToolResult(
                data=None,
                success=False,
                error="pymatgen is required for XRD simulation. Install with: pip install pymatgen",
            )

        struct, err = self._load_structure(input_data)
        if err is not None:
            return err

        try:
            calc = XRDCalculator(wavelength=input_data.wavelength)
            pattern = calc.get_pattern(
                struct,
                two_theta_range=(input_data.two_theta_min, input_data.two_theta_max),
            )
        except Exception as e:
            return ToolResult(
                data=None,
                success=False,
                error=f"XRDCalculator failed: {e}",
            )

        peaks = []
        for i in range(len(pattern.x)):
            hkl_list = []
            if hasattr(pattern, "hkls") and i < len(pattern.hkls):
                for hkl_dict in pattern.hkls[i]:
                    hkl_list.append(hkl_dict.get("hkl", []))
            peaks.append({
                "two_theta": round(float(pattern.x[i]), 4),
                "intensity": round(float(pattern.y[i]), 2),
                "hkls": hkl_list,
            })

        return ToolResult(
            data={
                "peaks": peaks,
                "wavelength": input_data.wavelength,
                "structure": struct.composition.reduced_formula,
                "n_peaks": len(peaks),
            },
            success=True,
        )

    def _parse_pattern(self, input_data: XrdSimToolInput) -> ToolResult:
        try:
            from scipy.signal import find_peaks
        except ImportError:
            return ToolResult(
                data=None,
                success=False,
                error="scipy is required for peak detection. Install with: pip install scipy",
            )

        try:
            data = np.loadtxt(input_data.file_path, delimiter=",", comments="#")
        except Exception as e:
            return ToolResult(
                data=None,
                success=False,
                error=f"Failed to parse XRD data file: {e}",
            )

        if data.ndim == 1:
            data = data.reshape(-1, 2)
        two_theta = data[:, 0]
        intensity = data[:, 1]

        # Height-based peak detection at 5% of max
        threshold = 0.05 * float(np.max(intensity))
        peak_indices, _ = find_peaks(intensity, height=threshold, distance=5)
        peak_positions = [round(float(two_theta[i]), 4) for i in peak_indices]

        return ToolResult(
            data={
                "two_theta": two_theta.tolist(),
                "intensity": intensity.tolist(),
                "peaks": peak_positions,
                "n_peaks": len(peak_positions),
            },
            success=True,
        )

    def _compare_patterns(
        self, input_data: XrdSimToolInput
    ) -> ToolResult:
        sim_peaks = input_data.simulated_peaks or []
        sim_positions = [p["two_theta"] for p in sim_peaks]

        # Get experimental peak positions
        if input_data.experimental_file:
            exp_result = self._parse_pattern(input_data)
            if not exp_result.success:
                return exp_result
            exp_positions = exp_result.data["peaks"]
        else:
            exp_positions = list(input_data.peaks or [])

        tol = input_data.tolerance
        matched = []
        unmatched_exp = []
        for ep in exp_positions:
            best_match = None
            best_dist = tol
            for sp in sim_positions:
                dist = abs(ep - sp)
                if dist < best_dist:
                    best_dist = dist
                    best_match = sp
            if best_match is not None:
                hkl = next(
                    (p["hkls"] for p in sim_peaks if p["two_theta"] == best_match),
                    [],
                )
                matched.append({
                    "experimental": ep,
                    "simulated": best_match,
                    "delta": round(best_dist, 4),
                    "hkls": hkl,
                })
            else:
                unmatched_exp.append(ep)

        unmatched_sim = [sp for sp in sim_positions if sp not in {m["simulated"] for m in matched}]

        # Simple Rwp-like figure of merit (lower = better match)
        n_matched = len(matched)
        n_total = len(exp_positions) + len(sim_positions)
        overlap = (2 * n_matched / n_total) if n_total > 0 else 0.0

        return ToolResult(
            data={
                "matched_peaks": matched,
                "unmatched_experimental": unmatched_exp,
                "unmatched_simulated": unmatched_sim,
                "overlap_ratio": round(overlap, 4),
                "n_matched": n_matched,
                "n_experimental": len(exp_positions),
                "n_simulated": len(sim_positions),
            },
            success=True,
        )

    def _index_peaks(self, input_data: XrdSimToolInput) -> ToolResult:
        """Assign Miller indices to observed peaks by matching against simulated pattern."""
        try:
            from pymatgen.analysis.diffraction.xrd import XRDCalculator
        except ImportError:
            return ToolResult(
                data=None,
                success=False,
                error="pymatgen is required for peak indexing. Install with: pip install pymatgen",
            )

        struct, err = self._load_structure(input_data)
        if err is not None:
            return err

        try:
            calc = XRDCalculator(wavelength=input_data.wavelength)
            pattern = calc.get_pattern(
                struct,
                two_theta_range=(input_data.two_theta_min, input_data.two_theta_max),
            )
        except Exception as e:
            return ToolResult(
                data=None,
                success=False,
                error=f"XRDCalculator failed: {e}",
            )

        # Build lookup: 2θ -> hkl list
        sim_lookup = {}
        for i in range(len(pattern.x)):
            tt = round(float(pattern.x[i]), 4)
            hkls = []
            if hasattr(pattern, "hkls") and i < len(pattern.hkls):
                for hkl_dict in pattern.hkls[i]:
                    hkls.append(hkl_dict.get("hkl", []))
            sim_lookup[tt] = hkls

        sim_positions = sorted(sim_lookup.keys())
        observed = input_data.peaks or []
        tol = input_data.tolerance

        indexed = []
        for obs_tt in observed:
            best_match = None
            best_dist = tol
            for sp in sim_positions:
                dist = abs(obs_tt - sp)
                if dist < best_dist:
                    best_dist = dist
                    best_match = sp
            if best_match is not None:
                indexed.append({
                    "two_theta": obs_tt,
                    "hkl": sim_lookup[best_match],
                    "simulated_2theta": best_match,
                    "delta": round(best_dist, 4),
                })
            else:
                indexed.append({
                    "two_theta": obs_tt,
                    "hkl": None,
                    "simulated_2theta": None,
                    "delta": None,
                })

        return ToolResult(
            data={
                "indexed_peaks": indexed,
                "structure": struct.composition.reduced_formula,
                "n_indexed": sum(1 for p in indexed if p["hkl"] is not None),
                "n_observed": len(observed),
            },
            success=True,
        )
