"""Descriptor tool — compute numerical descriptors for materials.

Provides composition-based features and, when optional dependencies are
available, structure-based descriptors (SOAP, MBTR, etc.) for machine-learning
screening workflows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool
from huginn.types import ToolContext, ToolResult


class DescriptorInput(BaseModel):
    action: Literal["composition", "soap"] = Field(...)
    formula: str | None = Field(default=None, description="Chemical formula")
    structure_file: str | None = Field(
        default=None, description="Path to structure file"
    )
    descriptor_type: Literal["composition_basic", "soap"] | None = Field(
        default="composition_basic"
    )


class DescriptorOutput(BaseModel):
    action: str
    descriptor_type: str
    count: int
    features: dict[str, Any]
    warnings: list[str] = []


class DescriptorTool(HuginnTool):
    """Compute material descriptors for ML screening."""

    name = "descriptor_tool"
    description = (
        "Compute composition and structure descriptors for machine-learning "
        "materials screening."
    )
    input_schema = DescriptorInput
    output_schema = DescriptorOutput
    read_only = True

    def is_read_only(self, args: DescriptorInput) -> bool:
        return True

    async def call(self, args: DescriptorInput, context: ToolContext) -> ToolResult:
        try:
            if args.action == "composition":
                features, warnings = self._composition_features(args)
            elif args.action == "soap":
                features, warnings = self._soap_descriptor(args)
            else:  # pragma: no cover
                raise ValueError(f"Unknown action: {args.action}")

            output = DescriptorOutput(
                action=args.action,
                descriptor_type=args.descriptor_type or args.action,
                count=len(features),
                features=features,
                warnings=warnings,
            )
            return ToolResult(data=output.model_dump(exclude_none=True))
        except Exception as exc:  # pragma: no cover
            return ToolResult(data=None, success=False, error=str(exc))

    def _composition_features(
        self, args: DescriptorInput
    ) -> tuple[dict[str, Any], list[str]]:
        formula = args.formula
        if not formula and args.structure_file:
            path = Path(args.structure_file)
            if not path.exists():
                raise FileNotFoundError(f"Structure file not found: {path}")
            try:
                from pymatgen.core import Structure

                structure = Structure.from_file(str(path))
                formula = structure.formula.replace(" ", "")
            except Exception as exc:
                raise RuntimeError(f"Could not read structure file: {exc}") from exc
        if not formula:
            raise ValueError("Provide formula or structure_file")

        try:
            from pymatgen.core import Composition

            comp = Composition(formula)
            total_atoms = float(comp.num_atoms)
            atomic_fractions = {
                str(el): float(frac) / total_atoms for el, frac in comp.items()
            }
            features = {
                "formula": comp.formula.replace(" ", ""),
                "num_elements": len(comp),
                "total_atoms": total_atoms,
                "atomic_fractions": atomic_fractions,
                "avg_atomic_mass": float(
                    sum(el.atomic_mass * frac for el, frac in comp.items())
                    / total_atoms
                ),
                "avg_electronegativity": _safe_average(
                    [el.X for el in comp.elements],
                    [float(frac) for frac in comp.values()],
                ),
            }
            return features, []
        except ImportError:
            return self._fallback_composition_features(formula)

    def _fallback_composition_features(
        self, formula: str
    ) -> tuple[dict[str, Any], list[str]]:
        warnings = [
            "pymatgen not installed; using lightweight fallback composition features"
        ]
        try:
            comp = _parse_formula(formula)
        except Exception as exc:
            raise ValueError(f"Could not parse formula '{formula}': {exc}") from exc
        total = sum(comp.values())
        fractions = {k: v / total for k, v in comp.items()}
        return {
            "formula": formula,
            "num_elements": len(comp),
            "total_atoms": float(total),
            "atomic_fractions": fractions,
            "avg_atomic_mass": None,
            "avg_electronegativity": None,
        }, warnings

    def _soap_descriptor(
        self, args: DescriptorInput
    ) -> tuple[dict[str, Any], list[str]]:
        if not args.structure_file:
            raise ValueError("soap descriptor requires structure_file")
        path = Path(args.structure_file)
        if not path.exists():
            raise FileNotFoundError(f"Structure file not found: {path}")
        try:
            from ase.io import read
            from dscribe.descriptors import SOAP

            atoms = read(str(path))
            species = sorted(set(atoms.get_chemical_symbols()))
            soap = SOAP(
                species=species,
                r_cut=6.0,
                n_max=8,
                l_max=6,
                average="outer",
                sparse=False,
            )
            vec = soap.create(atoms)
            return {
                "structure_file": str(path),
                "descriptor": "soap",
                "dim": int(vec.size),
                "mean": float(vec.mean()),
                "std": float(vec.std()),
            }, []
        except ImportError as exc:
            raise RuntimeError(
                "SOAP descriptor requires optional dependencies: dscribe, ase"
            ) from exc


def _safe_average(values: list[float], weights: list[float]) -> float | None:
    try:
        return float(np.average(values, weights=weights))
    except Exception:
        return None


def _parse_formula(formula: str) -> dict[str, int]:
    """Very small formula parser for fallback mode (e.g. 'H2O', 'CaCO3')."""
    import re

    tokens = re.findall(r"([A-Z][a-z]*)(\d*\.?\d*)", formula)
    if not tokens:
        raise ValueError(f"Invalid formula: {formula}")
    result: dict[str, int] = {}
    for elem, count in tokens:
        result[elem] = result.get(elem, 0) + (int(count) if count else 1)
    return result
