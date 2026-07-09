"""Sim-to-Real correction factor table — track calc vs experiment gaps.

DFT and other computational methods have systematic biases:
  - PBE bandgap underestimates by ~50%
  - GGA lattice constants overestimate by ~1-2%
  - LDA underestimates lattice constants by ~1-2%

This module maintains a correction table so the agent can:
  1. Report corrected values alongside raw calculated values
  2. Accumulate user-provided experimental references
  3. Learn correction factors from paired calc/exp observations

Usage:
    table = CorrectionTable.shared()
    corrected = table.apply_correction("Si", "band_gap", 0.61, "PBE")
    # Returns 1.17 (close to experimental 1.12 eV)
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CorrectionEntry:
    """One observed or known calc-to-experiment correction."""

    material: str  # e.g. "Si", "GaAs"
    property_name: str  # e.g. "band_gap", "lattice_constant"
    calc_value: float
    exp_value: float
    method: str = ""  # e.g. "PBE", "LDA", "HSE06", "GGA"
    source: str = ""  # citation or "user" or "builtin"
    registered_at: float = field(default_factory=time.time)

    @property
    def correction_factor(self) -> float:
        """Multiplicative factor: exp / calc. >1 means calc underestimates."""
        if self.calc_value == 0:
            return 1.0
        return self.exp_value / self.calc_value

    @property
    def offset(self) -> float:
        """Additive offset: exp - calc."""
        return self.exp_value - self.calc_value

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["correction_factor"] = self.correction_factor
        d["offset"] = self.offset
        return d


# Built-in corrections from literature. Only well-established values.
# ponytail: hardcoded common cases, extend via register() for the rest
_BUILTIN_CORRECTIONS: list[CorrectionEntry] = [
    # Band gaps (eV) — PBE famously underestimates
    CorrectionEntry("Si", "band_gap", 0.61, 1.12, "PBE", "builtin (exp)"),
    CorrectionEntry("Si", "band_gap", 1.17, 1.12, "HSE06", "builtin (exp)"),
    CorrectionEntry("GaAs", "band_gap", 0.55, 1.42, "PBE", "builtin (exp)"),
    CorrectionEntry("GaAs", "band_gap", 1.27, 1.42, "HSE06", "builtin (exp)"),
    CorrectionEntry("ZnO", "band_gap", 0.73, 3.37, "PBE", "builtin (exp)"),
    CorrectionEntry("TiO2", "band_gap", 1.83, 3.03, "PBE", "builtin (exp)"),
    # Lattice constants (Å) — GGA overestimates ~1-2%, LDA underestimates ~1-2%
    CorrectionEntry("Si", "lattice_constant", 5.47, 5.43, "PBE", "builtin (exp)"),
    CorrectionEntry("Si", "lattice_constant", 5.40, 5.43, "LDA", "builtin (exp)"),
    CorrectionEntry("Cu", "lattice_constant", 3.68, 3.61, "PBE", "builtin (exp)"),
    CorrectionEntry("Cu", "lattice_constant", 3.57, 3.61, "LDA", "builtin (exp)"),
    # Bulk modulus (GPa) — PBE tends to underestimate
    CorrectionEntry("Si", "bulk_modulus", 88.0, 99.0, "PBE", "builtin (exp)"),
]


class CorrectionTable:
    """Lookup table for sim-to-real corrections.

    Stores entries in memory + JSON file. Keyed by (material, property, method).
    If multiple entries exist for the same key, averages the correction factors.
    """

    _instance: CorrectionTable | None = None

    def __init__(self, persist_path: str | Path | None = None) -> None:
        self._entries: list[CorrectionEntry] = list(_BUILTIN_CORRECTIONS)
        self._persist_path: Path | None = Path(persist_path) if persist_path else None
        self._loaded = False
        if self._persist_path is not None:
            self._load()

    @classmethod
    def shared(cls) -> CorrectionTable:
        if cls._instance is None:
            cache_dir = os.environ.get("HUGINN_CACHE_DIR", "")
            if cache_dir:
                path = Path(cache_dir) / "corrections.json"
            else:
                path = Path.home() / ".huginn" / "corrections.json"
            cls._instance = cls(persist_path=path)
        return cls._instance

    def register(
        self,
        material: str,
        property_name: str,
        calc_value: float,
        exp_value: float,
        method: str = "",
        source: str = "user",
    ) -> CorrectionEntry:
        """Add a correction entry. Returns the created entry."""
        entry = CorrectionEntry(
            material=material,
            property_name=property_name,
            method=method,
            calc_value=calc_value,
            exp_value=exp_value,
            source=source,
        )
        self._entries.append(entry)
        self._save()
        logger.info(
            "Correction registered: %s %s (%s) calc=%.4f exp=%.4f factor=%.3f",
            material, property_name, method, calc_value, exp_value, entry.correction_factor,
        )
        return entry

    def get_corrections(
        self,
        material: str,
        property_name: str,
        method: str = "",
    ) -> list[CorrectionEntry]:
        """All matching entries for a (material, property) pair."""
        results = []
        for e in self._entries:
            if e.material != material:
                continue
            if e.property_name != property_name:
                continue
            if method and e.method and e.method != method:
                continue
            results.append(e)
        return results

    def get_avg_correction_factor(
        self,
        material: str,
        property_name: str,
        method: str = "",
    ) -> float | None:
        """Average multiplicative correction factor. None if no data."""
        entries = self.get_corrections(material, property_name, method)
        if not entries:
            return None
        return sum(e.correction_factor for e in entries) / len(entries)

    def apply_correction(
        self,
        material: str,
        property_name: str,
        calc_value: float,
        method: str = "",
    ) -> float:
        """Apply average correction factor to a calculated value.

        If no correction data exists, returns the raw value unchanged.
        """
        factor = self.get_avg_correction_factor(material, property_name, method)
        if factor is None:
            return calc_value
        return calc_value * factor

    def to_context_block(self) -> str:
        """Generate context string for agent prompts."""
        if not self._entries:
            return ""
        # Group by material+property for compactness
        seen: dict[str, list[CorrectionEntry]] = {}
        for e in self._entries:
            key = f"{e.material}/{e.property_name}"
            seen.setdefault(key, []).append(e)
        lines = ["### Sim-to-Real Correction Table:"]
        for key, entries in sorted(seen.items()):
            parts = []
            for e in entries:
                parts.append(
                    f"{e.method}: {e.calc_value:.2f}→{e.exp_value:.2f} (×{e.correction_factor:.2f})"
                )
            lines.append(f"  {key}: {'; '.join(parts)}")
        return "\n".join(lines)

    def list_materials(self) -> list[str]:
        return sorted({e.material for e in self._entries})

    def list_properties(self, material: str) -> list[str]:
        return sorted({
            e.property_name for e in self._entries if e.material == material
        })

    def summary(self) -> dict[str, Any]:
        return {
            "total_entries": len(self._entries),
            "materials": self.list_materials(),
            "properties": sorted({e.property_name for e in self._entries}),
        }

    def _save(self) -> None:
        if self._persist_path is None:
            return
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            # Only save user-registered entries, builtins are in code
            user_entries = [e for e in self._entries if e.source != "builtin (exp)"]
            data = {
                "version": "1.0",
                "saved_at": time.time(),
                "entries": [asdict(e) for e in user_entries],
            }
            self._persist_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            logger.debug("correction persist failed", exc_info=True)

    def _load(self) -> None:
        if self._persist_path is None or not self._persist_path.exists():
            return
        try:
            data = json.loads(self._persist_path.read_text(encoding="utf-8"))
            for e_dict in data.get("entries", []):
                self._entries.append(CorrectionEntry(
                    material=e_dict["material"],
                    property_name=e_dict["property_name"],
                    method=e_dict.get("method", ""),
                    calc_value=e_dict["calc_value"],
                    exp_value=e_dict["exp_value"],
                    source=e_dict.get("source", "user"),
                    registered_at=e_dict.get("registered_at", time.time()),
                ))
            loaded = len(data.get("entries", []))
            if loaded:
                logger.info("CorrectionTable: loaded %d user entries", loaded)
        except Exception:
            logger.debug("correction load failed", exc_info=True)

    def clear_user_entries(self) -> None:
        """Remove all non-builtin entries."""
        self._entries = [e for e in self._entries if e.source == "builtin (exp)"]
        self._save()
