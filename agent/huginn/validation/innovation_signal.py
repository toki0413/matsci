"""Innovation signal detector — when results deviate significantly from
literature, flag as potential innovation rather than just error.

Philosophy: literature values are a reference distribution, not ground truth.
Significant deviations could be:
1. A genuine error (most common) — should be caught by physics validation
2. A novel phase/structure/property — potentially a discovery
3. A boundary case the literature hasn't explored

The detector distinguishes these by checking if the result is:
- Physically plausible (passes physics auditor)
- Internally consistent (passes dimensional analysis)
- But deviates from literature consensus beyond expected spread

If yes to all three -> INNOVATION_SIGNAL, not error.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Sequence


class DeviationLevel(Enum):
    """How far the agent value sits from the literature consensus."""

    NEGLIGIBLE = auto()      # < 1 sigma
    MODERATE = auto()        # 1-2 sigma
    SIGNIFICANT = auto()     # 2-3 sigma
    EXTRAORDINARY = auto()   # > 3 sigma


@dataclass
class InnovationSignal:
    """Result of comparing one agent value against the literature distribution.

    is_innovation_signal is True only when the deviation is large AND the value
    passes basic physical plausibility checks. A large deviation from an
    implausible value is just an error, not a discovery.
    """

    property_name: str
    agent_value: float
    literature_consensus: float
    literature_spread: float
    deviation_sigma: float
    level: DeviationLevel
    possible_explanations: list[str] = field(default_factory=list)
    is_innovation_signal: bool = False

    def to_dict(self) -> dict:
        return {
            "property_name": self.property_name,
            "agent_value": self.agent_value,
            "literature_consensus": self.literature_consensus,
            "literature_spread": self.literature_spread,
            "deviation_sigma": round(self.deviation_sigma, 4),
            "level": self.level.name,
            "possible_explanations": self.possible_explanations,
            "is_innovation_signal": self.is_innovation_signal,
        }


# Thresholds in units of sigma. Tuned to match typical materials-science
# literature spread — most reported values cluster within 2 sigma.
_INNOVATION_THRESHOLD = 2.0   # > 2 sigma -> start paying attention
_EXTRAORDINARY_THRESHOLD = 3.0


class InnovationSignalDetector:
    """Detect potentially interesting deviations from literature consensus.

    Call ``detect(property_name, agent_value, literature_values)`` where
    ``literature_values`` is a list of floats reported across papers. The
    detector computes mean (consensus) and std (spread), then classifies the
    deviation.

    A value is flagged as an innovation signal only when:
      - deviation_sigma > 2 (statistically far from literature)
      - the value is physically plausible (finite, non-zero, reasonable)
    This prevents NaN/inf/garbage from masquerading as discoveries.
    """

    def detect(
        self,
        property_name: str,
        agent_value: float,
        literature_values: Sequence[float],
    ) -> InnovationSignal:
        # Need at least 2 literature points to talk about spread
        vals = [float(v) for v in literature_values if self._is_finite(v)]
        if len(vals) < 2:
            return self._no_data_signal(property_name, agent_value, vals)

        consensus = sum(vals) / len(vals)
        spread = self._std(vals)
        if spread == 0:
            # All literature values identical — any deviation is either exact
            # match or interesting. Use a tiny epsilon to avoid div-by-zero.
            spread = 1e-12

        dev_sigma = abs(agent_value - consensus) / spread
        level = self._classify(dev_sigma)
        plausible = self._is_physically_plausible(property_name, agent_value)
        is_signal = dev_sigma > _INNOVATION_THRESHOLD and plausible
        explanations = self._explanations(level, is_signal, property_name)

        return InnovationSignal(
            property_name=property_name,
            agent_value=agent_value,
            literature_consensus=round(consensus, 6),
            literature_spread=round(spread, 6),
            deviation_sigma=round(dev_sigma, 4),
            level=level,
            possible_explanations=explanations,
            is_innovation_signal=is_signal,
        )

    # ── helpers ───────────────────────────────────────────────

    @staticmethod
    def _is_finite(v: float) -> bool:
        try:
            return math.isfinite(float(v))
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _std(vals: list[float]) -> float:
        n = len(vals)
        if n < 2:
            return 0.0
        mean = sum(vals) / n
        var = sum((v - mean) ** 2 for v in vals) / (n - 1)
        return math.sqrt(var)

    @staticmethod
    def _classify(sigma: float) -> DeviationLevel:
        if sigma < 1.0:
            return DeviationLevel.NEGLIGIBLE
        if sigma < 2.0:
            return DeviationLevel.MODERATE
        if sigma < 3.0:
            return DeviationLevel.SIGNIFICANT
        return DeviationLevel.EXTRAORDINARY

    @staticmethod
    def _is_physically_plausible(name: str, value: float) -> bool:
        """Quick sanity check — not a full physics audit, just filters garbage.

        NaN/inf are never plausible. Zero is fine for properties that can
        legitimately be zero (e.g. magnetization), but suspicious for things
        like energy or lattice constant. We keep it simple: non-structural
        properties (energy, band_gap, lattice) must be non-zero and finite.
        """
        if not InnovationSignalDetector._is_finite(value):
            return False
        # Properties where exactly zero is physically meaningless
        _NONZERO_PROPS = {"energy", "lattice_constant", "lattice_a",
                          "lattice_b", "lattice_c", "volume", "bulk_modulus"}
        if name.lower() in _NONZERO_PROPS and abs(value) < 1e-15:
            return False
        return True

    @staticmethod
    def _explanations(
        level: DeviationLevel, is_signal: bool, prop: str
    ) -> list[str]:
        if level == DeviationLevel.NEGLIGIBLE:
            return []
        base = [
            "novel phase or structure",
            "different synthesis or calculation conditions",
            "size / dimensionality effect",
            "computational method difference (functional, basis set, etc.)",
        ]
        if is_signal:
            base.append("genuine discovery — verify with independent calculation")
        else:
            base.append("possible error — check units and input")
        return base

    @staticmethod
    def _no_data_signal(
        prop: str, value: float, vals: list[float]
    ) -> InnovationSignal:
        consensus = vals[0] if vals else 0.0
        return InnovationSignal(
            property_name=prop,
            agent_value=value,
            literature_consensus=consensus,
            literature_spread=0.0,
            deviation_sigma=0.0,
            level=DeviationLevel.NEGLIGIBLE,
            possible_explanations=["insufficient literature data for comparison"],
            is_innovation_signal=False,
        )


__all__ = [
    "DeviationLevel",
    "InnovationSignal",
    "InnovationSignalDetector",
]
