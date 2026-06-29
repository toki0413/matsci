"""Dempster-Shafer evidence theory tool for multi-source evidence fusion.

Combines evidence from independent sources (DFT, MD, experiments, …) using
Dempster's rule of combination, and exposes belief/plausibility intervals,
pignistic probabilities, and conflict diagnostics. All operations are pure
— the tool never touches external state.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool, ResearchPhase, ToolProfile
from huginn.types import ToolContext, ToolResult, ValidationResult

# Sources rarely sum to exactly 1.0; give them a little slack.
_MASS_EPS = 1e-9


class EvidenceItem(BaseModel):
    """One focal element contributed by a source."""

    hypotheses: list[str] = Field(
        default_factory=list,
        description="Hypotheses this mass is assigned to. Empty list is "
        "shorthand for the full frame (ignorance).",
    )
    mass: float = Field(..., ge=0.0, le=1.0, description="Mass value in [0, 1].")
    source: str = Field(..., description="Name of the evidence source.")
    weight: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Reliability weight for the source (weighted_combine only).",
    )


class MassItem(BaseModel):
    """A focal element of a standalone mass function."""

    hypotheses: list[str] = Field(
        default_factory=list,
        description="Hypotheses this mass is on. Empty list = full frame.",
    )
    mass: float = Field(..., ge=0.0, le=1.0)


class EvidenceFusionToolInput(BaseModel):
    action: Literal[
        "combine",
        "belief_plausibility",
        "pignistic",
        "conflict_analysis",
        "weighted_combine",
    ] = Field(default="combine")
    evidence: list[EvidenceItem] | None = Field(
        default=None,
        description="Evidence items tagged by source. Required for combine, "
        "conflict_analysis, weighted_combine.",
    )
    mass_function: list[MassItem] | None = Field(
        default=None,
        description="A single mass function as focal elements. Required for "
        "belief_plausibility and pignistic.",
    )


class EvidenceFusionTool(HuginnTool):
    """Fuse multi-source evidence with Dempster-Shafer theory."""

    name = "evidence_fusion_tool"
    category = "sci"
    profile = ToolProfile(phases=frozenset({ResearchPhase.VALIDATION}))
    description = (
        "Combine evidence from multiple sources using Dempster-Shafer theory. "
        "Supports Dempster's combination rule, belief/plausibility intervals, "
        "the pignistic probability transform, pairwise conflict analysis, and "
        "weighted (discounted) combination for sources of differing reliability."
    )
    input_schema = EvidenceFusionToolInput
    read_only = True

    def is_read_only(self, args: EvidenceFusionToolInput) -> bool:
        return True

    async def validate_input(
        self, args: EvidenceFusionToolInput, context: ToolContext | None = None
    ) -> ValidationResult:
        if args.action in ("combine", "conflict_analysis", "weighted_combine"):
            if not args.evidence:
                return ValidationResult(
                    result=False,
                    message=f"{args.action} requires a non-empty 'evidence' list.",
                )
            sources = {ev.source for ev in args.evidence}
            # combine / conflict_analysis are inherently pairwise
            if args.action in ("combine", "conflict_analysis") and len(sources) < 2:
                return ValidationResult(
                    result=False,
                    message=f"{args.action} needs at least two distinct sources.",
                )
        elif args.action in ("belief_plausibility", "pignistic"):
            if not args.mass_function:
                return ValidationResult(
                    result=False,
                    message=f"{args.action} requires a non-empty 'mass_function'.",
                )
        return ValidationResult(result=True)

    async def call(
        self, args: dict[str, Any], context: ToolContext | None = None
    ) -> ToolResult:
        input_data = EvidenceFusionToolInput(**args)
        try:
            if input_data.action == "combine":
                return self._combine(input_data.evidence or [])
            if input_data.action == "weighted_combine":
                return self._weighted_combine(input_data.evidence or [])
            if input_data.action == "conflict_analysis":
                return self._conflict_analysis(input_data.evidence or [])
            if input_data.action == "belief_plausibility":
                return self._belief_plausibility(input_data.mass_function or [])
            if input_data.action == "pignistic":
                return self._pignistic(input_data.mass_function or [])
            return ToolResult(
                data=None,
                success=False,
                error=f"Unknown action: {input_data.action}",
            )
        except Exception as e:
            return ToolResult(
                data=None, success=False, error=f"Evidence fusion failed: {e}"
            )

    # ── mass-function helpers ─────────────────────────────────────────

    @staticmethod
    def _frame_from_evidence(evidence: list[EvidenceItem]) -> frozenset[str]:
        frame: set[str] = set()
        for ev in evidence:
            frame.update(ev.hypotheses)
        return frozenset(frame)

    @staticmethod
    def _frame_from_mass(items: list[MassItem]) -> frozenset[str]:
        frame: set[str] = set()
        for it in items:
            frame.update(it.hypotheses)
        return frozenset(frame)

    @staticmethod
    def _key(hypotheses: list[str], frame: frozenset[str]) -> frozenset[str]:
        # Empty hypothesis list = the whole frame (ignorance).
        return frame if not hypotheses else frozenset(hypotheses)

    def _build_source_mass(
        self, items: list[EvidenceItem], frame: frozenset[str]
    ) -> dict[frozenset[str], float]:
        mass: dict[frozenset[str], float] = {}
        for ev in items:
            k = self._key(ev.hypotheses, frame)
            mass[k] = mass.get(k, 0.0) + ev.mass
        total = sum(mass.values())
        if total > 1.0 + _MASS_EPS:
            # Over-committed source: renormalise so the maths stays well-defined.
            mass = {k: v / total for k, v in mass.items()}
            total = 1.0
        # Leftover mass is genuine ignorance — park it on the frame.
        if total < 1.0 - _MASS_EPS:
            mass[frame] = mass.get(frame, 0.0) + (1.0 - total)
        return mass

    def _build_mass_function(
        self, items: list[MassItem], frame: frozenset[str]
    ) -> dict[frozenset[str], float]:
        mass: dict[frozenset[str], float] = {}
        for it in items:
            k = self._key(it.hypotheses, frame)
            mass[k] = mass.get(k, 0.0) + it.mass
        total = sum(mass.values())
        if total > 1.0 + _MASS_EPS:
            mass = {k: v / total for k, v in mass.items()}
            total = 1.0
        if total < 1.0 - _MASS_EPS:
            mass[frame] = mass.get(frame, 0.0) + (1.0 - total)
        return mass

    @staticmethod
    def _combine_two(
        m1: dict[frozenset[str], float],
        m2: dict[frozenset[str], float],
    ) -> tuple[dict[frozenset[str], float], float]:
        """Dempster's rule for two mass functions.

        Returns (combined_mass, conflict_K). combined_mass is empty when the
        two sources are in total conflict (K == 1).
        """
        combined: dict[frozenset[str], float] = {}
        conflict = 0.0
        for b, mb in m1.items():
            for c, mc in m2.items():
                inter = b & c
                prod = mb * mc
                if not inter:
                    # Disjoint focal elements -> pure conflict.
                    conflict += prod
                else:
                    combined[inter] = combined.get(inter, 0.0) + prod
        if conflict >= 1.0 - _MASS_EPS:
            return {}, conflict
        norm = 1.0 - conflict
        for k in combined:
            combined[k] /= norm
        return combined, conflict

    @staticmethod
    def _serialize_mass(
        mass: dict[frozenset[str], float], frame: frozenset[str]
    ) -> list[dict[str, Any]]:
        # Highest mass first, then alphabetical for stable output.
        ordered = sorted(mass.items(), key=lambda kv: (-kv[1], sorted(kv[0])))
        return [
            {
                "hypotheses": sorted(k),
                "mass": round(v, 6),
                "is_frame": k == frame,
            }
            for k, v in ordered
        ]

    @staticmethod
    def _belief_plausibility_singletons(
        mass: dict[frozenset[str], float], frame: frozenset[str]
    ) -> dict[str, dict[str, Any]]:
        """Bel/Pl for every singleton hypothesis in the frame."""
        result: dict[str, dict[str, Any]] = {}
        for h in frame:
            singleton = frozenset({h})
            belief = 0.0
            plausibility = 0.0
            for b, mb in mass.items():
                if not b:
                    continue
                if b <= singleton:
                    belief += mb
                if b & singleton:
                    plausibility += mb
            result[h] = {
                "belief": round(belief, 6),
                "plausibility": round(plausibility, 6),
                "uncertainty_interval": [round(belief, 6), round(plausibility, 6)],
            }
        return result

    @staticmethod
    def _pignistic_probs(
        mass: dict[frozenset[str], float], frame: frozenset[str]
    ) -> dict[str, float]:
        """Smets' pignistic transform -> a Bayesian distribution for decisions."""
        probs: dict[str, float] = dict.fromkeys(frame, 0.0)
        for b, mb in mass.items():
            if not b or mb == 0.0:
                continue
            share = mb / len(b)
            for h in b:
                probs[h] += share
        return {h: round(p, 6) for h, p in probs.items()}

    @staticmethod
    def _interpret(
        conflict: float, plaus: dict[str, dict[str, Any]]
    ) -> str:
        if conflict >= 1.0 - _MASS_EPS:
            return (
                "Total conflict (K=1): sources are mutually exclusive, so the "
                "combination is not meaningful."
            )
        top = max(plaus.items(), key=lambda kv: kv[1]["plausibility"])
        h, info = top
        if conflict < 0.1:
            level = "low conflict, sources agree well"
        elif conflict < 0.3:
            level = "moderate conflict"
        elif conflict < 0.5:
            level = "notable conflict — interpret with care"
        else:
            level = "HIGH conflict — combination may be unreliable"
        return (
            f"Combined evidence favours '{h}' "
            f"(Bel={info['belief']}, Pl={info['plausibility']}). "
            f"Conflict K={conflict:.3f}: {level}."
        )

    # ── actions ───────────────────────────────────────────────────────

    def _combine(self, evidence: list[EvidenceItem]) -> ToolResult:
        frame = self._frame_from_evidence(evidence)
        if not frame:
            return ToolResult(
                data=None,
                success=False,
                error="No hypotheses found in evidence — frame of discernment is empty.",
            )
        by_source: dict[str, list[EvidenceItem]] = {}
        for ev in evidence:
            by_source.setdefault(ev.source, []).append(ev)
        sources = list(by_source.keys())
        source_masses = {
            s: self._build_source_mass(by_source[s], frame) for s in sources
        }

        # Fold sources together pairwise.
        combined = source_masses[sources[0]]
        last_conflict = 0.0
        for s in sources[1:]:
            combined, last_conflict = self._combine_two(combined, source_masses[s])
            if not combined:
                return ToolResult(
                    data=None,
                    success=False,
                    error=f"Total conflict (K=1) when combining with source '{s}'. "
                    "Sources are mutually exclusive.",
                )

        plaus = self._belief_plausibility_singletons(combined, frame)
        data: dict[str, Any] = {
            "action": "combine",
            "frame_of_discernment": sorted(frame),
            "sources": sources,
            "combined_mass": self._serialize_mass(combined, frame),
            "belief_plausibility": plaus,
            "conflict": round(last_conflict, 6),
            "interpretation": self._interpret(last_conflict, plaus),
        }
        if last_conflict > 0.5:
            data["warning"] = (
                f"High conflict (K={last_conflict:.3f} > 0.5). Run "
                "conflict_analysis to find the disagreeing source, or use "
                "weighted_combine to discount unreliable sources."
            )
        return ToolResult(data=data, success=True)

    def _weighted_combine(self, evidence: list[EvidenceItem]) -> ToolResult:
        frame = self._frame_from_evidence(evidence)
        if not frame:
            return ToolResult(
                data=None,
                success=False,
                error="No hypotheses found in evidence — frame of discernment is empty.",
            )
        by_source: dict[str, list[EvidenceItem]] = {}
        weights: dict[str, float] = {}
        for ev in evidence:
            by_source.setdefault(ev.source, []).append(ev)
            if ev.weight is not None and ev.source not in weights:
                weights[ev.source] = ev.weight
        # No weight supplied == fully trusted.
        for s in by_source:
            weights.setdefault(s, 1.0)

        # Discount each source, then combine.
        discounted: dict[str, dict[frozenset[str], float]] = {}
        for s, items in by_source.items():
            base = self._build_source_mass(items, frame)
            w = weights[s]
            if w >= 1.0 - _MASS_EPS:
                discounted[s] = base
                continue
            new_mass: dict[frozenset[str], float] = {}
            for k, v in base.items():
                if k == frame:
                    continue
                new_mass[k] = v * w
            # m'(Theta) = w * m(Theta) + (1 - w)
            new_mass[frame] = (
                new_mass.get(frame, 0.0) + base.get(frame, 0.0) * w + (1.0 - w)
            )
            discounted[s] = new_mass

        sources = list(by_source.keys())
        combined = discounted[sources[0]]
        last_conflict = 0.0
        for s in sources[1:]:
            combined, last_conflict = self._combine_two(combined, discounted[s])
            if not combined:
                return ToolResult(
                    data=None,
                    success=False,
                    error=f"Total conflict (K=1) when combining with source '{s}'.",
                )

        plaus = self._belief_plausibility_singletons(combined, frame)
        data: dict[str, Any] = {
            "action": "weighted_combine",
            "frame_of_discernment": sorted(frame),
            "sources": sources,
            "weights": {s: weights[s] for s in sources},
            "combined_mass": self._serialize_mass(combined, frame),
            "belief_plausibility": plaus,
            "conflict": round(last_conflict, 6),
            "interpretation": self._interpret(last_conflict, plaus),
        }
        if last_conflict > 0.5:
            data["warning"] = (
                f"High conflict (K={last_conflict:.3f} > 0.5) even after discounting."
            )
        return ToolResult(data=data, success=True)

    def _conflict_analysis(self, evidence: list[EvidenceItem]) -> ToolResult:
        frame = self._frame_from_evidence(evidence)
        if not frame:
            return ToolResult(
                data=None,
                success=False,
                error="No hypotheses found in evidence.",
            )
        by_source: dict[str, list[EvidenceItem]] = {}
        for ev in evidence:
            by_source.setdefault(ev.source, []).append(ev)
        sources = sorted(by_source.keys())
        masses = {
            s: self._build_source_mass(by_source[s], frame) for s in sources
        }

        n = len(sources)
        matrix: list[list[float]] = [[0.0] * n for _ in range(n)]
        worst: tuple[str | None, str | None, float] = (None, None, -1.0)
        per_source: dict[str, list[float]] = {s: [] for s in sources}

        for i in range(n):
            for j in range(i + 1, n):
                _, k = self._combine_two(masses[sources[i]], masses[sources[j]])
                matrix[i][j] = round(k, 6)
                matrix[j][i] = round(k, 6)
                per_source[sources[i]].append(k)
                per_source[sources[j]].append(k)
                if k > worst[2]:
                    worst = (sources[i], sources[j], k)

        avg = {
            s: round(sum(v) / len(v), 6) if v else 0.0
            for s, v in per_source.items()
        }
        suspect = max(avg.items(), key=lambda kv: kv[1])
        if suspect[1] < 1e-6:
            recommendation = (
                "No significant conflict detected — all sources are consistent."
            )
        else:
            recommendation = (
                f"Source '{suspect[0]}' has the highest average pairwise conflict "
                f"({suspect[1]}); it may be the least reliable. Consider "
                "discounting it via weighted_combine."
            )

        data = {
            "action": "conflict_analysis",
            "frame_of_discernment": sorted(frame),
            "sources": sources,
            "conflict_matrix": matrix,
            "average_conflict": avg,
            "most_conflicting_pair": (
                {
                    "sources": [worst[0], worst[1]],
                    "conflict": round(worst[2], 6),
                }
                if worst[0]
                else None
            ),
            "recommendation": recommendation,
        }
        return ToolResult(data=data, success=True)

    def _belief_plausibility(self, items: list[MassItem]) -> ToolResult:
        frame = self._frame_from_mass(items)
        if not frame:
            return ToolResult(
                data=None,
                success=False,
                error="No hypotheses found in mass_function.",
            )
        mass = self._build_mass_function(items, frame)
        bp = self._belief_plausibility_singletons(mass, frame)
        data = {
            "action": "belief_plausibility",
            "frame_of_discernment": sorted(frame),
            "mass_function": self._serialize_mass(mass, frame),
            "belief_plausibility": bp,
        }
        return ToolResult(data=data, success=True)

    def _pignistic(self, items: list[MassItem]) -> ToolResult:
        frame = self._frame_from_mass(items)
        if not frame:
            return ToolResult(
                data=None,
                success=False,
                error="No hypotheses found in mass_function.",
            )
        mass = self._build_mass_function(items, frame)
        probs = self._pignistic_probs(mass, frame)
        top = max(probs.items(), key=lambda kv: kv[1])
        data = {
            "action": "pignistic",
            "frame_of_discernment": sorted(frame),
            "mass_function": self._serialize_mass(mass, frame),
            "pignistic_probability": probs,
            "decision": {"top_hypothesis": top[0], "probability": top[1]},
        }
        return ToolResult(data=data, success=True)
