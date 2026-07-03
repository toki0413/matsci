"""Cross-Modal Adapter (M5).

Bridges the textual and visual / data layers of the document graph.
Text blocks make claims ("a distinct peak at 2theta = 28.4 deg"); figures
and tables carry the ground-truth data points. This module pulls claims
out of text, binds them to the data extracted from figures/tables, and
cross-validates the two, emitting typed SUPPORTS / CONTRADICTS /
INCONCLUSIVE edges back into the graph.

The pipeline is deliberately tolerant: materials data is noisy, unit
conventions vary across papers, and a claim that's off by 0.3 degrees on
2theta is usually a rounding artefact, not a real contradiction. The
tolerance table encodes the domain-specific judgement about what "close
enough" means for each common metric.

An LLM client is optional. When supplied it produces richer claim
structures (conclusions, qualifiers); when absent we fall back to pure
regex, which is good enough for the common "value + unit" pattern.
"""

from __future__ import annotations

import json
import re
from typing import Any

from huginn.perception.doc_types import (
    DocumentElement,
    EdgeType,
    ElementType,
    GraphEdge,
)
from huginn.perception.document_graph import DocumentGraph


# ---------------------------------------------------------------------------
# Detection patterns.
#
# The numeric regex is intentionally wide -- papers mix "28.4 deg", "28.4deg",
# "28.4 degree", and the occasional "2theta = 28.4" with a unicode degree sign
# glued to the number. We capture the number and the unit token separately so
# the normaliser can fold variants later.
# ---------------------------------------------------------------------------
_NUMERIC_RE = re.compile(
    r"(\d+\.?\d*)\s*"
    r"(°|degree|degrees|deg|Å|angstrom|angstroms|nm|nanometer|nanometers|"
    r"eV|GHz|MHz|GPa|MPa|K|℃|°C|cm-1|cm⁻¹)",
    re.IGNORECASE,
)

# Qualitative keywords straight from the M5 spec. These signal that the text
# is making a judgement call, not just reporting a number.
_QUALITATIVE_RE = re.compile(
    r"(significant|显著|improve|提高|enhance|增强|decrease|降低|"
    r"confirm|确认)",
    re.IGNORECASE,
)

# Peak-shape descriptors. Not in the spec's qualitative list but extremely
# common in XRD / Raman / IR claims ("a sharp peak", "broad shoulder").
# Keeping them separate from _QUALITATIVE_RE lets us fill the qualifier field
# with something meaningful even when the LLM isn't around.
_PEAK_DESCRIPTOR_RE = re.compile(
    r"(sharp|broad|distinct|strong|weak|intense|prominent|narrow)",
    re.IGNORECASE,
)

# Verbs that introduce a conclusion clause. Used to pull a short conclusion
# string out of the surrounding sentence instead of dumping the whole window.
_CONCLUSION_VERB_RE = re.compile(
    r"\b(?:confirm|confirms|confirming|indicate|indicates|indicates|"
    r"suggest|suggests|demonstrate|demonstrates|reveal|reveals|show|shows|"
    r"prove|proves|verify|verifies|证实|确认|表明|说明|证实)\s+"
    r"(.{5,120}?)(?:[.;,]|\bbut\b|$)",
    re.IGNORECASE,
)

# Metric -> context keywords. We scan a window around each numeric hit and
# pick the first metric whose keywords appear. Order matters: longer phrases
# go first so "lattice constant" wins over a bare "a =".
_METRIC_KEYWORDS: list[tuple[str, list[str]]] = [
    ("2theta", ["2theta", "2θ", "2 theta", "2 θ", "diffraction angle", "bragg angle"]),
    ("d_spacing", ["d-spacing", "d spacing", "d=", "interplanar"]),
    ("lattice_constant", ["lattice constant", "lattice parameter", "a ="]),
    ("band_gap", ["band gap", "bandgap", "eg", "e_g"]),
    ("crystallite_size", ["crystallite size", "grain size", "scherrer"]),
    ("intensity", ["intensity", "peak intensity", "relative intensity", "counts"]),
    ("temperature", ["temperature", "annealing", "synthesized at", "calcined at"]),
    ("pressure", ["pressure", "applied pressure"]),
    ("frequency", ["frequency", "raman shift", "wavenumber", "cm-1", "cm⁻¹", "ir band"]),
]

# Unit normalisation. Different papers spell the same unit differently; we fold
# variants onto a canonical token before comparing claim vs data units.
_UNIT_ALIASES: dict[str, str] = {
    "°": "degree", "°c": "degree", "℃": "degree", "deg": "degree",
    "degree": "degree", "degrees": "degree",
    "å": "angstrom", "angstrom": "angstrom", "angstroms": "angstrom",
    "nm": "nm", "nanometer": "nm", "nanometers": "nm",
    "ev": "eV",
    "ghz": "GHz", "mhz": "MHz",
    "gpa": "GPa", "mpa": "MPa",
    "k": "K",
    "cm-1": "cm-1", "cm⁻¹": "cm-1",
}

# Units that can be compared at all. If two units aren't in each other's
# compatible set, we treat the comparison as a hard contradiction -- there's
# no way "28.4 deg" supports data recorded in "nm" without a conversion we
# don't perform.
_COMPATIBLE_UNITS: dict[str, set[str]] = {
    "degree": {"degree"},
    "angstrom": {"angstrom", "nm"},
    "nm": {"nm", "angstrom"},
    "eV": {"eV"},
    "GHz": {"GHz", "MHz"},
    "MHz": {"MHz", "GHz"},
    "GPa": {"GPa", "MPa"},
    "MPa": {"MPa", "GPa"},
    "K": {"K"},
    "cm-1": {"cm-1"},
}

# Qualifiers that imply a strong / high-intensity signal vs a weak one.
# Used by the qualitative cross-check when the data carries an intensity.
_HIGH_INTENSITY_QUALIFIERS = frozenset(
    {"sharp", "distinct", "strong", "intense", "prominent", "narrow", "significant"}
)
_LOW_INTENSITY_QUALIFIERS = frozenset({"weak", "broad"})


class CrossModalAdapter:
    """Extracts claims from text and cross-validates them against chart data.

    The adapter is the M5 stage of the document pipeline. It reads the
    heterogeneous document graph built by M3 (and optionally enriched with
    REFERENCES edges by M4), mines TEXT blocks for quantitative / qualitative
    claims, walks the graph to find the data points those claims refer to,
    and emits typed validation edges (SUPPORTS / CONTRADICTS / INCONCLUSIVE)
    back into the graph.

    Usage::

        adapter = CrossModalAdapter(llm_client=my_llm)
        new_edges = adapter.process(graph)
        # graph now has CLAIM nodes and SUPPORTS/CONTRADICTS/INCONCLUSIVE edges
    """

    # Per-metric tolerances. Values are in the canonical unit for each metric.
    # "default" is a 10% relative fallback for metrics we don't recognise.
    TOLERANCE_TABLE: dict[str, float] = {
        "2theta": 0.5,           # degree
        "d_spacing": 0.05,       # angstrom
        "lattice_constant": 0.01,  # angstrom
        "band_gap": 0.05,        # eV
        "crystallite_size": 1.0,   # nm
        "intensity": 0.15,      # relative, 15%
        "temperature": 2.0,     # K
        "pressure": 0.1,        # GPa
        "frequency": 0.5,      # cm^-1 (Raman/IR)
        "default": 0.1,        # 10% relative tolerance
    }

    def __init__(self, llm_client: Any = None) -> None:
        # LLM is optional -- regex extraction works fine without it, it just
        # produces less structured conclusions / qualifiers.
        self.llm = llm_client
        self._claim_counter = 0

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def process(self, graph: DocumentGraph) -> list[GraphEdge]:
        """Run claim extraction -> data binding -> cross-validation.

        Adds CLAIM elements and SUPPORTS / CONTRADICTS / INCONCLUSIVE edges
        to the graph. Returns the list of new validation edges. Structural
        edges (e.g. CONTAINS from text to claim) are also added for graph
        integrity but not returned -- only the validation verdicts are the
        M5 output that downstream stages care about.
        """
        new_edges: list[GraphEdge] = []

        claims = self._extract_claims(graph)
        bindings = self._bind_claims_to_data(graph, claims)

        for claim, data_points in bindings:
            result = self._cross_validate(claim, data_points)

            # Stash the verdict on the claim so callers can inspect it
            # without re-walking the edges.
            claim.metadata["validation"] = {
                "verdict": result["verdict"],
                "score": result["score"],
                "evidence": result["evidence"],
            }

            matched = result.get("matched_data")
            if matched is None:
                # No data to point at -- skip the edge. The claim node is
                # still in the graph for provenance.
                continue

            target_id = matched.get("_source_element_id")
            if not target_id:
                continue

            edge_type = self._verdict_edge_type(result["verdict"])
            edge = GraphEdge(
                source=claim.element_id,
                target=target_id,
                edge_type=edge_type,
                weight=round(result["score"], 4),
                confidence=round(result["score"], 4),
                metadata={
                    "evidence": result["evidence"],
                    "score": round(result["score"], 4),
                    "verdict": result["verdict"],
                },
            )
            graph.add_edge(edge)
            new_edges.append(edge)

        return new_edges

    # ------------------------------------------------------------------
    # 1. Claim extraction
    # ------------------------------------------------------------------

    def _extract_claims(self, graph: DocumentGraph) -> list[DocumentElement]:
        """Extract quantitative / qualitative claims from TEXT blocks.

        For each text element we run the LLM (if available) and the regex
        extractor, then deduplicate by (metric, value, unit, qualifier).
        Each surviving claim becomes a CLAIM element wired back to its source
        text via a CONTAINS edge, mirroring how mentions are handled.
        """
        claims: list[DocumentElement] = []
        seen_keys: set[tuple] = set()

        for text_el in graph.get_elements(ElementType.TEXT):
            content = text_el.content
            if not isinstance(content, str) or not content.strip():
                continue

            raw_claims: list[dict[str, Any]] = []

            # LLM first -- it tends to produce the richer structures (real
            # conclusions, better qualifiers). Regex fills the gaps and
            # runs unconditionally so we never miss a "value + unit" hit.
            if self.llm is not None:
                llm_claim = self._llm_extract_claim(content)
                if llm_claim:
                    raw_claims.append(llm_claim)

            raw_claims.extend(self._regex_extract_claims(content))

            for claim_data in raw_claims:
                key = (
                    claim_data.get("metric"),
                    claim_data.get("value"),
                    claim_data.get("unit"),
                    claim_data.get("qualifier"),
                )
                if key in seen_keys:
                    continue
                seen_keys.add(key)

                self._claim_counter += 1
                cid = f"{text_el.element_id}__claim_{self._claim_counter}"
                claim = DocumentElement(
                    element_id=cid,
                    element_type=ElementType.CLAIM,
                    content=claim_data.get("source_text", content[:200]),
                    page=text_el.page,
                    # Inherit the parent's bbox -- we don't have char-level
                    # offsets, same trade-off the mention extractor makes.
                    bbox=text_el.bbox,
                    claim_data=claim_data,
                    metadata={"parent_text_id": text_el.element_id},
                )
                claims.append(claim)
                graph.add_element(claim)
                graph.add_edge(GraphEdge(
                    source=text_el.element_id,
                    target=cid,
                    edge_type=EdgeType.CONTAINS,
                ))

        return claims

    def _regex_extract_claims(self, text: str) -> list[dict[str, Any]]:
        """Pull claims out of text using the numeric + qualitative regexes."""
        claims: list[dict[str, Any]] = []
        # Track the context windows of numeric hits so we can skip
        # qualitative words that were already absorbed as qualifiers.
        consumed: list[tuple[int, int]] = []

        for m in _NUMERIC_RE.finditer(text):
            value = float(m.group(1))
            raw_unit = m.group(2)
            unit = _UNIT_ALIASES.get(raw_unit.lower(), raw_unit.lower())

            ctx_start = max(0, m.start() - 80)
            ctx_end = min(len(text), m.end() + 80)
            window = text[ctx_start:ctx_end]

            metric = self._detect_metric(window)
            qualifier = self._detect_qualifier(window)
            conclusion = self._extract_conclusion(window)

            src_start = max(0, m.start() - 40)
            src_end = min(len(text), m.end() + 40)

            claims.append({
                "metric": metric,
                "value": value,
                "unit": unit,
                "qualifier": qualifier,
                "conclusion": conclusion,
                "source_text": text[src_start:src_end].strip(),
            })
            consumed.append((ctx_start, ctx_end))

        # Qualitative-only claims: judgement words that aren't near a number.
        # These capture things like "the results confirm the cubic phase"
        # where there's nothing quantitative to anchor on.
        for qm in _QUALITATIVE_RE.finditer(text):
            if any(s <= qm.start() < e for s, e in consumed):
                continue
            win_start = max(0, qm.start() - 60)
            win_end = min(len(text), qm.end() + 60)
            window = text[win_start:win_end]
            claims.append({
                "metric": None,
                "value": None,
                "unit": None,
                "qualifier": qm.group(1).lower(),
                "conclusion": self._extract_conclusion(window),
                "source_text": window.strip(),
            })

        return claims

    def _llm_extract_claim(self, text: str) -> dict[str, Any] | None:
        """Ask the LLM for a structured claim. Returns None on any failure."""
        prompt = (
            "Extract the quantitative or qualitative claim from the "
            "following materials science text. Return a single JSON object "
            "with keys: metric, value, unit, qualifier, conclusion, "
            "source_text. Use null for any field you can't fill. If no claim "
            "is present, return the literal string null.\n\n"
            f"Text:\n{text[:1200]}"
        )
        raw = self._llm_call(prompt)
        if not raw:
            return None
        return self._parse_llm_json(raw)

    def _llm_call(self, prompt: str) -> str | None:
        """Best-effort call into whatever LLM client shape we were given."""
        if self.llm is None:
            return None
        try:
            # Support a few common client shapes without coupling to any one.
            if callable(self.llm):
                return str(self.llm(prompt))
            if hasattr(self.llm, "complete"):
                return str(self.llm.complete(prompt))
            if hasattr(self.llm, "chat"):
                resp = self.llm.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                )
                return str(resp.choices[0].message.content)
        except Exception:
            # Any LLM hiccup -> silently fall back to regex. The pipeline
            # should never hard-fail just because the model is flaky.
            return None
        return None

    @staticmethod
    def _parse_llm_json(raw: str) -> dict[str, Any] | None:
        """Fish a JSON object out of an LLM response.

        Models love to wrap JSON in prose and markdown fences, so we try
        fenced extraction first, then a bare brace scan, then give up.
        """
        candidates: list[str] = []
        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if fence:
            candidates.append(fence.group(1))
        brace = re.search(r"\{.*\}", raw, re.DOTALL)
        if brace:
            candidates.append(brace.group(0))
        for cand in candidates:
            try:
                obj = json.loads(cand)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue
        return None

    # -- small context helpers -----------------------------------------

    @staticmethod
    def _detect_metric(window: str) -> str | None:
        lower = window.lower()
        for metric, keywords in _METRIC_KEYWORDS:
            for kw in keywords:
                if kw in lower:
                    return metric
        return None

    @staticmethod
    def _detect_qualifier(window: str) -> str | None:
        m = _PEAK_DESCRIPTOR_RE.search(window)
        if m:
            return m.group(1).lower()
        m = _QUALITATIVE_RE.search(window)
        if m:
            return m.group(1).lower()
        return None

    @staticmethod
    def _extract_conclusion(window: str) -> str:
        m = _CONCLUSION_VERB_RE.search(window)
        if m:
            return m.group(1).strip()
        # No conclusion verb found -- just hand back a trimmed snippet.
        return window.strip()[:120]

    # ------------------------------------------------------------------
    # 2. Data binding
    # ------------------------------------------------------------------

    def _bind_claims_to_data(
        self, graph: DocumentGraph, claims: list[DocumentElement]
    ) -> list[tuple[DocumentElement, list[dict[str, Any]]]]:
        """Find data points associated with each claim via REFERENCES edges.

        Returns a list of (claim, data_points) tuples. Each data point dict
        is tagged with ``_source_element_id`` so the validator (and the edge
        builder in process()) know which graph element owns the data.
        """
        # Build an id->element lookup once so we don't scan the whole element
        # list for every claim's parent text.
        id_index = {e.element_id: e for e in graph.get_elements()}

        bindings: list[tuple[DocumentElement, list[dict[str, Any]]]] = []
        for claim in claims:
            data_points = self._find_data_for_claim(graph, claim, id_index)
            bindings.append((claim, data_points))
        return bindings

    def _find_data_for_claim(
        self,
        graph: DocumentGraph,
        claim: DocumentElement,
        id_index: dict[str, DocumentElement],
    ) -> list[dict[str, Any]]:
        """Walk claim -> text -> mention -> figure/table -> data points."""
        parent_id = claim.metadata.get("parent_text_id")
        parent = id_index.get(parent_id) if parent_id else None

        figures: list[DocumentElement] = []

        if parent is not None:
            # text --CONTAINS--> mentions
            for nb in graph.get_neighbors(parent.element_id, EdgeType.CONTAINS):
                if nb.element_type is not ElementType.MENTION:
                    continue
                # mention --REFERENCES--> figure/table (predicted by M4)
                for ref in graph.get_neighbors(nb.element_id, EdgeType.REFERENCES):
                    if ref.element_type in (ElementType.FIGURE, ElementType.TABLE):
                        figures.append(ref)

        # Fallback when M4 hasn't run yet: consider every figure / table in
        # the graph. Coarser, but still lets us validate against whatever
        # data was extracted from charts.
        if not figures:
            figures = [
                e for e in id_index.values()
                if e.element_type in (ElementType.FIGURE, ElementType.TABLE)
            ]

        data: list[dict[str, Any]] = []

        # Prefer first-class DATA_POINT nodes (added by the graph builder).
        # These are the aggregate children of figures that carry data_points.
        for fig in figures:
            for dp_el in graph.get_neighbors(fig.element_id, EdgeType.EXTRACTED_FROM):
                if dp_el.element_type is not ElementType.DATA_POINT:
                    continue
                if not dp_el.data_points:
                    continue
                for dp in dp_el.data_points:
                    d = dict(dp)  # shallow copy -- don't mutate the original
                    d["_source_element_id"] = dp_el.element_id
                    data.append(d)

        # If no DATA_POINT children, read data_points straight off the
        # figure / table element. Some callers populate data_points there
        # without the graph builder synthesising a child node.
        if not data:
            for fig in figures:
                if not fig.data_points:
                    continue
                for dp in fig.data_points:
                    d = dict(dp)
                    d["_source_element_id"] = fig.element_id
                    data.append(d)

        return data

    # ------------------------------------------------------------------
    # 3. Cross-validation
    # ------------------------------------------------------------------

    def _cross_validate(
        self, claim: DocumentElement, data_points: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Compare a claim against its candidate data points.

        Returns a dict with verdict / score / evidence / matched_data.
        ``matched_data`` carries the ``_source_element_id`` tag set during
        binding so process() can build the edge target.
        """
        cd = claim.claim_data or {}

        if not data_points:
            return {
                "verdict": "inconclusive",
                "score": 0.5,
                "evidence": "No data points found to validate against.",
                "matched_data": None,
            }

        best_score = -1.0
        best_data: dict[str, Any] | None = None
        best_evidence = "no strong signal"

        for dp in data_points:
            score, evidence = self._score_match(cd, dp)
            if score > best_score:
                best_score = score
                best_data = dp
                best_evidence = evidence

        # Context completion: if the claim is missing a value but the
        # matched data has one, fill it in. This turns a vague qualitative
        # claim into a concrete one for downstream consumers.
        if best_data is not None and cd.get("value") is None:
            dv = self._data_value(best_data, cd.get("metric"))
            if dv is not None:
                cd["value"] = dv
                if not cd.get("unit"):
                    cd["unit"] = best_data.get("unit")

        return {
            "verdict": self._verdict_from_score(best_score),
            "score": round(best_score, 4),
            "evidence": best_evidence,
            "matched_data": best_data,
        }

    def _score_match(
        self, claim_data: dict[str, Any], dp: dict[str, Any]
    ) -> tuple[float, str]:
        """Score how well a single data point matches a claim.

        Scoring is additive from a 0.5 neutral base:
          - metric agreement      +/- 0.15 / -0.25
          - value within tol      +0.35  / beyond tol  -0.35
          - unit agreement        +0.05  / incompatible -0.30
          - qualitative agreement +/- 0.10 / -0.15

        The result is clamped to [0, 1].
        """
        score = 0.5
        reasons: list[str] = []

        claim_metric = claim_data.get("metric")
        claim_value = claim_data.get("value")
        claim_unit = claim_data.get("unit")
        claim_qualifier = claim_data.get("qualifier")

        dp_metric = dp.get("metric") or dp.get("label")
        dp_value = self._data_value(dp, claim_metric)
        dp_unit = dp.get("unit")
        dp_intensity = dp.get("intensity")
        if dp_intensity is None:
            dp_intensity = dp.get("y")

        # --- metric ---------------------------------------------------
        if claim_metric and dp_metric:
            if claim_metric == dp_metric:
                score += 0.15
                reasons.append("metric matches")
            else:
                score -= 0.25
                reasons.append(f"metric mismatch ({claim_metric} vs {dp_metric})")
        elif claim_metric and not dp_metric:
            # Data has no metric label -- can't confirm or deny, stay neutral.
            pass

        # --- numerical ------------------------------------------------
        if claim_value is not None and dp_value is not None:
            try:
                cv = float(claim_value)
                dv = float(dp_value)
            except (TypeError, ValueError):
                pass
            else:
                tol = self._get_tolerance(claim_metric or "")
                diff = abs(cv - dv)
                if diff <= tol:
                    score += 0.35
                    reasons.append(f"value within tolerance (Δ={diff:.4g} ≤ {tol})")
                else:
                    score -= 0.35
                    reasons.append(f"value out of tolerance (Δ={diff:.4g} > {tol})")
        elif claim_value is None and dp_value is not None:
            # Claim has no number -- data exists under the same metric, so
            # give a small nudge. The real verdict will hinge on units /
            # qualitative signals.
            score += 0.05

        # --- unit -----------------------------------------------------
        if claim_unit and dp_unit:
            cu = _UNIT_ALIASES.get(claim_unit.lower(), claim_unit.lower())
            du = _UNIT_ALIASES.get(dp_unit.lower(), dp_unit.lower())
            if cu == du:
                score += 0.05
                reasons.append("units match")
            elif du in _COMPATIBLE_UNITS.get(cu, set()):
                # Compatible but not identical (e.g. angstrom vs nm).
                # No bonus, no penalty -- a conversion would reconcile them.
                pass
            else:
                score -= 0.30
                reasons.append(f"incompatible units ({claim_unit} vs {dp_unit})")

        # --- qualitative ----------------------------------------------
        if claim_qualifier and dp_intensity is not None:
            try:
                intensity = float(dp_intensity)
            except (TypeError, ValueError):
                pass
            else:
                # Only judge intensity when it looks normalised (0-1 range).
                # Raw counts need a baseline we don't have, so we skip rather
                # than guess.
                if 0.0 <= intensity <= 1.0:
                    if claim_qualifier in _HIGH_INTENSITY_QUALIFIERS:
                        if intensity >= 0.5:
                            score += 0.10
                            reasons.append("qualifier matches high intensity")
                        else:
                            score -= 0.15
                            reasons.append("qualifier expects high intensity, data is low")
                    elif claim_qualifier in _LOW_INTENSITY_QUALIFIERS:
                        if intensity < 0.5:
                            score += 0.05
                            reasons.append("qualifier matches low intensity")
                        else:
                            score -= 0.10
                            reasons.append("qualifier expects low intensity, data is high")

        score = max(0.0, min(1.0, score))
        evidence = "; ".join(reasons) if reasons else "no strong signal"
        return score, evidence

    # ------------------------------------------------------------------
    # 4. Tolerance + verdict helpers
    # ------------------------------------------------------------------

    def _get_tolerance(self, metric: str | None) -> float:
        """Look up tolerance for a given metric, falling back to default."""
        if metric and metric in self.TOLERANCE_TABLE:
            return self.TOLERANCE_TABLE[metric]
        return self.TOLERANCE_TABLE["default"]

    @staticmethod
    def _verdict_from_score(score: float) -> str:
        if score >= 0.8:
            return "supports"
        if score <= 0.3:
            return "contradicts"
        return "inconclusive"

    @staticmethod
    def _verdict_edge_type(verdict: str) -> EdgeType:
        return {
            "supports": EdgeType.SUPPORTS,
            "contradicts": EdgeType.CONTRADICTS,
            "inconclusive": EdgeType.INCONCLUSIVE,
        }[verdict]

    @staticmethod
    def _data_value(dp: dict[str, Any], claim_metric: str | None) -> float | None:
        """Extract the comparable value from a data point dict.

        Chart data is often stored as {x, y} pairs rather than {value}. For
        intensity claims we want y; for everything else x is the independent
        variable (the one the claim is making an assertion about).
        """
        if "value" in dp and dp["value"] is not None:
            return dp["value"]
        if claim_metric == "intensity":
            return dp.get("y") or dp.get("intensity")
        return dp.get("x")
