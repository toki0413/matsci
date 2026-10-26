"""Lightweight benchmark framework for HuginnAgent.

A benchmark case is a task plus an evaluator. The suite runs each case
against an agent, scores the result, and the self-improvement loop stores
failures in long-term memory so the agent can learn over time.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from huginn.memory.manager import MemoryManager

EvaluatorT = Callable[[str, "BenchmarkCase"], tuple[bool, float]]
"""Evaluator(response, case) -> (success, score)."""


def keyword_evaluator(response: str, case: BenchmarkCase) -> tuple[bool, float]:
    """Pass if all expected keywords are present (case-insensitive)."""
    text = response.lower()
    matches = sum(1 for kw in case.expected_keywords if kw.lower() in text)
    if not case.expected_keywords:
        return True, 1.0
    success = matches == len(case.expected_keywords)
    score = matches / len(case.expected_keywords)
    return success, round(score, 2)


def numeric_evaluator(response: str, case: BenchmarkCase) -> tuple[bool, float]:
    """Pass if a number near ``expected_value`` appears in the response.

    Tolerance defaults to 1% relative or absolute, whichever is larger.
    """
    expected = case.expected_value
    if expected is None:
        return keyword_evaluator(response, case)

    numbers = [
        float(m) for m in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", response)
    ]
    if not numbers:
        return False, 0.0

    tolerance = case.tolerance or max(abs(expected) * 0.01, 0.01)
    best = min(abs(n - expected) for n in numbers)
    success = best <= tolerance
    score = max(0.0, 1.0 - best / (tolerance * 2)) if tolerance > 0 else 0.0
    return success, round(score, 2)


def llm_judge_evaluator(
    judge_model: Callable[[str], str],
) -> EvaluatorT:
    """Return an evaluator that asks a small LLM to score the answer 0-1.

    The judge is given the task, rubric, and response, and must reply with a
    single JSON object: {"score": float, "reason": str}.
    """

    def evaluate(response: str, case: BenchmarkCase) -> tuple[bool, float]:
        prompt = (
            "You are a strict but fair grader. Evaluate the answer below for the task.\n\n"
            f"Task: {case.task}\n"
            f"Rubric: {case.rubric or 'Answer should be correct and complete.'}\n\n"
            f"Answer: {response[:2000]}\n\n"
            'Respond ONLY with JSON: {"score": float between 0 and 1, "reason": string}'
        )
        try:
            raw = judge_model(prompt)
            # Extract JSON from possible markdown code block.
            if "```" in raw:
                raw = raw.split("```")[1].strip("json").strip()
            data = json.loads(raw)
            score = float(data.get("score", 0.0))
            return score >= 0.8, round(score, 2)
        except Exception:
            return False, 0.0

    return evaluate


def rubric_evaluator(response: str, case: BenchmarkCase) -> tuple[bool, float]:
    """Score against weighted rubric items (RCBench-style).

    Each rubric item has: criterion (str), weight (float), keywords (list[str]).
    An item is "met" if all its keywords appear in the response (case-insensitive).
    Score = sum(met weights) / sum(total weights) * 100.
    Falls back to keyword_evaluator if rubric_items is empty.
    """
    if not case.rubric_items:
        return keyword_evaluator(response, case)

    text = response.lower()
    total_weight = 0.0
    met_weight = 0.0
    for item in case.rubric_items:
        weight = float(item.get("weight", 1.0))
        keywords = item.get("keywords", [])
        total_weight += weight
        if not keywords:
            # no keywords = criterion met by default (e.g. code execution succeeded)
            met_weight += weight
        elif all(kw.lower() in text for kw in keywords):
            met_weight += weight

    if total_weight == 0:
        return True, 100.0
    score = round(met_weight / total_weight * 100, 1)
    # ponytail: pass threshold at 50 (RCBench's "matches paper" anchor)
    return score >= 50, score


@dataclass
class BenchmarkCase:
    """A single benchmark task."""

    task: str
    expected_keywords: list[str] = field(default_factory=list)
    expected_value: float | None = None
    tolerance: float | None = None
    rubric: str | None = None
    evaluator: EvaluatorT = keyword_evaluator
    category: str = "general"
    tags: list[str] = field(default_factory=list)
    case_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    # RCBench-style weighted rubric items. Each item:
    # {"criterion": str, "weight": float, "keywords": [str], "type": "text"|"image"}
    # When populated, rubric_evaluator scores each criterion by keyword presence
    # and computes a weighted sum scaled to 0-100.
    rubric_items: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class BenchmarkResult:
    """Result of running one benchmark case."""

    case_id: str
    task: str
    success: bool
    score: float
    response: str
    duration_ms: float
    error: str | None = None
    cost: float = 0.0


@dataclass
class CaseTrialResult:
    """One case run N times — feeds pass^3 / pass@3 computation."""

    case_id: str
    task: str
    category: str
    trials: list[BenchmarkResult]
    pass_all: bool  # pass^3: every trial passed
    pass_any: bool  # pass@3: at least one trial passed
    avg_score: float
    max_score: float


@dataclass
class MultiTrialResult:
    """Suite-level multi-trial result with ClawBench metrics."""

    case_results: list[CaseTrialResult]
    trials: int
    avg_score: float  # S: mean score across all trials, normalized 0-1
    pass_all_rate: float  # pass^3 fraction of cases
    pass_any_rate: float  # pass@3 fraction of cases
    final_score: float  # 100 * S^0.40 * r_all^0.45 * r_any^0.15
    total_cost: float
    avg_latency_ms: float
    coverage: float  # fraction of categories with >=1 pass


class BenchmarkSuite:
    """Collection of benchmark cases with execution and scoring."""

    def __init__(self, name: str = "default") -> None:
        self.name = name
        self.cases: list[BenchmarkCase] = []

    def add(self, case: BenchmarkCase) -> BenchmarkSuite:
        self.cases.append(case)
        return self

    def add_defaults(self) -> BenchmarkSuite:
        """Register a small set of materials-science sanity checks."""
        self.add(
            BenchmarkCase(
                task="What is the crystal structure of silicon at room temperature?",
                expected_keywords=["diamond", "cubic"],
                category="materials",
                tags=["structure"],
            )
        )
        self.add(
            BenchmarkCase(
                task="Calculate the band gap of silicon in eV.",
                expected_value=1.1,
                tolerance=0.05,
                evaluator=numeric_evaluator,
                category="materials",
                tags=["electronic"],
            )
        )
        self.add(
            BenchmarkCase(
                task="Set up a harmonic oscillator model with mass 1 and k=4.",
                expected_value=2.0,
                tolerance=0.1,
                evaluator=numeric_evaluator,
                category="unified",
                tags=["math"],
            )
        )
        return self

    def materials_science_research_cases(self) -> BenchmarkSuite:
        """Register 11 cases covering the matsci agent's skill surface.

        Spans structure queries, band gaps, DB retrieval, symbolic math,
        literature review, multiscale modeling, phase diagrams, thermo,
        degradation analysis, research design, and elastic properties.
        """
        # 1. structure query
        self._msrc_basics()
        self._msrc_computation()
        self._msrc_thermo_phase()
        self._msrc_design()
        self._msrc_inverse()
        return self

    def _msrc_basics(self) -> None:
        """Cases 1-3: structure query / band gap / DB retrieval."""
        # 1. structure query
        self.add(
            BenchmarkCase(
                task="硅在室温下的晶体结构是什么？给出空间群和晶格常数。",
                expected_keywords=["diamond", "cubic", "fd-3m"],
                category="structure",
                tags=["crystal", "silicon"],
            )
        )
        # 2. band gap calculation
        self.add(
            BenchmarkCase(
                task="计算硅的带隙（eV），并说明是直接还是间接带隙。",
                expected_value=1.12,
                tolerance=0.05,
                evaluator=numeric_evaluator,
                category="electronic",
                tags=["bandgap", "silicon"],
            )
        )
        # 3. materials database retrieval
        self.add(
            BenchmarkCase(
                task="从Materials Project查询BaTiO3的基本信息：空间群、带隙、形成能。",
                expected_keywords=["batio3", "perovskite", "p4mm"],
                category="database",
                tags=["mp", "query"],
            )
        )

    def _msrc_computation(self) -> None:
        """Cases 4-6: symbolic math / literature review / multiscale modeling."""
        BenchmarkSuite._msrc_computation_sympy(self)
        BenchmarkSuite._msrc_computation_battery(self)

    def _msrc_computation_sympy(self) -> None:
        """Register cases (sympy, review)."""
        # 4. symbolic computation
        self.add(
            BenchmarkCase(
                task="用SymPy计算 x^2 在区间 [0, 1] 上的定积分。",
                expected_value=0.333,
                tolerance=0.01,
                evaluator=numeric_evaluator,
                category="symbolic",
                tags=["sympy", "integral"],
            )
        )
        # 5. literature review
        self.add(
            BenchmarkCase(
                task="综述钙钛矿太阳能电池最近五年的效率进展，列出关键里程碑。",
                expected_keywords=["perovskite", "solar", "efficiency", "25"],
                category="literature",
                tags=["review", "photovoltaic"],
            )
        )

    def _msrc_computation_battery(self) -> None:
        """Register cases (battery)."""
        # 6. multiscale modeling advice
        self.add(
            BenchmarkCase(
                task="为Li离子电池正极材料设计多尺度建模方案，从DFT到连续介质。",
                rubric_items=[
                    {
                        "criterion": "mentions DFT/ab initio",
                        "weight": 2,
                        "keywords": ["dft", "ab initio"],
                    },
                    {
                        "criterion": "mentions molecular dynamics",
                        "weight": 2,
                        "keywords": ["molecular dynamics", "md "],
                    },
                    {
                        "criterion": "mentions continuum/FEM",
                        "weight": 2,
                        "keywords": ["continuum", "finite element", "fem"],
                    },
                    {
                        "criterion": "mentions scale bridging",
                        "weight": 1,
                        "keywords": ["bridge", "coupling", "handoff", "multiscale"],
                    },
                ],
                evaluator=rubric_evaluator,
                category="multiscale",
                tags=["battery", "modeling"],
            )
        )

    def _msrc_thermo_phase(self) -> None:
        """Cases 7-9: phase diagram / thermodynamics / degradation."""
        # 7. phase diagram analysis
        self.add(
            BenchmarkCase(
                task="分析Fe-C二元相图中的共析反应，给出反应温度和产物。",
                expected_keywords=["eutectoid", "austenite", "pearlite", "727"],
                category="phase_diagram",
                tags=["fe-c", "metallurgy"],
            )
        )
        # 8. thermodynamic properties
        self.add(
            BenchmarkCase(
                task="计算水在298 K下的标准生成焓（kJ/mol）。",
                expected_value=-285.8,
                tolerance=2.0,
                evaluator=numeric_evaluator,
                category="thermodynamics",
                tags=["enthalpy", "water"],
            )
        )
        # 9. degradation mechanism analysis
        self.add(
            BenchmarkCase(
                task="分析PVDF聚合物在紫外光照射下的降解机制。",
                expected_keywords=["radical", "dehydrofluorination", "chain scission"],
                category="degradation",
                tags=["polymer", "uv"],
            )
        )

    def _msrc_design(self) -> None:
        """Cases 10-11: research proposal design / elastic properties."""
        # 10. research proposal design
        self.add(
            BenchmarkCase(
                task="设计一个高通量计算筛选固态电解质的流程，包含描述符和筛选标准。",
                rubric_items=[
                    {
                        "criterion": "mentions high-throughput workflow",
                        "weight": 2,
                        "keywords": ["high-throughput", "high throughput", "screening"],
                    },
                    {
                        "criterion": "mentions ionic conductivity descriptor",
                        "weight": 2,
                        "keywords": ["ionic conductivity", "conductivity"],
                    },
                    {
                        "criterion": "mentions stability criterion",
                        "weight": 2,
                        "keywords": ["stability", "electrochemical window"],
                    },
                    {
                        "criterion": "mentions candidate material families",
                        "weight": 1,
                        "keywords": [
                            "li7p3s11",
                            "lgps",
                            "argyrodite",
                            "garnet",
                            "llzo",
                        ],
                    },
                ],
                evaluator=rubric_evaluator,
                category="research_design",
                tags=["high-throughput", "electrolyte"],
            )
        )
        # 11. elastic properties (bonus)
        self.add(
            BenchmarkCase(
                task="计算铜的体弹模量（GPa）。",
                expected_value=140.0,
                tolerance=15.0,
                evaluator=numeric_evaluator,
                category="mechanical",
                tags=["elastic", "copper"],
            )
        )

    def _msrc_inverse(self) -> None:
        """Cases 12-13: inverse design via reverse reasoning."""
        BenchmarkSuite._msrc_inverse_reverse_reasoning(self)
        BenchmarkSuite._msrc_inverse_reverse_reasoning_2(self)

    def _msrc_inverse_reverse_reasoning(self) -> None:
        """Register cases (reverse_reasoning)."""
        # 12. inverse design: target property → candidate composition (reverse reasoning)
        self.add(
            BenchmarkCase(
                task="给定目标带隙为 1.5 eV 的直接带隙半导体，反推一个可能的钙钛矿组成，"
                "并说明从目标性质到候选材料的逆向推理链。",
                rubric_items=[
                    {
                        "criterion": "mentions specific candidate composition",
                        "weight": 2,
                        "keywords": ["ba", "sn", "in", "ga", "masn", "fassn", "cs sn"],
                    },
                    {
                        "criterion": "mentions reverse reasoning chain (property→structure→composition)",
                        "weight": 2,
                        "keywords": [
                            "bandgap",
                            "tolerance factor",
                            "octahedral",
                            "direct gap",
                            "absorption",
                            "perovskite",
                        ],
                    },
                    {
                        "criterion": "mentions verification method",
                        "weight": 1,
                        "keywords": [
                            "dft",
                            "verify",
                            "calculate",
                            "confirm",
                            "validate",
                        ],
                    },
                ],
                evaluator=rubric_evaluator,
                category="inverse_design",
                tags=["reverse_reasoning", "perovskite", "bandgap"],
            )
        )

    def _msrc_inverse_reverse_reasoning_2(self) -> None:
        # 13. inverse design: target performance → structural features (reverse reasoning)
        self.add(
            BenchmarkCase(
                task="设计一个热导率低于 1 W/mK 的热电材料，反推所需的微观结构特征，"
                "并给出至少一个候选材料体系及其结构设计依据。",
                rubric_items=[
                    {
                        "criterion": "mentions low thermal conductivity mechanism",
                        "weight": 2,
                        "keywords": [
                            "phonon",
                            "scattering",
                            "defect",
                            "grain boundary",
                            "anharmonic",
                            "rattling",
                            "complex structure",
                        ],
                    },
                    {
                        "criterion": "mentions candidate material system",
                        "weight": 2,
                        "keywords": [
                            "skutterudite",
                            "clathrate",
                            "pbte",
                            "bi2te3",
                            "half-heusler",
                            "zintl",
                        ],
                    },
                    {
                        "criterion": "mentions reverse design logic (target→structure→material)",
                        "weight": 1,
                        "keywords": [
                            "structural",
                            "design",
                            "tailor",
                            "engineer",
                            "optimize",
                            "hierarchical",
                        ],
                    },
                ],
                evaluator=rubric_evaluator,
                category="inverse_design",
                tags=["reverse_reasoning", "thermoelectric", "thermal"],
            )
        )

    def matscibench_cases(self) -> BenchmarkSuite:
        """MatSciBench: 31 subfields × 3 difficulty levels = 93 cases.
        Source: arXiv:2510.12171, KDD 2026.
        6 domains, 31 subfields, 3 difficulty levels.
        """
        for build in (
            self._msb_materials,
            self._msb_properties,
            self._msb_structures,
            self._msb_mechanisms,
            self._msb_processes,
            self._msb_failures,
        ):
            build()
        return self

    def _msb_materials(self) -> None:
        """MatSciBench Materials domain."""
        # ═══ Materials domain ═══
        self._msbm_classes()
        self._msbm_ceramics()
        self._msbm_polymers()
        self._msbm_composites()
        self._msbm_semiconductors()

    def _msbm_classes(self) -> None:
        """Material classes & bonding: metals/ceramics/polymers (001-003)."""
        BenchmarkSuite._msbm_classes_metals(self)
        BenchmarkSuite._msbm_classes_metals_2(self)

    def _msbm_classes_metals(self) -> None:
        """Register matscibench_001, matscibench_002 (metals)."""
        self.add(
            BenchmarkCase(
                task="What are the three primary classes of engineering materials? Give one "
                "example of each.",
                expected_keywords=["metals", "ceramics", "polymers"],
                category="materials",
                tags=["metals", "classification", "easy"],
                case_id="matscibench_001",
            )
        )
        self.add(
            BenchmarkCase(
                task="Compare the bonding characteristics of metals, ceramics, and polymers. "
                "How does bonding explain the difference in ductility?",
                expected_keywords=[
                    "metallic",
                    "covalent",
                    "ionic",
                    "ductile",
                    "brittle",
                ],
                category="materials",
                tags=["metals", "bonding", "medium"],
                case_id="matscibench_002",
            )
        )

    def _msbm_classes_metals_2(self) -> None:
        """Register matscibench_003 (metals)."""
        self.add(
            BenchmarkCase(
                task="Design a high-strength low-alloy (HSLA) steel for a structural beam. "
                "What alloying elements would you add and why? Discuss the role of "
                "microalloying elements (Nb, V, Ti) in grain refinement and precipitation "
                "strengthening.",
                rubric_items=[
                    {
                        "criterion": "mentions microalloying elements",
                        "weight": 2,
                        "keywords": ["nb", "v", "ti"],
                    },
                    {
                        "criterion": "grain refinement mechanism",
                        "weight": 2,
                        "keywords": ["grain", "refinement"],
                    },
                    {
                        "criterion": "precipitation strengthening",
                        "weight": 2,
                        "keywords": ["precipitation", "strengthening"],
                    },
                    {
                        "criterion": "carbon content control",
                        "weight": 1,
                        "keywords": ["carbon", "low"],
                    },
                ],
                evaluator=rubric_evaluator,
                category="materials",
                tags=["metals", "alloy_design", "hard"],
                case_id="matscibench_003",
            )
        )

    def _msbm_ceramics(self) -> None:
        """Ceramics: bonding/toughening/ZrO2 transformation (004-006)."""
        BenchmarkSuite._msbm_ceramics_ceramics(self)
        BenchmarkSuite._msbm_ceramics_ceramics_2(self)

    def _msbm_ceramics_ceramics(self) -> None:
        """Register matscibench_004, matscibench_005 (ceramics)."""
        self.add(
            BenchmarkCase(
                task="What is a ceramic material? Describe the type of bonding (ionic vs "
                "covalent) and give two examples.",
                expected_keywords=["ionic", "covalent", "ceramic"],
                category="materials",
                tags=["ceramics", "bonding", "easy"],
                case_id="matscibench_004",
            )
        )
        self.add(
            BenchmarkCase(
                task="Describe three toughening mechanisms used in structural ceramics. How "
                "does each increase fracture toughness?",
                rubric_items=[
                    {
                        "criterion": "transformation toughening",
                        "weight": 2,
                        "keywords": ["transformation", "zirconia"],
                    },
                    {
                        "criterion": "crack deflection",
                        "weight": 2,
                        "keywords": ["crack", "deflection"],
                    },
                    {
                        "criterion": "fiber/whisker reinforcement",
                        "weight": 2,
                        "keywords": ["fiber", "reinforcement"],
                    },
                ],
                evaluator=rubric_evaluator,
                category="materials",
                tags=["ceramics", "toughening", "medium"],
                case_id="matscibench_005",
            )
        )

    def _msbm_ceramics_ceramics_2(self) -> None:
        """Register matscibench_006 (ceramics)."""
        self.add(
            BenchmarkCase(
                task="Explain the mechanism of transformation toughening in ZrO2. Describe the "
                "tetragonal-to-monoclinic phase transformation, the volume change "
                "involved (~3-5%), and the stress field around a crack tip. Calculate the "
                "critical grain size below which the tetragonal phase is retained at room "
                "temperature.",
                rubric_items=[
                    {
                        "criterion": "t→m transformation described",
                        "weight": 2,
                        "keywords": ["tetragonal", "monoclinic"],
                    },
                    {
                        "criterion": "volume change mentioned",
                        "weight": 2,
                        "keywords": ["volume", "expansion"],
                    },
                    {
                        "criterion": "crack tip stress field",
                        "weight": 2,
                        "keywords": ["crack", "stress"],
                    },
                    {
                        "criterion": "critical grain size",
                        "weight": 1,
                        "keywords": ["grain", "critical"],
                    },
                ],
                evaluator=rubric_evaluator,
                category="materials",
                tags=["ceramics", "zro2", "transformation", "hard"],
                case_id="matscibench_006",
            )
        )

    def _msbm_polymers(self) -> None:
        """Polymers: thermoplastic/Tg/Flory-Fox (007-009)."""
        self.add(
            BenchmarkCase(
                task="What is the difference between a thermoplastic and a thermoset polymer? "
                "Give one example of each.",
                expected_keywords=["thermoplastic", "thermoset", "crosslink"],
                category="materials",
                tags=["polymers", "classification", "easy"],
                case_id="matscibench_007",
            )
        )
        self.add(
            BenchmarkCase(
                task="Explain the glass transition temperature (Tg). How does it differ from "
                "the melting temperature (Tm)? What happens to the polymer chains at Tg?",
                expected_keywords=["glass transition", "tg", "amorphous", "chain"],
                category="materials",
                tags=["polymers", "tg", "medium"],
                case_id="matscibench_008",
            )
        )
        self.add(
            BenchmarkCase(
                task="How does molecular weight affect the glass transition temperature and "
                "mechanical strength of polymers? Using the Flory-Fox equation Tg = Tg(∞) "
                "- K/Mn, calculate Tg for polystyrene with Mn = 50000 g/mol. Given Tg(∞) "
                "= 373 K and K = 1.0×10⁵ K·g/mol. Answer in K.",
                expected_value=371.0,
                tolerance=1.0,
                evaluator=numeric_evaluator,
                category="materials",
                tags=["polymers", "molecular_weight", "flory_fox", "hard"],
                case_id="matscibench_009",
            )
        )

    def _msbm_composites(self) -> None:
        """Composites: rule of mixtures/fiber reinforcement/interface (010-012)."""
        BenchmarkSuite._msbm_composites_composites(self)
        BenchmarkSuite._msbm_composites_composites_2(self)

    def _msbm_composites_composites(self) -> None:
        """Register matscibench_010, matscibench_011 (composites)."""
        self.add(
            BenchmarkCase(
                task="A composite contains 60% glass fiber (E = 72 GPa) and 40% epoxy matrix "
                "(E = 3.5 GPa) by volume. Calculate the longitudinal modulus using the "
                "rule of mixtures: E_c = V_f·E_f + V_m·E_m. Answer in GPa.",
                expected_value=44.6,
                tolerance=1.0,
                evaluator=numeric_evaluator,
                category="materials",
                tags=["composites", "rule_of_mixtures", "easy"],
                case_id="matscibench_010",
            )
        )
        self.add(
            BenchmarkCase(
                task="Explain how fiber length, orientation, and volume fraction affect the "
                "mechanical properties of a fiber-reinforced composite. What is the "
                "critical fiber length and why does it matter?",
                rubric_items=[
                    {
                        "criterion": "fiber length effect",
                        "weight": 2,
                        "keywords": ["length", "critical"],
                    },
                    {
                        "criterion": "orientation effect",
                        "weight": 2,
                        "keywords": ["orientation", "aligned"],
                    },
                    {
                        "criterion": "volume fraction",
                        "weight": 2,
                        "keywords": ["volume", "fraction"],
                    },
                ],
                evaluator=rubric_evaluator,
                category="materials",
                tags=["composites", "fiber_reinforced", "medium"],
                case_id="matscibench_011",
            )
        )

    def _msbm_composites_composites_2(self) -> None:
        """Register matscibench_012 (composites)."""
        self.add(
            BenchmarkCase(
                task="Design the fiber-matrix interface for a carbon fiber/epoxy composite. "
                "Discuss sizing chemistry, interfacial shear strength testing (single "
                "fiber fragmentation), and the trade-off between strong and weak "
                "interfaces for toughness. How would you optimize the interphase for both "
                "strength and toughness?",
                rubric_items=[
                    {
                        "criterion": "sizing chemistry",
                        "weight": 2,
                        "keywords": ["sizing", "coupling"],
                    },
                    {
                        "criterion": "interfacial shear strength",
                        "weight": 2,
                        "keywords": ["interfacial", "shear"],
                    },
                    {
                        "criterion": "strength vs toughness trade-off",
                        "weight": 2,
                        "keywords": ["toughness", "strength"],
                    },
                ],
                evaluator=rubric_evaluator,
                category="materials",
                tags=["composites", "carbon_fiber", "interface", "hard"],
                case_id="matscibench_012",
            )
        )

    def _msbm_semiconductors(self) -> None:
        """Semiconductors: doping/p-n junction/band gap engineering (013-015)."""
        self.add(
            BenchmarkCase(
                task="Explain the difference between n-type and p-type doping in "
                "semiconductors. Give one dopant example for each type in silicon.",
                expected_keywords=["n-type", "p-type", "dopant", "phosphorus", "boron"],
                category="materials",
                tags=["semiconductors", "doping", "easy"],
                case_id="matscibench_013",
            )
        )
        self.add(
            BenchmarkCase(
                task="Calculate the built-in potential V_bi of a silicon p-n junction at 300 "
                "K. Given N_A = 1×10¹⁷ cm⁻³, N_D = 1×10¹⁶ cm⁻³, and n_i = 1.5×10¹⁰ cm⁻³. "
                "Use V_bi = (kT/q)·ln(N_A·N_D/n_i²). kT/q = 0.0259 V. Answer in V.",
                expected_value=0.75,
                tolerance=0.03,
                evaluator=numeric_evaluator,
                category="materials",
                tags=["semiconductors", "pn_junction", "built_in_potential", "medium"],
                case_id="matscibench_014",
            )
        )
        self.add(
            BenchmarkCase(
                task="Explain band gap engineering in semiconductor heterostructures. How does "
                "quantum confinement in a Type-I quantum well affect the effective band "
                "gap? Calculate the ground state confinement energy for an electron in a "
                "GaAs quantum well of width 5 nm. Use m* = 0.067m_e, ℏ = 1.055×10⁻³⁴ J·s. "
                "Answer in eV.",
                expected_value=0.045,
                tolerance=0.01,
                evaluator=numeric_evaluator,
                category="materials",
                tags=["semiconductors", "band_gap_engineering", "quantum_well", "hard"],
                case_id="matscibench_015",
            )
        )

    def _msb_properties(self):
        """MatSciBench Properties domain."""
        self._msb_properties_mechanical()
        self._msb_properties_thermal()
        self._msb_properties_electrical()
        self._msb_properties_magnetic()
        self._msb_properties_optical()
        self._msb_properties_dielectric()

    def _msb_properties_mechanical(self) -> None:
        """Mechanical properties: Hooke/stress-strain/strain hardening (016-018)."""
        # ═══ Properties domain ═══
        self.add(
            BenchmarkCase(
                task="A steel wire of length 2 m and cross-sectional area 1 mm² is subjected "
                "to a tensile force of 200 N. If Young's modulus is 200 GPa, calculate "
                "the elongation. Answer in mm.",
                expected_value=2.0,
                tolerance=0.1,
                evaluator=numeric_evaluator,
                category="properties",
                tags=["mechanical", "hooke_law", "easy"],
                case_id="matscibench_016",
            )
        )
        self.add(
            BenchmarkCase(
                task="From a tensile test, the stress-strain curve shows proportional limit at "
                "250 MPa, yield point at 280 MPa, and UTS at 450 MPa. The strain at UTS "
                "is 0.15. Calculate the Young's modulus from the elastic region if the "
                "strain at proportional limit is 0.00125. Answer in GPa.",
                expected_value=200.0,
                tolerance=10.0,
                evaluator=numeric_evaluator,
                category="properties",
                tags=["mechanical", "yield_strength", "stress_strain", "medium"],
                case_id="matscibench_017",
            )
        )
        self.add(
            BenchmarkCase(
                task="From tensile test data: true stress σ = K·ε^n. Given K = 1500 MPa and "
                "data points (ε=0.1, σ=1190 MPa) and (ε=0.3, σ=1490 MPa), calculate the "
                "strain hardening exponent n.",
                expected_value=0.3,
                tolerance=0.03,
                evaluator=numeric_evaluator,
                category="properties",
                tags=["mechanical", "strain_hardening", "hard"],
                case_id="matscibench_018",
            )
        )

    def _msb_properties_thermal(self) -> None:
        """Thermal properties: expansion/composite conductivity/shock (019-021)."""
        self.add(
            BenchmarkCase(
                task="An aluminum rod (α = 23×10⁻⁶ /°C) is 1 m long at 20°C. Calculate its "
                "length at 120°C. Answer in m.",
                expected_value=1.0023,
                tolerance=0.0001,
                evaluator=numeric_evaluator,
                category="properties",
                tags=["thermal", "expansion", "easy"],
                case_id="matscibench_019",
            )
        )
        self.add(
            BenchmarkCase(
                task="A composite wall consists of 10 mm steel (k=50 W/m·K) and 20 mm "
                "insulation (k=0.04 W/m·K). Calculate the heat flux through the wall with "
                "ΔT = 100 K. Answer in W/m².",
                expected_value=196.0,
                tolerance=5.0,
                evaluator=numeric_evaluator,
                category="properties",
                tags=["thermal", "conductivity", "composite", "medium"],
                case_id="matscibench_020",
            )
        )
        self.add(
            BenchmarkCase(
                task="Calculate the thermal shock resistance parameter R = σ_f(1-ν)/E·α for "
                "alumina. Given σ_f = 300 MPa, ν = 0.22, E = 380 GPa, α = 8×10⁻⁶ /°C. "
                "Answer in °C.",
                expected_value=770.0,
                tolerance=20.0,
                evaluator=numeric_evaluator,
                category="properties",
                tags=["thermal", "shock_resistance", "hard"],
                case_id="matscibench_021",
            )
        )

    def _msb_properties_electrical(self) -> None:
        """Electrical properties: conductivity/resistivity/Hall (022-024)."""
        self.add(
            BenchmarkCase(
                task="Calculate the electrical conductivity of copper with resistivity "
                "1.7×10⁻⁸ Ω·m. Answer in S/m.",
                expected_value=58800000.0,
                tolerance=5000000.0,
                evaluator=numeric_evaluator,
                category="properties",
                tags=["electrical", "conductivity", "easy"],
                case_id="matscibench_022",
            )
        )
        self.add(
            BenchmarkCase(
                task="The resistance of a copper wire is 10 Ω at 20°C. If the temperature "
                "coefficient of resistivity α = 0.00393/°C, calculate the resistance at "
                "100°C. Answer in Ω.",
                expected_value=13.14,
                tolerance=0.2,
                evaluator=numeric_evaluator,
                category="properties",
                tags=["electrical", "resistivity", "temperature", "medium"],
                case_id="matscibench_023",
            )
        )
        self.add(
            BenchmarkCase(
                task="In a Hall effect experiment on a semiconductor, a current of 10 mA flows "
                "through a sample of thickness 0.5 mm. A magnetic field of 0.5 T produces "
                "a Hall voltage of 5 mV. Calculate the Hall coefficient R_H = V_H·t/(I·B) "
                "in m³/C.",
                expected_value=0.0005,
                tolerance=0.0001,
                evaluator=numeric_evaluator,
                category="properties",
                tags=["electrical", "hall_effect", "hard"],
                case_id="matscibench_024",
            )
        )

    def _msb_properties_magnetic(self) -> None:
        """Magnetic properties: ferromagnetism/hysteresis/Curie (025-027)."""
        self.add(
            BenchmarkCase(
                task="What is ferromagnetism? Name three ferromagnetic elements at room "
                "temperature and describe what happens above the Curie temperature.",
                expected_keywords=[
                    "ferromagnetic",
                    "curie",
                    "iron",
                    "nickel",
                    "cobalt",
                ],
                category="properties",
                tags=["magnetic", "ferromagnetism", "easy"],
                case_id="matscibench_025",
            )
        )
        self.add(
            BenchmarkCase(
                task="A magnetic core has a B-H loop with B_max = 1.5 T and H_c = 200 A/m. "
                "Estimate the hysteresis loss per cycle using the approximation W_h ≈ "
                "4·B_max·H_c. Answer in J/m³.",
                expected_value=1200.0,
                tolerance=50.0,
                evaluator=numeric_evaluator,
                category="properties",
                tags=["magnetic", "hysteresis", "loss", "medium"],
                case_id="matscibench_026",
            )
        )
        self.add(
            BenchmarkCase(
                task="Explain the exchange interaction responsible for ferromagnetism. How "
                "does the Curie temperature relate to the exchange energy? Calculate the "
                "Curie temperature for a material with exchange integral J = 5 meV and S "
                "= 1/2. Use T_C = 2JS(S+1)/(3k_B). Answer in K.",
                expected_value=43.0,
                tolerance=5.0,
                evaluator=numeric_evaluator,
                category="properties",
                tags=["magnetic", "curie", "exchange", "hard"],
                case_id="matscibench_027",
            )
        )

    def _msb_properties_optical(self) -> None:
        """Optical properties: reflectance/absorption/PLQY (028-030)."""
        self.add(
            BenchmarkCase(
                task="A material has refractive index n = 1.5. Calculate the reflectance at "
                "normal incidence from air (n=1) using R = ((n-1)/(n+1))². Express as "
                "percentage.",
                expected_value=4.0,
                tolerance=0.5,
                evaluator=numeric_evaluator,
                category="properties",
                tags=["optical", "reflectance", "easy"],
                case_id="matscibench_028",
            )
        )
        self.add(
            BenchmarkCase(
                task="A semiconductor wafer of thickness 0.5 mm transmits 60% of incident "
                "light at a specific wavelength. Calculate the absorption coefficient α "
                "using I/I₀ = exp(-αt). Answer in cm⁻¹.",
                expected_value=10.2,
                tolerance=0.5,
                evaluator=numeric_evaluator,
                category="properties",
                tags=["optical", "absorption", "coefficient", "medium"],
                case_id="matscibench_029",
            )
        )
        self.add(
            BenchmarkCase(
                task="A phosphor absorbs 80% of excitation photons and emits 70% of absorbed "
                "energy as fluorescence. Calculate the photoluminescence quantum yield "
                "(PLQY) as a percentage.",
                expected_value=56.0,
                tolerance=2.0,
                evaluator=numeric_evaluator,
                category="properties",
                tags=["optical", "plqy", "quantum_yield", "hard"],
                case_id="matscibench_030",
            )
        )

    def _msb_properties_dielectric(self) -> None:
        """Dielectric properties: polarization/capacitance/ferroelectric (031-033)."""
        BenchmarkSuite._msb_properties_dielectric_dielectric(self)
        BenchmarkSuite._msb_properties_dielectric_dielectric_2(self)

    def _msb_properties_dielectric_dielectric(self) -> None:
        """Register matscibench_031, matscibench_032 (dielectric)."""
        self.add(
            BenchmarkCase(
                task="Name the four types of polarization in dielectric materials. Which type "
                "is dominant at optical frequencies?",
                expected_keywords=["electronic", "ionic", "orientational", "space"],
                category="properties",
                tags=["dielectric", "polarization", "easy"],
                case_id="matscibench_031",
            )
        )
        self.add(
            BenchmarkCase(
                task="A parallel plate capacitor with plate area 1 cm² and separation 0.1 mm "
                "is filled with a dielectric (ε_r = 10). Calculate the capacitance. ε₀ = "
                "8.854×10⁻¹² F/m. Answer in pF.",
                expected_value=88.5,
                tolerance=1.0,
                evaluator=numeric_evaluator,
                category="properties",
                tags=["dielectric", "capacitance", "medium"],
                case_id="matscibench_032",
            )
        )

    def _msb_properties_dielectric_dielectric_2(self) -> None:
        """Register matscibench_033 (dielectric)."""
        self.add(
            BenchmarkCase(
                task="Describe the ferroelectric phase transition in BaTiO3. What happens to "
                "the crystal structure and spontaneous polarization at the Curie "
                "temperature (~120°C)? Why is the tetragonal phase ferroelectric while "
                "the cubic phase is not?",
                rubric_items=[
                    {
                        "criterion": "cubic to tetragonal transition",
                        "weight": 2,
                        "keywords": ["cubic", "tetragonal"],
                    },
                    {
                        "criterion": "spontaneous polarization",
                        "weight": 2,
                        "keywords": ["spontaneous", "polarization"],
                    },
                    {
                        "criterion": "centrosymmetric vs non-centrosymmetric",
                        "weight": 2,
                        "keywords": ["centrosymmetric", "non-centrosymmetric"],
                    },
                ],
                evaluator=rubric_evaluator,
                category="properties",
                tags=["dielectric", "ferroelectric", "batio3", "hard"],
                case_id="matscibench_033",
            )
        )

    def _msb_structures(self) -> None:
        """MatSciBench Structures domain."""
        # ═══ Structures domain ═══
        self._msbs_crystal()
        self._msbs_amorphous()
        self._msbs_defects()
        self._msbs_surfaces()
        self._msbs_grain_boundaries()

    def _msbs_crystal(self) -> None:
        """Crystal structure: Bragg/Miller indices/space group (034-036)."""
        BenchmarkSuite._msbs_crystal_crystal(self)
        BenchmarkSuite._msbs_crystal_crystal_2(self)

    def _msbs_crystal_crystal(self) -> None:
        """Register matscibench_034, matscibench_035 (crystal)."""
        self.add(
            BenchmarkCase(
                task="Using Bragg's law (nλ = 2d·sinθ), calculate the d-spacing for a first- "
                "order peak at 2θ = 28.4° using Cu Kα radiation (λ = 1.5406 Å). Answer in "
                "nm.",
                expected_value=0.314,
                tolerance=0.005,
                evaluator=numeric_evaluator,
                category="structures",
                tags=["crystal", "bragg", "xrd", "easy"],
                case_id="matscibench_034",
            )
        )
        self.add(
            BenchmarkCase(
                task="In a cubic crystal, the Miller indices of a plane are (1,1,1). Calculate "
                "the angle between the (111) and (100) planes using cos θ = "
                "(h₁h₂+k₁k₂+l₁l₂)/√(h₁²+k₁²+l₁²)·√(h₂²+k₂²+l₂²). Answer in degrees.",
                expected_value=54.7,
                tolerance=1.0,
                evaluator=numeric_evaluator,
                category="structures",
                tags=["crystal", "miller", "angle", "medium"],
                case_id="matscibench_035",
            )
        )

    def _msbs_crystal_crystal_2(self) -> None:
        """Register matscibench_036 (crystal)."""
        self.add(
            BenchmarkCase(
                task="A crystal has a 4-fold rotation axis along [001], mirror planes "
                "perpendicular to [100] and [010], and a body-centering translation. "
                "Determine the space group and explain why the presence of these symmetry "
                "elements constrains the lattice type.",
                rubric_items=[
                    {
                        "criterion": "identifies tetragonal system",
                        "weight": 2,
                        "keywords": ["tetragonal", "4-fold"],
                    },
                    {
                        "criterion": "identifies body-centering",
                        "weight": 2,
                        "keywords": ["body-centered", "i"],
                    },
                    {
                        "criterion": "space group notation",
                        "weight": 2,
                        "keywords": ["space group", "i4"],
                    },
                ],
                evaluator=rubric_evaluator,
                category="structures",
                tags=["crystal", "space_group", "symmetry", "hard"],
                case_id="matscibench_036",
            )
        )

    def _msbs_amorphous(self) -> None:
        """Amorphous: crystalline vs amorphous/glass formation/SRO (037-039)."""
        BenchmarkSuite._msbs_amorphous_amorphous(self)
        BenchmarkSuite._msbs_amorphous_amorphous_2(self)

    def _msbs_amorphous_amorphous(self) -> None:
        """Register matscibench_037, matscibench_038 (amorphous)."""
        self.add(
            BenchmarkCase(
                task="What is the difference between a crystalline and an amorphous material "
                "in terms of atomic arrangement? Give one example of each.",
                expected_keywords=["crystalline", "amorphous", "order", "random"],
                category="structures",
                tags=["amorphous", "vs_crystalline", "easy"],
                case_id="matscibench_037",
            )
        )
        self.add(
            BenchmarkCase(
                task="Explain the criteria for glass formation. What is the reduced glass "
                "transition temperature T_rg = Tg/Tm, and why does a high T_rg favor "
                "glass formation?",
                rubric_items=[
                    {
                        "criterion": "reduced Tg concept",
                        "weight": 2,
                        "keywords": ["reduced", "tg"],
                    },
                    {
                        "criterion": "nucleation avoidance",
                        "weight": 2,
                        "keywords": ["nucleation", "avoid"],
                    },
                    {
                        "criterion": "critical cooling rate",
                        "weight": 2,
                        "keywords": ["cooling", "rate"],
                    },
                ],
                evaluator=rubric_evaluator,
                category="structures",
                tags=["amorphous", "glass_formation", "medium"],
                case_id="matscibench_038",
            )
        )

    def _msbs_amorphous_amorphous_2(self) -> None:
        """Register matscibench_039 (amorphous)."""
        self.add(
            BenchmarkCase(
                task="Describe the short-range order (SRO) in metallic glasses. How does the "
                "solute-center cluster model explain the topological packing in Cu-Zr "
                "amorphous alloys? What is the relationship between icosahedral SRO and "
                "glass-forming ability?",
                rubric_items=[
                    {
                        "criterion": "short-range order concept",
                        "weight": 2,
                        "keywords": ["short-range", "cluster"],
                    },
                    {
                        "criterion": "icosahedral packing",
                        "weight": 2,
                        "keywords": ["icosahedral", "packing"],
                    },
                    {
                        "criterion": "glass-forming ability link",
                        "weight": 2,
                        "keywords": ["glass-forming", "ability"],
                    },
                ],
                evaluator=rubric_evaluator,
                category="structures",
                tags=["amorphous", "metallic_glass", "sro", "hard"],
                case_id="matscibench_039",
            )
        )

    def _msbs_defects(self) -> None:
        """Defects: point defects/dislocations/stacking fault (040-042)."""
        self.add(
            BenchmarkCase(
                task="Describe the difference between a Schottky defect and a Frenkel defect. "
                "Which is more likely in a ceramic with large anions?",
                expected_keywords=["schottky", "frenkel", "vacancy", "interstitial"],
                category="structures",
                tags=["defects", "point_defects", "easy"],
                case_id="matscibench_040",
            )
        )
        self.add(
            BenchmarkCase(
                task="Describe the structure of edge and screw dislocations. How does the "
                "Burgers vector relate to the dislocation line for each type?",
                expected_keywords=[
                    "edge",
                    "screw",
                    "burgers",
                    "perpendicular",
                    "parallel",
                ],
                category="structures",
                tags=["defects", "dislocation", "medium"],
                case_id="matscibench_041",
            )
        )
        self.add(
            BenchmarkCase(
                task="Explain how stacking fault energy (SFE) affects the dissociation of "
                "perfect dislocations into partials in FCC metals. For an FCC metal with "
                "SFE = 45 mJ/m², G = 48 GPa, and b = 0.25 nm, estimate the equilibrium "
                "separation distance d between partials using d = Gb²/(2π·SFE). Answer in "
                "nm.",
                expected_value=5.0,
                tolerance=1.0,
                evaluator=numeric_evaluator,
                category="structures",
                tags=["defects", "stacking_fault", "partials", "hard"],
                case_id="matscibench_042",
            )
        )

    def _msbs_surfaces(self) -> None:
        """Surfaces: surface energy/grain boundary/interface (043-045)."""
        BenchmarkSuite._msbs_surfaces_surfaces(self)
        BenchmarkSuite._msbs_surfaces_surfaces_2(self)

    def _msbs_surfaces_surfaces(self) -> None:
        """Register matscibench_043, matscibench_044 (surfaces)."""
        self.add(
            BenchmarkCase(
                task="What is surface energy? Why do liquids tend to minimize their surface "
                "area?",
                expected_keywords=["surface energy", "surface tension", "minimize"],
                category="structures",
                tags=["surfaces", "surface_energy", "easy"],
                case_id="matscibench_043",
            )
        )
        self.add(
            BenchmarkCase(
                task="Explain the concept of grain boundary energy. How does it vary with "
                "misorientation angle in low-angle grain boundaries? Use the Read- "
                "Shockley equation.",
                rubric_items=[
                    {
                        "criterion": "Read-Shockley equation",
                        "weight": 2,
                        "keywords": ["read-shockley", "misorientation"],
                    },
                    {
                        "criterion": "low vs high angle",
                        "weight": 2,
                        "keywords": ["low-angle", "high-angle"],
                    },
                    {
                        "criterion": "energy vs angle relationship",
                        "weight": 2,
                        "keywords": ["energy", "angle"],
                    },
                ],
                evaluator=rubric_evaluator,
                category="structures",
                tags=["surfaces", "grain_boundary_energy", "medium"],
                case_id="matscibench_044",
            )
        )

    def _msbs_surfaces_surfaces_2(self) -> None:
        """Register matscibench_045 (surfaces)."""
        self.add(
            BenchmarkCase(
                task="Compare coherent, semi-coherent, and incoherent interfaces. How does "
                "lattice misfit determine the type of interface? Calculate the critical "
                "misfit below which a coherent interface is stable if the critical misfit "
                "strain is ε_c = b/(2·d), where b = 0.25 nm and d = 10 nm. Express as "
                "percentage.",
                expected_value=1.25,
                tolerance=0.2,
                evaluator=numeric_evaluator,
                category="structures",
                tags=["surfaces", "interface", "coherent", "hard"],
                case_id="matscibench_045",
            )
        )

    def _msbs_grain_boundaries(self) -> None:
        """Grain boundaries: low/high angle/Hall-Petch/superplasticity (046-048)."""
        BenchmarkSuite._msbs_grain_boundaries_grain_boundaries(self)
        BenchmarkSuite._msbs_grain_boundaries_grain_boundaries_2(self)

    def _msbs_grain_boundaries_grain_boundaries(self) -> None:
        """Register matscibench_046, matscibench_047 (grain_boundaries)."""
        self.add(
            BenchmarkCase(
                task="What is the difference between a low-angle and a high-angle grain "
                "boundary? At what misorientation angle does the transition typically "
                "occur?",
                expected_keywords=["low-angle", "high-angle", "15", "misorientation"],
                category="structures",
                tags=["grain_boundaries", "low_high_angle", "easy"],
                case_id="matscibench_046",
            )
        )
        self.add(
            BenchmarkCase(
                task="Using the Hall-Petch equation σ_y = σ₀ + k·d^(-1/2), calculate the yield "
                "strength for σ₀ = 150 MPa, k = 0.45 MPa·m^(1/2), and grain size d = 10 "
                "μm. Answer in MPa.",
                expected_value=292.3,
                tolerance=5.0,
                evaluator=numeric_evaluator,
                category="structures",
                tags=["grain_boundaries", "hall_petch", "medium"],
                case_id="matscibench_047",
            )
        )

    def _msbs_grain_boundaries_grain_boundaries_2(self) -> None:
        """Register matscibench_048 (grain_boundaries)."""
        self.add(
            BenchmarkCase(
                task="Explain grain boundary sliding in superplasticity. What is the strain "
                "rate sensitivity exponent m, and why must it be >0.3 for superplastic "
                "behavior? How do grain boundary diffusion and grain rotation contribute?",
                rubric_items=[
                    {
                        "criterion": "strain rate sensitivity m",
                        "weight": 2,
                        "keywords": ["strain rate", "sensitivity"],
                    },
                    {
                        "criterion": "grain boundary sliding",
                        "weight": 2,
                        "keywords": ["sliding", "boundary"],
                    },
                    {
                        "criterion": "diffusion/rotation mechanism",
                        "weight": 2,
                        "keywords": ["diffusion", "rotation"],
                    },
                ],
                evaluator=rubric_evaluator,
                category="structures",
                tags=["grain_boundaries", "superplasticity", "sliding", "hard"],
                case_id="matscibench_048",
            )
        )

    def _msb_mechanisms(self):
        """MatSciBench Fundamental mechanisms domain."""
        self._msb_mechanisms_thermodynamics()
        self._msb_mechanisms_kinetics()
        self._msb_mechanisms_diffusion()
        self._msb_mechanisms_phase_transformations()
        self._msb_mechanisms_electronic_structure()

    def _msb_mechanisms_thermodynamics(self) -> None:
        """Thermodynamics: Gibbs/phase rule/Ellingham (049-051)."""
        BenchmarkSuite._msb_mechanisms_thermodynamics_thermodynamics(self)
        BenchmarkSuite._msb_mechanisms_thermodynamics_thermodynamics_2(self)

    def _msb_mechanisms_thermodynamics_thermodynamics(self) -> None:
        """Register matscibench_049, matscibench_050 (thermodynamics)."""
        # ═══ Fundamental mechanisms domain ═══
        self.add(
            BenchmarkCase(
                task="Calculate ΔG for a process with ΔH = -100 kJ/mol and ΔS = -50 J/(mol·K) "
                "at T = 500 K. Use ΔG = ΔH - TΔS. Answer in J/mol.",
                expected_value=-75000,
                tolerance=1000,
                evaluator=numeric_evaluator,
                category="fundamental_mechanisms",
                tags=["thermodynamics", "gibbs", "easy"],
                case_id="matscibench_049",
            )
        )
        self.add(
            BenchmarkCase(
                task="Apply the Gibbs phase rule F = C - P + 2 to a binary eutectic system "
                "(C=2) at the eutectic point where three phases coexist (L, α, β). How "
                "many degrees of freedom exist?",
                expected_value=1.0,
                tolerance=0.1,
                evaluator=numeric_evaluator,
                category="fundamental_mechanisms",
                tags=["thermodynamics", "phase_rule", "medium"],
                case_id="matscibench_050",
            )
        )

    def _msb_mechanisms_thermodynamics_thermodynamics_2(self) -> None:
        """Register matscibench_051 (thermodynamics)."""
        self.add(
            BenchmarkCase(
                task="Interpret an Ellingham diagram: at what temperature does the line for "
                "2Al + 3/2 O₂ → Al₂O₃ cross the line for 2Mg + O₂ → 2MgO? Below this "
                "temperature, which metal is the stronger reducing agent? Explain the "
                "significance for metal extraction.",
                rubric_items=[
                    {
                        "criterion": "Ellingham diagram interpretation",
                        "weight": 2,
                        "keywords": ["ellingham", "crossing"],
                    },
                    {
                        "criterion": "reducing agent comparison",
                        "weight": 2,
                        "keywords": ["reducing", "agent"],
                    },
                    {
                        "criterion": "extraction significance",
                        "weight": 2,
                        "keywords": ["extraction", "reduction"],
                    },
                ],
                evaluator=rubric_evaluator,
                category="fundamental_mechanisms",
                tags=["thermodynamics", "ellingham", "reduction", "hard"],
                case_id="matscibench_051",
            )
        )

    def _msb_mechanisms_kinetics(self) -> None:
        """Kinetics: Arrhenius/rate law/nucleation (052-054)."""
        self.add(
            BenchmarkCase(
                task="Calculate the activation energy for a reaction with k₁ = 1.0×10⁻³ at T₁ "
                "= 300 K and k₂ = 5.0×10⁻³ at T₂ = 350 K. Use ln(k₂/k₁) = Ea/R × (1/T₁ - "
                "1/T₂), R = 8.314 J/(mol·K). Answer in kJ/mol.",
                expected_value=21.6,
                tolerance=2.0,
                evaluator=numeric_evaluator,
                category="fundamental_mechanisms",
                tags=["kinetics", "arrhenius", "activation_energy", "easy"],
                case_id="matscibench_052",
            )
        )
        self.add(
            BenchmarkCase(
                task="A reaction has rate constants k = 0.01 M/s at [A] = 0.1 M and k = 0.04 "
                "M/s at [A] = 0.2 M. Determine the rate law order n using rate = k[A]^n.",
                expected_value=2.0,
                tolerance=0.1,
                evaluator=numeric_evaluator,
                category="fundamental_mechanisms",
                tags=["kinetics", "rate_law", "order", "medium"],
                case_id="matscibench_053",
            )
        )
        self.add(
            BenchmarkCase(
                task="Write the steady-state nucleation rate equation I = "
                "N_s·β*·exp(-ΔG*/kT)·exp(-τ/t). Explain each term. For homogeneous "
                "nucleation in a metal melt, estimate the critical radius r* = 2γ/(ΔGv) "
                "with γ = 0.2 J/m² and ΔGv = -1×10⁹ J/m³. Answer in nm.",
                expected_value=0.4,
                tolerance=0.05,
                evaluator=numeric_evaluator,
                category="fundamental_mechanisms",
                tags=["kinetics", "nucleation", "rate", "hard"],
                case_id="matscibench_054",
            )
        )

    def _msb_mechanisms_diffusion(self) -> None:
        """Diffusion: Fick I/Fick II/Kirkendall (055-057)."""
        BenchmarkSuite._msb_mechanisms_diffusion_diffusion(self)
        BenchmarkSuite._msb_mechanisms_diffusion_diffusion_2(self)

    def _msb_mechanisms_diffusion_diffusion(self) -> None:
        """Register matscibench_055, matscibench_056 (diffusion)."""
        self.add(
            BenchmarkCase(
                task="Using Fick's first law, calculate the diffusion flux through a membrane. "
                "D = 2×10⁻¹⁴ m²/s, dC/dx = -5×10²⁸ atoms/m⁴. J = -D(dC/dx). Answer in "
                "atoms/m²·s.",
                expected_value=1000000000000000.0,
                tolerance=200000000000000.0,
                evaluator=numeric_evaluator,
                category="fundamental_mechanisms",
                tags=["diffusion", "fick_first_law", "easy"],
                case_id="matscibench_055",
            )
        )
        self.add(
            BenchmarkCase(
                task="A carburization process uses Fick's second law: C(x,t) = Cs - "
                "(Cs-C0)·erf(x/(2√Dt)). Given Cs = 1.2%, C0 = 0.2%, D = 1.28×10⁻¹¹ m²/s "
                "at 927°C, find the depth x (in mm) where C = 0.8% after t = 5 hours. "
                "erf⁻¹(0.4) ≈ 0.37. Answer in mm.",
                expected_value=0.57,
                tolerance=0.05,
                evaluator=numeric_evaluator,
                category="fundamental_mechanisms",
                tags=["diffusion", "fick_second_law", "carburization", "medium"],
                case_id="matscibench_056",
            )
        )

    def _msb_mechanisms_diffusion_diffusion_2(self) -> None:
        """Register matscibench_057 (diffusion)."""
        self.add(
            BenchmarkCase(
                task="Explain the Kirkendall effect. In a Cu/Zn diffusion couple, markers "
                "placed at the original interface move toward the Zn side. What does this "
                "reveal about the relative diffusivities of Cu and Zn? How does this lead "
                "to pore formation?",
                rubric_items=[
                    {
                        "criterion": "marker movement explanation",
                        "weight": 2,
                        "keywords": ["marker", "movement"],
                    },
                    {
                        "criterion": "relative diffusivity",
                        "weight": 2,
                        "keywords": ["diffusivity", "unequal"],
                    },
                    {
                        "criterion": "pore formation mechanism",
                        "weight": 2,
                        "keywords": ["pore", "vacancy"],
                    },
                ],
                evaluator=rubric_evaluator,
                category="fundamental_mechanisms",
                tags=["diffusion", "kirkendall", "hard"],
                case_id="matscibench_057",
            )
        )

    def _msb_mechanisms_phase_transformations(self) -> None:
        """Phase transformations: nucleation/growth/martensite (058-060)."""
        BenchmarkSuite._msb_mechanisms_phase_transformations_phase_transformations(self)
        BenchmarkSuite._msb_mechanisms_phase_transformations_phase_transformations_2(
            self
        )

    def _msb_mechanisms_phase_transformations_phase_transformations(self) -> None:
        """Register matscibench_058, matscibench_059 (phase_transformations)."""
        self.add(
            BenchmarkCase(
                task="Explain the difference between homogeneous and heterogeneous nucleation. "
                "Why is heterogeneous nucleation more common in practice?",
                expected_keywords=[
                    "homogeneous",
                    "heterogeneous",
                    "surface",
                    "catalyst",
                ],
                category="fundamental_mechanisms",
                tags=["phase_transformations", "nucleation", "easy"],
                case_id="matscibench_058",
            )
        )
        self.add(
            BenchmarkCase(
                task="For diffusion-controlled growth of a precipitate, the growth rate v = "
                "D·(ΔC)/(C_p·δ). Explain each term and how temperature affects the growth "
                "rate through both D and ΔC (supersaturation).",
                rubric_items=[
                    {
                        "criterion": "growth rate equation",
                        "weight": 2,
                        "keywords": ["growth", "rate"],
                    },
                    {
                        "criterion": "diffusion coefficient D",
                        "weight": 2,
                        "keywords": ["diffusion", "coefficient"],
                    },
                    {
                        "criterion": "supersaturation ΔC",
                        "weight": 2,
                        "keywords": ["supersaturation", "concentration"],
                    },
                ],
                evaluator=rubric_evaluator,
                category="fundamental_mechanisms",
                tags=["phase_transformations", "diffusion_growth", "medium"],
                case_id="matscibench_059",
            )
        )

    def _msb_mechanisms_phase_transformations_phase_transformations_2(self) -> None:
        """Register matscibench_060 (phase_transformations)."""
        self.add(
            BenchmarkCase(
                task="Explain the martensitic transformation in steel. Why is it "
                "diffusionless? How does the Bain correspondence model describe the "
                "FCC→BCT lattice distortion? Calculate the shear strain for the "
                "martensitic transformation given the lattice correspondence. What is the "
                "maximum shear strain (~0.22 for Fe-C)?",
                rubric_items=[
                    {
                        "criterion": "diffusionless nature",
                        "weight": 2,
                        "keywords": ["diffusionless", "military"],
                    },
                    {
                        "criterion": "Bain distortion",
                        "weight": 2,
                        "keywords": ["bain", "distortion"],
                    },
                    {
                        "criterion": "shear strain value",
                        "weight": 2,
                        "keywords": ["shear", "strain"],
                    },
                ],
                evaluator=rubric_evaluator,
                category="fundamental_mechanisms",
                tags=["phase_transformations", "martensitic", "bain", "hard"],
                case_id="matscibench_060",
            )
        )

    def _msb_mechanisms_electronic_structure(self) -> None:
        """Electronic structure: band gap/DOS/Fermi level (061-063)."""
        BenchmarkSuite._msb_mechanisms_electronic_structure_electronic_structure(self)
        BenchmarkSuite._msb_mechanisms_electronic_structure_electronic_structure_2(self)

    def _msb_mechanisms_electronic_structure_electronic_structure(self) -> None:
        """Register matscibench_061, matscibench_062 (electronic_structure)."""
        self.add(
            BenchmarkCase(
                task="Calculate the maximum wavelength absorbed by Si (Eg = 1.12 eV). Use λ = "
                "hc/Eg, h = 6.626×10⁻³⁴ J·s, c = 3×10⁸ m/s. Answer in nm.",
                expected_value=1107,
                tolerance=20,
                evaluator=numeric_evaluator,
                category="fundamental_mechanisms",
                tags=["electronic_structure", "band_gap", "wavelength", "easy"],
                case_id="matscibench_061",
            )
        )
        self.add(
            BenchmarkCase(
                task="Explain the concept of density of states (DOS) in a solid. How does the "
                "3D free-electron DOS scale with energy? Sketch the DOS near a band edge "
                "for a semiconductor.",
                rubric_items=[
                    {
                        "criterion": "DOS definition",
                        "weight": 2,
                        "keywords": ["density", "states"],
                    },
                    {
                        "criterion": "E^(1/2) scaling",
                        "weight": 2,
                        "keywords": ["energy", "square"],
                    },
                    {
                        "criterion": "band edge behavior",
                        "weight": 2,
                        "keywords": ["band", "edge"],
                    },
                ],
                evaluator=rubric_evaluator,
                category="fundamental_mechanisms",
                tags=["electronic_structure", "dos", "medium"],
                case_id="matscibench_062",
            )
        )

    def _msb_mechanisms_electronic_structure_electronic_structure_2(self) -> None:
        """Register matscibench_063 (electronic_structure)."""
        self.add(
            BenchmarkCase(
                task="A silicon sample is doped with phosphorus at N_D = 1×10¹⁶ cm⁻³. "
                "Calculate the Fermi level position relative to the conduction band at "
                "300 K. Use E_C - E_F = kT·ln(N_C/N_D) with N_C = 2.8×10¹⁹ cm⁻³ and kT = "
                "0.0259 eV. Answer in eV.",
                expected_value=0.207,
                tolerance=0.01,
                evaluator=numeric_evaluator,
                category="fundamental_mechanisms",
                tags=["electronic_structure", "fermi_level", "doped", "hard"],
                case_id="matscibench_063",
            )
        )

    def _msb_processes(self) -> None:
        """MatSciBench Processes domain."""
        # ═══ Processes domain ═══
        self._msb_processes_casting()
        self._msb_processes_forming()
        self._msb_processes_heat_treatment()
        self._msb_processes_sintering()
        self._msb_processes_deposition()

    def _msb_processes_casting(self) -> None:
        """Casting / solidification (064-066)."""
        BenchmarkSuite._msb_processes_casting_casting(self)
        BenchmarkSuite._msb_processes_casting_casting_2(self)

    def _msb_processes_casting_casting(self) -> None:
        """Register matscibench_064, matscibench_065 (casting)."""
        self.add(
            BenchmarkCase(
                task="Describe the basic solidification process in metal casting. What are the "
                "three zones of a typical ingot structure?",
                expected_keywords=["chill", "columnar", "equiaxed"],
                category="processes",
                tags=["casting", "solidification", "easy"],
                case_id="matscibench_064",
            )
        )
        self.add(
            BenchmarkCase(
                task="Explain dendritic growth during solidification. How does the "
                "constitutional undercooling criterion determine whether a planar or "
                "dendritic interface forms? What role does the temperature gradient G and "
                "growth rate R play?",
                rubric_items=[
                    {
                        "criterion": "constitutional undercooling",
                        "weight": 2,
                        "keywords": ["undercooling", "constitutional"],
                    },
                    {
                        "criterion": "G/R ratio",
                        "weight": 2,
                        "keywords": ["gradient", "ratio"],
                    },
                    {
                        "criterion": "planar vs dendritic",
                        "weight": 2,
                        "keywords": ["planar", "dendritic"],
                    },
                ],
                evaluator=rubric_evaluator,
                category="processes",
                tags=["casting", "dendritic", "medium"],
                case_id="matscibench_065",
            )
        )

    def _msb_processes_casting_casting_2(self) -> None:
        """Register matscibench_066 (casting)."""
        self.add(
            BenchmarkCase(
                task="Explain microsegregation (coring) in as-cast alloys. How does the "
                "partition coefficient k < 1 lead to solute enrichment at the dendrite "
                "boundaries? Describe the Scheil equation and how homogenization "
                "annealing is used to reduce segregation.",
                rubric_items=[
                    {
                        "criterion": "partition coefficient k",
                        "weight": 2,
                        "keywords": ["partition", "coefficient"],
                    },
                    {
                        "criterion": "Scheil equation",
                        "weight": 2,
                        "keywords": ["scheil", "segregation"],
                    },
                    {
                        "criterion": "homogenization anneal",
                        "weight": 2,
                        "keywords": ["homogenization", "anneal"],
                    },
                ],
                evaluator=rubric_evaluator,
                category="processes",
                tags=["casting", "microsegregation", "coring", "hard"],
                case_id="matscibench_066",
            )
        )

    def _msb_processes_forming(self) -> None:
        """Metal forming: rolling/forging/cold work/FLD (067-069)."""
        BenchmarkSuite._msb_processes_forming_forming(self)
        BenchmarkSuite._msb_processes_forming_forming_2(self)

    def _msb_processes_forming_forming(self) -> None:
        """Register matscibench_067, matscibench_068 (forming)."""
        self.add(
            BenchmarkCase(
                task="Compare rolling and forging as metal forming processes. What are the "
                "main differences in deformation mode and typical products?",
                expected_keywords=["rolling", "forging", "compressive", "deformation"],
                category="processes",
                tags=["forming", "rolling_forging", "easy"],
                case_id="matscibench_067",
            )
        )
        self.add(
            BenchmarkCase(
                task="In cold working, a metal sheet is reduced from 2.0 mm to 1.6 mm "
                "thickness. Calculate the cold work percentage and explain how this "
                "affects the hardness and ductility.",
                expected_value=20.0,
                tolerance=1.0,
                evaluator=numeric_evaluator,
                category="processes",
                tags=["forming", "cold_work", "strain_hardening", "medium"],
                case_id="matscibench_068",
            )
        )

    def _msb_processes_forming_forming_2(self) -> None:
        """Register matscibench_069 (forming)."""
        self.add(
            BenchmarkCase(
                task="Describe the forming limit diagram (FLD) for sheet metal forming. How do "
                "major and minor strains define the safe and failure zones? What is the "
                "FLD0 criterion and how does the strain hardening exponent n affect the "
                "forming limit?",
                rubric_items=[
                    {
                        "criterion": "major/minor strain concept",
                        "weight": 2,
                        "keywords": ["major", "minor", "strain"],
                    },
                    {
                        "criterion": "safe vs failure zone",
                        "weight": 2,
                        "keywords": ["safe", "failure"],
                    },
                    {
                        "criterion": "n-value effect on FLD0",
                        "weight": 2,
                        "keywords": ["forming limit", "n-value"],
                    },
                ],
                evaluator=rubric_evaluator,
                category="processes",
                tags=["forming", "fld", "deep_drawing", "hard"],
                case_id="matscibench_069",
            )
        )

    def _msb_processes_heat_treatment(self) -> None:
        """Heat treatment: anneal/normalize/quench/TTT/TME (070-072)."""
        BenchmarkSuite._msb_processes_heat_treatment_heat_treatment(self)
        BenchmarkSuite._msb_processes_heat_treatment_heat_treatment_2(self)

    def _msb_processes_heat_treatment_heat_treatment(self) -> None:
        """Register matscibench_070, matscibench_071 (heat_treatment)."""
        self.add(
            BenchmarkCase(
                task="Describe the difference between annealing, normalizing, and quenching of "
                "steel. What microstructure does each produce?",
                expected_keywords=[
                    "annealing",
                    "normalizing",
                    "quenching",
                    "pearlite",
                    "martensite",
                ],
                category="processes",
                tags=["heat_treatment", "anneal_normalize", "easy"],
                case_id="matscibench_070",
            )
        )
        self.add(
            BenchmarkCase(
                task="Interpret a TTT (time-temperature-transformation) diagram for eutectoid "
                "steel. Explain the difference between the nose of the curve and the "
                "Ms/Mf lines. How does the cooling path determine the final "
                "microstructure (pearlite vs bainite vs martensite)?",
                rubric_items=[
                    {
                        "criterion": "TTT diagram structure",
                        "weight": 2,
                        "keywords": ["ttt", "time"],
                    },
                    {
                        "criterion": "pearlite/bainite/martensite",
                        "weight": 2,
                        "keywords": ["pearlite", "bainite", "martensite"],
                    },
                    {
                        "criterion": "Ms and Mf lines",
                        "weight": 2,
                        "keywords": ["ms", "mf"],
                    },
                ],
                evaluator=rubric_evaluator,
                category="processes",
                tags=["heat_treatment", "ttt", "medium"],
                case_id="matscibench_071",
            )
        )

    def _msb_processes_heat_treatment_heat_treatment_2(self) -> None:
        """Register matscibench_072 (heat_treatment)."""
        self.add(
            BenchmarkCase(
                task="Explain tempered martensite embrittlement (TME, 350°C embrittlement) in "
                "steels. What happens to the microstructure during tempering in the "
                "250-400°C range? How does cementite film formation at prior austenite "
                "grain boundaries contribute to intergranular fracture?",
                rubric_items=[
                    {
                        "criterion": "TME temperature range",
                        "weight": 2,
                        "keywords": ["350", "tempered"],
                    },
                    {
                        "criterion": "cementite film formation",
                        "weight": 2,
                        "keywords": ["cementite", "film"],
                    },
                    {
                        "criterion": "intergranular fracture",
                        "weight": 2,
                        "keywords": ["intergranular", "fracture"],
                    },
                ],
                evaluator=rubric_evaluator,
                category="processes",
                tags=["heat_treatment", "tempering", "embrittlement", "hard"],
                case_id="matscibench_072",
            )
        )

    def _msb_processes_sintering(self) -> None:
        """Sintering: stages/mechanisms/liquid phase (073-075)."""
        BenchmarkSuite._msb_processes_sintering_sintering(self)
        BenchmarkSuite._msb_processes_sintering_sintering_2(self)

    def _msb_processes_sintering_sintering(self) -> None:
        """Register matscibench_073, matscibench_074 (sintering)."""
        self.add(
            BenchmarkCase(
                task="What is sintering? Describe the three stages and the driving force for "
                "each.",
                expected_keywords=["sintering", "neck", "densification", "surface"],
                category="processes",
                tags=["sintering", "stages", "easy"],
                case_id="matscibench_073",
            )
        )
        self.add(
            BenchmarkCase(
                task="Describe the mechanisms of sintering (diffusion, evaporation- "
                "condensation, plastic flow). Which mechanism dominates in the initial "
                "stage vs the intermediate stage? How does the activation energy for "
                "grain boundary diffusion compare to that for volume diffusion?",
                rubric_items=[
                    {
                        "criterion": "sintering mechanisms listed",
                        "weight": 2,
                        "keywords": ["diffusion", "mechanism"],
                    },
                    {
                        "criterion": "stage dominance",
                        "weight": 2,
                        "keywords": ["initial", "intermediate"],
                    },
                    {
                        "criterion": "activation energy comparison",
                        "weight": 2,
                        "keywords": ["activation", "energy"],
                    },
                ],
                evaluator=rubric_evaluator,
                category="processes",
                tags=["sintering", "mechanisms", "medium"],
                case_id="matscibench_074",
            )
        )

    def _msb_processes_sintering_sintering_2(self) -> None:
        """Register matscibench_075 (sintering)."""
        self.add(
            BenchmarkCase(
                task="Explain liquid phase sintering (LPS). What are the three stages "
                "(rearrangement, solution-precipitation, solid state sintering)? How does "
                "the solubility ratio between solid and liquid determine densification? "
                "Give an example system (e.g., WC-Co).",
                rubric_items=[
                    {
                        "criterion": "three LPS stages",
                        "weight": 2,
                        "keywords": ["rearrangement", "solution"],
                    },
                    {
                        "criterion": "solubility requirement",
                        "weight": 2,
                        "keywords": ["solubility", "liquid"],
                    },
                    {
                        "criterion": "WC-Co example",
                        "weight": 2,
                        "keywords": ["wc", "cobalt"],
                    },
                ],
                evaluator=rubric_evaluator,
                category="processes",
                tags=["sintering", "liquid_phase", "lps", "hard"],
                case_id="matscibench_075",
            )
        )

    def _msb_processes_deposition(self) -> None:
        """Deposition: PVD/CVD/sputter/ALD (076-078)."""
        BenchmarkSuite._msb_processes_deposition_deposition(self)
        BenchmarkSuite._msb_processes_deposition_deposition_2(self)

    def _msb_processes_deposition_deposition(self) -> None:
        """Register matscibench_076, matscibench_077 (deposition)."""
        self.add(
            BenchmarkCase(
                task="Compare PVD (physical vapor deposition) and CVD (chemical vapor "
                "deposition). How do they differ in mechanism and typical applications?",
                expected_keywords=["pvd", "cvd", "physical", "chemical", "vapor"],
                category="processes",
                tags=["deposition", "pvd_cvd", "easy"],
                case_id="matscibench_076",
            )
        )
        self.add(
            BenchmarkCase(
                task="Explain sputter deposition. How does the sputter yield depend on ion "
                "energy, target material, and incident angle? What is the difference "
                "between DC sputtering and RF sputtering for insulating targets?",
                rubric_items=[
                    {
                        "criterion": "sputter yield factors",
                        "weight": 2,
                        "keywords": ["sputter", "yield"],
                    },
                    {
                        "criterion": "DC vs RF sputtering",
                        "weight": 2,
                        "keywords": ["dc", "rf"],
                    },
                    {
                        "criterion": "insulating target handling",
                        "weight": 2,
                        "keywords": ["insulating", "dielectric"],
                    },
                ],
                evaluator=rubric_evaluator,
                category="processes",
                tags=["deposition", "sputter", "medium"],
                case_id="matscibench_077",
            )
        )

    def _msb_processes_deposition_deposition_2(self) -> None:
        """Register matscibench_078 (deposition)."""
        self.add(
            BenchmarkCase(
                task="Explain atomic layer deposition (ALD). How does self-limiting surface "
                "chemistry enable sub-nm thickness control? Describe the ALD cycle for "
                "Al₂O₃ from TMA and H₂O. What is the growth per cycle (GPC) and how does "
                "it differ from CVD growth rate?",
                rubric_items=[
                    {
                        "criterion": "self-limiting chemistry",
                        "weight": 2,
                        "keywords": ["self-limiting", "surface"],
                    },
                    {
                        "criterion": "TMA/H₂O cycle",
                        "weight": 2,
                        "keywords": ["tma", "h₂o"],
                    },
                    {
                        "criterion": "GPC vs CVD rate",
                        "weight": 2,
                        "keywords": ["growth", "cycle"],
                    },
                ],
                evaluator=rubric_evaluator,
                category="processes",
                tags=["deposition", "ald", "thin_film", "hard"],
                case_id="matscibench_078",
            )
        )

    def _msb_failures(self):
        """MatSciBench Failure mechanisms domain."""
        self._msb_failures_fracture()
        self._msb_failures_fatigue()
        self._msb_failures_creep()
        self._msb_failures_corrosion()
        self._msb_failures_wear()

    def _msb_failures_fracture(self) -> None:
        """Fracture: ductile-brittle/Griffith/K_IC (079-081)."""
        # ═══ Failure mechanisms domain ═══
        self.add(
            BenchmarkCase(
                task="Explain the difference between ductile and brittle fracture. What "
                "microstructural features distinguish the fracture surfaces?",
                expected_keywords=["ductile", "brittle", "dimples", "cleavage"],
                category="failure_mechanisms",
                tags=["fracture", "ductile_brittle", "easy"],
                case_id="matscibench_079",
            )
        )
        self.add(
            BenchmarkCase(
                task="Using the Griffith criterion σ_f = √(2Eγ/πa), calculate the fracture "
                "stress for a glass with E = 70 GPa, γ = 1 J/m², and an internal crack of "
                "length 2a = 10 μm. Answer in MPa.",
                expected_value=66.8,
                tolerance=5.0,
                evaluator=numeric_evaluator,
                category="failure_mechanisms",
                tags=["fracture", "griffith", "medium"],
                case_id="matscibench_080",
            )
        )
        self.add(
            BenchmarkCase(
                task="A compact tension specimen of a steel alloy has K_IC = 60 MPa·m^(1/2). "
                "If the applied stress is 400 MPa and the largest flaw is a = 2 mm, "
                "determine if the component will fail. Use K = Y·σ·√(πa) with Y = 1.12. "
                "Calculate K and compare to K_IC.",
                expected_value=35.5,
                tolerance=2.0,
                evaluator=numeric_evaluator,
                category="failure_mechanisms",
                tags=["fracture", "kic", "fracture_toughness", "hard"],
                case_id="matscibench_081",
            )
        )

    def _msb_failures_fatigue(self) -> None:
        """Fatigue: S-N/Paris law/notch (082-084)."""
        BenchmarkSuite._msb_failures_fatigue_fatigue(self)
        BenchmarkSuite._msb_failures_fatigue_fatigue_2(self)

    def _msb_failures_fatigue_fatigue(self) -> None:
        """Register matscibench_082, matscibench_083 (fatigue)."""
        self.add(
            BenchmarkCase(
                task="Describe the S-N curve for a metal. What is the difference between the "
                "fatigue behavior of ferrous and non-ferrous metals regarding the "
                "endurance limit?",
                expected_keywords=["s-n", "fatigue", "endurance", "ferrous"],
                category="failure_mechanisms",
                tags=["fatigue", "s-n_curve", "easy"],
                case_id="matscibench_082",
            )
        )
        self.add(
            BenchmarkCase(
                task="Using the Paris law da/dN = C(ΔK)^m, with C = 1×10⁻¹² m/cycle and m = 3, "
                "calculate the crack growth rate da/dN when ΔK = 20 MPa·m^(1/2). Answer "
                "in m/cycle.",
                expected_value=8e-09,
                tolerance=1e-09,
                evaluator=numeric_evaluator,
                category="failure_mechanisms",
                tags=["fatigue", "paris_law", "crack_growth", "medium"],
                case_id="matscibench_083",
            )
        )

    def _msb_failures_fatigue_fatigue_2(self) -> None:
        """Register matscibench_084 (fatigue)."""
        self.add(
            BenchmarkCase(
                task="Explain notch fatigue and the role of the stress concentration factor "
                "Kt. A plate with a circular hole has Kt = 3. If the nominal stress "
                "amplitude is 100 MPa, what is the local stress at the notch root? How "
                "does Neuber's rule relate Kt to the fatigue notch factor Kf?",
                rubric_items=[
                    {
                        "criterion": "stress concentration Kt",
                        "weight": 2,
                        "keywords": ["stress", "concentration"],
                    },
                    {
                        "criterion": "local stress calculation",
                        "weight": 2,
                        "keywords": ["local", "notch"],
                    },
                    {
                        "criterion": "Neuber's rule / Kf",
                        "weight": 2,
                        "keywords": ["neuber", "kf"],
                    },
                ],
                evaluator=rubric_evaluator,
                category="failure_mechanisms",
                tags=["fatigue", "notch", "kt", "hard"],
                case_id="matscibench_084",
            )
        )

    def _msb_failures_creep(self) -> None:
        """Creep: stages/Norton/Coble-Nabarro (085-087)."""
        BenchmarkSuite._msb_failures_creep_creep(self)
        BenchmarkSuite._msb_failures_creep_creep_2(self)

    def _msb_failures_creep_creep(self) -> None:
        """Register matscibench_085, matscibench_086 (creep)."""
        self.add(
            BenchmarkCase(
                task="Describe the three stages of creep. What happens to the strain rate in "
                "each stage?",
                rubric_items=[
                    {
                        "criterion": "primary creep",
                        "weight": 2,
                        "keywords": ["primary", "decreasing"],
                    },
                    {
                        "criterion": "secondary creep",
                        "weight": 2,
                        "keywords": ["secondary", "steady"],
                    },
                    {
                        "criterion": "tertiary creep",
                        "weight": 2,
                        "keywords": ["tertiary", "accelerating"],
                    },
                ],
                evaluator=rubric_evaluator,
                category="failure_mechanisms",
                tags=["creep", "stages", "easy"],
                case_id="matscibench_085",
            )
        )
        self.add(
            BenchmarkCase(
                task="Using Norton's creep law ε̇ = A·σ^n·exp(-Q/RT), with n = 5, calculate "
                "the steady-state creep rate at a stress of 50 MPa if ε̇ = 1×10⁻⁸ /s at "
                "100 MPa (same temperature). Answer in /s.",
                expected_value=3.125e-10,
                tolerance=5e-11,
                evaluator=numeric_evaluator,
                category="failure_mechanisms",
                tags=["creep", "norton", "steady_state", "medium"],
                case_id="matscibench_086",
            )
        )

    def _msb_failures_creep_creep_2(self) -> None:
        """Register matscibench_087 (creep)."""
        self.add(
            BenchmarkCase(
                task="Compare Coble creep and Nabarro-Herring creep. How do they differ in "
                "diffusion path (grain boundary vs lattice) and in the grain size "
                "dependence of the strain rate? Write the strain rate equations and "
                "identify the d⁻² vs d⁻³ dependence.",
                rubric_items=[
                    {
                        "criterion": "Cobble: grain boundary diffusion",
                        "weight": 2,
                        "keywords": ["coble", "boundary"],
                    },
                    {
                        "criterion": "N-H: lattice diffusion",
                        "weight": 2,
                        "keywords": ["nabarro", "lattice"],
                    },
                    {
                        "criterion": "grain size exponent d⁻² vs d⁻³",
                        "weight": 2,
                        "keywords": ["grain", "exponent"],
                    },
                ],
                evaluator=rubric_evaluator,
                category="failure_mechanisms",
                tags=["creep", "coble", "nabarro_herring", "hard"],
                case_id="matscibench_087",
            )
        )

    def _msb_failures_corrosion(self) -> None:
        """Corrosion: galvanic/pitting/SCC (088-090)."""
        BenchmarkSuite._msb_failures_corrosion_corrosion(self)
        BenchmarkSuite._msb_failures_corrosion_corrosion_2(self)

    def _msb_failures_corrosion_corrosion(self) -> None:
        """Register matscibench_088, matscibench_089 (corrosion)."""
        self.add(
            BenchmarkCase(
                task="Explain galvanic corrosion. Given Zn (E = -0.76 V) and Cu (E = +0.34 V) "
                "coupled together, which metal corrodes? Calculate the cell potential. "
                "Answer in V.",
                expected_value=1.1,
                tolerance=0.05,
                evaluator=numeric_evaluator,
                category="failure_mechanisms",
                tags=["corrosion", "galvanic", "easy"],
                case_id="matscibench_088",
            )
        )
        self.add(
            BenchmarkCase(
                task="Describe pitting corrosion. How does the breakdown of passivity initiate "
                "a pit? Explain the autocatalytic mechanism inside the pit involving Cl⁻, "
                "H⁺, and metal dissolution.",
                rubric_items=[
                    {
                        "criterion": "passivity breakdown",
                        "weight": 2,
                        "keywords": ["passivity", "breakdown"],
                    },
                    {
                        "criterion": "autocatalytic mechanism",
                        "weight": 2,
                        "keywords": ["autocatalytic", "pit"],
                    },
                    {
                        "criterion": "Cl⁻ role",
                        "weight": 2,
                        "keywords": ["chloride", "cl"],
                    },
                ],
                evaluator=rubric_evaluator,
                category="failure_mechanisms",
                tags=["corrosion", "pitting", "medium"],
                case_id="matscibench_089",
            )
        )

    def _msb_failures_corrosion_corrosion_2(self) -> None:
        """Register matscibench_090 (corrosion)."""
        self.add(
            BenchmarkCase(
                task="Explain stress corrosion cracking (SCC). How does the synergistic effect "
                "of tensile stress and corrosive environment lead to crack propagation? "
                "Compare anodic dissolution and hydrogen embrittlement mechanisms. What "
                "is the role of the film rupture rate at the crack tip?",
                rubric_items=[
                    {
                        "criterion": "synergistic stress + environment",
                        "weight": 2,
                        "keywords": ["synergistic", "stress"],
                    },
                    {
                        "criterion": "anodic dissolution mechanism",
                        "weight": 2,
                        "keywords": ["anodic", "dissolution"],
                    },
                    {
                        "criterion": "hydrogen embrittlement",
                        "weight": 2,
                        "keywords": ["hydrogen", "embrittlement"],
                    },
                ],
                evaluator=rubric_evaluator,
                category="failure_mechanisms",
                tags=["corrosion", "scc", "stress_corrosion", "hard"],
                case_id="matscibench_090",
            )
        )

    def _msb_failures_wear(self) -> None:
        """Wear: abrasive-adhesive/Archard/fretting (091-093)."""
        BenchmarkSuite._msb_failures_wear_wear(self)
        BenchmarkSuite._msb_failures_wear_wear_2(self)

    def _msb_failures_wear_wear(self) -> None:
        """Register matscibench_091, matscibench_092 (wear)."""
        self.add(
            BenchmarkCase(
                task="Describe abrasive wear and adhesive wear. How do they differ in "
                "mechanism? Give one strategy to reduce each.",
                expected_keywords=["abrasive", "adhesive", "hardness", "lubrication"],
                category="failure_mechanisms",
                tags=["wear", "abrasive_adhesive", "easy"],
                case_id="matscibench_091",
            )
        )
        self.add(
            BenchmarkCase(
                task="Using Archard's wear equation V = K·F·s/H, where V is wear volume, F is "
                "normal load (50 N), s is sliding distance (1000 m), H is hardness (300 "
                "HV), and K = 1×10⁻⁴. Calculate the wear volume in mm³. Note: convert H "
                "to MPa (1 HV ≈ 9.81 MPa).",
                expected_value=1.7,
                tolerance=0.2,
                evaluator=numeric_evaluator,
                category="failure_mechanisms",
                tags=["wear", "archard", "medium"],
                case_id="matscibench_092",
            )
        )

    def _msb_failures_wear_wear_2(self) -> None:
        """Register matscibench_093 (wear)."""
        self.add(
            BenchmarkCase(
                task="Explain fretting wear. How does small-amplitude oscillatory motion (μm "
                "range) between two surfaces cause damage? Describe the role of oxide "
                "debris generation, debris ejection vs retention, and the transition from "
                "fretting wear to fretting fatigue. What surface treatments can mitigate "
                "fretting?",
                rubric_items=[
                    {
                        "criterion": "oscillatory motion mechanism",
                        "weight": 2,
                        "keywords": ["oscillatory", "amplitude"],
                    },
                    {
                        "criterion": "oxide debris role",
                        "weight": 2,
                        "keywords": ["oxide", "debris"],
                    },
                    {
                        "criterion": "fretting fatigue transition",
                        "weight": 2,
                        "keywords": ["fretting", "fatigue"],
                    },
                ],
                evaluator=rubric_evaluator,
                category="failure_mechanisms",
                tags=["wear", "fretting", "hard"],
                case_id="matscibench_093",
            )
        )

    def csmbench_cases(self) -> BenchmarkSuite:
        """CSMBench-style cases — cross-scale material science perception.
        Source: arXiv:2603.19327. 4 physical scales + cross-scale = 5 categories.
        8 cases per category = 40 total.
        """
        for build in (
            self._csm_atomic_a,
            self._csm_atomic_b,
            self._csm_micro_a,
            self._csm_micro_b,
            self._csm_meso_a,
            self._csm_meso_b,
            self._csm_macro_a,
            self._csm_macro_b,
            self._csm_cross_a,
            self._csm_cross_b,
        ):
            build()
        return self

    def _csm_atomic_a(self) -> None:
        """Atomic scale 1/2: crystal structure, diffraction."""
        BenchmarkSuite._csm_atomic_a_xrd(self)
        BenchmarkSuite._csm_atomic_a_silicon(self)

    def _csm_atomic_a_xrd(self) -> None:
        """Register csmbench_001, csmbench_002, csmbench_003 (xrd, fcc, vacancy)."""
        # atomic scale: crystal structure, diffraction, point defects
        self.add(
            BenchmarkCase(
                case_id="csmbench_001",
                category="atomic",
                task="GaN crystallizes in the wurtzite structure. From XRD data the (100) "
                "peak is identified. Calculate the lattice parameter a in Å and "
                "compare with the accepted literature value.",
                expected_value=3.25,
                tolerance=0.05,
                evaluator=numeric_evaluator,
                tags=["xrd", "gan", "wurtzite", "lattice-parameter"],
            )
        )
        self.add(
            BenchmarkCase(
                case_id="csmbench_002",
                category="atomic",
                task="For FCC aluminum with lattice parameter a = 4.05 Å, calculate the "
                "interplanar spacing d_{111} in nm using d = a / sqrt(h² + k² + l²).",
                expected_value=0.234,
                tolerance=0.005,
                evaluator=numeric_evaluator,
                tags=["fcc", "interplanar-spacing", "aluminum"],
            )
        )
        self.add(
            BenchmarkCase(
                case_id="csmbench_003",
                category="atomic",
                task="Calculate the equilibrium vacancy fraction n_v/n in copper at 1000 K "
                "using n_v/n = exp(−Q_v / kT). Given Q_v = 0.9 eV and "
                "k = 8.617×10⁻⁵ eV/K.",
                expected_value=2.9e-5,
                tolerance=1e-5,
                evaluator=numeric_evaluator,
                tags=["vacancy", "thermodynamics", "copper", "point-defect"],
            )
        )

    def _csm_atomic_a_silicon(self) -> None:
        """Register csmbench_004 (silicon)."""
        self.add(
            BenchmarkCase(
                case_id="csmbench_004",
                category="atomic",
                task="Describe the crystal structure of silicon. Identify the Bravais "
                "lattice type, the number of atoms per conventional unit cell, and "
                "the local coordination geometry around each atom.",
                expected_keywords=["diamond", "cubic", "8", "tetrahedral"],
                evaluator=keyword_evaluator,
                tags=["silicon", "diamond-cubic", "crystal-structure"],
            )
        )

    def _csm_atomic_b(self) -> None:
        """Atomic scale 2/2: point defects."""
        BenchmarkSuite._csm_atomic_b_nacl(self)
        BenchmarkSuite._csm_atomic_b_apf(self)

    def _csm_atomic_b_nacl(self) -> None:
        """Register csmbench_005, csmbench_006, csmbench_007 (nacl, bcc, bragg-law)."""
        self.add(
            BenchmarkCase(
                case_id="csmbench_005",
                category="atomic",
                task="NaCl has the rocksalt structure with lattice parameter a = 5.64 Å. "
                "What is the nearest-neighbor Na–Cl distance in nm?",
                expected_value=0.282,
                tolerance=0.005,
                evaluator=numeric_evaluator,
                tags=["nacl", "rocksalt", "nearest-neighbor"],
            )
        )
        self.add(
            BenchmarkCase(
                case_id="csmbench_006",
                category="atomic",
                task="Calculate the atomic packing factor (APF) of a BCC crystal. Show "
                "the derivation from the ratio of atom volume to unit cell volume.",
                expected_value=0.68,
                tolerance=0.01,
                evaluator=numeric_evaluator,
                tags=["bcc", "apf", "packing-factor"],
            )
        )
        self.add(
            BenchmarkCase(
                case_id="csmbench_007",
                category="atomic",
                task="A diffraction peak is observed at 2θ = 28.4° using Cu Kα radiation "
                "(λ = 1.5406 Å). Calculate the d-spacing in nm using Bragg's law "
                "nλ = 2d sinθ (assume n = 1).",
                expected_value=0.314,
                tolerance=0.005,
                evaluator=numeric_evaluator,
                tags=["bragg-law", "xrd", "d-spacing", "copper-radiation"],
            )
        )

    def _csm_atomic_b_apf(self) -> None:
        """Register csmbench_008 (apf)."""
        self.add(
            BenchmarkCase(
                case_id="csmbench_008",
                category="atomic",
                task="Compare the atomic packing factors of FCC, BCC, and HCP crystal "
                "structures. Provide the numerical APF value for each and explain "
                "why FCC and HCP have the same packing efficiency.",
                rubric="Evaluate APF values for FCC, BCC, and HCP with explanation.",
                rubric_items=[
                    {
                        "criterion": "FCC APF value",
                        "weight": 0.33,
                        "keywords": ["fcc", "0.74"],
                    },
                    {
                        "criterion": "BCC APF value",
                        "weight": 0.34,
                        "keywords": ["bcc", "0.68"],
                    },
                    {
                        "criterion": "HCP APF value",
                        "weight": 0.33,
                        "keywords": ["hcp", "0.74"],
                    },
                ],
                evaluator=rubric_evaluator,
                tags=["apf", "fcc", "bcc", "hcp", "comparison"],
            )
        )

    def _csm_micro_a(self) -> None:
        """Micro scale 1/2: dislocations, TEM."""
        BenchmarkSuite._csm_micro_a_tem(self)
        BenchmarkSuite._csm_micro_a_tem_2(self)
        BenchmarkSuite._csm_micro_a_nucleation(self)

    def _csm_micro_a_tem(self) -> None:
        """Register csmbench_009, csmbench_010 (tem, burgers-vector)."""
        # micro scale: dislocations, TEM, precipitates, interfaces
        self.add(
            BenchmarkCase(
                case_id="csmbench_009",
                category="micro",
                task="An HRTEM image shows lattice fringes with a measured spacing of "
                "0.334 nm. Identify the material and the specific crystallographic "
                "plane that produces this spacing.",
                expected_keywords=["graphite", "0.334", "d-spacing", "002"],
                evaluator=keyword_evaluator,
                tags=["tem", "graphite", "lattice-fringes", "hrtem"],
            )
        )
        self.add(
            BenchmarkCase(
                case_id="csmbench_010",
                category="micro",
                task="For an FCC crystal (Al, a = 4.05 Å), calculate the magnitude of "
                "the perfect dislocation Burgers vector |b| = a/√2 in nm.",
                expected_value=0.286,
                tolerance=0.005,
                evaluator=numeric_evaluator,
                tags=["burgers-vector", "fcc", "dislocation", "aluminum"],
            )
        )

    def _csm_micro_a_tem_2(self) -> None:
        """Register csmbench_011 (tem)."""
        self.add(
            BenchmarkCase(
                case_id="csmbench_011",
                category="micro",
                task="Compare TEM and SEM in terms of working principle, achievable "
                "resolution, and sample preparation requirements.",
                rubric="Evaluate TEM vs SEM comparison across principle, resolution, and sample prep.",
                rubric_items=[
                    {
                        "criterion": "TEM working principle",
                        "weight": 0.3,
                        "keywords": ["transmission", "thin"],
                    },
                    {
                        "criterion": "SEM working principle",
                        "weight": 0.3,
                        "keywords": ["scanning", "secondary"],
                    },
                    {
                        "criterion": "Resolution comparison",
                        "weight": 0.2,
                        "keywords": ["resolution", "tem"],
                    },
                    {
                        "criterion": "Sample preparation",
                        "weight": 0.2,
                        "keywords": ["sample", "preparation"],
                    },
                ],
                evaluator=rubric_evaluator,
                tags=["tem", "sem", "comparison", "characterization"],
            )
        )

    def _csm_micro_a_nucleation(self) -> None:
        """Register csmbench_012 (nucleation)."""
        self.add(
            BenchmarkCase(
                case_id="csmbench_012",
                category="micro",
                task="Calculate the volume free energy change ΔGv (the driving force for "
                "precipitate nucleation) using ΔGv = −ΔHf·ΔT/Tm. Given "
                "ΔHf = 2.5×10⁹ J/m³, ΔT = 300 K, Tm = 1500 K. Express the "
                "result in J/m³.",
                expected_value=-5.0e8,
                tolerance=1e8,
                evaluator=numeric_evaluator,
                tags=["nucleation", "precipitate", "thermodynamics", "driving-force"],
            )
        )

    def _csm_micro_b(self) -> None:
        """Micro scale 2/2: precipitates, interfaces."""
        BenchmarkSuite._csm_micro_b_hrtem(self)
        BenchmarkSuite._csm_micro_b_precipitate(self)

    def _csm_micro_b_hrtem(self) -> None:
        """Register csmbench_013, csmbench_014, csmbench_015 (hrtem, dislocation, stacking-fault)."""
        self.add(
            BenchmarkCase(
                case_id="csmbench_013",
                category="micro",
                task="Explain how HRTEM imaging combined with FFT analysis is used to "
                "identify crystal defects at the atomic scale.",
                expected_keywords=["hrtem", "fft", "fourier", "lattice"],
                evaluator=keyword_evaluator,
                tags=["hrtem", "fft", "defect", "characterization"],
            )
        )
        self.add(
            BenchmarkCase(
                case_id="csmbench_014",
                category="micro",
                task="Estimate the dislocation line energy per unit length for copper "
                "(G ≈ 48 GPa, b ≈ 0.256 nm) using E ≈ Gb²/2. Give the result "
                "in J/m.",
                expected_value=2.0e-9,
                tolerance=1e-9,
                evaluator=numeric_evaluator,
                tags=["dislocation", "line-energy", "copper", "elastic"],
            )
        )
        self.add(
            BenchmarkCase(
                case_id="csmbench_015",
                category="micro",
                task="Discuss how stacking fault energy varies among FCC metals and its "
                "influence on the deformation behavior of austenitic stainless steels.",
                expected_keywords=["stacking fault", "energy", "fcc", "austenitic"],
                evaluator=keyword_evaluator,
                tags=["stacking-fault", "energy", "fcc", "austenitic"],
            )
        )

    def _csm_micro_b_precipitate(self) -> None:
        """Register csmbench_016 (precipitate)."""
        self.add(
            BenchmarkCase(
                case_id="csmbench_016",
                category="micro",
                task="Explain the difference between coherent, semi-coherent, and "
                "incoherent precipitate–matrix interfaces. Discuss lattice strain "
                "and misfit dislocations for each case.",
                rubric="Evaluate coherent vs incoherent interface understanding.",
                rubric_items=[
                    {
                        "criterion": "Coherent interface + strain",
                        "weight": 0.4,
                        "keywords": ["coherent", "strain"],
                    },
                    {
                        "criterion": "Incoherent interface",
                        "weight": 0.3,
                        "keywords": ["incoherent", "strain"],
                    },
                    {
                        "criterion": "Misfit / semi-coherent",
                        "weight": 0.3,
                        "keywords": ["misfit", "semi-coherent"],
                    },
                ],
                evaluator=rubric_evaluator,
                tags=["precipitate", "interface", "coherent", "incoherent", "misfit"],
            )
        )

    def _csm_meso_a(self) -> None:
        """Meso scale 1/2: grains, porosity."""
        BenchmarkSuite._csm_meso_a_sintering(self)
        BenchmarkSuite._csm_meso_a_hall_petch(self)

    def _csm_meso_a_sintering(self) -> None:
        """Register csmbench_017 (sintering)."""
        # meso scale: grains, porosity, fracture, phase distribution
        self.add(
            BenchmarkCase(
                case_id="csmbench_017",
                category="meso",
                task="Analyze the relationship between sintering parameters, grain growth, "
                "density, and mechanical properties via the Hall-Petch relation. "
                "Discuss the trade-off between densification and grain coarsening.",
                rubric="Evaluate sintering–grain growth–Hall-Petch chain reasoning.",
                rubric_items=[
                    {
                        "criterion": "Grain growth during sintering",
                        "weight": 0.25,
                        "keywords": ["grain", "growth"],
                    },
                    {
                        "criterion": "Density and densification",
                        "weight": 0.25,
                        "keywords": ["density", "sintering"],
                    },
                    {
                        "criterion": "Hall-Petch strengthening",
                        "weight": 0.25,
                        "keywords": ["hall-petch", "strength"],
                    },
                    {
                        "criterion": "Trade-off discussion",
                        "weight": 0.25,
                        "keywords": ["trade-off", "balance"],
                    },
                ],
                evaluator=rubric_evaluator,
                tags=["sintering", "hall-petch", "grain-growth", "density"],
            )
        )

    def _csm_meso_a_hall_petch(self) -> None:
        """Register csmbench_018, csmbench_019, csmbench_020 (hall-petch, ebsd, porosity)."""
        self.add(
            BenchmarkCase(
                case_id="csmbench_018",
                category="meso",
                task="Calculate the yield strength using the Hall-Petch relation "
                "σ_y = σ₀ + k/√d. Given σ₀ = 150 MPa, k = 0.45 MPa·m^0.5, "
                "and grain size d = 10 μm. Give the result in MPa.",
                expected_value=292.3,
                tolerance=5.0,
                evaluator=numeric_evaluator,
                tags=["hall-petch", "yield-strength", "grain-size"],
            )
        )
        self.add(
            BenchmarkCase(
                case_id="csmbench_019",
                category="meso",
                task="Describe how a bimodal grain size distribution, as revealed by EBSD, "
                "affects the strength-ductility balance in titanium alloys.",
                expected_keywords=[
                    "bimodal",
                    "grain",
                    "titanium",
                    "strength",
                    "ductility",
                ],
                evaluator=keyword_evaluator,
                tags=["ebsd", "bimodal", "titanium", "grain-size"],
            )
        )
        self.add(
            BenchmarkCase(
                case_id="csmbench_020",
                category="meso",
                task="Al₂O₃ has a theoretical (fully dense) elastic modulus E₀ = 400 GPa. "
                "Using the Spriggs equation E = E₀(1 − 1.9P + 0.9P²), calculate the "
                "modulus at 16% porosity. Give the result in GPa.",
                expected_value=284.8,
                tolerance=5.0,
                evaluator=numeric_evaluator,
                tags=["porosity", "elastic-modulus", "alumina", "spriggs"],
            )
        )

    def _csm_meso_b(self) -> None:
        """Meso scale 2/2: fracture, phase distribution."""
        BenchmarkSuite._csm_meso_b_intergranular(self)
        BenchmarkSuite._csm_meso_b_phase_distribution(self)

    def _csm_meso_b_intergranular(self) -> None:
        """Register csmbench_021, csmbench_022 (intergranular, grain-growth)."""
        self.add(
            BenchmarkCase(
                case_id="csmbench_021",
                category="meso",
                task="Explain the mechanisms of intergranular fracture, including the role "
                "of grain boundary segregation and impurity embrittlement.",
                expected_keywords=[
                    "intergranular",
                    "grain boundary",
                    "segregation",
                    "embrittlement",
                ],
                evaluator=keyword_evaluator,
                tags=["intergranular", "fracture", "grain-boundary", "embrittlement"],
            )
        )
        self.add(
            BenchmarkCase(
                case_id="csmbench_022",
                category="meso",
                task="In the grain growth kinetics relation d^n = Kt, what is the grain "
                "growth exponent n for normal grain growth in pure single-phase "
                "materials?",
                expected_value=2.0,
                tolerance=0.5,
                evaluator=numeric_evaluator,
                tags=["grain-growth", "kinetics", "exponent"],
            )
        )

    def _csm_meso_b_phase_distribution(self) -> None:
        """Register csmbench_023, csmbench_024 (phase-distribution, crack)."""
        self.add(
            BenchmarkCase(
                case_id="csmbench_023",
                category="meso",
                task="Describe the experimental methods for determining phase fractions "
                "in a multiphase alloy. Cover microscopy-based and diffraction-based "
                "techniques.",
                rubric="Evaluate phase fraction characterization methods.",
                rubric_items=[
                    {
                        "criterion": "Image analysis / microscopy",
                        "weight": 0.25,
                        "keywords": ["image", "microscopy"],
                    },
                    {
                        "criterion": "EBSD phase mapping",
                        "weight": 0.25,
                        "keywords": ["ebsd", "phase"],
                    },
                    {
                        "criterion": "XRD Rietveld refinement",
                        "weight": 0.25,
                        "keywords": ["xrd", "rietveld"],
                    },
                    {
                        "criterion": "Phase fraction quantification",
                        "weight": 0.25,
                        "keywords": ["fraction", "quantification"],
                    },
                ],
                evaluator=rubric_evaluator,
                tags=["phase-distribution", "multiphase", "ebsd", "xrd"],
            )
        )
        self.add(
            BenchmarkCase(
                case_id="csmbench_024",
                category="meso",
                task="Describe how weak grain boundaries influence crack propagation "
                "paths and crack deflection in polycrystalline ceramics.",
                expected_keywords=["crack", "grain boundary", "path", "deflection"],
                evaluator=keyword_evaluator,
                tags=["crack", "grain-boundary", "propagation", "deflection"],
            )
        )

    def _csm_macro_a(self) -> None:
        """Macro scale 1/2: mechanical testing, thermal."""
        BenchmarkSuite._csm_macro_a_tensile_test(self)
        BenchmarkSuite._csm_macro_a_chevron(self)

    def _csm_macro_a_tensile_test(self) -> None:
        """Register csmbench_025, csmbench_026 (tensile-test, vickers)."""
        # macro scale: mechanical testing, thermal, fatigue, creep
        self.add(
            BenchmarkCase(
                case_id="csmbench_025",
                category="macro",
                task="A tensile test is performed on a high-strength steel sample. From "
                "the stress-strain curve, determine the Young's modulus (~120 GPa), "
                "the ultimate tensile strength (~1200 MPa), and the elongation at "
                "failure (~16%). Report all three values.",
                rubric="Evaluate extraction of Young's modulus, UTS, and elongation from tensile data.",
                rubric_items=[
                    {
                        "criterion": "Young's modulus",
                        "weight": 0.33,
                        "keywords": ["120", "gpa"],
                    },
                    {
                        "criterion": "Ultimate tensile strength",
                        "weight": 0.34,
                        "keywords": ["1200", "mpa"],
                    },
                    {
                        "criterion": "Elongation at failure",
                        "weight": 0.33,
                        "keywords": ["16", "elongation"],
                    },
                ],
                evaluator=rubric_evaluator,
                tags=["tensile-test", "youngs-modulus", "uts", "elongation"],
            )
        )
        self.add(
            BenchmarkCase(
                case_id="csmbench_026",
                category="macro",
                task="A Vickers hardness test uses a load of 98.1 N (10 kgf). The average "
                "diagonal of the indentation measures 0.300 mm. Calculate the "
                "Vickers hardness number HV using HV = 1.8544·F/d² (F in kgf, "
                "d in mm).",
                expected_value=206.0,
                tolerance=5.0,
                evaluator=numeric_evaluator,
                tags=["vickers", "hardness", "indentation"],
            )
        )

    def _csm_macro_a_chevron(self) -> None:
        """Register csmbench_027, csmbench_028 (chevron, bending)."""
        self.add(
            BenchmarkCase(
                case_id="csmbench_027",
                category="macro",
                task="Describe the chevron fracture pattern. What does it reveal about "
                "crack origin and propagation direction in brittle materials?",
                expected_keywords=[
                    "chevron",
                    "crack",
                    "origin",
                    "propagation",
                    "brittle",
                ],
                evaluator=keyword_evaluator,
                tags=["chevron", "fracture", "crack", "brittle"],
            )
        )
        self.add(
            BenchmarkCase(
                case_id="csmbench_028",
                category="macro",
                task="A 3-point bending test is performed with load F = 800 N, support "
                "span L = 100 mm, and a rectangular cross-section b = 10 mm, "
                "h = 5 mm. Calculate the maximum bending stress σ = 3FL/(2bh²) "
                "in MPa.",
                expected_value=480.0,
                tolerance=10.0,
                evaluator=numeric_evaluator,
                tags=["bending", "stress", "three-point", "flexural"],
            )
        )

    def _csm_macro_b(self) -> None:
        """Macro scale 2/2: fatigue, creep."""
        BenchmarkSuite._csm_macro_b_thermal_expansion(self)
        BenchmarkSuite._csm_macro_b_creep(self)

    def _csm_macro_b_thermal_expansion(self) -> None:
        """Register csmbench_029, csmbench_030, csmbench_031 (thermal-expansion, charpy, fatigue)."""
        self.add(
            BenchmarkCase(
                case_id="csmbench_029",
                category="macro",
                task="A steel rod is heated from 25°C to 225°C (ΔT = 200 K). Its length "
                "changes from 100.000 mm to 100.192 mm. Calculate the coefficient "
                "of thermal expansion in units of 10⁻⁶ /°C.",
                expected_value=9.6,
                tolerance=0.2,
                evaluator=numeric_evaluator,
                tags=["thermal-expansion", "cte", "steel"],
            )
        )
        self.add(
            BenchmarkCase(
                case_id="csmbench_030",
                category="macro",
                task="Describe the Charpy V-notch impact test and explain the "
                "ductile-to-brittle transition temperature phenomenon.",
                expected_keywords=[
                    "charpy",
                    "impact",
                    "toughness",
                    "ductile",
                    "brittle",
                ],
                evaluator=keyword_evaluator,
                tags=["charpy", "impact", "toughness", "ductile-brittle"],
            )
        )
        self.add(
            BenchmarkCase(
                case_id="csmbench_031",
                category="macro",
                task="Using the Basquin relation σa = σ′f · (Nf)^b with σa = 300 MPa, "
                "σ′f = 1200 MPa, b = −0.10, calculate the fatigue life Nf "
                "(cycles to failure).",
                expected_value=1e6,
                tolerance=2e5,
                evaluator=numeric_evaluator,
                tags=["fatigue", "basquin", "s-n", "life-prediction"],
            )
        )

    def _csm_macro_b_creep(self) -> None:
        """Register csmbench_032 (creep)."""
        self.add(
            BenchmarkCase(
                case_id="csmbench_032",
                category="macro",
                task="Describe the three stages of a typical creep curve (strain vs time "
                "at constant stress and temperature). Explain the strain-rate "
                "behavior in each stage.",
                rubric="Evaluate identification of three creep stages and their strain-rate behavior.",
                rubric_items=[
                    {
                        "criterion": "Primary creep",
                        "weight": 0.33,
                        "keywords": ["primary", "decreasing"],
                    },
                    {
                        "criterion": "Secondary creep",
                        "weight": 0.34,
                        "keywords": ["secondary", "steady"],
                    },
                    {
                        "criterion": "Tertiary creep",
                        "weight": 0.33,
                        "keywords": ["tertiary", "accelerating"],
                    },
                ],
                evaluator=rubric_evaluator,
                tags=["creep", "deformation", "stages", "high-temperature"],
            )
        )

    def _csm_cross_a(self) -> None:
        """Cross-scale 1/2: multi-scale causal chains."""
        BenchmarkSuite._csm_cross_a_bonding(self)
        BenchmarkSuite._csm_cross_a_dopant(self)
        BenchmarkSuite._csm_cross_a_vacancy(self)
        BenchmarkSuite._csm_cross_a_sintering(self)

    def _csm_cross_a_bonding(self) -> None:
        """Register csmbench_033 (bonding)."""
        # cross-scale reasoning: multi-scale causal chains
        self.add(
            BenchmarkCase(
                case_id="csmbench_033",
                category="cross_scale",
                task="Trace the chain from atomic bonding type to dislocation behavior to "
                "macroscopic ductility. How does the bonding character (covalent vs "
                "metallic) determine whether a material is brittle or ductile?",
                rubric="Evaluate cross-scale bonding → dislocation → ductility reasoning.",
                rubric_items=[
                    {
                        "criterion": "Bonding character",
                        "weight": 0.25,
                        "keywords": ["covalent", "metallic"],
                    },
                    {
                        "criterion": "Dislocation mobility",
                        "weight": 0.25,
                        "keywords": ["dislocation", "mobility"],
                    },
                    {
                        "criterion": "Grain boundary barriers",
                        "weight": 0.25,
                        "keywords": ["grain", "boundary"],
                    },
                    {
                        "criterion": "Macroscopic ductility",
                        "weight": 0.25,
                        "keywords": ["brittle", "ductile"],
                    },
                ],
                evaluator=rubric_evaluator,
                tags=["bonding", "dislocation", "ductility", "cross-scale"],
            )
        )

    def _csm_cross_a_dopant(self) -> None:
        """Register csmbench_034 (dopant)."""
        self.add(
            BenchmarkCase(
                case_id="csmbench_034",
                category="cross_scale",
                task="Trace the causal chain from dopant substitution to electrical "
                "conductivity: dopant incorporation → carrier concentration → "
                "carrier scattering → macroscopic conductivity.",
                rubric="Evaluate dopant → carrier → scattering → conductivity chain.",
                rubric_items=[
                    {
                        "criterion": "Dopant substitution",
                        "weight": 0.25,
                        "keywords": ["substitution", "dopant"],
                    },
                    {
                        "criterion": "Carrier concentration",
                        "weight": 0.25,
                        "keywords": ["carrier", "electron"],
                    },
                    {
                        "criterion": "Scattering and mobility",
                        "weight": 0.25,
                        "keywords": ["scattering", "mobility"],
                    },
                    {
                        "criterion": "Conductivity outcome",
                        "weight": 0.25,
                        "keywords": ["conductivity", "resistivity"],
                    },
                ],
                evaluator=rubric_evaluator,
                tags=["dopant", "carrier", "scattering", "conductivity"],
            )
        )

    def _csm_cross_a_vacancy(self) -> None:
        """Register csmbench_035 (vacancy)."""
        self.add(
            BenchmarkCase(
                case_id="csmbench_035",
                category="cross_scale",
                task="Explain how atomic vacancies, porosity, and grain boundaries "
                "collectively affect thermal conductivity across length scales. "
                "Discuss phonon scattering at each scale.",
                rubric="Evaluate vacancy + porosity → thermal conductivity cross-scale reasoning.",
                rubric_items=[
                    {
                        "criterion": "Vacancy phonon scattering",
                        "weight": 0.25,
                        "keywords": ["vacancy", "phonon"],
                    },
                    {
                        "criterion": "Porosity effect",
                        "weight": 0.25,
                        "keywords": ["porosity", "pore"],
                    },
                    {
                        "criterion": "Grain boundary scattering",
                        "weight": 0.25,
                        "keywords": ["grain", "boundary"],
                    },
                    {
                        "criterion": "Temperature dependence",
                        "weight": 0.25,
                        "keywords": ["temperature", "thermal"],
                    },
                ],
                evaluator=rubric_evaluator,
                tags=["vacancy", "porosity", "thermal-conductivity", "phonon"],
            )
        )

    def _csm_cross_a_sintering(self) -> None:
        """Register csmbench_036 (sintering)."""
        self.add(
            BenchmarkCase(
                case_id="csmbench_036",
                category="cross_scale",
                task="Connect sintering to mechanical strength through the chain: "
                "diffusion → grain growth → Hall-Petch strengthening → yield "
                "strength. Explain each link.",
                rubric="Evaluate sintering → diffusion → grain → strength chain.",
                rubric_items=[
                    {
                        "criterion": "Diffusion mechanism",
                        "weight": 0.25,
                        "keywords": ["diffusion", "fick"],
                    },
                    {
                        "criterion": "Grain growth",
                        "weight": 0.25,
                        "keywords": ["grain", "growth"],
                    },
                    {
                        "criterion": "Hall-Petch strengthening",
                        "weight": 0.25,
                        "keywords": ["hall-petch", "strength"],
                    },
                    {
                        "criterion": "Yield strength outcome",
                        "weight": 0.25,
                        "keywords": ["yield", "strength"],
                    },
                ],
                evaluator=rubric_evaluator,
                tags=["sintering", "diffusion", "grain-growth", "hall-petch"],
            )
        )

    def _csm_cross_b(self) -> None:
        """Cross-scale 2/2: multi-scale causal chains."""
        BenchmarkSuite._csm_cross_b_processing(self)
        BenchmarkSuite._csm_cross_b_band_structure(self)
        BenchmarkSuite._csm_cross_b_dislocation(self)
        BenchmarkSuite._csm_cross_b_composition(self)

    def _csm_cross_b_processing(self) -> None:
        """Register csmbench_037 (processing)."""
        self.add(
            BenchmarkCase(
                case_id="csmbench_037",
                category="cross_scale",
                task="Describe how processing parameters (e.g., cooling rate) determine "
                "microstructure, which in turn controls material properties. Give "
                "a concrete example of the processing–structure–property chain.",
                rubric="Evaluate processing → microstructure → property chain reasoning.",
                rubric_items=[
                    {
                        "criterion": "Cooling rate effect",
                        "weight": 0.25,
                        "keywords": ["cooling", "rate"],
                    },
                    {
                        "criterion": "Microstructure formation",
                        "weight": 0.25,
                        "keywords": ["microstructure", "phase"],
                    },
                    {
                        "criterion": "Grain and phase control",
                        "weight": 0.25,
                        "keywords": ["grain", "phase"],
                    },
                    {
                        "criterion": "Property prediction",
                        "weight": 0.25,
                        "keywords": ["property", "strength"],
                    },
                ],
                evaluator=rubric_evaluator,
                tags=["processing", "microstructure", "property", "chain"],
            )
        )

    def _csm_cross_b_band_structure(self) -> None:
        """Register csmbench_038 (band-structure)."""
        self.add(
            BenchmarkCase(
                case_id="csmbench_038",
                category="cross_scale",
                task="Trace how atomic-level crystal structure determines the electronic "
                "band structure, which in turn governs optical absorption and "
                "emission properties. Cover band gap, DOS, and optical transitions.",
                rubric="Evaluate atomic structure → electronic → optical chain.",
                rubric_items=[
                    {
                        "criterion": "Band structure / band gap",
                        "weight": 0.25,
                        "keywords": ["band", "structure"],
                    },
                    {
                        "criterion": "Density of states",
                        "weight": 0.25,
                        "keywords": ["density", "states"],
                    },
                    {
                        "criterion": "Optical absorption",
                        "weight": 0.25,
                        "keywords": ["absorption", "optical"],
                    },
                    {
                        "criterion": "Emission / luminescence",
                        "weight": 0.25,
                        "keywords": ["emission", "luminescence"],
                    },
                ],
                evaluator=rubric_evaluator,
                tags=["band-structure", "density-of-states", "optical", "absorption"],
            )
        )

    def _csm_cross_b_dislocation(self) -> None:
        """Register csmbench_039 (dislocation)."""
        self.add(
            BenchmarkCase(
                case_id="csmbench_039",
                category="cross_scale",
                task="Describe the cross-scale progression from dislocation pile-up at "
                "grain boundaries to crack initiation, crack propagation, and "
                "final macroscopic fracture.",
                rubric="Evaluate defect → meso crack → macro fracture chain.",
                rubric_items=[
                    {
                        "criterion": "Dislocation pile-up",
                        "weight": 0.25,
                        "keywords": ["dislocation", "pile-up"],
                    },
                    {
                        "criterion": "Crack initiation",
                        "weight": 0.25,
                        "keywords": ["crack", "initiation"],
                    },
                    {
                        "criterion": "Crack propagation",
                        "weight": 0.25,
                        "keywords": ["crack", "propagation"],
                    },
                    {
                        "criterion": "Final fracture",
                        "weight": 0.25,
                        "keywords": ["fracture", "failure"],
                    },
                ],
                evaluator=rubric_evaluator,
                tags=["dislocation", "crack", "fracture", "cross-scale"],
            )
        )

    def _csm_cross_b_composition(self) -> None:
        """Register csmbench_040 (composition)."""
        self.add(
            BenchmarkCase(
                case_id="csmbench_040",
                category="cross_scale",
                task="Explain how composition and phase diagram analysis are used to "
                "predict equilibrium phases and resulting material properties. "
                "Give an example of composition → phase diagram → property "
                "prediction.",
                rubric="Evaluate composition → phase diagram → property chain.",
                rubric_items=[
                    {
                        "criterion": "Composition analysis",
                        "weight": 0.25,
                        "keywords": ["composition", "alloy"],
                    },
                    {
                        "criterion": "Phase diagram usage",
                        "weight": 0.25,
                        "keywords": ["phase", "diagram"],
                    },
                    {
                        "criterion": "Equilibrium phases",
                        "weight": 0.25,
                        "keywords": ["equilibrium", "phase"],
                    },
                    {
                        "criterion": "Property prediction",
                        "weight": 0.25,
                        "keywords": ["property", "prediction"],
                    },
                ],
                evaluator=rubric_evaluator,
                tags=["composition", "phase-diagram", "equilibrium", "property"],
            )
        )

    def tool_calling_cases(self) -> BenchmarkSuite:
        """Tool-calling benchmarks: pick the right method/code for a matsci problem.

        10 cases covering DFT, MD, XRD, CALPHAD, phonons, NEB, elastic constants,
        surface energy, GW correction, and ELF analysis. Each case checks whether
        the agent names the appropriate computational method or software.
        """
        BenchmarkSuite._tool_calling_cases_dft(self)
        BenchmarkSuite._tool_calling_cases_xrd(self)
        BenchmarkSuite._tool_calling_cases_phonon(self)
        BenchmarkSuite._tool_calling_cases_elastic(self)
        BenchmarkSuite._tool_calling_cases_surface_energy(self)
        BenchmarkSuite._tool_calling_cases_gw(self)
        return self

    def _tool_calling_cases_dft(self) -> None:
        """Register tc_001, tc_002 (dft, md)."""
        # DFT band structure
        self.add(
            BenchmarkCase(
                case_id="tc_001",
                category="tool_calling",
                task="我想计算硅的能带结构，应该用什么计算方法和软件？",
                rubric_items=[
                    {"criterion": "recommends DFT", "weight": 3, "keywords": ["dft"]},
                    {"criterion": "mentions VASP", "weight": 1, "keywords": ["vasp"]},
                    {
                        "criterion": "mentions Quantum ESPRESSO",
                        "weight": 1,
                        "keywords": ["quantum espresso"],
                    },
                ],
                evaluator=rubric_evaluator,
                tags=["dft", "band-structure", "silicon"],
            )
        )
        # MD diffusion coefficient
        self.add(
            BenchmarkCase(
                case_id="tc_002",
                category="tool_calling",
                task="如何计算锂离子在固态电解质中的扩散系数？",
                rubric_items=[
                    {
                        "criterion": "recommends molecular dynamics",
                        "weight": 3,
                        "keywords": ["molecular dynamics"],
                    },
                    {
                        "criterion": "mentions LAMMPS",
                        "weight": 1,
                        "keywords": ["lammps"],
                    },
                    {
                        "criterion": "mentions diffusion coefficient / MSD",
                        "weight": 1,
                        "keywords": ["diffusion coefficient"],
                    },
                ],
                evaluator=rubric_evaluator,
                tags=["md", "diffusion", "lammps"],
            )
        )

    def _tool_calling_cases_xrd(self) -> None:
        """Register tc_003, tc_004 (xrd, calphad)."""
        # XRD pattern analysis
        self.add(
            BenchmarkCase(
                case_id="tc_003",
                category="tool_calling",
                task="我有一组粉末衍射数据，如何进行物相鉴定和结构精修？",
                rubric_items=[
                    {"criterion": "mentions XRD", "weight": 2, "keywords": ["xrd"]},
                    {
                        "criterion": "mentions Bragg law",
                        "weight": 1,
                        "keywords": ["bragg"],
                    },
                    {
                        "criterion": "mentions Rietveld refinement",
                        "weight": 2,
                        "keywords": ["rietveld"],
                    },
                ],
                evaluator=rubric_evaluator,
                tags=["xrd", "rietveld", "phase-id"],
            )
        )
        # CALPHAD phase diagram
        self.add(
            BenchmarkCase(
                case_id="tc_004",
                category="tool_calling",
                task="如何计算Cu-Ag二元合金的相图？",
                rubric_items=[
                    {
                        "criterion": "recommends CALPHAD",
                        "weight": 3,
                        "keywords": ["calphad"],
                    },
                    {
                        "criterion": "mentions phase diagram",
                        "weight": 2,
                        "keywords": ["phase diagram"],
                    },
                ],
                evaluator=rubric_evaluator,
                tags=["calphad", "phase-diagram", "alloy"],
            )
        )

    def _tool_calling_cases_phonon(self) -> None:
        """Register tc_005, tc_006 (phonon, neb)."""
        # phonon dispersion
        self.add(
            BenchmarkCase(
                case_id="tc_005",
                category="tool_calling",
                task="如何计算金刚石的声子色散关系？",
                rubric_items=[
                    {
                        "criterion": "mentions phonon",
                        "weight": 2,
                        "keywords": ["phonon"],
                    },
                    {"criterion": "mentions DFPT", "weight": 2, "keywords": ["dfpt"]},
                    {
                        "criterion": "mentions finite displacement",
                        "weight": 1,
                        "keywords": ["finite displacement"],
                    },
                ],
                evaluator=rubric_evaluator,
                tags=["phonon", "dfpt", "diamond"],
            )
        )
        # NEB migration barrier
        self.add(
            BenchmarkCase(
                case_id="tc_006",
                category="tool_calling",
                task="如何计算锂离子在FePO4中的迁移势垒？",
                rubric_items=[
                    {"criterion": "mentions NEB", "weight": 3, "keywords": ["neb"]},
                    {
                        "criterion": "mentions nudged elastic band",
                        "weight": 1,
                        "keywords": ["nudged elastic band"],
                    },
                    {
                        "criterion": "mentions migration barrier",
                        "weight": 1,
                        "keywords": ["migration barrier"],
                    },
                ],
                evaluator=rubric_evaluator,
                tags=["neb", "migration-barrier", "lithium"],
            )
        )

    def _tool_calling_cases_elastic(self) -> None:
        """Register tc_007 (elastic)."""
        # elastic constants
        self.add(
            BenchmarkCase(
                case_id="tc_007",
                category="tool_calling",
                task="如何计算Mg合金的弹性常数？",
                rubric_items=[
                    {
                        "criterion": "mentions elastic constants",
                        "weight": 2,
                        "keywords": ["elastic constants"],
                    },
                    {
                        "criterion": "mentions stress-strain method",
                        "weight": 2,
                        "keywords": ["stress-strain"],
                    },
                ],
                evaluator=rubric_evaluator,
                tags=["elastic", "stress-strain", "magnesium"],
            )
        )

    def _tool_calling_cases_surface_energy(self) -> None:
        """Register tc_008 (surface-energy)."""
        # surface energy
        self.add(
            BenchmarkCase(
                case_id="tc_008",
                category="tool_calling",
                task="如何计算Pt(111)表面的表面能？",
                rubric_items=[
                    {
                        "criterion": "mentions surface energy",
                        "weight": 2,
                        "keywords": ["surface energy"],
                    },
                    {
                        "criterion": "mentions slab model",
                        "weight": 2,
                        "keywords": ["slab"],
                    },
                    {
                        "criterion": "mentions vacuum spacing",
                        "weight": 1,
                        "keywords": ["vacuum"],
                    },
                ],
                evaluator=rubric_evaluator,
                tags=["surface-energy", "slab", "platinum"],
            )
        )

    def _tool_calling_cases_gw(self) -> None:
        """Register tc_009, tc_010 (gw, elf)."""
        # GW correction
        self.add(
            BenchmarkCase(
                case_id="tc_009",
                category="tool_calling",
                task="DFT-PBE低估了ZnO的带隙，如何获得更准确的带隙？",
                rubric_items=[
                    {"criterion": "mentions GW", "weight": 3, "keywords": ["gw"]},
                    {
                        "criterion": "mentions quasiparticle",
                        "weight": 1,
                        "keywords": ["quasiparticle"],
                    },
                    {
                        "criterion": "mentions many-body perturbation",
                        "weight": 1,
                        "keywords": ["many-body perturbation"],
                    },
                ],
                evaluator=rubric_evaluator,
                tags=["gw", "quasiparticle", "zno"],
            )
        )
        # ELF analysis
        self.add(
            BenchmarkCase(
                case_id="tc_010",
                category="tool_calling",
                task="如何分析MgB2中B-B键的共价性？",
                rubric_items=[
                    {"criterion": "mentions ELF", "weight": 3, "keywords": ["elf"]},
                    {
                        "criterion": "mentions electron localization function",
                        "weight": 2,
                        "keywords": ["electron localization function"],
                    },
                ],
                evaluator=rubric_evaluator,
                tags=["elf", "bonding", "mgb2"],
            )
        )

    def multi_step_reasoning_cases(self):
        """Multi-step reasoning benchmarks: 3+ step quantitative problems.

        10 cases that require chaining calculation steps (density, packing factor,
        Bragg angle, Hall-Petch, conductivity, Gibbs free energy, etc.). Each
        rubric item maps to one reasoning step; expected_value records the target
        numeric answer as metadata.
        """
        return (
            self._msr_add_density()
            ._msr_add_structure_property()
            ._msr_add_thermo_electronic()
            ._msr_add_property_trends()
        )

    def _msr_add_density(self):
        """Density from crystal structure: Si/Cu/APF (msr_001-003)."""
        BenchmarkSuite._msr_add_density_density(self)
        BenchmarkSuite._msr_add_density_density_2(self)
        BenchmarkSuite._msr_add_density_packing_factor(self)
        return self

    def _msr_add_density_density(self) -> None:
        """Register msr_001 (density)."""
        # Si diamond cubic density (~2.33 g/cm3)
        self.add(
            BenchmarkCase(
                case_id="msr_001",
                category="multi_step",
                task="硅采用金刚石立方结构，晶格常数a=5.43 Å，原子量28.09，计算其理论密度（g/cm³）。要求写出计算步骤。",
                expected_value=2.33,
                tolerance=0.05,
                rubric_items=[
                    {
                        "criterion": "确定晶胞原子数(8)",
                        "weight": 1,
                        "keywords": ["8", "atoms"],
                    },
                    {"criterion": "计算晶胞质量", "weight": 1, "keywords": ["mass"]},
                    {
                        "criterion": "除以体积得到密度",
                        "weight": 1,
                        "keywords": ["density"],
                    },
                ],
                evaluator=rubric_evaluator,
                tags=["density", "silicon", "diamond-cubic"],
            )
        )

    def _msr_add_density_density_2(self) -> None:
        """Register msr_002 (density)."""
        # Cu FCC density (~8.96 g/cm3)
        self.add(
            BenchmarkCase(
                case_id="msr_002",
                category="multi_step",
                task="铜为FCC结构，晶格常数a=3.61 Å，原子量63.55，计算理论密度（g/cm³）。要求写出计算步骤。",
                expected_value=8.96,
                tolerance=0.1,
                rubric_items=[
                    {
                        "criterion": "确定FCC晶胞原子数(4)",
                        "weight": 1,
                        "keywords": ["4", "atoms"],
                    },
                    {"criterion": "计算晶胞质量", "weight": 1, "keywords": ["mass"]},
                    {
                        "criterion": "除以体积得到密度",
                        "weight": 1,
                        "keywords": ["density"],
                    },
                ],
                evaluator=rubric_evaluator,
                tags=["density", "copper", "fcc"],
            )
        )

    def _msr_add_density_packing_factor(self) -> None:
        """Register msr_003 (packing-factor)."""
        # atomic packing factor
        self.add(
            BenchmarkCase(
                case_id="msr_003",
                category="multi_step",
                task="FCC金属原子半径r=1.28 Å，晶格常数a=3.61 Å，计算原子堆积因子。要求写出计算步骤。",
                expected_value=0.74,
                tolerance=0.02,
                rubric_items=[
                    {
                        "criterion": "计算单个原子体积",
                        "weight": 1,
                        "keywords": ["volume", "atom"],
                    },
                    {
                        "criterion": "计算晶胞体积",
                        "weight": 1,
                        "keywords": ["volume", "cell"],
                    },
                    {"criterion": "计算堆积因子", "weight": 1, "keywords": ["packing"]},
                ],
                evaluator=rubric_evaluator,
                tags=["packing-factor", "fcc"],
            )
        )

    def _msr_add_structure_property(self):
        """Structure-derived properties: Bragg/Hall-Petch/conductivity (msr_004-006)."""
        BenchmarkSuite._msr_add_structure_property_bragg(self)
        BenchmarkSuite._msr_add_structure_property_hall_petch(self)
        BenchmarkSuite._msr_add_structure_property_conductivity(self)
        return self

    def _msr_add_structure_property_bragg(self) -> None:
        """Register msr_004 (bragg)."""
        # Bragg diffraction angle
        self.add(
            BenchmarkCase(
                case_id="msr_004",
                category="multi_step",
                task="Cu Kα辐射(λ=1.54 Å)照射FCC铝(a=4.05 Å)，计算(111)面的Bragg衍射角2θ（度）。要求写出计算步骤。",
                expected_value=38.5,
                tolerance=1.0,
                rubric_items=[
                    {
                        "criterion": "计算d间距",
                        "weight": 1,
                        "keywords": ["d-spacing", "d spacing", "interplanar"],
                    },
                    {"criterion": "应用Bragg方程", "weight": 1, "keywords": ["bragg"]},
                    {
                        "criterion": "得到衍射角2θ",
                        "weight": 1,
                        "keywords": ["2θ", "2theta", "diffraction angle"],
                    },
                ],
                evaluator=rubric_evaluator,
                tags=["bragg", "xrd", "aluminum"],
            )
        )

    def _msr_add_structure_property_hall_petch(self) -> None:
        """Register msr_005 (hall-petch)."""
        # Hall-Petch yield strength
        self.add(
            BenchmarkCase(
                case_id="msr_005",
                category="multi_step",
                task="某低碳钢屈服强度σ₀=150 MPa，Hall-Petch系数k=0.5 MPa·√m，晶粒尺寸d=10 μm，计算屈服强度（MPa）。要求写出计算步骤。",
                expected_value=308,
                tolerance=15,
                rubric_items=[
                    {
                        "criterion": "识别Hall-Petch关系",
                        "weight": 1,
                        "keywords": ["hall-petch", "hall petch"],
                    },
                    {
                        "criterion": "代入晶粒尺寸计算",
                        "weight": 1,
                        "keywords": ["grain size"],
                    },
                    {"criterion": "得到屈服强度", "weight": 1, "keywords": ["yield"]},
                ],
                evaluator=rubric_evaluator,
                tags=["hall-petch", "yield-strength", "steel"],
            )
        )

    def _msr_add_structure_property_conductivity(self) -> None:
        """Register msr_006 (conductivity)."""
        # conductivity from carrier concentration
        self.add(
            BenchmarkCase(
                case_id="msr_006",
                category="multi_step",
                task="某半导体载流子浓度n=1e16 cm⁻³，迁移率μ=1000 cm²/V·s，基本电荷e=1.6e-19 C，计算电导率σ=neμ（S/cm）。要求写出计算步骤。",
                expected_value=1.6,
                tolerance=0.1,
                rubric_items=[
                    {
                        "criterion": "识别载流子浓度",
                        "weight": 1,
                        "keywords": ["concentration"],
                    },
                    {"criterion": "使用迁移率", "weight": 1, "keywords": ["mobility"]},
                    {
                        "criterion": "计算电导率",
                        "weight": 1,
                        "keywords": ["conductivity"],
                    },
                ],
                evaluator=rubric_evaluator,
                tags=["conductivity", "semiconductor"],
            )
        )

    def _msr_add_thermo_electronic(self):
        """Thermodynamics + doping: Gibbs/carrier concentration (msr_007-008)."""
        BenchmarkSuite._msr_add_thermo_electronic_gibbs(self)
        BenchmarkSuite._msr_add_thermo_electronic_doping(self)
        return self

    def _msr_add_thermo_electronic_gibbs(self) -> None:
        """Register msr_007 (gibbs)."""
        # Gibbs free energy and phase stability
        self.add(
            BenchmarkCase(
                case_id="msr_007",
                category="multi_step",
                task="某相变反应ΔH=5 kJ/mol，ΔS=10 J/(mol·K)，计算临界温度Tc=ΔH/ΔS（K），并判断300K时哪相更稳定。要求写出推理步骤。",
                expected_value=500,
                tolerance=20,
                rubric_items=[
                    {
                        "criterion": "计算临界温度Tc",
                        "weight": 1,
                        "keywords": ["critical temperature", "tc"],
                    },
                    {
                        "criterion": "计算ΔG判断方向",
                        "weight": 1,
                        "keywords": ["gibbs", "free energy"],
                    },
                    {
                        "criterion": "判断相稳定性",
                        "weight": 1,
                        "keywords": ["stable", "stability"],
                    },
                ],
                evaluator=rubric_evaluator,
                tags=["gibbs", "phase-stability", "thermodynamics"],
            )
        )

    def _msr_add_thermo_electronic_doping(self) -> None:
        """Register msr_008 (doping)."""
        # carrier concentration from doping
        self.add(
            BenchmarkCase(
                case_id="msr_008",
                category="multi_step",
                task="硅中掺磷，掺杂浓度Nd=1e17 cm⁻³，本征载流子浓度ni=1.5e10 cm⁻³，计算室温下多数载流子浓度（cm⁻³）。要求写出推理步骤。",
                expected_value=1e17,
                tolerance=1e16,
                rubric_items=[
                    {
                        "criterion": "识别多数载流子类型",
                        "weight": 1,
                        "keywords": ["electron", "majority"],
                    },
                    {
                        "criterion": "使用掺杂浓度",
                        "weight": 1,
                        "keywords": ["doping", "donor"],
                    },
                    {
                        "criterion": "计算载流子浓度",
                        "weight": 1,
                        "keywords": ["carrier concentration"],
                    },
                ],
                evaluator=rubric_evaluator,
                tags=["doping", "carrier-concentration", "silicon"],
            )
        )

    def _msr_add_property_trends(self):
        """Property trends: thermal expansion/Curie-Weiss (msr_009-010)."""
        BenchmarkSuite._msr_add_property_trends_thermal_expansion(self)
        BenchmarkSuite._msr_add_property_trends_curie(self)
        return self

    def _msr_add_property_trends_thermal_expansion(self) -> None:
        """Register msr_009 (thermal-expansion)."""
        # thermal expansion lattice parameter
        self.add(
            BenchmarkCase(
                case_id="msr_009",
                category="multi_step",
                task="铝的晶格常数a₀=4.05 Å，线膨胀系数α=2.3e-5 K⁻¹，计算500K时的晶格常数（Å）。要求写出计算步骤。",
                expected_value=4.069,
                tolerance=0.01,
                rubric_items=[
                    {
                        "criterion": "使用热膨胀系数",
                        "weight": 1,
                        "keywords": ["expansion", "thermal"],
                    },
                    {
                        "criterion": "计算温度变化ΔT",
                        "weight": 1,
                        "keywords": ["temperature"],
                    },
                    {
                        "criterion": "计算新晶格常数",
                        "weight": 1,
                        "keywords": ["lattice"],
                    },
                ],
                evaluator=rubric_evaluator,
                tags=["thermal-expansion", "lattice-parameter", "aluminum"],
            )
        )

    def _msr_add_property_trends_curie(self) -> None:
        """Register msr_010 (curie)."""
        # Curie temperature from Curie-Weiss law
        self.add(
            BenchmarkCase(
                case_id="msr_010",
                category="multi_step",
                task="某铁磁材料遵循Curie-Weiss定律χ=C/(T-Tc)，测得T=400K时χ=0.01，T=600K时χ=0.005，计算Curie温度Tc（K）。要求写出推导步骤。",
                expected_value=200,
                tolerance=10,
                rubric_items=[
                    {
                        "criterion": "识别Curie-Weiss定律",
                        "weight": 1,
                        "keywords": ["curie-weiss", "curie weiss"],
                    },
                    {
                        "criterion": "建立方程组求解",
                        "weight": 1,
                        "keywords": ["equation", "solve"],
                    },
                    {
                        "criterion": "计算Curie温度",
                        "weight": 1,
                        "keywords": ["curie temperature"],
                    },
                ],
                evaluator=rubric_evaluator,
                tags=["curie", "curie-weiss", "magnetism"],
            )
        )

    def experiment_design_cases(self) -> BenchmarkSuite:
        """Experiment design benchmarks: plan a real characterization experiment.

        10 cases covering band gap measurement, thin film characterization,
        corrosion testing, fatigue life, thermal conductivity, electrochemistry,
        catalysis, polymer degradation, nanoindentation, and phase identification.
        Each rubric checks method selection, sample prep, parameters, and analysis.
        """
        # semiconductor band gap measurement
        self._ed_spectroscopy()
        self._ed_mechanical()
        self._ed_thermal_electrochem()
        self._ed_catalysis_polymer()
        self._ed_nano_xrd()
        return self

    def _ed_spectroscopy(self) -> None:
        """ed_001-002: optical band gap / ITO thin film characterization."""
        BenchmarkSuite._ed_spectroscopy_band_gap(self)
        BenchmarkSuite._ed_spectroscopy_thin_film(self)

    def _ed_spectroscopy_band_gap(self) -> None:
        """Register ed_001 (band-gap)."""
        # semiconductor band gap measurement
        self.add(
            BenchmarkCase(
                case_id="ed_001",
                category="experiment_design",
                task="设计一个测量ZnO薄膜光学带隙的实验方案，需包含方法选择、样品制备、测试参数和数据分析方法。",
                rubric_items=[
                    {
                        "criterion": "选择UV-Vis吸收光谱法",
                        "weight": 2,
                        "keywords": ["uv-vis", "uv vis", "absorption"],
                    },
                    {
                        "criterion": "使用Tauc plot分析",
                        "weight": 2,
                        "keywords": ["tauc"],
                    },
                    {
                        "criterion": "样品制备要求",
                        "weight": 1,
                        "keywords": ["substrate", "sample"],
                    },
                    {
                        "criterion": "数据分析方法",
                        "weight": 1,
                        "keywords": ["extrapolat", "linear"],
                    },
                ],
                evaluator=rubric_evaluator,
                tags=["band-gap", "uv-vis", "tauc-plot"],
            )
        )

    def _ed_spectroscopy_thin_film(self) -> None:
        """Register ed_002 (thin-film)."""
        # thin film characterization
        self.add(
            BenchmarkCase(
                case_id="ed_002",
                category="experiment_design",
                task="设计一个表征ITO透明导电薄膜的实验方案，需测定厚度、光学和电学性能。",
                rubric_items=[
                    {
                        "criterion": "厚度测量方法",
                        "weight": 1,
                        "keywords": ["profilometer", "ellipsometry", "step height"],
                    },
                    {
                        "criterion": "光学透过率测量",
                        "weight": 1,
                        "keywords": ["transmittance", "optical"],
                    },
                    {
                        "criterion": "电学性能测量",
                        "weight": 1,
                        "keywords": ["hall effect", "sheet resistance", "four-point"],
                    },
                    {
                        "criterion": "结构表征",
                        "weight": 1,
                        "keywords": ["xrd", "sem", "afm"],
                    },
                ],
                evaluator=rubric_evaluator,
                tags=["thin-film", "ito", "characterization"],
            )
        )

    def _ed_mechanical(self) -> None:
        """ed_003-004: corrosion rate / fatigue S-N life."""
        BenchmarkSuite._ed_mechanical_corrosion(self)
        BenchmarkSuite._ed_mechanical_fatigue(self)

    def _ed_mechanical_corrosion(self) -> None:
        """Register ed_003 (corrosion)."""
        # corrosion rate testing
        self.add(
            BenchmarkCase(
                case_id="ed_003",
                category="experiment_design",
                task="设计一个碳钢在3.5% NaCl溶液中腐蚀速率的测试实验方案。",
                rubric_items=[
                    {
                        "criterion": "选择测试方法",
                        "weight": 2,
                        "keywords": ["potentiodynamic", "polarization", "weight loss"],
                    },
                    {
                        "criterion": "电解池配置",
                        "weight": 1,
                        "keywords": [
                            "electrolyte",
                            "reference electrode",
                            "counter electrode",
                        ],
                    },
                    {
                        "criterion": "温度/时间控制",
                        "weight": 1,
                        "keywords": ["temperature", "immersion"],
                    },
                    {
                        "criterion": "腐蚀速率计算",
                        "weight": 1,
                        "keywords": ["corrosion rate", "mm/year", "mpy"],
                    },
                ],
                evaluator=rubric_evaluator,
                tags=["corrosion", "nacl", "polarization"],
            )
        )

    def _ed_mechanical_fatigue(self) -> None:
        """Register ed_004 (fatigue)."""
        # fatigue life testing
        self.add(
            BenchmarkCase(
                case_id="ed_004",
                category="experiment_design",
                task="设计一个铝合金疲劳寿命(S-N曲线)测试实验方案。",
                rubric_items=[
                    {
                        "criterion": "选择疲劳试验机类型",
                        "weight": 2,
                        "keywords": [
                            "rotating bending",
                            "servohydraulic",
                            "fatigue test",
                        ],
                    },
                    {
                        "criterion": "应力水平设置",
                        "weight": 1,
                        "keywords": ["stress amplitude", "stress level"],
                    },
                    {
                        "criterion": "样品制备标准",
                        "weight": 1,
                        "keywords": ["surface finish", "gauge", "standard"],
                    },
                    {
                        "criterion": "S-N曲线分析",
                        "weight": 1,
                        "keywords": ["s-n", "wohler", "fatigue limit"],
                    },
                ],
                evaluator=rubric_evaluator,
                tags=["fatigue", "s-n-curve", "aluminum"],
            )
        )

    def _ed_thermal_electrochem(self) -> None:
        """ed_005-006: thermal conductivity / LiCoO2 electrochemistry."""
        BenchmarkSuite._ed_thermal_electrochem_thermal_conductivity(self)
        BenchmarkSuite._ed_thermal_electrochem_battery(self)

    def _ed_thermal_electrochem_thermal_conductivity(self) -> None:
        """Register ed_005 (thermal-conductivity)."""
        # thermal conductivity measurement
        self.add(
            BenchmarkCase(
                case_id="ed_005",
                category="experiment_design",
                task="设计一个测量陶瓷材料热导率的实验方案。",
                rubric_items=[
                    {
                        "criterion": "选择测量方法",
                        "weight": 2,
                        "keywords": ["laser flash", "hot disk", "guarded hot plate"],
                    },
                    {
                        "criterion": "样品尺寸要求",
                        "weight": 1,
                        "keywords": ["thickness", "diameter", "sample size"],
                    },
                    {
                        "criterion": "温度控制",
                        "weight": 1,
                        "keywords": ["temperature", "furnace"],
                    },
                    {
                        "criterion": "数据处理",
                        "weight": 1,
                        "keywords": ["thermal diffusivity", "specific heat", "density"],
                    },
                ],
                evaluator=rubric_evaluator,
                tags=["thermal-conductivity", "laser-flash", "ceramic"],
            )
        )

    def _ed_thermal_electrochem_battery(self) -> None:
        """Register ed_006 (battery)."""
        # battery electrochemical characterization
        self.add(
            BenchmarkCase(
                case_id="ed_006",
                category="experiment_design",
                task="设计一个LiCoO2正极材料的电化学表征实验方案。",
                rubric_items=[
                    {
                        "criterion": "组装半电池",
                        "weight": 2,
                        "keywords": ["coin cell", "half cell", "half-cell"],
                    },
                    {
                        "criterion": "CV测试",
                        "weight": 1,
                        "keywords": ["cyclic voltammetry"],
                    },
                    {
                        "criterion": "恒流充放电",
                        "weight": 1,
                        "keywords": ["galvanostatic", "charge-discharge", "gitt"],
                    },
                    {
                        "criterion": "EIS测试",
                        "weight": 1,
                        "keywords": ["eis", "impedance"],
                    },
                ],
                evaluator=rubric_evaluator,
                tags=["battery", "electrochemistry", "licoo2"],
            )
        )

    def _ed_catalysis_polymer(self) -> None:
        """ed_007-008: ORR catalyst screening / PLA degradation."""
        BenchmarkSuite._ed_catalysis_polymer_catalysis(self)
        BenchmarkSuite._ed_catalysis_polymer_polymer(self)

    def _ed_catalysis_polymer_catalysis(self) -> None:
        """Register ed_007 (catalysis)."""
        # catalyst activity screening
        self.add(
            BenchmarkCase(
                case_id="ed_007",
                category="experiment_design",
                task="设计一个Pt/C催化剂氧还原反应(ORR)活性筛选实验方案。",
                rubric_items=[
                    {
                        "criterion": "选择测试方法(RDE/RRDE)",
                        "weight": 2,
                        "keywords": ["rde", "rrde", "rotating disk"],
                    },
                    {
                        "criterion": "电解液配置",
                        "weight": 1,
                        "keywords": ["electrolyte", "koh", "0.1 m"],
                    },
                    {
                        "criterion": "参比电极校准",
                        "weight": 1,
                        "keywords": ["reference electrode", "rhe", "calibration"],
                    },
                    {
                        "criterion": "活性指标分析",
                        "weight": 1,
                        "keywords": ["onset potential", "half-wave", "mass activity"],
                    },
                ],
                evaluator=rubric_evaluator,
                tags=["catalysis", "orr", "rde"],
            )
        )

    def _ed_catalysis_polymer_polymer(self) -> None:
        """Register ed_008 (polymer)."""
        # polymer degradation
        self.add(
            BenchmarkCase(
                case_id="ed_008",
                category="experiment_design",
                task="设计一个评估PLA聚合物在土壤中降解速率的实验方案。",
                rubric_items=[
                    {
                        "criterion": "降解环境设置",
                        "weight": 2,
                        "keywords": ["soil", "compost", "burial"],
                    },
                    {
                        "criterion": "时间梯度设计",
                        "weight": 1,
                        "keywords": ["time point", "sampling", "interval"],
                    },
                    {
                        "criterion": "质量损失监测",
                        "weight": 1,
                        "keywords": ["weight loss", "mass loss"],
                    },
                    {
                        "criterion": "降解表征方法",
                        "weight": 1,
                        "keywords": ["gpc", "sem", "ftir"],
                    },
                ],
                evaluator=rubric_evaluator,
                tags=["polymer", "degradation", "pla"],
            )
        )

    def _ed_nano_xrd(self) -> None:
        """ed_009-010: nanoindentation / XRD phase identification."""
        BenchmarkSuite._ed_nano_xrd_nanoindentation(self)
        BenchmarkSuite._ed_nano_xrd_xrd(self)

    def _ed_nano_xrd_nanoindentation(self) -> None:
        """Register ed_009 (nanoindentation)."""
        # nanoindentation
        self.add(
            BenchmarkCase(
                case_id="ed_009",
                category="experiment_design",
                task="设计一个用纳米压痕测量薄膜硬度和弹性模量的实验方案。",
                rubric_items=[
                    {
                        "criterion": "选择压头类型",
                        "weight": 2,
                        "keywords": ["berkovich", "indentor", "tip"],
                    },
                    {
                        "criterion": "载荷-位移曲线",
                        "weight": 1,
                        "keywords": ["load-displacement", "p-h"],
                    },
                    {
                        "criterion": "Oliver-Pharr方法",
                        "weight": 2,
                        "keywords": ["oliver-pharr", "oliver pharr"],
                    },
                    {
                        "criterion": "校准与基线",
                        "weight": 1,
                        "keywords": ["calibration", "fused silica"],
                    },
                ],
                evaluator=rubric_evaluator,
                tags=["nanoindentation", "hardness", "oliver-pharr"],
            )
        )

    def _ed_nano_xrd_xrd(self) -> None:
        """Register ed_010 (xrd)."""
        # phase identification
        self.add(
            BenchmarkCase(
                case_id="ed_010",
                category="experiment_design",
                task="设计一个用XRD鉴定未知粉末样品物相的实验方案。",
                rubric_items=[
                    {
                        "criterion": "XRD测试参数设置",
                        "weight": 1,
                        "keywords": ["scan rate", "step size", "2θ"],
                    },
                    {
                        "criterion": "使用PDF数据库",
                        "weight": 2,
                        "keywords": ["pdf", "icdd", "database"],
                    },
                    {
                        "criterion": "衍射峰匹配",
                        "weight": 1,
                        "keywords": ["peak", "match", "indexing"],
                    },
                    {
                        "criterion": "定量分析",
                        "weight": 1,
                        "keywords": ["rietveld", "quantitative"],
                    },
                ],
                evaluator=rubric_evaluator,
                tags=["xrd", "phase-id", "pdf-database"],
            )
        )

    def reverse_reasoning_cases(self) -> BenchmarkSuite:
        """Reverse-reasoning benchmarks: given an effect, infer the cause.

        10 cases covering XRD structure ID, density-based material ID,
        phonon scattering mechanism, deformation mode, semiconductor ID,
        Curie temperature material ID, heat-treatment strengthening,
        corrosion type, eutectoid composition, and fracture mode.
        Each rubric checks identification + physical reasoning.
        """
        # XRD d-spacing ratio → FCC vs BCC
        self._rr_structure_id()
        self._rr_transport_deformation()
        self._rr_semiconductor_magnetic()
        self._rr_strengthening_corrosion()
        self._rr_phase_fracture()
        return self

    def _rr_structure_id(self) -> None:
        """rr_001-002: XRD ratio → FCC/BCC / density → iron."""
        # XRD d-spacing ratio → FCC vs BCC
        self.add(
            BenchmarkCase(
                case_id="rr_001",
                category="reverse_reasoning",
                task="某立方晶系晶体XRD前三条衍射峰的d-spacing比值为1:0.577:0.516。"
                "请根据该比值序列判断是FCC还是BCC结构，并说明判断依据。",
                rubric_items=[
                    {"criterion": "识别立方晶系", "weight": 1, "keywords": ["cubic"]},
                    {
                        "criterion": "判断为FCC",
                        "weight": 2,
                        "keywords": ["fcc", "face-centered"],
                    },
                    {
                        "criterion": "说明d-spacing比值依据",
                        "weight": 2,
                        "keywords": ["1/sqrt(3)", "ratio", "sequence"],
                    },
                ],
                evaluator=rubric_evaluator,
                tags=["xrd", "crystal-structure", "fcc"],
            )
        )
        # density + magnetism → material ID
        self.add(
            BenchmarkCase(
                case_id="rr_002",
                category="reverse_reasoning",
                task="某金属材料室温密度为7.87 g/cm³，具有铁磁性。请判断该材料是什么，"
                "并说明其晶体结构和磁性来源。",
                rubric_items=[
                    {"criterion": "识别为铁", "weight": 3, "keywords": ["iron", "fe"]},
                    {
                        "criterion": "提到BCC结构",
                        "weight": 2,
                        "keywords": ["bcc", "body-centered"],
                    },
                    {
                        "criterion": "提到铁磁性",
                        "weight": 1,
                        "keywords": ["ferromagnetic"],
                    },
                ],
                evaluator=rubric_evaluator,
                tags=["density", "material-id", "iron"],
            )
        )

    def _rr_transport_deformation(self) -> None:
        """rr_003-004: phonon scattering / deformation mode."""
        BenchmarkSuite._rr_transport_deformation_thermal_conductivity(self)
        BenchmarkSuite._rr_transport_deformation_stress_strain(self)

    def _rr_transport_deformation_thermal_conductivity(self) -> None:
        """Register rr_003 (thermal-conductivity)."""
        # thermal conductivity trend → scattering mechanism
        self.add(
            BenchmarkCase(
                case_id="rr_003",
                category="reverse_reasoning",
                task="某绝缘晶体在高温区(>Debye温度)热导率随温度升高按1/T趋势下降。"
                "请判断主导的声子散射机制，并解释其物理起源。",
                rubric_items=[
                    {
                        "criterion": "识别Umklapp散射",
                        "weight": 3,
                        "keywords": ["umklapp"],
                    },
                    {
                        "criterion": "解释高温1/T依赖",
                        "weight": 2,
                        "keywords": ["1/t", "phonon", "scattering"],
                    },
                ],
                evaluator=rubric_evaluator,
                tags=["thermal-conductivity", "phonon", "umklapp"],
            )
        )

    def _rr_transport_deformation_stress_strain(self) -> None:
        """Register rr_004 (stress-strain)."""
        # stress-strain curve → deformation mechanism
        self.add(
            BenchmarkCase(
                case_id="rr_004",
                category="reverse_reasoning",
                task="某金属拉伸试验中无明显屈服点，均匀塑性变形量超过40%，加工硬化率较高。"
                "请判断该金属可能的晶体结构类型及变形机制。",
                rubric_items=[
                    {
                        "criterion": "判断为FCC金属(如Cu/Al)",
                        "weight": 2,
                        "keywords": ["fcc", "copper", "aluminum"],
                    },
                    {
                        "criterion": "提到应变硬化",
                        "weight": 2,
                        "keywords": ["strain", "hardening", "work"],
                    },
                    {
                        "criterion": "解释多滑移系",
                        "weight": 1,
                        "keywords": ["slip", "system"],
                    },
                ],
                evaluator=rubric_evaluator,
                tags=["stress-strain", "deformation", "fcc"],
            )
        )

    def _rr_semiconductor_magnetic(self) -> None:
        """rr_005-006: band gap → silicon / Curie → iron."""
        # band gap → semiconductor ID
        self.add(
            BenchmarkCase(
                case_id="rr_005",
                category="reverse_reasoning",
                task="某半导体材料室温带隙为1.12 eV，为间接带隙，外观为深灰色晶体。"
                "请判断该材料是什么，并说明其晶体结构特征。",
                rubric_items=[
                    {
                        "criterion": "识别为硅",
                        "weight": 3,
                        "keywords": ["silicon", "si"],
                    },
                    {
                        "criterion": "提到间接带隙",
                        "weight": 2,
                        "keywords": ["indirect"],
                    },
                    {
                        "criterion": "提到金刚石立方结构",
                        "weight": 1,
                        "keywords": ["diamond", "cubic"],
                    },
                ],
                evaluator=rubric_evaluator,
                tags=["band-gap", "semiconductor", "silicon"],
            )
        )
        # Curie temperature → magnetic material ID
        self.add(
            BenchmarkCase(
                case_id="rr_006",
                category="reverse_reasoning",
                task="某铁磁材料Curie温度为770°C，室温饱和磁化强度约1.7×10⁶ A/m。"
                "请判断该材料是什么，并描述其磁性转变过程。",
                rubric_items=[
                    {"criterion": "识别为铁", "weight": 3, "keywords": ["iron", "fe"]},
                    {
                        "criterion": "提到铁磁-顺磁转变",
                        "weight": 2,
                        "keywords": ["ferromagnetic", "paramagnetic"],
                    },
                    {"criterion": "提到BCC结构", "weight": 1, "keywords": ["bcc"]},
                ],
                evaluator=rubric_evaluator,
                tags=["curie-temperature", "magnetism", "iron"],
            )
        )

    def _rr_strengthening_corrosion(self) -> None:
        """rr_007-008: heat-treatment strengthening / pitting."""
        BenchmarkSuite._rr_strengthening_corrosion_heat_treatment(self)
        BenchmarkSuite._rr_strengthening_corrosion_corrosion(self)

    def _rr_strengthening_corrosion_heat_treatment(self) -> None:
        """Register rr_007 (heat-treatment)."""
        # hardness change → strengthening mechanism
        self.add(
            BenchmarkCase(
                case_id="rr_007",
                category="reverse_reasoning",
                task="某低碳钢淬火后硬度从120HV升至650HV，随后200°C回火降至550HV。"
                "请分析各阶段硬度变化的强化机制。",
                rubric_items=[
                    {
                        "criterion": "识别马氏体强化",
                        "weight": 3,
                        "keywords": ["martensite"],
                    },
                    {
                        "criterion": "解释淬火产生马氏体",
                        "weight": 2,
                        "keywords": ["quench", "rapid cooling"],
                    },
                    {
                        "criterion": "解释回火析出碳化物",
                        "weight": 2,
                        "keywords": ["temper", "carbide"],
                    },
                ],
                evaluator=rubric_evaluator,
                tags=["heat-treatment", "martensite", "strengthening"],
            )
        )

    def _rr_strengthening_corrosion_corrosion(self) -> None:
        """Register rr_008 (corrosion)."""
        # corrosion morphology → corrosion type
        self.add(
            BenchmarkCase(
                case_id="rr_008",
                category="reverse_reasoning",
                task="某不锈钢在含Cl⁻环境中服役后，表面出现孤立深坑，其余区域基本完好。"
                "请判断腐蚀类型，并分析Cl⁻在其中的作用。",
                rubric_items=[
                    {
                        "criterion": "识别为点蚀",
                        "weight": 3,
                        "keywords": ["pitting", "pit"],
                    },
                    {
                        "criterion": "解释Cl⁻的作用",
                        "weight": 2,
                        "keywords": ["chloride", "cl"],
                    },
                    {
                        "criterion": "提到钝化膜破坏",
                        "weight": 2,
                        "keywords": ["passive", "film", "breakdown"],
                    },
                ],
                evaluator=rubric_evaluator,
                tags=["corrosion", "pitting", "chloride"],
            )
        )

    def _rr_phase_fracture(self) -> None:
        """rr_009-010: eutectoid composition / ductile fracture."""
        BenchmarkSuite._rr_phase_fracture_phase_diagram(self)
        BenchmarkSuite._rr_phase_fracture_fracture(self)

    def _rr_phase_fracture_phase_diagram(self) -> None:
        """Register rr_009 (phase-diagram)."""
        # phase diagram microstructure → composition
        self.add(
            BenchmarkCase(
                case_id="rr_009",
                category="reverse_reasoning",
                task="某Fe-C合金在727°C发生共析转变，室温显微组织为珠光体(铁素体+渗碳体层片)。"
                "请判断该合金的碳含量，并说明共析转变过程。",
                rubric_items=[
                    {
                        "criterion": "识别共析成分0.77wt%C",
                        "weight": 3,
                        "keywords": ["0.77", "eutectoid"],
                    },
                    {"criterion": "提到珠光体", "weight": 2, "keywords": ["pearlite"]},
                    {
                        "criterion": "提到铁素体+渗碳体",
                        "weight": 2,
                        "keywords": ["ferrite", "cementite"],
                    },
                ],
                evaluator=rubric_evaluator,
                tags=["phase-diagram", "eutectoid", "fe-c"],
            )
        )

    def _rr_phase_fracture_fracture(self) -> None:
        """Register rr_010 (fracture)."""
        # fracture surface → failure mode
        self.add(
            BenchmarkCase(
                case_id="rr_010",
                category="reverse_reasoning",
                task="某金属构件断口呈杯锥状，中心区域有大量韧窝，边缘呈45°剪切唇。"
                "请判断失效模式，并解释断口形貌的形成过程。",
                rubric_items=[
                    {
                        "criterion": "识别为延性断裂",
                        "weight": 3,
                        "keywords": ["ductile", "延性"],
                    },
                    {
                        "criterion": "提到韧窝",
                        "weight": 2,
                        "keywords": ["dimple", "韧窝"],
                    },
                    {
                        "criterion": "解释杯锥状断口",
                        "weight": 2,
                        "keywords": ["cup", "cone", "shear"],
                    },
                ],
                evaluator=rubric_evaluator,
                tags=["fracture", "ductile", "cup-cone"],
            )
        )

    async def run(
        self,
        agent: Any,
        thread_id: str = "benchmark",
    ) -> list[BenchmarkResult]:
        """Run all cases against ``agent`` and return scored results."""
        results: list[BenchmarkResult] = []
        for case in self.cases:
            start = time.time()
            response = ""
            error: str | None = None
            try:
                response = await self._invoke_agent(agent, case.task, thread_id)
            except Exception as exc:
                error = str(exc)
            duration_ms = round((time.time() - start) * 1000, 2)

            if error:
                results.append(
                    BenchmarkResult(
                        case_id=case.case_id,
                        task=case.task,
                        success=False,
                        score=0.0,
                        response=response,
                        duration_ms=duration_ms,
                        error=error,
                    )
                )
                continue

            success, score = case.evaluator(response, case)
            results.append(
                BenchmarkResult(
                    case_id=case.case_id,
                    task=case.task,
                    success=success,
                    score=score,
                    response=response,
                    duration_ms=duration_ms,
                )
            )
        return results

    async def run_multi_trial(
        self,
        agent: Any,
        trials: int = 3,
        thread_id: str = "benchmark",
        checkpoint_path: str | None = None,
    ) -> MultiTrialResult:
        """Run each case ``trials`` times and compute pass^3 / pass@3 / FinalScore.

        If *checkpoint_path* is set, completed trials are persisted after each
        round so a crash or Ctrl-C can resume without redoing finished work.
        """
        from pathlib import Path

        trial_runs: list[list[BenchmarkResult]] = []

        # --- resume from checkpoint ---
        if checkpoint_path and Path(checkpoint_path).exists():
            saved = json.loads(Path(checkpoint_path).read_text(encoding="utf-8"))
            for run_data in saved.get("trial_runs", []):
                trial_runs.append([BenchmarkResult(**r) for r in run_data])

        start_trial = len(trial_runs)
        for t in range(start_trial, trials):
            run_results = await self.run(agent, thread_id=f"{thread_id}_t{t}")
            trial_runs.append(run_results)
            if checkpoint_path:
                Path(checkpoint_path).write_text(
                    json.dumps(
                        {
                            "trial_runs": [
                                [asdict(r) for r in run] for run in trial_runs
                            ]
                        },
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )

        return self._compile_multi_trial(trial_runs[:trials], trials)

    def _compile_multi_trial(
        self,
        trial_runs: list[list[BenchmarkResult]],
        trials: int,
    ) -> MultiTrialResult:
        """Group trial runs by case and compute ClawBench metrics."""
        (
            case_results,
            all_scores,
            all_latencies,
            total_cost,
            covered_cats,
        ) = BenchmarkSuite._aggregate_case_trials(self, trial_runs)

        n = len(self.cases) or 1
        pass_all_rate = sum(1 for cr in case_results if cr.pass_all) / n
        pass_any_rate = sum(1 for cr in case_results if cr.pass_any) / n

        S = sum(all_scores) / len(all_scores) if all_scores else 0.0
        r_all = pass_all_rate ** (1.0 / 3.0)
        r_any = 1.0 - (1.0 - pass_any_rate) ** (1.0 / 3.0)
        final_score = round(100.0 * (S**0.40) * (r_all**0.45) * (r_any**0.15), 2)

        all_cats = {c.category for c in self.cases}
        coverage = len(covered_cats) / len(all_cats) if all_cats else 0.0

        return MultiTrialResult(
            case_results=case_results,
            trials=trials,
            avg_score=round(S, 4),
            pass_all_rate=round(pass_all_rate, 4),
            pass_any_rate=round(pass_any_rate, 4),
            final_score=final_score,
            total_cost=round(total_cost, 4),
            avg_latency_ms=round(sum(all_latencies) / len(all_latencies), 2)
            if all_latencies
            else 0.0,
            coverage=round(coverage, 4),
        )

    def _aggregate_case_trials(
        self,
        trial_runs: list[list[BenchmarkResult]],
    ) -> tuple[list[CaseTrialResult], list[float], list[float], float, set[str]]:
        """Group trial runs by case and accumulate scores, latency, cost, coverage."""
        all_scores: list[float] = []
        all_latencies: list[float] = []
        total_cost = 0.0
        covered_cats: set[str] = set()

        case_results: list[CaseTrialResult] = []
        for i, case in enumerate(self.cases):
            trial_results = [run[i] for run in trial_runs]
            successes = [r.success for r in trial_results]
            scores = [r.score for r in trial_results]

            # normalize rubric scores (0-100) to 0-1 for the FinalScore formula
            norm = [s / 100.0 if s > 1.0 else s for s in scores]
            all_scores.extend(norm)
            all_latencies.extend(r.duration_ms for r in trial_results)
            total_cost += sum(getattr(r, "cost", 0.0) for r in trial_results)

            pass_all = all(successes)
            pass_any = any(successes)
            if pass_any:
                covered_cats.add(case.category)

            case_results.append(
                CaseTrialResult(
                    case_id=case.case_id,
                    task=case.task,
                    category=case.category,
                    trials=trial_results,
                    pass_all=pass_all,
                    pass_any=pass_any,
                    avg_score=round(sum(scores) / len(scores), 3) if scores else 0.0,
                    max_score=max(scores) if scores else 0.0,
                )
            )

        return case_results, all_scores, all_latencies, total_cost, covered_cats

    @staticmethod
    async def _invoke_agent(agent: Any, task: str, thread_id: str) -> str:
        """Collect the final response from an async agent."""
        final_response = ""
        async for state in agent.chat(task, thread_id=thread_id):
            messages = state.get("messages", [])
            for msg in messages:
                content = getattr(msg, "content", None)
                if content:
                    final_response = str(content)
        return final_response

    def summary(
        self,
        results: list[BenchmarkResult] | MultiTrialResult,
    ) -> dict[str, Any]:
        """Return aggregate statistics for a result set.

        Pass a :class:`MultiTrialResult` to get ClawBench FinalScore and
        multi-trial metrics (pass^3, pass@3, coverage, cost).
        """
        if isinstance(results, MultiTrialResult):
            return {
                "trials": results.trials,
                "total_cases": len(results.case_results),
                "avg_score": results.avg_score,
                "pass_all_rate": results.pass_all_rate,  # pass^3
                "pass_any_rate": results.pass_any_rate,  # pass@3
                "final_score": results.final_score,
                "total_cost": results.total_cost,
                "avg_latency_ms": results.avg_latency_ms,
                "coverage": results.coverage,
                "case_results": [
                    {
                        "case_id": cr.case_id,
                        "task": cr.task,
                        "category": cr.category,
                        "pass_all": cr.pass_all,
                        "pass_any": cr.pass_any,
                        "avg_score": cr.avg_score,
                        "max_score": cr.max_score,
                        "n_trials": len(cr.trials),
                    }
                    for cr in results.case_results
                ],
            }

        if not results:
            return {"total": 0, "passed": 0, "failed": 0, "avg_score": 0.0}
        passed = sum(1 for r in results if r.success)
        return {
            "total": len(results),
            "passed": passed,
            "failed": len(results) - passed,
            "avg_score": round(sum(r.score for r in results) / len(results), 3),
            "avg_duration_ms": round(
                sum(r.duration_ms for r in results) / len(results), 2
            ),
        }


class SelfImprovementLoop:
    """Run benchmarks and feed failures back into long-term memory."""

    def __init__(
        self,
        suite: BenchmarkSuite,
        memory_manager: MemoryManager,
    ) -> None:
        self.suite = suite
        self.memory = memory_manager

    async def evaluate(
        self,
        agent: Any,
        store_failures: bool = True,
    ) -> dict[str, Any]:
        """Run the suite and optionally memorize failures for future learning."""
        results = await self.suite.run(agent)
        summary = self.suite.summary(results)

        if store_failures:
            for r in results:
                if not r.success:
                    self.memory.remember(
                        content=(
                            f"Benchmark failure [{r.case_id}]: task='{r.task}' "
                            f"score={r.score} response='{r.response[:500]}'"
                        ),
                        category="benchmark_failure",
                        tags=["benchmark", r.task[:20]],
                        importance=0.7,
                        tier="mid",
                    )

        return {"summary": summary, "results": results}
