"""Lightweight benchmark framework for HuginnAgent.

A benchmark case is a task plus an evaluator. The suite runs each case
against an agent, scores the result, and the self-improvement loop stores
failures in long-term memory so the agent can learn over time.
"""

from __future__ import annotations

import json
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
    import re

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
    pass_all: bool   # pass^3: every trial passed
    pass_any: bool   # pass@3: at least one trial passed
    avg_score: float
    max_score: float


@dataclass
class MultiTrialResult:
    """Suite-level multi-trial result with ClawBench metrics."""

    case_results: list[CaseTrialResult]
    trials: int
    avg_score: float         # S: mean score across all trials, normalized 0-1
    pass_all_rate: float     # pass^3 fraction of cases
    pass_any_rate: float     # pass@3 fraction of cases
    final_score: float       # 100 * S^0.40 * r_all^0.45 * r_any^0.15
    total_cost: float
    avg_latency_ms: float
    coverage: float           # fraction of categories with >=1 pass


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
        self.add(BenchmarkCase(
            task="硅在室温下的晶体结构是什么？给出空间群和晶格常数。",
            expected_keywords=["diamond", "cubic", "fd-3m"],
            category="structure",
            tags=["crystal", "silicon"],
        ))
        # 2. band gap calculation
        self.add(BenchmarkCase(
            task="计算硅的带隙（eV），并说明是直接还是间接带隙。",
            expected_value=1.12,
            tolerance=0.05,
            evaluator=numeric_evaluator,
            category="electronic",
            tags=["bandgap", "silicon"],
        ))
        # 3. materials database retrieval
        self.add(BenchmarkCase(
            task="从Materials Project查询BaTiO3的基本信息：空间群、带隙、形成能。",
            expected_keywords=["batio3", "perovskite", "p4mm"],
            category="database",
            tags=["mp", "query"],
        ))
        # 4. symbolic computation
        self.add(BenchmarkCase(
            task="用SymPy计算 x^2 在区间 [0, 1] 上的定积分。",
            expected_value=0.333,
            tolerance=0.01,
            evaluator=numeric_evaluator,
            category="symbolic",
            tags=["sympy", "integral"],
        ))
        # 5. literature review
        self.add(BenchmarkCase(
            task="综述钙钛矿太阳能电池最近五年的效率进展，列出关键里程碑。",
            expected_keywords=["perovskite", "solar", "efficiency", "25"],
            category="literature",
            tags=["review", "photovoltaic"],
        ))
        # 6. multiscale modeling advice
        self.add(BenchmarkCase(
            task="为Li离子电池正极材料设计多尺度建模方案，从DFT到连续介质。",
            rubric_items=[
                {"criterion": "mentions DFT/ab initio", "weight": 2, "keywords": ["dft", "ab initio"]},
                {"criterion": "mentions molecular dynamics", "weight": 2, "keywords": ["molecular dynamics", "md "]},
                {"criterion": "mentions continuum/FEM", "weight": 2, "keywords": ["continuum", "finite element", "fem"]},
                {"criterion": "mentions scale bridging", "weight": 1, "keywords": ["bridge", "coupling", "handoff", "multiscale"]},
            ],
            evaluator=rubric_evaluator,
            category="multiscale",
            tags=["battery", "modeling"],
        ))
        # 7. phase diagram analysis
        self.add(BenchmarkCase(
            task="分析Fe-C二元相图中的共析反应，给出反应温度和产物。",
            expected_keywords=["eutectoid", "austenite", "pearlite", "727"],
            category="phase_diagram",
            tags=["fe-c", "metallurgy"],
        ))
        # 8. thermodynamic properties
        self.add(BenchmarkCase(
            task="计算水在298 K下的标准生成焓（kJ/mol）。",
            expected_value=-285.8,
            tolerance=2.0,
            evaluator=numeric_evaluator,
            category="thermodynamics",
            tags=["enthalpy", "water"],
        ))
        # 9. degradation mechanism analysis
        self.add(BenchmarkCase(
            task="分析PVDF聚合物在紫外光照射下的降解机制。",
            expected_keywords=["radical", "dehydrofluorination", "chain scission"],
            category="degradation",
            tags=["polymer", "uv"],
        ))
        # 10. research proposal design
        self.add(BenchmarkCase(
            task="设计一个高通量计算筛选固态电解质的流程，包含描述符和筛选标准。",
            rubric_items=[
                {"criterion": "mentions high-throughput workflow", "weight": 2, "keywords": ["high-throughput", "high throughput", "screening"]},
                {"criterion": "mentions ionic conductivity descriptor", "weight": 2, "keywords": ["ionic conductivity", "conductivity"]},
                {"criterion": "mentions stability criterion", "weight": 2, "keywords": ["stability", "electrochemical window"]},
                {"criterion": "mentions candidate material families", "weight": 1, "keywords": ["li7p3s11", "lgps", "argyrodite", "garnet", "llzo"]},
            ],
            evaluator=rubric_evaluator,
            category="research_design",
            tags=["high-throughput", "electrolyte"],
        ))
        # 11. elastic properties (bonus)
        self.add(BenchmarkCase(
            task="计算铜的体弹模量（GPa）。",
            expected_value=140.0,
            tolerance=15.0,
            evaluator=numeric_evaluator,
            category="mechanical",
            tags=["elastic", "copper"],
        ))
        return self

    def matscibench_cases(self) -> BenchmarkSuite:
        """MatSciBench-style cases — open-ended reasoning questions.

        Source: arXiv:2510.12171, KDD 2026.
        6 primary domains, 31 subfields, 3 difficulty levels.
        Covers: materials classification, properties (mechanical/thermal/
        electrical/magnetic/optical), structures (crystal/amorphous/defects),
        fundamental mechanisms (thermo/kinetics/diffusion/electronic/bonding),
        processes (casting/forming/heat treatment/sintering/deposition),
        failure mechanisms (fracture/fatigue/creep/corrosion/wear).
        """
        # ═══ Materials domain (5 cases) ═══
        self.add(BenchmarkCase(
            task="What are the three primary classes of engineering materials? "
             "Give one example of each.",
            expected_keywords=["metals", "ceramics", "polymers"],
            category="materials", tags=["classification", "easy"],
            case_id="matscibench_001",
        ))
        self.add(BenchmarkCase(
            task="Compare the bonding characteristics of metals, ceramics, and "
             "polymers. How does bonding explain the difference in ductility?",
            expected_keywords=["metallic", "covalent", "ionic", "ductile", "brittle"],
            category="materials", tags=["bonding", "medium"],
            case_id="matscibench_002",
        ))
        self.add(BenchmarkCase(
            task="What is a composite material? Give two examples and explain "
             "how the matrix and reinforcement interact.",
            expected_keywords=["matrix", "reinforcement", "fiber", "particle"],
            category="materials", tags=["composites", "easy"],
            case_id="matscibench_003",
        ))
        self.add(BenchmarkCase(
            task="Explain why semiconductor materials are doped. What is the "
             "difference between n-type and p-type doping? Give examples.",
            expected_keywords=["n-type", "p-type", "dopant", "phosphorus", "boron"],
            category="materials", tags=["semiconductors", "medium"],
            case_id="matscibench_004",
        ))
        self.add(BenchmarkCase(
            task="What makes a material biocompatible? List three requirements "
             "for a biomaterial used in bone implants.",
            expected_keywords=["biocompatible", "toxicity", "corrosion", "mechanical"],
            category="materials", tags=["biomaterials", "hard"],
            case_id="matscibench_005",
        ))

        # ═══ Properties domain (6 cases) ═══
        self.add(BenchmarkCase(
            task="Calculate the theoretical density of copper (FCC, atomic weight "
             "63.55 g/mol, lattice parameter 0.3615 nm, 4 atoms per unit cell). "
             "Answer in g/cm³.",
            expected_value=8.96, tolerance=0.1,
            evaluator=numeric_evaluator,
            category="properties", tags=["density", "fcc", "medium"],
            case_id="matscibench_006",
        ))
        self.add(BenchmarkCase(
            task="A metal rod (length 1 m, cross-section 1 cm²) is heated from "
             "20°C to 120°C. If α = 23×10⁻⁶ /°C and E = 70 GPa, calculate the "
             "thermal stress if the rod is fully constrained. Answer in MPa.",
            expected_value=161.0, tolerance=5.0,
            evaluator=numeric_evaluator,
            category="properties", tags=["thermal", "stress", "medium"],
            case_id="matscibench_007",
        ))
        self.add(BenchmarkCase(
            task="Calculate the electrical conductivity of a material with "
             "resistivity 1.7×10⁻⁸ Ω·m. Answer in S/m.",
            expected_value=5.88e7, tolerance=0.5e7,
            evaluator=numeric_evaluator,
            category="properties", tags=["electrical", "conductivity", "easy"],
            case_id="matscibench_008",
        ))
        self.add(BenchmarkCase(
            task="A ferromagnetic material has a saturation magnetization of "
             "1.7×10⁶ A/m. Calculate the magnetic flux density B_s in Tesla "
             "in vacuum (μ₀ = 4π×10⁻⁷ T·m/A).",
            expected_value=2.14, tolerance=0.05,
            evaluator=numeric_evaluator,
            category="properties", tags=["magnetic", "hard"],
            case_id="matscibench_009",
        ))
        self.add(BenchmarkCase(
            task="A material has a refractive index n = 1.5. Calculate the "
             "reflectance at normal incidence from air (n_air = 1.0). "
             "Use R = ((n-1)/(n+1))². Answer as a percentage.",
            expected_value=4.0, tolerance=0.5,
            evaluator=numeric_evaluator,
            category="properties", tags=["optical", "reflectance", "easy"],
            case_id="matscibench_010",
        ))
        self.add(BenchmarkCase(
            task="Write the expression for the stress-strain relationship in the "
             "linear elastic regime according to Hooke's law for a 3D isotropic "
             "material. Express it in terms of Young's modulus and Poisson's ratio.",
            rubric="Should include σ = Eε for uniaxial, or the full 3D tensor "
                   "form with Lamé constants or E, ν. Must mention Young's modulus "
                   "and Poisson's ratio.",
            evaluator=keyword_evaluator,
            expected_keywords=["young", "poisson", "stress", "strain"],
            category="properties", tags=["elasticity", "formula", "medium"],
            case_id="matscibench_011",
        ))

        # ═══ Structures domain (5 cases) ═══
        self.add(BenchmarkCase(
            task="The Miller indices of a plane in a cubic crystal are (1,1,1). "
             "What is the angle between this plane and the (1,0,0) plane? "
             "Answer in degrees.",
            expected_value=54.7, tolerance=1.0,
            evaluator=numeric_evaluator,
            category="structures", tags=["miller", "cubic", "medium"],
            case_id="matscibench_012",
        ))
        self.add(BenchmarkCase(
            task="For a BCC crystal with lattice parameter a = 0.2866 nm, "
             "calculate the atomic packing factor (APF).",
            expected_value=0.68, tolerance=0.01,
            evaluator=numeric_evaluator,
            category="structures", tags=["bcc", "packing", "hard"],
            case_id="matscibench_013",
        ))
        self.add(BenchmarkCase(
            task="Describe the difference between a Schottky defect and a Frenkel "
             "defect. Which is more likely in a ceramic with large anions?",
            expected_keywords=["schottky", "frenkel", "vacancy", "interstitial"],
            category="structures", tags=["defects", "point", "medium"],
            case_id="matscibench_014",
        ))
        self.add(BenchmarkCase(
            task="What is the difference between a grain boundary and a phase "
             "boundary? How do low-angle and high-angle grain boundaries differ?",
            expected_keywords=["grain boundary", "phase boundary", "low-angle",
                            "high-angle", "misorientation"],
            category="structures", tags=["grain_boundary", "interface", "medium"],
            case_id="matscibench_015",
        ))
        self.add(BenchmarkCase(
            task="Describe the structure of a screw dislocation. How does it "
             "differ from an edge dislocation in terms of Burgers vector "
             "direction relative to the dislocation line?",
            expected_keywords=["screw", "edge", "burgers", "parallel", "perpendicular"],
            category="structures", tags=["dislocation", "defect", "hard"],
            case_id="matscibench_016",
        ))

        # ═══ Fundamental mechanisms domain (6 cases) ═══
        self.add(BenchmarkCase(
            task="Calculate the change in Gibbs free energy (J/mol) for a process "
             "where ΔH = -100 kJ/mol and ΔS = -50 J/(mol·K) at T = 500 K. "
             "Use ΔG = ΔH - TΔS.",
            expected_value=-75000, tolerance=1000,
            evaluator=numeric_evaluator,
            category="fundamental_mechanisms", tags=["thermodynamics", "gibbs", "easy"],
            case_id="matscibench_017",
        ))
        self.add(BenchmarkCase(
            task="Using Fick's first law, calculate the diffusion flux (atoms/m²·s) "
             "through a membrane. D = 2×10⁻¹⁴ m²/s, concentration gradient "
             "dC/dx = -5×10²⁸ atoms/m⁴. J = -D(dC/dx).",
            expected_value=1.0e15, tolerance=0.2e15,
            evaluator=numeric_evaluator,
            category="fundamental_mechanisms", tags=["diffusion", "fick", "medium"],
            case_id="matscibench_018",
        ))
        self.add(BenchmarkCase(
            task="Calculate the activation energy (kJ/mol) for a reaction with "
             "rate constant k₁ = 1.0×10⁻³ at T₁ = 300 K and k₂ = 5.0×10⁻³ "
             "at T₂ = 350 K. Use Arrhenius: ln(k₂/k₁) = Ea/R × (1/T₁ - 1/T₂). "
             "R = 8.314 J/(mol·K).",
            expected_value=21.6, tolerance=2.0,
            evaluator=numeric_evaluator,
            category="fundamental_mechanisms", tags=["kinetics", "arrhenius", "hard"],
            case_id="matscibench_019",
        ))
        self.add(BenchmarkCase(
            task="Explain the difference between intrinsic and extrinsic "
             "semiconductors. How does temperature affect carrier concentration "
             "in each case?",
            expected_keywords=["intrinsic", "extrinsic", "carrier", "temperature",
                            "dopant"],
            category="fundamental_mechanisms", tags=["electronic", "semiconductor", "medium"],
            case_id="matscibench_020",
        ))
        self.add(BenchmarkCase(
            task="What is the relationship between band gap energy and the "
             "wavelength of light absorbed by a semiconductor? Calculate the "
             "maximum wavelength (nm) absorbed by Si (Eg = 1.12 eV). "
             "Use λ = hc/Eg, h = 6.626×10⁻³⁴ J·s, c = 3×10⁸ m/s.",
            expected_value=1107, tolerance=20,
            evaluator=numeric_evaluator,
            category="fundamental_mechanisms", tags=["electronic", "band_gap", "hard"],
            case_id="matscibench_021",
        ))
        self.add(BenchmarkCase(
            task="Describe the three primary types of atomic bonding (ionic, "
             "covalent, metallic). For each, state: directionality, electron "
             "sharing vs transfer, and typical material examples.",
            rubric_items=[
                {"criterion": "ionic bonding described", "weight": 2,
                 "keywords": ["ionic", "transfer"]},
                {"criterion": "covalent bonding described", "weight": 2,
                 "keywords": ["covalent", "share", "directional"]},
                {"criterion": "metallic bonding described", "weight": 2,
                 "keywords": ["metallic", "sea", "delocalized"]},
                {"criterion": "examples given", "weight": 1,
                 "keywords": ["nacl", "diamond", "copper", "iron"]},
            ],
            evaluator=rubric_evaluator,
            category="fundamental_mechanisms", tags=["bonding", "easy"],
            case_id="matscibench_022",
        ))

        # ═══ Processes domain (5 cases) ═══
        self.add(BenchmarkCase(
            task="In heat treatment of steel, what is the critical cooling rate "
             "and why is it important? Explain the relationship between cooling "
             "rate and microstructure formation.",
            expected_keywords=["critical cooling rate", "martensite", "austenite",
                            "transformation"],
            category="processes", tags=["heat_treatment", "steel", "hard"],
            case_id="matscibench_023",
        ))
        self.add(BenchmarkCase(
            task="Describe the difference between annealing, normalizing, and "
             "quenching. What microstructure does each produce in steel?",
            expected_keywords=["annealing", "normalizing", "quenching",
                            "pearlite", "martensite", "ferrite"],
            category="processes", tags=["heat_treatment", "medium"],
            case_id="matscibench_024",
        ))
        self.add(BenchmarkCase(
            task="What is sintering? Describe the three stages of sintering "
             "and the driving force for each.",
            expected_keywords=["sintering", "driving force", "surface area",
                            "neck", "densification"],
            category="processes", tags=["sintering", "ceramic", "medium"],
            case_id="matscibench_025",
        ))
        self.add(BenchmarkCase(
            task="Compare casting and forming as manufacturing processes. "
             "When would you choose each? Give one advantage and one "
             "disadvantage of each.",
            expected_keywords=["casting", "forming", "advantage", "disadvantage",
                            "liquid", "plastic"],
            category="processes", tags=["casting", "forming", "easy"],
            case_id="matscibench_026",
        ))
        self.add(BenchmarkCase(
            task="Describe physical vapor deposition (PVD) and chemical vapor "
             "deposition (CVD). How do they differ in mechanism and typical "
             "applications?",
            expected_keywords=["pvd", "cvd", "physical", "chemical", "vapor",
                            "deposition"],
            category="processes", tags=["deposition", "thin_film", "hard"],
            case_id="matscibench_027",
        ))

        # ═══ Failure mechanisms domain (5 cases) ═══
        self.add(BenchmarkCase(
            task="A steel component fails by fatigue after 10⁶ cycles at a stress "
             "amplitude of 250 MPa. If the S-N curve follows σ_a = σ_f' * (2N_f)^b "
             "with σ_f' = 1200 MPa and b = -0.10, verify the predicted life. "
             "Calculate the predicted number of reversals to failure.",
            expected_value=1.0e6, tolerance=2.0e5,
            evaluator=numeric_evaluator,
            category="failure_mechanisms", tags=["fatigue", "s-n", "hard"],
            case_id="matscibench_028",
        ))
        self.add(BenchmarkCase(
            task="Explain the difference between ductile and brittle fracture. "
             "What microstructural features distinguish the fracture surfaces?",
            expected_keywords=["ductile", "brittle", "dimples", "cleavage",
                            "microvoid"],
            category="failure_mechanisms", tags=["fracture", "medium"],
            case_id="matscibench_029",
        ))
        self.add(BenchmarkCase(
            task="Describe the three stages of creep. What happens in each stage? "
             "Sketch (describe) a typical creep strain vs time curve.",
            rubric_items=[
                {"criterion": "primary creep described", "weight": 2,
                 "keywords": ["primary", "decreasing", "strain hardening"]},
                {"criterion": "secondary creep described", "weight": 2,
                 "keywords": ["secondary", "steady", "constant"]},
                {"criterion": "tertiary creep described", "weight": 2,
                 "keywords": ["tertiary", "accelerating", "necking", "rupture"]},
            ],
            evaluator=rubric_evaluator,
            category="failure_mechanisms", tags=["creep", "hard"],
            case_id="matscibench_030",
        ))
        self.add(BenchmarkCase(
            task="Explain the mechanism of galvanic corrosion. Given two metals "
             "(Zn, E = -0.76 V and Cu, E = +0.34 V), which will corrode when "
             "coupled? Calculate the cell potential.",
            expected_value=1.10, tolerance=0.05,
            evaluator=numeric_evaluator,
            category="failure_mechanisms", tags=["corrosion", "galvanic", "medium"],
            case_id="matscibench_031",
        ))
        self.add(BenchmarkCase(
            task="Describe abrasive wear and adhesive wear. How do they differ "
             "in mechanism? Give one engineering strategy to reduce each.",
            expected_keywords=["abrasive", "adhesive", "wear", "hardness",
                            "lubrication"],
            category="failure_mechanisms", tags=["wear", "medium"],
            case_id="matscibench_032",
        ))
        return self

    def csmbench_cases(self) -> BenchmarkSuite:
        """CSMBench-style cases — cross-scale material science perception.

        Source: arXiv:2603.19327. 4 physical scales: atomic→micro→meso→macro.
        Adapted to text-based format since BenchmarkCase has no image field.
        Focus: can the agent reason about structure-property across scales?

        Atomic (Å): crystal structure, XRD, point defects, bonding
        Micro (nm): TEM, dislocations, precipitates, nanoscale interfaces
        Meso (μm): SEM, grain structure, porosity, cracks, phase distribution
        Macro (cm-m): tensile test, hardness, fracture surface, thermal
        Cross-scale: causal chain from atomic bonding to macro properties
        """
        # ═══ Atomic scale (5 cases) ═══
        self.add(BenchmarkCase(
            task="A material shows an XRD pattern with peaks at 2θ = 35°, 38°, 40° "
             "(Cu Kα, λ=1.5406 Å). The structure is hexagonal (wurtzite). "
             "Identify the material and calculate the lattice parameter a.",
            expected_value=3.25, tolerance=0.05,
            evaluator=numeric_evaluator,
            category="atomic", tags=["xrd", "wurtzite", "gan", "diffraction"],
            case_id="csmbench_001",
        ))
        self.add(BenchmarkCase(
            task="Calculate the interplanar spacing d_hkl for the (111) planes "
             "of an FCC crystal with lattice parameter a = 0.405 nm. "
             "Use d = a/√(h²+k²+l²). Answer in nm.",
            expected_value=0.234, tolerance=0.005,
            evaluator=numeric_evaluator,
            category="atomic", tags=["fcc", "interplanar", "easy"],
            case_id="csmbench_002",
        ))
        self.add(BenchmarkCase(
            task="The energy of a vacancy in copper is 0.9 eV. Calculate the "
             "equilibrium vacancy fraction at 1000 K. "
             "Use n_v/N = exp(-Q_v/kT), k = 8.617×10⁻⁵ eV/K.",
            expected_value=2.9e-5, tolerance=1.0e-5,
            evaluator=numeric_evaluator,
            category="atomic", tags=["vacancy", "defect", "boltzmann", "hard"],
            case_id="csmbench_003",
        ))
        self.add(BenchmarkCase(
            task="Describe the crystal structure of diamond cubic silicon. "
             "How many atoms per unit cell? What is the coordination number?",
            expected_keywords=["diamond", "cubic", "8", "4", "tetrahedral",
                            "covalent"],
            category="atomic", tags=["crystal", "silicon", "structure", "easy"],
            case_id="csmbench_004",
        ))
        self.add(BenchmarkCase(
            task="NaCl has a rock salt structure with lattice parameter a = 0.564 nm. "
             "Calculate the distance between nearest Na⁺ and Cl⁻ neighbors. "
             "Answer in nm.",
            expected_value=0.282, tolerance=0.005,
            evaluator=numeric_evaluator,
            category="atomic", tags=["nacl", "rocksalt", "ionic", "medium"],
            case_id="csmbench_005",
        ))

        # ═══ Micro scale (5 cases) ═══
        self.add(BenchmarkCase(
            task="A TEM image shows lattice fringes with spacing 0.334 nm. "
             "What crystallographic plane and material is this likely to be? "
             "Explain the relationship between fringe spacing and d-spacing.",
            expected_keywords=["graphite", "0.334", "d-spacing", "002",
                            "lattice"],
            category="micro", tags=["tem", "lattice_fringe", "carbon"],
            case_id="csmbench_006",
        ))
        self.add(BenchmarkCase(
            task="The Burgers vector of an edge dislocation in an FCC crystal "
             "is b = a/√2. If a = 0.405 nm, calculate the magnitude of b. "
             "Answer in nm.",
            expected_value=0.286, tolerance=0.005,
            evaluator=numeric_evaluator,
            category="micro", tags=["dislocation", "burgers", "fcc", "medium"],
            case_id="csmbench_007",
        ))
        self.add(BenchmarkCase(
            task="Describe how TEM differs from SEM in terms of imaging "
             "principle, resolution, and sample preparation. When would you "
             "choose TEM over SEM?",
            rubric_items=[
                {"criterion": "TEM imaging principle", "weight": 2,
                 "keywords": ["transmission", "electron", "thin"]},
                {"criterion": "SEM imaging principle", "weight": 2,
                 "keywords": ["scanning", "secondary", "backscatter"]},
                {"criterion": "resolution comparison", "weight": 2,
                 "keywords": ["resolution", "TEM", "higher", "sub-nm"]},
                {"criterion": "sample preparation difference", "weight": 1,
                 "keywords": ["thin", "bulk", "preparation"]},
            ],
            evaluator=rubric_evaluator,
            category="micro", tags=["tem", "sem", "characterization", "medium"],
            case_id="csmbench_008",
        ))
        self.add(BenchmarkCase(
            task="A precipitate has a critical radius r* = 2 nm and the "
             "interface energy γ = 0.5 J/m². Calculate the volume free energy "
             "ΔGv (J/m³) using r* = -2γ/ΔGv.",
            expected_value=-5.0e8, tolerance=1.0e8,
            evaluator=numeric_evaluator,
            category="micro", tags=["precipitate", "nucleation", "hard"],
            case_id="csmbench_009",
        ))
        self.add(BenchmarkCase(
            task="Explain how HRTEM can be used to identify crystal defects at "
             "the atomic scale. What information does the FFT of an HRTEM "
             "image provide?",
            expected_keywords=["hrtem", "fft", "defect", "fourier",
                            "atomic", "lattice"],
            category="micro", tags=["hrtem", "fft", "defect", "hard"],
            case_id="csmbench_010",
        ))

        # ═══ Meso scale (5 cases) ═══
        self.add(BenchmarkCase(
            task="An SEM image of a sintered ceramic shows grain sizes ranging "
             "from 2 to 10 μm with some porosity (~5%). Predict how increasing "
             "the sintering temperature would affect: (a) grain size, "
             "(b) density, (c) mechanical strength. Explain the Hall-Petch "
             "relationship.",
            rubric_items=[
                {"criterion": "grain growth with temperature", "weight": 2,
                 "keywords": ["grain growth", "larger"]},
                {"criterion": "density increase", "weight": 2,
                 "keywords": ["density", "densification"]},
                {"criterion": "Hall-Petch relationship", "weight": 2,
                 "keywords": ["hall-petch", "strength"]},
                {"criterion": "trade-off strength vs toughness", "weight": 1,
                 "keywords": ["trade-off", "brittle"]},
            ],
            evaluator=rubric_evaluator,
            category="meso", tags=["sem", "grain", "sintering", "hall-petch"],
            case_id="csmbench_011",
        ))
        self.add(BenchmarkCase(
            task="Using the Hall-Petch equation σ_y = σ₀ + k·d^(-1/2), "
             "calculate the yield strength for a material with σ₀ = 150 MPa, "
             "k = 0.45 MPa·m^1/2, and grain size d = 10 μm. Answer in MPa.",
            expected_value=292.3, tolerance=5.0,
            evaluator=numeric_evaluator,
            category="meso", tags=["hall-petch", "grain", "medium"],
            case_id="csmbench_012",
        ))
        self.add(BenchmarkCase(
            task="An EBSD map shows a bimodal grain size distribution (5 μm and "
             "50 μm) in a titanium alloy. Explain how this affects mechanical "
             "properties compared to a uniform grain structure.",
            expected_keywords=["bimodal", "grain", "titanium", "strength",
                            "ductility", "trade-off"],
            category="meso", tags=["ebsd", "titanium", "bimodal", "hard"],
            case_id="csmbench_013",
        ))
        self.add(BenchmarkCase(
            task="Describe how porosity affects the elastic modulus of a "
             "ceramic. If Young's modulus of fully dense Al₂O₃ is 380 GPa "
             "and the material has 15% porosity, estimate the modulus using "
             "E = E₀(1-1.9P+0.9P²). Answer in GPa.",
            expected_value=284.8, tolerance=5.0,
            evaluator=numeric_evaluator,
            category="meso", tags=["porosity", "modulus", "ceramic", "medium"],
            case_id="csmbench_014",
        ))
        self.add(BenchmarkCase(
            task="A meso-scale crack propagates along grain boundaries in a "
             "polycrystalline alloy. What is this type of fracture called? "
             "What microstructural features promote this failure mode?",
            expected_keywords=["intergranular", "grain boundary", "segregation",
                            "embrittlement"],
            category="meso", tags=["fracture", "intergranular", "medium"],
            case_id="csmbench_015",
        ))

        # ═══ Macro scale (5 cases) ═══
        self.add(BenchmarkCase(
            task="A tensile test on a metal specimen (gauge length 50 mm, "
             "cross-section 12.5 mm²) yields: at 2 mm extension the load is "
             "15 kN, at fracture the load is 12 kN and total extension is 8 mm. "
             "Calculate: (a) Young's modulus, (b) UTS, (c) elongation at fracture.",
            rubric_items=[
                {"criterion": "Young's modulus", "weight": 3,
                 "keywords": ["young", "modulus", "120", "gpa"]},
                {"criterion": "UTS calculation", "weight": 3,
                 "keywords": ["uts", "tensile", "1200", "mpa"]},
                {"criterion": "elongation at fracture", "weight": 2,
                 "keywords": ["elongation", "16%", "0.16"]},
            ],
            evaluator=rubric_evaluator,
            category="macro", tags=["tensile", "mechanical", "stress-strain"],
            case_id="csmbench_016",
        ))
        self.add(BenchmarkCase(
            task="A Vickers hardness test uses a load of 10 kgf and produces "
             "a diagonal of 0.3 mm. Calculate HV using "
             "HV = 1.8544 × P/d² where P is in kgf and d in mm.",
            expected_value=206, tolerance=5,
            evaluator=numeric_evaluator,
            category="macro", tags=["hardness", "vickers", "easy"],
            case_id="csmbench_017",
        ))
        self.add(BenchmarkCase(
            task="A macro-scale fracture surface shows chevron patterns pointing "
             "toward the origin. What does this indicate about the crack "
             "propagation direction and loading mode?",
            expected_keywords=["chevron", "crack", "origin", "propagation",
                            "brittle", "direction"],
            category="macro", tags=["fracture", "chevron", "medium"],
            case_id="csmbench_018",
        ))
        self.add(BenchmarkCase(
            task="A steel beam (length 2 m, rectangular cross-section 50×25 mm) "
             "is loaded in 3-point bending with a central load of 5000 N. "
             "Calculate the maximum stress. Use σ = 3FL/(2bd²) with b=50mm, "
             "d=25mm. Answer in MPa.",
            expected_value=480.0, tolerance=10.0,
            evaluator=numeric_evaluator,
            category="macro", tags=["bending", "stress", "beam", "hard"],
            case_id="csmbench_019",
        ))
        self.add(BenchmarkCase(
            task="A metal rod of length 2 m is heated from 25°C to 425°C. "
             "If α = 12×10⁻⁶ /°C, calculate the thermal expansion in mm.",
            expected_value=9.6, tolerance=0.2,
            evaluator=numeric_evaluator,
            category="macro", tags=["thermal", "expansion", "easy"],
            case_id="csmbench_020",
        ))

        # ═══ Cross-scale reasoning (4 cases) ═══
        self.add(BenchmarkCase(
            task="Explain how the atomic-scale bonding (covalent vs metallic) "
             "influences the macro-scale mechanical properties (brittleness vs "
             "ductility) of ceramics vs metals. Trace the causal chain through "
             "micro-scale dislocation mobility and meso-scale grain boundary "
             "behavior.",
            rubric_items=[
                {"criterion": "atomic bonding difference", "weight": 2,
                 "keywords": ["covalent", "metallic", "directional"]},
                {"criterion": "dislocation mobility link", "weight": 3,
                 "keywords": ["dislocation", "mobility", "slip"]},
                {"criterion": "grain boundary role", "weight": 2,
                 "keywords": ["grain boundary", "barrier", "block"]},
                {"criterion": "macro property connection", "weight": 2,
                 "keywords": ["brittle", "ductile"]},
            ],
            evaluator=rubric_evaluator,
            category="cross_scale",
            tags=["bonding", "dislocation", "cross-scale", "hard"],
            case_id="csmbench_021",
        ))
        self.add(BenchmarkCase(
            task="Trace the causal chain from atomic-scale dopant addition "
             "(e.g., P in Si) to macro-scale electrical conductivity. "
             "What happens at each scale: atomic, micro, meso, macro?",
            rubric_items=[
                {"criterion": "atomic: dopant substitution", "weight": 2,
                 "keywords": ["substitution", "dopant", "phosphorus"]},
                {"criterion": "micro: carrier generation", "weight": 2,
                 "keywords": ["carrier", "electron", "donor"]},
                {"criterion": "meso: grain boundary scattering", "weight": 2,
                 "keywords": ["scattering", "mobility", "grain"]},
                {"criterion": "macro: conductivity measurement", "weight": 2,
                 "keywords": ["conductivity", "resistivity", "macro"]},
            ],
            evaluator=rubric_evaluator,
            category="cross_scale",
            tags=["doping", "semiconductor", "conductivity", "cross-scale", "hard"],
            case_id="csmbench_022",
        ))
        self.add(BenchmarkCase(
            task="A polycrystalline ceramic has 5% porosity and average grain "
             "size 2 μm. Explain how BOTH atomic-scale (vacancy concentration) "
             "and meso-scale (porosity + grain size) features independently "
             "affect the macro-scale thermal conductivity. Which scale "
             "dominates at room temperature vs high temperature?",
            rubric_items=[
                {"criterion": "vacancy scattering at atomic scale", "weight": 2,
                 "keywords": ["vacancy", "phonon", "scattering"]},
                {"criterion": "porosity effect at meso scale", "weight": 2,
                 "keywords": ["porosity", "pore", "thermal resistance"]},
                {"criterion": "grain boundary scattering", "weight": 2,
                 "keywords": ["grain boundary", "phonon", "scattering"]},
                {"criterion": "temperature dependence", "weight": 2,
                 "keywords": ["temperature", "room", "high", "dominates"]},
            ],
            evaluator=rubric_evaluator,
            category="cross_scale",
            tags=["thermal", "conductivity", "porosity", "vacancy", "cross-scale", "hard"],
            case_id="csmbench_023",
        ))
        self.add(BenchmarkCase(
            task="Explain how processing (sintering temperature) affects atomic "
             "diffusion → grain growth → meso grain size → macro mechanical "
             "strength. Give the governing equation at each scale.",
            rubric_items=[
                {"criterion": "atomic: diffusion equation", "weight": 2,
                 "keywords": ["diffusion", "fick", "d = d0"]},
                {"criterion": "grain growth kinetics", "weight": 2,
                 "keywords": ["grain growth", "d^n", "time"]},
                {"criterion": "meso: Hall-Petch", "weight": 2,
                 "keywords": ["hall-petch", "grain size", "strength"]},
                {"criterion": "macro: strength measurement", "weight": 2,
                 "keywords": ["yield", "strength", "modulus"]},
            ],
            evaluator=rubric_evaluator,
            category="cross_scale",
            tags=["sintering", "diffusion", "hall-petch", "cross-scale", "hard"],
            case_id="csmbench_024",
        ))
        return self

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
                        {"trial_runs": [[asdict(r) for r in run] for run in trial_runs]},
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

            case_results.append(CaseTrialResult(
                case_id=case.case_id,
                task=case.task,
                category=case.category,
                trials=trial_results,
                pass_all=pass_all,
                pass_any=pass_any,
                avg_score=round(sum(scores) / len(scores), 3) if scores else 0.0,
                max_score=max(scores) if scores else 0.0,
            ))

        n = len(self.cases) or 1
        pass_all_rate = sum(1 for cr in case_results if cr.pass_all) / n
        pass_any_rate = sum(1 for cr in case_results if cr.pass_any) / n

        S = sum(all_scores) / len(all_scores) if all_scores else 0.0
        r_all = pass_all_rate ** (1.0 / 3.0)
        r_any = 1.0 - (1.0 - pass_any_rate) ** (1.0 / 3.0)
        final_score = round(100.0 * (S ** 0.40) * (r_all ** 0.45) * (r_any ** 0.15), 2)

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
            avg_latency_ms=round(sum(all_latencies) / len(all_latencies), 2) if all_latencies else 0.0,
            coverage=round(coverage, 4),
        )

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
                "pass_all_rate": results.pass_all_rate,   # pass^3
                "pass_any_rate": results.pass_any_rate,   # pass@3
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
