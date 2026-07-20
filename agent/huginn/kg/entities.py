"""Entity/relation types and helpers for the project knowledge graph."""

from __future__ import annotations

from typing import Any


class EntityType:
    TOPIC = "Topic"
    MATERIAL = "Material"
    TOOL = "Tool"
    METHOD = "Method"
    ERROR_PATTERN = "ErrorPattern"
    FACT = "Fact"
    SESSION = "Session"
    RESOURCE = "Resource"
    LITERATURE = "Literature"
    # ── Materials science GraphRAG extensions ──
    ELEMENT = "Element"
    COMPOUND = "Compound"
    PROPERTY = "Property"
    CRYSTAL_STRUCTURE = "CrystalStructure"
    APPLICATION = "Application"
    # ── Math concept layer (MathConceptGraph) ──
    MATH_CONCEPT = "MathConcept"
    # ── P13 CrossDomain transfer history ──
    CROSS_DOMAIN_TRANSFER = "cross_domain_transfer"


class Relation:
    MENTIONS = "mentions"
    USED = "used"
    FAILED_WITH = "failed_with"
    SOLVED_BY = "solved_by"
    APPLIES = "applies"
    RELATED_TO = "related_to"
    DERIVED_FROM = "derived_from"
    RUNS_ON = "runs_on"
    CITES = "cites"
    CITED_BY = "cited_by"
    REPRODUCES = "reproduces"
    USES_METHOD_FROM = "uses_method_from"
    EXTENDS = "extends"
    CONTRADICTS = "contradicts"
    # ── Materials science relations ──
    HAS_ELEMENT = "has_element"          # Compound → Element (with stoichiometry attr)
    HAS_PROPERTY = "has_property"        # Compound → Property (with value+unit attrs)
    HAS_STRUCTURE = "has_structure"      # Compound → CrystalStructure
    USED_FOR = "used_for"                # Compound → Application
    COMPUTED_WITH = "computed_with"      # Property → Tool
    VALIDATES = "validates"              # Paper → Property
    # ── Math concept relations (MathConceptGraph) ──
    DEPENDS_ON = "depends_on"            # Hausdorff → Compact (Hausdorff 需要 Topology)
    GENERALIZES = "generalizes"          # Metric → Topological (Metric 是 Topological 的特例)
    DUAL_TO = "dual_to"                  # Maximize stability ↔ Minimize instability path
    # ── P13 CrossDomain transfer relations ──
    CROSS_DOMAIN_ANALOGY = "cross_domain_analogy"      # source problem → transfer node
    TRANSFERS_TO = "transfers_to"                      # transfer node → target domain
    STRUCTURALLY_ISOMORPHIC = "structurally_isomorphic"  # transfer ↔ math concept


# Common keywords used for rule-based extraction.
TOOL_KEYWORDS: set[str] = {
    "VASP",
    "LAMMPS",
    "Quantum ESPRESSO",
    "QE",
    "CP2K",
    "ORCA",
    "Gaussian",
    "ABAQUS",
    "Abaqus",
    "OpenFOAM",
    "GROMACS",
    "NWChem",
    "Psi4",
    "GAMESS",
    "CASTEP",
    "Siesta",
    "Octopus",
    "DFTB+",
    "Materials Project",
    "AFLOW",
    "OQMD",
    "NOMAD",
}


METHOD_KEYWORDS: set[str] = {
    "DFT",
    "MD",
    "FEM",
    "CFD",
    "QC",
    "QM",
    "Monte Carlo",
    "molecular dynamics",
    "density functional theory",
    "finite element",
    "computational fluid dynamics",
}


def node_id(label: str, entity_type: str) -> str:
    """Stable node id from label and type."""
    return f"{entity_type}:{label.strip()}"


def normalize_props(props: dict[str, Any]) -> dict[str, Any]:
    """Strip and stringify props for JSON serialization."""
    clean: dict[str, Any] = {}
    for k, v in props.items():
        if v is None:
            continue
        if isinstance(v, str):
            v = v.strip()
        clean[k] = v
    return clean
