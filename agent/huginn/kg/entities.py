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
