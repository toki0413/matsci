"""Rule-based entity and relation extraction for the knowledge graph."""

from __future__ import annotations

import re
from typing import Any

from huginn.kg.entities import METHOD_KEYWORDS, TOOL_KEYWORDS

# Chemical formula regex: captures sequences like LiCoO2, Si, TiO2, Fe2O3, C60.
_CHEMICAL_FORMULA_RE = re.compile(r"\b(?:[A-Z][a-z]?\d*)+(?:[A-Z][a-z]?\d*)*\b")
# Filter out common false positives.
_FALSE_FORMULAS = {
    "DFT",
    "MD",
    "FEM",
    "CFD",
    "QC",
    "QM",
    "GPU",
    "CPU",
    "API",
    "JSON",
    "XML",
    "HTML",
    "SQL",
    "SSH",
    "SCP",
    "PDF",
    "CSV",
    "PNG",
    "JPG",
    "HTTP",
    "URL",
    "VASP",
    "LAMMPS",
    "QE",
    "CP2K",
    "ORCA",
    "ABAQUS",
    "GROMACS",
    "NWChem",
}


def extract_tools(text: str) -> set[str]:
    """Extract known tool/software names from text."""
    found: set[str] = set()
    for keyword in TOOL_KEYWORDS:
        if keyword.lower() in text.lower():
            found.add(keyword)
    return found


def extract_methods(text: str) -> set[str]:
    """Extract known method names from text."""
    found: set[str] = set()
    lower = text.lower()
    for keyword in METHOD_KEYWORDS:
        if keyword.lower() in lower:
            found.add(keyword)
    return found


def extract_materials(text: str) -> set[str]:
    """Extract candidate chemical formulas from text."""
    found: set[str] = set()
    for match in _CHEMICAL_FORMULA_RE.finditer(text):
        token = match.group(0)
        if token in _FALSE_FORMULAS:
            continue
        if len(token) == 1 and token.isupper():
            found.add(token)
            continue
        # Simple heuristic: must contain at least one lowercase letter or digit
        # and start with an uppercase letter, or be a single uppercase element.
        if any(c.islower() or c.isdigit() for c in token):
            found.add(token)
    return found


def extract_error_pattern(error_message: str | None) -> str | None:
    """Map a raw error message to a canonical error pattern label."""
    if not error_message:
        return None
    lower = error_message.lower()
    if any(k in lower for k in ("scf", "electronic", "convergence")):
        return "SCF convergence failure"
    if any(k in lower for k in ("ionic", "relaxation", "geometry")):
        return "Geometry relaxation failure"
    if any(k in lower for k in ("memory", "oom", "out of memory")):
        return "Out of memory"
    if any(k in lower for k in ("timeout", "timed out")):
        return "Timeout"
    if any(k in lower for k in ("lost atoms", "bond atoms missing")):
        return "Lost atoms / broken topology"
    if "permission" in lower or "access" in lower:
        return "Permission / access denied"
    if "file" in lower and ("not found" in lower or "missing" in lower):
        return "Missing file"
    return "Unknown error"


def extract_entities(text: str) -> dict[str, set[str]]:
    """Extract all entity mentions from a text."""
    return {
        "tools": extract_tools(text),
        "methods": extract_methods(text),
        "materials": extract_materials(text),
    }


def extract_relations(text: str) -> list[dict[str, Any]]:
    """Extract simple (src_label, relation, dst_label) triples from text.

    First version only links the first mentioned method/tool to materials.
    """
    entities = extract_entities(text)
    relations: list[dict[str, Any]] = []
    materials = entities["materials"]
    for tool in entities["tools"]:
        for mat in materials:
            relations.append(
                {
                    "src": tool,
                    "src_type": "Tool",
                    "relation": "applies",
                    "dst": mat,
                    "dst_type": "Material",
                }
            )
    for method in entities["methods"]:
        for mat in materials:
            relations.append(
                {
                    "src": method,
                    "src_type": "Method",
                    "relation": "applies",
                    "dst": mat,
                    "dst_type": "Material",
                }
            )
    return relations
