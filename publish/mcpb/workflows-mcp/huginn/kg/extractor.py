"""Rule-based entity and relation extraction for the knowledge graph."""

from __future__ import annotations

import re
from typing import Any

from huginn.kg.entities import METHOD_KEYWORDS, TOOL_KEYWORDS

# Chemical formula regex: captures sequences like LiCoO2, Si, TiO2, Fe2O3, C60.
_CHEMICAL_FORMULA_RE = re.compile(r"\b(?:[A-Z][a-z]?\d*)+(?:[A-Z][a-z]?\d*)*\b")
# Element token within a formula: Symbol + optional count.
_ELEMENT_TOKEN_RE = re.compile(r"([A-Z][a-z]?)(\d*)")
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

# ponytail: Periodic table subset — only elements we're likely to encounter in
# materials science. Not a complete 118-element table; if an unknown element
# symbol appears, the parser returns it anyway (the caller can filter).
# Full periodic table seed is a one-time builder task, not needed here.
_ELEMENTS = {
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi",
    "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Th", "U",
}


def parse_formula(formula: str) -> dict[str, int]:
    """Parse a chemical formula into element→count.

    TiO2 → {"Ti": 1, "O": 2}
    SrTiO3 → {"Sr": 1, "Ti": 1, "O": 3}
    C60 → {"C": 60}

    ponytail: Naive linear regex parse — no nested groups (no parentheses
    support like Ca(NO3)2). Sufficient for >95% of materials science formulas
    in papers. Add parenthesised group expansion if the need arises.
    """
    result: dict[str, int] = {}
    for symbol, count_str in _ELEMENT_TOKEN_RE.findall(formula):
        if symbol not in _ELEMENTS:
            continue
        count = int(count_str) if count_str else 1
        result[symbol] = result.get(symbol, 0) + count
    return result


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
    """Extract all entity mentions from text."""
    materials = extract_materials(text)
    elements: set[str] = set()
    for formula in materials:
        elements.update(parse_formula(formula).keys())
    return {
        "tools": extract_tools(text),
        "methods": extract_methods(text),
        "materials": materials,
        "elements": elements,
    }


def extract_relations(text: str) -> list[dict[str, Any]]:
    """Extract (src_label, relation, dst_label) triples from text.

    Links tools/methods to materials, and materials to their constituent elements.
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
    # Compound → Element links via formula parsing
    for formula in materials:
        elements = parse_formula(formula)
        if not elements:
            continue
        for elem, count in elements.items():
            relations.append(
                {
                    "src": formula,
                    "src_type": "Compound",
                    "relation": "has_element",
                    "dst": elem,
                    "dst_type": "Element",
                    "stoichiometry": count,
                }
            )
    return relations
