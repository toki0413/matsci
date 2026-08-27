#!/usr/bin/env python3
"""Deterministic smoke test for structure-encoder (no real files needed)."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

from structure_embed import (
    build_descriptor,
    formula_descriptor,
    structure_file_descriptor,
)


def test_formula() -> None:
    """A chemical formula should normalize to reduced formula + species shares."""
    desc = formula_descriptor("LiFePO4")
    assert desc["formula"] == "LiFePO4"
    assert set(desc["elements"]) == {"Li", "Fe", "P", "O"}
    assert desc["species_shares"]["O"] == 0.571429


def test_structure_file(tmpdir: Path) -> None:
    """A synthetic cubic Si structure should parse into a full descriptor."""
    from pymatgen.core import Lattice, Structure

    lattice = Lattice.cubic(5.43)
    structure = Structure(lattice, ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]])
    cif = tmpdir / "Si.cif"
    structure.to(fmt="cif", filename=str(cif))

    desc = structure_file_descriptor(cif)
    assert desc["formula"] == "Si"
    assert desc["n_atoms"] == 2
    assert set(desc["elements"]) == {"Si"}
    assert desc["lattice_abc"] == [5.43, 5.43, 5.43]


def test_route() -> None:
    """build_descriptor routes a real path to file parsing."""
    with tempfile.TemporaryDirectory() as d:
        cif = Path(d) / "x.cif"
        from pymatgen.core import Lattice, Structure

        structure = Structure(Lattice.cubic(4.0), ["Fe", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]])
        structure.to(fmt="cif", filename=str(cif))
        desc = build_descriptor(str(cif))
        assert desc["source"] == "structure_file"

    # a pure formula string stays on the formula path
    desc2 = build_descriptor("NaCl")
    assert desc2["source"] == "formula"
    assert desc2["formula"] == "NaCl"


def test_json_serializable() -> None:
    """Every descriptor field must survive json.dumps (no Element objects)."""
    for desc in (formula_descriptor("LiFePO4"),):
        json.dumps(desc, ensure_ascii=False)
    # build one from a file too
    with tempfile.TemporaryDirectory() as d:
        from pymatgen.core import Lattice, Structure

        structure = Structure(Lattice.orthorhombic(10, 6, 4), ["Li"], [[0, 0, 0]])
        cif = Path(d) / "x.cif"
        structure.to(fmt="cif", filename=str(cif))
        json.dumps(structure_file_descriptor(cif), ensure_ascii=False)


if __name__ == "__main__":
    test_formula()
    test_structure_file(Path(tempfile.mkdtemp(prefix="struct_smoke_")))
    test_route()
    test_json_serializable()
    print("structure-encoder smoke test OK")