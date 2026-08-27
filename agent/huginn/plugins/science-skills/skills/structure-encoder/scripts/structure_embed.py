#!/usr/bin/env python3
"""Normalize a crystal-structure file or chemical formula into a unified descriptor JSON.

Retrieval-scope "heterogeneous material encoder": instead of training a joint
embedding, we normalize any structure input (CIF / POSCAR / xyz / formula) into
a small, stable JSON descriptor — composition, space group, cell metric, and a
species-share fingerprint. That descriptor can then flow into Huginn's existing
text/knowledge search pipeline.

Deliberately deterministic and dependency-light (pymatgen only). No neural
weights here; when a real trained encoder lands later, swapping this function's
internals is enough — the CLI/output contract stays the same.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def formula_descriptor(formula: str) -> dict:
    """Parse a chemical formula like LiFePO4 into a composition descriptor."""
    try:
        from pymatgen.core import Composition
    except ImportError as exc:
        raise RuntimeError("需要 pymatgen；请在 skill 目录运行 `uv run`。") from exc
    comp = Composition(formula)
    return {
        "source": "formula",
        "formula": comp.reduced_formula,
        "elements": sorted(e.symbol for e in comp.elements),
        "species_shares": {
            str(e): round(float(comp.get_atomic_fraction(e)), 6)
            for e in comp.elements
        },
        "normalized_formula": comp.reduced_formula,
    }


def structure_file_descriptor(path: Path) -> dict:
    """Parse a structure file (CIF/POSCAR/xyz/json) into a unified descriptor."""
    try:
        from pymatgen.core import Structure
    except ImportError as exc:
        raise RuntimeError("需要 pymatgen；请在 skill 目录运行 `uv run`。") from exc

    try:
        structure = Structure.from_file(str(path))
    except Exception as exc:
        raise ValueError(f"无法解析结构文件 {path}: {exc}") from exc

    comp = structure.composition
    species_shares = {str(e): round(float(comp.get_atomic_fraction(e)), 6) for e in comp.elements}
    # 空间群判定可能因对称性分析失败而抛错(比如非周期零碎结构), 单独兜底.
    try:
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        analyzer = SpacegroupAnalyzer(structure)
        group = analyzer.get_space_group_symbol()
    except Exception:
        group = None

    return {
        "source": "structure_file",
        "file": path.name,
        "formula": comp.reduced_formula,
        "n_atoms": int(structure.num_sites),
        "elements": sorted(e.symbol for e in comp.elements),
        "species_shares": species_shares,
        "density": round(float(structure.density), 6),
        "volume_ang3": round(float(structure.volume), 6),
        "space_group": group,
        "lattice_abc": [round(float(x), 6) for x in structure.lattice.abc],
        "lattice_angles": [round(float(x), 6) for x in structure.lattice.angles],
    }


def build_descriptor(input_arg: str) -> dict:
    """Route an input to the right descriptor path.

    A string that matches an existing file is treated as a structure file;
    otherwise it is parsed as a chemical formula.
    """
    inp = Path(input_arg)
    if inp.is_file():
        return structure_file_descriptor(inp)
    return formula_descriptor(input_arg)


def main(argv: list[str] | None = None) -> int:
    # --query 接受结构文件路径或化学式, 与 science_skills_bridge 的泛化参数骨架对齐
    # (bridge 会用 action + --query <input> + --output <path> 调用我们).
    ap = argparse.ArgumentParser(
        description="Normalize a structure file or formula into a unified descriptor JSON."
    )
    ap.add_argument("--query", default=None, help="结构文件路径 (CIF/POSCAR/xyz) 或化学式, 如 LiFePO4")
    ap.add_argument("input", nargs="?", help="(旧) 位置参数形式的结构输入")
    ap.add_argument("--output", help="写入 JSON 文件的路径; 缺省打到 stdout")
    # 兼容 action 子命令(如 encode); bridge 可能追加一个 action, 这里吸收忽略即可
    args = ap.parse_args(argv)
    if not hasattr(args, "action"):
        # bridge 会把 args.action 作为第一个位置参添上; 我们是通过 --query 取输入, 忽略它
        pass

    input_arg = args.query or args.input
    if not input_arg:
        print("ERROR: 需要提供 --query", file=sys.stderr)
        return 2

    try:
        descriptor = build_descriptor(input_arg)
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    payload = json.dumps(descriptor, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())