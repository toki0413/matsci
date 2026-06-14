#!/usr/bin/env python3
"""
Hierarchical Router Retriever for Huginn.

Routes queries to appropriate sub-indexes based on software/method detection,
then falls back to general search. Improves retrieval precision by
narrowing the search space before semantic matching.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class HierarchicalRetriever:
    """Multi-level retriever that routes queries to domain-specific indexes."""

    # Software keywords for routing
    SOFTWARE_KEYWORDS: dict[str, list[str]] = {
        "Multiwfn": ["multiwfn", "波函数分析", "wfn", "fchk", "molden", "cube"],
        "Gaussian": ["gaussian", "g16", "g09", "gjf", "chk", "fch", "route section"],
        "ORCA": ["orca", "gbw", "molden", "inp", "%pal", "%scf"],
        "VMD": ["vmd", "可视化", "render", "snapshot", "tcl"],
        "CP2K": ["cp2k", "inp", "pbe", "dft-d3", "npt", "nvt"],
        "GROMACS": ["gromacs", "gro", "top", "mdp", "tpr", "gmx"],
        "VASP": ["vasp", "incar", "poscar", "kpoints", "potcar", "nbands"],
        "LAMMPS": ["lammps", "in.", "data.", "dump", "fix", "pair_style"],
        # FEA / Solid Mechanics
        "ABAQUS": ["abaqus", "cae", "odb", "inp", "umat", "vumat", "job", "assembly"],
        "ANSYS": ["ansys", "mapdl", "workbench", "apdl", "db", "rst", "cfd-post"],
        "COMSOL": ["comsol", "mph", "multiphysics", "physics", "feature"],
        "DAMASK": ["damask", "cpfem", "crystal plasticity", "spectral", "grid"],
        # CFD / Fluid Mechanics
        "OpenFOAM": ["openfoam", "foam", "simplefoam", "pimplefoam", "interfoam", "fvscheme", "fvsolution"],
        "Fluent": ["fluent", "udf", "scheme", "journal", "case", "data"],
    }

    METHOD_KEYWORDS: dict[str, list[str]] = {
        # Quantum Chemistry
        "ESP": ["esp", "静电势", "electrostatic potential", "mep"],
        "RESP": ["resp", "电荷拟合", "charge fitting", "restrained esp"],
        "NCI": ["nci", "弱相互作用", "rdg", "non-covalent"],
        "IGMH": ["igmh", "弱相互作用", "independent gradient"],
        "NTO": ["nto", "自然跃迁轨道", "natural transition"],
        "TDDFT": ["tddft", "激发态", "excited state", "cios"],
        "Hirshfeld": ["hirshfeld", "原子电荷", "atomic charge", "hirshfeld-i"],
        "NICS": ["nics", "芳香性", "aromaticity", "磁屏蔽"],
        "ALIE": ["alie", "平均局部离子化能", "ionization energy"],
        "LEAE": ["leae", "局部电子附着能", "electron attachment"],
        "Fukui": ["fukui", "福井函数", "反应位点", "reactivity"],
        "DualDescriptor": ["dual descriptor", "双描述符", "dd"],
        # FEA / Solid Mechanics
        "FEA": ["fea", "finite element", "有限元", "structural analysis", "stress", "strain", "deformation"],
        "CPFEM": ["cpfem", "crystal plasticity", "晶体塑性", "slip system", "taylor model"],
        "Fracture": ["fracture", "j-integral", "ctod", "stress intensity", "crack", "断裂"],
        "Modal": ["modal", "eigenvalue", "natural frequency", "mode shape", "振型"],
        "Buckling": ["buckling", "eigenvalue buckling", "critical load", "失稳"],
        # CFD / Fluid Mechanics
        "RANS": ["rans", "k-epsilon", "k-omega", "sst", "spalart-allmaras", "湍流"],
        "LES": ["les", "large eddy simulation", "smagorinsky", "wale", "subgrid"],
        "Multiphase": ["multiphase", "vof", "euler-euler", "dpm", "两相流", "interface"],
        "HeatTransfer": ["heat transfer", "conjugate heat transfer", "cht", "nusselt", "传热"],
    }

    def __init__(self, vector_store: Any, index_path: str | None = None):
        self.store = vector_store
        self._index: dict[str, Any] = {"software": {}, "method": {}}
        self._load_index(index_path)

    def _load_index(self, path: str | None) -> None:
        if path and Path(path).exists():
            with open(path, "r", encoding="utf-8") as f:
                self._index = json.load(f)
            return
        # Try default location
        repo_root = Path(__file__).resolve().parent.parent.parent.parent
        default = repo_root / "Sobko_MCP_project" / "advanced_optimization" / "hierarchical_index.json"
        if default.exists():
            with open(default, "r", encoding="utf-8") as f:
                self._index = json.load(f)

    def _detect_route(self, query: str) -> tuple[str | None, str | None]:
        """Detect which software and/or method the query is about."""
        q_lower = query.lower()
        detected_sw = None
        detected_method = None
        max_sw_score = 0
        max_method_score = 0

        for sw, keywords in self.SOFTWARE_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in q_lower)
            if score > max_sw_score:
                max_sw_score = score
                detected_sw = sw

        for method, keywords in self.METHOD_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in q_lower)
            if score > max_method_score:
                max_method_score = score
                detected_method = method

        return detected_sw, detected_method

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_dict: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Search with hierarchical routing.

        Returns:
            {
                "route": {"software": str|None, "method": str|None},
                "results": list[dict],
                "general_results": list[dict],
                "routing_reason": str,
            }
        """
        sw, method = self._detect_route(query)
        route_info = {"software": sw, "method": method}

        # Build targeted filter
        targeted_filter = dict(filter_dict) if filter_dict else {}
        if sw:
            # Narrow to software-specific entries if store supports metadata filtering
            targeted_filter["software_tags"] = sw

        # Primary search: targeted
        try:
            targeted_results = self.store.search(
                query=query,
                top_k=top_k,
                filter_dict=targeted_filter if targeted_filter else None,
            )
        except Exception:
            targeted_results = []

        # Fallback search: general
        try:
            general_results = self.store.search(
                query=query,
                top_k=top_k,
                filter_dict=filter_dict,
            )
        except Exception:
            general_results = []

        # Deduplicate and merge
        seen_ids = {r.get("id", r.get("chunk_id", "")) for r in targeted_results}
        merged = list(targeted_results)
        for r in general_results:
            rid = r.get("id", r.get("chunk_id", ""))
            if rid not in seen_ids:
                merged.append(r)

        # Build routing reason
        if sw and method:
            reason = f"Routed to {sw} software index + {method} method index"
        elif sw:
            reason = f"Routed to {sw} software index"
        elif method:
            reason = f"Routed to {method} method index"
        else:
            reason = "No specific route detected; using general search"

        return {
            "route": route_info,
            "results": merged[:top_k],
            "targeted_count": len(targeted_results),
            "general_count": len(general_results),
            "routing_reason": reason,
        }
