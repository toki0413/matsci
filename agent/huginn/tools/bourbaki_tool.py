"""Bourbaki Tool — Mathematical Structure Modeling for Huginn.

Wraps Bourbaki (math_anything) 3-layer architecture:
  Foundation (algorithms) → Structures (types) → Domains (physics)

Exposes 7 physics domains (DFT, CFD, MD, FEM, EM, QC, PhaseField),
18 conservation fields, morphism chains, type theory verification,
dimensional analysis, and symbolic regression as Huginn tools.

Usage:
    BourbakiTool().call(BourbakiInput(action="analyze_domain", domain="dft"), ctx)
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool
from huginn.types import ToolContext, ToolResult


class BourbakiInput(BaseModel):
    """Input schema for Bourbaki mathematical structure tool."""

    action: Literal[
        "analyze_domain",
        "compare_domains",
        "build_conservation_field",
        "analyze_morphism_chain",
        "buckingham_pi",
        "discover_equations",
        "extract_engine",
        "list_domains",
        "list_equation_types",
    ] = Field(description="Bourbaki operation to perform")

    # Domain analysis
    domain: str | None = Field(
        default=None, description="Physics domain name (dft, cfd, md, fem, em, qc, phase_field)"
    )
    domain_b: str | None = Field(
        default=None, description="Second domain for comparison"
    )
    parameters: dict[str, Any] | None = Field(
        default=None, description="Domain-specific parameters"
    )
    parameters_b: dict[str, Any] | None = Field(
        default=None, description="Second domain parameters for comparison"
    )

    # Conservation field
    equation_type: str | None = Field(
        default=None,
        description="Equation system type (navier_stokes, schrodinger, maxwell, elasticity, heat, wave, kohn_sham, ...)",
    )

    # Morphism chain
    chain: list[str] | None = Field(
        default=None, description="Specific morphism names to trace"
    )

    # Buckingham Pi
    variables: list[tuple[str, str]] | list[dict[str, Any]] | None = Field(
        default=None, description="List of (name, unit) tuples or dicts with 'name', 'symbol', 'dimensions' for dimensional analysis"
    )
    target: str | None = Field(
        default=None, description="Target variable for Buckingham Pi grouping"
    )

    # Equation discovery
    data: list[dict[str, float]] | None = Field(
        default=None, description="Tabular data for symbolic regression"
    )
    target_variable: str | None = Field(
        default=None, description="Column name to predict"
    )
    max_complexity: int = Field(default=5, ge=1, le=20)

    # Engine extraction
    engine: str | None = Field(
        default=None,
        description="Computational engine (vasp, lammps, abaqus, ansys, comsol, gromacs, multiwfn)",
    )
    engine_params: dict[str, Any] | None = Field(
        default=None, description="Engine parameters dict (e.g., {'ENCUT': 520, 'SIGMA': 0.05})"
    )


class BourbakiTool(HuginnTool[BourbakiInput, BaseModel]):
    """Bourbaki — Mathematical Structure Modeling for Computational Science."""

    name = "bourbaki"
    description = (
        "Analyze the mathematical structure behind physics simulations. "
        "Actions: analyze_domain (dft/cfd/md/fem/em/qc/phase_field), "
        "compare_domains, build_conservation_field (navier_stokes/schrodinger/maxwell/...), "
        "analyze_morphism_chain, buckingham_pi (dimensional analysis), "
        "discover_equations (symbolic regression from data), extract_engine (vasp/lammps/...), "
        "list_domains, list_equation_types."
    )
    read_only = True

    input_schema = BourbakiInput

    def __init__(self) -> None:
        """Initialize with lazy imports to avoid heavy deps at registry time."""
        self._ma: Any | None = None
        self._domains: Any | None = None
        self._structures: Any | None = None
        self._dimensional: Any | None = None
        self._eml: Any | None = None

    def _ensure_loaded(self) -> None:
        """Lazy-load math_anything modules."""
        if self._ma is not None:
            return
        from math_anything import MathAnything
        from math_anything import domains as _domains
        from math_anything import structures as _structures
        from math_anything import dimensional as _dimensional
        from math_anything import eml_v2 as _eml

        self._ma = MathAnything()
        self._domains = _domains
        self._structures = _structures
        self._dimensional = _dimensional
        self._eml = _eml

    async def call(self, args: BourbakiInput, context: ToolContext) -> ToolResult:
        self._ensure_loaded()
        try:
            result = self._dispatch(args)
            return ToolResult(
                success=True,
                data={"result": result},
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Bourbaki {args.action} failed: {e}",
            )

    def _dispatch(self, args: BourbakiInput) -> str:
        action = args.action

        if action == "analyze_domain":
            return self._analyze_domain(args)
        if action == "compare_domains":
            return self._compare_domains(args)
        if action == "build_conservation_field":
            return self._build_conservation_field(args)
        if action == "analyze_morphism_chain":
            return self._analyze_morphism_chain(args)
        if action == "buckingham_pi":
            return self._buckingham_pi(args)
        if action == "discover_equations":
            return self._discover_equations(args)
        if action == "extract_engine":
            return self._extract_engine(args)
        if action == "list_domains":
            return self._list_domains()
        if action == "list_equation_types":
            return self._list_equation_types()

        raise ValueError(f"Unknown Bourbaki action: {action}")

    def _analyze_domain(self, args: BourbakiInput) -> str:
        domain = args.domain or ""
        params = args.parameters or {}
        registry = self._domains.DOMAIN_REGISTRY
        if domain not in registry:
            available = sorted(registry.keys())
            return json.dumps({"error": f"Unknown domain: {domain}", "available": available}, indent=2)
        dom = registry[domain](params)
        analysis = dom.analyze()
        return json.dumps(analysis.to_dict(), indent=2, ensure_ascii=False, default=str)

    def _compare_domains(self, args: BourbakiInput) -> str:
        a = args.domain or ""
        b = args.domain_b or ""
        registry = self._domains.DOMAIN_REGISTRY
        for name in [a, b]:
            if name not in registry:
                return json.dumps({"error": f"Unknown domain: {name}", "available": sorted(registry.keys())}, indent=2)
        dom_a = registry[a](args.parameters or {})
        dom_b = registry[b](args.parameters_b or {})
        comparison = dom_a.compare_with(dom_b)
        return json.dumps(comparison, indent=2, ensure_ascii=False, default=str)

    def _build_conservation_field(self, args: BourbakiInput) -> str:
        eq_type = (args.equation_type or "").lower()
        params = args.parameters or {}
        from math_anything.structures.conservation_field import ConservationMatrixField

        field = ConservationMatrixField()
        builder_map = {
            "navier_stokes": lambda: field.build_from_navier_stokes(**params),
            "euler": lambda: field.build_from_euler_equations(**params),
            "schrodinger": lambda: field.build_from_schrodinger(**params),
            "maxwell": lambda: field.build_from_maxwell(**params),
            "elasticity": lambda: field.build_from_elasticity(**params),
            "mhd": lambda: field.build_from_mhd(**params),
            "heat": lambda: field.build_from_heat_equation(**params),
            "dirac": lambda: field.build_from_dirac(**params),
            "einstein_field": lambda: field.build_from_einstein_field(**params),
            "klein_gordon": lambda: field.build_from_klein_gordon(**params),
            "wave": lambda: field.build_from_wave_equation(**params),
            "kohn_sham": lambda: field.build_from_kohn_sham(**params),
            "boltzmann": lambda: field.build_from_boltzmann(**params),
            "shallow_water": lambda: field.build_from_shallow_water(**params),
            "schrodinger_nonlinear": lambda: field.build_from_schrodinger_nonlinear(**params),
            "vlasov": lambda: field.build_from_vlasov(**params),
            "hartree_fock": lambda: field.build_from_hartree_fock(**params),
            "advection_diffusion": lambda: field.build_from_advection_diffusion(**params),
        }
        if eq_type not in builder_map:
            return json.dumps({"error": f"Unknown equation type: {eq_type}", "supported": list(builder_map.keys())}, indent=2)
        builder_map[eq_type]()
        result = field.to_dict()
        result["equation_type"] = eq_type
        return json.dumps(result, indent=2, ensure_ascii=False, default=str)

    def _analyze_morphism_chain(self, args: BourbakiInput) -> str:
        domain = args.domain or ""
        registry = self._domains.DOMAIN_REGISTRY
        if domain not in registry:
            return json.dumps({"error": f"Unknown domain: {domain}"}, indent=2)
        dom = registry[domain](args.parameters or {})
        full_chain = dom.build_morphism_chain()
        if args.chain:
            chain_filter = {c.lower() for c in args.chain}
            full_chain = [step for step in full_chain if step.get("name", "").lower() in chain_filter]
        all_preserved = set()
        all_lost = set()
        all_introduced = set()
        for step in full_chain:
            for inv in step.get("invariants_kept", []):
                if inv:
                    all_preserved.add(inv)
            for inv in step.get("invariants_lost", []):
                if inv:
                    all_lost.add(inv)
            for inv in step.get("invariants_introduced", []):
                if inv:
                    all_introduced.add(inv)
        return json.dumps({
            "domain": domain,
            "chain_length": len(full_chain),
            "steps": full_chain,
            "summary": {
                "preserved_throughout": sorted(list(all_preserved - all_lost)),
                "lost_somewhere": sorted(list(all_lost)),
                "introduced_somewhere": sorted(list(all_introduced)),
            },
        }, indent=2, ensure_ascii=False, default=str)

    def _buckingham_pi(self, args: BourbakiInput) -> str:
        from math_anything.dimensional.scaling_group import BuckinghamPiEngine, PhysicalQuantity, BUILTIN_QUANTITIES
        engine = BuckinghamPiEngine()
        raw_vars = args.variables or []
        quantities = []
        for v in raw_vars:
            if isinstance(v, dict):
                # Full dict format: {name, symbol, dimensions, ...}
                q = PhysicalQuantity(
                    name=v.get("name", "unknown"),
                    symbol=v.get("symbol", v.get("name", "unknown")),
                    dimensions=v.get("dimensions", {}),
                    canonical_unit=v.get("unit", ""),
                    physical_role=v.get("role", "state_variable"),
                    description=v.get("description", ""),
                )
                quantities.append(q)
            elif isinstance(v, (list, tuple)) and len(v) >= 2:
                name, unit = v[0], v[1]
                # Try to match from BUILTIN_QUANTITIES by name
                if name in BUILTIN_QUANTITIES:
                    quantities.append(BUILTIN_QUANTITIES[name])
                else:
                    # Fallback: create with empty dimensions (user should pass dict for full control)
                    quantities.append(PhysicalQuantity(name=name, symbol=name, dimensions={}, canonical_unit=unit))
            else:
                raise ValueError(f"Invalid variable format: {v}")

        if not quantities:
            return json.dumps({"error": "No variables provided."}, indent=2)

        groups = engine.compute(quantities)
        result = {
            "pi_groups": [g.to_dict() for g in groups],
            "target": args.target,
            "variables": [q.name for q in quantities],
        }
        return json.dumps(result, indent=2, ensure_ascii=False, default=str)

    def _discover_equations(self, args: BourbakiInput) -> str:
        data = args.data or []
        target = args.target_variable or ""
        if not data or not target:
            return json.dumps({"error": "data and target_variable are required"}, indent=2)
        # Placeholder: real PSRN integration would go here
        return json.dumps({
            "status": "discover_equations requires PSRN engine",
            "data_points": len(data),
            "target": target,
            "max_complexity": args.max_complexity,
        }, indent=2, ensure_ascii=False)

    def _extract_engine(self, args: BourbakiInput) -> str:
        engine = args.engine or ""
        params = args.engine_params or {}
        if not engine:
            return json.dumps({"error": "engine is required"}, indent=2)
        try:
            result = self._ma.extract(engine, params)
            return json.dumps({
                "engine": result.engine,
                "success": result.success,
                "schema": result.schema,
                "warnings": result.warnings,
                "summary": result.summary(),
            }, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            return json.dumps({"error": str(e), "engine": engine}, indent=2)

    def _list_domains(self) -> str:
        registry = self._domains.DOMAIN_REGISTRY
        domains = []
        for name, cls in registry.items():
            try:
                dom = cls()
                domains.append({
                    "name": name,
                    "description": getattr(cls, "description", "No description"),
                    "equation_type": getattr(cls, "equation_type", "unknown"),
                    "morphism_chain_length": len(dom.build_morphism_chain()),
                })
            except Exception:
                domains.append({"name": name, "description": "(failed to instantiate)"})
        return json.dumps({"domains": domains, "total": len(domains)}, indent=2, ensure_ascii=False, default=str)

    def _list_equation_types(self) -> str:
        types = [
            "navier_stokes", "euler", "schrodinger", "maxwell", "elasticity",
            "mhd", "heat", "dirac", "einstein_field", "klein_gordon", "wave",
            "kohn_sham", "boltzmann", "shallow_water", "schrodinger_nonlinear",
            "vlasov", "hartree_fock", "advection_diffusion",
        ]
        return json.dumps({"equation_types": types, "total": len(types)}, indent=2)
