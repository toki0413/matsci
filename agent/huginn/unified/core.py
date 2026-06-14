"""Core mathematical structures for the unified framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import sympy as sp


class FieldKind(str, Enum):
    """Kind of physical field."""

    SCALAR = "scalar"
    VECTOR = "vector"
    TENSOR = "tensor"
    PHASE_SPACE = "phase_space"


class DomainType(str, Enum):
    """Type of computational domain."""

    CONTINUUM = "continuum"
    LATTICE = "lattice"
    PARTICLES = "particles"
    MESH = "mesh"


class VariationalPrinciple(str, Enum):
    """Principle from which governing equations are derived."""

    STATIONARY = "stationary"           # δE = 0
    MINIMUM = "minimum"                 # min E
    MAXIMUM = "maximum"                 # max E
    HAMILTONIAN = "hamiltonian"         # Hamilton's equations / least action
    SELF_CONSISTENT = "self_consistent" # Kohn-Sham style
    CONSERVATION = "conservation"       # balance laws (mass, momentum, energy)
    DISSIPATIVE = "dissipative"         # entropy production / Onsager


@dataclass
class Field:
    """A physical field or state variable."""

    name: str
    kind: FieldKind
    symbols: list[sp.Symbol]
    domain: Domain
    units: str = ""
    description: str = ""

    def expr(self) -> sp.Expr:
        """Return the primary symbol/expression for the field."""
        if len(self.symbols) == 1:
            return self.symbols[0]
        return sp.Matrix(self.symbols)


@dataclass
class Domain:
    """Spatial or abstract domain on which fields live."""

    name: str
    kind: DomainType
    dimension: int
    bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    coordinates: list[sp.Symbol] = field(default_factory=list)

    @classmethod
    def continuum_1d(cls, x: sp.Symbol | None = None, bounds: tuple[float, float] = (0.0, 1.0)) -> Domain:
        x = x or sp.Symbol("x")
        return cls(name="continuum_1d", kind=DomainType.CONTINUUM, dimension=1, bounds={str(x): bounds}, coordinates=[x])

    @classmethod
    def continuum_2d(
        cls,
        x: sp.Symbol | None = None,
        y: sp.Symbol | None = None,
        bounds: tuple[tuple[float, float], tuple[float, float]] = ((0.0, 1.0), (0.0, 1.0)),
    ) -> Domain:
        x = x or sp.Symbol("x")
        y = y or sp.Symbol("y")
        return cls(
            name="continuum_2d",
            kind=DomainType.CONTINUUM,
            dimension=2,
            bounds={str(x): bounds[0], str(y): bounds[1]},
            coordinates=[x, y],
        )

    @classmethod
    def particles(cls, n: int, dim: int = 3) -> Domain:
        return cls(name=f"particles_{n}_{dim}d", kind=DomainType.PARTICLES, dimension=n * dim)


@dataclass
class EnergyFunctional:
    """Symbolic energy or action functional."""

    name: str
    expression: sp.Expr
    variables: list[sp.Symbol]
    parameters: dict[str, Any] = field(default_factory=dict)
    description: str = ""


@dataclass
class Operator:
    """A differential/integral operator."""

    name: str
    symbol: sp.Symbol
    expression: sp.Expr


@dataclass
class ConstitutiveModel:
    """Constitutive relation or approximation (XC, force field, elastic law, EOS)."""

    name: str
    expression: sp.Expr
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class UnifiedProblem:
    """A scientific computing problem expressed in the unified language."""

    name: str
    fields: dict[str, Field]
    principle: VariationalPrinciple
    energy: EnergyFunctional | None = None
    operators: dict[str, Operator] = field(default_factory=dict)
    domain: Domain | None = None
    constitutive: ConstitutiveModel | None = None
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dict."""
        return {
            "name": self.name,
            "description": self.description,
            "principle": self.principle.value,
            "domain": {
                "name": self.domain.name if self.domain else None,
                "kind": self.domain.kind.value if self.domain else None,
                "dimension": self.domain.dimension if self.domain else None,
            },
            "fields": {k: {"kind": v.kind.value, "description": v.description} for k, v in self.fields.items()},
            "energy": {"name": self.energy.name, "expression": str(self.energy.expression)} if self.energy else None,
            "constitutive": {"name": self.constitutive.name} if self.constitutive else None,
        }
