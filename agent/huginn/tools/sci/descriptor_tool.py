"""Descriptor tool — compute numerical descriptors for materials.

Provides composition-based features and, when optional dependencies are
available, structure-based descriptors (SOAP, MBTR, ACSF, Coulomb matrix,
matminer stats) for machine-learning screening workflows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool, ResearchPhase, ToolProfile
from huginn.types import ToolContext, ToolResult


class DescriptorInput(BaseModel):
    action: Literal[
        "composition",
        "soap",
        "matminer",
        "mbtr",
        "acsf",
        "coulomb_matrix",
        "ibp",
    ] = Field(...)
    formula: str | None = Field(default=None, description="Chemical formula")
    structure_file: str | None = Field(
        default=None, description="Path to structure file"
    )
    structure: dict[str, Any] | None = Field(
        default=None,
        description="Structure dict with 'lattice' and 'sites' keys",
    )
    descriptor_type: Literal[
        "composition_basic",
        "soap",
        "matminer",
        "mbtr",
        "acsf",
        "coulomb_matrix",
        "ibp",
    ] | None = Field(default="composition_basic")

    # MBTR parameters
    k: int = Field(default=2, ge=1, le=3, description="MBTR k-term (1, 2, or 3)")
    n_grid: int = Field(default=100, description="MBTR grid points per axis")
    geometry: Literal["cosine", "inverse_distance", "distance"] = Field(
        default="cosine", description="MBTR geometry function"
    )
    weighting: Literal["exp", "unity"] = Field(
        default="exp", description="MBTR weighting function"
    )

    # ACSF parameters
    rcut: float = Field(default=6.0, description="ACSF cutoff radius")
    g2_params: list[list[float]] | None = Field(
        default=None, description="ACSF G2 parameters as [eta, Rs] pairs"
    )
    g4_params: list[list[float]] | None = Field(
        default=None, description="ACSF G4 parameters as [eta, zeta, lambda] triples"
    )

    # Coulomb matrix parameters
    n_atoms_max: int = Field(default=50, description="Max atoms for Coulomb matrix padding")
    permutation: Literal["sorted_l2", "random", "eigenspectrum", "none"] = Field(
        default="sorted_l2", description="Coulomb matrix permutation mode"
    )

    # IBP parameters
    data: list[list[float]] | None = Field(
        default=None,
        description="Input data for IBP (N samples x D observed features)",
    )
    alpha: float = Field(
        default=1.0, gt=0, description="IBP concentration parameter"
    )
    n_iterations: int = Field(
        default=100, ge=1, description="Number of Gibbs sampling iterations"
    )
    n_init_features: int = Field(
        default=5, ge=1, description="Initial number of latent features"
    )
    beta: float = Field(
        default=1.0,
        gt=0,
        description="Prior probability parameter for feature activation",
    )
    seed: int | None = Field(
        default=None, description="Random seed for reproducibility"
    )


class DescriptorOutput(BaseModel):
    action: str
    descriptor_type: str
    count: int
    features: dict[str, Any]
    warnings: list[str] = []


class DescriptorTool(HuginnTool):
    """Compute material descriptors for ML screening."""

    name = "descriptor_tool"
    category = "sci"
    profile = ToolProfile(phases=frozenset({ResearchPhase.HYPOTHESIS, ResearchPhase.PLANNING}))
    description = (
        "Compute composition and structure descriptors for machine-learning "
        "materials screening."
    )
    input_schema = DescriptorInput
    output_schema = DescriptorOutput
    read_only = True

    def is_read_only(self, args: DescriptorInput) -> bool:
        return True

    async def call(self, args: DescriptorInput, context: ToolContext) -> ToolResult:
        try:
            if args.action == "composition":
                features, warnings = self._composition_features(args)
            elif args.action == "soap":
                features, warnings = self._soap_descriptor(args)
            elif args.action == "matminer":
                features, warnings = self._matminer_features(args)
            elif args.action == "mbtr":
                features, warnings = self._mbtr_descriptor(args)
            elif args.action == "acsf":
                features, warnings = self._acsf_descriptor(args)
            elif args.action == "coulomb_matrix":
                features, warnings = self._coulomb_matrix_descriptor(args)
            elif args.action == "ibp":
                features, warnings = self._ibp_features(args)
            else:  # pragma: no cover
                raise ValueError(f"Unknown action: {args.action}")

            output = DescriptorOutput(
                action=args.action,
                descriptor_type=args.descriptor_type or args.action,
                count=len(features),
                features=features,
                warnings=warnings,
            )
            return ToolResult(data=output.model_dump(exclude_none=True))
        except Exception as exc:  # pragma: no cover
            return ToolResult(data=None, success=False, error=str(exc))

    # ------------------------------------------------------------------ #
    # composition
    # ------------------------------------------------------------------ #

    def _composition_features(
        self, args: DescriptorInput
    ) -> tuple[dict[str, Any], list[str]]:
        formula = args.formula
        if not formula and args.structure_file:
            path = Path(args.structure_file)
            if not path.exists():
                raise FileNotFoundError(f"Structure file not found: {path}")
            try:
                from pymatgen.core import Structure

                structure = Structure.from_file(str(path))
                formula = structure.formula.replace(" ", "")
            except Exception as exc:
                raise RuntimeError(f"Could not read structure file: {exc}") from exc
        if not formula:
            raise ValueError("Provide formula or structure_file")

        try:
            from pymatgen.core import Composition

            comp = Composition(formula)
            total_atoms = float(comp.num_atoms)
            atomic_fractions = {
                str(el): float(frac) / total_atoms for el, frac in comp.items()
            }
            features = {
                "formula": comp.formula.replace(" ", ""),
                "num_elements": len(comp),
                "total_atoms": total_atoms,
                "atomic_fractions": atomic_fractions,
                "avg_atomic_mass": float(
                    sum(el.atomic_mass * frac for el, frac in comp.items())
                    / total_atoms
                ),
                "avg_electronegativity": _safe_average(
                    [el.X for el in comp.elements],
                    [float(frac) for frac in comp.values()],
                ),
            }
            return features, []
        except ImportError:
            return self._fallback_composition_features(formula)

    def _fallback_composition_features(
        self, formula: str
    ) -> tuple[dict[str, Any], list[str]]:
        warnings = [
            "pymatgen not installed; using lightweight fallback composition features"
        ]
        try:
            comp = _parse_formula(formula)
        except Exception as exc:
            raise ValueError(f"Could not parse formula '{formula}': {exc}") from exc
        total = sum(comp.values())
        fractions = {k: v / total for k, v in comp.items()}
        return {
            "formula": formula,
            "num_elements": len(comp),
            "total_atoms": float(total),
            "atomic_fractions": fractions,
            "avg_atomic_mass": None,
            "avg_electronegativity": None,
        }, warnings

    # ------------------------------------------------------------------ #
    # soap
    # ------------------------------------------------------------------ #

    def _soap_descriptor(
        self, args: DescriptorInput
    ) -> tuple[dict[str, Any], list[str]]:
        if not args.structure_file:
            raise ValueError("soap descriptor requires structure_file")
        path = Path(args.structure_file)
        if not path.exists():
            raise FileNotFoundError(f"Structure file not found: {path}")
        try:
            from ase.io import read
            from dscribe.descriptors import SOAP

            atoms = read(str(path))
            species = sorted(set(atoms.get_chemical_symbols()))
            soap = SOAP(
                species=species,
                r_cut=6.0,
                n_max=8,
                l_max=6,
                average="outer",
                sparse=False,
            )
            vec = soap.create(atoms)
            return {
                "structure_file": str(path),
                "descriptor": "soap",
                "dim": int(vec.size),
                "mean": float(vec.mean()),
                "std": float(vec.std()),
            }, []
        except ImportError as exc:
            raise RuntimeError(
                "SOAP descriptor requires optional dependencies: dscribe, ase"
            ) from exc

    # ------------------------------------------------------------------ #
    # matminer — Magpie-style element property statistics
    # ------------------------------------------------------------------ #

    def _matminer_features(
        self, args: DescriptorInput
    ) -> tuple[dict[str, Any], list[str]]:
        formula = args.formula
        if not formula:
            raise ValueError("matminer action requires formula")
        try:
            from matminer.featurizers.composition import ElementProperty
            from pymatgen.core import Composition
        except ImportError as exc:
            raise RuntimeError(
                "matminer action requires the matminer package. "
                "Install it with: pip install matminer"
            ) from exc

        comp = Composition(formula)
        properties = [
            "Number",
            "AtomicWeight",
            "Electronegativity",
            "Column",
            "Row",
            "CovalentRadius",
        ]
        stats = ["mean", "minimum", "maximum", "avg_dev", "mode"]

        ep = ElementProperty(
            data_source="magpie",
            features=properties,
            stats=stats,
        )

        values = ep.featurize(comp)
        labels = ep.feature_labels()

        # Group by property so callers get {prop: {mean, min, max, std, ...}}
        grouped: dict[str, dict[str, float | None]] = {}
        for label, val in zip(labels, values):
            # Labels look like "MagpieData mean Number"
            parts = label.rsplit(" ", 2)
            if len(parts) != 3:
                continue
            _, stat, prop = parts
            stat_key = {"minimum": "min", "maximum": "max"}.get(stat, stat)
            grouped.setdefault(prop, {})[stat_key] = (
                float(val) if val is not None and not np.isnan(val) else None
            )

        # std isn't in the Magpie stat set, so compute it from element values
        for prop in properties:
            entry = grouped.setdefault(prop, {})
            vals: list[float] = []
            weights: list[float] = []
            for el, frac in comp.items():
                v = _magpie_value(el, prop)
                if v is not None:
                    vals.append(v)
                    weights.append(float(frac))
            if vals:
                arr = np.asarray(vals, dtype=float)
                w = np.asarray(weights, dtype=float)
                w = w / w.sum()
                mean = float(np.sum(arr * w))
                var = float(np.sum(w * (arr - mean) ** 2))
                entry["std"] = float(np.sqrt(var))
            else:
                entry["std"] = None

        return {
            "formula": comp.formula.replace(" ", ""),
            "properties": grouped,
        }, []

    # ------------------------------------------------------------------ #
    # mbtr — Many-Body Tensor Representation
    # ------------------------------------------------------------------ #

    def _mbtr_descriptor(
        self, args: DescriptorInput
    ) -> tuple[dict[str, Any], list[str]]:
        try:
            from dscribe.descriptors import MBTR
        except ImportError as exc:
            raise RuntimeError(
                "mbtr action requires the dscribe package. "
                "Install it with: pip install dscribe"
            ) from exc

        atoms = self._get_atoms(args)
        species = sorted(set(atoms.get_chemical_symbols()))

        k = args.k
        n = args.n_grid

        grid: dict[str, dict[str, float]] = {}
        if k >= 1:
            grid["k1"] = {
                "min": 1,
                "max": max(len(species), 2),
                "n": n,
                "sigma": 0.1,
            }
        if k >= 2:
            grid["k2"] = {"min": -1.0, "max": 1.0, "n": n, "sigma": 0.1}
        if k >= 3:
            grid["k3"] = {"min": -1.0, "max": 1.0, "n": n, "sigma": 0.1}

        if args.weighting == "exp":
            weighting: dict[str, Any] = {
                "function": "exp",
                "scale": 0.5,
                "cutoff": 1e-3,
            }
        else:
            weighting = {"function": "unity"}

        mbtr = MBTR(
            species=species,
            geometry={"function": args.geometry},
            grid=grid,
            weighting=weighting,
            periodic=True,
        )

        vec = np.asarray(mbtr.create(atoms)).flatten()
        return {
            "descriptor": "mbtr",
            "k": k,
            "n_grid": n,
            "geometry": args.geometry,
            "weighting": args.weighting,
            "species": species,
            "dim": int(vec.size),
            "vector": vec.tolist(),
            "mean": float(vec.mean()),
            "std": float(vec.std()),
        }, []

    # ------------------------------------------------------------------ #
    # acsf — Atom-Centered Symmetry Functions
    # ------------------------------------------------------------------ #

    def _acsf_descriptor(
        self, args: DescriptorInput
    ) -> tuple[dict[str, Any], list[str]]:
        try:
            from dscribe.descriptors import ACSF
        except ImportError as exc:
            raise RuntimeError(
                "acsf action requires the dscribe package. "
                "Install it with: pip install dscribe"
            ) from exc

        atoms = self._get_atoms(args)
        species = sorted(set(atoms.get_chemical_symbols()))

        g2 = args.g2_params or [[0.05, 0.0], [0.05, 2.0], [0.05, 4.0]]
        g4 = args.g4_params or [
            [0.005, 1.0, 1.0],
            [0.005, 1.0, -1.0],
            [0.005, 4.0, 1.0],
        ]

        acsf = ACSF(
            species=species,
            rcut=args.rcut,
            g2_params=g2,
            g4_params=g4,
        )

        vec = np.asarray(acsf.create(atoms))
        return {
            "descriptor": "acsf",
            "rcut": args.rcut,
            "g2_params": g2,
            "g4_params": g4,
            "species": species,
            "shape": list(vec.shape),
            "descriptor_array": vec.tolist(),
            "mean": float(vec.mean()),
            "std": float(vec.std()),
        }, []

    # ------------------------------------------------------------------ #
    # coulomb_matrix
    # ------------------------------------------------------------------ #

    def _coulomb_matrix_descriptor(
        self, args: DescriptorInput
    ) -> tuple[dict[str, Any], list[str]]:
        try:
            from dscribe.descriptors import CoulombMatrix
        except ImportError as exc:
            raise RuntimeError(
                "coulomb_matrix action requires the dscribe package. "
                "Install it with: pip install dscribe"
            ) from exc

        atoms = self._get_atoms(args)
        species = sorted(set(atoms.get_chemical_symbols()))

        cm = CoulombMatrix(
            n_atoms_max=args.n_atoms_max,
            permutation=args.permutation,
            sparse=False,
        )

        vec = np.asarray(cm.create(atoms)).flatten()
        return {
            "descriptor": "coulomb_matrix",
            "n_atoms_max": args.n_atoms_max,
            "permutation": args.permutation,
            "species": species,
            "num_atoms": len(atoms),
            "dim": int(vec.size),
            "vector": vec.tolist(),
            "mean": float(vec.mean()),
            "std": float(vec.std()),
        }, []

    # ------------------------------------------------------------------ #
    # ibp - Indian Buffet Process for nonparametric feature discovery
    # ------------------------------------------------------------------ #

    def _ibp_features(
        self, args: DescriptorInput
    ) -> tuple[dict[str, Any], list[str]]:
        if not args.data:
            raise ValueError("ibp action requires data (list of lists)")

        X = np.asarray(args.data, dtype=float)
        if X.ndim != 2:
            raise ValueError("data must be a 2D array (N samples x D features)")

        N, D = X.shape
        if N < 2:
            raise ValueError("ibp requires at least 2 samples")

        # Standardize columns so the fixed noise/prior variances behave
        # sensibly regardless of the raw feature scales.
        col_mean = X.mean(axis=0)
        col_std = X.std(axis=0)
        col_std[col_std == 0] = 1.0
        X_norm = (X - col_mean) / col_std

        rng = np.random.default_rng(args.seed)

        alpha = float(args.alpha)
        n_iter = int(args.n_iterations)
        K = max(1, int(args.n_init_features))
        beta = float(args.beta)

        # Noise and weight-prior variances. Kept fixed - learning them is
        # possible but adds complexity without much benefit for discovery.
        sigma_x2 = 1.0
        sigma_w2 = 1.0

        # Gamma(1, 1) prior on alpha - standard weakly-informative choice
        a_prior, b_prior = 1.0, 1.0

        Z = (rng.random((N, K)) < 0.5).astype(float)
        W = rng.normal(0.0, np.sqrt(sigma_w2), size=(K, D))

        alpha_trace: list[float] = []
        # Harmonic number H_N shows up in the alpha posterior
        H_N = float(np.sum(1.0 / np.arange(1, N + 1)))

        for _ in range(n_iter):
            # -- Block sample W | Z via conjugate Gaussian --
            ZtZ = Z.T @ Z
            Sigma_inv = np.eye(K) / sigma_w2 + ZtZ / sigma_x2
            Sigma_inv += 1e-6 * np.eye(K)  # jitter to dodge singular matrices
            Sigma = np.linalg.inv(Sigma_inv)
            L = np.linalg.cholesky(Sigma)
            Mu = Sigma @ (Z.T @ X_norm) / sigma_x2
            W = Mu + L @ rng.standard_normal(size=(K, D))

            # -- Sample Z | W, one row at a time (vectorised over features) --
            pred = Z @ W
            for n in range(N):
                # Residual with feature k removed: r_{n,-k} = x_n - pred_n + z_nk w_k
                r = X_norm[n] - pred[n] + Z[n][:, None] * W  # (K, D)

                log_lik1 = -0.5 * np.sum((r - W) ** 2, axis=1) / sigma_x2
                log_lik0 = -0.5 * np.sum(r ** 2, axis=1) / sigma_x2

                # IBP prior with Beta(beta, beta) smoothing on the activation prob
                m_minus = Z.sum(axis=0) - Z[n]
                p_prior1 = (m_minus + beta) / (N - 1 + 2 * beta)
                p_prior1 = np.clip(p_prior1, 1e-10, 1 - 1e-10)
                log_prior1 = np.log(p_prior1)
                log_prior0 = np.log(1.0 - p_prior1)

                log_diff = (log_lik1 + log_prior1) - (log_lik0 + log_prior0)
                log_diff = np.clip(log_diff, -500, 500)
                p1 = 1.0 / (1.0 + np.exp(-log_diff))

                Z[n] = (rng.random(K) < p1).astype(float)
                pred[n] = Z[n] @ W

            # -- Drop features that no sample owns --
            non_empty = Z.sum(axis=0) > 0
            if not non_empty.all():
                Z = Z[:, non_empty]
                W = W[non_empty]
                K = Z.shape[1]
                if K == 0:
                    # Re-seed one feature so the sampler doesn't collapse
                    K = 1
                    Z = np.zeros((N, K))
                    Z[0, 0] = 1.0
                    W = rng.normal(0.0, np.sqrt(sigma_w2), size=(K, D))

            # -- Propose new features via the IBP Chinese restaurant process --
            # Cap K so a runaway alpha can't blow up memory. The Z sweep
            # above is what really decides which features survive.
            max_features = 2 * N
            for n in range(N):
                if max_features <= K:
                    break
                n_new = rng.poisson(alpha / N)
                if n_new <= 0:
                    continue
                # Don't overshoot the cap
                n_new = min(n_new, max_features - K)
                new_Z = np.zeros((N, n_new))
                new_Z[n] = 1.0
                # Sample weights from the prior, not the singleton posterior.
                # Posterior-sampled weights fit the residual too well and the
                # features never get pruned, which causes K to explode.
                new_W = rng.normal(
                    0.0, np.sqrt(sigma_w2), size=(n_new, D)
                )
                Z = np.hstack([Z, new_Z])
                W = np.vstack([W, new_W])
                K = Z.shape[1]

            # -- Update alpha from its Gamma posterior --
            K_plus = int((Z.sum(axis=0) > 0).sum())
            alpha = rng.gamma(a_prior + K_plus, 1.0 / (b_prior + H_N))
            alpha_trace.append(float(alpha))

        # Strip any features that died out in the final sweep
        non_empty = Z.sum(axis=0) > 0
        Z = Z[:, non_empty]
        W = W[non_empty]
        K = Z.shape[1]

        pred = Z @ W
        recon_error = float(np.mean((X_norm - pred) ** 2))
        feature_importance = Z.sum(axis=0).astype(int).tolist()

        # For each latent feature, list the top-3 observed features by |weight|
        # so the user has a rough idea what the feature encodes.
        feature_interpretation: list[list[dict[str, Any]]] = []
        for k in range(K):
            top_idx = np.argsort(np.abs(W[k]))[-3:][::-1]
            feature_interpretation.append(
                [
                    {"observed_feature_index": int(i), "weight": float(W[k, i])}
                    for i in top_idx
                ]
            )

        return {
            "n_samples": N,
            "n_observed_features": D,
            "n_features": int(K),
            "Z": Z.astype(int).tolist(),
            "W": W.tolist(),
            "feature_importance": feature_importance,
            "feature_interpretation": feature_interpretation,
            "reconstruction_error": recon_error,
            "alpha_trace": alpha_trace,
            "alpha_final": float(alpha),
        }, []

    # ------------------------------------------------------------------ #
    # helpers for building ASE Atoms from the various inputs
    # ------------------------------------------------------------------ #

    def _get_atoms(self, args: DescriptorInput):
        """Build an ASE Atoms object from whichever input the caller provided."""
        from ase import Atoms

        if args.structure:
            return self._structure_dict_to_atoms(args.structure)

        if args.structure_file:
            path = Path(args.structure_file)
            if not path.exists():
                raise FileNotFoundError(f"Structure file not found: {path}")
            from ase.io import read

            return read(str(path))

        if args.formula:
            from ase.build import bulk

            try:
                return bulk(args.formula)
            except Exception:
                # bulk() only covers simple cases — fall back to a bare Atoms
                return Atoms(args.formula)

        raise ValueError("Provide structure, structure_file, or formula")

    @staticmethod
    def _structure_dict_to_atoms(structure: dict[str, Any]):
        """Convert a {lattice, sites} dict into ASE Atoms.

        Accepts either a pymatgen-style dict (lattice params + species list)
        or a simpler variant with a 3x3 lattice matrix and element strings.
        """
        from ase import Atoms
        from ase.cell import Cell

        lattice = structure.get("lattice")
        sites = structure.get("sites", [])
        if not sites:
            raise ValueError("Structure dict contains no sites")

        symbols: list[str] = []
        positions: list[list[float]] = []
        use_fractional = False
        for site in sites:
            sp = site.get("species", site.get("element"))
            if isinstance(sp, list):
                # pymatgen format: [{"element": "Fe", "occu": 1.0}]
                symbols.append(sp[0]["element"])
            elif isinstance(sp, dict):
                symbols.append(sp.get("element", sp.get("symbol", "")))
            else:
                symbols.append(str(sp))

            if "xyz" in site:
                positions.append(list(site["xyz"]))
            elif "abc" in site:
                positions.append(list(site["abc"]))
                use_fractional = True
            else:
                positions.append([0.0, 0.0, 0.0])

        positions_arr = np.asarray(positions, dtype=float)

        if isinstance(lattice, dict):
            cell = Cell.fromcellpar([
                lattice.get("a", 1.0),
                lattice.get("b", 1.0),
                lattice.get("c", 1.0),
                lattice.get("alpha", 90.0),
                lattice.get("beta", 90.0),
                lattice.get("gamma", 90.0),
            ])
            if use_fractional:
                positions_arr = positions_arr @ np.asarray(cell)
            return Atoms(symbols=symbols, positions=positions_arr, cell=cell, pbc=True)

        if isinstance(lattice, list):
            cell = np.asarray(lattice, dtype=float)
            if use_fractional:
                positions_arr = positions_arr @ cell
            return Atoms(symbols=symbols, positions=positions_arr, cell=cell, pbc=True)

        raise ValueError("Invalid lattice format in structure dict")


def _magpie_value(el, prop: str) -> float | None:
    """Look up a Magpie-style element property from a pymatgen Element."""
    try:
        if prop == "Number":
            return float(el.Z)
        if prop == "AtomicWeight":
            return float(el.atomic_mass)
        if prop == "Electronegativity":
            return float(el.X) if el.X is not None else None
        if prop == "Column":
            return float(el.group)
        if prop == "Row":
            return float(el.row)
        if prop == "CovalentRadius":
            return float(el.covalent_radius) if el.covalent_radius else None
    except (TypeError, AttributeError):
        return None
    return None


def _safe_average(values: list[float], weights: list[float]) -> float | None:
    try:
        return float(np.average(values, weights=weights))
    except Exception:
        return None


def _parse_formula(formula: str) -> dict[str, int]:
    """Very small formula parser for fallback mode (e.g. 'H2O', 'CaCO3')."""
    import re

    tokens = re.findall(r"([A-Z][a-z]*)(\d*\.?\d*)", formula)
    if not tokens:
        raise ValueError(f"Invalid formula: {formula}")
    result: dict[str, int] = {}
    for elem, count in tokens:
        result[elem] = result.get(elem, 0) + (int(count) if count else 1)
    return result
