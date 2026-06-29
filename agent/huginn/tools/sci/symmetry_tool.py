"""Symmetry analysis tool — space groups, point groups, Wyckoff positions, k-paths.

Wraps pymatgen's SpacegroupAnalyzer (spglib backend) to give the agent
on-demand access to crystallographic symmetry data. Read-only and safe
to auto-execute.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool, ResearchPhase, ToolProfile
from huginn.types import ToolContext, ToolResult, ValidationResult

# Rough spin-only magnetic moment estimates for common transition metals
# and lanthanides, in Bohr magnetons.  These are typical values for common
# oxidation states — actual moments depend on the local environment.
_MAGNETIC_MOMENTS: dict[str, float] = {
    "Fe": 2.2,
    "Co": 1.7,
    "Ni": 0.6,
    "Mn": 3.5,
    "Cr": 3.0,
    "V": 1.0,
    "Gd": 7.0,
    "Tb": 6.0,
    "Dy": 5.0,
    "Ho": 4.0,
    "Er": 3.0,
    "Tm": 2.0,
    "Nd": 2.0,
    "Pr": 1.0,
    "Eu": 3.0,
    "Ce": 1.0,
}


class SymmetryToolInput(BaseModel):
    action: Literal[
        "analyze",
        "operations",
        "primitive",
        "conventional",
        "kpath",
        "site_symmetry",
        "irreducible",
        "subgroups",
        "wyckoff_split",
        "magnetic",
    ] = Field(...)
    file_path: str = Field(
        ..., description="Path to structure file (POSCAR, CIF, etc.)"
    )
    symprec: float = Field(
        default=0.01,
        description="Symmetry tolerance in Angstroms for spglib.",
    )
    angle_tolerance: float = Field(
        default=5.0,
        description="Angle tolerance in degrees for spglib.",
    )
    site_index: int | None = Field(
        default=None,
        description="Atom index (0-based) for site_symmetry action.",
    )
    kpath_density: int = Field(
        default=20,
        description="Points per segment for irreducible k-path generation.",
    )
    index: int = Field(
        default=2,
        description="Subgroup index for the subgroups action (e.g. 2 for "
        "index-2 subgroups). Use 0 or a negative value to list all detected "
        "subgroups regardless of index.",
    )
    subgroup_number: int | None = Field(
        default=None,
        description="Target subgroup space group number (1-230) for the "
        "wyckoff_split action.",
    )


class SymmetryTool(HuginnTool):
    """Analyze crystal symmetry: space groups, point groups, Wyckoff sites, k-paths."""

    name = "symmetry_tool"
    category = "sci"
    profile = ToolProfile(phases=frozenset({ResearchPhase.PLANNING, ResearchPhase.VALIDATION}))
    description = (
        "Analyze crystal symmetry: space group, point group, Wyckoff positions, "
        "symmetry operations, primitive/conventional cells, and high-symmetry "
        "k-point paths for band structure calculations."
    )
    input_schema = SymmetryToolInput

    def is_read_only(self, args: SymmetryToolInput) -> bool:
        return True

    async def validate_input(
        self, args: SymmetryToolInput, context: ToolContext | None = None
    ) -> ValidationResult:
        if args.action == "site_symmetry" and args.site_index is None:
            return ValidationResult(
                result=False,
                message="site_index is required for site_symmetry action.",
            )
        if args.action == "wyckoff_split" and args.subgroup_number is None:
            return ValidationResult(
                result=False,
                message="subgroup_number is required for wyckoff_split action.",
            )
        return ValidationResult(result=True)

    async def call(
        self, args: dict[str, Any], context: ToolContext | None = None
    ) -> ToolResult:
        input_data = SymmetryToolInput(**args)

        try:
            from pymatgen.core import Structure
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        except ImportError:
            return ToolResult(
                data=None,
                success=False,
                error="pymatgen is required for symmetry analysis. Install it with: pip install pymatgen",
            )

        try:
            structure = Structure.from_file(input_data.file_path)
        except Exception as e:
            return ToolResult(
                data=None,
                success=False,
                error=f"Failed to read structure from {input_data.file_path}: {e}",
            )

        try:
            sga = SpacegroupAnalyzer(
                structure,
                symprec=input_data.symprec,
                angle_tolerance=input_data.angle_tolerance,
            )
        except Exception as e:
            return ToolResult(
                data=None,
                success=False,
                error=f"SpacegroupAnalyzer failed: {e}. Try adjusting symprec.",
            )

        if input_data.action == "analyze":
            return self._analyze(sga, structure)
        elif input_data.action == "operations":
            return self._operations(sga)
        elif input_data.action == "primitive":
            return self._primitive(sga)
        elif input_data.action == "conventional":
            return self._conventional(sga)
        elif input_data.action == "kpath":
            return self._kpath(sga, structure, input_data.kpath_density)
        elif input_data.action == "site_symmetry":
            return self._site_symmetry(sga, structure, input_data.site_index)
        elif input_data.action == "irreducible":
            return self._irreducible(sga, input_data.kpath_density)
        elif input_data.action == "subgroups":
            return self._subgroups(sga, structure, input_data.index)
        elif input_data.action == "wyckoff_split":
            return self._wyckoff_split(sga, structure, input_data.subgroup_number)
        elif input_data.action == "magnetic":
            return self._magnetic(sga, structure)

        return ToolResult(
            data=None,
            success=False,
            error=f"Unknown action: {input_data.action}",
        )

    def _analyze(self, sga, structure) -> ToolResult:
        """Full symmetry summary: space group, point group, crystal system, Wyckoff."""
        try:
            spg_number = sga.get_space_group_number()
            spg_symbol = sga.get_space_group_symbol()
            pg_symbol = sga.get_point_group_symbol()
            crystal_system = sga.get_crystal_system()
            laue = sga.get_laue()
            wyckoff = sga.get_symmetry_dataset().get("wyckoffs", [])
            equivalents = sga.get_symmetry_dataset().get("equivalent_atoms", [])

            # Count unique Wyckoff sites
            unique_sites = sorted(set(zip(equivalents, wyckoff)))

            return ToolResult(
                data={
                    "formula": structure.composition.reduced_formula,
                    "space_group_number": spg_number,
                    "space_group_symbol": spg_symbol,
                    "point_group": pg_symbol,
                    "crystal_system": crystal_system,
                    "laue_class": laue,
                    "n_symmetry_operations": len(sga.get_symmetry_operations()),
                    "wyckoff_positions": wyckoff,
                    "equivalent_atoms": equivalents.tolist() if hasattr(equivalents, "tolist") else list(equivalents),
                    "unique_sites": [
                        {"atom_index": int(eq), "wyckoff": wy}
                        for eq, wy in unique_sites
                    ],
                    "n_atoms": len(structure),
                    "n_equivalent_atoms": len(set(equivalents.tolist() if hasattr(equivalents, "tolist") else equivalents)),
                },
                success=True,
            )
        except Exception as e:
            return ToolResult(data=None, success=False, error=f"Analysis failed: {e}")

    def _operations(self, sga) -> ToolResult:
        """List all symmetry operations (rotation matrices + translations)."""
        try:
            ops = sga.get_symmetry_operations()
            operations = []
            for i, op in enumerate(ops):
                operations.append({
                    "index": i,
                    "rotation": op.rotation_matrix.tolist(),
                    "translation": op.translation_vector.tolist(),
                })
            return ToolResult(
                data={
                    "n_operations": len(operations),
                    "operations": operations[:48],  # cap at 48 (max for cubic)
                    "truncated": len(operations) > 48,
                },
                success=True,
            )
        except Exception as e:
            return ToolResult(data=None, success=False, error=f"Failed to get operations: {e}")

    def _primitive(self, sga) -> ToolResult:
        """Get the primitive cell."""
        try:
            prim = sga.get_primitive_standard_structure()
            lattice = prim.lattice
            return ToolResult(
                data={
                    "n_atoms_primitive": len(prim),
                    "formula": prim.composition.reduced_formula,
                    "lattice": {
                        "a": lattice.a,
                        "b": lattice.b,
                        "c": lattice.c,
                        "alpha": lattice.alpha,
                        "beta": lattice.beta,
                        "gamma": lattice.gamma,
                    },
                    "matrix": lattice.matrix.tolist(),
                    "sites": [
                        {
                            "species": str(site.specie),
                            "frac_coords": site.frac_coords.tolist(),
                        }
                        for site in prim
                    ],
                },
                success=True,
            )
        except Exception as e:
            return ToolResult(data=None, success=False, error=f"Failed to get primitive cell: {e}")

    def _conventional(self, sga) -> ToolResult:
        """Get the conventional standard cell."""
        try:
            conv = sga.get_conventional_standard_structure()
            lattice = conv.lattice
            return ToolResult(
                data={
                    "n_atoms_conventional": len(conv),
                    "formula": conv.composition.reduced_formula,
                    "lattice": {
                        "a": lattice.a,
                        "b": lattice.b,
                        "c": lattice.c,
                        "alpha": lattice.alpha,
                        "beta": lattice.beta,
                        "gamma": lattice.gamma,
                    },
                    "matrix": lattice.matrix.tolist(),
                    "sites": [
                        {
                            "species": str(site.specie),
                            "frac_coords": site.frac_coords.tolist(),
                        }
                        for site in conv
                    ],
                },
                success=True,
            )
        except Exception as e:
            return ToolResult(data=None, success=False, error=f"Failed to get conventional cell: {e}")

    def _kpath(self, sga, structure, density: int) -> ToolResult:
        """Generate high-symmetry k-point path for band structure calculations."""
        try:
            # pymatgen's HighSymmPath gives the standard k-path per crystal system
            from pymatgen.symmetry.bandstructure import HighSymmPath

            hsp = HighSymmPath(structure)
            kpath = hsp.get_kpoints(
                line_density=density,
                coords_are_cartesian=False,
            )

            # get_kpoints returns (kpoints, labels) — labels may have empty strings
            kpoints, labels = kpath

            # Build segments from the path
            segments = []
            current_segment = []
            for i, (kp, label) in enumerate(zip(kpoints, labels)):
                current_segment.append({
                    "coords": kp.tolist(),
                    "label": label if label else None,
                })
                if label and i > 0 and len(current_segment) > 1:
                    segments.append(current_segment)
                    current_segment = [current_segment[-1]]

            if current_segment:
                segments.append(current_segment)

            return ToolResult(
                data={
                    "n_kpoints": len(kpoints),
                    "kpoints": [kp.tolist() for kp in kpoints],
                    "labels": labels,
                    "segments": segments,
                    "density": density,
                },
                success=True,
            )
        except Exception as e:
            return ToolResult(data=None, success=False, error=f"Failed to generate k-path: {e}")

    def _site_symmetry(self, sga, structure, site_index: int) -> ToolResult:
        """Get the site symmetry group for a specific atom."""
        try:
            if site_index < 0 or site_index >= len(structure):
                return ToolResult(
                    data=None,
                    success=False,
                    error=f"site_index {site_index} out of range (0-{len(structure) - 1})",
                )

            symm_ops = sga.get_site_symmetry_operations(site_index)
            wyckoff = sga.get_symmetry_dataset().get("wyckoffs", [])
            equivalents = sga.get_symmetry_dataset().get("equivalent_atoms", [])

            site = structure[site_index]
            return ToolResult(
                data={
                    "site_index": site_index,
                    "species": str(site.specie),
                    "frac_coords": site.frac_coords.tolist(),
                    "n_site_symmetry_ops": len(symm_ops),
                    "wyckoff_letter": wyckoff[site_index] if site_index < len(wyckoff) else None,
                    "equivalent_to": int(equivalents[site_index]) if site_index < len(equivalents) else None,
                    "site_symmetry_operations": [
                        {
                            "rotation": op.rotation_matrix.tolist(),
                            "translation": op.translation_vector.tolist(),
                        }
                        for op in symm_ops[:24]
                    ],
                },
                success=True,
            )
        except Exception as e:
            return ToolResult(data=None, success=False, error=f"Site symmetry failed: {e}")

    def _irreducible(self, sga, density: int) -> ToolResult:
        """Get irreducible reciprocal space points (for Brillouin zone sampling)."""
        try:
            # Use the primitive structure for irreducible k-points
            prim = sga.get_primitive_standard_structure()
            rec_lattice = prim.lattice.reciprocal_lattice

            # Generate a uniform grid and reduce by symmetry
            from pymatgen.symmetry.bandstructure import HighSymmPath

            hsp = HighSymmPath(prim)
            kpoints, labels = hsp.get_kpoints(
                line_density=density,
                coords_are_cartesian=False,
            )

            return ToolResult(
                data={
                    "n_irreducible_kpoints": len(kpoints),
                    "kpoints": [kp.tolist() for kp in kpoints],
                    "labels": labels,
                    "reciprocal_lattice": rec_lattice.matrix.tolist(),
                    "density": density,
                },
                success=True,
            )
        except Exception as e:
            return ToolResult(data=None, success=False, error=f"Irreducible k-points failed: {e}")

    # ── Subgroup / supergroup analysis ────────────────────────────────

    def _symmetry_reduced_variants(self, structure):
        """Generate perturbed copies to probe subgroup symmetries.

        Each variant breaks a different subset of symmetry operations
        (lattice equality, angular constraints, mirror planes, inversion)
        so that spglib reports the corresponding subgroup.
        """
        import numpy as np
        from pymatgen.core import Lattice, Structure

        lattice = structure.lattice
        species = [site.specie for site in structure]
        coords = structure.frac_coords
        variants = []

        # Lattice stretches — break cubic / tetragonal / hexagonal equality
        for axis, delta in [("a", 0.02), ("b", -0.015), ("c", 0.025)]:
            params = {
                "a": lattice.a, "b": lattice.b, "c": lattice.c,
                "alpha": lattice.alpha, "beta": lattice.beta, "gamma": lattice.gamma,
            }
            params[axis] *= (1 + delta)
            new_lat = Lattice.from_parameters(**params)
            variants.append((f"stretch_{axis}", Structure(new_lat, species, coords)))

        # Angular tilts — break higher-symmetry angle constraints
        for angle, delta in [("alpha", 1.5), ("beta", -2.0), ("gamma", 2.5)]:
            params = {
                "a": lattice.a, "b": lattice.b, "c": lattice.c,
                "alpha": lattice.alpha, "beta": lattice.beta, "gamma": lattice.gamma,
            }
            params[angle] += delta
            new_lat = Lattice.from_parameters(**params)
            variants.append((f"tilt_{angle}", Structure(new_lat, species, coords)))

        # Atomic displacements — break mirror / glide / inversion symmetry
        rng = np.random.default_rng(42)
        for trial in range(8):
            disp = rng.standard_normal((len(structure), 3)) * 0.03
            variants.append(
                (f"displace_{trial}", Structure(lattice, species, coords + disp))
            )

        return variants

    def _subgroups(self, sga, structure, index: int) -> ToolResult:
        """Find subgroups of the space group by testing symmetry reductions.

        spglib doesn't expose a direct subgroup API, so we perturb the
        structure to break specific symmetry elements and let spglib
        identify what space group remains.  The ratio of operation counts
        gives the subgroup index.
        """
        try:
            import numpy as np
            import spglib

            dataset = sga.get_symmetry_dataset()
            current_spg = dataset["number"]
            current_sym = dataset["international"]
            current_n_ops = len(dataset["rotations"])

            found: dict[int, dict] = {}
            for name, perturbed in self._symmetry_reduced_variants(structure):
                cell = (
                    np.asarray(perturbed.lattice.matrix, dtype=float),
                    np.asarray(perturbed.frac_coords, dtype=float),
                    np.array([site.specie.Z for site in perturbed]),
                )
                ds = spglib.get_symmetry_dataset(cell, symprec=0.05)
                if ds is None:
                    continue
                sg_num = ds["number"]
                if sg_num == current_spg:
                    continue
                n_ops = len(ds["rotations"])
                detected_index = current_n_ops / n_ops if n_ops else float("inf")

                # Filter by requested index (skip filter when index <= 0)
                if index > 0 and abs(detected_index - index) > 0.01:
                    continue

                if sg_num not in found:
                    found[sg_num] = {
                        "number": sg_num,
                        "hermann_mauguin": ds["international"],
                        "n_operations": n_ops,
                        "index": round(detected_index, 3),
                        "method": name,
                    }

            return ToolResult(
                data={
                    "current_spacegroup": current_spg,
                    "current_symbol": current_sym,
                    "current_n_operations": current_n_ops,
                    "requested_index": index,
                    "subgroups": sorted(found.values(), key=lambda x: x["number"]),
                },
                success=True,
            )
        except Exception as e:
            return ToolResult(data=None, success=False, error=f"Subgroup analysis failed: {e}")

    def _wyckoff_split(self, sga, structure, subgroup_number: int | None) -> ToolResult:
        """Show how Wyckoff positions split when going to a subgroup.

        Atom indices are preserved across the perturbation, so we can
        compare Wyckoff letters position-by-position to build the split
        mapping.
        """
        try:
            import numpy as np
            import spglib

            if subgroup_number is None:
                return ToolResult(
                    data=None,
                    success=False,
                    error="subgroup_number is required for wyckoff_split.",
                )

            orig_dataset = sga.get_symmetry_dataset()
            orig_spg = orig_dataset["number"]
            orig_sym = orig_dataset["international"]
            orig_wyckoffs = list(orig_dataset["wyckoffs"])

            # Search for a perturbation that reduces symmetry to the target subgroup
            target_dataset = None
            for _name, perturbed in self._symmetry_reduced_variants(structure):
                cell = (
                    np.asarray(perturbed.lattice.matrix, dtype=float),
                    np.asarray(perturbed.frac_coords, dtype=float),
                    np.array([site.specie.Z for site in perturbed]),
                )
                ds = spglib.get_symmetry_dataset(cell, symprec=0.05)
                if ds is not None and ds["number"] == subgroup_number:
                    target_dataset = ds
                    break

            if target_dataset is None:
                return ToolResult(
                    data=None,
                    success=False,
                    error=(
                        f"Could not reduce structure to space group "
                        f"{subgroup_number} via symmetry reduction. Try a "
                        f"different subgroup or adjust perturbations."
                    ),
                )

            new_wyckoffs = list(target_dataset["wyckoffs"])
            new_spg = target_dataset["number"]
            new_sym = target_dataset["international"]

            # Map each original Wyckoff letter to the set of letters it
            # becomes in the subgroup
            splits: dict[str, set[str]] = {}
            for orig_wy, new_wy in zip(orig_wyckoffs, new_wyckoffs):
                splits.setdefault(orig_wy, set()).add(new_wy)

            return ToolResult(
                data={
                    "original_spacegroup": orig_spg,
                    "original_symbol": orig_sym,
                    "subgroup_spacegroup": new_spg,
                    "subgroup_symbol": new_sym,
                    "wyckoff_splits": {
                        wy: sorted(new_wys) for wy, new_wys in splits.items()
                    },
                    "original_wyckoffs": orig_wyckoffs,
                    "new_wyckoffs": new_wyckoffs,
                },
                success=True,
            )
        except Exception as e:
            return ToolResult(data=None, success=False, error=f"Wyckoff split analysis failed: {e}")

    def _magnetic(self, sga, structure) -> ToolResult:
        """Check for potentially magnetic sites and estimate moments."""
        try:
            magnetic_sites = []
            for i, site in enumerate(structure):
                element = str(site.specie)
                if element in _MAGNETIC_MOMENTS:
                    moment = _MAGNETIC_MOMENTS[element]
                    if moment > 0:
                        magnetic_sites.append({
                            "index": i,
                            "element": element,
                            "estimated_moment_muB": moment,
                            "frac_coords": site.frac_coords.tolist(),
                        })

            return ToolResult(
                data={
                    "has_magnetic_sites": len(magnetic_sites) > 0,
                    "n_magnetic_sites": len(magnetic_sites),
                    "magnetic_sites": magnetic_sites,
                    "note": (
                        "Moment estimates are rough typical values in Bohr "
                        "magnetons. Actual moments depend on oxidation state, "
                        "coordination, and local environment."
                    ),
                },
                success=True,
            )
        except Exception as e:
            return ToolResult(data=None, success=False, error=f"Magnetic analysis failed: {e}")
