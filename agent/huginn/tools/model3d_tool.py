"""3D model generation and manipulation tool for materials science.

Builds real 3D meshes using trimesh: primitives, ball-and-stick models from
crystal structures, 4D trajectory animations, lattice wireframes, and mesh
operations (boolean, transform, subdivide, simplify). Exports to
STL / OBJ / GLTF / GLB / PLY for 3D printing, Blender, and web (three.js).

Dependencies (all lazy-loaded so the tool imports even without them installed):
    trimesh  >= 4.0
    pymatgen >= 2023.1
    numpy
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field, model_validator

from huginn.tools.base import HuginnTool, ResearchPhase, ToolProfile
from huginn.types import ToolContext, ToolResult, ValidationResult

logger = logging.getLogger(__name__)

# ── Covalent radii (Å) — compact table, enough for common elements ──
# Unknown species falls back to 1.5 Å.
_COVALENT_RADII: dict[str, float] = {
    "H": 0.31, "He": 0.28, "Li": 1.28, "Be": 0.96, "B": 0.84,
    "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57, "Ne": 0.58,
    "Na": 1.66, "Mg": 1.41, "Al": 1.21, "Si": 1.11, "P": 1.07,
    "S": 1.05, "Cl": 1.02, "Ar": 1.06, "K": 2.03, "Ca": 1.76,
    "Sc": 1.70, "Ti": 1.60, "V": 1.53, "Cr": 1.39, "Mn": 1.39,
    "Fe": 1.32, "Co": 1.26, "Ni": 1.24, "Cu": 1.32, "Zn": 1.22,
    "Ga": 1.22, "Ge": 1.20, "As": 1.19, "Se": 1.20, "Br": 1.20,
    "Kr": 1.16, "Rb": 2.20, "Sr": 1.95, "Y": 1.90, "Zr": 1.75,
    "Nb": 1.64, "Mo": 1.54, "Tc": 1.47, "Ru": 1.46, "Rh": 1.42,
    "Pd": 1.39, "Ag": 1.45, "Cd": 1.44, "In": 1.42, "Sn": 1.39,
    "Sb": 1.39, "Te": 1.38, "I": 1.39, "Xe": 1.40, "Cs": 2.44,
    "Ba": 2.15, "La": 2.07, "Ce": 2.04, "Pr": 2.03, "Nd": 2.01,
    "Pm": 1.99, "Sm": 1.98, "Eu": 1.98, "Gd": 1.96, "Tb": 1.94,
    "Dy": 1.92, "Ho": 1.92, "Er": 1.89, "Tm": 1.90, "Yb": 1.87,
    "Lu": 1.87, "Hf": 1.75, "Ta": 1.70, "W": 1.62, "Re": 1.51,
    "Os": 1.44, "Ir": 1.41, "Pt": 1.36, "Au": 1.36, "Hg": 1.32,
    "Tl": 1.45, "Pb": 1.46, "Bi": 1.48, "Po": 1.40, "At": 1.50,
    "Rn": 1.50,
}

# ── CPK colors (RGB 0-1) for ball-and-stick rendering ──
# Following Jmol conventions for the common elements.
_ELEMENT_COLORS: dict[str, tuple[float, float, float]] = {
    "H": (1.00, 1.00, 1.00), "He": (0.85, 1.00, 1.00),
    "Li": (0.80, 0.50, 0.25), "Be": (0.76, 1.00, 0.00),
    "B": (1.00, 0.71, 0.71), "C": (0.30, 0.30, 0.30),
    "N": (0.19, 0.31, 0.97), "O": (1.00, 0.05, 0.05),
    "F": (0.56, 0.87, 0.31), "Ne": (0.70, 0.89, 0.96),
    "Na": (0.67, 0.36, 0.95), "Mg": (0.54, 1.00, 0.00),
    "Al": (0.75, 0.65, 0.65), "Si": (0.94, 0.78, 0.63),
    "P": (1.00, 0.50, 0.00), "S": (1.00, 1.00, 0.19),
    "Cl": (0.12, 0.94, 0.12), "Ar": (0.50, 0.81, 0.89),
    "K": (0.56, 0.25, 0.83), "Ca": (0.24, 1.00, 0.00),
    "Sc": (0.90, 0.90, 0.90), "Ti": (0.75, 0.76, 0.78),
    "V": (0.65, 0.65, 0.67), "Cr": (0.54, 0.60, 0.78),
    "Mn": (0.61, 0.48, 0.78), "Fe": (0.88, 0.40, 0.20),
    "Co": (0.94, 0.56, 0.63), "Ni": (0.31, 0.82, 0.22),
    "Cu": (0.78, 0.50, 0.20), "Zn": (0.49, 0.50, 0.69),
    "Ga": (0.76, 0.56, 0.56), "Ge": (0.40, 0.56, 0.56),
    "As": (0.74, 0.50, 0.89), "Se": (1.00, 0.63, 0.00),
    "Br": (0.65, 0.16, 0.16), "Kr": (0.36, 0.72, 0.82),
    "Rb": (0.44, 0.18, 0.69), "Sr": (0.00, 1.00, 0.00),
    "Y": (0.58, 1.00, 1.00), "Zr": (0.58, 0.88, 0.88),
    "Nb": (0.45, 0.76, 0.79), "Mo": (0.33, 0.71, 0.71),
    "Tc": (0.23, 0.62, 0.62), "Ru": (0.14, 0.56, 0.56),
    "Rh": (0.04, 0.49, 0.55), "Pd": (0.00, 0.41, 0.52),
    "Ag": (0.75, 0.75, 0.75), "Cd": (0.72, 0.72, 0.82),
    "In": (0.65, 0.46, 0.45), "Sn": (0.40, 0.50, 0.50),
    "Sb": (0.62, 0.39, 0.71), "Te": (0.83, 0.48, 0.00),
    "I": (0.58, 0.00, 0.58), "Xe": (0.26, 0.62, 0.69),
    "Cs": (0.34, 0.09, 0.56), "Ba": (0.00, 0.79, 0.00),
    "La": (0.44, 0.83, 1.00), "Ce": (1.00, 1.00, 0.78),
    "Pr": (0.85, 1.00, 0.78), "Nd": (0.78, 1.00, 0.78),
    "Pm": (0.64, 1.00, 0.78), "Sm": (0.56, 1.00, 0.78),
    "Eu": (0.38, 1.00, 0.78), "Gd": (0.27, 1.00, 0.78),
    "Tb": (0.19, 1.00, 0.78), "Dy": (0.12, 1.00, 0.78),
    "Ho": (0.00, 1.00, 0.61), "Er": (0.00, 0.90, 0.46),
    "Tm": (0.00, 0.83, 0.32), "Yb": (0.00, 0.75, 0.22),
    "Lu": (0.00, 0.67, 0.14), "Hf": (0.30, 0.76, 1.00),
    "Ta": (0.49, 0.69, 0.81), "W": (0.13, 0.58, 0.84),
    "Re": (0.15, 0.49, 0.67), "Os": (0.15, 0.40, 0.59),
    "Ir": (0.09, 0.33, 0.53), "Pt": (0.81, 0.82, 0.87),
    "Au": (1.00, 0.82, 0.14), "Hg": (0.72, 0.72, 0.82),
    "Tl": (0.65, 0.33, 0.30), "Pb": (0.34, 0.35, 0.38),
    "Bi": (0.54, 0.31, 0.89), "Po": (0.70, 0.50, 0.83),
    "At": (0.42, 0.45, 0.45), "Rn": (0.26, 0.00, 0.40),
}

# bond cylinder radius relative to the smaller atom radius
_BOND_RADIUS_FACTOR = 0.15
# default bond cutoff multiplier — atoms are bonded if d < (r1 + r2) * mult
_BOND_CUTOFF_MULT = 1.3
# lattice wireframe edge thickness (Å)
_LATTICE_EDGE_RADIUS = 0.08


# ═══════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════

def _atomic_radius(species: str) -> float:
    """Look up covalent radius by element symbol, stripping oxidation etc."""
    # pymatgen Species/Element str might look like "Fe2+" or "Fe3+"
    sym = "".join(c for c in species if c.isalpha())
    return _COVALENT_RADII.get(sym, 1.5)


def _element_color(species: str) -> tuple[float, float, float]:
    """CPK color for an element symbol."""
    sym = "".join(c for c in species if c.isalpha())
    return _ELEMENT_COLORS.get(sym, (0.8, 0.4, 0.8))


def _to_rgba(rgb: tuple[float, float, float], alpha: float = 1.0) -> np.ndarray:
    """Convert an (r,g,b) tuple in 0-1 to a uint8 RGBA array."""
    return np.array(
        [int(r * 255) for r in rgb] + [int(alpha * 255)], dtype=np.uint8
    )


def _lattice_matrix(
    a: float, b: float, c: float,
    alpha: float, beta: float, gamma: float,
) -> np.ndarray:
    """Build a 3x3 lattice matrix from a, b, c (Å) and angles (degrees).

    Standard crystallographic convention: a along x, b in xy-plane.
    """
    alpha_r = math.radians(alpha)
    beta_r = math.radians(beta)
    gamma_r = math.radians(gamma)

    cos_a, cos_b, cos_g = math.cos(alpha_r), math.cos(beta_r), math.cos(gamma_r)
    sin_g = math.sin(gamma_r)

    vol = a * b * c * math.sqrt(
        1 - cos_a**2 - cos_b**2 - cos_g**2 + 2 * cos_a * cos_b * cos_g
    )

    vec_a = np.array([a, 0.0, 0.0])
    vec_b = np.array([b * cos_g, b * sin_g, 0.0])
    vec_c = np.array([
        c * cos_b,
        c * (cos_a - cos_b * cos_g) / sin_g,
        vol / (a * b * sin_g),
    ])
    return np.stack([vec_a, vec_b, vec_c], axis=0)


def _make_atom_sphere(
    center: np.ndarray,
    radius: float,
    color: tuple[float, float, float],
    subdivisions: int = 3,
):
    """Create a single atom as a colored UV sphere mesh."""
    import trimesh

    sphere = trimesh.creation.uv_sphere(radius=radius, count=[24, 16])
    sphere.apply_translation(center)
    sphere.visual.vertex_colors = np.tile(
        _to_rgba(color), (len(sphere.vertices), 1)
    )
    return sphere


def _make_bond_cylinder(
    p1: np.ndarray,
    p2: np.ndarray,
    radius: float,
    color: tuple[float, float, float],
    segments: int = 16,
):
    """Create a bond as a cylinder stretching from p1 to p2."""
    import trimesh

    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    vec = p2 - p1
    height = float(np.linalg.norm(vec))

    if height < 1e-8:
        return None

    cyl = trimesh.creation.cylinder(
        radius=radius, height=height, sections=segments
    )
    # cylinder is centered at origin along +z by default
    cyl.apply_translation((p1 + p2) / 2.0)

    # rotate so its local z-axis aligns with vec
    direction = vec / height
    z_axis = np.array([0.0, 0.0, 1.0])
    if not np.allclose(direction, z_axis, atol=1e-6):
        rot_axis = np.cross(z_axis, direction)
        rot_norm = float(np.linalg.norm(rot_axis))
        if rot_norm > 1e-8:
            rot_axis = rot_axis / rot_norm
            angle = float(math.acos(np.clip(np.dot(z_axis, direction), -1, 1)))
            # trimesh.transformations gives a proper 4x4 homogeneous matrix
            R = trimesh.transformations.rotation_matrix(angle, rot_axis)
            cyl.apply_transform(R)

    cyl.visual.vertex_colors = np.tile(
        _to_rgba(color), (len(cyl.vertices), 1)
    )
    return cyl


def _parse_structure(
    path: str | None = None,
    data: dict[str, Any] | None = None,
):
    """Parse a crystal structure from file path or inline dict via pymatgen.

    Inline dict should have keys: lattice (3x3 list), species (list[str]),
    coords (list of [x,y,z] fractional), coords_are_cartesian (bool, default False).
    """
    from pymatgen.core import Lattice, Structure

    if data is not None:
        lattice = np.array(data["lattice"], dtype=float)
        species = data["species"]
        coords = data["coords"]
        cart = data.get("coords_are_cartesian", False)
        struct = Structure(
            Lattice(lattice), species, coords,
            coords_are_cartesian=cart,
        )
        return struct

    if path is None:
        raise ValueError("Either path or data must be provided for structure")

    return Structure.from_file(path)


def _build_ball_and_stick(
    struct,
    atom_scale: float = 0.5,
    bond_cutoff: float | None = None,
    show_bonds: bool = True,
):
    """Build a trimesh Scene from a pymatgen Structure (ball-and-stick).

    Returns a trimesh.Scene with one geometry per atom + bond, or a
    concatenated Trimesh if Scene export isn't supported for the target format.
    """
    import trimesh

    meshes: list = []
    coords = np.asarray(struct.cart_coords)
    species = [str(sp) for sp in struct.species]

    # atoms
    radii = [_atomic_radius(s) * atom_scale for s in species]
    for i, (pos, sp) in enumerate(zip(coords, species)):
        sphere = _make_atom_sphere(pos, radii[i], _element_color(sp))
        meshes.append(sphere)

    # bonds — brute-force O(n^2), fine for unit cells (<200 atoms)
    if show_bonds and len(species) > 1:
        n = len(species)
        for i in range(n):
            ri = _atomic_radius(species[i])
            for j in range(i + 1, n):
                rj = _atomic_radius(species[j])
                d = float(np.linalg.norm(coords[j] - coords[i]))
                cutoff = bond_cutoff if bond_cutoff else (ri + rj) * _BOND_CUTOFF_MULT
                if d < cutoff and d > 0.5:
                    bond_r = min(radii[i], radii[j]) * _BOND_RADIUS_FACTOR
                    # bond color: blend of the two atom colors
                    ci = _element_color(species[i])
                    cj = _element_color(species[j])
                    bond_color = tuple((a + b) / 2 for a, b in zip(ci, cj))
                    cyl = _make_bond_cylinder(
                        coords[i], coords[j], bond_r, bond_color
                    )
                    if cyl is not None:
                        meshes.append(cyl)

    if not meshes:
        raise RuntimeError("No meshes generated from structure")

    scene = trimesh.Scene(meshes)
    # also stash a concatenated version for single-mesh export formats
    combined = trimesh.util.concatenate(meshes)
    scene.metadata["_combined"] = combined
    return scene


def _build_lattice_wireframe(
    lattice: np.ndarray,
    edge_radius: float = _LATTICE_EDGE_RADIUS,
    color: tuple[float, float, float] = (0.5, 0.5, 0.5),
):
    """Build thin cylinders along the 12 edges of a unit cell."""
    import trimesh

    corners = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1],
    ], dtype=float)
    cart = corners @ lattice

    edges = [
        (0, 1), (2, 4), (3, 5), (6, 7),  # a direction
        (0, 2), (1, 4), (3, 6), (5, 7),  # b direction
        (0, 3), (1, 5), (2, 6), (4, 7),  # c direction
    ]

    meshes = []
    for i, j in edges:
        cyl = _make_bond_cylinder(
            cart[i], cart[j], edge_radius, color, segments=8
        )
        if cyl is not None:
            meshes.append(cyl)

    if meshes:
        return trimesh.util.concatenate(meshes)
    return trimesh.Trimesh()


# ═══════════════════════════════════════════════════════════════════════
# Pydantic input model
# ═══════════════════════════════════════════════════════════════════════

class Model3DInput(BaseModel):
    """Input schema for the 3D model tool.

    All actions share this flat model; the ``action`` field determines which
    parameters are relevant. A model_validator enforces required fields per
    action so missing data is caught at schema time, not deep in call().
    """

    action: Literal[
        "create_primitive",
        "create_from_structure",
        "generate_4d_animation",
        "export_model",
        "modify_mesh",
        "create_lattice_model",
    ] = Field(..., description="Which 3D operation to perform")

    output_path: str = Field(
        ..., description="Output file path (or directory for 4d animation)"
    )

    # ── create_primitive ──
    shape: Literal[
        "box", "sphere", "cylinder", "cone",
        "torus", "capsule", "icosahedron",
    ] | None = Field(
        default=None, description="Primitive shape (create_primitive only)"
    )
    dimensions: list[float] | None = Field(
        default=None,
        description=(
            "Shape-specific sizes: box=[w,h,d], sphere=[r], "
            "cylinder/cone/capsule=[r,h], torus=[R,r], icosahedron=[r]"
        ),
    )
    resolution: int | None = Field(
        default=None,
        description="Mesh resolution: sphere subdivisions, cylinder sections, etc.",
    )
    position: list[float] | None = Field(
        default=None, description="Translate the mesh to this [x,y,z] position"
    )

    # ── create_from_structure / create_lattice_model ──
    structure_path: str | None = Field(
        default=None, description="Path to POSCAR/CIF/CONTCAR file"
    )
    structure_data: dict[str, Any] | None = Field(
        default=None,
        description="Inline structure dict: {lattice, species, coords, ...}",
    )
    atom_scale: float = Field(
        default=0.5, description="Scale factor for atom sphere radii"
    )
    bond_cutoff: float | None = Field(
        default=None,
        description="Max bond distance (Å); default = sum of covalent radii * 1.3",
    )
    show_bonds: bool = Field(default=True, description="Draw bond cylinders")

    # ── create_lattice_model ──
    lattice_params: dict[str, float] | None = Field(
        default=None,
        description="Lattice: {a, b, c, alpha, beta, gamma} in Å and degrees",
    )
    supercell: list[int] | None = Field(
        default=None, description="Supercell expansion [nx, ny, nz]"
    )
    species: list[str] | None = Field(
        default=None, description="Element symbols at lattice sites"
    )
    fractional_coords: list[list[float]] | None = Field(
        default=None, description="Fractional coords [[x,y,z], ...] for lattice atoms"
    )

    # ── generate_4d_animation ──
    trajectory_path: str | None = Field(
        default=None, description="Path to XDATCAR / trajectory file"
    )
    trajectory_data: list[dict[str, Any]] | None = Field(
        default=None,
        description="Inline trajectory: list of {structure_data or lattice+coords}",
    )
    trajectory_format: Literal["xdatcar", "md", "neb", "inline"] | None = Field(
        default=None, description="Trajectory source type"
    )
    num_frames: int | None = Field(
        default=None,
        description="Max frames to render (subsamples long trajectories)",
    )
    fps: int = Field(default=10, description="Animation FPS for metadata")

    # ── export_model ──
    input_path: str | None = Field(
        default=None, description="Existing mesh file to re-export"
    )
    export_format: Literal["stl", "obj", "gltf", "glb", "ply"] | None = Field(
        default=None, description="Target format for export_model"
    )

    # ── modify_mesh ──
    operation: Literal[
        "union", "intersection", "difference",
        "translate", "rotate", "scale",
        "subdivide", "simplify", "texture",
    ] | None = Field(
        default=None, description="Mesh operation (modify_mesh only)"
    )
    mesh_paths: list[str] | None = Field(
        default=None,
        description="Mesh files for boolean ops: [mesh_a, mesh_b]",
    )
    transform: list[float] | None = Field(
        default=None,
        description=(
            "translate=[x,y,z], rotate=[angle_deg, ax, ay, az], "
            "scale=[sx, sy, sz] or [s] for uniform"
        ),
    )
    face_count: int | None = Field(
        default=None, description="Target face count for simplify"
    )
    color: list[float] | None = Field(
        default=None, description="Color [r,g,b] 0-1 for texture operation"
    )

    @model_validator(mode="after")
    def _check_required_fields(self) -> "Model3DInput":
        """Validate that the right fields are present for the chosen action."""
        a = self.action

        if a == "create_primitive":
            if not self.shape:
                raise ValueError("create_primitive requires 'shape'")
            if self.dimensions is None:
                raise ValueError("create_primitive requires 'dimensions'")

        elif a == "create_from_structure":
            if not self.structure_path and not self.structure_data:
                raise ValueError(
                    "create_from_structure requires 'structure_path' or 'structure_data'"
                )

        elif a == "generate_4d_animation":
            if not self.trajectory_path and not self.trajectory_data:
                raise ValueError(
                    "generate_4d_animation requires 'trajectory_path' or 'trajectory_data'"
                )

        elif a == "export_model":
            if not self.input_path:
                raise ValueError("export_model requires 'input_path'")
            if not self.export_format:
                # infer from output extension
                ext = Path(self.output_path).suffix.lower().lstrip(".")
                if ext in ("stl", "obj", "gltf", "glb", "ply"):
                    self.export_format = ext  # type: ignore[assignment]
                else:
                    raise ValueError("export_model requires 'export_format'")

        elif a == "modify_mesh":
            if not self.operation:
                raise ValueError("modify_mesh requires 'operation'")
            if self.operation in ("union", "intersection", "difference"):
                if not self.mesh_paths or len(self.mesh_paths) < 2:
                    raise ValueError(
                        f"{self.operation} requires 'mesh_paths' with >= 2 files"
                    )
            else:
                if not self.input_path:
                    raise ValueError(
                        f"{self.operation} requires 'input_path'"
                    )
                if self.operation in ("translate", "rotate", "scale") and not self.transform:
                    raise ValueError(
                        f"{self.operation} requires 'transform'"
                    )

        elif a == "create_lattice_model":
            if not self.lattice_params and not self.structure_path and not self.structure_data:
                raise ValueError(
                    "create_lattice_model requires 'lattice_params', 'structure_path', or 'structure_data'"
                )

        return self


# ═══════════════════════════════════════════════════════════════════════
# Tool implementation
# ═══════════════════════════════════════════════════════════════════════

class Model3DTool(HuginnTool):
    """3D model generation and manipulation tool.

    Uses trimesh as the primary 3D library. Can create primitives (box, sphere,
    cylinder, cone, torus, capsule, icosahedron), build ball-and-stick models
    from crystal structures (POSCAR/CIF), generate 4D animations from molecular
    dynamics trajectories (XDATCAR/MD/NEB), perform mesh operations (boolean,
    transform, subdivide, simplify), and build periodic lattice wireframes.

    All outputs are real 3D mesh files (STL/OBJ/GLTF/GLB/PLY) suitable for
    3D printing, Blender, or web (three.js) rendering.
    """

    name = "model3d_tool"
    category = "cv"
    # trimesh ops are fast (seconds), but not trivially "light" — none keeps
    # them allowed by default without the light-path bookkeeping
    profile = ToolProfile(
        cost_tier="none",
        phases=frozenset({
            ResearchPhase.PLANNING,
            ResearchPhase.EXECUTION,
            ResearchPhase.VALIDATION,
            ResearchPhase.REPORTING,
        }),
    )
    description = (
        "Generate and manipulate 3D mesh models: primitives, ball-and-stick "
        "from crystal structures (POSCAR/CIF), 4D trajectory animations "
        "(XDATCAR/MD/NEB), lattice wireframes, and mesh operations (boolean, "
        "transform, subdivide, simplify). Exports STL/OBJ/GLTF/GLB/PLY."
    )
    input_schema = Model3DInput
    read_only = False  # creates files on disk

    # ── dispatch ──────────────────────────────────────────────────────

    async def call(self, args: Model3DInput, context: ToolContext) -> ToolResult:
        dispatch = {
            "create_primitive": self._create_primitive,
            "create_from_structure": self._create_from_structure,
            "generate_4d_animation": self._generate_4d_animation,
            "export_model": self._export_model,
            "modify_mesh": self._modify_mesh,
            "create_lattice_model": self._create_lattice_model,
        }
        handler = dispatch.get(args.action)
        if handler is None:
            return ToolResult(
                data=None, success=False,
                error=f"Unknown action: {args.action}",
            )
        try:
            return handler(args)
        except Exception as exc:
            logger.warning("model3d_tool %s failed: %s", args.action, exc, exc_info=True)
            return ToolResult(data=None, success=False, error=str(exc))

    # ── create_primitive ────────────────────────────────────────────

    def _create_primitive(self, args: Model3DInput) -> ToolResult:
        import trimesh

        dims = args.dimensions or []
        res = args.resolution
        shape = args.shape

        if shape == "box":
            ext = dims if len(dims) >= 3 else [1.0, 1.0, 1.0]
            mesh = trimesh.creation.box(extents=ext[:3])

        elif shape == "sphere":
            radius = dims[0] if dims else 1.0
            sub = res if res else 3
            mesh = trimesh.creation.icosphere(subdivisions=sub, radius=radius)

        elif shape == "cylinder":
            radius = dims[0] if len(dims) >= 1 else 0.5
            height = dims[1] if len(dims) >= 2 else 2.0
            sections = res if res else 32
            mesh = trimesh.creation.cylinder(
                radius=radius, height=height, sections=sections
            )

        elif shape == "cone":
            radius = dims[0] if len(dims) >= 1 else 0.5
            height = dims[1] if len(dims) >= 2 else 2.0
            sections = res if res else 32
            mesh = trimesh.creation.cone(
                radius=radius, height=height, sections=sections
            )

        elif shape == "torus":
            major = dims[0] if len(dims) >= 1 else 1.0
            minor = dims[1] if len(dims) >= 2 else 0.3
            maj_sec = res if res else 32
            min_sec = max(res or 16, 8)
            mesh = trimesh.creation.torus(
                major_radius=major, minor_radius=minor,
                major_sections=maj_sec, minor_sections=min_sec,
            )

        elif shape == "capsule":
            radius = dims[0] if len(dims) >= 1 else 0.5
            height = dims[1] if len(dims) >= 2 else 2.0
            segments = res if res else 32
            try:
                mesh = trimesh.creation.capsule(
                    height=height, radius=radius, segments=segments
                )
            except (AttributeError, TypeError):
                # older trimesh — build from cylinder + two spheres
                cyl = trimesh.creation.cylinder(
                    radius=radius, height=height, sections=segments
                )
                top = trimesh.creation.uv_sphere(radius=radius)
                top.apply_translation([0, 0, height / 2])
                bot = trimesh.creation.uv_sphere(radius=radius)
                bot.apply_translation([0, 0, -height / 2])
                mesh = trimesh.util.concatenate([cyl, top, bot])

        elif shape == "icosahedron":
            radius = dims[0] if dims else 1.0
            # subdivisions=0 gives the base 20-face icosahedron
            mesh = trimesh.creation.icosphere(subdivisions=0, radius=radius)

        else:
            return ToolResult(
                data=None, success=False,
                error=f"Unsupported shape: {shape}",
            )

        # apply position offset
        if args.position:
            mesh.apply_translation(args.position[:3])

        Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
        mesh.export(args.output_path)

        return self._mesh_result(mesh, args.output_path, extra={
            "shape": shape,
            "dimensions": dims,
        })

    # ── create_from_structure ───────────────────────────────────────

    def _create_from_structure(self, args: Model3DInput) -> ToolResult:
        struct = _parse_structure(args.structure_path, args.structure_data)
        scene = _build_ball_and_stick(
            struct,
            atom_scale=args.atom_scale,
            bond_cutoff=args.bond_cutoff,
            show_bonds=args.show_bonds,
        )

        Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
        fmt = Path(args.output_path).suffix.lower().lstrip(".")

        # GLTF/GLB works with Scene (preserves per-geometry colors);
        # STL/OBJ/PLY need a single mesh — fall back to concatenated version
        if fmt in ("gltf", "glb"):
            scene_without_meta = type(scene)(scene.geometry)
            scene_without_meta.export(args.output_path)
            combined = scene.metadata.get("_combined")
            stats = self._mesh_stats(combined) if combined is not None else {}
        else:
            combined = scene.metadata.get("_combined")
            if combined is None:
                # fallback: concatenate all geometries manually
                combined = self._concat_scene(scene)
            combined.export(args.output_path)
            stats = self._mesh_stats(combined)

        return ToolResult(
            data={
                "output_path": args.output_path,
                "format": fmt,
                "formula": struct.composition.reduced_formula,
                "num_atoms": len(struct),
                "num_bonds": max(0, len(scene.geometry) - len(struct)),
                **stats,
            },
            success=True,
            side_effects=[args.output_path],
        )

    # ── generate_4d_animation ────────────────────────────────────────

    def _generate_4d_animation(self, args: Model3DInput) -> ToolResult:
        out_dir = Path(args.output_path)
        out_dir.mkdir(parents=True, exist_ok=True)

        frames = self._load_trajectory(args)
        if not frames:
            return ToolResult(
                data=None, success=False,
                error="No frames found in trajectory",
            )

        # subsample if too many frames
        if args.num_frames and len(frames) > args.num_frames:
            indices = np.linspace(0, len(frames) - 1, args.num_frames, dtype=int)
            frames = [frames[i] for i in indices]

        frame_files: list[dict[str, Any]] = []
        for idx, struct in enumerate(frames):
            scene = _build_ball_and_stick(
                struct, atom_scale=args.atom_scale,
                bond_cutoff=args.bond_cutoff, show_bonds=args.show_bonds,
            )
            frame_path = str(out_dir / f"frame_{idx:04d}.obj")
            combined = scene.metadata.get("_combined") or self._concat_scene(scene)
            combined.export(frame_path)

            frame_files.append({
                "index": idx,
                "file": frame_path,
                "formula": struct.composition.reduced_formula,
                "num_atoms": len(struct),
            })

        # write animation metadata
        meta = {
            "type": "trajectory_animation",
            "format": args.trajectory_format or "inline",
            "num_frames": len(frame_files),
            "fps": args.fps,
            "frames": frame_files,
            "created": datetime.now(timezone.utc).isoformat(),
        }
        meta_path = str(out_dir / "animation_meta.json")
        Path(meta_path).write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )

        return ToolResult(
            data={
                "output_dir": str(out_dir),
                "metadata_file": meta_path,
                "num_frames": len(frame_files),
                "fps": args.fps,
                "frame_files": [f["file"] for f in frame_files],
            },
            success=True,
            side_effects=[str(out_dir)],
        )

    def _load_trajectory(self, args: Model3DInput) -> list:
        """Load trajectory frames as a list of pymatgen Structures."""
        if args.trajectory_data is not None:
            frames = []
            for entry in args.trajectory_data:
                struct = _parse_structure(data=entry)
                frames.append(struct)
            return frames

        path = args.trajectory_path
        if not path:
            return []

        fmt = args.trajectory_format
        # auto-detect from filename
        if fmt is None:
            fname = Path(path).name.upper()
            if "XDATCAR" in fname:
                fmt = "xdatcar"
            else:
                fmt = "md"

        if fmt == "xdatcar":
            from pymatgen.io.vasp.outputs import Xdatcar
            xdat = Xdatcar(path)
            return list(xdat.structures)

        if fmt == "neb":
            # NEB: each directory frame*/CONTCAR or a list of CONTCAR files
            base = Path(path)
            if base.is_dir():
                contcar_files = sorted(base.glob("*/CONTCAR")) + \
                                sorted(base.glob("*/CONTCAR.*"))
                structs = []
                for cf in contcar_files:
                    structs.append(_parse_structure(path=str(cf)))
                return structs
            # single file with multiple structures — try as XDATCAR
            from pymatgen.io.vasp.outputs import Xdatcar
            return list(Xdatcar(path).structures)

        if fmt == "md":
            # try XDATCAR format first, then fall back to multi-structure CIF
            try:
                from pymatgen.io.vasp.outputs import Xdatcar
                return list(Xdatcar(path).structures)
            except Exception:
                from pymatgen.core import Structure
                # generic: try reading as a sequence — if it's a single
                # structure, just return [struct]
                return [Structure.from_file(path)]

        return []

    # ── export_model ────────────────────────────────────────────────

    def _export_model(self, args: Model3DInput) -> ToolResult:
        import trimesh

        if not Path(args.input_path).exists():
            return ToolResult(
                data=None, success=False,
                error=f"Input mesh not found: {args.input_path}",
            )

        loaded = trimesh.load(args.input_path, force='mesh')
        Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
        loaded.export(args.output_path)

        return ToolResult(
            data={
                "input_path": args.input_path,
                "output_path": args.output_path,
                "format": args.export_format,
                **self._mesh_stats(loaded),
            },
            success=True,
            side_effects=[args.output_path],
        )

    # ── modify_mesh ─────────────────────────────────────────────────

    def _modify_mesh(self, args: Model3DInput) -> ToolResult:
        import trimesh

        op = args.operation

        # ── boolean operations ──
        if op in ("union", "intersection", "difference"):
            meshes = [trimesh.load(p, force='mesh') for p in args.mesh_paths]
            try:
                from trimesh import boolean
                if op == "union":
                    result = boolean.union(meshes)
                elif op == "intersection":
                    result = boolean.intersection(meshes)
                else:
                    result = boolean.difference(meshes)
            except Exception as exc:
                if op == "union":
                    # union without a boolean backend = just concatenate
                    logger.info("boolean backend unavailable, falling back to concatenate: %s", exc)
                    result = trimesh.util.concatenate(meshes)
                else:
                    return ToolResult(
                        data=None, success=False,
                        error=(
                            f"{op} requires a boolean backend (blender/manifold3d). "
                            f"Install: pip install manifold3d  —  {exc}"
                        ),
                    )

            Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
            result.export(args.output_path)
            return self._mesh_result(result, args.output_path, extra={
                "operation": op, "inputs": args.mesh_paths,
            })

        # ── single-mesh operations ──
        mesh = trimesh.load(args.input_path, force='mesh')

        if op == "translate":
            mesh.apply_translation(args.transform[:3])
        elif op == "rotate":
            angle_rad = math.radians(args.transform[0])
            axis = np.array(args.transform[1:4], dtype=float)
            axis = axis / (np.linalg.norm(axis) or 1.0)
            R = trimesh.transformations.rotation_matrix(angle_rad, axis)
            mesh.apply_transform(R)
        elif op == "scale":
            t = args.transform
            if len(t) == 1:
                mesh.apply_scale(t[0])
            else:
                mesh.apply_scale(t[:3])
        elif op == "subdivide":
            mesh = mesh.subdivide()
        elif op == "simplify":
            target = args.face_count or max(len(mesh.faces) // 2, 100)
            # trimesh 4.x: face_count is a keyword arg, not positional.
            # Positional would be interpreted as 'percent' (0-1 ratio).
            try:
                mesh = mesh.simplify_quadric_decimation(face_count=target)
            except TypeError:
                # older trimesh — positional face_count
                mesh = mesh.simplify_quadric_decimation(target)
        elif op == "texture":
            if args.color:
                rgba = _to_rgba(tuple(args.color[:3]))
                mesh.visual.vertex_colors = np.tile(rgba, (len(mesh.vertices), 1))
        else:
            return ToolResult(
                data=None, success=False,
                error=f"Unsupported operation: {op}",
            )

        Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
        mesh.export(args.output_path)
        return self._mesh_result(mesh, args.output_path, extra={
            "operation": op, "input": args.input_path,
        })

    # ── create_lattice_model ────────────────────────────────────────

    def _create_lattice_model(self, args: Model3DInput) -> ToolResult:
        import trimesh

        # get lattice matrix
        if args.structure_path or args.structure_data:
            struct = _parse_structure(args.structure_path, args.structure_data)
            lattice = np.array(struct.lattice.matrix)
            species = [str(sp) for sp in struct.species]
            frac_coords = struct.frac_coords.tolist()
        else:
            lp = args.lattice_params or {}
            a = lp.get("a", 5.0)
            b = lp.get("b", 5.0)
            c = lp.get("c", 5.0)
            alpha = lp.get("alpha", 90.0)
            beta = lp.get("beta", 90.0)
            gamma = lp.get("gamma", 90.0)
            lattice = _lattice_matrix(a, b, c, alpha, beta, gamma)
            species = args.species or []
            frac_coords = args.fractional_coords or []

        # supercell expansion
        sc = args.supercell or [1, 1, 1]
        nx, ny, nz = sc[0], sc[1], sc[2]

        meshes: list = []

        # unit cell wireframes for every supercell tile
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    offset = (
                        ix * lattice[0] + iy * lattice[1] + iz * lattice[2]
                    )
                    wf = _build_lattice_wireframe(
                        lattice, edge_radius=_LATTICE_EDGE_RADIUS
                    )
                    wf.apply_translation(offset)
                    meshes.append(wf)

        # atoms at all supercell positions
        for sp, fc in zip(species, frac_coords):
            r = _atomic_radius(sp) * args.atom_scale
            color = _element_color(sp)
            for ix in range(nx):
                for iy in range(ny):
                    for iz in range(nz):
                        cart = (
                            (fc[0] + ix) * lattice[0]
                            + (fc[1] + iy) * lattice[1]
                            + (fc[2] + iz) * lattice[2]
                        )
                        sphere = _make_atom_sphere(cart, r, color)
                        meshes.append(sphere)

        if not meshes:
            return ToolResult(
                data=None, success=False,
                error="No meshes generated — check lattice params and species",
            )

        combined = trimesh.util.concatenate(meshes)
        Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
        combined.export(args.output_path)

        return self._mesh_result(combined, args.output_path, extra={
            "lattice_params": args.lattice_params,
            "supercell": sc,
            "num_unit_cells": nx * ny * nz,
            "num_atoms_total": len(species) * nx * ny * nz,
        })

    # ── utility methods ─────────────────────────────────────────────

    def _mesh_stats(self, mesh) -> dict[str, Any]:
        """Return useful stats about a trimesh object."""
        try:
            import trimesh
            if isinstance(mesh, trimesh.Scene):
                return {"type": "scene", "geometries": len(mesh.geometry)}
            return {
                "vertices": int(len(mesh.vertices)),
                "faces": int(len(mesh.faces)),
                "volume": float(mesh.volume) if mesh.is_watertight else None,
                "watertight": bool(mesh.is_watertight),
                "bounds": mesh.bounds.tolist(),
            }
        except Exception:
            return {}

    def _mesh_result(
        self, mesh, output_path: str, extra: dict | None = None
    ) -> ToolResult:
        """Build a standard success ToolResult for a mesh export."""
        data = {
            "output_path": output_path,
            "exists": Path(output_path).exists(),
            "file_size_bytes": Path(output_path).stat().st_size
            if Path(output_path).exists() else 0,
            **self._mesh_stats(mesh),
        }
        if extra:
            data.update(extra)
        return ToolResult(
            data=data, success=True, side_effects=[output_path]
        )

    def _concat_scene(self, scene):
        """Concatenate all geometries in a Scene into a single Trimesh."""
        import trimesh
        meshes = list(scene.geometry.values())
        if not meshes:
            return trimesh.Trimesh()
        return trimesh.util.concatenate(meshes)
