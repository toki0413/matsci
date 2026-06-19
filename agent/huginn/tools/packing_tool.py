"""Unified packing and preview tool for molecules, particles, and future geometries.

The tool performs deterministic placement (random rotation + translation with a
distance constraint) so the LLM only needs to choose parameters.  It can also
export a native Packmol input file and, if Packmol is installed, delegate to it.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field, model_validator

from huginn.security.sandbox import SandboxConfig, SandboxExecutor
from huginn.tools.base import HuginnTool
from huginn.types import ToolContext, ToolResult

_ELEMENT_COLORS: dict[str, str] = {
    "H": "#FFFFFF",
    "C": "#909090",
    "N": "#3050F8",
    "O": "#FF0D0D",
    "F": "#90E050",
    "P": "#FF8000",
    "S": "#FFFF30",
    "Cl": "#1FF01F",
    "Fe": "#E06633",
    "Cu": "#C78033",
    "Zn": "#7D80B0",
    "Si": "#F0C8A0",
    "Au": "#FFD123",
}
_DEFAULT_COLOR = "#FF69B4"


class ComponentSpec(BaseModel):
    source: str = Field(
        ..., description="Molecule/particle source (XYZ path, SMILES, name, or JSON)"
    )
    count: int = Field(default=1, ge=1)
    source_type: Literal["auto", "xyz", "smiles", "name", "particle"] = Field(
        default="auto"
    )


class PackingToolInput(BaseModel):
    action: Literal["pack", "generate", "preview"] = Field(default="pack")
    mode: Literal["molecules", "particles"] = Field(default="molecules")
    components: list[ComponentSpec] = Field(default_factory=list)
    box: list[float] = Field(
        default=[20.0, 20.0, 20.0], description="Box dimensions in Angstrom"
    )
    tolerance: float = Field(
        default=2.0, gt=0, description="Minimum inter-object distance"
    )
    max_trials: int = Field(default=1000, ge=1)
    seed: int | None = Field(default=None)
    output_format: Literal["xyz", "pdb", "lammps-data"] = Field(default="xyz")
    output_prefix: str = Field(default="packed")
    visualize: bool = Field(default=True, description="Render a 3D PNG preview")
    write_packmol_input: bool = Field(
        default=False, description="Also write a Packmol input file (molecules mode)"
    )
    structure_file: str | None = Field(
        default=None, description="File to preview (XYZ/PDB)"
    )
    working_dir: str | None = Field(default=None)

    @model_validator(mode="after")
    def _check_box(self) -> PackingToolInput:
        if len(self.box) != 3 or any(v <= 0 for v in self.box):
            raise ValueError("box must be a positive 3-vector")
        return self


class PackingTool(HuginnTool):
    """Pack molecules or particles into a box and render previews."""

    name = "packing_tool"
    description = (
        "Pack molecules or particles into a simulation box and render 3D previews. "
        "Outputs XYZ/PDB/LAMMPS-data and can export a Packmol input file."
    )
    input_schema = PackingToolInput

    def __init__(
        self,
        packmol_executable: str | None = None,
        sandbox: SandboxExecutor | None = None,
    ) -> None:
        super().__init__()
        self.packmol_executable = packmol_executable or self._find_packmol()
        self.sandbox = sandbox or SandboxExecutor()

    def _find_packmol(self) -> str | None:
        env_path = os.environ.get("PACKMOL_EXECUTABLE")
        if env_path and Path(env_path).exists():
            return env_path
        return shutil.which("packmol")

    def call(
        self, args: dict[str, Any], context: ToolContext | None = None
    ) -> ToolResult:
        input_data = PackingToolInput(**args)
        work_dir = (
            Path(input_data.working_dir) if input_data.working_dir else Path.cwd()
        )
        work_dir.mkdir(parents=True, exist_ok=True)

        try:
            if input_data.action == "preview":
                return self._preview(input_data, work_dir)
            if input_data.action == "generate":
                return self._generate_packmol_input(input_data, work_dir)
            return self._pack(input_data, work_dir)
        except Exception as e:
            return ToolResult(
                data=None, success=False, error=f"Packing tool failed: {e}"
            )

    # ------------------------------------------------------------------
    # Source loading
    # ------------------------------------------------------------------

    def _detect_source_type(self, spec: ComponentSpec) -> str:
        if spec.source_type != "auto":
            return spec.source_type
        source = spec.source.strip()
        if source.startswith("{"):
            return "particle"
        if source.endswith(".xyz"):
            return "xyz"
        if any(c in source for c in "=#[](.)@/\\"):
            return "smiles"
        return "name"

    def _load_component(
        self,
        spec: ComponentSpec,
        work_dir: Path,
    ) -> list[tuple[str, np.ndarray]]:
        source_type = self._detect_source_type(spec)
        if source_type == "xyz":
            return self._load_xyz(self._resolve_path(spec.source, work_dir))
        if source_type == "smiles":
            return self._smiles_to_geometry(spec.source)
        if source_type == "name":
            return self._name_to_geometry(spec.source)
        if source_type == "particle":
            return self._particle_to_geometry(spec.source)
        raise ValueError(f"Unsupported source type: {source_type}")

    def _resolve_path(self, source: str, work_dir: Path) -> Path:
        path = Path(source)
        if not path.is_absolute():
            path = work_dir / path
        return path

    def _load_xyz(self, path: Path) -> list[tuple[str, np.ndarray]]:
        if not path.exists():
            raise FileNotFoundError(f"XYZ file not found: {path}")
        lines = path.read_text(encoding="utf-8").splitlines()
        if len(lines) < 3:
            raise ValueError(f"Invalid XYZ file: {path}")
        try:
            n_atoms = int(lines[0].strip())
        except ValueError as exc:
            raise ValueError(f"Invalid atom count in XYZ file: {path}") from exc
        atoms: list[tuple[str, np.ndarray]] = []
        for line in lines[2 : 2 + n_atoms]:
            parts = line.split()
            if len(parts) < 4:
                continue
            symbol = parts[0]
            coord = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            atoms.append((symbol, coord))
        return atoms

    def _smiles_to_geometry(self, smiles: str) -> list[tuple[str, np.ndarray]]:
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
        except ImportError as exc:
            raise RuntimeError(
                "SMILES input requires RDKit. Install it or provide an XYZ file."
            ) from exc
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Could not parse SMILES: {smiles}")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=0xF00D)
        AllChem.MMFFOptimizeMolecule(mol)
        conf = mol.GetConformer()
        atoms: list[tuple[str, np.ndarray]] = []
        for i in range(mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            atoms.append(
                (mol.GetAtomWithIdx(i).GetSymbol(), np.array([pos.x, pos.y, pos.z]))
            )
        return atoms

    def _name_to_geometry(self, name: str) -> list[tuple[str, np.ndarray]]:
        builtin = self._builtin_molecule(name)
        try:
            from ase.build import molecule
            from ase.io import write

            atoms = molecule(name)
            tmp_xyz = Path(f"__tmp_{name}.xyz")
            write(tmp_xyz, atoms)
            geo = self._load_xyz(tmp_xyz)
            tmp_xyz.unlink(missing_ok=True)
            return geo
        except ImportError:
            if builtin is not None:
                return builtin
            raise RuntimeError(
                f"Named molecule '{name}' requires ASE or RDKit. "
                "Install ase/rdkit or provide an XYZ file."
            ) from None
        except Exception:
            # ASE may raise KeyError for unknown names; fall back to built-ins.
            if builtin is not None:
                return builtin
            raise

    def _builtin_molecule(self, name: str) -> list[tuple[str, np.ndarray]] | None:
        lowered = name.lower()
        if lowered in ("water", "h2o"):
            return [
                ("O", np.array([0.0, 0.0, 0.0])),
                ("H", np.array([0.96, 0.0, 0.0])),
                ("H", np.array([-0.24, 0.93, 0.0])),
            ]
        if lowered in ("methane", "ch4"):
            return [
                ("C", np.array([0.0, 0.0, 0.0])),
                ("H", np.array([0.63, 0.63, 0.63])),
                ("H", np.array([-0.63, -0.63, 0.63])),
                ("H", np.array([-0.63, 0.63, -0.63])),
                ("H", np.array([0.63, -0.63, -0.63])),
            ]
        return None

    def _particle_to_geometry(self, source: str) -> list[tuple[str, np.ndarray]]:
        try:
            params = json.loads(source)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Particle source must be JSON: {source}") from exc
        shape = params.get("shape", "sphere")
        symbol = params.get("symbol", "P")
        n_points = int(params.get("n_points", 50))
        if shape == "sphere":
            radius = float(params.get("radius", 1.0))
            return [(symbol, p) for p in self._fibonacci_sphere(n_points, radius)]
        if shape == "cube":
            size = float(params.get("size", 2.0))
            return [(symbol, p) for p in self._cube_surface(n_points, size)]
        raise ValueError(f"Unsupported particle shape: {shape}")

    @staticmethod
    def _fibonacci_sphere(n: int, radius: float) -> np.ndarray:
        indices = np.arange(n, dtype=float) + 0.5
        phi = np.arccos(1 - 2 * indices / n)
        theta = np.pi * (1 + 5**0.5) * indices
        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)
        return np.column_stack([x, y, z])

    @staticmethod
    def _cube_surface(n: int, size: float) -> np.ndarray:
        # Uniformly sample points on a cube surface.
        half = size / 2.0
        points_per_face = max(1, n // 6)
        points: list[np.ndarray] = []
        faces = [
            ("x", half),
            ("x", -half),
            ("y", half),
            ("y", -half),
            ("z", half),
            ("z", -half),
        ]
        for axis, value in faces:
            u = np.linspace(-half, half, int(np.sqrt(points_per_face)))
            v = np.linspace(-half, half, int(np.sqrt(points_per_face)))
            for ui in u:
                for vi in v:
                    if axis == "x":
                        points.append(np.array([value, ui, vi]))
                    elif axis == "y":
                        points.append(np.array([ui, value, vi]))
                    else:
                        points.append(np.array([ui, vi, value]))
        return np.array(points) if points else np.array([[0.0, 0.0, 0.0]])

    # ------------------------------------------------------------------
    # Packing core
    # ------------------------------------------------------------------

    @staticmethod
    def _random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
        m = rng.standard_normal((3, 3))
        q, r = np.linalg.qr(m)
        d = np.diag(np.sign(np.diag(r)))
        rot = q @ d
        if np.linalg.det(rot) < 0:
            rot[:, 0] *= -1
        return rot

    def _check_distance(
        self,
        candidate: np.ndarray,
        placed: list[tuple[str, np.ndarray]],
        tolerance: float,
    ) -> bool:
        if not placed:
            return True
        placed_coords = np.stack([c for _, c in placed])
        diff = placed_coords[:, np.newaxis, :] - candidate[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        return bool(np.min(dists) >= tolerance)

    def _pack(self, args: PackingToolInput, work_dir: Path) -> ToolResult:
        rng = np.random.default_rng(args.seed)
        box = np.asarray(args.box, dtype=float)

        placed: list[tuple[str, np.ndarray]] = []
        placed_object_ids: list[int] = []
        object_metadata: dict[int, dict[str, Any]] = {}
        component_summaries: list[dict[str, Any]] = []
        object_id = 0

        for comp_index, spec in enumerate(args.components):
            base = self._load_component(spec, work_dir)
            base_coords = np.stack([c for _, c in base])
            base_symbols = [s for s, _ in base]
            source_type = self._detect_source_type(spec)
            particle_meta = (
                self._particle_meta(spec.source) if source_type == "particle" else {}
            )
            success_count = 0
            fail_count = 0

            for _ in range(spec.count):
                accepted = False
                for _trial in range(args.max_trials):
                    rot = self._random_rotation_matrix(rng)
                    rotated = (rot @ base_coords.T).T
                    centroid = rotated.mean(axis=0)
                    centered = rotated - centroid
                    low = centered.min(axis=0)
                    high = centered.max(axis=0)
                    trans_min = -low
                    trans_max = box - high
                    if np.any(trans_min > trans_max):
                        continue
                    translation = rng.uniform(trans_min, trans_max)
                    candidate = centered + translation

                    if not self._check_distance(candidate, placed, args.tolerance):
                        continue

                    placed.extend(zip(base_symbols, candidate))
                    placed_object_ids.extend([object_id] * len(base_symbols))
                    object_metadata[object_id] = {
                        "component_index": comp_index,
                        "source_type": source_type,
                        "symbol": particle_meta.get("symbol", base_symbols[0]),
                        "radius": particle_meta.get("radius"),
                        "size": particle_meta.get("size"),
                    }
                    object_id += 1
                    success_count += 1
                    accepted = True
                    break

                if not accepted:
                    fail_count += 1

            summary: dict[str, Any] = {
                "source": spec.source,
                "source_type": source_type,
                "requested": spec.count,
                "placed": success_count,
                "failed": fail_count,
            }
            if particle_meta:
                summary["particle"] = particle_meta
            component_summaries.append(summary)

        if not placed:
            return ToolResult(
                data={
                    "components": component_summaries,
                    "message": "No objects placed.",
                },
                success=False,
            )

        objects = self._build_objects(placed, placed_object_ids, object_metadata)

        base_path = work_dir / args.output_prefix
        output_files: dict[str, str] = {}

        structure_path = self._write_structure(
            base_path, args.output_format, placed, box
        )
        output_files["structure"] = str(structure_path)

        if args.mode == "molecules" and args.write_packmol_input:
            packmol_path = base_path.with_suffix(".inp")
            self._write_packmol_input_file(packmol_path, args, work_dir)
            output_files["packmol_input"] = str(packmol_path)

        image_path: str | None = None
        if args.visualize:
            image_path = str(base_path.with_suffix(".png"))
            self._render_image(placed, image_path)
            output_files["image"] = image_path

        return ToolResult(
            data={
                "output_files": output_files,
                "components": component_summaries,
                "objects": objects,
                "total_atoms": len(placed),
                "box": box.tolist(),
                "message": f"Packed {len(placed)} atoms/points into {args.output_format} file.",
            },
            success=True,
        )

    def _particle_meta(self, source: str) -> dict[str, Any]:
        try:
            params = json.loads(source)
        except json.JSONDecodeError:
            return {}
        meta: dict[str, Any] = {"shape": params.get("shape", "sphere")}
        if meta["shape"] == "sphere":
            meta["radius"] = float(params.get("radius", 1.0))
        elif meta["shape"] == "cube":
            meta["size"] = float(params.get("size", 2.0))
            meta["radius"] = meta["size"] / 2.0
        meta["symbol"] = params.get("symbol", "P")
        return meta

    def _build_objects(
        self,
        placed: list[tuple[str, np.ndarray]],
        placed_object_ids: list[int],
        object_metadata: dict[int, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not placed:
            return []
        ids = np.array(placed_object_ids)
        coords = np.stack([c for _, c in placed])
        objects: list[dict[str, Any]] = []
        for oid, meta in sorted(object_metadata.items()):
            mask = ids == oid
            obj_coords = coords[mask]
            center = obj_coords.mean(axis=0).tolist()
            approx_radius = float(np.linalg.norm(obj_coords - center, axis=1).max())
            radius = meta.get("radius")
            objects.append(
                {
                    "id": oid,
                    "component_index": meta["component_index"],
                    "source_type": meta["source_type"],
                    "symbol": meta.get("symbol"),
                    "center": center,
                    "radius": radius if radius is not None else approx_radius,
                }
            )
        return objects

    # ------------------------------------------------------------------
    # Output writers
    # ------------------------------------------------------------------

    def _write_structure(
        self,
        base_path: Path,
        output_format: str,
        atoms: list[tuple[str, np.ndarray]],
        box: np.ndarray,
    ) -> Path:
        if output_format == "xyz":
            path = base_path.with_suffix(".xyz")
            self._write_xyz(path, atoms)
        elif output_format == "pdb":
            path = base_path.with_suffix(".pdb")
            self._write_pdb(path, atoms)
        elif output_format == "lammps-data":
            path = base_path.with_suffix(".data")
            self._write_lammps_data(path, atoms, box)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
        return path

    def _write_xyz(self, path: Path, atoms: list[tuple[str, np.ndarray]]) -> None:
        lines = [str(len(atoms)), "Packed by huginn-agent packing_tool"]
        for symbol, coord in atoms:
            lines.append(
                f"{symbol:>3} {coord[0]:12.6f} {coord[1]:12.6f} {coord[2]:12.6f}"
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _write_pdb(self, path: Path, atoms: list[tuple[str, np.ndarray]]) -> None:
        lines = ["REMARK   Generated by huginn-agent packing_tool"]
        for i, (symbol, coord) in enumerate(atoms, start=1):
            atom_name = symbol.upper()
            lines.append(
                f"ATOM  {i:5d} {atom_name:4s} MOL A   1    "
                f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}"
                f"  1.00  0.00          {atom_name:>2s}  "
            )
        lines.append("END")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _write_lammps_data(
        self,
        path: Path,
        atoms: list[tuple[str, np.ndarray]],
        box: np.ndarray,
    ) -> None:
        symbols = [s for s, _ in atoms]
        types = sorted(set(symbols))
        type_map = {s: i + 1 for i, s in enumerate(types)}
        lines = [
            "LAMMPS data file generated by huginn-agent packing_tool",
            "",
            f"{len(atoms)} atoms",
            f"{len(types)} atom types",
            "",
            f"0.0 {box[0]:.6f} xlo xhi",
            f"0.0 {box[1]:.6f} ylo yhi",
            f"0.0 {box[2]:.6f} zlo zhi",
            "",
            "Masses",
            "",
        ]
        for s in types:
            lines.append(f"{type_map[s]} {self._approx_mass(s):.3f}")
        lines.extend(["", "Atoms # atomic", ""])
        for i, (symbol, coord) in enumerate(atoms, start=1):
            lines.append(
                f"{i} {type_map[symbol]} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}"
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    @staticmethod
    def _approx_mass(symbol: str) -> float:
        table = {
            "H": 1.008,
            "C": 12.011,
            "N": 14.007,
            "O": 15.999,
            "F": 18.998,
            "P": 30.974,
            "S": 32.065,
            "Cl": 35.45,
            "Fe": 55.845,
            "Cu": 63.546,
            "Zn": 65.38,
            "Si": 28.085,
            "Au": 196.967,
        }
        return table.get(symbol, 12.011)

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def _render_image(
        self, atoms: list[tuple[str, np.ndarray]], image_path: str
    ) -> None:
        try:
            import matplotlib

            matplotlib.use("Agg")
            from matplotlib import pyplot as plt
        except ImportError as exc:
            raise RuntimeError(
                "Visualization requires matplotlib. Install with: pip install matplotlib"
            ) from exc

        coords = np.stack([c for _, c in atoms])
        colors = [_ELEMENT_COLORS.get(s, _DEFAULT_COLOR) for s, _ in atoms]

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            c=colors,
            s=80,
            alpha=0.8,
            edgecolors="k",
        )
        ax.set_xlabel("X (Å)")
        ax.set_ylabel("Y (Å)")
        ax.set_zlabel("Z (Å)")
        ax.set_title("Packed system")
        plt.tight_layout()
        plt.savefig(image_path, dpi=150)
        plt.close(fig)

    # ------------------------------------------------------------------
    # Packmol input generation / execution
    # ------------------------------------------------------------------

    def _generate_packmol_input(
        self, args: PackingToolInput, work_dir: Path
    ) -> ToolResult:
        path = work_dir / f"{args.output_prefix}.inp"
        self._write_packmol_input_file(path, args, work_dir)
        return ToolResult(
            data={
                "packmol_input": str(path),
                "packmol_available": self.packmol_executable is not None,
                "message": "Generated Packmol input file.",
            },
            success=True,
        )

    def _write_packmol_input_file(
        self,
        path: Path,
        args: PackingToolInput,
        work_dir: Path,
    ) -> None:
        lines = [
            "# Packmol input generated by huginn-agent packing_tool",
            f"tolerance {args.tolerance}",
            "filetype xyz",
            f"output {args.output_prefix}_packmol.xyz",
            "",
        ]
        for spec in args.components:
            source_type = self._detect_source_type(spec)
            if source_type != "xyz":
                continue
            src = self._resolve_path(spec.source, work_dir)
            lines.extend(
                [
                    f"structure {src}",
                    f"  number {spec.count}",
                    f"  inside box 0.0 0.0 0.0 {args.box[0]} {args.box[1]} {args.box[2]}",
                    "end structure",
                    "",
                ]
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _run_packmol(self, args: PackingToolInput, work_dir: Path) -> ToolResult:
        if not self.packmol_executable:
            return ToolResult(
                data=None,
                success=False,
                error="Packmol executable not found.",
            )
        packmol_path = work_dir / f"{args.output_prefix}.inp"
        self._write_packmol_input_file(packmol_path, args, work_dir)

        cfg = SandboxConfig(
            dry_run=False,
            allowed_executables=self.sandbox.config.allowed_executables | {"packmol"},
        )
        with open(packmol_path, encoding="utf-8") as stdin_file:
            result = self.sandbox.run(
                [self.packmol_executable],
                cwd=work_dir,
                config=cfg,
                stdin=stdin_file,
            )
        return ToolResult(
            data={
                "packmol_input": str(packmol_path),
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            },
            success=result.success,
        )

    # ------------------------------------------------------------------
    # Preview existing structure
    # ------------------------------------------------------------------

    def _preview(self, args: PackingToolInput, work_dir: Path) -> ToolResult:
        structure_path: Path | None = None
        if args.structure_file:
            structure_path = self._resolve_path(args.structure_file, work_dir)
        else:
            for ext in (".xyz", ".pdb"):
                candidate = work_dir / f"{args.output_prefix}{ext}"
                if candidate.exists():
                    structure_path = candidate
                    break

        if structure_path is None or not structure_path.exists():
            return ToolResult(
                data=None,
                success=False,
                error="No structure file found to preview.",
            )

        if structure_path.suffix.lower() == ".xyz":
            atoms = self._load_xyz(structure_path)
        elif structure_path.suffix.lower() == ".pdb":
            atoms = self._load_pdb(structure_path)
        else:
            return ToolResult(
                data=None,
                success=False,
                error=f"Unsupported preview format: {structure_path.suffix}",
            )

        image_path = str(work_dir / f"{args.output_prefix}.png")
        self._render_image(atoms, image_path)
        return ToolResult(
            data={
                "structure": str(structure_path),
                "image": image_path,
                "message": f"Rendered preview for {structure_path.name}.",
            },
            success=True,
        )

    def _load_pdb(self, path: Path) -> list[tuple[str, np.ndarray]]:
        atoms: list[tuple[str, np.ndarray]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            try:
                symbol = line[76:78].strip() or line[12:16].strip()
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                atoms.append((symbol, np.array([x, y, z])))
            except (ValueError, IndexError):
                continue
        return atoms
