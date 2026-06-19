"""Input File Generator — creates calculation inputs from high-level specs.

Instead of the user manually writing POSCAR/INCAR/Gaussian input files,
the Agent generates them from natural language or structured parameters.

Supported software:
  - VASP (POSCAR, INCAR, KPOINTS, POTCAR hints)
  - Gaussian (.gjf route sections)
  - LAMMPS (in.* scripts, data files)
  - ORCA (.inp files)
  - ABAQUS (.inp decks)
  - OpenFOAM (dict files: controlDict, fvSchemes, fvSolution, etc.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class GeneratedInput:
    """A generated input file with metadata."""

    software: str
    filename: str
    content: str
    description: str
    validation_notes: list[str] = field(default_factory=list)


class InputFileGenerator:
    """Generate computational software input files from high-level specifications.

    Usage:
        gen = InputFileGenerator()
        inputs = gen.generate_vasp_inputs(
            system="Si bulk",
            structure={"lattice": 5.43, "basis": [[0,0,0]], "species": ["Si"]},
            task="band_structure",
            params={"ENCUT": 520, "KPOINTS": "9 9 9"}
        )
        for inp in inputs:
            Path(inp.filename).write_text(inp.content)
    """

    def __init__(self, template_dir: str | None = None):
        self.template_dir = Path(template_dir) if template_dir else None

    # ------------------------------------------------------------------
    # VASP
    # ------------------------------------------------------------------

    def generate_vasp_inputs(
        self,
        system: str,
        structure: dict[str, Any],
        task: str,
        params: dict[str, Any] | None = None,
        potcar_hints: dict[str, str] | None = None,
    ) -> list[GeneratedInput]:
        """Generate VASP input set for a given task.

        Args:
            system: System description (e.g., "Si bulk FCC")
            structure: {"lattice": float or [[a,b,c],...], "basis": [[x,y,z],...], "species": ["Si", "Si"]}
            task: "relax", "scf", "band", "dos", "md", "phonon"
            params: Override dict for INCAR tags
            potcar_hints: {"Si": "Si_PBE", "O": "O_PBE"} — POTCAR recommendations
        """
        params = params or {}
        potcar_hints = potcar_hints or {}
        inputs = []

        # POSCAR
        poscar = self._make_poscar(system, structure)
        inputs.append(GeneratedInput("VASP", "POSCAR", poscar, "VASP structure file"))

        # INCAR — task-specific defaults + user overrides
        incar = self._make_incar(task, params)
        inputs.append(GeneratedInput("VASP", "INCAR", incar, f"VASP input for {task}"))

        # KPOINTS
        kpoints = self._make_kpoints(task, params.get("KPOINTS", "automatic"))
        inputs.append(GeneratedInput("VASP", "KPOINTS", kpoints, "K-point mesh"))

        # POTCAR hint file
        if potcar_hints:
            potcar_info = self._make_potcar_info(potcar_hints)
            inputs.append(
                GeneratedInput(
                    "VASP", "POTCAR_HINTS", potcar_info, "POTCAR selection hints"
                )
            )

        return inputs

    def _make_poscar(self, system: str, structure: dict[str, Any]) -> str:
        lines = [system, "1.0"]
        lattice = structure.get("lattice", 1.0)
        if isinstance(lattice, (int, float)):
            # Cubic
            lines.append(f"  {lattice:.6f}  0.0  0.0")
            lines.append(f"  0.0  {lattice:.6f}  0.0")
            lines.append(f"  0.0  0.0  {lattice:.6f}")
        else:
            for row in lattice:
                lines.append("  " + "  ".join(f"{x:.6f}" for x in row))

        species = structure.get("species", [])
        basis = structure.get("basis", [])
        from collections import Counter

        counts = Counter(species)
        lines.append("  " + "  ".join(counts.keys()))
        lines.append("  " + "  ".join(str(counts[s]) for s in counts))
        lines.append("Direct")
        for coord in basis:
            lines.append("  " + "  ".join(f"{x:.6f}" for x in coord))
        return "\n".join(lines) + "\n"

    def _make_incar(self, task: str, overrides: dict[str, Any]) -> str:
        defaults = {
            "relax": {
                "ISIF": 3,
                "IBRION": 2,
                "EDIFFG": -0.01,
                "NSW": 100,
                "ISMEAR": 0,
                "SIGMA": 0.05,
                "ENCUT": 520,
                "PREC": "Normal",
            },
            "scf": {
                "ISMEAR": 0,
                "SIGMA": 0.05,
                "ENCUT": 520,
                "PREC": "Normal",
                "EDIFF": 1e-6,
            },
            "band": {
                "ISMEAR": 0,
                "SIGMA": 0.05,
                "ENCUT": 520,
                "PREC": "Accurate",
                "LORBIT": 11,
                "ICHARG": 11,
            },
            "dos": {
                "ISMEAR": -5,
                "ENCUT": 520,
                "LORBIT": 11,
                "NEDOS": 3001,
            },
            "md": {
                "IBRION": 0,
                "NSW": 1000,
                "POTIM": 1.0,
                "TEBEG": 300,
                "TEEND": 300,
                "ISMEAR": 0,
                "SIGMA": 0.05,
            },
            "phonon": {
                "IBRION": 8,
                "ENCUT": 520,
                "PREC": "Accurate",
            },
        }
        tags = dict(defaults.get(task, defaults["scf"]))
        tags.update(overrides)
        lines = [f"{k} = {v}" for k, v in tags.items()]
        return "\n".join(lines) + "\n"

    def _make_kpoints(self, task: str, spec: Any) -> str:
        if isinstance(spec, str) and " " in spec:
            # "9 9 9" format
            return f"Automatic mesh\n0\nMonkhorst-Pack\n{spec}\n0 0 0\n"
        if task in ("band", "dos"):
            return "Line-mode\n20\nReciprocal\n0 0 0  G\n0.5 0 0  X\n"
        return "Automatic mesh\n0\nGamma\n3 3 3\n0 0 0\n"

    def _make_potcar_info(self, hints: dict[str, str]) -> str:
        lines = [
            "# POTCAR selection hints",
            "# Use 'cat' to concatenate in species order",
        ]
        for elem, pot in hints.items():
            lines.append(f"{elem}: {pot}")
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # Gaussian
    # ------------------------------------------------------------------

    def generate_gaussian_input(
        self,
        task: str,
        method: str,
        basis: str,
        structure: str,  # xyz format or z-matrix
        extras: dict[str, Any] | None = None,
    ) -> GeneratedInput:
        """Generate Gaussian .gjf input file.

        Args:
            task: "opt", "sp", "freq", "td", "nbo"
            method: "B3LYP", "CAM-B3LYP", "M06-2X", etc.
            basis: "6-31G(d)", "def2-TZVP", etc.
            structure: XYZ string or z-matrix
            extras: Additional route section keywords
        """
        extras = extras or {}
        route = f"#{method}/{basis}"

        task_map = {
            "opt": "opt",
            "sp": "sp",
            "freq": "freq",
            "td": "td=(nstates=10,root=1)",
            "nbo": "pop=(nbo,full)",
        }
        route += f" {task_map.get(task, 'sp')}"

        if extras.get("scf_xqc"):
            route += " scf=xqc"
        if extras.get("integral"):
            route += f" integral={extras['integral']}"
        if extras.get("iop"):
            route += f" IOp({extras['iop']})"

        content = f"%chk=job.chk\n%mem=8GB\n%nprocshared=4\n{route}\n\nTitle\n\n0 1\n{structure}\n\n"
        return GeneratedInput(
            "Gaussian", "job.gjf", content, f"Gaussian input for {task}"
        )

    # ------------------------------------------------------------------
    # LAMMPS
    # ------------------------------------------------------------------

    def generate_lammps_input(
        self,
        task: str,
        potential: str,
        structure_file: str,
        temperature: float = 300.0,
        steps: int = 100000,
        timestep: float = 1.0,
        ensemble: str = "nvt",
    ) -> GeneratedInput:
        """Generate LAMMPS input script."""
        lines = [
            f"# LAMMPS input for {task}",
            "units metal",
            "atom_style atomic",
            f"read_data {structure_file}",
            "",
            f"pair_style {potential}",
            "pair_coeff * * potential_file.element1 element2",
            "",
            f"timestep {timestep}",
            f"fix 1 all {ensemble} temp {temperature} {temperature} 0.1",
            "",
            "thermo 100",
            "thermo_style custom step temp pe ke etotal press",
            "",
            f"run {steps}",
            "write_data final.data",
        ]
        return GeneratedInput(
            "LAMMPS", "in.lammps", "\n".join(lines) + "\n", f"LAMMPS input for {task}"
        )

    # ------------------------------------------------------------------
    # ABAQUS
    # ------------------------------------------------------------------

    def generate_abaqus_input(
        self,
        job_name: str,
        element_type: str = "C3D8R",
        material: dict[str, Any] | None = None,
        bc: list[dict[str, Any]] | None = None,
    ) -> GeneratedInput:
        """Generate ABAQUS .inp deck skeleton."""
        mat = material or {"name": "Steel", "E": 210000, "nu": 0.3}
        lines = [
            f"*Heading\n{job_name}",
            "*Preprint, echo=NO, model=NO, history=NO, contact=NO",
            "**",
            "*Part, name=Part-1",
            "*End Part",
            "**",
            "*Assembly, name=Assembly",
            "*Instance, name=Part-1-1, part=Part-1",
            "*End Instance",
            "*End Assembly",
            "**",
            f"*Material, name={mat['name']}",
            f"*Elastic\n{mat['E']}, {mat['nu']}",
            "**",
            "*Step, name=Step-1, nlgeom=NO",
            "*Static",
            "1., 1., 1e-05, 1.",
        ]
        if bc:
            for b in bc:
                lines.append(
                    f"*Boundary\n{b['node_set']}, {b['dof']}, {b['dof']}, {b.get('value', 0)}"
                )
        lines.extend(
            [
                "*Output, field, variable=PRESELECT",
                "*Output, history, variable=PRESELECT",
                "*End Step",
            ]
        )
        return GeneratedInput(
            "ABAQUS", f"{job_name}.inp", "\n".join(lines) + "\n", "ABAQUS input deck"
        )

    # ------------------------------------------------------------------
    # OpenFOAM
    # ------------------------------------------------------------------

    def generate_openfoam_dicts(
        self,
        solver: str = "simpleFoam",
        turbulence: str = "kOmegaSST",
        geometry: dict[str, Any] | None = None,
        mesh: dict[str, Any] | None = None,
        transport_properties: dict[str, Any] | None = None,
    ) -> list[GeneratedInput]:
        """Generate essential OpenFOAM dictionary files."""
        geometry = geometry or {"length": 2.0, "width": 0.5, "height": 0.5}
        mesh = mesh or {"cells": [20, 8, 8]}
        transport_properties = transport_properties or {"nu": 1e-05}
        inputs = []

        L = float(geometry.get("length", 2.0))
        W = float(geometry.get("width", 0.5))
        H = float(geometry.get("height", 0.5))
        nx, ny, nz = mesh.get("cells", [20, 8, 8])

        # controlDict
        control = f"""application     {solver};
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         1000;
deltaT          1;
writeControl    timeStep;
writeInterval   100;
purgeWrite      3;
writeFormat     ascii;
writePrecision  6;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;
"""
        inputs.append(
            GeneratedInput("OpenFOAM", "controlDict", control, "Time control")
        )

        # fvSchemes (simplified)
        schemes = """ddtSchemes
{
    default         steadyState;
}
gradSchemes
{
    default         Gauss linear;
}
divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss linearUpwind grad(U);
    div(phi,k)      bounded Gauss upwind;
    div(phi,omega)  bounded Gauss upwind;
}
laplacianSchemes
{
    default         Gauss linear corrected;
}
"""
        inputs.append(
            GeneratedInput("OpenFOAM", "fvSchemes", schemes, "Discretization schemes")
        )

        # fvSolution (simplified)
        solution = """solvers
{
    p
    {
        solver          GAMG;
        tolerance       1e-07;
        relTol          0.1;
    }
    U
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-08;
        relTol          0.1;
    }
}
SIMPLE
{
    nNonOrthogonalCorrectors 0;
    consistent      yes;
    residualControl
    {
        p               1e-5;
        U               1e-5;
    }
}
relaxationFactors
{
    fields
    {
        p               0.3;
    }
    equations
    {
        U               0.7;
    }
}
"""
        inputs.append(
            GeneratedInput("OpenFOAM", "fvSolution", solution, "Solver settings")
        )

        # turbulenceProperties
        turb = f"""simulationType  RAS;
RAS
{{
    RASModel        {turbulence};
    turbulence      on;
    printCoeffs     on;
}}
"""
        inputs.append(
            GeneratedInput("OpenFOAM", "turbulenceProperties", turb, "Turbulence model")
        )

        # blockMeshDict
        block_mesh = f"""convertToMeters 1;

vertices
(
    (0 0 0)
    ({L} 0 0)
    ({L} {W} 0)
    (0 {W} 0)
    (0 0 {H})
    ({L} 0 {H})
    ({L} {W} {H})
    (0 {W} {H})
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1)
);

boundary
(
    inlet
    {{
        type patch;
        faces
        (
            (0 4 7 3)
        );
    }}
    outlet
    {{
        type patch;
        faces
        (
            (1 2 6 5)
        );
    }}
    walls
    {{
        type wall;
        faces
        (
            (0 1 5 4)
            (3 7 6 2)
            (0 3 2 1)
            (4 5 6 7)
        );
    }}
);
"""
        inputs.append(
            GeneratedInput(
                "OpenFOAM", "blockMeshDict", block_mesh, "Block mesh definition"
            )
        )

        # transportProperties
        nu = transport_properties.get("nu", 1e-05)
        transport = f"""transportModel  Newtonian;
nu              {nu};
"""
        inputs.append(
            GeneratedInput(
                "OpenFOAM", "transportProperties", transport, "Transport properties"
            )
        )

        # Initial fields
        p_field = """dimensions      [0 2 -2 0 0 0 0];

internalField   uniform 0;

boundaryField
{
    inlet
    {
        type            zeroGradient;
    }
    outlet
    {
        type            fixedValue;
        value           uniform 0;
    }
    walls
    {
        type            zeroGradient;
    }
}
"""
        inputs.append(
            GeneratedInput("OpenFOAM", "p", p_field, "Pressure initial/boundary field")
        )

        u_field = """dimensions      [0 1 -1 0 0 0 0];

internalField   uniform (0 0 0);

boundaryField
{
    inlet
    {
        type            fixedValue;
        value           uniform (1 0 0);
    }
    outlet
    {
        type            zeroGradient;
    }
    walls
    {
        type            noSlip;
    }
}
"""
        inputs.append(
            GeneratedInput("OpenFOAM", "U", u_field, "Velocity initial/boundary field")
        )

        return inputs

    # ------------------------------------------------------------------
    # Generic dispatcher
    # ------------------------------------------------------------------

    def generate(
        self,
        software: str,
        task: str,
        **kwargs: Any,
    ) -> list[GeneratedInput]:
        """Dispatch to the appropriate generator based on software name."""
        sw = software.lower()
        if sw == "vasp":
            return self.generate_vasp_inputs(task=task, **kwargs)
        if sw == "gaussian":
            return [self.generate_gaussian_input(task=task, **kwargs)]
        if sw == "lammps":
            return [self.generate_lammps_input(task=task, **kwargs)]
        if sw == "abaqus":
            return [self.generate_abaqus_input(**kwargs)]
        if sw == "openfoam":
            return self.generate_openfoam_dicts(**kwargs)
        raise ValueError(f"Unsupported software: {software}")
