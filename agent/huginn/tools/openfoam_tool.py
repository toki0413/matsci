"""OpenFOAM CFD tool — generate case files and run via CLI.

When OpenFOAM is not installed, the tool falls back to case-export mode.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.security import SandboxConfig, SandboxExecutor
from huginn.tools.base import HuginnTool
from huginn.types import ToolResult, ToolContext


class BoundaryCondition(BaseModel):
    patch: str = Field(..., description="Patch name")
    type: Literal["wall", "inlet", "outlet", "symmetry", "empty"] = Field(...)
    value: dict = Field(default_factory=dict, description="OpenFOAM field value dict")


class OpenFoamToolInput(BaseModel):
    action: Literal["generate", "run", "parse", "set_fields"] = Field(default="run")
    solver: Literal[
        "icoFoam", "simpleFoam", "pimpleFoam", "laplacianFoam", "potentialFoam"
    ] = Field(default="icoFoam")
    case_name: str = Field(default="openfoam_case")
    geometry: dict = Field(
        default_factory=lambda: {
            "type": "block",
            "length": 2.0,
            "width": 0.5,
            "height": 0.5,
        }
    )
    mesh: dict = Field(default_factory=lambda: {"cells": [20, 8, 8]})
    boundary_conditions: list[BoundaryCondition] = Field(default_factory=list)
    transport_properties: dict = Field(
        default_factory=lambda: {"nu": 1e-05, "rho": 1.0}
    )
    turbulence: str = Field(default="laminar")
    end_time: float = Field(default=1.0)
    delta_t: float = Field(default=0.005)
    write_interval: int = Field(default=20)
    # set_fields action parameters
    packing_data: dict[str, Any] | str | None = Field(
        default=None,
        description="Packing result dict or path to JSON (requires 'objects' list)",
    )
    field_name: str = Field(default="alpha.water")
    default_value: float = Field(default=0.0)
    set_value: float = Field(default=1.0)
    working_dir: str | None = Field(default=None)
    result_files: list[str] = Field(default_factory=list)


class OpenFoamTool(HuginnTool):
    """Generate and run OpenFOAM CFD cases."""

    name = "openfoam_tool"
    description = (
        "Generate and run OpenFOAM CFD cases. "
        "Falls back to exporting case files when OpenFOAM is not installed."
    )
    input_schema = OpenFoamToolInput

    def __init__(self, openfoam_dir: str | None = None, sandbox: SandboxExecutor | None = None):
        super().__init__()
        self.openfoam_dir = openfoam_dir or self._find_openfoam()
        self.sandbox = sandbox or SandboxExecutor()

    def _find_openfoam(self) -> str | None:
        env_path = os.environ.get("OPENFOAM_DIR")
        if env_path and Path(env_path).exists():
            return env_path
        if shutil.which("blockMesh"):
            return "system"
        # Common Linux/macOS install paths
        for base in ["/usr/lib/openfoam", "/opt/openfoam"]:
            base_path = Path(base)
            if base_path.exists():
                for version in base_path.iterdir():
                    if version.is_dir() and (version / "bin" / "blockMesh").exists():
                        return str(version)
        return None

    def _openfoam_cmd(self, name: str) -> str | None:
        if self.openfoam_dir == "system":
            return shutil.which(name)
        if self.openfoam_dir:
            candidate = Path(self.openfoam_dir) / "bin" / name
            if candidate.exists():
                return str(candidate)
        return shutil.which(name)

    def call(self, args: dict[str, Any], context: ToolContext | None = None) -> ToolResult:
        input_data = OpenFoamToolInput(**args)
        work_dir = Path(input_data.working_dir) if input_data.working_dir else Path.cwd()
        work_dir.mkdir(parents=True, exist_ok=True)
        case_dir = work_dir / input_data.case_name

        try:
            if input_data.action == "parse":
                return self._parse_results(input_data, work_dir)

            if input_data.action == "set_fields":
                return self._set_fields(input_data, case_dir)

            self._generate_case(input_data, case_dir)

            if input_data.action == "generate":
                return ToolResult(
                    data={
                        "case_dir": str(case_dir),
                        "openfoam_available": self._openfoam_cmd("blockMesh") is not None,
                        "message": "Generated OpenFOAM case directory.",
                    },
                    success=True,
                )

            return self._run_openfoam(input_data, case_dir)
        except Exception as e:
            return ToolResult(data=None, success=False, error=f"OpenFOAM tool failed: {e}")

    def _generate_case(self, args: OpenFoamToolInput, case_dir: Path) -> None:
        """Generate a minimal OpenFOAM case directory."""
        system_dir = case_dir / "system"
        constant_dir = case_dir / "constant"
        zero_dir = case_dir / "0"

        system_dir.mkdir(parents=True, exist_ok=True)
        constant_dir.mkdir(parents=True, exist_ok=True)
        zero_dir.mkdir(parents=True, exist_ok=True)

        geom = args.geometry
        L = float(geom.get("length", 2.0))
        W = float(geom.get("width", 0.5))
        H = float(geom.get("height", 0.5))
        nx, ny, nz = args.mesh.get("cells", [20, 8, 8])

        # system/controlDict
        (system_dir / "controlDict").write_text(
            self._control_dict(args),
            encoding="utf-8",
        )

        # system/fvSchemes
        (system_dir / "fvSchemes").write_text(
            self._fv_schemes(args),
            encoding="utf-8",
        )

        # system/fvSolution
        (system_dir / "fvSolution").write_text(
            self._fv_solution(args),
            encoding="utf-8",
        )

        # system/blockMeshDict
        (system_dir / "blockMeshDict").write_text(
            self._block_mesh_dict(L, W, H, nx, ny, nz),
            encoding="utf-8",
        )

        # constant/transportProperties
        (constant_dir / "transportProperties").write_text(
            self._transport_properties(args),
            encoding="utf-8",
        )

        # constant/turbulenceProperties for RAS solvers
        if args.solver in ("simpleFoam", "pimpleFoam"):
            (constant_dir / "turbulenceProperties").write_text(
                self._turbulence_properties(args),
                encoding="utf-8",
            )

        # Initial fields
        (zero_dir / "p").write_text(self._field_p(), encoding="utf-8")
        (zero_dir / "U").write_text(self._field_u(), encoding="utf-8")

        if args.solver in ("simpleFoam", "pimpleFoam"):
            (zero_dir / "k").write_text(self._field_k(), encoding="utf-8")
            (zero_dir / "omega").write_text(self._field_omega(), encoding="utf-8")
            (zero_dir / "nut").write_text(self._field_nut(), encoding="utf-8")

    def _control_dict(self, args: OpenFoamToolInput) -> str:
        return f"""/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Version:  compliant
     \\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      controlDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

application     {args.solver};
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {args.end_time};
deltaT          {args.delta_t};
writeControl    timeStep;
writeInterval   {args.write_interval};
purgeWrite      3;
writeFormat     ascii;
writePrecision  6;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;

// ************************************************************************* //
"""

    def _fv_schemes(self, args: OpenFoamToolInput) -> str:
        ddt = "Euler" if args.solver in ("icoFoam", "pimpleFoam") else "steadyState"
        return f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      fvSchemes;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

ddtSchemes
{{
    default         {ddt};
}}

gradSchemes
{{
    default         Gauss linear;
}}

divSchemes
{{
    default         none;
    div(phi,U)      bounded Gauss linearUpwind grad(U);
}}

laplacianSchemes
{{
    default         Gauss linear corrected;
}}

interpolationSchemes
{{
    default         linear;
}}

snGradSchemes
{{
    default         corrected;
}}

// ************************************************************************* //
"""

    def _fv_solution(self, args: OpenFoamToolInput) -> str:
        if args.solver == "simpleFoam":
            return """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      fvSolution;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

solvers
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
    k
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-08;
        relTol          0.1;
    }
    omega
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
        k               0.7;
        omega           0.7;
    }
}

// ************************************************************************* //
"""
        return """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      fvSolution;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

solvers
{
    p
    {
        solver          PCG;
        preconditioner  DIC;
        tolerance       1e-06;
        relTol          0.05;
    }
    U
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-05;
        relTol          0;
    }
}

PISO
{
    nCorrectors     2;
    nNonOrthogonalCorrectors 0;
    pRefCell        0;
    pRefValue       0;
}

// ************************************************************************* //
"""

    def _block_mesh_dict(self, L: float, W: float, H: float, nx: int, ny: int, nz: int) -> str:
        return f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      blockMeshDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

convertToMeters 1;

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

edges
(
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

// ************************************************************************* //
"""

    def _transport_properties(self, args: OpenFoamToolInput) -> str:
        nu = args.transport_properties.get("nu", 1e-05)
        return f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "constant";
    object      transportProperties;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

transportModel  Newtonian;
nu              {nu};

// ************************************************************************* //
"""

    def _turbulence_properties(self, args: OpenFoamToolInput) -> str:
        return f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "constant";
    object      turbulenceProperties;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

simulationType  RAS;
RAS
{{
    RASModel        {args.turbulence};
    turbulence      on;
    printCoeffs     on;
}}

// ************************************************************************* //
"""

    def _field_p(self) -> str:
        return """FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    location    "0";
    object      p;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 2 -2 0 0 0 0];

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

// ************************************************************************* //
"""

    def _field_u(self) -> str:
        return """FoamFile
{
    version     2.0;
    format      ascii;
    class       volVectorField;
    location    "0";
    object      U;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 1 -1 0 0 0 0];

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

// ************************************************************************* //
"""

    def _field_k(self) -> str:
        return """FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    location    "0";
    object      k;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 2 -2 0 0 0 0];

internalField   uniform 0.00375;

boundaryField
{
    inlet
    {
        type            fixedValue;
        value           uniform 0.00375;
    }
    outlet
    {
        type            zeroGradient;
    }
    walls
    {
        type            kqRWallFunction;
        value           uniform 0.00375;
    }
}

// ************************************************************************* //
"""

    def _field_omega(self) -> str:
        return """FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    location    "0";
    object      omega;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 0 -1 0 0 0 0];

internalField   uniform 3.375;

boundaryField
{
    inlet
    {
        type            fixedValue;
        value           uniform 3.375;
    }
    outlet
    {
        type            zeroGradient;
    }
    walls
    {
        type            omegaWallFunction;
        value           uniform 3.375;
    }
}

// ************************************************************************* //
"""

    def _field_nut(self) -> str:
        return """FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    location    "0";
    object      nut;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 2 -1 0 0 0 0];

internalField   uniform 0;

boundaryField
{
    inlet
    {
        type            calculated;
        value           uniform 0;
    }
    outlet
    {
        type            calculated;
        value           uniform 0;
    }
    walls
    {
        type            nutkWallFunction;
        value           uniform 0;
    }
}

// ************************************************************************* //
"""

    def _run_openfoam(self, args: OpenFoamToolInput, case_dir: Path) -> ToolResult:
        block_mesh_cmd = self._openfoam_cmd("blockMesh")
        solver_cmd = self._openfoam_cmd(args.solver)

        if not block_mesh_cmd or not solver_cmd:
            return ToolResult(
                data={
                    "case_dir": str(case_dir),
                    "openfoam_available": False,
                    "message": (
                        "OpenFOAM executable not found. Case files exported; "
                        "run manually with: blockMesh && " + args.solver
                    ),
                },
                success=True,
            )

        from huginn.security.sandbox import SandboxConfig

        cfg = SandboxConfig(
            dry_run=False,
            allowed_executables=self.sandbox.config.allowed_executables | {
                "blockmesh", "icofoam", "simplefoam", "pimplefoam",
                "laplacianfoam", "potentialfoam",
            },
        )
        log_path = case_dir / f"{args.solver}.log"

        # Run blockMesh
        with open(log_path, "w", encoding="utf-8") as log_file:
            bm_result = self.sandbox.run(
                [block_mesh_cmd],
                cwd=case_dir,
                config=cfg,
                capture_output=False,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )

        if bm_result.returncode != 0:
            return ToolResult(
                data={
                    "case_dir": str(case_dir),
                    "openfoam_available": True,
                    "log_path": str(log_path),
                    "message": "blockMesh failed; see log.",
                },
                success=False,
            )

        # Run solver
        with open(log_path, "a", encoding="utf-8") as log_file:
            solver_result = self.sandbox.run(
                [solver_cmd],
                cwd=case_dir,
                config=cfg,
                capture_output=False,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )

        parsed = self._parse_log_file(log_path)
        success = solver_result.returncode == 0
        return ToolResult(
            data={
                "case_dir": str(case_dir),
                "log_path": str(log_path),
                "openfoam_available": True,
                "parsed": parsed,
                "message": "OpenFOAM execution completed." if success else "OpenFOAM solver failed; see log.",
            },
            success=success,
        )

    def _set_fields(self, args: OpenFoamToolInput, case_dir: Path) -> ToolResult:
        """Write setFieldsDict and initial phase field from packing output."""
        work_dir = Path(args.working_dir) if args.working_dir else Path.cwd()
        objects = self._load_packing_data(args.packing_data, work_dir)
        if not objects:
            return ToolResult(
                data=None,
                success=False,
                error="set_fields requires packing_data with a non-empty 'objects' list.",
            )

        # Ensure case directory has the minimum structure
        (case_dir / "system").mkdir(parents=True, exist_ok=True)
        (case_dir / "0").mkdir(parents=True, exist_ok=True)

        set_fields_path = self._write_set_fields_dict(
            case_dir,
            args.field_name,
            objects,
            args.default_value,
            args.set_value,
        )
        field_path = self._write_vol_scalar_field(
            case_dir / "0" / args.field_name,
            args.field_name,
            args.default_value,
        )

        setfields_cmd = self._openfoam_cmd("setFields")
        ran_setfields = False
        setfields_log = ""
        if setfields_cmd:
            from huginn.security.sandbox import SandboxConfig

            cfg = SandboxConfig(
                dry_run=False,
                allowed_executables=self.sandbox.config.allowed_executables | {"setfields"},
            )
            result = self.sandbox.run(
                [setfields_cmd],
                cwd=case_dir,
                config=cfg,
                capture_output=True,
                text=True,
            )
            ran_setfields = result.success
            setfields_log = result.stdout + result.stderr

        return ToolResult(
            data={
                "case_dir": str(case_dir),
                "set_fields_dict": str(set_fields_path),
                "initial_field": str(field_path),
                "objects_used": len(objects),
                "openfoam_available": setfields_cmd is not None,
                "setfields_executed": ran_setfields,
                "setfields_log": setfields_log,
                "message": (
                    "setFields configuration written"
                    + (" and executed." if ran_setfields else "; run setFields manually or after blockMesh.")
                ),
            },
            success=True,
        )

    def _load_packing_data(
        self,
        packing_data: dict[str, Any] | str | None,
        work_dir: Path,
    ) -> list[dict[str, Any]]:
        if packing_data is None:
            return []
        if isinstance(packing_data, str):
            path = Path(packing_data)
            if not path.is_absolute():
                path = work_dir / path
            data = json.loads(path.read_text(encoding="utf-8"))
        else:
            data = packing_data
        return data.get("objects", [])

    def _write_set_fields_dict(
        self,
        case_dir: Path,
        field_name: str,
        objects: list[dict[str, Any]],
        default_value: float,
        set_value: float,
    ) -> Path:
        region_blocks = []
        for obj in objects:
            centre = " ".join(f"{c:.6f}" for c in obj["center"])
            radius = float(obj["radius"])
            region_blocks.append(
                f"""    sphereToCell
    {{
        centre ({centre});
        radius {radius:.6f};
        fieldValues
        (
            volScalarFieldValue {field_name} {set_value};
        );
    }}"""
            )

        content = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      setFieldsDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

defaultFieldValues
(
    volScalarFieldValue {field_name} {default_value};
);

regions
(
{chr(10).join(region_blocks)}
);

// ************************************************************************* //
"""
        path = case_dir / "system" / "setFieldsDict"
        path.write_text(content, encoding="utf-8")
        return path

    def _write_vol_scalar_field(
        self,
        path: Path,
        field_name: str,
        default_value: float,
    ) -> Path:
        content = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    location    "0";
    object      {field_name};
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 0 0 0 0 0 0];

internalField   uniform {default_value};

boundaryField
{{
    inlet
    {{
        type            zeroGradient;
    }}
    outlet
    {{
        type            zeroGradient;
    }}
    walls
    {{
        type            zeroGradient;
    }}
}}

// ************************************************************************* //
"""
        path.write_text(content, encoding="utf-8")
        return path

    def _parse_log_file(self, log_path: Path) -> dict[str, Any]:
        if not log_path.exists():
            return {"error": "Log file not found"}
        content = log_path.read_text(encoding="utf-8", errors="ignore")
        return self._parse_log(content)

    def _parse_log(self, content: str) -> dict[str, Any]:
        result: dict[str, Any] = {
            "final_time": None,
            "continuity_errors": [],
            "final_residuals": {},
            "converged": False,
        }

        lines = content.splitlines()

        for line in lines:
            if "Time =" in line:
                parts = line.split("=")
                if len(parts) > 1:
                    try:
                        result["final_time"] = float(parts[-1].strip())
                    except (ValueError, IndexError):
                        pass

            if "continuity errors" in line.lower():
                # Extract the "global" continuity error value, e.g.
                # "time step continuity errors : sum local = 1.2e-06, global = -4.3e-08"
                parts = line.split("global =")
                if len(parts) > 1:
                    try:
                        result["continuity_errors"].append(float(parts[-1].strip().split(",")[0].split()[0]))
                    except (ValueError, IndexError):
                        pass

            # Final residuals: "Solving for Ux, Initial residual = ..., Final residual = ..."
            if "Solving for" in line and "Final residual" in line:
                parts = line.split(",")
                var_part = parts[0].split("for")[-1].strip()
                for part in parts:
                    if "Final residual" in part:
                        try:
                            val = float(part.split("=")[-1].strip())
                            result["final_residuals"][var_part] = val
                        except (ValueError, IndexError):
                            pass

        result["converged"] = "End" in content or "Finalising parallel run" in content
        return result

    def _parse_results(self, args: OpenFoamToolInput, work_dir: Path) -> ToolResult:
        parsed: dict[str, Any] = {}
        for file_name in args.result_files:
            file_path = work_dir / file_name
            parsed[file_name] = self._parse_log_file(file_path)

        return ToolResult(
            data={"results": parsed, "message": f"Parsed {len(parsed)} OpenFOAM log files."},
            success=True,
        )
