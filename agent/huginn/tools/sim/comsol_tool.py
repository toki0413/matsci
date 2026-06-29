"""COMSOL Multiphysics tool — generate models and run via CLI when available.

When COMSOL is not installed, the tool falls back to script-export mode and
returns the generated COMSOL Java script so the user can run it manually.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.security import SandboxConfig, SandboxExecutor
from huginn.tools.base import HuginnTool, ResearchPhase, ToolProfile
from huginn.types import ToolContext, ToolResult


class BoundaryCondition(BaseModel):
    type: Literal["fixed", "force", "pressure", "temperature", "velocity"] = Field(
        ..., description="Boundary condition type"
    )
    selection: str = Field(default="all", description="Boundary selection expression")
    value: list[float] | float = Field(default=0.0, description="BC value(s)")
    unit: str = Field(default="", description="Unit for the BC value")


class ComsolToolInput(BaseModel):
    action: Literal["generate", "run", "parse", "import_packing"] = Field(
        default="run",
        description="generate script, run COMSOL, parse results, or import packing data",
    )
    physics: Literal["solid_mechanics", "thermal", "cfd", "modal", "buckling"] = Field(
        default="solid_mechanics", description="Physics interface"
    )
    geometry: dict = Field(
        default_factory=lambda: {
            "type": "block",
            "width": 1.0,
            "height": 0.1,
            "depth": 0.1,
        },
        description="Geometry description",
    )
    mesh: dict = Field(
        default_factory=lambda: {"element_size": "normal"},
        description="Mesh settings",
    )
    material: dict = Field(
        default_factory=lambda: {
            "youngs_modulus": 200e9,
            "poissons_ratio": 0.3,
            "density": 7850.0,
        },
        description="Material properties",
    )
    boundary_conditions: list[BoundaryCondition] = Field(
        default_factory=list,
        description="List of boundary conditions",
    )
    solver: Literal["stationary", "transient", "eigenfrequency"] = Field(
        default="stationary", description="Solver study type"
    )
    packing_data: dict[str, Any] | str | None = Field(
        default=None,
        description="Packing result dict or path to JSON (for import_packing)",
    )
    working_dir: str | None = Field(default=None)
    output_prefix: str = Field(default="comsol_out")
    result_files: list[str] = Field(
        default_factory=list,
        description="Result files to parse (for action=parse)",
    )


class ComsolTool(HuginnTool):
    """Generate and run COMSOL Multiphysics models."""

    name = "comsol_tool"
    category = "sim"
    profile = ToolProfile(
        cost_tier="heavy",
        phases=frozenset({ResearchPhase.EXECUTION}),
        constraint_scope="fea",
        light_alternatives=("symbolic_math_tool", "numerical_tool"),
    )
    description = (
        "Generate and run COMSOL Multiphysics finite element models. "
        "Falls back to exporting the generated Java script when COMSOL is not installed."
    )
    input_schema = ComsolToolInput

    def __init__(
        self,
        comsol_executable: str | None = None,
        sandbox: SandboxExecutor | None = None,
    ):
        super().__init__()
        self.comsol_executable = comsol_executable or self._find_comsol()
        self.sandbox = sandbox or SandboxExecutor()

    def _find_comsol(self) -> str | None:
        """Find COMSOL executable on the system."""
        env_path = os.environ.get("COMSOL_EXECUTABLE")
        if env_path and Path(env_path).exists():
            return env_path

        for cmd in ["comsol", "comsolmph"]:
            if shutil.which(cmd):
                return cmd

        # Common Windows install locations
        program_files = os.environ.get("PROGRAMFILES", "C:\\Program Files")
        program_files_x86 = os.environ.get(
            "PROGRAMFILES(X86)", "C:\\Program Files (x86)"
        )
        for base in [program_files, program_files_x86]:
            base_path = Path(base)
            if not base_path.exists():
                continue
            for candidate in base_path.glob(
                "COMSOL/COMSOL*/Multiphysics/bin/win64/comsol.exe"
            ):
                return str(candidate)

        return None

    def call(
        self, args: dict[str, Any], context: ToolContext | None = None
    ) -> ToolResult:
        """Dispatch COMSOL action."""
        input_data = ComsolToolInput(**args)
        work_dir = (
            Path(input_data.working_dir) if input_data.working_dir else Path.cwd()
        )
        work_dir.mkdir(parents=True, exist_ok=True)

        try:
            if input_data.action == "parse":
                return self._parse_results(input_data, work_dir)

            if input_data.action == "import_packing":
                return self._import_packing(input_data, work_dir)

            script_path = self._generate_script(
                input_data, work_dir, input_data.output_prefix
            )

            if input_data.action == "generate":
                return ToolResult(
                    data={
                        "script_path": str(script_path),
                        "comsol_available": self.comsol_executable is not None,
                        "message": "Generated COMSOL Java script.",
                    },
                    success=True,
                )

            if input_data.action == "run":
                return self._run_comsol(input_data, work_dir, script_path)

            return ToolResult(
                data=None, success=False, error=f"Unknown action: {input_data.action}"
            )
        except Exception as e:
            return ToolResult(
                data=None, success=False, error=f"COMSOL tool failed: {e}"
            )

    def _generate_script(
        self, args: ComsolToolInput, work_dir: Path, prefix: str
    ) -> Path:
        """Generate a COMSOL Java script for the requested model."""
        script_path = work_dir / f"{prefix}.java"

        if args.physics == "solid_mechanics":
            script = self._generate_solid_mechanics_script(args, work_dir, prefix)
        elif args.physics == "thermal":
            script = self._generate_thermal_script(args, work_dir, prefix)
        elif args.physics == "modal":
            script = self._generate_modal_script(args, work_dir, prefix)
        else:
            script = self._generate_solid_mechanics_script(args, work_dir, prefix)

        script_path.write_text(script, encoding="utf-8")
        return script_path

    def _generate_solid_mechanics_script(
        self, args: ComsolToolInput, work_dir: Path, prefix: str
    ) -> str:
        """Generate a Java script for a basic solid mechanics simulation."""
        geom = args.geometry
        width = float(geom.get("width", 1.0))
        height = float(geom.get("height", 0.1))
        depth = float(geom.get("depth", 0.1))
        E = float(args.material.get("youngs_modulus", 200e9))
        nu = float(args.material.get("poissons_ratio", 0.3))
        rho = float(args.material.get("density", 7850.0))

        bc_lines = []
        for bc in args.boundary_conditions:
            if bc.type == "fixed":
                bc_lines.append(
                    f'model.physics("solid").feature().create("fix{bc_lines.__len__()}", "FixedConstraint", 2);'
                )
                bc_lines.append(
                    f'model.physics("solid").feature("fix{bc_lines.__len__()-1}").selection().named("geom1_{bc.selection}");'
                )
            elif bc.type == "force":
                val = bc.value if isinstance(bc.value, list) else [bc.value, 0.0, 0.0]
                bc_lines.append(
                    f'model.physics("solid").feature().create("load{bc_lines.__len__()}", "BoundaryLoad", 2);'
                )
                bc_lines.append(
                    f'model.physics("solid").feature("load{bc_lines.__len__()-1}").set("F", new String[][]{{"{val[0]}[N]", "{val[1]}[N]", "{val[2]}[N]"}});'
                )

        bc_script = "\n    ".join(bc_lines)

        return f"""import com.comsol.model.*;
import com.comsol.model.util.*;

public class {prefix} {{
  public static void main(String[] args) {{
    Model model = ModelUtil.create("{prefix}");
    model.modelPath("{work_dir.as_posix()}");

    // Geometry
    GeomSequence g = model.geom().create("geom1", 3);
    g.feature().create("blk1", "Block");
    g.feature("blk1").set("size", new String[]{{"{width}", "{height}", "{depth}"}});
    g.run();

    // Material
    Material mat = model.material().create("mat1");
    mat.set("E", "{E}[Pa]");
    mat.set("nu", "{nu}");
    mat.set("density", "{rho}[kg/m^3]");

    // Physics
    model.physics().create("solid", "SolidMechanics", "geom1");
    {bc_script}

    // Mesh
    model.mesh().create("mesh1", "geom1");
    model.mesh("mesh1").feature().create("size1", "Size");
    model.mesh("mesh1").feature("size1").set("custom", "on");
    model.mesh("mesh1").feature("size1").set("hmax", "{height / 5.0}");
    model.mesh("mesh1").run();

    // Study
    Study study = model.study().create("std1");
    study.create("stat", "Stationary");
    study.run();

    // Export results
    model.result().export().create("plot1", "Image3D");
    model.result().export("plot1").set("plotgroup", "pg1");
    model.save("{work_dir.as_posix()}/{prefix}.mph");
    System.out.println("COMSOL model saved to {work_dir.as_posix()}/{prefix}.mph");
  }}
}}
"""

    def _generate_thermal_script(
        self, args: ComsolToolInput, work_dir: Path, prefix: str
    ) -> str:
        """Generate a Java script for a basic thermal simulation."""
        geom = args.geometry
        width = float(geom.get("width", 1.0))
        height = float(geom.get("height", 0.1))
        depth = float(geom.get("depth", 0.1))
        k = float(args.material.get("thermal_conductivity", 400.0))
        rho = float(args.material.get("density", 7850.0))
        cp = float(args.material.get("specific_heat", 500.0))

        return f"""import com.comsol.model.*;
import com.comsol.model.util.*;

public class {prefix} {{
  public static void main(String[] args) {{
    Model model = ModelUtil.create("{prefix}");
    model.modelPath("{work_dir.as_posix()}");

    GeomSequence g = model.geom().create("geom1", 3);
    g.feature().create("blk1", "Block");
    g.feature("blk1").set("size", new String[]{{"{width}", "{height}", "{depth}"}});
    g.run();

    Material mat = model.material().create("mat1");
    mat.set("thermalconductivity", "{k}[W/(m*K)]");
    mat.set("density", "{rho}[kg/m^3]");
    mat.set("heatcapacity", "{cp}[J/(kg*K)]");

    model.physics().create("ht", "HeatTransfer", "geom1");

    model.mesh().create("mesh1", "geom1");
    model.mesh("mesh1").run();

    Study study = model.study().create("std1");
    study.create("stat", "Stationary");
    study.run();

    model.save("{work_dir.as_posix()}/{prefix}.mph");
    System.out.println("COMSOL thermal model saved to {work_dir.as_posix()}/{prefix}.mph");
  }}
}}
"""

    def _generate_modal_script(
        self, args: ComsolToolInput, work_dir: Path, prefix: str
    ) -> str:
        """Generate a Java script for a modal analysis."""
        return self._generate_solid_mechanics_script(args, work_dir, prefix)

    def _import_packing(self, args: ComsolToolInput, work_dir: Path) -> ToolResult:
        """Generate a COMSOL Java script that imports packed particle data and run it if available."""
        objects = self._load_packing_data(args.packing_data, work_dir)
        if not objects:
            return ToolResult(
                data=None,
                success=False,
                error="import_packing requires packing_data with a non-empty 'objects' list.",
            )

        prefix = args.output_prefix
        script_path = work_dir / f"{prefix}.java"
        script = self._generate_packing_import_script(prefix, objects)
        script_path.write_text(script, encoding="utf-8")

        exec_result = self._execute_script(script_path, prefix, work_dir)
        data = exec_result.data or {}
        data["script_path"] = str(script_path)
        data["objects_imported"] = len(objects)
        return ToolResult(
            data=data,
            success=exec_result.success,
        )

    def _load_packing_data(
        self, packing_data: dict[str, Any] | str | None, work_dir: Path
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

    def _generate_packing_import_script(
        self, prefix: str, objects: list[dict[str, Any]]
    ) -> str:
        centres = ", ".join(
            f"{{{obj['center'][0]:.6f}, {obj['center'][1]:.6f}, {obj['center'][2]:.6f}}}"
            for obj in objects
        )
        radii = ", ".join(f"{float(obj['radius']):.6f}" for obj in objects)

        return f"""import com.comsol.model.*;
import com.comsol.model.util.*;

public class {prefix} {{
  public static void main(String[] args) {{
    Model model = ModelUtil.create("{prefix}");
    GeomSequence g = model.geom().create("geom1", 3);

    double[][] centres = {{{centres}}};
    double[] radii = {{{radii}}};

    for (int i = 0; i < centres.length; i++) {{
      String tag = "sph" + i;
      g.feature().create(tag, "Sphere");
      g.feature(tag).set("r", radii[i]);
      g.feature(tag).set("pos", centres[i]);
    }}

    g.run();
    model.geom().run("geom1");
    ModelUtil.save("{prefix}.mph");
  }}
}}
"""

    def _run_comsol(
        self, args: ComsolToolInput, work_dir: Path, script_path: Path
    ) -> ToolResult:
        """Run the generated COMSOL script via CLI, or fall back to script export."""
        return self._execute_script(script_path, args.output_prefix, work_dir)

    def _execute_script(
        self, script_path: Path, output_prefix: str, work_dir: Path
    ) -> ToolResult:
        """Run an existing COMSOL Java script via CLI, or fall back to export."""
        base_data = {"script_path": str(script_path)}
        if not self.comsol_executable:
            return ToolResult(
                data={
                    **base_data,
                    "comsol_available": False,
                    "stdout": "",
                    "stderr": "",
                    "message": (
                        "COMSOL executable not found. Generated script is available; "
                        "run it manually with: comsol batch -inputfile "
                        f"{script_path.name} -outputfile {output_prefix}.mph"
                    ),
                },
                success=True,
            )

        mph_path = work_dir / f"{output_prefix}.mph"
        cmd = [
            self.comsol_executable,
            "batch",
            "-inputfile",
            str(script_path),
            "-outputfile",
            str(mph_path),
            "-tmpdir",
            str(work_dir),
        ]

        cfg = SandboxConfig(
            dry_run=False,
            allowed_executables=self.sandbox.config.allowed_executables
            | {"comsol", "comsolmph"},
        )
        result = self.sandbox.run(cmd, cwd=work_dir, config=cfg)

        success = result.returncode == 0
        return ToolResult(
            data={
                **base_data,
                "mph_path": str(mph_path) if mph_path.exists() else None,
                "comsol_available": True,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "message": (
                    "COMSOL execution completed."
                    if success
                    else "COMSOL execution failed; see stderr."
                ),
            },
            success=success,
        )

    def _parse_results(self, args: ComsolToolInput, work_dir: Path) -> ToolResult:
        """Parse COMSOL-exported CSV/TXT result files."""
        parsed: dict[str, Any] = {}
        for file_name in args.result_files:
            file_path = work_dir / file_name
            if not file_path.exists():
                parsed[file_name] = {"error": "File not found"}
                continue

            content = file_path.read_text(encoding="utf-8", errors="ignore")
            lines = [line.strip() for line in content.splitlines() if line.strip()]

            if not lines:
                parsed[file_name] = {"error": "Empty file"}
                continue

            # Simple CSV/space-delimited parser
            header = lines[0].replace(",", " ").split()
            rows = []
            for line in lines[1:]:
                parts = line.replace(",", " ").split()
                if len(parts) >= len(header):
                    row = {}
                    for h, p in zip(header, parts):
                        try:
                            row[h] = float(p)
                        except ValueError:
                            row[h] = p
                    rows.append(row)

            parsed[file_name] = {
                "header": header,
                "n_rows": len(rows),
                "rows": rows[:100],  # Limit to first 100 rows
            }

        return ToolResult(
            data={
                "results": parsed,
                "message": f"Parsed {len(parsed)} result files.",
            },
            success=True,
        )
