"""Native Abaqus helper tool — generate scripts from packing data and more.

This tool does not replace the `.abaqus-mcp` MCP server; it complements it by
producing standalone Abaqus Python scripts (e.g. to import packed particles)
that can be run via `abaqus cae noGUI=script.py` or sent to the MCP server.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from matsci_agent.security import SandboxConfig, SandboxExecutor
from matsci_agent.tools.base import MatSciTool
from matsci_agent.types import ToolResult, ToolContext


class AbaqusToolInput(BaseModel):
    action: Literal["import_packing", "run"] = Field(default="import_packing")
    packing_data: dict[str, Any] | str | None = Field(
        default=None,
        description="Packing result dict or path to JSON (requires 'objects' list)",
    )
    script_path: str | None = Field(
        default=None,
        description="Path to an existing Abaqus Python script (for run action)",
    )
    particle_shape: Literal["reference_point", "sphere"] = Field(
        default="reference_point",
        description="How to represent each packed object in Abaqus",
    )
    base_model: str = Field(default="Model-1", description="Abaqus model name")
    output_prefix: str = Field(default="abaqus_particles")
    working_dir: str | None = Field(default=None)


class AbaqusTool(MatSciTool):
    """Generate Abaqus Python scripts from agent outputs and run them when available."""

    name = "abaqus_tool"
    description = (
        "Generate Abaqus Python scripts, e.g. to import packed particles as "
        "reference points or spherical inclusions, and run them via the Abaqus "
        "CLI when installed. Falls back to exporting the script."
    )
    input_schema = AbaqusToolInput

    def __init__(
        self,
        abaqus_executable: str | None = None,
        sandbox: SandboxExecutor | None = None,
    ) -> None:
        super().__init__()
        self.abaqus_executable = abaqus_executable or self._find_abaqus()
        self.sandbox = sandbox or SandboxExecutor()

    def _find_abaqus(self) -> str | None:
        env_path = os.environ.get("ABAQUS_EXECUTABLE")
        if env_path and Path(env_path).exists():
            return env_path
        return shutil.which("abaqus")

    def call(self, args: dict[str, Any], context: ToolContext | None = None) -> ToolResult:
        input_data = AbaqusToolInput(**args)
        work_dir = Path(input_data.working_dir) if input_data.working_dir else Path.cwd()
        work_dir.mkdir(parents=True, exist_ok=True)

        try:
            if input_data.action == "import_packing":
                return self._import_packing(input_data, work_dir)
            if input_data.action == "run":
                script_path = Path(input_data.script_path) if input_data.script_path else None
                if script_path is None or not script_path.exists():
                    return ToolResult(
                        data=None,
                        success=False,
                        error="run action requires an existing script_path.",
                    )
                return self._execute_script(script_path, work_dir)
            return ToolResult(data=None, success=False, error=f"Unknown action: {input_data.action}")
        except Exception as e:
            return ToolResult(data=None, success=False, error=f"Abaqus tool failed: {e}")

    def _import_packing(self, args: AbaqusToolInput, work_dir: Path) -> ToolResult:
        objects = self._load_packing_data(args.packing_data, work_dir)
        if not objects:
            return ToolResult(
                data=None,
                success=False,
                error="import_packing requires packing_data with a non-empty 'objects' list.",
            )

        script_path = work_dir / f"{args.output_prefix}.py"
        script = self._generate_import_script(args, objects)
        script_path.write_text(script, encoding="utf-8")

        exec_result = self._execute_script(script_path, work_dir)
        data = exec_result.data or {}
        data["script_path"] = str(script_path)
        data["objects_imported"] = len(objects)
        return ToolResult(data=data, success=exec_result.success)

    def _execute_script(self, script_path: Path, work_dir: Path) -> ToolResult:
        """Run an Abaqus Python script via CLI, or fall back to export."""
        if not self.abaqus_executable:
            return ToolResult(
                data={
                    "abaqus_available": False,
                    "stdout": "",
                    "stderr": "",
                    "message": (
                        "Abaqus executable not found. Generated script is available; "
                        f"run it manually with: abaqus cae noGUI={script_path.name}"
                    ),
                },
                success=True,
            )

        cmd = [self.abaqus_executable, "cae", f"noGUI={script_path}"]
        cfg = SandboxConfig(
            dry_run=False,
            allowed_executables=self.sandbox.config.allowed_executables | {"abaqus", "abaqus.bat"},
        )
        result = self.sandbox.run(cmd, cwd=work_dir, config=cfg)

        success = result.returncode == 0
        return ToolResult(
            data={
                "abaqus_available": True,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "message": (
                    "Abaqus execution completed."
                    if success
                    else "Abaqus execution failed; see stderr."
                ),
            },
            success=success,
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

    def _generate_import_script(
        self, args: AbaqusToolInput, objects: list[dict[str, Any]]
    ) -> str:
        if args.particle_shape == "sphere":
            body = self._sphere_part_body(args.base_model, objects)
        else:
            body = self._reference_point_body(args.base_model, objects)

        return f"""# Abaqus Python script generated by matsci-agent abaqus_tool
# Run with: abaqus cae noGUI={args.output_prefix}.py

from abaqus import *
from abaqusConstants import *
from caeModules import *

{body}
"""

    def _reference_point_body(self, base_model: str, objects: list[dict[str, Any]]) -> str:
        lines = [
            f"model = mdb.models['{base_model}']",
            "assy = model.rootAssembly",
            "particles = [",
        ]
        for obj in objects:
            c = obj["center"]
            r = float(obj["radius"])
            lines.append(f"    {{'center': ({c[0]:.6f}, {c[1]:.6f}, {c[2]:.6f}), 'radius': {r:.6f}}},")
        lines.extend([
            "]",
            "",
            "for i, p in enumerate(particles):",
            "    x, y, z = p['center']",
            "    rp = assy.ReferencePoint(point=(x, y, z))",
            "    assy.Set(referencePoints=(rp,), name='Particle_%d' % i)",
            "    # Radius stored in set description for later use",
            "    assy.sets['Particle_%d' % i].description = 'radius=%g' % p['radius']",
        ])
        return "\n".join(lines)

    def _sphere_part_body(self, base_model: str, objects: list[dict[str, Any]]) -> str:
        lines = [
            f"model = mdb.models['{base_model}']",
            "assy = model.rootAssembly",
            "particles = [",
        ]
        for obj in objects:
            c = obj["center"]
            r = float(obj["radius"])
            lines.append(f"    {{'center': ({c[0]:.6f}, {c[1]:.6f}, {c[2]:.6f}), 'radius': {r:.6f}}},")
        lines.extend([
            "]",
            "",
            "for i, p in enumerate(particles):",
            "    r = p['radius']",
            "    x, y, z = p['center']",
            "    part_name = 'Particle_%d' % i",
            "    sketch = model.ConstrainedSketch(name='__sphere_profile', sheetSize=2*r)",
            "    sketch.ConstructionLine(point1=(0.0, -r), point2=(0.0, r))",
            "    sketch.Line(point1=(0.0, -r), point2=(0.0, r))",
            "    sketch.ArcByCenterEnds(center=(0.0, 0.0), point1=(0.0, -r), point2=(0.0, r), direction=CLOCKWISE)",
            "    part = model.Part(name=part_name, dimensionality=THREE_D, type=DEFORMABLE_BODY)",
            "    part.BaseSolidRevolve(sketch=sketch, angle=360.0)",
            "    instance = assy.Instance(name=part_name, part=part, dependent=ON)",
            "    assy.translate(instanceList=(part_name,), vector=(x, y, z))",
        ])
        return "\n".join(lines)
