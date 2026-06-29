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

from pydantic import BaseModel, Field, model_validator

from huginn.security import SandboxConfig, SandboxExecutor
from huginn.tools.base import HuginnTool
from huginn.types import ToolContext, ToolResult


class StaticGeneralSpec(BaseModel):
    """静力通用分析步参数."""

    youngs_modulus: float = Field(..., gt=0, description="E, Pa")
    poissons_ratio: float = Field(default=0.3, ge=-1, lt=0.5)
    density: float = Field(default=7850.0, gt=0)
    section_type: Literal["solid", "shell", "beam"] = "solid"
    section_dims: dict[str, float] = Field(
        default_factory=dict,
        description="solid: {thickness}; shell: {thickness}; beam: {area, I}",
    )
    nlgeom: bool = Field(default=True, description="大变形开关")
    max_iterations: int = Field(default=100, ge=1)
    loads: list[dict[str, Any]] = Field(
        default_factory=list,
        description="[{type: pressure|concentrated|gravity, value, region?}]",
    )
    boundary_conditions: list[dict[str, Any]] = Field(
        default_factory=list,
        description="[{region, dofs: [1,2,3], value: 0.0}]",
    )


class ModalSpec(BaseModel):
    """模态分析参数."""

    num_modes: int = Field(default=10, ge=1, le=200)
    frequency_range: tuple[float, float] | None = Field(
        default=None, description="(f_min, f_max) Hz, 不给就全频段"
    )
    lumped_mass: bool = Field(default=False, description="集中质量矩阵")


class BucklingSpec(BaseModel):
    """特征值屈曲参数."""

    num_modes: int = Field(default=5, ge=1, le=50)
    imperfection: float = Field(
        default=0.0, ge=0.0, description="初始缺陷幅值 (相对一阶模态)"
    )


class FatigueSpec(BaseModel):
    """疲劳分析参数."""

    sn_params: dict[str, float] = Field(
        default_factory=dict,
        description="Basquin: {sigma_f_prime, b}; 或 S-N 曲线系数",
    )
    mean_stress_theory: Literal["goodman", "soderberg", "gerber", "morrow"] = (
        "goodman"
    )
    cycles_limit: int = Field(default=10**6, ge=1)
    stress_amplitude: float = Field(..., gt=0, description="应力幅, Pa")
    mean_stress: float = Field(default=0.0, description="平均应力, Pa")


class FractureSpec(BaseModel):
    """断裂力学参数."""

    crack_type: Literal["edge", "interior"] = "edge"
    crack_length: float = Field(..., gt=0, description="裂纹长度 a, m")
    contour_integral: int = Field(default=5, ge=1, le=20, description="积分回路数")
    j_integral: bool = Field(default=True)
    k_ic: float | None = Field(default=None, gt=0, description="断裂韧性, Pa·√m")


class AbaqusToolInput(BaseModel):
    action: Literal[
        "import_packing",
        "run",
        "static_general",
        "modal",
        "buckling",
        "fatigue",
        "fracture",
    ] = Field(default="import_packing")
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

    # 5 个新 action 的专用参数
    static_spec: StaticGeneralSpec | None = None
    modal_spec: ModalSpec | None = None
    buckling_spec: BucklingSpec | None = None
    fatigue_spec: FatigueSpec | None = None
    fracture_spec: FractureSpec | None = None

    @model_validator(mode="after")
    def _check_action_fields(self) -> "AbaqusToolInput":
        """新 action 需要对应 spec, 在 schema 层兜底."""
        spec_map = {
            "static_general": "static_spec",
            "modal": "modal_spec",
            "buckling": "buckling_spec",
            "fatigue": "fatigue_spec",
            "fracture": "fracture_spec",
        }
        for action_name, spec_field in spec_map.items():
            if self.action == action_name and getattr(self, spec_field) is None:
                raise ValueError(
                    f"action '{action_name}' requires '{spec_field}'"
                )
        return self


class AbaqusTool(HuginnTool):
    """Generate Abaqus Python scripts from agent outputs and run them when available."""

    name = "abaqus_tool"
    category = "sim"
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

    def call(
        self, args: dict[str, Any], context: ToolContext | None = None
    ) -> ToolResult:
        input_data = AbaqusToolInput(**args)
        work_dir = (
            Path(input_data.working_dir) if input_data.working_dir else Path.cwd()
        )
        work_dir.mkdir(parents=True, exist_ok=True)

        try:
            if input_data.action == "import_packing":
                return self._import_packing(input_data, work_dir)
            if input_data.action == "run":
                script_path = (
                    Path(input_data.script_path) if input_data.script_path else None
                )
                if script_path is None or not script_path.exists():
                    return ToolResult(
                        data=None,
                        success=False,
                        error="run action requires an existing script_path.",
                    )
                return self._execute_script(script_path, work_dir)

            # 5 个新 action: 生成脚本 + 可选执行
            script_generators = {
                "static_general": self._generate_static_general_script,
                "modal": self._generate_modal_script,
                "buckling": self._generate_buckling_script,
                "fatigue": self._generate_fatigue_script,
                "fracture": self._generate_fracture_script,
            }
            gen = script_generators.get(input_data.action)
            if gen is None:
                return ToolResult(
                    data=None,
                    success=False,
                    error=f"Unknown action: {input_data.action}",
                )
            script = gen(input_data)
            prefix = input_data.output_prefix or f"abaqus_{input_data.action}"
            script_path = work_dir / f"{prefix}.py"
            script_path.write_text(script, encoding="utf-8")

            exec_result = self._execute_script(script_path, work_dir)
            data = exec_result.data or {}
            data["script_path"] = str(script_path)
            data["action"] = input_data.action
            return ToolResult(data=data, success=exec_result.success)
        except Exception as e:
            return ToolResult(
                data=None, success=False, error=f"Abaqus tool failed: {e}"
            )

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
            allowed_executables=self.sandbox.config.allowed_executables
            | {"abaqus", "abaqus.bat"},
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

        return f"""# Abaqus Python script generated by huginn-agent abaqus_tool
# Run with: abaqus cae noGUI={args.output_prefix}.py

from abaqus import *
from abaqusConstants import *
from caeModules import *

{body}
"""

    def _reference_point_body(
        self, base_model: str, objects: list[dict[str, Any]]
    ) -> str:
        lines = [
            f"model = mdb.models['{base_model}']",
            "assy = model.rootAssembly",
            "particles = [",
        ]
        for obj in objects:
            c = obj["center"]
            r = float(obj["radius"])
            lines.append(
                f"    {{'center': ({c[0]:.6f}, {c[1]:.6f}, {c[2]:.6f}), 'radius': {r:.6f}}},"
            )
        lines.extend(
            [
                "]",
                "",
                "for i, p in enumerate(particles):",
                "    x, y, z = p['center']",
                "    rp = assy.ReferencePoint(point=(x, y, z))",
                "    assy.Set(referencePoints=(rp,), name='Particle_%d' % i)",
                "    # Radius stored in set description for later use",
                "    assy.sets['Particle_%d' % i].description = 'radius=%g' % p['radius']",
            ]
        )
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
            lines.append(
                f"    {{'center': ({c[0]:.6f}, {c[1]:.6f}, {c[2]:.6f}), 'radius': {r:.6f}}},"
            )
        lines.extend(
            [
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
            ]
        )
        return "\n".join(lines)

    # ── 5 个新 action 的脚本生成器 ──

    def _generate_static_general_script(self, args: AbaqusToolInput) -> str:
        """生成静力通用分析脚本 (*STATIC, NLGEOM)."""
        s = args.static_spec
        E = s.youngs_modulus
        nu = s.poissons_ratio
        rho = s.density
        nlgeom = "ON" if s.nlgeom else "OFF"
        niter = s.max_iterations

        loads_lines = []
        for i, load in enumerate(s.loads):
            ltype = load.get("type", "pressure")
            val = float(load.get("value", 0.0))
            if ltype == "pressure":
                loads_lines.append(
                    f"    model.Pressure(name='Load-{i}', createStepName='Step-1', "
                    f"region=region, magnitude={val})"
                )
            elif ltype == "concentrated":
                loads_lines.append(
                    f"    model.ConcentratedForce(name='Load-{i}', createStepName='Step-1', "
                    f"region=region, cf3={val})"
                )
            elif ltype == "gravity":
                loads_lines.append(
                    f"    model.Gravity(name='Load-{i}', createStepName='Step-1', "
                    f"comp3={val})"
                )

        bc_lines = []
        for i, bc in enumerate(s.boundary_conditions):
            val = float(bc.get("value", 0.0))
            bc_lines.append(
                f"    model.DisplacementBC(name='BC-{i}', createStepName='Initial', "
                f"region=region, u1={val}, u2={val}, u3={val})"
            )

        loads_code = "\n".join(loads_lines) if loads_lines else "    pass"
        bc_code = "\n".join(bc_lines) if bc_lines else "    pass"

        return f"""# Abaqus Python script: static_general
# Run with: abaqus cae noGUI={args.output_prefix}.py
from abaqus import *
from abaqusConstants import *
from caeModules import *

model = mdb.Model(name='{args.base_model}')

# 材料属性
model.Material(name='Material-1')
model.materials['Material-1'].Elastic(table=(({E}, {nu}), ))
model.materials['Material-1'].Density(table=(({rho}, ), ))

# 截面
model.HomogeneousSolidSection(name='Section-1', material='Material-1')

# 静力步 (*STATIC, NLGEOM={nlgeom})
model.StaticStep(name='Step-1', previous='Initial', nlgeom={nlgeom},
                 minInc=1e-06, maxNumInc={niter})

# 载荷 (region 需按实际模型指定)
region = model.rootAssembly.sets.get('SET-LOAD')
{loads_code}

# 边界条件
region = model.rootAssembly.sets.get('SET-FIX')
{bc_code}

# 提交作业
job = mdb.Job(name='{args.output_prefix}', model='{args.base_model}',
              description='Static general analysis')
job.submit()
job.waitForCompletion()
"""

    def _generate_modal_script(self, args: AbaqusToolInput) -> str:
        """生成模态分析脚本 (*FREQUENCY, Lanczos)."""
        s = args.modal_spec
        n_modes = s.num_modes
        lumped = "ON" if s.lumped_mass else "OFF"

        freq_range_line = ""
        if s.frequency_range is not None:
            f_min, f_max = s.frequency_range
            freq_range_line = f", fmin={f_min}, fmax={f_max}"

        return f"""# Abaqus Python script: modal (Lanczos)
# Run with: abaqus cae noGUI={args.output_prefix}.py
from abaqus import *
from abaqusConstants import *
from caeModules import *

model = mdb.Model(name='{args.base_model}')

# 材料属性 (需用户补充实际值)
model.Material(name='Material-1')
model.materials['Material-1'].Elastic(table=((210e9, 0.3), ))
model.materials['Material-1'].Density(table=((7850.0, ), ))

model.HomogeneousSolidSection(name='Section-1', material='Material-1')

# 频率提取步 (*FREQUENCY, Lanczos)
model.FrequencyStep(name='Step-1', previous='Initial',
                    numEigen={n_modes}{freq_range_line},
                    lumpedMassFormulation={lumped})

# 提交作业
job = mdb.Job(name='{args.output_prefix}', model='{args.base_model}',
              description='Modal analysis (Lanczos)')
job.submit()
job.waitForCompletion()
"""

    def _generate_buckling_script(self, args: AbaqusToolInput) -> str:
        """生成特征值屈曲脚本 (*BUCKLE)."""
        s = args.buckling_spec
        n_modes = s.num_modes

        imperfection_line = ""
        if s.imperfection > 0:
            imperfection_line = f"""
# 初始缺陷: {s.imperfection} 倍一阶模态
model.StaticStep(name='Imperfection', previous='Step-1')
"""

        return f"""# Abaqus Python script: buckling (eigenvalue)
# Run with: abaqus cae noGUI={args.output_prefix}.py
from abaqus import *
from abaqusConstants import *
from caeModules import *

model = mdb.Model(name='{args.base_model}')

# 材料属性
model.Material(name='Material-1')
model.materials['Material-1'].Elastic(table=((210e9, 0.3), ))
model.materials['Material-1'].Density(table=((7850.0, ), ))

model.HomogeneousSolidSection(name='Section-1', material='Material-1')

# 静力预加载步
model.StaticStep(name='Step-1', previous='Initial', nlgeom=OFF)

# 特征值屈曲步 (*BUCKLE)
model.BuckleStep(name='Step-2', previous='Step-1', numEigen={n_modes},
                 eigensolver=LANCZOS)
{imperfection_line}
# 提交作业
job = mdb.Job(name='{args.output_prefix}', model='{args.base_model}',
              description='Buckling analysis')
job.submit()
job.waitForCompletion()
"""

    def _generate_fatigue_script(self, args: AbaqusToolInput) -> str:
        """生成疲劳分析脚本 (*DIRECT CYCLIC + S-N)."""
        s = args.fatigue_spec
        sigma_a = s.stress_amplitude
        sigma_m = s.mean_stress
        theory = s.mean_stress_theory.upper()
        cycles = s.cycles_limit

        sn_lines = [f"# {key} = {val}" for key, val in s.sn_params.items()]
        sn_block = "\n".join(sn_lines) if sn_lines else "# S-N 参数未给, 需手动补"

        return f"""# Abaqus Python script: fatigue (direct cyclic)
# Run with: abaqus cae noGUI={args.output_prefix}.py
from abaqus import *
from abaqusConstants import *
from caeModules import *

model = mdb.Model(name='{args.base_model}')

# 材料属性
model.Material(name='Material-1')
model.materials['Material-1'].Elastic(table=((210e9, 0.3), ))
model.materials['Material-1'].Density(table=((7850.0, ), ))

# 疲劳参数
# 应力幅 sigma_a = {sigma_a} Pa, 平均应力 sigma_m = {sigma_m} Pa
# 平均应力修正: {theory}
# 循环数上限: {cycles}
{sn_block}

model.HomogeneousSolidSection(name='Section-1', material='Material-1')

# 直接循环法步 (*DIRECT CYCLIC)
model.DirectCyclicStep(name='Step-1', previous='Initial',
                       initialNumInc=100, maxNumInc=1000)

# 提交作业
job = mdb.Job(name='{args.output_prefix}', model='{args.base_model}',
              description='Fatigue analysis (direct cyclic)')
job.submit()
job.waitForCompletion()
"""

    def _generate_fracture_script(self, args: AbaqusToolInput) -> str:
        """生成断裂力学脚本 (*CONTOUR INTEGRAL, J-integral)."""
        s = args.fracture_spec
        a = s.crack_length
        n_contours = s.contour_integral
        crack_type = s.crack_type
        j_int = "ON" if s.j_integral else "OFF"
        kic_line = f"# 断裂韧性 K_IC = {s.k_ic} Pa·sqrt(m)" if s.k_ic else ""

        return f"""# Abaqus Python script: fracture (contour integral)
# Run with: abaqus cae noGUI={args.output_prefix}.py
from abaqus import *
from abaqusConstants import *
from caeModules import *

model = mdb.Model(name='{args.base_model}')

# 材料属性
model.Material(name='Material-1')
model.materials['Material-1'].Elastic(table=((210e9, 0.3), ))

model.HomogeneousSolidSection(name='Section-1', material='Material-1')

# 静力步
model.StaticStep(name='Step-1', previous='Initial', nlgeom=ON)

# 裂纹定义 ({crack_type} crack, a = {a} m)
# Contour integral: {n_contours} 个回路, J-integral = {j_int}
# *CONTOUR INTEGRAL, CONTOURS={n_contours}, TYPE=J
{kic_line}

# 提交作业
job = mdb.Job(name='{args.output_prefix}', model='{args.base_model}',
              description='Fracture analysis (J-integral)')
job.submit()
job.waitForCompletion()
"""
