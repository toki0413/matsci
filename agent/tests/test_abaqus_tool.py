"""Tests for the native Abaqus helper tool."""

from pathlib import Path

from huginn.tools.abaqus_tool import AbaqusTool, AbaqusToolInput


def test_abaqus_tool_import_packing_reference_points(tmp_path: Path) -> None:
    """AbaqusTool import_packing should generate a script with reference points."""
    tool = AbaqusTool()
    packing_data = {
        "objects": [
            {"id": 0, "center": [1.0, 2.0, 3.0], "radius": 0.5, "symbol": "Si"},
            {"id": 1, "center": [4.0, 5.0, 6.0], "radius": 0.8, "symbol": "Si"},
        ]
    }
    result = tool.call(
        {
            "action": "import_packing",
            "packing_data": packing_data,
            "particle_shape": "reference_point",
            "output_prefix": "particle_abaqus",
            "working_dir": str(tmp_path),
        }
    )
    assert result.success is True
    script_path = Path(result.data["script_path"])
    assert script_path.exists()
    script = script_path.read_text(encoding="utf-8")
    assert "ReferencePoint" in script
    assert "Particle_" in script
    assert "1.000000, 2.000000, 3.000000" in script


def test_abaqus_tool_import_packing_spheres(tmp_path: Path) -> None:
    """AbaqusTool import_packing with sphere shape should generate sphere parts."""
    tool = AbaqusTool()
    packing_data = {
        "objects": [
            {"id": 0, "center": [0.0, 0.0, 0.0], "radius": 1.0, "symbol": "Si"},
        ]
    }
    result = tool.call(
        {
            "action": "import_packing",
            "packing_data": packing_data,
            "particle_shape": "sphere",
            "output_prefix": "sphere_abaqus",
            "working_dir": str(tmp_path),
        }
    )
    assert result.success is True
    script = Path(result.data["script_path"]).read_text(encoding="utf-8")
    assert "BaseSolidRevolve" in script


def test_abaqus_tool_input_schema() -> None:
    """AbaqusToolInput should accept valid parameters."""
    inp = AbaqusToolInput(
        action="import_packing",
        particle_shape="sphere",
        base_model="Model-1",
    )
    assert inp.particle_shape == "sphere"
    assert inp.base_model == "Model-1"


def test_abaqus_tool_static_general(tmp_path: Path) -> None:
    """static_general action 应生成包含 StaticStep + nlgeom=ON 的脚本."""
    tool = AbaqusTool()
    result = tool.call(
        {
            "action": "static_general",
            "static_spec": {
                "youngs_modulus": 210e9,
                "poissons_ratio": 0.3,
                "density": 7850.0,
                "nlgeom": True,
                "max_iterations": 50,
                "loads": [
                    {"type": "pressure", "value": 1e6},
                    {"type": "gravity", "value": 9.81},
                ],
                "boundary_conditions": [
                    {"region": "fix", "dofs": [1, 2, 3], "value": 0.0}
                ],
            },
            "output_prefix": "static_test",
            "working_dir": str(tmp_path),
        }
    )
    assert result.success is True
    script = Path(result.data["script_path"]).read_text(encoding="utf-8")
    assert "StaticStep" in script
    assert "nlgeom=ON" in script
    assert "maxNumInc=50" in script
    assert "Pressure" in script
    assert "Gravity" in script
    assert "DisplacementBC" in script


def test_abaqus_tool_modal(tmp_path: Path) -> None:
    """modal action 应生成包含 FrequencyStep + numEigen 的脚本."""
    tool = AbaqusTool()
    result = tool.call(
        {
            "action": "modal",
            "modal_spec": {
                "num_modes": 5,
                "lumped_mass": False,
            },
            "output_prefix": "modal_test",
            "working_dir": str(tmp_path),
        }
    )
    assert result.success is True
    script = Path(result.data["script_path"]).read_text(encoding="utf-8")
    assert "FrequencyStep" in script
    assert "numEigen=5" in script
    assert "lumpedMassFormulation=OFF" in script


def test_abaqus_tool_buckling(tmp_path: Path) -> None:
    """buckling action 应生成包含 BuckleStep + LANCZOS 的脚本."""
    tool = AbaqusTool()
    result = tool.call(
        {
            "action": "buckling",
            "buckling_spec": {
                "num_modes": 3,
                "imperfection": 0.0,
            },
            "output_prefix": "buckling_test",
            "working_dir": str(tmp_path),
        }
    )
    assert result.success is True
    script = Path(result.data["script_path"]).read_text(encoding="utf-8")
    assert "BuckleStep" in script
    assert "numEigen=3" in script
    assert "LANCZOS" in script
    # 预加载静力步
    assert "StaticStep" in script


def test_abaqus_tool_fatigue(tmp_path: Path) -> None:
    """fatigue action 应生成包含 DirectCyclicStep 的脚本."""
    tool = AbaqusTool()
    result = tool.call(
        {
            "action": "fatigue",
            "fatigue_spec": {
                "sn_params": {"sigma_f_prime": 1100e6, "b": -0.1},
                "mean_stress_theory": "goodman",
                "cycles_limit": 100000,
                "stress_amplitude": 200e6,
                "mean_stress": 50e6,
            },
            "output_prefix": "fatigue_test",
            "working_dir": str(tmp_path),
        }
    )
    assert result.success is True
    script = Path(result.data["script_path"]).read_text(encoding="utf-8")
    assert "DirectCyclicStep" in script
    assert "sigma_f_prime" in script  # S-N 参数注释
    assert "GOODMAN" in script  # 平均应力修正


def test_abaqus_tool_fracture(tmp_path: Path) -> None:
    """fracture action 应生成包含 Contour Integral + J-integral 的脚本."""
    tool = AbaqusTool()
    result = tool.call(
        {
            "action": "fracture",
            "fracture_spec": {
                "crack_type": "edge",
                "crack_length": 0.01,
                "contour_integral": 5,
                "j_integral": True,
                "k_ic": 80e6,
            },
            "output_prefix": "fracture_test",
            "working_dir": str(tmp_path),
        }
    )
    assert result.success is True
    script = Path(result.data["script_path"]).read_text(encoding="utf-8")
    script_lower = script.lower()
    assert "contour integral" in script_lower
    assert "CONTOURS=5" in script
    assert "J-integral" in script
    assert "K_IC" in script  # 断裂韧性注释
