"""Machine-learning potential tool (MACE, CHGNet, NEP).

Provides a unified interface for energy/force prediction and fine-tuning with
popular ML potentials. Heavy dependencies are imported lazily so the agent can
still start when they are not installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool, ResearchPhase, ToolProfile
from huginn.types import ToolContext, ToolResult


# 统一 MLIP 注册表: agent 可以查哪些 U-MLIP 可用 + 怎么导入
UMLIP_REGISTRY: dict[str, dict[str, str]] = {
    "mace": {"module": "mace.calculators", "class": "MACECalculator", "model": "MACE-MP-0"},
    "grace": {"module": "fairchem.core", "class": "FAIRChemCalculator", "model": "GRACE-2L"},
    "chgnet": {"module": "chgnet.model", "class": "CHGNet", "model": "0.3.0"},
    "nep": {"module": "pynep", "class": "Nep", "model": "user-trained"},
}


class MLPotentialInput(BaseModel):
    backend: Literal["mace", "chgnet", "nep", "grace"] = Field(
        ..., description="ML potential backend"
    )
    action: Literal["predict", "fine_tune", "relax", "energy_landscape"] = Field(
        default="predict"
    )
    structure_file: str = Field(..., description="Path to structure file")
    model_path: str | None = Field(
        default=None,
        description="Path to a trained model; uses built-in pretrained model if omitted",
    )
    output_path: str | None = Field(
        default=None, description="Where to save the relaxed structure"
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Backend-specific kwargs (max_steps, fmax, epochs, etc.)",
    )
    # ---- energy_landscape 专用 ----
    perturbation_directions: list[list[int]] | None = Field(
        default=None,
        description=(
            "energy_landscape 用: 指定扰动方向列表, 每项 [atom_idx, axis(0/1/2)]. "
            "不传就走随机方向."
        ),
    )
    n_samples: int = Field(
        default=10,
        ge=1,
        description="energy_landscape 采样数 (随机方向时生效)",
    )
    displacement_magnitude: float = Field(
        default=0.1,
        ge=1e-4,
        description="energy_landscape 每次扰动的位移幅度 (Å)",
    )


class MLPotentialTool(HuginnTool):
    """Run MACE, CHGNet, or NEP machine-learning potentials."""

    name = "ml_potential_tool"
    category = "sci"
    profile = ToolProfile(
        cost_tier="heavy",
        phases=frozenset({ResearchPhase.EXECUTION}),
        heavy_actions=frozenset({"train", "fit", "training"}),
        light_alternatives=("materials_database_tool", "numerical_tool"),
    )
    description = (
        "Predict energy/forces/stress with MACE, CHGNet, or NEP ML potentials "
        "and optionally relax or fine-tune structures."
    )
    input_schema = MLPotentialInput
    read_only = True

    def is_read_only(self, args: MLPotentialInput) -> bool:
        # predict / energy_landscape 都只读结构文件, 不写回
        return args.action in ("predict", "energy_landscape")

    async def call(self, args: MLPotentialInput, context: ToolContext) -> ToolResult:
        if not Path(args.structure_file).exists():
            return ToolResult(
                data=None,
                success=False,
                error=f"Structure file not found: {args.structure_file}",
            )

        try:
            if args.action == "energy_landscape":
                return self._run_energy_landscape(args)
            if args.action == "fine_tune":
                return self._run_fine_tune(args)
            if args.backend == "mace":
                return self._run_mace(args)
            if args.backend == "chgnet":
                return self._run_chgnet(args)
            if args.backend == "nep":
                return self._run_nep(args)
            if args.backend == "grace":
                return self._run_grace(args)
        except Exception as exc:
            return ToolResult(data=None, success=False, error=str(exc))

        return ToolResult(
            data=None, success=False, error=f"Unknown backend: {args.backend}"
        )

    def _run_mace(self, args: MLPotentialInput) -> ToolResult:
        try:
            from ase.io import read, write
            from ase.optimize import BFGS
            from mace.calculators import MACECalculator
        except ImportError as exc:
            raise RuntimeError(
                "MACE backend requires mace-torch and ase. "
                "Install: pip install mace-torch ase"
            ) from exc

        atoms = read(args.structure_file)
        calc = MACECalculator(
            model_paths=args.model_path or "small",
            device=args.parameters.get("device", "cpu"),
            default_dtype=args.parameters.get("default_dtype", "float64"),
        )
        atoms.calc = calc

        if args.action == "relax":
            fmax = float(args.parameters.get("fmax", 0.05))
            max_steps = int(args.parameters.get("max_steps", 500))
            opt = BFGS(atoms)
            opt.run(fmax=fmax, steps=max_steps)
            if args.output_path:
                write(args.output_path, atoms)

        energy = float(atoms.get_potential_energy())
        forces = atoms.get_forces().tolist()
        stress = atoms.get_stress(voigt=True).tolist()
        return ToolResult(
            data={
                "backend": "mace",
                "action": args.action,
                "energy": energy,
                "forces": forces,
                "stress": stress,
                "output_path": args.output_path,
            }
        )

    def _run_chgnet(self, args: MLPotentialInput) -> ToolResult:
        try:
            from ase.io import read, write
            from chgnet.model import CHGNet, StructOptimizer
        except ImportError as exc:
            raise RuntimeError(
                "CHGNet backend requires chgnet and ase. "
                "Install: pip install chgnet ase"
            ) from exc

        atoms = read(args.structure_file)
        model = CHGNet.load(args.model_path) if args.model_path else CHGNet.load()
        prediction = model.predict_structure(atoms)

        if args.action == "relax":
            relaxer = StructOptimizer()
            result = relaxer.relax(
                atoms,
                fmax=float(args.parameters.get("fmax", 0.05)),
                steps=int(args.parameters.get("max_steps", 500)),
            )
            atoms = result["final_structure"]
            if args.output_path:
                write(args.output_path, atoms)
            energy = float(result["trajectory"].energies[-1])
        else:
            energy = float(prediction["e"])

        return ToolResult(
            data={
                "backend": "chgnet",
                "action": args.action,
                "energy": energy,
                "forces": prediction.get("f", []),
                "stress": prediction.get("s", []),
                "magmom": prediction.get("m", []),
                "output_path": args.output_path,
            }
        )

    def _run_nep(self, args: MLPotentialInput) -> ToolResult:
        try:
            from ase.io import read, write
            from pynep import Nep
        except ImportError as exc:
            raise RuntimeError(
                "NEP backend requires pynep and ase. " "Install: pip install pynep ase"
            ) from exc

        atoms = read(args.structure_file)
        nep = Nep(args.model_path or "nep.txt")
        atoms.calc = nep

        if args.action == "relax":
            from ase.optimize import BFGS

            fmax = float(args.parameters.get("fmax", 0.05))
            max_steps = int(args.parameters.get("max_steps", 500))
            BFGS(atoms).run(fmax=fmax, steps=max_steps)
            if args.output_path:
                write(args.output_path, atoms)

        return ToolResult(
            data={
                "backend": "nep",
                "action": args.action,
                "energy": float(atoms.get_potential_energy()),
                "forces": atoms.get_forces().tolist(),
                "stress": atoms.get_stress(voigt=True).tolist(),
                "output_path": args.output_path,
            }
        )

    def _run_grace(self, args: MLPotentialInput) -> ToolResult:
        """GRACE-2L 走 fairchem-core. 装了才能用, 没装降级 not_available."""
        try:
            from ase.io import read, write
            from fairchem.core import FAIRChemCalculator  # noqa: F401
        except ImportError as exc:
            return ToolResult(
                data={"backend": "grace", "status": "not_available"},
                success=False,
                error=(
                    "GRACE backend requires fairchem-core and ase. "
                    "Install: pip install fairchem-core ase"
                ),
            )

        atoms = read(args.structure_file)
        calc = FAIRChemCalculator(
            model=args.model_path or "GRACE-2L",
            task=args.parameters.get("task", "omol"),
        )
        atoms.calc = calc

        if args.action == "relax":
            from ase.optimize import BFGS

            fmax = float(args.parameters.get("fmax", 0.05))
            max_steps = int(args.parameters.get("max_steps", 500))
            BFGS(atoms).run(fmax=fmax, steps=max_steps)
            if args.output_path:
                write(args.output_path, atoms)

        return ToolResult(
            data={
                "backend": "grace",
                "action": args.action,
                "energy": float(atoms.get_potential_energy()),
                "forces": atoms.get_forces().tolist(),
                "stress": atoms.get_stress(voigt=True).tolist(),
                "output_path": args.output_path,
            }
        )

    def _run_fine_tune(self, args: MLPotentialInput) -> ToolResult:
        """Fine-tune an MLIP.  MACE is wired to its ``mace.finetune.run`` API;
        other backends don't expose a programmatic fine-tune entry point yet."""
        if args.backend == "mace":
            try:
                from mace.finetune import run as run_finetune
            except ImportError:
                return ToolResult(
                    data={
                        "backend": "mace",
                        "action": "fine_tune",
                        "status": "not_available",
                        "message": (
                            "MACE fine-tuning requires the mace.finetune module. "
                            "Install a recent mace-torch build that ships it."
                        ),
                    },
                    success=False,
                    error="mace.finetune not importable",
                )

            config_path = args.parameters.get("config_path")
            if not config_path:
                return ToolResult(
                    data={
                        "backend": "mace",
                        "action": "fine_tune",
                        "status": "missing_config",
                        "message": (
                            "MACE fine-tune needs a YAML config file. "
                            "Pass parameters.config_path pointing to the training config."
                        ),
                    },
                    success=False,
                    error="parameters.config_path is required for MACE fine-tune",
                )

            cli_args = ["--config", str(config_path)]
            for key in ("device", "seed", "default_dtype", "log_level"):
                val = args.parameters.get(key)
                if val is not None:
                    cli_args += [f"--{key}", str(val)]

            import traceback
            try:
                run_finetune(cli_args)
                return ToolResult(
                    data={
                        "backend": "mace",
                        "action": "fine_tune",
                        "status": "completed",
                        "config_path": str(config_path),
                        "message": "MACE fine-tune finished. Check the log for the model checkpoint path.",
                    },
                    success=True,
                )
            except SystemExit:
                return ToolResult(
                    data={
                        "backend": "mace",
                        "action": "fine_tune",
                        "status": "completed",
                        "config_path": str(config_path),
                        "message": "MACE fine-tune finished (argparse exit).",
                    },
                    success=True,
                )
            except Exception as exc:
                return ToolResult(
                    data={
                        "backend": "mace",
                        "action": "fine_tune",
                        "status": "failed",
                        "config_path": str(config_path),
                        "traceback": traceback.format_exc()[-1000:],
                    },
                    success=False,
                    error=f"MACE fine-tune failed: {exc}",
                )

        if args.backend == "chgnet":
            return ToolResult(
                data={
                    "backend": "chgnet",
                    "action": "fine_tune",
                    "status": "not_supported",
                    "message": (
                        "CHGNet fine-tuning is not exposed as a programmatic API. "
                        "Use the chgnet.train.Trainer CLI directly: "
                        "python -m chgnet.train --train-set train.json --val-set val.json"
                    ),
                },
                success=False,
                error="fine_tune not supported for chgnet via this tool",
            )

        return ToolResult(
            data={
                "backend": args.backend,
                "action": "fine_tune",
                "status": "not_supported",
                "message": (
                    f"fine_tune is not implemented for backend '{args.backend}'. "
                    f"Supported backends: mace (requires config_path)."
                ),
            },
            success=False,
            error=f"fine_tune not supported for {args.backend}",
        )

    # ------------------------------------------------------------------
    # energy_landscape: 沿指定/随机方向扰动结构, 采样能量地形
    # ------------------------------------------------------------------

    def _attach_calculator(self, atoms, args):
        """按 backend 给 atoms 挂上 ASE calculator, energy_landscape 复用."""
        if args.backend == "mace":
            from mace.calculators import MACECalculator

            calc = MACECalculator(
                model_paths=args.model_path or "small",
                device=args.parameters.get("device", "cpu"),
                default_dtype=args.parameters.get("default_dtype", "float64"),
            )
        elif args.backend == "chgnet":
            from chgnet.model import CHGNet
            from chgnet.model.dynamics import CHGNetCalculator

            model = CHGNet.load(args.model_path) if args.model_path else CHGNet.load()
            calc = CHGNetCalculator(model)
        elif args.backend == "nep":
            from pynep import Nep

            calc = Nep(args.model_path or "nep.txt")
        elif args.backend == "grace":
            from fairchem.core import FAIRChemCalculator

            calc = FAIRChemCalculator(
                model=args.model_path or "GRACE-2L",
                task=args.parameters.get("task", "omol"),
            )
        else:
            raise RuntimeError(f"Unknown backend: {args.backend}")
        atoms.calc = calc
        return atoms

    def _run_energy_landscape(self, args: MLPotentialInput) -> ToolResult:
        """沿随机/指定方向微扰结构, 用 ML 势评估能量, 输出能量地形采样点.

        每个采样点记录: displacement (Å, 相对初始位置的总位移矢量展平)、
        energy (eV)、force (eV/Å, 当前构型受力展平). 不修改原结构文件.
        """
        try:
            from ase.io import read
        except ImportError as exc:
            raise RuntimeError(
                "energy_landscape requires ase. Install: pip install ase"
            ) from exc

        atoms = read(args.structure_file)
        self._attach_calculator(atoms, args)

        positions0 = atoms.get_positions().copy()
        n_atoms = len(atoms)
        magnitude = float(args.displacement_magnitude)

        # 准备扰动方向列表: 用户指定的优先, 否则随机生成
        directions = []
        if args.perturbation_directions:
            for atom_idx, axis in args.perturbation_directions:
                if not (0 <= atom_idx < n_atoms) or axis not in (0, 1, 2):
                    raise RuntimeError(
                        f"perturbation_directions 含非法项 [{atom_idx}, {axis}]"
                    )
                vec = np.zeros((n_atoms, 3))
                vec[atom_idx, axis] = 1.0
                directions.append(vec)
        else:
            rng = np.random.default_rng(42)
            for _ in range(int(args.n_samples)):
                vec = rng.standard_normal((n_atoms, 3))
                norm = np.linalg.norm(vec)
                if norm < 1e-12:
                    vec = np.zeros((n_atoms, 3))
                    vec[0, 0] = 1.0
                    norm = 1.0
                vec = vec / norm
                directions.append(vec)

        samples: list[dict[str, Any]] = []
        # 第一个采样点用原始位置做基准 (零位移)
        try:
            e0 = float(atoms.get_potential_energy())
            f0 = atoms.get_forces()
        except Exception as exc:
            raise RuntimeError(f"ML 势评估失败: {exc}") from exc

        samples.append(
            {
                "displacement": [0.0] * (3 * n_atoms),
                "displacement_magnitude": 0.0,
                "energy": e0,
                "force": np.asarray(f0).ravel().tolist(),
            }
        )

        # 沿每个方向扰动一次 (固定幅度), 评估并复位
        for vec in directions:
            atoms.set_positions(positions0 + magnitude * vec)
            try:
                e = float(atoms.get_potential_energy())
                f = atoms.get_forces()
            except Exception as exc:
                # 单点失败就跳过, 不让整个 landscape 挂掉
                continue
            disp = (magnitude * vec).ravel().tolist()
            samples.append(
                {
                    "displacement": disp,
                    "displacement_magnitude": float(magnitude),
                    "energy": e,
                    "force": np.asarray(f).ravel().tolist(),
                }
            )

        # 复位, 避免影响后续调用 (虽然本方法不写回文件, 但 atoms 可能被复用)
        atoms.set_positions(positions0)

        energies = [s["energy"] for s in samples]
        e_min = min(energies)
        e_max = max(energies)

        return ToolResult(
            data={
                "backend": args.backend,
                "action": "energy_landscape",
                "structure_file": args.structure_file,
                "n_samples": len(samples),
                "displacement_magnitude": magnitude,
                "samples": samples,
                "energy_range": [e_min, e_max],
                "energy_span": e_max - e_min,
            },
            success=True,
        )
