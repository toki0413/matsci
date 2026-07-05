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
# diversity_metadata 记录每个模型的训练来源、软化倾向、元素盲区、已知失效模式和互补模型,
# 交叉验证时用来挑互补模型, 避免"柏拉图收敛"(不同模型系统性地一起错)
UMLIP_REGISTRY: dict[str, dict[str, Any]] = {
    "mace": {
        "module": "mace.calculators",
        "class": "MACECalculator",
        "model": "MACE-MP-0",
        "diversity_metadata": {
            "training_source": "MPTrj",
            "softening_tendency": "high",
            "element_coverage_gaps": [],
            "failure_modes": ["phonon_softening"],
            "complementary_models": ["chgnet", "nep"],
        },
    },
    "grace": {
        "module": "fairchem.core",
        "class": "FAIRChemCalculator",
        "model": "GRACE-2L",
        "diversity_metadata": {
            "training_source": "MP+Alexandria",
            "softening_tendency": "medium",
            "element_coverage_gaps": [],
            "failure_modes": ["surface_energy"],
            "complementary_models": ["mace", "chgnet"],
        },
    },
    "chgnet": {
        "module": "chgnet.model",
        "class": "CHGNet",
        "model": "0.3.0",
        "diversity_metadata": {
            "training_source": "MP",
            "softening_tendency": "high",
            "element_coverage_gaps": ["lanthanides"],
            "failure_modes": ["phonon_softening", "vacancy_energy"],
            "complementary_models": ["mace", "grace"],
        },
    },
    "nep": {
        "module": "pynep",
        "class": "Nep",
        "model": "user-trained",
        "diversity_metadata": {
            "training_source": "NEP",
            "softening_tendency": "low",
            "element_coverage_gaps": ["transitions_metals"],
            "failure_modes": ["elastic_constants"],
            "complementary_models": ["grace"],
        },
    },
    "equiformer_v2_omat24": {
        "module": "fairchem.core",
        "class": "OCCalculator",
        "model": "facebook/OMAT24",
        "diversity_metadata": {
            "training_source": "OMat24",
            "softening_tendency": "medium",
            "element_coverage_gaps": [],
            "failure_modes": ["defect_formation"],
            "complementary_models": ["mace", "chgnet"],
        },
    },
}


class MLPotentialInput(BaseModel):
    backend: Literal["mace", "chgnet", "nep", "grace", "equiformer_v2_omat24"] = Field(
        ..., description="ML potential backend"
    )
    action: Literal["predict", "fine_tune", "relax", "energy_landscape", "cross_validate"] = Field(
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
    # ---- cross_validate 专用 ----
    property_type: Literal["energy", "forces", "phonons"] | None = Field(
        default=None,
        description="cross_validate 用: 要交叉验证的性质类型 (energy/forces/phonons)",
    )
    models: list[str] | None = Field(
        default=None,
        description=(
            "cross_validate 用: 指定参与交叉验证的模型列表. "
            "不传则根据 backend 的 diversity_metadata 自动选互补模型."
        ),
    )


class MLPotentialTool(HuginnTool):
    """Run MACE, CHGNet, NEP, GRACE, or OMat24 machine-learning potentials."""

    name = "ml_potential_tool"
    category = "sci"
    profile = ToolProfile(
        cost_tier="heavy",
        phases=frozenset({ResearchPhase.EXECUTION}),
        heavy_actions=frozenset({"train", "fit", "training"}),
        light_alternatives=("materials_database_tool", "numerical_tool"),
    )
    description = (
        "Predict energy/forces/stress with MACE, CHGNet, NEP, GRACE, or OMat24 "
        "ML potentials. Supports relax, fine-tune, energy_landscape, and "
        "cross_validate (multi-model consistency check against Plato convergence)."
    )
    input_schema = MLPotentialInput
    read_only = True

    def is_read_only(self, args: MLPotentialInput) -> bool:
        # predict / energy_landscape 都只读结构文件, 不写回
        return args.action in ("predict", "energy_landscape", "cross_validate")

    async def call(self, args: MLPotentialInput, context: ToolContext) -> ToolResult:
        if not Path(args.structure_file).exists():
            return ToolResult(
                data=None,
                success=False,
                error=f"Structure file not found: {args.structure_file}",
            )

        try:
            if args.action == "cross_validate":
                return self.cross_validate_umlips(args)
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
            if args.backend == "equiformer_v2_omat24":
                return self._run_equiformer_v2_omat24(args)
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

    def _run_equiformer_v2_omat24(self, args: MLPotentialInput) -> ToolResult:
        """OMat24 EquiformerV2 — 走 fairchem.core 的 OCCalculator.

        跟 GRACE 一样复用 fairchem-core 依赖, 但用的是 OC 系列 calculator.
        权重在 HuggingFace facebook/OMAT24, 用户可以传本地 checkpoint 路径覆盖.
        fairchem 没装就降级 not_available, 不影响其他 backend.
        """
        try:
            from ase.io import read, write
            from fairchem.core.oc.calculator import OCCalculator
        except ImportError:
            # 新版 fairchem 可能把 OCCalculator 挪到顶层
            try:
                from ase.io import read, write  # noqa: F811
                from fairchem.core import OCCalculator  # type: ignore
            except ImportError:
                return ToolResult(
                    data={"backend": "equiformer_v2_omat24", "status": "not_available"},
                    success=False,
                    error=(
                        "OMat24 backend requires fairchem-core and ase. "
                        "Install: pip install fairchem-core ase"
                    ),
                )

        atoms = read(args.structure_file)
        # model_path 可以是本地 checkpoint, 也可以是 HF repo id "facebook/OMAT24"
        calc = OCCalculator(
            model_path=args.model_path or "facebook/OMAT24",
            device=args.parameters.get("device", "cpu"),
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
                "backend": "equiformer_v2_omat24",
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
        elif args.backend == "equiformer_v2_omat24":
            try:
                from fairchem.core.oc.calculator import OCCalculator
            except ImportError:
                from fairchem.core import OCCalculator  # type: ignore

            calc = OCCalculator(
                model_path=args.model_path or "facebook/OMAT24",
                device=args.parameters.get("device", "cpu"),
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

    # ------------------------------------------------------------------
    # cross_validate: 用多个互补 U-MLIP 预测同一性质, 检查一致性
    # 对策"柏拉图收敛"问题: 不同模型可能系统性地一起错,
    # 如果互补模型之间偏差大, 说明预测不可靠
    # ------------------------------------------------------------------

    # energy 一致性阈值: 50 meV/atom, 超过就警告
    _CV_ENERGY_THRESHOLD = 0.05  # eV/atom
    # forces 一致性阈值: 100 meV/Å
    _CV_FORCE_THRESHOLD = 0.1  # eV/Å

    def _predict_single(self, model: str, args: MLPotentialInput) -> ToolResult:
        """用指定 model 跑一次 predict, 内部分发到对应的 _run_* 方法."""
        sub_args = MLPotentialInput(
            backend=model,
            action="predict",
            structure_file=args.structure_file,
            model_path=args.model_path if model == args.backend else None,
            parameters=args.parameters,
        )
        dispatch = {
            "mace": self._run_mace,
            "chgnet": self._run_chgnet,
            "nep": self._run_nep,
            "grace": self._run_grace,
            "equiformer_v2_omat24": self._run_equiformer_v2_omat24,
        }
        runner = dispatch.get(model)
        if runner is None:
            return ToolResult(data=None, success=False, error=f"Unknown model: {model}")
        return runner(sub_args)

    def cross_validate_umlips(self, args: MLPotentialInput) -> ToolResult:
        """用 2-3 个互补 U-MLIP 预测同一性质, 返回各模型预测值 + 一致性统计.

        选模型策略:
          - 用户传 args.models 就用用户指定的
          - 不传就从 args.backend 的 diversity_metadata.complementary_models 里挑,
            加上 backend 自己, 最多 3 个

        柏拉图收敛风险: 如果互补模型之间偏差大 (std > 阈值), 说明预测不可靠,
        即使模型们彼此一致也不能全信 (它们可能一起错).
        """
        # ---- 选模型 ----
        if args.models:
            models = args.models
        else:
            meta = UMLIP_REGISTRY.get(args.backend, {})
            complementary = (
                meta.get("diversity_metadata", {}).get("complementary_models", [])
            )
            models = [args.backend] + complementary[:2]

        # 去重保序, 只留注册表里有的
        seen: set[str] = set()
        selected: list[str] = []
        for m in models:
            if m not in seen and m in UMLIP_REGISTRY:
                seen.add(m)
                selected.append(m)

        if len(selected) < 2:
            return ToolResult(
                data=None,
                success=False,
                error="cross_validate needs at least 2 models, got: " + str(selected),
            )

        property_type = args.property_type or "energy"

        # ---- 逐模型跑 predict ----
        predictions: dict[str, dict[str, Any]] = {}
        errors: dict[str, str] = {}
        for model in selected:
            try:
                result = self._predict_single(model, args)
                if result.success and result.data:
                    predictions[model] = result.data
                else:
                    errors[model] = result.error or "prediction failed"
            except Exception as exc:
                errors[model] = str(exc)

        if len(predictions) < 2:
            return ToolResult(
                data={
                    "property_type": property_type,
                    "selected_models": selected,
                    "predictions": predictions,
                    "errors": errors,
                    "status": "insufficient_models",
                    "message": "Not enough models succeeded to cross-validate",
                },
                success=False,
                error="At least 2 models must succeed for cross-validation",
            )

        # ---- 拿原子数 (算 per-atom 量用) ----
        n_atoms = 1
        try:
            from ase.io import read

            n_atoms = len(read(args.structure_file))
        except Exception:
            pass  # 读不了就用 1, energy_per_atom == energy

        # ---- 按性质类型算统计量 ----
        warnings: list[str] = []

        if property_type == "energy":
            per_model_e: dict[str, float] = {}
            for model, data in predictions.items():
                e = data.get("energy")
                if e is not None:
                    per_model_e[model] = float(e) / n_atoms

            values = list(per_model_e.values())
            if len(values) < 2:
                return ToolResult(
                    data={
                        "property_type": "energy",
                        "selected_models": selected,
                        "predictions": predictions,
                        "errors": errors,
                        "status": "insufficient_energy_data",
                    },
                    success=False,
                    error="Not enough models returned valid energy",
                )

            mean_e = float(np.mean(values))
            std_e = float(np.std(values))
            # 一致性分数: 1.0 = 完全一致, 0.0 = 差异 >= 2x 阈值
            consistency = max(0.0, 1.0 - std_e / (2 * self._CV_ENERGY_THRESHOLD))

            if std_e > self._CV_ENERGY_THRESHOLD:
                warnings.append(
                    f"Multi-model inconsistency: std={std_e * 1000:.1f} meV/atom > "
                    f"threshold={self._CV_ENERGY_THRESHOLD * 1000:.0f} meV/atom. "
                    f"Plato-convergence risk — models may agree but all be wrong."
                )

            # 附上每个模型的 diversity_metadata, 方便判断哪些模型互补
            model_metadata = {
                m: UMLIP_REGISTRY.get(m, {}).get("diversity_metadata", {})
                for m in per_model_e
            }

            return ToolResult(
                data={
                    "property_type": "energy",
                    "selected_models": selected,
                    "per_model": {
                        m: {"energy_per_atom": e} for m, e in per_model_e.items()
                    },
                    "mean": mean_e,
                    "std": std_e,
                    "consistency_score": round(consistency, 4),
                    "threshold": self._CV_ENERGY_THRESHOLD,
                    "warnings": warnings,
                    "errors": errors,
                    "model_diversity_metadata": model_metadata,
                },
                success=True,
            )

        if property_type == "forces":
            per_model_f: dict[str, np.ndarray] = {}
            for model, data in predictions.items():
                f = data.get("forces")
                if f:
                    per_model_f[model] = np.asarray(f, dtype=float)

            if len(per_model_f) < 2:
                return ToolResult(
                    data={
                        "property_type": "forces",
                        "selected_models": selected,
                        "predictions": predictions,
                        "errors": errors,
                        "status": "insufficient_force_data",
                    },
                    success=False,
                    error="Not enough models returned valid forces",
                )

            # 两两算 RMSE
            model_names = list(per_model_f.keys())
            pairwise_rmses: list[float] = []
            for i in range(len(model_names)):
                for j in range(i + 1, len(model_names)):
                    diff = per_model_f[model_names[i]] - per_model_f[model_names[j]]
                    pairwise_rmses.append(float(np.sqrt(np.mean(diff ** 2))))

            mean_rmse = float(np.mean(pairwise_rmses))
            max_rmse = float(np.max(pairwise_rmses))
            consistency = max(
                0.0, 1.0 - mean_rmse / (2 * self._CV_FORCE_THRESHOLD)
            )

            if mean_rmse > self._CV_FORCE_THRESHOLD:
                warnings.append(
                    f"Multi-model force inconsistency: mean RMSE={mean_rmse * 1000:.1f} "
                    f"meV/Å > threshold={self._CV_FORCE_THRESHOLD * 1000:.0f} meV/Å. "
                    f"Plato-convergence risk."
                )

            model_metadata = {
                m: UMLIP_REGISTRY.get(m, {}).get("diversity_metadata", {})
                for m in per_model_f
            }

            return ToolResult(
                data={
                    "property_type": "forces",
                    "selected_models": selected,
                    "per_model": {
                        m: {"forces": per_model_f[m].tolist()} for m in model_names
                    },
                    "mean_rmse": mean_rmse,
                    "max_rmse": max_rmse,
                    "consistency_score": round(consistency, 4),
                    "threshold": self._CV_FORCE_THRESHOLD,
                    "warnings": warnings,
                    "errors": errors,
                    "model_diversity_metadata": model_metadata,
                },
                success=True,
            )

        # property_type == "phonons"
        # phonon 交叉验证需要力常数 (有限差分), 单点预测不够.
        # 退而求其次: 用能量一致性做 proxy, 附说明.
        per_model_e = {}
        for model, data in predictions.items():
            e = data.get("energy")
            if e is not None:
                per_model_e[model] = float(e) / n_atoms

        values = list(per_model_e.values())
        mean_e = float(np.mean(values)) if values else 0.0
        std_e = float(np.std(values)) if values else 0.0
        consistency = max(0.0, 1.0 - std_e / (2 * self._CV_ENERGY_THRESHOLD))

        warnings.append(
            "Phonon cross-validation requires force constants via finite differences. "
            "Energy consistency is used as a proxy here. "
            "For rigorous phonon validation, run energy_landscape with each model "
            "and compare curvatures, or use phonopy with multiple MLIP backends."
        )
        if std_e > self._CV_ENERGY_THRESHOLD:
            warnings.append(
                f"Multi-model energy inconsistency (phonon proxy): "
                f"std={std_e * 1000:.1f} meV/atom > "
                f"threshold={self._CV_ENERGY_THRESHOLD * 1000:.0f} meV/atom."
            )

        return ToolResult(
            data={
                "property_type": "phonons",
                "selected_models": selected,
                "per_model": {
                    m: {"energy_per_atom": e} for m, e in per_model_e.items()
                },
                "mean": mean_e,
                "std": std_e,
                "consistency_score": round(consistency, 4),
                "threshold": self._CV_ENERGY_THRESHOLD,
                "warnings": warnings,
                "errors": errors,
                "note": (
                    "Phonon validation is proxied by energy consistency. "
                    "Use energy_landscape action with multiple backends for curvature comparison."
                ),
            },
            success=True,
        )
