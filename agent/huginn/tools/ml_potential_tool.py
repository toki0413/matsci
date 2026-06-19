"""Machine-learning potential tool (MACE, CHGNet, NEP).

Provides a unified interface for energy/force prediction and fine-tuning with
popular ML potentials. Heavy dependencies are imported lazily so the agent can
still start when they are not installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool
from huginn.types import ToolContext, ToolResult


class MLPotentialInput(BaseModel):
    backend: Literal["mace", "chgnet", "nep"] = Field(
        ..., description="ML potential backend"
    )
    action: Literal["predict", "fine_tune", "relax"] = Field(default="predict")
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


class MLPotentialTool(HuginnTool):
    """Run MACE, CHGNet, or NEP machine-learning potentials."""

    name = "ml_potential_tool"
    description = (
        "Predict energy/forces/stress with MACE, CHGNet, or NEP ML potentials "
        "and optionally relax or fine-tune structures."
    )
    input_schema = MLPotentialInput
    read_only = True

    def is_read_only(self, args: MLPotentialInput) -> bool:
        return args.action == "predict"

    async def call(self, args: MLPotentialInput, context: ToolContext) -> ToolResult:
        if not Path(args.structure_file).exists():
            return ToolResult(
                data=None,
                success=False,
                error=f"Structure file not found: {args.structure_file}",
            )

        try:
            if args.backend == "mace":
                return self._run_mace(args)
            if args.backend == "chgnet":
                return self._run_chgnet(args)
            if args.backend == "nep":
                return self._run_nep(args)
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
