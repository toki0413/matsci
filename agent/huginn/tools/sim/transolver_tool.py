"""Transolver++ PDE surrogate solver tool.

Wraps the Transolver++ neural operator (github.com/thuml/Transolver_plus, MIT)
as a PDE surrogate: predict fields on arbitrary meshes, fine-tune on new
data, and list available checkpoints. Torch and the transolver package are
imported lazily, so the tool loads even when they're absent — those code
paths return a helpful install hint instead of crashing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool, ResearchPhase, ToolProfile
from huginn.types import ToolContext, ToolResult

logger = logging.getLogger(__name__)


# Transolver++ ships the model class as `Model` under models/Transolver_plus.py.
# Try a few import paths so a local clone, a pip install, or a PYTHONPATH entry
# all resolve.
_TRANSOLVER_IMPORT_PATHS = (
    ("transolver_plus.models.Transolver_plus", "Model"),
    ("models.Transolver_plus", "Model"),
    ("Transolver_plus.models.Transolver_plus", "Model"),
)

_INSTALL_HINT = (
    "Transolver++ is not importable. Install from source:\n"
    "  git clone https://github.com/thuml/Transolver_plus\n"
    "  cd Transolver_plus && pip install -e .\n"
    "It also needs torch, timm, and einops (see its requirements.txt)."
)


class TransolverToolInput(BaseModel):
    action: Literal["predict", "train", "list_models"] = Field(
        ...,
        description=(
            "predict: run inference on a mesh; "
            "train: fine-tune on new data; "
            "list_models: list available checkpoints"
        ),
    )
    model_name: str = Field(
        default="default",
        description="Checkpoint name (stem, no extension) under the transolver model dir",
    )
    # Mesh data, inlined as lists (matches the gp_tool / dynamics pattern).
    # coords: (n_nodes, space_dim), features: (n_nodes, fun_dim)
    coords: list[list[float]] = Field(
        default_factory=list,
        description="Node coordinates, shape (n_nodes, space_dim)",
    )
    features: list[list[float]] = Field(
        default_factory=list,
        description="Per-node input features, shape (n_nodes, fun_dim)",
    )
    target: list[list[float]] = Field(
        default_factory=list,
        description="Ground-truth field for training, shape (n_nodes, out_dim)",
    )
    condition: list[float] | None = Field(
        default=None,
        description="Optional per-batch conditioning vector (length must match the "
        "model embedding input; Transolver++ uses 3 by default)",
    )
    # Model config — mirrors the Transolver++ Model constructor.
    space_dim: int = Field(default=3, ge=1, description="Spatial dimension of the mesh")
    n_hidden: int = Field(default=256, ge=8)
    n_layers: int = Field(default=5, ge=1)
    n_head: int = Field(default=8, ge=1)
    fun_dim: int = Field(default=1, ge=1, description="Input feature channels per node")
    out_dim: int = Field(default=1, ge=1, description="Output field channels per node")
    slice_num: int = Field(default=32, ge=1)
    # Training hyperparameters
    epochs: int = Field(default=10, ge=1, le=10000)
    learning_rate: float = Field(default=1e-4, gt=0)
    device: str = Field(default="cuda", description="cuda | cpu (falls back to cpu if cuda is unavailable)")
    checkpoint_dir: str | None = Field(
        default=None,
        description="Override checkpoint dir (defaults to workspace/.huginn/models/transolver)",
    )


class TransolverToolOutput(BaseModel):
    status: Literal["ok", "error", "no_models"] = "ok"
    predictions: list[list[float]] = []
    available_models: list[str] = []
    checkpoint_path: str | None = None
    train_loss: list[float] = []
    warnings: list[str] = []


class TransolverTool(HuginnTool):
    """Transolver++ neural PDE surrogate: predict / train / list_models."""

    name = "transolver_tool"
    category = "sim"
    profile = ToolProfile(
        cost_tier="heavy",
        phases=frozenset({ResearchPhase.EXECUTION}),
        constraint_scope="pde",
        light_alternatives=("numerical_tool", "symbolic_math_tool"),
        # train is the only action that burns serious GPU; predict/list are cheap
        heavy_actions=frozenset({"train"}),
    )
    description = (
        "Transolver++ neural PDE surrogate. Predict fields on arbitrary meshes, "
        "fine-tune on new data, and manage checkpoints. Degrades gracefully when "
        "torch/transolver are not installed."
    )
    input_schema = TransolverToolInput
    _init_kwargs_map = {"workspace": "workspace"}

    def __init__(self, workspace: str | None = None):
        super().__init__()
        self._workspace = workspace
        self._torch_ok: bool | None = None
        self._model_cls: Any = None

    # ── lazy dependency checks ────────────────────────────────────

    def _resolve_model_dir(self, args: TransolverToolInput) -> Path:
        if args.checkpoint_dir:
            d = Path(args.checkpoint_dir)
        elif self._workspace:
            d = Path(self._workspace) / ".huginn" / "models" / "transolver"
        else:
            d = Path.cwd() / ".huginn" / "models" / "transolver"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _check_torch(self) -> bool:
        if self._torch_ok is not None:
            return self._torch_ok
        try:
            import torch  # noqa: F401

            self._torch_ok = True
        except ImportError:
            self._torch_ok = False
        return self._torch_ok

    def _load_model_class(self) -> Any:
        """Lazily resolve the Transolver++ Model class. Returns the class or None."""
        if self._model_cls is not None:
            return self._model_cls
        import importlib

        for mod_path, cls_name in _TRANSOLVER_IMPORT_PATHS:
            try:
                mod = importlib.import_module(mod_path)
                cls = getattr(mod, cls_name, None)
                if cls is not None:
                    self._model_cls = cls
                    return cls
            except Exception:
                continue
        return None

    def _build_model(self, args: TransolverToolInput):
        """Construct a Transolver++ model from the input config, return (model, device)."""
        import torch

        Model = self._load_model_class()
        if Model is None:
            raise RuntimeError("Transolver++ Model class not found on the import paths")
        model = Model(
            space_dim=args.space_dim,
            n_layers=args.n_layers,
            n_hidden=args.n_hidden,
            n_head=args.n_head,
            fun_dim=args.fun_dim,
            out_dim=args.out_dim,
            slice_num=args.slice_num,
        )
        # The model's get_grid hardcodes .cuda() only under unified_pos, which we
        # leave off; cpu is safe otherwise.
        device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
        return model.to(device), device

    @staticmethod
    def _to_tensor(arr, device, dtype):
        import torch

        return torch.tensor(arr, dtype=dtype, device=device)

    # ── entry point ───────────────────────────────────────────────

    async def call(self, args: TransolverToolInput, context: ToolContext) -> ToolResult:
        # list_models is pure filesystem — no torch needed.
        if args.action == "list_models":
            return self._list_models(args)
        # predict / train need torch + the transolver package.
        if not self._check_torch():
            return ToolResult(data=None, success=False, error=_INSTALL_HINT)
        if args.action == "predict":
            return self._predict(args)
        if args.action == "train":
            return self._train(args)
        return ToolResult(data=None, success=False, error=f"unknown action: {args.action}")

    # ── actions ───────────────────────────────────────────────────

    def _list_models(self, args: TransolverToolInput) -> ToolResult:
        model_dir = self._resolve_model_dir(args)
        models = sorted({
            p.stem for pat in ("*.pt", "*.pth") for p in model_dir.glob(pat)
        })
        out = TransolverToolOutput(
            status="ok" if models else "no_models",
            available_models=models,
        )
        return ToolResult(data=out.model_dump(), success=True)

    def _predict(self, args: TransolverToolInput) -> ToolResult:
        if not args.coords or not args.features:
            return ToolResult(
                data=None, success=False,
                error="predict needs coords and features",
            )
        if self._load_model_class() is None:
            return ToolResult(data=None, success=False, error=_INSTALL_HINT)

        try:
            import torch
            model, device = self._build_model(args)
        except Exception as e:
            return ToolResult(data=None, success=False, error=f"model build failed: {e}")

        model_dir = self._resolve_model_dir(args)
        ckpt_path = model_dir / f"{args.model_name}.pt"
        if not ckpt_path.exists():
            ckpt_path = model_dir / f"{args.model_name}.pth"
        if not ckpt_path.exists():
            return ToolResult(
                data=None, success=False,
                error=f"checkpoint '{args.model_name}' not found in {model_dir}",
            )
        try:
            state = torch.load(ckpt_path, map_location=device)
            sd = state.get("model", state) if isinstance(state, dict) else state
            model.load_state_dict(sd, strict=False)
            model.eval()
        except Exception as e:
            return ToolResult(data=None, success=False, error=f"checkpoint load failed: {e}")

        try:
            x = self._to_tensor(args.features, device, torch.float32).unsqueeze(0)
            pos = self._to_tensor(args.coords, device, torch.float32).unsqueeze(0)
            cond = None
            if args.condition is not None:
                cond = torch.tensor(
                    [args.condition], dtype=torch.float32, device=device
                )
            with torch.no_grad():
                pred = model((x, pos, cond))
            preds = pred.squeeze(0).cpu().tolist()
        except Exception as e:
            return ToolResult(data=None, success=False, error=f"inference failed: {e}")

        out = TransolverToolOutput(
            status="ok",
            predictions=preds,
            checkpoint_path=str(ckpt_path),
        )
        data = out.model_dump()

        # Physics audit — NaN in predictions, checkpoint mismatch
        try:
            from huginn.execution.physics_auditor import PhysicsAuditor

            auditor = PhysicsAuditor()
            audit_report = auditor.audit("transolver_tool", args.action, data, args.model_dump())
            data["physics_audit"] = audit_report.to_dict()
        except Exception:
            logger.debug("audit failure can't block result delivery", exc_info=True)

        return ToolResult(data=data, success=True)

    def _train(self, args: TransolverToolInput) -> ToolResult:
        if not args.coords or not args.features or not args.target:
            return ToolResult(
                data=None, success=False,
                error="train needs coords, features, and target",
            )
        if self._load_model_class() is None:
            return ToolResult(data=None, success=False, error=_INSTALL_HINT)

        try:
            import torch
            import torch.nn as nn
            model, device = self._build_model(args)
        except Exception as e:
            return ToolResult(data=None, success=False, error=f"model build failed: {e}")

        # Warm-start from an existing checkpoint if one exists.
        model_dir = self._resolve_model_dir(args)
        ckpt_path = model_dir / f"{args.model_name}.pt"
        warm_started = False
        if ckpt_path.exists():
            try:
                state = torch.load(ckpt_path, map_location=device)
                sd = state.get("model", state) if isinstance(state, dict) else state
                model.load_state_dict(sd, strict=False)
                warm_started = True
            except Exception:
                logger.debug("load failed", exc_info=True)

        try:
            x = self._to_tensor(args.features, device, torch.float32).unsqueeze(0)
            pos = self._to_tensor(args.coords, device, torch.float32).unsqueeze(0)
            y = self._to_tensor(args.target, device, torch.float32).unsqueeze(0)
            cond = None
            if args.condition is not None:
                cond = torch.tensor(
                    [args.condition], dtype=torch.float32, device=device
                )
        except Exception as e:
            return ToolResult(data=None, success=False, error=f"tensor prep failed: {e}")

        opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        loss_fn = nn.MSELoss()
        model.train()
        losses: list[float] = []
        for _ in range(args.epochs):
            opt.zero_grad()
            pred = model((x, pos, cond))
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

        save_path = model_dir / f"{args.model_name}.pt"
        try:
            torch.save({"model": model.state_dict(), "config": args.model_dump()}, save_path)
        except Exception as e:
            return ToolResult(data=None, success=False, error=f"save failed: {e}")

        warnings: list[str] = []
        if not warm_started:
            warnings.append("trained from scratch (no warm-start checkpoint)")
        out = TransolverToolOutput(
            status="ok",
            checkpoint_path=str(save_path),
            train_loss=losses,
            warnings=warnings,
        )
        data = out.model_dump()

        # Physics audit — loss divergence, NaN/Inf, gradient explosion
        try:
            from huginn.execution.physics_auditor import PhysicsAuditor

            auditor = PhysicsAuditor()
            audit_report = auditor.audit("transolver_tool", args.action, data, args.model_dump())
            data["physics_audit"] = audit_report.to_dict()
        except Exception:
            logger.debug("audit failure can't block result delivery", exc_info=True)

        return ToolResult(data=data, success=True)

    def estimate_cost(self, args: TransolverToolInput) -> dict[str, float] | None:
        if args.action == "train":
            return {"gpu_hours": args.epochs * 0.05, "walltime_hours": args.epochs * 0.05}
        if args.action == "predict":
            return {"gpu_hours": 0.01, "walltime_hours": 0.01}
        return None
