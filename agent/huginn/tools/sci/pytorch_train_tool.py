"""Generic PyTorch training wrapper.

Trains a small MLP or CNN for classification, returns weights + train curve
+ metrics. Uses standard nn.Module / optim / DataLoader. CPU-only by design.

torch is an optional import — if it's missing the tool reports unavailable
and self-check skips cleanly instead of crashing the registry.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool, ResearchPhase, ToolProfile
from huginn.types import ToolContext, ToolResult

# ponytail: import torch lazily so the module loads even without torch.
# The registry can still import this file; is_available() gates real use.
try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only on torch-less envs
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    TensorDataset = None  # type: ignore[assignment]

from huginn.tools.sci import get_torch_device

_ACTIVATIONS: dict[str, Any] = {
    "relu": lambda: nn.ReLU(),
    "tanh": lambda: nn.Tanh(),
    "sigmoid": lambda: nn.Sigmoid(),
    "gelu": lambda: nn.GELU(),
}


class PyTorchTrainToolInput(BaseModel):
    action: Literal["train_mlp", "train_cnn", "evaluate"] = Field(
        default="train_mlp", description="Training or evaluation action"
    )
    # MLP tabular data
    X: list[list[float]] = Field(
        default_factory=list, description="MLP input features, shape [N, D]"
    )
    # CNN image data, shape [N, C, H, W]
    X_images: list[list[list[list[float]]]] = Field(
        default_factory=list, description="CNN image batch as nested lists"
    )
    y: list[int] = Field(default_factory=list, description="Class labels")
    # MLP architecture
    input_dim: int = Field(default=0, gt=0, description="MLP input feature dim")
    hidden_dims: list[int] = Field(
        default_factory=lambda: [64, 32], description="MLP hidden layer widths"
    )
    activation: Literal["relu", "tanh", "sigmoid", "gelu"] = "relu"
    # CNN architecture
    image_channels: int = Field(default=1, gt=0)
    image_size: int = Field(default=28, gt=0, description="Square image side length")
    conv_channels: list[int] = Field(
        default_factory=lambda: [16, 32],
        description="CNN conv output channels (2-3 layers recommended)",
    )
    kernel_size: int = Field(default=3, gt=0)
    n_classes: int = Field(default=2, gt=0)
    # Hyperparams
    epochs: int = Field(default=10, ge=1)
    lr: float = Field(default=1e-3, gt=0)
    batch_size: int = Field(default=32, ge=1)
    seed: int = Field(default=0)
    # Weights I/O
    weights: dict[str, list] | None = Field(
        default=None,
        description="Inline state_dict (name -> nested list) for evaluate / warm-start",
    )
    weights_path: str | None = Field(
        default=None,
        description="Load state_dict from a .pt file instead of inline weights",
    )
    save_path: str | None = Field(
        default=None,
        description="If set, save trained state_dict here and return the path",
    )
    working_dir: str | None = Field(default=None)


class PyTorchTrainTool(HuginnTool):
    """Train a small MLP/CNN classifier with PyTorch and evaluate it."""

    name = "pytorch_train_tool"
    category = "sci"
    profile = ToolProfile(phases=frozenset({ResearchPhase.VALIDATION}))
    description = (
        "Train a configurable MLP or CNN classifier with PyTorch, or evaluate "
        "a saved model on a test set. Returns weights, per-epoch loss curve, "
        "and accuracy."
    )
    input_schema = PyTorchTrainToolInput

    def __init__(self) -> None:
        super().__init__()

    def is_available(self) -> bool:
        return _TORCH_AVAILABLE

    # ── public entry ──────────────────────────────────────────────

    def call(
        self, args: dict[str, Any], context: ToolContext | None = None
    ) -> ToolResult:
        if not _TORCH_AVAILABLE:
            return ToolResult(
                data=None, success=False, error="torch not available"
            )
        try:
            parsed = PyTorchTrainToolInput(**args)
            torch.manual_seed(parsed.seed)
            # ponytail: assume CPU — agent picks the device and migrates
            # the state_dict itself if it wants GPU. Adding device auto-detect
            # here would hide CUDA-availability bugs from the caller.
            if parsed.action == "train_mlp":
                return self._train_mlp(parsed)
            if parsed.action == "train_cnn":
                return self._train_cnn(parsed)
            return self._evaluate(parsed)
        except Exception as e:
            return ToolResult(
                data=None, success=False, error=f"pytorch_train_tool failed: {e}"
            )

    # ── model builders ────────────────────────────────────────────

    @staticmethod
    def _build_mlp(
        input_dim: int,
        hidden_dims: list[int],
        n_classes: int,
        activation: str,
    ) -> nn.Module:
        act_fn = _ACTIVATIONS.get(activation, _ACTIVATIONS["relu"])
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(act_fn())
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        return nn.Sequential(*layers)

    @staticmethod
    def _build_cnn(
        in_channels: int,
        image_size: int,
        conv_channels: list[int],
        kernel_size: int,
        n_classes: int,
    ) -> nn.Module:
        # ponytail: simple VGG-style stack — Conv/ReLU/MaxPool × len(conv_channels),
        # then a single FC head. No BatchNorm/Dropout; agent can swap a custom
        # module in if it needs regularization. Ceiling: degrades on deep nets
        # (>4 conv) where BN starts to matter; upgrade path = expose a full
        # layer-spec list.
        pad = kernel_size // 2
        layers: list[nn.Module] = []
        prev_c = in_channels
        spatial = image_size
        for c in conv_channels:
            layers.append(nn.Conv2d(prev_c, c, kernel_size, padding=pad))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2))
            prev_c = c
            spatial = max(spatial // 2, 1)
        feat = nn.Sequential(*layers)
        head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(prev_c * spatial * spatial, n_classes),
        )
        return nn.Sequential(feat, head)

    # ── train / eval core ─────────────────────────────────────────

    @staticmethod
    def _make_loader(X: torch.Tensor, y: torch.Tensor, batch_size: int,
                     shuffle: bool) -> DataLoader:
        ds = TensorDataset(X, y)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    @staticmethod
    def _train_loop(model: nn.Module, loader: DataLoader,
                    epochs: int, lr: float, device: str = "cpu") -> list[float]:
        # ponytail: no early stopping — agent inspects train_curve and
        # decides whether to retrain with fewer epochs. Adding patience
        # here would hide the raw curve from the caller.
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        model.train()
        curve: list[float] = []
        for _ in range(epochs):
            total, n = 0.0, 0
            for Xb, yb in loader:
                Xb = Xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(Xb), yb)
                loss.backward()
                optimizer.step()
                total += loss.item() * len(Xb)
                n += len(Xb)
            curve.append(total / max(n, 1))
        return curve

    @staticmethod
    def _eval_loop(model: nn.Module, loader: DataLoader,
                   device: str = "cpu") -> dict[str, Any]:
        model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss, correct, total = 0.0, 0, 0
        preds: list[int] = []
        with torch.no_grad():
            for Xb, yb in loader:
                Xb = Xb.to(device)
                yb = yb.to(device)
                out = model(Xb)
                total_loss += criterion(out, yb).item() * len(Xb)
                pred = out.argmax(dim=1)
                correct += (pred == yb).sum().item()
                total += len(yb)
                preds.extend(pred.tolist())
        return {
            "loss": total_loss / max(total, 1),
            "accuracy": correct / max(total, 1),
            "predictions": preds,
        }

    @staticmethod
    def _state_to_lists(model: nn.Module) -> dict[str, list]:
        return {
            k: v.detach().cpu().numpy().tolist()
            for k, v in model.state_dict().items()
        }

    @staticmethod
    def _load_state(model: nn.Module, weights: dict[str, list]) -> None:
        state = model.state_dict()
        loaded = {
            k: torch.tensor(v, dtype=state[k].dtype) if k in state else torch.tensor(v)
            for k, v in weights.items()
        }
        model.load_state_dict(loaded, strict=True)

    def _resolve_weights(self, model: nn.Module,
                         args: PyTorchTrainToolInput) -> ToolResult | None:
        if args.weights_path:
            state = torch.load(args.weights_path, map_location="cpu")
            model.load_state_dict(state)
            return None
        if args.weights:
            self._load_state(model, args.weights)
            return None
        return ToolResult(
            data=None,
            success=False,
            error="evaluate requires weights or weights_path",
        )

    def _finalize_weights(self, model: nn.Module,
                          args: PyTorchTrainToolInput) -> dict[str, Any]:
        if args.save_path:
            path = Path(args.save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), path)
            return {"saved_to": str(path)}
        return {"weights": self._state_to_lists(model)}

    # ── actions ───────────────────────────────────────────────────

    def _train_mlp(self, args: PyTorchTrainToolInput) -> ToolResult:
        if not args.X or not args.y:
            return ToolResult(
                data=None, success=False,
                error="train_mlp requires non-empty X and y",
            )
        if len(args.X) != len(args.y):
            return ToolResult(
                data=None, success=False,
                error="X and y must have the same length",
            )
        X = torch.tensor(args.X, dtype=torch.float32)
        y = torch.tensor(args.y, dtype=torch.long)
        model = self._build_mlp(
            args.input_dim, args.hidden_dims, args.n_classes, args.activation
        )
        _dev = get_torch_device()
        model = model.to(_dev)
        loader = self._make_loader(X, y, args.batch_size, shuffle=True)
        curve = self._train_loop(model, loader, args.epochs, args.lr, _dev)

        train = self._eval_loop(model, loader, _dev)
        weight_out = self._finalize_weights(model, args)
        return ToolResult(
            data={
                **weight_out,
                "train_curve": curve,
                "metrics": {
                    "train_loss": train["loss"],
                    "train_accuracy": train["accuracy"],
                },
                "architecture": {
                    "type": "mlp",
                    "input_dim": args.input_dim,
                    "hidden_dims": args.hidden_dims,
                    "activation": args.activation,
                    "n_classes": args.n_classes,
                },
                "epochs": args.epochs,
                "lr": args.lr,
                "batch_size": args.batch_size,
            },
            success=True,
        )

    def _train_cnn(self, args: PyTorchTrainToolInput) -> ToolResult:
        if not args.X_images or not args.y:
            return ToolResult(
                data=None, success=False,
                error="train_cnn requires non-empty X_images and y",
            )
        if len(args.X_images) != len(args.y):
            return ToolResult(
                data=None, success=False,
                error="X_images and y must have the same length",
            )
        X = torch.tensor(args.X_images, dtype=torch.float32)
        y = torch.tensor(args.y, dtype=torch.long)
        model = self._build_cnn(
            args.image_channels, args.image_size,
            args.conv_channels, args.kernel_size, args.n_classes,
        )
        _dev = get_torch_device()
        model = model.to(_dev)
        loader = self._make_loader(X, y, args.batch_size, shuffle=True)
        curve = self._train_loop(model, loader, args.epochs, args.lr, _dev)

        train = self._eval_loop(model, loader, _dev)
        weight_out = self._finalize_weights(model, args)
        return ToolResult(
            data={
                **weight_out,
                "train_curve": curve,
                "metrics": {
                    "train_loss": train["loss"],
                    "train_accuracy": train["accuracy"],
                },
                "architecture": {
                    "type": "cnn",
                    "image_channels": args.image_channels,
                    "image_size": args.image_size,
                    "conv_channels": args.conv_channels,
                    "kernel_size": args.kernel_size,
                    "n_classes": args.n_classes,
                },
                "epochs": args.epochs,
                "lr": args.lr,
                "batch_size": args.batch_size,
            },
            success=True,
        )

    def _evaluate(self, args: PyTorchTrainToolInput) -> ToolResult:
        if args.X and not args.X_images:
            if args.input_dim <= 0:
                return ToolResult(
                    data=None, success=False,
                    error="evaluate on MLP requires input_dim",
                )
            X = torch.tensor(args.X, dtype=torch.float32)
            model = self._build_mlp(
                args.input_dim, args.hidden_dims,
                args.n_classes, args.activation,
            )
        elif args.X_images and not args.X:
            X = torch.tensor(args.X_images, dtype=torch.float32)
            model = self._build_cnn(
                args.image_channels, args.image_size,
                args.conv_channels, args.kernel_size, args.n_classes,
            )
        else:
            return ToolResult(
                data=None, success=False,
                error="evaluate requires exactly one of X (MLP) or X_images (CNN)",
            )

        if not args.y:
            return ToolResult(
                data=None, success=False,
                error="evaluate requires y labels",
            )
        y = torch.tensor(args.y, dtype=torch.long)

        err = self._resolve_weights(model, args)
        if err is not None:
            return err

        loader = self._make_loader(X, y, args.batch_size, shuffle=False)
        _dev = get_torch_device()
        model = model.to(_dev)
        metrics = self._eval_loop(model, loader, _dev)
        return ToolResult(
            data={
                "metrics": {"loss": metrics["loss"],
                            "accuracy": metrics["accuracy"]},
                "predictions": metrics["predictions"],
                "n_samples": len(args.y),
            },
            success=True,
        )


# ── self-check ───────────────────────────────────────────────────
# Run: python -m huginn.tools.sci.pytorch_train_tool
# Verifies the train/evaluate round-trip on a tiny sklearn dataset.
# If torch isn't installed, prints a skip line and exits 0 so the
# registry import doesn't blow up in CI.

if __name__ == "__main__":
    if not _TORCH_AVAILABLE:
        print("torch not available, skip demo")
        raise SystemExit(0)

    from sklearn.datasets import make_classification

    X_np, y_np = make_classification(
        n_samples=200, n_features=20, n_informative=10,
        n_classes=2, random_state=0,
    )

    tool = PyTorchTrainTool()

    # 1-epoch MLP train
    res = tool.call({
        "action": "train_mlp",
        "X": X_np.tolist(),
        "y": y_np.tolist(),
        "input_dim": 20,
        "hidden_dims": [32, 16],
        "activation": "relu",
        "n_classes": 2,
        "epochs": 1,
        "batch_size": 32,
        "lr": 1e-3,
        "seed": 0,
    })
    assert res.success, f"train_mlp failed: {res.error}"
    assert len(res.data["train_curve"]) == 1, "train_curve should have 1 entry"
    assert 0.0 <= res.data["metrics"]["train_accuracy"] <= 1.0
    assert "weights" in res.data, "inline weights missing"
    print("train_mlp OK:", res.data["metrics"])

    # evaluate with the trained weights
    res2 = tool.call({
        "action": "evaluate",
        "X": X_np.tolist(),
        "y": y_np.tolist(),
        "input_dim": 20,
        "hidden_dims": [32, 16],
        "n_classes": 2,
        "weights": res.data["weights"],
        "batch_size": 64,
    })
    assert res2.success, f"evaluate failed: {res2.error}"
    assert len(res2.data["predictions"]) == len(y_np)
    assert 0.0 <= res2.data["metrics"]["accuracy"] <= 1.0
    print("evaluate OK:", res2.data["metrics"])

    # tiny CNN sanity: 8 random 1×8×8 images, 2 classes
    rng = np.random.default_rng(0)
    Xcnn = rng.standard_normal((8, 1, 8, 8)).tolist()
    ycnn = rng.integers(0, 2, size=8).tolist()
    res3 = tool.call({
        "action": "train_cnn",
        "X_images": Xcnn,
        "y": ycnn,
        "image_channels": 1,
        "image_size": 8,
        "conv_channels": [4, 8],
        "kernel_size": 3,
        "n_classes": 2,
        "epochs": 1,
        "batch_size": 4,
    })
    assert res3.success, f"train_cnn failed: {res3.error}"
    assert "weights" in res3.data
    print("train_cnn OK:", res3.data["metrics"])

    print("OK: pytorch_train_tool self-check passed")
