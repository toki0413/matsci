"""Variational autoencoder tool.

Trains a vanilla VAE on tabular data and exposes encode / decode / sample.
Pure PyTorch, standard reparameterization trick. torch is imported lazily so
the module still loads when torch is not installed.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool, ResearchPhase, ToolProfile
from huginn.types import ToolContext, ToolResult


class VAEToolInput(BaseModel):
    action: Literal["train", "encode", "decode", "sample"] = Field(default="train")
    X: list[list[float]] = Field(
        default_factory=list, description="Training data, rows x features"
    )
    X_encode: list[list[float]] = Field(
        default_factory=list, description="Data to encode"
    )
    Z: list[list[float]] = Field(
        default_factory=list, description="Latent vectors to decode"
    )
    latent_dim: int = Field(default=8, gt=0, description="Latent space dimension")
    hidden_dim: int = Field(default=64, gt=0, description="Hidden layer width")
    n_layers: int = Field(
        default=2, ge=1, description="Encoder/decoder hidden layer count"
    )
    input_dim: int | None = Field(
        default=None, gt=0, description="Feature count, required for decode/sample"
    )
    epochs: int = Field(default=50, ge=1)
    lr: float = Field(default=1e-3, gt=0)
    batch_size: int = Field(default=32, ge=1)
    n_samples: int = Field(default=10, ge=1, description="Samples for the sample action")
    state_dict: dict[str, Any] | None = Field(
        default=None, description="Trained weights returned by the train action"
    )
    seed: int | None = Field(default=None)


def _build_vae(input_dim: int, latent_dim: int, hidden_dim: int, n_layers: int):
    """Build a fresh VAE nn.Module. Imports torch here, not at module load."""
    import torch
    import torch.nn as nn

    class VAE(nn.Module):
        def __init__(self):
            super().__init__()
            enc, in_d = [], input_dim
            for _ in range(n_layers):
                enc += [nn.Linear(in_d, hidden_dim), nn.ReLU()]
                in_d = hidden_dim
            self.encoder = nn.Sequential(*enc)
            self.fc_mu = nn.Linear(in_d, latent_dim)
            self.fc_logvar = nn.Linear(in_d, latent_dim)
            dec, in_d = [], latent_dim
            for _ in range(n_layers):
                dec += [nn.Linear(in_d, hidden_dim), nn.ReLU()]
                in_d = hidden_dim
            self.decoder = nn.Sequential(*dec, nn.Linear(in_d, input_dim))

        def encode(self, x):
            h = self.encoder(x)
            return self.fc_mu(h), self.fc_logvar(h)

        def reparameterize(self, mu, logvar):
            std = (0.5 * logvar).exp()
            return mu + torch.randn_like(std) * std

        def decode(self, z):
            return self.decoder(z)

        def forward(self, x):
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            return self.decode(z), mu, logvar

    return VAE()


def _vae_loss(recon_x, x, mu, logvar):
    """MSE reconstruction + analytical KL to N(0, I)."""
    import torch
    import torch.nn.functional as F  # noqa: N812

    mse = F.mse_loss(recon_x, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return mse + kl


class VAETool(HuginnTool):
    """Variational autoencoder for tabular data (train / encode / decode / sample)."""

    name = "vae_tool"
    category = "sci"
    profile = ToolProfile(
        cost_tier="heavy",
        phases=frozenset({ResearchPhase.EXECUTION}),
        heavy_actions=frozenset({"train"}),
    )
    description = (
        "Train a variational autoencoder on tabular data, then encode, decode, "
        "or sample from the latent space. Returns model state_dict, loss curve, "
        "and latent-space statistics."
    )
    input_schema = VAEToolInput

    # ponytail: vanilla VAE only — no β-VAE, no conditional VAE, no VampPrior.
    # If the agent needs a variant it can scale the KL term (β) or concatenate a
    # condition tensor to x and z itself; this wrapper stays minimal on purpose.

    def call(
        self, args: dict[str, Any], context: ToolContext | None = None
    ) -> ToolResult:
        try:
            import torch  # noqa: F401
        except ImportError:
            return ToolResult(
                data=None, success=False, error="torch not available"
            )

        data = VAEToolInput(**args)
        try:
            if data.action == "train":
                return self._train(data)
            if data.action == "encode":
                return self._encode(data)
            if data.action == "decode":
                return self._decode(data)
            if data.action == "sample":
                return self._sample(data)
        except Exception as e:
            return ToolResult(data=None, success=False, error=f"VAE tool failed: {e}")
        return ToolResult(
            data=None, success=False, error=f"Unknown action: {data.action}"
        )

    def _train(self, args: VAEToolInput) -> ToolResult:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        if not args.X:
            return ToolResult(data=None, success=False, error="train requires X.")
        X = np.asarray(args.X, dtype=np.float32)
        input_dim = int(X.shape[1])
        if args.seed is not None:
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)

        from huginn.tools.sci import get_torch_device
        _dev = get_torch_device()
        model = _build_vae(input_dim, args.latent_dim, args.hidden_dim, args.n_layers)
        model = model.to(_dev)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(X)),
            batch_size=args.batch_size,
            shuffle=True,
        )

        loss_curve: list[float] = []
        for _ in range(args.epochs):
            total = 0.0
            for (xb,) in loader:
                xb = xb.to(_dev)
                opt.zero_grad()
                recon, mu, logvar = model(xb)
                loss = _vae_loss(recon, xb, mu, logvar) / xb.size(0)
                loss.backward()
                opt.step()
                total += loss.item() * xb.size(0)
            loss_curve.append(total / len(X))

        # latent stats over the full training set
        with torch.no_grad():
            mu_all, _ = model.encode(torch.from_numpy(X).to(_dev))
            latent_mean = mu_all.mean(dim=0).tolist()
            latent_var = mu_all.var(dim=0).tolist()

        return ToolResult(
            data={
                "state_dict": {k: v.tolist() for k, v in model.state_dict().items()},
                "loss_curve": loss_curve,
                "latent_mean": latent_mean,
                "latent_var": latent_var,
                "input_dim": input_dim,
                "latent_dim": args.latent_dim,
                "hidden_dim": args.hidden_dim,
                "n_layers": args.n_layers,
                "n_train": len(X),
                "message": (
                    f"VAE trained {args.epochs} epoch(s), "
                    f"final loss {loss_curve[-1]:.6f}."
                ),
            },
            success=True,
        )

    def _load_model(self, args: VAEToolInput, input_dim: int):
        """Rebuild the architecture and load weights. Returns model or error result."""
        import torch

        if not args.state_dict:
            return ToolResult(
                data=None,
                success=False,
                error=f"{args.action} requires state_dict from a previous train call.",
            )
        from huginn.tools.sci import get_torch_device
        _dev = get_torch_device()
        model = _build_vae(input_dim, args.latent_dim, args.hidden_dim, args.n_layers)
        sd = {
            k: torch.tensor(v, dtype=torch.float32)
            for k, v in args.state_dict.items()
        }
        model.load_state_dict(sd)
        model = model.to(_dev)
        model.eval()
        return model

    @staticmethod
    def _is_error(m: Any) -> bool:
        return isinstance(m, ToolResult)

    def _encode(self, args: VAEToolInput) -> ToolResult:
        import torch

        if not args.X_encode:
            return ToolResult(
                data=None, success=False, error="encode requires X_encode."
            )
        X = np.asarray(args.X_encode, dtype=np.float32)
        model = self._load_model(args, int(X.shape[1]))
        if self._is_error(model):
            return model  # type: ignore[return-value]
        from huginn.tools.sci import get_torch_device
        _dev = get_torch_device()
        with torch.no_grad():
            mu, logvar = model.encode(torch.from_numpy(X).to(_dev))
        return ToolResult(
            data={
                "z_mean": mu.tolist(),
                "z_logvar": logvar.tolist(),
                "message": f"Encoded {len(X)} samples to latent dim {args.latent_dim}.",
            },
            success=True,
        )

    def _decode(self, args: VAEToolInput) -> ToolResult:
        import torch

        if not args.Z:
            return ToolResult(data=None, success=False, error="decode requires Z.")
        if args.input_dim is None:
            return ToolResult(
                data=None, success=False, error="decode requires input_dim."
            )
        Z = np.asarray(args.Z, dtype=np.float32)
        model = self._load_model(args, args.input_dim)
        if self._is_error(model):
            return model  # type: ignore[return-value]
        from huginn.tools.sci import get_torch_device
        _dev = get_torch_device()
        with torch.no_grad():
            recon = model.decode(torch.from_numpy(Z).to(_dev))
        return ToolResult(
            data={
                "X_recon": recon.tolist(),
                "message": f"Decoded {len(Z)} latent vectors.",
            },
            success=True,
        )

    def _sample(self, args: VAEToolInput) -> ToolResult:
        import torch

        if args.input_dim is None:
            return ToolResult(
                data=None, success=False, error="sample requires input_dim."
            )
        if args.seed is not None:
            torch.manual_seed(args.seed)
        model = self._load_model(args, args.input_dim)
        if self._is_error(model):
            return model  # type: ignore[return-value]
        from huginn.tools.sci import get_torch_device
        _dev = get_torch_device()
        with torch.no_grad():
            z = torch.randn(args.n_samples, args.latent_dim, device=_dev)
            samples = model.decode(z)
        return ToolResult(
            data={
                "samples": samples.tolist(),
                "message": (
                    f"Sampled {args.n_samples} points from N(0, I) latent prior."
                ),
            },
            success=True,
        )


# ── self-check: train 1 epoch on make_classification data ────────────
if __name__ == "__main__":
    try:
        import torch  # noqa: F401
    except ImportError:
        print("torch not available, skip demo")
        raise SystemExit(0) from None  # noqa: B904

    from sklearn.datasets import make_classification

    X, _ = make_classification(n_samples=200, n_features=20, random_state=0)
    tool = VAETool()

    res = tool.call({
        "action": "train",
        "X": X.tolist(),
        "latent_dim": 4,
        "hidden_dim": 32,
        "n_layers": 2,
        "epochs": 1,
        "batch_size": 32,
        "seed": 0,
    })
    assert res.success, res.error
    d = res.data
    assert "state_dict" in d
    assert "loss_curve" in d
    assert "latent_mean" in d
    assert "latent_var" in d
    assert len(d["loss_curve"]) == 1
    assert len(d["latent_mean"]) == d["latent_dim"]

    # round-trip: encode -> decode, shapes must line up
    enc = tool.call({
        "action": "encode",
        "X_encode": X[:5].tolist(),
        "state_dict": d["state_dict"],
        "latent_dim": d["latent_dim"],
        "hidden_dim": d["hidden_dim"],
        "n_layers": d["n_layers"],
    })
    assert enc.success, enc.error
    assert len(enc.data["z_mean"]) == 5

    dec = tool.call({
        "action": "decode",
        "Z": enc.data["z_mean"],
        "state_dict": d["state_dict"],
        "latent_dim": d["latent_dim"],
        "hidden_dim": d["hidden_dim"],
        "n_layers": d["n_layers"],
        "input_dim": d["input_dim"],
    })
    assert dec.success, dec.error
    assert len(dec.data["X_recon"]) == 5

    smp = tool.call({
        "action": "sample",
        "n_samples": 3,
        "state_dict": d["state_dict"],
        "latent_dim": d["latent_dim"],
        "hidden_dim": d["hidden_dim"],
        "n_layers": d["n_layers"],
        "input_dim": d["input_dim"],
        "seed": 1,
    })
    assert smp.success, smp.error
    assert len(smp.data["samples"]) == 3

    print("[vae_tool] self-check OK, train loss =", round(d["loss_curve"][0], 4))
