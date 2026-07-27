"""Small-scale Transformer training/prediction wrapper.

Trains a tiny encoder-only Transformer (next-token LM style) on token-id
sequences and returns state_dict + loss curve; or reloads a state_dict and
predicts tokens. Hard cap of 10M parameters — beyond that the agent should
dispatch to HPC, this wrapper is the wrong tool.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool, ResearchPhase, ToolProfile
from huginn.types import ToolContext, ToolResult

# torch is optional — keep the module importable even when it's missing so
# tool registration doesn't crash; the call() path returns a clean error.
try:
    import torch
    from torch import nn

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on env
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

from huginn.tools.sci import get_torch_device


# ponytail: 10M param ceiling — RCB-scale sequence tasks fit here; for real
# LLM-scale work the agent dispatches to HPC, this wrapper is not the tool.
_MAX_PARAMS = 10_000_000


class TransformerToolInput(BaseModel):
    action: Literal["train", "predict"] = Field(default="train")
    # Token-id sequences (B, T). For train: input batch. For predict: prompts
    # to extend autoregressively.
    sequences: list[list[int]] = Field(default_factory=list)
    # Optional explicit targets (B, T). If empty, next-token shift of
    # `sequences` is used.
    targets: list[list[int]] = Field(default_factory=list)
    # Model spec
    vocab_size: int = Field(default=32, gt=0)
    d_model: int = Field(default=64, gt=0)
    nhead: int = Field(default=4, gt=0)
    num_layers: int = Field(default=2, gt=0)
    dim_ff: int = Field(default=128, gt=0)
    max_seq_len: int = Field(default=64, gt=0)
    # Training
    epochs: int = Field(default=1, ge=1)
    lr: float = Field(default=3e-4, gt=0)
    batch_size: int = Field(default=8, ge=1)
    seed: int = Field(default=0)
    # Predict: state_dict from a prior train call (tensors flattened to lists).
    state_dict: dict[str, list[float]] | None = Field(default=None)
    predict_len: int = Field(default=8, gt=0)


def _count_params(model: "nn.Module") -> int:
    return sum(p.numel() for p in model.parameters())


# Gate the model class on torch availability so the module imports cleanly
# even when torch is missing — call()/is_available() handle the runtime path.
if _TORCH_AVAILABLE:

    class _SeqLM(nn.Module):
        """Encoder-only Transformer LM: token + positional embedding, encoder, linear head.

        ponytail: no FlashAttention, no KV cache — those matter at scale, and
        nothing in the <=10M regime benefits enough to justify the extra
        surface here. Add them when the agent promotes the model to HPC.
        """

        def __init__(
            self,
            vocab_size: int,
            d_model: int,
            nhead: int,
            num_layers: int,
            dim_ff: int,
            max_seq_len: int,
        ) -> None:
            super().__init__()
            self.tok_embed = nn.Embedding(vocab_size, d_model)
            self.pos_embed = nn.Embedding(max_seq_len, d_model)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_ff,
                batch_first=True,
                activation="gelu",
            )
            # ponytail: encoder-only. nn.TransformerDecoder can be slotted in
            # for true seq2seq later; for next-token LM the head over encoder
            # output is enough and keeps param count predictable.
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
            self.head = nn.Linear(d_model, vocab_size)
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

        def forward(self, tokens: "torch.Tensor") -> "torch.Tensor":
            B, T = tokens.shape
            pos = torch.arange(T, device=tokens.device).unsqueeze(0).expand(B, T)
            x = self.tok_embed(tokens) + self.pos_embed(pos)
            x = self.encoder(x)
            return self.head(x)


class TransformerTool(HuginnTool):
    """Tiny Transformer trainer/predictor, bounded to 10M parameters."""

    name = "transformer_tool"
    category = "sci"
    profile = ToolProfile(phases=frozenset({ResearchPhase.VALIDATION}))
    description = (
        "Train a small (<=10M parameter) Transformer encoder on token-id "
        "sequences and return the state_dict + loss curve; or predict from "
        "a previously trained state_dict. For larger models, dispatch to HPC."
    )
    input_schema = TransformerToolInput

    def __init__(self) -> None:
        super().__init__()
        self._torch_available = _TORCH_AVAILABLE

    def is_available(self) -> bool:
        return _TORCH_AVAILABLE

    def call(
        self, args: dict[str, Any], context: ToolContext | None = None
    ) -> ToolResult:
        if not _TORCH_AVAILABLE:
            return ToolResult(
                data=None,
                success=False,
                error="torch not available; install torch to use transformer_tool",
            )
        try:
            inp = TransformerToolInput(**args)
        except Exception as e:
            return ToolResult(data=None, success=False, error=f"invalid input: {e}")

        if inp.action == "predict":
            return self._predict(inp)
        return self._train(inp)

    # ── train ───────────────────────────────────────────────────────────
    def _train(self, inp: TransformerToolInput) -> ToolResult:
        torch.manual_seed(inp.seed)
        model = _SeqLM(
            inp.vocab_size,
            inp.d_model,
            inp.nhead,
            inp.num_layers,
            inp.dim_ff,
            inp.max_seq_len,
        )
        n_params = _count_params(model)
        if n_params > _MAX_PARAMS:
            return ToolResult(
                data=None,
                success=False,
                error=(
                    f"model has {n_params:,} params > {_MAX_PARAMS:,} cap; "
                    "reduce d_model/num_layers/vocab_size or dispatch to HPC"
                ),
            )

        if not inp.sequences:
            return ToolResult(data=None, success=False, error="sequences is empty")

        # ponytail: device由入口检测后通过 env var 决定, 这里不重复判断.
        # 升级路径: 按 n_params 自动选 device (大 model 强制 GPU).
        _dev = get_torch_device()
        device = torch.device(_dev)
        model = model.to(device)
        try:
            X = torch.tensor(inp.sequences, dtype=torch.long, device=device)
        except Exception as e:
            return ToolResult(data=None, success=False, error=f"bad sequences: {e}")
        if X.dim() != 2:
            return ToolResult(
                data=None,
                success=False,
                error=f"sequences must be 2D (B, T); got {tuple(X.shape)}",
            )
        if X.size(1) > inp.max_seq_len:
            return ToolResult(
                data=None,
                success=False,
                error=f"seq_len {X.size(1)} > max_seq_len {inp.max_seq_len}",
            )

        if inp.targets:
            Y = torch.tensor(inp.targets, dtype=torch.long, device=device)
            if Y.shape != X.shape:
                return ToolResult(
                    data=None,
                    success=False,
                    error=(
                        f"targets shape {tuple(Y.shape)} != "
                        f"sequences {tuple(X.shape)}"
                    ),
                )
        else:
            # next-token shift
            Y = X[:, 1:]
            X = X[:, :-1]
            if X.size(1) == 0:
                return ToolResult(
                    data=None,
                    success=False,
                    error="sequences too short for next-token shift",
                )

        opt = torch.optim.AdamW(model.parameters(), lr=inp.lr)
        loss_fn = nn.CrossEntropyLoss()
        model.train()

        B = X.size(0)
        bs = min(inp.batch_size, B)
        loss_curve: list[float] = []
        for _ in range(inp.epochs):
            perm = torch.randperm(B)
            ep_loss = 0.0
            n_batches = 0
            for i in range(0, B, bs):
                idx = perm[i : i + bs]
                xb = X[idx]
                yb = Y[idx]
                logits = model(xb)  # (b, T, V)
                loss = loss_fn(
                    logits.reshape(-1, logits.size(-1)), yb.reshape(-1)
                )
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                ep_loss += float(loss.item())
                n_batches += 1
            loss_curve.append(ep_loss / max(n_batches, 1))

        # Flatten tensors to plain lists so the result survives JSON round-trip.
        sd = {
            k: v.detach().cpu().reshape(-1).tolist()
            for k, v in model.state_dict().items()
        }
        shapes = {k: list(v.shape) for k, v in model.state_dict().items()}

        return ToolResult(
            data={
                "action": "train",
                "n_params": n_params,
                "under_cap": n_params <= _MAX_PARAMS,
                "epochs": inp.epochs,
                "loss_curve": loss_curve,
                "final_loss": loss_curve[-1] if loss_curve else None,
                "state_dict": sd,
                "state_dict_shapes": shapes,
                "model_spec": {
                    "vocab_size": inp.vocab_size,
                    "d_model": inp.d_model,
                    "nhead": inp.nhead,
                    "num_layers": inp.num_layers,
                    "dim_ff": inp.dim_ff,
                    "max_seq_len": inp.max_seq_len,
                },
                "message": (
                    f"trained {n_params:,} param Transformer for "
                    f"{inp.epochs} epoch(s); final loss {loss_curve[-1]:.4f}"
                ),
            },
            success=True,
        )

    # ── predict ─────────────────────────────────────────────────────────
    def _predict(self, inp: TransformerToolInput) -> ToolResult:
        if not inp.sequences:
            return ToolResult(data=None, success=False, error="sequences is empty")
        if not inp.state_dict:
            return ToolResult(
                data=None,
                success=False,
                error="predict requires state_dict from a prior train call",
            )

        torch.manual_seed(inp.seed)
        model = _SeqLM(
            inp.vocab_size,
            inp.d_model,
            inp.nhead,
            inp.num_layers,
            inp.dim_ff,
            inp.max_seq_len,
        )
        # Rebuild tensors from the flattened state_dict using the model's own
        # current shapes — they're authoritative for this model spec.
        shapes = {k: list(v.shape) for k, v in model.state_dict().items()}
        sd: dict[str, "torch.Tensor"] = {}
        for k, flat in inp.state_dict.items():
            t = torch.tensor(flat, dtype=torch.float32)
            sd[k] = t.reshape(shapes[k]) if k in shapes else t
        try:
            model.load_state_dict(sd)
        except Exception as e:
            return ToolResult(
                data=None, success=False, error=f"state_dict load failed: {e}"
            )
        model.eval()

        _dev = get_torch_device()
        device = torch.device(_dev)
        model = model.to(device)
        X = torch.tensor(inp.sequences, dtype=torch.long, device=device)
        if X.size(1) > inp.max_seq_len:
            return ToolResult(
                data=None,
                success=False,
                error=f"seq_len {X.size(1)} > max_seq_len {inp.max_seq_len}",
            )

        # Greedy autoregressive extension.
        cur = X
        steps: list[list[int]] = []
        with torch.no_grad():
            for _ in range(inp.predict_len):
                if cur.size(1) > inp.max_seq_len:
                    cur = cur[:, -inp.max_seq_len :]
                logits = model(cur)  # (B, T, V)
                nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (B, 1)
                steps.append(nxt.squeeze(1).tolist())
                cur = torch.cat([cur, nxt], dim=1)
        # transpose: list[step] of B-list -> B-list of predict_len
        extended = [[step[i] for step in steps] for i in range(X.size(0))]

        return ToolResult(
            data={
                "action": "predict",
                "input_sequences": inp.sequences,
                "predicted_tokens": extended,
                "predict_len": inp.predict_len,
                "message": (
                    f"predicted {inp.predict_len} tokens for "
                    f"{X.size(0)} sequence(s)"
                ),
            },
            success=True,
        )


# ── self-check: tiny doubling-mod task, 1 epoch, param count under cap ────
def _selfcheck() -> None:
    if not _TORCH_AVAILABLE:
        print("torch not available, skip demo")
        return

    # Synthetic next-token task: each sequence is [0, 2, 4, ...] mod vocab.
    # The model should pick up the +2 (mod vocab) pattern.
    vocab = 16
    seq_len = 8
    n_seq = 32
    seqs = [[(i * 2) % vocab for i in range(seq_len)] for _ in range(n_seq)]

    tool = TransformerTool()
    train_res = tool.call(
        {
            "action": "train",
            "sequences": seqs,
            "vocab_size": vocab,
            "d_model": 32,
            "nhead": 4,
            "num_layers": 1,
            "dim_ff": 64,
            "max_seq_len": seq_len + 4,
            "epochs": 1,
            "lr": 5e-3,
            "batch_size": 8,
            "seed": 0,
        }
    )
    if not train_res.success:
        raise SystemExit(f"train failed: {train_res.error}")
    data = train_res.data
    assert data is not None
    print(
        f"[train] params={data['n_params']:,}  "
        f"final_loss={data['final_loss']:.4f}  epochs={data['epochs']}"
    )
    assert data["n_params"] <= _MAX_PARAMS, "param cap violated"
    final_loss = data["final_loss"]
    assert final_loss == final_loss, "loss is NaN"  # NaN != NaN
    assert final_loss < 5.0, f"loss {final_loss} too large; training broken"

    # Round-trip: predict using the just-trained state_dict.
    pred_res = tool.call(
        {
            "action": "predict",
            "sequences": [[0, 2, 4, 6]],
            "state_dict": data["state_dict"],
            "vocab_size": vocab,
            "d_model": 32,
            "nhead": 4,
            "num_layers": 1,
            "dim_ff": 64,
            "max_seq_len": seq_len + 4,
            "predict_len": 4,
            "seed": 0,
        }
    )
    if not pred_res.success:
        raise SystemExit(f"predict failed: {pred_res.error}")
    pdata = pred_res.data
    assert pdata is not None
    print(
        f"[predict] input={pdata['input_sequences'][0]}  "
        f"predicted={pdata['predicted_tokens'][0]}"
    )
    print("self-check OK")


if __name__ == "__main__":
    _selfcheck()
