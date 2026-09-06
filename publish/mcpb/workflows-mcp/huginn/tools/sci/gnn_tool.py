"""Graph neural network training wrapper.

Trains a Graph Convolutional Network (GCN) for transductive node
classification and exposes train / predict / embed actions. Uses PyTorch
Geometric when both torch and torch_geometric are importable; otherwise
falls back to a NetworkX spectral embedding + sklearn LogisticRegression
pipeline so the tool still produces embeddings and predictions without
PyG installed.

torch / torch_geometric are imported lazily — importing this module never
requires them.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from huginn.core_types import ToolContext, ToolResult
from huginn.tools.base import HuginnTool, ResearchPhase, ToolProfile


class GNNToolInput(BaseModel):
    action: Literal["train", "predict", "embed"] = Field(default="train")
    edge_index: list[list[int]] = Field(
        default_factory=list,
        description="2 x E edge list as [[src...], [dst...]]; undirected edges should be added in both directions",
    )
    node_features: list[list[float]] = Field(
        default_factory=list,
        description="N x F node features; empty = identity matrix (one-hot nodes)",
    )
    labels: list[int] = Field(
        default_factory=list,
        description="N integer node labels; required for train, ignored by embed",
    )
    train_mask: list[bool] | None = Field(
        default=None,
        description="N bool, which nodes are supervised; None = all nodes",
    )
    hidden_dim: int = Field(default=16, gt=0)
    num_layers: int = Field(default=2, ge=1)
    epochs: int = Field(default=100, ge=1)
    lr: float = Field(default=0.01, gt=0)
    dropout: float = Field(default=0.0, ge=0, le=1)
    num_classes: int | None = Field(
        default=None, gt=0, description="Inferred from labels if omitted"
    )
    state_dict: dict[str, Any] | None = Field(
        default=None, description="Weights returned by the train action"
    )
    seed: int | None = Field(default=None)


# ── model builders ─────────────────────────────────────────────────


def _build_gcn(in_dim: int, hidden_dim: int, num_classes: int, num_layers: int, dropout: float):
    """Build a vanilla stacked-GCNConv classifier. Imports torch lazily."""
    import torch.nn as nn
    import torch.nn.functional as F  # noqa: N812
    from torch_geometric.nn import GCNConv

    class GCN(nn.Module):
        def __init__(self):
            super().__init__()
            convs = [GCNConv(in_dim, hidden_dim)]
            for _ in range(max(0, num_layers - 2)):
                convs.append(GCNConv(hidden_dim, hidden_dim))
            convs.append(GCNConv(hidden_dim, num_classes))
            self.convs = nn.ModuleList(convs)
            self.drop = dropout

        def forward(self, x, edge_index):
            for conv in self.convs[:-1]:
                x = F.relu(conv(x, edge_index))
                x = F.dropout(x, p=self.drop, training=self.training)
            return self.convs[-1](x, edge_index)

        def embed(self, x, edge_index):
            for conv in self.convs[:-1]:
                x = F.relu(conv(x, edge_index))
            return x

    return GCN()


def _spectral_embedding(edge_index: np.ndarray, n: int, k: int) -> tuple[np.ndarray, int]:
    """NetworkX + scipy spectral embedding on the normalized Laplacian.

    Returns (embedding n x k_eff, k_eff). Falls back to dense eigh when the
    graph is too small for sparse eigsh or eigsh fails to converge.
    """
    import networkx as nx
    from scipy.sparse.linalg import eigsh

    G = nx.Graph()
    G.add_nodes_from(range(n))
    if edge_index.shape[1] > 0:
        src = edge_index[0].tolist()
        dst = edge_index[1].tolist()
        G.add_edges_from(zip(src, dst))

    k_eff = max(1, min(k, max(1, n - 1)))
    L = nx.normalized_laplacian_matrix(G)
    try:
        # smallest eigenvalues — 'SM' is robust for small k on sparse Laplacians
        _, vecs = eigsh(L, k=k_eff, which="SM")
    except Exception:
        # dense fallback: tiny / disconnected graphs, numerical issues
        from scipy.linalg import eigh as dense_eigh

        vals, vecs = dense_eigh(L.toarray())
        vecs = vecs[:, :k_eff]
    return np.asarray(vecs, dtype=np.float64), int(vecs.shape[1])


# ── tool ───────────────────────────────────────────────────────────


class GNNTool(HuginnTool):
    """Graph neural network tool — GCN node classification with PyG or NetworkX+sklearn fallback."""

    name = "gnn_tool"
    category = "sci"
    profile = ToolProfile(
        cost_tier="heavy",
        phases=frozenset({ResearchPhase.EXECUTION}),
        heavy_actions=frozenset({"train"}),
    )
    description = (
        "Train a Graph Convolutional Network for node classification, then "
        "predict labels or pull node embeddings. Uses PyTorch Geometric when "
        "available; falls back to NetworkX spectral embedding + sklearn "
        "LogisticRegression so the tool still works without PyG."
    )
    input_schema = GNNToolInput

    # ponytail: GCN only — no GraphSAGE / GAT / GIN. The agent can swap the
    # conv layer in _build_gcn if it needs a different message-passing variant;
    # this wrapper pins one architecture so state_dict shapes are predictable
    # and the fallback path stays 1:1 with the PyG path.
    #
    # ponytail: transductive setting — predict/embed recompute the forward
    # pass on whatever edge_index is passed, but the fallback LR coefficients
    # are tied to the training graph's spectral basis. For genuinely inductive
    # prediction on unseen nodes, re-train. Ceiling: O(N) eigsh on each call.

    def call(
        self, args: dict[str, Any], context: ToolContext | None = None
    ) -> ToolResult:
        try:
            data = GNNToolInput(**args)
        except Exception as e:
            return ToolResult(data=None, success=False, error=f"invalid args: {e}")
        try:
            if data.action == "train":
                return self._train(data)
            if data.action == "predict":
                return self._predict(data)
            if data.action == "embed":
                return self._embed(data)
        except Exception as e:
            return ToolResult(data=None, success=False, error=f"GNN tool failed: {e}")
        return ToolResult(
            data=None, success=False, error=f"unknown action: {data.action}"
        )

    # ── input helpers ──────────────────────────────────────────────

    @staticmethod
    def _parse_graph(args: GNNToolInput) -> tuple[np.ndarray, np.ndarray, int]:
        if len(args.edge_index) != 2:
            raise ValueError("edge_index must be a 2 x E list")
        edge_index = np.asarray(args.edge_index, dtype=np.int64)
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E]")
        if args.node_features:
            feats = np.asarray(args.node_features, dtype=np.float32)
            n = int(feats.shape[0])
        else:
            # identity features — standard for node classification when no
            # external node attributes exist; matches the karate-club setup
            n = int(edge_index.max()) + 1 if edge_index.size else 1
            feats = np.eye(n, dtype=np.float32)
        if edge_index.size and (edge_index.max() >= n or edge_index.min() < 0):
            raise ValueError("edge_index references nodes outside [0, N-1]")
        return edge_index, feats, n

    @staticmethod
    def _pyg_available() -> bool:
        try:
            import torch  # noqa: F401
            import torch_geometric  # noqa: F401

            return True
        except ImportError:
            return False

    @staticmethod
    def _load_pyg_state(model, sd: dict[str, Any]) -> None:
        """Rebuild weight tensors from JSON lists, skipping the _meta key."""
        import torch

        tensors = {
            k: torch.tensor(v, dtype=torch.float32)
            for k, v in sd.items()
            if k != "_meta"
        }
        model.load_state_dict(tensors, strict=True)
        model.eval()

    # ── train ──────────────────────────────────────────────────────

    def _train(self, args: GNNToolInput) -> ToolResult:
        edge_index, feats, n = self._parse_graph(args)
        if not args.labels:
            return ToolResult(
                data=None, success=False, error="train requires labels for all nodes"
            )
        labels = np.asarray(args.labels, dtype=np.int64)
        if len(labels) != n:
            return ToolResult(
                data=None,
                success=False,
                error=f"labels length {len(labels)} != N {n}",
            )
        mask = (
            np.ones(n, dtype=bool)
            if args.train_mask is None
            else np.asarray(args.train_mask, dtype=bool)
        )
        n_classes = args.num_classes or int(labels.max()) + 1
        if n_classes < 2:
            return ToolResult(
                data=None,
                success=False,
                error="need at least 2 classes for classification",
            )

        if self._pyg_available():
            return self._train_pyg(args, edge_index, feats, labels, mask, n_classes)
        return self._train_fallback(args, edge_index, feats, labels, mask, n_classes)

    def _train_pyg(
        self, args, edge_index, feats, labels, mask, n_classes
    ) -> ToolResult:
        import torch
        import torch.nn.functional as F  # noqa: N812

        if args.seed is not None:
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)

        from huginn.tools.sci import get_torch_device
        _dev = get_torch_device()
        x = torch.from_numpy(feats).to(_dev)
        ei = torch.from_numpy(edge_index).to(_dev)
        y = torch.from_numpy(labels).to(_dev)
        m = torch.from_numpy(mask).to(_dev)

        model = _build_gcn(
            int(feats.shape[1]), args.hidden_dim, n_classes, args.num_layers, args.dropout
        )
        model = model.to(_dev)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)

        loss_curve: list[float] = []
        for _ in range(args.epochs):
            model.train()
            opt.zero_grad()
            out = model(x, ei)
            loss = F.cross_entropy(out[m], y[m])
            loss.backward()
            opt.step()
            loss_curve.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            emb = model.embed(x, ei).cpu().numpy()
            preds = model(x, ei).argmax(dim=1).cpu().numpy()

        state = {k: v.tolist() for k, v in model.state_dict().items()}
        state["_meta"] = {
            "in_dim": int(feats.shape[1]),
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "num_classes": int(n_classes),
        }

        return ToolResult(
            data={
                "state_dict": state,
                "loss_curve": loss_curve,
                "embeddings": emb.tolist(),
                "predictions": preds.tolist(),
                "num_classes": int(n_classes),
                "fallback_used": False,
                "n_nodes": int(len(labels)),
                "message": (
                    f"PyG GCN trained {args.epochs} epoch(s), "
                    f"final loss {loss_curve[-1]:.6f}."
                ),
            },
            success=True,
        )

    def _train_fallback(
        self, args, edge_index, feats, labels, mask, n_classes
    ) -> ToolResult:
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import log_loss
        except ImportError as e:
            return ToolResult(
                data=None,
                success=False,
                error=(
                    f"neither PyG nor the fallback stack "
                    f"(networkx/scipy/sklearn) is available: {e}"
                ),
            )

        if args.seed is not None:
            np.random.seed(args.seed)

        emb, k_eff = _spectral_embedding(edge_index, len(labels), args.hidden_dim)
        X_train = emb[mask]
        y_train = labels[mask]
        if len(np.unique(y_train)) < 2:
            return ToolResult(
                data=None,
                success=False,
                error="train_mask covers only one class; cannot fit LogisticRegression",
            )

        # max_iter ≈ epochs so the LR solver budget scales with the request
        clf = LogisticRegression(max_iter=max(args.epochs, 100), random_state=args.seed)
        clf.fit(X_train, y_train)

        probs = clf.predict_proba(X_train)
        try:
            loss = float(log_loss(y_train, probs, labels=list(range(n_classes))))
        except ValueError:
            loss = float(log_loss(y_train, probs))
        preds = clf.predict(emb)

        state = {
            "method": "spectral_lr",
            "coef": clf.coef_.tolist(),
            "intercept": clf.intercept_.tolist(),
            "classes": [int(c) for c in clf.classes_.tolist()],
            "k": int(k_eff),
            "num_classes": int(n_classes),
        }

        return ToolResult(
            data={
                "state_dict": state,
                "loss_curve": [loss],
                "embeddings": emb.tolist(),
                "predictions": preds.tolist(),
                "num_classes": int(n_classes),
                "fallback_used": True,
                "n_nodes": int(len(labels)),
                "message": (
                    f"Fallback (NetworkX spectral k={k_eff} + sklearn "
                    f"LogisticRegression), train log-loss {loss:.6f}."
                ),
            },
            success=True,
        )

    # ── predict ────────────────────────────────────────────────────

    def _predict(self, args: GNNToolInput) -> ToolResult:
        if not args.state_dict:
            return ToolResult(
                data=None,
                success=False,
                error="predict requires state_dict from a previous train call",
            )
        edge_index, feats, n = self._parse_graph(args)
        sd = args.state_dict

        if sd.get("method") == "spectral_lr":
            return self._predict_fallback(args, edge_index, sd, n)

        return self._predict_pyg(args, edge_index, feats, sd, n)

    def _predict_fallback(self, args, edge_index, sd, n) -> ToolResult:
        emb, _ = _spectral_embedding(edge_index, n, int(sd["k"]))
        coef = np.asarray(sd["coef"], dtype=np.float64)
        intercept = np.asarray(sd["intercept"], dtype=np.float64)
        classes = np.asarray(sd["classes"], dtype=np.int64)
        logits = emb @ coef.T + intercept
        preds = classes[np.argmax(logits, axis=1)]
        return ToolResult(
            data={
                "predictions": preds.tolist(),
                "fallback_used": True,
                "message": f"Predicted labels for {n} nodes (fallback path).",
            },
            success=True,
        )

    def _predict_pyg(self, args, edge_index, feats, sd, n) -> ToolResult:
        if not self._pyg_available():
            return ToolResult(
                data=None,
                success=False,
                error="state_dict was trained with PyG but torch_geometric is not available",
            )
        import torch

        meta = sd.get("_meta", {})
        in_dim = int(meta.get("in_dim", feats.shape[1]))
        if in_dim != feats.shape[1]:
            return ToolResult(
                data=None,
                success=False,
                error=f"node_features dim {feats.shape[1]} != trained in_dim {in_dim}",
            )
        model = _build_gcn(
            in_dim,
            int(meta.get("hidden_dim", args.hidden_dim)),
            int(meta.get("num_classes", args.num_classes or 2)),
            int(meta.get("num_layers", args.num_layers)),
            args.dropout,
        )
        self._load_pyg_state(model, sd)
        from huginn.tools.sci import get_torch_device
        _dev = get_torch_device()
        model = model.to(_dev)
        x = torch.from_numpy(feats).to(_dev)
        ei = torch.from_numpy(edge_index).to(_dev)
        with torch.no_grad():
            preds = model(x, ei).argmax(dim=1).cpu().numpy()
        return ToolResult(
            data={
                "predictions": preds.tolist(),
                "fallback_used": False,
                "message": f"Predicted labels for {n} nodes (PyG path).",
            },
            success=True,
        )

    # ── embed ──────────────────────────────────────────────────────

    def _embed(self, args: GNNToolInput) -> ToolResult:
        if not args.state_dict:
            return ToolResult(
                data=None,
                success=False,
                error="embed requires state_dict from a previous train call",
            )
        edge_index, feats, n = self._parse_graph(args)
        sd = args.state_dict

        if sd.get("method") == "spectral_lr":
            emb, _ = _spectral_embedding(edge_index, n, int(sd["k"]))
            return ToolResult(
                data={
                    "embeddings": emb.tolist(),
                    "fallback_used": True,
                    "message": f"Spectral embeddings for {n} nodes (fallback path).",
                },
                success=True,
            )

        if not self._pyg_available():
            return ToolResult(
                data=None,
                success=False,
                error="state_dict was trained with PyG but torch_geometric is not available",
            )
        import torch

        meta = sd.get("_meta", {})
        in_dim = int(meta.get("in_dim", feats.shape[1]))
        if in_dim != feats.shape[1]:
            return ToolResult(
                data=None,
                success=False,
                error=f"node_features dim {feats.shape[1]} != trained in_dim {in_dim}",
            )
        model = _build_gcn(
            in_dim,
            int(meta.get("hidden_dim", args.hidden_dim)),
            int(meta.get("num_classes", args.num_classes or 2)),
            int(meta.get("num_layers", args.num_layers)),
            args.dropout,
        )
        self._load_pyg_state(model, sd)
        from huginn.tools.sci import get_torch_device
        _dev = get_torch_device()
        model = model.to(_dev)
        x = torch.from_numpy(feats).to(_dev)
        ei = torch.from_numpy(edge_index).to(_dev)
        with torch.no_grad():
            emb = model.embed(x, ei).cpu().numpy()
        return ToolResult(
            data={
                "embeddings": emb.tolist(),
                "fallback_used": False,
                "message": f"GCN hidden embeddings for {n} nodes (PyG path).",
            },
            success=True,
        )


# ── self-check: karate club, 1 epoch, PyG or fallback ──────────────
if __name__ == "__main__":
    try:
        import networkx as nx  # noqa: F401
    except ImportError:
        print("networkx not available, skip demo")
        raise SystemExit(0) from None  # noqa: B904

    # need at least one stack: PyG or (scipy + sklearn)
    has_pyg = GNNTool._pyg_available()
    try:
        import scipy  # noqa: F401
        import sklearn  # noqa: F401

        has_fallback = True
    except ImportError:
        has_fallback = False
    if not has_pyg and not has_fallback:
        print("neither PyG nor scipy+sklearn available, skip demo")
        raise SystemExit(0)

    G = nx.karate_club_graph()
    edges = list(G.edges())
    src = [u for u, _ in edges]
    dst = [v for _, v in edges]
    # undirected → add both directions for message passing
    edge_index = [src + dst, dst + src]
    labels = [0 if G.nodes[i]["club"] == "Mr. Hi" else 1 for i in G.nodes()]
    n = G.number_of_nodes()

    tool = GNNTool()
    res = tool.call({
        "action": "train",
        "edge_index": edge_index,
        "labels": labels,
        "epochs": 1,
        "hidden_dim": 8,
        "seed": 0,
    })
    assert res.success, res.error
    d = res.data
    assert "state_dict" in d
    assert "loss_curve" in d
    assert "embeddings" in d
    assert "predictions" in d
    assert len(d["predictions"]) == n
    assert len(d["embeddings"]) == n
    assert d["num_classes"] == 2
    print(f"train ok: fallback={d['fallback_used']}, loss_curve={d['loss_curve']}")
    print(f"  message: {d['message']}")

    # round-trip: predict + embed using the returned state_dict
    pred = tool.call({
        "action": "predict",
        "edge_index": edge_index,
        "state_dict": d["state_dict"],
        "hidden_dim": 8,
    })
    assert pred.success, pred.error
    assert len(pred.data["predictions"]) == n
    print(f"predict ok: {len(pred.data['predictions'])} predictions")

    emb = tool.call({
        "action": "embed",
        "edge_index": edge_index,
        "state_dict": d["state_dict"],
        "hidden_dim": 8,
    })
    assert emb.success, emb.error
    assert len(emb.data["embeddings"]) == n
    print(f"embed ok: {len(emb.data['embeddings'])} embeddings")

    print("self-check passed")
