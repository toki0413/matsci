"""离线 JEPA predictor 训练管线 (阶段2-A) — 离线训练, 运行时冻结.

把:
    冻结编码器 φ (sentence-transformers, 已预训练, 不更新)
    + 小型 predictor g (只做潜空间映射, 可训练)
学成一个 L2 目标: g(φ(prediction_context)) ≈ φ(actual).

训出的 g 在运行时**只前向不更新**, 贴合 §4.3 非学习红线.
语料: ``{runtime_home}/corpus/jepa_pairs.jsonl`` (真实 plan→actual).

⚠ 诚实边界: 当前真实语料仅 4 条, 远不足以训出泛化 predictor.
本脚本是**管线落地 + 可运行演示**: 验证 数据加载→冻结编码→训练→前向冻结 完整闭环,
并输出真实语料上的语义 surprise 重标定参考 (供 §4.15 阈值积累). 不作"已训好模型"之断言.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from huginn.utils.runtime import get_runtime_home

# JEPA 专属编码器 (与共享 RAG/taste 的 EMBED_MODEL 解耦). 可用 HUGINN_JEPA_EMBED_MODEL 覆盖.
# 默认升级到更高容量多语言 mpnet(768) —— 相对 MiniLM-L12(384), 更大模型应改善短数值型
# actual 的可辨识度 (阶段2-C 弱族实验的编码侧对照).
JEPA_EMBED_MODEL = os.environ.get(
    "HUGINN_JEPA_EMBED_MODEL",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
)


def load_pairs(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_encoder(offline: bool = True):
    if offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(JEPA_EMBED_MODEL)


def semantic_dist_survey(predictor_pred: np.ndarray, actual: np.ndarray,
                         pairs, encoder_num: list[str]) -> dict:
    """真实语料语义 surprise 重标定参考 (同配 vs 跨配分布)."""
    def d(a, b):
        a = a / (np.linalg.norm(a) + 1e-12)
        b = b / (np.linalg.norm(b) + 1e-12)
        return float(1 - float(np.dot(a, b)))
    same = [d(predictor_pred[i], actual[i]) for i in range(len(pairs))]
    cross = []
    for i in range(len(pairs)):
        for j in range(len(pairs)):
            if i != j:
                cross.append(d(predictor_pred[i], actual[j]))
    return {
        "same_mean": float(np.mean(same)),
        "same_p90": float(np.percentile(same, 90)),
        "cross_mean": float(np.mean(cross)),
        "cross_p10": float(np.percentile(cross, 10)),
        "n": len(pairs),
    }


def train_predictor(X: np.ndarray, Y: np.ndarray, dim: int,
                    steps: int = 200, lr: float = 1e-3, seed: int = 0) -> tuple:
    """冻结 φ 只看数据, 训 predictor g. 返回 (g 权重 pack, 训练 loss 曲线)."""
    rng = np.random.default_rng(seed)
    # 单隐层 MLP: dim -> 64 -> dim, tanh. weight init small.
    W1 = rng.standard_normal((dim, 64)) * 0.02
    b1 = np.zeros(64)
    W2 = rng.standard_normal((64, dim)) * 0.02
    b2 = np.zeros(dim)
    losses = []
    for _ in range(steps):
        h = np.tanh(X @ W1 + b1)
        pred = h @ W2 + b2
        grad = 2 * (pred - Y) / max(1, len(X))
        gradW2 = h.T @ grad
        gradb2 = grad.sum(axis=0)
        gradh = grad @ W2.T
        gradb1 = gradh * (1 - h * h)
        gradW1 = X.T @ gradb1
        W1 -= lr * gradW1
        b1 -= lr * gradb1.sum(axis=0)
        W2 -= lr * gradW2
        b2 -= lr * gradb2
        losses.append(float(np.mean((pred - Y) ** 2)))
    pack = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    return pack, losses


def forward_frozen(pack, X):
    h = np.tanh(X @ pack["W1"] + pack["b1"])
    return h @ pack["W2"] + pack["b2"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=None, help="覆盖 jepa_pairs.jsonl 路径")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    corpus = Path(args.corpus) if args.corpus else (get_runtime_home() / "corpus" / "jepa_pairs.jsonl")
    pairs = load_pairs(corpus)
    print(f"[jepa] corpus={corpus} pairs={len(pairs)}")
    if not pairs:
        print("[jepa] 无语料, 跳过训练 (先运行真实采集)。")
        return

    enc = load_encoder()
    _get_dim = getattr(enc, "get_embedding_dimension", None) or getattr(enc, "get_sentence_embedding_dimension")
    dim = int(_get_dim())
    pred_embs = enc.encode([p["prediction"] for p in pairs], normalize_embeddings=True)
    act_embs = enc.encode([p["actual"] for p in pairs], normalize_embeddings=True)
    pred_embs = np.asarray(pred_embs, dtype=np.float64)
    act_embs = np.asarray(act_embs, dtype=np.float64)

    survey = semantic_dist_survey(pred_embs, act_embs, pairs, [])
    print(f"[jepa] 语义 surprise 重标定: "
          f"same_mean={survey['same_mean']:.3f} same_p90={survey['same_p90']:.3f} "
          f"cross_mean={survey['cross_mean']:.3f} cross_p10={survey['cross_p10']:.3f} (n={survey['n']})")
    print("[jepa] 注意: 样本过少, 同/跨仍有重叠, 阈值须继续累积真实配对再精标.")

    # 训练 (用全部样例做演示性拟合; 真实场景须更多语料并留独立验证集)
    pack, losses = train_predictor(pred_embs, act_embs, dim,
                                   steps=args.steps, seed=args.seed)
    print(f"[jepa] predictor 训练完成: 起始loss={losses[0]:.4f} 末loss={losses[-1]:.4f}")
    if losses:
        import statistics
        print(f"[jepa] loss 末段均值={sum(losses[-20:])/len(losses[-20:]):.4f}")

    # 运行时冻结前向: 对每条真实预测投影到潜空间, 报告它能多近地还原 actual.
    fwd = forward_frozen(pack, pred_embs)
    def d(a, b):
        a = a / (np.linalg.norm(a) + 1e-12)
        b = b / (np.linalg.norm(b) + 1e-12)
        return float(1 - float(np.dot(a, b)))
    errs = [d(fwd[i], act_embs[i]) for i in range(len(pairs))]
    print(f"[jepa] 冻结 predictor 前向重构 surprise: " +
          ", ".join(f"{e:.3f}" for e in errs) +
          f" (mean={np.mean(errs):.3f}, 理论下限即编码后不可还原的部分)")

    # 落盘 predictor (jsonl) 作为"离线产物", 运行时只读不更新.
    out = (get_runtime_home() / "models" / "jepa_predictor.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "embed_model": JEPA_EMBED_MODEL,
        "dim": dim,
        "weights": {k: v.tolist() for k, v in pack.items()},
        "loss_history": losses,
        "surprise_survey": survey,
        "note": "离线训练产物; 运行时只前向冻结, 不更新权重 (§4.3 红线).",
    }
    out.write_text(json.dumps(payload), encoding="utf-8")
    print(f"[jepa] predictor 已落盘: {out} (离线构建产物)")


if __name__ == "__main__":
    main()