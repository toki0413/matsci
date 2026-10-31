"""阶段2-B: JEPA surprise 绝对阈值精标 (离线, 用当前冻结 predictor).

目标: 把 predictor 前向 surprise 从"相对排名信号"升级为"可用于运行时的绝对阈值分离器".
方法: 把每个样本作(pred, actual)当正例(标签0, surprise小), 跨配错配(pred, other_actual)当反例(标签1, surprise大),
      对阈值 τ 扫描, 计算 TPR/FPR/F1, 输出最优 τ (以 Youden/F1 max 为准) 与分离质量(AUC, Brier).
诚实边界: 语料仍是有限真实配对, 阈值是"在这批数据上可分离的参考值", 需继续累积后复标.
          且 §4.3 隔离红线: 运行时仍只前向冻结 predictor, 阈值只是消费其 surprise 的判定, 不更新权重.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from huginn.utils.runtime import get_runtime_home


def load_pairs(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_encoder(offline: bool = True):
    import os

    if offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )


def load_predictor_weights() -> dict | None:
    p = get_runtime_home() / "models" / "jepa_predictor.json"
    if not p.exists():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    return {k: np.asarray(v, dtype=np.float64) for k, v in data["weights"].items()}


def cosine_distance(a, b) -> float:
    a = np.asarray(a) / (np.linalg.norm(a) + 1e-12)
    b = np.asarray(b) / (np.linalg.norm(b) + 1e-12)
    return float(1.0 - float(np.dot(a, b)))


def forward(pred_emb: np.ndarray, w: dict) -> np.ndarray:
    h = np.tanh(pred_emb @ w["W1"] + w["b1"])
    return h @ w["W2"] + w["b2"]


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--steps", type=int, default=None)  # 未用: 阈值判定不重新训练
    return ap


def _same_cross(fwd: np.ndarray, act_embs: np.ndarray, n: int) -> tuple:
    """正例: 同配 surprise (对齐好 => 距离小); 反例: 跨配 (错配 => 距离大)."""
    same = np.array([cosine_distance(fwd[i], act_embs[i]) for i in range(n)])
    cross = np.array(
        [
            cosine_distance(fwd[i], act_embs[j])
            for i in range(n)
            for j in range(n)
            if i != j
        ]
    )
    return same, cross


def _auc(s: np.ndarray, c: np.ndarray) -> float:
    """AUC (Mann-Whitney U): P(跨配 > 同配)."""
    # subsample if huge
    if len(s) * len(c) > 5_000_000:
        rng = np.random.default_rng(0)
        s = rng.choice(s, 2000, replace=False)
        c = rng.choice(c, 2000, replace=False)
    pos = 0.0
    for x in s:
        pos += float(np.sum(c > x))
    n_pairs = len(s) * len(c)
    ties = 0.0
    for x in s:
        ties += float(np.sum(c == x))
    return (pos + 0.5 * ties) / n_pairs


def _scan_threshold(same: np.ndarray, cross: np.ndarray) -> dict:
    """阈值扫描: τ ∈ [p5, p95 of all surprise]; labels 跨配=1, 同配=0."""
    all_s = np.concatenate([same, cross])
    grid = np.linspace(min(all_s) + 1e-6, max(all_s) - 1e-6, 200)
    best = None
    for tau in grid:
        tp = float(np.sum(cross >= tau))  # 跨配被正确判异常
        fn = float(np.sum(cross < tau))
        fp = float(np.sum(same >= tau))  # 同配被误判异常
        tn = float(np.sum(same < tau))
        fpr = fp / max(1e-9, fp + tn)
        tpr = tp / max(1e-9, tp + fn)
        prec = tp / max(1e-9, tp + fp)
        f1 = 2 * prec * tpr / max(1e-9, prec + tpr)
        youden = tpr - fpr
        if best is None or f1 > best["f1"]:
            best = {
                "tau": float(tau),
                "f1": float(f1),
                "tpr": float(tpr),
                "fpr": float(fpr),
                "prec": float(prec),
                "youden": float(youden),
            }
    return best


def _report_distributions(same: np.ndarray, cross: np.ndarray) -> None:
    print(
        f"[t2b] same(n={len(same)}) mean={same.mean():.3f} p10={np.percentile(same, 10):.3f} p50={np.percentile(same, 50):.3f} p90={np.percentile(same, 90):.3f}"
    )
    print(
        f"[t2b] cross(n={len(cross)}) mean={cross.mean():.3f} p10={np.percentile(cross, 10):.3f} p50={np.percentile(cross, 50):.3f} p90={np.percentile(cross, 90):.3f}"
    )


def _report_best(best: dict, same: np.ndarray, cross: np.ndarray) -> None:
    print(
        f"[t2b] 最优τ: surprise<={best['tau']:.4f} 判正常 (同配类); >={best['tau']:.4f} 判异常/错配"
    )
    print(
        f"[t2b]   F1={best['f1']:.3f}  TPR={best['tpr']:.3f}  FPR={best['fpr']:.3f}  Precision={best['prec']:.3f}  Youden={best['youden']:.3f}"
    )
    # 覆盖: 在同配里, 多少低于阈值(应为低误报); 在跨配里多少高于(应为高捕获)
    print(
        f"[t2b]   同配低于τ(正确放行)比例={float(np.sum(same < best['tau'])) / len(same):.3f}"
    )
    print(
        f"[t2b]   跨配高于τ(正确拦截)比例={float(np.sum(cross >= best['tau'])) / len(cross):.3f}"
    )


def main() -> None:
    args = _build_parser().parse_args()
    corpus = (
        Path(args.corpus)
        if args.corpus
        else (get_runtime_home() / "corpus" / "jepa_pairs.jsonl")
    )
    pairs = load_pairs(corpus)
    print(f"[t2b] corpus={corpus} pairs={len(pairs)}")
    if len(pairs) < 5:
        print("[t2b] 样本过少, 无法精标.")
        return

    w = load_predictor_weights()
    enc = load_encoder()
    pred_embs = np.asarray(
        enc.encode([p["prediction"] for p in pairs], normalize_embeddings=True),
        dtype=np.float64,
    )
    act_embs = np.asarray(
        enc.encode([p["actual"] for p in pairs], normalize_embeddings=True),
        dtype=np.float64,
    )

    fwd = forward(pred_embs, w)  # shape (N, dim), 冻结前向
    same, cross = _same_cross(fwd, act_embs, len(pairs))
    _report_distributions(same, cross)

    auc_val = _auc(same, cross)
    print(f"[t2b] AUC(跨配>同配)={auc_val:.4f}  (>0.5 => 可分离; 1.0 = 完全分离)")

    best = _scan_threshold(same, cross)
    _report_best(best, same, cross)

    # 运行时建议: 把阈值写回? 不 —— 阈值精标是消费端决策, 干净起见只打印并可由调用方落盘.
    print(
        "[t2b] 注意: 这是离线参考阈值; 运行时只前向冻结 predictor + 用此阈值判异常, 不更新权重(§4.3 红线)."
    )


if __name__ == "__main__":
    main()
