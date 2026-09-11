"""排序/对比 目标 —— 方案① 最后一条与"相对秩是唯一稳健信号"同向的路.

背景: 回归型(逐 span MLP → 重建 actual 文本嵌入)与增强 prediction 信息(文本/数值)
双通道都卡在 ~0.28 表示级上限。不再尝试降绝对 surprise, 而是**训 predictor 让正确
actual 的相对秩排最前** —— 消费端本就吃相对秩, 这里把它变成可训目标。

目标 (margin contrastive, 可微 strep):
    p_i = MLP( mean{预测 span 的 ST 嵌入} )   (d -> 64 -> d)
    pos = dist(p_i, a_i);  neg = dist(p_i, 最难跨配 a_j)  (最近那个)
    loss = max(0, pos - neg + margin)            → 只压"同配更近、跨配更远"
  实际表示 a 用冻结 ST 的 pooled 均值(参数固定即固定), 只有 p 端 MLP 可训。

指标 (吃相对秩的证词): held-out 上正确 actual 在所有 actual 里的排名 →
    MRR = mean(1/rank), rank-1 = 正确即最近 占比, 以及分离带
    (同配 pos 均距 vs 跨配最近 neg 均距); 对照未训基线 (p = 输入身份)。

诚实边界:
  - 用 pooled(整 pred 均值)单向量打分, 丢 span 结构 —— 首测对比机制够用; 若有效再上 span 级。
  - 最难负样本 = 全局跨配 argmin, 训练压力强但也易记住单个对的怪癖, 需 held-out 目标级判断。
"""
from __future__ import annotations

import argparse
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from huginn.utils.runtime import get_runtime_home
from scripts.train_jepa_predictor import load_pairs, load_encoder
from scripts.train_jepa_span_predictor import _spans


def _pool(text: str, enc) -> np.ndarray:
    spans = [s for s in _spans(text)]
    if not spans:
        return np.zeros(enc.get_embedding_dimension(), dtype=np.float64)
    v = np.asarray(enc.encode(spans, normalize_embeddings=True), dtype=np.float64)
    return v.mean(axis=0)


def _dist(a, b):
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(1 - float(np.dot(a, b)))


def _train_rank(X, Y, d, steps=300, lr=1e-3, margin=0.1, seed=0, neg_k=8, alpha=0.1):
    """margin contrastive (residual): P = X + alpha*MLP(X). 每样本取最近跨配作最难负样本.

    residual 保证初始 == 未训基线(identity), 只学"把正确 actual 的秩再提高"的小增量,
    避免从零初始化坍缩到随机方向。
    """
    rng = np.random.default_rng(seed)
    W1 = rng.standard_normal((d, 64)) * 0.02
    b1 = np.zeros(64)
    W2 = rng.standard_normal((64, d)) * 0.02
    b2 = np.zeros(d)
    N = X.shape[0]
    Yn = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12)

    def _forward(_w1, _b1, _w2, _b2):
        H = np.tanh(X @ _w1 + _b1) if True else None
        _g = H @ _w2 + _b2
        return (X + alpha * _g), H

    for it in range(steps):
        P, H = _forward(W1, b1, W2, b2)
        Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)
        pos = 1.0 - np.einsum("nd,nd->n", Pn, Yn)
        sim = np.where(~np.eye(N, dtype=bool), Pn @ Yn.T, -1e18)
        hard_j = np.argmax(sim, axis=1)
        neg = 1.0 - sim[np.arange(N), hard_j]
        margin_term = (margin - (pos - neg)).clip(min=0.0)
        act = margin_term > 0.0
        if it == 0 or (it + 1) % 100 == 0:
            print(f"      step {it+1}: pos={pos.mean():.3f} neg={neg.mean():.3f} "
                  f"active={act.mean():.1%}")

        gnorm = act[:, None] * (Yn[hard_j] - Yn)                 # dloss/du, u=归一P
        r = np.linalg.norm(P, axis=1, keepdims=True) + 1e-12
        r3 = np.maximum(r, 1e-8) ** 3
        dP = gnorm / r - P * np.einsum("nd,nd->n", P, gnorm)[:, None] / r3   # dloss/dP
        dP = alpha * dP                                           # P = X + alpha*g
        gH = dP @ W2.T
        gh = gH * (1.0 - H * H)
        gradW2 = H.T @ dP
        gradb2 = dP.sum(axis=0)
        gradW1 = X.T @ gh
        gradb1 = gh.sum(axis=0)
        W1 -= lr * gradW1
        b1 -= lr * gradb1
        W2 -= lr * gradW2
        b2 -= lr * gradb2
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "alpha": alpha}


def _fwd(pack, X):
    H = np.tanh(X @ pack["W1"] + pack["b1"])
    return X + pack["alpha"] * (H @ pack["W2"] + pack["b2"])


def _rank_metrics(pi, acts, i):
    """pi: 预测池化(d). acts: 全部 actual 池化 (N,d). 返回 (rank, reciprocal_rank)."""
    ds = [_dist(pi, a) for a in acts]
    order = sorted(range(len(ds)), key=lambda k: ds[k])
    rr = order.index(i) + 1
    return rr, 1.0 / rr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--splits", type=int, default=12)
    ap.add_argument("--holdout-frac", type=float, default=0.25)
    ap.add_argument("--margin", type=float, default=0.1)
    ap.add_argument("--neg-k", type=int, default=8)
    args = ap.parse_args()

    corpus = Path(args.corpus) if args.corpus else (get_runtime_home() / "corpus" / "jepa_pairs.jsonl")
    pairs = load_pairs(corpus)
    enc = load_encoder()
    d = int(enc.get_embedding_dimension())
    X = np.stack([_pool(p["prediction"], enc) for p in pairs]).astype(np.float64)
    Y = np.stack([_pool(p["actual"], enc) for p in pairs]).astype(np.float64)
    # 归一化 X 输入 (ST 已归一, mean 后重新归一确保量级一致)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

    objs = list({p.get("objective", f"o{i}") for i, p in enumerate(pairs)})
    by_obj = defaultdict(list)
    for i, p in enumerate(pairs):
        by_obj[p.get("objective", f"o{i}")].append(i)

    rng = np.random.default_rng(args.seed)
    mrr_base, mrr_rank = [], []
    rank1_base = rank1_rank = 0
    sep_base_pos, sep_base_neg = [], []
    sep_rank_pos, sep_rank_neg = [], []
    n_ho = max(1, int(round(len(objs) * args.holdout_frac)))
    tot = 0
    for _ in range(args.splits):
        ho = set(rng.choice(objs, size=n_ho, replace=False))
        te = [i for o, ids in by_obj.items() if o in ho for i in ids]
        tr = [i for o, ids in by_obj.items() if o not in ho for i in ids]
        if len(tr) < 5 or not te:
            continue
        pack = _train_rank(Xn[tr], Y[tr], d, steps=args.steps, margin=args.margin,
                           seed=int(rng.integers(1, 1 << 30)))
        P_te = _fwd(pack, Xn[te])          # 测试行前向 (len=len(te))
        acts = Y[te]
        for idx, i in enumerate(te):
            tot += 1
            pbase = Xn[i]                     # 未训基线 (身份)
            r_b, rr_b = _rank_metrics(pbase, acts, idx)
            mrr_base.append(rr_b)
            if r_b == 1:
                rank1_base += 1
            # 分离带: 对未训, pos = dist(pbase, 自己的 actual), neg = 最近跨配(在 te 内)
            _a_self = acts[idx]
            sep_base_pos.append(_dist(pbase, _a_self))
            negb = min((_dist(pbase, acts[j]) for j in range(len(acts)) if j != idx))
            sep_base_neg.append(negb)
            # trained
            p_i = P_te[idx]
            r_a, rr_a = _rank_metrics(p_i, acts, idx)
            mrr_rank.append(rr_a)
            if r_a == 1:
                rank1_rank += 1
            sep_rank_pos.append(_dist(p_i, _a_self))
            negr = min((_dist(p_i, acts[j]) for j in range(len(acts)) if j != idx))
            sep_rank_neg.append(negr)

    n = max(1, len(mrr_base))
    print(f"[rank] 目标级留出 {args.holdout_frac:.0%} ({n_ho}/{len(objs)}), {args.splits} 次, "
          f"steps={args.steps}, margin={args.margin}")
    print(f"[base] MRR={statistics.mean(mrr_base):.3f}  rank-1={100*rank1_base/tot:.1f}%  "
          f"分离带 pos={statistics.mean(sep_base_pos):.3f} vs neg={statistics.mean(sep_base_neg):.3f}")
    print(f"[rank] MRR={statistics.mean(mrr_rank):.3f}  rank-1={100*rank1_rank/tot:.1f}%  "
          f"分离带 pos={statistics.mean(sep_rank_pos):.3f} vs neg={statistics.mean(sep_rank_neg):.3f}")
    print(f"[rank-base] dMRR={statistics.mean(mrr_rank)-statistics.mean(mrr_base):+.3f}  "
          f"drank1={100*(rank1_rank-rank1_base)/tot:+.1f}pp")


if __name__ == "__main__":
    main()