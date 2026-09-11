"""Attention 对齐的 span predictor (阶段2-C-3 方向验证).

动机: 逐 span 共享 MLP + 硬最近邻在"拼通道"时没真正消费结构(负结果)。这里给预测 span
一个**自注意力聚合层**: 每个 span 由同 prediction 其他 span 加权(只读预测内结构, 不看 actual),
再经 MLP 回归到最近的真实 span。注意力是可学习的结构消费机制。

前向(单头, h 维): 
  Q=X@Wq, K=X@Wk, V=X@Wv;  attn=softmax(Q@Kt/sqrt(h), mask 掉非法的预测 span);
  Z=attn@V;  ref=X + Z@Wo(残差);  out=tanh(ref@W1+b1)@W2+b2
损失/惊讶: 掩码均值 over 预测 span of dist(out, 最近真实 span)。
"""
from __future__ import annotations

import argparse
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from huginn.utils.runtime import get_runtime_home
from scripts.train_jepa_predictor import load_pairs, load_encoder
from scripts.train_jepa_span_predictor import _spans, _padded, _dist, K


def _softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _forward(X, M, P, d, h):
    """X:(n,K,d) 预测 span; M:(n,K) 掩码. 返回 out:(n,K,d)."""
    Wq, Wk, Wv, Wo, W1, b1, W2, b2 = (
        P["Wq"], P["Wk"], P["Wv"], P["Wo"], P["W1"], P["b1"], P["W2"], P["b2"])
    Q = X @ Wq
    Km = X @ Wk
    V = X @ Wv
    sc = Q @ Km.transpose(0, 2, 1) / np.sqrt(h)
    mask = (M[:, :, None] & M[:, None, :])
    sc = np.where(mask, sc, -1e9)
    attn = _softmax(sc)
    Z = attn @ V
    Zproj = Z @ Wo
    ref = X + Zproj
    H = np.tanh(ref @ W1 + b1)
    out = H @ W2 + b2
    return out, (Q, Km, V, attn, Z, Zproj, ref, H)


def _train(X, T, M, d, h, steps=200, lr=2e-3, seed=0):
    rng = np.random.default_rng(seed)
    P = {
        "Wq": rng.standard_normal((d, h)) * 0.04, "Wk": rng.standard_normal((d, h)) * 0.04,
        "Wv": rng.standard_normal((d, h)) * 0.04, "Wo": rng.standard_normal((h, d)) * 0.04,
        "W1": rng.standard_normal((d, 64)) * 0.02, "b1": np.zeros(64),
        "W2": rng.standard_normal((64, d)) * 0.02, "b2": np.zeros(d),
    }
    n = X.shape[0]
    for _ in range(steps):
        out, (Q, Km, V, attn, Z, Zproj, ref, H) = _forward(X, M, P, d, h)
        cnt = np.clip(M.sum(axis=-1, keepdims=True), 1, None)
        # 损失: 掩码 L2 (只对有效行)
        dl_dout = 2 * (out - T) * M[..., None] / cnt[..., None]
        # MLP 反传
        gW2 = H.transpose(0, 2, 1) @ dl_dout
        gb2 = dl_dout.sum(axis=(0, 1))
        gH = dl_dout @ P["W2"].T
        gb1n = (gH * (1 - H * H))
        gW1 = np.einsum("nke,nkd->ed", gb1n, ref)
        # 对 ref(X+Zproj) 的梯度
        gRef = gb1n @ P["W1"].T          # (n,K,d)
        gZproj = gRef                    # ref = X + Zproj
        gWo = Z.transpose(0, 2, 1) @ gZproj
        gZ = gZproj @ P["Wo"].T
        gV = attn.transpose(0, 2, 1) @ gZ
        gattn = gZ @ V.transpose(0, 2, 1)
        # softmax 反传
        g_logits = attn * (gattn - (gattn * attn).sum(axis=2, keepdims=True))
        g_logits = np.where(M[:, :, None] & M[:, None, :], g_logits, 0.0)
        gQ = g_logits @ Km
        gKm = g_logits.transpose(0, 2, 1) @ Q
        gWq = X.transpose(0, 2, 1) @ gQ
        gWk = X.transpose(0, 2, 1) @ gKm
        gWv = X.transpose(0, 2, 1) @ gV
        # 更新
        P["Wq"] -= lr * gWq.sum(axis=0)
        P["Wk"] -= lr * gWk.sum(axis=0)
        P["Wv"] -= lr * gWv.sum(axis=0)
        P["Wo"] -= lr * gWo.sum(axis=0)
        P["W1"] -= lr * gW1.T
        P["b1"] -= lr * gb1n.sum(axis=(0, 1))
        P["W2"] -= lr * gW2.sum(axis=0)
        P["b2"] -= lr * gb2
    return P


def _nearest_target(Xi, Ai, Mi, asl_i):
    T = np.zeros_like(Xi)
    kk = min(asl_i, K)
    for k in range(K):
        if not Mi[k] or kk == 0:
            continue
        T[k] = min((Ai[j] for j in range(kk)), key=lambda a: _dist(Xi[k], a))
    return T


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--splits", type=int, default=12)
    ap.add_argument("--holdout-frac", type=float, default=0.25)
    ap.add_argument("--heads", type=int, default=1)
    ap.add_argument("--hidden", type=int, default=32)
    args = ap.parse_args()
    h = args.hidden

    corpus = Path(args.corpus) if args.corpus else (get_runtime_home() / "corpus" / "jepa_pairs.jsonl")
    pairs = load_pairs(corpus)
    n = len(pairs)
    enc = load_encoder()
    d = int(enc.get_embedding_dimension())

    pred_span = [_spans(p["prediction"]) for p in pairs]
    act_span = [_spans(p["actual"]) for p in pairs]
    asl = [len(a) for a in act_span]
    uniq = sorted({s for i in range(n) for s in pred_span[i] + act_span[i]})
    vs = enc.encode(uniq, normalize_embeddings=True)
    emap = {s: np.asarray(v, dtype=np.float32) for s, v in zip(uniq, vs)}

    X = np.stack([_padded([emap[s] for s in r], d) for r in pred_span])
    A = np.stack([_padded([emap[s] for s in r], d) for r in act_span])
    M = np.stack([np.append(np.ones(min(len(r), K), bool), np.zeros(K - min(len(r), K), bool)) for r in pred_span])

    sent_p = np.asarray(enc.encode([p["prediction"] for p in pairs], normalize_embeddings=True), dtype=np.float32)
    sent_a = np.asarray(enc.encode([p["actual"] for p in pairs], normalize_embeddings=True), dtype=np.float32)

    objs = list({p.get("objective", f"o{i}") for i, p in enumerate(pairs)})
    by_obj = defaultdict(list)
    for i, p in enumerate(pairs):
        by_obj[p.get("objective", f"o{i}")].append(i)

    rng = np.random.default_rng(args.seed)
    met, baseline, chance, win_split, win_sample = [], [], [], 0, 0
    n_ho = max(1, int(round(len(objs) * args.holdout_frac)))
    for _ in range(args.splits):
        ho = set(rng.choice(objs, size=n_ho, replace=False))
        tr = [i for o, ids in by_obj.items() if o not in ho for i in ids]
        te = [i for o, ids in by_obj.items() if o in ho for i in ids]
        if len(tr) < 5 or not te:
            continue
        Tt = np.array([_nearest_target(X[i], A[i], M[i], asl[i]) for i in tr])
        P = _train(X[tr], Tt, M[tr], d, h, steps=args.steps, seed=int(rng.integers(1, 1 << 30)))
        out, _ = _forward(X[te], M[te], P, d, h)
        pm = []
        for i in te:
            ds = []
            kk = min(asl[i], K)
            for k in range(K):
                if M[i][k] and kk > 0:
                    ds.append(min(_dist(out[te.index(i), k], A[i][j]) for j in range(kk)))
            if ds:
                pm.append(float(np.mean(ds)))
        met.append(float(np.mean(pm)) if pm else 1.0)
        baseline.append(float(np.mean([_dist(sent_p[i], sent_a[i]) for i in te])))
        chance.append(float(np.mean([_dist(sent_p[i], sent_a[j]) for i in te for j in tr])))
        for idx, i in enumerate(te):
            if idx < len(pm) and pm[idx] < _dist(sent_p[i], sent_a[i]):
                win_sample += 1
        if float(np.mean(pm)) < baseline[-1]:
            win_split += 1

    nv = len(met)
    print(f"[attn] 目标级留出 {args.holdout_frac:.0%} ({n_ho}/{len(objs)}), 单头 h={h}")
    print(f"[attn] attention span surprise: mean={sum(met)/nv:.3f} ±{statistics.stdev(met)/(nv**0.5):.3f} (split 胜句子基线 {win_split}/{nv})")
    print(f"[attn] 基线(直接用 pred 句向量取 actual): mean={sum(baseline)/nv:.3f}")
    print(f"[attn] 转机(跨配 chance): mean={sum(chance)/nv:.3f}")
    print(f"[attn] 逐留出样本上 attn<句子基线 次数: {win_sample}")


if __name__ == "__main__":
    main()