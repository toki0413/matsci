"""数值感知编码的 span-level JEPA predictor — 方案① 表征侧重实验.

动机: 0.28 信息下限在纯文本通道下是冻结 mpnet 对数值的**压损** —— 数字当普通
token, 词向量不保量级。这里给 predictor 一条**绕过 ST 的数值旁路**: 把每个 span 的
字面 float 用 RFF(正弦核, log 量级)精确向量化, 与 span 文本嵌入拼接后进逐 span 共享
MLP。目标不变 (最近真实 span 文本嵌入), actual 侧不经数值通道 —— 保证对照公平。

对照 (同分划、同 seed, 只变 prediction 侧输入):
  - pure-text: 输入 = span 文本嵌入 (d)  -> 现有 0.287 基线
  - numeric:   输入 = [span 文本嵌入 ∥ RFF 数值特征] (d + 2F+2)

同一套 held-out 目标级分划, 各训各的 MLP, 同 target (actual 文本嵌入)。若 numeric 的
目标级留出 surprise < pure-text, 说明数值旁路被 predictor 学会读了 → 机制成立。

设计细节 (定稿):
  - 数值向量: 每值 v → sign(s) + 量级 a=log10|v| (归一化 u=clip(a/24)) →
    [cos(w_j·π·u), sin(w_j·π·u)]_{j=1..F}; w_j 固定 random 高斯频率 (seed)。
    无值 span → 全零 + count=0。多值 span 取均值 + count + sign 均值。
  - 融合: 输入维度 D = d + (2F+2); MLP D -> 64 -> d (输出仍 d, 对齐目标)。
"""
from __future__ import annotations

import argparse
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from huginn.utils.runtime import get_runtime_home
from scripts.train_jepa_predictor import load_pairs, load_encoder, JEPA_EMBED_MODEL

K = 8          # 与 span predictor 一致的定长张量槽数
U_SCALE = 24.0  # log10 量级归一上界 (覆盖 ~1e-12 .. 1e12)

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")


def _spans(text: str) -> list[str]:
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def _dist(a, b):
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(1 - float(np.dot(a, b)))


def _rff_bank(freqs: int, seed: int = 123) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(freqs)


def _numeric_feat(span_text: str, W: np.ndarray) -> np.ndarray:
    """span 的 RFF 数值特征: (2F+2,). [cos/sin] xF + sign均值 + count. 输出缩放到 ~0.2 级, 防输入不稳定."""
    F = int(len(W))
    out = np.zeros(2 * F + 2, dtype=np.float32)
    nums = [float(m) for m in _NUM_RE.findall(span_text or "")]
    if not nums:
        return out
    out[-1] = min(1.0, len(nums) / 20.0)  # 归一化 count
    inv = 1.0 / len(nums)
    ssum = 0.0
    for v in nums:
        a = math.log10(abs(v) + 1e-12)
        u = max(-1.0, min(1.0, a / U_SCALE))
        s = 1.0 if v > 0 else (-1.0 if v < 0 else 0.0)
        ssum += s
        for j in range(F):
            ang = W[j] * math.pi * u
            out[2 * j] += math.cos(ang)
            out[2 * j + 1] += math.sin(ang)
    out[: 2 * F] *= inv
    out[2 * F] = ssum / len(nums)
    return out * 0.2


def _padded(vecs: list[np.ndarray], dim: int) -> np.ndarray:
    m = np.zeros((K, dim), dtype=np.float32)
    for k, v in enumerate(vecs[:K]):
        m[k] = v
    return m


def _train_mlp(X, T, M, d_in: int, d_out: int, steps=200, lr=1e-3, seed=0) -> dict:
    """逐 span 共享 MLP: d_in -> 64 -> d_out. 仅掩码 span 贡献梯度."""
    rng = np.random.default_rng(seed)
    W1 = rng.standard_normal((d_in, 64)) * 0.02
    b1 = np.zeros(64)
    W2 = rng.standard_normal((64, d_out)) * 0.02
    b2 = np.zeros(d_out)
    for _ in range(steps):
        H = np.tanh(X @ W1 + b1)
        out = H @ W2 + b2
        g2 = 2 * (out - T) * M[..., None]
        gradW2 = H.transpose(0, 2, 1) @ g2
        gradb2 = g2.sum(axis=(0, 1))
        gradH = g2 @ W2.T
        gradb1_nk = gradH * (1 - H * H)             # (N,K,64)
        gradb1 = gradb1_nk.sum(axis=(0, 1))
        gradW1 = np.einsum("nke,nkd->ed", gradb1_nk, X)  # (64,d_in)
        W1 -= lr * gradW1.T
        b1 -= lr * gradb1
        W2 -= lr * gradW2.sum(axis=0)
        b2 -= lr * gradb2
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


def _fwd(pack, X):
    H = np.tanh(X @ pack["W1"] + pack["b1"])
    return H @ pack["W2"] + pack["b2"]


def _nearest_target(Xi, Ai, Mi, kk):
    """pair i: 每非掩码预测 span 目标 = 同 pair 最近真实 span (返回 (K,d))."""
    T = np.zeros_like(Xi)
    for k in range(K):
        if not Mi[k] or kk == 0:
            continue
        T[k] = min((Ai[j] for j in range(kk)), key=lambda a: _dist(Xi[k], a))
    return T


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--splits", type=int, default=12)
    ap.add_argument("--holdout-frac", type=float, default=0.25)
    ap.add_argument("--freqs", type=int, default=16, help="RFF 频率数 F")
    args = ap.parse_args()

    corpus = Path(args.corpus) if args.corpus else (get_runtime_home() / "corpus" / "jepa_pairs.jsonl")
    pairs = load_pairs(corpus)
    n = len(pairs)
    enc = load_encoder()
    d = int(enc.get_embedding_dimension())

    pred_span_list = [_spans(p["prediction"]) for p in pairs]
    act_span_list = [_spans(p["actual"]) for p in pairs]
    asl = [len(a) for a in act_span_list]

    pieces = sorted({s for r in pred_span_list + act_span_list for s in r})
    vs = enc.encode(pieces, normalize_embeddings=True)
    emap = {s: np.asarray(v, dtype=np.float32) for s, v in zip(pieces, vs)}

    def _rowv(span: str) -> np.ndarray:
        return emap[span]

    E = np.stack([_padded([_rowv(s) for s in pred_span_list[i]], d) for i in range(n)])
    A = np.stack([_padded([_rowv(s) for s in act_span_list[i]], d) for i in range(n)])
    M = np.stack([
        np.concatenate([np.ones(min(len(pred_span_list[i]), K), bool),
                        np.zeros(K - min(len(pred_span_list[i]), K), bool)])
        for i in range(n)
    ])

    W = _rff_bank(args.freqs, seed=1234)
    dn = 2 * int(len(W)) + 2
    Num = np.stack([
        _padded([_numeric_feat(s, W) for s in pred_span_list[i]], dn)
        for i in range(n)
    ])
    Xnum = np.concatenate([E, Num.astype(np.float32)], axis=-1).astype(np.float64)

    objs = list({p.get("objective", f"o{i}") for i, p in enumerate(pairs)})
    by_obj = defaultdict(list)
    for i, p in enumerate(pairs):
        by_obj[p.get("objective", f"o{i}")].append(i)

    rng = np.random.default_rng(args.seed)
    # 两个 variant 共享同分划, 对比公平
    met_text, met_num = [], []
    win_split = 0
    n_ho = max(1, int(round(len(objs) * args.holdout_frac)))
    for _ in range(args.splits):
        ho = set(rng.choice(objs, size=n_ho, replace=False))
        tr = [i for o, ids in by_obj.items() if o not in ho for i in ids]
        te = [i for o, ids in by_obj.items() if o in ho for i in ids]
        if len(tr) < 5 or not te:
            continue
        T = np.array([_nearest_target(E[i], A[i], M[i], asl[i]) for i in tr])

        pack_t = _train_mlp(E[tr], T, M[tr], d, d, steps=args.steps,
                            seed=int(rng.integers(1, 1 << 30)))
        pack_n = _train_mlp(Xnum[tr], T, M[tr], d + dn, d, steps=args.steps,
                            seed=int(rng.integers(1, 1 << 30)))

        def _surprise(pack, Xin):
            pm = []
            for i in te:
                fwd = _fwd(pack, Xin[[i]])[0]
                ds = []
                kk = min(asl[i], K)
                for k in range(K):
                    if M[i][k] and kk > 0:
                        ds.append(min(_dist(fwd[k], A[i][j]) for j in range(kk)))
                if ds:
                    pm.append(float(np.mean(ds)))
            return float(np.mean(pm)) if pm else 1.0

        st = _surprise(pack_t, E)
        sn = _surprise(pack_n, Xnum)
        met_text.append(st)
        met_num.append(sn)
        if sn < st - 1e-9:
            win_split += 1

    nv = len(met_text)
    print(f"[numeric] 目标级留出 {args.holdout_frac:.0%} ({n_ho}/{len(objs)}), {args.splits} 次, "
          f"F={args.freqs}, 数值通道 dim={dn}")
    print(f"[text]   span surprise: mean={statistics.mean(met_text):.4f} ±{statistics.stdev(met_text)/(nv**0.5):.4f}")
    print(f"[num]    span surprise: mean={statistics.mean(met_num):.4f} ±{statistics.stdev(met_num)/(nv**0.5):.4f}")
    print(f"[num-text] delta = {statistics.mean(met_num) - statistics.mean(met_text):+.4f}; "
          f"split 胜纯文本 {win_split}/{nv}")


if __name__ == "__main__":
    main()