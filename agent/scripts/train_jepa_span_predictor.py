"""Span-level 结构化输出编码的 JEPA 预测器 (阶段2-C 方向验证).

动机: 句子级句向量把"多行数值 + 判据" actual 压成一个向量, 导致短数值型目标在潜空间
互塌缩 (阶段2-C 弱族). 这里把 prediction/actual 拆成**行级 span 向量**, 保"数字行"与
"判据行"分立, 用逐 span 共享 MLP 预测: 每个预测 span 回归到其所在 pair 的最近真实 span.

表示:
  text → splitlines → 非空 span → span 向量 (768, mpnet)
  每条 = 定长张量 (K, d), span 按其文本排序(确定性), 尾部零填充 + 掩码.

训练: 单隐层逐 span 共享 MLP (d→64→d). 对每个非掩码预测 span x:
       目标 t = 同 pair actual 里离 x 最近的 span;  L2(g(x), t) 回归 (掩码外不贡献).

诊断:
  - 目标级留出: span predictor 前向 surprise 与 句子级检出混淆。
  - 全程配对混淆率: 每个 预测span前向 的最邻近真实span 是否来自本pair (=是否被别的目标吸走).
"""
from __future__ import annotations

import argparse
import re
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from huginn.utils.runtime import get_runtime_home
from scripts.train_jepa_predictor import load_pairs, load_encoder

K = 8

_NUM_RE = re.compile(r"-?\d+\.\d+(?:[eE][+-]?\d+)?|-?\d+[eE][+-]?\d+")


def _canon(text: str) -> str:
    """把数字词元规范成统一 %g 格式, 消除 pred/actual 的数值格式噪声(1.0 vs 1.0000)。
    只改数值 token, 不改标签/单位/判据文本 —— 结构化字段保持不变."""
    return _NUM_RE.sub(lambda m: f"{float(m.group(0)):.6g}", text)


def _dist(a, b):
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(1 - float(np.dot(a, b)))


def _spans(text: str) -> list[str]:
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def _padded(vecs: list[np.ndarray], d: int) -> np.ndarray:
    m = np.zeros((K, d), dtype=np.float32)
    for k, v in enumerate(vecs[:K]):
        m[k] = v
    return m


def _train_mlp(X, T, M, d, steps=200, lr=1e-3, seed=0) -> dict:
    """X,T:(N,K,d) 逐 span; M:(N,K) bool 掩码; 逐 span 共享 MLP d->64->d."""
    rng = np.random.default_rng(seed)
    W1 = rng.standard_normal((d, 64)) * 0.02
    b1 = np.zeros(64)
    W2 = rng.standard_normal((64, d)) * 0.02
    b2 = np.zeros(d)
    for _ in range(steps):
        H = np.tanh(X @ W1 + b1)
        out = H @ W2 + b2
        g2 = 2 * (out - T) * M[..., None]
        gradW2 = H.transpose(0, 2, 1) @ g2
        gradb2 = g2.sum(axis=(0, 1))
        gradH = g2 @ W2.T
        gradb1_nk = gradH * (1 - H * H)          # (N,K,64)
        gradb1 = gradb1_nk.sum(axis=(0, 1))       # (64,)
        gradW1 = np.einsum("nke,nkd->ed", gradb1_nk, X)  # (64,d)
        W1 -= lr * gradW1.T
        b1 -= lr * gradb1
        W2 -= lr * gradW2.sum(axis=0)
        b2 -= lr * gradb2
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


def _fwd(pack, X):
    H = np.tanh(X @ pack["W1"] + pack["b1"])
    return H @ pack["W2"] + pack["b2"]


def _nearest_target(Xi, Ai, Mi, ass_i_len):
    """对 pair i 的每个非掩码预测 span, 目标 = 同 pair 最近真实 span."""
    T = np.zeros_like(Xi)
    kk = min(ass_i_len, K)
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
    ap.add_argument("--canon", action="store_true", help="结构化数值规范化: 统一数字 token 格式")
    args = ap.parse_args()

    corpus = Path(args.corpus) if args.corpus else (get_runtime_home() / "corpus" / "jepa_pairs.jsonl")
    pairs = load_pairs(corpus)
    n = len(pairs)
    enc = load_encoder()
    d = int(enc.get_embedding_dimension())

    pred_span_list = [_spans(p["prediction"]) for p in pairs]
    act_span_list = [_spans(p["actual"]) for p in pairs]
    asl = [len(a) for a in act_span_list]  # 每 pair 真实 span 数
    if args.canon:
        pred_span_list = [[_canon(s) for s in r] for r in pred_span_list]
        act_span_list = [[_canon(s) for s in r] for r in act_span_list]

    uniq = sorted({s for idx in range(n) for s in pred_span_list[idx] + act_span_list[idx]})
    vs = enc.encode(uniq, normalize_embeddings=True)
    emap = {s: np.asarray(v, dtype=np.float32) for s, v in zip(uniq, vs)}

    X = np.stack([_padded([emap[s] for s in pred_span_list[i]], d) for i in range(n)])
    A = np.stack([_padded([emap[s] for s in act_span_list[i]], d) for i in range(n)])
    M = np.stack([np.concatenate([np.ones(min(len(pred_span_list[i]), K), bool), np.zeros(K - min(len(pred_span_list[i]), K), bool)]) for i in range(n)])

    # 句子级对照基线 (同一预处理: canon 时同样规范化)
    _sp = [p["prediction"] for p in pairs]
    _sa = [p["actual"] for p in pairs]
    if args.canon:
        _sp = [_canon(t) for t in _sp]
        _sa = [_canon(t) for t in _sa]
    sent_p = np.asarray(enc.encode(_sp, normalize_embeddings=True), dtype=np.float32)
    sent_a = np.asarray(enc.encode(_sa, normalize_embeddings=True), dtype=np.float32)

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
        pack = _train_mlp(X[tr], Tt, M[tr], d, steps=args.steps, seed=int(rng.integers(1, 1 << 30)))
        fwd = _fwd(pack, X[te])
        pm = []
        for i in te:
            ds = []
            kk = min(asl[i], K)
            for k in range(K):
                if M[i][k] and kk > 0:
                    ds.append(min(_dist(fwd[te.index(i), k], A[i][j]) for j in range(kk)))
            if ds:
                pm.append(float(np.mean(ds)))
        met.append(float(np.mean(pm)) if pm else 1.0)
        ba = float(np.mean([_dist(sent_p[i], sent_a[i]) for i in te]))
        baseline.append(ba)
        chance.append(float(np.mean([_dist(sent_p[i], sent_a[j]) for i in te for j in tr])))
        for idx, i in enumerate(te):
            if idx < len(pm) and pm[idx] < _dist(sent_p[i], sent_a[i]):
                win_sample += 1
        if float(np.mean(pm)) < ba:
            win_split += 1

    nv = len(met)
    print(f"[span] 目标级留出 {args.holdout_frac:.0%} ({n_ho}/{len(objs)}), {args.splits} 次")
    print(f"[span] span predictor 前向 surprise: mean={sum(met)/nv:.3f} ±{statistics.stdev(met)/(nv**0.5):.3f} (split 胜句子基线 {win_split}/{nv})")
    print(f"[span] 基线(直接用 pred 句向量取 actual): mean={sum(baseline)/nv:.3f}")
    print(f"[span] 转机(跨配 chance): mean={sum(chance)/nv:.3f}")
    print(f"[span] 逐留出样本上 span<句子基线 次数: {win_sample}")

    # 全程配对混淆诊断: 预测 span 前向的最近真实 span 是否来自本 pair
    T_full = np.array([_nearest_target(X[i], A[i], M[i], asl[i]) for i in range(n)])
    pack_full = _train_mlp(X, T_full, M, d, steps=args.steps, seed=args.seed)
    fwdf = _fwd(pack_full, X)
    self_confused, all_span = 0, 0
    for i in range(n):
        kk = min(asl[i], K)
        for k in range(K):
            if not M[i][k] or kk == 0:
                continue
            all_span += 1
            # 最近真实 span 的 (pair, j)
            best = min(
                ((_dist(fwdf[i, k], A[j][jj]), (j, jj))
                 for j in range(n) for jj in range(min(asl[j], K))),
                key=lambda t: t[0],
            )
            self_confused += best[1][0] != i
    print(f"[span-混淆] 全程: {self_confused}/{all_span} 预测span 最近真实span 来自别的 pair "
          f"({100*self_confused/max(1,all_span):.1f}%). 短数值型跨目标吸附为 {self_confused} 处."
          f"\n   (对比: 句子级阶段2-C 时对抗样本 10/10 confused)")


if __name__ == "__main__":
    main()