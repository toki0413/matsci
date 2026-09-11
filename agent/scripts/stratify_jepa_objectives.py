"""目标难度分层 + 高混淆定位 (阶段2-C 补采对抗样本的前置).

做法(全部用冻结 predictor 前向, 守 §4.3 只前向不更新):
  对每个目标 i:
    self_err   = dist( g(φ(pred_i)), φ(actual_i) )          # 重建自己 actual 的误差
    nearest_j  = argmin_{j≠i} dist( g(φ(pred_i)), φ(actual_j) ) # 最像的"他人 actual"
    margin     = self_err - dist(g(φ(pred_i)), φ(actual_{nearest_j}))
  margin>0 ⇒ 自己的 actual 比任何他人的都近 ⇒ 易/自信;
  margin<0 ⇒ predictor 把该预测投影得离"别人的 actual"更近 ⇒ 可混淆/难 = 对抗弱点候选.

按 margin 分桶 easy/medium/hard, 并按目标名粗族聚合输出最弱族(供设计对抗样本).
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from huginn.utils.runtime import get_runtime_home
from scripts.train_jepa_predictor import (
    load_pairs, load_encoder, train_predictor, forward_frozen,
)
from scripts.collect_jepa_batch import OBJECTIVES


def _dist(a, b):
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(1 - float(np.dot(a, b)))


def _family(name: str) -> str:
    low = name.lower()
    words = [
        "pendulum", "spring", "projectile", "wave", "rc", "capac", "gas",
        "work", "energy", "collision", "momentum", "thermal", "gravity",
        "integral", "matrix", "eigen", "newton", "grad", "jacob", "interpol",
        "gradient", "fourier", "simpson", "markov", "curl", "det", "inner",
        "cross", "cauchy", "vector", "maxwell", "bessel", "monte", "stat",
        "erro", "converg", "stabil", "harmon", "distribution",
    ]
    for w in words:
        if w in low:
            return w
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top", type=int, default=20, help="打印最难/最易前 N")
    args = ap.parse_args()

    corpus = Path(args.corpus) if args.corpus else (get_runtime_home() / "corpus" / "jepa_pairs.jsonl")
    pairs = load_pairs(corpus)
    print(f"[strat] corpus={corpus} rows={len(pairs)}")
    if len(pairs) < 8:
        print("[strat] 语料太少, 跳过分层")
        return

    enc = load_encoder()
    dim = int(enc.get_embedding_dimension())
    pred = np.asarray(enc.encode([p["prediction"] for p in pairs], normalize_embeddings=True), dtype=np.float64)
    act = np.asarray(enc.encode([p["actual"] for p in pairs], normalize_embeddings=True), dtype=np.float64)

    pack, _ = train_predictor(pred, act, dim, steps=args.steps, seed=args.seed)
    fwd = forward_frozen(pack, pred)

    n = len(pairs)
    # 全行两两 dist(g(pred_i), actual_j)
    D = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            D[i, j] = _dist(fwd[i], act[j])

    rows = []
    for i in range(n):
        self_err = D[i, i]
        others = [j for j in range(n) if j != i]
        jn = min(others, key=lambda j: D[i, j])
        nearest_err = D[i, jn]
        margin = self_err - nearest_err
        is_confused = jn != i  # nearest actual 不是自己的
        obj = (pairs[i].get("objective") or f"obj_{i}")
        rows.append({
            "i": i, "objective": obj, "family": _family(obj),
            "self_err": self_err, "nearest_err": nearest_err,
            "nearest_obj": (pairs[jn].get("objective") or f"obj_{jn}"),
            "margin": margin, "confused": is_confused,
        })

    margins = np.array([r["margin"] for r in rows])
    lo, hi = np.percentile(margins, [33, 67])
    for r in rows:
        r["bucket"] = "hard" if r["margin"] <= lo else ("med" if r["margin"] <= hi else "easy")

    print("\n── 分桶统计 ────────────────────────────")
    for b in ("easy", "med", "hard"):
        sub = [r for r in rows if r["bucket"] == b]
        print(f"{b:5s}: {len(sub):3d} 目标 | mean_margin={np.mean([r['margin'] for r in sub]):+.3f} "
              f"confused={sum(1 for r in sub if r['confused'])}")

    print("\n── 最易 Top", args.top, "────────────────────")
    for r in sorted(rows, key=lambda r: -r["margin"])[: args.top]:
        print(f"  {r['objective']:32s} self={r['self_err']:.3f} margin={r['margin']:+.3f} [{r['bucket']}]")

    print("\n── 最难/高混淆 Top", args.top, "────────────")
    for r in sorted(rows, key=lambda r: r["margin"])[: args.top]:
        mark = " 🚩confused" if r["confused"] else ""
        print(f"  {r['objective']:32s} self={r['self_err']:.3f} nearest={r['nearest_err']:.3f}→{r['nearest_obj'][:24]:24s} "
              f"margin={r['margin']:+.3f} [{r['bucket']}]{mark}")

    print("\n── 族级聚合(按 margin 均值)────────────")
    fam = defaultdict(list)
    for r in rows:
        fam[r["family"]].append(r["margin"])
    for f, ms in sorted(fam.items(), key=lambda kv: np.mean(kv[1])):
        print(f"  {f:16s} n={len(ms):3d} mean_margin={np.mean(ms):+.3f}"
              + ("  ⚠弱族" if np.mean(ms) < 0 else ""))


if __name__ == "__main__":
    main()