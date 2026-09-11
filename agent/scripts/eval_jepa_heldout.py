"""按物理目标做留出验证: 测 JEPA predictor 泛化 与 语义 surprise 的可辨性.

把语料按 `objective` 分组; 随机分多次 train/test **目标级** 拆分(留出的目标在训练里完全没见过),
在留出目标上回答两个问题:
  (a) 语义 surprise 是否有用: 留出目标里 pred↔自己actual 的距离 显著低于 pred↔其他actual 吗?
  (b) predictor 是否泛化: 冻结前向在留出目标上的重建 surprise 显著低于基线(直接用pred)与转机(跨配)吗?

⚠ 诚实: n=36、目标少, 这是管线级/信号级验证, 不是"模型已可靠"的证明.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from huginn.utils.runtime import get_runtime_home
from scripts.train_jepa_predictor import load_pairs, load_encoder, train_predictor, forward_frozen
from scripts.collect_jepa_batch import OBJECTIVES

_LEGACY = ["orig_pendulum", "orig_pendulum", "orig_freefall", "orig_spring"]


def _tag_objectives(rows: list[dict]) -> None:
    """按采集顺序补 objective 标签; 已带则跳过."""
    if rows and all("objective" in r for r in rows):
        return
    for i, r in enumerate(rows):
        if "objective" in r:
            continue
        if i < len(_LEGACY):
            r["objective"] = _LEGACY[i]
        else:
            j = i - len(_LEGACY)
            r["objective"] = OBJECTIVES[j][0] if j < len(OBJECTIVES) else f"extra_{i}"


def _dist(a, b):
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(1 - float(np.dot(a, b)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--splits", type=int, default=12, help="随机目标级拆分次数")
    ap.add_argument("--holdout-frac", type=float, default=0.25, help="留出目标比例")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    corpus = Path(args.corpus) if args.corpus else (get_runtime_home() / "corpus" / "jepa_pairs.jsonl")
    rows = load_pairs(corpus)
    _tag_objectives(rows)
    # 持久化 objective 标签回语料 (自描述)
    corpus.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")

    by_obj: dict[str, list] = defaultdict(list)
    for i, r in enumerate(rows):
        by_obj[r.get("objective", f"obj_{i}")].append(i)
    obj_names = list(by_obj.keys())
    print(f"[heldout] corpus={corpus} rows={len(rows)} objectives={len(obj_names)}")

    enc = load_encoder()
    all_pred = np.asarray(enc.encode([r["prediction"] for r in rows], normalize_embeddings=True), dtype=np.float64)
    all_act = np.asarray(enc.encode([r["actual"] for r in rows], normalize_embeddings=True), dtype=np.float64)
    dim = int((getattr(enc, "get_embedding_dimension", None) or getattr(enc, "get_sentence_embedding_dimension"))())

    rng = np.random.default_rng(args.seed)
    met, baseline, chance, win_pred, win_metric = [], [], [], 0, 0
    n_ho_per_split = max(1, int(round(len(obj_names) * args.holdout_frac)))
    for _ in range(args.splits):
        ho = set(rng.choice(obj_names, size=n_ho_per_split, replace=False))
        tr_idx = [i for o, ids in by_obj.items() if o not in ho for i in ids]
        te_idx = [i for o, ids in by_obj.items() if o in ho for i in ids]
        if len(tr_idx) < 5 or not te_idx:
            continue
        X, Y = all_pred[tr_idx], all_act[tr_idx]
        pack, _ = train_predictor(X, Y, dim, steps=args.steps, seed=int(rng.integers(1, 1 << 30)))
        # (a) 语义 surprise 信号: 留出目标 pred↔自己actual vs pred↔train的其他actual
        md = np.mean([_dist(all_pred[i], all_act[i]) for i in te_idx])
        # 同目标内 pred↔other-actual (近期"不一致"参考, 用跨目标 actual)[注: 拿不到同目标负例, 用跨目标]
        cross = [_dist(all_pred[i], all_act[j]) for i in te_idx for j in tr_idx]
        ba = float(np.mean([_dist(all_pred[i], all_act[i]) for i in te_idx]))  # baseline = 直接用pred
        # predictor 冻结前向: 留出目标重建
        fwd = forward_frozen(pack, all_pred[te_idx])
        pm = float(np.mean([_dist(fwd[k], all_act[te_idx[k]]) for k in range(len(te_idx))]))
        met.append(pm)
        baseline.append(ba)
        chance.append(float(np.mean(cross)))
        if pm < ba:
            win_pred += 1
        if md <= float(np.percentile(cross, 10)):
            win_metric += 1

    if not met:
        print("[heldout] 无有效拆分, 中止")
        return
    n = len(met)
    print(f"[heldout] 目标级留出 {args.holdout_frac:.0%} ({n_ho_per_split}/{len(obj_names)}), {args.splits} 次随机拆分")
    print(f"[heldout] predictor 冻结前向重建 surprise: mean={sum(met)/n:.3f} ±{statistics.stdev(met)/ (n**0.5):.3f} (win vs 基线: {win_pred}/{n})")
    print(f"[heldout] 基线(直接用pred取actual): mean={sum(baseline)/n:.3f}")
    print(f"[heldout] 转机(跨配 chance):         mean={sum(chance)/n:.3f}")
    print(f"[heldout] 语义 surprise 可辨性: 留出目标同配距离落在跨配 p10 以里的次数 {win_metric}/{n}")

    # 汇总留出目标上的语义度量可辨性(合并所有拆分)
    print("\n[小结] predictor 是否泛化取决于: 重建 surprise < 跨配转机 且 显著 < 基线; 若≈跨配则只是记住了训练目标.")


if __name__ == "__main__":
    main()