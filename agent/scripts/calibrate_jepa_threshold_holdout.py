"""阶段2-B 补充: 留出式阈值分离验证.

上一脚本用全量 same vs cross 得到 AUC~0.50 (不可分离). 这里做更严谨的验证:
对 12 次目标级留出拆分, 在训练部分上只前向冻结 predictor, 对**留出(未见)目标**求 same vs cross,
看本轮扩采后留出式 AUC 是否>0.5. 若留出目标上 AUC 仍 ~0.5, 则绝对阈值不可用是本质性的,
predictor 只适合相对秩信号 —— 这是阶段2-B 的可证伪结论.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from huginn.utils.runtime import get_runtime_home

from train_jepa_predictor import load_pairs, load_encoder

def load_predictor_weights() -> dict:
    p = get_runtime_home() / "models" / "jepa_predictor.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    return {k: np.asarray(v, dtype=np.float64) for k, v in data["weights"].items()}

def cd(a, b) -> float:
    a = a / (np.linalg.norm(a) + 1e-12); b = b / (np.linalg.norm(b) + 1e-12)
    return float(1.0 - float(np.dot(a, b)))

def forward(pred_emb, w) -> np.ndarray:
    h = np.tanh(pred_emb @ w["W1"] + w["b1"])
    return h @ w["W2"] + w["b2"]

def auc(s, c):
    rng = np.random.default_rng(0)
    s = np.asarray(s, float); c = np.asarray(c, float)
    if len(s) * len(c) > 4_000_000:
        s = rng.choice(s, min(3000, len(s)), replace=False)
        c = rng.choice(c, min(3000, len(c)), replace=False)
    pos = 0.0; ties = 0.0
    for x in s:
        pos += float(np.sum(c > x)); ties += float(np.sum(c == x))
    return (pos + 0.5 * ties) / (len(s) * len(c))

def main() -> None:
    pairs = load_pairs(get_runtime_home() / "corpus" / "jepa_pairs.jsonl")
    print(f"[t2b-holdout] pairs={len(pairs)}")
    by_obj: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(pairs):
        by_obj[r.get("objective", f"obj_{i}")].append(i)
    objs = list(by_obj.keys())
    w = load_predictor_weights()
    enc = load_encoder()
    pe = np.asarray(enc.encode([p["prediction"] for p in pairs], normalize_embeddings=True), float)
    ae = np.asarray(enc.encode([p["actual"] for p in pairs], normalize_embeddings=True), float)
    fwd = forward(pe, w)
    rng = np.random.default_rng(0)
    aucs = []
    for split in range(8):
        test_objs = set(rng.choice(objs, size=max(1, len(objs)//4), replace=False))
        test_idx = [i for o, idxs in by_obj.items() if o in test_objs for i in idxs]
        same = [cd(fwd[i], ae[i]) for i in test_idx]
        cross = [cd(fwd[i], ae[j]) for i in test_idx for j in test_idx if i != j]
        if not same:
            continue
        a = auc(same, cross)
        aucs.append(a)
        print(f"  split{split}: holdout={len(test_idx)} AUC={a:.4f} same_mean={np.mean(same):.3f} cross_mean={np.mean(cross):.3f}")
    if aucs:
        print(f"[t2b-holdout] mean AUC over splits = {np.mean(aucs):.4f} (0.5=不可分离)")

if __name__ == "__main__":
    main()