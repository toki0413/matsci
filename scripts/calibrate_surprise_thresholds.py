#!/usr/bin/env python3
"""阶段0：用预置知识(seed/*.md)标定 surprise 语义距离阈值.

把种子文档分块，构造两类配对并统计语义距离分布：
  - 同主题对：同一篇文档内的两个块（应语义相近 → 距离小）
  - 跨主题对：不同篇文档的两个块（应语义相异 → 距离大）

据两类的分布给出 surprise 标签(high/low)的建议阈值，供下游
（encounter_space 的 [0,1] clip、`/flow` 的 high>0.6/low<0.3 等）参考。

距离度量：
  优先冻结句向量(sentence-transformers)余弦距离，与 engine_reflect 阶段1 同语义空间；
  ST 不可用时回退词元 Jaccard 距离（纯标准库），并如实标注当前模式。

纯离线、自包含：不 import huginn.*，不触碰单网关门禁(ADR-0001)，
仅在目标部署(已装 ST)产出语义阈值，其余环境产出回退距离阈值。

用法:
  python scripts/calibrate_surprise_thresholds.py [--seed-dir DIR] [--out OUT] [--pairs N]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path

# 与 knowledge 缺省同源的多语言小模型；可用环境变量覆盖。
EMBED_MODEL = os.environ.get(
    "HUGINN_EMBED_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"
)

_ST = None  # 惰性单例


def _load_st():
    global _ST
    if _ST is not None:
        return _ST
    try:
        from sentence_transformers import SentenceTransformer
        _ST = SentenceTransformer(EMBED_MODEL)
    except Exception:  # noqa: BLE001 — ST 不可用即回退 Jaccard
        _ST = False
    return _ST


def chunk_doc(text: str, max_chars: int = 700, max_chunks: int = 6) -> list[str]:
    """按空行拆段，再合并到 max_chars 窗口；每篇限 max_chunks，控制规模."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    cur = ""
    for p in paras:
        if cur and len(cur) + len(p) > max_chars:
            chunks.append(cur)
            cur = p
        else:
            cur = (cur + " " + p).strip() if cur else p
    if cur:
        chunks.append(cur)
    return chunks[:max_chunks]


def jaccard_dist(a: str, b: str) -> float:
    """词元 Jaccard 距离 = 1 - |A∩B|/|A∪B| (与 engine_reflect 旧 surprise 同语义)."""
    ta = set(re.findall(r"[A-Za-z0-9_]+", a.lower()))
    tb = set(re.findall(r"[A-Za-z0-9_]+", b.lower()))
    if not ta or not tb:
        return 0.0
    return 1.0 - len(ta & tb) / len(ta | tb)


def _vec(text: str):
    st = _load_st()
    if not st:
        return None
    import numpy as np
    return np.asarray(st.encode([text], normalize_embeddings=True)[0], dtype=np.float32)


def dist(a: str, b: str) -> float:
    """语义优先(余弦距离)，ST 不可用回退 Jaccard；返回 [0,1]."""
    va = _vec(a)
    vb = _vec(b)
    if va is None or vb is None:
        return jaccard_dist(a, b)
    import numpy as np
    na = va / (np.linalg.norm(va) + 1e-12)
    nb = vb / (np.linalg.norm(vb) + 1e-12)
    return max(0.0, min(1.0, 1.0 - float(np.dot(na, nb))))


def mode() -> str:
    return "semantic" if _load_st() else "jaccard"


def _percentile(vals: list[float], pct: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    idx = min(len(s) - 1, int(round(pct / 100.0 * (len(s) - 1))))
    return s[idx]


def _stats(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0, "mean": 0.0, "p10": 0.0, "median": 0.0, "p90": 0.0}
    return {
        "n": len(vals),
        "mean": round(sum(vals) / len(vals), 4),
        "p10": round(_percentile(vals, 10), 4),
        "median": round(_percentile(vals, 50), 4),
        "p90": round(_percentile(vals, 90), 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed-dir", default=str(Path(__file__).resolve().parents[1]
                                              / "agent/huginn/knowledge/seed"))
    ap.add_argument("--out", default="", help="可选: 写 JSON 结果到该路径")
    ap.add_argument("--pairs", type=int, default=3000, help="跨主题对采样上限")
    ap.add_argument("--seed", type=int, default=0, help="RNG 种子，保证可复现")
    args = ap.parse_args()

    seed_dir = Path(args.seed_dir)
    files = sorted(seed_dir.glob("*.md"))
    if not files:
        print(f"seed 目录无 .md: {seed_dir}", file=__import__("sys").stderr)
        return 1

    random.seed(args.seed)
    blocks: list[tuple[str, int, str]] = []  # (doc_stem, chunk_idx, text)
    for fp in files:
        for i, ch in enumerate(chunk_doc(fp.read_text(encoding="utf-8", errors="ignore"))):
            blocks.append((fp.stem, i, ch))

    # 分文档索引
    by_doc: dict[str, list[tuple[int, str]]] = {}
    for stem, idx, ch in blocks:
        by_doc.setdefault(stem, []).append((idx, ch))

    # 同主题对（篇内块对）→ 期望低距离
    same: list[float] = []
    for stem, chs in by_doc.items():
        for i in range(len(chs)):
            for j in range(i + 1, len(chs)):
                same.append(dist(chs[i][1], chs[j][1]))
    # 跨主题对（篇间块对）→ 期望高距离
    stems = list(by_doc)
    cross: list[float] = []
    attempts = 0
    while len(cross) < args.pairs and attempts < args.pairs * 8:
        attempts += 1
        s1 = random.choice(stems)
        s2 = random.choice(stems)
        if s1 == s2:
            continue
        a_idx, a_text = random.choice(by_doc[s1])
        b_idx, b_text = random.choice(by_doc[s2])
        cross.append(dist(a_text, b_text))

    same_s, cross_s = _stats(same), _stats(cross)
    low_cut = same_s["p90"]     # 高于它多半是"不相似"
    high_cut = cross_s["p10"]   # 低于它多半是"相似"
    sep = "清晰可分" if low_cut < high_cut else "存在重叠带"

    print(f"[calibrate] mode={mode()} | 文档={len(files)} 块={len(blocks)} "
          f"同主题对={same_s['n']} 跨主题对={cross_s['n']}")
    print(f"  同主题距离  {same_s}")
    print(f"  跨主题距离  {cross_s}")
    print(f"  → 建议阈值: low_cutoff(低于=相似)={low_cut:.3g} "
          f"high_cutoff(高于=相异)={high_cut:.3g}  [{sep}]")
    print("  注: 语义距离分布与 Jaccard 不同; 若 mode=jaccard 仅作管线演示, 阈值需 ST 环境重标.")

    result = {
        "mode": mode(),
        "docs": len(files),
        "blocks": len(blocks),
        "same_topic": same_s,
        "cross_topic": cross_s,
        "suggested_low_cutoff": low_cut,
        "suggested_high_cutoff": high_cut,
        "separation": sep,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())