# -*- coding: utf-8 -*-
"""把 sobereva 决策启发式规则生成成 Huginn seed 预置知识.

从方法一览帖抽 '场景→方法' 决策规则 (heuristic_distiller),
输出为 seed/heuristics_chunks.jsonl, 随 agent 安装包分发.
用户检索到规则后, agent 能像 sobereva 一样按场景选方法.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\wanzh\Desktop\matsci-agent")
sys.path.insert(0, str(ROOT / "agent"))

from huginn.evolution.heuristic_distiller import distill_posts  # noqa: E402

POSTS = ROOT / "tmpSobko" / "data_sources" / "sobko_sources" / "思想家公社帖子" / "posts" / "academic"
SEED = ROOT / "agent" / "huginn" / "knowledge" / "seed" / "heuristics_chunks.jsonl"


def to_seed_line(rule) -> dict:
    methods = "、".join(rule.methods)
    return {
        "text": f"决策启发式: 场景『{rule.scenario}』推荐方法: {methods}。"
                f"当遇到『{rule.scenario}』的分析需求时, 优先考虑选用这些方法。"
                f"(来源: {rule.source})",
        "title": f"决策规则: {rule.scenario}",
        "source": "sobko_heuristic",
        "source_type": "decision_heuristic",
        "authority_level": "A",
        "canonical_url": rule.source,
        "scenario": rule.scenario,
        "method_tags": rule.methods,
        "domain": "波函数分析",
        "pre_chunked": "1",
    }


def main() -> None:
    n, rules = distill_posts(POSTS, SEED)
    # 直接用 to_seed_line 重写为带决策语义的 seed 文本
    with SEED.open("w", encoding="utf-8") as f:
        for r in rules:
            f.write(json.dumps(to_seed_line(r), ensure_ascii=False) + "\n")
    print(f"决策启发式 seed 写入 {SEED}: {len(rules)} 条")


if __name__ == "__main__":
    main()