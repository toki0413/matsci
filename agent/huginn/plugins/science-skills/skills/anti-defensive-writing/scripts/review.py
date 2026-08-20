#!/usr/bin/env python3
"""anti-defensive-writing 检测脚本 (启发式).

定位常见"防御性写作"措辞: 免责声明 / not-X-but-Y / 过度 hedge / 首句限制 /
冗余过渡。只做命中定位, 改写交给 LLM 依据 hit 清单 + SKILL.md 规则完成。
只依赖标准库, 供 science-skills bridge 以 `uv run review.py --query ...` 调用。
"""
from __future__ import annotations

import argparse
import json
import re
import sys

# 各类防御性措辞 (小写匹配). pattern -> 修复提示.
_PATTERNS = [
    # 免责/否定主张
    (re.compile(r"\b(?:we|this paper|this study|this work|the (?:review|model|method|analysis))\s+(?:do(?:es)? not|don'?t|is not|are not)\s+(?:claim|intend(?:ed)?|attempt|aim)\b"),
     "免责声明: 不要宣称‘不声称什么’, 改为正面说明文本‘考察/证明/贡献’什么"),
    (re.compile(r"\bthis (?:is|should) not (?:to say|be taken to mean)\b"), "冗余否定: 删掉 / 改为正面主张"),
    (re.compile(r"\bnot intended to provide\b"), "负向框定: 直接给出本文的贡献"),
    # not-X-but-Y / rather-than
    (re.compile(r"\bnot (.+?) but (.+?)\b", re.IGNORECASE), "not-X-but-Y 结构: 只保留正向那一半"),
    (re.compile(r"\brather than\b|\binstead of\b", re.IGNORECASE), "对比式防御: 仅在对比本身属于论证时才保留"),
    # 过度 hedge
    (re.compile(r"\b(potentially|might|could|may)\b", re.IGNORECASE),
     "hedge: 若不确定性真实, 指名来源与范围; 否则去掉用精确主张替代"),
    # 以限制/转折开头的段落
    (re.compile(r"^(?:while |although |though |despite |even though )", re.IGNORECASE | re.MULTILINE),
     "以限制开头: 先给出主张, 限制移到方法/讨论/局限节"),
    # 冗余过渡/圈定
    (re.compile(r"\bit (?:is|should be) (?:worth )?noting that\b", re.IGNORECASE), "冗余过渡: 删除"),
    (re.compile(r"\bto be clear\b", re.IGNORECASE), "冗余圈定: 删除"),
    (re.compile(r"\bof course, (?:this|the) does not fully capture\b", re.IGNORECASE), "自我削弱: 删除或改用积极限定"),
    (re.compile(r"\b(?:although|however|nevertheless)\b", re.IGNORECASE), "冗余转折: 仅在需要真实对比时保留"),
]

_SENT_BOUND = re.compile(r"(?<=[.!?。！？])\s+")


def _positions(text: str) -> list[dict]:
    """逐句启发式扫描, 返回命中列表."""
    hits: list[dict] = []
    for si, sent in enumerate(_SENT_BOUND.split(text)):
        low = sent.lower()
        for pat, hint in _PATTERNS:
            m = pat.search(low)
            if m:
                hits.append({
                    "sentence_index": si,
                    "pattern": pat.pattern,
                    "matched_span": sent[max(0, m.start()):m.end()],
                    "sentence": sent.strip()[:280],
                    "fix_hint": hint,
                    "level": "warning",
                })
    return hits


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--query", default=None, help="要审查的文本")
    ap.add_argument("--output", default=None, help="JSON 输出路径 (默认 stdout)")
    args = ap.parse_args()

    text = args.query
    # 无 --query 时允许从 stdin 读, 便于管道
    if text is None and not sys.stdin.isatty():
        text = sys.stdin.read()
    if not text:
        print("no text given (use --query or stdin)", file=sys.stderr)
        return 2

    result = {
        "skill": "anti-defensive-writing",
        "hits": _positions(text),
        "count": 0,
        "summary": None,
    }
    result["count"] = len(result["hits"])
    result["summary"] = (
        f"found {len(result['hits'])} defensive-writing marker(s). "
        f"Review each hit; keep only limits that affect accuracy/scope/method, "
        f"and revise the rest claim-first per SKILL.md."
    )

    blob = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(blob)
    else:
        print(blob)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
