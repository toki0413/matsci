"""涌现复杂度 (Emergent Complexity) 指标.

灵感来自 Cell Patterns 论文 "Quantifying emergent complexity in science":
复杂系统的行为不能用单一维度衡量, 需要多维度组合.

对 agent 输出做 4 个维度的度量:
  1. tool_diversity — 用了多少种不同的工具 (Shannon 熵)
  2. reasoning_entropy — 推理文本的词频分布熵 (越高越不像模板)
  3. cross_domain — 输出是否跨了多个材料科学子领域
  4. novelty — 是否包含 KB/记忆里没有的新结构 (近似: 非常见关键词比例)

最终 EC = 各维度的几何平均 (任一维度为 0 就拉低总分).
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any


def _shannon_entropy(items: list[str]) -> float:
    """Shannon 熵, 以 ln 为底. 空列表返回 0."""
    if not items:
        return 0.0
    counts = Counter(items)
    n = len(items)
    h = 0.0
    for c in counts.values():
        p = c / n
        h -= p * math.log(p)
    return h


# 材料科学子领域关键词, 用于 cross_domain 判定
# 覆盖 5 大类, 命中不同类越多 cross_domain 越高
_DOMAIN_KEYWORDS: list[set[str]] = [
    # 结构
    {"crystal", "lattice", "symmetry", "space group", "diffraction", "晶格", "晶体", "对称"},
    # 热力学
    {"phase", "energy", "thermodynamic", "enthalpy", "gibbs", "相图", "热力学", "焓"},
    # 电子结构
    {"band", "dos", "electronic", "gap", "fermi", "能带", "电子", "带隙"},
    # 力学
    {"stress", "strain", "modulus", "elastic", "mechanical", "应力", "应变", "模量"},
    # 催化/化学
    {"catalys", "adsorption", "reaction", "active site", "催化", "吸附", "反应"},
]


def _extract_tool_names(execution_result: dict[str, Any]) -> list[str]:
    """从 execution_result 抽工具名列表."""
    calls = (
        execution_result.get("tool_calls")
        or execution_result.get("steps")
        or execution_result.get("actions")
        or []
    )
    if not isinstance(calls, list):
        return []
    names: list[str] = []
    for call in calls:
        if isinstance(call, dict):
            name = call.get("tool") or call.get("name") or call.get("action") or ""
            if name:
                names.append(str(name))
    return names


def _extract_text(execution_result: Any) -> str:
    """从 execution_result 抽文本."""
    if execution_result is None:
        return ""
    if isinstance(execution_result, str):
        return execution_result
    if not isinstance(execution_result, dict):
        return str(execution_result)
    parts: list[str] = []
    for key in ("summary", "description", "result_data", "output", "reasoning", "plan", "hypothesis"):
        v = execution_result.get(key)
        if v:
            parts.append(str(v))
    return " ".join(parts)


def compute_ec(
    execution_result: Any,
    validation_results: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """计算涌现复杂度, 返回各维度分数 + 总分.

    EC 越高表示 agent 行为越"涌现" — 工具多样、推理非模板化、跨域连接、有新东西.
    范围 [0, 1], 几何平均.
    """
    if not isinstance(execution_result, dict):
        execution_result = {"_text": str(execution_result)} if execution_result else {}

    text = _extract_text(execution_result)
    text_lower = text.lower()

    # 1. tool diversity: 不同工具数的 Shannon 熵, 归一化到 [0, 1]
    # ponytail: 用 ln 熵 / ln(n) 归一化, n=10 时 ln(10)≈2.3 足够覆盖常见场景
    tool_names = _extract_tool_names(execution_result)
    if tool_names:
        raw_h = _shannon_entropy(tool_names)
        max_h = math.log(len(set(tool_names))) if len(set(tool_names)) > 1 else 1.0
        tool_diversity = raw_h / max_h if max_h > 0 else 0.0
    else:
        tool_diversity = 0.0

    # 2. reasoning entropy: 词频分布熵, 高熵 = 非模板化
    # ponytail: 只取 >3 字符的词, 过滤停用词太重, 用 unique_ratio 近似
    words = [w for w in text_lower.split() if len(w) > 3]
    if words:
        raw_h = _shannon_entropy(words)
        # ln(n) 是最大熵, 归一化; n 太小时上限低, 所以 clamp
        max_h = math.log(len(words))
        reasoning_entropy = min(1.0, raw_h / max_h) if max_h > 0 else 0.0
    else:
        reasoning_entropy = 0.0

    # 3. cross_domain: 命中了几个不同的子领域
    domains_hit = 0
    for keyword_set in _DOMAIN_KEYWORDS:
        if any(kw in text_lower for kw in keyword_set):
            domains_hit += 1
    # 5 个类, 命中 2+ 才算跨域
    cross_domain = min(1.0, domains_hit / 3.0) if domains_hit >= 2 else 0.0

    # 4. novelty: 非常见词的比例 (出现 <= 1 次的词 / 总词数)
    # 高 novelty = 不是在重复已知结论
    if words:
        word_counts = Counter(words)
        novel = sum(1 for c in word_counts.values() if c <= 1)
        novelty = min(1.0, novel / len(words))
    else:
        novelty = 0.0

    # 几何平均: 任一维度为 0 就大幅拉低总分
    dims = [tool_diversity, reasoning_entropy, cross_domain, novelty]
    nonzero = [d for d in dims if d > 0]
    if nonzero:
        ec = math.exp(sum(math.log(d) for d in nonzero) / len(nonzero))
    else:
        ec = 0.0

    return {
        "ec_score": round(ec, 4),
        "ec_tool_diversity": round(tool_diversity, 4),
        "ec_reasoning_entropy": round(reasoning_entropy, 4),
        "ec_cross_domain": round(cross_domain, 4),
        "ec_novelty": round(novelty, 4),
        "ec_domains_hit": domains_hit,
    }


__all__ = ["compute_ec"]
