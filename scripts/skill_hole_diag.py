"""用 networkx 复现 skill_audit.py 的 β_1 计算, 并解释环.

skill_audit.py 走 compute_exact_betti 的 fallback (无 gudhi): 它只取
len(simplex)==2 的 simplex 作为边, 丢弃 3+ 维 simplex 的边界. 这意味着
每个技能的工具集若 ≥3 个, fallback 不会把它的完整 2-simplex 面加进去,
可能漏掉真实填充或产生伪环. 本脚本分别用两种构造对比, 解释 β_1=1.
"""
from __future__ import annotations

import os
import sys
from itertools import combinations

_AGENT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "agent")
)
if _AGENT_DIR not in sys.path:
    sys.path.insert(0, _AGENT_DIR)

import networkx as nx  # noqa: E402

from huginn.metacog.simplicial_homology import compute_exact_betti  # noqa: E402
from huginn.skills.registry import SkillRegistry  # noqa: E402


def load():
    import huginn.skills  # noqa: F401
    seen = {}
    for s in SkillRegistry.get_all_definitions():
        seen[s.name] = s
    skills = list(seen.values())
    tool_of = {}
    for s in skills:
        tools = set()
        for t in (list(getattr(s, "required_tools", None) or [])
                  + [st.tool for st in (getattr(s, "steps", None) or [])
                     if getattr(st, "tool", None)]):
            if t:
                tools.add(t)
        tool_of[s.name] = tools
    all_tools = sorted({t for ts in tool_of.values() for t in ts})
    tool_idx = {t: i for i, t in enumerate(all_tools)}
    return tool_idx, {i: t for t, i in tool_idx.items()}, tool_of


def main() -> None:
    tool_idx, int_to_tool, tool_of = load()

    # 构造 A: 完整复形 (每个 simplex 含边界) — 正确语义
    full = []
    for ts in tool_of.values():
        vs = sorted(tool_idx[t] for t in ts)
        if vs:
            full.append(tuple(vs))
    # 需展开边界: GUDHI 自动做, fallback 不做. 手动补全部面
    full_simplices = set()
    for simplex in full:
        vs = list(simplex)
        for r in range(2, len(vs) + 1):
            for c in combinations(vs, r):
                full_simplices.add(tuple(c))

    # 构造 B: skill_audit fallback 实际传的 (把每技能工具集当 simplex, 不展开)
    as_is = [tuple(sorted(tool_idx[t] for t in ts)) for ts in tool_of.values() if ts]

    bA = compute_exact_betti(list(full_simplices), max_dim=2) if full_simplices else {}
    bB = compute_exact_betti(as_is, max_dim=2) if as_is else {}
    print(f"构造A(完整复形+边界展开) β: {bA}")
    print(f"构造B(原样 simplex, 即 skill_audit 传入) β: {bB}")

    # networkx 图论环: 用技能为节点的共享工具图
    print("\n[技能为节点 / 共享工具为边的图: 环路]")
    G = nx.Graph()
    names = list(tool_of.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            inter = tool_of[names[i]] & tool_of[names[j]]
            if inter:
                G.add_edge(names[i], names[j], shared=sorted(inter))
    cycles = []
    for a, b in G.edges():
        try:
            p = nx.shortest_path(G, a, b)
        except Exception:
            continue
        if len(p) >= 4:
            cycles.append((len(p), p))
    cycles.sort()
    seen = set()
    for ln, p in cycles[:12]:
        key = tuple(sorted(p))
        if key in seen or ln != len(p):
            continue
        seen.add(key)
        print(f"  环路(len {ln}): {' -> '.join(p)}")
        # 显示每条边共享的工具
        for x, y in zip(p, p[1:]):
            sh = sorted(tool_of[x] & tool_of[y])
            print(f"      {x} --{sh}-- {y}")
    print(f"  不同环路数: {len(seen)}")


if __name__ == "__main__":
    main()