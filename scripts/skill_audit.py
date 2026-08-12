"""huginn 技能库盘点 + 图论/拓扑诊断 (可复用 / CI).

把技能库建模成 技能-工具 二分图, 再用仓库已有的高级图论/拓扑能力
(trace_topology 的 β_0 弱连通分量 + 语义重叠, simplicial_homology 的
真正 betti) 分析技能库的模块结构、冗余、孤立、hub 工具与技能树。

只读诊断, 不改任何代码。

用法:
  python scripts/skill_audit.py                    基础盘点
  python scripts/skill_audit.py --tree             额外打印技能树
  python scripts/skill_audit.py --fail-on-hole     若 β_1>0 则以退出码 1 结束 (CI 门禁)
  python scripts/skill_audit.py --json             输出机器可读 JSON 摘要
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

# 让脚本可随处运行 (本地 / CI): agent 目录 = 脚本所在仓库根下的 agent/.
# 不再硬编码 /workspace, 否则 CI runner 上必挂.
_AGENT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "agent")
)
if _AGENT_DIR not in sys.path:
    sys.path.insert(0, _AGENT_DIR)

import networkx as nx  # noqa: E402  (agent 目录需先行注入 sys.path)

from huginn.metacog.simplicial_homology import compute_exact_betti  # noqa: E402
from huginn.metacog.trace_topology import _sem_overlap  # noqa: E402
from huginn.skills.registry import SkillRegistry  # noqa: E402


def build_tool_graph(skills):
    """返回 (tool_of, tool2skill, all_tools)."""
    tool_of = {}
    tool2skill = defaultdict(set)
    for s in skills:
        tools = set()
        for t in (list(getattr(s, "required_tools", None) or [])
                  + [st.tool for st in (getattr(s, "steps", None) or [])
                     if getattr(st, "tool", None)]):
            if t:
                tools.add(t)
        tool_of[s.name] = tools
        for t in tools:
            tool2skill[t].add(s.name)
    return tool_of, tool2skill, set(tool2skill)


def load_skills():
    import huginn.skills  # noqa: F401  触发 presets/composite 注册
    seen = {}
    for s in SkillRegistry.get_all_definitions():
        seen[s.name] = s
    return list(seen.values())


def print_tree() -> None:
    """打印整棵技能树 (parent → children, 含顶层)."""
    print("\n[技能树 (parent → children)]")
    tree = SkillRegistry.tree()
    top = tree.get("", [])
    print(f"  顶层技能 {len(top)} 项:")
    for n in top:
        kids = SkillRegistry.children(n)
        suffix = f"  → {len(kids)} children" if kids else ""
        print(f"    - {n}{suffix}")
    for parent, kids in tree.items():
        if parent == "":
            continue
        print(f"  {parent} ({len(kids)}):")
        for k in kids:
            print(f"    - {k}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="huginn 技能库盘点诊断")
    ap.add_argument("--tree", action="store_true", help="打印技能树")
    ap.add_argument("--fail-on-hole", action="store_true",
                    help="β_1>0 时以退出码 1 结束 (CI 门禁)")
    ap.add_argument("--json", action="store_true",
                    help="输出机器可读 JSON 摘要到 stdout 末尾")
    args = ap.parse_args(argv)

    skills = load_skills()
    print("=" * 70)
    print(f"技能总数: {len(skills)}")

    cat = Counter(s.category for s in skills)
    print("\n[category 分布]")
    for c, n in cat.most_common():
        print(f"  {c:24s} {n}")

    tags = Counter(t for s in skills for t in (getattr(s, "tags", None) or []))
    print("\n[tag 分布 top20]")
    for t, n in tags.most_common(20):
        print(f"  {t:28s} {n}")

    tool_of, tool2skill, all_tools = build_tool_graph(skills)
    print(f"\n涉及工具数: {len(all_tools)}")

    print("\n[hub 工具 top15 → 复用/共享热点]")
    for t, ss in sorted(tool2skill.items(), key=lambda kv: -len(kv[1]))[:15]:
        lst = sorted(ss)
        print(f"  {t:34s} used_by {len(ss)}: {lst[:6]}{'...' if len(lst) > 6 else ''}")

    print("\n[技能复杂度 top15 (依赖工具数最多)]")
    for n, tools in sorted(tool_of.items(), key=lambda kv: -len(kv[1]))[:15]:
        print(f"  {n:34s} tools={len(tools)}")

    G = nx.Graph()
    for s in skills:
        G.add_node(s.name)
    pairs = defaultdict(set)
    names = [s.name for s in skills]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            inter = tool_of[names[i]] & tool_of[names[j]]
            if inter:
                pairs[(names[i], names[j])] = inter
    for (a, b), inter in pairs.items():
        G.add_edge(a, b, weight=len(inter), shared=sorted(inter))

    comps = list(nx.connected_components(G))
    print("\n[共享工具图拓扑]")
    print(f"  弱连通分量数 β_0 = {len(comps)}  (技能被共享工具聚成 {len(comps)} 簇)")
    isolated = [n for n, d in G.degree() if d == 0]
    print(f"  孤立技能 (不与其他技能共享工具): {len(isolated)}")
    for n in sorted(isolated):
        print(f"    - {n}  (tools={sorted(tool_of[n])})")
    if comps:
        sizes = sorted((len(c) for c in comps), reverse=True)
        print(f"  最大簇规模: {sizes[:8]}")

    print("\n[高共享技能对 → 冗余/合并候选 (共享≥2 工具)]")
    hi = sorted(pairs.items(), key=lambda kv: -len(kv[1]))[:20]
    for (a, b), inter in hi:
        if len(inter) >= 2:
            print(f"  {a}  <->  {b}  shared={len(inter)} {sorted(inter)}")

    print("\n[语义重叠技能对 → 重复/冲突候选 (overlap>0.5)]")
    sem = []
    for i in range(len(skills)):
        for j in range(i + 1, len(skills)):
            a, b = skills[i], skills[j]
            ov = _sem_overlap(a.description, b.description)
            if ov > 0.5:
                sem.append((ov, a.name, b.name))
    sem.sort(reverse=True)
    for ov, a, b in sem[:25]:
        print(f"  [{ov:.2f}] {a}  <->  {b}")

    print("\n[高阶拓扑 (simplicial homology on 技能→工具) — 已装依赖时计算]")
    betti: dict[int, int] = {}
    try:
        tool_idx = {t: i for i, t in enumerate(all_tools)}
        simplices = []
        for s in skills:
            vs = sorted(tool_idx[t] for t in tool_of[s.name])
            if vs:
                simplices.append(tuple(vs))
        betti = compute_exact_betti(simplices, max_dim=2)
        print(f"  betti (技能→工具 单纯复形): {betti}")
        if betti.get(1, 0) > 0:
            print("  → β_1>0: 工具组合网络中有'洞', 存在未被任何技能覆盖的工具组合环")
    except Exception as e:  # pragma: no cover
        print(f"  betti 计算失败: {e}")

    if args.tree:
        print_tree()

    print("\n" + "=" * 70)

    if args.json:
        summary = {
            "skills": len(skills),
            "tools": len(all_tools),
            "categories": dict(cat),
            "betti": {str(k): v for k, v in betti.items()},
            "isolated_skills": sorted(isolated),
            "top_hub_tools": sorted(
                ((t, len(ss)) for t, ss in tool2skill.items()),
                key=lambda kv: -kv[1],
            )[:15],
        }
        print("\n[JSON]\n" + json.dumps(summary, ensure_ascii=False))

    if args.fail_on_hole and betti.get(1, 0) > 0:
        print("\n[CI] 检测到 β_1>0 工具组合洞 → 退出码 1")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
