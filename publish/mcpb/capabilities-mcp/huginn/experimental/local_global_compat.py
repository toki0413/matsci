"""局部-整体兼容实验 — 用朗兰兹"局部-整体 + 保持不变量"校准 agent 结构层.

背景 / hypothesis:
  朗兰兹纲领对 agent 的启示不是复现 Galois/自守计算, 而是两个可测原则:
    (1) 局部-整体 (Local-Global): 整体语义 = 所有"局部自由度条件"兼容的系统,
        缺哪个庶处 (自由度) 没定义, 整体就不能假装完整.
    (2) 保持不变量 (Functoriality): 组合/校验映射须保持某"意义中心", 不随身体漂移.

  本实验把这两条钉进已建的结构层, 用可复现的合成数据验证 agent 是否
  "自发实现了局部-整体兼容", 还是只是在打一致性标签:

    - 局部化: group_reported 按自由度 (方法族/泛函/温度/单位) 分组 —
              = 朗兰兹里的 local field Q_p.
    - 局部成立: 每组内跑 _annotate_value_consistency → 组内是否收敛 (局部解存在).
    - 缺失自由度: 汇总各组 missing_dims → = "哪些素数没定义" (局部化完整性诊断).
    - 整体兼容: 各条件组中位数互洽性检查 → 组间偏差按 consistent/moderate/
              conflicting 判据分类; **不假装一致** (restricted product 精神).
    - 判别对比: 若把所有值塞进一组 (不按自由度), 一个 plausilly "conflicting"
              的数据会怎样: 未分组误标 conflicting vs 分组后拆成组内一致、组间
              归因为方法自由度 (LDA/GGA 系统性低估带隙). 这是"保持结构"的直接证据.

本模块纯结构层: 零网络/零 LLM, 用真实生产函数 _annotate_value_consistency +
condition_normalize.group_reported. 固定 seed 完全可复现.
可独立运行:  python -m huginn.experimental.local_global_compat
"""

from __future__ import annotations

import random
from typing import Any

from huginn.tools.literature.condition_normalize import group_reported
from huginn.tools.literature.tool import _annotate_value_consistency

# ── 组间互洽判据 (整体兼容性 — 本实验新增) ────────────────────────────────

# 与 _annotate_value_consistency 同源的偏差档
_CONSISTENT_REL = 0.05
_MODERATE_REL = 0.20


def _median(vals: list[float]) -> float:
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return 0.0
    if n % 2:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


def _rel_dev(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1.0)


def assess_group_compatibility(
    group_stats: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """组间互洽检查 — 整体语义是否兼容, 且不把"缺自由度"伪装成一致.

    对每个条件组取稳健中位数, 两两同 unit 比较:
      rel_dev ≤ consistent_rel / moderate_rel → 兼容或大致兼容
      > moderate_rel                            → 组间冲突 (归因为自由度未归一)

    返回组间 pairwise 结果 + 总体判定. 只比对同 unit 组; 不同 unit 不混判.
    """
    units: dict[str, list[tuple[str, float]]] = {}
    for key, st in group_stats.items():
        unit = st.get("unit")
        med = st.get("median")
        if unit is not None and med is not None:
            units.setdefault(unit, []).append((key, med))

    pairs: list[dict[str, Any]] = []
    stat = {"consistent": 0, "moderate": 0, "conflicting": 0, "n_pairs": 0}
    for unit, items in units.items():
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                k_a, v_a = items[i]
                k_b, v_b = items[j]
                dev = _rel_dev(v_a, v_b)
                if dev <= _CONSISTENT_REL:
                    label = "consistent"
                elif dev <= _MODERATE_REL:
                    label = "moderate"
                else:
                    label = "conflicting"
                stat[label] += 1
                stat["n_pairs"] += 1
                pairs.append(
                    {
                        "unit": unit,
                        "group_a": k_a,
                        "group_b": k_b,
                        "median_a": round(v_a, 6),
                        "median_b": round(v_b, 6),
                        "rel_dev": round(dev, 4),
                        "label": label,
                    }
                )

    if stat["conflicting"] > 0:
        verdict = "conflicting_across_groups"
    elif stat["n_pairs"] and stat["consistent"] == stat["n_pairs"]:
        verdict = "all_compatible"
    elif stat["conflicting"] == 0 and stat["n_pairs"]:
        verdict = "partially_compatible"
    else:
        verdict = "no_shared_unit_for_compare"
    stat["verdict"] = verdict
    return {
        "pairs": sorted(pairs, key=lambda p: (p["unit"], p["group_a"], p["group_b"])),
        "overall": stat,
    }


# ── 主流程 ────────────────────────────────────────────────────────────────


def _local_stats(items: list[dict[str, Any]]) -> dict[str, Any]:
    """单条件组内的收敛统计 (局部解是否成立)."""
    tagged = list(items)
    consistency = _annotate_value_consistency(
        [
            {
                "value": r["value"],
                "unit": r.get("unit", ""),
                "method": r.get("method", ""),
                "note": r.get("note", ""),
                "source_paper": r.get("source_paper", ""),
                "doi": r.get("doi"),
            }
            for r in tagged
        ]
    )
    vals = [r["value"] for r in tagged]
    cond = items[0].get("_condition") or {}
    return {
        "group": cond.get("condition_key", ""),
        "n_sources": len(vals),
        "median": round(_median(vals), 6),
        "unit": (vals and (tagged[0].get("unit") or None)),
        "group_verdict": consistency["overall"].get("verdict"),
        "consistency": consistency,
        "missing_dims": cond.get("missing_dims", []),
    }


def run_compat_experiment(
    reported: list[dict[str, Any]],
) -> dict[str, Any]:
    """局部-整体兼容实验主入口. reported: 各条含 value/unit/method/note/source_paper/doi.

    返回:
      - local: {condition_key: 组内统计 (是否收敛 + missing_dims)}
      - global_compat: assess_group_compatibility 结果 (组间互洽)
      - flat: 把所有值不分组 (旧行为) 直接标注 → 判别"分组 vs 不分组"差异
      - verdict 汇总
    """
    # 局部化: 按自由度分组, 每条注入 _condition
    groups = group_reported(reported)  # {key: [enriched items]}

    local: dict[str, dict[str, Any]] = {}
    all_missing: list[str] = []
    for key, items in groups.items():
        st = _local_stats(items)
        local[key] = st
        all_missing.extend(st["missing_dims"])

    # 组间互洽
    global_compat = assess_group_compatibility(
        {
            k: {"unit": v.get("unit"), "median": v.get("median")}
            for k, v in local.items()
        }
    )

    # 判别对比: 不分组 (旧 _annotate_value_consistency 直接跑全部) → 是否误把
    # "自由度差异"当成"数值冲突"
    flat_consistency = _annotate_value_consistency(
        [{"value": r["value"], "unit": r.get("unit", "")} for r in reported]
    )

    # 缺失自由度去重汇总
    missing_unique = sorted({m for m in all_missing if m})

    verdict = global_compat["overall"]["verdict"]
    # 若全部组内本身收敛, 且缺自由度非空 → 局部成立但整体需补自由度
    return {
        "local": local,
        "n_condition_groups": len(local),
        "missing_dims": missing_unique,
        "global_compat": global_compat,
        "flat_confounding": flat_consistency,
        "overall_verdict": verdict,
    }


# ── 合成数据: 故意缺失/混杂自由度 ─────────────────────────────────────────


def _synthetic_mixed(seed: int = 7) -> list[dict[str, Any]]:
    """一批"应分组的"文献值: 同体系 Li2O band_gap, 但方法自由度不同.

    - PBE @ 0K      : ~5.20 eV (LDA/GGA 系统性低估带隙)
    - experiment @ 室温: ~5.81 eV
    - HSE06 @ 0K     : ~6.00 eV (杂化泛函更准)
    - 一条 shear 量故意不标 method/note → 缺失自由度
    故意让"未分组"时会误判成 conflicting, 而"分组后"能拆开归因.
    """
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    specs = [
        ("PBE", "", ["4.1", "4.05", "4.12"]),  # LDA/GGA 严重低估带隙
        ("experiment", "room temperature", ["5.81", "5.79", "5.82"]),
        ("HSE06", "", ["5.98", "6.02", "5.96"]),
    ]
    idx = 0
    for method, note, vals in specs:
        for vs in vals:
            idx += 1
            rows.append(
                {
                    "value": round(float(vs) + rng.uniform(-0.01, 0.01), 4),
                    "unit": "eV",
                    "method": method,
                    "note": note,
                    "source_paper": f"Paper-{idx}",
                    "doi": f"10.1000/p{idx:03d}",
                }
            )
    # shear modulus 缺失 method/note → 缺 method_family + temperature
    rows.append(
        {
            "value": 47.5,
            "unit": "GPa",
            "method": "",
            "note": "",
            "source_paper": "Paper-shear",
            "doi": "10.1000/s01",
        }
    )
    return rows


# ── __main__ 自检 ─────────────────────────────────────────────────────────


if __name__ == "__main__":  # pragma: no cover - 可独立运行
    data = _synthetic_mixed()
    res = run_compat_experiment(data)
    print("== 局部化 (按自由度分组) ==")
    for key, st in res["local"].items():
        print(
            f"  {key:26s} n={st['n_sources']} median={st['median']:>7}"
            f" unit={st['unit']} 组内verdict={st['consistency']['overall']['verdict']}"
            f" 缺={st['missing_dims'] or '无'}"
        )
    print("== 组间互洽 (整体兼容) ==")
    for p in res["global_compat"]["pairs"]:
        print(
            f"  {p['group_a']} vs {p['group_b']} ({p['unit']}): "
            f"rel_dev={p['rel_dev']} → {p['label']}"
        )
    print(f"  OVERALL: {res['global_compat']['overall']['verdict']}")
    print("== 缺自由度汇总 ==", res["missing_dims"] or "无")
    print("== 判别：若不分group_direct 全塞一起会怎样 ==")
    print("  verdict =", res["flat_confounding"]["overall"]["verdict"])
