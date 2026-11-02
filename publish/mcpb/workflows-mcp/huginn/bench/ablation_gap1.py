"""Gap 1 测试: 结构对象直传 vs 文本中转 (不经 hint_coordinator 渲染).

v3 LLM ablation 发现: 几何 hint (自然语言+结构信息) F1=0.950 < 文本 hint F1=0.974.
但上一轮的"几何 hint"还是把结构信息拼进了自然语言段落 (hint_coordinator 风格).

Gap 1 真正问题: 如果 LLM 收到的是纯结构 JSON (无自然语言包装), 会怎样?
这是"结构对象直传"的极端测试 — 模拟 manifold.Hypothesis 对象直接序列化给 LLM,
不经过 hint_coordinator 的文本渲染.

三组对比:
  A 组 (纯文本): 只看 description (baseline, 跟 v3 一致)
  B 组 (自然语言+结构): description + 结构信息拼成段落 (hint_coordinator 风格, 跟 v3 一致)
  C 组 (纯结构 JSON): Hypothesis 对象直接 JSON 序列化, 无自然语言包装

如果 C 组 >> B 组 → 结构直传有优势 (LLM 更会读 JSON 结构)
如果 C 组 ≈ B 组 ≤ A 组 → 结构信息本身没用, 怎么传都一样
如果 C 组 << A 组 → LLM 不会读纯结构, 反而需要自然语言包装

这是 Gap 1 的真正裁判: "结构对象不经过文本渲染" 是否有净增益.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from ablation_geom_vs_text import TestCase, build_test_cases
from ablation_llm_loop import _build_geom_hint, _build_text_hint, _make_model

from huginn.metacog.hypothesis_manifold import HypothesisManifold


def _build_pure_struct_hint(c: TestCase) -> str:
    """C 组: 纯结构 JSON, 无自然语言包装.

    模拟 Hypothesis 对象直接序列化, 不经过 hint_coordinator 的文本渲染.
    """
    m = HypothesisManifold()
    m.add(c.a)
    m.add(c.b)
    fisher = m.fisher_distance(c.a.h_id, c.b.h_id)
    struct = {
        "hypothesis_a": {
            "id": c.a.h_id,
            "description": c.a.description,
            "predictions": c.a.predictions,
            "n_params": c.a.n_params,
        },
        "hypothesis_b": {
            "id": c.b.h_id,
            "description": c.b.description,
            "predictions": c.b.predictions,
            "n_params": c.b.n_params,
        },
        "geometry": {
            "fisher_distance": round(fisher, 6),
            "complexity_diff": abs(c.a.n_params - c.b.n_params),
        },
    }
    return "```json\n" + json.dumps(struct, indent=2, default=str) + "\n```"


_PROMPT_TEMPLATE = """{hint}

判断这两个假设是否"本质不同" (非换名归约, 非同义改写).

本质不同 = 不同物理理论/机制/结构, 即使能 fit 同样数据
本质相同 = 同一理论的不同表述, 或同义改写

只回答 JSON, 不要解释:
{{"different": true/false, "reason": "一句话理由"}}"""


def _ask(model, hint: str) -> bool | None:
    from langchain_core.messages import HumanMessage
    prompt = _PROMPT_TEMPLATE.format(hint=hint)
    try:
        resp = model.invoke([HumanMessage(content=prompt)])
        text = str(resp.content).strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        data = json.loads(text)
        return bool(data.get("different", False))
    except Exception as e:
        print(f"  LLM 解析失败: {e}")
        return None


def _gap1_banner(n: int) -> None:
    """打印实验头: 分组说明 + 样本数."""
    print("=" * 80)
    print(f"Gap 1 测试: 结构对象直传 vs 文本中转 (deepseek, n={n})")
    print("=" * 80)
    print("A 组: 纯文本 (只 description)")
    print("B 组: 自然语言+结构 (hint_coordinator 风格)")
    print("C 组: 纯结构 JSON (无自然语言包装, 模拟对象直传)")
    print()


def _gap1_probe_model() -> Any | None:
    """连一次 deepseek 做存活探测; 失败返 None (整轮跳过)."""
    try:
        model = _make_model()
        from langchain_core.messages import HumanMessage
        model.invoke([HumanMessage(content="hi")])
        print("deepseek 连接 OK\n")
        return model
    except Exception as e:
        print(f"deepseek 连接失败: {e}")
        return None


def _gap1_ask_group(model, c: TestCase, hint_builder, stats: dict, grp: str) -> bool:
    """跑单个分组; 解析失败计入 fail 并按 False 参与混淆矩阵."""
    pred = _ask(model, hint_builder(c))
    if pred is None:
        stats[grp]["fail"] += 1
        pred = False
    time.sleep(0.3)
    return pred


def _gap1_score(stats: dict, grp: str, pred: bool, actual: bool) -> None:
    """累计单样本的 correct / tp / fp / fn / tn."""
    s = stats[grp]
    if pred == actual:
        s["correct"] += 1
    if pred and actual:
        s["tp"] += 1
    elif pred and not actual:
        s["fp"] += 1
    elif not pred and actual:
        s["fn"] += 1
    else:
        s["tn"] += 1


def _gap1_run_samples(model, sample: list, stats: dict) -> int:
    """逐样本跑 A/B/C 三组, 返回 B vs C 的判断不一致数."""
    disagreements_bc = 0
    for i, c in enumerate(sample):
        actual = c.human_label
        pred_a = _gap1_ask_group(model, c, _build_text_hint, stats, "A")
        pred_b = _gap1_ask_group(model, c, _build_geom_hint, stats, "B")
        pred_c = _gap1_ask_group(model, c, _build_pure_struct_hint, stats, "C")

        for grp, pred in [("A", pred_a), ("B", pred_b), ("C", pred_c)]:
            _gap1_score(stats, grp, pred, actual)

        if pred_b != pred_c:
            disagreements_bc += 1

        if (i + 1) % 10 == 0:
            n = i + 1
            print(f"  进度: {n}/{len(sample)}  "
                  f"A={stats['A']['correct']/n:.3f}  "
                  f"B={stats['B']['correct']/n:.3f}  "
                  f"C={stats['C']['correct']/n:.3f}")
    return disagreements_bc


def _wilson(p: float, nn: int, z: float = 1.96) -> tuple[float, float]:
    """二项比例的 Wilson 95% 置信区间."""
    from math import sqrt

    if nn == 0:
        return (0, 0)
    denom = 1 + z * z / nn
    center = (p + z * z / (2 * nn)) / denom
    spread = z * sqrt(p * (1 - p) / nn + z * z / (4 * nn * nn)) / denom
    return (max(0, center - spread), min(1, center + spread))


def _gap1_report(stats: dict, n: int) -> None:
    """算 acc/p/r/f1 并打印对比表 + 各组 F1 的 Wilson CI."""
    print()
    print("=" * 80)
    print("结果")
    print("=" * 80)
    print(f"{'指标':<20} {'A(纯文本)':>14} {'B(自然+结构)':>14} {'C(纯结构JSON)':>16}")
    print("-" * 66)
    for grp, _label in [("A", "纯文本"), ("B", "自然+结构"), ("C", "纯结构JSON")]:
        s = stats[grp]
        s["acc"] = s["correct"] / n
        s["p"] = s["tp"] / (s["tp"] + s["fp"]) if (s["tp"] + s["fp"]) else 0
        s["r"] = s["tp"] / (s["tp"] + s["fn"]) if (s["tp"] + s["fn"]) else 0
        s["f1"] = (
            2 * s["p"] * s["r"] / (s["p"] + s["r"]) if (s["p"] + s["r"]) else 0
        )

    for metric, key in [
        ("准确率", "acc"), ("F1", "f1"), ("Precision", "p"), ("Recall", "r"),
    ]:
        print(f"{metric:<20} {stats['A'][key]:>14.3f} {stats['B'][key]:>14.3f} "
              f"{stats['C'][key]:>16.3f}")
    for metric, key in [
        ("TP", "tp"), ("FP", "fp"), ("FN", "fn"), ("TN", "tn"), ("解析失败", "fail"),
    ]:
        print(f"{metric:<20} {stats['A'][key]:>14} {stats['B'][key]:>14} "
              f"{stats['C'][key]:>16}")
    print()

    for grp in ["A", "B", "C"]:
        lo, hi = _wilson(stats[grp]["f1"], n)
        print(f"  {grp} 组 F1 Wilson 95% CI: [{lo:.3f}, {hi:.3f}]")
    print()


def _gap1_conclude(stats: dict, n: int, disagreements_bc: int) -> None:
    """打印 ΔF1 + CI 重叠判断 + 最终结论."""
    print("=" * 80)
    print("Gap 1 结论")
    print("=" * 80)
    delta_bc = stats["C"]["f1"] - stats["B"]["f1"]
    delta_ac = stats["C"]["f1"] - stats["A"]["f1"]
    print(f"  C(纯结构JSON) - B(自然+结构) ΔF1 = {delta_bc:+.3f}  ← Gap 1 核心指标")
    print(f"  C(纯结构JSON) - A(纯文本)    ΔF1 = {delta_ac:+.3f}")
    print()

    # B vs C CI 重叠判断
    b_lo, b_hi = _wilson(stats["B"]["f1"], n)
    c_lo, c_hi = _wilson(stats["C"]["f1"], n)
    bc_overlap = not (c_lo > b_hi or b_lo > c_hi)

    if not bc_overlap and delta_bc > 0.05:
        print("  → 结构直传 (C) 显著优于文本中转 (B), CI 不重叠")
        print("  → Gap 1 有净增益: 结构对象不经文本渲染直接给 LLM 更好")
    elif not bc_overlap and delta_bc < -0.05:
        print("  → 结构直传 (C) 显著差于文本中转 (B), CI 不重叠")
        print("  → LLM 需要自然语言包装, 纯结构 JSON 反而干扰")
    else:
        print(f"  → B vs C CI 重叠 (ΔF1={delta_bc:+.3f}), 结构传递方式无显著影响")
        print("  → Gap 1 (结构直传 vs 文本中转) 无净增益, hint_coordinator 文本渲染不是瓶颈")
        if disagreements_bc > 0:
            print(f"  → 但 {disagreements_bc} 个样本传递方式改变了判断, 看具体样本")
    print()
    if delta_ac < -0.02 and delta_bc < 0.02:
        print("  → 综合判断: 结构信息对 LLM 判断无净增益 (C≈B≤A)")
        print("  → hint_coordinator 的几何 hint 是负优化, 应简化回纯文本 hint")
        print("  → 这是几何通信方向的终止信号")


def run_gap1_ablation(n_samples: int = 60, seed: int = 42):
    """Gap 1: 三组对比 (纯文本 / 自然语言+结构 / 纯结构 JSON)."""
    cases = build_test_cases()
    rnd = random.Random(seed)
    sample = rnd.sample(cases, min(n_samples, len(cases)))
    _gap1_banner(len(sample))

    model = _gap1_probe_model()
    if model is None:
        return None

    # 三组统计
    stats = {k: {"correct": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0, "fail": 0}
             for k in ["A", "B", "C"]}

    disagreements_bc = _gap1_run_samples(model, sample, stats)

    n = len(sample)
    _gap1_report(stats, n)
    print(f"B vs C 不一致 (结构传递方式的影响): "
          f"{disagreements_bc}/{n} ({disagreements_bc/n:.1%})")
    print()
    _gap1_conclude(stats, n, disagreements_bc)
    return stats


if __name__ == "__main__":
    n = 60
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    run_gap1_ablation(n_samples=n)
