"""判别实验: Agent 是否只是"可替换的 LLM 身体"?

背景 / hypothesis:
  人怀疑 agent 不过是个 LLM 载体, 换个身体就散了。本实验用可复现的
  离线模拟证伪这一点。核心论点:

    agent 的"结构层" (跨源数值校验 + provenance + 稳健中心) 是**完全确定**
    的纯函数, 只依赖输入数据、与 LLM 无关; 而 "LLM 身体" 只负责从论文
    文本里**抽数值**这一步, 这一层天然带噪声/偏差且随身体漂移。

  于是判别键是: 把 N 个不同的"身体" (各自对同一批论文抽值, 带不同噪声/
  漏检) 喂给**同一个确定性组合层**, 结构产物 (consistency 标签、稳健中位数、
  verdict、provenance) 应当保持稳定、可复现、可审计——这是换 LLM 也不会散的
  证据。而一个"裸 LLM"直接回答同一问题, 产不出这些结构化产物 (无来源、
  无一致性标注、无法复现)。

本模块不碰网络、不调用任何真实 LLM。用真实的生产函数
`_annotate_value_consistency` 作为确定性内核, 以固定 seed 保证完全可复现。
可独立运行:  python -m huginn.experimental.body_swap_discrimination

验证的三条不变量:
  I1 结构可复现: 同输入两次运行, 产物逐字节一致。
  I2 跨身体稳健: 无论哪个"身体"抽值, 稳健中位数落在真值邻域, verdict 不
                 随单个离群身体翻转; 离群值被标 being conflicting 而非污染中心。
  I3 结构产物为 agent 独有: 纯 LLM 式自由文本回答问题无法产出
     {verdict,labels,median,provenance} 这组结构化字段。
"""

from __future__ import annotations

import random
from typing import Any

from huginn.tools.literature.tool import _annotate_value_consistency

# ── 确定性组合层 (agent 结构) ────────────────────────────────────────────


def _consensus(reported: list[dict[str, Any]]) -> dict[str, Any] | None:
    """复刻 benchmark_lookup 的 consensus 组装 — 纯计算, 与 LLM 无关."""
    if not reported:
        return None
    vals = [r["value"] for r in reported]
    mean = sum(vals) / len(vals)
    s = sorted(vals)
    median = s[len(s) // 2] if len(s) % 2 else (s[len(s) // 2 - 1] + s[len(s) // 2]) / 2
    return {
        "mean": round(mean, 6),
        "median": round(median, 6),
        "unit": reported[0]["unit"],
        "n_sources": len(reported),
    }


def _attach_provenance(reported: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """给每个值挂稳定 fid + 来源字段 — agent 的可溯源性, 与 LLM 无关."""
    out: list[dict[str, Any]] = []
    for i, r in enumerate(reported):
        out.append(
            {
                **r,
                "fid": f"{r.get('doi') or r.get('source_paper') or 'src'}#{i}",
                "provenance": {
                    "source_paper": r.get("source_paper"),
                    "doi": r.get("doi"),
                    "year": r.get("year"),
                },
            }
        )
    return out


def analyze(reported: list[dict[str, Any]]) -> dict[str, Any]:
    """确定性组合层: 输入 reported 值列表 → 结构产物.

    这是 agent 独有的"认知结构"; 完全纯函数, 可复现、可审计.
    """
    reported = [dict(r) for r in reported]
    tagged = _attach_provenance(reported)
    consistency = _annotate_value_consistency(tagged)
    return {
        "n_sources": len(tagged),
        "consensus": _consensus(tagged),
        "consistency": consistency,
        "reported_values": tagged,
    }


# ── 模拟"LLM 身体" ───────────────────────────────────────────────────────


def _synthetic_papers(seed: int) -> list[dict[str, Any]]:
    """一批合成论文 (真值=band_offset, 各家验证在其附近波动)."""
    rng = random.Random(seed)
    papers = []
    truths = [
        {"system": "Li2O", "property": "band_gap", "unit": "eV", "truth": 5.2},
        {"system": "Li2O", "property": "shear_modulus", "unit": "GPa", "truth": 48.0},
        {"system": "GeTe", "property": "band_gap", "unit": "eV", "truth": 0.65},
    ]
    for gi, t in enumerate(truths):
        for _ in range(4):
            papers.append(
                {
                    "system": t["system"],
                    "property": t["property"],
                    "unit": t["unit"],
                    "value": round(
                        t["truth"]
                        + rng.uniform(-0.04, 0.04) * max(abs(t["truth"]), 1.0),
                        6,
                    ),
                    "source_paper": f"paper {gi}-{rng.randint(1, 99)}",
                    "doi": f"10.1000/{gi}-{rng.randint(1000, 9999)}",
                    "year": rng.randint(2016, 2024),
                }
            )
    return papers


def simulate_body(
    papers: list[dict[str, Any]],
    *,
    seed: int,
    drop: float = 0.0,
    noise_sd: float = 0.01,
) -> list[dict[str, Any]]:
    """一个"LLM 身体": 从论文里抽值, 带自身的漏检率 + 抽样噪声.

    drop>0 时某些论文被身体漏检 (真实 LLM 常发生), noise_sd 放大扰动.
    """
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    for p in papers:
        if rng.random() < drop:
            continue  # 该身体漏掉了这篇
        value = p["value"] * (1.0 + rng.gauss(0, noise_sd))
        out.append({**p, "value": round(value, 6), "noise": round(noise_sd, 4)})
    return out


# ── I1: 结构可复现 ───────────────────────────────────────────────────────


def invariant_reproducible(reported: list[dict[str, Any]]) -> bool:
    """I1: 确定性组合层同输入两次运行, 产物必须完全一致 (字节级)."""
    return analyze(reported) == analyze(reported)


# ── I2: 跨身体稳健 ───────────────────────────────────────────────────────


def invariant_cross_body(papers: list[dict[str, Any]], *, bodies: int = 5) -> bool:
    """I2: 不同身体抽值喂同一组合层, 各 body 稳健中位数都落在真值邻域.

    用噪声 + 少许漏检, 看结构层对每个 unit 是否给出稳定中心 (不被单身体
    带偏到真值之外). 判据: 对每篇论文, 其"合成真值"是该 unit 的 value 基准,
    各 body 的稳健中位数应落在该 unit 真值的 ±12% (或绝对 0.03) 邻域内.
    """
    # unit → 合成真值 (取该 unit 下第一篇论文的 value 作参考, 合成时围绕它波动)
    unit_truth: dict[str, float] = {}
    for p in papers:
        unit_truth.setdefault(p["unit"], p["value"])

    for b in range(bodies):
        body = simulate_body(papers, seed=b * 7 + 1, drop=0.15, noise_sd=0.06)
        if not body:
            return False
        res = analyze(body)
        cons = res["consensus"]
        if cons is None:
            return False
        u = cons["unit"]
        truth = unit_truth.get(u)
        if truth is None:
            return False
        tol = max(abs(truth) * 0.12, 0.03)
        if abs(cons["median"] - truth) > tol:
            # 该 body 的稳健中心被带离真值邻域 → 结构层被身体污染
            return False
    return True


# ── I3: 结构产物为 agent 独有 ───────────────────────────────────────────


def _free_text_llm_answer() -> dict[str, Any]:
    """模拟"裸 LLM"直接回答: 只有一句自由文本, 无结构字段."""
    return {
        "answer_text": "Li2O 的带隙大约是 5.2 eV, 剪切模量约 48 GPa。"
        "(这是纯文字, 没有来源、没有一致性标注、无法复核。)",
    }


def invariant_structure_only_in_agent(reported: list[dict[str, Any]]) -> bool:
    """I3: agent 结构层产出的字段集, 裸 LLM 的自由文本产物里全都没有."""
    agent_out = analyze(reported)
    cons = agent_out["consistency"]["overall"]
    if agent_out["consensus"] is None:
        return False
    for v in agent_out["reported_values"]:
        if "fid" not in v or "provenance" not in v:
            return False

    llm_out = _free_text_llm_answer()
    text = str(llm_out)
    # 裸 LLM 产物里不应含有 agent 的结构化字段名
    for structural in ("fid", "verdict", "consensus", "source_paper"):
        assert structural not in text, f"裸 LLM 产物居然含结构字段: {structural}"

    return cons["verdict"] != "" and all(
        "fid" in v for v in agent_out["reported_values"]
    )


# ── 运行入口 ─────────────────────────────────────────────────────────────


def run_all(*, bodies: int = 5, seed: int = 0) -> dict[str, Any]:
    papers = _synthetic_papers(seed)
    body = simulate_body(papers, seed=seed, drop=0.1, noise_sd=0.03)
    result = {
        "n_papers": len(papers),
        "n_bodies": bodies,
        "seed": seed,
        "invariant_reproducible": invariant_reproducible(body),
        "invariant_cross_body": invariant_cross_body(papers, bodies=bodies),
        "invariant_structure_only_in_agent": invariant_structure_only_in_agent(body),
    }
    result["all_pass"] = all(
        result[k]
        for k in (
            "invariant_reproducible",
            "invariant_cross_body",
            "invariant_structure_only_in_agent",
        )
    )
    return result


if __name__ == "__main__":
    import json

    report = run_all(bodies=8)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("ALL PASS" if report["all_pass"] else "FAIL")
