"""判别实验 — 真实身体接入变体.

上一版 (body_swap_discrimination.py) 只在"结构层纯函数"上验证三条不变量,
没有走真实的 agent 生产代码路径. 这一版把**可配置的假 LLM 身体**注入
`_do_benchmark_lookup`, 让真实生产函数 `_annotate_value_consistency`、真实
`_get_model`→`_llm_invoke`→`ainvoke`→`_parse_json`→paper 映射这一整条链都在
跑, 唯一被替换的是"LLM 抽数值"这一步 (换成可复现的假身体).

这回答一个更尖锐的判别问题:
  同样是这个 agent 结构, 换一个"身体" (带不同抽样噪声/漏检的 LLM), 最后
  产出的结构化结论 (consensus、consistency、verdict) 是否仍然稳健、可复核?
  如果连真实的生产代码路径下都稳, "agent 是可替换身壳"就进一步被削弱.

本模块不碰网络; 假 model 的 ainvoke 完全由 seed 决定, 可复现.
可独立运行:  python -m huginn.experimental.real_body_run
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from huginn.core_types import ToolContext, ToolResult
from huginn.experimental.body_swap_discrimination import _synthetic_papers
from huginn.tools.literature.tool import (
    LiteratureInput,
    LiteratureTool,
)

# ── 可配置假 LLM 身体 ─────────────────────────────────────────────────────


class FakeBodyModel:
    """一个扮演"LLM 身体"的假 model.

    它的 ainvoke 返回一个从论文文本里"抽数值"的 JSON——带抽样噪声和漏检,
    完全由 seed 决定. 走的是真实代码路径 `_llm_invoke(model).ainvoke(...)`,
    但数值来源是可复现的假身体.
    """

    def __init__(
        self,
        papers_with_text: list[tuple[int, dict]],
        *,
        seed: int,
        noise_sd: float,
        drop: float,
    ) -> None:
        rng = _RNG(seed)
        self._values: list[tuple[int, dict[str, Any]]] = []
        for i, p in papers_with_text:
            if rng.random() < drop:
                continue  # 身体漏掉了这篇
            value = p["value"] * (1.0 + rng.gauss(0, noise_sd))
            method = p.get("method") or "DFT"
            self._values.append(
                (
                    i,
                    {
                        "value": round(value, 6),
                        "unit": p["unit"],
                        "method": method,
                        "note": "",
                        "paper_idx": i,
                    },
                )
            )

    async def ainvoke(self, messages: list[Any]) -> Any:
        """返回"抽取数值"的 JSON 字符串 (真实 _parse_json 会解析它)."""

        class _Resp:
            def __init__(self, content: str) -> None:
                self.content = content

        return _Resp(json.dumps({"values": [v for _, v in self._values]}))


class _RNG:
    """极简可复现随机源, 避免依赖 random/gauss 的全局状态差异."""

    def __init__(self, seed: int) -> None:
        self.state = seed

    def random(self) -> float:
        # xorshift64 简版, 确定性
        self.state ^= (self.state << 13) & 0xFFFFFFFFFFFFFFFF
        self.state ^= self.state >> 7
        self.state ^= (self.state << 17) & 0xFFFFFFFFFFFFFFFF
        return (self.state % 10000) / 10000.0

    def gauss(self, _mu: float, sigma: float) -> float:
        # 用两个 U, 中心极限近似 N(0,1) → 足够可复现
        s = sum(self.random() for _ in range(12)) - 6.0
        return sigma * s


# ── 从合成论文构造"带文本"输入, 并生成假身体返回值 ─────────────────────


def _papers_with_text(papers: list[dict[str, Any]]) -> list[tuple[int, dict]]:
    out: list[tuple[int, dict]] = []
    for i, p in enumerate(papers, 1):
        out.append(
            (
                i,
                {
                    **p,
                    "title": p["source_paper"],
                    "abstract": (
                        f"我们报道 {p['system']} 的 {p['property']} 为 "
                        f"{p['value']} {p['unit']}, 用 {p.get('method') or 'DFT'} 计算."
                    ),
                },
            )
        )
    return out


def _benchmark_args(system: str, property_: str, papers_with_text) -> LiteratureInput:
    paper_dicts = [p for _, p in papers_with_text]
    return LiteratureInput(
        action="benchmark_lookup",
        system=system,
        property=property_,
        papers=paper_dicts,
    )


# ── 主张真实生产路径 ─────────────────────────────────────────────────────


async def run_real_body(
    *,
    papers,
    system: str,
    property_: str,
    seed: int,
    noise_sd: float = 0.04,
    drop: float = 0.1,
    model=None,
) -> dict[str, Any]:
    """用 (假或真) 身体驱动真实 `_do_benchmark_lookup`.

    model 缺省时用 `FakeBodyModel` (可复现, 离线); 传真 model (如真实
    ChatOpenAI) 时就跑真 LLM 身体. 其余 (搜索/过滤/后端路由/校验) 全走生产代码.
    """
    tool = LiteratureTool()
    ctx = ToolContext(session_id="real-body", workspace=".")
    # 注入身体: 替换 _get_model, 其余完全不碰
    pts = _papers_with_text(papers)
    if model is None:
        model = FakeBodyModel(
            [(i, p) for i, p in pts if p],
            seed=seed,
            noise_sd=noise_sd,
            drop=drop,
        )
    tool._get_model = lambda context: model  # type: ignore[method-assign]
    args = _benchmark_args(system, property_, pts)
    result: ToolResult = await tool._do_benchmark_lookup(args, ctx)
    return result.data if isinstance(result.data, dict) else {}


def _real_deepseek_model(
    api_key: str, *, temperature: float, model: str = "deepseek-chat"
):
    """构造真实 deepseek 身体 (走 huginn/bench/llm_judge 同款封装)."""
    from langchain_openai import ChatOpenAI

    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY 未设置")
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",
        temperature=temperature,
        max_tokens=2000,
    )


# ── 内联驱动 (无重依赖) ──────────────────────────────────────────────────


def run(*, seed: int = 0, noise_sd: float = 0.04, drop: float = 0.1) -> dict[str, Any]:
    papers = _synthetic_papers(seed)
    # 取 Li2O band_gap 单体系, 保证单一 unit 语义
    system, property_ = "Li2O", "band_gap"
    subset = [p for p in papers if p["system"] == system and p["property"] == property_]
    if not subset:
        return {"ok": False, "error": "no papers for system"}

    data = asyncio.run(
        run_real_body(
            papers=subset,
            system=system,
            property_=property_,
            seed=seed,
            noise_sd=noise_sd,
            drop=drop,
        )
    )

    reported = data.get("reported_values", [])
    consistency = data.get("consistency") or {}
    consensus = data.get("consensus")
    return {
        "ok": True,
        "system": system,
        "property": property_,
        "n_papers_searched": data.get("n_papers_searched", 0),
        "n_reported": len(reported),
        "consensus": consensus,
        "consistency": consistency,
        "invariant_reproducible_seed": seed,
    }


def run_deepseek(
    *,
    api_key: str,
    system: str = "Li2O",
    property_: str = "band_gap",
    seed: int = 0,
    model: str = "deepseek-chat",
    papers: list[dict[str, Any]] | None = None,
    temps: tuple[float, float] = (0.2, 0.7),
    full_text: dict[str, str] | None = None,
) -> dict[str, Any]:
    """用真实 deepseek 身体跑多次 (不同 temperature), 判断结构层收敛性.

    诚实标注: 这里两个"身体"是**同一个真实 LLM、不同解码温度**, 不是两个
    不同品牌模型. 因此它在真实生产路径下验证的是"读取误差(解码随机性)下,
    agent 结构层仍把分散抽取收敛成同一稳健结论"; 它**不**声称验证"跨品牌
    模型一致性" (那需要两套独立 API key, 本环境不具备).

    判别: 各 body 的 consensus 中位数应落在同一真值邻域、verdict 一致.
    """
    if papers is None:
        papers = [
            p
            for p in _synthetic_papers(seed)
            if p["system"] == system and p["property"] == property_
        ]
    if not papers:
        return {"ok": False, "error": f"no {system} {property_} papers"}

    base = os.environ.get("DEEPSEEK_API_KEY", api_key)
    runs = []
    for t in temps:
        model_obj = _real_deepseek_model(base, temperature=t, model=model)
        aug = []
        for p in papers:
            item = dict(p)
            if full_text:
                key = p.get("doi") or p.get("source_paper") or ""
                item["full_text"] = full_text.get(key, item.get("full_text", ""))
            aug.append(item)
        data = asyncio.run(
            run_real_body(
                papers=aug,
                system=system,
                property_=property_,
                model=model_obj,
                seed=seed,
                drop=0.0,
                noise_sd=0.0,
            )
        )
        runs.append(
            {
                "temperature": t,
                "reported_values": data.get("reported_values", []),
                "consensus": data.get("consensus"),
                "consistency": data.get("consistency"),
                "n_reported": len(data.get("reported_values", [])),
            }
        )

    # 判别: 跨两身体稳健中心一致性
    medians = [
        r["consensus"]["median"]
        for r in runs
        if r["consensus"] and r["consensus"]["median"] is not None
    ]
    verdicts = [
        (r["consistency"] or {}).get("overall", {}).get("verdict") for r in runs
    ]
    tol = max(abs(papers[0]["value"]) * 0.15, 0.05) if papers else 0.05
    converge = False
    if len(medians) == len(runs) and len(medians) >= 2:
        converge = max(medians) - min(medians) <= tol
    return {
        "ok": True,
        "system": system,
        "property": property_,
        "model": model,
        "n_real_bodies": len(runs),
        "runs": runs,
        "real_bodies_converge": converge,
        "median_spread": (
            round(max(medians) - min(medians), 6) if len(medians) >= 2 else None
        ),
        "verdicts": verdicts,
    }


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--deepseek":
        key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not key:
            sys.exit("set DEEPSEEK_API_KEY and pass --deepseek")
        print(
            json.dumps(
                run_deepseek(api_key=key, temps=(0.2, 0.7)),
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(json.dumps(run(seed=1), ensure_ascii=False, indent=2))
