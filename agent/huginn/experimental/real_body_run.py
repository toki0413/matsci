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
) -> dict[str, Any]:
    """用假身体驱动真实 `_do_benchmark_lookup`, 返回结构化结果."""
    tool = LiteratureTool()
    ctx = ToolContext(session_id="real-body", workspace=".")
    # 注入假身体: 替换 _get_model, 其余 (搜索/过滤/后端路由) 完全不碰
    pts = _papers_with_text(papers)
    model = FakeBodyModel(
        [(i, p) for i, p in pts if p], seed=seed, noise_sd=noise_sd, drop=drop
    )
    tool._get_model = lambda context: model  # type: ignore[method-assign]
    args = _benchmark_args(system, property_, pts)
    result: ToolResult = await tool._do_benchmark_lookup(args, ctx)
    return result.data if isinstance(result.data, dict) else {}


# ── 内联驱动 (无重依赖) ──────────────────────────────────────────────────


def run(*, seed: int = 0, noise_sd: float = 0.04, drop: float = 0.1) -> dict[str, Any]:
    papers = _synthetic_papers(seed)
    # 取 Li2O band_gap 单体系, 保证单一 unit 语义
    system, property_ = "Li2O", "band_gap"
    subset = [p for p in papers if p["system"] == system]
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


if __name__ == "__main__":
    import json

    print(json.dumps(run(seed=1), ensure_ascii=False, indent=2))
