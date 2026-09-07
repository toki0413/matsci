"""Huginn Research — 域无关的自主深研统一管线(产品能力).

这不是参赛演示, 而是 Huginn 产品里的一条通用能力:
给定一个研究目标, 自动完成 假说生成 → 真实实验执行 → Pareto 演化/淘汰 → 
批判综合 → 声明门禁 → 兜底组装 的完整深研闭环.

设计(对齐 2026 Co-Scientist / GPT-6 Astra / Fable 的 product-grade 做法):
  - 域无关: 只依赖 ``Experiment`` 抽象与一个可插拔执行函数, 不碰具体物理.
  - 复用真实 ``ExplorationOrchestrator``(min_iterations 防早停 + max_iterations 预算).
  - 批判/演化: Pareto 剪枝维持前沿 = "debate 淘汰"的存活假说.
  - 综合: supervisor 可选(传 client 则真机成文; 否则确定性组装), 再走 claim_grounding 门禁.
  - 兜底: 真机交不出可落地报告 → 确定性组装(数值全来自真实执行, 必过门禁).
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass
class Experiment:
    """一个可执行的研究假说/实验.

    name: 唯一标识; hypothesis: 人类可读假说; run: 执行真实实验,
        返回 dict(含 summary 与 objectives[magnet maximize]).
    """
    name: str
    hypothesis: str
    run: Callable[[], dict[str, Any]]


@dataclass
class ResearchOutcome:
    converred: str = ""
    explored: int = 0
    pruned: int = 0
    pareto_front: list[dict] = field(default_factory=list)   # 存活假说 [{name, hypothesis, objectives}]
    cache: dict[str, dict] = field(default_factory=dict)     # name -> {summary, objectives}
    report: str = ""
    verdict: str = "needs_grounding"
    ungrounded: list = field(default_factory=list)
    report_source: str = "deterministic"


def _build_trace(cache: dict[str, dict]) -> list[str]:
    return [json.dumps(v, ensure_ascii=False) for v in cache.values()]


def run_research_program(
    goal: str,
    experiments: list[Experiment],
    objectives_config: dict[str, str],
    *,
    max_iterations: int = 40,
    min_iterations: int = 6,
    max_parallel: int = 2,
    client: Any = None,            # OpenAI 兼容 client; None → 确定性综合
    model: str = "intern-s2-preview",
    base_url: str | None = None,
    verify: Callable[[str, list[str]], dict] | None = None,
    out_md: Path | None = None,
) -> ResearchOutcome:
    """跑一条完整深研管线并返回结果."""
    from huginn.exploration.orchestrator import ExplorationOrchestrator
    import huginn.exploration.strategies as S

    cache: dict[str, dict] = {}
    spec_by_name = {e.name: e for e in experiments}

    async def executor(branch):
        spec = spec_by_name.get(branch.name)
        if spec is None:
            return {"success": False, "objectives": {}, "results": {}}
        res = await asyncio.to_thread(spec.run)
        cache[branch.name] = res
        return {"success": bool(res.get("success", True)),
                "objectives": res.get("objectives", {}),
                "results": res.get("summary", {})}

    orch = ExplorationOrchestrator(
        strategy=S.ParetoPruningStrategy(max_active=max_parallel + 6),
        branch_executor=executor,
        max_parallel=max_parallel,
    )

    # 声明门禁: 默认从产品内加载 claim_grounding
    def _default_verify(text, trace):
        try:
            from huginn.validation.claim_grounding import verify_claims
            return verify_claims(text, trace, allow_derived=True)
        except Exception:
            from importlib import util
            src = Path(__file__).resolve().parents[1] / "validation/claim_grounding.py"
            spec = util.spec_from_file_location("_cg", str(src))
            mod = util.module_from_spec(spec); spec.loader.exec_module(mod)
            return mod.verify_claims(text, trace, allow_derived=True)

    verify = verify or _default_verify

    result = asyncio.run(orch.explore(
        objective=goal,
        initial_branches=[{"name": e.name, "hypothesis": e.hypothesis} for e in experiments],
        objectives_config=objectives_config,
        max_iterations=max_iterations,
        min_iterations=min_iterations,
    ))

    trace = _build_trace(cache)
    front = result.pareto_front or []
    out = ResearchOutcome(converred=result.convergence_reason,
                          explored=result.n_branches_explored,
                          pruned=result.n_branches_pruned,
                          pareto_front=front, cache=cache)
    survivors = [(b["name"], cache.get(b["name"], {})) for b in front]

    def _synthesize() -> str:
        L = [f"# 自主深研 — {goal}", "",
             "> 存活假说(Pareto 前沿, debate 淘汰后): " + ", ".join(n for n, _ in survivors) + "", ""]
        for name, res in survivors:
            hyp = spec_by_name.get(name).hypothesis if spec_by_name.get(name) else ""
            L.append(f"## {name}")
            L.append(f"- 假说: {hyp}")
            if res.get("summary"):
                L.append(f"- 真实结果: {json.dumps(res['summary'], ensure_ascii=False)}")
            L.append("")
        L.append("## 结论(开放)")
        L.append("存活假说覆盖目标下的多证据方向, 数值均来自真实执行、可复现。")
        return "\n".join(L)

    final, verdict, ungrounded = "", "needs_grounding", []
    if client is not None:
        survivors_text = "\n".join(
            f"- {name}: {json.dumps(res.get('summary', {}), ensure_ascii=False)}" for name, res in survivors)
        prompt = (f"你是 Huginn 科研智能体。目标: {goal}。\n"
                  f"以下是存活假说的真实数值证据(可复现、非伪造):\n{survivors_text}\n\n"
                  f"请撰写跨学科深度研究报告(研究问题/数据与方法/结果分析/对账与局限/下一步)。"
                  f"每个数值必须来自上面真实结果, 不许编造。把最终报告放在 <report> 与 </report> 之间。")
        for _ in range(3):
            r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}],
                                               max_tokens=4000, temperature=0.2)
            final = r.choices[0].message.content or ""
            m = re.search(r"<report>(.*?)</report>", final, flags=re.DOTALL | re.IGNORECASE)
            if m:
                final = m.group(1).strip()
            g = verify(final, trace)
            if g["verdict"] == "pass" and len(final.strip()) > 200:
                verdict, ungrounded = "pass", []; break
            ungrounded = g["unsubstantiated"]
            prompt = f"未交付(未落地:{ungrounded})。请只用真实证据重写:\n" + prompt
        else:
            final = _synthesize()      # L3b 兜底组装, 必过门禁
            out.report_source = "fallback_assembly(agent failed)"
    else:
        final = _synthesize()
        out.report_source = "deterministic"

    g2 = verify(final, trace)
    if g2["verdict"] == "pass":
        verdict, ungrounded = "pass", []
    else:
        ungrounded = g2["unsubstantiated"]

    out.report = final
    out.verdict, out.ungrounded = verdict, ungrounded

    if out_md is not None:
        header = (f"# 自主深研(Huginn×书生)\n\n> **门禁: {verdict}** (未落地: {ungrounded or '无'})\n"
                  f"> real orchestration: explored={out.explored} pruned={out.pruned} "
                  f"convergence={out.converred}\n> 报告来源: {out.report_source}\n\n")
        out_md.write_text(header + final.strip() + "\n", encoding="utf-8")
    return out