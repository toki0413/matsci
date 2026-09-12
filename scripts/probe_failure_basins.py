"""probe_failure_basins —— 分形吸引域假说实验的 agent 端 harness（方案①b, 接入真分类逻辑）。

阶段更新:
  方案①a(FailureSim 假模型)已验证"估 β 管道"可分形/光滑。
  方案①b 把假 FailureSim 换成**真 completion_auditor**, 让 harness 用真实分类逻辑、
  假 agent 跑通 —— 仍零 LLM, 但"失败指纹"来自真实代码而非手工造。

失败指纹(连续向量, 长度=5):
  [effort_deficit, equivalence_trap, missing_confession, veto, complete]
  由 CompletionAuditor.audit() 返回的 CompletionChecklist 的真实字段计算。
  两态碰撞 = 指纹(基础态) 与 指纹(带扰动 ε 的态) 的类别不同。
  扰动模拟: 对 audit 的输入参数加一个随机偏移 ∈[-ε,+ε], 表示"初始假设/努力程度
  的微小扰动"; 用真分类器重算指纹, 判是否落入不同失败类别。

零 LLM: audit 不依赖 engine/LLM(equivalence_auditor 不传 model 走规则)。
跑通后, 把"输入扰动+重跑"换成真 agent 交互即可上真数据。
"""
from __future__ import annotations

import math
import random
from pathlib import Path

# 真分类器
from huginn.metacog.completion_auditor import CompletionAuditor


def _fingerprint(c) -> tuple[int, ...]:
    """把 CompletionChecklist 压成 5 维 0/1 失败指纹.

    顺序固定: 努力不足, 等价陷阱, 缺自白, 对抗否决, 完成.
    """
    return (
        0 if c.effort_floor_passed else 1,
        1 if c.equivalence_traps_remaining else 0,
        1 if not c.unexplored_count else 0,
        1 if c.adversarial_veto else 0,
        1 if c.is_complete else 0,
    )


class RealClassifierAgent:
    """用真 completion_auditor 做"agent", 对绕组后的输入做分类."""

    def __init__(self, auditor: CompletionAuditor | None = None) -> None:
        self._auditor = auditor or CompletionAuditor()
        self._rng = random.Random(0)

    def result_at(self, base: dict, eps: float) -> tuple[int, ...]:
        """在 base 参数上叠加 ±eps 随机扰动, 用真分类器算出失败指纹."""
        d = dict(base)
        d["iteration"] = base["iteration"] + self._rng.uniform(-eps, eps) * 8
        d["families_explored"] = base["families_explored"] + self._rng.uniform(-eps, eps) * 3
        d["live_components"] = base["live_components"] + self._rng.uniform(-eps, eps) * 2
        # candidate_finding/original/reduction 是字符串, 扰动只作用数值型工作量参数
        c = self._auditor.audit(
            iteration=int(max(0, d["iteration"])),
            families_explored=int(max(0, d["families_explored"])),
            live_components=int(max(0, d["live_components"])),
            total_iterations=d.get("total_iterations", 10),
            candidate_finding=d.get("candidate_finding", ""),
            original_problem=d.get("original_problem", ""),
            reduction_chain=d.get("reduction_chain", ""),
            unexplored_declaration=d.get("unexplored_declaration", ""),
        )
        return _fingerprint(c)


def _f(eps: float, agent: RealClassifierAgent, base: dict, n: int = 1200,
       seed_offset: int = 0) -> float:
    """对扰动 eps, 基础态与扰动态落入不同失败指纹的概率."""
    agent._rng = random.Random(seed_offset)
    k0 = agent.result_at(base, 0.0)
    diff = 0
    for _ in range(n):
        k2 = agent.result_at(base, eps)
        if k0 != k2:
            diff += 1
    return diff / n


def _fit_beta(epss, fs) -> float:
    pts = [(math.log(e), math.log(f)) for e, f in zip(epss, fs) if f > 0 and e > 0]
    n = len(pts)
    if n < 2:
        return float("nan")
    sx = sum(p[0] for p in pts)
    sy = sum(p[1] for p in pts)
    sxx = sum(p[0] * p[0] for p in pts)
    sxy = sum(p[0] * p[1] for p in pts)
    denom = n * sxx - sx * sx
    return (n * sxy - sx * sy) / denom if denom else float("nan")


def probe_with_real_classifier(base: dict, epss=None, n_per_eps: int = 1200,
                               seed: int = 1) -> dict:
    """扫 ε → 用真分类器估 f(ε) → 拟合 β."""
    epss = epss or [0.5, 0.25, 0.12, 0.06, 0.03, 0.015]
    agent = RealClassifierAgent()
    fs = [_f(e, agent, base, n=n_per_eps, seed_offset=seed + i)
          for i, e in enumerate(epss)]
    beta = _fit_beta(epss, fs)
    verdict = "FRACTAL(β<0.9)" if beta < 0.9 else ("NOT_FRACTAL(β≈1)" if beta > 0.95 else "BORDERLINE")
    return {
        "beta": beta,
        "verdict": verdict,
        "f_by_eps": dict(zip([f"{e:.4g}" for e in epss], [f"{f:.4f}" for f in fs])),
    }


# ── 三种"工作量画像"基态, 对应 agent 不同深度 ───────────────────
BASES = {
    # 基线: 中期, 及格工作量, 有自白无否决 —— 真实分类器最常见画像
    "baseline": dict(
        iteration=6, families_explored=3, live_components=2, total_iterations=10,
        candidate_finding="用主动学习找到相场边界",
        original_problem="找相场边界", reduction_chain="",
        unexplored_declaration="- 高压相图\n- 多组分体系",
    ),
    # 浅: 早停边缘, 贴近努力下限 —— 扰动易跨越 fraction
    "shallow": dict(
        iteration=3, families_explored=3, live_components=2, total_iterations=10,
        candidate_finding="稳定性由形成能决定",
        original_problem="预测材料稳定性", reduction_chain="通过形成能",

        unexplored_declaration="- 缺陷容忍度",
    ),
    # 深: 后期, 该收敛时, 扰动小
    "deep": dict(
        iteration=9, families_explored=4, live_components=2, total_iterations=10,
        candidate_finding="主动学习锁定相场边界",
        original_problem="找相场边界", reduction_chain="",
        unexplored_declaration="- 高压区",
    ),
}


if __name__ == "__main__":
    print("=" * 68)
    print("probe_failure_basins 方案①b —— 用【真 completion_auditor】当失败分类器")
    print("零 LLM: 真分类逻辑 + 扰动重跑; 通完才上真数据")
    print("=" * 68)
    keys = list(BASES)
    for name in keys:
        r = probe_with_real_classifier(BASES[name])
        print(f"\n[{name}] 基态: iter={BASES[name]['iteration']} "
              f"fam={BASES[name]['families_explored']} "
              f"live={BASES[name]['live_components']}")
        print(f"  β = {r['beta']:.3f}  →  {r['verdict']}")
        print(f"  f(ε): {r['f_by_eps']}")