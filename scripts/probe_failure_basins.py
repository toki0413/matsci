"""probe_failure_basins —— 分形吸引域假说实验的 agent 端 harness（方案①, 零 LLM 成本）。

对齐方式:
  牛顿法载体(curve已校准, β=0.565<1)   agent 端 harness
  --------------------------------------------------------
  初始点 z                           初始假设扰动 ε (输入微扰)
  根的吸引域                          block_reason() 的失败类别域
  收敛到哪个根                         agent 最终卡在哪类失败/is_complete
  不确定指数 β                        f(ε)≈ε^β, β<1 ⇒ 失败边界分形

本 harness 用「确定性假模型」模拟"agent 对初始扰动 ε 敏感 → 落入不同失败类别"
的统计行为, 把"估 β"的整条管道(扰动→分类→碰撞概率→幂律拟合)在零 LLM 成本下
跑通。走通后, 把 `failure_classifier` 换成真 agent 的 block_reason() 即可上真数据。

不依赖 engine/LLM。复用 completion_auditor 的分类语义: 失败类别取四个可读标签
(effort / equivalence / confession / veto / ok)。
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

# ── 1. 失败类别锚: 复用 completion_auditor 的 block_reason 语义 ──────
# 与 CompletionChecklist.block_reason() 的四类 + 通过态对齐, 便于日后真接
FAIL_KEYS = ("effort", "equivalence", "confession", "veto")
OK_KEY = "ok"


# ── 2. 扰动 → 模拟 agent 落点 ──────────────────────────────────────
class FailureSim:
    """确定性伪随机 agent 失败模型.

    对每个扰动样本(基础态 + eps 偏移), 依据一个"分形边界发生器"判断两态
    是否落入不同失败类别。用于在 harness 阶段验证 β 拟合管道, 不声称反映真 agent。
    """

    def __init__(self, seed: int = 0, fractal: bool = True) -> None:
        self._rng = random.Random(seed)
        # 分形/光滑 开关: fractal=True → 两态在边界处判不同类的概率按幂律缩放
        self._fractal = fractal

    def category(self) -> str:
        """基础态失败类别(确定性), 作为"吸引子"之一。"""
        return FAIL_KEYS[self._rng.randrange(len(FAIL_KEYS))]

    def differs(self, cat: str, eps: float) -> bool:
        """对扰动 eps, 是否跳到另一个类别。"""
        if not self._fractal:
            # 光滑域: 概率 ~ eps (β≈1)
            return self._rng.random() < min(1.0, eps)
        # 分形域: 概率 ~ eps^0.55 (β=0.55, 更慢衰减 → 大尺度仍不同类)
        return self._rng.random() < min(1.0, eps ** 0.55)


# ── 3. 估 f(ε): 碰撞概率 ─────────────────────────────────────────
def _f(eps: float, sim: FailureSim, n: int = 4000) -> float:
    diff = 0
    for _ in range(n):
        c = sim.category()
        if sim.differs(c, eps):
            diff += 1
    return diff / n


# ── 4. 幂律拟合 β (同校准探针的最小二乘) ───────────────────────────
def _fit_beta(epss, fs) -> float:
    pts = [(math.log(e), math.log(f)) for e, f in zip(epss, fs)
           if f > 0 and e > 0]
    n = len(pts)
    if n < 2:
        return float("nan")
    sx = sum(p[0] for p in pts)
    sy = sum(p[1] for p in pts)
    sxx = sum(p[0] * p[0] for p in pts)
    sxy = sum(p[0] * p[1] for p in pts)
    denom = n * sxx - sx * sx
    return (n * sxy - sx * sy) / denom if denom else float("nan")


# ── 5. harness 主入口 ────────────────────────────────────────────
def probe_failure_basins(epss=None, fractal: bool = True, seed: int = 0,
                         n_per_eps: int = 4000) -> dict:
    """扫 ε → 估 f(ε) → 拟合 β.

    返回 {beta, fractal_verdict, f_by_eps}。这是 harness 验收输出:
      若 fractal=True  → β 应明显 <1 (管道能测出分形)
      若 fractal=False → β 应≈1    (管道能测出光滑=假说测伪)
    """
    epss = epss or [0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]
    sim = FailureSim(seed=seed, fractal=fractal)
    fs = [_f(e, sim, n=n_per_eps) for e in epss]
    beta = _fit_beta(epss, fs)
    verdict = "FRACTAL(β<0.9)" if beta < 0.9 else ("NOT_FRACTAL(β≈1)" if beta > 0.95 else "BORDERLINE")
    return {
        "beta": beta,
        "verdict": verdict,
        "fractal_ground_truth": fractal,
        "f_by_eps": dict(zip([f"{e:.6g}" for e in epss],
                             [f"{f:.4f}" for f in fs])),
    }


if __name__ == "__main__":
    print("=" * 64)
    print("probe_failure_basins harness —— 零 LLM 验证估 β 管道")
    print("=" * 64)
    for fractal in (True, False):
        r = probe_failure_basins(fractal=fractal, seed=42)
        print(f"\n[ground_truth=fractal={fractal}]")
        print(f"  β = {r['beta']:.3f}  →  {r['verdict']}")
        print(f"  f(ε): {r['f_by_eps']}")
    print("\n预期: fractal=True→β<1; fractal=False→β≈1。管道若都吻合, harness 校验通过。")