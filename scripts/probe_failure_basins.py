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

import numpy as np

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


# ── 交叉验证版: 确定性 2D 晶格 Chern–Markus β (与 research_a 同一协议) ──
# research_a 把语义 cosine 当终态; 这里把布尔 completion_auditor 指纹当终态,
# 用尽对同一测量协议(P_cross(δ)~δ^β + 光滑/噪声对照)做双路交叉验证。
#   1. 二维参数 (u,v)∈[0,1]² 确定性扰动审计输入(u→工作量, v→自白条目)
#   2. 终态 = completion_auditor 的 5 维失败指纹 (int 编码)
#   3. P_cross(δ): 网格上相距 δ 两点落入不同指纹类别比例
#   4. 拟合 log P_cross ~ β log δ; 对照: 光滑→β≈1, 噪声→β≈0
# 关键: 扰动是确定性连续函数(无 per-cell 随机抖动), 避免把噪声当分形。


def _fp_int(fp: tuple[int, ...]) -> int:
    """5 维 0/1 指纹 → 整数标签 (2^5 个可能吸引域)."""
    return sum(b << i for i, b in enumerate(fp))


def _decl_count(text: str) -> int:
    return sum(1 for ln in str(text).splitlines() if ln.strip().startswith("- "))


def _decl(count: int) -> str:
    if count <= 0:
        return "已充分探索"
    return "\n".join(f"- 未探索项{i}" for i in range(count))


def _u01(n):
    return [i / (n - 1) if n > 1 else 0.5 for i in range(n)]


def _labels_grid(base: dict, grid_n: int, agent: RealClassifierAgent) -> np.ndarray:
    """在 [0,1]² 上逐点审计 → (grid_n,grid_n) int 指纹标签阵."""
    g = np.empty((grid_n, grid_n), dtype=int)
    us, vs = _u01(grid_n), _u01(grid_n)
    agent._rng = random.Random(0)  # audit 不接受随机位移, 无害固定
    for i, u in enumerate(us):
        for j, v in enumerate(vs):
            c = agent._auditor.audit(
                iteration=int(max(0, base["iteration"] + u * 8)),
                families_explored=int(max(0, base["families_explored"] + u * 3)),
                live_components=int(max(0, base["live_components"] + u * 2)),
                total_iterations=base.get("total_iterations", 10),
                candidate_finding=base.get("candidate_finding", ""),
                original_problem=base.get("original_problem", ""),
                reduction_chain=base.get("reduction_chain", ""),
                unexplored_declaration=_decl(
                    int(round(_decl_count(base.get("unexplored_declaration", "")) + v * 3))
                ),
            )
            g[i, j] = _fp_int(_fingerprint(c))
    return g


def _p_cross(g: np.ndarray, grid_n: int, delta_frac: float, rng, samples: int = 8000) -> float:
    d = max(1, int(round(delta_frac * (grid_n - 1))))
    steps = [(-d, 0), (d, 0), (0, -d), (0, d), (d, d), (-d, d), (d, -d), (-d, -d)]
    cnt = tot = 0
    for _ in range(samples):
        i = int(rng.integers(0, grid_n))
        j = int(rng.integers(0, grid_n))
        di, dj = rng.choice(steps)
        i2, j2 = i + di, j + dj
        if 0 <= i2 < grid_n and 0 <= j2 < grid_n:
            tot += 1
            if g[i, j] != g[i2, j2]:
                cnt += 1
    return cnt / tot if tot else 0.0


DELTAS = [0.02, 0.04, 0.06, 0.08, 0.12, 0.16, 0.25, 0.33]


def _fit_beta_r2(deltas, pcs) -> dict:
    pts = [(math.log(d), math.log(max(p, 1e-9))) for d, p in zip(deltas, pcs)
           if 0.0 < p < 1.0]
    if len(pts) < 3:
        return {"beta": float("nan"), "r2": float("nan"), "n_used": len(pts)}
    xs = [x for x, _ in pts]
    ys = [y for _, y in pts]
    slope, intercept = np.polyfit(xs, ys, 1)
    yhat = [slope * x + intercept for x in xs]
    ss_res = sum((y - yh) ** 2 for y, yh in zip(ys, yhat))
    ss_tot = sum((y - sum(ys) / len(ys)) ** 2 for y in ys)
    r2 = 1.0 - ss_res / ss_tot if ss_tot else float("nan")
    return {"beta": float(slope), "r2": float(r2), "n_used": len(pts)}


def estimate_beta_2d(base: dict, grid_n: int = 150, seed: int = 3) -> dict:
    """对单个基态跑 2D 晶格 β: 真标签 + 光滑对照 + 噪声对照."""
    agent = RealClassifierAgent()
    g_true = _labels_grid(base, grid_n, agent)

    g_smooth = np.asarray([[0 if u < 0.5 else 1 for _ in _u01(grid_n)]
                           for u in _u01(grid_n)], dtype=int)
    rng_noise = random.Random(seed)
    g_noise = np.asarray(
        [[rng_noise.randint(0, 1) for _ in _u01(grid_n)] for _ in _u01(grid_n)],
        dtype=int,
    )

    out = {}
    for name, gg in [("真实指纹吸引域", g_true), ("光滑对照", g_smooth), ("噪声对照", g_noise)]:
        rng = np.random.default_rng(seed)
        pcs = [_p_cross(gg, grid_n, d, rng) for d in DELTAS]
        fit = _fit_beta_r2(DELTAS, pcs)
        out[name] = {**fit}
    return {"grid_n": grid_n, "beta": out["真实指纹吸引域"]["beta"], **out}


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
    import sys

    # ── 双路交叉验证: 确定性 2D 晶格 β (同一协议, 与 research_a 对齐) ──
    if "--beta2d" in sys.argv:
        print("=" * 68)
        print("布尔分类器交叉验证 —— 确定性 2D 晶格 P_cross(δ)~δ^β (Chern–Markus)")
        print("同一测量协议, 与 research_a 语义余弦路对齐")
        print("=" * 68)
        for name in BASES:
            r = estimate_beta_2d(BASES[name])
            print(f"\n[{name}] 基态: iter={BASES[name]['iteration']} "
                  f"fam={BASES[name]['families_explored']} "
                  f"live={BASES[name]['live_components']}")
            for k in ("真实指纹吸引域", "光滑对照", "噪声对照"):
                item = r[k]
                b_k, r2 = item["beta"], item["r2"]
                if math.isnan(b_k):
                    sign = "无边界(P_cross=0/指纹恒定)"
                else:
                    sign = ("分形" if b_k < 0.85 else
                            "平滑(β≈1)" if b_k > 0.95 else "模糊")
                print(f"  {k:<8} β={b_k if not math.isnan(b_k) else float('nan'):+.3f} "
                      f"R²={'%.3f' % r2 if not math.isnan(r2) else '--'} "
                      f"({item['n_used']}/{len(DELTAS)}点)  → {sign}")
            b = r["beta"]
            if math.isnan(b):
                print(f"  → 布尔分类器该基态无失败边界可测(指纹对扰动恒定), 与语义路不可比")
            else:
                print(f"  → 布尔分类器 β={b:+.3f} "
                      f"({'分形, 与语义路不一致' if b < 0.85 else '平滑, 与语义路一致(β≈1)'})")
        sys.exit(0)

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