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


def _decl_count(text: str) -> int:
    return sum(1 for ln in str(text).splitlines() if ln.strip().startswith("- "))


def _u01(n):
    return [i / (n - 1) if n > 1 else 0.5 for i in range(n)]


def _labels_grid(base: dict, grid_n: int, agent: RealClassifierAgent) -> np.ndarray:
    """向量化: 指纹按其独立性分解为 u-工作量位 与 v-自白位 的组合.

    fingerprint 位 = F(iteration(u), families(u), live(u), unexplored_count(v)):
      bit0 努力达标   → 只依赖 u
      bit1 等价陷阱   → 依赖字符串(网格内恒定)
      bit2 缺自白     → 只依赖 v
      bit3 对抗否决   → 恒定
      bit4 完成(is_complete) → 耦合: 努力&非陷阱&非否决 & 自白≥1
    故对唯一 u 调用一次 audit(读数 effort 位), 对唯一 v 分离计算自白位,
    numpy 增广组合即可, 复杂度 O(grid_n) 而非 O(grid_n²).
    """
    us, vs = _u01(grid_n), _u01(grid_n)
    const_trap = False
    const_veto = False
    eff_passed = np.zeros(grid_n, dtype=bool)   # bit0 per u
    for i, u in enumerate(us):
        c = agent._auditor.audit(
            iteration=int(max(0, base["iteration"] + u * 8)),
            families_explored=int(max(0, base["families_explored"] + u * 3)),
            live_components=int(max(0, base["live_components"] + u * 2)),
            total_iterations=base.get("total_iterations", 10),
            candidate_finding=base.get("candidate_finding", ""),
            original_problem=base.get("original_problem", ""),
            reduction_chain=base.get("reduction_chain", ""),
            unexplored_declaration="已充分探索",
        )
        eff_passed[i] = c.effort_floor_passed
        const_trap = bool(c.equivalence_traps_remaining)
        const_veto = bool(c.adversarial_veto)
    base_n = _decl_count(base.get("unexplored_declaration", ""))
    confess_bit = np.zeros(grid_n, dtype=bool)   # bit2 = 缺自白(1: 无条目) per v
    for j, v in enumerate(vs):
        n_self = int(max(0, round(base_n - v * 3)))  # 向下扰动: 自白条目减少→翻到0
        confess_bit[j] = (n_self < 1)
    # 增广组合 → (grid_n,grid_n) 指纹 int
    # bit0 随u横扩, bit2 随v竖扩, bit1/bit3 常量
    bit0 = eff_passed[:, None]
    bit2 = confess_bit[None, :]
    bit4 = bit0 & (not const_trap) & (not const_veto) & (~bit2)
    g = (np.logical_not(bit0).astype(int) << 0) | (int(const_trap) << 1) \
        | (bit2.astype(int) << 2) | (int(const_veto) << 3) | (bit4.astype(int) << 4)
    return g


def _p_cross(g: np.ndarray, grid_n: int, delta_frac: float, rng, samples: int = 8000) -> float:
    """向量化采样相距δ的点对, 统计落入不同指纹类别占比."""
    d = max(1, int(round(delta_frac * (grid_n - 1))))
    steps = np.asarray([(-d, 0), (d, 0), (0, -d), (0, d),
                        (d, d), (-d, d), (d, -d), (-d, -d)], dtype=int)
    i = rng.integers(0, grid_n, size=samples)
    j = rng.integers(0, grid_n, size=samples)
    s = rng.integers(0, len(steps), size=samples)
    i2 = i + steps[s, 0]
    j2 = j + steps[s, 1]
    ok = (i2 >= 0) & (i2 < grid_n) & (j2 >= 0) & (j2 < grid_n)
    if not ok.any():
        return 0.0
    same = g[i[ok], j[ok]] == g[i2[ok], j2[ok]]
    return float(1.0 - same.mean())


DELTAS = [0.02, 0.04, 0.06, 0.08, 0.12, 0.16, 0.25, 0.33]
D_FINE = [0.02, 0.03, 0.045, 0.06]     # 渐近细 δ 窗: 分解有限直边 vs 自相似
D_COARSE = [0.12, 0.16, 0.22, 0.33]   # 粗 δ 窗


def _slope_deltas(g: np.ndarray, grid_n: int, deltas, seed: int) -> float:
    """对给定 δ 集合拟合局部斜率 β (分隔判定用)."""
    rng = np.random.default_rng(seed)
    pcs = [_p_cross(g, grid_n, d, rng) for d in deltas]
    return _fit_beta_r2(deltas, pcs)["beta"]


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


def estimate_beta_scaled(base: dict, grid_ns=(80, 120, 160, 200, 260, 320),
                         seed: int = 3) -> dict:
    """深挖: 同一基态下随网格分辨率放大, 测 β 收敛 + 细/粗 δ 局部斜率.

    判读:
      β 随 grid_n 收敛到 1        → 纯量化伪影
      β 稳定 <1 且细/粗斜率一致      → 自相似分形
      β 稳定 <1 但细斜率→1/粗斜率→0  → 有限直边交叉网(非分形, 窗口混叠)
    """
    rows = {}
    for gn in grid_ns:
        agent = RealClassifierAgent()
        g_true = _labels_grid(base, gn, agent)
        r = estimate_beta_2d(base, grid_n=gn, seed=seed)
        r["fine"] = _slope_deltas(g_true, gn, D_FINE, seed)
        r["coarse"] = _slope_deltas(g_true, gn, D_COARSE, seed)
        rows[gn] = r
    return {"grid_ns": grid_ns, "by_resolution": rows}


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

    # ── 深挖: shallow 基态随网格分辨率放大, β 收敛行为 ──
    if "--deepdive" in sys.argv:
        print("=" * 68)
        print("深挖 shallow —— 网格分辨率扫描 β 收敛性 + 细/粗δ 局部斜率")
        print("分形=细/粗斜率一致且稳定<1; 有限交叉网=细→1粗→0; 收敛到1=量化伪影")
        print("=" * 68)
        base = BASES["shallow"]
        res = estimate_beta_scaled(base)
        print(f"\n基态[shallow]: iter={base['iteration']} fam={base['families_explored']} "
              f"live={base['live_components']}\n")
        print(f"  {'grid_n':>6}  {'总体β':>7} {'R²':>6}  {'细δβ':>7}  {'粗δβ':>7}")
        for gn in res["grid_ns"]:
            r = res["by_resolution"][gn]
            print(f"  {gn:>6}  {r['beta']:>+7.3f} {r['真实指纹吸引域']['r2']:>6.3f}  "
                  f"{r['fine']:>+7.3f}  {r['coarse']:>+7.3f}")
        betas = [res["by_resolution"][gn]["beta"] for gn in res["grid_ns"]]
        fines = [res["by_resolution"][gn]["fine"] for gn in res["grid_ns"]]
        coarses = [res["by_resolution"][gn]["coarse"] for gn in res["grid_ns"]]
        if all(math.isnan(b) for b in betas):
            print("\n→ 所有分辨率下真实β=nan: 无边界可测")
        else:
            bs = [b for b in betas if not math.isnan(b)]
            fs = [b for b in fines if not math.isnan(b)]
            cs = [b for b in coarses if not math.isnan(b)]
            drift = max(bs) - min(bs)
            print(f"\n→ 真实β: min={min(bs):.3f} max={max(bs):.3f} 漂移={drift:+.3f} "
                  f"末分辨率β={bs[-1]:+.3f}")
            mean_f, mean_c = sum(fs) / len(fs), sum(cs) / len(cs)
            spread = mean_c - mean_f
            print(f"  细δβ均值={mean_f:+.3f}  粗δβ均值={mean_c:+.3f}  细粗差={spread:+.3f}")
            if spread > 0.35 or mean_f > 0.9:
                print("\n 判读: 细δ斜率→1、粗δ→0 → 有限直边交叉网, 非分形 (特征尺度过有限)")
            elif drift < 0.1 and abs(spread) < 0.25 and max(bs) < 0.95:
                print("\n 判读: β 稳定<1 且细/粗斜率接近 → 自相似分形信号(需更大δ窗确认)")
            else:
                print("\n 判读: 介于两者之间, 需更大分辨率或更宽δ窗确认")
        sys.exit(0)

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