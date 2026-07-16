"""OptimBench — 算法优化能力评测.

对标 MLE-Bench: 给定优化目标, 让 agent 写代码/算参数来优化结果.
材料科学导向: 成分优化、晶格优化、超参优化、带隙最大化.

evaluator 用数值容差判分, 不依赖 LLM judge.
"""

from __future__ import annotations

import re

from .task import BenchmarkTask


def _extract_number(text: str, pattern: str) -> float | None:
    m = re.search(pattern, text, re.IGNORECASE)
    if not m:
        return None
    try:
        return float(m.group(1))
    except (ValueError, IndexError):
        return None


def _num_close(value: float, expected: float, tol: float) -> bool:
    return abs(value - expected) <= tol


# ── T1: 合金成分优化 ─────────────────────────────────────────────

def _eval_alloy_optim(output: str) -> tuple[bool, str, float]:
    """目标: 最大化硬度 H = 5*x + 3*y, 约束 x+y<=10, x,y>=0.
    最优解 x=10, y=0, H=50. 容差 ±2."""
    x = _extract_number(output, r"(?:x|铜|Cu)[^0-9]*(\d+\.?\d*)")
    y = _extract_number(output, r"(?:y|锌|Zn)[^0-9]*(\d+\.?\d*)")
    h = _extract_number(output, r"(?:H|硬度|hardness)[^0-9]*(\d+\.?\d*)")
    # 优先看 H
    if h is not None:
        if _num_close(h, 50.0, 2.0):
            return True, f"H={h:.1f} (最优 ~50)", 1.0
        if _num_close(h, 50.0, 5.0):
            return False, f"H={h:.1f}, 接近但未达最优", 0.5
        return False, f"H={h:.1f}, 期望 ~50", 0.2
    # 没 H, 看 x,y
    if x is not None and y is not None:
        calc_h = 5 * x + 3 * y
        if _num_close(calc_h, 50.0, 2.0):
            return True, f"x={x}, y={y}, H={calc_h:.1f}", 1.0
        return False, f"x={x}, y={y}, H={calc_h:.1f}, 期望 ~50", 0.3
    return False, "未找到优化结果 (H 或 x,y)", 0.0


# ── T2: 晶格常数优化 ─────────────────────────────────────────────

def _eval_lattice_optim(output: str) -> tuple[bool, str, float]:
    """Lennard-Jones 势能: E(a) = 4ε[(σ/a)¹² - (σ/a)⁶], ε=1, σ=1.
    平衡晶格常数 a₀ = 2^(1/6) ≈ 1.1225. 最小能量 E_min = -ε = -1.
    判 a₀ 或 E_min."""
    a = _extract_number(output, r"(?:a|晶格|lattice)[^0-9]*(\d+\.?\d*)")
    e = _extract_number(output, r"(?:E|能量|energy)[^0-9]*(-?\d+\.?\d*)")
    if a is not None and _num_close(a, 1.1225, 0.01):
        return True, f"a₀={a:.4f} (期望 ~1.1225)", 1.0
    if e is not None and _num_close(e, -1.0, 0.05):
        return True, f"E_min={e:.2f} (期望 ~-1.0)", 1.0
    if a is not None and _num_close(a, 1.1225, 0.05):
        return False, f"a₀={a:.4f}, 接近 1.1225", 0.5
    return False, f"a={a}, E={e}, 期望 a₀~1.1225 或 E~-1.0", 0.2


# ── T3: KRR 超参优化 ─────────────────────────────────────────────

def _eval_krr_optim(output: str) -> tuple[bool, str, float]:
    """Kernel Ridge Regression: K(x,x') = exp(-||x-x'||²/(2σ²)).
    给定 5 折 CV, 最优 σ≈0.5, C≈10. 判 σ 和 C."""
    sigma = _extract_number(output, r"(?:σ|sigma|带宽|bandwidth)[^0-9]*(\d+\.?\d*)")
    c = _extract_number(output, r"(?:C|正则|regular)[^0-9]*(\d+\.?\d*)")
    score = 0.0
    if sigma is not None and _num_close(sigma, 0.5, 0.1):
        score += 0.5
    if c is not None and _num_close(c, 10.0, 2.0):
        score += 0.5
    if score >= 0.5:
        return True, f"σ={sigma}, C={c}", score
    return False, f"σ={sigma}, C={c}, 期望 σ~0.5, C~10", score


# ── T4: 带隙最大化 ───────────────────────────────────────────────

def _eval_bandgap_optim(output: str) -> tuple[bool, str, float]:
    """合金带隙 Eg(x) = 1.0 + 0.5*x - 0.1*x² (x: 掺杂浓度 0~5).
    最优 x=2.5, Eg_max=1.625 eV. 判 x 或 Eg."""
    x = _extract_number(output, r"(?:x|掺杂|doping|浓度)[^0-9]*(\d+\.?\d*)")
    eg = _extract_number(output, r"(?:Eg|带隙|band.?gap)[^0-9]*(\d+\.?\d*)")
    if eg is not None and _num_close(eg, 1.625, 0.05):
        return True, f"Eg={eg:.3f} eV (最优 ~1.625)", 1.0
    if x is not None and _num_close(x, 2.5, 0.2):
        return True, f"x={x:.2f} (最优 ~2.5)", 1.0
    if eg is not None and _num_close(eg, 1.625, 0.1):
        return False, f"Eg={eg:.3f}, 接近 1.625", 0.5
    return False, f"x={x}, Eg={eg}, 期望 x~2.5, Eg~1.625", 0.2


# ── T5: 0-1 背包问题 ─────────────────────────────────────────────

def _eval_knapsack(output: str) -> tuple[bool, str, float]:
    """物品 w=[2,3,4,5], v=[3,4,5,6], 容量 C=5. 最优: 选物品0+1, v=7."""
    val = _extract_number(output, r"(?:V|总价值|value|最优)[^0-9]*(\d+\.?\d*)")
    if val is None:
        nums = [float(x) for x in re.findall(r"\d+\.?\d*", output)]
        close = [n for n in nums if _num_close(n, 7, 0.5)]
        if close:
            val = close[0]
        else:
            return False, "未找到最优价值 (~7)", 0.0
    if _num_close(val, 7, 0.5):
        return True, f"V={val} (最优 ~7)", 1.0
    return False, f"V={val}, 期望 ~7", 0.3


# ── T6: 梯度下降 ─────────────────────────────────────────────────

def _eval_gradient_descent(output: str) -> tuple[bool, str, float]:
    """f(x)=x²+2x+1, f'(x)=2x+2, α=0.1, x0=5. 最小值 x=-1."""
    val = _extract_number(output, r"(?:x|最小值|min)[\s=:]*(-?\d+\.?\d*)")
    if val is None:
        nums = [float(x) for x in re.findall(r"[-+]?\d+\.?\d*", output)]
        close = [n for n in nums if _num_close(n, -1.0, 0.1)]
        if close:
            val = close[0]
        else:
            return False, "未找到最小值 x (~-1.0)", 0.0
    if _num_close(val, -1.0, 0.1):
        return True, f"x={val} (最优 ~-1.0)", 1.0
    return False, f"x={val}, 期望 ~-1.0", 0.3


# ── T7: Pareto 前沿 ──────────────────────────────────────────────

def _eval_pareto(output: str) -> tuple[bool, str, float]:
    """5 个解 (f1,f2): A(1,5) B(2,3) C(3,4) D(4,2) E(5,1).
    C 被 B 支配 (2<3, 3<4). Pareto 前沿: A,B,D,E = 4 个."""
    val = _extract_number(output, r"(?:N|个数|数量|count|解)[^0-9]*(\d+\.?\d*)")
    if val is None:
        nums = [float(x) for x in re.findall(r"\d+\.?\d*", output)]
        close = [n for n in nums if _num_close(n, 4, 0.5)]
        if close:
            val = close[0]
        else:
            return False, "未找到 Pareto 前沿个数 (~4)", 0.0
    if _num_close(val, 4, 0.5):
        return True, f"Pareto 前沿 {int(val)} 个 (期望 4)", 1.0
    return False, f"{int(val)} 个, 期望 4", 0.3


# ── T8: 蒙特卡洛 π 估算 ──────────────────────────────────────────

def _eval_monte_carlo_pi(output: str) -> tuple[bool, str, float]:
    """蒙特卡洛 N=10000 估算 π. 答案 ~3.14, 容差 ±0.3."""
    val = _extract_number(output, r"(?:π|pi|估算|estimate)[^0-9]*(\d+\.?\d*)")
    if val is None:
        nums = [float(x) for x in re.findall(r"[-+]?\d+\.?\d*", output)]
        close = [n for n in nums if _num_close(n, 3.14, 0.3)]
        if close:
            val = close[0]
        else:
            return False, "未找到 π 估算值 (~3.14)", 0.0
    if _num_close(val, 3.14159, 0.3):
        return True, f"π≈{val:.3f} (期望 ~3.14)", 1.0
    return False, f"π≈{val:.3f}, 期望 ~3.14", 0.3


def build_optim_tasks() -> list[BenchmarkTask]:
    """算法优化 benchmark: 8 题, 覆盖 LP/势能/超参/二次/背包/梯度/Pareto/MC."""
    return [
        BenchmarkTask(
            id="optim-alloy",
            category="optimization",
            prompt=(
                "合金硬度 H = 5x + 3y，其中 x 是铜含量，y 是锌含量。"
                "约束: x + y ≤ 10，x ≥ 0，y ≥ 0。"
                "求使硬度 H 最大的成分 x 和 y，以及最大硬度 H。"
            ),
            evaluator=_eval_alloy_optim,
            tags=["optimization", "linear", "alloy"],
            requires_api_key=True,
            reference="x=10, y=0, H=50 (LP: max 5x+3y, x+y<=10)",
        ),
        BenchmarkTask(
            id="optim-lattice",
            category="optimization",
            prompt=(
                "原子间势能用 Lennard-Jones 势描述: "
                "E(a) = 4ε[(σ/a)¹² - (σ/a)⁶]，ε=1.0，σ=1.0。"
                "求平衡晶格常数 a₀（使 E 最小）和最小能量 E_min。"
            ),
            evaluator=_eval_lattice_optim,
            tags=["optimization", "potential", "lattice"],
            requires_api_key=True,
            reference="a₀=2^(1/6)=1.1225, E_min=-1.0 (LJ 势能最小值)",
        ),
        BenchmarkTask(
            id="optim-krr",
            category="optimization",
            prompt=(
                "用 Kernel Ridge Regression 做回归，核函数 K(x,x')=exp(-||x-x'||²/(2σ²))。"
                "数据集 5 折交叉验证，网格搜索最优超参 σ ∈ {0.1, 0.3, 0.5, 0.7, 1.0} "
                "和 C ∈ {1, 5, 10, 50, 100}。"
                "最优组合是 σ=0.5, C=10。请验证这个结论并说明理由。"
            ),
            evaluator=_eval_krr_optim,
            tags=["optimization", "hyperparameter", "ml"],
            requires_api_key=True,
            reference="σ=0.5, C=10 (5折CV网格搜索最优)",
        ),
        BenchmarkTask(
            id="optim-bandgap",
            category="optimization",
            prompt=(
                "半导体合金带隙 Eg(x) = 1.0 + 0.5x - 0.1x²，x 是掺杂浓度 (0 ≤ x ≤ 5)。"
                "求使带隙最大的掺杂浓度 x 和最大带隙 Eg (eV)。"
            ),
            evaluator=_eval_bandgap_optim,
            tags=["optimization", "quadratic", "electronic"],
            requires_api_key=True,
            reference="x=2.5, Eg=1.625 eV (二次函数顶点)",
        ),
        BenchmarkTask(
            id="optim-knapsack",
            category="optimization",
            prompt=(
                "0-1 背包问题: 有 4 个物品，重量 w=[2, 3, 4, 5]，价值 v=[3, 4, 5, 6]。"
                "背包容量 C=5。每个物品最多选一次。"
                "求使总价值最大的选择方案，及最大总价值 V。"
            ),
            evaluator=_eval_knapsack,
            tags=["optimization", "dp", "combinatorial"],
            requires_api_key=True,
            reference="选物品0+1, V=7 (w=[2,3], v=[3,4], C=5)",
        ),
        BenchmarkTask(
            id="optim-gradient-descent",
            category="optimization",
            prompt=(
                "用梯度下降法优化 f(x) = x² + 2x + 1。"
                "导数 f'(x) = 2x + 2，学习率 α=0.1，初始点 x₀=5。"
                "迭代 100 步后，x 收敛到什么值？"
            ),
            evaluator=_eval_gradient_descent,
            tags=["optimization", "gradient", "continuous"],
            requires_api_key=True,
            reference="x=-1.0 (f(x)=x²+2x+1=(x+1)² 的最小值)",
        ),
        BenchmarkTask(
            id="optim-pareto",
            category="optimization",
            prompt=(
                "多目标优化问题，5 个候选解 (目标值越小越好):\n"
                "  A: (f1=1, f2=5)\n"
                "  B: (f1=2, f2=3)\n"
                "  C: (f1=3, f2=4)\n"
                "  D: (f1=4, f2=2)\n"
                "  E: (f1=5, f2=1)\n"
                "求 Pareto 前沿中有几个解？"
            ),
            evaluator=_eval_pareto,
            tags=["optimization", "multi_objective", "pareto"],
            requires_api_key=True,
            reference="Pareto 前沿 4 个: A,B,D,E (C 被 B 支配)",
        ),
        BenchmarkTask(
            id="optim-monte-carlo",
            category="optimization",
            prompt=(
                "用蒙特卡洛方法估算 π: 在单位正方形 [0,1]×[0,1] 内随机撒 N=10000 个点，"
                "统计落在四分之一圆 (x²+y²≤1) 内的点数 M，则 π ≈ 4M/N。"
                "请给出估算结果 π。"
            ),
            evaluator=_eval_monte_carlo_pi,
            tags=["optimization", "monte_carlo", "stochastic"],
            requires_api_key=True,
            reference="π ≈ 3.14 (N=10000 蒙特卡洛估算)",
        ),
    ]


# ── self-check ─────────────────────────────────────────────────

def _selfcheck() -> None:
    """验证 8 题: 正确答案 pass, 错误答案 fail."""
    tasks = build_optim_tasks()
    assert len(tasks) == 8, f"expected 8 optim tasks, got {len(tasks)}"

    # T1 合金: H=50, x=10, y=0
    t1 = tasks[0]
    r = t1.evaluate("x=10, y=0, H=50")
    assert r.passed, f"T1 correct should pass: {r.reason}"
    r = t1.evaluate("H=20")
    assert not r.passed, "T1 wrong should fail"

    # T2 晶格: a0=1.1225, E=-1
    t2 = tasks[1]
    r = t2.evaluate("a₀ = 1.1225, E_min = -1.0")
    assert r.passed, f"T2 correct should pass: {r.reason}"
    r = t2.evaluate("a=2.0")
    assert not r.passed, "T2 wrong should fail"

    # T3 KRR: σ=0.5, C=10
    t3 = tasks[2]
    r = t3.evaluate("σ=0.5, C=10")
    assert r.passed, f"T3 correct should pass: {r.reason}"
    r = t3.evaluate("σ=1.0, C=1")
    assert not r.passed, "T3 wrong should fail"

    # T4 带隙: x=2.5, Eg=1.625
    t4 = tasks[3]
    r = t4.evaluate("x=2.5, Eg=1.625 eV")
    assert r.passed, f"T4 correct should pass: {r.reason}"
    r = t4.evaluate("Eg=0.5")
    assert not r.passed, "T4 wrong should fail"

    # T5 背包: V=7
    t5 = tasks[4]
    r = t5.evaluate("最大总价值 V=7")
    assert r.passed, f"T5 correct should pass: {r.reason}"
    r = t5.evaluate("V=5")
    assert not r.passed, "T5 wrong should fail"

    # T6 梯度下降: x=-1
    t6 = tasks[5]
    r = t6.evaluate("x = -1.0")
    assert r.passed, f"T6 correct should pass: {r.reason}"
    r = t6.evaluate("x = 3.0")
    assert not r.passed, "T6 wrong should fail"

    # T7 Pareto: 4 个
    t7 = tasks[6]
    r = t7.evaluate("Pareto 前沿有 4 个解")
    assert r.passed, f"T7 correct should pass: {r.reason}"
    r = t7.evaluate("有 5 个")
    assert not r.passed, "T7 wrong should fail"

    # T8 蒙特卡洛: π≈3.14
    t8 = tasks[7]
    r = t8.evaluate("π ≈ 3.14")
    assert r.passed, f"T8 correct should pass: {r.reason}"
    r = t8.evaluate("π ≈ 2.0")
    assert not r.passed, "T8 wrong should fail"

    print(f"PASS: optim_bench ({len(tasks)} tasks)")


if __name__ == "__main__":
    _selfcheck()
