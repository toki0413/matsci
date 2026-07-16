"""PaperReproBench — 论文复现能力评测.

对标 PaperBench: 给一段论文描述/数据, 让 agent 复现关键结果.
不限于材料计算, 覆盖公式提取、数据拟合、代码重建、图表复现、ML 模型复现.

evaluator 用数值容差或代码结构判分, 不依赖 LLM judge.
"""

from __future__ import annotations

import re
from typing import Any

from .task import BenchmarkTask


def _extract_number(text: str, pattern: str) -> float | None:
    """从文本里提取第一个匹配 pattern 的数字."""
    m = re.search(pattern, text, re.IGNORECASE)
    if not m:
        return None
    try:
        return float(m.group(1))
    except (ValueError, IndexError):
        return None


def _num_close(value: float, expected: float, tol: float) -> bool:
    return abs(value - expected) <= tol


# ── T1: Hall-Petch 公式复现 ──────────────────────────────────────

def _eval_hall_petch(output: str) -> tuple[bool, str, float]:
    """论文描述 Hall-Petch: σ = σ0 + k/√d, σ0=50MPa, k=0.4MPa·√m.
    问 d=10μm 时 σ. 答案: 50 + 0.4/sqrt(10e-6) = 50 + 126.5 = 176.5 MPa.
    容差 ±15 MPa (sqrt 计算精度差异)."""
    val = _extract_number(output, r"(\d+\.?\d*)\s*MPa")
    if val is None:
        return False, "未找到 MPa 数值", 0.0
    if _num_close(val, 176.5, 15):
        return True, f"Hall-Petch σ={val:.1f} MPa (期望 ~176.5)", 1.0
    return False, f"σ={val:.1f} MPa, 期望 ~176.5 (±15)", 0.3


# ── T2: Arrhenius 活化能拟合 ─────────────────────────────────────

def _eval_arrhenius(output: str) -> tuple[bool, str, float]:
    """给 4 个温度-速率数据点, 让 agent 做 Arrhenius 拟合算活化能 Ea.
    ln(k) = ln(A) - Ea/(RT). 数据: T=300,400,500,600K, k=0.01,0.1,1,10.
    线性拟合 ln(k) vs 1/T, 斜率 = -Ea/R. Ea ≈ 23.0 kJ/mol."""
    # agent 可能输出 Ea, 活化能, activation energy 等
    val = _extract_number(output, r"(?:Ea|活化能|activation)[^0-9]*(\d+\.?\d*)")
    if val is None:
        # 找所有数字, 看有没有接近 23 的
        nums = [float(x) for x in re.findall(r"(\d+\.?\d*)", output)]
        close = [n for n in nums if _num_close(n, 23.0, 3.0)]
        if close:
            val = close[0]
        else:
            return False, "未找到活化能数值 (~23 kJ/mol)", 0.0
    if _num_close(val, 23.0, 3.0):
        return True, f"Ea={val:.1f} kJ/mol (期望 ~23)", 1.0
    return False, f"Ea={val:.1f}, 期望 ~23 (±3)", 0.3


# ── T3: Debye 比热代码重建 ───────────────────────────────────────

def _eval_debye_code(output: str) -> tuple[bool, str, float]:
    """论文给出 Debye 模型 Cv = 9NkB(T/θD)³ ∫₀^θD/T x⁴eˣ/(eˣ-1)² dx.
    让 agent 写 Python 函数 debye_cv(T, theta_D, N, kB).
    判: 函数定义 + 积分结构 + 关键变量."""
    has_def = "def debye_cv" in output or "def debye" in output
    has_integral = "quad" in output or "integrate" in output or "∫" in output or "trapz" in output
    has_theta = "theta_D" in output or "theta" in output or "θD" in output.lower()
    has_exp = "exp" in output
    score = sum([has_def, has_integral, has_theta, has_exp]) / 4.0
    if score >= 0.75:
        return True, f"Debye 代码结构完整 ({score*100:.0f}%)", score
    return False, f"Debye 代码缺失关键部分 ({score*100:.0f}%)", score


# ── T4: XRD Bragg 角计算 ─────────────────────────────────────────

def _eval_bragg(output: str) -> tuple[bool, str, float]:
    """论文: Si 立方晶系 a=5.43Å, (111) 面, λ=1.54Å (Cu Kα).
    Bragg: 2d sinθ = nλ, d = a/√(h²+k²+l²) = 5.43/√3 = 3.135Å.
    sinθ = λ/(2d) = 1.54/(2×3.135) = 0.2456, θ = 14.22°."""
    val = _extract_number(output, r"(\d+\.?\d*)\s*(?:°|度|deg)")
    if val is None:
        val = _extract_number(output, r"θ\s*[=:]\s*(\d+\.?\d*)")
    if val is None:
        return False, "未找到 Bragg 角 (°)", 0.0
    if _num_close(val, 14.22, 1.0):
        return True, f"θ={val:.2f}° (期望 ~14.22°)", 1.0
    return False, f"θ={val:.2f}°, 期望 ~14.22° (±1°)", 0.3


# ── T5: ML 线性回归模型复现 ──────────────────────────────────────

def _eval_linear_regression(output: str) -> tuple[bool, str, float]:
    """论文: 用线性回归 y = w·x + b 预测硬度. 数据: x=[1,2,3,4], y=[2,4,6,8].
    复现: w=2.0, b=0.0. 让 agent 算 w 和 b."""
    w = _extract_number(output, r"(?:w|斜率|slope)[^0-9]*(\d+\.?\d*)")
    b = _extract_number(output, r"(?:b|截距|intercept)[^0-9]*(\d+\.?\d*)")
    if w is None:
        return False, "未找到斜率 w (~2.0)", 0.0
    score = 0.0
    if _num_close(w, 2.0, 0.1):
        score += 0.6
    if b is not None and _num_close(b, 0.0, 0.5):
        score += 0.4
    elif b is None:
        score += 0.2  # 只找到 w, 部分分
    if score >= 0.6:
        return True, f"w={w}, b={b}", score
    return False, f"w={w}, b={b}, 期望 w=2.0, b=0.0", score


# ── T6: Curie-Weiss 磁化率 ───────────────────────────────────────

def _eval_curie_weiss(output: str) -> tuple[bool, str, float]:
    """χ = C/(T-θ), C=1.0 K·emu/mol, θ=50K, T=300K → χ=0.004 emu/mol."""
    val = _extract_number(output, r"(?:χ|chi|磁化率|susceptibility)[^0-9]*(\d+\.?\d*e?[-+]?\d*)")
    if val is None:
        # 找所有数字, 看有没有接近 0.004 的
        nums = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", output)]
        close = [n for n in nums if _num_close(n, 0.004, 0.001)]
        if close:
            val = close[0]
        else:
            return False, "未找到磁化率 (~0.004)", 0.0
    if _num_close(val, 0.004, 0.001):
        return True, f"χ={val} (期望 ~0.004)", 1.0
    return False, f"χ={val}, 期望 ~0.004", 0.3


# ── T7: 胡克定律应力应变 ─────────────────────────────────────────

def _eval_hooke(output: str) -> tuple[bool, str, float]:
    """σ = Eε, E=210 GPa, ε=0.002 → σ=0.42 GPa = 420 MPa."""
    val = _extract_number(output, r"(?:σ|stress|应力)[^0-9]*(\d+\.?\d*)\s*(?:MPa|GPa)?")
    if val is None:
        return False, "未找到应力数值", 0.0
    # 允许 GPa 或 MPa
    if _num_close(val, 420, 20) or _num_close(val, 0.42, 0.02):
        return True, f"σ={val} (期望 ~420 MPa 或 0.42 GPa)", 1.0
    return False, f"σ={val}, 期望 ~420 MPa", 0.3


# ── T8: 德拜温度 ─────────────────────────────────────────────────

def _eval_debye_temp(output: str) -> tuple[bool, str, float]:
    """θD = hνmax/kB, νmax=10 THz → θD ≈ 479.6 K."""
    val = _extract_number(output, r"(?:θD|theta_D|德拜温度|Debye)[^0-9]*(\d+\.?\d*)\s*K?")
    if val is None:
        nums = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+", output)]
        close = [n for n in nums if _num_close(n, 480, 15)]
        if close:
            val = close[0]
        else:
            return False, "未找到德拜温度 (~480 K)", 0.0
    if _num_close(val, 479.6, 15):
        return True, f"θD={val:.1f} K (期望 ~479.6)", 1.0
    return False, f"θD={val:.1f} K, 期望 ~480 K", 0.3


# ── T9: 费米能级 ─────────────────────────────────────────────────

def _eval_fermi_energy(output: str) -> tuple[bool, str, float]:
    """3D 自由电子气: EF = (ℏ²/2m)(3π²n)^(2/3), n=1e28 m⁻³ → EF ≈ 1.66 eV."""
    val = _extract_number(output, r"(?:EF|费米|Fermi)[^0-9]*(\d+\.?\d*)\s*(?:eV|J)?")
    if val is None:
        nums = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+", output)]
        close = [n for n in nums if _num_close(n, 1.66, 0.3)]
        if close:
            val = close[0]
        else:
            return False, "未找到费米能级 (~1.66 eV)", 0.0
    if _num_close(val, 1.66, 0.3):
        return True, f"EF={val:.2f} eV (期望 ~1.66)", 1.0
    return False, f"EF={val:.2f}, 期望 ~1.66 eV", 0.3


# ── T10: Nernst 方程 ─────────────────────────────────────────────

def _eval_nernst(output: str) -> tuple[bool, str, float]:
    """E = E0 - (RT/nF)lnQ, E0=1.1V, T=298K, n=2, Q=0.01 → E ≈ 1.159 V."""
    val = _extract_number(output, r"(?:E|电位|potential)[^0-9]* (\d+\.?\d*)\s*V")
    if val is None:
        nums = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+", output)]
        close = [n for n in nums if _num_close(n, 1.159, 0.03)]
        if close:
            val = close[0]
        else:
            return False, "未找到电位 (~1.159 V)", 0.0
    if _num_close(val, 1.159, 0.03):
        return True, f"E={val:.3f} V (期望 ~1.159)", 1.0
    return False, f"E={val:.3f} V, 期望 ~1.159 V", 0.3


def build_repro_tasks() -> list[BenchmarkTask]:
    """论文复现 benchmark: 10 题, 覆盖公式/拟合/代码/图表/ML/磁/热/电."""
    return [
        BenchmarkTask(
            id="repro-hall-petch",
            category="reproduction",
            prompt=(
                "一篇论文研究了低碳钢的屈服强度与晶粒尺寸的关系，提出 Hall-Petch 关系: "
                "σ = σ₀ + k/√d，其中 σ₀=50 MPa，k=0.4 MPa·√m。"
                "请计算当晶粒尺寸 d=10 μm 时的屈服强度 σ（MPa）。"
            ),
            evaluator=_eval_hall_petch,
            tags=["reproduction", "formula", "metallurgy"],
            requires_api_key=True,
            reference="σ = 50 + 0.4/sqrt(10e-6) = 176.5 MPa",
        ),
        BenchmarkTask(
            id="repro-arrhenius",
            category="reproduction",
            prompt=(
                "论文测量了某反应在不同温度下的速率常数:\n"
                "  T=300K, k=0.01 s⁻¹\n"
                "  T=400K, k=0.1 s⁻¹\n"
                "  T=500K, k=1.0 s⁻¹\n"
                "  T=600K, k=10.0 s⁻¹\n"
                "用 Arrhenius 方程 k = A·exp(-Ea/(RT)) 拟合，计算活化能 Ea (kJ/mol)。"
                "R=8.314 J/(mol·K)。"
            ),
            evaluator=_eval_arrhenius,
            tags=["reproduction", "fitting", "kinetics"],
            requires_api_key=True,
            reference="Ea ≈ 23.0 kJ/mol (ln(k) vs 1/T 线性拟合, 斜率=-Ea/R)",
        ),
        BenchmarkTask(
            id="repro-debye-code",
            category="reproduction",
            prompt=(
                "论文中用 Debye 模型计算晶格比热容:\n"
                "  Cv = 9Nk_B(T/θ_D)³ ∫₀^(θ_D/T) x⁴eˣ/(eˣ-1)² dx\n"
                "请用 Python 实现函数 debye_cv(T, theta_D, N, kB)，"
                "返回给定温度下的比热容。回复只包含代码块。"
            ),
            evaluator=_eval_debye_code,
            tags=["reproduction", "code", "thermal"],
            requires_api_key=True,
            is_code_task=True,
            reference="Python 函数 def debye_cv + scipy.integrate.quad + exp + theta_D",
        ),
        BenchmarkTask(
            id="repro-bragg",
            category="reproduction",
            prompt=(
                "论文用 XRD 表征了硅样品。硅为立方晶系，晶格常数 a=5.43 Å。"
                "使用 Cu Kα 辐射 (λ=1.54 Å)，计算 (111) 晶面的 Bragg 衍射角 θ（度）。"
                "Bragg 公式: 2d sinθ = nλ，n=1，d = a/√(h²+k²+l²)。"
            ),
            evaluator=_eval_bragg,
            tags=["reproduction", "diffraction", "crystallography"],
            requires_api_key=True,
            reference="d=3.135Å, sinθ=0.2456, θ=14.22°",
        ),
        BenchmarkTask(
            id="repro-linear-regression",
            category="reproduction",
            prompt=(
                "论文用线性回归模型 y = w·x + b 预测合金硬度。训练数据:\n"
                "  x=[1, 2, 3, 4] (成分比例)\n"
                "  y=[2, 4, 6, 8] (硬度 GPa)\n"
                "用最小二乘法计算回归系数 w 和 b。"
            ),
            evaluator=_eval_linear_regression,
            tags=["reproduction", "ml", "regression"],
            requires_api_key=True,
            reference="w=2.0, b=0.0",
        ),
        BenchmarkTask(
            id="repro-curie-weiss",
            category="reproduction",
            prompt=(
                "论文测量了某顺磁材料的磁化率，符合 Curie-Weiss 定律: "
                "χ = C/(T-θ)，其中 C=1.0 K·emu/mol，θ=50 K。"
                "计算 T=300 K 时的磁化率 χ (emu/mol)。"
            ),
            evaluator=_eval_curie_weiss,
            tags=["reproduction", "magnetic"],
            requires_api_key=True,
            reference="χ = 1.0/(300-50) = 0.004 emu/mol",
        ),
        BenchmarkTask(
            id="repro-hooke",
            category="reproduction",
            prompt=(
                "论文测量了钢的弹性形变，符合胡克定律 σ = Eε。"
                "弹性模量 E=210 GPa，应变 ε=0.002。"
                "计算应力 σ (MPa)。"
            ),
            evaluator=_eval_hooke,
            tags=["reproduction", "mechanical"],
            requires_api_key=True,
            reference="σ = 210*0.002 = 0.42 GPa = 420 MPa",
        ),
        BenchmarkTask(
            id="repro-debye-temp",
            category="reproduction",
            prompt=(
                "论文用德拜模型描述晶格振动，德拜温度 θD = hνmax/kB，"
                "其中 h=6.626×10⁻³⁴ J·s，kB=1.381×10⁻²³ J/K，"
                "最大声子频率 νmax=10 THz (10×10¹² Hz)。"
                "计算德拜温度 θD (K)。"
            ),
            evaluator=_eval_debye_temp,
            tags=["reproduction", "thermal", "phonon"],
            requires_api_key=True,
            reference="θD = 6.626e-34*10e12/1.381e-23 = 479.6 K",
        ),
        BenchmarkTask(
            id="repro-fermi-energy",
            category="reproduction",
            prompt=(
                "论文计算了铜的费米能级，用自由电子气模型: "
                "EF = (ℏ²/2m)(3π²n)^(2/3)，"
                "其中 ℏ=1.055×10⁻³⁴ J·s，m=9.11×10⁻³¹ kg，"
                "电子浓度 n=1.0×10²⁸ m⁻³。"
                "计算费米能级 EF (eV)。1 eV = 1.602×10⁻¹⁹ J。"
            ),
            evaluator=_eval_fermi_energy,
            tags=["reproduction", "electronic"],
            requires_api_key=True,
            reference="EF ≈ 1.66 eV",
        ),
        BenchmarkTask(
            id="repro-nernst",
            category="reproduction",
            prompt=(
                "论文研究了电化学反应，电池电动势符合 Nernst 方程: "
                "E = E₀ - (RT/nF)lnQ，"
                "其中 E₀=1.1 V，T=298 K，n=2，"
                "R=8.314 J/(mol·K)，F=96485 C/mol，Q=0.01。"
                "计算电池电动势 E (V)。"
            ),
            evaluator=_eval_nernst,
            tags=["reproduction", "electrochemistry"],
            requires_api_key=True,
            reference="E = 1.1 + (8.314*298/(2*96485))*ln(100) = 1.159 V",
        ),
    ]


# ── self-check ─────────────────────────────────────────────────

def _selfcheck() -> None:
    """验证 10 题: 正确答案 pass, 错误答案 fail."""
    tasks = build_repro_tasks()
    assert len(tasks) == 10, f"expected 10 repro tasks, got {len(tasks)}"

    # T1 Hall-Petch: 正确 ~176.5
    t1 = tasks[0]
    r = t1.evaluate("屈服强度 σ = 176.5 MPa")
    assert r.passed, f"T1 correct should pass: {r.reason}"

    r = t1.evaluate("σ = 50 MPa")
    assert not r.passed, "T1 wrong should fail"

    # T2 Arrhenius: 正确 ~23
    t2 = tasks[1]
    r = t2.evaluate("活化能 Ea = 23.0 kJ/mol")
    assert r.passed, f"T2 correct should pass: {r.reason}"

    r = t2.evaluate("Ea = 100 kJ/mol")
    assert not r.passed, "T2 wrong should fail"

    # T3 Debye code: 正确结构
    t3 = tasks[2]
    r = t3.evaluate(
        "def debye_cv(T, theta_D, N, kB):\n"
        "    from scipy.integrate import quad\n"
        "    x_max = theta_D / T\n"
        "    integrand = lambda x: x**4 * np.exp(x) / (np.exp(x)-1)**2\n"
        "    integral, _ = quad(integrand, 0, x_max)\n"
        "    return 9 * N * kB * (T/theta_D)**3 * integral"
    )
    assert r.passed, f"T3 correct should pass: {r.reason}"

    r = t3.evaluate("print('hello')")
    assert not r.passed, "T3 wrong should fail"

    # T4 Bragg: 正确 ~14.22
    t4 = tasks[3]
    r = t4.evaluate("θ = 14.22°")
    assert r.passed, f"T4 correct should pass: {r.reason}"

    r = t4.evaluate("θ = 30°")
    assert not r.passed, "T4 wrong should fail"

    # T5 Linear regression: 正确 w=2, b=0
    t5 = tasks[4]
    r = t5.evaluate("w = 2.0, b = 0.0")
    assert r.passed, f"T5 correct should pass: {r.reason}"

    r = t5.evaluate("w = 5.0, b = 1.0")
    assert not r.passed, "T5 wrong should fail"

    # T6 Curie-Weiss: 正确 ~0.004
    t6 = tasks[5]
    r = t6.evaluate("磁化率 χ = 0.004 emu/mol")
    assert r.passed, f"T6 correct should pass: {r.reason}"
    r = t6.evaluate("χ = 0.1")
    assert not r.passed, "T6 wrong should fail"

    # T7 Hooke: 正确 ~420 MPa
    t7 = tasks[6]
    r = t7.evaluate("应力 σ = 420 MPa")
    assert r.passed, f"T7 correct should pass: {r.reason}"
    r = t7.evaluate("σ = 100 MPa")
    assert not r.passed, "T7 wrong should fail"

    # T8 Debye temp: 正确 ~480 K
    t8 = tasks[7]
    r = t8.evaluate("德拜温度 θD = 479.6 K")
    assert r.passed, f"T8 correct should pass: {r.reason}"
    r = t8.evaluate("θD = 200 K")
    assert not r.passed, "T8 wrong should fail"

    # T9 Fermi energy: 正确 ~1.66 eV
    t9 = tasks[8]
    r = t9.evaluate("费米能级 EF = 1.66 eV")
    assert r.passed, f"T9 correct should pass: {r.reason}"
    r = t9.evaluate("EF = 5.0 eV")
    assert not r.passed, "T9 wrong should fail"

    # T10 Nernst: 正确 ~1.159 V
    t10 = tasks[9]
    r = t10.evaluate("电池电动势 E = 1.159 V")
    assert r.passed, f"T10 correct should pass: {r.reason}"
    r = t10.evaluate("E = 1.0 V")
    assert not r.passed, "T10 wrong should fail"

    print(f"PASS: paper_repro_bench ({len(tasks)} tasks)")


if __name__ == "__main__":
    _selfcheck()
