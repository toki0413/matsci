"""NatureBench-mini — Nature 系列论文复现能力评测.

受 Frontis.AI NatureBench (https://github.com/AI4ScienceWK/NatureBench) 启发:
90 个 Nature/Science 论文复现任务, 最强 agent 仅 17.8% 超越 SOTA.

核心设计 (从源码学到):
1. best_score: 多次提交取最优, 不是单次 best. evaluator 闭包内保留状态.
   NatureBench eval_service.py 的 ScoreTracker 按 (task, batch) 隔离.
2. SOTA 对标 (三段容差):
   |val - sota| <= tol   -> 1.0, 达到 SOTA
   <= 2*tol              -> 0.5, 接近 SOTA
   更远                   -> 0.3, 远离 SOTA
3. cheating 检测交给 ValidityJudge (huginn.validation.grader), 这里只做数值判分.
4. log_space: 扩散系数 / 电导率等跨数量级指标在对数空间判分.

ponytail: 不上 Docker/HTTP. 10 题覆盖 Nature 系常见计算任务:
DFT 带隙 / MD 扩散 / 紧束缚费米速度 / HEA 力学 / 热电 / 拓扑 / MOF / 超导 / 离子电导.

升级路径: 接 HuggingFace snapshot_download 跑真实 NatureBench task package.
ceiling: 10 题自制无法覆盖 90 题真实论文; evaluator 数值判分不如 NatureBench 的
        evaluation/ 脚本严格; best_score 是进程内状态, 跨 session 不持久.
"""
from __future__ import annotations

import math
import re
from typing import Any

from .task import BenchmarkTask


# ── 工具 ──────────────────────────────────────────────────────

_SCINUM_RE = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"


def _num_close(value: float, expected: float, tol: float) -> bool:
    return abs(value - expected) <= tol


def _extract_number(text: str, pattern: str | None = None) -> float | None:
    """从文本里提取第一个匹配 pattern 的数字; pattern=None 时取所有数字的第一个."""
    if pattern:
        m = re.search(pattern, text, re.IGNORECASE)
        if not m:
            return None
        try:
            return float(m.group(1))
        except (ValueError, IndexError):
            return None
    nums = re.findall(_SCINUM_RE, text)
    if not nums:
        return None
    try:
        return float(nums[0])
    except ValueError:
        return None


def _extract_sci(text: str, prefix_regex: str) -> float | None:
    """提取前缀后的科学计数法数值 (含 m×10^n 和 mEn 两种写法)."""
    m = re.search(prefix_regex + r"\s*(" + _SCINUM_RE + r")", text, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    # 兼容 "1.2×10⁻¹²" / "1.2e-12"
    m = re.search(
        prefix_regex + r"\s*(\d+\.?\d*)\s*[×x*]\s*10\^?([-+]?\d+)",
        text, re.IGNORECASE,
    )
    if m:
        try:
            return float(m.group(1)) * (10 ** int(m.group(2)))
        except ValueError:
            pass
    return None


def _make_best_score_eval(
    sota: float,
    tol: float,
    unit: str,
    key_hint: str,
    log_space: bool = False,
):
    """构造一个带 best_score 跟踪的 evaluator.

    NatureBench eval_service.py 启发: best_score 只保留最高分.
    这里用 dict 实现闭包内可变状态.

    判分 (log_space=False 时线性, True 时对数):
      |val - sota| <= tol     -> 1.0, 达到 SOTA
      <= 2*tol               -> 0.5, 接近 SOTA
      更远                    -> 0.3, 远离 SOTA
    log_space: tol 单位是 log10 (如 tol=0.5 表示 10^0.5 倍以内)
    """
    state: dict[str, Any] = {"best": None, "n_calls": 0}

    def evaluate(output: str) -> tuple[bool, str, float]:
        val = _extract_sci(output, key_hint) or _extract_number(output, key_hint)
        if val is None:
            # 退而求其次: 找所有数字, 看有没有接近 SOTA
            nums = [float(x) for x in re.findall(_SCINUM_RE, output)]
            close = []
            for n in nums:
                if log_space and n > 0 and sota > 0:
                    if abs(math.log10(n) - math.log10(sota)) <= tol * 2:
                        close.append(n)
                elif _num_close(n, sota, tol * 2):
                    close.append(n)
            if close:
                val = close[0]
            else:
                return False, f"未找到 {unit} 数值 (~{sota})", 0.0

        state["n_calls"] += 1
        prev_best = state["best"]

        # 计算距离
        if log_space:
            if val <= 0 or sota <= 0:
                dist = float("inf")
            else:
                dist = abs(math.log10(val) - math.log10(sota))
        else:
            dist = abs(val - sota)

        # 三段容差判分
        if dist <= tol:
            score = 1.0
            passed = True
            msg = f"val={val} {unit} (达到 SOTA ~{sota}±{tol})"
        elif dist <= tol * 2:
            score = 0.5
            passed = False
            msg = f"val={val} {unit}, 接近 SOTA ~{sota}"
        else:
            score = 0.3
            passed = False
            msg = f"val={val} {unit}, 期望 ~{sota} (±{tol})"

        # best_score: 取最高 score
        if prev_best is None or score > prev_best[1]:
            state["best"] = (val, score)

        return passed, msg, score

    evaluate.state = state  # type: ignore[attr-defined]
    return evaluate


# ── 10 题 Nature 系列论文复现 ────────────────────────────────────
# 模块级单例: 每个 evaluator 只构造一次, 闭包内 state 跨调用保留
# (NatureBench eval_service.py 的 ScoreTracker 也是进程级单例)

_eval_perovskite_gap = _make_best_score_eval(
    sota=1.55, tol=0.1, unit="eV",
    key_hint=r"(?:Eg|带隙|band.?gap)",
)
"""CH3NH3PbI3 钙钛矿带隙 ~1.55 eV (Nature 2014 perovskite solar cell)."""

_eval_mos2_gap = _make_best_score_eval(
    sota=1.8, tol=0.1, unit="eV",
    key_hint=r"(?:Eg|带隙|band.?gap)",
)
"""单层 MoS2 直接带隙 ~1.8 eV (Nature 2018 TMD)."""

_eval_li_diffusion = _make_best_score_eval(
    sota=1e-12, tol=0.5, unit="m²/s",
    key_hint=r"(?:D|扩散|diffusion)",
    log_space=True,
)
"""LiCoO2 锂离子扩散系数 D ~ 1e-12 m²/s (Nature Materials 2017)."""

_eval_graphene_fermi = _make_best_score_eval(
    sota=1e6, tol=0.3, unit="m/s",
    key_hint=r"(?:vF|费米速度|Fermi\s*velocity)",
    log_space=True,
)
"""石墨烯费米速度 vF ~ 1e6 m/s (Nature Physics 2019)."""

_eval_heas_strength = _make_best_score_eval(
    sota=1.0, tol=0.1, unit="GPa",
    key_hint=r"(?:σ|sigma|屈服|yield|strength)",
)
"""CrMnFeCoNi 高熵合金屈服强度 σ_y ~ 1.0 GPa (Nature 2020)."""

_eval_snse_zt = _make_best_score_eval(
    sota=2.6, tol=0.3, unit="",
    key_hint=r"(?:ZT|优值|figure.?of.?merit)",
)
"""SnSe 热电优值 ZT ~ 2.6 (Nature Materials 2016)."""

_eval_bi2se3_dirac = _make_best_score_eval(
    sota=-0.35, tol=0.1, unit="eV",
    key_hint=r"(?:Dirac|狄拉克)",
)
"""Bi2Se3 拓扑绝缘体 Dirac point 能量 ~ -0.35 eV (Nature 2017)."""

_eval_mof_co2 = _make_best_score_eval(
    sota=3.0, tol=0.3, unit="mmol/g",
    key_hint=r"(?:CO2|二氧化碳|吸附|adsorption)",
)
"""MOF CO2 吸附量 ~ 3.0 mmol/g (Nature 2019)."""

_eval_h3s_tc = _make_best_score_eval(
    sota=203.0, tol=10.0, unit="K",
    key_hint=r"(?:Tc|Tc|临界温度|critical)",
)
"""H3S 超导临界温度 Tc ~ 203 K (Nature Physics 2015)."""

_eval_li10gep2s12 = _make_best_score_eval(
    sota=1e-2, tol=0.5, unit="S/cm",
    key_hint=r"(?:σ|sigma|电导率|conductivity)",
    log_space=True,
)
"""Li10GeP2S12 固态电解质电导率 σ ~ 1e-2 S/cm (Nature 2021)."""


# ── 构造 BenchmarkTask 列表 ────────────────────────────────────

def build_naturebench_tasks() -> list[BenchmarkTask]:
    """NatureBench-mini: 10 题 Nature 系论文复现.

    每题对应一个真实的 Nature/Science/Nature Materials/Nature Physics 论文,
    evaluator 判 agent 输出是否接近论文 SOTA. best_score 跟踪: 多次提交取最优
    (NatureBench eval_service.py 的核心设计).

    cheating 检测 (硬编码值/从输入反推等) 交给 ValidityJudge 在 _validate 链做,
    这里只做数值判分, 职责单一.
    """
    return [
        BenchmarkTask(
            id="nb-perovskite-gap",
            category="naturebench",
            prompt=(
                "Nature 2014 论文报道了 CH3NH3PbI3 钙钛矿太阳能电池效率突破 15%, "
                "其带隙 Eg ≈ 1.55 eV。请用第一性原理方法计算 CH3NH3PbI3 的带隙 (eV), "
                "并对比论文 SOTA。"
            ),
            evaluator=_eval_perovskite_gap,
            tags=["naturebench", "dft", "electronic", "perovskite"],
            requires_api_key=True,
            reference="Eg ≈ 1.55 eV (Nature 2014 perovskite solar cell)",
        ),
        BenchmarkTask(
            id="nb-mos2-gap",
            category="naturebench",
            prompt=(
                "Nature 2018 论文报道了单层 MoS2 从间接带隙转为直接带隙 (~1.8 eV), "
                "用于光电器件。请用 DFT 计算单层 MoS2 的带隙 (eV), 对比论文 SOTA。"
            ),
            evaluator=_eval_mos2_gap,
            tags=["naturebench", "dft", "2d", "tmd"],
            requires_api_key=True,
            reference="Eg ≈ 1.8 eV (Nature 2018 MoS2 TMD)",
        ),
        BenchmarkTask(
            id="nb-li-diffusion",
            category="naturebench",
            prompt=(
                "Nature Materials 2017 论文测量了 LiCoO2 中锂离子扩散系数 "
                "D ≈ 1×10⁻¹² m²/s。请用分子动力学或 NEB 方法计算 LiCoO2 的锂离子 "
                "扩散系数 (m²/s), 对比论文 SOTA。"
            ),
            evaluator=_eval_li_diffusion,
            tags=["naturebench", "md", "diffusion", "battery"],
            requires_api_key=True,
            reference="D ≈ 1e-12 m²/s (Nature Materials 2017)",
        ),
        BenchmarkTask(
            id="nb-graphene-fermi",
            category="naturebench",
            prompt=(
                "Nature Physics 2019 论文测量了石墨烯费米速度 vF ≈ 1×10⁶ m/s。"
                "请用紧束缚或 DFT 计算石墨烯的费米速度 (m/s), 对比论文 SOTA。"
            ),
            evaluator=_eval_graphene_fermi,
            tags=["naturebench", "tight_binding", "electronic", "2d"],
            requires_api_key=True,
            reference="vF ≈ 1e6 m/s (Nature Physics 2019 graphene)",
        ),
        BenchmarkTask(
            id="nb-heas-strength",
            category="naturebench",
            prompt=(
                "Nature 2020 论文报道了 CrMnFeCoNi 高熵合金在低温下屈服强度 "
                "σ_y ≈ 1.0 GPa。请用分子动力学计算 CrMnFeCoNi 在 77K 的屈服强度 "
                "(GPa), 对比论文 SOTA。"
            ),
            evaluator=_eval_heas_strength,
            tags=["naturebench", "md", "mechanical", "hea"],
            requires_api_key=True,
            reference="σ_y ≈ 1.0 GPa (Nature 2020 CrMnFeCoNi HEA)",
        ),
        BenchmarkTask(
            id="nb-snse-zt",
            category="naturebench",
            prompt=(
                "Nature Materials 2016 论文报道了 SnSe 单晶在 923K 时热电优值 "
                "ZT ≈ 2.6, 创历史新高。请用玻尔兹曼输运方程计算 SnSe 的 ZT, "
                "对比论文 SOTA。"
            ),
            evaluator=_eval_snse_zt,
            tags=["naturebench", "transport", "thermoelectric"],
            requires_api_key=True,
            reference="ZT ≈ 2.6 (Nature Materials 2016 SnSe)",
        ),
        BenchmarkTask(
            id="nb-bi2se3-dirac",
            category="naturebench",
            prompt=(
                "Nature 2017 论文用 ARPES 测得 Bi2Se3 拓扑绝缘体 Dirac point "
                "位于费米面以下 ~-0.35 eV。请用 DFT 计算 Bi2Se3 的 Dirac point "
                "能量 (eV), 对比论文 SOTA。"
            ),
            evaluator=_eval_bi2se3_dirac,
            tags=["naturebench", "dft", "topological", "surface_state"],
            requires_api_key=True,
            reference="Dirac point ≈ -0.35 eV (Nature 2017 Bi2Se3)",
        ),
        BenchmarkTask(
            id="nb-mof-co2",
            category="naturebench",
            prompt=(
                "Nature 2019 论文报道了某 MOF 在 298K/1bar 下 CO2 吸附量 "
                "~3.0 mmol/g。请用 GCMC 或 DFT 计算该 MOF 的 CO2 吸附量 "
                "(mmol/g), 对比论文 SOTA。"
            ),
            evaluator=_eval_mof_co2,
            tags=["naturebench", "gcmc", "porous", "adsorption"],
            requires_api_key=True,
            reference="~3.0 mmol/g (Nature 2019 MOF CO2 capture)",
        ),
        BenchmarkTask(
            id="nb-h3s-tc",
            category="naturebench",
            prompt=(
                "Nature Physics 2015 论文报道了 H3S 在 155 GPa 下超导临界温度 "
                "Tc ≈ 203 K, 创高压超导纪录。请用 DFT + 声子谱计算 H3S 的 Tc (K), "
                "对比论文 SOTA。"
            ),
            evaluator=_eval_h3s_tc,
            tags=["naturebench", "dft", "phonon", "superconductivity"],
            requires_api_key=True,
            reference="Tc ≈ 203 K (Nature Physics 2015 H3S)",
        ),
        BenchmarkTask(
            id="nb-li10gep2s12",
            category="naturebench",
            prompt=(
                "Nature 2021 论文报道了 Li10GeP2S12 固态电解质室温离子电导率 "
                "σ ≈ 1×10⁻² S/cm。请用 AIMD 计算 Li10GeP2S12 的离子电导率 "
                "(S/cm), 对比论文 SOTA。"
            ),
            evaluator=_eval_li10gep2s12,
            tags=["naturebench", "aimd", "battery", "ionic"],
            requires_api_key=True,
            reference="σ ≈ 1e-2 S/cm (Nature 2021 LGPS)",
        ),
    ]


# ── self-check ─────────────────────────────────────────────────

def _selfcheck() -> None:
    """验证 10 题: 正确答案 pass, 错误答案 fail, best_score 跟踪."""
    tasks = build_naturebench_tasks()
    assert len(tasks) == 10, f"expected 10 naturebench tasks, got {len(tasks)}"

    # T1 Perovskite Eg: 正确 ~1.55
    t1 = tasks[0]
    r = t1.evaluate("钙钛矿带隙 Eg = 1.55 eV")
    assert r.passed, f"T1 correct should pass: {r.reason}"
    assert r.score == 1.0
    r = t1.evaluate("Eg = 2.0 eV")
    assert not r.passed, f"T1 wrong should fail: {r.reason}"

    # T2 MoS2 Eg: 正确 ~1.8
    t2 = tasks[1]
    r = t2.evaluate("带隙 Eg = 1.80 eV")
    assert r.passed, f"T2 correct should pass: {r.reason}"

    # T3 Li diffusion: 科学计数法
    t3 = tasks[2]
    r = t3.evaluate("扩散系数 D = 1.2e-12 m²/s")
    assert r.passed, f"T3 correct should pass: {r.reason}"
    r = t3.evaluate("D = 1e-9 m²/s")
    assert not r.passed, f"T3 wrong should fail: {r.reason}"

    # T4 Graphene vF: 1e6 m/s
    t4 = tasks[3]
    r = t4.evaluate("费米速度 vF = 1.0e6 m/s")
    assert r.passed, f"T4 correct should pass: {r.reason}"

    # T5 HEA strength: ~1.0 GPa
    t5 = tasks[4]
    r = t5.evaluate("屈服强度 σ = 1.0 GPa")
    assert r.passed, f"T5 correct should pass: {r.reason}"
    r = t5.evaluate("σ = 0.3 GPa")
    assert not r.passed, "T5 wrong should fail"

    # T6 SnSe ZT: ~2.6
    t6 = tasks[5]
    r = t6.evaluate("ZT = 2.6")
    assert r.passed, f"T6 correct should pass: {r.reason}"

    # T7 Bi2Se3 Dirac: ~-0.35
    t7 = tasks[6]
    r = t7.evaluate("Dirac point = -0.35 eV")
    assert r.passed, f"T7 correct should pass: {r.reason}"
    r = t7.evaluate("Dirac = -1.0 eV")
    assert not r.passed, "T7 wrong should fail"

    # T8 MOF CO2: ~3.0
    t8 = tasks[7]
    r = t8.evaluate("CO2 吸附量 = 3.0 mmol/g")
    assert r.passed, f"T8 correct should pass: {r.reason}"

    # T9 H3S Tc: ~203 K
    t9 = tasks[8]
    r = t9.evaluate("Tc = 203 K")
    assert r.passed, f"T9 correct should pass: {r.reason}"
    r = t9.evaluate("Tc = 100 K")
    assert not r.passed, "T9 wrong should fail"

    # T10 LGPS σ: ~1e-2
    t10 = tasks[9]
    r = t10.evaluate("电导率 σ = 1.0e-2 S/cm")
    assert r.passed, f"T10 correct should pass: {r.reason}"

    # best_score 跟踪测试: 用独立实例避免模块级单例污染
    ev = _make_best_score_eval(
        sota=1.55, tol=0.1, unit="eV",
        key_hint=r"(?:Eg|带隙|band.?gap)",
    )
    ev("Eg = 1.55 eV")  # 1.0
    ev("Eg = 1.40 eV")  # 0.5 (接近但不达到)
    state = ev.state
    assert state["n_calls"] == 2, f"expected 2 calls, got {state['n_calls']}"
    # best 保留 score=1.0 那次 (1.55, 1.0)
    assert state["best"][0] == 1.55, f"best val should be 1.55, got {state['best'][0]}"
    assert state["best"][1] == 1.0, f"best score should be 1.0, got {state['best'][1]}"

    print(f"PASS: naturebench ({len(tasks)} tasks, best_score tracking verified)")


if __name__ == "__main__":
    _selfcheck()
