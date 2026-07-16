"""参数化题目生成器 — 批量生成 1100+ 道 benchmark 题.

5 个生成器:
  gen_knowledge:  元素性质/物理常数/单位换算 (350 题)
  gen_physics:    Bragg/Hall-Petch/胡克/热膨胀/磁化率 (300 题)
  gen_repro:      Arrhenius/回归/Nernst/德拜温度 (200 题)
  gen_optim:      LP/背包/梯度/二次优化 (200 题)
  gen_chemistry:  摩尔质量/化学计量/反应平衡 (100 题)

用固定随机种子保证可复现. evaluator 用闭包捕获参数.
"""

from __future__ import annotations

import math
import random
import re
from typing import Any

from .task import BenchmarkTask

_RNG = random.Random(42)  # 固定种子, 保证可复现


def _extract_nums(text: str) -> list[float]:
    """从文本提取所有数值."""
    return [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)]


def _num_close(val: float, expected: float, tol: float) -> bool:
    return abs(val - expected) <= tol


def _eval_num(expected: float, tol: float):
    """生成一个 '找最接近 expected 的数值' evaluator."""
    def evaluate(output: str) -> tuple[bool, str, float]:
        nums = _extract_nums(output)
        if not nums:
            return False, f"未找到数值 (期望 {expected})", 0.0
        for n in nums:
            if _num_close(n, expected, tol):
                return True, f"got {n} (期望 {expected}±{tol})", 1.0
        return False, f"got {nums[0]}, 期望 {expected}±{tol}", 0.3
    return evaluate


def _eval_keyword(keyword: str, expected: str):
    """生成关键词匹配 evaluator."""
    def evaluate(output: str) -> tuple[bool, str, float]:
        if keyword.lower() in output.lower():
            return True, f"找到 {keyword}", 1.0
        return False, f"未找到 {keyword}, 期望 {expected}", 0.0
    return evaluate


# ── 元素性质数据表 (30 个常见元素) ──────────────────────────────

ELEMENTS = [
    # (符号, 中文名, 英文名, 原子序数, 密度 g/cm³, 晶体结构, 带隙 eV 或 None)
    ("H",  "氢",  "Hydrogen",  1,  0.00009, "gas", None),
    ("He", "氦",  "Helium",    2,  0.00018, "gas", None),
    ("Li", "锂",  "Lithium",   3,  0.534,   "BCC", None),
    ("Be", "铍",  "Beryllium", 4,  1.848,   "HCP", None),
    ("B",  "硼",  "Boron",     5,  2.34,    "rhombohedral", None),
    ("C",  "碳",  "Carbon",    6,  2.27,    "graphite", None),
    ("N",  "氮",  "Nitrogen",  7,  0.00125, "gas", None),
    ("O",  "氧",  "Oxygen",    8,  0.00143, "gas", None),
    ("F",  "氟",  "Fluorine",  9,  0.00170, "gas", None),
    ("Ne", "氖",  "Neon",     10,  0.00090, "gas", None),
    ("Na", "钠",  "Sodium",   11,  0.971,   "BCC", None),
    ("Mg", "镁",  "Magnesium",12,  1.738,   "HCP", None),
    ("Al", "铝",  "Aluminum", 13,  2.70,    "FCC", None),
    ("Si", "硅",  "Silicon",  14,  2.33,    "diamond", 1.12),
    ("P",  "磷",  "Phosphorus",15, 1.82,    "orthorhombic", None),
    ("S",  "硫",  "Sulfur",   16,  2.07,    "orthorhombic", None),
    ("Cl", "氯",  "Chlorine", 17,  0.00321, "gas", None),
    ("Ar", "氩",  "Argon",    18,  0.00178, "gas", None),
    ("K",  "钾",  "Potassium",19,  0.862,   "BCC", None),
    ("Ca", "钙",  "Calcium",  20,  1.55,    "FCC", None),
    ("Ti", "钛",  "Titanium", 22,  4.506,   "HCP", None),
    ("V",  "钒",  "Vanadium", 23,  6.0,     "BCC", None),
    ("Cr", "铬",  "Chromium", 24,  7.19,    "BCC", None),
    ("Mn", "锰",  "Manganese",25,  7.21,    "BCC", None),
    ("Fe", "铁",  "Iron",     26,  7.874,   "BCC", None),
    ("Co", "钴",  "Cobalt",   27,  8.90,    "HCP", None),
    ("Ni", "镍",  "Nickel",   28,  8.908,   "FCC", None),
    ("Cu", "铜",  "Copper",   29,  8.96,    "FCC", None),
    ("Zn", "锌",  "Zinc",     30,  7.134,   "HCP", None),
    ("Ge", "锗",  "Germanium",32,  5.323,   "diamond", 0.67),
]

PHYSICAL_CONSTANTS = [
    ("光速 c", 2.998e8, "m/s", 0.001e8),
    ("普朗克常数 h", 6.626e-34, "J·s", 0.01e-34),
    ("约化普朗克常数 ℏ", 1.055e-34, "J·s", 0.01e-34),
    ("玻尔兹曼常数 kB", 1.381e-23, "J/K", 0.01e-23),
    ("阿伏伽德罗常数 NA", 6.022e23, "mol⁻¹", 0.01e23),
    ("基本电荷 e", 1.602e-19, "C", 0.01e-19),
    ("真空介电常数 ε₀", 8.854e-12, "F/m", 0.01e-12),
    ("真空磁导率 μ₀", 1.257e-6, "H/m", 0.01e-6),
    ("电子质量 me", 9.109e-31, "kg", 0.01e-31),
    ("质子质量 mp", 1.673e-27, "kg", 0.01e-27),
    ("中子质量 mn", 1.675e-27, "kg", 0.01e-27),
    ("万有引力常数 G", 6.674e-11, "N·m²/kg²", 0.01e-11),
    ("玻尔半径 a₀", 5.292e-11, "m", 0.01e-11),
    ("里德伯常数 R∞", 1.097e7, "m⁻¹", 0.01e7),
    ("法拉第常数 F", 96485, "C/mol", 1),
    ("摩尔气体常数 R", 8.314, "J/(mol·K)", 0.01),
    ("斯特藩-玻尔兹曼常数 σ", 5.670e-8, "W/(m²·K⁴)", 0.01