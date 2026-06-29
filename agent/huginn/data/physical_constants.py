"""CODATA 2018 物理常数库.

统一管理物理计算里常用的基本常数, 带单位、不确定度和来源.
所有工具引用同一份, 避免到处硬编码值不一致.

值来自 NIST CODATA 2018 推荐值. 2019 SI 重定义后 c / h / e / k_B / N_A
是精确定义的 (不确定度 0), 其余常数有测量不确定度.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


# ---------------------------------------------------------------------------
# 数据类
# ---------------------------------------------------------------------------


Category = Literal["fundamental", "electromagnetic", "atomic", "thermodynamic"]


@dataclass(frozen=True)
class PhysicalConstant:
    """单个物理常数的元数据.

    uncertainty 是标准不确定度 (1 sigma), 精确常数填 0.0.
    """

    name: str
    symbol: str
    value: float
    unit: str
    uncertainty: float = 0.0
    source: str = "CODATA 2018"
    category: Category = "fundamental"

    def __repr__(self) -> str:
        # 方便调试, 顺手带上单位
        return (
            f"PhysicalConstant({self.symbol}={self.value!r} {self.unit}, "
            f"unc={self.uncertainty}, src={self.source!r})"
        )


# ---------------------------------------------------------------------------
# 常数表. symbol 作为 key.
# ---------------------------------------------------------------------------

CONSTANTS: dict[str, PhysicalConstant] = {
    # --- fundamental: SI 定义常数 + 时空基本量 ---
    "c": PhysicalConstant(
        name="speed of light in vacuum",
        symbol="c",
        value=299792458.0,
        unit="m s^-1",
        uncertainty=0.0,
        category="fundamental",
    ),
    "G": PhysicalConstant(
        name="Newtonian constant of gravitation",
        symbol="G",
        value=6.67430e-11,
        unit="m^3 kg^-1 s^-2",
        uncertainty=1.5e-15,
        category="fundamental",
    ),
    "h": PhysicalConstant(
        name="Planck constant",
        symbol="h",
        value=6.62607015e-34,
        unit="J Hz^-1",
        uncertainty=0.0,  # SI 2019 精确定义
        category="fundamental",
    ),
    "hbar": PhysicalConstant(
        name="reduced Planck constant",
        symbol="ℏ",
        value=1.054571817e-34,
        unit="J s",
        uncertainty=0.0,  # 由 h/(2π) 推得, h 精确所以也精确
        category="fundamental",
    ),
    "alpha": PhysicalConstant(
        name="fine-structure constant",
        symbol="α",
        value=7.2973525693e-3,
        unit="dimensionless",
        uncertainty=1.1e-12,
        category="fundamental",
    ),
    # --- electromagnetic: 电磁学相关 ---
    "mu_0": PhysicalConstant(
        name="vacuum magnetic permeability",
        symbol="μ0",
        value=1.25663706212e-6,
        unit="N A^-2",
        uncertainty=1.9e-16,
        category="electromagnetic",
    ),
    "eps_0": PhysicalConstant(
        name="vacuum electric permittivity",
        symbol="ε0",
        value=8.8541878128e-12,
        unit="F m^-1",
        uncertainty=1.3e-21,
        category="electromagnetic",
    ),
    "k_e": PhysicalConstant(
        name="Coulomb constant",
        symbol="k_e",
        value=8.9875517923e9,
        unit="N m^2 C^-2",
        uncertainty=1.3e-3,  # 由 ε0 推得
        category="electromagnetic",
    ),
    "e": PhysicalConstant(
        name="elementary charge",
        symbol="e",
        value=1.602176634e-19,
        unit="C",
        uncertainty=0.0,  # SI 2019 精确定义
        category="electromagnetic",
    ),
    "mu_B": PhysicalConstant(
        name="Bohr magneton",
        symbol="μ_B",
        value=9.2740100783e-24,
        unit="J T^-1",
        uncertainty=2.8e-33,
        category="electromagnetic",
    ),
    "mu_N": PhysicalConstant(
        name="nuclear magneton",
        symbol="μ_N",
        value=5.0507837461e-27,
        unit="J T^-1",
        uncertainty=1.5e-36,
        category="electromagnetic",
    ),
    "F": PhysicalConstant(
        name="Faraday constant",
        symbol="F",
        value=96485.33212331001,
        unit="C mol^-1",
        uncertainty=0.0,  # N_A * e, 两者都精确
        category="electromagnetic",
    ),
    # --- atomic: 粒子质量 / 原子单位制 ---
    "m_e": PhysicalConstant(
        name="electron mass",
        symbol="m_e",
        value=9.1093837015e-31,
        unit="kg",
        uncertainty=2.8e-40,
        category="atomic",
    ),
    "m_p": PhysicalConstant(
        name="proton mass",
        symbol="m_p",
        value=1.67262192369e-27,
        unit="kg",
        uncertainty=5.1e-37,
        category="atomic",
    ),
    "m_n": PhysicalConstant(
        name="neutron mass",
        symbol="m_n",
        value=1.67492749804e-27,
        unit="kg",
        uncertainty=9.5e-37,
        category="atomic",
    ),
    "m_e_c2": PhysicalConstant(
        name="electron rest energy",
        symbol="m_e c^2",
        value=8.1871057769e-14,
        unit="J",
        uncertainty=2.5e-23,
        category="atomic",
    ),
    "a_0": PhysicalConstant(
        name="Bohr radius",
        symbol="a_0",
        value=5.29177210903e-11,
        unit="m",
        uncertainty=8.0e-21,
        category="atomic",
    ),
    "R_inf": PhysicalConstant(
        name="Rydberg constant",
        symbol="R_∞",
        value=10973731.568160,
        unit="m^-1",
        uncertainty=2.1e-5,
        category="atomic",
    ),
    "E_h": PhysicalConstant(
        name="Hartree energy",
        symbol="E_h",
        value=4.3597447222071e-18,
        unit="J",
        uncertainty=2.8e-31,
        category="atomic",
    ),
    "u": PhysicalConstant(
        name="atomic mass unit",
        symbol="u",
        value=1.66053906660e-27,
        unit="kg",
        uncertainty=5.0e-37,
        category="atomic",
    ),
    "eV": PhysicalConstant(
        name="electron volt",
        symbol="eV",
        value=1.602176634e-19,
        unit="J",
        uncertainty=0.0,  # 由 e 定义, 精确
        category="atomic",
    ),
    # --- thermodynamic: 热力学 / 摩尔 ---
    "N_A": PhysicalConstant(
        name="Avogadro constant",
        symbol="N_A",
        value=6.02214076e23,
        unit="mol^-1",
        uncertainty=0.0,  # SI 2019 精确定义
        category="thermodynamic",
    ),
    "k_B": PhysicalConstant(
        name="Boltzmann constant",
        symbol="k_B",
        value=1.380649e-23,
        unit="J K^-1",
        uncertainty=0.0,  # SI 2019 精确定义
        category="thermodynamic",
    ),
    "R": PhysicalConstant(
        name="molar gas constant",
        symbol="R",
        value=8.31446261815324,
        unit="J mol^-1 K^-1",
        uncertainty=0.0,  # N_A * k_B, 两者都精确
        category="thermodynamic",
    ),
    "sigma": PhysicalConstant(
        name="Stefan-Boltzmann constant",
        symbol="σ",
        value=5.670374419e-8,
        unit="W m^-2 K^-4",
        uncertainty=0.0,  # 由 h, c, k_B 推得, 都精确
        category="thermodynamic",
    ),
    "b": PhysicalConstant(
        name="Wien displacement constant",
        symbol="b",
        value=2.897771955e-3,
        unit="m K",
        uncertainty=0.0,  # 由 h, c, k_B 推得, 都精确
        category="thermodynamic",
    ),
}


# ---------------------------------------------------------------------------
# 查询函数
# ---------------------------------------------------------------------------


def get(symbol: str) -> PhysicalConstant:
    """按 symbol 取常数, 不存在给友好报错."""
    if symbol not in CONSTANTS:
        # 别让 KeyError 抛原始堆栈, 给点提示
        available = ", ".join(sorted(CONSTANTS.keys()))
        raise KeyError(
            f"Unknown physical constant symbol: {symbol!r}. "
            f"Available symbols: {available}"
        )
    return CONSTANTS[symbol]


def get_value(symbol: str) -> float:
    """直接拿数值, 单位/不确定度丢掉."""
    return get(symbol).value


def get_with_unit(symbol: str) -> tuple[float, str]:
    """返回 (value, unit) 二元组."""
    c = get(symbol)
    return c.value, c.unit


def list_all() -> list[PhysicalConstant]:
    """列全部常数, 按 symbol 排序."""
    return [CONSTANTS[k] for k in sorted(CONSTANTS.keys())]


def list_category(category: str) -> list[PhysicalConstant]:
    """按分类过滤: fundamental / electromagnetic / atomic / thermodynamic."""
    valid = {"fundamental", "electromagnetic", "atomic", "thermodynamic"}
    if category not in valid:
        raise ValueError(
            f"Unknown category: {category!r}. Valid: {sorted(valid)}"
        )
    return [c for c in CONSTANTS.values() if c.category == category]


# ---------------------------------------------------------------------------
# 便捷导出: 顶部直接 from huginn.data.physical_constants import K_B, H, E
# ---------------------------------------------------------------------------

C = CONSTANTS["c"].value
G = CONSTANTS["G"].value
H = CONSTANTS["h"].value
HBAR = CONSTANTS["hbar"].value
ALPHA = CONSTANTS["alpha"].value
MU_0 = CONSTANTS["mu_0"].value
EPS_0 = CONSTANTS["eps_0"].value
K_E = CONSTANTS["k_e"].value
E = CONSTANTS["e"].value
MU_B = CONSTANTS["mu_B"].value
MU_N = CONSTANTS["mu_N"].value
F = CONSTANTS["F"].value
M_E = CONSTANTS["m_e"].value
M_P = CONSTANTS["m_p"].value
M_N = CONSTANTS["m_n"].value
M_E_C2 = CONSTANTS["m_e_c2"].value
A_0 = CONSTANTS["a_0"].value
R_INF = CONSTANTS["R_inf"].value
E_H = CONSTANTS["E_h"].value
U = CONSTANTS["u"].value
EV = CONSTANTS["eV"].value
N_A = CONSTANTS["N_A"].value
K_B = CONSTANTS["k_B"].value
R = CONSTANTS["R"].value
SIGMA = CONSTANTS["sigma"].value
B_WIEN = CONSTANTS["b"].value  # 别跟玻尔半径冲突, 加个后缀


# ---------------------------------------------------------------------------
# 单位转换辅助
# ---------------------------------------------------------------------------

# 转换因子表: (from_unit, to_unit) -> multiplier
# 物理计算里最常用的十来种, 不搞通用单位系统
_UNIT_FACTORS: dict[tuple[str, str], float] = {
    # 能量
    ("eV", "J"): EV,                       # 1 eV = 1.602e-19 J
    ("J", "eV"): 1.0 / EV,
    ("eV", "kJ/mol"): EV * N_A / 1000.0,   # 1 eV = 96.485 kJ/mol
    ("kJ/mol", "eV"): 1000.0 / (EV * N_A),
    ("eV", "kcal/mol"): EV * N_A / 4184.0,
    ("kcal/mol", "eV"): 4184.0 / (EV * N_A),
    ("Hartree", "eV"): E_H / EV,           # 27.2114
    ("eV", "Hartree"): EV / E_H,
    ("Hartree", "J"): E_H,
    ("J", "Hartree"): 1.0 / E_H,
    ("Ry", "eV"): 0.5 * E_H / EV,          # 1 Ry = 13.606 eV
    ("eV", "Ry"): EV / (0.5 * E_H),
    ("Ry", "J"): 0.5 * E_H,
    ("J", "Ry"): 1.0 / (0.5 * E_H),
    # 长度
    ("Å", "m"): 1.0e-10,
    ("m", "Å"): 1.0e10,
    ("Å", "nm"): 0.1,
    ("nm", "Å"): 10.0,
    ("bohr", "Å"): A_0 * 1.0e10,           # 0.5292
    ("Å", "bohr"): 1.0 / (A_0 * 1.0e10),
    ("bohr", "m"): A_0,
    ("m", "bohr"): 1.0 / A_0,
    # 时间
    ("fs", "s"): 1.0e-15,
    ("s", "fs"): 1.0e15,
    ("ps", "s"): 1.0e-12,
    ("s", "ps"): 1.0e12,
    # 质量
    ("u", "kg"): U,
    ("kg", "u"): 1.0 / U,
    ("m_e", "kg"): M_E,
    ("kg", "m_e"): 1.0 / M_E,
    # 压强
    ("GPa", "Pa"): 1.0e9,
    ("Pa", "GPa"): 1.0e-9,
    # 电荷
    ("e", "C"): E,
    ("C", "e"): 1.0 / E,
}

# 温度 <-> 能量 这种需要乘 k_B 的, 单独处理, 不在表里
_TEMP_ENERGY_UNITS = {"K", "eV", "J", "meV", "Hartree", "Ry"}


def convert(value: float, from_unit: str, to_unit: str) -> float:
    """常见物理单位换算.

    覆盖能量 (eV/J/Hartree/Ry/kJ·mol/kcal·mol)、长度 (Å/m/nm/bohr)、
    时间 (fs/ps/s)、质量 (u/kg/m_e)、压强 (GPa/Pa)、电荷 (e/C),
    以及温度↔能量 (经 k_B, 如 1 eV ↔ 11604 K).

    不支持的换算抛 ValueError.
    """
    if from_unit == to_unit:
        return float(value)

    # 先查直接因子表
    key = (from_unit, to_unit)
    if key in _UNIT_FACTORS:
        return value * _UNIT_FACTORS[key]

    # 温度 <-> 能量: 1 K = k_B J, 1 eV = 1.16045e4 K
    # 先把 from_unit 归一到 J, 再换算到 to_unit
    def _to_joule(v: float, unit: str) -> float:
        if unit == "J":
            return v
        if unit == "K":
            return v * K_B  # 热能 k_B T
        if unit == "meV":
            return v * EV * 1e-3
        if (unit, "J") in _UNIT_FACTORS:
            return v * _UNIT_FACTORS[(unit, "J")]
        raise ValueError(f"Cannot convert from {unit!r} to Joule")

    def _from_joule(v_j: float, unit: str) -> float:
        if unit == "J":
            return v_j
        if unit == "K":
            return v_j / K_B
        if unit == "meV":
            return v_j / (EV * 1e-3)
        if ("J", unit) in _UNIT_FACTORS:
            return v_j * _UNIT_FACTORS[("J", unit)]
        raise ValueError(f"Cannot convert from Joule to {unit!r}")

    # 只在温度/能量范畴内尝试中间换算
    if from_unit in _TEMP_ENERGY_UNITS and to_unit in _TEMP_ENERGY_UNITS:
        return _from_joule(_to_joule(value, from_unit), to_unit)

    raise ValueError(
        f"Unsupported unit conversion: {from_unit!r} -> {to_unit!r}. "
        f"Supported units: Å, m, nm, bohr, eV, J, meV, Hartree, Ry, "
        f"kJ/mol, kcal/mol, K, fs, ps, s, u, kg, m_e, GPa, Pa, e, C."
    )
