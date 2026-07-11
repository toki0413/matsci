"""材料性质名称别名映射 — 让用户和各数据库的异构字段名统一到 canonical name.

借鉴 AGAPI 的 PROPERTY_ALIASES 思路, 但覆盖我们自己的工具链字段.
用法: normalize_property_name("bg") -> "band_gap"
"""

from __future__ import annotations

import re

# canonical name -> 别名集合
# 别名匹配时全部转小写 + 去空格/连字符, 所以这里只写小写形式
_PROPERTY_ALIASES: dict[str, set[str]] = {
    "band_gap": {"bg", "eg", "bandgap", "band gap", "electronic_gap", "gap"},
    "formation_energy": {"delta_e", "formation energy", "formationenergy", "ef", "e_form", "energy_per_atom"},
    "bulk_modulus": {"bulk modulus", "bulkmodulus", "k_vrh", "k_voigt", "bv"},
    "shear_modulus": {"shear modulus", "shearmodulus", "g_vrh", "g_voigt", "gv"},
    "youngs_modulus": {"youngs modulus", "young modulus", "e_modulus", "ey"},
    "poisson_ratio": {"poisson", "nu", "poissonratio"},
    "debye_temperature": {"debye temp", "debye temperature", "theta_d", "debye"},
    "thermal_conductivity": {"thermal conductivity", "kappa", "k_thermal"},
    "fermi_energy": {"fermi energy", "fermi level", "e_fermi", "ef_fermi"},
    "magnetic_moment": {"magnetic moment", "magmom", "total magnetization", "magnetization"},
    "dielectric_constant": {"dielectric constant", "dielectric", "epsilon", "eps", "epsilon_static"},
    "refractive_index": {"refractive index", "n_refractive", "refractiveindex"},
    "spacegroup": {"space group", "spacegroup", "sg", "symmetry"},
    "lattice_parameters": {"lattice params", "lattice", "lattice constants", "cell parameters"},
    "density": {"rho", "mass density", "volumetric density"},
    "melting_point": {"melting temperature", "melting point", "t_melt", "tm"},
    "cohesive_energy": {"cohesive energy", "ecoh", "e_coh"},
    "phonon_frequency": {"phonon frequency", "phonon", "omega_ph"},
    "surface_energy": {"surface energy", "gamma_surface", "surface_energy"},
    "work_function": {"work function", "wf", "workfunction"},
}

# 预编译反向索引: alias_lower -> canonical
_ALIAS_INDEX: dict[str, str] = {}
for _canonical, _aliases in _PROPERTY_ALIASES.items():
    _ALIAS_INDEX[_canonical] = _canonical  # canonical 自身也注册
    for _a in _aliases:
        _ALIAS_INDEX[_a] = _canonical


def _normalize_key(s: str) -> str:
    """统一大小写和分隔符: 小写 + 去空格/连字符/下划线."""
    return re.sub(r"[\s\-_]+", "", s.lower())


# 用 _normalize_key 重建索引, 这样 "band-gap" "band_gap" "bandgap" 都能匹配
_NORMALIZED_INDEX: dict[str, str] = {}
for _canonical, _aliases in _PROPERTY_ALIASES.items():
    _NORMALIZED_INDEX[_normalize_key(_canonical)] = _canonical
    for _a in _aliases:
        _NORMALIZED_INDEX[_normalize_key(_a)] = _canonical


def normalize_property_name(raw: str) -> str:
    """把用户输入或数据库字段名归一化到 canonical property name.

    >>> normalize_property_name("bg")
    'band_gap'
    >>> normalize_property_name("Eg")
    'band_gap'
    >>> normalize_property_name("band-gap")
    'band_gap'
    >>> normalize_property_name("unknown_prop")
    'unknown_prop'
    """
    if not raw:
        return raw
    key = _normalize_key(raw)
    return _NORMALIZED_INDEX.get(key, raw)


def known_properties() -> list[str]:
    """返回所有已注册的 canonical property name."""
    return list(_PROPERTY_ALIASES.keys())


if __name__ == "__main__":
    # 快速自检
    assert normalize_property_name("bg") == "band_gap"
    assert normalize_property_name("Eg") == "band_gap"
    assert normalize_property_name("band-gap") == "band_gap"
    assert normalize_property_name("delta_e") == "formation_energy"
    assert normalize_property_name("Bulk Modulus") == "bulk_modulus"
    assert normalize_property_name("unknown") == "unknown"
    print(f"OK — {len(known_properties())} canonical properties registered")
