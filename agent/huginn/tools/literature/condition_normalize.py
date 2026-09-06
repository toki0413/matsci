"""结构层 condition 归一 — 把文献报道值的物理自由度归一成语义条件键.

不依赖 LLM 抽取, 纯规则 + 别名映射. 解决"同一物理量缺自由度 → 分歧无法归因":
`_annotate_value_consistency` 目前只按 unit 分组, 把 method/note 里隐含的
温度/计算方法/测量方式这些自由度当成了同质性对待, 于是"PBE vs HSE"、"实验 vs
DFT"、"0K vs 室温"报出的不同数值被误打成 conflicting, 而非"条件不同".

本模块从 method + note + unit 三个已有字段里**确定性地**抽出一组 condition 键:

- ``method_family``     : 实验 / DFT / MD / 半经验(微扰+M+N) / unknown
- ``functional``        : DFT 泛函族 (PBE/HSE/revPBE/...), 非 DFT 置 None
- ``temperature``       : 归一温度档 (0K近似/室温/其他/未声明), 从 note 提取
- ``basis``             : 基组/赝势 (GW/PBE+U/MD-basin-hopping...), 保留细粒度
- ``condition_key``     : 组合上述维度的稳定键, 用作一致性分组维度

设计约束 (对齐项目风格, 参见 property_aliases.py):
- 纯函数, 零网络/零 LLM/零 I/O, 幂等 (同输入必同输出).
- 别名索引预处理规范化 (小写 + 去空格/连字符/下划线), 容忍书写差异.
- 无法归一的一律落 "unknown"/None, 不臆造; 缺失自由度显性暴露, 供上层诊断.
"""

from __future__ import annotations

import re
from typing import Any

# ────────────────────────── 温度归一 ──────────────────────────

# 温度档: 只有"显式 T=(值)"或"值+K/Kelvin 单位"才算温度,
# 避免把方法字符串里的数字 (如 HSE06 / PBE+U) 误判成温度.
_TEMP_VALUE_PATTERN = re.compile(
    r"(?:T\s*[:=]\s*)([0-9]+(?:\.[0-9]+)?)|" r"([0-9]+(?:\.[0-9]+)?)\s*(?:K|Kelvin)\b",
    re.IGNORECASE,
)

# 语义温度档: 关键词 → 归一温度键 (不在数值内判断, 兼顾 "0K"/"RT"/"室温")
# 由 normalize_temperature 用 word 边界正则判断, 这里保留映射便于扩展.
_SEMANTIC_TEMP: dict[str, str] = {
    "0k": "t_zero",
    "at 0k": "t_zero",
    "t=0": "t_zero",
    "t=0k": "t_zero",
    "rt": "t_room",
    "room temperature": "t_room",
    "室温": "t_room",
    "300k": "t_room",
    "t=300k": "t_room",
}

# 数值温度: 落入哪个档 (K)
_TEMP_BANDS: list[tuple[float, float, str]] = [
    (0, 1, "t_zero"),  # 0K 附近的 DFT 惯用值
    (250, 350, "t_room"),  # 室温附近 (RT ~293-300 K)
]


def normalize_method_family(method: str) -> str:
    """把 method 字符串归一到高频方法族.

    只识别可信的硬规则; 其余一律 "unknown" (不臆造, 方便上层标 ungrouped).
    """
    m = (method or "").strip()
    if not m:
        return "unknown"
    norm = m.lower()
    # 实验/测量优先 (可能混在 "experiment, DFT crosscheck" 里)
    if any(
        k in norm
        for k in (
            "experiment",
            "measurement",
            "experimental",
            "measured",
            "x-ray",
            "synthe",
        )
    ):
        return "experiment"
    # DFT 家族 (含泛函/软件写法, 如 HSE06 / PBE / GGA-PBE / VASP). 词边界允许
    # 数字后缀, 避免 "HSE06" 因 \b 断词失败而漏判.
    if re.search(
        r"(?<![a-z])(dft|pbe|pbesol|pwc|gga|lda|hse\d*|revpbe|r2scan|mbe|b3lyp|scan|vasp|qe|abinit|siesta|cp2k)(?![a-z])",
        norm,
    ):
        return "dft"
    if "md" in norm or "basin-hopping" in norm or "molecular dynamics" in norm:
        return "md"
    if (
        "semiempirical" in norm
        or "semi-empirical" in norm
        or "am1" in norm
        or "pm3" in norm
    ):
        return "semiempirical"
    return "unknown"


def normalize_functional(method: str, family: str) -> str | None:
    """DFT 泛函族归一 (PBE/HSE/revPBE...); 非 DFT 或未识别返回 None."""
    if family != "dft":
        return None
    m = (method or "").lower()
    # 匹配泛函 token (数字后缀如 HSE06 也命中), 顺序: 更长/更具体在前避免子串误判
    for name in (
        "revpbe",
        "pbesol",
        "hse06",
        "r2scan",
        "b3lyp",
        "hse",
        "pbe",
        "pwc",
        "scan",
        "lda",
        "gga",
    ):
        if re.search(rf"(?<![a-z]){name}(?![a-z])", m):
            return name
    return None


def normalize_temperature(note: str, method: str = "") -> str:
    """从 note(优先) + method 提取温度档.

    优先级: 语义关键词 > 显式数值 (按 _TEMP_BANDS 归档) > unknown.
    """
    text = f"{note or ''} {method or ''}".strip()
    low = text.lower()
    # 语义档 (0K/RT/室温等) — 需要 word 边界, 避免 "HSE06" 里的 "0" 命中 "0k"
    if re.search(r"\b0\s*k\b", low) or "t=0" in low:
        return "t_zero"
    if re.search(r"\b(rt|room[ -]?temperature|room[ -]?temp|室温)\b", low):
        return "t_room"
    # 显式数值 (T=300 / "300 K") 按档归档
    m = _TEMP_VALUE_PATTERN.search(text)
    if m:
        value = float(m.group(1) or m.group(2))
        for lo, hi, key in _TEMP_BANDS:
            if value > lo and value <= hi:
                return key
        return "t_other"
    return "unknown"


def normalize_basis(method: str, note: str) -> str | None:
    """基组/方法细粒度 token (GW / PBE+U / MD-basin-hopping). None 若无法识别."""
    m = (method or "").strip()
    n = (note or "").strip()
    full = f"{m} {n}".lower()
    basis: list[str] = []
    for tok in (
        "pbe+u",
        "pbe-u",
        "gw",
        "g0w0",
        "mbj",
        "tddft",
        "u-hubbard",
        "spin-orbit",
        "+u",
        "-u",
    ):
        if tok in full:
            basis.append(tok.lstrip("+-"))
    return "+".join(sorted(set(basis))) if basis else None


def condition_key(method: str, note: str, unit: str = "") -> tuple[str, str]:
    """组合多个维度成为稳定 condition 键, 用作一致性分组.

    返回 (condition_key_str, diagnostics):
      - condition_key: 稳定分组键, 例如 "dft/pbe@t_zero" 或 "experiment@unknown"
      - diagnostics: 记录了哪些维度缺失 (供上层"缺自由度"诊断), 如 "functional,temperature"

    设计: 只聚合**已能确定**的维度; 缺失的暴露在 diagnostics 里, 不塞进 key
    充当"shared"假同质. 这使得"同 unit 但 PBE vs HSE"各自成组而非伪装一致.
    """
    family = normalize_method_family(method)
    functional = normalize_functional(method, family)
    temperature = normalize_temperature(note, method)
    basis = normalize_basis(method, note)

    # 缺失自由度诊断 (真缺清单, 供上层判断要不要补表征)
    missing: list[str] = []
    if family == "unknown":
        missing.append("method_family")
    if family == "dft" and functional is None:
        missing.append("functional")
    if temperature == "unknown":
        missing.append("temperature")

    dims: list[str] = []
    if family != "unknown":
        dims.append(family)
    if functional:
        dims.append(functional)
    if basis:
        dims.append(basis)
    # 温度只在实际偏离默认 (室温/其他明确温度) 时才单独成维度.
    # t_zero / unknown 都是"未特别声明", 不塞进 key 当假同质 — 缺失由 missing 暴露.
    if temperature in ("t_room", "t_other"):
        dims.append(temperature)
    elif temperature == "t_zero":
        pass  # DFT 默认 0K, 不单列
    # 注意: temperature == "unknown" 时既不入 dims, 也由 missing 标记, 不伪造.
    dims.append(unit.strip() or "unitless")

    return "/".join(dims), ",".join(missing)


def feature_vector(method: str, note: str, unit: str = "") -> dict[str, Any]:
    """单条 reported 的归一化结构特征 (供调用方直接使用)."""
    family = normalize_method_family(method)
    functional = normalize_functional(method, family)
    temperature = normalize_temperature(note, method)
    basis = normalize_basis(method, note)
    key, missing_dims = condition_key(method, note, unit)
    return {
        "method_family": family,
        "functional": functional,
        "temperature": temperature,
        "basis": basis,
        "condition_key": key,
        "missing_dims": missing_dims.split(",") if missing_dims else [],
        "group": key,
    }


def group_reported(reported: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """把 reported 列表按结构层 condition_key 分组 (替代仅按 unit 分组).

    返回 {condition_key: [reported items...]}; 各条已注入 feature_vector 字段.
    未改动原 dict, 返回浅拷贝并附加 feature 键.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in reported:
        unit = str(item.get("unit") or "")
        method = str(item.get("method") or "")
        note = str(item.get("note") or "")
        feat = feature_vector(method, note, unit)
        enriched = dict(item)
        enriched["_condition"] = feat
        groups.setdefault(feat["condition_key"], []).append(enriched)
    return groups


if __name__ == "__main__":  # pragma: no cover - 快速自检
    cases = [
        ("dft-pbe", "", "eV"),
        ("DFT-PBE", "T=0K", "eV"),
        ("HSE06", "", "eV"),
        ("experiment", "room temperature", "eV"),
        ("", "", "eV"),
    ]
    for method, note, unit in cases:
        print(method or "∅", "|", note or "∅", "→", condition_key(method, note, unit))
