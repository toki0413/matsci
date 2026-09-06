"""缺度追问 — 把局部-整体实验发现的"缺失自由度"翻译成可执行的补全检索.

背景 / hypothesis:
  局部-整体兼容实验里 `missing_dims` 暴露"哪些素数没定义" (temperature /
  functional / method_family 缺明)。缺度追问就是: 结构层**不靠 LLM 臆测**,
  而是把缺失自由度翻译成一句可知可检索的**补全查询**, 指明它补哪个自由度、
  为什么补、对整体判定有多关键。上层可直接把 query 喂给 `_do_search` 或
  web_search 做真实条件补全检索。

设计约束 (对齐 condition_normalize):
  - 纯函数, 零网络 / 零 LLM / 零 I/O, 幂等 (同输入必同输出)。
  - 各维度缺度有独立模板; 只生成**已缺**自由度的查询。
  - 给 urgency 分级: 该自由度缺失对"整体互洽判定"影响多大 (温度最关键——
    前面实验中缺温度让 PBE@0K 与实验/HSE 无法对齐)。

可独立运行:  python -m huginn.tools.literature.query_completion
"""

from __future__ import annotations

from typing import Any

# 各缺失自由度 → 补全查询模板 (ai_version: 无 LLM 依赖, 纯字符串组装)
# {system} {property} {unit} 由调用方传入; {target} 是补全对象
_TEMPLATES: dict[str, str] = {
    "temperature": "{system} {property} value at temperature T {unit}",
    "functional": "{system} {property} HSE vs PBE band gap comparison {unit}",
    "method_family": "{system} {property} value {unit} method cross-check",
}

# 缺度 → 对整体互洽判定的关键度 (3=最影响判定, 1=补充信息)
_URGENCY: dict[str, int] = {
    "temperature": 3,  # 不补, 组间中心无法对齐 (0K vs 室温)
    "functional": 2,  # 不补, 无法区分泛函系统性偏差
    "method_family": 1,  # 不补, 只能归 unknown, 但不致误判数值冲突
}


def _clean(s: str) -> str:
    s = (s or "").strip()
    return " ".join(s.split())


def _known_condition_tail(known: dict[str, str]) -> str:
    """把已知自由度拼成查询尾部 (如 'experiment at room temperature')."""
    if not known:
        return ""
    parts: list[str] = []
    fam = known.get("method_family")
    if fam and fam != "unknown":
        parts.append(fam)
    func = known.get("functional")
    if func:
        parts.append(func.upper())
    temp = known.get("temperature")
    if temp == "t_room":
        parts.append("room temperature")
    elif temp == "t_other":
        parts.append("non-standard temperature")
    elif temp == "t_zero":
        parts.append("at 0 K")
    return " ".join(parts) if parts else ""


def completion_query(
    missing_dims: list[str],
    system: str,
    property: str,
    unit: str = "",
    *,
    known: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """为缺失自由度的生成补全查询 (按 urgency 排序).

    missing_dims: condition_normalize 报出的缺度 (如 ["temperature", "functional"])
    system/property/unit: 待补齐的体系/性质/单位
    known: 已知自由度 (method_family 等), 用于让补全查询更聚焦.

    返回: [{target_dim, query, urgency, reason, should_retrieve}], 按 urgency 降序.
    """
    dims = [d for d in missing_dims if d in _TEMPLATES]
    if not dims:
        return []
    sys_name = _clean(system)
    prop_name = _clean(property)
    if not sys_name or not prop_name:
        return []
    unit_s = _clean(unit)
    tail = _known_condition_tail(known or {})
    base_fields = {
        "system": sys_name,
        "property": prop_name,
        "unit": unit_s,
    }
    out: list[dict[str, Any]] = []
    for dim in dims:
        # 已知的自由度不视为"缺", 跳过 (method_family 已从 known 拿到 → 不再判缺)
        if known and _known_satisfies(know=known, dim=dim):
            continue
        template = _TEMPLATES[dim]
        query = template.format(**base_fields)
        if tail:
            query = f"{query} {tail}"
        reason = _reason(dim)
        urgency = _URGENCY.get(dim, 1)
        out.append(
            {
                "target_dim": dim,
                "query": _clean(query),
                "urgency": urgency,
                "reason": reason,
                "should_retrieve": urgency >= 2,  # 关键缺度才值得发起真实检索
            }
        )
    out.sort(key=lambda q: q["urgency"], reverse=True)
    return out


def _known_satisfies(*, know: dict[str, str], dim: str) -> bool:
    """该自由度是否已被 known 里的条件满足 (从而不再缺)."""
    if dim == "method_family":
        fam = know.get("method_family")
        return bool(fam and fam != "unknown")
    if dim == "functional":
        return bool(know.get("functional"))
    if dim == "temperature":
        temp = know.get("temperature")
        return temp in ("t_room", "t_other", "t_zero")
    return False


def _reason(dim: str) -> str:
    mapping = {
        "temperature": "温度是组间互洽判定的关键自由度: 0K vs 室温的同类性质数值不可直接对账",
        "functional": "泛函决定带隙等电子性质的系统性偏差方向: 缺它无法区分 LDA/GGA vs HSE 差异",
        "method_family": "方法族 (实验/DFT/MD) 决定数值的物理含义: 缺它只能归 unknown, 无法跨源对比",
    }
    return mapping.get(dim, "")


def build_followup_input(
    system: str,
    property: str,
    unit: str,
    missing_dims: list[str],
    *,
    method_family: str | None = None,
    functional: str | None = None,
    temperature: str | None = None,
) -> dict[str, Any]:
    """一个便捷入口: 把不足度 + 已知条件打包成"应发起的补全检索"描述.

    返回:
      - target_query: 最关键的补全查询 (供直接检索)
      - queries: 全部补全查询
      - fills: 这些查询期望补全的自由度
      - known_conditions_tail: 已锁定条件 (供检索时附上)
    """
    known: dict[str, str] = {}
    if method_family:
        known["method_family"] = method_family
    if functional:
        known["functional"] = functional
    if temperature:
        known["temperature"] = temperature
    queries = completion_query(missing_dims, system, property, unit, known=known)
    return {
        "target_query": queries[0]["query"] if queries else None,
        "queries": queries,
        "fills": [q for q in queries if q["should_retrieve"]],
        "known_conditions_tail": _known_condition_tail(known),
    }


# ───────────────────────────── 可读摘要 ─────────────────────────────


def summarize_missing(missing_dims: list[str]) -> dict[str, str]:
    """缺度的可读中文解释 (供输出给上层/用户) — 纯映射, 无 LLM."""
    mapping = {
        "temperature": "未声明测量/计算温度",
        "functional": "未声明 DFT 泛函",
        "method_family": "未声明方法族 (实验/DFT/MD)",
    }
    return {d: mapping.get(d, d) for d in missing_dims}


if __name__ == "__main__":  # pragma: no cover - 可独立运行
    # 已知 method_family=experiment → 不应再生成 method_family 补全查询
    qs = completion_query(
        ["temperature", "method_family"],
        "Li2O",
        "band_gap",
        "eV",
        known={"method_family": "experiment"},
    )
    for q in qs:
        print(
            f"[{q['urgency']}] {q['target_dim']:15s}: {q['query']}  retr={q['should_retrieve']}"
        )
