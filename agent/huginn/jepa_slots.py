"""结构化计划槽 (structured plan slots) — 采集/序列化/防泄漏.

动机: JEPA 0.28 信息下限的根因是 prediction 侧缺字面数字(plan 文本多是
自由描述), 而 actual 侧有完整输出。方案①在 plan 侧加一套"plan 时已知的输入
参量 + 意图公式"槽, 显式把缺失信息补进 prediction, 让 predictor 有据此重建
actual 的信息。

诚实红线 (最重要):
  - 槽只采集 **执行前已知的输入/公式**, 绝不采集"预测的输出值"。
  - 若槽里的数值 == actual 的结果码(最后数值), 判为泄漏, 丢弃该对 —— 否则
    prediction 变成 ground-truth 本身, surprise 退化成循环自证。

序列化形态: ``PLAN_SLOTS: a=3; b=4 unit=...; formula=RMS(a,b)`` 追加到
prediction 文本尾。现行 span predictor 按行切 span, 零改动即可消费槽块。

用法:
    slots = [{"label":"a","value":3}, {"label":"b","value":4, "unit":"eV"}]
    formula = "RMS(a,b)"
    text = append_slots(prediction, slots, formula)
    if detect_leak(slots, actual_text):
        # 槽泄漏了答案 → 丢弃该对, 不进训练
        drop()
"""
from __future__ import annotations

import re

# 与 train_jepa_span_predictor 同款数值 token 正则 (含科学计数法)
_NUM_RE = re.compile(r"-?\d+\.\d+(?:[eE][+-]?\d+)?|-?\d+[eE][+-]?\d+")


def slots_block(inputs) -> str | None:
    """把 slots(list[dict]) + 隐含 formula 序列化成规范 PLAN_SLOTS 文本块.

    slot dict: {label?, value, unit?}. 无任何槽/公式返回 None。
    """
    if not inputs:
        return None
    parts: list[str] = []
    for s in inputs or []:
        if not isinstance(s, dict):
            continue
        value = s.get("value")
        if value is None:
            continue
        label = str(s.get("label", "")).strip()
        unit = str(s.get("unit", "")).strip()
        toks: list[str] = []
        if label:
            toks.append(f"{label}={value}")
        else:
            toks.append(str(value))
        if unit and unit.lower() not in ("", "unit", "none"):
            toks.append(f"unit={unit}")
        if toks:
            parts.append(" ".join(toks))
    if not parts:
        return None
    return "PLAN_SLOTS: " + "; ".join(parts)


def append_slots(prediction: str, inputs, formula: str = "") -> str:
    """把槽块(含可选 formula)追加到 prediction 尾; 无槽则原样返回."""
    blk = slots_block(inputs)
    if not blk:
        return prediction or ""
    if formula:
        blk = f"{blk}; formula={formula}"
    p = (prediction or "").strip()
    return (p + "\n" + blk) if p else blk


def _last_numeric(text: str) -> float | None:
    """actual 文本最后一个数值 token (多为结果码). 无数值返回 None."""
    ms = list(_NUM_RE.finditer((text or "").strip()))
    if not ms:
        return None
    try:
        return float(ms[-1].group(0))
    except ValueError:
        return None


def detect_leak(inputs, actual: str) -> bool:
    """防答案泄漏: 任一槽数值 ≈ actual 的结果码(末位数值) → 判泄漏.

    输入参量(a=3,b=4)会合法地出现在 actual 前面(因为它就是 givens),
    但"答案"是 actual 的最后一个数值; 若槽把它写了进来, 说明 LLM 把预测
    输出当槽塞了, 该对 surprise 无信息量, 判泄漏。
    """
    ans = _last_numeric(actual)
    if ans is None:
        return False
    for s in inputs or []:
        if not isinstance(s, dict):
            continue
        try:
            v = float(s.get("value"))
        except (TypeError, ValueError):
            v = None
        if v is not None and abs(v - ans) <= max(1e-6, abs(ans) * 1e-4):
            return True
    return False


def parse_slots_line(text: str) -> tuple[list[dict], str]:
    """解析 LLM 输出的 SLOTS: 行 → (slots, formula).

    支持 ``k=v`` 槽、裸数值、``unit=...`` 追加到上一个槽、``formula=<expr>``。
    """
    slots: list[dict] = []
    formula = ""
    for tok in (text or "").split(";"):
        tok = tok.strip()
        if not tok:
            continue
        low = tok.lower()
        if low.startswith("formula="):
            formula = tok.split("=", 1)[1].strip()
            continue
        if low.startswith("unit="):
            if slots:
                slots[-1]["unit"] = tok.split("=", 1)[1].strip()
            continue
        m = re.match(r"^([^=\s]+)\s*=\s*(.+)$", tok)
        if m:
            slots.append({"label": m.group(1).strip(), "value": m.group(2).strip()})
        else:
            slots.append({"value": tok})
    return slots, formula