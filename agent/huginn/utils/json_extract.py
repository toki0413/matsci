"""json_extract — 从 LLM 输出文本里提取 JSON 的单一实现.

合并前全仓有五套策略互不相同的实现 (plan_check / text_to_json / text_types /
autoresearch / vision_locate), 同一个模型输出在不同路径上可能得到不同结果.
这里只保留一套原语, 各调用点退化为薄适配, 保留各自原有签名与返回类型.

约束: 仅标准库; 纯函数; 解析失败一律返回 None / 空串, 不抛异常.
"""

from __future__ import annotations

import json
import re
from typing import Any, NamedTuple

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

OBJECT_OPENERS: tuple[str, ...] = ("{",)
ALL_OPENERS: tuple[str, ...] = ("{", "[")


class JsonSpan(NamedTuple):
    """一个平衡 JSON 片段的区间 (end 为开区间) 与顶层容器类型."""

    start: int
    end: int
    opener: str  # "{" 或 "["


def strip_code_fence(text: str, *, prefer: str = "last") -> str:
    """去掉 markdown 围栏(```json ... ``` / ``` ... ```), 取指定一段内容.

    没有围栏时原样返回. prefer="last" 取最后一段 (思考叙事在前、结果在后).
    """
    blocks = _FENCE_RE.findall(text)
    if not blocks:
        return text
    return blocks[-1] if prefer == "last" else blocks[0]


def find_json_spans(text: str) -> list[JsonSpan]:
    """扫描文本, 返回**所有**平衡 {...} / [...] 的区间, 按出现顺序.

    跳过字符串字面量里的括号与转义引号; 括号不匹配时丢弃该候选.
    包含嵌套片段 (数组里的对象也会被列出).
    """
    spans: list[JsonSpan] = []
    stack: list[tuple[str, int]] = []
    in_str = False
    escaped = False
    for i, ch in enumerate(text):
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append((ch, i))
        elif ch in "}]":
            if not stack:
                continue
            opener, start = stack.pop()
            if (opener == "{" and ch != "}") or (opener == "[" and ch != "]"):
                stack.clear()
                continue
            spans.append(JsonSpan(start, i + 1, opener))
    spans.sort(key=lambda s: s.start)
    return spans


def top_level_spans(spans: list[JsonSpan]) -> list[JsonSpan]:
    """只保留不被其它片段包含的最外层片段."""
    return [
        s
        for s in spans
        if not any(o.start < s.start and s.end <= o.end for o in spans)
    ]


def _pick_spans(
    text: str, *, prefer: str, openers: tuple[str, ...]
) -> list[JsonSpan]:
    """顶层片段按 opener 过滤后, 依 prefer 返回候选顺序 ("last" 逆序)."""
    spans = [s for s in top_level_spans(find_json_spans(text)) if s.opener in openers]
    if prefer == "last":
        return list(reversed(spans))
    return spans


def extract_json(
    text: str,
    *,
    prefer: str = "last",
    openers: tuple[str, ...] = ALL_OPENERS,
    strip_fences: bool = True,
) -> Any | None:
    """从任意文本取出一个 JSON 容器值 (dict / list), 找不到返回 None.

    先试整段即 JSON; 再去 markdown 围栏; 最后按 prefer 从顶层平衡片段里挑.
    openers 限定可接受的容器类型, 例如 ("{",) 只收对象.
    """
    if not text or not isinstance(text, str):
        return None
    t = text.strip()
    try:  # fast-path: 整段本身就是 JSON
        value = json.loads(t)
    except Exception:  # noqa: BLE001 — 落到片段扫描
        value = None
    else:
        if isinstance(value, dict) and "{" in openers:
            return value
        if isinstance(value, list) and "[" in openers:
            return value
    if strip_fences:
        t = strip_code_fence(t, prefer=prefer)
    for span in _pick_spans(t, prefer=prefer, openers=openers):
        try:
            return json.loads(t[span.start : span.end])
        except Exception:  # noqa: BLE001 — 单个坏片段跳过
            continue
    return None


def extract_json_text(
    text: str,
    *,
    prefer: str = "first",
    openers: tuple[str, ...] = ALL_OPENERS,
    strip_fences: bool = True,
) -> tuple[str, str]:
    """返回 (原文片段, 错误原因); 成功时错误为空串.

    只做定位, 不校验能否解析 —— 调用方需要原文自己 json.loads 时用它.
    错误原因: "no json"(文本里没有开括号) / "no closing brace"(有开无闭).
    """
    if not text or not isinstance(text, str):
        return "", "no json"
    t = text.strip()
    try:  # fast-path: 整段即合法 JSON, 直接回原文
        value = json.loads(t)
    except Exception:  # noqa: BLE001
        pass
    else:
        if (isinstance(value, dict) and "{" in openers) or (
            isinstance(value, list) and "[" in openers
        ):
            return t, ""
    if strip_fences:
        t = strip_code_fence(t, prefer=prefer)
    spans = _pick_spans(t, prefer=prefer, openers=openers)
    if not spans:
        # 有平衡片段但不在顶层 (如数组里的对象) -> 视为没有目标容器
        if any(s.opener in openers for s in find_json_spans(t)):
            return "", "no json"
        return "", "no closing brace" if any(o in t for o in openers) else "no json"
    span = spans[0]
    return t[span.start : span.end], ""
