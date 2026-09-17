"""text_to_json —— 模型无关的鲁棒结构化输出提取层.

为什么需要: 不同 LLM 对 STRICT JSON 契约的遵循差异极大且不稳定
(书生 s1 ~10%, DeepSeek 100%; 同一模型还会随 prompt 复杂度切换行为).
不能靠 per-model 白名单 / 单独配置(那不可泛化).

本模块把"赌模型会输出干净 JSON"转成: **无论输出多脏, 管线都能尽力取出
结构化对象** —— 它不认识任何模型, 对任意 OpenAI 兼容端点的返回文本都适用,
因此是一个与模型无关的泛化提取层.

提取策略递进(泛化, 不依赖某个模型偏好):
  1. fast-path: 整段即合法 JSON
  2. 去 markdown 围栏后取首个 / 末个平衡 {...}
  3. 宽松字段抽取: 无完整 JSON 时, 用 schema 字段名在任意文本里抓 k:v(数值/字符串),
     容忍散文 / 半 JSON / 截断 —— 这对"思考+结果混排"的输出尤其有用.

调用方只需声明 schema(字段名 -> 期望类型/提取器), 不关心具体模型.

设计约束(贴合项目偏好):
  纯函数, 零 LLM / 零网络 / 零 numpy 依赖; 仅标准库.
  失败一律返回 None, 不抛异常, 由调用层的"遇错升级"策略处理重试.
"""
from __future__ import annotations

import contextlib
import json
import re
from collections.abc import Callable
from typing import Any

# ── 1. 工具: 去围栏 / 找平衡 JSON ────────────────────────────────

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def strip_fence(text: str) -> str:
    """去掉 markdown 围栏(```json ... ``` / ``` ... ```), 取最后一段代码内容."""
    blocks = _FENCE_RE.findall(text)
    return blocks[-1] if blocks else text


def find_json_objects(text: str) -> list[dict]:
    """扫描文本, 返回**所有**平衡 {...} 的已解析 dict(按出现顺序).

    遍历一次找出每个最外层平衡对象; 单对象解析失败不影响其它.
    返回空表 = 文本里没有完整可解析的 JSON 对象.
    """
    out: list[dict] = []
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    cand = text[start:i + 1]
                    try:
                        obj = json.loads(cand)
                        if isinstance(obj, dict):
                            out.append(obj)
                    except Exception:  # noqa: BLE001 — 单个坏对象跳过
                        pass
    return out


def extract_json_object(text: str, prefer: str = "last") -> dict | None:
    """从任意文本取出一个 JSON 对象 dict.

    prefer: "last" 取最后一个平衡对象(think 叙事在前、结果在后时更稳);
            "first" 取第一个. 任何情况无 -> None.
    先试 fast-path(整段即 JSON); 再去围栏; 再按 prefer 从平衡对象里挑.
    """
    if not text or not isinstance(text, str):
        return None
    t = text.strip()
    try:  # fast-path: 整段本身就是 JSON
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except Exception:  # noqa: BLE001
        pass
    t = strip_fence(t)
    objs = find_json_objects(t)
    if not objs:
        return None
    return objs[-1] if prefer == "last" else objs[0]


# ── 2. 宽松字段抽取(容忍无完整 JSON 的散文/半 JSON) ──────────────
# 对"思考+结果混排、JSON 被截断/拆散"的输出, 按字段名直接抓值.

_STR_RE = re.compile(r'["\']?([\w]+)["\']?\s*[:：=]\s*"([^"]*)"')
_NUM_RE = re.compile(r'["\']?([\w]+)["\']?\s*[:：=]\s*(-?\d*\.?\d+(?:[eE][+-]?\d+)?)')


def loose_scalar_fields(text: str) -> dict:
    """宽松抓所有 'key: value' 对: 值优先解释为 number, 否则原始串."""
    out: dict[str, Any] = {}
    for key, num in _NUM_RE.findall(text):
        with contextlib.suppress(ValueError):
            out[key] = float(num)
    for key, val in _STR_RE.findall(text):
        out.setdefault(key, val)  # 字符串弱优先级, 数值优先
    return out


def loose_json_like(text: str) -> dict | None:
    """若是'首个顶层对象被某些语言省略,但整体像JSON'的多行文本, 尝试整段补齐.

    保守启发: 文本以 { 开头、以 } 结尾, 但 json 失败时, 逐个字段抓并组装.
    """
    t = text.strip()
    if not (t.startswith("{") and t.endswith("}")):
        return None
    fields = loose_scalar_fields(t)
    return fields or None


# ── 3. schema 驱动的鲁棒提取(调用方声明字段类型) ─────────────────
# schema: dict[str, extractor]
#   extractor 可以是类型构造器(如 float / int / str), 或一个 callable(str)->value
# 返回 dict(仅含成功满足 schema 的字段) 或 None(未取到任何字段).

_TypeOrCallable = type | Callable[[str], Any]


def _coerce(raw: str, desired: _TypeOrCallable) -> Any:
    if desired is float:
        return float(raw)
    if desired is int:
        return int(raw)
    if desired is str:
        return raw
    if callable(desired):
        return desired(raw)
    return raw


def robust_extract(
    text: str,
    schema: dict[str, _TypeOrCallable],
    prefer: str = "last",
) -> dict | None:
    """综合提取: JSON 对象优先; 无完整 JSON 时按 schema 宽松抓字段.

    Result 只对 schema 里能找到的字段生效; 一个字段都没有 -> None
    (此时调用层应判断这次输出不可用, 触发升级重试).
    """
    if not text:
        return None
    obj = extract_json_object(text, prefer=prefer)
    if obj is not None and isinstance(obj, dict):
        return obj

    # 退化到宽松字段抽取
    flat = loose_scalar_fields(text)
    picked: dict[str, Any] = {}
    for field, desired in schema.items():
        if field in flat:
            try:
                picked[field] = _coerce(str(flat[field]), desired)
            except Exception:  # noqa: BLE001
                continue
    return picked or None


# ── self-check ───────────────────────────────────────────────────

def _selfcheck() -> None:
    clean = '{"a": 1, "b": 2.5}'
    assert extract_json_object(clean) == {"a": 1, "b": 2.5}
    assert extract_json_object('<text>```json\n{"x": 3}\n``` tail') == {"x": 3}

    # thinking 叙事在前 + 结果 JSON 在后 -> "last" 应取后者
    noisy = ("Let me think: this could be one mechanism {maybe not json at all} "
             'and here is the answer {"a": 7, "b": 8}.')
    assert extract_json_object(noisy, prefer="last") == {"a": 7, "b": 8}

    # 半 JSON / 截断 -> robust_extract 可按 schema 宽松抓
    torn = 'We tried and the prediction was conductivity=1.5 and mobility=150.'
    schema = {"conductivity": float, "mobility": float}
    got = robust_extract(torn, schema)
    assert got and got.get("conductivity") == 1.5 and got.get("mobility") == 150.0

    # 无任何可用字段 -> None
    assert robust_extract("nothing useful here", schema) is None

    # find_json_objects 应取出多个平衡对象, 跳过中间散词
    objs = find_json_objects('{"ok": 1} and some_words {"ok2": 2} tail')
    assert len(objs) == 2 and objs[0] == {"ok": 1} and objs[1] == {"ok2": 2}

    print("OK text_to_json self-check passed (model-agnostic robust extraction)")


if __name__ == "__main__":
    _selfcheck()
