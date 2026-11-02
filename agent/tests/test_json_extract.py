"""B2: json_extract 单一实现 + 五处薄适配的等价性测试.

合并前五套实现策略互不相同; 这里锁住"同一份模型输出在各路径上得到同一结果"
的行为, 避免以后任一路径又长出私有解析.
"""

from __future__ import annotations

import json

import pytest

from huginn.autoloop.plan_check import _extract_json_object as plan_check_extract
from huginn.metacog.text_to_json import (
    extract_json_object,
    find_json_objects,
    strip_fence,
)
from huginn.plugins.autoresearch import AutoresearchTool
from huginn.tools.vision_locate_tool import _extract_json as vision_extract
from huginn.utils.json_extract import (
    extract_json,
    extract_json_text,
    find_json_spans,
    strip_code_fence,
    top_level_spans,
)
from huginn.utils.text_types import TextHandler

# ── 唯一实现 ────────────────────────────────────────────────────


def test_strip_code_fence_prefers_last_block():
    text = '```json\n{"a": 1}\n```\nnoise\n```json\n{"b": 2}\n```'
    assert json.loads(strip_code_fence(text)) == {"b": 2}
    assert json.loads(strip_code_fence(text, prefer="first")) == {"a": 1}
    assert strip_fence(text) == strip_code_fence(text, prefer="last")
    assert strip_code_fence("plain text") == "plain text"


def test_find_json_spans_skips_strings_and_lists_nesting():
    text = 'x {"a": "brace } and { [ quote", "b": [1, 2]} y [3]'
    spans = find_json_spans(text)
    assert [s.opener for s in spans] == ["{", "[", "["]
    outer = top_level_spans(spans)
    assert [s.opener for s in outer] == ["{", "["]


def test_extract_json_prefer_fast_path_and_openers():
    two = '{"a": 1} then {"b": 2}'
    assert extract_json(two, prefer="last") == {"b": 2}
    assert extract_json(two, prefer="first") == {"a": 1}
    assert extract_json("[1, 2]") == [1, 2]
    assert extract_json("[1, 2]", openers=("{",)) is None
    assert extract_json("null") is None
    assert extract_json("") is None
    assert extract_json('{"a": 1') is None


def test_extract_json_handles_braces_inside_strings():
    tricky = '{"code": "def f(): return {}", "n": 1}'
    assert extract_json(tricky) == {"code": "def f(): return {}", "n": 1}
    escaped = '{"q": "he said \\"hi\\" {x}"}'
    assert extract_json(escaped) == {"q": 'he said "hi" {x}'}


def test_extract_json_text_contract():
    assert extract_json_text('{"a": 1}') == ('{"a": 1}', "")
    assert extract_json_text("no json here") == ("", "no json")
    assert extract_json_text('{"a": 1') == ("", "no closing brace")
    blob, err = extract_json_text('noise {"a": 1, "b": 2} tail')
    assert err == "" and json.loads(blob) == {"a": 1, "b": 2}
    # 只定位不校验: 平衡但非法的片段照样返回原文, 由调用方决定
    blob, err = extract_json_text('{"a": }')
    assert err == "" and blob == '{"a": }'
    # 数组里的对象不算目标容器
    blob, err = extract_json_text('[{"a": 1}]', openers=("{",))
    assert blob == "" and err == "no json"


def test_find_json_objects_keeps_dicts_only():
    objs = find_json_objects('{"ok": 1} and some_words {"ok2": 2} tail')
    assert objs == [{"ok": 1}, {"ok2": 2}]
    assert find_json_objects("[1, 2]") == []
    assert find_json_objects('{"broken": }') == []


# ── 五处薄适配的等价性 ──────────────────────────────────────────

_OBJECT_SAMPLES = [
    '{"a": 1}',
    '```json\n{"a": 1}\n```',
    "thinking... {\"a\": 7} answer",
    '{"code": "return {}", "n": 1}',
]


@pytest.mark.parametrize("text", _OBJECT_SAMPLES)
def test_adapters_agree_on_objects(text):
    expected = extract_json(text, prefer="last", openers=("{",))
    assert expected is not None

    assert extract_json_object(text) == expected          # metacog
    assert vision_extract(text) == expected               # vision_locate
    assert TextHandler(text).json() == expected           # text_types

    # plan_check / autoresearch 返回原文, 由调用方 json.loads
    blob, err = plan_check_extract(text)
    assert err == "" and json.loads(blob) == expected

    raw = AutoresearchTool.__new__(AutoresearchTool)._extract_json(text)
    assert json.loads(raw) == expected


_ARRAY = '[{"x1": 1, "y1": 2, "x2": 3, "y2": 4}]'


def test_adapters_keep_their_historical_array_semantics():
    # vision 需要数组 (detect 的元素清单)
    assert vision_extract(_ARRAY) == [{"x1": 1, "y1": 2, "x2": 3, "y2": 4}]
    # text_to_json 面向"对象", 数组里的对象历史上会被取出
    assert extract_json_object(_ARRAY) == {"x1": 1, "y1": 2, "x2": 3, "y2": 4}
    # plan_check 只要对象: 纯数组 -> 跳过校验
    blob, err = plan_check_extract(_ARRAY)
    assert blob == "" and "skip" in err


def test_plan_check_error_messages_preserved():
    _, err = plan_check_extract("no json here")
    assert err == "no json, skip"
    _, err = plan_check_extract('{"is_valid": false')
    assert err == "no closing brace, skip"
