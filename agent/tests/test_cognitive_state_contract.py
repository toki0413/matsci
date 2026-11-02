"""C1 门禁: ``cog`` 的字面量键必须落在 ``CognitiveState`` 声明内.

``cog`` 是 run_cognitive 各 phase 之间传递的中间结果. 此前注解为裸 ``dict``,
键名拼错 (如 ``cog["validaton"]``) 只会在运行时静默返回 None. 本测试把
改/读点用到的字面量键与 TypedDict 声明对齐, 新增键必须先进 ``CognitiveState``.
"""

from __future__ import annotations

import re
from pathlib import Path

from huginn.autoloop.types import CognitiveState

_HUGINN_ROOT = Path(__file__).resolve().parents[1] / "huginn"

#: 使用 cog 的模块 (C1 收敛范围).
_COG_MODULES = (
    "autoloop/cognitive_loop.py",
    "autoloop/engine.py",
    "autoloop/engine_reflect.py",
    "autoloop/cognitive_checks.py",
    "metacog/three_cabin_reflector.py",
)

#: ``cog["key"]`` / ``cog.get("key")`` / ``cog.pop|setdefault("key")``.
_COG_KEY_RE = re.compile(
    r"\bcog\s*(?:\[\s*[\"']([a-z_]+)[\"']\s*\]"
    r"|\.(?:get|pop|setdefault)\(\s*[\"']([a-z_]+)[\"'])"
)


def _declared_keys() -> set[str]:
    return set(CognitiveState.__annotations__)


def test_cog_literal_keys_are_declared():
    declared = _declared_keys()
    for rel in _COG_MODULES:
        text = (_HUGINN_ROOT / rel).read_text(encoding="utf-8")
        used = {m.group(1) or m.group(2) for m in _COG_KEY_RE.finditer(text)}
        undeclared = used - declared
        assert not undeclared, (
            f"{rel} 使用了未声明的 cog 键: {sorted(undeclared)}; "
            "请先在 huginn/autoloop/types.py::CognitiveState 补声明"
        )


def test_rc_new_cog_keys_are_declared():
    """初始化 cog 的键必须全部已声明, 且覆盖 8 个 phase 骨架槽."""
    from huginn.autoloop.cognitive_loop import CognitiveRunner

    keys = set(CognitiveRunner._rc_new_cog())
    assert keys <= _declared_keys()
    assert {
        "context",
        "hypothesis",
        "plan",
        "execution_result",
        "validation",
        "current_hyp_id",
        "phases",
        "completed_steps",
    } <= keys
