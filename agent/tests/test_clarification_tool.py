"""Tests for clarification_tool — ask/confirm_* 回归 + socratic_probes/decision_tree 新 action.

用 FakeClarificationManager 注入, 不依赖全局单例和 asyncio.Future.
feature flag 用 autouse fixture 强开, 避免 clarification 被默认关掉.
"""

from __future__ import annotations

import asyncio

import pytest
from pydantic import ValidationError

from huginn.tools.clarification_tool import ClarificationInput, ClarificationTool


# ── Fake manager ─────────────────────────────────────────────────────


class FakeClarificationManager:
    """按预设答案队列回答, 不走 asyncio.Future. 队列空了返回 default_answer."""

    def __init__(self, answers=None):
        self._answers = list(answers or [])
        self.asked: list[str] = []

    async def ask(
        self,
        thread_id="",
        question="",
        options=None,
        context="",
        default_answer="",
        timeout=None,
        metadata=None,
    ):
        self.asked.append(question)
        if self._answers:
            return self._answers.pop(0)
        return default_answer

    def should_ask(self, question_type, context=None):
        return True


@pytest.fixture(autouse=True)
def _enable_clarification(monkeypatch):
    """强制 clarification feature flag 开启, 否则 call() 会直接走 default 返回."""
    try:
        from huginn.feature_flags import FeatureFlags

        class _FakeFlags:
            def is_enabled(self, name):
                return True

        monkeypatch.setattr(FeatureFlags, "shared", lambda: _FakeFlags())
    except Exception:
        # FeatureFlags 不可用时, tool 内部的 try/except 也会跳过检查
        pass


# ── 回归: 原 4 个 action ─────────────────────────────────────────────


def test_ask_action_still_works():
    fake = FakeClarificationManager(answers=["用 PBE"])
    tool = ClarificationTool(manager=fake)
    args = ClarificationInput(action="ask", question="用 PBE 还是 HSE06?")
    result = asyncio.run(tool.call(args, context=None))
    assert result.success is True
    assert result.data["answer"] == "用 PBE"


def test_confirm_destructive_still_works():
    fake = FakeClarificationManager(answers=["取消"])
    tool = ClarificationTool(manager=fake)
    args = ClarificationInput(action="confirm_destructive", question="删除 POSCAR")
    result = asyncio.run(tool.call(args, context=None))
    assert result.success is True
    assert result.data["answer"] == "取消"


# ── socratic_probes ──────────────────────────────────────────────────


def test_socratic_probes_runs_all_probes():
    """3 个探针, Fake 给 3 答案, 验证全部问完且 answers 有 3 项."""
    fake = FakeClarificationManager(answers=["目标", "PBE", "520eV"])
    tool = ClarificationTool(manager=fake)
    args = ClarificationInput(
        action="socratic_probes",
        probes=["优化目标是什么?", "用什么泛函?", "ENCUT 取多少?"],
    )
    result = asyncio.run(tool.call(args, context=None))
    assert result.success is True
    assert len(result.data["answers"]) == 3
    assert len(fake.asked) == 3


def test_socratic_probes_returns_answers_dict():
    """验证 answers 是 {probe: answer} 映射, 不是列表."""
    fake = FakeClarificationManager(answers=["a1", "a2"])
    tool = ClarificationTool(manager=fake)
    args = ClarificationInput(
        action="socratic_probes",
        probes=["q1", "q2"],
    )
    result = asyncio.run(tool.call(args, context=None))
    assert result.data["answers"] == {"q1": "a1", "q2": "a2"}


def test_socratic_probes_missing_probes_raises():
    """model_validator: action=socratic_probes 但 probes=[] 应报错."""
    with pytest.raises(ValidationError):
        ClarificationInput(action="socratic_probes", probes=[])


def test_socratic_probes_too_many_raises():
    """model_validator: 超过 5 个探针应报错."""
    with pytest.raises(ValidationError):
        ClarificationInput(
            action="socratic_probes",
            probes=["q1", "q2", "q3", "q4", "q5", "q6"],
        )


# ── decision_tree ────────────────────────────────────────────────────


def test_decision_tree_traverses_to_leaf():
    """3 节点树 root→func→pbe_leaf, Fake 给 2 答案, 验证走到叶子."""
    fake = FakeClarificationManager(answers=["DFT", "PBE"])
    tool = ClarificationTool(manager=fake)
    tree = {
        "root": {
            "question": "DFT 还是 ML 势?",
            "options": {"DFT": "func", "ML 势": "ml_leaf"},
        },
        "func": {
            "question": "用什么泛函?",
            "options": {"PBE": "pbe_leaf", "HSE06": "hse_leaf"},
        },
        "pbe_leaf": {"leaf": True, "result": "PBE relaxation"},
        "hse_leaf": {"leaf": True, "result": "HSE06 relaxation"},
        "ml_leaf": {"leaf": True, "result": "ML potential"},
    }
    args = ClarificationInput(
        action="decision_tree",
        tree_nodes=tree,
        start_node="root",
    )
    result = asyncio.run(tool.call(args, context=None))
    assert result.success is True
    assert result.data["final_result"] == "PBE relaxation"
    assert result.data["tree_path"] == ["root", "func", "pbe_leaf"]


def test_decision_tree_missing_start_node_raises():
    """model_validator: start_node 不在 tree_nodes 里应报错."""
    with pytest.raises(ValidationError):
        ClarificationInput(
            action="decision_tree",
            tree_nodes={"root": {"leaf": True, "result": "x"}},
            start_node="nonexistent",
        )


def test_decision_tree_cycle_guard():
    """构造环 A→B→A, 验证 visited set 检测到环后退出, 不死循环."""
    fake = FakeClarificationManager(
        answers=["go to B", "go to A", "go to B", "go to A"]
    )
    tool = ClarificationTool(manager=fake)
    tree = {
        "A": {"question": "A?", "options": {"go to B": "B"}},
        "B": {"question": "B?", "options": {"go to A": "A"}},
    }
    args = ClarificationInput(
        action="decision_tree",
        tree_nodes=tree,
        start_node="A",
    )
    result = asyncio.run(tool.call(args, context=None))
    assert result.success is True
    # A → B → (A 已访问, 退出)
    assert result.data["tree_path"] == ["A", "B"]


def test_decision_tree_unknown_option_falls_back():
    """用户给了一个不在 options 里的答案, 应走 default_answer 对应的边."""
    fake = FakeClarificationManager(answers=["maybe"])  # "maybe" 不在 options
    tool = ClarificationTool(manager=fake)
    tree = {
        "root": {
            "question": "选哪个?",
            "options": {"A": "leaf_a", "B": "leaf_b"},
        },
        "leaf_a": {"leaf": True, "result": "got A"},
        "leaf_b": {"leaf": True, "result": "got B"},
    }
    args = ClarificationInput(
        action="decision_tree",
        tree_nodes=tree,
        start_node="root",
        default_answer="B",
    )
    result = asyncio.run(tool.call(args, context=None))
    assert result.success is True
    assert result.data["final_result"] == "got B"
