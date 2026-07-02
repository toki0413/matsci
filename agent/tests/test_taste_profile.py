"""Tests for TasteProfile, OnboardingQuestionnaire, and OnboardingTool.

Covers P2 (W1): onboarding questionnaire + taste extraction.
- TestTasteProfile: to_directive() branches, skipped, save/load roundtrip
- TestOnboardingQuestionnaire: 12-question nav, out-of-range, extract voting, partial skip
- TestOnboardingTool: start/answer/skip/get_profile/update_profile
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from huginn.personalization.taste_profile import (
    Imagination,
    OnboardingQuestionnaire,
    ResearchStyle,
    TasteProfile,
    ThinkingMode,
    load_profile,
    save_profile,
)
from huginn.tools.onboarding_tool import (
    OnboardingTool,
    OnboardingToolInput,
    set_shared_questionnaire,
)


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _isolate_questionnaire():
    """每个测试前重置共享问卷单例, 避免跨测试污染."""
    set_shared_questionnaire(None)
    yield
    set_shared_questionnaire(None)


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    """把 HUGINN_CACHE_DIR 指向 tmp_path, 防止污染用户 home 目录."""
    monkeypatch.setenv("HUGINN_CACHE_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def tool():
    return OnboardingTool()


@pytest.fixture
def ctx(tmp_path):
    """ToolContext for tool.call(). 名字用 ctx 不是 context (避 pytest 保留名)."""
    from huginn.types import ToolContext

    return ToolContext(session_id="test", workspace=str(tmp_path))


def _call(tool, args, ctx=None):
    """同步包装 async call, 简化测试."""
    return asyncio.run(tool.call(args, ctx))


# ── TestTasteProfile ────────────────────────────────────────────────


class TestTasteProfile:
    def test_skipped_returns_empty_directive(self):
        p = TasteProfile(completed=False, skipped=True)
        assert p.to_directive() == ""

    def test_default_profile_not_skipped_has_directive(self):
        # 默认 profile (systems/balanced/mixed) 不 skipped, 应有指令
        p = TasteProfile(completed=False, skipped=False)
        d = p.to_directive()
        assert "taste profile" in d
        assert "系统性思考" in d
        assert "平衡的推测" in d
        assert "混合风格" in d

    def test_analytical_intuitive_visual_branches(self):
        for mode, key in [
            (ThinkingMode.ANALYTICAL, "严谨的逻辑分析"),
            (ThinkingMode.INTUITIVE, "直觉式理解"),
            (ThinkingMode.VISUAL, "可视化思维"),
        ]:
            p = TasteProfile(thinking_mode=mode, skipped=False)
            assert key in p.to_directive()

    def test_imagination_branches(self):
        for img, key in [
            (Imagination.CONSERVATIVE, "保守推断"),
            (Imagination.SPECULATIVE, "大胆推测"),
            (Imagination.VISIONARY, "颠覆性想象"),
        ]:
            p = TasteProfile(imagination=img, skipped=False)
            assert key in p.to_directive()

    def test_research_style_branches(self):
        for rs, key in [
            (ResearchStyle.THEORETICAL, "理论驱动"),
            (ResearchStyle.EMPIRICAL, "实验驱动"),
            (ResearchStyle.ENGINEERING, "工程导向"),
        ]:
            p = TasteProfile(research_style=rs, skipped=False)
            assert key in p.to_directive()

    def test_save_load_roundtrip(self, cache_dir):
        p = TasteProfile(
            thinking_mode=ThinkingMode.ANALYTICAL,
            imagination=Imagination.VISIONARY,
            research_style=ResearchStyle.THEORETICAL,
            completed=True,
            skipped=False,
        )
        path = save_profile(p)
        assert path.exists()
        loaded = load_profile()
        assert loaded is not None
        assert loaded.thinking_mode == ThinkingMode.ANALYTICAL
        assert loaded.imagination == Imagination.VISIONARY
        assert loaded.research_style == ResearchStyle.THEORETICAL
        assert loaded.completed is True
        assert loaded.skipped is False

    def test_load_missing_returns_none(self, cache_dir):
        assert load_profile() is None


# ── TestGetTasteDirective (P3) ─────────────────────────────────────


class TestGetTasteDirective:
    """P3: get_taste_directive() 把持久化的 profile 转成 system prompt 片段."""

    def test_no_profile_returns_empty(self, cache_dir):
        from huginn.personalization import get_taste_directive

        assert get_taste_directive() == ""

    def test_skipped_profile_returns_empty(self, cache_dir):
        from huginn.personalization import get_taste_directive

        save_profile(TasteProfile(skipped=True))
        assert get_taste_directive() == ""

    def test_completed_profile_returns_directive(self, cache_dir):
        from huginn.personalization import get_taste_directive

        save_profile(
            TasteProfile(
                thinking_mode=ThinkingMode.ANALYTICAL,
                imagination=Imagination.VISIONARY,
                research_style=ResearchStyle.THEORETICAL,
                completed=True,
                skipped=False,
            )
        )
        d = get_taste_directive()
        assert "taste profile" in d
        assert "严谨的逻辑分析" in d
        assert "颠覆性想象" in d
        assert "理论驱动" in d

    def test_corrupt_json_returns_empty(self, cache_dir):
        from huginn.personalization import get_taste_directive

        # 写一个坏 JSON 到 taste_profile.json
        bad = cache_dir / "taste_profile.json"
        bad.write_text("{not valid json", encoding="utf-8")
        # load_profile 内部 catch 了 JSONDecodeError 返回 None,
        # get_taste_directive 再 catch → 空串
        assert get_taste_directive() == ""

    def test_directive_changes_after_reprofile(self, cache_dir):
        from huginn.personalization import get_taste_directive

        save_profile(
            TasteProfile(thinking_mode=ThinkingMode.ANALYTICAL, skipped=False)
        )
        d1 = get_taste_directive()
        assert "严谨的逻辑分析" in d1

        save_profile(
            TasteProfile(thinking_mode=ThinkingMode.INTUITIVE, skipped=False)
        )
        d2 = get_taste_directive()
        assert "直觉式理解" in d2
        assert "严谨的逻辑分析" not in d2


# ── TestOnboardingQuestionnaire ─────────────────────────────────────


class TestOnboardingQuestionnaire:
    def test_total_questions_is_12(self):
        q = OnboardingQuestionnaire()
        assert q.total_questions == 12
        assert q.answered_count == 0
        assert q.is_complete is False

    def test_get_question_first(self):
        q = OnboardingQuestionnaire()
        first = q.get_question(0)
        assert first is not None
        assert first["index"] == 0
        assert first["id"] == 1
        assert first["dimension"] == "thinking_mode"
        assert len(first["options"]) == 4

    def test_get_question_out_of_range(self):
        q = OnboardingQuestionnaire()
        assert q.get_question(-1) is None
        assert q.get_question(12) is None
        assert q.get_question(99) is None

    def test_submit_answer_invalid_option(self):
        q = OnboardingQuestionnaire()
        assert q.submit_answer(1, -1) is False
        assert q.submit_answer(1, 4) is False
        assert q.submit_answer(1, 0) is True

    def test_submit_answer_invalid_question_id(self):
        q = OnboardingQuestionnaire()
        assert q.submit_answer(999, 0) is False
        assert q.submit_answer(0, 0) is False

    def test_next_index_after_answer(self):
        q = OnboardingQuestionnaire()
        assert q.next_index() == 0
        q.submit_answer(1, 0)  # 答 Q1
        assert q.next_index() == 1
        q.submit_answer(2, 1)  # 答 Q2
        assert q.next_index() == 2

    def test_next_index_none_when_complete(self):
        q = OnboardingQuestionnaire()
        for i in range(1, 13):
            q.submit_answer(i, 0)
        assert q.is_complete is True
        assert q.next_index() is None

    def test_extract_empty_is_skipped(self):
        q = OnboardingQuestionnaire()
        p = q.extract()
        assert p.skipped is True
        assert p.completed is False

    def test_extract_votes_by_majority(self):
        # Q1-Q4 都是 thinking_mode. 答 3 题 analytical (opt 0), 1 题 intuitive (opt 1)
        # analytical 应胜出
        q = OnboardingQuestionnaire()
        q.submit_answer(1, 0)  # analytical
        q.submit_answer(2, 0)  # analytical
        q.submit_answer(3, 0)  # analytical
        q.submit_answer(4, 1)  # intuitive
        q.submit_answer(5, 0)  # conservative
        q.submit_answer(6, 0)  # conservative
        q.submit_answer(7, 0)  # conservative
        q.submit_answer(8, 1)  # balanced
        q.submit_answer(9, 0)  # theoretical
        q.submit_answer(10, 0)  # theoretical
        q.submit_answer(11, 0)  # theoretical
        q.submit_answer(12, 1)  # empirical
        p = q.extract()
        assert p.completed is True
        assert p.thinking_mode == ThinkingMode.ANALYTICAL
        assert p.imagination == Imagination.CONSERVATIVE
        assert p.research_style == ResearchStyle.THEORETICAL

    def test_extract_partial_uses_defaults_for_unanswered_dims(self):
        # 只答 thinking_mode 4 题, 其它两维度未答 -> 用默认值
        q = OnboardingQuestionnaire()
        q.submit_answer(1, 2)  # visual
        q.submit_answer(2, 2)  # visual
        q.submit_answer(3, 2)  # visual
        q.submit_answer(4, 2)  # visual
        p = q.extract()
        assert p.completed is False  # 没答完 12 题
        assert p.thinking_mode == ThinkingMode.VISUAL
        # 未答的维度保持默认
        assert p.imagination == Imagination.BALANCED
        assert p.research_style == ResearchStyle.MIXED


# ── TestOnboardingTool ──────────────────────────────────────────────


class TestOnboardingTool:
    def test_start_returns_first_question(self, tool, ctx, cache_dir):
        result = _call(tool, {"action": "start"}, ctx)
        assert result.success is True
        assert result.data["total_questions"] == 12
        assert result.data["answered"] == 0
        q = result.data["current_question"]
        assert q["id"] == 1
        assert q["dimension"] == "thinking_mode"

    def test_answer_without_start_works(self, tool, ctx, cache_dir):
        # 不先 start 也能直接 answer (共享单例懒创建)
        result = _call(
            tool,
            {"action": "answer", "question_id": 1, "option_index": 0},
            ctx,
        )
        assert result.success is True
        assert result.data["answered"] == 1
        assert result.data["completed"] is False
        assert result.data["next_question"]["id"] == 2

    def test_answer_missing_params(self, tool, ctx, cache_dir):
        result = _call(tool, {"action": "answer"}, ctx)
        assert result.success is False
        assert "question_id" in result.error

    def test_answer_invalid_option(self, tool, ctx, cache_dir):
        result = _call(
            tool,
            {"action": "answer", "question_id": 1, "option_index": 9},
            ctx,
        )
        assert result.success is False
        assert "无效答案" in result.error

    def test_answer_completes_and_saves(self, tool, ctx, cache_dir):
        # 答 12 题, 最后一题应触发存盘
        for i in range(1, 12):
            r = _call(tool, {"action": "answer", "question_id": i, "option_index": 0}, ctx)
            assert r.success is True
            assert r.data["completed"] is False
        # 第 12 题
        r = _call(tool, {"action": "answer", "question_id": 12, "option_index": 0}, ctx)
        assert r.success is True
        assert r.data["completed"] is True
        assert r.data["profile"]["completed"] is True
        # 存盘文件存在
        profile_path = cache_dir / "taste_profile.json"
        assert profile_path.exists()
        data = json.loads(profile_path.read_text(encoding="utf-8"))
        assert data["completed"] is True

    def test_skip_persists_skipped_profile(self, tool, ctx, cache_dir):
        result = _call(tool, {"action": "skip"}, ctx)
        assert result.success is True
        assert result.data["skipped"] is True
        # 存盘的 profile 应该 skipped=True
        loaded = load_profile()
        assert loaded is not None
        assert loaded.skipped is True
        assert loaded.to_directive() == ""

    def test_get_profile_when_none(self, tool, ctx, cache_dir):
        result = _call(tool, {"action": "get_profile"}, ctx)
        assert result.success is True
        assert result.data["exists"] is False

    def test_get_profile_after_completion(self, tool, ctx, cache_dir):
        # 先答完问卷
        for i in range(1, 13):
            _call(tool, {"action": "answer", "question_id": i, "option_index": 0}, ctx)
        # 再查
        result = _call(tool, {"action": "get_profile"}, ctx)
        assert result.success is True
        assert result.data["exists"] is True
        assert result.data["profile"]["completed"] is True
        assert "taste profile" in result.data["directive"]

    def test_update_profile_creates_new(self, tool, ctx, cache_dir):
        result = _call(
            tool,
            {
                "action": "update_profile",
                "thinking_mode": "analytical",
                "imagination": "visionary",
                "research_style": "theoretical",
            },
            ctx,
        )
        assert result.success is True
        assert result.data["updated"] is True
        assert result.data["profile"]["thinking_mode"] == "analytical"
        assert result.data["profile"]["imagination"] == "visionary"
        assert result.data["profile"]["research_style"] == "theoretical"
        assert result.data["profile"]["completed"] is True

    def test_update_profile_invalid_value(self, tool, ctx, cache_dir):
        result = _call(
            tool,
            {"action": "update_profile", "thinking_mode": "bogus"},
            ctx,
        )
        assert result.success is False
        assert "thinking_mode" in result.error

    def test_update_profile_partial_overrides_existing(self, tool, ctx, cache_dir):
        # 先全量建一个
        _call(
            tool,
            {"action": "update_profile", "thinking_mode": "analytical", "imagination": "balanced"},
            ctx,
        )
        # 再只改一个维度
        result = _call(
            tool,
            {"action": "update_profile", "research_style": "empirical"},
            ctx,
        )
        assert result.success is True
        assert result.data["profile"]["thinking_mode"] == "analytical"  # 保留
        assert result.data["profile"]["research_style"] == "empirical"  # 改了

    def test_unknown_action_rejected_by_schema(self, tool, ctx, cache_dir):
        # Literal 校验在 Pydantic 层直接拒绝, 不进 call() 主体.
        # 这与 git_tool 等约定一致: schema 错误上抛, 不吞成 ToolResult.
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            _call(tool, {"action": "bogus"}, ctx)
