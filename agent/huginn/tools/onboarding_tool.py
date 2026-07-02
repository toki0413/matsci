"""onboarding_tool — 首次使用时的 taste 问卷工具.

agent 调它带用户走 12 题问卷, 提取 TasteProfile, 持久化到本地 JSON.
可跳过, 跳过时 to_directive() 返回空串, 不注入 system prompt.
问卷进度走进程内共享单例, 跨 tool 调用保持连续.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.personalization.taste_profile import (
    Imagination,
    OnboardingQuestionnaire,
    ResearchStyle,
    TasteProfile,
    ThinkingMode,
    load_profile,
    save_profile,
)
from huginn.phases import ResearchPhase
from huginn.tools.base import HuginnTool
from huginn.tools.profile import ToolProfile
from huginn.types import ToolContext, ToolResult


class OnboardingToolInput(BaseModel):
    action: Literal["start", "answer", "skip", "get_profile", "update_profile"] = Field(
        default="start",
        description=(
            "start: 开始/重启问卷, 返回第一题; "
            "answer: 提交一题答案, 返回下一题; "
            "skip: 跳过问卷, 用默认行为; "
            "get_profile: 查已保存的 profile; "
            "update_profile: 直接覆盖某维度, 不走问卷"
        ),
    )
    question_id: int | None = Field(default=None, description="answer 用: 题号 1-12")
    option_index: int | None = Field(default=None, description="answer 用: 选项 0-3")
    thinking_mode: str | None = Field(
        default=None, description="update_profile 用: analytical/intuitive/visual/systems"
    )
    imagination: str | None = Field(
        default=None, description="update_profile 用: conservative/balanced/speculative/visionary"
    )
    research_style: str | None = Field(
        default=None, description="update_profile 用: theoretical/empirical/engineering/mixed"
    )


# 进程内共享问卷单例, 跨 tool 调用保持答题进度.
# 测试可通过 set_shared_questionnaire 注入干净实例.
_shared_questionnaire: OnboardingQuestionnaire | None = None


def get_shared_questionnaire() -> OnboardingQuestionnaire:
    global _shared_questionnaire
    if _shared_questionnaire is None:
        _shared_questionnaire = OnboardingQuestionnaire()
    return _shared_questionnaire


def set_shared_questionnaire(q: OnboardingQuestionnaire | None) -> None:
    global _shared_questionnaire
    _shared_questionnaire = q


class OnboardingTool(HuginnTool):
    """带用户走 taste 问卷, 提取并持久化 TasteProfile."""

    name = "onboarding_tool"
    category = "meta"
    description = (
        "首次使用时的 taste 问卷. 12 题提取思维模式/想象力/研究风格, "
        "可跳过. 完成后 profile 注入 system prompt 定制 agent 行为."
    )
    input_schema = OnboardingToolInput
    profile = ToolProfile(
        cost_tier="light",
        phases=frozenset({ResearchPhase.OPEN}),
    )

    async def call(
        self, args: dict[str, Any], context: ToolContext | None = None
    ) -> ToolResult:
        input_data = OnboardingToolInput(**args)
        q = get_shared_questionnaire()
        try:
            if input_data.action == "start":
                return self._start(q)
            if input_data.action == "answer":
                return self._answer(q, input_data)
            if input_data.action == "skip":
                return self._skip()
            if input_data.action == "get_profile":
                return self._get_profile()
            if input_data.action == "update_profile":
                return self._update_profile(input_data)
            return ToolResult(
                data=None, success=False, error=f"未知 action: {input_data.action}"
            )
        except Exception as exc:
            return ToolResult(data=None, success=False, error=f"Onboarding tool failed: {exc}")

    # ── action 实现 ──────────────────────────────────────────────

    def _start(self, q: OnboardingQuestionnaire) -> ToolResult:
        # start 显式重置, 从头开始. 已答的进度清掉.
        fresh = OnboardingQuestionnaire()
        set_shared_questionnaire(fresh)
        first = fresh.get_question(0)
        return ToolResult(
            data={
                "total_questions": fresh.total_questions,
                "answered": fresh.answered_count,
                "current_question": first,
                "message": "问卷开始. 用 action=answer 提交答案 (question_id + option_index), 或 action=skip 跳过.",
            },
            success=True,
        )

    def _answer(
        self, q: OnboardingQuestionnaire, input_data: OnboardingToolInput
    ) -> ToolResult:
        if input_data.question_id is None or input_data.option_index is None:
            return ToolResult(
                data=None,
                success=False,
                error="answer 需要 question_id (1-12) 和 option_index (0-3)",
            )
        ok = q.submit_answer(input_data.question_id, input_data.option_index)
        if not ok:
            return ToolResult(
                data=None,
                success=False,
                error=(
                    f"无效答案: question_id={input_data.question_id}, "
                    f"option_index={input_data.option_index}"
                ),
            )

        if q.is_complete:
            profile = q.extract()
            save_profile(profile)
            return ToolResult(
                data={
                    "answered": q.answered_count,
                    "total": q.total_questions,
                    "completed": True,
                    "profile": _profile_to_dict(profile),
                    "message": "问卷完成, profile 已保存.",
                },
                success=True,
            )

        next_idx = q.next_index()
        next_q = q.get_question(next_idx) if next_idx is not None else None
        return ToolResult(
            data={
                "answered": q.answered_count,
                "total": q.total_questions,
                "completed": False,
                "next_question": next_q,
            },
            success=True,
        )

    def _skip(self) -> ToolResult:
        # 跳过: 标记 skipped, 存盘. to_directive() 会返回空串.
        profile = TasteProfile(completed=False, skipped=True)
        save_profile(profile)
        # 重置内存问卷, 避免残留半截进度
        set_shared_questionnaire(OnboardingQuestionnaire())
        return ToolResult(
            data={
                "skipped": True,
                "message": "已跳过问卷. agent 用默认行为, 不注入 taste 指令.",
            },
            success=True,
        )

    def _get_profile(self) -> ToolResult:
        loaded = load_profile()
        if loaded is None:
            return ToolResult(
                data={
                    "exists": False,
                    "message": "尚未完成问卷. 用 action=start 开始, 或 action=skip 跳过.",
                },
                success=True,
            )
        return ToolResult(
            data={
                "exists": True,
                "profile": _profile_to_dict(loaded),
                "directive": loaded.to_directive(),
            },
            success=True,
        )

    def _update_profile(self, input_data: OnboardingToolInput) -> ToolResult:
        # 直接覆盖维度, 不走问卷. 已有 profile 就在其上改, 没有就新建.
        loaded = load_profile() or TasteProfile(completed=True, skipped=False)

        if input_data.thinking_mode:
            try:
                loaded.thinking_mode = ThinkingMode(input_data.thinking_mode)
            except ValueError:
                return ToolResult(
                    data=None,
                    success=False,
                    error=f"无效 thinking_mode: {input_data.thinking_mode}",
                )
        if input_data.imagination:
            try:
                loaded.imagination = Imagination(input_data.imagination)
            except ValueError:
                return ToolResult(
                    data=None,
                    success=False,
                    error=f"无效 imagination: {input_data.imagination}",
                )
        if input_data.research_style:
            try:
                loaded.research_style = ResearchStyle(input_data.research_style)
            except ValueError:
                return ToolResult(
                    data=None,
                    success=False,
                    error=f"无效 research_style: {input_data.research_style}",
                )

        # 手动更新视为完成, 撤销 skipped 标记
        loaded.completed = True
        loaded.skipped = False
        save_profile(loaded)
        return ToolResult(
            data={"updated": True, "profile": _profile_to_dict(loaded)},
            success=True,
        )


def _profile_to_dict(p: TasteProfile) -> dict:
    return {
        "thinking_mode": p.thinking_mode.value,
        "imagination": p.imagination.value,
        "research_style": p.research_style.value,
        "completed": p.completed,
        "skipped": p.skipped,
    }
