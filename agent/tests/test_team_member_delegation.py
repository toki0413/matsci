"""Tests for ModelTeam._run_member vision delegation fix.

核心缺陷: CV_TOOLS 场景委托 VISION member 时, _run_member 只把文本任务传给
agent.chat(), image_path 没透传 → 多模态模型只能看到"图片路径"字符串, 无法
真正看图. 修复后 image_path 应原样透传给 agent.chat(image_path=...).
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from huginn.agents.team import ModelTeam, TeamMember, TeamRole
from huginn.models.registry import ModelCaps


class _RecordingChat:
    """记录 chat() 调用的 async generator 假实现."""

    def __init__(self, messages: list | None = None):
        self.calls: list[tuple] = []
        self.kwargs_list: list[dict] = []
        self._messages = messages or []

    async def __call__(self, *args, **kwargs):
        self.calls.append(args)
        self.kwargs_list.append(kwargs)
        yield {"messages": self._messages}

    @property
    def call_count(self) -> int:
        return len(self.calls)

    @property
    def last_kwargs(self) -> dict:
        return self.kwargs_list[-1] if self.kwargs_list else {}


def _make_member(role: TeamRole, chat: _RecordingChat) -> TeamMember:
    """构造一个绑定了 fake agent 的成员, 绕过 get_agent 的真 agent 创建."""
    agent = MagicMock()
    agent.chat = chat
    member = TeamMember(
        name=f"{role.value}-test",
        profile_id="test",
        role=role,
        model_name="qwen2.5-vl",
        caps=ModelCaps(vision=True, tools=False, reasoning=False, streaming=True),
    )
    member._agent = agent  # 直接注入 fake agent, 不触发 from_config
    return member


def _run_member(member: TeamMember, task: str, ctx: dict) -> str:
    """pack _run_member 为可同步等待的入口."""
    team = ModelTeam([])
    traces: list = []
    return asyncio.run(team._run_member(member, task, ctx, traces))


class TestRunMemberImagePathPassthrough:
    def test_image_path_passed_to_chat(self):
        """ctx 带 image_path → agent.chat() 收到 image_path 参数."""
        chat = _RecordingChat()
        member = _make_member(TeamRole.VISION, chat)

        result = _run_member(
            member, "describe this image", {"image_path": "/data/sem_001.png"}
        )

        assert result == ""
        assert chat.call_count == 1
        kwargs = chat.last_kwargs
        assert kwargs.get("image_path") == "/data/sem_001.png"
        assert kwargs.get("thread_id") == "team-vision"

    def test_no_image_path_omits_param(self):
        """ctx 无 image_path → 不传 image_path 参数 (走纯文本旧路径)."""
        chat = _RecordingChat()
        member = _make_member(TeamRole.VISION, chat)

        _run_member(member, "describe", {"original_task": "x"})

        assert chat.call_count == 1
        assert "image_path" not in chat.last_kwargs

    def test_final_output_extracted_from_messages(self):
        """_run_member 仍能从 stream 的 messages 里提取最终文本."""
        msg = MagicMock()
        msg.content = "这是 SEM 图像, 平均粒径 120nm"
        chat = _RecordingChat(messages=[msg])
        member = _make_member(TeamRole.VISION, chat)

        result = _run_member(
            member, "describe", {"image_path": "/data/tem_001.tif"}
        )
        assert "120nm" in result
