"""autoloop 的 LLM 调用必须走统一重试层.

回归 run74: `engine_act._llm_chat` 此前绕开 `huginn.llm_retry`, 一次 provider
限流 (HTTP 400 + code=-20048 + "请求过于频繁") 就被当成阶段无产出, 经
redirect→pivot 清空状态 → "no hyp to pivot from" 停机, 全程 0 工具调用.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from huginn.autoloop import engine_act as ea


class _BadRequestError(Exception):
    """通用坏请求异常 (类名不含 rate, 只能靠状态码/文案识别)."""


class _ScriptedLLM:
    """按脚本依次返回/抛出的假模型; 无 astream → 走非流式 ainvoke 路径."""

    def __init__(self, seq):
        self._seq = list(seq)
        self.calls = 0

    async def ainvoke(self, messages):  # noqa: ANN001
        item = self._seq[self.calls]
        self.calls += 1
        if isinstance(item, BaseException):
            raise item
        return SimpleNamespace(content=item, usage_metadata={"total": 1})


class _FakeEngine:
    def __init__(self, llm):
        self.model = llm
        self._current_phase = ""
        self.usage = []

    async def _track_llm_usage(self, meta):  # noqa: ANN001
        self.usage.append(meta)


def _provider_rate_limit() -> _BadRequestError:
    exc = _BadRequestError(
        "Error code: 400 - {'error': {'code': '-20048', "
        "'message': '请求过于频繁，请稍后再试'}}"
    )
    exc.status_code = 400
    return exc


@pytest.mark.asyncio
async def test_llm_chat_retries_provider_rate_limit(monkeypatch):
    import huginn.llm_retry as lr

    monkeypatch.setattr(lr, "_sleep_with_log", AsyncMock())
    llm = _ScriptedLLM([_provider_rate_limit(), "假设文本"])
    act = ea.EngineAct(_FakeEngine(llm))

    out = await act._llm_chat("prompt")

    assert out == "假设文本"
    assert llm.calls == 2  # 第一次限流被重试, 第二次成功


@pytest.mark.asyncio
async def test_llm_chat_exhausts_retries_then_raises(monkeypatch):
    import huginn.llm_retry as lr

    monkeypatch.setattr(lr, "_sleep_with_log", AsyncMock())
    llm = _ScriptedLLM([_provider_rate_limit() for _ in range(10)])
    act = ea.EngineAct(_FakeEngine(llm))

    with pytest.raises(_BadRequestError):
        await act._llm_chat("prompt")


@pytest.mark.asyncio
async def test_llm_chat_non_retryable_raises_immediately(monkeypatch):
    import huginn.llm_retry as lr

    sleep = AsyncMock()
    monkeypatch.setattr(lr, "_sleep_with_log", sleep)
    llm = _ScriptedLLM([ValueError("bad input")])
    act = ea.EngineAct(_FakeEngine(llm))

    with pytest.raises(ValueError):
        await act._llm_chat("prompt")

    assert llm.calls == 1  # 不可重试 → 不重试
    sleep.assert_not_called()
