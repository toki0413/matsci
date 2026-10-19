"""离线契约门禁: HUGINN_DISABLE_WEB_SEARCH=1 时 literature 层不得发起真实网络请求.

锁定"离线即真离线"这一不变量, 防止再次出现"只在部分调用点设守卫、内部路径
静默联网"的回归 —— 该缺陷曾让 CI 的 test job 因真实网络扇出而挂起.

门禁落在唯一网络漏斗 `_OPENER.open`: 无论走 `_http_request` 还是各源函数里
直接调用 (XML/atom/分页抓取), 都必须被拦. 全部本地可跑, 不打真实 API.
"""

from __future__ import annotations

import pytest

from huginn.core_types import ToolContext
from huginn.tools.literature._http import (
    _DISABLED_HINT,
    _OPENER,
    HttpError,
    _http_get_json,
)
from huginn.tools.literature.tool import LiteratureInput, LiteratureTool


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HUGINN_DISABLE_WEB_SEARCH", "1")


def test_raw_opener_blocked_offline() -> None:
    """直接 _OPENER.open (各源函数走的就是这条) 必须抛 HttpError, 不触网."""
    with pytest.raises(HttpError):
        _OPENER.open("https://example.com")


@pytest.mark.asyncio
async def test_http_request_funnel_blocked_offline() -> None:
    """经 _http_request -> _fetch_sync 的常规抓取路径同样被拦."""
    with pytest.raises(HttpError):
        await _http_get_json("https://example.com")


@pytest.mark.asyncio
async def test_do_search_blocked_offline() -> None:
    """tool 级 search 在离线时返回失败并带 hint, 不进入 15 源扇出."""
    tool = LiteratureTool()
    res = await tool._do_search(
        LiteratureInput(action="search", query="li2o band gap"),
        ToolContext(session_id="offline", workspace="."),
    )
    assert res.success is False
    assert res.error == _DISABLED_HINT
