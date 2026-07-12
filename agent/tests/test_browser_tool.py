"""Tests for BrowserTool — validation, mock mode, and action handling."""

from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, patch

import pytest

from huginn.tools.browser_tool import BrowserAction, BrowserTool, BrowserToolInput
from huginn.types import ToolContext, ValidationResult


def _ctx() -> ToolContext:
    return ToolContext(session_id="test", workspace=".")


def _run(coro):
    """Run an async coroutine synchronously."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is None:
        return asyncio.run(coro)
    return coro


@pytest.fixture
def tool():
    return BrowserTool()


# ── Tool creation ────────────────────────────────────────────────────


class TestBrowserToolCreation:
    def test_tool_name(self, tool):
        assert tool.name == "browser_tool"

    def test_tool_description(self, tool):
        assert "browser" in tool.description.lower()

    def test_tool_has_input_schema(self, tool):
        assert tool.input_schema is BrowserToolInput

    def test_browser_initially_none(self, tool):
        assert tool._browser is None
        assert tool._page is None


# ── BrowserAction enum ──────────────────────────────────────────────


class TestBrowserActionEnum:
    def test_all_actions_present(self):
        expected = {"navigate", "click", "type", "fill_form", "scroll",
                    "screenshot", "extract", "state", "login", "wait", "close"}
        actual = {a.value for a in BrowserAction}
        assert actual == expected

    def test_action_is_string_enum(self):
        assert isinstance(BrowserAction.NAVIGATE, str)
        assert BrowserAction.NAVIGATE == "navigate"


# ── Input validation ─────────────────────────────────────────────────


class TestValidateInput:
    @pytest.mark.asyncio
    async def test_navigate_requires_url(self, tool):
        args = BrowserToolInput(action=BrowserAction.NAVIGATE)
        result = await tool.validate_input(args, _ctx())
        assert result.result is False
        assert "URL required" in result.message

    @pytest.mark.asyncio
    async def test_navigate_with_url(self, tool):
        args = BrowserToolInput(action=BrowserAction.NAVIGATE, url="https://example.com")
        result = await tool.validate_input(args, _ctx())
        assert result.result is True

    @pytest.mark.asyncio
    async def test_click_requires_selector(self, tool):
        args = BrowserToolInput(action=BrowserAction.CLICK)
        result = await tool.validate_input(args, _ctx())
        assert result.result is False
        assert "Selector required" in result.message

    @pytest.mark.asyncio
    async def test_type_requires_selector(self, tool):
        args = BrowserToolInput(action=BrowserAction.TYPE, text="hello")
        result = await tool.validate_input(args, _ctx())
        assert result.result is False
        assert "Selector required" in result.message

    @pytest.mark.asyncio
    async def test_type_requires_text(self, tool):
        args = BrowserToolInput(action=BrowserAction.TYPE, selector="#input")
        result = await tool.validate_input(args, _ctx())
        assert result.result is False
        assert "Text required" in result.message

    @pytest.mark.asyncio
    async def test_type_valid(self, tool):
        args = BrowserToolInput(action=BrowserAction.TYPE, selector="#input", text="hello")
        result = await tool.validate_input(args, _ctx())
        assert result.result is True

    @pytest.mark.asyncio
    async def test_login_requires_url(self, tool):
        args = BrowserToolInput(
            action=BrowserAction.LOGIN,
            credentials={"username": "user", "password": "pass"},
        )
        result = await tool.validate_input(args, _ctx())
        assert result.result is False
        assert "URL required" in result.message

    @pytest.mark.asyncio
    async def test_login_requires_credentials(self, tool):
        args = BrowserToolInput(action=BrowserAction.LOGIN, url="https://example.com/login")
        result = await tool.validate_input(args, _ctx())
        assert result.result is False
        assert "Username and password" in result.message

    @pytest.mark.asyncio
    async def test_login_valid(self, tool):
        args = BrowserToolInput(
            action=BrowserAction.LOGIN,
            url="https://example.com/login",
            credentials={"username": "user", "password": "pass"},
        )
        result = await tool.validate_input(args, _ctx())
        assert result.result is True

    @pytest.mark.asyncio
    async def test_fill_form_requires_data(self, tool):
        args = BrowserToolInput(action=BrowserAction.FILL_FORM)
        result = await tool.validate_input(args, _ctx())
        assert result.result is False
        assert "form_data required" in result.message

    @pytest.mark.asyncio
    async def test_fill_form_valid(self, tool):
        args = BrowserToolInput(
            action=BrowserAction.FILL_FORM,
            form_data={"#name": "Alice", "#email": "alice@example.com"},
        )
        result = await tool.validate_input(args, _ctx())
        assert result.result is True

    @pytest.mark.asyncio
    async def test_url_must_have_scheme(self, tool):
        args = BrowserToolInput(action=BrowserAction.NAVIGATE, url="ftp://bad.com")
        result = await tool.validate_input(args, _ctx())
        assert result.result is False
        assert "http://" in result.message or "https://" in result.message

    @pytest.mark.asyncio
    async def test_http_url_ok(self, tool):
        args = BrowserToolInput(action=BrowserAction.NAVIGATE, url="http://example.com")
        result = await tool.validate_input(args, _ctx())
        assert result.result is True

    @pytest.mark.asyncio
    async def test_screenshot_no_validation_error(self, tool):
        args = BrowserToolInput(action=BrowserAction.SCREENSHOT)
        result = await tool.validate_input(args, _ctx())
        assert result.result is True

    @pytest.mark.asyncio
    async def test_scroll_valid(self, tool):
        args = BrowserToolInput(action=BrowserAction.SCROLL, direction="up", scroll_amount=300)
        result = await tool.validate_input(args, _ctx())
        assert result.result is True

    @pytest.mark.asyncio
    async def test_wait_valid(self, tool):
        args = BrowserToolInput(action=BrowserAction.WAIT, wait_ms=500)
        result = await tool.validate_input(args, _ctx())
        assert result.result is True

    @pytest.mark.asyncio
    async def test_close_valid(self, tool):
        args = BrowserToolInput(action=BrowserAction.CLOSE)
        result = await tool.validate_input(args, _ctx())
        assert result.result is True

    @pytest.mark.asyncio
    async def test_extract_valid(self, tool):
        args = BrowserToolInput(action=BrowserAction.EXTRACT, extract_selector=".item")
        result = await tool.validate_input(args, _ctx())
        assert result.result is True


# ── Mock mode (no Playwright) ────────────────────────────────────────


class TestMockMode:
    """When Playwright is not installed, tool returns mock results."""

    @pytest.fixture(autouse=True)
    def _force_mock_mode(self):
        """Patch _ensure_browser to return False, simulating no Playwright."""
        with patch.object(BrowserTool, "_ensure_browser", new_callable=AsyncMock, return_value=False):
            yield

    @pytest.mark.asyncio
    async def test_navigate_mock(self, tool):
        args = BrowserToolInput(action=BrowserAction.NAVIGATE, url="https://example.com")
        result = await tool.call(args, _ctx())
        assert result.success is True
        assert result.data["status"] == "mock"
        assert result.data["action"] == "navigate"
        assert "url" in result.data

    @pytest.mark.asyncio
    async def test_extract_mock(self, tool):
        args = BrowserToolInput(action=BrowserAction.EXTRACT)
        result = await tool.call(args, _ctx())
        assert result.success is True
        assert result.data["status"] == "mock"
        assert "elements" in result.data
        assert result.data["count"] == 2

    @pytest.mark.asyncio
    async def test_login_mock(self, tool):
        args = BrowserToolInput(
            action=BrowserAction.LOGIN,
            url="https://example.com/login",
            credentials={"username": "user", "password": "pass"},
        )
        result = await tool.call(args, _ctx())
        assert result.success is True
        assert result.data["status"] == "mock"
        assert "login_url" in result.data

    @pytest.mark.asyncio
    async def test_fill_form_mock(self, tool):
        args = BrowserToolInput(
            action=BrowserAction.FILL_FORM,
            form_data={"#name": "Alice"},
        )
        result = await tool.call(args, _ctx())
        assert result.success is True
        assert result.data["status"] == "mock"
        assert "fields_filled" in result.data

    @pytest.mark.asyncio
    async def test_click_mock(self, tool):
        args = BrowserToolInput(action=BrowserAction.CLICK, selector="#btn")
        result = await tool.call(args, _ctx())
        assert result.success is True
        assert result.data["status"] == "mock"
        assert result.data["action"] == "click"

    @pytest.mark.asyncio
    async def test_type_mock(self, tool):
        args = BrowserToolInput(action=BrowserAction.TYPE, selector="#input", text="hi")
        result = await tool.call(args, _ctx())
        assert result.success is True
        assert result.data["status"] == "mock"

    @pytest.mark.asyncio
    async def test_scroll_mock(self, tool):
        args = BrowserToolInput(action=BrowserAction.SCROLL, direction="down")
        result = await tool.call(args, _ctx())
        assert result.success is True
        assert result.data["status"] == "mock"
        assert result.data["action"] == "scroll"

    @pytest.mark.asyncio
    async def test_screenshot_mock(self, tool):
        args = BrowserToolInput(action=BrowserAction.SCREENSHOT)
        result = await tool.call(args, _ctx())
        assert result.success is True
        assert result.data["status"] == "mock"

    @pytest.mark.asyncio
    async def test_wait_mock(self, tool):
        args = BrowserToolInput(action=BrowserAction.WAIT, wait_ms=0)
        result = await tool.call(args, _ctx())
        assert result.success is True
        assert result.data["status"] == "mock"

    @pytest.mark.asyncio
    async def test_close_mock(self, tool):
        args = BrowserToolInput(action=BrowserAction.CLOSE)
        result = await tool.call(args, _ctx())
        assert result.success is True
        assert result.data["status"] == "mock"


# ── BrowserToolInput defaults ────────────────────────────────────────


class TestBrowserToolInputDefaults:
    def test_default_direction(self):
        inp = BrowserToolInput(action=BrowserAction.SCROLL)
        assert inp.direction == "down"

    def test_default_scroll_amount(self):
        inp = BrowserToolInput(action=BrowserAction.SCROLL)
        assert inp.scroll_amount == 500

    def test_default_wait_ms(self):
        inp = BrowserToolInput(action=BrowserAction.WAIT)
        assert inp.wait_ms == 1000

    def test_default_headless(self):
        inp = BrowserToolInput(action=BrowserAction.NAVIGATE, url="https://x.com")
        assert inp.headless is True

    def test_form_data_default_empty(self):
        inp = BrowserToolInput(action=BrowserAction.FILL_FORM)
        assert inp.form_data == {}

    def test_credentials_default_empty(self):
        inp = BrowserToolInput(action=BrowserAction.LOGIN)
        assert inp.credentials == {}


# ── Scroll directions ────────────────────────────────────────────────


class TestScrollDirections:
    def test_all_directions_accepted(self):
        for d in ("up", "down", "left", "right"):
            inp = BrowserToolInput(action=BrowserAction.SCROLL, direction=d)
            assert inp.direction == d


# ── Remote connection features ───────────────────────────────────────


class TestRemoteConnection:
    """Tests for remote browser connection support."""

    def test_ws_endpoint_field_accepted(self):
        inp = BrowserToolInput(
            action=BrowserAction.NAVIGATE,
            url="https://example.com",
            ws_endpoint="ws://remote-host:3000",
        )
        assert inp.ws_endpoint == "ws://remote-host:3000"

    def test_browser_url_field_accepted(self):
        inp = BrowserToolInput(
            action=BrowserAction.NAVIGATE,
            url="https://example.com",
            browser_url="http://selenium-hub:4444/wd/hub",
        )
        assert inp.browser_url == "http://selenium-hub:4444/wd/hub"

    def test_ws_endpoint_default_none(self):
        inp = BrowserToolInput(action=BrowserAction.NAVIGATE, url="https://x.com")
        assert inp.ws_endpoint is None

    def test_browser_url_default_none(self):
        inp = BrowserToolInput(action=BrowserAction.NAVIGATE, url="https://x.com")
        assert inp.browser_url is None

    @pytest.mark.asyncio
    async def test_ws_endpoint_validation_bad_scheme(self, tool):
        args = BrowserToolInput(
            action=BrowserAction.NAVIGATE,
            url="https://example.com",
            ws_endpoint="http://not-ws:3000",
        )
        result = await tool.validate_input(args, _ctx())
        assert result.result is False
        assert "ws://" in result.message

    @pytest.mark.asyncio
    async def test_ws_endpoint_validation_wss_ok(self, tool):
        args = BrowserToolInput(
            action=BrowserAction.NAVIGATE,
            url="https://example.com",
            ws_endpoint="wss://secure-host:3000",
        )
        result = await tool.validate_input(args, _ctx())
        assert result.result is True

    def test_resolve_ws_endpoint_from_arg(self, tool):
        args = BrowserToolInput(
            action=BrowserAction.NAVIGATE, url="https://x.com",
            ws_endpoint="ws://my-host:3000",
        )
        assert tool._resolve_ws_endpoint(args) == "ws://my-host:3000"

    def test_resolve_ws_endpoint_from_env(self, tool):
        args = BrowserToolInput(action=BrowserAction.NAVIGATE, url="https://x.com")
        with patch.dict(os.environ, {"BROWSER_WS_ENDPOINT": "ws://env-host:3000"}):
            assert tool._resolve_ws_endpoint(args) == "ws://env-host:3000"

    def test_resolve_ws_endpoint_arg_overrides_env(self, tool):
        args = BrowserToolInput(
            action=BrowserAction.NAVIGATE, url="https://x.com",
            ws_endpoint="ws://arg-host:3000",
        )
        with patch.dict(os.environ, {"BROWSER_WS_ENDPOINT": "ws://env-host:3000"}):
            assert tool._resolve_ws_endpoint(args) == "ws://arg-host:3000"

    def test_resolve_selenium_url_from_arg(self, tool):
        args = BrowserToolInput(
            action=BrowserAction.NAVIGATE, url="https://x.com",
            browser_url="http://selenium:4444",
        )
        assert tool._resolve_selenium_url(args) == "http://selenium:4444"

    def test_resolve_selenium_url_from_env(self, tool):
        args = BrowserToolInput(action=BrowserAction.NAVIGATE, url="https://x.com")
        with patch.dict(os.environ, {"SELENIUM_REMOTE_URL": "http://env-selenium:4444"}):
            assert tool._resolve_selenium_url(args) == "http://env-selenium:4444"

    @pytest.mark.asyncio
    async def test_mock_result_mentions_remote_options(self, tool):
        """Mock result note should mention remote connection options."""
        args = BrowserToolInput(action=BrowserAction.NAVIGATE, url="https://example.com")
        result = tool._mock_result(args)
        assert result.success is True
        assert "BROWSER_WS_ENDPOINT" in result.data["note"]
        assert "SELENIUM_REMOTE_URL" in result.data["note"]
        assert result.data["mode"] == "mock"

    def test_connection_mode_initial(self, tool):
        assert tool._connection_mode == "none"

