"""Browser automation tool — simulate human browser interactions.

Supports login, click, scroll, input, form filling, and information extraction.
Prefers connecting to a remote browser service (via ws_endpoint or browser_url);
falls back to local Playwright launch, then mock mode.
"""
from __future__ import annotations
import asyncio
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal
from pydantic import BaseModel, Field
from huginn.tools.base import HuginnTool
from huginn.types import ToolContext, ToolResult, ValidationResult
from huginn.validation.handle_validator import HandleValidator

class BrowserAction(str, Enum):
    NAVIGATE = "navigate"
    CLICK = "click"
    TYPE = "type"
    FILL_FORM = "fill_form"
    SCROLL = "scroll"
    SCREENSHOT = "screenshot"
    EXTRACT = "extract"
    LOGIN = "login"
    WAIT = "wait"
    CLOSE = "close"

class BrowserToolInput(BaseModel):
    action: BrowserAction = Field(...)
    url: str | None = Field(default=None, description="URL to navigate to")
    selector: str | None = Field(default=None, description="CSS selector for element")
    text: str | None = Field(default=None, description="Text to type or value to fill")
    form_data: dict[str, str] = Field(default_factory=dict, description="Form field -> value mapping")
    credentials: dict[str, str] = Field(default_factory=dict, description="Login credentials (username, password)")
    direction: Literal["up", "down", "left", "right"] = Field(default="down")
    scroll_amount: int = Field(default=500, description="Pixels to scroll")
    extract_selector: str | None = Field(default=None, description="CSS selector for elements to extract")
    extract_attributes: list[str] = Field(default_factory=list, description="Attributes to extract")
    wait_ms: int = Field(default=1000, ge=0, le=30000)
    timeout_ms: int = Field(default=30000, ge=1000, le=120000)
    headless: bool = Field(default=True)
    ws_endpoint: str | None = Field(
        default=None,
        description="WebSocket endpoint for remote Playwright browser (e.g. ws://remote-host:3000). "
                    "Falls back to BROWSER_WS_ENDPOINT env var, then local launch.",
    )
    browser_url: str | None = Field(
        default=None,
        description="Remote Selenium WebDriver URL (e.g. http://selenium-hub:4444/wd/hub). "
                    "Falls back to SELENIUM_REMOTE_URL env var.",
    )

class BrowserTool(HuginnTool):
    """Automate browser interactions: navigate, click, type, fill forms, extract data."""

    name = "browser_tool"
    category = "search"
    description = "Simulate human browser interactions — login, click, scroll, input, fill forms, and extract information from web pages"
    input_schema = BrowserToolInput

    def __init__(self):
        super().__init__()
        self._browser = None
        self._page = None
        self._context = None
        self._pw = None
        self._driver = None  # Selenium WebDriver
        self._connection_mode: str = "none"  # remote_playwright | remote_selenium | local | mock

    async def validate_input(self, args: BrowserToolInput, context: ToolContext) -> ValidationResult:
        if args.action == BrowserAction.NAVIGATE and not args.url:
            return ValidationResult(result=False, message="URL required for navigate", error_code=400)
        if args.action in (BrowserAction.CLICK, BrowserAction.TYPE) and not args.selector:
            return ValidationResult(result=False, message=f"Selector required for {args.action.value}", error_code=400)
        if args.action == BrowserAction.TYPE and not args.text:
            return ValidationResult(result=False, message="Text required for type action", error_code=400)
        if args.action == BrowserAction.LOGIN:
            if not args.url:
                return ValidationResult(result=False, message="URL required for login", error_code=400)
            if not args.credentials.get("username") or not args.credentials.get("password"):
                return ValidationResult(result=False, message="Username and password required for login", error_code=400)
        if args.action == BrowserAction.FILL_FORM and not args.form_data:
            return ValidationResult(result=False, message="form_data required for fill_form", error_code=400)
        if args.url and not args.url.startswith(("http://", "https://")):
            return ValidationResult(result=False, message="URL must start with http:// or https://", error_code=400)
        if args.ws_endpoint and not args.ws_endpoint.startswith(("ws://", "wss://")):
            return ValidationResult(result=False, message="ws_endpoint must start with ws:// or wss://", error_code=400)
        return ValidationResult(result=True)

    def _resolve_ws_endpoint(self, args: BrowserToolInput) -> str | None:
        """Resolve Playwright WS endpoint: arg > env var > None."""
        return args.ws_endpoint or os.environ.get("BROWSER_WS_ENDPOINT")

    def _resolve_selenium_url(self, args: BrowserToolInput) -> str | None:
        """Resolve Selenium remote URL: arg > env var > None."""
        return args.browser_url or os.environ.get("SELENIUM_REMOTE_URL")

    async def _ensure_browser(self, args: BrowserToolInput) -> bool:
        """Ensure browser is available. Priority: remote Playwright > remote Selenium > local > mock."""
        if self._page is not None or self._driver is not None:
            return True

        # 1. Remote Playwright via WebSocket
        ws_endpoint = self._resolve_ws_endpoint(args)
        if ws_endpoint:
            try:
                from playwright.async_api import async_playwright
                self._pw = await async_playwright().start()
                self._browser = await self._pw.chromium.connect(ws_endpoint)
                self._context = self._browser.contexts[0] if self._browser.contexts else await self._browser.new_context()
                self._page = self._context.pages[0] if self._context.pages else await self._context.new_page()
                self._connection_mode = "remote_playwright"
                return True
            except Exception:
                self._pw = None
                self._browser = None

        # 2. Remote Selenium WebDriver
        selenium_url = self._resolve_selenium_url(args)
        if selenium_url:
            try:
                from selenium import webdriver
                from selenium.webdriver.remote.webdriver import WebDriver
                options = webdriver.ChromeOptions()
                if args.headless:
                    options.add_argument("--headless")
                self._driver = webdriver.Remote(command_executor=selenium_url, options=options)
                self._connection_mode = "remote_selenium"
                return True
            except Exception:
                self._driver = None

        # 3. Local Playwright launch
        # 注意: 只 catch ImportError 会漏掉浏览器二进制缺失的情况
        # playwright python 包装了, 但 chromium 浏览器没装时会抛 FileNotFoundError
        # 必须用 Exception 兜底, 让它降级到 mock 模式而不是炸飞整个 agent
        try:
            from playwright.async_api import async_playwright
            self._pw = await async_playwright().start()
            self._browser = await self._pw.chromium.launch(headless=args.headless)
            self._context = await self._browser.new_context(
                viewport={"width": 1280, "height": 720},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            )
            self._page = await self._context.new_page()
            self._connection_mode = "local"
            return True
        except Exception:
            # playwright 没装 / chromium 没下载 / 启动失败 —— 都降级到 mock
            self._pw = None
            self._browser = None

        # 4. Mock mode
        self._connection_mode = "mock"
        return False

    async def call(self, args: BrowserToolInput, context: ToolContext) -> ToolResult:
        has_browser = await self._ensure_browser(args)

        if not has_browser:
            return self._mock_result(args)

        # Dispatch to Selenium if connected that way
        if self._driver is not None:
            return await self._call_selenium(args)

        # Playwright path
        try:
            if args.action == BrowserAction.NAVIGATE:
                await self._page.goto(args.url, timeout=args.timeout_ms)
                title = await self._page.title()
                return ToolResult(data={"status": "navigated", "url": args.url, "title": title, "mode": self._connection_mode}, success=True)

            elif args.action == BrowserAction.CLICK:
                await self._page.click(args.selector, timeout=args.timeout_ms)
                return ToolResult(data={"status": "clicked", "selector": args.selector, "mode": self._connection_mode}, success=True)

            elif args.action == BrowserAction.TYPE:
                await self._page.fill(args.selector, "", timeout=args.timeout_ms)
                await self._page.type(args.selector, args.text, delay=50)
                return ToolResult(data={"status": "typed", "selector": args.selector, "length": len(args.text), "mode": self._connection_mode}, success=True)

            elif args.action == BrowserAction.FILL_FORM:
                filled = []
                for sel, val in args.form_data.items():
                    await self._page.fill(sel, val, timeout=args.timeout_ms)
                    filled.append(sel)
                return ToolResult(data={"status": "form_filled", "fields": filled, "mode": self._connection_mode}, success=True)

            elif args.action == BrowserAction.SCROLL:
                scroll_js = {
                    "down": f"window.scrollBy(0, {args.scroll_amount})",
                    "up": f"window.scrollBy(0, -{args.scroll_amount})",
                    "left": f"window.scrollBy(-{args.scroll_amount}, 0)",
                    "right": f"window.scrollBy({args.scroll_amount}, 0)",
                }
                await self._page.evaluate(scroll_js[args.direction])
                return ToolResult(data={"status": "scrolled", "direction": args.direction, "amount": args.scroll_amount, "mode": self._connection_mode}, success=True)

            elif args.action == BrowserAction.SCREENSHOT:
                screenshot_bytes = await self._page.screenshot(full_page=True)
                import base64
                b64 = base64.b64encode(screenshot_bytes).decode()
                return ToolResult(data={"status": "screenshot", "image_base64": b64[:100] + "...", "size_bytes": len(screenshot_bytes), "mode": self._connection_mode}, success=True)

            elif args.action == BrowserAction.EXTRACT:
                if args.extract_selector:
                    elements = await self._page.query_selector_all(args.extract_selector)
                    extracted = []
                    for el in elements:
                        item = {}
                        if args.extract_attributes:
                            for attr in args.extract_attributes:
                                item[attr] = await el.get_attribute(attr)
                        else:
                            item["text"] = await el.inner_text()
                            item["tag"] = await el.evaluate("el => el.tagName")
                        extracted.append(item)
                    return ToolResult(data={"status": "extracted", "count": len(extracted), "elements": extracted, "mode": self._connection_mode}, success=True)
                else:
                    content = await self._page.content()
                    return ToolResult(data={"status": "extracted", "content_length": len(content), "content_preview": content[:500], "mode": self._connection_mode}, success=True)

            elif args.action == BrowserAction.LOGIN:
                await self._page.goto(args.url, timeout=args.timeout_ms)
                username = args.credentials["username"]
                password = args.credentials["password"]
                for user_sel in ['input[name="username"]', 'input[type="email"]', '#username', 'input[name="user"]']:
                    try:
                        await self._page.fill(user_sel, username, timeout=3000)
                        break
                    except Exception:
                        continue
                for pass_sel in ['input[name="password"]', 'input[type="password"]', '#password']:
                    try:
                        await self._page.fill(pass_sel, password, timeout=3000)
                        break
                    except Exception:
                        continue
                for submit_sel in ['button[type="submit"]', 'input[type="submit"]', 'button:has-text("Login")', 'button:has-text("Sign in")']:
                    try:
                        await self._page.click(submit_sel, timeout=3000)
                        break
                    except Exception:
                        continue
                await asyncio.sleep(1)
                title = await self._page.title()
                return ToolResult(data={"status": "login_attempted", "url": args.url, "title": title, "mode": self._connection_mode}, success=True)

            elif args.action == BrowserAction.WAIT:
                await asyncio.sleep(args.wait_ms / 1000)
                return ToolResult(data={"status": "waited", "ms": args.wait_ms, "mode": self._connection_mode}, success=True)

            elif args.action == BrowserAction.CLOSE:
                await self._close_all()
                return ToolResult(data={"status": "closed", "mode": self._connection_mode}, success=True)

            return ToolResult(data=None, success=False, error=f"Unknown action: {args.action}")

        except Exception as e:
            return ToolResult(data=None, success=False, error=f"Browser error: {e}")

    async def _call_selenium(self, args: BrowserToolInput) -> ToolResult:
        """Execute actions via Selenium WebDriver."""
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC

        try:
            driver = self._driver
            if args.action == BrowserAction.NAVIGATE:
                driver.get(args.url)
                return ToolResult(data={"status": "navigated", "url": args.url, "title": driver.title, "mode": "remote_selenium"}, success=True)

            elif args.action == BrowserAction.CLICK:
                el = driver.find_element(By.CSS_SELECTOR, args.selector)
                el.click()
                return ToolResult(data={"status": "clicked", "selector": args.selector, "mode": "remote_selenium"}, success=True)

            elif args.action == BrowserAction.TYPE:
                el = driver.find_element(By.CSS_SELECTOR, args.selector)
                el.clear()
                el.send_keys(args.text)
                return ToolResult(data={"status": "typed", "selector": args.selector, "length": len(args.text), "mode": "remote_selenium"}, success=True)

            elif args.action == BrowserAction.FILL_FORM:
                filled = []
                for sel, val in args.form_data.items():
                    el = driver.find_element(By.CSS_SELECTOR, sel)
                    el.clear()
                    el.send_keys(val)
                    filled.append(sel)
                return ToolResult(data={"status": "form_filled", "fields": filled, "mode": "remote_selenium"}, success=True)

            elif args.action == BrowserAction.SCROLL:
                scroll_js = {
                    "down": f"window.scrollBy(0, {args.scroll_amount})",
                    "up": f"window.scrollBy(0, -{args.scroll_amount})",
                    "left": f"window.scrollBy(-{args.scroll_amount}, 0)",
                    "right": f"window.scrollBy({args.scroll_amount}, 0)",
                }
                driver.execute_script(scroll_js[args.direction])
                return ToolResult(data={"status": "scrolled", "direction": args.direction, "amount": args.scroll_amount, "mode": "remote_selenium"}, success=True)

            elif args.action == BrowserAction.SCREENSHOT:
                import base64
                b64 = driver.get_screenshot_as_base64()
                return ToolResult(data={"status": "screenshot", "image_base64": b64[:100] + "...", "size_bytes": len(b64), "mode": "remote_selenium"}, success=True)

            elif args.action == BrowserAction.EXTRACT:
                if args.extract_selector:
                    elements = driver.find_elements(By.CSS_SELECTOR, args.extract_selector)
                    extracted = []
                    for el in elements:
                        item = {}
                        if args.extract_attributes:
                            for attr in args.extract_attributes:
                                item[attr] = el.get_attribute(attr)
                        else:
                            item["text"] = el.text
                            item["tag"] = el.tag_name
                        extracted.append(item)
                    return ToolResult(data={"status": "extracted", "count": len(extracted), "elements": extracted, "mode": "remote_selenium"}, success=True)
                else:
                    content = driver.page_source
                    return ToolResult(data={"status": "extracted", "content_length": len(content), "content_preview": content[:500], "mode": "remote_selenium"}, success=True)

            elif args.action == BrowserAction.LOGIN:
                driver.get(args.url)
                username = args.credentials["username"]
                password = args.credentials["password"]
                for user_sel in ['input[name="username"]', 'input[type="email"]', '#username', 'input[name="user"]']:
                    try:
                        el = driver.find_element(By.CSS_SELECTOR, user_sel)
                        el.clear()
                        el.send_keys(username)
                        break
                    except Exception:
                        continue
                for pass_sel in ['input[name="password"]', 'input[type="password"]', '#password']:
                    try:
                        el = driver.find_element(By.CSS_SELECTOR, pass_sel)
                        el.clear()
                        el.send_keys(password)
                        break
                    except Exception:
                        continue
                for submit_sel in ['button[type="submit"]', 'input[type="submit"]']:
                    try:
                        driver.find_element(By.CSS_SELECTOR, submit_sel).click()
                        break
                    except Exception:
                        continue
                await asyncio.sleep(1)
                return ToolResult(data={"status": "login_attempted", "url": args.url, "title": driver.title, "mode": "remote_selenium"}, success=True)

            elif args.action == BrowserAction.WAIT:
                await asyncio.sleep(args.wait_ms / 1000)
                return ToolResult(data={"status": "waited", "ms": args.wait_ms, "mode": "remote_selenium"}, success=True)

            elif args.action == BrowserAction.CLOSE:
                await self._close_all()
                return ToolResult(data={"status": "closed", "mode": "remote_selenium"}, success=True)

            return ToolResult(data=None, success=False, error=f"Unknown action: {args.action}")

        except Exception as e:
            return ToolResult(data=None, success=False, error=f"Selenium error: {e}")

    async def _close_all(self):
        """Close all browser resources."""
        if self._driver:
            try:
                self._driver.quit()
            except Exception:
                pass
            self._driver = None
        if self._browser:
            try:
                await self._browser.close()
            except Exception:
                pass
            self._browser = None
        self._page = None
        self._context = None
        if self._pw:
            try:
                await self._pw.stop()
            except Exception:
                pass
            self._pw = None
        self._connection_mode = "none"

    def _mock_result(self, args: BrowserToolInput) -> ToolResult:
        """Return mock results when no browser backend is available."""
        mock_data: dict[str, Any] = {
            "status": "mock",
            "action": args.action.value,
            "mode": "mock",
            "note": "No browser backend available. Options: "
                    "(1) Set BROWSER_WS_ENDPOINT for remote Playwright, "
                    "(2) Set SELENIUM_REMOTE_URL for remote Selenium, "
                    "(3) Install Playwright locally: pip install playwright && playwright install chromium",
        }
        if args.action == BrowserAction.NAVIGATE:
            mock_data.update({"url": args.url, "title": "Mock Page Title"})
        elif args.action == BrowserAction.EXTRACT:
            mock_data.update({"elements": [{"text": "Mock element 1"}, {"text": "Mock element 2"}], "count": 2})
        elif args.action == BrowserAction.LOGIN:
            mock_data.update({"login_url": args.url, "login_status": "mock_success"})
        elif args.action == BrowserAction.FILL_FORM:
            mock_data.update({"fields_filled": list(args.form_data.keys())})
        return ToolResult(data=mock_data, success=True)
