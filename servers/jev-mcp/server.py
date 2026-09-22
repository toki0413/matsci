"""JEV MCP Server — 把 TypeSafe 的 System One 决策模型暴露成 MCP 工具.

让任意 MCP host (含 Huginn 的 mcp_client) 直接以普通工具消费 JEV 的三个原语:

  - ``jev_noul``   : 并行 yes/no 命题 → {name: {noul, confidence}}
  - ``jev_choice`` : 从候选里选一 → {choice, probabilities, confidence}
  - ``jev_score``  : 按自定义等级打分 → {score, confidence}

设计:
  - 自包含, 只依赖 ``mcp`` SDK + httpx; 不 import huginn.*.
  - fail-open: 无 ``TYPESAFE_API_KEY`` / 网络异常 → 返回显式错误文本, 不抛穿.
  - 传输: stdio (MCP host 以子进程拉起的默认形态).
  - 单测可注入 ``_api`` (见 ``JevApi.transport``), 无需真实网络.

通过 ``/v1/systemone`` 调用: body ``{"state": …, "questions": {name: {type, …}}}``.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

app = Server("jev-mcp")

_DEFAULT_BASE_URL = "https://api.typesafe.ai"
_DEFAULT_TIMEOUT = 5.0
_MAX_CHOICE_OPTIONS = 250


def _api_key() -> str | None:
    return os.environ.get("TYPESAFE_API_KEY") or os.environ.get("TYPESAFE_MANAGED_KEY") or None


def _base_url() -> str:
    return os.environ.get("TYPESAFE_BASE_URL") or _DEFAULT_BASE_URL


def _model() -> str | None:
    return os.environ.get("TYPESAFE_MODEL") or None


class JevApi:
    """最小 JEV 客户端 (MCP server 内部使用). transport 可注入供测试 mock."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
        transport: Any | None = None,
    ) -> None:
        self._api_key = api_key or _api_key()
        self._base_url = (base_url or _base_url()).rstrip("/")
        self._timeout = timeout
        self._transport = transport  # post(url, headers, json, timeout) -> resp(.json())

    @property
    def available(self) -> bool:
        return bool(self._api_key)

    def system_one(self, state: Any, questions: dict[str, Any]) -> dict[str, Any] | None:
        """POST /v1/systemone. 无 key / 网络异常 → None (fail-open)."""
        if not self.available or not questions:
            return None
        body: dict[str, Any] = {"state": state, "questions": questions}
        mod = _model()
        if mod:
            body["model"] = mod
        try:
            payload = self._post(body)
        except Exception:  # noqa: BLE001 — 网络/超时/状态码一律 fail-open
            return None
        return payload if isinstance(payload, dict) else None

    def _post(self, body: dict[str, Any]) -> Any:
        url = f"{self._base_url}/v1/systemone"
        headers = {"Authorization": f"Bearer {self._api_key}"}
        if self._transport is not None:
            return self._transport(
                url=url, headers=headers, json=body, timeout=self._timeout
            ).json()
        import httpx

        with httpx.Client(timeout=self._timeout) as client:
            resp = client.post(url, headers=headers, json=body)
            resp.raise_for_status()
            return resp.json()


# 供工具实现共用; 单测可替换为注入 transport 的实例.
_api = JevApi()


# ---------------------------------------------------------------------------
# 工具定义
# ---------------------------------------------------------------------------

TOOLS: list[Tool] = [
    Tool(
        name="jev_noul",
        description=(
            "System One 并行判定: 对 {name: proposition} 逐个输出 0..1 成立概率. "
            "返回 {name: {noul, confidence}}. 高置信高概率 → 命题更可能成立."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "state": {"type": "object", "description": "结构化程序状态/上下文"},
                "questions": {
                    "type": "object",
                    "description": "{命题名: 命题文本}, 并行求值",
                    "additionalProperties": {"type": "string"},
                },
            },
            "required": ["state", "questions"],
        },
    ),
    Tool(
        name="jev_choice",
        description=(
            "System One 单选: 从 options 中选一. 返回 {choice, probabilities, confidence}."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "state": {"type": "object"},
                "instruction": {"type": "string", "description": "选择任务描述"},
                "options": {"type": "array", "items": {"type": "string"}},
                "question_name": {"type": "string", "default": "choice"},
            },
            "required": ["state", "instruction", "options"],
        },
    ),
    Tool(
        name="jev_score",
        description=(
            "System One 打分: 按你描述的等级对 state 打分. 返回 {score, confidence}."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "state": {"type": "object"},
                "instruction": {"type": "string", "description": "评分标准描述"},
                "levels": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "可选的等级标签(自低到高)",
                },
                "question_name": {"type": "string", "default": "score"},
            },
            "required": ["state", "instruction"],
        },
    ),
]


@app.list_tools()
async def list_tools() -> list[Tool]:
    return TOOLS


@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any] | None) -> list[TextContent]:
    arguments = arguments or {}
    if name == "jev_noul":
        return _handle_noul(arguments)
    if name == "jev_choice":
        return _handle_choice(arguments)
    if name == "jev_score":
        return _handle_score(arguments)
    raise ValueError(f"Unknown tool: {name}")


# ---------------------------------------------------------------------------
# 工具实现
# ---------------------------------------------------------------------------


def _text(payload: dict[str, Any], *, is_error: bool = False) -> list[TextContent]:
    return [TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, default=str))]


def _unavailable(reason: str) -> list[TextContent]:
    return _text({"success": False, "error": f"jev unavailable: {reason}"})


def _handle_noul(args: dict[str, Any]) -> list[TextContent]:
    state = args.get("state", {})
    questions: Any = args.get("questions", {})
    if not isinstance(questions, dict) or not questions:
        return _text({"success": False, "error": "jev_noul requires non-empty questions"})
    if not _api.available:
        return _unavailable("no TYPESAFE_API_KEY")
    qq = {name: {"type": "noul", "instructions": instr} for name, instr in questions.items()}
    payload = _api.system_one(state, qq)
    if payload is None:
        return _unavailable("request failed / timed out (fail-open)")
    answers = payload.get("answers", {})
    return _text({"success": True, "answers": answers})


def _handle_choice(args: dict[str, Any]) -> list[TextContent]:
    state = args.get("state", {})
    instruction = args.get("instruction", "")
    options = args.get("options", [])
    qname = args.get("question_name", "choice")
    if not instruction or not isinstance(options, list) or not options:
        return _text({"success": False, "error": "jev_choice requires instruction + non-empty options"})
    if not _api.available:
        return _unavailable("no TYPESAFE_API_KEY")
    eff_options = options[:_MAX_CHOICE_OPTIONS]
    qq = {qname: {"type": "choice", "instructions": instruction, "options": list(eff_options)}}
    payload = _api.system_one(state, qq)
    if payload is None:
        return _unavailable("request failed / timed out (fail-open)")
    answers = payload.get("answers", {})
    ans = answers.get(qname, {})
    return _text({"success": True, "answers": answers, "answer": ans})


def _handle_score(args: dict[str, Any]) -> list[TextContent]:
    state = args.get("state", {})
    instruction = args.get("instruction", "")
    levels = args.get("levels")
    qname = args.get("question_name", "score")
    if not instruction:
        return _text({"success": False, "error": "jev_score requires instruction"})
    if not _api.available:
        return _unavailable("no TYPESAFE_API_KEY")
    q: dict[str, Any] = {"type": "score", "instructions": instruction}
    if isinstance(levels, list) and levels:
        q["levels"] = list(levels)
    payload = _api.system_one(state, {qname: q})
    if payload is None:
        return _unavailable("request failed / timed out (fail-open)")
    answers = payload.get("answers", {})
    ans = answers.get(qname, {})
    return _text({"success": True, "answers": answers, "answer": ans})


# ---------------------------------------------------------------------------
# 传输
# ---------------------------------------------------------------------------


async def main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="jev-mcp",
                server_version="0.1.0",
                capabilities=app.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


def console_main() -> None:
    """PyPI console-script 入口 (pip install 后由 jev-mcp 命令调用)."""
    asyncio.run(main())


if __name__ == "__main__":
    console_main()