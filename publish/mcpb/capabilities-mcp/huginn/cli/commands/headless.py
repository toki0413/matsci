"""Headless exec command — CI-friendly single-run agent.

对标 Codex CLI 的 ``codex exec``: 跑一次 prompt 就退出, 不进入交互循环.
stdout 输出固定 JSON 契约, 退出码标准化, 便于 shell 判断成功/失败:

    exit 0  → agent 成功产出了答复
    exit 1  → agent 构建或运行出错 (``error`` 字段带原因)
    exit 2  → 未配置任何 provider

stdout JSON 契约 (稳定, 供 CI 解析):
    {
      "version": 1,
      "success": bool,
      "exit_code": int,
      "thread_id": str,
      "answer": str,
      "tool_calls": [{"name": str}, ...],
      "messages": [{"role": str, "content": str}, ...],
      "error": str | null
    }
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

import click

from huginn.cli.context import CliContext, build_agent_from_ctx


def _msg_text(msg: Any) -> str:
    """把消息 content 拍平成纯文本 (数组/blocks 场景)."""
    content = getattr(msg, "content", "")
    if isinstance(content, list):
        return "".join(
            b.get("text", "") for b in content if isinstance(b, dict)
        )
    return str(content)


def _assistant_answer(messages: list) -> str:
    """取最后一条带文本的 assistant 答复, 没有返回空串."""
    for msg in reversed(messages):
        if getattr(msg, "type", "") != "ai":
            continue
        text = _msg_text(msg)
        if text.strip():
            return text
    return ""


async def _drive(agent: Any, message: str, thread_id: str) -> dict[str, Any]:
    """流式跑一轮, 收集最终消息 / 答复 / 工具调用.

    复用 agent.chat 的流式链路 (与 chat.py 一致), 但只沉淀结构, 不做
    Rich 渲染 — 输出留给调用方统一打 JSON.
    """
    tool_calls: list[str] = []
    messages: list = []
    shown = 0
    async for state in agent.chat(message, thread_id=thread_id):
        if not isinstance(state, dict) or not state.get("messages"):
            # 边信道事件/状态标记不是完整 state, 跳过
            continue
        new_msgs = state["messages"]
        for m in new_msgs[shown:]:
            if getattr(m, "tool_calls", None):
                for tc in m.tool_calls or []:
                    tool_calls.append(tc.get("name", "unknown"))
        shown = len(new_msgs)
        messages = new_msgs  # 流式 messages 是累积的, 直接覆盖即可

    return {"messages": messages, "tool_names": tool_calls}


@click.command("exec")
@click.argument("prompt")
@click.option("--thread-id", "thread_id", default="default", help="会话 id, 用于恢复同一上下文")
@click.option("--json", "as_json", is_flag=True, default=True, help="JSON 输出 (默认开启)")
@click.pass_obj
def exec_cmd(ctx: CliContext, prompt: str, thread_id: str, as_json: bool) -> None:
    """Run the agent once, non-interactively, and print a JSON result."""
    code = 1  # 默认失败, 成功路径再覆盖
    payload: dict[str, Any] = {
        "version": 1,
        "success": False,
        "exit_code": code,
        "thread_id": thread_id,
        "answer": "",
        "tool_calls": [],
        "messages": [],
        "error": None,
    }

    agent = build_agent_from_ctx(ctx)
    if agent is None:
        payload["error"] = "no provider configured"
        code = 2
    else:
        try:
            agent.register_tools_from_registry()
            result = asyncio.run(_drive(agent, prompt, thread_id))
            payload["messages"] = [
                {
                    "role": getattr(m, "type", ""),
                    "content": _msg_text(m),
                }
                for m in result["messages"]
            ]
            payload["answer"] = _assistant_answer(result["messages"])
            payload["tool_calls"] = [{"name": n} for n in result["tool_names"]]
            payload["success"] = bool(payload["answer"])
            code = 0 if payload["success"] else 1
            if not payload["success"]:
                payload["error"] = "agent produced no answer"
        except Exception as exc:  # noqa: BLE001 — 崩溃也要给 CI 可解析的 JSON
            payload["error"] = str(exc)
            code = 1

    payload["exit_code"] = code
    _emit(payload, as_json)
    sys.exit(code)


def _emit(payload: dict[str, Any], as_json: bool) -> None:
    """输出结果: JSON 到 stdout (UTF-8, 无 Rich 标记污染)."""
    if as_json:
        sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
        sys.stdout.flush()
    else:
        # 人读模式: 失败打 stderr, 答复打 stdout
        if payload.get("error"):
            sys.stderr.write(f"error: {payload['error']}\n")
        if payload.get("answer"):
            sys.stdout.write(payload["answer"] + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    # 纯函数自检: 从 messages 里抽答复
    class _Fake:
        def __init__(self, type_, content, tool_calls=None):
            self.type = type_
            self.content = content
            self.tool_calls = tool_calls or []

    msgs = [
        _Fake("human", "hi"),
        _Fake("ai", "", [{"name": "search"}]),
        _Fake("ai", "找到结果了"),
    ]
    assert _assistant_answer(msgs) == "找到结果了", _assistant_answer(msgs)

    empty = [_Fake("human", "hi")]
    assert _assistant_answer(empty) == "", _assistant_answer(empty)

    print("headless.py self-check OK")
