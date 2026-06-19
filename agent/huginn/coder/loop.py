"""Autonomous coder loop for huginn-agent.

The loop is designed to feel like OpenAI Codex: the model can read, write,
edit, run shell commands, and inspect git state. It stops when it decides it
is done or after a maximum number of iterations.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool

from huginn.config import CoderSettings, get_settings
from huginn.llm import get_model
from huginn.permissions import PermissionConfig
from huginn.prompts import CODER_SYSTEM_PROMPT
from huginn.tools.adapter import ApprovalCallback, ToolAdapter


def _clean_args(raw: dict[str, Any]) -> dict[str, Any]:
    """Drop None values and coerce args for tool input."""
    return {k: v for k, v in raw.items() if v is not None}


class CoderRunner:
    """Run an autonomous coding session.

    Parameters
    ----------
    tools
        Optional list of tools. If ``None``, the default coder toolset is built
        from the registry.
    settings
        Optional application settings. Defaults are loaded from the environment.
    permission_config
        Permission configuration controlling read-only / destructive tool
        behavior. Defaults to the global default rules.
    approval_callback
        Optional callback invoked when a tool requires user approval. If
        ``None`` and a tool is in ``ASK`` mode, the call is denied.
    """

    def __init__(
        self,
        tools: list[BaseTool] | None = None,
        settings: Any | None = None,
        permission_config: PermissionConfig | None = None,
        approval_callback: ApprovalCallback | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.permission_config = permission_config or PermissionConfig()
        self.approval_callback = approval_callback
        self.tools = self._build_tools(tools)
        self.tool_map = {tool.name: tool for tool in self.tools}
        self.model = get_model(self.settings)

    def _build_tools(
        self,
        tools: list[BaseTool] | None,
    ) -> list[BaseTool]:
        """Return adapted LangChain tools."""
        if tools is not None:
            return tools

        from huginn.tools.bash_tool import BashTool
        from huginn.tools.code_tool import CodeTool
        from huginn.tools.file_edit_tool import FileEditTool
        from huginn.tools.file_read_tool import FileReadTool
        from huginn.tools.file_write_tool import FileWriteTool
        from huginn.tools.git_tool import GitTool

        originals = [
            FileReadTool(),
            FileWriteTool(),
            FileEditTool(),
            BashTool(),
            GitTool(),
            CodeTool(),
        ]
        return [
            ToolAdapter.adapt(
                tool,
                permission_config=self.permission_config,
                approval_callback=self.approval_callback,
            )
            for tool in originals
        ]

    def _execute_tool_call(self, call: dict[str, Any]) -> dict[str, Any]:
        name = call.get("name")
        args = call.get("args", {})
        call_id = call.get("id", "unknown")
        if name not in self.tool_map:
            return {
                "tool_call_id": call_id,
                "role": "tool",
                "name": name or "unknown",
                "content": json.dumps({"error": f"Tool '{name}' not found."}),
            }
        tool = self.tool_map[name]
        cleaned = _clean_args(args)

        try:
            result = tool.invoke(cleaned)
            return {
                "tool_call_id": call_id,
                "role": "tool",
                "name": name,
                "content": json.dumps(result),
            }
        except Exception as exc:  # pragma: no cover - defensive
            return {
                "tool_call_id": call_id,
                "role": "tool",
                "name": name,
                "content": json.dumps({"error": str(exc)}),
            }

    def run(self, task: str, max_iterations: int | None = None) -> dict[str, Any]:
        """Run the coder loop on a user task.

        Parameters
        ----------
        task
            Natural language description of the change to make.
        max_iterations
            Maximum number of model iterations. Defaults to the value in
            :class:`~huginn.config.CoderSettings`.

        Returns
        -------
        dict
            Dictionary containing ``final_answer`` and ``messages``.
        """
        coder_cfg: CoderSettings = self.settings.coder
        max_iter = max_iterations or coder_cfg.max_iterations

        messages: list[BaseMessage] = [
            SystemMessage(content=CODER_SYSTEM_PROMPT),
            HumanMessage(content=task),
        ]

        bound_model = self.model.bind_tools(self.tools)

        for _ in range(max_iter):
            response: AIMessage = bound_model.invoke(messages)
            messages.append(response)

            if response.tool_calls:
                for call in response.tool_calls:
                    tool_result = self._execute_tool_call(call)
                    messages.append(ToolMessage(**tool_result))
                continue

            content = response.content or ""
            if isinstance(content, list):
                content = "\n".join(str(c) for c in content)

            if coder_cfg.done_marker in content:
                final = content.split(coder_cfg.done_marker, 1)[0].strip()
                return {"final_answer": final, "messages": messages}

            # No tool calls and no done marker: treat response as final.
            return {"final_answer": content, "messages": messages}

        # Hit iteration limit.
        final_msg = messages[-1]
        final_text = final_msg.content if isinstance(final_msg.content, str) else ""
        return {
            "final_answer": f"Coder reached the maximum iteration limit.\n{final_text}",
            "messages": messages,
        }
