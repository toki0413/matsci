"""共享 LLM 调用 — 把 model 统一唤起为纯文本.

DRY: ``agentic_search_tool`` 与 ``literature.tool`` 曾各复制一份逐字相同的
``_llm_invoke`` (同步/异步模型兼容 + content 归一化). 任一处对多模态 content 的
处理变化都会让两条检索链给出不同的综合文本, 因此收敛为单一实现.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any


async def invoke_llm_text(model: Any, system_prompt: str, user_prompt: str) -> str:
    """调 model 并返回纯文本; 非字符串 content 统一 JSON 序列化."""
    from langchain_core.messages import HumanMessage, SystemMessage

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    if hasattr(model, "ainvoke"):
        response = await model.ainvoke(messages)
    else:
        response = await asyncio.to_thread(model.invoke, messages)
    content = response.content if hasattr(response, "content") else str(response)
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False)
    return content
