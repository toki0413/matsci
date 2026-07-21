"""验证 _PatchedModel 不破坏 langgraph 工具调用循环.

如果 _PatchedBound.invoke 返回的 AIMessage 丢了 tool_calls,
create_react_agent 会在第一轮就结束 (text-only response).
"""
import os, sys
sys.path.insert(0, "agent")

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

os.environ["DEEPSEEK_API_KEY"] = "sk-462dae89e16941e4b05c62f0e8e76aa6"

from huginn.agent.middlewares import FixDanglingToolCallsMiddleware

mw = FixDanglingToolCallsMiddleware()

class _PatchedBound:
    def __init__(self, bound, mw):
        self._bound = bound
        self._mw = mw
    def invoke(self, input, config=None, **kw):
        if isinstance(input, list):
            input = self._mw._patch_messages(list(input))
        return self._bound.invoke(input, config=config, **kw)
    async def ainvoke(self, input, config=None, **kw):
        if isinstance(input, list):
            input = self._mw._patch_messages(list(input))
        return await self._bound.ainvoke(input, config=config, **kw)
    def __getattr__(self, name):
        return getattr(self._bound, name)

class _PatchedModel:
    def __init__(self, model, mw):
        self._model = model
        self._mw = mw
    def bind_tools(self, tools, **kw):
        return _PatchedBound(self._model.bind_tools(tools, **kw), self._mw)
    def with_structured_output(self, *a, **kw):
        return self._model.with_structured_output(*a, **kw)
    def __getattr__(self, name):
        return getattr(self._model, name)

@tool
def list_dir(path: str) -> str:
    """List files in a directory."""
    import os
    return str(os.listdir(path))

model = ChatOpenAI(
    model="deepseek-chat",
    api_key="sk-462dae89e16941e4b05c62f0e8e76aa6",
    base_url="https://api.deepseek.com/v1",
)

print("=== 原始 model (无 _PatchedModel 包装) ===")
bound = model.bind_tools([list_dir])
r = bound.invoke([HumanMessage(content="Use the list_dir tool to list the current directory. You MUST call the tool, do not just describe what you would do.")])
print(f"type: {type(r).__name__}")
print(f"tool_calls: {getattr(r, 'tool_calls', None)}")
print(f"content: {str(r.content)[:200]}")

print("\n=== _PatchedModel 包装 ===")
pm = _PatchedModel(model, mw)
pbound = pm.bind_tools([list_dir])
r2 = pbound.invoke([HumanMessage(content="Use the list_dir tool to list the current directory. You MUST call the tool, do not just describe what you would do.")])
print(f"type: {type(r2).__name__}")
print(f"tool_calls: {getattr(r2, 'tool_calls', None)}")
print(f"content: {str(r2.content)[:200]}")

print("\n=== 对比 ===")
tc1 = getattr(r, 'tool_calls', None) or []
tc2 = getattr(r2, 'tool_calls', None) or []
if tc1 and tc2:
    print(f"OK: 两者都有 tool_calls (原始 {len(tc1)}, 包装 {len(tc2)})")
elif tc1 and not tc2:
    print("FAIL: 原始有 tool_calls, 包装丢失 — _PatchedModel 破坏了工具调用!")
elif not tc1 and not tc2:
    print("WARN: 两者都没 tool_calls — DeepSeek 本次不调工具 (可能 prompt 问题)")
else:
    print(f"UNEXPECTED: 原始 {len(tc1)}, 包装 {len(tc2)}")
