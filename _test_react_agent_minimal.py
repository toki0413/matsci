"""最小测试: 直接 create_react_agent + DeepSeek + code_tool.
绕开 HuginnAgent 复杂层, 看工具是否真执行."""
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

os.environ.setdefault("HUGINN_RATE_LIMIT_TOKENS_PER_SECOND", "50000")
os.environ.setdefault("HUGINN_RATE_LIMIT_TOKENS_PER_TURN", "500000")
os.environ.setdefault("HUGINN_HEALTH_MONITOR", "0")
os.environ.setdefault("HUGINN_ALLOW_UNRESTRICTED_READ", "1")
os.environ.setdefault("HUGINN_ALLOW_LOCAL_BASH", "1")
os.environ.setdefault("HUGINN_CACHE_DIR", str(Path(__file__).parent / "_test_cache2"))

try:
    import huginn.security.restricted_python as _rp
    _rp.validate_code = lambda code: None
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).parent / "agent"))

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

from huginn.config import HuginnConfig
from huginn.models.registry import ModelRegistry
from huginn.tools import register_all_tools
from huginn.tools.code_tool import CodeTool
from huginn.tools.file_write_tool import FileWriteTool


async def main():
    cfg = HuginnConfig.from_env()
    registry = ModelRegistry.from_config(cfg)
    alias = registry.default_alias()
    model = registry.resolve(alias) if alias else registry.resolve(f"{cfg.provider}/{cfg.model}")
    print(f"Model: {type(model).__name__} ({getattr(model, 'model', 'unknown')})")

    ws = Path(__file__).parent / "_test_ws2"
    ws.mkdir(exist_ok=True)
    (ws / "report").mkdir(exist_ok=True)
    os.chdir(ws)

    # 用最小 code_tool + file_write_tool, 不要 HuginnAgent 的复杂 tool registry
    register_all_tools()
    code_tool = CodeTool()
    file_write = FileWriteTool()

    # 转 langchain tool 格式 (auto_approve_all=True 模拟 RCBench 场景)
    from huginn.tools.adapter import ToolAdapter
    from huginn.permissions import PermissionConfig
    _perm = PermissionConfig(auto_approve_all=True)
    lc_code = ToolAdapter().adapt(code_tool, permission_config=_perm)
    lc_write = ToolAdapter().adapt(file_write, permission_config=_perm)
    print(f"Tools: {lc_code.name}, {lc_write.name}")

    sys_prompt = (
        "You are a test agent. Use code_tool to print 'hello world', "
        "then use file_write_tool to write 'test ok' to report/report.md. "
        "ACTUALLY call the tools, do not just describe."
    )

    agent = create_react_agent(
        model=model,
        tools=[lc_code, lc_write],
        prompt=SystemMessage(content=sys_prompt),
        checkpointer=None,
    )

    print("\n--- streaming ---")
    inputs = {"messages": [HumanMessage(content="Do it now.")]}

    tool_calls = 0
    tool_msgs = 0
    final_ai = ""
    async for mode, data in agent.astream(
        inputs, {"recursion_limit": 50},
        stream_mode=["values", "messages"],
    ):
        if mode == "messages":
            chunk, _meta = data
            ctype = type(chunk).__name__
            if ctype.startswith("AIMessage"):
                if getattr(chunk, "tool_calls", None):
                    tool_calls += len(chunk.tool_calls)
                    print(f"  [AIMessage tool_calls: {[(tc['name'], str(tc['args'])[:80]) for tc in chunk.tool_calls]}]")
                elif chunk.content:
                    print(f"  [AIMessage text: {str(chunk.content)[:150]}]")
                    final_ai = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
        else:
            state = data
            msgs = state.get("messages", [])
            for m in msgs[-2:]:
                mtype = type(m).__name__
                if mtype == "ToolMessage":
                    tool_msgs += 1
                    print(f"  [ToolMessage: {str(m.content)[:200]}]")

    print(f"\ntool_calls={tool_calls}, tool_msgs={tool_msgs}")
    print(f"final: {final_ai[-300:]}")
    rep = ws / "report" / "report.md"
    print(f"report.md: {rep.read_text() if rep.exists() else 'NOT EXIST'}")


if __name__ == "__main__":
    import io
    import contextlib
    _buf = io.StringIO()
    with contextlib.redirect_stdout(_buf), contextlib.redirect_stderr(_buf):
        try:
            asyncio.run(main())
        except Exception as e:
            import traceback
            _buf.write(f"\nEXCEPTION: {e}\n")
            _buf.write(traceback.format_exc())
    Path("_test_react_output.txt").write_text(_buf.getvalue(), encoding="utf-8")
