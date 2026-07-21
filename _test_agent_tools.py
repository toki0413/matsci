"""快速测 agent 能否真调工具 — 构造 HuginnAgent, 发一个简单 prompt."""
import asyncio
import os
import sys
from pathlib import Path

# 加载 .env
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

# RCBench 沙箱配置 (复用 rcb_huginn.py 的环境)
os.environ.setdefault("HUGINN_RATE_LIMIT_TOKENS_PER_SECOND", "50000")
os.environ.setdefault("HUGINN_RATE_LIMIT_TOKENS_PER_TURN", "500000")
os.environ.setdefault("HUGINN_HEALTH_MONITOR", "0")
os.environ.setdefault("HUGINN_ALLOW_UNRESTRICTED_READ", "1")
os.environ.setdefault("HUGINN_ALLOW_LOCAL_BASH", "1")
os.environ.setdefault("HUGINN_CACHE_DIR", str(Path(__file__).parent / "_test_cache"))

# 关 RestrictedPython
try:
    import huginn.security.restricted_python as _rp
    _rp.validate_code = lambda code: None
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).parent / "agent"))

from huginn.agent.core import HuginnAgent
from huginn.config import HuginnConfig
from huginn.memory.manager import MemoryManager, MemoryConfig
from huginn.models.registry import ModelRegistry
from huginn.skills.base import DeclarativeSkillExecutor
from huginn.tools import register_all_tools
from huginn.tools.registry import ToolRegistry


async def main():
    cfg = HuginnConfig.from_env()
    registry = ModelRegistry.from_config(cfg)
    alias = registry.default_alias()
    if alias:
        model = registry.resolve(alias)
    else:
        model = registry.resolve(f"{cfg.provider}/{cfg.model or 'auto'}")
    print(f"Model: {type(model).__name__} ({getattr(model, 'model', 'unknown')})")

    # 验证 model 有 bind_tools
    assert hasattr(model, "bind_tools"), "model has no bind_tools!"
    print("model.bind_tools: OK")

    ws = Path(__file__).parent / "_test_workspace"
    ws.mkdir(exist_ok=True)
    (ws / "report").mkdir(exist_ok=True)

    memory_dir = ws / ".memory"
    memory_dir.mkdir(exist_ok=True)
    memory_manager = MemoryManager(
        config=MemoryConfig(memory_dir=memory_dir, auto_promote_to_longterm=False),
        llm=model,
    )
    skill_executor = DeclarativeSkillExecutor(ToolRegistry)

    agent = HuginnAgent(
        model=model,
        system_prompt=(
            "You are a test agent. Use code_tool to print 'hello world'. "
            "Then use file_write_tool to write 'test ok' to report/report.md. "
            "DO NOT just describe what you would do — ACTUALLY call the tools."
        ),
        memory_manager=memory_manager,
        skill_executor=skill_executor,
        max_tool_calls=20,
        max_tool_calls_per_tool=10,
        auto_approve=True,
        tool_filter=["code_tool", "file_write_tool"],
        workspace=str(ws.resolve()),
    )
    register_all_tools()
    agent.register_tools_from_registry()

    print(f"\nTools registered: {len(agent.langchain_tools)}")
    for t in agent.langchain_tools:
        print(f"  - {t.name}")

    print("\n--- sending chat ---")
    tool_calls_seen = 0
    tool_results_seen = 0
    final_text = ""
    async for chunk in agent.chat(
        "Use code_tool to print 'hello world', then file_write_tool to write 'test ok' to report/report.md.",
        thread_id="test",
    ):
        msgs = chunk.get("messages", []) if isinstance(chunk, dict) else []
        for m in msgs:
            mtype = type(m).__name__
            if mtype == "AIMessage" and getattr(m, "tool_calls", None):
                tool_calls_seen += len(m.tool_calls)
                print(f"  [AIMessage tool_calls: {[(tc['name'], tc['args']) for tc in m.tool_calls]}]")
            elif mtype == "ToolMessage":
                tool_results_seen += 1
                content = str(m.content)[:200]
                print(f"  [ToolMessage: {content}]")
            elif mtype == "AIMessage":
                content = m.content if isinstance(m.content, str) else str(m.content)
                if content:
                    final_text = content
        # Also handle _token events
        if isinstance(chunk, dict) and "_token" in chunk:
            pass  # skip streaming tokens

    print(f"\ntool_calls_seen={tool_calls_seen}, tool_results_seen={tool_results_seen}")
    print(f"final_text (last 200): {final_text[-200:]}")

    report = ws / "report" / "report.md"
    if report.exists():
        print(f"report.md exists, content: {report.read_text()!r}")
    else:
        print("report.md does NOT exist")


if __name__ == "__main__":
    asyncio.run(main())
