"""书生 × Huginn 科学计算 MCP 码头 (stdio / HTTP-streamable).

把深研管线的域科学诊断工具 (probe_*: C* 容量 / 零空间方向分解 / PDE 约束
求解 / PINN 优化器财政) 装箱成标准 MCP server —— 任意 MCP host (Claude
Desktop / Cursor / Trae / 书生自己的 MCP 客户端) 用 ``tools/list`` +
``tools/call`` 即可召唤**真实**科学计算, 不 import 我们的 Python 包.

独立性红线:
  - 工具数值全部来自确定性科学内核 (最小二乘 / 零空间 SVD / ODE 算子),
    **不经任何 LLM**; 书生只在这里取证据, 不参与数值产生;
  - MCP 层只做 schema 往返 (tools/list / tools/call), 与声明门禁同一证据
    轨迹 —— 远端拿到什么, 深研 trace 里就是什么;
  - 默认只暴露只读能力 (allow_write=False), 远端 host 无法动磁盘/跑命令.

用法:
  python -m huginn.research.science_mcp --check     # 打印 tools/list 自检
  python -m huginn.research.science_mcp             # stdio 传输, 供 MCP host 连接
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path


def _science_tools() -> list[dict]:
    """工具面与深研管线同源 (examples 域诊断工具), 一处定义不漂移."""
    _EX = Path(__file__).resolve().parents[3] / "examples"
    if str(_EX) not in sys.path:
        sys.path.insert(0, str(_EX))
    from shusheng_huginn_workflow import _diagnostic_tools  # type: ignore[import-not-found]
    return _diagnostic_tools()


def _ensure_registered() -> None:
    """把科学诊断工具注册进 ToolRegistry → CapabilityRegistry (幂等)."""
    from huginn.research.tool_surface import register_diagnostic_tools
    from huginn.capabilities.registry import CapabilityRegistry
    register_diagnostic_tools(_science_tools())
    CapabilityRegistry.scan_tool_registry()


def build_server(server_name: str = "huginn-science"):
    """把科学诊断工具注册进 ToolRegistry → CapabilityRegistry → MCP server."""
    from huginn.capabilities import mcp_export
    _ensure_registered()
    return mcp_export.build_server(
        mcp_export.CapabilityMCPBackend(), server_name=server_name)


async def _check() -> int:
    """自检: 打印 MCP tools/list 可见的科学工具清单 (真实可用性)."""
    from huginn.capabilities import mcp_export
    from huginn.capabilities.registry import CapabilityRegistry
    _ensure_registered()
    backend = mcp_export.CapabilityMCPBackend()
    tools = await mcp_export.list_tools_handler(backend)
    print(f"== 书生 × Huginn 科学 MCP: tools/list 共 {len(tools)} 个工具 ==")
    for t in tools:
        print(f"  - {t.name}: {(t.description or '').splitlines()[0][:78]}")
    print(f"\n能力清单: {list(CapabilityRegistry.list_capabilities())[:12]} ...")
    # 真实 tool/call 冒烟: 调方向分解探针, 数值必须来自确定性科学内核.
    r = await mcp_export.call_tool_handler("probe_direction_g",
                                           {"config": {"order": 2, "ctype": "derivative",
                                                       "positions": [0.0, 0.97]}},
                                           backend)
    print(f"\n冒烟 probe_direction_g(导数约束) → {r.content[0].text[:160]}")
    return 0


async def _serve() -> int:
    from huginn.capabilities import mcp_export
    server = build_server()
    await mcp_export.serve_stdio(server)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="只打印 tools/list 并退出")
    args = ap.parse_args()
    try:
        return asyncio.run(_check() if args.check else _serve())
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())