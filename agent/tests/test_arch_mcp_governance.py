"""制度化闭环：MCP server 连接必须走单入口，禁止散落硬编码。

单网关原则的延伸：真正的 MCP client 接线（构造 MCPServerConfig、拉起 stdio/sse
client、register_mcp_tools）只允许出现在治理白名单文件里。业务模块想接 MCP，
只能通过 huginn.server 的 `/v1/mcp/*` API 或 lifespan 统一接线——否则又会回到
"各处直接 spawn MCP server" 的多入口老问题（鉴权/审计/生命周期不统一）。

本门禁扫描 agent/huginn/** 源码：任何非白名单文件出现 MCP 接线 token 即 fail。
"""

from __future__ import annotations

from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
HUGINN_ROOT = AGENT_ROOT / "huginn"

# MCP client 侧接线 token（server 实现里的 Server() 不算，那是服务端）
_MCP_WIRING_TOKENS = (
    "MCPServerConfig(",
    "register_mcp_tools(",
    "stdio_client(",
    "sse_client(",
)

# 治理白名单：MCP 接线只允许出现在这些文件（相对 agent/）。
# 新增接线入口必须显式加入，否则 CI 拦截。
_ALLOWED_WIRING_FILES = frozenset({
    "huginn/mcp_client.py",          # 客户端/管理器本体
    "huginn/routes/mcp.py",          # /v1/mcp/* 管理 API
    "huginn/lifespan.py",            # 启动统一接线
    "huginn/tools/mcp_adapter.py",   # register_mcp_tools 定义地
    "huginn/cli/context.py",         # CLI 本地 MCP 兜底
})


def _iter_py_files() -> list[Path]:
    return sorted(HUGINN_ROOT.rglob("*.py"))


def test_mcp_wiring_is_single_gateway():
    """任何非白名单文件不得出现 MCP 接线 token。"""
    violations: list[str] = []
    for path in _iter_py_files():
        rel = path.relative_to(AGENT_ROOT).as_posix()
        if rel in _ALLOWED_WIRING_FILES:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for token in _MCP_WIRING_TOKENS:
            if token in text:
                violations.append(f"{rel}: 出现 MCP 接线 token {token!r}")
    assert not violations, (
        "MCP server 连接必须走单入口（mcp_client / routes/mcp.py / lifespan / "
        "mcp_adapter / cli.context），业务模块禁止直接 spawn MCP client。"
        f"\n违规:\n" + "\n".join(violations)
    )


def test_allowlist_is_nonempty():
    """守卫有效性检查：白名单非空，防止误删后失去意义。"""
    assert _ALLOWED_WIRING_FILES, "MCP 接线白名单不应为空"


def test_governance_scan_actually_covers_allowlist():
    """确保白名单文件真实存在（都在扫描范围内），否则守卫形同虚设。"""
    existing = {p.relative_to(AGENT_ROOT).as_posix() for p in _iter_py_files()}
    missing = _ALLOWED_WIRING_FILES - existing
    assert not missing, f"白名单里不存在的文件: {missing}"