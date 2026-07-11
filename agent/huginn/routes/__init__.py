"""Route module registry.

Each submodule exposes an ``APIRouter`` named ``router``.
``ALL_ROUTERS`` collects them in registration order for the FastAPI app.
"""

from __future__ import annotations

import logging
import threading

from fastapi import APIRouter, FastAPI
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from huginn.routes.advisor import router as advisor_router
from huginn.routes.agents import router as agents_router
from huginn.routes.auth import router as auth_router
from huginn.routes.autoloop import router as autoloop_router
from huginn.routes.bench import router as bench_router
from huginn.routes.bot import router as bot_router
from huginn.routes.checkpoints import router as checkpoints_router
from huginn.routes.codebase import router as codebase_router
from huginn.routes.coder import router as coder_router
from huginn.routes.compat import router as compat_router
from huginn.routes.config import router as config_router
from huginn.routes.diagnostics import router as diagnostics_router
from huginn.routes.eval import router as eval_router
from huginn.routes.events import router as events_router
from huginn.routes.event_stream import router as event_stream_router
from huginn.routes.execution import router as execution_router
from huginn.routes.health import router as health_router
from huginn.routes.hpc import router as hpc_router
from huginn.routes.credentials import router as credentials_router
from huginn.routes.interaction import router as interaction_router
from huginn.routes.knowledge import router as knowledge_router
from huginn.routes.metrics import router as metrics_router
from huginn.routes.provenance import router as provenance_router
from huginn.routes.mcp import router as mcp_router
from huginn.routes.memory import router as memory_router
from huginn.routes.pet import router as pet_router
from huginn.routes.planner import router as planner_router
from huginn.routes.project import router as project_router
from huginn.routes.research_project import router as research_project_router
from huginn.routes.skills import router as skills_router
from huginn.routes.system import router as system_router
from huginn.routes.team import router as team_router
from huginn.routes.threads import router as threads_router
from huginn.routes.tools import router as tools_router
from huginn.routes.unified import router as unified_router
from huginn.routes.users import router as users_router
from huginn.routes.workflows import router as workflows_router
from huginn.routes.data_dict import router as data_dict_router
from huginn.routes.document import router as document_router
from huginn.routes.ws import router as ws_router
from huginn.routes.live_script import router as live_script_router
from huginn.routes.parameters import router as parameters_router
from huginn.routes.side import router as side_router
from huginn.routes.admin import router as admin_router
from huginn.routes.visual import router as visual_router
from huginn.routes.export_share import router as export_share_router
# 远程计算增强: 隧道 / 文件传输 / 远程终端 (仿 MobaXterm)
from huginn.routes.tunnels import router as tunnels_router
from huginn.routes.transfer import router as transfer_router
from huginn.routes.terminal import router as terminal_router
# 实时 3D 分子查看器: REST (load/trajectory/elements) + WebSocket (/ws/viewer3d)
from huginn.routes.viewer3d import router as viewer3d_router
from huginn.routes.viewer3d import ws_router as viewer3d_ws_router
# 有状态 ipykernel 会话
from huginn.routes.kernel import router as kernel_router

ALL_ROUTERS = [
    advisor_router,
    # auth must come early so /auth/* is registered before any catch-all
    auth_router,
    health_router,
    config_router,
    project_router,
    research_project_router,
    planner_router,
    codebase_router,
    knowledge_router,
    tools_router,
    events_router,
    # 实时事件 SSE: 订阅全局 EventBus 推送所有 agent 生命周期事件
    event_stream_router,
    pet_router,
    memory_router,
    agents_router,
    autoloop_router,
    bench_router,
    eval_router,
    # Bot bridge: OneBot v11 QQ/WeChat 接入
    bot_router,
    execution_router,
    unified_router,
    workflows_router,
    skills_router,
    mcp_router,
    threads_router,
    users_router,
    checkpoints_router,
    compat_router,
    hpc_router,
    credentials_router,
    team_router,
    system_router,
    coder_router,
    data_dict_router,
    ws_router,
    live_script_router,
    parameters_router,
    # 互动层: SSE chat / 中途干预 / 主动提问 / 进度展示
    interaction_router,
    # 侧边对话: 不打断主任务的并行 Q&A
    side_router,
    # 视觉感知: I-JEPA 图像编码 / 检索 / 索引
    visual_router,
    # 文档理解: PDF 解析 / DocGraph / 信息包
    document_router,
    # Prometheus /metrics 抓取端点
    metrics_router,
    # 计算溯源: 文件产出关系 / 谱系追溯 / 全文搜索
    provenance_router,
    diagnostics_router,
    # Admin endpoints (maintenance mode, etc.)
    admin_router,
    # 导出 / 导入 / 分享: 全量打包、单组件导出、归档导入
    export_share_router,
    # 实时 3D 分子查看器: 结构加载 / 轨迹 / 元素表 / WebSocket 流
    viewer3d_router,
    viewer3d_ws_router,
    # 远程计算增强 (仿 MobaXterm): SSH 隧道 / 文件传输 / 远程终端
    tunnels_router,
    transfer_router,
    terminal_router,
    # 有状态 ipykernel 会话: 创建 / 执行 / 查状态 / 关闭
    kernel_router,
]

# ── API versioning ──────────────────────────────────────────────────

_logger = logging.getLogger("huginn.api.deprecation")

# Paths served at root that have no /v1 counterpart (infra / docs / auth).
# Hitting these should NOT emit a deprecation warning.
_ROOT_ONLY_PATHS = frozenset(
    {
        "/health",
        "/health/rust",
        "/health/guidance",
        "/docs",
        "/openapi.json",
        "/redoc",
        "/metrics",
        "/diagnostics",
        "/diagnostics/tools",
        "/diagnostics/circuit",
        "/diagnostics/trace",
        "/auth/token",
        "/auth/login",
        "/",
    }
)

# Track which route groups we've already warned about so the log doesn't
# drown in repeats under load. Keyed by the first path segment, which is
# bounded by the number of route groups (~35).
_warned_segments: set[str] = set()
_warn_lock = threading.Lock()


async def _deprecation_dispatch(request: Request, call_next):
    """Nudge callers away from the legacy root mount toward /v1.

    Adds a ``Deprecation`` response header on every root-level API response
    and logs a warning the first time we see a given route group. /v1 calls
    and infra paths are passed through silently.
    """
    path = request.url.path
    is_v1 = path == "/v1" or path.startswith("/v1/")
    if is_v1 or path in _ROOT_ONLY_PATHS:
        return await call_next(request)

    response = await call_next(request)

    # Signal deprecation to clients via standard headers.
    segment = "/" + path.strip("/").split("/", 1)[0] if path.strip("/") else "/"
    response.headers["Deprecation"] = "true"
    response.headers["Link"] = f'</v1{segment}>; rel="successor-version"'

    # Log once per route group to keep the log readable.
    with _warn_lock:
        if segment in _warned_segments:
            return response
        _warned_segments.add(segment)
    _logger.warning(
        "Deprecated root API path %s — use /v1%s instead. "
        "The root mount is kept for backward compatibility and may be "
        "removed in a future release.",
        segment,
        segment,
    )
    return response


def include_v1_routes(app: FastAPI, *, keep_root_compat: bool = True) -> None:
    """Mount every router under the ``/v1`` version prefix.

    This is the canonical API surface going forward. When
    *keep_root_compat* is True (the default) the same routers are also
    mounted at the root path so existing clients keep working, with a
    deprecation warning logged and a ``Deprecation`` header set on each
    root-level response.
    """
    v1_router = APIRouter(prefix="/v1")
    for router in ALL_ROUTERS:
        v1_router.include_router(router)
    app.include_router(v1_router)

    if keep_root_compat:
        # Keep the un-versioned mount alive so older clients don't break.
        for router in ALL_ROUTERS:
            app.include_router(router)
        app.add_middleware(BaseHTTPMiddleware, dispatch=_deprecation_dispatch)
