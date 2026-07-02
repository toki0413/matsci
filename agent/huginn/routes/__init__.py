"""Route module registry.

Each submodule exposes an ``APIRouter`` named ``router``.
``ALL_ROUTERS`` collects them in registration order for the FastAPI app.
"""

from huginn.routes.advisor import router as advisor_router
from huginn.routes.agents import router as agents_router
from huginn.routes.bench import router as bench_router
from huginn.routes.checkpoints import router as checkpoints_router
from huginn.routes.codebase import router as codebase_router
from huginn.routes.coder import router as coder_router
from huginn.routes.compat import router as compat_router
from huginn.routes.config import router as config_router
from huginn.routes.events import router as events_router
from huginn.routes.execution import router as execution_router
from huginn.routes.health import router as health_router
from huginn.routes.hpc import router as hpc_router
from huginn.routes.interaction import router as interaction_router
from huginn.routes.knowledge import router as knowledge_router
from huginn.routes.mcp import router as mcp_router
from huginn.routes.memory import router as memory_router
from huginn.routes.pet import router as pet_router
from huginn.routes.planner import router as planner_router
from huginn.routes.project import router as project_router
from huginn.routes.skills import router as skills_router
from huginn.routes.system import router as system_router
from huginn.routes.team import router as team_router
from huginn.routes.threads import router as threads_router
from huginn.routes.tools import router as tools_router
from huginn.routes.unified import router as unified_router
from huginn.routes.workflows import router as workflows_router
from huginn.routes.data_dict import router as data_dict_router
from huginn.routes.ws import router as ws_router
from huginn.routes.live_script import router as live_script_router
from huginn.routes.parameters import router as parameters_router
from huginn.routes.side import router as side_router

ALL_ROUTERS = [
    advisor_router,
    health_router,
    config_router,
    project_router,
    planner_router,
    codebase_router,
    knowledge_router,
    tools_router,
    events_router,
    pet_router,
    memory_router,
    agents_router,
    bench_router,
    execution_router,
    unified_router,
    workflows_router,
    skills_router,
    mcp_router,
    threads_router,
    checkpoints_router,
    compat_router,
    hpc_router,
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
]
