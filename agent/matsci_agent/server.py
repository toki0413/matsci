"""FastAPI + WebSocket server for MatSci-Agent.

Serves the desktop frontend with:
- HTTP API for tools, workflows, and health checks
- WebSocket endpoint for real-time Agent chat
- Compatibility stubs for math-anything frontend APIs
"""

from __future__ import annotations

import json
import os
import asyncio
import difflib
import tempfile
import traceback
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from langchain_core.messages import AIMessage, ToolMessage

from matsci_agent import __version__
from matsci_agent.agent import MatSciAgent
from matsci_agent.tools.registry import ToolRegistry
from matsci_agent.tools.structure_tool import StructureTool
from matsci_agent.tools.extract_tool import ExtractTool
from matsci_agent.tools.job_tool import JobTool
from matsci_agent.tools.database_tool import DatabaseTool
from matsci_agent.tools.potential_tool import PotentialTool
from matsci_agent.tools.diff_tool import DiffTool
from matsci_agent.tools.validate_tool import ValidateTool
from matsci_agent.tools.diagnose_tool import DiagnoseTool
from matsci_agent.tools.vasp_tool import VaspTool
from matsci_agent.tools.lammps_tool import LammpsTool
from matsci_agent.tools.symbolic_regression_tool import SymbolicRegressionTool
from matsci_agent.tools.report_tool import ReportTool
from matsci_agent.tools.comsol_tool import ComsolTool
from matsci_agent.tools.qe_tool import QuantumEspressoTool
from matsci_agent.tools.cp2k_tool import Cp2kTool
from matsci_agent.tools.uq_tool import UQTool
from matsci_agent.tools.gp_tool import GPTool
from matsci_agent.rag.rag_tool import RAGTool
from matsci_agent.evaluation.evaluation_tool import EvaluationTool
from matsci_agent.config import MatSciConfig
from matsci_agent.personas import PERSONAS
from matsci_agent.types import ToolContext
from matsci_agent.workflows.engine import WorkflowEngine
from matsci_agent.workflows.templates import get_template
from matsci_agent.hpc.client import HPCClient, HPCConfig
from matsci_agent.skills.base import DeclarativeSkillExecutor
from matsci_agent.skills.registry import SkillRegistry


# Register all tools
for T in [StructureTool, ExtractTool, JobTool, DatabaseTool, PotentialTool, DiffTool, ValidateTool, DiagnoseTool, VaspTool, LammpsTool, ComsolTool, QuantumEspressoTool, Cp2kTool, UQTool, GPTool, RAGTool, EvaluationTool, SymbolicRegressionTool, ReportTool]:
    ToolRegistry.register(T())

# Global agent and MCP manager
_agent: MatSciAgent | None = None
_mcp_manager = None

# In-memory checkpoints for diff review (snapshot path -> content)
_checkpoints: dict[str, tuple[Path, dict[str, str]]] = {}


async def _init_mcp_tools():
    """Connect to local MCP servers and register their tools."""
    global _mcp_manager
    try:
        from matsci_agent.mcp_client import MCPClientManager, MCPServerConfig
        from matsci_agent.tools.mcp_adapter import register_mcp_tools
        from pathlib import Path

        _mcp_manager = MCPClientManager()
        base = Path(__file__).parent.parent.parent  # repo root

        # Try mat-db-mcp
        mat_db_path = base / "servers" / "mat-db-mcp" / "server.py"
        if mat_db_path.exists():
            await _mcp_manager.connect(MCPServerConfig(
                name="mat-db",
                command="python",
                args=[str(mat_db_path)],
            ))

        # Try math-anything-mcp
        math_path = base / "servers" / "math-anything-mcp" / "server.py"
        if math_path.exists():
            await _mcp_manager.connect(MCPServerConfig(
                name="math-anything",
                command="python",
                args=[str(math_path)],
            ))

        registered = register_mcp_tools(_mcp_manager)
        print(f"[MCP] Registered {len(registered)} tools from MCP servers")
    except Exception as e:
        print(f"[MCP] Warning: Could not initialize MCP tools: {e}")


async def _shutdown_mcp():
    """Disconnect all MCP servers."""
    global _mcp_manager
    if _mcp_manager:
        await _mcp_manager.disconnect_all()
        _mcp_manager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    await _init_mcp_tools()
    yield
    await _shutdown_mcp()


app = FastAPI(title="MatSci-Agent Server", version=__version__, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _check_ollama_available(base_url: str, timeout: float = 2.0) -> bool:
    """Quick check if Ollama is responding."""
    try:
        import urllib.request
        req = urllib.request.Request(f"{base_url}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def get_agent() -> MatSciAgent:
    """Get or create the MatSciAgent instance."""
    global _agent
    if _agent is not None:
        return _agent
    
    cfg = MatSciConfig.from_env()
    system_prompt = PERSONAS.get(cfg.persona, PERSONAS["default"])
    
    # No provider configured → mock mode
    if cfg.provider == "default":
        _agent = MatSciAgent(model=None, system_prompt=system_prompt)
        _agent.register_tools_from_registry()
        return _agent
    
    # Ollama: check availability first
    if cfg.provider == "ollama":
        if not _check_ollama_available(cfg.ollama_host):
            print(f"Warning: Ollama not responding at {cfg.ollama_host}")
            print("Falling back to mock mode (no LLM)")
            _agent = MatSciAgent(model=None, system_prompt=system_prompt)
            _agent.register_tools_from_registry()
            return _agent
    
    # Try unified provider factory
    try:
        _agent = MatSciAgent.from_provider(
            provider=cfg.provider,
            model=cfg.model,
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            system_prompt=system_prompt,
        )
    except ImportError as e:
        print(f"Warning: Missing dependency for {cfg.provider}: {e}")
        print("Falling back to mock mode (no LLM)")
        _agent = MatSciAgent(model=None, system_prompt=system_prompt)
    except ValueError as e:
        print(f"Warning: {e}")
        print("Falling back to mock mode (no LLM)")
        _agent = MatSciAgent(model=None, system_prompt=system_prompt)
    except Exception as e:
        print(f"Warning: Failed to initialize {cfg.provider} model: {e}")
        print("Falling back to mock mode (no LLM)")
        _agent = MatSciAgent(model=None, system_prompt=system_prompt)
    
    _agent.register_tools_from_registry()
    return _agent


# ── Health & Info ──────────────────────────────────────────────

@app.get("/health")
async def health() -> dict[str, Any]:
    cfg = MatSciConfig.from_env()
    return {
        "status": "ok",
        "version": __version__,
        "provider": cfg.provider,
        "model": cfg.model,
        "configured": cfg.provider != "default" and bool(cfg.resolved_api_key),
    }


@app.get("/config")
async def get_config() -> dict[str, Any]:
    """Return current server-side configuration (API key masked)."""
    return MatSciConfig.from_env().to_dict(mask_key=True)


@app.post("/config")
async def update_config(params: dict[str, Any]) -> dict[str, Any]:
    """Update server-side configuration and reset the agent so changes take effect."""
    global _agent

    if "provider" in params:
        os.environ["MATSCI_PROVIDER"] = str(params["provider"])
    if "model" in params:
        os.environ["MATSCI_MODEL"] = str(params["model"])
    if "api_key" in params:
        if params["api_key"]:
            os.environ["MATSCI_API_KEY"] = str(params["api_key"])
        else:
            os.environ.pop("MATSCI_API_KEY", None)
    if "base_url" in params:
        if params["base_url"]:
            os.environ["MATSCI_BASE_URL"] = str(params["base_url"])
        else:
            os.environ.pop("MATSCI_BASE_URL", None)
    if "ollama_host" in params:
        os.environ["OLLAMA_HOST"] = str(params["ollama_host"])
    if "persona" in params:
        os.environ["MATSCI_PERSONA"] = str(params["persona"])

    _agent = None  # force re-initialization with new config
    cfg = MatSciConfig.from_env()
    return {"success": True, "config": cfg.to_dict(mask_key=True)}


@app.get("/tools")
async def list_tools() -> list[dict[str, Any]]:
    """List all available tools with their schemas."""
    return ToolRegistry.get_all_schemas()


@app.post("/tools/{tool_name}")
async def call_tool(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Call a tool directly via HTTP."""
    from matsci_agent.types import ToolContext
    
    tool = ToolRegistry.get(tool_name)
    if not tool:
        return {"error": f"Tool '{tool_name}' not found"}
    
    if not tool.input_schema:
        return {"error": f"Tool '{tool_name}' has no input schema"}
    
    try:
        input_data = tool.input_schema(**args)
    except Exception as e:
        return {"error": f"Invalid input: {e}"}
    
    context = ToolContext(session_id="http", workspace=".")
    result = await tool.call(input_data, context)
    
    return {
        "success": result.success,
        "data": result.data,
        "error": result.error,
    }


@app.get("/workflows")
async def list_workflows() -> list[str]:
    from matsci_agent.workflows.templates import list_templates
    return list_templates()


@app.post("/workflows/execute")
async def execute_workflow(params: dict[str, Any]) -> dict[str, Any]:
    """Execute a workflow template.
    
    Args:
        template: Template name (e.g., "standard_dft", "aimd")
        args: Arguments passed to the template function
    """
    template_name = params.get("template")
    template_args = params.get("args", {})
    
    template_fn = get_template(template_name)
    if not template_fn:
        return {"error": f"Template '{template_name}' not found"}
    
    try:
        stages = template_fn(**template_args)
    except Exception as e:
        return {"error": f"Failed to build workflow: {e}"}
    
    engine = WorkflowEngine(ToolRegistry)
    context = ToolContext(session_id="http", workspace=".")
    
    result = await engine.execute(stages, context)
    
    return {
        "success": result.success,
        "total_walltime": result.total_walltime,
        "stages": {
            sid: {
                "name": s.name,
                "status": s.status,
                "attempts": s.attempts,
                "error": s.result.error if s.result else None,
            }
            for sid, s in result.stages.items()
        },
        "outputs": result.outputs,
        "error": result.error,
    }


# ── Skills ─────────────────────────────────────────────────────

@app.get("/skills")
async def list_skills() -> list[dict[str, Any]]:
    """List all registered skills."""
    # Ensure presets are loaded and registered
    from matsci_agent.skills import presets  # noqa: F401

    return [
        {
            "name": skill.name,
            "description": skill.description,
            "category": skill.category,
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type,
                    "description": p.description,
                    "required": p.required,
                    "default": p.default,
                }
                for p in skill.parameters
            ],
            "tags": skill.tags,
        }
        for skill in SkillRegistry.get_all_definitions()
    ]


@app.post("/skills/execute")
async def execute_skill(params: dict[str, Any]) -> dict[str, Any]:
    """Execute a skill by name with the provided arguments."""
    from matsci_agent.skills import presets  # noqa: F401

    skill_name = params.get("skill")
    skill_args = params.get("args", {})

    skill = SkillRegistry.get(skill_name)
    if not skill:
        return {"error": f"Skill '{skill_name}' not found"}

    executor = DeclarativeSkillExecutor(ToolRegistry)
    context = ToolContext(session_id="http", workspace=".")
    result = await executor.execute(skill, skill_args, {})
    return result


# ── WebSocket ──────────────────────────────────────────────────

@app.websocket("/ws/agent")
async def agent_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time Agent chat."""
    await websocket.accept()
    
    try:
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)
            
            msg_type = data.get("type", "user_input")
            content = data.get("content", "")
            thread_id = data.get("thread_id", "default")
            
            if msg_type == "user_input":
                agent = get_agent()
                
                if agent.model is None:
                    await websocket.send_json({
                        "type": "error",
                        "error": "No LLM configured. Set MATSCI_PROVIDER and API keys, or start Ollama."
                    })
                    continue
                
                # Stream agent responses
                try:
                    full_response = ""
                    seen_tool_calls: set[str] = set()
                    seen_tool_results: set[str] = set()
                    async for state in agent.chat(content, thread_id):
                        messages = state.get("messages", [])
                        if not messages:
                            continue
                        last_msg = messages[-1]

                        # Emit tool-call cards
                        if isinstance(last_msg, AIMessage):
                            for tc in getattr(last_msg, "tool_calls", []) or []:
                                tid = tc.get("id")
                                if tid and tid not in seen_tool_calls:
                                    seen_tool_calls.add(tid)
                                    await websocket.send_json({
                                        "type": "tool_call",
                                        "id": tid,
                                        "name": tc.get("name", "unknown"),
                                        "args": tc.get("args", {}),
                                    })

                        # Emit tool results
                        if isinstance(last_msg, ToolMessage):
                            tid = getattr(last_msg, "tool_call_id", None)
                            if tid and tid not in seen_tool_results:
                                seen_tool_results.add(tid)
                                await websocket.send_json({
                                    "type": "tool_result",
                                    "id": tid,
                                    "content": str(getattr(last_msg, "content", "")),
                                })

                        # Only send text delta for assistant content
                        if hasattr(last_msg, "content") and not isinstance(last_msg, ToolMessage):
                            # Only send delta (new content)
                            delta = last_msg.content[len(full_response):]
                            if delta:
                                full_response = last_msg.content
                                await websocket.send_json({
                                    "type": "text_delta",
                                    "text": delta,
                                })
                    
                    # Signal completion
                    await websocket.send_json({"type": "done"})
                    
                except Exception as e:
                    traceback.print_exc()
                    await websocket.send_json({
                        "type": "error",
                        "error": f"Agent error: {str(e)}"
                    })
            
            elif msg_type == "explore_start":
                # Exploration mode — run real exploration engine
                await websocket.send_json({
                    "type": "text_delta",
                    "text": f"🚀 Starting exploration: {content}\n"
                })
                try:
                    from matsci_agent.exploration.orchestrator import ExplorationOrchestrator
                    from matsci_agent.exploration.strategies import ParetoPruningStrategy

                    orch = ExplorationOrchestrator(
                        strategy=ParetoPruningStrategy(max_active=5),
                        max_parallel=3,
                    )

                    # Parse exploration config from message if provided
                    config = data.get("config", {})
                    initial_branches = config.get("initial_branches", [
                        {"name": "baseline", "hypothesis": f"Baseline for: {content}"}
                    ])
                    objectives = config.get("objectives", {"score": "maximize"})

                    result = await orch.explore(
                        objective=content,
                        initial_branches=initial_branches,
                        objectives_config=objectives,
                        max_iterations=config.get("max_iterations", 10),
                    )

                    # Stream results
                    await websocket.send_json({
                        "type": "text_delta",
                        "text": f"\n✅ Exploration complete!\n"
                                f"• Branches explored: {result.n_branches_explored}\n"
                                f"• Branches pruned: {result.n_branches_pruned}\n"
                                f"• Pareto front size: {len(result.pareto_front)}\n"
                                f"• Convergence: {result.convergence_reason}\n",
                    })

                    if result.best_branch:
                        await websocket.send_json({
                            "type": "text_delta",
                            "text": f"\n🏆 Best branch: {result.best_branch['name']}\n"
                                    f"   Hypothesis: {result.best_branch['hypothesis']}\n"
                                    f"   Objectives: {result.best_branch['objectives']}\n",
                        })

                    # Send structured data as final message
                    await websocket.send_json({
                        "type": "exploration_result",
                        "data": {
                            "pareto_front": result.pareto_front,
                            "best_branch": result.best_branch,
                            "convergence_reason": result.convergence_reason,
                        }
                    })

                except Exception as e:
                    traceback.print_exc()
                    await websocket.send_json({
                        "type": "error",
                        "error": f"Exploration failed: {str(e)}"
                    })
                await websocket.send_json({"type": "done"})
            
            elif msg_type == "ping":
                await websocket.send_json({"type": "pong"})
    
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")


# ── math-anything compatibility stubs ──────────────────────────

@app.get("/firewall/status")
async def firewall_status() -> dict[str, Any]:
    return {"enabled": False}


@app.post("/sandbox/execute")
async def sandbox_execute(params: dict[str, Any]) -> dict[str, Any]:
    """Execute Python code in a sandbox."""
    code = params.get("code", "")
    timeout = params.get("timeout_seconds", 10)

    # Pre-validate code against restricted execution policy
    try:
        from matsci_agent.security import validate_code, RestrictedPythonError
        validate_code(code)
    except RestrictedPythonError as e:
        return {"success": False, "error": f"Policy violation: {e}"}

    try:
        import subprocess
        from matsci_agent.security import SandboxExecutor, SandboxConfig
        # Write code to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()
            tmp_path = f.name

        sandbox = SandboxExecutor(
            SandboxConfig(
                allowed_executables={"python", "python3"},
                default_timeout=min(float(timeout), 300.0),
                max_timeout=300.0,
                max_output_bytes=10 * 1024 * 1024,
            )
        )
        sb_result = sandbox.run(
            ["python", tmp_path],
            timeout=min(float(timeout), 300.0),
        )
        result = sb_result

        os.unlink(tmp_path)

        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"Execution timed out after {timeout}s"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/analyze/symmetry")
async def analyze_symmetry(params: dict[str, Any]) -> dict[str, Any]:
    return {"error": "Not yet implemented. Install spglib for symmetry analysis."}


@app.post("/analyze/spectral")
async def analyze_spectral(params: dict[str, Any]) -> dict[str, Any]:
    return {"error": "Not yet implemented"}


@app.post("/analyze/dynamics")
async def analyze_dynamics(params: dict[str, Any]) -> dict[str, Any]:
    return {"error": "Not yet implemented"}


@app.post("/analyze/tda")
async def analyze_tda(params: dict[str, Any]) -> dict[str, Any]:
    return {"error": "Not yet implemented"}


@app.post("/analyze/sindy")
async def analyze_sindy(params: dict[str, Any]) -> dict[str, Any]:
    return {"error": "Not yet implemented"}


@app.post("/viz/dos")
async def viz_dos(params: dict[str, Any]) -> dict[str, Any]:
    return {"fallback": True, "html": "<div>DOS visualization stub</div>"}


@app.post("/viz/phase")
async def viz_phase(params: dict[str, Any]) -> dict[str, Any]:
    return {"fallback": True, "html": "<div>Phase portrait stub</div>"}


@app.post("/viz/persistence")
async def viz_persistence(params: dict[str, Any]) -> dict[str, Any]:
    return {"fallback": True, "html": "<div>Persistence diagram stub</div>"}


@app.post("/viz/sindy")
async def viz_sindy(params: dict[str, Any]) -> dict[str, Any]:
    return {"fallback": True, "html": "<div>SINDy visualization stub</div>"}


# ── HPC endpoints ──────────────────────────────────────────────

@app.post("/hpc/test")
async def hpc_test_connection(params: dict[str, Any]) -> dict[str, Any]:
    """Test SSH connection to an HPC cluster."""
    cfg = HPCConfig(
        host=params.get("host", ""),
        username=params.get("username", ""),
        scheduler=params.get("scheduler", "slurm"),
        key_path=params.get("key_path"),
        port=params.get("port", 22),
    )
    
    if not cfg.host or not cfg.username:
        return {"success": False, "error": "host and username are required"}
    
    try:
        with HPCClient(cfg) as client:
            stdout, stderr, rc = client._exec("hostname")
            if rc == 0:
                return {
                    "success": True,
                    "hostname": stdout,
                    "scheduler": cfg.scheduler,
                }
            else:
                return {"success": False, "error": stderr or "Connection failed"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/hpc/submit")
async def hpc_submit(params: dict[str, Any]) -> dict[str, Any]:
    """Submit a job to remote HPC."""
    cfg = HPCConfig(
        host=params.get("host", ""),
        username=params.get("username", ""),
        scheduler=params.get("scheduler", "slurm"),
        key_path=params.get("key_path"),
        remote_work_dir=params.get("remote_work_dir", "~/matsci_jobs"),
    )
    
    if not cfg.host or not cfg.username:
        return {"success": False, "error": "host and username are required"}
    
    try:
        with HPCClient(cfg) as client:
            script = client.generate_job_script(
                command=params.get("command", "echo 'Hello HPC'"),
                job_name=params.get("job_name", "matsci_job"),
                walltime=params.get("walltime", "01:00:00"),
                nodes=params.get("nodes", 1),
                ntasks_per_node=params.get("ntasks_per_node", 4),
                queue=params.get("queue"),
                modules=params.get("modules", []),
                env_vars=params.get("env_vars", {}),
            )
            job_id = client.submit_job(script, job_name=params.get("job_name", "matsci_job"))
            return {"success": True, "job_id": job_id, "host": cfg.host}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/hpc/status")
async def hpc_status(params: dict[str, Any]) -> dict[str, Any]:
    """Poll status of a remote HPC job."""
    cfg = HPCConfig(
        host=params.get("host", ""),
        username=params.get("username", ""),
        scheduler=params.get("scheduler", "slurm"),
        key_path=params.get("key_path"),
    )
    
    job_id = params.get("job_id")
    if not job_id:
        return {"success": False, "error": "job_id is required"}
    
    try:
        with HPCClient(cfg) as client:
            status = client.poll_status(job_id)
            return {
                "success": True,
                "job_id": status.job_id,
                "state": status.state,
                "exit_code": status.exit_code,
                "runtime": status.runtime,
                "message": status.message,
            }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ── Checkpoint / Diff Review ───────────────────────────────────

def _snapshot_directory(base: Path) -> dict[str, str]:
    """Snapshot text files under base into a dict keyed by relative path."""
    snapshot: dict[str, str] = {}
    if not base.exists():
        return snapshot
    for path in base.rglob("*"):
        if not path.is_file():
            continue
        # Skip obvious binary / large files
        try:
            if path.stat().st_size > 2 * 1024 * 1024:
                continue
            data = path.read_bytes()
            if b"\x00" in data:
                continue
            snapshot[str(path.relative_to(base))] = data.decode("utf-8", errors="ignore")
        except Exception:
            continue
    return snapshot


@app.post("/checkpoints")
async def create_checkpoint(params: dict[str, Any]) -> dict[str, Any]:
    """Create a checkpoint of the given directory for later diff review."""
    base = Path(params.get("path", ".")).resolve()
    snapshot = _snapshot_directory(base)
    cp_id = uuid.uuid4().hex[:8]
    _checkpoints[cp_id] = (base, snapshot)
    return {"id": cp_id, "base": str(base), "files": len(snapshot)}


@app.get("/checkpoints/{cp_id}")
async def get_checkpoint(cp_id: str) -> dict[str, Any]:
    if cp_id not in _checkpoints:
        return {"error": "checkpoint not found"}
    base, snapshot = _checkpoints[cp_id]
    return {"id": cp_id, "base": str(base), "files": list(snapshot.keys())}


@app.get("/checkpoints/{cp_id}/diff")
async def checkpoint_diff(cp_id: str) -> dict[str, Any]:
    if cp_id not in _checkpoints:
        return {"error": "checkpoint not found"}
    base, snapshot = _checkpoints[cp_id]
    current = _snapshot_directory(base)
    diffs = []
    all_files = set(snapshot.keys()) | set(current.keys())
    for rel in sorted(all_files):
        old = snapshot.get(rel, "")
        new = current.get(rel, "")
        if old == new:
            continue
        status = "added" if rel not in snapshot else "deleted" if rel not in current else "modified"
        diff_text = "\n".join(
            difflib.unified_diff(
                old.splitlines(),
                new.splitlines(),
                fromfile=f"a/{rel}",
                tofile=f"b/{rel}",
                lineterm="",
            )
        )
        diffs.append({
            "path": rel,
            "status": status,
            "diff": diff_text,
            "old": old,
            "new": new,
        })
    return {"id": cp_id, "base": str(base), "diffs": diffs}


@app.post("/checkpoints/{cp_id}/accept")
async def accept_checkpoint(cp_id: str) -> dict[str, Any]:
    if cp_id not in _checkpoints:
        return {"error": "checkpoint not found"}
    del _checkpoints[cp_id]
    return {"success": True}


@app.post("/checkpoints/{cp_id}/reject")
async def reject_checkpoint(cp_id: str) -> dict[str, Any]:
    if cp_id not in _checkpoints:
        return {"error": "checkpoint not found"}
    base, snapshot = _checkpoints[cp_id]
    current = _snapshot_directory(base)
    for rel, content in snapshot.items():
        if rel in current:
            path = base / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
    # Remove files that were added after the checkpoint
    for rel in current:
        if rel not in snapshot:
            path = base / rel
            try:
                path.unlink()
            except Exception:
                pass
    del _checkpoints[cp_id]
    return {"success": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
