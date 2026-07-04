"""Multi-provider, multi-agent, personas, orchestration, telemetry, and swarm endpoints."""

from __future__ import annotations

import asyncio
import os
import traceback
from typing import Any

from fastapi import APIRouter
from pydantic import ValidationError

from huginn.config import HuginnConfig, get_config
from huginn.models.registry import ModelRegistry
from huginn.routes.schemas import ChatRequest
from huginn.server_core import (
    get_agent,
    get_agent_factory,
    get_context,
    get_memory_manager,
)

router = APIRouter(tags=["agents"])


# ── Models & Agents ──────────────────────────────────────────────


@router.get("/models")
async def list_models() -> dict[str, Any]:
    """List configured model aliases."""
    try:
        cfg = get_config()
        registry = ModelRegistry.from_config(cfg)
        return {"models": [m.__dict__ for m in registry.list()]}
    except Exception as e:
        return {"error": str(e)}


@router.get("/agents")
async def list_agents() -> dict[str, Any]:
    """List configured agent profiles."""
    try:
        factory = get_agent_factory()
        profiles = factory.list_profiles()
        return {
            "agents": [
                {
                    "id": p.id,
                    "name": p.name or p.id,
                    "model_alias": p.model_alias,
                    "persona": p.persona,
                    "tools": p.tools,
                    "enabled": p.enabled,
                    "max_steps": p.max_steps,
                }
                for p in profiles
            ]
        }
    except Exception as e:
        return {"error": str(e)}


@router.post("/agents/{agent_id}/chat")
async def chat_with_agent(agent_id: str, params: dict[str, Any]) -> dict[str, Any]:
    """Send a single-turn message to a specific agent profile."""
    # Validate the request body before touching any downstream code.
    try:
        req = ChatRequest.model_validate(params)
    except ValidationError as exc:
        return {"error": f"Invalid request: {exc.errors()}"}

    user_message = req.content
    thread_id = req.thread_id

    try:
        # sidecar serve 是非交互 API 服务, dev 模式下既没容器运行时也没审批回调
        # 自动放行本地沙箱, 不然 code_tool/bash_tool 调用全卡在 SandboxError
        # 真正生产环境应该走容器执行, 那时再显式配 HUGINN_CONTAINER_RUNTIME
        if os.environ.get("HUGINN_DEV_MODE", "").lower() in ("1", "true", "yes"):
            os.environ.setdefault("HUGINN_ALLOW_LOCAL_BASH", "1")

        factory = get_agent_factory()
        agent = factory.create(
            agent_id,
            thread_id=thread_id,
            thinking=req.thinking,
            max_tokens=req.max_tokens,
        )
        # sidecar serve 是非交互 API 服务, 不可能有审批回调, 强制自动批准所有工具
        # 不然 ASK 模式的工具会被拒, agent 只能回退到手动计算
        agent._permission_config.auto_approve_all = True
        # invoke 内部用 asyncio.run, 直接在 async 端点里调会炸, 放线程池里跑
        # thread_id 必须传进去, 否则 invoke 用默认 "default", 不同对话历史全混一起
        # 180s 超时兜底: DeepSeek 对长输出推理慢 + 内部 3 次重试退避可能累计很久
        timeout = float(params.get("timeout", 180))
        state = await asyncio.wait_for(
            asyncio.to_thread(
                agent.invoke,
                user_message,
                thread_id=thread_id,
            ),
            timeout=timeout,
        )
        messages = state.get("messages", [])
        content = ""
        if messages and hasattr(messages[-1], "content"):
            content = messages[-1].content
        # Questions 机制: chat() 提前返回时 state 带 clarify_questions + needs_clarification
        # 透传给调用方, 让测试脚本能识别这是追问而非正常回复
        clarify_questions = state.get("clarify_questions") if isinstance(state, dict) else None
        needs_clarification = bool(clarify_questions)
        # 提取工具调用 trace, 让测试脚本能评估工具是否被调用
        # 遍历全部消息, AIMessage.tool_calls 记调用请求, ToolMessage 记返回结果
        tool_trace = []
        for msg in messages:
            msg_type = getattr(msg, "type", None)
            if msg_type == "tool":
                tool_trace.append({
                    "type": "tool_result",
                    "name": getattr(msg, "name", "unknown"),
                    "content_preview": str(getattr(msg, "content", ""))[:200],
                })
            elif hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_trace.append({
                        "type": "tool_call",
                        "name": tc.get("name", "unknown"),
                        "args_preview": str(tc.get("args", ""))[:200],
                    })
        # 空回复通常是前一轮 timeout 后 checkpointer 状态不一致,
        # 返回明确错误让测试脚本能识别, 而不是当作正常空回复
        # 但 Questions 机制的空 content 是正常的(追问走 clarify_questions 字段)
        # 例外: tool_trace 非空说明 agent 调过工具但 LLM 没生成总结文本,
        # 这种不算空回复(工具执行了), 返回提示让调用方看 tool_trace
        if not content and not needs_clarification:
            if tool_trace:
                tool_names_seen = sorted({t.get("name", "?") for t in tool_trace if t.get("name")})
                # LLM 调完工具没写总结时, 把 tool_result 摘要拼进 content,
                # 调用方不用再翻 tool_trace 就能看到实际结果
                result_summaries = [
                    f"[{t.get('name', '?')}] {(t.get('content_preview', '') or '')[:200]}"
                    for t in tool_trace
                    if t.get("type") == "tool_result"
                ][:5]
                result_block = "\n".join(result_summaries) if result_summaries else "(无工具结果返回)"
                return {
                    "agent_id": agent_id,
                    "content": (
                        f"本轮调用了 {len(tool_trace)} 个工具操作 "
                        f"({', '.join(tool_names_seen[:5])}), LLM 未生成总结文本, "
                        f"以下是工具返回结果:\n\n{result_block}"
                    ),
                    "thread_id": thread_id,
                    "tool_trace": tool_trace,
                }
            return {
                "agent_id": agent_id,
                "content": "",
                "tool_trace": tool_trace,
                "error": "empty_reply: agent returned no content (state may be corrupted by prior timeout)",
            }
        result = {
            "agent_id": agent_id,
            "content": content,
            "thread_id": thread_id,
            "tool_trace": tool_trace,
        }
        if needs_clarification:
            result["clarify_questions"] = clarify_questions
            result["needs_clarification"] = True
        return result
    except asyncio.TimeoutError:
        # 分段返回: 超时不等于全白干. LangGraph checkpointer 每个 superstep
        # 都落了盘 (SqliteSaver 走文件锁, InMemorySaver 走 factory 共享实例),
        # 同一 thread_id 下次请求会从 checkpoint 接着跑, 不用从头再来.
        # 这里把 thread_id + resume_hint 带回去, 调用方说声"继续"就能接续.
        #
        # 注意: asyncio.to_thread 超时后底层线程杀不掉, 还会继续跑一阵子
        # 写 checkpoint. 调用方最好等一下再续, 避免两个 run 抢同一个
        # thread_id 的 checkpoint. 彻底解决要换进程隔离/任务队列, 那是
        # round8 计划里的 Layer 3 流式返回, 这里先做最小可行的分段返回.
        msg_text = user_message or ""
        # 检测多语言混合: 非拉丁字符集种类多时 API 处理可能异常慢
        # 中文很常见不算触发条件, 只数非中非拉丁的文字 (阿拉伯/韩/俄/日等)
        non_latin_blocks: set[str] = set()
        for ch in msg_text:
            cp = ord(ch)
            if 0x0600 <= cp <= 0x06FF:
                non_latin_blocks.add("Arabic")
            elif 0xAC00 <= cp <= 0xD7AF:
                non_latin_blocks.add("Hangul")
            elif 0x0400 <= cp <= 0x04FF:
                non_latin_blocks.add("Cyrillic")
            elif 0x3040 <= cp <= 0x309F:
                non_latin_blocks.add("Hiragana")
            elif 0x30A0 <= cp <= 0x30FF:
                non_latin_blocks.add("Katakana")
        multilingual_hint = ""
        if len(non_latin_blocks) >= 3:
            multilingual_hint = (
                f" 检测到输入含多种非拉丁文字 ({', '.join(sorted(non_latin_blocks))}), "
                f"这类多语言混合可能让 API 处理异常慢, 建议拆分输入或换中英文表达."
            )
        return {
            "error": f"agent invoke timed out after {timeout}s",
            "agent_id": agent_id,
            "thread_id": thread_id,
            "partial": True,
            "elapsed": timeout,
            "resume_hint": (
                f"本轮 {timeout}s 没跑完, 但已完成的步骤已存入 checkpoint "
                f"(thread_id={thread_id}). 用同一个 thread_id 再发一条消息 "
                f"(比如'继续上一轮的 LLZO 结构构建'), agent 会从中断处接续, "
                f"不会从头再来. 建议稍等几秒再续, 避免跟还在跑的上一个请求抢状态."
                f"{multilingual_hint}"
            ),
        }
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


# ── Personas ─────────────────────────────────────────────────────


@router.get("/personas")
async def list_personas() -> dict[str, Any]:
    """List available personas and the current default."""
    from huginn.personas import PersonaManager

    try:
        mgr = PersonaManager(workspace=get_context().config.workspace)
        return {
            "default": mgr.get_default_name(),
            "personas": [
                {
                    "name": name,
                    "system_prompt": mgr.get(name).system_prompt[:200],
                    "begin_dialogs": mgr.get(name).begin_dialogs,
                    "avatar": mgr.get(name).avatar,
                }
                for name in mgr.list()
            ],
        }
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": str(e)}


@router.get("/personas/templates")
async def list_persona_templates() -> dict[str, Any]:
    """列出内置 persona 模板库."""
    from huginn.personas import PersonaManager

    try:
        mgr = PersonaManager(workspace=get_context().config.workspace)
        templates = mgr.list_templates()
        return {
            "success": True,
            "count": len(templates),
            "templates": [
                {
                    "name": t.get("name"),
                    "description": t.get("description", ""),
                    "when_to_use": t.get("when_to_use", []),
                    "default_values": t.get("default_values", {}),
                }
                for t in templates
            ],
        }
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": str(e)}


@router.post("/personas/from-template")
async def create_persona_from_template(params: dict[str, Any]) -> dict[str, Any]:
    """从模板实例化 persona. body: template_name + 可选 overrides (占位符值/name/description)."""
    from huginn.personas import PersonaManager

    try:
        template_name = params.get("template_name") or params.get("template")
        if not template_name:
            return {"success": False, "error": "template_name is required"}
        overrides = params.get("overrides", {}) or {}
        # 也允许把顶层字段直接当 overrides 传
        for key in ("name", "description", "when_to_use", "user_name", "language", "audience", "target"):
            if key in params and key not in overrides:
                overrides[key] = params[key]

        mgr = PersonaManager(workspace=get_context().config.workspace)
        p = mgr.instantiate_template(template_name, **overrides)
        return {"success": True, "persona": p.to_dict()}
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": str(e)}


@router.post("/personas/import")
async def import_persona(params: dict[str, Any]) -> dict[str, Any]:
    """从 Nuwa 格式 markdown 文本导入 persona. body: markdown + overwrite."""
    from huginn.personas import PersonaManager

    try:
        markdown = params.get("markdown") or params.get("content")
        if not markdown:
            return {"success": False, "error": "markdown is required"}
        overwrite = bool(params.get("overwrite", False))

        mgr = PersonaManager(workspace=get_context().config.workspace)
        p = mgr.import_persona(markdown, overwrite=overwrite)
        return {"success": True, "persona": p.to_dict()}
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": str(e)}


@router.get("/personas/{name}")
async def get_persona(name: str) -> dict[str, Any]:
    """Get a single persona."""
    from huginn.personas import PersonaManager

    try:
        mgr = PersonaManager(workspace=get_context().config.workspace)
        p = mgr.get(name)
        return {
            "success": True,
            "name": p.name,
            "system_prompt": p.system_prompt,
            "begin_dialogs": p.begin_dialogs,
            "mood_dialogs": p.mood_dialogs,
            "variables": p.variables,
            "avatar": p.avatar,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.put("/personas/{name}")
async def update_persona(name: str, params: dict[str, Any]) -> dict[str, Any]:
    """更新一个已存在的 persona (内置 persona 拒绝修改)."""
    from huginn.personas import PersonaManager

    try:
        mgr = PersonaManager(workspace=get_context().config.workspace)
        p = mgr.update_persona(name, **params)
        return {"success": True, "persona": p.to_dict()}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/personas/{name}/export")
async def export_persona(name: str) -> dict[str, Any]:
    """把 persona 导出为 Nuwa 格式 markdown 字符串."""
    from huginn.personas import PersonaManager

    try:
        mgr = PersonaManager(workspace=get_context().config.workspace)
        markdown = mgr.export_persona(name)
        return {"success": True, "name": name, "markdown": markdown}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/personas")
async def create_persona(params: dict[str, Any]) -> dict[str, Any]:
    """Create a new persona."""
    from huginn.personas import PersonaManager

    try:
        mgr = PersonaManager(workspace=get_context().config.workspace)
        p = mgr.create(
            name=params["name"],
            system_prompt=params.get("system_prompt", ""),
            begin_dialogs=params.get("begin_dialogs", []),
            mood_dialogs=params.get("mood_dialogs", []),
            variables=params.get("variables", {}),
            avatar=params.get("avatar"),
        )
        return {"success": True, "persona": p.to_dict()}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/personas/match")
async def match_persona(params: dict[str, Any]) -> dict[str, Any]:
    """Match a query to the most suitable persona."""
    from huginn.persona_matcher import PersonaMatcher
    from huginn.personas import PersonaManager

    try:
        mgr = PersonaManager(workspace=get_context().config.workspace)
        matcher = PersonaMatcher(manager=mgr)
        results = matcher.match(
            params.get("query", ""),
            top_k=int(params.get("top_k", 3)),
            score_threshold=float(params.get("threshold", 0.3)),
        )
        return {
            "success": True,
            "matches": [
                {
                    "name": p.name,
                    "score": float(score),
                    "description": p.description,
                    "when_to_use": p.when_to_use,
                }
                for p, score in results
            ],
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.patch("/personas/{name}/default")
async def set_default_persona(name: str) -> dict[str, Any]:
    """Set the default persona."""
    from huginn.personas import PersonaManager

    try:
        mgr = PersonaManager(workspace=get_context().config.workspace)
        mgr.set_default(name)
        return {"success": True, "default": mgr.get_default_name()}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.delete("/personas/{name}")
async def delete_persona(name: str) -> dict[str, Any]:
    """Delete a user-defined persona."""
    from huginn.personas import PersonaManager

    try:
        mgr = PersonaManager(workspace=get_context().config.workspace)
        mgr.delete(name)
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/personas/{name}/switch")
async def switch_persona(name: str, params: dict[str, Any]) -> dict[str, Any]:
    """Switch the active persona for the current chat session."""
    from huginn.persona_emotion import EmotionTracker
    from huginn.personas import PersonaManager

    try:
        mgr = PersonaManager(workspace=get_context().config.workspace)
        p = mgr.get(name)
        os.environ["HUGINN_PERSONA"] = name
        get_context().agent = None  # force re-init with new default persona
        tracker = EmotionTracker(name, workspace=get_context().config.workspace)
        return {
            "success": True,
            "persona": p.name,
            "system_prompt": p.system_prompt,
            "begin_dialogs": p.begin_dialogs,
            "emotion": tracker.current_state().to_dict(),
            "context_prompt": tracker.context_prompt(),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/personas/{name}/emotion")
async def get_persona_emotion(name: str) -> dict[str, Any]:
    """Return the current emotional trajectory for a persona."""
    from huginn.persona_emotion import EmotionTracker

    try:
        tracker = EmotionTracker(name, workspace=get_context().config.workspace)
        state = tracker.current_state()
        return {
            "success": True,
            "persona": name,
            "state": state.to_dict(),
            "context_prompt": tracker.context_prompt(),
            "trajectory": tracker.trajectory(),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ── Orchestration ────────────────────────────────────────────────


@router.post("/orchestrate")
async def orchestrate(params: dict[str, Any]) -> dict[str, Any]:
    """Run the multi-agent orchestrator on an objective."""
    try:
        factory = get_agent_factory()
        from huginn.agents.orchestrator import Orchestrator

        orch = Orchestrator(
            factory=factory,
            memory_manager=get_memory_manager(),
            max_concurrent=params.get(
                "max_concurrent", factory.config.max_concurrent_subagents
            ),
        )
        result = await orch.run(params.get("objective", ""))
        return {
            "success": result.success,
            "objective": result.objective,
            "plan": [
                {
                    "task_id": t.task_id,
                    "agent_id": t.agent_id,
                    "status": t.status,
                    "prompt": t.prompt,
                }
                for t in result.plan.tasks
            ],
            "outputs": result.outputs,
            "summary": result.summary,
            "error": result.error,
        }
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": str(e)}


# ── Telemetry ────────────────────────────────────────────────────


@router.get("/telemetry/summary")
async def telemetry_summary() -> dict[str, Any]:
    """Return coarse telemetry summary for the global agent."""
    try:
        agent = await get_agent()
        return {"summary": agent.telemetry_summary()}
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


@router.get("/telemetry/spans")
async def telemetry_spans() -> dict[str, Any]:
    """Return all recorded telemetry spans for the global agent."""
    try:
        agent = await get_agent()
        return {"spans": agent.telemetry_spans()}
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


# ── Swarm ────────────────────────────────────────────────────────


@router.post("/swarm/run")
async def swarm_run(params: dict[str, Any]) -> dict[str, Any]:
    """Run a task through the multi-agent swarm."""
    from huginn.agents.swarm import AgentRole, HuginnSwarm, SwarmAgent

    try:
        agent = await get_agent()
        task = params.get("task", "")
        if not task:
            return {"error": "task is required"}

        workers = [
            SwarmAgent(
                "planner", AgentRole.PLANNER, agent, "Break the task into steps."
            ),
            SwarmAgent(
                "scientist", AgentRole.SCIENTIST, agent, "Choose physical models."
            ),
            SwarmAgent("coder", AgentRole.CODER, agent, "Write code or tool calls."),
            SwarmAgent("executor", AgentRole.EXECUTOR, agent, "Run the solution."),
            SwarmAgent("critic", AgentRole.CRITIC, agent, "Review correctness."),
        ]
        result = await HuginnSwarm(workers).run(task)
        return {"success": True, **result}
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": str(e)}


# ── Personalization ──────────────────────────────────────────────


@router.get("/personalization/style")
async def get_style_profile() -> dict[str, Any]:
    """返回当前用户的语言偏好 profile."""
    from dataclasses import asdict

    from huginn.personalization import get_shared_style_learner

    try:
        learner = get_shared_style_learner()
        return {"success": True, "profile": asdict(learner.get_profile())}
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": str(e)}


@router.post("/personalization/style/reset")
async def reset_style_profile() -> dict[str, Any]:
    """重置用户语言偏好 profile, 清掉所有学习结果和手动设置."""
    from huginn.personalization import get_shared_style_learner

    try:
        learner = get_shared_style_learner()
        learner.reset()
        return {"success": True}
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": str(e)}


@router.post("/personalization/style/feedback")
async def style_feedback(params: dict[str, Any]) -> dict[str, Any]:
    """用户显式反馈.

    body 两种形式:
    - {term: "顺便说一下", action: "avoid"} — 标记避免用词
    - {dimension: "verbosity", value: "concise"} — 手动设某维度, 覆盖学习结果
    """
    from dataclasses import asdict

    from huginn.personalization import get_shared_style_learner

    try:
        learner = get_shared_style_learner()
        learner.apply_feedback(
            term=params.get("term"),
            action=params.get("action"),
            dimension=params.get("dimension"),
            value=params.get("value"),
        )
        return {"success": True, "profile": asdict(learner.get_profile())}
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": str(e)}


@router.get("/personalization/style/directive")
async def get_style_directive() -> dict[str, Any]:
    """返回当前 style directive 文本, 调试用."""
    from huginn.personalization import get_shared_style_learner

    try:
        learner = get_shared_style_learner()
        return {"success": True, "directive": learner.get_style_directive()}
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": str(e)}
