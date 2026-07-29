"""Skill tool — lets the LLM invoke preset scientific workflow skills.

Wraps DeclarativeSkillExecutor so the agent can list, describe, and run
named skills (DFT, MD, phonon, band structure, etc.) during a conversation.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Literal

from pydantic import BaseModel, Field

# Importing presets has a side effect: every SkillDefinition passed to
# register_skill() lands in SkillRegistry. Keep this import above the
# tool class so the registry is populated before the first call.
import huginn.skills.presets  # noqa: F401
from huginn.skills.base import (
    DeclarativeSkillExecutor,
    SkillDefinition,
    SkillParameter,
    SkillStep,
)
from huginn.skills.registry import SkillRegistry
from huginn.tools.base import HuginnTool
from huginn.tools.registry import ToolRegistry
from huginn.types import ToolContext, ToolResult
from huginn.memory.longterm import LongTermMemory

logger = logging.getLogger(__name__)


# ponytail: memory 单例懒加载, 第一次初始化失败就标死不再试.
# 升级路径: 加 healthcheck 探活 + 降级到 in-memory 缓存.
_memory_singleton: LongTermMemory | None = None
_memory_broken: bool = False


def _get_memory() -> LongTermMemory | None:
    """拿 LongTermMemory 单例. SQLite 文件路径走默认 (~/.huginn/memory.db),
    多个调用方共享同一份库. 任何初始化异常都吞掉返回 None."""
    global _memory_singleton, _memory_broken
    if _memory_broken:
        return None
    if _memory_singleton is None:
        try:
            _memory_singleton = LongTermMemory()
        except Exception:
            _memory_broken = True
            return None
    return _memory_singleton


class SkillToolInput(BaseModel):
    action: Literal["list", "execute", "describe", "distill_workflow"] = Field(
        default="list"
    )
    skill_name: str | None = Field(
        default=None,
        description=(
            "Name of the skill to execute/describe, OR override name for a "
            "distilled skill (distill_workflow). Optional for distill_workflow."
        ),
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Parameters to pass to the skill"
    )
    thread_id: str | None = Field(
        default=None, description="Thread ID for context isolation"
    )
    conversation_trace: list[dict[str, Any]] | None = Field(
        default=None,
        description=(
            "Conversation history for distill_workflow. Each entry: "
            "{role, content, tool_calls?}. Ignored by other actions."
        ),
    )


class SkillToolOutput(BaseModel):
    success: bool
    action: str
    skill_name: str | None = None
    result: Any = None
    error: str | None = None
    available_skills: list[dict] | None = None


def _skill_summary(skill: SkillDefinition) -> dict:
    """Compact dict representation used by list and describe actions."""
    return {
        "name": skill.name,
        "description": skill.description,
        "category": skill.category,
        "tags": list(skill.tags),
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
        "required_tools": list(skill.required_tools),
        "required_env_vars": list(skill.required_env_vars),
        "references": list(skill.references),
        "domain": skill.domain,
        "stage": skill.stage,
        "function": skill.function,
        "steps": [
            {
                "name": s.name,
                "tool": s.tool,
                "output_key": s.output_key,
                "on_failure": s.on_failure,
            }
            for s in skill.steps
        ],
    }


# ── distill_workflow: 从对话 trace 蒸馏 SkillDefinition ────────────────────
# 设计参考 huginn.tools.literature.schema_extractor.chat_json 的 LLM 调用约定:
# model.ainvoke(messages) + 去 code fence + json.loads. 不引入新依赖.
# ponytail: 与 schema_extractor._strip_code_fences 重复, 升级路径是抽到公共 util.
DISTILL_SYSTEM_PROMPT = """You are a skill distiller. Given a conversation trace, extract a reusable SkillDefinition as JSON.

A SkillDefinition describes a reproducible workflow made of tool-call steps. Output STRICTLY a JSON object with this schema:
{
  "name": "snake_case skill name",
  "description": "one-sentence purpose",
  "category": "computation | analysis | diagnostics | reporting | distilled",
  "parameters": [
    {"name": "param_name", "type": "string|int|float|bool|list|dict", "description": "...", "required": true, "default": null}
  ],
  "steps": [
    {
      "name": "step name",
      "tool": "tool name exactly as seen in trace",
      "input_mapping": {"tool_arg": "$context_key_or_literal"},
      "output_key": "context_key_for_result",
      "on_failure": "abort | skip | retry"
    }
  ],
  "required_tools": ["tool names referenced by steps"],
  "tags": ["distilled"]
}

Rules:
- Only include steps that are reproducible. Skip one-off clarifications, chit-chat, and failed explorations.
- Reuse tool names exactly as they appear in the trace's tool_calls.
- input_mapping values starting with "$" reference context keys (e.g. "$structure_path"); other values are literals.
- Output ONLY the JSON object. No prose, no code fences, no explanation."""


def _strip_code_fences(text: str) -> str:
    """去掉 ```json ... ``` 包裹."""
    if not text:
        return ""
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]
            if text.startswith("json"):
                text = text[4:]
    return text.strip()


def _get_llm_model(context: ToolContext) -> Any:
    """从 context.config 拿默认 LLM 实例. config 没有就走 from_env.
    复用 LiteraturePipelineTool._get_model_for_extract 的同款 pattern."""
    from huginn.models.registry import ModelRegistry

    cfg = getattr(context, "config", None)
    if cfg is None:
        from huginn.config import HuginnConfig
        cfg = HuginnConfig.from_env()
    registry = ModelRegistry.from_config(cfg)
    alias = registry.default_alias()
    if not alias:
        raise ValueError("registry 无可用 alias, 请配置 models")
    return registry.get(alias)


async def _llm_distill_trace(model: Any, trace: list[dict]) -> dict:
    """调 LLM 把对话 trace 蒸馏成 SkillDefinition 字段. 失败返回 {}.
    trace 截断到 12k 字符防止单次请求超长."""
    trace_text = json.dumps(trace, ensure_ascii=False, default=str)[:12000]
    messages = [
        {"role": "system", "content": DISTILL_SYSTEM_PROMPT},
        {"role": "user", "content": f"Conversation trace:\n{trace_text}\n\nReturn JSON now."},
    ]
    try:
        resp = await model.ainvoke(messages)
    except Exception as exc:
        logger.warning("distill_workflow LLM 调用失败: %s", exc)
        return {}
    text = str(resp.content).strip() if hasattr(resp, "content") else str(resp).strip()
    text = _strip_code_fences(text)
    if not text:
        return {}
    try:
        result = json.loads(text)
        return result if isinstance(result, dict) else {}
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.warning("distill_workflow JSON 解析失败: %s", exc)
        return {}


def _heuristic_distill_trace(trace: list[dict]) -> dict:
    """LLM 不可用时的 fallback: 用正则从 trace 的 tool_calls 抽 step.
    每个唯一 tool 名产一条 step, 不解析参数 (LLM 该干)."""
    steps: list[dict] = []
    required_tools: list[str] = []
    seen: set[str] = set()
    for entry in trace:
        if not isinstance(entry, dict):
            continue
        tool_calls = entry.get("tool_calls") or []
        if not isinstance(tool_calls, list):
            continue
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            name = tc.get("name") or tc.get("tool") or ""
            if not name or name in seen:
                continue
            seen.add(name)
            idx = len(steps)
            steps.append({
                "name": f"step_{idx}_{name}",
                "tool": name,
                "input_mapping": {},
                "output_key": f"step_{idx}_out",
                "on_failure": "skip",
            })
            required_tools.append(name)
    if not steps:
        return {}
    # 名字从第一条 user 消息抽, 抽不到用默认
    name = "distilled_workflow"
    for entry in trace:
        if isinstance(entry, dict) and entry.get("role") == "user":
            content = str(entry.get("content", ""))[:60]
            cleaned = re.sub(r"[^a-zA-Z0-9_]", "_", content).strip("_").lower()
            if cleaned and len(cleaned) >= 4:
                name = f"distilled_{cleaned[:40]}"
            break
    return {
        "name": name,
        "description": "Heuristically distilled from conversation trace (LLM unavailable).",
        "category": "distilled",
        "parameters": [],
        "steps": steps,
        "required_tools": required_tools,
        "tags": ["distilled", "heuristic"],
    }


def _build_skill_definition(spec: dict) -> SkillDefinition:
    """从 LLM/heuristic 返回的 dict 构造 SkillDefinition. 字段缺失抛 ValueError."""
    name = spec.get("name")
    if not name:
        raise ValueError("skill spec missing 'name'")
    steps: list[SkillStep] = []
    for i, s in enumerate(spec.get("steps") or []):
        if not isinstance(s, dict):
            continue
        tool = s.get("tool")
        if not tool:
            continue
        steps.append(SkillStep(
            name=s.get("name") or f"step_{i}",
            tool=tool,
            input_mapping=s.get("input_mapping") or {},
            output_key=s.get("output_key") or f"step_{i}_out",
            on_failure=s.get("on_failure") or "skip",
        ))
    if not steps:
        raise ValueError("skill spec has no valid steps")
    params: list[SkillParameter] = []
    for p in (spec.get("parameters") or []):
        if not isinstance(p, dict):
            continue
        pname = p.get("name")
        if not pname:
            continue
        params.append(SkillParameter(
            name=pname,
            type=p.get("type") or "any",
            description=p.get("description") or "",
            required=bool(p.get("required", True)),
            default=p.get("default"),
        ))
    return SkillDefinition(
        name=name,
        description=spec.get("description") or "",
        category=spec.get("category") or "distilled",
        parameters=params,
        steps=steps,
        required_tools=list(spec.get("required_tools") or []),
        tags=list(spec.get("tags") or ["distilled"]),
    )


class SkillTool(HuginnTool[SkillToolInput, SkillToolOutput]):
    """Bridge between the LLM and the declarative skill executor."""

    name = "skill"
    category = "meta"
    description = (
        "Execute preset scientific workflow skills (DFT, MD, phonon, band "
        "structure, etc.). Use 'list' action to see available skills."
    )
    destructive = False
    read_only = False  # skills run real tools, so they can have side effects
    input_schema = SkillToolInput
    output_schema = SkillToolOutput

    def __init__(self, skill_executor: DeclarativeSkillExecutor | None = None) -> None:
        super().__init__()
        if skill_executor is None:
            skill_executor = DeclarativeSkillExecutor(ToolRegistry)
        self._executor = skill_executor

    async def call(self, args: SkillToolInput, context: ToolContext) -> ToolResult:
        if args.action == "list":
            return self._list_skills()
        if args.action == "describe":
            return self._describe_skill(args)
        if args.action == "execute":
            return await self._execute_skill(args, context)
        if args.action == "distill_workflow":
            return await self._distill_workflow(args, context)

        msg = f"Unknown action: {args.action}"
        return ToolResult(
            data=SkillToolOutput(success=False, action=args.action, error=msg).model_dump(),
            success=False,
            error=msg,
        )

    # -- actions -----------------------------------------------------------

    def _list_skills(self) -> ToolResult:
        skills = SkillRegistry.get_all_definitions()
        summaries = [_skill_summary(s) for s in skills]
        out = SkillToolOutput(
            success=True,
            action="list",
            available_skills=summaries,
            result={"count": len(summaries)},
        )
        return ToolResult(data=out.model_dump(), success=True)

    def _describe_skill(self, args: SkillToolInput) -> ToolResult:
        if not args.skill_name:
            return self._missing_name("describe")

        skill = SkillRegistry.get(args.skill_name)
        if skill is None:
            return self._not_found(args.skill_name, "describe")

        out = SkillToolOutput(
            success=True,
            action="describe",
            skill_name=skill.name,
            result=_skill_summary(skill),
        )
        return ToolResult(data=out.model_dump(), success=True)

    async def _execute_skill(
        self, args: SkillToolInput, context: ToolContext
    ) -> ToolResult:
        if not args.skill_name:
            return self._missing_name("execute")

        skill = SkillRegistry.get(args.skill_name)
        if skill is None:
            return self._not_found(args.skill_name, "execute")

        # Build a context dict for the executor. Params are merged on top
        # of this inside DeclarativeSkillExecutor.execute, so user-supplied
        # values always win.
        exec_context: dict[str, Any] = {}
        if args.thread_id:
            exec_context["thread_id"] = args.thread_id
        if context.session_id:
            exec_context["session_id"] = context.session_id

        # 调 skill 前先 recall 历史, 塞到 exec_context 给 executor 当 hint.
        # ponytail: memory recall/remember 用 try/except 包失败静默;
        # 升级路径是加 memory 可用性检查 + 降级策略
        history_hint = self._recall_skill_history(args.skill_name)
        if history_hint:
            exec_context["_skill_history_hint"] = history_hint

        try:
            result = await self._executor.execute(skill, args.parameters, exec_context)
        except Exception as exc:
            self._record_skill_invocation(args.skill_name, success=False, error=str(exc))
            out = SkillToolOutput(
                success=False,
                action="execute",
                skill_name=skill.name,
                error=str(exc),
            )
            return ToolResult(data=out.model_dump(), success=False, error=str(exc))

        success = bool(result.get("success", False))
        self._record_skill_invocation(args.skill_name, success=success, result=result)

        out = SkillToolOutput(
            success=success,
            action="execute",
            skill_name=skill.name,
            result=result,
        )
        return ToolResult(data=out.model_dump(), success=out.success)

    async def _distill_workflow(
        self, args: SkillToolInput, context: ToolContext
    ) -> ToolResult:
        """从对话 trace 蒸馏出 SkillDefinition 并注册到 SkillRegistry.
        走 LLM 抽取; LLM 不可用 / 失败时 fallback 到正则抽 tool_calls."""
        trace = args.conversation_trace or []
        if not trace:
            msg = "conversation_trace is required for distill_workflow"
            return ToolResult(
                data=SkillToolOutput(
                    success=False, action="distill_workflow", error=msg
                ).model_dump(),
                success=False,
                error=msg,
            )

        # 先试 LLM 蒸馏; 失败/空结果 fallback 到 heuristic
        spec: dict = {}
        try:
            model = _get_llm_model(context)
            spec = await _llm_distill_trace(model, trace)
        except Exception as exc:
            logger.warning("distill_workflow 取模型/调用失败, 走 heuristic: %s", exc)
            spec = {}

        if not spec:
            spec = _heuristic_distill_trace(trace)

        if not spec or not spec.get("steps"):
            msg = "failed to distill skill from trace (no reproducible steps)"
            self._record_skill_invocation("distill_workflow", success=False, error=msg)
            return ToolResult(
                data=SkillToolOutput(
                    success=False, action="distill_workflow", error=msg
                ).model_dump(),
                success=False,
                error=msg,
            )

        # 用户指定的 skill_name 覆盖 LLM/heuristic 给的
        if args.skill_name:
            spec["name"] = args.skill_name

        try:
            skill = _build_skill_definition(spec)
        except (KeyError, TypeError, ValueError) as exc:
            msg = f"invalid skill spec from trace: {exc}"
            self._record_skill_invocation("distill_workflow", success=False, error=msg)
            return ToolResult(
                data=SkillToolOutput(
                    success=False, action="distill_workflow", error=msg
                ).model_dump(),
                success=False,
                error=msg,
            )

        try:
            SkillRegistry.register(skill)
        except Exception as exc:
            msg = f"failed to register distilled skill: {exc}"
            self._record_skill_invocation("distill_workflow", success=False, error=msg)
            return ToolResult(
                data=SkillToolOutput(
                    success=False, action="distill_workflow", error=msg
                ).model_dump(),
                success=False,
                error=msg,
            )

        self._record_skill_invocation(
            "distill_workflow", success=True, result={"name": skill.name}
        )
        out = SkillToolOutput(
            success=True,
            action="distill_workflow",
            skill_name=skill.name,
            result=_skill_summary(skill),
        )
        return ToolResult(data=out.model_dump(), success=True)

    # -- memory helpers ----------------------------------------------------

    def _recall_skill_history(self, skill_name: str) -> str | None:
        """从长期记忆拉该 skill 的历史调用记录, 拼成提示串.
        memory 不可用 / 出错 / 没历史都返回 None, 不影响 skill 主流程."""
        mem = _get_memory()
        if mem is None:
            return None
        try:
            rows = mem.retrieve(
                query=skill_name,
                category="skill_invocation",
                top_k=3,
            )
        except Exception:
            return None
        if not rows:
            return None
        lines = []
        for r in rows:
            ts = (r.get("created_at") or "")[:19]
            content = r.get("content") or ""
            lines.append(f"[{ts}] {content}")
        return "历史调用记录:\n" + "\n".join(lines)

    def _record_skill_invocation(
        self,
        skill_name: str,
        success: bool,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        """落一条 skill_invocation 记忆. 任何异常都吞掉, 不阻断 skill 调用."""
        mem = _get_memory()
        if mem is None:
            return
        summary = f"success={success}"
        if error:
            summary += f" error={error[:200]}"
        elif result is not None:
            # 截断防止单条记忆太长, 300 字符够复盘也够 FTS 检索
            summary += f" result={str(result)[:300]}"
        content = f"skill={skill_name} {summary}"
        try:
            mem.store(
                content=content,
                category="skill_invocation",
                tags=[skill_name],
                source=f"skill_tool:{skill_name}",
                importance=0.5,
                tier="mid",
            )
        except Exception:
            pass

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _missing_name(action: str) -> ToolResult:
        msg = f"skill_name is required for {action} action"
        return ToolResult(
            data=SkillToolOutput(success=False, action=action, error=msg).model_dump(),
            success=False,
            error=msg,
        )

    @staticmethod
    def _not_found(skill_name: str, action: str) -> ToolResult:
        msg = f"Skill '{skill_name}' not found"
        return ToolResult(
            data=SkillToolOutput(
                success=False,
                action=action,
                skill_name=skill_name,
                error=msg,
            ).model_dump(),
            success=False,
            error=msg,
        )


if __name__ == "__main__":
    # 自检: 不调真实 LLM/网络, 只验证蒸馏的纯逻辑路径.
    # 1. _strip_code_fences
    assert _strip_code_fences("```json\n{\"a\":1}\n```") == '{"a":1}'
    assert _strip_code_fences('{"a":1}') == '{"a":1}'
    assert _strip_code_fences("") == ""

    # 2. _build_skill_definition 合法 spec
    spec = {
        "name": "test_distill",
        "description": "self-check skill",
        "category": "distilled",
        "parameters": [
            {"name": "x", "type": "int", "description": "in", "required": True},
            {"name": "y", "type": "str", "description": "opt", "required": False, "default": "dft"},
        ],
        "steps": [
            {"name": "s1", "tool": "relax_tool", "input_mapping": {"x": "$x"},
             "output_key": "relaxed", "on_failure": "abort"},
            {"name": "s2", "tool": "scf_tool", "input_mapping": {"struct": "$relaxed"},
             "output_key": "scf_out", "on_failure": "skip"},
        ],
        "required_tools": ["relax_tool", "scf_tool"],
        "tags": ["distilled", "test"],
    }
    skill = _build_skill_definition(spec)
    assert skill.name == "test_distill"
    assert len(skill.steps) == 2
    assert skill.steps[0].tool == "relax_tool"
    assert skill.steps[0].on_failure == "abort"
    assert len(skill.parameters) == 2
    assert skill.parameters[1].default == "dft"
    assert "test" in skill.tags

    # 3. _build_skill_definition 缺 name 抛 ValueError
    try:
        _build_skill_definition({"steps": [{"tool": "x"}]})
        raise AssertionError("应抛 ValueError")
    except ValueError:
        pass

    # 4. _build_skill_definition 空 steps 抛 ValueError
    try:
        _build_skill_definition({"name": "empty"})
        raise AssertionError("应抛 ValueError")
    except ValueError:
        pass

    # 5. _heuristic_distill_trace 从 trace 抽 tool_calls
    trace = [
        {"role": "user", "content": "Relax SiO2 structure and run SCF"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"name": "relax_tool", "args": {"formula": "SiO2"}},
            {"name": "scf_tool", "args": {}},
        ]},
        {"role": "assistant", "content": "", "tool_calls": [
            {"name": "relax_tool", "args": {}},  # 重复, 应被去重
        ]},
    ]
    out = _heuristic_distill_trace(trace)
    assert out["name"].startswith("distilled_")
    assert len(out["steps"]) == 2  # relax_tool + scf_tool, 重复的去掉了
    assert out["steps"][0]["tool"] == "relax_tool"
    assert out["required_tools"] == ["relax_tool", "scf_tool"]
    assert "heuristic" in out["tags"]

    # 6. _heuristic_distill_trace 无 tool_calls 返回 {}
    assert _heuristic_distill_trace([{"role": "user", "content": "hi"}]) == {}

    print("skill_tool self-check OK")

