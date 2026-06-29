"""多模型团队编排 —— 让不同 LLM 各司其职, 协作完成任务.

和老的 HuginnSwarm 区别在于: swarm 是一个模型扮多个角色,
team 是真的有多个不同模型的 agent 组队, 每个成员用自己的 API
和擅长的能力 (coding / reasoning / vision) 承担不同步骤.

能力路由基于 models/registry.py 的 ModelCaps 声明:
  - planner  → reasoning 优先 (deepseek-reasoner / o1 / o3)
  - coder    → tools 必须 (claude-sonnet / gpt-4o / deepseek-coder)
  - scientist→ tools 优先, reasoning 加分
  - executor → tools 必须
  - critic   → 跟 planner 不同的模型, 避免自己审自己

只有一个模型配置时退化为单模型多角色 (向后兼容 swarm 行为).
"""

from __future__ import annotations

import asyncio
import enum
import json
import time
from dataclasses import dataclass, field
from typing import Any

from huginn.models.registry import ModelCaps, get_model_capabilities


class TeamRole(enum.StrEnum):
    PLANNER = "planner"
    SCIENTIST = "scientist"
    CODER = "coder"
    EXECUTOR = "executor"
    CRITIC = "critic"


# 角色 → 需要的能力 (按优先级排序)
# 第一个 bool 是"必须满足", 后面是"加分项"
ROLE_REQUIREMENTS: dict[TeamRole, tuple[set[str], set[str]]] = {
    # 规划阶段需要强推理, 不一定需要工具调用
    TeamRole.PLANNER: ({"reasoning"}, set()),
    # 科学分析需要工具调用 + 推理加分
    TeamRole.SCIENTIST: ({"tools"}, {"reasoning"}),
    # 写代码需要工具调用能力
    TeamRole.CODER: ({"tools"}, set()),
    # 执行工具同上
    TeamRole.EXECUTOR: ({"tools"}, set()),
    # 审查用不同模型即可, 没有硬性能力要求
    TeamRole.CRITIC: (set(), set()),
}


@dataclass
class TeamMember:
    """团队成员: 一个 agent 绑定一个角色.

    agent 在首次使用时才创建 (lazy), 避免启动时把所有模型的
    LangChain 实例都拉起来.
    """

    name: str
    profile_id: str
    role: TeamRole
    model_name: str = ""
    caps: ModelCaps = field(default_factory=ModelCaps)
    _agent: Any = None
    _config: Any = None

    def get_agent(self) -> Any:
        """延迟创建 agent 实例."""
        if self._agent is None:
            from huginn.agent import HuginnAgent

            if self._config is None:
                raise RuntimeError(
                    f"TeamMember '{self.name}' 没有关联 config, 无法创建 agent"
                )
            self._agent = HuginnAgent.from_config(
                self._config, profile_id=self.profile_id
            )
        return self._agent


@dataclass
class TeamStep:
    """执行计划中的一步."""

    id: str
    role: TeamRole
    task: str
    depends_on: list[str] = field(default_factory=list)


@dataclass
class TeamTrace:
    """一步执行的记录."""

    role: TeamRole
    member_name: str
    model_name: str
    input_task: str
    output: str
    duration_ms: float = 0.0


class ModelTeam:
    """多模型团队编排器.

    用法::

        team = ModelTeam.from_config(cfg)
        result = await team.run("帮我算一下 Si 的声子谱")
    """

    def __init__(self, members: list[TeamMember]) -> None:
        self.members: dict[TeamRole, TeamMember] = {}
        for m in members:
            self.assign(m)

    def assign(self, member: TeamMember) -> "ModelTeam":
        """把成员绑定到其声明的角色 (覆盖同角色的旧成员)."""
        self.members[member.role] = member
        return self

    @classmethod
    def from_config(cls, config: Any) -> "ModelTeam":
        """根据 HuginnConfig.agents 中的 profile 自动组建团队.

        策略:
        1. 遍历所有 enabled 的 agent profile
        2. 用 profile.id 做角色匹配 (profile id == 角色名直接绑定)
        3. 匹配不上的按 ModelCaps 自动分配到最合适的角色
        4. 只有一个 profile 时所有角色都用它 (兼容老 swarm)
        """
        from huginn.config import HuginnConfig

        if not isinstance(config, HuginnConfig):
            raise TypeError("需要 HuginnConfig 实例")

        profiles = [a for a in config.agents if a.enabled]
        if not profiles:
            return cls([])

        # 只有一个 profile: 所有角色都用它
        if len(profiles) == 1:
            p = profiles[0]
            model_name = _resolve_model_name(config, p.model_alias)
            caps = get_model_capabilities(model_name) if model_name else ModelCaps()
            members = [
                TeamMember(
                    name=f"{role.value}-{p.id}",
                    profile_id=p.id,
                    role=role,
                    model_name=model_name,
                    caps=caps,
                    _config=config,
                )
                for role in TeamRole
            ]
            return cls(members)

        # 多 profile: 先按 id 直接匹配角色, 剩下的按能力路由
        members: list[TeamMember] = []
        used_profiles: set[str] = set()
        assigned_roles: set[TeamRole] = set()

        # 第一轮: profile.id 和角色名同名的直接绑定
        for p in profiles:
            try:
                role = TeamRole(p.id)
            except ValueError:
                continue
            model_name = _resolve_model_name(config, p.model_alias)
            caps = get_model_capabilities(model_name) if model_name else ModelCaps()
            members.append(
                TeamMember(
                    name=f"{role.value}-{p.id}",
                    profile_id=p.id,
                    role=role,
                    model_name=model_name,
                    caps=caps,
                    _config=config,
                )
            )
            used_profiles.add(p.id)
            assigned_roles.add(role)

        # 第二轮: 剩余角色按能力从剩余 profile 中挑最合适的
        remaining_profiles = [p for p in profiles if p.id not in used_profiles]
        for role in TeamRole:
            if role in assigned_roles:
                continue
            best = _pick_best_profile(role, remaining_profiles, config)
            if best is None:
                # 实在没人了, 从已分配的里面借一个 (planner 和 critic 不能同一个)
                best = _pick_fallback_profile(role, members, assigned_roles)
                if best is None:
                    continue
                # 复用已有成员的 profile, 但起个新名字
                members.append(
                    TeamMember(
                        name=f"{role.value}-{best.profile_id}",
                        profile_id=best.profile_id,
                        role=role,
                        model_name=best.model_name,
                        caps=best.caps,
                        _config=config,
                    )
                )
            else:
                model_name = _resolve_model_name(config, best.model_alias)
                caps = (
                    get_model_capabilities(model_name) if model_name else ModelCaps()
                )
                members.append(
                    TeamMember(
                        name=f"{role.value}-{best.id}",
                        profile_id=best.id,
                        role=role,
                        model_name=model_name,
                        caps=caps,
                        _config=config,
                    )
                )
                remaining_profiles.remove(best)

        return cls(members)

    # ── 运行 ──────────────────────────────────────────────

    async def run(
        self, task: str, context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """跑完整流水线: 规划 → 按步骤执行 → 审查."""
        traces: list[TeamTrace] = []
        ctx = dict(context or {})
        ctx["original_task"] = task

        # 1. 规划
        plan_text = await self._delegate(TeamRole.PLANNER, task, ctx, traces)
        ctx["planner_output"] = plan_text
        steps = self._parse_plan(plan_text)
        if not steps:
            steps = self._default_plan(task)
        ctx["plan"] = self._plan_to_text(steps)

        # 2. 执行
        step_outputs = await self._execute_plan(steps, ctx, traces)
        role_outputs: dict[TeamRole, str] = {}
        for step, output in zip(steps, step_outputs):
            role_outputs[step.role] = output

        ctx["scientific_reasoning"] = role_outputs.get(TeamRole.SCIENTIST, "")
        ctx["code"] = role_outputs.get(TeamRole.CODER, "")
        ctx["execution_result"] = role_outputs.get(
            TeamRole.EXECUTOR, "No executor step completed."
        )

        # 3. 审查 (如果计划里没有)
        critic_output = role_outputs.get(TeamRole.CRITIC, "")
        if not critic_output and TeamRole.CRITIC in self.members:
            critic_input = (
                f"Task: {task}\n"
                f"Plan: {ctx['plan']}\n"
                f"Execution result: {ctx['execution_result']}"
            )
            critic_output = await self._delegate(
                TeamRole.CRITIC, critic_input, ctx, traces
            )
        ctx["review"] = critic_output

        return {
            "task": task,
            "context": ctx,
            "members": [
                {"role": m.role.value, "name": m.name, "model": m.model_name}
                for m in self.members.values()
            ],
            "trace": [self._trace_to_dict(t) for t in traces],
            "final_output": ctx["execution_result"],
            "review": ctx["review"],
        }

    # ── 内部方法 ──────────────────────────────────────────

    async def _delegate(
        self,
        role: TeamRole,
        task: str,
        ctx: dict[str, Any],
        traces: list[TeamTrace],
    ) -> str:
        member = self.members.get(role)
        if member is None:
            return ""
        return await self._run_member(member, task, ctx, traces)

    async def _run_member(
        self,
        member: TeamMember,
        task: str,
        ctx: dict[str, Any],
        traces: list[TeamTrace],
    ) -> str:
        start = time.time()
        agent = member.get_agent()
        final_output = ""
        async for state in agent.chat(
            task, thread_id=ctx.get("thread_id", f"team-{member.role.value}")
        ):
            messages = state.get("messages", [])
            for msg in messages:
                content = getattr(msg, "content", None)
                if content:
                    final_output = str(content)
        duration_ms = round((time.time() - start) * 1000, 2)
        traces.append(
            TeamTrace(
                role=member.role,
                member_name=member.name,
                model_name=member.model_name,
                input_task=task,
                output=final_output,
                duration_ms=duration_ms,
            )
        )
        return final_output

    async def _execute_plan(
        self,
        steps: list[TeamStep],
        ctx: dict[str, Any],
        traces: list[TeamTrace],
    ) -> list[str]:
        results: dict[str, str] = {}
        pending = {s.id: s for s in steps}

        while pending:
            ready = [
                s
                for s in pending.values()
                if all(dep in results for dep in s.depends_on)
            ]
            if not ready:
                ready = list(pending.values())

            async def run_one(step: TeamStep) -> tuple[str, str]:
                member = self.members.get(step.role)
                if member is None:
                    return step.id, ""
                dep_text = "\n".join(
                    f"{dep}: {results[dep]}"
                    for dep in step.depends_on
                    if dep in results
                )
                task = step.task
                if dep_text:
                    task = f"{task}\n\nContext from previous steps:\n{dep_text}"
                output = await self._run_member(member, task, ctx, traces)
                return step.id, output

            batch = await asyncio.gather(*(run_one(s) for s in ready))
            for step_id, output in batch:
                results[step_id] = output
                pending.pop(step_id)

        return [results[s.id] for s in steps]

    def _parse_plan(self, text: str) -> list[TeamStep]:
        if not text:
            return []
        if "```" in text:
            parts = text.split("```")
            if len(parts) >= 3:
                text = parts[1].strip("json").strip()
        try:
            data = json.loads(text)
        except Exception:
            return []
        if not isinstance(data, list):
            return []

        steps: list[TeamStep] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            try:
                steps.append(
                    TeamStep(
                        id=str(item.get("id", f"step{len(steps)+1}")),
                        role=TeamRole(item.get("role", "executor")),
                        task=str(item.get("task", "")),
                        depends_on=[
                            str(d) for d in item.get("depends_on", []) if d
                        ],
                    )
                )
            except Exception:
                continue
        return steps

    def _default_plan(self, task: str) -> list[TeamStep]:
        steps: list[TeamStep] = []
        order = [
            TeamRole.SCIENTIST,
            TeamRole.CODER,
            TeamRole.EXECUTOR,
            TeamRole.CRITIC,
        ]
        prev_id: str | None = None
        for role in order:
            if role not in self.members:
                continue
            step_id = f"{role.value}_step"
            depends = [prev_id] if prev_id else []
            steps.append(
                TeamStep(
                    id=step_id,
                    role=role,
                    task=f"{role.value} for: {task}",
                    depends_on=depends,
                )
            )
            prev_id = step_id
        return steps

    @staticmethod
    def _plan_to_text(steps: list[TeamStep]) -> str:
        lines = []
        for s in steps:
            deps = f" (after {', '.join(s.depends_on)})" if s.depends_on else ""
            lines.append(f"{s.id}: [{s.role.value}] {s.task}{deps}")
        return "\n".join(lines)

    @staticmethod
    def _trace_to_dict(t: TeamTrace) -> dict[str, Any]:
        return {
            "role": t.role.value,
            "member": t.member_name,
            "model": t.model_name,
            "input": t.input_task,
            "output": t.output,
            "duration_ms": t.duration_ms,
        }

    def list_members(self) -> list[dict[str, Any]]:
        """返回团队成员清单 (供 CLI / 前端展示)."""
        return [
            {
                "role": m.role.value,
                "name": m.name,
                "profile": m.profile_id,
                "model": m.model_name,
                "caps": {
                    "vision": m.caps.vision,
                    "tools": m.caps.tools,
                    "reasoning": m.caps.reasoning,
                    "streaming": m.caps.streaming,
                },
            }
            for m in self.members.values()
        ]


# ── 辅助函数 ──────────────────────────────────────────────


def _resolve_model_name(config: Any, alias: str) -> str:
    """从 config 的 model pool 里按 alias 找到真实 model 名."""
    for m in config.models:
        if m.alias == alias and m.enabled:
            return m.model or ""
    return ""


def _pick_best_profile(
    role: TeamRole,
    profiles: list[Any],
    config: Any,
) -> Any | None:
    """从候选 profile 中挑能力最匹配的那个."""
    if not profiles:
        return None
    required, bonus = ROLE_REQUIREMENTS.get(role, (set(), set()))

    best = None
    best_score = -1.0
    for p in profiles:
        model_name = _resolve_model_name(config, p.model_alias)
        caps = get_model_capabilities(model_name) if model_name else ModelCaps()
        caps_dict = {
            "vision": caps.vision,
            "tools": caps.tools,
            "reasoning": caps.reasoning,
            "streaming": caps.streaming,
        }
        # 必须满足的硬性要求
        if not all(caps_dict.get(r, False) for r in required):
            continue
        # 加分项
        score = sum(1.0 for b in bonus if caps_dict.get(b, False))
        # 能力越全越好 (作为 tiebreaker)
        score += sum(caps_dict.values()) * 0.1
        if score > best_score:
            best_score = score
            best = p
    return best


def _pick_fallback_profile(
    role: TeamRole,
    existing_members: list[TeamMember],
    assigned_roles: set[TeamRole],
) -> TeamMember | None:
    """实在没多余 profile 了, 从已有成员里借一个.

    critic 不能跟 planner 用同一个 (避免自我审查).
    其他角色可以随便借.
    """
    if role == TeamRole.CRITIC:
        planner = next(
            (m for m in existing_members if m.role == TeamRole.PLANNER), None
        )
        candidates = [m for m in existing_members if m is not planner]
    else:
        candidates = existing_members

    return candidates[0] if candidates else None
