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
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from huginn.models.registry import ModelCaps, get_model_capabilities

logger = logging.getLogger(__name__)


class TeamRole(enum.StrEnum):
    PLANNER = "planner"
    SCIENTIST = "scientist"
    CODER = "coder"
    EXECUTOR = "executor"
    CRITIC = "critic"
    VISION = "vision"          # 本地多模态: 处理图像/视觉理解
    SYNTHESIZER = "synthesizer" # 合成器: 跨模型信息融合


# NonAuthoritativeProvenance: 子 agent 是非权威来源, 不能继承主会话的
# auto_approve_all 高权限. 按 role 限定工具子集, 避免子 agent 调危险工具.
# None = 不限 (继承 profile 配置); list = 只给这些工具.
# ceiling: 硬编码工具名, 工具改名时需同步更新. 升级: 从 ModelCaps 推导.
_ROLE_TOOL_FILTER: dict[TeamRole, list[str] | None] = {
    TeamRole.VISION: ["vision_describe", "image_analysis_tool", "file_read_tool"],
    TeamRole.CRITIC: ["file_read_tool", "grep", "glob", "diff_tool"],
    TeamRole.PLANNER: None,   # planner 需要全工具视野来规划
    TeamRole.SYNTHESIZER: None,
}


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
    # 视觉角色必须能看图 (本地多模态小模型)
    TeamRole.VISION: ({"vision"}, set()),
    # 合成器无硬性要求, 但推理加分
    TeamRole.SYNTHESIZER: (set(), {"reasoning"}),
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
        """延迟创建 agent 实例.

        NonAuthoritativeProvenance: 子 agent 不继承主会话的 auto_approve
        高权限, 强制降级. 按 role 限定工具子集 (VISION 只给视觉+读文件,
        CRITIC 只给只读工具). approval_callback 从 config 继承, 保证
        ASK 模式有回调可走而非卡死.
        """
        if self._agent is None:
            import dataclasses

            from huginn.agent import HuginnAgent

            if self._config is None:
                raise RuntimeError(
                    f"TeamMember '{self.name}' 没有关联 config, 无法创建 agent"
                )
            overrides: dict[str, Any] = {}
            _filter = _ROLE_TOOL_FILTER.get(self.role)
            if _filter is not None:
                overrides["tool_filter"] = _filter
            self._agent = HuginnAgent.from_config(
                self._config, profile_id=self.profile_id, **overrides
            )
            # 强制降级: 子 agent 不继承主会话 auto_approve_all 高权限,
            # 但保留 path_rules / sandbox_mode 等其他权限设置.
            _perm = getattr(self._agent, "_permission_config", None)
            if _perm is not None:
                self._agent._permission_config = dataclasses.replace(
                    _perm, auto_approve_all=False
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

    每次 run/fusion 会往 EventBus 发 ``team.*`` 事件 (run/member start-done +
    tool_call), 前端通过 /events/stream 的 SSE 实时渲染子任务面板.
    """

    def __init__(self, members: list[TeamMember]) -> None:
        self.members: dict[TeamRole, TeamMember] = {}
        for m in members:
            self.assign(m)

    # ── 实时事件 (子任务面板数据源) ─────────────────────────

    async def _publish(self, event_type: str, data: dict[str, Any]) -> None:
        """发布 team.* 事件到全局 EventBus.

        后端失败不阻塞团队执行 (best-effort): 前端面板挂了不影响流水线.
        """
        try:
            import time as _time

            from huginn.events.event_bus import AgentEvent, EventBus

            await EventBus.shared().publish(
                AgentEvent(
                    type=event_type,
                    timestamp=_time.time(),
                    data=data,
                    source="model_team",
                )
            )
        except Exception:
            logger.debug("team event publish failed: %s", event_type, exc_info=True)

    def assign(self, member: TeamMember) -> ModelTeam:
        """把成员绑定到其声明的角色 (覆盖同角色的旧成员)."""
        self.members[member.role] = member
        return self

    @classmethod
    def from_config(cls, config: Any) -> ModelTeam:
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
                logger.debug("best-effort op failed", exc_info=True)
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
        import uuid

        run_id = context.get("team_run_id") if context else None
        if not run_id:
            run_id = uuid.uuid4().hex[:12]
        traces: list[TeamTrace] = []
        ctx = dict(context or {})
        ctx["original_task"] = task
        ctx["team_run_id"] = run_id
        t0 = time.time()

        await self._publish("team.run.start", {
            "run_id": run_id,
            "task": task,
            "members": [m.role.value for m in self.members.values()],
        })

        # 1. 规划
        plan_text = await self._delegate(TeamRole.PLANNER, task, ctx, traces, run_id)
        ctx["planner_output"] = plan_text
        steps = self._parse_plan(plan_text)
        if not steps:
            steps = self._default_plan(task)
        ctx["plan"] = self._plan_to_text(steps)

        # 2. 执行
        step_outputs = await self._execute_plan(steps, ctx, traces, run_id)
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
                TeamRole.CRITIC, critic_input, ctx, traces, run_id
            )
        ctx["review"] = critic_output

        await self._publish("team.run.done", {
            "run_id": run_id,
            "task": task,
            "duration_ms": round((time.time() - t0) * 1000, 2),
            "steps": len(traces),
        })

        return {
            "task": task,
            "run_id": run_id,
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
        run_id: str = "",
    ) -> str:
        member = self.members.get(role)
        if member is None:
            return ""
        return await self._run_member(member, task, ctx, traces, run_id)

    async def _run_member(
        self,
        member: TeamMember,
        task: str,
        ctx: dict[str, Any],
        traces: list[TeamTrace],
        run_id: str = "",
    ) -> str:
        start = time.time()
        agent = member.get_agent()
        base = {
            "run_id": run_id,
            "role": member.role.value,
            "member": member.name,
            "model": member.model_name,
        }
        await self._publish("team.member.start", {**base, "status": "running", "task": task[:500]})
        final_output = ""
        # VISION member 需要把真实图像透传给 agent.chat(), 否则多模态模型
        # 只能看到文本拼的"图片路径", 无法真正看图. ctx 里由调用方注入
        # image_path / image_bytes, 这里原样透传 (None 时走纯文本旧路径).
        chat_kwargs: dict[str, Any] = {
            "thread_id": ctx.get("thread_id", f"team-{member.role.value}")
        }
        if ctx.get("image_path") is not None:
            chat_kwargs["image_path"] = ctx["image_path"]
        # 记录该成员这轮用掉的 token (best-effort, 拿不到就不带).
        usage: dict[str, int] = {}
        try:
            async for state in agent.chat(task, **chat_kwargs):
                # 工具调用 → team.member.tool 事件 (前端节点展开看调用序列)
                for msg in state.get("messages", []):
                    tcs = getattr(msg, "tool_calls", None)
                    if tcs:
                        for tc in tcs:
                            name = "?"
                            if isinstance(tc, dict):
                                name = tc.get("name") or (tc.get("function") or {}).get("name", "?")
                            else:
                                name = getattr(tc, "name", "?")
                            args = ""
                            if isinstance(tc, dict):
                                args = tc.get("args") or (tc.get("function") or {}).get("arguments", "")
                            await self._publish(
                                "team.member.tool",
                                {**base, "tool": str(name), "args": str(args)[:300]},
                            )
                    content = getattr(msg, "content", None)
                    if content:
                        final_output = str(content)
                # 尝试从 state 拿 token usage (各 agent 字段名不完全一致)
                for key in ("usage", "_usage", "token_usage", "_tokens"):
                    u = state.get(key) if isinstance(state, dict) else None
                    if isinstance(u, dict) and any(u.values()):
                        usage.update({k: int(v) for k, v in u.items() if isinstance(v, (int, float))})
        except Exception as exc:
            await self._publish("team.member.done", {
                **base,
                "status": "failed",
                "error": str(exc)[:300],
                "duration_ms": round((time.time() - start) * 1000, 2),
                "tokens": usage,
            })
            raise
        duration_ms = round((time.time() - start) * 1000, 2)
        await self._publish("team.member.done", {
            **base,
            "status": "done",
            "duration_ms": duration_ms,
            "tokens": usage,
        })
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
        run_id: str = "",
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

            # 并行 batch 开始 → 前端可看到哪些 step 同时在跑.
            await self._publish("team.batch.start", {
                "run_id": run_id,
                "steps": [{"id": s.id, "role": s.role.value, "task": s.task[:200]} for s in ready],
            })

            async def run_one(step: TeamStep) -> tuple[str, str]:
                member = self.members.get(step.role)
                if member is None:
                    return step.id, ""
                try:
                    dep_text = "\n".join(
                        f"{dep}: {results[dep]}"
                        for dep in step.depends_on
                        if dep in results
                    )
                    task = step.task
                    if dep_text:
                        task = f"{task}\n\nContext from previous steps:\n{dep_text}"
                    output = await self._run_member(member, task, ctx, traces, run_id)
                    return step.id, output
                except Exception as exc:
                    # 失败隔离: 单个 member 异常不应团灭整个并行 batch.
                    # 与 parallel_executor._run_one / swarm.run_one 对齐.
                    logger.warning(
                        "team run_one step=%s failed (isolated): %s",
                        step.id, exc, exc_info=True,
                    )
                    return step.id, f"[ERROR] step {step.id} failed: {exc}"

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
                logger.debug("best-effort op failed", exc_info=True)
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

    # ── Fusion 模式 ────────────────────────────────────────

    async def fusion_query(
        self,
        query: str,
        context: dict[str, Any] | None = None,
        *,
        panel_roles: list[TeamRole] | None = None,
        synthesizer_role: TeamRole = TeamRole.CRITIC,
        max_panel: int = 5,
        rounds: int = 1,
    ) -> dict[str, Any]:
        """Fusion 模式: 并行分发 → 多轮讨论 → 裁判合成.

        灵感来自 OpenRouter Fusion (2026.06):
        1. 将用户 query 并行分发给 panel 成员 (默认所有 tools-capable 成员)
        2. 每个成员独立推理, 可以调工具
        3. rounds > 1 时, 后续轮次每个成员能看到其他人的回答, 进行补充/修正
        4. 所有轮次结束后, 由 synthesizer 成员做差异驱动的信息蒸馏:
           - 共识提取: 多个模型一致的结论
           - 分歧标注: 矛盾的观点 + 证据判断
           - 盲区补全: 单一模型独有的有效信息

        和 ModelTeam.run() 的区别:
        - run() 是串行流水线 (plan → execute → review)
        - fusion_query() 是并行 fan-out + (可选)多轮讨论 + 合成 fan-in
        - fusion 适合开放性研究问题, run 适合有明确步骤的执行任务

        参数:
            panel_roles: 参与并行回答的角色, 默认所有 tools-capable 成员
            synthesizer_role: 负责合成的角色, 默认 critic
            max_panel: 最多并行成员数 (成本控制)
            rounds: 讨论轮数, 1=单轮(纯并行), 2+=多轮讨论(协作)

        返回:
            dict 包含 final_answer, panel_responses, all_rounds, consensus, dissent
        """
        traces: list[TeamTrace] = []
        ctx = dict(context or {})
        ctx["original_task"] = query

        # 1. 选择 panel 成员
        if panel_roles is None:
            # 默认: 所有有 tools 能力的成员 (不含 synthesizer)
            panel_roles = [
                role for role, member in self.members.items()
                if role != synthesizer_role and member.caps.tools
            ]
        if not panel_roles:
            # 退化为所有非 synthesizer 成员
            panel_roles = [
                role for role in self.members if role != synthesizer_role
            ]

        # 限制 panel 大小
        panel_roles = panel_roles[:max_panel]
        if not panel_roles:
            return {
                "task": query,
                "final_answer": "No panel members available for fusion.",
                "panel_responses": [],
                "consensus": "",
                "dissent": "",
            }

        # 2. 并行分发 (多轮讨论)
        import uuid as _uuid
        fusion_run_id = (context or {}).get("team_run_id") or _uuid.uuid4().hex[:12]
        await self._publish("team.run.start", {
            "run_id": fusion_run_id,
            "task": query,
            "members": [r.value for r in panel_roles],
        })

        async def _parallel_delegate(
            role: TeamRole, task: str
        ) -> tuple[TeamRole, str, float]:
            start = time.time()
            output = await self._delegate(role, task, ctx, traces, fusion_run_id)
            duration = round((time.time() - start) * 1000, 2)
            return role, output, duration

        all_rounds: list[list[dict]] = []
        panel_responses: list[dict] = []

        for round_idx in range(rounds):
            if round_idx == 0:
                # 第一轮: 原始 query
                task_for_round = query
            else:
                # 后续轮次: 把上一轮所有回答作为同侪上下文
                peer_text = "\n\n".join(
                    f"[{r['role']} ({r['model']})]\n{r['answer']}"
                    for r in panel_responses
                )
                task_for_round = (
                    f"{query}\n\n"
                    f"=== 同侪回答 (第 {round_idx} 轮) ===\n"
                    f"{peer_text}\n\n"
                    f"=== 你的任务 ===\n"
                    f"请审视其他模型的回答, 补充遗漏的信息, "
                    f"修正错误, 或坚持你的观点并给出理由."
                )

            round_results = await asyncio.gather(
                *(_parallel_delegate(role, task_for_round) for role in panel_roles)
            )

            panel_responses = []
            for role, output, duration in round_results:
                member = self.members.get(role)
                panel_responses.append({
                    "role": role.value,
                    "model": member.model_name if member else "?",
                    "answer": output,
                    "duration_ms": duration,
                    "round": round_idx + 1,
                })
            all_rounds.append(list(panel_responses))

        # panel_responses 现在是最后一轮的结果

        # 3. 裁判合成
        synthesizer_member = self.members.get(synthesizer_role)
        if synthesizer_member is None:
            # 没有 synthesizer, 退化为拼接
            combined = "\n\n---\n\n".join(
                f"[{r['role']} ({r['model']})]\n{r['answer']}"
                for r in panel_responses
            )
            return {
                "task": query,
                "final_answer": combined,
                "panel_responses": panel_responses,
                "all_rounds": all_rounds,
                "rounds": rounds,
                "consensus": "No synthesizer available — raw concatenation.",
                "dissent": "",
                "trace": [self._trace_to_dict(t) for t in traces],
            }

        # 构建合成 prompt
        panel_text = "\n\n".join(
            f"=== [{r['role']} ({r['model']})] ===\n{r['answer']}"
            for r in panel_responses
        )

        synthesis_prompt = f"""以下是多个独立模型对同一问题的回答:

原始问题: {query}

{panel_text}

请作为综合分析师, 完成以下任务:
1. **共识提取**: 提取多个模型一致认同的结论 (标注哪些模型支持)
2. **分歧标注**: 标注存在矛盾或不同观点的地方, 分析哪方证据更充分
3. **盲区补全**: 补充单一模型遗漏但有价值的信息
4. **最终结论**: 输出一个结构完整、逻辑清晰的综合答案

请按以下格式输出:

## 共识
(多个模型一致的结论)

## 分歧
(矛盾的观点 + 判断)

## 综合答案
(最终结论)
"""

        synth_output = await self._delegate(
            synthesizer_role, synthesis_prompt, ctx, traces, fusion_run_id
        )

        # 解析合成结果
        consensus = ""
        dissent = ""
        final_answer = synth_output
        if "## 共识" in synth_output:
            parts = synth_output.split("## 共识", 1)
            if len(parts) > 1:
                rest = "## 共识" + parts[1]
                if "## 分歧" in rest:
                    consensus_part = rest.split("## 分歧", 1)
                    consensus = consensus_part[0].replace("## 共识", "").strip()
                    if len(consensus_part) > 1 and "## 综合答案" in consensus_part[1]:
                        dissent_part = consensus_part[1].split("## 综合答案", 1)
                        dissent = dissent_part[0].strip()
                        final_answer = dissent_part[1].strip()
                    else:
                        dissent = consensus_part[1].strip()
                else:
                    consensus = rest.replace("## 共识", "").strip()
        elif "## 综合答案" in synth_output:
            final_answer = synth_output.split("## 综合答案", 1)[-1].strip()

        return {
            "task": query,
            "final_answer": final_answer,
            "panel_responses": panel_responses,
            "all_rounds": all_rounds,
            "rounds": rounds,
            "consensus": consensus,
            "dissent": dissent,
            "synthesizer": {
                "role": synthesizer_role.value,
                "model": synthesizer_member.model_name,
            },
            "trace": [self._trace_to_dict(t) for t in traces],
        }


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
