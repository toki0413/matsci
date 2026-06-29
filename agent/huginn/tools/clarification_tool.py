"""ClarificationTool —— agent 主动向用户提问的桥接工具.

LLM 在执行过程中遇到不确定的情况 (参数缺失 / 多路径选择 / 长任务确认),
调用这个工具向用户提问. 工具内部走 ClarificationManager.ask 阻塞等回答,
超时 (默认 5 分钟) 自动返回 default_answer, 让 agent 不至于卡死.

跟 clarify_questions_hook 的区别: hook 在用户提交消息时做规则匹配, 是
被动追问; 这个工具是 agent 在执行过程中主动调用的, 是 LLM 自驱动的提问.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from huginn.interaction.clarification import (
    ClarificationManager,
    get_clarification_manager,
)
from huginn.tools.base import HuginnTool
from huginn.types import ToolContext, ToolResult


class ClarificationInput(BaseModel):
    """工具输入. action=ask 时 question 必填, 其它 action 自动生成."""

    action: Literal[
        "ask",
        "confirm_destructive",
        "confirm_plan",
        "confirm_cost",
        "socratic_probes",
        "decision_tree",
    ] = Field(
        default="ask",
        description=(
            "提问类型. ask=普通提问; confirm_destructive=破坏性操作确认"
            "(安全默认=取消); confirm_plan=多步计划审批; "
            "confirm_cost=高成本操作确认(安全默认=取消); "
            "socratic_probes=按预设探针列表逐个追问, 收集所有答案; "
            "decision_tree=按决策树遍历, 每节点问一次, 跟着答案走到叶子."
        ),
    )
    question: str = Field(
        default="",
        description=(
            "要问用户的具体问题. action=ask 时必填, 必须是完整句子. "
            "其它 action 时可留空, 工具会自动生成确认语."
        ),
    )
    options: list[str] = Field(
        default_factory=list,
        description=(
            "可选的候选答案列表. 为空表示开放性问题. "
            "confirm_* action 会自动覆盖为 [确认, 取消]."
        ),
    )
    context: str = Field(
        default="",
        description="提问背景说明, 帮助用户理解为什么要问.",
    )
    default_answer: str = Field(
        default="",
        description=(
            "用户超时未回答时用的默认值. confirm_destructive/confirm_cost "
            "自动设为 '取消' (安全默认)."
        ),
    )
    timeout_seconds: float = Field(
        default=300.0,
        ge=10.0,
        le=3600.0,
        description="等待用户回答的超时时间(秒). 默认 300.",
    )
    plan_steps: list[str] = Field(
        default_factory=list,
        description="confirm_plan 专用: 要展示给用户审批的步骤列表.",
    )
    probes: list[str] = Field(
        default_factory=list,
        description=(
            "socratic_probes 专用: 预设的探针问题列表 (按顺序逐个问). "
            "最多 5 个, 避免疲劳轰炸. 每个必须是完整句子."
        ),
    )
    tree_nodes: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description=(
            "decision_tree 专用: 节点表. key=node_id, value={"
            "'question': str, 'options': {option_text: next_node_id}, "
            "'leaf': bool, 'result': str (leaf 时必填)}."
        ),
    )
    start_node: str = Field(
        default="",
        description="decision_tree 专用: 起始节点 id, 必须在 tree_nodes 里.",
    )

    @model_validator(mode="after")
    def _validate_action_fields(self):
        # socratic_probes / decision_tree 的必填字段校验,
        # 避免运行到一半才发现参数缺了
        if self.action == "socratic_probes":
            if not self.probes:
                raise ValueError(
                    "action=socratic_probes 需要 probes 字段 (非空列表)"
                )
            if len(self.probes) > 5:
                raise ValueError(
                    "socratic_probes 最多 5 个探针, 避免疲劳轰炸"
                )
        elif self.action == "decision_tree":
            if not self.tree_nodes:
                raise ValueError(
                    "action=decision_tree 需要 tree_nodes 字段 (非空)"
                )
            if not self.start_node:
                raise ValueError(
                    "action=decision_tree 需要 start_node 字段"
                )
            if self.start_node not in self.tree_nodes:
                raise ValueError(
                    f"start_node '{self.start_node}' 不在 tree_nodes 里"
                )
        return self


class ClarificationOutput(BaseModel):
    """工具输出. answer 是用户回答或超时默认值."""

    answer: str
    question_id: str
    timed_out: bool = False
    # socratic_probes: {probe_text: user_answer}
    answers: dict[str, str] = Field(
        default_factory=dict,
        description="socratic_probes 的 {probe: answer} 映射",
    )
    # decision_tree: 经过的节点 id 序列 + 叶子结果
    tree_path: list[str] = Field(
        default_factory=list,
        description="decision_tree 经过的节点 id 序列",
    )
    final_result: str = Field(
        default="",
        description="decision_tree 叶子节点的 result",
    )


class ClarificationTool(HuginnTool[ClarificationInput, ClarificationOutput]):
    """让 agent 在不确定时主动向用户提问.

    使用场景 (在 prompts.py 里也有引导):
    - 任务描述模糊 ("算一下" 没说算什么)
    - 参数不明确 (ENCUT 没说取多少)
    - 多个可能路径 (DFT 还是 ML 势)
    - 长任务前确认 (VASP 预计 2h)

    工具会阻塞等待用户回答, 默认超时 5 分钟. 超时返回 default_answer,
    agent 拿到后按默认值继续执行.
    """

    name = "clarification_tool"
    category = "meta"
    description = (
        "Ask the user a clarifying question or request confirmation. "
        "Actions: 'ask' (open question, blocks until answer/timeout), "
        "'confirm_destructive' (irreversible op, safe default=cancel), "
        "'confirm_cost' (high-cost compute, safe default=cancel), "
        "'confirm_plan' (present multi-step plan for approval), "
        "'socratic_probes' (ask 1-5 layered probes in sequence, returns "
        "{probe: answer} map — use when one ask would need 3+ rounds), "
        "'decision_tree' (walk a pre-mapped decision graph node by node, "
        "returns leaf result + path — use for nested branching choices). "
        "Use sparingly — only when a wrong guess wastes significant "
        "compute or time, or before irreversible operations."
    )
    destructive = False
    read_only = True  # 只问问题, 不改任何状态
    input_schema = ClarificationInput
    output_schema = ClarificationOutput

    def __init__(self, manager: ClarificationManager | None = None) -> None:
        # 允许调用方注入自定义 manager (测试用), 默认走进程级单例
        self._manager = manager

    @property
    def manager(self) -> ClarificationManager:
        """懒加载 manager 单例, 避免在 __init__ 时就拉起全局状态."""
        if self._manager is None:
            self._manager = get_clarification_manager()
        return self._manager

    async def call(
        self, args: ClarificationInput, context: ToolContext
    ) -> ToolResult:
        """提问并等回答. 按 action 自动生成提问模板."""
        # flag 关掉时不问, 直接走默认行为
        try:
            from huginn.feature_flags import FeatureFlags
            if not FeatureFlags.shared().is_enabled("clarification"):
                return ToolResult(
                    data=ClarificationOutput(
                        answer=args.default_answer or "",
                        question_id="",
                        timed_out=True,
                    ).model_dump(),
                    success=True,
                    side_effects=["clarification disabled by feature flag"],
                )
        except Exception:
            pass

        thread_id = getattr(context, "session_id", None) or "default"

        # socratic_probes / decision_tree 走专用路径, 不用 _prepare 模板
        if args.action == "socratic_probes":
            return await self._run_socratic_probes(args, thread_id)
        if args.action == "decision_tree":
            return await self._run_decision_tree(args, thread_id)

        # 按 action 准备提问参数
        question, options, default_answer = self._prepare(args)
        if question is None:
            return ToolResult(
                data=ClarificationOutput(
                    answer=default_answer, question_id="", timed_out=True,
                ).model_dump(),
                success=False,
                error="action=ask 需要 question 字段",
            )

        # 队列满了直接走默认值
        if not self.manager.should_ask(
            "agent_initiated",
            context={"thread_id": thread_id},
        ):
            return ToolResult(
                data=ClarificationOutput(
                    answer=default_answer, question_id="", timed_out=True,
                ).model_dump(),
                success=True,
                side_effects=["clarification skipped: too many pending"],
            )

        try:
            answer = await self.manager.ask(
                thread_id=thread_id,
                question=question,
                options=options,
                context=args.context,
                default_answer=default_answer,
                timeout=args.timeout_seconds,
                metadata={
                    "engine_kind": "clarification_tool",
                    "session_id": thread_id,
                    "action": args.action,
                },
            )
            timed_out = answer == default_answer and default_answer != ""
            return ToolResult(
                data=ClarificationOutput(
                    answer=answer, question_id="", timed_out=timed_out,
                ).model_dump(),
                success=True,
                side_effects=[
                    f"asked user ({args.action}): {question[:80]}",
                    f"got answer: {answer[:80]}",
                ],
            )
        except Exception as exc:
            return ToolResult(
                data=ClarificationOutput(
                    answer=default_answer, question_id="", timed_out=True,
                ).model_dump(),
                success=False,
                error=f"clarification failed: {exc}",
            )

    def _prepare(
        self, args: ClarificationInput
    ) -> tuple[str | None, list[str], str]:
        """按 action 生成提问文案/选项/默认值. 返回 (question, options, default)."""
        if args.action == "ask":
            if not args.question:
                return None, [], args.default_answer or ""
            return args.question, args.options, args.default_answer

        if args.action == "confirm_destructive":
            q = args.question or "即将执行破坏性操作"
            question = f"⚠️ 破坏性操作确认: {q}. 此操作不可逆, 确认执行？"
            return question, ["确认执行", "取消"], "取消"

        if args.action == "confirm_cost":
            q = args.question or "即将执行高成本计算"
            question = f"⏱ 高成本操作确认: {q}. 确认执行？"
            return question, ["确认执行", "取消"], "取消"

        if args.action == "confirm_plan":
            steps = args.plan_steps or []
            if steps:
                numbered = "\n".join(
                    f"  {i}. {s}" for i, s in enumerate(steps, 1)
                )
                question = f"📋 执行计划审批:\n{numbered}\n\n确认执行此计划？"
            else:
                question = "📋 执行计划审批: 确认执行？"
            return question, ["确认执行此计划", "修改计划", "取消"], "取消"

        return args.question, args.options, args.default_answer

    async def _run_socratic_probes(
        self, args: ClarificationInput, thread_id: str
    ) -> ToolResult:
        """按 probes 列表逐个问, 收集所有答案. 超时用 default_answer 填."""
        answers: dict[str, str] = {}
        timed_out_any = False
        for probe in args.probes:
            # 队列满了就不再问, 直接用默认值填
            if not self.manager.should_ask(
                "socratic_probe", context={"thread_id": thread_id}
            ):
                answers[probe] = args.default_answer
                timed_out_any = True
                continue
            try:
                ans = await self.manager.ask(
                    thread_id=thread_id,
                    question=probe,
                    options=args.options,
                    context=args.context,
                    default_answer=args.default_answer,
                    timeout=args.timeout_seconds,
                    metadata={
                        "engine_kind": "clarification_tool",
                        "action": "socratic_probes",
                        "session_id": thread_id,
                    },
                )
                answers[probe] = ans
                # 超时退回默认值时标记一下
                if ans == args.default_answer and args.default_answer:
                    timed_out_any = True
            except Exception:
                answers[probe] = args.default_answer
                timed_out_any = True
        return ToolResult(
            data=ClarificationOutput(
                answer=answers.get(args.probes[-1], ""),
                question_id="",
                timed_out=timed_out_any,
                answers=answers,
            ).model_dump(),
            success=True,
            side_effects=[
                f"socratic_probes: asked {len(answers)} probes",
            ],
        )

    async def _run_decision_tree(
        self, args: ClarificationInput, thread_id: str
    ) -> ToolResult:
        """从 start_node 遍历, 每节点问一次, 跟答案走到叶子.

        防环: visited set + max_depth=20 兜底. 断链 (选项指向不存在的节点)
        或用户给了一个不在 options 里的答案时, 走 default_answer 对应的边.
        """
        path: list[str] = []
        current = args.start_node
        visited: set[str] = set()
        max_depth = 20

        for _ in range(max_depth):
            if current in visited:
                break  # 检测到环, 退出
            visited.add(current)
            path.append(current)

            node = args.tree_nodes.get(current)
            if node is None:
                break  # 断链: 指向不存在的节点

            if node.get("leaf", False):
                final = node.get("result", "")
                return ToolResult(
                    data=ClarificationOutput(
                        answer=final,
                        question_id="",
                        timed_out=False,
                        tree_path=path,
                        final_result=final,
                    ).model_dump(),
                    success=True,
                    side_effects=[
                        f"decision_tree: reached leaf '{current}'",
                    ],
                )

            question = node.get("question", "")
            options = list(node.get("options", {}).keys())
            if not question or not options:
                break  # 节点结构不完整

            if not self.manager.should_ask(
                "decision_tree", context={"thread_id": thread_id}
            ):
                break

            try:
                ans = await self.manager.ask(
                    thread_id=thread_id,
                    question=question,
                    options=options,
                    context=args.context,
                    default_answer=args.default_answer or options[0],
                    timeout=args.timeout_seconds,
                    metadata={
                        "engine_kind": "clarification_tool",
                        "action": "decision_tree",
                        "session_id": thread_id,
                        "node": current,
                    },
                )
            except Exception:
                break

            # 用户答案不在 options 里 -> 走 default_answer 对应的边, 再不行取第一个
            next_node = node["options"].get(ans)
            if next_node is None:
                next_node = node["options"].get(
                    args.default_answer, options[0]
                )
            if next_node is None:
                break
            current = next_node

        # 走到上限或断链: 用当前节点的 result 或 default 兜底
        node = args.tree_nodes.get(current, {})
        final = node.get("result", args.default_answer)
        return ToolResult(
            data=ClarificationOutput(
                answer=final,
                question_id="",
                timed_out=True,
                tree_path=path,
                final_result=final,
            ).model_dump(),
            success=True,
            side_effects=[
                f"decision_tree: stopped at '{current}' (max depth or broken link)",
            ],
        )
