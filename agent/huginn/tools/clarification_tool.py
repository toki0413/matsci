"""ClarificationTool —— agent 主动向用户提问的桥接工具.

LLM 在执行过程中遇到不确定的情况 (参数缺失 / 多路径选择 / 长任务确认),
调用这个工具向用户提问. 工具内部走 ClarificationManager.ask 阻塞等回答,
超时 (默认 5 分钟) 自动返回 default_answer, 让 agent 不至于卡死.

跟 clarify_questions_hook 的区别: hook 在用户提交消息时做规则匹配, 是
被动追问; 这个工具是 agent 在执行过程中主动调用的, 是 LLM 自驱动的提问.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from huginn.interaction.clarification import (
    ClarificationManager,
    get_clarification_manager,
)
from huginn.tools.base import HuginnTool
from huginn.types import ToolContext, ToolResult


class ClarificationInput(BaseModel):
    """工具输入. question 必填, 其它都可选."""

    question: str = Field(
        description=(
            "要问用户的具体问题. 必须是完整句子, 让用户能直接回答. "
            "例如: 'ENCUT 取 400 还是 520 eV?' 而不是 'ENCUT?'."
        )
    )
    options: list[str] = Field(
        default_factory=list,
        description=(
            "可选的候选答案列表. 用户可以选其中一个, 也可以自己输入. "
            "为空表示开放性问题, 用户自由作答."
        ),
    )
    context: str = Field(
        default="",
        description=(
            "提问背景说明, 帮助用户理解为什么要问. 例如: "
            "'需要确定基组截断能, 影响计算精度和成本'."
        ),
    )
    default_answer: str = Field(
        default="",
        description=(
            "用户超时未回答时用的默认值. 必须给一个合理的默认, "
            "避免 agent 永远等下去. 例如: '520'."
        ),
    )
    timeout_seconds: float = Field(
        default=300.0,
        ge=10.0,
        le=3600.0,
        description=(
            "等待用户回答的超时时间(秒). 默认 300 (5 分钟), "
            "长任务确认可以设大一点."
        ),
    )


class ClarificationOutput(BaseModel):
    """工具输出. answer 是用户回答或超时默认值."""

    answer: str
    question_id: str
    timed_out: bool = False


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
        "Ask the user a clarifying question when you are uncertain about "
        "task scope, parameters, or which approach to take. Blocks until "
        "the user answers or the timeout expires. Use sparingly — only "
        "when a wrong guess would waste significant compute or time."
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
        """提问并等回答.

        thread_id 从 ToolContext.session_id 拿, 这样问题跟当前会话绑定.
        没有 session_id 时退化为 "default", 跟 chat 默认 thread 一致.
        """
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
            # flag 层挂了不能带挂业务, 继续走原逻辑
            pass

        thread_id = getattr(context, "session_id", None) or "default"

        # 先判断要不要问 (避免在已有 3 个未答问题时还堆新的)
        if not self.manager.should_ask(
            "agent_initiated",
            context={"thread_id": thread_id},
        ):
            # 队列满了, 直接走默认值, 别再问
            return ToolResult(
                data=ClarificationOutput(
                    answer=args.default_answer or "",
                    question_id="",
                    timed_out=True,
                ).model_dump(),
                success=True,
                side_effects=["clarification skipped: too many pending"],
            )

        try:
            # 调 manager.ask 阻塞等回答
            # 这里没法精确知道是不是超时, manager 内部已经处理了,
            # 我们通过比对返回值跟 default_answer 来粗略判断
            answer = await self.manager.ask(
                thread_id=thread_id,
                question=args.question,
                options=args.options,
                context=args.context,
                default_answer=args.default_answer,
                timeout=args.timeout_seconds,
                metadata={
                    "engine_kind": "clarification_tool",
                    "session_id": thread_id,
                },
            )
            timed_out = (
                answer == args.default_answer
                and args.default_answer != ""
            )
            # 拿 question_id: 列 pending 取第一个, 应该就是我们刚问的
            # (ask 返回后问题已从 pending 列表移除, 这里取不到也无所谓)
            qid = ""
            return ToolResult(
                data=ClarificationOutput(
                    answer=answer,
                    question_id=qid,
                    timed_out=timed_out,
                ).model_dump(),
                success=True,
                side_effects=[
                    f"asked user: {args.question[:80]}",
                    f"got answer: {answer[:80]}",
                ],
            )
        except Exception as exc:
            # 异常时退回 default_answer, 不让 agent 卡死
            return ToolResult(
                data=ClarificationOutput(
                    answer=args.default_answer or "",
                    question_id="",
                    timed_out=True,
                ).model_dump(),
                success=False,
                error=f"clarification failed: {exc}",
            )
