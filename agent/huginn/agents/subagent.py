"""Subagent isolation dispatch -- keep main context clean by offloading
long-running or context-heavy tasks to isolated subagent sessions.

Inspired by Kimi Code's coder/explore/plan first-class subagent pattern.
Each subagent gets its own message list and tool budget, results are
summarized back to the main conversation.

核心思路: 主 agent 把 "探索代码库" / "写一段代码" / "分析数据" 这种会
产生大量中间输出的任务丢给子 agent. 子 agent 在独立的 thread_id 下跑,
产出的中间 tool_call / 文件内容不会污染主对话. 跑完后只把压缩后的
摘要塞回主上下文, 完整输出留档备查.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# 超过这个字符数才触发 LLM 摘要, 短输出直接原样返回省一次 API 调用
_SUMMARIZE_THRESHOLD = 2000

# 摘要时喂给 LLM 的原文截断长度, 避免超长输出把摘要请求也撑爆
_SUMMARIZE_INPUT_LIMIT = 8000


@dataclass
class SubagentSpec:
    """Declarative spec for a subagent type.

    allowed_tools 是白名单, 填了就只给子 agent 这些工具. 空列表 = 所有
    已注册工具都可用. 白名单里不存在的工具名会被静默跳过 (跟
    HuginnAgent.tool_filter 行为一致).

    summary_format 控制 _summarize 用哪种 prompt 压缩:
    - "free" (默认): 散文摘要, <500 词, 适合 explore/coder/analyst
    - "json": 结构化 JSON, 保留 findings/evidence/limitations 字段,
      适合 support spec (Oxelra Core+Support 模式)
    ponytail: 不引入 schema validator, 让 LLM 自律输出 JSON.
      升级路径: pydantic schema validate + retry on parse fail.
    """
    name: str
    description: str
    system_prompt: str
    allowed_tools: list[str]
    max_tool_calls: int = 10
    # ponytail: max_iterations 没有直接对应的 langgraph 限制器, 实际靠
    # max_tool_calls 预算间接限制. 多数情况下一轮迭代至少调一次工具,
    # 所以 tool_calls 预算耗尽 ≈ 迭代耗尽. 要精确限制迭代数得改 graph
    # 的 recursion_limit, 但那是 agent.chat() 内部硬编码的, 不值得为此
    # 动 agent 的代码. 升级路径: 给 chat() 加 recursion_limit 参数.
    max_iterations: int = 5
    summarize_result: bool = True
    summary_format: str = "free"


@dataclass
class SubagentResult:
    """Result returned by a subagent dispatch.

    summary 是喂回主上下文的压缩结果, full_output 是完整输出留档.
    """
    summary: str
    full_output: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tokens_used: int = 0
    success: bool = True
    error: str | None = None
    spec_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "full_output": self.full_output,
            "tool_calls": self.tool_calls,
            "tokens_used": self.tokens_used,
            "success": self.success,
            "error": self.error,
            "spec_name": self.spec_name,
        }


class SubagentDispatch:
    """Manages subagent execution with isolated context.

    用法::

        dispatch = SubagentDispatch()
        result = await dispatch.dispatch(
            "explore",
            "找出项目里所有 DFT 计算的入口文件",
            context={"agent_factory": factory, "session_id": "xxx"},
        )
        print(result.summary)
    """

    # Built-in subagent types (Kimi Code inspired)
    BUILTIN_SPECS: dict[str, SubagentSpec] = {
        "explore": SubagentSpec(
            name="explore",
            description="探索代码库/文件系统/文档，返回发现摘要",
            system_prompt=(
                "You are an exploration agent. Read files, search code, "
                "and summarize findings. Do not modify anything."
            ),
            # glob / grep 不是独立工具, 填了也只是静默跳过, 留着是给将来加的
            allowed_tools=[
                "file_read_tool", "glob", "grep",
                "web_search_tool", "literature_tool",
            ],
            max_tool_calls=8,
            max_iterations=3,
        ),
        "coder": SubagentSpec(
            name="coder",
            description="执行代码编写和修改任务",
            system_prompt="You are a coding agent. Write and modify code files as requested.",
            # ponytail: 去掉 file_write_tool/file_edit_tool — Windows 路径 bug (§2.1).
            # 子 agent 用 code_tool 的 open() 写文件, 和主 agent 一致.
            # ponytail: 去掉 bench_infra (plot/training_matrix) — 让 coder 自己用 code_tool 实现
            allowed_tools=[
                "code_tool", "bash_tool",
                "file_read_tool", "glob", "grep",
            ],
            # benchmark 需要更多预算: 训练循环 + 画图 + 调试
            max_tool_calls=50,
            max_iterations=10,
        ),
        "analyst": SubagentSpec(
            name="analyst",
            description="分析计算结果/数据，返回结构化洞察",
            system_prompt=(
                "You are a data analysis agent. Analyze results, compute "
                "statistics, and return structured insights. "
                "Implement metrics (C2ST/MCMC) in code_tool, don't expect pre-built tools."
            ),
            allowed_tools=[
                "code_tool", "bash_tool", "file_read_tool",
                "numerical_tool", "visualize_tool",
            ],
            max_tool_calls=20,
            max_iterations=5,
        ),
        # v7 P2: Oxelra Core+Support 模式 — Support 子代理在隔离上下文里做重活,
        # 只把结构化 finding 喂回 Core, raw trace 不污染 Core context.
        # ponytail: 不引入新隔离机制, 复用现有 thread_id + 工具白名单 + LLM 摘要.
        # 升级路径: pydantic schema 校验 JSON summary, 失败重试.
        "support": SubagentSpec(
            name="support",
            description="Oxelra-style Support 子代理: 隔离上下文做重活, 返回结构化 finding",
            system_prompt=(
                "You are a Support research agent. Do the heavy lifting in isolation "
                "(run experiments, analyze data, read papers). "
                "Return ONLY structured findings: key results, evidence, limitations, artifacts. "
                "Do not include reasoning process or intermediate steps in your final output."
            ),
            allowed_tools=[
                "code_tool", "bash_tool",
                "file_read_tool", "glob", "grep",
                "web_search_tool", "numerical_tool",
            ],
            max_tool_calls=30,
            max_iterations=8,
            summary_format="json",
        ),
    }

    def __init__(self) -> None:
        # copy 一份内置 spec, register_spec 不会改类属性
        self._specs: dict[str, SubagentSpec] = dict(self.BUILTIN_SPECS)

    # ------------------------------------------------------------------ API

    async def dispatch(
        self,
        spec_name: str,
        task: str,
        context: dict | None = None,
        on_state: Any = None,
    ) -> SubagentResult:
        """Dispatch a subagent to handle a task in isolated context.

        context 里需要带 agent_factory (AgentFactory 实例), 没有就报错.
        on_state: optional async callback(state_dict) called for each
        intermediate agent state — lets callers stream subagent progress.
        """
        spec = self._specs.get(spec_name)
        if spec is None:
            return SubagentResult(
                summary="", full_output="",
                success=False,
                error=f"Unknown subagent spec: {spec_name}. "
                      f"Available: {sorted(self._specs.keys())}",
                spec_name=spec_name,
            )

        ctx = context or {}
        factory = ctx.get("agent_factory")
        if factory is None:
            return SubagentResult(
                summary="", full_output="",
                success=False,
                error="agent_factory not available in context. "
                      "Configure models and agent profiles first.",
                spec_name=spec_name,
            )

        # v7: 从 context 拿父 agent 的 approval_callback, 透传给子 agent.
        # 之前不传, 子 agent 调 vasp_tool 等 ASK 工具会被静默拒绝.
        approval_callback = ctx.get("approval_callback")

        # 独立 thread_id, 跟主对话完全隔离
        thread_id = f"subagent_{spec_name}_{uuid.uuid4().hex[:8]}"

        try:
            profile_id = self._pick_profile(factory)
            agent = factory.create(
                profile_id=profile_id,
                thread_id=thread_id,
                system_prompt_override=spec.system_prompt,
                approval_callback=approval_callback,
            )
        except Exception as exc:
            logger.debug("subagent creation failed", exc_info=True)
            return SubagentResult(
                summary="", full_output="",
                success=False,
                error=f"Failed to create subagent: {exc}",
                spec_name=spec_name,
            )

        # 应用工具白名单: 清掉 factory 注册的默认工具, 按白名单重新注册
        if spec.allowed_tools:
            agent.tool_filter = set(spec.allowed_tools)
            agent.langchain_tools.clear()
            agent.register_tools_from_registry()

        # 设工具调用预算, chat() 里会据此建 ToolCallBudget
        agent._max_tool_calls = spec.max_tool_calls

        try:
            final_state = None
            async for state in agent.chat(task, thread_id):
                if isinstance(state, dict):
                    final_state = state
                    if on_state is not None:
                        try:
                            await on_state(state)
                        except Exception:
                            logger.debug("on_state callback failed", exc_info=True)

            output = self._extract_output(final_state)
            tool_calls = self._extract_tool_calls(final_state)
            tokens = self._estimate_tokens(final_state)

            if spec.summarize_result and len(output) > _SUMMARIZE_THRESHOLD:
                summary = await self._summarize(
                    factory, output, task, summary_format=spec.summary_format,
                )
            else:
                summary = output

            return SubagentResult(
                summary=summary,
                full_output=output,
                tool_calls=tool_calls,
                tokens_used=tokens,
                success=True,
                spec_name=spec_name,
            )
        except Exception as exc:
            logger.debug("subagent execution failed", exc_info=True)
            return SubagentResult(
                summary="", full_output="",
                success=False,
                error=f"Subagent execution failed: {exc}",
                spec_name=spec_name,
            )

    def register_spec(self, spec: SubagentSpec) -> None:
        """Register a custom subagent type. 覆盖同名 spec."""
        self._specs[spec.name] = spec

    def list_specs(self) -> list[dict[str, Any]]:
        """返回所有可用 spec 的摘要, 给 LLM 展示用."""
        return [
            {
                "name": s.name,
                "description": s.description,
                "allowed_tools": s.allowed_tools,
                "max_tool_calls": s.max_tool_calls,
                "max_iterations": s.max_iterations,
            }
            for s in self._specs.values()
        ]

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _pick_profile(factory: Any) -> str:
        """挑一个可用的 agent profile, 优先 lead / default."""
        for preferred in ("lead", "default"):
            if factory.get_profile(preferred):
                return preferred
        profiles = factory.list_profiles()
        if profiles:
            return profiles[0].id
        # 没有 profile 让 factory.create 自己报错, 信息更清晰
        return "lead"

    @staticmethod
    def _extract_output(state: Any) -> str:
        """从 agent 最终 state 里取最后一条消息的文本."""
        if not isinstance(state, dict):
            return str(state) if state else ""
        messages = state.get("messages", [])
        if not messages:
            return ""
        last = messages[-1]
        if hasattr(last, "content"):
            return str(last.content)
        return str(last)

    @staticmethod
    def _extract_tool_calls(state: Any) -> list[dict[str, Any]]:
        """从 state messages 里提取所有 tool_call 记录."""
        if not isinstance(state, dict):
            return []
        calls: list[dict[str, Any]] = []
        for msg in state.get("messages", []):
            tool_calls = getattr(msg, "tool_calls", None) or []
            for tc in tool_calls:
                if isinstance(tc, dict):
                    calls.append({
                        "name": tc.get("name", "unknown"),
                        "args": tc.get("args", {}),
                    })
                else:
                    calls.append({"name": str(tc), "args": {}})
        return calls

    @staticmethod
    def _estimate_tokens(state: Any) -> int:
        """粗估 subagent 消耗的 token 数.

        ponytail: 用通用 tokenizer 而非模型精确 tokenizer, 误差 10-20%.
        精确值要按 model_name 调 count_tokens 的 model_name 参数, 但
        subagent 可能用了跟主 agent 不同的模型, 拿不准就用粗估值.
        """
        if not isinstance(state, dict):
            return 0
        try:
            from huginn.utils.tokens import count_tokens
        except ImportError:
            return 0
        total = 0
        for msg in state.get("messages", []):
            content = getattr(msg, "content", str(msg))
            total += count_tokens(str(content))
        return total

    @staticmethod
    async def _summarize(
        factory: Any, output: str, task: str, *, summary_format: str = "free",
    ) -> str:
        """用 LLM 压缩子 agent 输出, 拿不到模型就截断兜底.

        summary_format:
        - "free": 散文摘要 <500 词 (默认, explore/coder/analyst 用)
        - "json": 结构化 JSON {findings, evidence, limitations, artifacts}
                  (support spec 用, Oxelra Core+Support 模式)
        ponytail: json 模式不校验 schema, LLM 自律输出. 升级路径 pydantic + retry.
        """
        try:
            alias = factory.model_registry.default_alias()
            if not alias:
                return output[:_SUMMARIZE_THRESHOLD] + "..."
            model = factory.model_registry.resolve(alias)
        except Exception:
            logger.debug("resolve model for summarize failed", exc_info=True)
            return output[:_SUMMARIZE_THRESHOLD] + "..."

        if summary_format == "json":
            system_content = (
                "Extract structured findings from the subagent output as JSON. "
                "Schema: {\"findings\": str, \"evidence\": [str], "
                "\"limitations\": [str], \"artifacts\": [str]}. "
                "findings = 主要结论一句话; evidence = 支撑证据/数据来源列表; "
                "limitations = 适用边界/未验证项; artifacts = 产出文件路径列表. "
                "Output ONLY the JSON object, no prose, no markdown fences."
            )
        else:
            system_content = (
                "Summarize the following subagent output concisely. "
                "Focus on key findings, results, and any errors. "
                "Keep it under 500 words."
            )

        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            messages = [
                SystemMessage(content=system_content),
                HumanMessage(
                    content=(
                        f"Task: {task}\n\n"
                        f"Output:\n{output[:_SUMMARIZE_INPUT_LIMIT]}"
                    )
                ),
            ]
            result = await asyncio.to_thread(model.invoke, messages)
            return result.content if hasattr(result, "content") else str(result)
        except Exception:
            logger.debug("summarize LLM call failed", exc_info=True)
            return output[:_SUMMARIZE_THRESHOLD] + "..."


# ── self-check ────────────────────────────────────────────────────────────
# 最小验证: spec 注册 / 查询 / 未知 spec 报错. 不依赖 LLM 和 agent factory,
# 只验数据结构和控制流. 放在 __main__ 里, `python -m huginn.agents.subagent`
# 就能跑.

if __name__ == "__main__":
    import sys

    d = SubagentDispatch()

    # 1. 内置 spec 存在 (含 v7 P2 新增 support)
    specs = d.list_specs()
    names = [s["name"] for s in specs]
    assert "explore" in names and "coder" in names and "analyst" in names, names
    assert "support" in names, f"support spec missing: {names}"
    print(f"[ok] builtin specs: {names}")

    # 1b. v7 P2: support spec 配置正确
    support_spec = d._specs["support"]
    assert support_spec.summary_format == "json", support_spec.summary_format
    assert support_spec.max_tool_calls == 30
    assert "code_tool" in support_spec.allowed_tools
    # 其他 spec 默认 free
    assert d._specs["explore"].summary_format == "free"
    assert d._specs["coder"].summary_format == "free"
    print("[ok] support spec summary_format=json, others free")

    # 2. 自定义 spec 注册
    custom = SubagentSpec(
        name="test_custom",
        description="test",
        system_prompt="test",
        allowed_tools=["file_read_tool"],
        max_tool_calls=3,
    )
    d.register_spec(custom)
    assert "test_custom" in [s["name"] for s in d.list_specs()]
    print("[ok] custom spec registered")

    # 3. register_spec 不污染类属性
    assert "test_custom" not in SubagentDispatch.BUILTIN_SPECS
    print("[ok] class attr not mutated")

    # 4. 未知 spec 返回错误
    result = asyncio.run(d.dispatch("nonexistent", "test"))
    assert not result.success
    assert "Unknown subagent spec" in result.error
    print(f"[ok] unknown spec rejected: {result.error}")

    # 5. 缺 agent_factory 报错
    result = asyncio.run(d.dispatch("explore", "test"))
    assert not result.success
    assert "agent_factory" in result.error
    print(f"[ok] missing factory rejected: {result.error}")

    # 6. _extract_output / _extract_tool_calls 处理空 state
    assert SubagentDispatch._extract_output(None) == ""
    assert SubagentDispatch._extract_tool_calls(None) == []
    assert SubagentDispatch._estimate_tokens(None) == 0
    print("[ok] empty state handled")

    print("\nAll self-checks passed.")
    sys.exit(0)
