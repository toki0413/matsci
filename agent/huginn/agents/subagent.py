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
import json
import logging
import re
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 超过这个字符数才触发 LLM 摘要, 短输出直接原样返回省一次 API 调用
_SUMMARIZE_THRESHOLD = 2000

# 摘要时喂给 LLM 的原文截断长度, 避免超长输出把摘要请求也撑爆
_SUMMARIZE_INPUT_LIMIT = 8000

# G1: 当前 dispatch 递归深度. 主 agent 调 dispatch 时 _depth=0, 子 agent
# 再调 subagent_tool 时从这里读到 _depth=1, 透传给 dispatch 守卫.
# ponytail: 用 contextvar 而非给 ToolContext 加字段, LLM 看不见也改不了.
_current_depth: ContextVar[int] = ContextVar("_current_depth", default=0)


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
    # G1: 递归深度 cap. 1=单层 (不能再委派), 2=可委派单层 sub-sub, 3=硬 cap.
    # ponytail: 默认 1, 防 subagent 递归失控. 升级: M4 budget_decomp 推荐配置.
    max_depth: int = 1


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
        # P1: 盲重建 subagent (chaoxu 启发). 拿 hypothesis statement 独立推导,
        # 不看原 proof/evidence. 两阶段 verification 的第二阶段 (第一阶段是
        # adversarial_critique). mismatch → refute, match → support.
        "blind_reconstructor": SubagentSpec(
            name="blind_reconstructor",
            description="从 hypothesis statement 独立推导, 不参考原 proof (blind reconstruction)",
            system_prompt=(
                "You are a blind reconstruction agent. You receive a hypothesis "
                "statement and must independently derive whether it holds, from first "
                "principles. You do NOT see the original proof, evidence, or reasoning.\n\n"
                "Output a JSON with fields:\n"
                "- \"holds\": true/false — does the statement hold under your derivation?\n"
                "- \"derivation\": your independent reasoning (concise, <300 words)\n"
                "- \"key_assumption\": the one assumption your derivation relies on\n"
                "- \"confidence\": 0.0-1.0\n"
                "Be rigorous. If you cannot derive it, say holds=false with reason."
            ),
            allowed_tools=[
                "code_tool", "bash_tool",
                "file_read_tool", "numerical_tool",
            ],
            max_tool_calls=15,
            max_iterations=5,
            summary_format="json",
        ),
        # Task 3: 失败推理反推 (failure trace inversion). 拿 (input, failed_result)
        # 反推"为什么这个 input 会导致这个 failed result". 跟 blind_reconstructor
        # 同款 dispatch 路径, 仅 system_prompt 反转: 不是判 holds, 是反推 failure.
        "failure_inverter": SubagentSpec(
            name="failure_inverter",
            description="从 (input, failed_result) 反推为什么这个 input 会导致这个 failed result (failure trace inversion)",
            system_prompt=(
                "You are a failure inversion agent. You receive (input parameters, "
                "failed result, failure mode) and must INFER the failure reasoning: "
                "trace back from the failed result to which step / assumption broke, "
                "and what counterfactual change could make it work.\n\n"
                "Output a JSON with fields:\n"
                "- \"failure_reasoning\": the full reverse-engineered failure path (<300 words)\n"
                "- \"failure_point\": at which step / assumption does it break\n"
                "- \"counterfactual\": what condition could change to make it work\n"
                "- \"confidence\": 0.0-1.0\n"
                "Be rigorous. If you cannot invert, return empty failure_reasoning with confidence=0.0."
            ),
            allowed_tools=[
                "code_tool", "bash_tool",
                "file_read_tool", "numerical_tool",
            ],
            max_tool_calls=15,
            max_iterations=5,
            summary_format="json",
        ),
    }

    def __init__(self) -> None:
        # copy 一份内置 spec, register_spec 不会改类属性.
        # H4 试点: toggle harness_phase_evolve 开启时从 PhaseRegistry 取
        # (baseline + 用户 override 合成), 没开保留 class attr 路径 (零开销).
        # ponytail: 不在 import 时就拉起 PhaseRegistry, 避免循环 import +
        # 单例污染. toggle off 时 get_subagent_specs_for_dispatch 返回 None.
        specs_override = None
        try:
            from huginn.harness.phase_spec import (
                get_subagent_specs_for_dispatch,
            )
            specs_override = get_subagent_specs_for_dispatch()
        except Exception:
            pass
        if specs_override is not None:
            self._specs: dict[str, SubagentSpec] = dict(specs_override)
        else:
            self._specs: dict[str, SubagentSpec] = dict(self.BUILTIN_SPECS)

    # ------------------------------------------------------------------ API

    async def dispatch(
        self,
        spec_name: str,
        task: str,
        context: dict | None = None,
        on_state: Any = None,
        _depth: int = 0,
    ) -> SubagentResult:
        """Dispatch a subagent to handle a task in isolated context.

        context 里需要带 agent_factory (AgentFactory 实例), 没有就报错.
        on_state: optional async callback(state_dict) called for each
        intermediate agent state — lets callers stream subagent progress.

        _depth: 递归深度 (G1 守卫). 主 agent 调 dispatch 时 _depth=0,
        subagent 内再调 dispatch 时 _depth=1, 以此类推. 超 spec.max_depth 拒.
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

        # G1: 递归深度守卫. 超 max_depth 拒绝, 防 subagent 无限递归.
        if _depth >= spec.max_depth:
            return SubagentResult(
                summary="", full_output="",
                success=False,
                error=f"depth {_depth} >= max_depth {spec.max_depth} "
                      f"(spec={spec_name}). 递归过深, 拒绝 dispatch.",
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

        # G1: 把 _depth+1 注入 contextvar, 子 agent 调 subagent_tool 时
        # 能读到正确的递归深度. chat() 里的 tool call 同 task, contextvar 可见.
        token = _current_depth.set(_depth + 1)
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
        finally:
            _current_depth.reset(token)

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


# ── v14 Task 10: Čech H¹ finding 一致性检查 ────────────────────────────────

# ponytail: 真正 sheaf cohomology 算不动 (O(n^ω) 级), 用"语义相似度 + 数值冲突"
# 做 obstruction 代理. spec 诚实边界 §3 明说这是工程近似.
# 升级路径: 在 simplicial complex 上算真实 H¹, 需要 persistent homology 库.

# H¹ 检查的相似度阈值 — spec SubTask 10.2 规定
_H1_HIGH_OVERLAP = 0.8   # > 此值 + 数值不同 → obstruction
_H1_LOW_OVERLAP = 0.5    # ≤ 此值视为新信息, 无 obstruction
# 中间区 (0.5, 0.8] 保守不 reject

# 数值相近判据: 相对误差 < 10% 视为一致
_H1_NUM_REL_TOL = 0.1


def _is_close(a: float, b: float, rel_tol: float = _H1_NUM_REL_TOL) -> bool:
    """相对误差 < rel_tol 视为相近.

    ponytail: math.isclose 默认 abs_tol=1e-9 + rel_tol=1e-5 太严, 不适合工程近似.
    这里用纯相对误差, 分母 max(|a|,|b|,1e-9) 防 0 除. 升级: 跟着数值量级动态调 tol.
    """
    denom = max(abs(a), abs(b), 1e-9)
    return abs(a - b) / denom < rel_tol


def _check_finding_consistency(
    finding: dict | str,
    core_context: str,
) -> tuple[bool, str]:
    """v14 Task 10: Čech H¹ 一致性代理 — finding vs Core context claim.

    返回 (h1_zero, reason):
    - h1_zero=True  → 无 obstruction, finding 可拼入 Core context
    - h1_zero=False → obstruction, finding 应被 reject 写入 support_rejections.jsonl

    判据 (spec §"Support finding Čech H¹ 一致性检查"):
    - overlap ≤ 0.5         → 新信息, H¹=0
    - 0.5 < overlap ≤ 0.8   → 中等重叠, 保守不 reject, H¹=0
    - overlap > 0.8         → 高重叠, 必须数值一致:
        * finding 每个数值在 core_context 找不到相近的 (差 ≥ 10%) → H¹≠0
        * 否则 H¹=0
    """
    if isinstance(finding, dict):
        finding_str = json.dumps(finding, ensure_ascii=False, sort_keys=True)
    else:
        finding_str = str(finding) if finding else ""

    if not finding_str or not core_context:
        return True, "empty input, no obstruction"

    from huginn.context_builder import _compute_semantic_overlap
    overlap = _compute_semantic_overlap(finding_str, core_context)

    if overlap <= _H1_LOW_OVERLAP:
        return True, f"new info, no overlap (sim={overlap:.2f})"

    if overlap <= _H1_HIGH_OVERLAP:
        # ponytail: 中等重叠保守不 reject. 真正 obstruction 检测要算 sheaf H¹,
        # 成本太高. 升级: 加 LLM judge 做语义一致性判定.
        return True, f"moderate overlap, assuming consistent (sim={overlap:.2f})"

    # 高重叠: 数值必须一致, 否则 obstruction
    finding_nums = [float(x) for x in re.findall(r"\d+\.?\d*", finding_str)]
    core_nums = [float(x) for x in re.findall(r"\d+\.?\d*", core_context)]

    if not finding_nums:
        return True, f"high overlap, no numbers to conflict (sim={overlap:.2f})"

    for n in finding_nums:
        if not any(_is_close(n, m) for m in core_nums):
            closest = min(core_nums, key=lambda m: abs(n - m)) if core_nums else None
            return False, (
                f"numeric conflict: finding has {n}, core has {closest} "
                f"(sim={overlap:.2f})"
            )

    return True, f"high overlap, numbers consistent (sim={overlap:.2f})"


def _write_support_rejection(
    workspace: str | Path,
    finding: dict | str,
    reason: str,
    core_context: str,
) -> Path | None:
    """H¹≠0 时把 finding 落盘到 .huginn/support_rejections.jsonl 供后续 review.

    返回写入的 Path, workspace 无效时返回 None 静默跳过.
    """
    if not workspace:
        return None
    ws = Path(workspace)
    rejection_path = ws / ".huginn" / "support_rejections.jsonl"
    rejection_path.parent.mkdir(parents=True, exist_ok=True)
    snippet = core_context[:500] if isinstance(core_context, str) else str(core_context)[:500]
    with rejection_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "ts": datetime.now().isoformat(),
            "finding": finding,
            "reason": reason,
            "core_context_snippet": snippet,
        }, ensure_ascii=False, default=str) + "\n")
    return rejection_path


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

    # ── v14 Task 10: H¹ 一致性检查 self-check ────────────────────────────
    # 3 cases: numeric conflict / numbers consistent / new info
    import tempfile

    # case 1: 高重叠 + 数值冲突 (0.05 vs 0.5 差 ≥ 10%) → H¹≠0
    # 用长共现文本撑高 TF-IDF cosine 到 >0.8 阈值
    f1 = "the model achieved MAE=0.05 on test set with high confidence"
    c1 = "the model achieved MAE=0.5 on test set with high confidence"
    h1_zero, reason1 = _check_finding_consistency(f1, c1)
    print(f"[case1] h1_zero={h1_zero}, reason={reason1}")
    assert not h1_zero, f"0.05 vs 0.5 应触发 obstruction, got h1_zero={h1_zero}"
    assert "numeric conflict" in reason1, f"reason 应含 numeric conflict: {reason1}"

    # 验证 rejection 文件会被写入
    with tempfile.TemporaryDirectory() as tmp_ws:
        rej_path = _write_support_rejection(tmp_ws, f1, reason1, c1)
        assert rej_path is not None and rej_path.exists(), "rejection 文件应被写入"
        content = rej_path.read_text(encoding="utf-8").strip()
        assert "numeric conflict" in content, f"rejection 内容应含 reason: {content}"
        print(f"[case1] rejection written to {rej_path.name}")

    # case 2: 高重叠 + 数值一致 → H¹=0
    f2 = "the model achieved MAE=0.5 on test set with high confidence"
    c2 = "the model achieved MAE=0.5 on test set with high confidence"
    h1_zero, reason2 = _check_finding_consistency(f2, c2)
    print(f"[case2] h1_zero={h1_zero}, reason={reason2}")
    assert h1_zero, f"数值一致应 H¹=0, got h1_zero={h1_zero}, reason={reason2}"
    assert "consistent" in reason2, f"reason 应含 consistent: {reason2}"

    # case 3: 低重叠, 新信息 → H¹=0
    f3 = "quantum tunneling effect observed in semiconductor"
    c3 = "chemical reaction kinetics analysis"
    h1_zero, reason3 = _check_finding_consistency(f3, c3)
    print(f"[case3] h1_zero={h1_zero}, reason={reason3}")
    assert h1_zero, f"新信息应 H¹=0, got h1_zero={h1_zero}"
    assert "new info" in reason3, f"reason 应含 new info: {reason3}"

    print("\nv14 Task 10 self-check PASSED")

    # ── G1: 递归深度守卫 self-check ─────────────────────────────────────
    # spec.max_depth 默认 1 (单层), _depth >= max_depth 拒绝
    assert d._specs["explore"].max_depth == 1, "默认 max_depth 应为 1"

    # case A: _depth=1 >= max_depth=1 → 拒 (即使有 factory 也不让过)
    result = asyncio.run(d.dispatch("explore", "test", _depth=1))
    assert not result.success, f"_depth=1 应被拒, got success={result.success}"
    assert "depth" in (result.error or "").lower(), f"error 应含 depth: {result.error}"
    print(f"[ok] G1 depth guard 拒绝 _depth=1: {result.error}")

    # case B: _depth=0 < max_depth=1 → 不被 depth guard 拦, 走到 factory 检查
    result = asyncio.run(d.dispatch("explore", "test", _depth=0))
    assert not result.success, "无 factory 应失败"
    assert "depth" not in (result.error or "").lower(), \
        f"_depth=0 不该触发 depth guard: {result.error}"
    assert "agent_factory" in (result.error or ""), \
        f"_depth=0 应走到 factory 检查: {result.error}"
    print(f"[ok] G1 _depth=0 放行 depth guard, 走到 factory 检查")

    # case C: contextvar 默认 0
    assert _current_depth.get() == 0, "contextvar 默认应为 0"
    # 模拟 dispatch 设置后, contextvar 应 +1
    token = _current_depth.set(5)
    assert _current_depth.get() == 5
    _current_depth.reset(token)
    assert _current_depth.get() == 0
    print("[ok] G1 _current_depth contextvar set/reset 正常")

    # case D: max_depth 可配置
    deep_spec = SubagentSpec(
        name="deep_test",
        description="test deep",
        system_prompt="test",
        allowed_tools=["file_read_tool"],
        max_depth=3,
    )
    d.register_spec(deep_spec)
    assert d._specs["deep_test"].max_depth == 3
    # _depth=2 < 3 放行 (走到 factory 检查)
    result = asyncio.run(d.dispatch("deep_test", "test", _depth=2))
    assert not result.success and "depth" not in (result.error or "").lower(), \
        f"_depth=2 < max_depth=3 不该被 depth 拒: {result.error}"
    # _depth=3 >= 3 拒
    result = asyncio.run(d.dispatch("deep_test", "test", _depth=3))
    assert not result.success and "depth" in (result.error or "").lower(), \
        f"_depth=3 >= max_depth=3 应被拒: {result.error}"
    print("[ok] G1 max_depth 可配置, _depth=2 放行, _depth=3 拒")

    print("\nG1 self-check PASSED")

    print("\nAll self-checks passed.")
    sys.exit(0)
