"""PRM-style step verifier (P1).

参考 Awesome-Long-Horizon-Agents survey 里 Pillar I 的 Verification 章节:
- Math-Shepherd: step-level reward
- MCTS-Judge: 树搜索中的 step verifier
- Process Reward Model: 中间步骤打分, 早停坏 step

huginn 已有 outcome-level 验证 (KRCL plan_check + Darwin ratchet + surprise),
本模块补 step-level 验证. 对长轨迹 (H2/H3) 早停坏 step, 不烧完整 trajectory.

设计:
- 不训练专用 PRM 模型 (成本高). 直接调现有 LLM 给 step 评分.
- 只对"重步" (VASP / LAMMPS / QE / Gaussian / 任何高 IO 工具) 评分, 跳过
  轻工具 (read / list / grep / 文档查询).
- 不强制 block 工具结果, 只往 ctx.metadata 塞 step_score / step_concerns.
- 累计 N 步低分时, Engine 自己走 should_pause_for_decision 触发 pause.

接入: HookManager.register(POST_TOOL_USE, StepVerifierHook(llm_chat_fn)).

升级路径:
1. 缓存评分结果 (相同 args + result hash 不重复调 LLM)
2. 训练专用 7B PRM 模型 (替代 LLM 评分, 速度↑ 成本↓)
3. MCTS 式树搜索 (用 verifier 给候选 step 打分, 选最优 branch)
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)


# === 配置 ===

# 需要评分的工具: 高成本 / 高 IO / 不可逆的工具
# 跳过: read / list / glob / grep / 文档查询等轻工具
_HEAVY_TOOL_PREFIXES: tuple[str, ...] = (
    "vasp_", "lammps_", "qe_", "quantum_", "gaussian_", "orca_",
    "cp2k_", "gromacs_", "abacus_", "abinit_",
    "structure_tool", "validate_tool", "numerical_tool",
    "hpc_submit", "hpc_run", "remote_exec",
)

# 评分缓存 (args+result 哈希 → score). 简单 dict, 进程级.
# ponytail: 不上 LRU, 重步工具调用一般不会无限重复, dict 够用.
_SCORE_CACHE: dict[str, float] = {}

# 低分阈值: < 0.4 视为坏 step
_LOW_SCORE_THRESHOLD = 0.4
# 累计触发阈值: 连续 N 步低分 → 该 pause (Engine 走 should_pause_for_decision)
_LOW_SCORE_STREAK = 3


def _is_heavy_tool(tool_name: str) -> bool:
    """判断是否需要评分. ponytail: 简单前缀匹配."""
    name = (tool_name or "").lower()
    return any(name.startswith(p) for p in _HEAVY_TOOL_PREFIXES)


def _step_signature(tool_name: str, args: Any, result: Any) -> str:
    """生成 step 签名, 用于缓存.

    args + result 序列化后取 md5, 避免相同输入重复调 LLM.
    ponytail: 用 default=str 容错非可序列化对象.
    """
    try:
        payload = json.dumps(
            {"tool": tool_name, "args": args, "result": result},
            default=str, sort_keys=True, ensure_ascii=False,
        )
    except Exception:
        return f"{tool_name}:{id(args)}:{id(result)}"
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


# === 评分结果 ===


@dataclass
class StepScore:
    """单步评分结果.

    score: 0.0-1.0, 越高越好. < 0.4 视为坏 step.
    concerns: 担心点列表 (如 "ENCUT=200 太低, 可能不收敛").
    action: 建议动作 ("proceed" / "warn" / "pause" / "redo").
    reasoning: LLM 给的简短理由.
    """
    score: float = 1.0
    concerns: list[str] = field(default_factory=list)
    action: str = "proceed"  # proceed / warn / pause / redo
    reasoning: str = ""

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 3),
            "concerns": list(self.concerns),
            "action": self.action,
            "reasoning": self.reasoning,
        }


# === 评分 Prompt ===

_SCORE_PROMPT_TEMPLATE = """You are a process reward model (PRM) for a materials science agent.

Score the following tool step on a scale of 0.0 to 1.0:
- 1.0 = perfect step, definitely advances the goal correctly
- 0.7 = good step, minor concerns
- 0.4 = questionable, may be wrong or wasteful
- 0.0 = clearly wrong, will fail or derail the goal

Tool: {tool_name}
Args: {args}
Result: {result}
Previous step context (last 3 lines): {prev_context}

Respond JSON only, no markdown fences:
{{"score": 0.0, "concerns": ["..."], "action": "proceed|warn|pause|redo", "reasoning": "1 sentence why"}}

Criteria for low score:
- Wrong physics (wrong functional, wrong ensemble, wrong unit conversion)
- Numerically unstable (k-spacing too sparse, time step too large, basis too small)
- Wastes compute (redundant calculation, should reuse prior result)
- Mismatched with goal (tool is right but args don't serve stated goal)
- Result itself reports failure or non-convergence

Criteria for high score:
- Args match best practice for the calculation type
- Result looks physically reasonable (energy negative, band gap plausible)
- Step directly serves the stated goal
"""


# === Hook ===


class StepVerifierHook:
    """POST_TOOL_USE hook: 给重步工具调用评分.

    用法:
        hook = StepVerifierHook(llm_chat_fn=your_async_llm)
        hook_manager.register(POST_TOOL_USE, hook)

    ponytail: 不强依赖 LLM 可用. llm_chat_fn=None 或失败时跳过评分,
    只记 log. 不可逆操作的拦截仍由 sandbox / permissions 兜底.
    """

    def __init__(
        self,
        llm_chat_fn: Callable[[str], Awaitable[str]] | None = None,
        *,
        low_score_threshold: float = _LOW_SCORE_THRESHOLD,
        prev_context_fn: Callable[[], str] | None = None,
    ) -> None:
        self._llm = llm_chat_fn
        self._low_threshold = low_score_threshold
        # 抽取之前步骤上下文给 LLM 看. 默认 None, Engine 可注入.
        self._prev_context_fn = prev_context_fn

    async def __call__(self, ctx) -> None:
        """POST_TOOL_USE hook callback. 修改 ctx.metadata, 不返回."""
        try:
            if not _is_heavy_tool(ctx.tool_name):
                return
            if ctx.error is not None:
                # 工具自身失败, 不评分 (AnomalyDetectionHook 会处理)
                return
            if self._llm is None:
                # 无 LLM 不评分, 不写 metadata, 完全跳过
                return

            score = await self._score_step(ctx)
            if score is None:
                # 评分跳过 (无 LLM / 异常 / 解析失败), 不写 metadata
                return
            ctx.metadata["step_score"] = score.to_dict()
            if score.score < self._low_threshold:
                ctx.metadata["step_low_score"] = True
                logger.warning(
                    "PRM step verifier: %s low score %.2f — %s",
                    ctx.tool_name, score.score, score.reasoning,
                )
        except Exception:
            # 评分失败不能把 agent 搞挂
            logger.warning("StepVerifierHook raised", exc_info=True)

    async def _score_step(self, ctx) -> StepScore | None:
        """调 LLM 给单步评分. 失败/无 LLM → None (跳过, 不写 metadata)."""
        if self._llm is None:
            return None

        sig = _step_signature(ctx.tool_name, ctx.args, ctx.result)
        cached = _SCORE_CACHE.get(sig)
        if cached is not None:
            # 缓存命中: 返回简化对象 (concerns / reasoning 缓存里没存)
            return StepScore(score=cached)

        prev_ctx = ""
        if self._prev_context_fn is not None:
            try:
                prev_ctx = self._prev_context_fn()[:500]
            except Exception:
                prev_ctx = ""

        prompt = self._build_prompt(ctx, prev_ctx)
        try:
            resp = await self._llm(prompt)
        except Exception:
            logger.debug("LLM step scoring failed (non-fatal)", exc_info=True)
            return None

        score = self._parse_response(resp)
        if score is None:
            return None
        _SCORE_CACHE[sig] = score.score
        return score

    def _build_prompt(self, ctx, prev_ctx: str) -> str:
        """构造 PRM 评分 prompt. 字段截断避免 prompt 爆炸."""
        args_str = self._truncate(json.dumps(ctx.args, default=str, ensure_ascii=False), 800)
        result_str = self._truncate(
            json.dumps(ctx.result, default=str, ensure_ascii=False), 1500)
        return _SCORE_PROMPT_TEMPLATE.format(
            tool_name=ctx.tool_name,
            args=args_str,
            result=result_str,
            prev_context=prev_ctx or "(no prior context)",
        )

    @staticmethod
    def _truncate(s: str, n: int) -> str:
        return s if len(s) <= n else s[:n] + "..."

    @staticmethod
    def _parse_response(resp: str) -> StepScore | None:
        """从 LLM response 抽 StepScore. 解析失败 → None (跳过, 不写 metadata)."""
        if not resp:
            return None
        resp = resp.strip()
        # 去 markdown fence (LLM 偶尔加)
        if resp.startswith("```"):
            resp = resp.strip("`")
            if resp.lower().startswith("json"):
                resp = resp[4:]
        start = resp.find("{")
        end = resp.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            data = json.loads(resp[start:end + 1])
        except json.JSONDecodeError:
            return None
        try:
            score = float(data.get("score", 1.0))
        except (TypeError, ValueError):
            return None
        # 钳位 [0, 1]
        score = max(0.0, min(1.0, score))
        concerns = data.get("concerns") or []
        if not isinstance(concerns, list):
            concerns = [str(concerns)]
        action = str(data.get("action", "proceed")).lower()
        if action not in ("proceed", "warn", "pause", "redo"):
            action = "proceed"
        reasoning = str(data.get("reasoning", ""))[:300]
        return StepScore(
            score=score,
            concerns=[str(c)[:200] for c in concerns][:5],
            action=action,
            reasoning=reasoning,
        )


# === 默认 LLM chat fn (factory.py 注册时用) ===


def make_default_llm_chat_fn() -> Callable[[str], Awaitable[str]] | None:
    """懒加载一个默认 LLM (deepseek-chat) 给 StepVerifierHook 用.

    失败 (没 key / 包没装 / 网络挂) 返回 None, 调用方自己降级.

    ponytail: 复用 anomaly_llm_hook 的同款 deepseek-chat, 不另开模型.
    """
    try:
        from huginn.models.registry import create_langchain_model

        model = create_langchain_model(
            provider="deepseek",
            model_name="deepseek-chat",
            temperature=0.0,
            max_tokens=300,
        )
    except Exception as exc:
        logger.debug("StepVerifierHook 默认模型初始化失败: %s", exc)
        return None

    async def _chat(prompt: str) -> str:
        from langchain_core.messages import HumanMessage
        try:
            resp = await model.ainvoke([HumanMessage(content=prompt)])
            # langchain 返回的是 AIMessage, .content 是 str
            return getattr(resp, "content", str(resp))
        except Exception:
            raise

    return _chat


# === Engine 端累计低分检查 ===


def should_pause_for_low_score_streak(
    recent_scores: list[float],
    *,
    threshold: float = _LOW_SCORE_THRESHOLD,
    streak: int = _LOW_SCORE_STREAK,
) -> tuple[bool, str]:
    """检查最近 N 步评分是否连续低于阈值.

    Engine 持有 recent_scores 列表 (从 ctx.metadata['step_score']['score']
    累计), 每步后调本函数.

    Returns:
        (should_pause, reason)
    """
    if len(recent_scores) < streak:
        return (False, "")
    recent = recent_scores[-streak:]
    if all(s < threshold for s in recent):
        avg = sum(recent) / len(recent)
        return (
            True,
            f"连续 {streak} 步 PRM 评分 < {threshold} (最近均值 {avg:.2f})",
        )
    return (False, "")


# === 自检 ===

if __name__ == "__main__":
    import asyncio

    # 1. _is_heavy_tool
    assert _is_heavy_tool("vasp_run")
    assert _is_heavy_tool("lammps_simulate")
    assert _is_heavy_tool("structure_tool")
    assert _is_heavy_tool("validate_tool")
    assert not _is_heavy_tool("read_file")
    assert not _is_heavy_tool("list_dir")
    assert not _is_heavy_tool("")
    assert not _is_heavy_tool("grep")

    # 2. _step_signature — 相同输入相同签名
    sig1 = _step_signature("vasp_run", {"encut": 400}, {"energy": -10.5})
    sig2 = _step_signature("vasp_run", {"encut": 400}, {"energy": -10.5})
    assert sig1 == sig2, "相同输入应同签名"

    sig3 = _step_signature("vasp_run", {"encut": 500}, {"energy": -10.5})
    assert sig1 != sig3, "不同 args 应不同签名"

    # 3. StepScore.to_dict
    s = StepScore(score=0.3, concerns=["ENCUT 太低"], action="warn",
                  reasoning="basis too small")
    d = s.to_dict()
    assert d["score"] == 0.3
    assert d["concerns"] == ["ENCUT 太低"]
    assert d["action"] == "warn"

    # 4. _parse_response — 正常 JSON
    resp = '{"score": 0.6, "concerns": ["k-spacing sparse"], "action": "warn", "reasoning": "may not converge"}'
    score = StepVerifierHook._parse_response(resp)
    assert score.score == 0.6
    assert score.concerns == ["k-spacing sparse"]
    assert score.action == "warn"
    assert "converge" in score.reasoning

    # 4b. markdown fence
    resp_fenced = '```json\n{"score": 0.2, "action": "pause", "reasoning": "wrong"}\n```'
    score = StepVerifierHook._parse_response(resp_fenced)
    assert score.score == 0.2
    assert score.action == "pause"

    # 4c. 非 JSON → None (跳过, 不写 metadata)
    score = StepVerifierHook._parse_response("not json at all")
    assert score is None, "非 JSON 应返回 None"

    # 4d. score 越界钳位
    score = StepVerifierHook._parse_response('{"score": 1.5}')
    assert score is not None and score.score == 1.0, f"钳位到 1.0, got {score}"
    score = StepVerifierHook._parse_response('{"score": -0.5}')
    assert score is not None and score.score == 0.0

    # 4e. action 非法 → proceed
    score = StepVerifierHook._parse_response('{"score": 0.5, "action": "explode"}')
    assert score is not None and score.action == "proceed"

    # 4f. score 非法 (字符串) → None
    score = StepVerifierHook._parse_response('{"score": "abc"}')
    assert score is None, "score 非法应返回 None"

    # 4g. 空 response → None
    assert StepVerifierHook._parse_response("") is None
    assert StepVerifierHook._parse_response(None) is None

    # 5. should_pause_for_low_score_streak
    # 连续 3 步低分 → pause
    pause, reason = should_pause_for_low_score_streak(
        [0.3, 0.2, 0.1])
    assert pause and "3 步" in reason

    # 2 步低分但窗口不够 → 不触发
    pause, _ = should_pause_for_low_score_streak([0.3, 0.2])
    assert not pause

    # 3 步但只有 2 步低分 → 不触发
    pause, _ = should_pause_for_low_score_streak([0.3, 0.8, 0.2])
    assert not pause

    # 5 步最近 3 步都低 → 触发
    pause, reason = should_pause_for_low_score_streak(
        [0.9, 0.8, 0.2, 0.1, 0.3])
    assert pause and "3 步" in reason

    # 6. hook 调用 — 轻工具跳过 (mock ctx)
    class _MockCtx:
        def __init__(self, tool_name, args=None, result=None, error=None):
            self.tool_name = tool_name
            self.args = args or {}
            self.result = result or {}
            self.error = error
            self.metadata = {}

    async def _mock_llm_good(prompt: str) -> str:
        return '{"score": 0.8, "action": "proceed"}'

    async def _mock_llm_bad(prompt: str) -> str:
        return '{"score": 0.2, "concerns": ["ENCUT=200 太低"], "action": "warn"}'

    async def _mock_llm_raise(prompt: str) -> str:
        raise RuntimeError("LLM offline")

    async def _run_hook_test():
        # 轻工具不评分 (用 _mock_llm_bad 是故意的: 即使配了 LLM 也不评)
        hook_bad = StepVerifierHook(llm_chat_fn=_mock_llm_bad)
        ctx = _MockCtx("read_file", args={"path": "a.txt"}, result={"content": "hi"})
        await hook_bad(ctx)
        assert "step_score" not in ctx.metadata, "轻工具不应评分"

        # 工具失败不评分
        ctx = _MockCtx("vasp_run", error=RuntimeError("boom"))
        await hook_bad(ctx)
        assert "step_score" not in ctx.metadata, "失败工具不应评分"

        # 重工具 + LLM 给高分
        hook_good = StepVerifierHook(llm_chat_fn=_mock_llm_good)
        ctx = _MockCtx("vasp_run", args={"encut": 500}, result={"energy": -10.5})
        await hook_good(ctx)
        assert "step_score" in ctx.metadata
        assert ctx.metadata["step_score"]["score"] == 0.8
        assert "step_low_score" not in ctx.metadata

        # 重工具 + LLM 给低分
        ctx = _MockCtx("vasp_run", args={"encut": 200}, result={"energy": -5.0})
        hook_bad = StepVerifierHook(llm_chat_fn=_mock_llm_bad)
        await hook_bad(ctx)
        assert ctx.metadata["step_score"]["score"] == 0.2
        assert ctx.metadata.get("step_low_score") is True

        # LLM 不可用时跳过评分
        ctx = _MockCtx("vasp_run")
        hook_nolllm = StepVerifierHook(llm_chat_fn=None)
        await hook_nolllm(ctx)
        assert "step_score" not in ctx.metadata, "无 LLM 应跳过"

        # LLM 抛异常时不崩
        ctx = _MockCtx("vasp_run")
        hook_raise = StepVerifierHook(llm_chat_fn=_mock_llm_raise)
        await hook_raise(ctx)
        # 异常被吞, 不崩, 不写 metadata
        assert "step_score" not in ctx.metadata

        # 缓存命中: 同输入第二次调, 不再调 LLM
        call_count = [0]
        async def _count_llm(prompt: str) -> str:
            call_count[0] += 1
            return '{"score": 0.5}'

        hook_cached = StepVerifierHook(llm_chat_fn=_count_llm)
        ctx1 = _MockCtx("vasp_run", args={"x": 1}, result={"y": 2})
        ctx2 = _MockCtx("vasp_run", args={"x": 1}, result={"y": 2})
        await hook_cached(ctx1)
        await hook_cached(ctx2)
        assert call_count[0] == 1, f"缓存命中应只调 LLM 1次, got {call_count[0]}"
        assert ctx1.metadata["step_score"]["score"] == 0.5
        assert ctx2.metadata["step_score"]["score"] == 0.5

    asyncio.run(_run_hook_test())

    print("step_verifier selfcheck All passed")
