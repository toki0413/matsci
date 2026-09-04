"""PromptOptimizeTool 行为测试 — 全 mock, 不真调 LLM.

覆盖 (父任务要求 4 点):
  (a) recommend 返回建议结构 (无 LLM, 纯规则)
  (b) optimize 在 LLM 可用时调用 (monkeypatch _get_model 返回固定文本)
  (c) LLM 失败优雅降级为规则结果, 不崩
  (d) 三种 mode (replace/prepend/append) 行为

风格参考 test_meta_tools.py: async def + await tool.call + 断言 result.*.
"""

from __future__ import annotations

from types import SimpleNamespace

from huginn.tools.prompt_optimize_tool import PromptOptimizeTool


# 假 LLM: ainvoke 返回固定文本 content.
class FakeModel:
    def __init__(self, text):
        self._text = text

    async def ainvoke(self, messages):
        return SimpleNamespace(content=self._text)


def _ctx():
    return SimpleNamespace(session_id="t", config=None)


# ════════════════════════════════════════════════════════════════════
# (a) recommend — 纯规则, 返回建议结构
# ════════════════════════════════════════════════════════════════════


async def test_recommend_missing_objective_returns_error():
    """objective 为空 → 返回 success=False."""
    tool = PromptOptimizeTool()
    result = await tool.call({"action": "recommend", "objective": ""}, context=None)
    assert result.success is False
    assert "objective" in result.error


async def test_recommend_returns_suggestion_structure_no_llm():
    """recommend 返回结构/elements/context_refs/tool_usage 四个建议块 + 推荐工具.

    不依赖 LLM, source='rules'. 目标命中关键词时 recommended_tools 非空.
    """
    tool = PromptOptimizeTool()
    result = await tool.call(
        {"action": "recommend", "objective": "帮我验证计算 Si 的带隙"}, context=None
    )
    assert result.success is True
    d = result.data
    assert d["action"] == "recommend"
    assert d["source"] == "rules"
    for key in ("structure", "elements", "context_refs", "tool_usage"):
        assert isinstance(d[key], list) and len(d[key]) > 0, key
    assert isinstance(d["recommended_tools"], list)
    assert "验证" in str(d["recommended_tools"]) or "validate_tool" in str(
        d["recommended_tools"]
    )


# ════════════════════════════════════════════════════════════════════
# (b) optimize — LLM 可用时调用
# ════════════════════════════════════════════════════════════════════


async def test_optimize_uses_llm_replace(monkeypatch):
    """mode=replace, LLM 可用: 直接用 LLM 输出作为优化结果."""
    tool = PromptOptimizeTool()
    monkeypatch.setattr(
        tool, "_get_model", lambda ctx: FakeModel("LLM 重写后的完整提示词")
    )
    result = await tool.call(
        {
            "action": "optimize",
            "prompt": "计算 Si 带隙",
            "mode": "replace",
            "intent": "需要能量收敛阈值为 1e-5",
        },
        context=_ctx(),
    )
    assert result.success is True
    assert result.data["source"] == "llm"
    assert result.data["fallback"] is False
    assert result.data["optimized"] == "LLM 重写后的完整提示词"


async def test_optimize_uses_llm_prepend(monkeypatch):
    """mode=prepend, LLM 可用: LLM 片段 + '\\n' + 原提示词."""
    tool = PromptOptimizeTool()
    monkeypatch.setattr(tool, "_get_model", lambda ctx: FakeModel("前置引导语"))
    prompt = "计算 Si 带隙"
    result = await tool.call(
        {"action": "optimize", "prompt": prompt, "mode": "prepend"},
        context=_ctx(),
    )
    assert result.success is True
    assert result.data["source"] == "llm"
    assert result.data["optimized"] == "前置引导语\n" + prompt


# ════════════════════════════════════════════════════════════════════
# (c) LLM 失败 → 降级为规则结果
# ════════════════════════════════════════════════════════════════════


async def test_optimize_degrades_when_llm_fails(monkeypatch):
    """_get_model 抛异常 → 降级到规则模板, success 仍为 True 不崩."""
    tool = PromptOptimizeTool()

    def boom(ctx):
        raise RuntimeError("no llm available")

    monkeypatch.setattr(tool, "_get_model", boom)
    result = await tool.call(
        {"action": "optimize", "prompt": "计算 Si 带隙", "mode": "replace"},
        context=_ctx(),
    )
    assert result.success is True
    assert result.data["source"] == "rules"
    assert result.data["fallback"] is True
    assert result.data["applied"] == "rule_replace"
    assert "计算 Si 带隙" in result.data["optimized"]


async def test_optimize_missing_prompt_returns_error():
    """optimize 但 prompt 为空 → 返回 success=False."""
    tool = PromptOptimizeTool()
    result = await tool.call({"action": "optimize", "prompt": ""}, context=None)
    assert result.success is False
    assert "prompt" in result.error


# ════════════════════════════════════════════════════════════════════
# (d) 三种 mode 行为 (规则降级路径, 确定性)
# ════════════════════════════════════════════════════════════════════


async def test_optimize_three_modes_rule(monkeypatch):
    """LLM 失败下降级: replace 整体改写 / prepend 前缀 / append 后缀."""
    tool = PromptOptimizeTool()

    def boom(ctx):
        raise RuntimeError("no llm")

    monkeypatch.setattr(tool, "_get_model", boom)
    prompt = "原始提示词内容"
    r_replace = await tool.call(
        {"action": "optimize", "prompt": prompt, "mode": "replace"}, context=None
    )
    r_prepend = await tool.call(
        {"action": "optimize", "prompt": prompt, "mode": "prepend"}, context=None
    )
    r_append = await tool.call(
        {"action": "optimize", "prompt": prompt, "mode": "append"}, context=None
    )
    # replace: 不是原样, 是补结构后的全量文本
    assert r_replace.data["optimized"] != prompt
    assert r_replace.data["applied"] == "rule_replace"
    # prepend: 前缀 + 原提示词
    assert r_prepend.data["optimized"].endswith(prompt)
    assert r_prepend.data["applied"] == "rule_prepend"
    # append: 原提示词 + 后缀
    assert r_append.data["optimized"].startswith(prompt)
    assert r_append.data["applied"] == "rule_append"
    # 三者互不相同
    assert (
        len(
            {
                r_replace.data["optimized"],
                r_prepend.data["optimized"],
                r_append.data["optimized"],
            }
        )
        == 3
    )
