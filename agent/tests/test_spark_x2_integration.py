"""星火 X2.5-4B 接入 — 端侧 1M 上下文小模型的三处声明验证.

讯飞词元星火 2026-09 开源端侧 X2.5-4B (原生 1M 上下文, 混合注意力, 面向智能体/
代码/指令遵循优化)。接入走 `openai-compatible` 通道 (讯飞星辰 MaaS 或本地 vLLM/
SGLang), 这里验证接入层三件套: 上下文窗口 / 能力表 / provider 注册.
"""

from __future__ import annotations

from huginn.context_manager import (
    get_context_window,
    get_model_max_output_tokens,
)
from huginn.models.registry import (
    _DOMESTIC_OPENAI_COMPATIBLE,
    _PROVIDER_DEFAULTS,
    get_model_capabilities,
    list_providers,
)

_MODEL = "spark-x2.5-4b"


def test_spark_context_window_is_1m():
    """百万上下文被登记: agent 据此给满窗口而不误收进 128K 兜底."""
    assert get_context_window(_MODEL) == 1_000_000


def test_spark_max_output_tokens_conservative():
    """输出上限未公开, 保守登记 (8K 默认 / 32K 上限)."""
    assert get_model_max_output_tokens(_MODEL) == (8192, 32768)


def test_spark_capabilities_tools_and_streaming():
    """能力表: 支持工具调用 + 流式; 原生长链推理未知 → fail-closed False."""
    caps = get_model_capabilities(_MODEL)
    assert caps.tools is True
    assert caps.streaming is True
    assert caps.reasoning is False
    # tools=True 派生 structured_output / parallel_tool_calls
    assert caps.structured_output is True
    assert caps.parallel_tool_calls is True


def test_spark_provider_registered_through_openai_compatible():
    """spark provider 挂到 MaaS OpenAI 兼容端点, 可被 create_langchain_model 解析."""
    assert "spark" in _DOMESTIC_OPENAI_COMPATIBLE
    assert _DOMESTIC_OPENAI_COMPATIBLE["spark"]["env"] == "SPARK_API_KEY"
    assert _DOMESTIC_OPENAI_COMPATIBLE["spark"]["base_url"].endswith("/v2")
    assert _PROVIDER_DEFAULTS["spark"] == _MODEL


def test_spark_provider_listed_for_ui():
    """UI/CLI 能列出并默认选中星火."""
    ids = {e["id"]: e for e in list_providers()}
    assert "spark" in ids
    assert ids["spark"]["default_model"] == _MODEL
    assert ids["spark"]["env_var"] == "SPARK_API_KEY"
