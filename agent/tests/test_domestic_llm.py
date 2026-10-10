
from __future__ import annotations

import pytest

pytest.importorskip("openai", reason="openai SDK not installed")

"""Tests for domestic / OpenAI-compatible LLM providers."""

from typing import Any  # noqa: E402

import pytest  # noqa: E402
from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402

from huginn.config import HuginnConfig  # noqa: E402
from huginn.models.registry import (  # noqa: E402
    _merge_system_messages,
    create_langchain_model,
)


class _FakeChatOpenAI:
    """Capture kwargs passed to ChatOpenAI."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


def _patch_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("langchain_openai.ChatOpenAI", _FakeChatOpenAI)


class TestDomesticProviders:
    @pytest.mark.parametrize(
        "provider,env_var,expected_base,default_model",
        [
            (
                "deepseek",
                "DEEPSEEK_API_KEY",
                "https://api.deepseek.com",
                "deepseek-v4-flash",
            ),
            (
                "siliconflow",
                "SILICONFLOW_API_KEY",
                "https://api.siliconflow.cn/v1",
                "deepseek-ai/DeepSeek-V3",
            ),
            (
                "moonshot",
                "MOONSHOT_API_KEY",
                "https://api.moonshot.cn/v1",
                "kimi-k2.6",
            ),
            (
                "zhipu",
                "ZHIPU_API_KEY",
                "https://open.bigmodel.cn/api/paas/v4/",
                "glm-4-flash",
            ),
            (
                "baichuan",
                "BAICHUAN_API_KEY",
                "https://api.baichuan-ai.com/v1",
                "Baichuan4",
            ),
            (
                "dashscope",
                "DASHSCOPE_API_KEY",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "qwen3.5-plus",
            ),
            (
                "qianfan",
                "QIANFAN_API_KEY",
                "https://qianfan.baidubce.com/v2",
                "ernie-4.0-turbo-8k",
            ),
            (
                "doubao",
                "DOUBAO_API_KEY",
                "https://ark.cn-beijing.volces.com/api/v3",
                "doubao-pro-32k",
            ),
            (
                "hunyuan",
                "HUNYUAN_API_KEY",
                "https://api.hunyuan.tencentcloudapi.com/v1",
                "hunyuan-turbo",
            ),
        ],
    )
    def test_default_base_url_and_model(
        self,
        monkeypatch: pytest.MonkeyPatch,
        provider: str,
        env_var: str,
        expected_base: str,
        default_model: str,
    ):
        _patch_openai(monkeypatch)
        monkeypatch.setenv(env_var, "test-key")
        model = create_langchain_model(provider=provider)
        assert model.kwargs["model"] == default_model
        assert model.kwargs["base_url"] == expected_base
        assert model.kwargs["api_key"] == "test-key"

    def test_openai_compatible_requires_base_url(self, monkeypatch: pytest.MonkeyPatch):
        _patch_openai(monkeypatch)
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        with pytest.raises(ValueError, match="base_url"):
            create_langchain_model(provider="openai-compatible", model_name="my-model")

    def test_openai_compatible_requires_model(self, monkeypatch: pytest.MonkeyPatch):
        _patch_openai(monkeypatch)
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        with pytest.raises(ValueError, match="model name"):
            create_langchain_model(
                provider="openai-compatible", base_url="http://localhost:8000/v1"
            )

    def test_openai_compatible_uses_provided_values(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        _patch_openai(monkeypatch)
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        model = create_langchain_model(
            provider="openai-compatible",
            model_name="my-model",
            base_url="http://localhost:8000/v1",
            temperature=0.5,
        )
        assert model.kwargs["model"] == "my-model"
        assert model.kwargs["base_url"] == "http://localhost:8000/v1"
        assert model.kwargs["temperature"] == 0.5

    def test_missing_api_key_raises(self, monkeypatch: pytest.MonkeyPatch):
        _patch_openai(monkeypatch)
        monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
        with pytest.raises(ValueError, match="MOONSHOT_API_KEY"):
            create_langchain_model(provider="moonshot")

    def test_explicit_model_overrides_default(self, monkeypatch: pytest.MonkeyPatch):
        _patch_openai(monkeypatch)
        monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
        model = create_langchain_model(provider="dashscope", model_name="qwen-turbo")
        assert model.kwargs["model"] == "qwen-turbo"

    def test_explicit_base_url_overrides_default(self, monkeypatch: pytest.MonkeyPatch):
        _patch_openai(monkeypatch)
        monkeypatch.setenv("ZHIPU_API_KEY", "test-key")
        custom_url = "https://private.example.com/v1"
        model = create_langchain_model(provider="zhipu", base_url=custom_url)
        assert model.kwargs["base_url"] == custom_url


class TestSystemMessageNormalization:
    """回归: OpenAI 兼容端点在请求前把多条/错位 system 收敛为队首单条.

    internlm (intern-s2-preview) 只接受最多一条且位于首条的 system 消息,
    命中即回 "in prompt processing error". huginn 会在会话中注入多条 system,
    故在模型边界归一是必要的.
    """

    def test_merges_multiple_systems_to_single_leading(self):
        msgs = [
            HumanMessage(content="q"),
            SystemMessage(content="STYLE"),
            HumanMessage(content="inner"),
            SystemMessage(content="BUDGET"),
        ]
        out = _merge_system_messages(msgs)
        assert isinstance(out[0], SystemMessage)
        assert "STYLE" in out[0].content
        assert "BUDGET" in out[0].content
        assert sum(isinstance(m, SystemMessage) for m in out) == 1
        # 非 system 消息相对顺序不变
        assert [type(m) for m in out] == [
            SystemMessage,
            HumanMessage,
            HumanMessage,
        ]

    def test_single_leading_system_unchanged(self):
        msgs = [SystemMessage(content="SYS"), HumanMessage(content="q")]
        assert _merge_system_messages(msgs) is msgs

    def test_no_system_unchanged(self):
        msgs = [HumanMessage(content="q")]
        assert _merge_system_messages(msgs) is msgs

    def test_hoists_single_trailing_system(self):
        msgs = [HumanMessage(content="q"), SystemMessage(content="S")]
        out = _merge_system_messages(msgs)
        assert isinstance(out[0], SystemMessage)
        assert [type(m) for m in out] == [SystemMessage, HumanMessage]

    def test_payload_collapses_system_messages(self, monkeypatch: pytest.MonkeyPatch):
        # 不 mock ChatOpenAI: 走真实 _get_request_payload (不触发网络).
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        model = create_langchain_model(
            provider="openai-compatible",
            model_name="m",
            base_url="http://localhost:8000/v1",
        )
        payload = model._get_request_payload(
            [
                HumanMessage(content="q"),
                SystemMessage(content="STYLE"),
                HumanMessage(content="inner"),
                SystemMessage(content="BUDGET"),
            ]
        )
        roles = [m["role"] for m in payload["messages"]]
        assert roles[0] == "system"
        assert roles.count("system") == 1
        assert "STYLE" in payload["messages"][0]["content"]
        assert "BUDGET" in payload["messages"][0]["content"]


class TestConfigParsingDomestic:
    def test_legacy_env_path_moonshot(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("HUGINN_PROVIDER", "moonshot")
        monkeypatch.setenv("HUGINN_MODEL", "moonshot-v1-32k")
        monkeypatch.setenv("MOONSHOT_API_KEY", "test-key")
        cfg = HuginnConfig.from_env()
        assert cfg.provider == "moonshot"
        assert cfg.models[0].provider == "moonshot"
        assert cfg.models[0].model == "moonshot-v1-32k"

    def test_huginn_models_json_domestic(self, monkeypatch: pytest.MonkeyPatch):
        import json

        monkeypatch.setenv(
            "HUGINN_MODELS",
            json.dumps(
                [
                    {
                        "alias": "qwen",
                        "provider": "dashscope",
                        "model": "qwen-max",
                    }
                ]
            ),
        )
        monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
        cfg = HuginnConfig.from_env()
        assert cfg.models[0].provider == "dashscope"
