
from __future__ import annotations

import pytest
pytest.importorskip("openai", reason="openai SDK not installed")

"""Tests for domestic / OpenAI-compatible LLM providers."""

from typing import Any

import pytest

from huginn.config import HuginnConfig
from huginn.models.registry import create_langchain_model


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
                "deepseek-chat",
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
                "moonshot-v1-8k",
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
                "qwen-max",
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