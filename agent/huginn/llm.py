"""Convenience helpers for instantiating LLM clients."""

from __future__ import annotations

from typing import Any

from huginn.agent import _create_langchain_model
from huginn.config import HuginnConfig, Settings


def get_model(
    config: HuginnConfig | Settings | None = None,
    temperature: float = 0.2,
    thinking: Any | None = None,
    max_tokens: int | None = None,
) -> Any:
    """Return a LangChain chat model based on the active configuration.

    Parameters
    ----------
    config
        A :class:`~huginn.config.HuginnConfig` or
        :class:`~huginn.config.Settings` instance. If ``None``, settings are
        loaded from the environment.
    temperature
        Sampling temperature for the model.
    thinking
        Optional reasoning intensity override.
    max_tokens
        Optional max_tokens override.

    Returns
    -------
    Any
        A LangChain chat model with ``invoke`` / ``bind_tools`` support.
    """
    if config is None:
        cfg = HuginnConfig.from_env()
    elif isinstance(config, Settings):
        cfg = config.config
    else:
        cfg = config

    effective_thinking = thinking if thinking is not None else cfg.thinking
    effective_max_tokens = max_tokens if max_tokens is not None else cfg.max_tokens

    return _create_langchain_model(
        provider=cfg.provider,
        model_name=cfg.model,
        api_key=cfg.resolved_api_key,
        base_url=cfg.base_url,
        temperature=temperature,
        thinking=effective_thinking,
        max_tokens=effective_max_tokens,
    )
