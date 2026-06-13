"""Convenience helpers for instantiating LLM clients."""

from __future__ import annotations

from typing import Any

from matsci_agent.agent import _create_langchain_model
from matsci_agent.config import MatSciConfig, Settings


def get_model(config: MatSciConfig | Settings | None = None, temperature: float = 0.2) -> Any:
    """Return a LangChain chat model based on the active configuration.

    Parameters
    ----------
    config
        A :class:`~matsci_agent.config.MatSciConfig` or
        :class:`~matsci_agent.config.Settings` instance. If ``None``, settings are
        loaded from the environment.
    temperature
        Sampling temperature for the model.

    Returns
    -------
    Any
        A LangChain chat model with ``invoke`` / ``bind_tools`` support.
    """
    if config is None:
        cfg = MatSciConfig.from_env()
    elif isinstance(config, Settings):
        cfg = config.config
    else:
        cfg = config

    return _create_langchain_model(
        provider=cfg.provider,
        model_name=cfg.model,
        api_key=cfg.resolved_api_key,
        base_url=cfg.base_url,
        temperature=temperature,
    )
