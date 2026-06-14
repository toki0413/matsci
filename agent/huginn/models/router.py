"""Multi-LLM router — route tasks to the cheapest/strongest available model.

Huginn is single-model by default, but the router lets power users register
multiple models and pick one per task:

- ``coding`` / ``science`` / ``reasoning`` -> strongest cloud model
- ``summarize`` / ``format`` / ``cheap`` -> small/cheap model
- ``local`` -> Ollama / vLLM
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Literal

from huginn.models.registry import ProviderT, create_langchain_model


TaskT = Literal[
    "default",
    "agent",
    "coding",
    "science",
    "reasoning",
    "summarize",
    "format",
    "cheap",
    "local",
]


@dataclass
class RegisteredModel:
    """A model entry in the router pool."""

    name: str
    model: Any
    tags: set[str] = field(default_factory=set)
    cost_input: float = 0.0  # USD per 1M input tokens
    cost_output: float = 0.0  # USD per 1M output tokens
    priority: int = 0  # higher = preferred when multiple candidates match


class ModelRouter:
    """Pool of models with task-based selection and cost-aware fallback."""

    # Task -> preferred tags, in order of preference.
    _TASK_TAGS: dict[TaskT, list[str]] = {
        "default": ["default", "agent"],
        "agent": ["agent", "default"],
        "coding": ["coding", "agent", "default"],
        "science": ["science", "reasoning", "agent", "default"],
        "reasoning": ["reasoning", "science", "agent", "default"],
        "summarize": ["summarize", "cheap", "default"],
        "format": ["format", "cheap", "default"],
        "cheap": ["cheap", "summarize", "default"],
        "local": ["local", "default"],
    }

    def __init__(self, default_task: TaskT = "default") -> None:
        self._models: dict[str, RegisteredModel] = {}
        self._default_task: TaskT = default_task

    def register(
        self,
        name: str,
        model: Any,
        tags: set[str] | None = None,
        cost_input: float = 0.0,
        cost_output: float = 0.0,
        priority: int = 0,
    ) -> RegisteredModel:
        """Register an existing LangChain model instance."""
        entry = RegisteredModel(
            name=name,
            model=model,
            tags=tags or {"default"},
            cost_input=cost_input,
            cost_output=cost_output,
            priority=priority,
        )
        self._models[name] = entry
        return entry

    def register_provider(
        self,
        name: str,
        provider: ProviderT,
        model_name: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        tags: set[str] | None = None,
        cost_input: float = 0.0,
        cost_output: float = 0.0,
        priority: int = 0,
        temperature: float = 0.7,
    ) -> RegisteredModel:
        """Register a model by provider descriptor."""
        model = create_langchain_model(
            provider=provider,
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
        )
        return self.register(
            name=name,
            model=model,
            tags=tags,
            cost_input=cost_input,
            cost_output=cost_output,
            priority=priority,
        )

    def select(self, task: TaskT | str | None = None, prefer_cheap: bool = False) -> Any:
        """Pick the best model for ``task``.

        Falls back to the first registered model if no tag matches.
        """
        task = task or self._default_task
        task = task if isinstance(task, str) else task.value  # type: ignore[attr-defined]
        preferred_tags = self._TASK_TAGS.get(task, ["default"])  # type: ignore[arg-type]

        candidates: list[RegisteredModel] = []
        for tag in preferred_tags:
            matches = [m for m in self._models.values() if tag in m.tags]
            if matches:
                candidates = matches
                break

        if not candidates:
            candidates = list(self._models.values())
        if not candidates:
            raise RuntimeError("ModelRouter has no registered models")

        if prefer_cheap:
            candidates.sort(key=lambda m: (m.cost_input + m.cost_output, -m.priority))
        else:
            candidates.sort(key=lambda m: (-m.priority, m.cost_input + m.cost_output))

        return candidates[0].model

    def list_models(self) -> list[str]:
        """Return registered model names."""
        return list(self._models.keys())

    @classmethod
    def from_env(cls) -> "ModelRouter":
        """Build a router from environment variables.

        Example:
            HUGINN_MODEL_DEFAULT=openai:gpt-4o
            HUGINN_MODEL_CHEAP=openai:gpt-4o-mini
            HUGINN_MODEL_LOCAL=ollama:qwen2.5:14b
        """
        router = cls()
        prefix = "HUGINN_MODEL_"
        for key, value in os.environ.items():
            if not key.startswith(prefix):
                continue
            name = key[len(prefix) :].lower()
            if ":" not in value:
                continue
            provider, model_name = value.split(":", 1)
            try:
                model = create_langchain_model(provider=provider, model_name=model_name)
                router.register(name, model, tags={name})
            except Exception:
                # Skip providers that are not configured (missing API keys, etc.)
                continue
        return router
