"""Multi-LLM router — route tasks to the cheapest/strongest available model.

Huginn is single-model by default, but the router lets power users register
multiple models and pick one per task:

- ``coding`` / ``science`` / ``reasoning`` -> strongest cloud model
- ``summarize`` / ``format`` / ``cheap`` -> small/cheap model
- ``local`` -> Ollama / vLLM
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal

from huginn.config import ThinkingIntensity
from huginn.models.registry import ProviderT, create_langchain_model

logger = logging.getLogger(__name__)

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
    # Moonshine 三槽: main 生成假设, verification 用不同 LLM 独立验证,
    # archival 归档研究日志. 三槽避免"模型自己生成自己验证"的确认偏差.
    "verification",
    "archival",
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
        # verification 优先选独立 LLM, 没有就退回 reasoning 模型
        "verification": ["verification", "reasoning", "science", "default"],
        # archival 优先选便宜模型
        "archival": ["archival", "cheap", "summarize", "default"],
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
        thinking: ThinkingIntensity | dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> RegisteredModel:
        """Register a model by provider descriptor."""
        model = create_langchain_model(
            provider=provider,
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            thinking=thinking,
            max_tokens=max_tokens,
        )
        return self.register(
            name=name,
            model=model,
            tags=tags,
            cost_input=cost_input,
            cost_output=cost_output,
            priority=priority,
        )

    def select(
        self, task: TaskT | str | None = None, prefer_cheap: bool = False
    ) -> Any:
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

    def select_candidates(
        self, task: TaskT | str | None = None, top_n: int = 3
    ) -> list[Any]:
        """返回按优先级排序的候选模型列表, 供调用方做 fallback retry."""
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
            return []

        candidates.sort(key=lambda m: (-m.priority, m.cost_input + m.cost_output))
        return [m.model for m in candidates[:top_n]]

    async def select_with_fallback(
        self,
        task: TaskT | str | None,
        try_fn: Callable[[Any], Awaitable[Any]],
        top_n: int = 3,
    ) -> tuple[Any, Exception | None]:
        """按优先级依次 try, 返回第一个成功的结果.

        try_fn 接收模型实例, 返回调用结果. 全部失败时返回 (None, last_error).
        """
        models = self.select_candidates(task, top_n=top_n)
        last_err: Exception | None = None
        for model in models:
            try:
                result = await try_fn(model)
                return result, None
            except Exception as e:
                last_err = e
                logger.warning("model fallback: %s failed: %s", type(model).__name__, e)
        return None, last_err

    def select_verification(self) -> Any:
        """选验证用 LLM. 优先 verification 标签的独立模型,
        没注册就退回 reasoning/science, 最后退回 default.
        这样默认情况下 verification = main, 但用户只要注册一个
        带 verification 标签的模型就会自动启用独立验证."""
        return self.select("verification")

    def select_archival(self) -> Any:
        """选归档用 LLM. 优先 archival/cheap 模型, 降低归档成本."""
        return self.select("archival")

    def has_dedicated_verification(self) -> bool:
        """是否注册了独立的 verification 模型 (标签含 'verification')."""
        return any(
            "verification" in m.tags for m in self._models.values()
        )

    def list_models(self) -> list[str]:
        """Return registered model names."""
        return list(self._models.keys())

    @classmethod
    def from_env(cls) -> ModelRouter:
        """Build a router from environment variables.

        Example:
            HUGINN_MODEL_DEFAULT=openai:gpt-4o
            HUGINN_MODEL_CHEAP=openai:gpt-4o-mini
            HUGINN_MODEL_LOCAL=ollama:qwen2.5:14b
            # Moonshine 三槽: 注册独立验证/归档模型
            HUGINN_MODEL_VERIFICATION=deepseek:deepseek-chat
            HUGINN_MODEL_ARCHIVAL=openai:gpt-4o-mini
        """
        router = cls()
        prefix = "HUGINN_MODEL_"
        failed_providers: list[str] = []
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
            except Exception as e:
                failed_providers.append(f"{name}={provider}:{model_name} ({e})")
                logger.warning("model provider %s skipped: %s", name, e)
                continue
        # router 空时尝试 fallback 到本地 ollama
        if not router._models and failed_providers:
            logger.warning(
                "all model providers failed (%s), trying ollama fallback",
                failed_providers,
            )
            try:
                model = create_langchain_model(provider="ollama", model_name="qwen2.5:7b")
                router.register("default", model, tags={"default"})
            except Exception:
                pass  # ollama 也没装就只能让 select() 报 RuntimeError
        return router
