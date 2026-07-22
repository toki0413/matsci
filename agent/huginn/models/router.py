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
    # P3 (chaoxu 启发): model family 标识 (如 "openai"/"anthropic"/"deepseek"),
    # 用于 cross-family audit — verification 模型应来自不同 family, 避免
    # "模型自己验证自己"的确认偏差. 空串 = 未指定 (向后兼容).
    family: str = ""


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
        family: str = "",
    ) -> RegisteredModel:
        """Register an existing LangChain model instance."""
        entry = RegisteredModel(
            name=name,
            model=model,
            tags=tags or {"default"},
            cost_input=cost_input,
            cost_output=cost_output,
            priority=priority,
            family=family,
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
        family: str | None = None,
    ) -> RegisteredModel:
        """Register a model by provider descriptor.

        P3: family 默认从 provider 推断 (provider 就是 family 的天然来源),
        显式传 family 则覆盖. 这样调用方零改动即可获得 family 标签.
        """
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
            family=family if family is not None else str(provider),
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

    def select_verification(self, different_family_from: str | None = None) -> Any:
        """选验证用 LLM. 优先 verification 标签的独立模型,
        没注册就退回 reasoning/science, 最后退回 default.
        这样默认情况下 verification = main, 但用户只要注册一个
        带 verification 标签的模型就会自动启用独立验证.

        P3 (chaoxu 启发): different_family_from 传入主模型 family 时,
        优先选不同 family 的 verification 模型 (cross-family audit, 避免
        "模型自己验证自己"的确认偏差). 没有不同 family 的就退回原逻辑 (不阻塞,
        只降级 — family 多样性是 advisory, 不该因找不到异族模型就不验证).

        ponytail: 复用 _TASK_TAGS 的 tag 优先级 + family 过滤, 不改 select() 返回值.
        """
        if different_family_from:
            preferred_tags = self._TASK_TAGS.get("verification", ["verification"])
            for tag in preferred_tags:
                matches = [
                    m for m in self._models.values()
                    if tag in m.tags and m.family != different_family_from
                ]
                if matches:
                    matches.sort(key=lambda m: (-m.priority, m.cost_input + m.cost_output))
                    return matches[0].model
            # 没有不同 family 的, 退回原逻辑 (advisory, 不阻塞)
        return self.select("verification")

    def get_family(self, name: str) -> str:
        """查模型 family. 没注册返空串.

        P3: 供调用方查主模型 family, 传给 select_verification(different_family_from=...).
        """
        m = self._models.get(name)
        return m.family if m else ""

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


def _selfcheck() -> None:
    """P3 cross-family audit selfcheck.

    验证 RegisteredModel.family 字段 + select_verification(different_family_from)
    cross-family 过滤 + get_family 查询. 用 fake model 对象, 不依赖真 LLM.
    """
    r = ModelRouter()
    # fake model 对象 — select 只返实例不调, 任意 object 即可
    _m_openai = object()
    _m_anthropic = object()
    _m_deepseek = object()

    # P3a: register_provider 默认从 provider 推断 family
    # 不调真 create_langchain_model, 直接走 register 路径验证 family 逻辑
    r.register("main_openai", _m_openai, tags={"default", "agent"}, family="openai")
    r.register("verif_anthropic", _m_anthropic, tags={"verification"}, family="anthropic")
    r.register("verif_deepseek", _m_deepseek, tags={"verification", "reasoning"}, family="deepseek")
    assert r.get_family("main_openai") == "openai", "get_family 应返 openai"
    assert r.get_family("verif_anthropic") == "anthropic"
    assert r.get_family("nonexistent") == "", "未注册模型应返空串"
    print("P3a. family 字段注册 + get_family 查询 OK")

    # P3b: select_verification(different_family_from) 优先选不同 family
    # 主模型 family=openai, 应选 anthropic 或 deepseek (都 != openai)
    _v = r.select_verification(different_family_from="openai")
    assert _v is not _m_openai, "cross-family audit: 不应选同 family 的 main 模型"
    assert _v in (_m_anthropic, _m_deepseek), "应选不同 family 的 verification 模型"
    print("P3b. select_verification(different_family_from) cross-family 过滤 OK")

    # P3c: different_family_from 匹配所有 verification 时退回原逻辑 (advisory 不阻塞)
    # 主模型 family=anthropic, 但 verification 模型有 anthropic + deepseek,
    # 应选 deepseek (不同 family). 再试 family=deepseek, 应选 anthropic.
    _v2 = r.select_verification(different_family_from="deepseek")
    assert _v2 is _m_anthropic, f"应选 anthropic (非 deepseek), got {_v2}"
    print("P3c. cross-family 优先选异族 (deepseek 主 → anthropic 验) OK")

    # P3d: 没有不同 family 的 verification 时退回原逻辑 (不阻塞)
    # 只有 openai family 的 verification 模型, 主模型也是 openai → 退回原 select
    r2 = ModelRouter()
    r2.register("main", object(), tags={"default", "agent"}, family="openai")
    r2.register("verif", object(), tags={"verification"}, family="openai")
    _v3 = r2.select_verification(different_family_from="openai")
    # 退回原逻辑: select("verification") → 找 verification tag → 返 verif 模型
    # 同 family 也能用, advisory 不阻塞
    assert _v3 is not None, "无异族时退回原逻辑, 不应报错"
    print("P3d. 无异族 verification 退回原逻辑 (advisory 不阻塞) OK")

    # P3e: different_family_from=None 向后兼容 (原 select_verification 行为)
    _v4 = r.select_verification()
    # 原逻辑: verification tag 优先, priority 排序, 第一个匹配
    assert _v4 in (_m_anthropic, _m_deepseek), "None 时走原逻辑选 verification 模型"
    print("P3e. different_family_from=None 向后兼容 OK")

    print("ModelRouter selfcheck OK (P3a-P3e cross-family audit)")


if __name__ == "__main__":
    _selfcheck()
