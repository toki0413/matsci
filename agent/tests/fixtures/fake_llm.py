"""Unified fake LLM for tests — replaces 12+ incompatible mock classes.

Supports three usage patterns:

1. **Scripted responses** (agent flow / E2E tests):
   Feed a list of AIMessages, the model plays them back in order.

   .. code-block:: python

       model = FakeLLM(responses=[AIMessage(content="hello"), ...])
       agent = HuginnAgent(model=model, ...)

2. **Callable mode** (memory / summarizer / router tests):
   Pass a function that receives the prompt and returns a string/AIMessage.

   .. code-block:: python

       model = FakeLLM(func=lambda prompt: f"Summary of: {prompt[:50]}")

3. **Router stub** (config / router tests that only need a name):
   Works as a bare object with a ``name`` attribute, no LLM call needed.

   .. code-block:: python

       model = FakeLLM(name="cheap")
       router.register("cheap", model, tags={"summarize"})

Usage metadata (input/output tokens) is reported on every response so
the rate limiter and cost tracking code paths are exercised.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Union

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult


# Type for the callable mode: receives the formatted prompt, returns
# either a plain string or an AIMessage.
PromptFn = Callable[[str], Union[str, AIMessage]]


class FakeLLM(BaseChatModel):
    """Deterministic chat model for tests.

    Drop-in replacement for the various ``_FakeModel`` / ``MockLLM`` /
    ``FakeToolCallingModel`` classes scattered across the test suite.

    Three modes (checked in order):
      1. ``responses`` — scripted AIMessage list, played back round-robin.
      2. ``func``      — callable that receives the prompt text.
      3. neither       — returns a generic echo response (router stub mode).
    """

    responses: list[AIMessage] = []
    model_name: str = "fake-llm"
    # Alias for router tests that access .name — pydantic v2 doesn't play
    # well with @property on models, so we use a computed field alias.
    name: str = "fake-llm"
    _index: int = 0

    def __init__(
        self,
        responses: list[AIMessage] | None = None,
        func: PromptFn | None = None,
        name: str | None = None,
        usage: dict[str, int] | None = None,
        **kwargs: Any,
    ) -> None:
        # Set name before super().__init__ so pydantic picks it up
        if name is not None:
            kwargs["name"] = name
            kwargs["model_name"] = name
        super().__init__(responses=responses or [], **kwargs)
        # Store non-pydantic state via object.__setattr__ to bypass
        # pydantic's field validation — these are runtime-only attributes.
        object.__setattr__(self, "_func", func)
        object.__setattr__(self, "calls", [])
        object.__setattr__(self, "_lock", threading.Lock())
        object.__setattr__(self, "_index", 0)
        object.__setattr__(self, "_usage_override", usage)

    # ── public helpers ──────────────────────────────────────

    @property
    def call_count(self) -> int:
        with object.__getattribute__(self, "_lock"):
            return len(object.__getattribute__(self, "calls"))

    @property
    def last_prompt(self) -> str:
        with object.__getattribute__(self, "_lock"):
            calls = object.__getattribute__(self, "calls")
            if not calls:
                return ""
            msgs = calls[-1]
            if not msgs:
                return ""
            last = msgs[-1]
            return getattr(last, "content", str(last))

    def reset(self) -> None:
        """Clear call history and reset the scripted-response index."""
        with object.__getattribute__(self, "_lock"):
            object.__setattr__(self, "calls", [])
            object.__setattr__(self, "_index", 0)

    # ── BaseChatModel implementation ─────────────────────────

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        calls = object.__getattribute__(self, "calls")
        lock = object.__getattribute__(self, "_lock")
        with lock:
            calls.append(messages)
        response = self._next_response(messages)
        return ChatResult(generations=[ChatGeneration(message=response)])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return self._generate(messages, stop, run_manager, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "fake-llm"

    def bind_tools(self, tools, **kwargs):
        # Return self so the agent executor sees a model that accepts
        # tool definitions but still responds from the scripted list.
        return self

    # ── response selection ──────────────────────────────────

    def _next_response(self, messages: list[Any]) -> AIMessage:
        lock = object.__getattribute__(self, "_lock")
        with lock:
            idx_val = object.__getattribute__(self, "_index")
            responses = self.responses
            func = object.__getattribute__(self, "_func")
            object.__setattr__(self, "_index", idx_val + 1)

        # Mode 1: scripted responses
        if responses:
            idx = idx_val % len(responses)
            msg = responses[idx]
            return self._with_usage(msg)

        # Mode 2: callable
        if func is not None:
            prompt = self._extract_prompt(messages)
            result = func(prompt)
            if isinstance(result, AIMessage):
                return self._with_usage(result)
            return self._with_usage(AIMessage(content=str(result)))

        # Mode 3: router stub — echo the last user message
        prompt = self._extract_prompt(messages)
        return self._with_usage(AIMessage(content=f"[fake-llm] {prompt[:200]}"))

    def _extract_prompt(self, messages: list[Any]) -> str:
        """Pull the last human/user message content from the list."""
        for msg in reversed(messages):
            content = getattr(msg, "content", None)
            role = getattr(msg, "type", "") or getattr(msg, "role", "")
            if content and role in ("human", "user", ""):
                return content if isinstance(content, str) else str(content)
        # Fall back to the last message regardless of role
        if messages:
            return getattr(messages[-1], "content", str(messages[-1]))
        return ""

    def _with_usage(self, msg: AIMessage) -> AIMessage:
        """Attach usage_metadata so rate limiter / cost tracking is exercised."""
        if msg.usage_metadata is not None:
            return msg
        override = object.__getattribute__(self, "_usage_override")
        if override is not None:
            in_tok = override.get("input_tokens", 10)
            out_tok = override.get("output_tokens", 10)
        else:
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            in_tok = max(len(content) // 4, 1)
            out_tok = max(len(content) // 4, 1)
        # Pydantic v2 model — use model_construct to set without validation
        return msg.model_copy(update={
            "usage_metadata": {
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "total_tokens": in_tok + out_tok,
            }
        })

    # ── repr for debugging ─────────────────────────────────

    def __repr__(self) -> str:
        mode = "scripted" if self.responses else ("callable" if self._func else "stub")
        return f"FakeLLM(name={self.model_name!r}, mode={mode}, calls={self.call_count})"


# ── Factory helpers ─────────────────────────────────────────


def make_scripted_llm(responses: list[AIMessage], **kwargs: Any) -> FakeLLM:
    """Quick constructor for the scripted-response mode."""
    return FakeLLM(responses=responses, **kwargs)


def make_callable_llm(func: PromptFn, name: str = "callable-llm", **kwargs: Any) -> FakeLLM:
    """Quick constructor for the callable mode."""
    return FakeLLM(func=func, name=name, **kwargs)


def make_router_stub(name: str = "stub") -> FakeLLM:
    """Quick constructor for router/config tests that only need a name."""
    return FakeLLM(name=name)
