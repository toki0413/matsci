"""Agent factory — create HuginnAgent instances from configured profiles.

Each profile picks a model alias (or `provider/model`), a persona, and an
optional tool allowlist. The factory reuses the global ModelRegistry so
provider/model instances are cached.
"""

from __future__ import annotations

from typing import Any

from huginn.agent import HuginnAgent
from huginn.config import HuginnConfig, AgentProfileConfig, ThinkingIntensity
from huginn.models.registry import ModelRegistry
from huginn.personas import PersonaManager
from huginn.project_context import load_project_context


class AgentFactory:
    """Factory for creating configured agent instances."""

    def __init__(
        self,
        config: HuginnConfig,
        model_registry: ModelRegistry | None = None,
        memory_manager: Any | None = None,
    ):
        self.config = config
        self.model_registry = model_registry or ModelRegistry.from_config(config)
        self.memory_manager = memory_manager
        self.persona_manager = PersonaManager()
        self._profiles: dict[str, AgentProfileConfig] = {
            a.id: a for a in config.agents if a.enabled
        }

    def get_profile(self, profile_id: str) -> AgentProfileConfig | None:
        return self._profiles.get(profile_id)

    def list_profiles(self) -> list[AgentProfileConfig]:
        return list(self._profiles.values())

    def create(
        self,
        profile_id: str,
        thread_id: str | None = None,
        system_prompt_override: str | None = None,
        memory_manager: Any | None = None,
        thinking: ThinkingIntensity | dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> HuginnAgent:
        """Create a HuginnAgent for the given profile.

        ``thinking`` and ``max_tokens`` override the configured model/agent
        defaults for this request.
        """
        profile = self._profiles.get(profile_id)
        if profile is None:
            raise ValueError(f"Agent profile '{profile_id}' not found or disabled")

        model_alias = profile.model_alias or self.model_registry.default_alias()
        if not model_alias:
            raise ValueError(f"Profile '{profile_id}' has no model_alias and no default model is configured")

        effective_thinking = thinking if thinking is not None else profile.thinking
        model = self.model_registry.resolve(
            model_alias,
            thinking=effective_thinking,
            max_tokens=max_tokens,
        )

        begin_dialogs: list[tuple[str, str]] = []
        if system_prompt_override:
            prompt = system_prompt_override
        else:
            persona = self.persona_manager.get(profile.persona)
            prompt = persona.system_prompt
            begin_dialogs = [
                (d.get("role", "user"), d.get("content", ""))
                for d in persona.begin_dialogs
            ]
            # Inject project context if available
            try:
                ctx = load_project_context(self.config.workspace)
                if ctx.strip():
                    prompt = f"{prompt}\n\n# Project Context\n\n{ctx}"
            except Exception:
                pass

        agent = HuginnAgent(
            model=model,
            system_prompt=prompt,
            begin_dialogs=begin_dialogs,
            memory_manager=memory_manager if memory_manager is not None else self.memory_manager,
            profile_id=profile_id,
            thread_id=thread_id,
            tool_filter=profile.tools if profile.tools else None,
            agent_factory=self,
            privacy_redact_secrets=self.config.privacy_redact_secrets,
            privacy_block_on_secrets=self.config.privacy_block_on_secrets,
            max_tool_output_tokens=self.config.max_tool_output_tokens,
            context_budget_tokens=self.config.context_budget_tokens,
        )
        agent.register_tools_from_registry()
        return agent

    def create_lead(
        self,
        thread_id: str | None = None,
        thinking: ThinkingIntensity | dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> HuginnAgent:
        """Convenience: create the lead/default agent."""
        kwargs: dict[str, Any] = {}
        if thinking is not None:
            kwargs["thinking"] = thinking
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        for preferred in ("lead", "default"):
            if preferred in self._profiles:
                return self.create(preferred, thread_id=thread_id, **kwargs)
        # Fall back to first configured profile
        if self._profiles:
            return self.create(next(iter(self._profiles)), thread_id=thread_id, **kwargs)
        raise ValueError("No enabled agent profiles found")
