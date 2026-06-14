"""Configuration management for Huginn.

Supports environment variables, .env files, and config files.
"""

from __future__ import annotations

import json
import os
import pathlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from huginn.crypto import CryptoVault, EncryptedConfig, KeyManager


ThinkingIntensity = Literal["low", "medium", "high"]


@dataclass
class ModelConfig:
    """A single LLM provider/model entry in the model pool."""
    alias: str  # e.g. "gpt4o", "claude-sonnet"
    provider: Literal[
        "anthropic", "openai", "ollama", "deepseek",
        "google-genai", "openrouter", "nvidia", "vllm", "local", "default"
    ]
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.7
    enabled: bool = True
    # Reasoning / extended-thinking intensity. Provider-specific mapping happens
    # in the model registry. ``dict`` values are passed through verbatim.
    thinking: ThinkingIntensity | dict[str, Any] | None = None
    max_tokens: int | None = None  # must be > thinking budget for Anthropic


@dataclass
class AgentProfileConfig:
    """A reusable agent profile that maps to a model alias + persona + tools."""
    id: str  # e.g. "lead", "coder", "reviewer"
    name: str = ""
    model_alias: str = ""
    persona: str = "default"
    tools: list[str] = field(default_factory=list)
    enabled: bool = True
    max_steps: int = 10
    # Optional per-agent thinking override. Falls back to the model's setting.
    thinking: ThinkingIntensity | dict[str, Any] | None = None


@dataclass
class HuginnConfig:
    """Huginn configuration."""

    # Legacy single-model settings (kept for backward compatibility)
    provider: Literal[
        "anthropic", "openai", "ollama", "deepseek",
        "google-genai", "openrouter", "nvidia", "vllm", "local", "default"
    ] = "default"
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None

    # Multi-LLM / multi-agent settings
    models: list[ModelConfig] = field(default_factory=list)
    agents: list[AgentProfileConfig] = field(default_factory=list)
    team_mode_enabled: bool = False
    max_concurrent_subagents: int = 3

    # Ollama specific
    ollama_host: str = "http://localhost:11434"

    # Computational tools
    vasp_executable: str | None = None
    lammps_executable: str | None = None

    # HPC settings
    hpc_scheduler: Literal["slurm", "pbs", "local"] = "local"
    hpc_host: str | None = None
    hpc_username: str | None = None

    # MCP server paths
    abaqus_mcp_server: str | None = None

    # Workspace
    workspace: str = "."

    # Agent behavior
    auto_approve: bool = False
    enable_exploration: bool = True
    max_parallel_branches: int = 5
    persona: str = "default"
    rag_enabled: bool = False

    # Encryption settings
    encryption_enabled: bool = False
    encryption_password: str | None = None
    encryption_key_file: str | None = None
    encrypt_rag_documents: bool = True
    encrypt_rag_metadata: bool = True
    encrypt_config: bool = False

    # Privacy / data leakage controls
    privacy_redact_secrets: bool = True
    privacy_block_on_secrets: bool = False

    # Local-only / no-cloud mode
    local_only_mode: bool = False

    # Context/output budgets
    max_tool_output_tokens: int = 25000
    context_budget_tokens: int = 0

    # Prompt caching
    prompt_cache_control: bool = True

    # Checkpointer / persistence
    checkpointer_path: str | None = None

    # Telemetry
    telemetry_enabled: bool = True

    # Memory maintenance
    memory_decay_enabled: bool = False
    memory_decay_interval_turns: int = 0
    memory_decay_prune_threshold: float = 0.15

    # Tool output compression
    tool_compression_max_tokens: int = 8000

    # Default reasoning intensity for all models (can be overridden per model/agent).
    thinking: ThinkingIntensity | dict[str, Any] | None = None
    max_tokens: int | None = None

    # Pet customization
    pet_name: str = "Muninn"
    pet_personality: Literal["cheerful", "nerdy", "calm", "sassy"] = "cheerful"

    def build_agent_kwargs(self, profile_id: str = "lead") -> dict[str, Any]:
        """Build keyword arguments for constructing a HuginnAgent.

        Resolves the requested agent profile, builds the model router when a
        model pool is configured, and applies all agent-scoped settings.
        """
        from huginn.checkpointer import create_checkpointer, create_in_memory_checkpointer
        from huginn.models.router import ModelRouter

        profile = self.get_profile(profile_id)

        # Effective thinking for this profile: profile > model > global.
        effective_thinking = None
        if profile and profile.thinking is not None:
            effective_thinking = profile.thinking

        # Build model/router
        model = None
        model_router = None
        if self.models:
            model_router = ModelRouter()
            for m in self.models:
                if not m.enabled:
                    continue
                thinking = m.thinking if m.thinking is not None else effective_thinking if effective_thinking is not None else self.thinking
                try:
                    model_router.register_provider(
                        name=m.alias,
                        provider=m.provider,
                        model_name=m.model,
                        api_key=self.resolve_key(m.api_key),
                        base_url=m.base_url,
                        tags={m.alias, profile_id},
                        temperature=m.temperature,
                        thinking=thinking,
                        max_tokens=m.max_tokens if m.max_tokens is not None else self.max_tokens,
                    )
                except Exception:
                    # Skip models that cannot be initialized (missing keys, etc.)
                    continue
        else:
            # Legacy single-model path
            from huginn.models.registry import create_langchain_model
            provider = self.provider
            if provider and provider != "default":
                thinking = effective_thinking if effective_thinking is not None else self.thinking
                model = create_langchain_model(
                    provider=provider,
                    model_name=self.model or None,
                    api_key=self.resolved_api_key,
                    base_url=self.base_url,
                    thinking=thinking,
                    max_tokens=self.max_tokens,
                )

        # Checkpointer
        if self.checkpointer_path:
            checkpointer = create_checkpointer(self.checkpointer_path)
        else:
            checkpointer = create_in_memory_checkpointer()

        return {
            "model": model,
            "model_router": model_router,
            "checkpointer": checkpointer,
            "system_prompt": None,  # Loaded from persona by caller if desired
            "enable_exploration": self.enable_exploration,
            "privacy_redact_secrets": self.privacy_redact_secrets,
            "privacy_block_on_secrets": self.privacy_block_on_secrets,
            "max_tool_output_tokens": self.max_tool_output_tokens,
            "context_budget_tokens": self.context_budget_tokens,
            "prompt_cache_control": self.prompt_cache_control,
            "tool_filter": profile.tools if profile else None,
        }

    def get_profile(self, profile_id: str) -> AgentProfileConfig | None:
        """Return the agent profile with the given id, or None."""
        for a in self.agents:
            if a.id == profile_id and a.enabled:
                return a
        return None

    @classmethod
    def from_env(cls) -> HuginnConfig:
        """Load configuration from environment variables.

        API key resolution priority:
        1. HUGINN_API_KEY (generic)
        2. Provider-specific env vars (ANTHROPIC_API_KEY, OPENAI_API_KEY, etc.)
        """
        provider = os.environ.get("HUGINN_PROVIDER", "default").lower().strip()

        # Resolve API key with provider-specific fallback
        api_key = os.environ.get("HUGINN_API_KEY")
        if not api_key:
            provider_key_map = {
                "anthropic": "ANTHROPIC_API_KEY",
                "openai": "OPENAI_API_KEY",
                "deepseek": "DEEPSEEK_API_KEY",
                "google-genai": "GOOGLE_API_KEY",
                "openrouter": "OPENROUTER_API_KEY",
                "nvidia": "NVIDIA_API_KEY",
                "vllm": "OPENAI_API_KEY",
                "local": "OPENAI_API_KEY",
            }
            env_key = provider_key_map.get(provider)
            if env_key:
                api_key = os.environ.get(env_key)

        models = cls._parse_models_env()
        agents = cls._parse_agents_env()
        thinking = cls._parse_thinking_env()
        max_tokens = cls._parse_max_tokens_env()

        # If no model pool but legacy provider is set, synthesize a default model entry.
        if not models and provider != "default":
            models = [ModelConfig(
                alias="default",
                provider=provider,  # type: ignore[arg-type]
                model=os.environ.get("HUGINN_MODEL"),
                api_key=api_key,
                base_url=os.environ.get("HUGINN_BASE_URL"),
                thinking=thinking,
                max_tokens=max_tokens,
            )]

        # If no agent profiles, synthesize a default lead agent pointing at the default/only model.
        if not agents:
            model_alias = models[0].alias if models else "default"
            agents = [AgentProfileConfig(
                id="lead",
                name="Lead",
                model_alias=model_alias,
                persona=os.environ.get("HUGINN_PERSONA", "default").strip(),
                thinking=thinking,
            )]

        return cls(
            provider=provider,
            model=os.environ.get("HUGINN_MODEL"),
            api_key=api_key,
            base_url=os.environ.get("HUGINN_BASE_URL"),
            models=models,
            agents=agents,
            team_mode_enabled=os.environ.get("HUGINN_TEAM_MODE", "").lower() == "true",
            max_concurrent_subagents=int(os.environ.get("HUGINN_MAX_CONCURRENT_SUBAGENTS", "3")),
            ollama_host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
            vasp_executable=os.environ.get("VASP_EXECUTABLE"),
            lammps_executable=os.environ.get("LAMMPS_EXECUTABLE"),
            hpc_scheduler=os.environ.get("HPC_SCHEDULER", "local"),
            hpc_host=os.environ.get("HPC_HOST"),
            hpc_username=os.environ.get("HPC_USERNAME"),
            abaqus_mcp_server=os.environ.get("ABAQUS_MCP_SERVER_PATH"),
            workspace=os.environ.get("HUGINN_WORKSPACE", "."),
            auto_approve=os.environ.get("HUGINN_AUTO_APPROVE", "").lower() == "true",
            enable_exploration=os.environ.get("HUGINN_ENABLE_EXPLORATION", "true").lower() == "true",
            max_parallel_branches=int(os.environ.get("HUGINN_MAX_BRANCHES", "5")),
            encryption_enabled=os.environ.get("HUGINN_ENCRYPTION_ENABLED", "").lower() == "true",
            encryption_password=os.environ.get("HUGINN_ENCRYPTION_PASSWORD") or None,
            encryption_key_file=os.environ.get("HUGINN_ENCRYPTION_KEY_FILE") or None,
            encrypt_rag_documents=os.environ.get("HUGINN_ENCRYPT_RAG_DOCS", "true").lower() == "true",
            encrypt_rag_metadata=os.environ.get("HUGINN_ENCRYPT_RAG_META", "true").lower() == "true",
            encrypt_config=os.environ.get("HUGINN_ENCRYPT_CONFIG", "").lower() == "true",
            persona=os.environ.get("HUGINN_PERSONA", "default").strip(),
            rag_enabled=os.environ.get("HUGINN_RAG_ENABLED", "").lower() == "true",
            local_only_mode=os.environ.get("HUGINN_LOCAL_ONLY", "").lower() == "true",
            privacy_redact_secrets=os.environ.get("HUGINN_PRIVACY_REDACT_SECRETS", "true").lower() != "false",
            privacy_block_on_secrets=os.environ.get("HUGINN_PRIVACY_BLOCK_ON_SECRETS", "").lower() == "true",
            max_tool_output_tokens=int(os.environ.get("HUGINN_MAX_TOOL_OUTPUT_TOKENS", "25000")),
            context_budget_tokens=int(os.environ.get("HUGINN_CONTEXT_BUDGET_TOKENS", "0")),
            prompt_cache_control=os.environ.get("HUGINN_PROMPT_CACHE_CONTROL", "true").lower() != "false",
            checkpointer_path=os.environ.get("HUGINN_CHECKPOINTER_PATH") or None,
            telemetry_enabled=os.environ.get("HUGINN_TELEMETRY_ENABLED", "true").lower() != "false",
            memory_decay_enabled=os.environ.get("HUGINN_MEMORY_DECAY_ENABLED", "").lower() == "true",
            memory_decay_interval_turns=int(os.environ.get("HUGINN_MEMORY_DECAY_INTERVAL_TURNS", "0")),
            memory_decay_prune_threshold=float(os.environ.get("HUGINN_MEMORY_DECAY_PRUNE_THRESHOLD", "0.15")),
            tool_compression_max_tokens=int(os.environ.get("HUGINN_TOOL_COMPRESSION_MAX_TOKENS", "8000")),
            thinking=thinking,
            max_tokens=max_tokens,
            pet_name=os.environ.get("HUGINN_PET_NAME", "Muninn").strip() or "Muninn",
            pet_personality=os.environ.get("HUGINN_PET_PERSONALITY", "cheerful").strip().lower() or "cheerful",  # type: ignore[arg-type]
        )

    @staticmethod
    def _parse_thinking_env() -> ThinkingIntensity | dict[str, Any] | None:
        """Parse HUGINN_THINKING: a JSON object or one of low/medium/high."""
        raw = os.environ.get("HUGINN_THINKING", "").strip()
        if not raw:
            return None
        if raw in ("low", "medium", "high"):
            return raw  # type: ignore[return-value]
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
        return None

    @staticmethod
    def _parse_max_tokens_env() -> int | None:
        raw = os.environ.get("HUGINN_MAX_TOKENS", "").strip()
        if not raw:
            return None
        try:
            return int(raw)
        except ValueError:
            return None

    @staticmethod
    def _parse_models_env() -> list[ModelConfig]:
        raw = os.environ.get("HUGINN_MODELS", "")
        if not raw:
            return []
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                return [ModelConfig(**item) for item in data]
        except Exception:
            pass
        return []

    @staticmethod
    def _parse_agents_env() -> list[AgentProfileConfig]:
        raw = os.environ.get("HUGINN_AGENTS", "")
        if not raw:
            return []
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                return [AgentProfileConfig(**item) for item in data]
        except Exception:
            pass
        return []
    

    @staticmethod
    def resolve_key(raw: str | None) -> str | None:
        """Resolve a credential string supporting env: and keyring: prefixes.

        Supported formats:
        - "plain-secret"               -> returned as-is
        - "env:OPENAI_API_KEY"         -> read from environment variable
        - "keyring:huginn:openai"      -> read from OS keyring (requires 'keyring' package)
        - None                           -> None
        """
        if raw is None:
            return None
        raw = raw.strip()
        if raw.startswith("env:"):
            var_name = raw[4:]
            return os.environ.get(var_name) or None
        if raw.startswith("keyring:"):
            try:
                import keyring
            except ImportError:
                raise ImportError("pip install keyring")
            parts = raw[8:].split(":", 1)
            service = parts[0] if parts else "huginn"
            username = parts[1] if len(parts) > 1 else "default"
            return keyring.get_password(service, username)
        return raw

    @property
    def resolved_api_key(self) -> str | None:
        """Return the resolved API key (supports env:/keyring: prefixes)."""
        return self.resolve_key(self.api_key)

    def to_dict(self, mask_key: bool = True) -> dict[str, Any]:
        def mask(k: str | None) -> str | None:
            return "***" if mask_key and k else k

        return {
            "provider": self.provider,
            "model": self.model,
            "api_key": mask(self.api_key),
            "base_url": self.base_url,
            "models": [
                {
                    "alias": m.alias,
                    "provider": m.provider,
                    "model": m.model,
                    "api_key": mask(m.api_key),
                    "base_url": m.base_url,
                    "temperature": m.temperature,
                    "enabled": m.enabled,
                    "thinking": m.thinking,
                    "max_tokens": m.max_tokens,
                }
                for m in self.models
            ],
            "agents": [
                {
                    "id": a.id,
                    "name": a.name or a.id,
                    "model_alias": a.model_alias,
                    "persona": a.persona,
                    "tools": a.tools,
                    "enabled": a.enabled,
                    "max_steps": a.max_steps,
                    "thinking": a.thinking,
                }
                for a in self.agents
            ],
            "team_mode_enabled": self.team_mode_enabled,
            "max_concurrent_subagents": self.max_concurrent_subagents,
            "ollama_host": self.ollama_host,
            "vasp_executable": self.vasp_executable,
            "lammps_executable": self.lammps_executable,
            "hpc_scheduler": self.hpc_scheduler,
            "workspace": self.workspace,
            "auto_approve": self.auto_approve,
            "enable_exploration": self.enable_exploration,
            "max_parallel_branches": self.max_parallel_branches,
            "encryption_enabled": self.encryption_enabled,
            "encryption_password": mask(self.encryption_password),
            "encryption_key_file": self.encryption_key_file,
            "encrypt_rag_documents": self.encrypt_rag_documents,
            "encrypt_rag_metadata": self.encrypt_rag_metadata,
            "encrypt_config": self.encrypt_config,
            "persona": self.persona,
            "rag_enabled": self.rag_enabled,
            "privacy_redact_secrets": self.privacy_redact_secrets,
            "privacy_block_on_secrets": self.privacy_block_on_secrets,
            "local_only_mode": self.local_only_mode,
            "max_tool_output_tokens": self.max_tool_output_tokens,
            "context_budget_tokens": self.context_budget_tokens,
            "prompt_cache_control": self.prompt_cache_control,
            "checkpointer_path": self.checkpointer_path,
            "telemetry_enabled": self.telemetry_enabled,
            "memory_decay_enabled": self.memory_decay_enabled,
            "memory_decay_interval_turns": self.memory_decay_interval_turns,
            "memory_decay_prune_threshold": self.memory_decay_prune_threshold,
            "tool_compression_max_tokens": self.tool_compression_max_tokens,
            "thinking": self.thinking,
            "max_tokens": self.max_tokens,
            "pet_name": self.pet_name,
            "pet_personality": self.pet_personality,
        }

    @classmethod
    def from_dict(cls, data: dict) -> HuginnConfig:
        """Restore config from a plain dict."""
        models_raw = data.get("models")
        agents_raw = data.get("agents")
        kwargs = {k: v for k, v in data.items() if k not in ("models", "agents")}

        if models_raw is not None:
            kwargs["models"] = [ModelConfig(**m) for m in models_raw if isinstance(m, dict)]
        if agents_raw is not None:
            kwargs["agents"] = [AgentProfileConfig(**a) for a in agents_raw if isinstance(a, dict)]

        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in kwargs.items() if k in known}
        return cls(**filtered)

    def save(self, path: str | pathlib.Path, format: Literal["toml", "json"] = "toml") -> None:
        """Persist configuration to a file.

        When ``encrypt_config`` is enabled (or the path ends in ``.enc``),
        the file is encrypted with the configured password/key file.
        Otherwise API keys are written in plain text; ensure the file has
        restricted permissions.
        """
        target = pathlib.Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict(mask_key=False)

        if self.encrypt_config or str(target).endswith(".enc"):
            vault = self._get_vault()
            ec = EncryptedConfig(config_path=target, vault=vault)
            ec.save(data)
            return

        if format == "toml":
            try:
                import toml
            except ImportError:
                raise ImportError("pip install toml")
            target.write_text(toml.dumps(data), encoding="utf-8")
        else:
            import json
            target.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _get_vault(self) -> CryptoVault:
        """Return an unlocked CryptoVault for config encryption."""
        password = self.encryption_password or os.environ.get("HUGINN_ENCRYPTION_PASSWORD")
        if not password:
            raise RuntimeError(
                "Config encryption requires encryption_password or HUGINN_ENCRYPTION_PASSWORD."
            )

        if self.encryption_key_file:
            km = KeyManager(self.encryption_key_file)
            path = pathlib.Path(self.encryption_key_file)
            if not path.exists():
                km.create_key_file(password)
            else:
                km.load_key_file(password)
            return km.get_vault()

        return CryptoVault(master_password=password)

    @classmethod
    def load(
        cls,
        path: str | pathlib.Path,
        format: Literal["toml", "json"] | None = None,
        password: str | None = None,
    ) -> HuginnConfig:
        """Load configuration from a file.

        If the path ends in ``.enc`` it is decrypted using ``password`` or
        ``HUGINN_ENCRYPTION_PASSWORD``.
        """
        target = pathlib.Path(path)
        if not target.exists():
            raise FileNotFoundError(f"Config file not found: {target}")

        if str(target).endswith(".enc"):
            vault_password = password or os.environ.get("HUGINN_ENCRYPTION_PASSWORD")
            if not vault_password:
                raise RuntimeError(
                    "Encrypted config requires password or HUGINN_ENCRYPTION_PASSWORD."
                )
            vault = CryptoVault(master_password=vault_password)
            data = EncryptedConfig(config_path=target, vault=vault).load()
            return cls.from_dict(data)

        fmt = format or ("toml" if target.suffix in (".toml", ".tml") else "json")
        text = target.read_text(encoding="utf-8")
        if fmt == "toml":
            try:
                import toml
            except ImportError:
                raise ImportError("pip install toml")
            data = toml.loads(text)
        else:
            import json
            data = json.loads(text)
        return cls.from_dict(data)


@dataclass
class CoderSettings:
    """Settings for autonomous coder mode."""

    max_iterations: int = 50
    done_marker: str = "[DONE]"

    @classmethod
    def from_env(cls) -> "CoderSettings":
        """Load coder settings from environment variables."""
        return cls(
            max_iterations=int(os.environ.get("HUGINN_CODER_MAX_ITER", "50")),
            done_marker=os.environ.get("HUGINN_CODER_DONE_MARKER", "[DONE]"),
        )


@dataclass
class Settings:
    """Top-level application settings."""

    config: HuginnConfig = field(default_factory=HuginnConfig.from_env)
    coder: CoderSettings = field(default_factory=CoderSettings.from_env)


def get_settings() -> Settings:
    """Load application settings from environment."""
    return Settings()
