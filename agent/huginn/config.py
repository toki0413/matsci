"""Configuration management for Huginn.

Supports environment variables, .env files, and config files.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import pathlib
import shutil
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from huginn.crypto import CryptoVault, EncryptedConfig, KeyManager

logger = logging.getLogger(__name__)

ThinkingIntensity = Literal["low", "medium", "high"]


def _parse_queue_map(value: str | None) -> dict[str, str]:
    """Parse a JSON or comma-separated queue map from an environment variable."""
    if not value:
        return {}
    try:
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return {str(k): str(v) for k, v in parsed.items()}
    except json.JSONDecodeError:
        logger.debug("suppressed in _parse_queue_map", exc_info=True)
    result: dict[str, str] = {}
    for part in value.split(","):
        if "=" in part:
            key, val = part.split("=", 1)
            result[key.strip()] = val.strip()
    return result


# ---------------------------------------------------------------------------
# 模块级缓存 & 文件锁
# ---------------------------------------------------------------------------

# write-through 缓存：首次从磁盘加载后缓存，后续命中直接返回
_config_cache: HuginnConfig | None = None
_config_cache_path: pathlib.Path | None = None
_config_cache_mtime: float = 0.0  # 上次加载时磁盘文件的 mtime
_config_lock = threading.RLock()

# 备份轮转配置
_MAX_BACKUPS = 5
_BACKUP_COOLDOWN_SEC = 60.0
_last_backup_time: float = 0.0


@contextlib.contextmanager
def _file_lock(path: pathlib.Path):
    """跨平台文件锁，Windows 用 msvcrt，Unix 用 fcntl。

    对同一文件的并发写入做互斥，避免配置损坏。
    """
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        if sys.platform == "win32":
            import msvcrt
            # Windows 下用 msvcrt.locking 做独占锁
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            if sys.platform == "win32":
                import msvcrt
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            logger.debug("suppressed in _file_lock", exc_info=True)
        os.close(fd)
        try:
            lock_path.unlink()
        except OSError:
            logger.debug("suppressed in _file_lock", exc_info=True)


def _would_lose_auth_state(cached: HuginnConfig, fresh: HuginnConfig) -> bool:
    """检测新配置是否会丢失已有的认证状态。

    典型场景：用户之前配了 api_key，结果新配置忘了带，写入后就把 key 丢了。
    返回 True 表示有丢失风险，save() 应该拒绝并保留备份。
    """
    # 缓存有 api_key 但新配置没有
    if cached.api_key and not fresh.api_key:
        return True
    # 缓存有 models 列表但新配置是空的
    if cached.models and not fresh.models:
        return True
    # 逐个检查 models 里的 api_key 是否被清空
    cached_keys = {m.alias: m.api_key for m in cached.models if m.api_key}
    for m in fresh.models:
        if m.alias in cached_keys and not m.api_key:
            return True
    # HPC 密码丢失
    if cached.hpc_password and not fresh.hpc_password:
        return True
    return False


def _backup_before_save(path: pathlib.Path) -> None:
    """在写入前创建时间戳备份，保留最近 N 个，60秒内不重复。"""
    global _last_backup_time

    now = time.time()
    if now - _last_backup_time < _BACKUP_COOLDOWN_SEC:
        return

    if not path.exists():
        return

    # 备份路径: config.toml -> config.toml.bak.1706123456
    ts = int(now)
    bak = path.parent / f"{path.name}.bak.{ts}"
    shutil.copy2(str(path), str(bak))
    _last_backup_time = now

    # 轮转：只留最近的 _MAX_BACKUPS 个备份
    backups = sorted(
        path.parent.glob(f"{path.name}.bak.*"),
        key=lambda p: p.stat().st_mtime,
    )
    for old in backups[:-_MAX_BACKUPS]:
        try:
            old.unlink()
        except OSError:
            logger.debug("suppressed in _backup_before_save", exc_info=True)


def _check_disk_freshness(path: pathlib.Path | None) -> bool:
    """检查磁盘文件是否被其他进程修改过（mtime 变化）。

    返回 True 表示文件比缓存更新，需要重新加载。
    """
    global _config_cache_mtime

    if path is None or not path.exists():
        return False

    try:
        mtime = path.stat().st_mtime
    except OSError:
        return False

    return mtime > _config_cache_mtime


def _atomic_write(
    target: pathlib.Path, data: dict[str, Any], format: str
) -> None:
    """原子写入配置文件.

    流程: 序列化 -> 回读校验 -> 写临时文件 -> os.replace 原子替换.
    os.replace 在 Windows 和 POSIX 上都是原子操作, 不会出现半截文件.
    临时文件用 pid 后缀, 避免多进程撞车.
    """
    # 先序列化 + 回读校验, 拿到 content 字符串再落盘
    if format == "toml":
        try:
            import toml
        except ImportError as err:
            raise ImportError("pip install toml") from err
        content = toml.dumps(data)
        toml.loads(content)  # 回读校验, 防止序列化产物损坏
    else:
        content = json.dumps(data, indent=2)
        json.loads(content)  # 回读校验

    tmp_path = target.with_name(f"{target.name}.tmp.{os.getpid()}")
    try:
        tmp_path.write_text(content, encoding="utf-8")
        # 同目录 rename, 保证在同一文件系统上 -> 原子
        os.replace(str(tmp_path), str(target))
    except Exception:
        # 写入或替换失败, 清理临时文件, 原文件不受影响
        try:
            tmp_path.unlink()
        except OSError:
            logger.debug("suppressed in _atomic_write", exc_info=True)
        raise


@dataclass
class ModelConfig:
    """A single LLM provider/model entry in the model pool."""

    alias: str  # e.g. "gpt4o", "claude-sonnet"
    provider: Literal[
        "anthropic",
        "openai",
        "ollama",
        "deepseek",
        "google-genai",
        "openrouter",
        "nvidia",
        "vllm",
        "local",
        "default",
        "siliconflow",
        "moonshot",
        "zhipu",
        "baichuan",
        "dashscope",
        "qianfan",
        "doubao",
        "hunyuan",
        "openai-compatible",
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
    # Link to a stored LLM credential in CredentialStore.
    # When set, ModelRegistry.get() will use this credential's api_key
    # as fallback when api_key field is empty.
    credential_id: str | None = None


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
    # 自定义 system prompt 覆写 (AstrBot 会话级覆写模式)
    # 设为 None 时使用 persona 加载的 system prompt; 非 None 时直接覆盖
    system_prompt_override: str | None = None


@dataclass
class SecurityConfig:
    """Security, privacy, and encryption settings."""

    api_key: str | None = None
    encryption_enabled: bool = False
    encryption_password: str | None = None
    encryption_key_file: str | None = None
    encrypt_config: bool = False
    encrypt_rag_documents: bool = True
    encrypt_rag_metadata: bool = True
    privacy_redact_secrets: bool = True
    privacy_block_on_secrets: bool = False
    local_only_mode: bool = False


@dataclass
class PersistenceConfig:
    """Persistence settings for checkpoints, memory, and remote jobs."""

    checkpointer_path: str | None = None


@dataclass
class SandboxConfig:
    """Sandbox and container execution settings."""

    container_runtime: Literal[
        "none", "docker", "podman", "apptainer", "singularity"
    ] = "none"
    container_image: str | None = None
    max_tool_output_tokens: int = 25000
    context_budget_tokens: int = 0


@dataclass
class HuginnConfig:
    """Huginn configuration."""

    # 配置版本号 (AstrBot 风格自愈机制)
    # 从 v1 开始, 未来版本升级时用于触发迁移逻辑
    config_version: int = 1

    # Legacy single-model settings (kept for backward compatibility)
    provider: Literal[
        "anthropic",
        "openai",
        "ollama",
        "deepseek",
        "google-genai",
        "openrouter",
        "nvidia",
        "vllm",
        "local",
        "default",
        "siliconflow",
        "moonshot",
        "zhipu",
        "baichuan",
        "dashscope",
        "qianfan",
        "doubao",
        "hunyuan",
        "openai-compatible",
    ] = "default"
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None

    # Multi-LLM / multi-agent settings
    models: list[ModelConfig] = field(default_factory=list)
    agents: list[AgentProfileConfig] = field(default_factory=list)
    team_mode_enabled: bool = False
    max_concurrent_subagents: int = 3
    # Planner/executor 分离: auto_confirm=True 时跳过用户确认, 直接执行 plan.
    # 默认 False 走 human-in-the-loop, 对齐 Claude Code plan-mode UX.
    plan_auto_confirm: bool = False

    # Ollama specific
    ollama_host: str = "http://localhost:11434"

    # Computational tools
    vasp_executable: str | None = None
    lammps_executable: str | None = None

    # Public materials database API keys
    mp_api_key: str | None = None
    oqmd_api_key: str | None = None

    # HPC / remote execution settings
    execution_backend: Literal["local", "remote"] = "local"
    container_runtime: Literal[
        "none", "docker", "podman", "apptainer", "singularity"
    ] = "none"
    container_image: str | None = None
    hpc_scheduler: Literal["slurm", "pbs", "local"] = "local"
    hpc_host: str | None = None
    hpc_username: str | None = None
    hpc_key_path: str | None = None
    hpc_password: str | None = None
    hpc_port: int = 22
    remote_work_dir: str = "~/huginn_jobs"
    hpc_default_queue: str | None = None
    hpc_gpu_queue: str | None = None
    hpc_queue_map: dict[str, str] = field(default_factory=dict)
    hpc_default_walltime: str = "24:00:00"
    hpc_default_nodes: int = 1
    hpc_default_ntasks_per_node: int = 4
    hpc_default_gpus_per_node: int = 0
    hpc_max_retries: int = 3
    hpc_retry_backoff: float = 1.0
    hpc_strict_host_key_checking: bool = True

    # MCP server paths
    abaqus_mcp_server: str | None = None
    # Generic MCP server configurations (name -> {command, args, env, transport})
    mcp_servers: dict[str, dict[str, Any]] = field(default_factory=dict)

    # ToolUniverse MCP integration (curated: only materials-science tools pass
    # the whitelist in mcp_adapter.MATERIAL_SCIENCE_TOOL_WHITELIST).
    # Off by default — user must install `pip install tooluniverse` and set
    # HUGINN_TOOLUNIVERSE_ENABLED=1 to enable. Avoids startup errors when the
    # package is not installed.
    tooluniverse_enabled: bool = False
    tooluniverse_mcp_command: str = "python"
    tooluniverse_mcp_args: str = "-m tooluniverse.smcp_server"

    # Workspace
    workspace: str = "."

    # Agent behavior
    auto_approve: bool = False
    enable_exploration: bool = True
    max_parallel_branches: int = 5
    persona: str = "default"
    persona_auto_route: bool = True
    persona_auto_route_threshold: float = 0.3
    rag_enabled: bool = False

    # Knowledge graph
    kg_enabled: bool = False
    kg_depth: int = 1
    kg_top_k: int = 10

    # Speculative decoding (vLLM only).
    # Only works with a local vLLM backend; API providers (OpenAI/Anthropic/etc.)
    # handle caching/speculation internally so these settings are ignored there.
    # Requires a draft model — a small model from the same family as the target
    # (e.g. "Qwen/Qwen2.5-1.5B-Instruct" alongside a 7B/14B target).
    # Expected speedup: 2-3x inference throughput with negligible quality loss.
    # ponytail: config fields deleted — registry.py reads env vars directly
    # (HUGINN_SPECULATIVE_ENABLED/MODEL/DRAFT_TOKENS). Adding cfg params here
    # would require threading cfg through create_langchain_model, not worth it.

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

    # Allow local bash/code execution without container isolation.
    # Maps to HUGINN_ALLOW_LOCAL_BASH=1 for the sandbox executor.
    allow_local_bash: bool = False

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
    pet_name: str = "渡鸦"
    pet_personality: Literal["cheerful", "nerdy", "calm", "sassy"] = "cheerful"

    # 统一 opt-out 开关层, 见 huginn/feature_flags.py
    # 这里只存配置文件里的覆盖值, 运行时 toggle 不写回这里
    feature_flags: dict[str, bool] = field(default_factory=dict)

    def apply_overrides(
        self,
        *,
        provider: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        ollama_url: str | None = None,
        thinking: str | None = None,
    ) -> None:
        """Apply CLI flag overrides to this config instance in-place."""
        if provider:
            self.provider = provider  # type: ignore[assignment]
        if model:
            self.model = model
        if base_url:
            self.base_url = base_url
        if ollama_url:
            self.ollama_host = ollama_url
        if thinking:
            self.thinking = thinking  # type: ignore[assignment]

    def build_agent_kwargs(self, profile_id: str = "lead") -> dict[str, Any]:
        """Build keyword arguments for constructing a HuginnAgent.

        Resolves the requested agent profile, builds the model router when a
        model pool is configured, and applies all agent-scoped settings.
        """
        from huginn.checkpointer import (
            create_checkpointer,
            create_in_memory_checkpointer,
        )
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
                thinking = (
                    m.thinking
                    if m.thinking is not None
                    else (
                        effective_thinking
                        if effective_thinking is not None
                        else self.thinking
                    )
                )
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
                        max_tokens=(
                            m.max_tokens
                            if m.max_tokens is not None
                            else self.max_tokens
                        ),
                    )
                except Exception:
                    # Skip models that cannot be initialized (missing keys, etc.)
                    continue
        else:
            # Legacy single-model path
            from huginn.models.registry import create_langchain_model

            provider = self.provider
            if provider and provider != "default":
                thinking = (
                    effective_thinking
                    if effective_thinking is not None
                    else self.thinking
                )
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
            "workspace": self.workspace,
            "kg_enabled": self.kg_enabled,
            "kg_depth": self.kg_depth,
            "kg_top_k": self.kg_top_k,
            "auto_approve": self.auto_approve,
            "compression_max_tokens": self.tool_compression_max_tokens,
            "telemetry_enabled": self.telemetry_enabled,
            "memory_decay_enabled": self.memory_decay_enabled,
            "memory_decay_interval_turns": self.memory_decay_interval_turns,
            "memory_decay_prune_threshold": self.memory_decay_prune_threshold,
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
            models = [
                ModelConfig(
                    alias="default",
                    provider=provider,  # type: ignore[arg-type]
                    model=os.environ.get("HUGINN_MODEL"),
                    api_key=api_key,
                    base_url=os.environ.get("HUGINN_BASE_URL"),
                    thinking=thinking,
                    max_tokens=max_tokens,
                )
            ]

        # If no agent profiles, synthesize a default lead agent pointing at the default/only model.
        if not agents:
            model_alias = models[0].alias if models else "default"
            agents = [
                AgentProfileConfig(
                    id="lead",
                    name="Lead",
                    model_alias=model_alias,
                    persona=os.environ.get("HUGINN_PERSONA", "default").strip(),
                    thinking=thinking,
                )
            ]

        return cls(
            provider=provider,
            model=os.environ.get("HUGINN_MODEL"),
            api_key=api_key,
            base_url=os.environ.get("HUGINN_BASE_URL"),
            models=models,
            agents=agents,
            team_mode_enabled=os.environ.get("HUGINN_TEAM_MODE", "").lower() == "true",
            max_concurrent_subagents=int(
                os.environ.get("HUGINN_MAX_CONCURRENT_SUBAGENTS", "3")
            ),
            plan_auto_confirm=os.environ.get("HUGINN_PLAN_AUTO_CONFIRM", "0") == "1",
            ollama_host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
            vasp_executable=os.environ.get("VASP_EXECUTABLE"),
            lammps_executable=os.environ.get("LAMMPS_EXECUTABLE"),
            mp_api_key=os.environ.get("MP_API_KEY") or None,
            oqmd_api_key=os.environ.get("OQMD_API_KEY") or None,
            execution_backend=os.environ.get("HUGINN_EXECUTION_BACKEND", "local").lower(),  # type: ignore[arg-type]
            container_runtime=os.environ.get("HUGINN_CONTAINER_RUNTIME", "none").lower(),  # type: ignore[arg-type]
            container_image=os.environ.get("HUGINN_CONTAINER_IMAGE") or None,
            hpc_scheduler=os.environ.get("HPC_SCHEDULER", "local"),
            hpc_host=os.environ.get("HPC_HOST"),
            hpc_username=os.environ.get("HPC_USERNAME"),
            hpc_key_path=os.environ.get("HPC_KEY_PATH") or None,
            hpc_password=os.environ.get("HPC_PASSWORD") or None,
            hpc_port=int(os.environ.get("HPC_PORT", "22")),
            remote_work_dir=os.environ.get("HUGINN_REMOTE_WORK_DIR", "~/huginn_jobs"),
            hpc_default_queue=os.environ.get("HPC_DEFAULT_QUEUE") or None,
            hpc_gpu_queue=os.environ.get("HPC_GPU_QUEUE") or None,
            hpc_queue_map=_parse_queue_map(os.environ.get("HPC_QUEUE_MAP")),
            hpc_default_walltime=os.environ.get("HPC_DEFAULT_WALLTIME", "24:00:00"),
            hpc_default_nodes=int(os.environ.get("HPC_DEFAULT_NODES", "1")),
            hpc_default_ntasks_per_node=int(
                os.environ.get("HPC_DEFAULT_NTASKS_PER_NODE", "4")
            ),
            hpc_default_gpus_per_node=int(
                os.environ.get("HPC_DEFAULT_GPUS_PER_NODE", "0")
            ),
            hpc_max_retries=int(os.environ.get("HPC_MAX_RETRIES", "3")),
            hpc_retry_backoff=float(os.environ.get("HPC_RETRY_BACKOFF", "1.0")),
            hpc_strict_host_key_checking=os.environ.get(
                "HPC_STRICT_HOST_KEY_CHECKING", "true"
            ).lower()
            != "false",
            abaqus_mcp_server=os.environ.get("ABAQUS_MCP_SERVER_PATH"),
            workspace=os.environ.get("HUGINN_WORKSPACE", "."),
            auto_approve=os.environ.get("HUGINN_AUTO_APPROVE", "").lower() == "true",
            enable_exploration=os.environ.get(
                "HUGINN_ENABLE_EXPLORATION", "true"
            ).lower()
            == "true",
            max_parallel_branches=int(os.environ.get("HUGINN_MAX_BRANCHES", "5")),
            persona=os.environ.get("HUGINN_PERSONA", "default").strip(),
            persona_auto_route=os.environ.get(
                "HUGINN_PERSONA_AUTO_ROUTE", "true"
            ).lower()
            != "false",
            persona_auto_route_threshold=float(
                os.environ.get("HUGINN_PERSONA_AUTO_ROUTE_THRESHOLD", "0.3")
            ),
            rag_enabled=os.environ.get("HUGINN_RAG_ENABLED", "").lower() == "true",
            kg_enabled=os.environ.get("HUGINN_KG_ENABLED", "").lower() == "true",
            kg_depth=int(os.environ.get("HUGINN_KG_DEPTH", "1")),
            kg_top_k=int(os.environ.get("HUGINN_KG_TOP_K", "10")),
            local_only_mode=os.environ.get("HUGINN_LOCAL_ONLY", "").lower() == "true",
            allow_local_bash=os.environ.get("HUGINN_ALLOW_LOCAL_BASH", "").lower()
            in ("1", "true", "yes"),
            encryption_enabled=os.environ.get("HUGINN_ENCRYPTION_ENABLED", "").lower()
            == "true",
            encrypt_config=os.environ.get("HUGINN_ENCRYPT_CONFIG", "").lower()
            == "true",
            privacy_redact_secrets=os.environ.get(
                "HUGINN_PRIVACY_REDACT_SECRETS", "true"
            ).lower()
            != "false",
            encryption_password=os.environ.get("HUGINN_ENCRYPTION_PASSWORD") or None,
            encryption_key_file=os.environ.get("HUGINN_ENCRYPTION_KEY_FILE") or None,
            encrypt_rag_documents=os.environ.get(
                "HUGINN_ENCRYPT_RAG_DOCS", "true"
            ).lower()
            == "true",
            encrypt_rag_metadata=os.environ.get(
                "HUGINN_ENCRYPT_RAG_META", "true"
            ).lower()
            == "true",
            privacy_block_on_secrets=os.environ.get(
                "HUGINN_PRIVACY_BLOCK_ON_SECRETS", ""
            ).lower()
            == "true",
            max_tool_output_tokens=int(
                os.environ.get("HUGINN_MAX_TOOL_OUTPUT_TOKENS", "25000")
            ),
            context_budget_tokens=int(
                os.environ.get("HUGINN_CONTEXT_BUDGET_TOKENS", "0")
            ),
            prompt_cache_control=os.environ.get(
                "HUGINN_PROMPT_CACHE_CONTROL", "true"
            ).lower()
            != "false",
            checkpointer_path=os.environ.get("HUGINN_CHECKPOINTER_PATH") or None,
            telemetry_enabled=os.environ.get("HUGINN_TELEMETRY_ENABLED", "true").lower()
            != "false",
            memory_decay_enabled=os.environ.get(
                "HUGINN_MEMORY_DECAY_ENABLED", ""
            ).lower()
            == "true",
            memory_decay_interval_turns=int(
                os.environ.get("HUGINN_MEMORY_DECAY_INTERVAL_TURNS", "0")
            ),
            memory_decay_prune_threshold=float(
                os.environ.get("HUGINN_MEMORY_DECAY_PRUNE_THRESHOLD", "0.15")
            ),
            tool_compression_max_tokens=int(
                os.environ.get("HUGINN_TOOL_COMPRESSION_MAX_TOKENS", "8000")
            ),
            thinking=thinking,
            max_tokens=max_tokens,
            pet_name=os.environ.get("HUGINN_PET_NAME", "渡鸦").strip() or "渡鸦",
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
            logger.debug("suppressed in _parse_thinking_env", exc_info=True)
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
            logger.debug("suppressed in _parse_models_env", exc_info=True)
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
            logger.debug("suppressed in _parse_agents_env", exc_info=True)
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
            except ImportError as err:
                raise ImportError("pip install keyring") from err
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
                    "credential_id": m.credential_id,
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
                    "system_prompt_override": a.system_prompt_override,
                }
                for a in self.agents
            ],
            "team_mode_enabled": self.team_mode_enabled,
            "max_concurrent_subagents": self.max_concurrent_subagents,
            "plan_auto_confirm": self.plan_auto_confirm,
            "ollama_host": self.ollama_host,
            "mp_api_key": mask(self.mp_api_key),
            "oqmd_api_key": mask(self.oqmd_api_key),
            "vasp_executable": self.vasp_executable,
            "lammps_executable": self.lammps_executable,
            "execution_backend": self.execution_backend,
            "hpc_scheduler": self.hpc_scheduler,
            "hpc_host": self.hpc_host,
            "hpc_username": self.hpc_username,
            "hpc_key_path": self.hpc_key_path,
            "hpc_password": mask(self.hpc_password),
            "hpc_port": self.hpc_port,
            "remote_work_dir": self.remote_work_dir,
            "hpc_default_queue": self.hpc_default_queue,
            "hpc_gpu_queue": self.hpc_gpu_queue,
            "hpc_queue_map": self.hpc_queue_map,
            "hpc_default_walltime": self.hpc_default_walltime,
            "hpc_default_nodes": self.hpc_default_nodes,
            "hpc_default_ntasks_per_node": self.hpc_default_ntasks_per_node,
            "hpc_default_gpus_per_node": self.hpc_default_gpus_per_node,
            "hpc_max_retries": self.hpc_max_retries,
            "hpc_retry_backoff": self.hpc_retry_backoff,
            "hpc_strict_host_key_checking": self.hpc_strict_host_key_checking,
            "workspace": self.workspace,
            "auto_approve": self.auto_approve,
            "enable_exploration": self.enable_exploration,
            "max_parallel_branches": self.max_parallel_branches,
            "persona": self.persona,
            "persona_auto_route": self.persona_auto_route,
            "persona_auto_route_threshold": self.persona_auto_route_threshold,
            "rag_enabled": self.rag_enabled,
            "kg_enabled": self.kg_enabled,
            "kg_depth": self.kg_depth,
            "kg_top_k": self.kg_top_k,
            "local_only_mode": self.local_only_mode,
            "encryption_enabled": self.encryption_enabled,
            "encryption_password": mask(self.encryption_password),
            "encryption_key_file": self.encryption_key_file,
            "encrypt_rag_documents": self.encrypt_rag_documents,
            "encrypt_rag_metadata": self.encrypt_rag_metadata,
            "encrypt_config": self.encrypt_config,
            "privacy_redact_secrets": self.privacy_redact_secrets,
            "privacy_block_on_secrets": self.privacy_block_on_secrets,
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
            "feature_flags": dict(self.feature_flags),
            "config_version": self.config_version,
        }

    @classmethod
    def from_dict(cls, data: dict) -> HuginnConfig:
        """Restore config from a plain dict."""
        models_raw = data.get("models")
        agents_raw = data.get("agents")
        kwargs = {k: v for k, v in data.items() if k not in ("models", "agents")}

        if models_raw is not None:
            kwargs["models"] = [
                ModelConfig(**m) for m in models_raw if isinstance(m, dict)
            ]
        if agents_raw is not None:
            kwargs["agents"] = [
                AgentProfileConfig(**a) for a in agents_raw if isinstance(a, dict)
            ]

        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in kwargs.items() if k in known}
        return cls(**filtered)

    @property
    def security_config(self) -> SecurityConfig:
        """Return the security/privacy slice of the configuration."""
        return SecurityConfig(
            api_key=self.api_key,
            encryption_enabled=self.encryption_enabled,
            encryption_password=self.encryption_password,
            encryption_key_file=self.encryption_key_file,
            encrypt_config=self.encrypt_config,
            encrypt_rag_documents=self.encrypt_rag_documents,
            encrypt_rag_metadata=self.encrypt_rag_metadata,
            privacy_redact_secrets=self.privacy_redact_secrets,
            privacy_block_on_secrets=self.privacy_block_on_secrets,
            local_only_mode=self.local_only_mode,
        )

    @property
    def persistence_config(self) -> PersistenceConfig:
        """Return the persistence slice of the configuration."""
        return PersistenceConfig(checkpointer_path=self.checkpointer_path)

    @property
    def sandbox_config(self) -> SandboxConfig:
        """Return the sandbox/container execution slice of the configuration."""
        return SandboxConfig(
            container_runtime=self.container_runtime,
            container_image=self.container_image,
            max_tool_output_tokens=self.max_tool_output_tokens,
            context_budget_tokens=self.context_budget_tokens,
        )

    def save(
        self, path: str | pathlib.Path, format: Literal["toml", "json"] = "toml"
    ) -> None:
        """Persist configuration to a file.

        When ``encrypt_config`` is enabled (or the path ends in ``.enc``),
        the file is encrypted with the configured password/key file.
        Otherwise API keys are written in plain text; ensure the file has
        restricted permissions.

        写入流程：
        1. 加文件锁，防止并发写入互相覆盖；
        2. auth-loss guard：如果新配置会丢掉已有的 api_key / models，拒绝写入；
        3. 备份当前文件（带时间戳轮转）；
        4. 原子写入(tmp + os.replace)：先写临时文件, 回读校验格式,
           再 rename 替换原文件, 避免写入中断导致配置损坏；
        5. 同步更新内存缓存。
        """
        global _config_cache, _config_cache_path, _config_cache_mtime

        target = pathlib.Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)

        with _config_lock, _file_lock(target):
            # auth-loss guard：拿缓存里的旧配置做对比
            cached = _config_cache
            if cached is not None and _would_lose_auth_state(cached, self):
                raise RuntimeError(
                    "Refusing to save: new config would lose existing "
                    "api_key / models. Restore them or clear the cache first."
                )

            # 写入前备份
            _backup_before_save(target)

            data = self.to_dict(mask_key=False)

            if self.encrypt_config or str(target).endswith(".enc"):
                # 加密配置由 EncryptedConfig 自行处理写入, 不走原子写路径
                vault = self._get_vault()
                ec = EncryptedConfig(config_path=target, vault=vault)
                ec.save(data)
            else:
                _atomic_write(target, data, format)

            # 写盘成功后同步更新缓存
            _config_cache = self
            _config_cache_path = target
            try:
                _config_cache_mtime = target.stat().st_mtime
            except OSError:
                _config_cache_mtime = time.time()

    def _get_vault(self) -> CryptoVault:
        """Return an unlocked CryptoVault for config encryption."""
        password = self.encryption_password or os.environ.get(
            "HUGINN_ENCRYPTION_PASSWORD"
        )
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
            except ImportError as err:
                raise ImportError("pip install toml") from err
            data = toml.loads(text)
        else:
            import json

            data = json.loads(text)
        return cls.from_dict(data)

    @classmethod
    def check_and_heal(cls, path: str | pathlib.Path | None = None) -> list[str]:
        """检查并修复配置文件完整性 (AstrBot 风格自愈).

        加载配置文件 → 与默认值递归比对 → 补全缺失键 → 删除孤儿键 → 原子写回.

        Returns: 变更列表
        """
        from huginn.config_integrity import migrate_config

        if path is None:
            return []
        path = pathlib.Path(path)
        if not path.exists():
            return []

        # tomllib is stdlib in 3.11+; fall back to tomli for older Python
        try:
            import tomllib
        except ModuleNotFoundError:
            try:
                import tomli as tomllib
            except ModuleNotFoundError:
                logger.warning("Neither tomllib nor tomli available; TOML config healing skipped")
                return []

        try:
            if path.suffix == ".json":
                with open(path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
            else:
                with open(path, "rb") as f:
                    raw = tomllib.load(f)
        except Exception as e:
            logger.warning("Failed to load config for healing: %s", e)
            return []

        healed, changes = migrate_config(raw)

        if changes:
            _atomic_write(
                path,
                healed,
                format="json" if path.suffix == ".json" else "toml",
            )
            logger.info(
                "Config self-healed %d issues: %s", len(changes), "; ".join(changes)
            )

        return changes


@dataclass
class CoderSettings:
    """Settings for autonomous coder mode."""

    max_iterations: int = 50
    done_marker: str = "[DONE]"

    @classmethod
    def from_env(cls) -> CoderSettings:
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


_cached_settings: Settings | None = None

def get_settings() -> Settings:
    """Load application settings — cached after first call."""
    global _cached_settings
    if _cached_settings is None:
        _cached_settings = Settings()
    return _cached_settings


def get_config(
    path: str | pathlib.Path | None = None,
    *,
    force_reload: bool = False,
) -> HuginnConfig:
    """获取配置实例，read-through cache。

    首次调用从磁盘加载（或退回 from_env），之后直接命中内存缓存。
    如果磁盘文件被其他进程修改过（mtime 变化），自动刷新缓存。

    Args:
        path: 配置文件路径。None 时退回 from_env()。
        force_reload: True 时强制重新加载，忽略缓存。
    """
    global _config_cache, _config_cache_path, _config_cache_mtime

    with _config_lock:
        if path is None:
            # check HUGINN_CONFIG_FILE before falling back to from_env
            env_path = os.environ.get("HUGINN_CONFIG_FILE")
            if env_path:
                path = env_path
            elif _config_cache is None or force_reload:
                _config_cache = HuginnConfig.from_env()
                _config_cache_path = None
                _config_cache_mtime = 0.0
                return _config_cache
            else:
                return _config_cache

        target = pathlib.Path(path)

        # 跨进程新鲜度检测：磁盘文件被改过就刷新
        disk_changed = _check_disk_freshness(target)
        same_path = (
            _config_cache_path is not None
            and pathlib.Path(_config_cache_path).resolve() == target.resolve()
        )

        if _config_cache is None or force_reload or disk_changed or not same_path:
            if target.exists():
                try:
                    _config_cache = HuginnConfig.load(target)
                    _config_cache_path = target
                except Exception:
                    # toml parse error or missing dep — fall back to env
                    _config_cache = HuginnConfig.from_env()
                    _config_cache_path = None
                try:
                    _config_cache_mtime = target.stat().st_mtime
                except OSError:
                    _config_cache_mtime = time.time()
            else:
                # 文件不存在，退回环境变量
                _config_cache = HuginnConfig.from_env()
                _config_cache_path = None
                _config_cache_mtime = 0.0

        return _config_cache


def clear_config_cache() -> None:
    """清空配置缓存。下次 get_config() 会重新加载。"""
    global _config_cache, _config_cache_path, _config_cache_mtime
    with _config_lock:
        _config_cache = None
        _config_cache_path = None
        _config_cache_mtime = 0.0
