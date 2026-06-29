"""配置对象 for HuginnAgent.

把 HuginnAgent.__init__ 原本近 40 个散落的构造参数按职责分组到一组
dataclass 里, 集中环境变量默认值逻辑, 并提供构建器与从 HuginnConfig
派生的桥梁.

类名统一加 ``Agent`` 前缀, 避免和 ``huginn.config`` 里已有的
``ModelConfig`` / ``SecurityConfig`` / ``SandboxConfig`` 同名冲突.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from huginn.config import HuginnConfig
    from huginn.models.router import ModelRouter


def _env_bool(name: str, default: bool) -> bool:
    """读取布尔型环境变量, 默认值用 ``default``.

    兼容原 __init__ 里两种写法:
    - ``!= "0"``      -> 传 default=True,  命中 "0" 为 False
    - ``== "1"``      -> 传 default=False, 命中 "1" 为 True
    - ``.lower() in ("1","true","yes")`` -> 同上
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class AgentCoreConfig:
    """Agent 身份与顶层行为开关."""

    profile_id: str = "default"
    thread_id: str | None = None
    enable_exploration: bool = True
    agent_factory: Any | None = None
    skill_executor: Any | None = None


@dataclass
class AgentModelConfig:
    """模型 / 路由 / 系统提示相关配置."""

    model: Any | None = None
    model_router: "ModelRouter | None" = None
    system_prompt: str | None = None
    begin_dialogs: list[tuple[str, str]] | None = None
    prompt_cache_control: bool | None = None

    @classmethod
    def from_env(cls) -> AgentModelConfig:
        return cls(
            prompt_cache_control=(
                os.environ.get("HUGINN_PROMPT_CACHE_CONTROL", "1") != "0"
            ),
        )


@dataclass
class AgentToolConfig:
    """工具注册 / 调用预算 / 输出压缩配置."""

    tools: list[Any] | None = None
    tool_filter: list[str] | None = None
    max_tool_output_tokens: int | None = None
    max_tool_calls: int | None = None
    max_tool_calls_per_tool: int | None = None
    break_after_tool: bool = False
    compression_max_tokens: int | None = None

    @classmethod
    def from_env(cls) -> AgentToolConfig:
        return cls(
            max_tool_output_tokens=int(
                os.environ.get("HUGINN_MAX_TOOL_OUTPUT_TOKENS", "25000")
            ),
            max_tool_calls=int(os.environ.get("HUGINN_MAX_TOOL_CALLS", "15")),
            max_tool_calls_per_tool=int(
                os.environ.get("HUGINN_MAX_TOOL_CALLS_PER_TOOL", "5")
            ),
            compression_max_tokens=int(
                os.environ.get("HUGINN_TOOL_COMPRESSION_MAX_TOKENS", "8000")
            ),
        )


@dataclass
class AgentMemoryConfig:
    """内存管理器 / 检查点 / 记忆衰减配置."""

    memory_manager: Any | None = None
    checkpointer: Any | None = None
    checkpointer_path: str | None = None
    memory_decay_enabled: bool | None = None
    memory_decay_interval_turns: int | None = None
    memory_decay_prune_threshold: float | None = None

    @classmethod
    def from_env(cls) -> AgentMemoryConfig:
        return cls(
            checkpointer_path=os.environ.get("HUGINN_CHECKPOINTER_PATH") or None,
            memory_decay_enabled=(
                os.environ.get("HUGINN_MEMORY_DECAY_ENABLED", "").lower() == "true"
            ),
            memory_decay_interval_turns=int(
                os.environ.get("HUGINN_MEMORY_DECAY_INTERVAL_TURNS", "0")
            ),
            memory_decay_prune_threshold=float(
                os.environ.get("HUGINN_MEMORY_DECAY_PRUNE_THRESHOLD", "0.15")
            ),
        )


@dataclass
class AgentSecurityConfig:
    """沙箱 / 审计 / 隐私 / 审批配置."""

    sandbox: Any | None = None
    audit: Any | None = None
    privacy_redact_secrets: bool | None = None
    privacy_block_on_secrets: bool | None = None
    auto_approve: bool | None = None
    approval_callback: Callable[[str, str], bool] | None = None

    @classmethod
    def from_env(cls) -> AgentSecurityConfig:
        return cls(
            privacy_redact_secrets=(
                os.environ.get("HUGINN_PRIVACY_REDACT_SECRETS", "1") != "0"
            ),
            privacy_block_on_secrets=(
                os.environ.get("HUGINN_PRIVACY_BLOCK_ON_SECRETS", "0") == "1"
            ),
            auto_approve=_env_bool("HUGINN_AUTO_APPROVE", False),
        )


@dataclass
class AgentTelemetryConfig:
    """遥测开关配置."""

    telemetry_enabled: bool | None = None

    @classmethod
    def from_env(cls) -> AgentTelemetryConfig:
        return cls(
            telemetry_enabled=(
                os.environ.get("HUGINN_TELEMETRY_ENABLED", "1").lower() != "false"
            ),
        )


@dataclass
class AgentContextBudgetConfig:
    """上下文窗口预算配置.

    ``context_budget_tokens`` 为 None 或 <=0 且配置了 model 时, 由
    HuginnAgent 按模型名自动推断上下文窗口大小.
    """

    context_budget_tokens: int | None = None

    @classmethod
    def from_env(cls) -> AgentContextBudgetConfig:
        return cls(
            context_budget_tokens=int(
                os.environ.get("HUGINN_CONTEXT_BUDGET_TOKENS", "0")
            ),
        )


@dataclass
class AgentKnowledgeGraphConfig:
    """知识图谱 / workspace 配置."""

    workspace: str = "."
    kg_enabled: bool = False
    kg_depth: int = 1
    kg_top_k: int = 10


@dataclass
class AgentPersonalizationConfig:
    """人格 / 情感 / 风格学习配置."""

    persona_name: str | None = None
    emotion_tracker: Any | None = None
    style_learner: Any | None = None


@dataclass
class AgentConfig:
    """HuginnAgent 的聚合配置.

    用法::

        cfg = AgentConfig.from_env()
        cfg.model.model = my_model
        cfg.security.auto_approve = True
        agent = HuginnAgent(config=cfg)

    也兼容直接传旧的位置/关键字参数 (见 HuginnAgent.__init__).
    """

    core: AgentCoreConfig = field(default_factory=AgentCoreConfig)
    model: AgentModelConfig = field(default_factory=AgentModelConfig)
    tools: AgentToolConfig = field(default_factory=AgentToolConfig)
    memory: AgentMemoryConfig = field(default_factory=AgentMemoryConfig)
    security: AgentSecurityConfig = field(default_factory=AgentSecurityConfig)
    telemetry: AgentTelemetryConfig = field(default_factory=AgentTelemetryConfig)
    context_budget: AgentContextBudgetConfig = field(
        default_factory=AgentContextBudgetConfig
    )
    knowledge_graph: AgentKnowledgeGraphConfig = field(
        default_factory=AgentKnowledgeGraphConfig
    )
    personalization: AgentPersonalizationConfig = field(
        default_factory=AgentPersonalizationConfig
    )

    @classmethod
    def from_env(cls) -> AgentConfig:
        """从环境变量构建默认配置 (与原 __init__ 的 env 回退逻辑一致)."""
        return cls(
            model=AgentModelConfig.from_env(),
            tools=AgentToolConfig.from_env(),
            memory=AgentMemoryConfig.from_env(),
            security=AgentSecurityConfig.from_env(),
            telemetry=AgentTelemetryConfig.from_env(),
            context_budget=AgentContextBudgetConfig.from_env(),
        )

    @classmethod
    def from_huginn_config(
        cls, config: "HuginnConfig", profile_id: str = "lead"
    ) -> AgentConfig:
        """从全局 HuginnConfig 派生 AgentConfig.

        替代 HuginnConfig.build_agent_kwargs() 的 dict 中转方式,
        让 factory 可以直接构造 config 对象.
        """
        profile = config.get_profile(profile_id)
        return cls(
            core=AgentCoreConfig(
                profile_id=profile_id,
                enable_exploration=config.enable_exploration,
            ),
            model=AgentModelConfig(
                prompt_cache_control=config.prompt_cache_control,
            ),
            tools=AgentToolConfig(
                tool_filter=profile.tools if profile else None,
                max_tool_output_tokens=config.max_tool_output_tokens,
                compression_max_tokens=config.tool_compression_max_tokens,
            ),
            memory=AgentMemoryConfig(
                checkpointer_path=config.checkpointer_path,
                memory_decay_enabled=config.memory_decay_enabled,
                memory_decay_interval_turns=config.memory_decay_interval_turns,
                memory_decay_prune_threshold=config.memory_decay_prune_threshold,
            ),
            security=AgentSecurityConfig(
                privacy_redact_secrets=config.privacy_redact_secrets,
                privacy_block_on_secrets=config.privacy_block_on_secrets,
                auto_approve=config.auto_approve,
            ),
            telemetry=AgentTelemetryConfig(
                telemetry_enabled=config.telemetry_enabled,
            ),
            context_budget=AgentContextBudgetConfig(
                context_budget_tokens=config.context_budget_tokens,
            ),
            knowledge_graph=AgentKnowledgeGraphConfig(
                workspace=config.workspace,
                kg_enabled=config.kg_enabled,
                kg_depth=config.kg_depth,
                kg_top_k=config.kg_top_k,
            ),
            personalization=AgentPersonalizationConfig(
                persona_name=(profile.persona if profile else None),
            ),
        )


class _UnsetSentinel:
    """标记"调用方未传入该参数"的单例, 区分 None 与未指定."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - 调试辅助
        return "<UNSET>"


_UNSET_SENTINEL = _UnsetSentinel()


class HuginnAgentBuilder:
    """HuginnAgent 的流式构建器.

    用法::

        agent = (
            HuginnAgentBuilder()
            .with_model(model)
            .with_tools(tools)
            .with_profile("research", "session-123")
            .enable_knowledge_graph(depth=2, top_k=20)
            .build()
        )
    """

    def __init__(self, config: AgentConfig | None = None) -> None:
        self._config = config or AgentConfig.from_env()

    # ── 模型 ──────────────────────────────────────────────
    def with_model(self, model: Any) -> HuginnAgentBuilder:
        self._config.model.model = model
        return self

    def with_model_router(self, router: "ModelRouter") -> HuginnAgentBuilder:
        self._config.model.model_router = router
        return self

    def with_system_prompt(self, prompt: str) -> HuginnAgentBuilder:
        self._config.model.system_prompt = prompt
        return self

    def with_begin_dialogs(
        self, dialogs: list[tuple[str, str]]
    ) -> HuginnAgentBuilder:
        self._config.model.begin_dialogs = dialogs
        return self

    # ── 工具 ──────────────────────────────────────────────
    def with_tools(self, tools: list[Any]) -> HuginnAgentBuilder:
        self._config.tools.tools = tools
        return self

    def with_tool_filter(self, allowlist: list[str]) -> HuginnAgentBuilder:
        self._config.tools.tool_filter = allowlist
        return self

    def with_tool_budget(
        self, max_calls: int | None = None, max_per_tool: int | None = None
    ) -> HuginnAgentBuilder:
        if max_calls is not None:
            self._config.tools.max_tool_calls = max_calls
        if max_per_tool is not None:
            self._config.tools.max_tool_calls_per_tool = max_per_tool
        return self

    def break_after_tool(self, enabled: bool = True) -> HuginnAgentBuilder:
        self._config.tools.break_after_tool = enabled
        return self

    # ── 内存 ──────────────────────────────────────────────
    def with_memory_manager(self, manager: Any) -> HuginnAgentBuilder:
        self._config.memory.memory_manager = manager
        return self

    def with_checkpointer(self, checkpointer: Any) -> HuginnAgentBuilder:
        self._config.memory.checkpointer = checkpointer
        return self

    def with_checkpointer_path(self, path: str) -> HuginnAgentBuilder:
        self._config.memory.checkpointer_path = path
        return self

    def enable_memory_decay(
        self,
        interval_turns: int | None = None,
        prune_threshold: float | None = None,
    ) -> HuginnAgentBuilder:
        self._config.memory.memory_decay_enabled = True
        if interval_turns is not None:
            self._config.memory.memory_decay_interval_turns = interval_turns
        if prune_threshold is not None:
            self._config.memory.memory_decay_prune_threshold = prune_threshold
        return self

    # ── 安全 ──────────────────────────────────────────────
    def with_sandbox(self, sandbox: Any) -> HuginnAgentBuilder:
        self._config.security.sandbox = sandbox
        return self

    def with_audit(self, audit: Any) -> HuginnAgentBuilder:
        self._config.security.audit = audit
        return self

    def with_approval_callback(
        self, callback: Callable[[str, str], bool]
    ) -> HuginnAgentBuilder:
        self._config.security.approval_callback = callback
        return self

    def auto_approve(self, enabled: bool = True) -> HuginnAgentBuilder:
        self._config.security.auto_approve = enabled
        return self

    # ── 身份 ──────────────────────────────────────────────
    def with_profile(
        self, profile_id: str, thread_id: str | None = None
    ) -> HuginnAgentBuilder:
        self._config.core.profile_id = profile_id
        self._config.core.thread_id = thread_id
        return self

    def with_agent_factory(self, factory: Any) -> HuginnAgentBuilder:
        self._config.core.agent_factory = factory
        return self

    # ── 知识图谱 ──────────────────────────────────────────
    def enable_knowledge_graph(
        self, depth: int = 1, top_k: int = 10
    ) -> HuginnAgentBuilder:
        self._config.knowledge_graph.kg_enabled = True
        self._config.knowledge_graph.kg_depth = depth
        self._config.knowledge_graph.kg_top_k = top_k
        return self

    def with_workspace(self, workspace: str) -> HuginnAgentBuilder:
        self._config.knowledge_graph.workspace = workspace
        return self

    # ── 个性化 ────────────────────────────────────────────
    def with_persona(self, name: str) -> HuginnAgentBuilder:
        self._config.personalization.persona_name = name
        return self

    def with_emotion_tracker(self, tracker: Any) -> HuginnAgentBuilder:
        self._config.personalization.emotion_tracker = tracker
        return self

    def with_style_learner(self, learner: Any) -> HuginnAgentBuilder:
        self._config.personalization.style_learner = learner
        return self

    # ── 上下文预算 ────────────────────────────────────────
    def with_context_budget(self, tokens: int) -> HuginnAgentBuilder:
        self._config.context_budget.context_budget_tokens = tokens
        return self

    # ── 遥测 ──────────────────────────────────────────────
    def with_telemetry(self, enabled: bool = True) -> HuginnAgentBuilder:
        self._config.telemetry.telemetry_enabled = enabled
        return self

    # ── 构建 ──────────────────────────────────────────────
    @property
    def config(self) -> AgentConfig:
        return self._config

    def build(self) -> Any:
        """构造 HuginnAgent. 延迟导入避免循环依赖."""
        from huginn.agent import HuginnAgent

        return HuginnAgent(config=self._config)
