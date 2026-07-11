"""Agent factory — create HuginnAgent instances from configured profiles.

Each profile picks a model alias (or `provider/model`), a persona, and an
optional tool allowlist. The factory reuses the global ModelRegistry so
provider/model instances are cached.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from huginn.agent import HuginnAgent
from huginn.config import AgentProfileConfig, HuginnConfig, ThinkingIntensity
from huginn.models.registry import ModelRegistry
from huginn.persona_emotion import EmotionTracker
from huginn.personas import PersonaManager
from huginn.project_context import load_project_context
import logging
logger = logging.getLogger(__name__)



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
        self.persona_manager = PersonaManager(workspace=config.workspace)
        self._profiles: dict[str, AgentProfileConfig] = {
            a.id: a for a in config.agents if a.enabled
        }
        # 共享 checkpointer — 同一 thread_id 的对话历史跨请求保留
        # 配了 checkpointer_path 就走 SqliteSaver (文件锁, 并发安全)
        # 没配才退回 InMemorySaver (并发长程有竞态, 见问题23)
        # 注意: persistent_checkpointer 返回 context manager, 必须持有 cm 对象
        # 否则 GC 回收 cm 会触发 __exit__ 关闭数据库连接
        from huginn.checkpointer import persistent_checkpointer, create_in_memory_checkpointer
        if self.config.checkpointer_path:
            self._checkpointer_cm = persistent_checkpointer(self.config.checkpointer_path)
            self._shared_checkpointer = self._checkpointer_cm.__enter__()
        else:
            self._shared_checkpointer = create_in_memory_checkpointer()

        # PRT Level 0 异常登记表. db 放跟 checkpointer 同目录, 没配就放 workspace 下.
        # 生命周期跟 factory 一致, 所有 agent 共用一份.
        from huginn.anomaly_log import AnomalyLogStore
        anomaly_db = self._anomaly_db_path()
        self._anomaly_store = AnomalyLogStore(anomaly_db)

        # 共享工具调度器 — 同一 factory 下所有 agent (含 orchestrator/team/swarm
        # 派生的子 agent) 共用同一份 heavy/light 信号量 + 资源预算, 父子 agent
        # 提交重作业时由它仲裁并发. 持久后端用 SqliteCampaignStore (jobs 表),
        # 跟 anomalies.db 同目录; 构造失败退回 NullCampaignStore 纯内存模式.
        self._shared_scheduler = self._build_shared_scheduler()

        # 事件总线审计订阅 — 启动时装一次, 所有 publish 的事件落 audit.jsonl
        try:
            from huginn.events.audit_log import install_audit_subscriber
            self._audit_unsub = install_audit_subscriber()
        except Exception:
            logger.debug("audit subscriber install failed (non-fatal)", exc_info=True)
            self._audit_unsub = None

    def _build_shared_scheduler(self) -> Any:
        from huginn.persistence.campaign import (
            NullCampaignStore,
            SqliteCampaignStore,
        )
        from huginn.scheduling import AdmissionPolicy, ToolScheduler

        try:
            store = SqliteCampaignStore(self._campaign_db_path())
        except Exception:
            store = NullCampaignStore()
        try:
            return ToolScheduler(store=store, policy=AdmissionPolicy.from_env())
        except Exception:
            return None

    def _campaign_db_path(self) -> str:
        from pathlib import Path

        if self.config.checkpointer_path:
            return str(Path(self.config.checkpointer_path).expanduser().parent / "campaigns.sqlite")
        return str(Path(self.config.workspace).expanduser() / "campaigns.sqlite")

    def _anomaly_db_path(self) -> str:
        """anomalies.db 跟 checkpoints.db 放一个目录."""
        from pathlib import Path
        if self.config.checkpointer_path:
            return str(Path(self.config.checkpointer_path).expanduser().parent / "anomalies.db")
        # 没配 checkpointer 就退到 workspace
        return str(Path(self.config.workspace).expanduser() / "anomalies.db")

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
        approval_callback: Callable[[str, str], bool] | None = None,
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
            raise ValueError(
                f"Profile '{profile_id}' has no model_alias and no default model is configured"
            )

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
                logger.debug("load project context failed", exc_info=True)

        emotion_tracker = EmotionTracker(
            profile.persona, workspace=self.config.workspace
        )
        agent = HuginnAgent(
            model=model,
            system_prompt=prompt,
            begin_dialogs=begin_dialogs,
            memory_manager=(
                memory_manager if memory_manager is not None else self.memory_manager
            ),
            profile_id=profile_id,
            thread_id=thread_id,
            checkpointer=self._shared_checkpointer,
            tool_filter=profile.tools if profile.tools else None,
            agent_factory=self,
            privacy_redact_secrets=self.config.privacy_redact_secrets,
            privacy_block_on_secrets=self.config.privacy_block_on_secrets,
            max_tool_output_tokens=self.config.max_tool_output_tokens,
            context_budget_tokens=self.config.context_budget_tokens,
            workspace=self.config.workspace,
            auto_approve=self.config.auto_approve,
            compression_max_tokens=self.config.tool_compression_max_tokens,
            telemetry_enabled=self.config.telemetry_enabled,
            memory_decay_enabled=self.config.memory_decay_enabled,
            memory_decay_interval_turns=self.config.memory_decay_interval_turns,
            memory_decay_prune_threshold=self.config.memory_decay_prune_threshold,
            persona_name=profile.persona,
            emotion_tracker=emotion_tracker,
            approval_callback=approval_callback,
            scheduler=self._shared_scheduler,
        )
        agent.register_tools_from_registry()

        # 个人定制: 注入共享 StyleLearner, chat() 里会自动学用户语言偏好.
        # 共享单例保证同 workspace 下所有 agent 实例用同一份 profile.
        from huginn.personalization import get_shared_style_learner
        agent.set_style_learner(get_shared_style_learner())

        # PRT Level 0: 挂上异常检测钩子, 拦截工具输出做登记.
        # 钩子在工具调用时按 hook_manager 动态读取, 注册时机不敏感.
        from huginn.hooks import POST_TOOL_USE, AnomalyDetectionHook
        agent.register_hook(POST_TOOL_USE, AnomalyDetectionHook(self._anomaly_store))

        # PRT Level 1: LLM 异常判定, 默认关. 开启条件 HUGINN_PRT_LEVEL1=1.
        # 每次被观察工具调用都会打一次小模型(deepseek-chat), 有成本, 所以默认关.
        if os.environ.get("HUGINN_PRT_LEVEL1", "0") == "1":
            from huginn.hooks.anomaly_llm_hook import AnomalyLLMHook
            agent.register_hook(POST_TOOL_USE, AnomalyLLMHook(self._anomaly_store))

        # Prompt 引导钩子: 用户提问里命中"验证/计算/求解"等关键词时,
        # 强制要求走工具. 纯规则匹配零成本, 默认开, 不需要环境变量.
        from huginn.hooks import USER_PROMPT_SUBMIT
        from huginn.hooks.prompt_guidance_hook import PromptGuidanceHook
        agent.register_hook(USER_PROMPT_SUBMIT, PromptGuidanceHook())

        # Questions 机制: 信息不足时(目标模糊/参数缺失/输出未定/过短)
        # 生成结构化追问, 降低无效迭代. 每 thread 最多追问一次.
        from huginn.hooks.clarify_questions_hook import ClarifyQuestionsHook
        agent.register_hook(USER_PROMPT_SUBMIT, ClarifyQuestionsHook())

        # 工具名校验: 用户消息里点名的工具不在可用列表时, 注入 prompt_guidance
        # 让 agent 明确告知用户该工具不存在, 别默默调 ls/web_search 探索.
        from huginn.hooks.tool_name_validation_hook import ToolNameValidationHook
        agent.register_hook(USER_PROMPT_SUBMIT, ToolNameValidationHook())

        # Design Plan gate: 调用 vasp/lammps 等执行类工具前,
        # 建议先有用户确认的 plan. Chat 模式下 advisory (不硬阻塞),
        # 避免工作流堵点. autoloop 有自己的 plan 确认机制.
        from huginn.hooks import PRE_TOOL_USE
        from huginn.hooks.design_plan_gate_hook import DesignPlanGateHook
        agent.register_hook(PRE_TOOL_USE, DesignPlanGateHook(mode="advisory"))

        return agent

    def create_lead(
        self,
        thread_id: str | None = None,
        thinking: ThinkingIntensity | dict[str, Any] | None = None,
        max_tokens: int | None = None,
        approval_callback: Callable[[str, str], bool] | None = None,
        system_prompt_override: str | None = None,
    ) -> HuginnAgent:
        """Convenience: create the lead/default agent."""
        kwargs: dict[str, Any] = {}
        if thinking is not None:
            kwargs["thinking"] = thinking
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if approval_callback is not None:
            kwargs["approval_callback"] = approval_callback
        if system_prompt_override is not None:
            kwargs["system_prompt_override"] = system_prompt_override
        for preferred in ("lead", "default"):
            if preferred in self._profiles:
                return self.create(preferred, thread_id=thread_id, **kwargs)
        # Fall back to first configured profile
        if self._profiles:
            return self.create(
                next(iter(self._profiles)), thread_id=thread_id, **kwargs
            )
        raise ValueError("No enabled agent profiles found")
