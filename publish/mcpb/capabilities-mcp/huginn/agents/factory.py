"""Agent factory — create HuginnAgent instances from configured profiles.

Each profile picks a model alias (or `provider/model`), a persona, and an
optional tool allowlist. The factory reuses the global ModelRegistry so
provider/model instances are cached.
"""

from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import Callable
from typing import Any

from huginn.agent import HuginnAgent
from huginn.config import AgentProfileConfig, HuginnConfig, ThinkingIntensity
from huginn.models.registry import ModelRegistry
from huginn.persona_emotion import EmotionTracker
from huginn.personas import PersonaManager
from huginn.project_context import load_project_context

logger = logging.getLogger(__name__)


# 反跑题任务锚 — 挂在主 agent 系统提示词尾部, 防止在开放任务里被中间输出带偏.
# 三条硬约束: 锚定原目标 / 不回绕无关主题 / 收敛到交付件. 措辞直接, 不绕弯.
_TASK_ANCHOR = (
    "Stay anchored to the user's stated task. Re-read it after every tool call.\n"
    "If exploration or a tool result is drifting away from the goal, stop expanding "
    "scope and return to the original request.\n"
    "Do not chase tangential topics, newly-surfaced features, or adjacent problems "
    "unless the user asks. When the task is complete, say so and stop."
)



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
        self.persona_manager = self._init_persona_manager(config)
        self._profiles: dict[str, AgentProfileConfig] = {
            a.id: a for a in config.agents if a.enabled
        }
        self._init_checkpointer()
        self._init_observability()
        # 共享工具调度器 — 同一 factory 下所有 agent (含 orchestrator/team/swarm
        # 派生的子 agent) 共用同一份 heavy/light 信号量 + 资源预算, 父子 agent
        # 提交重作业时由它仲裁并发.
        self._shared_scheduler = self._build_shared_scheduler()
        # 事件总线审计订阅 — 启动时装一次, 所有 publish 的事件落 audit.jsonl
        self._audit_unsub = self._init_audit_subscribers()

    @staticmethod
    def _init_persona_manager(config: HuginnConfig) -> PersonaManager:
        """建 PersonaManager; 配了 persona_auto_sync 时先同步技能目录."""
        manager = PersonaManager(workspace=config.workspace)
        if config.persona_auto_sync and config.persona_sync_dir:
            try:
                from pathlib import Path
                manager.sync_skills(Path(config.persona_sync_dir))
            except Exception:  # 防御: 人格技能同步失败则忽略
                logger.debug("persona auto-sync failed (non-fatal)", exc_info=True)
        return manager

    def _init_checkpointer(self) -> None:
        """共享 checkpointer — 同一 thread_id 的对话历史跨请求保留.

        配了 checkpointer_path 就走 SqliteSaver (文件锁, 并发安全),
        没配才退回 InMemorySaver (并发长程有竞态, 见问题23).
        注意: persistent_checkpointer 返回 context manager, 必须持有 cm 对象,
        否则 GC 回收 cm 会触发 __exit__ 关闭数据库连接.
        """
        from huginn.checkpointer import (
            create_in_memory_checkpointer,
            persistent_checkpointer,
        )
        if self.config.checkpointer_path:
            self._checkpointer_cm = persistent_checkpointer(self.config.checkpointer_path)
            self._shared_checkpointer = self._checkpointer_cm.__enter__()
        else:
            self._shared_checkpointer = create_in_memory_checkpointer()

    def _init_observability(self) -> None:
        """PRT Level 0 异常登记表 + ISC 反渗透跨轮安全风险追踪 (工厂级共享)."""
        # db 放跟 checkpointer 同目录, 没配就放 workspace 下.
        # 生命周期跟 factory 一致, 所有 agent 共用一份.
        from huginn.anomaly_log import AnomalyLogStore
        self._anomaly_store = AnomalyLogStore(self._anomaly_db_path())

        # agent 按请求新建, 状态挂钩子实例上会跨轮丢失, 所以 tracker 放工厂级.
        from huginn.hooks.safety_collapse_hook import SafetyCollapseTracker
        self._safety_tracker = SafetyCollapseTracker()

    @staticmethod
    def _init_audit_subscribers() -> Any:
        """装审计订阅 (失败返 None); 返 unsubscribe 句柄供测试/关闭时摘除."""
        try:
            from huginn.events.audit_log import (
                install_audit_subscriber,
                install_campaign_subscriber,
            )
            unsub = install_audit_subscriber()
            # campaign.* / quality.check 业务订阅: 之前 emit→audit→dead, 现在至少 log 可见.
            install_campaign_subscriber()
            return unsub
        except Exception:  # 防御: 审计订阅安装失败则忽略
            logger.debug("audit subscriber install failed (non-fatal)", exc_info=True)
            return None

    def _build_shared_scheduler(self) -> Any:
        from huginn.persistence.campaign import (
            NullCampaignStore,
            SqliteCampaignStore,
        )
        from huginn.scheduling import AdmissionPolicy, ToolScheduler

        try:
            store = SqliteCampaignStore(self._campaign_db_path())
        except Exception:  # 防御: 持久存储不可用则退空实现
            store = NullCampaignStore()
        try:
            scheduler = ToolScheduler(
                store=store,
                policy=AdmissionPolicy.from_env(),
                hpc_layer=self._build_hpc_layer(),
            )
            # 重启后先对账: 遗留 running→orphaned、queued 重新挂起、current 模式下
            # 把远端还在烧机时的作业机时反推回预算占用, 防止新进程看到归零预算而超投.
            with contextlib.suppress(Exception):
                scheduler.recover()
            return scheduler
        except Exception:  # 防御: 调度器构造失败则禁用
            logger.debug("best-effort op failed", exc_info=True)
            return None

    def _build_hpc_layer(self) -> Any:
        """Build an HPC queue layer from configured HPC settings, or None.

        Only enabled when ``hpc_scheduler`` is remote (slurm/pbs) and a host +
        username are set. Building the client is cheap — connection is lazy
        (HPCClient connects on first submit), so an unreachable cluster merely
        disables remote admission rather than blocking startup.
        """
        cfg = self.config
        if cfg.hpc_scheduler == "local" or not cfg.hpc_host or not cfg.hpc_username:
            return None
        try:
            from huginn.hpc.client import HPCClient
            from huginn.scheduling import HpcQueueLayer

            layer = HpcQueueLayer(
                self._hpc_queue_config(cfg),
                client=HPCClient(self._hpc_client_config(cfg)),
            )
            logger.info("[scheduler] HPC queue layer enabled: %s", layer.config.name)
            return layer
        except Exception as e:  # noqa: BLE001 — never block startup on HPC config
            logger.warning("[scheduler] HPC queue layer disabled: %s", e)
            return None

    @staticmethod
    def _hpc_client_config(cfg: HuginnConfig) -> Any:
        """把 HuginnConfig 的 hpc_* 字段映射成 HPCConfig (连接仍懒建)."""
        from huginn.hpc.client import HPCConfig

        return HPCConfig(
            host=cfg.hpc_host,
            username=cfg.hpc_username,
            scheduler=cfg.hpc_scheduler,  # slurm / pbs
            key_path=cfg.hpc_key_path,
            password=cfg.hpc_password,
            port=cfg.hpc_port,
            remote_work_dir=cfg.remote_work_dir,
            default_queue=cfg.hpc_default_queue,
            gpu_queue=cfg.hpc_gpu_queue,
            queue_map=cfg.hpc_queue_map,
            default_walltime=cfg.hpc_default_walltime,
            default_nodes=cfg.hpc_default_nodes,
            default_ntasks_per_node=cfg.hpc_default_ntasks_per_node,
            default_gpus_per_node=cfg.hpc_default_gpus_per_node,
            max_retries=cfg.hpc_max_retries,
            retry_backoff=cfg.hpc_retry_backoff,
            strict_host_key_checking=cfg.hpc_strict_host_key_checking,
        )

    @staticmethod
    def _hpc_queue_config(cfg: HuginnConfig) -> Any:
        """HPC 队列层的并发/资源上限 (并发沿用 admission policy 的重作业配额)."""
        from huginn.scheduling import AdmissionPolicy, HpcQueueConfig

        return HpcQueueConfig(
            name=cfg.hpc_default_queue or cfg.hpc_host,
            max_concurrent=AdmissionPolicy.from_env().max_concurrent_heavy,
            queue=cfg.hpc_default_queue,
            walltime=cfg.hpc_default_walltime,
            nodes=cfg.hpc_default_nodes,
            ntasks_per_node=cfg.hpc_default_ntasks_per_node,
            gpus_per_node=cfg.hpc_default_gpus_per_node,
        )

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

        model = self._resolve_profile_model(profile, profile_id, thinking, max_tokens)
        prompt, begin_dialogs = self._build_profile_prompt(profile, system_prompt_override)
        emotion_tracker = EmotionTracker(
            profile.persona, workspace=self.config.workspace
        )
        agent = self._instantiate_agent(
            profile=profile,
            profile_id=profile_id,
            thread_id=thread_id,
            prompt=prompt,
            begin_dialogs=begin_dialogs,
            model=model,
            memory_manager=memory_manager,
            emotion_tracker=emotion_tracker,
            approval_callback=approval_callback,
        )
        agent.register_tools_from_registry()
        self._attach_agent_hooks(agent)
        return agent

    def _instantiate_agent(
        self,
        profile: AgentProfileConfig,
        profile_id: str,
        thread_id: str | None,
        prompt: str,
        begin_dialogs: list[tuple[str, str]],
        model: Any,
        memory_manager: Any | None,
        emotion_tracker: Any,
        approval_callback: Callable[[str, str], bool] | None,
    ) -> HuginnAgent:
        """按已解析好的模型/prompt/记忆构造 HuginnAgent (不注册工具与钩子)."""
        return HuginnAgent(
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

    def _resolve_profile_model(
        self,
        profile: AgentProfileConfig,
        profile_id: str,
        thinking: ThinkingIntensity | dict[str, Any] | None,
        max_tokens: int | None,
    ) -> Any:
        """按 profile 的 model_alias 解析模型; 无 alias 也无默认 → ValueError."""
        model_alias = profile.model_alias or self.model_registry.default_alias()
        if not model_alias:
            raise ValueError(
                f"Profile '{profile_id}' has no model_alias and no default model is configured"
            )
        effective_thinking = thinking if thinking is not None else profile.thinking
        return self.model_registry.resolve(
            model_alias,
            thinking=effective_thinking,
            max_tokens=max_tokens,
        )

    def _build_profile_prompt(
        self,
        profile: AgentProfileConfig,
        system_prompt_override: str | None,
    ) -> tuple[str, list[tuple[str, str]]]:
        """组装 system prompt (+ begin_dialogs); override 优先, 否则 persona + 上下文锚."""
        begin_dialogs: list[tuple[str, str]] = []
        if system_prompt_override:
            return system_prompt_override, begin_dialogs

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
        except Exception:  # 防御: 项目上下文载入失败则跳过
            logger.debug("load project context failed", exc_info=True)

        # 反跑题任务锚: 明确本轮目标并锁死, 防止 agent 被中间输出
        # (搜索结果/无关文件)带偏. 惩罚偏离, 约束结尾, 收敛工具使用.
        return f"{prompt}\n\n# Task Anchor\n" + _TASK_ANCHOR, begin_dialogs

    def _attach_agent_hooks(self, agent: HuginnAgent) -> None:
        """挂载个性化 / 异常检测 / PRM / prompt 引导 / 安全 / plan gate 钩子."""
        # 个人定制: 注入共享 StyleLearner, chat() 里会自动学用户语言偏好.
        # 共享单例保证同 workspace 下所有 agent 实例用同一份 profile.
        from huginn.personalization import get_shared_style_learner
        agent.set_style_learner(get_shared_style_learner())

        self._attach_anomaly_hooks(agent)
        self._attach_prm_hook(agent)
        self._attach_prompt_hooks(agent)

        # ISC 反渗透: TVD 结构检测 (USER_PROMPT_SUBMIT) + SafeRedirect 引导 +
        # 跨轮风险追踪 (STOP). 共享 tracker 保证 multi-turn 连续性.
        from huginn.hooks.safety_collapse_hook import register_safety_collapse_hooks
        register_safety_collapse_hooks(agent.hook_manager, self._safety_tracker)

        # Design Plan gate: 调用 vasp/lammps 等执行类工具前,
        # 建议先有用户确认的 plan. Chat 模式下 advisory (不硬阻塞),
        # 避免工作流堵点. autoloop 有自己的 plan 确认机制.
        from huginn.hooks import PRE_TOOL_USE
        from huginn.hooks.design_plan_gate_hook import DesignPlanGateHook
        agent.register_hook(PRE_TOOL_USE, DesignPlanGateHook(mode="advisory"))

    def _attach_anomaly_hooks(self, agent: HuginnAgent) -> None:
        """PRT Level 0/1: 工具输出异常检测 (L1 需 HUGINN_PRT_LEVEL1=1)."""
        # PRT Level 0: 挂上异常检测钩子, 拦截工具输出做登记.
        # 钩子在工具调用时按 hook_manager 动态读取, 注册时机不敏感.
        from huginn.hooks import POST_TOOL_USE, AnomalyDetectionHook
        agent.register_hook(POST_TOOL_USE, AnomalyDetectionHook(self._anomaly_store))

        # PRT Level 1: LLM 异常判定, 默认关. 开启条件 HUGINN_PRT_LEVEL1=1.
        # 每次被观察工具调用都会打一次小模型(deepseek-chat), 有成本, 所以默认关.
        if os.environ.get("HUGINN_PRT_LEVEL1", "0") == "1":
            from huginn.hooks.anomaly_llm_hook import AnomalyLLMHook
            agent.register_hook(POST_TOOL_USE, AnomalyLLMHook(self._anomaly_store))

    def _attach_prm_hook(self, agent: HuginnAgent) -> None:
        """PRM step verifier (P1): 重步工具调完打 0-1 分, 默认关."""
        # 累计低分由 Engine 自己走 should_pause_for_decision.
        # 跟 PRT Level 1 一样每次重步都打一次 deepseek-chat, 有成本.
        if os.environ.get("HUGINN_PRM_VERIFIER", "0") != "1":
            return
        from huginn.hooks import POST_TOOL_USE
        from huginn.runtime.step_verifier import (
            StepVerifierHook,
            make_default_llm_chat_fn,
        )
        prm_llm = make_default_llm_chat_fn()
        if prm_llm is not None:
            agent.register_hook(POST_TOOL_USE, StepVerifierHook(prm_llm))

    def _attach_prompt_hooks(self, agent: HuginnAgent) -> None:
        """Prompt 引导 + 结构化追问 + 工具名校验 (均挂在 USER_PROMPT_SUBMIT)."""
        from huginn.hooks import USER_PROMPT_SUBMIT
        from huginn.hooks.clarify_questions_hook import ClarifyQuestionsHook
        from huginn.hooks.prompt_guidance_hook import PromptGuidanceHook
        from huginn.hooks.tool_name_validation_hook import ToolNameValidationHook

        # Prompt 引导钩子: 用户提问里命中"验证/计算/求解"等关键词时,
        # 强制要求走工具. 纯规则匹配零成本, 默认开, 不需要环境变量.
        agent.register_hook(USER_PROMPT_SUBMIT, PromptGuidanceHook())

        # Questions 机制: 信息不足时(目标模糊/参数缺失/输出未定/过短)
        # 生成结构化追问, 降低无效迭代. 每 thread 最多追问一次.
        agent.register_hook(USER_PROMPT_SUBMIT, ClarifyQuestionsHook())

        # 工具名校验: 用户消息里点名的工具不在可用列表时, 注入 prompt_guidance
        # 让 agent 明确告知用户该工具不存在, 别默默调 ls/web_search 探索.
        agent.register_hook(USER_PROMPT_SUBMIT, ToolNameValidationHook())


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
