"""统一 opt-out 开关层.

agent 的增强功能散落在各模块 (投机执行 / provenance / 路由拦截 / 主动提问
/ 个性化 / 循环检测 ...), 用户没法一键关. 这里集中管理: 默认全开,
用户可通过配置文件 / 环境变量 / 运行时 API 关掉任意功能.

优先级 (从低到高):
  1. _DEFAULTS 硬编码默认值
  2. 配置文件 feature_flags 字段 (load_from_config 注入)
  3. 环境变量 HUGINN_FEATURE_<NAME>=false (shared() 初始化时读一次)
  4. 运行时 enable/disable/toggle (内存覆盖, 不写盘)

运行时改动不持久化, 要落盘用 persist_to_config(config, path).
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

logger = logging.getLogger(__name__)


class FeatureFlags:
    """统一 opt-out 开关层. 默认全开, 用户可关."""

    # 所有可关的功能, 默认全 True (privacy_* 三个除外, 互斥)
    _DEFAULTS: dict[str, bool] = {
        "speculator": True,            # 投机执行 (意图预测+工具预热)
        "provenance": True,            # 计算快照
        "tool_call_router": True,      # 重型工具 sanity check
        "clarification": True,         # 主动提问
        "personalization": True,       # 学习用户通信风格
        "loop_detector": True,         # 循环检测
        "external_thinking": False,    # deep_think 外部草稿纸 (默认关, 显式开启才注入系统提示)
        "system_health_monitor": True,  # 系统资源监控 (CPU/内存/磁盘)
        "system_health_auto_fix": False,  # 监控发现异常后自动熔断 (默认关, 只报告)
        # v23 Round 9: 两个 router 之前是 raw env var (HUGINN_CONTEXT_ROUTER /
        # HUGINN_TASK_TOOL_ROUTER), 极端模式 setdefault "1". 现纳入 FeatureFlags
        # 统一接管, 默认关 (普通模式不开), 极端模式通过 FeatureFlags.enable() 开.
        # 注意: 模块代码仍读 env var, FeatureFlags 这里只是登记, 不直接控制.
        # 升级路径: 模块代码改为读 FeatureFlags 后, 删除 env var setdefault.
        "context_router": False,       # P3 信息路径多样性稀疏化 (context_builder)
        "task_tool_router": False,     # task keyword → tool category 动态路由
        # harness 实验性栅栏 (默认 off, 显式开启才生效). 见 huginn/harness/_enabled.py.
        # 开启方式: huginn.toml [feature_flags] 字段, 或环境变量 HUGINN_FEATURE_<NAME>=true.
        "harness_workflow_evolution": False,  # H2: variant bandit 演化回路
        "harness_ood_holdout": False,         # H6: OOD 留出验证 (防背题补丁)
        "harness_significance_gate": False,   # H5: 结果显著性门 (统计检验)
        "harness_adoption_gate": False,       # 严格 gate 模式: RED 不自动采纳 (默认 advisory, 只评分不拦)
        "harness_joint_optimizer": False,     # 联合优化 (phase/block/params 协同)
        "harness_phase_evolve": False,        # 阶段规范演化
        "harness_prompt_patch": False,        # 提示补丁 (跨域提示增强)
        # ---- v24 契约收敛 Round 1: 登记散落的裸 bool env (HUGINN_* = 0/1) ----
        # 这些变量之前在各模块 os.environ.get 裸读, 无统一 schema. 现纳入
        # FeatureFlags 统一登记: 默认值与裸读默认一致, 既可通过
        # HUGINN_FEATURE_<NAME> 关/开, 也通过 _ENV_ALIASES 兼容旧变量名.
        # "module-read" 标记: 模块代码仍读旧变量, FeatureFlags 只登记不做控制;
        # 迁移 read 点后即改为 FeatureFlags 生效 (见 _ENV_ALIASES 注释).
        "json_logs": True,               # JSON 结构化日志 (HUGINN_JSON_LOGS)
        "prompt_cache_control": True,    # prompt-cache control 注入 (HUGINN_PROMPT_CACHE_CONTROL)
        "telemetry": True,               # 遥测采集 (HUGINN_TELEMETRY_ENABLED)
        "bandit_mdp": True,              # bandit MDP 决策 (HUGINN_BANDIT_MDP)
        "belief_update": True,           # 信念更新管线 (HUGINN_BELIEF_UPDATE)
        "belief_darwin": True,           # 信念 Darwin 演化 (HUGINN_BELIEF_DARWIN)
        "belief_mode_switch": True,      # 信念模式切换 (HUGINN_BELIEF_MODE_SWITCH)
        "memory_typing": True,           # 记忆类型标注 (HUGINN_USE_MEMORY_TYPING)
        "crdt_merge": True,              # CRDT 合并 (HUGINN_CRDT_MERGE)
        "crdt_branch_merge": True,       # CRDT 分支合并 (HUGINN_CRDT_BRANCH_MERGE)
        "ising_rerank": True,            # Ising 重排 (HUGINN_ISING_RERANK)
        "hils_attention": True,          # HILS 注意力 (HUGINN_HILS_ATTENTION)
        "ising_frontier": True,          # Ising 前沿 (HUGINN_ISING_FRONTIER)
        "fts_auto_rebuild": True,        # 全文检索自动重建 (HUGINN_FTS_AUTO_REBUILD)
        "privacy_redact_secrets": True,  # 检测到密钥时脱敏 (HUGINN_PRIVACY_REDACT_SECRETS)
        "extreme_dispatch": False,       # 极端模式分发 (HUGINN_EXTREME_DISPATCH)
        "three_cabin": False,            # 三舱模型 (HUGINN_USE_THREE_CABIN)
        "use_cognitive_map": False,      # 认知地图 (HUGINN_USE_COGNITIVE_MAP)
        "use_atomworld": False,          # AtomWorld 环境 (HUGINN_USE_ATOMWORLD)
        "persistent_terminal": False,    # 持久化终端 (HUGINN_PERSISTENT_TERMINAL)
        "world_model": True,             # 世界模型 (HUGINN_WORLD_MODEL)
        "curiosity_hint": True,          # 好奇心提示 (HUGINN_CURIOSITY_HINT)
        "privacy_block_on_secrets": False,  # 检测到密钥时阻断 (HUGINN_PRIVACY_BLOCK_ON_SECRETS)
        # P1#1 (v25): 假设维度/方法族/失败类型 关键词匹配 → LLM 语义判定.
        # 默认关 (与 harness 实验栅栏同款): 显式开启 + 有 model provider 才生效,
        # 无 model / 异常 / 输出非法标签时优雅降级回关键词匹配, 行为向后兼容.
        "hypothesis_llm_semantic": False,  # LLM 语义判定 (huginn/autoloop/hypothesis_semantic.py)
        # 隐私三档, 互斥. PrivacyGuard.set_level 负责保证同时只一个 True.
        # privacy_off 仅由 set_level 维护互斥, 外部设置无效.
        "privacy_off": True,           # 不脱敏 (默认)
        "privacy_redact": False,       # 脱敏后发云端
        "privacy_local_only": False,   # 完全本地, 不发云端
    }

    # 旧裸读 env 变量名 → flag 名. 迁移 read 点后仍保留旧变量兼容:
    # 用户设 HUGINN_USE_THREE_CABIN=1 与设 HUGINN_FEATURE_THREE_CABIN=true 等价.
    # 迁移完对应 read 点后, 该别名可保留 (向后兼容已有 shell/.env 配置).
    _ENV_ALIASES: dict[str, str] = {
        "HUGINN_JSON_LOGS": "json_logs",
        "HUGINN_PROMPT_CACHE_CONTROL": "prompt_cache_control",
        "HUGINN_TELEMETRY_ENABLED": "telemetry",
        "HUGINN_BANDIT_MDP": "bandit_mdp",
        "HUGINN_USE_MEMORY_TYPING": "memory_typing",
        "HUGINN_CRDT_BRANCH_MERGE": "crdt_branch_merge",
        "HUGINN_USE_THREE_CABIN": "three_cabin",
        "HUGINN_USE_COGNITIVE_MAP": "use_cognitive_map",
        "HUGINN_USE_ATOMWORLD": "use_atomworld",
        "HUGINN_PERSISTENT_TERMINAL": "persistent_terminal",
        "HUGINN_WORLD_MODEL": "world_model",
        "HUGINN_CURIOSITY_HINT": "curiosity_hint",
        "HUGINN_PRIVACY_REDACT_SECRETS": "privacy_redact_secrets",
        "HUGINN_PRIVACY_BLOCK_ON_SECRETS": "privacy_block_on_secrets",
        # 与各模块裸读点对齐的旧变量名, 迁移 read 点后旧配置仍生效.
        "HUGINN_BELIEF_UPDATE": "belief_update",
        "HUGINN_BELIEF_DARWIN": "belief_darwin",
        "HUGINN_BELIEF_MODE_SWITCH": "belief_mode_switch",
        "HUGINN_EXTREME_DISPATCH": "extreme_dispatch",
        "HUGINN_TASK_TOOL_ROUTER": "task_tool_router",
        "HUGINN_CONTEXT_ROUTER": "context_router",
        "HUGINN_ISING_FRONTIER": "ising_frontier",
        "HUGINN_ISING_RERANK": "ising_rerank",
        "HUGINN_HILS_ATTENTION": "hils_attention",
        "HUGINN_FTS_AUTO_REBUILD": "fts_auto_rebuild",
        "HUGINN_CRDT_MERGE": "crdt_merge",
    }

    # 给 list_flags 用的功能描述
    _DESCRIPTIONS: dict[str, str] = {
        "speculator": "投机执行 (意图预测+工具预热)",
        "provenance": "计算 provenance 快照",
        "tool_call_router": "重型工具 sanity check 路由",
        "clarification": "agent 主动向用户提问",
        "personalization": "学习用户通信风格",
        "loop_detector": "对话循环检测",
        "external_thinking": "外部草稿纸: 注入 deep_think 指令, 让模型动手前先写分析 (默认关)",
        "system_health_monitor": "系统资源监控 (CPU/内存/磁盘)",
        "system_health_auto_fix": "监控异常后自动熔断工具 (默认关)",
        "context_router": "P3 信息路径多样性稀疏化 (context_builder, 默认关)",
        "task_tool_router": "task keyword → tool category 动态路由 (默认关)",
        "harness_workflow_evolution": "H2 variant bandit 演化回路 (实验性, 默认关)",
        "harness_ood_holdout": "H6 OOD 留出验证, 防背题补丁 (实验性, 默认关)",
        "harness_significance_gate": "H5 结果显著性门, 统计检验 (实验性, 默认关)",
        "harness_adoption_gate": "严格 gate 模式: RED 不自动采纳 (实验性, 默认 advisory 只评分不拦)",
        "harness_joint_optimizer": "联合优化 phase/block/params (实验性, 默认关)",
        "harness_phase_evolve": "阶段规范演化 (实验性, 默认关)",
        "harness_prompt_patch": "提示补丁, 跨域提示增强 (实验性, 默认关)",
        "json_logs": "JSON 结构化日志 (HUGINN_JSON_LOGS)",
        "prompt_cache_control": "prompt-cache control 注入 (HUGINN_PROMPT_CACHE_CONTROL)",
        "telemetry": "遥测采集 (HUGINN_TELEMETRY_ENABLED)",
        "bandit_mdp": "bandit MDP 决策 (HUGINN_BANDIT_MDP)",
        "belief_update": "信念更新管线 (HUGINN_BELIEF_UPDATE)",
        "belief_darwin": "信念 Darwin 演化 (HUGINN_BELIEF_DARWIN)",
        "belief_mode_switch": "信念模式切换 (HUGINN_BELIEF_MODE_SWITCH)",
        "memory_typing": "记忆类型标注 (HUGINN_USE_MEMORY_TYPING)",
        "crdt_merge": "CRDT 合并 (HUGINN_CRDT_MERGE)",
        "crdt_branch_merge": "CRDT 分支合并 (HUGINN_CRDT_BRANCH_MERGE)",
        "ising_rerank": "Ising 重排 (HUGINN_ISING_RERANK)",
        "hils_attention": "HILS 注意力 (HUGINN_HILS_ATTENTION)",
        "ising_frontier": "Ising 前沿 (HUGINN_ISING_FRONTIER)",
        "fts_auto_rebuild": "全文检索自动重建 (HUGINN_FTS_AUTO_REBUILD)",
        "privacy_redact_secrets": "检测到密钥时脱敏 (HUGINN_PRIVACY_REDACT_SECRETS)",
        "extreme_dispatch": "极端模式分发 (HUGINN_EXTREME_DISPATCH)",
        "three_cabin": "三舱模型 (HUGINN_USE_THREE_CABIN)",
        "use_cognitive_map": "认知地图 (HUGINN_USE_COGNITIVE_MAP)",
        "use_atomworld": "AtomWorld 环境 (HUGINN_USE_ATOMWORLD)",
        "persistent_terminal": "持久化终端 (HUGINN_PERSISTENT_TERMINAL)",
        "world_model": "世界模型 (HUGINN_WORLD_MODEL)",
        "curiosity_hint": "好奇心提示 (HUGINN_CURIOSITY_HINT)",
        "privacy_block_on_secrets": "检测到密钥时阻断 (HUGINN_PRIVACY_BLOCK_ON_SECRETS)",
        "hypothesis_llm_semantic": "假设维度/方法族/失败类型 LLM 语义判定 (P1#1, 默认关, 优雅降级)",
        "privacy_off": "隐私级别: off (不脱敏, 默认. 仅由 set_level 维护互斥, 外部设置无效)",
        "privacy_redact": "隐私级别: redact (脱敏后发云端)",
        "privacy_local_only": "隐私级别: local_only (完全本地)",
    }

    _singleton_lock = threading.Lock()
    _singleton: FeatureFlags | None = None

    def __init__(self) -> None:
        self._lock = threading.RLock()
        # 三层覆盖, 优先级: defaults < config < env < runtime
        self._config_overrides: dict[str, bool] = {}
        self._env_overrides: dict[str, bool] = {}
        self._runtime_overrides: dict[str, bool] = {}
        # 启动时读一次环境变量
        self._load_env_overrides()

    @classmethod
    def shared(cls) -> FeatureFlags:
        """进程级单例. 首次调用读一次环境变量, 之后复用."""
        with cls._singleton_lock:
            if cls._singleton is None:
                cls._singleton = cls()
            return cls._singleton

    # ------------------------------------------------------------------ 查询

    def is_enabled(self, feature: str) -> bool:
        """查开关. 未知 feature 保守返回 True 并打 warning."""
        with self._lock:
            if feature in self._runtime_overrides:
                return self._runtime_overrides[feature]
            if feature in self._env_overrides:
                return self._env_overrides[feature]
            if feature in self._config_overrides:
                return self._config_overrides[feature]
            if feature in self._DEFAULTS:
                return self._DEFAULTS[feature]
        # 未知 feature: 保守开, 但提醒一下
        logger.warning("unknown feature flag '%s', treating as enabled", feature)
        return True

    def list_flags(self) -> list[dict[str, Any]]:
        """返回所有 flag 的当前状态, 按 _DEFAULTS 顺序."""
        with self._lock:
            result: list[dict[str, Any]] = []
            for name, default_val in self._DEFAULTS.items():
                result.append({
                    "name": name,
                    "enabled": self._effective_value_unlocked(name),
                    "description": self._DESCRIPTIONS.get(name, ""),
                    "default": default_val,
                })
            return result

    def to_dict(self) -> dict[str, bool]:
        """序列化当前所有 flag 的有效状态."""
        with self._lock:
            return {
                name: self._effective_value_unlocked(name)
                for name in self._DEFAULTS
            }

    # ------------------------------------------------------------------ 修改

    def enable(self, feature: str) -> None:
        """运行时打开某个功能."""
        self._set_runtime(feature, True)

    def disable(self, feature: str) -> None:
        """运行时关掉某个功能."""
        self._set_runtime(feature, False)

    def toggle(self, feature: str, enabled: bool) -> None:
        """显式设某个功能的开关."""
        self._set_runtime(feature, bool(enabled))

    def reset(self, feature: str | None = None) -> None:
        """重置到默认. 传 None 重置全部运行时覆盖."""
        with self._lock:
            if feature is None:
                self._runtime_overrides.clear()
            else:
                self._runtime_overrides.pop(feature, None)

    # ------------------------------------------------------------------ 配置注入

    def load_from_config(self, config: Any) -> None:
        """从 HuginnConfig.feature_flags 字段加载覆盖.

        config 没有 feature_flags 字段就当空覆盖, 不报错.
        """
        try:
            raw = getattr(config, "feature_flags", None) or {}
        except Exception:
            raw = {}
        if not isinstance(raw, dict):
            return
        with self._lock:
            # 只接受 _DEFAULTS 里已知的 flag, 防止外部塞脏数据
            self._config_overrides = {
                k: bool(v) for k, v in raw.items() if k in self._DEFAULTS
            }

    def persist_to_config(self, config: Any, path: Any) -> None:
        """把运行时覆盖写回 config.feature_flags 并落盘.

        config 需要是 HuginnConfig 实例 (有 save 方法). 已有的
        feature_flags 字段会合并, 不会清掉没动过的.
        """
        import pathlib

        target = pathlib.Path(path)
        with self._lock:
            snapshot = dict(self._runtime_overrides)
        try:
            existing = getattr(config, "feature_flags", None) or {}
            if not isinstance(existing, dict):
                existing = {}
            existing.update(snapshot)
            config.feature_flags = existing  # type: ignore[attr-defined]
            config.save(target, format="toml")
        except Exception as exc:
            logger.warning("persist_to_config failed: %s", exc)
            raise

    # ------------------------------------------------------------------ 内部

    def _set_runtime(self, feature: str, value: bool) -> None:
        with self._lock:
            self._runtime_overrides[feature] = value

    def _effective_value_unlocked(self, feature: str) -> bool:
        """计算最终生效值 (调用方必须持锁)."""
        if feature in self._runtime_overrides:
            return self._runtime_overrides[feature]
        if feature in self._env_overrides:
            return self._env_overrides[feature]
        if feature in self._config_overrides:
            return self._config_overrides[feature]
        return self._DEFAULTS.get(feature, True)

    @staticmethod
    def _parse_env_value(raw: str) -> bool:
        """把环境变量值解析成布尔. true/1/yes/on → True, 其余含无法识别的
        非空值保守当关 (False). 空串由调用方跳过, 不产生覆盖."""
        s = raw.strip().lower()
        return s in ("true", "1", "yes", "on")

    def _load_env_overrides(self) -> None:
        """读 HUGINN_FEATURE_<NAME> 环境变量. 大写名.

        false/0/no/off → 关, true/1/yes/on → 开, 其他非空值保守当关.
        同时兼容旧裸读变量名 (见 _ENV_ALIASES): 迁移 read 点后旧变量仍生效,
        已有 shell/.env 配置不因迁移而失效.
        """
        for name in self._DEFAULTS:
            env_name = f"HUGINN_FEATURE_{name.upper()}"
            raw = os.environ.get(env_name)
            if raw is not None and raw.strip():
                self._env_overrides[name] = self._parse_env_value(raw)
        for alias, name in self._ENV_ALIASES.items():
            raw = os.environ.get(alias)
            if raw is not None and raw.strip():
                self._env_overrides[name] = self._parse_env_value(raw)
