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
        "least_effort_path": True,     # prompt 决策树
        "data_postprocess_rule": True, # 数据后处理规则
        "tool_cache": True,            # 工具缓存
        "parallel_executor": True,     # 并行执行
        "scenario_tool": True,         # 场景预设
        "benchmark": True,             # 文献基准对比
        "uq_propagate": True,          # 误差传播 GUM
        "circuit_breaker": True,       # 熔断器
        "health_dashboard": True,      # 健康仪表盘
        # 隐私三档, 互斥. PrivacyGuard.set_level 负责保证同时只一个 True.
        "privacy_off": True,           # 不脱敏 (默认)
        "privacy_redact": False,       # 脱敏后发云端
        "privacy_local_only": False,   # 完全本地, 不发云端
    }

    # 给 list_flags 用的功能描述
    _DESCRIPTIONS: dict[str, str] = {
        "speculator": "投机执行 (意图预测+工具预热)",
        "provenance": "计算 provenance 快照",
        "tool_call_router": "重型工具 sanity check 路由",
        "clarification": "agent 主动向用户提问",
        "personalization": "学习用户通信风格",
        "loop_detector": "对话循环检测",
        "least_effort_path": "prompt 决策树 (最短路径)",
        "data_postprocess_rule": "数据后处理规则",
        "tool_cache": "工具结果缓存",
        "parallel_executor": "并行工具执行",
        "scenario_tool": "场景预设",
        "benchmark": "文献基准对比",
        "uq_propagate": "误差传播 (GUM)",
        "circuit_breaker": "熔断器",
        "health_dashboard": "健康仪表盘",
        "privacy_off": "隐私级别: off (不脱敏, 默认)",
        "privacy_redact": "隐私级别: redact (脱敏后发云端)",
        "privacy_local_only": "隐私级别: local_only (完全本地)",
    }

    _singleton_lock = threading.Lock()
    _singleton: "FeatureFlags | None" = None

    def __init__(self) -> None:
        self._lock = threading.RLock()
        # 三层覆盖, 优先级: defaults < config < env < runtime
        self._config_overrides: dict[str, bool] = {}
        self._env_overrides: dict[str, bool] = {}
        self._runtime_overrides: dict[str, bool] = {}
        # 启动时读一次环境变量
        self._load_env_overrides()

    @classmethod
    def shared(cls) -> "FeatureFlags":
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

    def _load_env_overrides(self) -> None:
        """读 HUGINN_FEATURE_<NAME> 环境变量. 大写名.

        false/0/no/off → 关, true/1/yes/on → 开, 其他非空值保守当关.
        """
        for name in self._DEFAULTS:
            env_name = f"HUGINN_FEATURE_{name.upper()}"
            raw = os.environ.get(env_name)
            if raw is None:
                continue
            s = raw.strip().lower()
            if s in ("false", "0", "no", "off"):
                self._env_overrides[name] = False
            elif s in ("true", "1", "yes", "on"):
                self._env_overrides[name] = True
            elif s:
                # 无法识别的非空值, 保守关掉
                self._env_overrides[name] = False
