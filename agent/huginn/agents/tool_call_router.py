"""最简路径决策路由 —— 在重型工具调用前做一道 sanity check.

ToolCallBudget 解决的是 "调太多次" 的问题, 这个 router 解决的是
"该不该调" 的问题. 常见失败: 用户问硅的带隙, LLM 起手就调 vasp_tool
跑 DFT, 而不是先查常量. 这里拦一道, 强制 LLM 先把轻量路径走一遍,
确认重型仿真确实必要后再放行.

设计要点:
  - 轻量工具调用一律放行, 只记录已尝试过的工具名
  - 重型工具调用先看是否已试过轻量替代, 没试过就拦下并把替代方案
    塞进 error 信息返回给 LLM
  - LLM 拿到拦下信息后可以: 1) 先调轻量工具 2) 在 tool_input 里加
    `__confirm_heavy=true` 显式跳过检查 (不硬拦, 避免误伤)
  - 路由状态按单轮 agent chat 生命周期管理, 跟 ToolCallBudget 一样
    由 agent 在 chat() 开头 set 进 ToolAdapter, 结束后清掉
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# 重型工具: 单次调用就要烧掉几小时 CPU/GPU 的仿真, 必须先确认无替代
# ml_potential_tool 只在 "train" 动作算重型, predict 不算 —— 这层判断
# 由 should_allow 看 tool_input["action"] 实现, 这里 name 先列上
# Populated by _rebuild_router_tables() from ToolProfile metadata.
HEAVY_TOOLS: set[str] = set()

# 轻量工具: 查询/解析/数值计算, 几秒内返回, 调用一律放行, 只记录
# Populated by _rebuild_router_tables() from ToolProfile metadata.
LIGHT_TOOLS: set[str] = set()

# 重型工具 → 推荐先试的轻量替代 (按优先级排序)
# Populated by _rebuild_router_tables() from ToolProfile metadata.
LIGHT_ALTERNATIVES: dict[str, list[str]] = {}

# ml_potential_tool 的 predict 动作不算重型, 只有 train 才拦
# Populated by _rebuild_router_tables() from ml_potential_tool.heavy_actions.
_MLP_HEAVY_ACTIONS: set[str] = set()


def _rebuild_router_tables() -> None:
    """Rebuild HEAVY_TOOLS / LIGHT_TOOLS / LIGHT_ALTERNATIVES /
    _MLP_HEAVY_ACTIONS in place from ToolProfile metadata.

    Called at the end of register_all_tools() so the router tracks the
    registered tools' declared cost tiers instead of a hand-maintained dict.
    """
    from huginn.tools.registry import ToolRegistry

    heavy = {t.name for t in ToolRegistry._tools.values() if t.cost_tier == "heavy"}
    light = {t.name for t in ToolRegistry._tools.values() if t.cost_tier == "light"}
    alternatives = {
        t.name: list(t.light_alternatives)
        for t in ToolRegistry._tools.values()
        if t.light_alternatives
    }

    mlp = ToolRegistry.get("ml_potential_tool")
    mlp_actions = set(mlp.heavy_actions) if mlp and mlp.heavy_actions else set()

    HEAVY_TOOLS.clear()
    HEAVY_TOOLS.update(heavy)
    LIGHT_TOOLS.clear()
    LIGHT_TOOLS.update(light)
    LIGHT_ALTERNATIVES.clear()
    LIGHT_ALTERNATIVES.update(alternatives)
    _MLP_HEAVY_ACTIONS.clear()
    _MLP_HEAVY_ACTIONS.update(mlp_actions)


class ToolCallRouter:
    """单轮 agent chat 内的轻量决策路由.

    用法::

        router = ToolCallRouter(budget=None)
        allowed, reason = router.should_allow("vasp_tool", {}, {})
        if not allowed:
            return {"error": reason}
        # ... 执行工具 ...
        if tool_name in LIGHT_TOOLS:
            router.record_light_attempt(tool_name)
    """

    def __init__(
        self,
        budget: Any | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        # budget 当前只做引用, 不直接消费 —— 留作未来跟 ToolCallBudget
        # 联动做更精细的拦截 (比如剩余预算少时更激进地拦重型)
        self.budget = budget
        self._logger = logger or globals()["logger"]
        # 本轮已尝试过的轻量工具名, 用来判断 LLM 有没有走过轻量路径
        self._attempted_light: set[str] = set()
        self._lock = __import__("threading").RLock()

    # ------------------------------------------------------------------ API

    def should_allow(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        context: dict[str, Any],
    ) -> tuple[bool, str]:
        """决定本次工具调用能不能放行.

        返回 (allowed, reason):
          - allowed=True  : 放行, reason 为空或 "confirmed heavy"
          - allowed=False : 拦下, reason 写明原因和替代方案, 喂回 LLM

        context 目前没强约束, 保留给上层塞额外信息 (比如 task 类型).
        轻量路径是否已尝试以内部 _attempted_light 为准, context["attempted_light"]
        如果给了也会并进来, 方便外部预热.
        """
        # flag 关掉时直接放行, 不做 sanity check
        try:
            from huginn.feature_flags import FeatureFlags
            if not FeatureFlags.shared().is_enabled("tool_call_router"):
                return True, ""
        except Exception:
            # flag 层挂了不能带挂业务, 继续走原逻辑
            pass

        # 轻量工具一律放行, 不拦
        if tool_name in LIGHT_TOOLS:
            return True, ""

        # ml_potential_tool 只拦训练动作, predict/evaluate 直接放
        if tool_name == "ml_potential_tool":
            action = str(tool_input.get("action", "")).lower()
            if action and action not in _MLP_HEAVY_ACTIONS:
                return True, ""

        # 非重型工具 (code_tool / bash_tool / file_*) 直接放, 由 prompt 约束
        if tool_name not in HEAVY_TOOLS:
            return True, ""

        # ---- 以下都是重型工具 ----

        # ponytail: removed __confirm_heavy bypass — LLM should not be able to
        # self-authorise heavy tool calls. Light-path attempt is the only gate.

        # 判断轻量路径是否已尝试: 内部记录 + context 里给的
        with self._lock:
            attempted = set(self._attempted_light)
        ext = context.get("attempted_light") if context else None
        if isinstance(ext, (list, tuple, set)):
            attempted.update(ext)

        if attempted:
            # 已试过轻量路径, 放行重型仿真
            return True, ""

        # 没试过轻量路径就上重型 → 拦下, 给替代方案
        alternatives = self.get_alternatives(tool_name)
        alt_text = ", ".join(alternatives) if alternatives else "查 knowledge seed / web_search_tool"
        reason = (
            f"Heavy tool {tool_name} requested but no light path attempted. "
            f"Consider: {alt_text}. "
            f"If you confirm heavy simulation is necessary, set "
            f"__confirm_heavy=true in tool_input."
        )
        self._logger.info(
            "router blocked %s: no light path attempted, alternatives=%s",
            tool_name,
            alternatives,
        )
        return False, reason

    def record_light_attempt(self, tool_name: str) -> None:
        """记录一次轻量工具调用, 后续重型工具可凭此放行."""
        if tool_name in LIGHT_TOOLS:
            with self._lock:
                self._attempted_light.add(tool_name)

    def get_alternatives(self, tool_name: str) -> list[str]:
        """返回该重型工具推荐的轻量替代, 没有就给空列表."""
        return list(LIGHT_ALTERNATIVES.get(tool_name, []))

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _is_confirmed_heavy(tool_input: dict[str, Any]) -> bool:
        """LLM 在 tool_input 里塞 __confirm_heavy=true 跳过检查.

        容忍各种 truthy 写法 (True / "true" / "True" / 1), 别因为类型
        差异把 LLM 的确认请求误判成没确认.
        """
        if not isinstance(tool_input, dict):
            return False
        val = tool_input.get("__confirm_heavy", False)
        if isinstance(val, bool):
            return val
        if isinstance(val, (int, float)):
            return val != 0
        if isinstance(val, str):
            return val.strip().lower() in ("true", "yes", "1")
        return False

    def reset(self) -> None:
        """清空本轮轻量路径记录 (下一轮 agent chat 开始时用)."""
        with self._lock:
            self._attempted_light.clear()

    def status(self) -> dict[str, Any]:
        """返回当前路由状态, 方便 debug / telemetry."""
        return {
            "attempted_light": sorted(self._attempted_light),
            "heavy_tools": sorted(HEAVY_TOOLS),
            "light_tools": sorted(LIGHT_TOOLS),
        }

    def __repr__(self) -> str:
        return (
            f"ToolCallRouter(attempted_light={sorted(self._attempted_light)})"
        )
