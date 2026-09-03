"""PhysicalWorkspace — 物理世界接入核心结构的编排层 (Cordis 时空可组合).

把核心的两个抽象统一到一个"物理实验工作台":
- **时间可组合** (RevertibleContext): 每个物理动作经世界模型推断逆, 由
  ``track_world_action`` 登记为 OP_ACTION 逆; 事务失败/回滚时, 物理执行器
  按 LIFO 执行逆动作, 把物理世界恢复到批次前.
- **空间可组合** (CoEffectRegistry): 实验协议组件声明 provides/requires;
  依赖缺失时下游自动停用 (degrade), 不停跑无依据的步骤.
- **感知确认**: 动作执行后用 verifier 校验状态 (视觉/力觉/传感器), 失败即
  抛异常触发回滚 — 物理动作"执行了"不等于"做对了".

本层只是把已有核心组合起来, 不修改 revertible/coeffect 本身 (第三次实例化).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import Any, Protocol

from huginn.security.actuator_model import ErrorModel, SensorModelExecutor
from huginn.security.coeffect import CoEffectRegistry
from huginn.security.physics_schema import (
    ActionSpec,
    matches_state,
)
from huginn.security.revertible import RevertibleContext, register_physical_executor
from huginn.security.world_model import (
    FORWARD_EFFECTS,
    ConstraintViolationError,
    PhysicalAction,
    WorldModel,
    apply_forward,
    check_constraints,
)

logger = logging.getLogger(__name__)


class WorkspaceConfirmError(Exception):
    """感知确认失败 — 动作已执行但状态未达预期, 触发事务回滚."""


class DependencyNotMetError(Exception):
    """声明式规格前置依赖不满足 — 动作不应执行.

    ``spec.preconditions`` 中的依赖 key 在此刻不可用 (AvailableDependency 未
    提供或被 degrade), 在 execute 之前拦截, 不产生任何副作用/逆.
    """


class ActionExecutor(Protocol):
    """VLA / 仿真 / Mock 执行器: 执行动作并读回当前状态.

    ``sensor_view``: 把一个动作执行后的**理想世界状态**投影到与该动作实际
    测量(**observe()**)一致的"读数"视角. 这是感知确认的视角一致性钩子 —
    microduck 的核心 sim2real 教训: 编码器读穿齿轮输出侧, 感知确认必须在观测
    同一视角下进行, 否则一个"偏置但正确补偿"的动作会被判为失败. 真实 VLA /
    执行器 adapter 用传感器模型(系统偏置/位姿噪声)实现它; 无偏置时恒等.
    """

    def execute(self, action: PhysicalAction) -> None: ...

    def observe(self) -> dict[str, Any]: ...

    def sensor_view(
        self, state: dict[str, Any], action_type: str
    ) -> dict[str, Any]: ...


class MockExecutor:
    """内存执行器 — 记录动作日志, 可配置失败; 用于测试与演示.

    ``fail_on``: 动作 type 集合, 命中即抛异常 (模拟物理执行失败).
    """

    def __init__(self, fail_on: set[str] | None = None) -> None:
        self.log: list[PhysicalAction] = []
        self.fail_on = set(fail_on or ())
        self.state: dict[str, Any] = {}

    def execute(self, action: PhysicalAction) -> None:
        if action.type in self.fail_on:
            raise RuntimeError(f"mock executor: action {action.type} failed")
        self.log.append(action)

    def observe(self) -> dict[str, Any]:
        return dict(self.state)

    def sensor_view(self, state: dict[str, Any], action_type: str) -> dict[str, Any]:
        """Mock 无传感器模型 — 读数与状态恒等."""
        del action_type
        return dict(state)


class SimExecutor(SensorModelExecutor):
    """状态转移仿真执行器 — 复用可复用执行器骨架 (阶段1 首个工具实例).

    前向真值 = ``world_model.apply_forward`` (规则表, 与 world model 共用唯一
    事实来源, 保证"世界模型预演"与"实际仿真结果"一致); 传感器键取自
    ``FORWARD_EFFECTS``; 执行误差仅在带 ``vol`` 参数的动作上注入 (量守恒).

    真实 VLA / 计算工具后端只需声明自己的 ``_forward`` 与 ``_sensor_keys`` +
    ``error_model``, 即可同享 execute / observe / sensor_view / calibrate 语义.
    """

    def __init__(
        self,
        fail_on: set[str] | None = None,
        *,
        initial: dict[str, Any] | None = None,
        error_model: ErrorModel | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            fail_on=fail_on,
            initial=initial
            or {"reagent_vol": 100.0, "sample_vol": 0.0, "tube_vol": 0.0},
            error_model=error_model,
            seed=seed,
        )
        self.state.setdefault("mixed", False)
        self.state.setdefault("aliquot_count", 0)

    def _forward(self, state: dict[str, Any], action: PhysicalAction) -> dict[str, Any]:
        return apply_forward(state, action)

    def _sensor_keys(self, action_type: str) -> tuple[str, str]:
        dec_key, inc_key, _ = FORWARD_EFFECTS.get(action_type, ("", "", ""))
        return (dec_key, inc_key)

    def _noise_eligible(self, action: PhysicalAction) -> bool:
        # 只对带体积参数的动作注入 (mix/aliquot 为离散量, 不注入).
        return bool(getattr(action, "params", None) and "vol" in action.params)


class PhysicalWorkspace:
    """一个可组合的物理实验工作台 (时间可逆 + 空间依赖 + 感知确认)."""

    def __init__(
        self,
        world_model: WorldModel,
        executor: ActionExecutor,
        revertible: RevertibleContext | None = None,
        *,
        schema: Mapping[str, Any] | None = None,
    ) -> None:
        self.world_model = world_model
        self.executor = executor
        # 复用外部逆上下文 (如 agent 的 ToolContext.revertible) 时, 物理逆进入
        # 统一的 agent 逆栈, 上层整体回滚会一并撤销实验副作用; 否则自建一个.
        self.revertible = revertible if revertible is not None else RevertibleContext()
        self.registry = CoEffectRegistry()
        # 当前物理状态给世界模型推断逆用.
        self.state: dict[str, Any] = dict(getattr(executor, "state", {}))
        # 感知确认器: key -> 校验当前状态是否满足.
        self._confirmers: dict[str, Callable[[], bool]] = {}
        # 世界表征闭环跟踪器 (advisory): 给 schema 则在逐轮后记录状态快照 +
        # 前向预测误差 (回流信号). world_model 作为解析前向真值注入.
        self._world_state = None
        if schema is not None:
            try:
                from huginn.security.world_state import WorldStateTracker

                self._world_state = WorldStateTracker(schema, world_model)
            except Exception:
                logger.debug("world-state tracker init failed", exc_info=True)
        # 全局物理执行器接管 OP_ACTION 逆的执行 (与 register_compensator 同构).
        register_physical_executor(self._run_inverse)

    # ── 空间可组合: 实验协议依赖声明 ─────────────────────────────
    def declare(
        self,
        component_id: str,
        *,
        provides: set[str] | tuple[str, ...] | list[str] = (),
        requires: set[str] | tuple[str, ...] | list[str] = (),
        on_change: Callable[[str, str], None] | None = None,
    ) -> None:
        self.registry.declare(
            component_id, provides=provides, requires=requires, on_change=on_change
        )

    def set_available(self, key: str, available: bool) -> None:
        self.registry.set_available(key, available)

    def is_active(self, component_id: str) -> bool:
        return self.registry.is_active(component_id)

    def is_available(self, key: str) -> bool:
        return self.registry.is_available(key)

    # ── 时间可组合: 物理动作 + 感知确认 ──────────────────────────
    def confirm(self, key: str, verifier: Callable[[], bool]) -> None:
        """登记一个感知确认器: 动作执行后校验 key 对应的状态谓词."""
        self._confirmers[key] = verifier

    def preflight(self, action: PhysicalAction) -> dict[str, Any]:
        """执行前预演: 物理约束校验 + 世界模型前向预测.

        - 违背第一性原理约束 → 抛 :class:`ConstraintViolationError`, 不执行.
        - 通过则返回 ``world_model.predict`` 的预期后状态, 供感知确认对比.
        """
        issues = check_constraints(self.state, action)
        if issues:
            raise ConstraintViolationError(action, issues)
        return self.world_model.predict(self.state, action)

    def execute(
        self,
        action: PhysicalAction,
        confirm_key: str | None = None,
        *,
        preflight: bool = False,
        expected: dict[str, Any] | None = None,
        spec: ActionSpec | None = None,
        tolerance: float = 1e-9,
    ) -> PhysicalAction:
        """执行物理动作: (可选预演) → 世界模型推断逆 → 执行 → 读状态 → 登记逆 → 确认.

        - ``spec``: 声明式动作规格. 提供时:
          - 用 ``spec.expect`` (StepResult 断言列表) 做状态级感知确认, 每字段自带
            相对/绝对容差;
          - 前置依赖 ``spec.preconditions`` 不满足 → 抛 :class:`DependencyNotMetError`,
            不执行.
        - ``preflight=True``: 执行前用世界模型对**当前状态**做约束校验 + 前向
          预测; 约束违规抛 :class:`ConstraintViolationError`, 不执行.
        - ``expected``: 给定的预期后状态 (dict, 统一绝对容差), 与 ``spec`` 互斥;
          执行后与 ``observe()`` 实测状态对比, 超容差抛 :class:`WorkspaceConfirmError`.
        - ``confirm_key``: 原有按 key 的确认器仍最后执行.

        确认失败触发外层事务回滚. 不可逆动作仍执行但不登记逆.
        """
        state_before = dict(self.state)

        if spec is not None:
            for dep in spec.preconditions:
                if not self.is_available(dep):
                    raise DependencyNotMetError(spec.id, dep)
            if expected is not None:
                raise ValueError("expected 与 spec 互斥, 二者只能传其一")

        if preflight:
            # 用当前状态预演: 先约束校验, 再取前向预测 (供 expected 对比).
            predicted = self.preflight(action)
            if (
                expected is None
                and spec is None
                or spec is not None
                and not spec.expect
            ):
                expected = predicted
                # 感知确认须与观测同一视角 (microduck: 编码器读穿输出侧). 预演得到
                # 的是"理想世界状态", 先投影到执行器读数视角再对比 observe(), 否则
                # 一个"偏置但正确"的动作会被误判失败. 真实 adapter 用 sensor_view
                # 建模传感器系统偏置/位姿噪声; 无偏置执行器走恒等, 行为不变.
                sensor_view = getattr(self.executor, "sensor_view", None)
                if sensor_view is not None:
                    expected = sensor_view(expected, action.type)

        inverse = self.world_model.infer_inverse(state_before, action)

        self.executor.execute(action)
        self.state = dict(self.executor.observe())

        if (
            spec is not None
            and spec.expect
            and not matches_state(spec.expect, self.state)
        ):
            raise WorkspaceConfirmError(
                f"感知确认失败: 状态偏离预期 (规格 {spec.id}, 动作 {action.type})"
            )
        elif expected is not None and not matches_state(
            expected, self.state, tolerance
        ):
            raise WorkspaceConfirmError(
                f"感知确认失败: 状态偏离预期 (动作 {action.type})"
            )

        if inverse is not None:
            # state_before 是动作执行前的状态, inverse 是"回到执行前"所需的逆动作.
            self.revertible.track_world_action(
                state_before, action.to_dict(), inverse.to_dict()
            )

        if confirm_key is not None:
            verifier = self._confirmers.get(confirm_key)
            if verifier is not None and not verifier():
                raise WorkspaceConfirmError(
                    f"感知确认失败: {confirm_key} (动作 {action.type})"
                )
        # 世界表征回流 (advisory): 记录本轮状态快照 + 前向预测误差, 供奖励/记忆消费.
        # 失败静默吞掉, 不阻塞实验执行.
        if self._world_state is not None:
            try:
                self._world_state.step(state_before, action)
                self._world_state.observe(self.state)
            except Exception:
                logger.debug("world-state step failed", exc_info=True)
        return action

    def last_world_snapshot(self):
        """最近一轮的世界状态快照 (含 predicted), 无跟踪器时返回 None."""
        return (
            self._world_state.last_snapshot if self._world_state is not None else None
        )

    def last_prediction_error(self) -> float | None:
        """最近一轮的前向预测误差 (预测 vs 实测, 相对 RMS), 回流给奖励用."""
        if self._world_state is None:
            return None
        return self._world_state.last_prediction_error

    def last_prediction_reward(self) -> float | None:
        """最近一轮的前向预测命中奖励 ([0,1]), 喂给 bandit/episodic 的 r_phys."""
        if self._world_state is None:
            return None
        return self._world_state.last_prediction_reward

    def transaction(self) -> Any:
        """事务边界: 块内异常 → 物理逆按 LIFO 执行, 工作台恢复到块前."""
        return self.revertible.transaction()

    def _run_inverse(self, inverse: dict[str, Any]) -> None:
        """内部: 把一个逆动作字典交给执行器执行 (OP_ACTION 逆的回调)."""
        self.executor.execute(PhysicalAction.from_dict(inverse))
