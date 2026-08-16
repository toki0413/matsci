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
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

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
    """VLA / 仿真 / Mock 执行器: 执行动作并读回当前状态."""

    def execute(self, action: PhysicalAction) -> None: ...

    def observe(self) -> dict[str, Any]: ...


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


@dataclass
class ErrorModel:
    """硬件执行误差模型 — 施加在体积相关量的实测值上.

    真实移液站有系统偏置 (accuracy) + 随机误差 (precision). 有误差时仿真执行器
    的实测状态偏离 world_model 的理想预测, 让状态级感知确认 (``spec.expect`` /
    ``expected``) 在纯软件下就能处理设备噪声:

    - ``systematic``: 系统偏置, 绝对体积单位 (uL), 每次固定叠加.
    - ``sigma``: 随机误差标准差, 绝对体积单位 (uL), 服从高斯.

    仅对 ``aspirate`` / ``dispense`` 这类带体积参数的动作注入; 不影响
    ``mix`` / ``aliquot`` 等离散量.
    """

    systematic: float = 0.0
    sigma: float = 0.0


class SimExecutor(MockExecutor):
    """状态转移仿真执行器 — 算法可微的物理后端占位 (真实 VLA/仿真器可替换).

    前向转移规则与 ``world_model.apply_forward`` 共用同一事实来源
    (``FORWARD_EFFECTS``), 保证"世界模型预演"与"实际仿真结果"一致 —
    感知确认才能对比预期 vs 实际. ``fail_on`` 与 ``MockExecutor`` 同语义.

    ``error_model``: 可注入的硬件误差 (系统偏置 + 随机 sigma, 均以绝对体积单位
    施加在体积相关量上). 有误差时执行结果与 world_model 的理想预测存在偏差,
    让状态级感知确认在纯软件下就能处理真实设备噪声. ``seed`` 保证同 seed 同
    误差序列 (测试可复现). ``error_model=None`` 时行为与旧完全一致 (确定性).
    """

    def __init__(
        self,
        fail_on: set[str] | None = None,
        *,
        initial: dict[str, Any] | None = None,
        error_model: ErrorModel | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(fail_on=fail_on)
        self.state = dict(initial or {"reagent_vol": 100.0, "sample_vol": 0.0, "tube_vol": 0.0})
        self.state.setdefault("mixed", False)
        self.state.setdefault("aliquot_count", 0)
        self.error_model = error_model
        self._rng = random.Random(seed)

    def execute(self, action: PhysicalAction) -> None:
        if action.type in self.fail_on:
            raise RuntimeError(f"sim executor: action {action.type} failed")
        self._apply(action)
        self.log.append(action)

    def _apply(self, action: PhysicalAction) -> None:
        self.state = apply_forward(self.state, action)
        # 体积相关动作注入硬件噪声: 目标增量 +noise, 源减量 -noise (等幅反相,
        # 保持量守恒 − 移错体积而非凭空增减). 物理上 = 一次移液量的测量误差.
        if self.error_model is not None and "vol" in action.params:
            dec_key, inc_key, _ = FORWARD_EFFECTS.get(action.type, ("", "", ""))
            noise = self.error_model.systematic + self._rng.gauss(0.0, self.error_model.sigma)
            if inc_key and isinstance(self.state.get(inc_key), (int, float)):
                self.state[inc_key] = float(self.state.get(inc_key, 0.0)) + noise
            if dec_key and isinstance(self.state.get(dec_key), (int, float)):
                self.state[dec_key] = float(self.state.get(dec_key, 0.0)) - noise


class PhysicalWorkspace:
    """一个可组合的物理实验工作台 (时间可逆 + 空间依赖 + 感知确认)."""

    def __init__(
        self,
        world_model: WorldModel,
        executor: ActionExecutor,
        revertible: RevertibleContext | None = None,
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
            if expected is None and spec is None or spec is not None and not spec.expect:
                expected = predicted

        inverse = self.world_model.infer_inverse(state_before, action)

        self.executor.execute(action)
        self.state = dict(self.executor.observe())

        if spec is not None and spec.expect and not matches_state(spec.expect, self.state):
            raise WorkspaceConfirmError(
                f"感知确认失败: 状态偏离预期 (规格 {spec.id}, 动作 {action.type})"
            )
        elif expected is not None and not matches_state(expected, self.state, tolerance):
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
        return action

    def transaction(self) -> Any:
        """事务边界: 块内异常 → 物理逆按 LIFO 执行, 工作台恢复到块前."""
        return self.revertible.transaction()

    def _run_inverse(self, inverse: dict[str, Any]) -> None:
        """内部: 把一个逆动作字典交给执行器执行 (OP_ACTION 逆的回调)."""
        self.executor.execute(PhysicalAction.from_dict(inverse))
