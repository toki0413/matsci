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
from collections.abc import Callable
from typing import Any, Protocol

from huginn.security.coeffect import CoEffectRegistry
from huginn.security.revertible import RevertibleContext, register_physical_executor
from huginn.security.world_model import PhysicalAction, WorldModel

logger = logging.getLogger(__name__)


class WorkspaceConfirmError(Exception):
    """感知确认失败 — 动作已执行但状态未达预期, 触发事务回滚."""


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


class SimExecutor(MockExecutor):
    """状态转移仿真执行器 — 算法可微的物理后端占位 (真实 VLA/仿真器可替换).

    目前以确定性规则近似 VLA 执行结果: 每个动作按 ``_EFFECTS`` 更新内部状态,
    ``observe()`` 读回当前状态供感知确认. ``fail_on`` 与 ``MockExecutor`` 同语义.
    这是"真实仿真后端"的接缝: 把 ``_EFFECTS`` / ``_apply`` 换成对接 LAMMPS /
    ASE / 真实 VLA 控制器即可, 对外 ``execute``/``observe`` 契约不变.

    示例状态:
      - aspire:   reagent_vol 减少, sample_vol 增加
      - dispense: sample_vol 减少, tube_vol 增加
      - mix:      mixed=True
      - aliquot:  aliquot_count 增加
    """

    _EFFECTS: dict[str, tuple[str, str, str]] = {
        # type -> (dec_key, inc_key, vol_param)
        "aspirate": ("reagent_vol", "sample_vol", "vol"),
        "dispense": ("sample_vol", "tube_vol", "vol"),
        "mix": ("", "mixed", ""),
        "aliquot": ("", "aliquot_count", ""),
    }

    def __init__(self, fail_on: set[str] | None = None, *, initial: dict[str, Any] | None = None) -> None:
        super().__init__(fail_on=fail_on)
        self.state = dict(initial or {"reagent_vol": 100.0, "sample_vol": 0.0, "tube_vol": 0.0})
        self.state.setdefault("mixed", False)
        self.state.setdefault("aliquot_count", 0)

    def execute(self, action: PhysicalAction) -> None:
        if action.type in self.fail_on:
            raise RuntimeError(f"sim executor: action {action.type} failed")
        self._apply(action)
        self.log.append(action)

    def _apply(self, action: PhysicalAction) -> None:
        effect = self._EFFECTS.get(action.type)
        dec_key, inc_key, vol_param = effect if effect else ("", "", "vol")
        if dec_key:
            v = float(action.params.get(vol_param, 0) or 0)
            self.state[dec_key] = max(0.0, float(self.state.get(dec_key, 0.0)) - v)
        if inc_key == "mixed":
            self.state["mixed"] = True
        elif inc_key == "aliquot_count":
            self.state["aliquot_count"] = int(self.state.get("aliquot_count", 0)) + 1
        elif inc_key:
            v = float(action.params.get(vol_param, 0) or 0)
            self.state[inc_key] = float(self.state.get(inc_key, 0.0)) + v


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

    def execute(
        self,
        action: PhysicalAction,
        confirm_key: str | None = None,
    ) -> PhysicalAction:
        """执行物理动作: 世界模型推断逆 → 执行 → 读状态 → 登记逆 → 感知确认.

        确认失败抛 :class:`WorkspaceConfirmError`, 触发外层事务回滚.
        不可逆动作 (infer_inverse 返回 None) 仍执行, 但不登记逆 (无法自动回滚).
        """
        state_before = dict(self.state)
        inverse = self.world_model.infer_inverse(state_before, action)

        self.executor.execute(action)
        self.state = dict(self.executor.observe())

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
