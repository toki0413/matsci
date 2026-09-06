"""一维力学振子前向真值执行器 — 跨物理域证明"不止 VASP、也不只热力学".

与理想气体/范德华 (热力学域) 共用同一个 ``SensorModelExecutor`` 骨架, 但这是
**不同物理域** (经典力学/运动学): 状态 ``{x, v}``, 动作 ``kick(dv)``(冲量改速度) 与
``displace(dx)``(位移). 这证明 Physics-AI 后端是**领域无关**的 — Huginn 的
真实计算工具 (DFT/MD/CFD/FEM/QM…) 都能以同种 ``ToolSpec`` 注册接入, 非单一工具.

对计算类工具,"传感器模型"= 数值/方法不确定性 (这里用位置/速度 gauge 偏置示意).
"""

from __future__ import annotations

from typing import Any

from huginn.security.actuator_model import ErrorModel, SensorModelExecutor
from huginn.security.behavior_lifecycle import (
    BehaviorArtifact,
    BehaviorLifecycle,
    InstallResult,
)
from huginn.security.world_model import PhysicalAction, WorldModel

# 物理合法边界 (超出即非物理, 健康门控拦截).
_X_MAX = 100.0
_V_MAX = 100.0

_INITIAL: dict[str, float] = {"x": 0.0, "v": 0.0}


def osc_forward(state: dict[str, Any], action: PhysicalAction) -> dict[str, Any]:
    """一维运动学前向 (纯函数): ``kick(dv)`` 瞬时冲量改速度, ``displace(dx)`` 位移."""
    s = dict(state)
    t = action.type
    if t == "kick":
        s["v"] = float(s["v"]) + float(action.params.get("dv", 0.0))
    elif t == "displace":
        s["x"] = float(s["x"]) + float(action.params.get("dx", 0.0))
    return s


class OscillatorWorldModel(WorldModel):
    """世界模型: predict 与执行器共用 osc_forward; 可逆动作 (kick/displace) 给精确逆."""

    def predict(
        self, state_before: dict[str, Any], action: PhysicalAction
    ) -> dict[str, Any]:
        return osc_forward(state_before, action)

    def infer_inverse(
        self, state_before: dict[str, Any], action: PhysicalAction
    ) -> PhysicalAction | None:
        if action.type == "kick":
            return PhysicalAction("kick", {"dv": -float(action.params.get("dv", 0.0))})
        if action.type == "displace":
            return PhysicalAction(
                "displace", {"dx": -float(action.params.get("dx", 0.0))}
            )
        return None


class OscillatorExecutor(SensorModelExecutor):
    """一维力学执行器 — 位置/速度 gauge 传感器 (偏置叠加在 x,v 读数上)."""

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
            initial=initial or dict(_INITIAL),
            error_model=error_model,
            seed=seed,
        )

    def _forward(self, state: dict[str, Any], action: PhysicalAction) -> dict[str, Any]:
        return osc_forward(state, action)

    def _sensor_keys(self, action_type: str) -> tuple[str, str]:
        del action_type
        return ("", "")

    def _noise_eligible(self, action: PhysicalAction) -> bool:
        del action
        return True

    def _apply_bias(
        self,
        state: dict[str, Any],
        action_type: str,
        magnitude: float,
    ) -> dict[str, Any]:
        del action_type
        view = dict(state)
        for key in ("x", "v"):
            if isinstance(view.get(key), int | float):
                view[key] = float(view.get(key, 0.0)) + magnitude
        return view


# ── 制品接缝 (同形) ─────────────────────────────────────────────
OSCILLATOR_CONTRACT_VERSION = 1


def build_osc_artifact(
    version: int,
    *,
    initial: dict[str, Any] | None = None,
    systematic: float = 0.0,
    sigma: float = 0.0,
) -> BehaviorArtifact:
    return BehaviorArtifact(
        name="mechanics_oscillator",
        version=version,
        contract_version=OSCILLATOR_CONTRACT_VERSION,
        config={
            "x_max": _X_MAX,
            "v_max": _V_MAX,
            "initial": dict(initial or _INITIAL),
            "error_model": {"systematic": systematic, "sigma": sigma},
        },
    )


def osc_executor_from_artifact(artifact: BehaviorArtifact) -> SensorModelExecutor:
    cfg = artifact.config
    em = ErrorModel(
        systematic=float(cfg["error_model"]["systematic"]),
        sigma=float(cfg["error_model"]["sigma"]),
    )
    return OscillatorExecutor(initial=dict(cfg["initial"]), error_model=em, seed=0)


def osc_health_check(artifact: BehaviorArtifact) -> bool:
    """物理健康门控: 状态有界 (|x|<=x_max, |v|<=v_max), 探测动作后仍在界内."""
    cfg = artifact.config
    ex = osc_executor_from_artifact(artifact)
    x_max, v_max = float(cfg["x_max"]), float(cfg["v_max"])
    if not (abs(ex.observe()["x"]) <= x_max and abs(ex.observe()["v"]) <= v_max):
        return False
    for action in (
        PhysicalAction("kick", {"dv": 10.0}),
        PhysicalAction("displace", {"dx": 5.0}),
    ):
        ex.execute(action)
        s = ex.observe()
        if not (abs(s["x"]) <= x_max and abs(s["v"]) <= v_max):
            return False
    return True


def install_osc(
    lifecycle: BehaviorLifecycle, artifact: BehaviorArtifact
) -> InstallResult:
    return lifecycle.install(
        artifact, health_check=lambda _v: osc_health_check(artifact)
    )
