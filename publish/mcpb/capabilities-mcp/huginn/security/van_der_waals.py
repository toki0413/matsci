"""范德华气体前向真值执行器 — M5"同接口换工具"的第二个真实(软件)物理工具.

与理想气体 (thermo_system) 共用同一个 ``SensorModelExecutor`` 骨架 — 只需换
``_forward`` 的前向物理定律, 传感器(gauge)/确认/制品生命周期/安全层全部复用.
这正回答了 M5: `同一接口下把 forward 换成另一个物理工具` 是否可行.

状态方程 (真实气体): ``p = nRT/(V - nb) - a·(n/V)²``, 其中 a 为分子间引力气(Pa·m⁶/mol²),
b 为分子本身体积 (m³/mol). 无论 ``heat``/``move`` 之后, 每步末统一起算 p,
且要求 ``V > nb`` (越界即非物理, 健康门控拦截).
"""

from __future__ import annotations

from typing import Any

from huginn.security.actuator_model import ErrorModel, SensorModelExecutor
from huginn.security.behavior_lifecycle import (
    BehaviorArtifact,
    BehaviorLifecycle,
    InstallResult,
)
from huginn.security.thermo_system import _CV, _R, ThermoExecutor
from huginn.security.world_model import PhysicalAction, WorldModel

# 范德华常数 (接近 CO2 量级): a (Pa·m⁶/mol²), b (m³/mol).
_A = 0.3658
_B = 4.5e-5

# 默认状态: 取 V >> nb 使其物理合法, p 为正.
_INITIAL: dict[str, float] = {
    "p": _R * 300.0 / (0.05 - _B) - _A * (1.0 / 0.05) ** 2,
    "V": 0.05,
    "T": 300.0,
    "n": 1.0,
}


def vdw_forward(state: dict[str, Any], action: PhysicalAction) -> dict[str, Any]:
    """范德华气体封闭体系解析前向 (纯函数, 不修改入参).

    ``heat(dq)`` 等容加热 (V 不变, T 升); ``move(v)`` 准静态等温变体积 (T 不变).
    每步末统一起算 ``p = nRT/(V-nb) - a(n/V)²``. 要求 ``V > nb``, 否则 p 无意义
    (会由健康门控拦截).
    """
    s = dict(state)
    t = action.type
    if t == "heat":
        s["T"] = float(s["T"]) + float(action.params.get("dq", 0.0)) / (
            float(s["n"]) * _CV
        )
    elif t == "move":
        s["V"] = float(action.params.get("v", s["V"]))
    nb = _B * float(s["n"])
    vol = max(float(s["V"]) - nb, 1e-12)  # 防除零; 物理性由健康门控校验
    s["p"] = (
        _R * float(s["T"]) * float(s["n"]) / vol
        - _A * (float(s["n"]) / float(s["V"])) ** 2
    )
    return s


class VanDerWaalsWorldModel(WorldModel):
    """世界模型: predict 与执行器共用 vdw_forward; infer_inverse 与理想气一致."""

    def predict(
        self, state_before: dict[str, Any], action: PhysicalAction
    ) -> dict[str, Any]:
        return vdw_forward(state_before, action)

    def infer_inverse(
        self, state_before: dict[str, Any], action: PhysicalAction
    ) -> PhysicalAction | None:
        if action.type == "heat":
            return PhysicalAction("heat", {"dq": -float(action.params.get("dq", 0.0))})
        if action.type == "move":
            return PhysicalAction("move", {"v": float(state_before["V"])})
        return None


class VanDerWaalsExecutor(ThermoExecutor):
    """范德华气体执行器 — 只换 ``_forward`` (前向物理定律), gauge 传感器/骨架复用."""

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
        return vdw_forward(state, action)


# ── M5 制品接缝 (与 thermo_system 同形) ─────────────────────────
VAN_DER_WAALS_CONTRACT_VERSION = 1


def build_vdw_artifact(
    version: int,
    *,
    initial: dict[str, Any] | None = None,
    systematic: float = 0.0,
    sigma: float = 0.0,
) -> BehaviorArtifact:
    return BehaviorArtifact(
        name="van_der_waals_gas",
        version=version,
        contract_version=VAN_DER_WAALS_CONTRACT_VERSION,
        config={
            "a": _A,
            "b": _B,
            "initial": dict(initial or _INITIAL),
            "error_model": {"systematic": systematic, "sigma": sigma},
        },
    )


def vdw_executor_from_artifact(artifact: BehaviorArtifact) -> SensorModelExecutor:
    cfg = artifact.config
    em = ErrorModel(
        systematic=float(cfg["error_model"]["systematic"]),
        sigma=float(cfg["error_model"]["sigma"]),
    )
    return VanDerWaalsExecutor(initial=dict(cfg["initial"]), error_model=em, seed=0)


def vdw_health_check(artifact: BehaviorArtifact) -> bool:
    """物理健康门控: 状态物理合法 (T>0, V>nb>0, p>0) 且 EOS 恒成立."""
    ex = vdw_executor_from_artifact(artifact)
    s = ex.observe()
    if not (s["T"] > 0 and s["V"] > _B * s["n"] and s["p"] > 0 and s["n"] > 0):
        return False
    for action in (
        PhysicalAction("heat", {"dq": 100.0}),
        PhysicalAction("move", {"v": float(s["V"]) * 2}),
    ):
        ex.execute(action)
        s = ex.observe()
        if not (s["T"] > 0 and s["V"] > _B * s["n"] and s["p"] > 0):
            return False
        nb = _B * s["n"]
        lhs = (s["p"] + _A * (s["n"] / s["V"]) ** 2) * (s["V"] - nb)
        rhs = _R * s["T"] * s["n"]
        if abs(lhs - rhs) > 1e-6 * max(lhs, rhs):
            return False
    return True


def install_vdw(
    lifecycle: BehaviorLifecycle, artifact: BehaviorArtifact
) -> InstallResult:
    return lifecycle.install(
        artifact, health_check=lambda _v: vdw_health_check(artifact)
    )
