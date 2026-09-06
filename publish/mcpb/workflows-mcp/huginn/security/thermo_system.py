"""热力学前向真值执行器 — 方案A 首个真实(软件)物理执行器.

microduck sim2real: 一个物理执行器 = 前向真值 + 传感器读数视图 + 域随机化 + 标定.
这里把前向真值从"规则表"升级为**第一性原理的理想气体状态方程 (pV = nRT)** 在
封闭体系 (n 固定) 上的确定性演化 — 纯 Python、毫秒级、CI 可跑, 作为未来换
VASP/DFT 真实计算的模板 (届时只替换 ``_forward`` 与配套世界模型, 其余层次复用).

可动作 (可逆, ``IdealGasWorldModel.infer_inverse`` 能给出精确逆):
- ``heat(dq)`` : 等容加热, T 与 p 升 (逆 = ``heat(-dq)`` 冷却, 精确还原).
- ``move(v)``  : 准静态等温变体积到目标 v, p 变 (逆 = ``move(V_prior)``, 精确还原).

传感器 (gauge 型): 压力表 + 温度计在输出侧读 ``p``/``T``, 各自带 systematic +
零均值 sigma 偏置 — 用 ``_apply_bias`` 覆写默认"守恒 dec/inc"为"每通道 gauge".
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

# 理想气体常数 J/(mol·K)
_R = 8.31446261815324
# 单原子理想气体定容摩尔热容 ≈ 3/2 R, J/(mol·K)
_CV = 1.5 * _R

# 标准状态 (1 mol, STP): V = 0.022414 m³, T = 273.15 K → p ≈ 101325 Pa.
_INITIAL: dict[str, float] = {
    "p": _R * 273.15 / 0.022414,
    "V": 0.022414,
    "T": 273.15,
    "n": 1.0,
}


def forward(state: dict[str, Any], action: PhysicalAction) -> dict[str, Any]:
    """理想气体封闭体系解析前向: 返回动作执行后的世界状态 (纯函数, 不修改入参).

    - ``heat(dq)``: 等容加热 ``dT = dq/(n·cv)``, V 不变, p = nRT/V 随之升.
    - ``move(v)`` : 准静态等温变体积, T、n 不变, p = nRT/V 反比于 V.
    每步末统一起算 p = nRT/V 保证 pV = nRT 恒成立 (第一性原理约束).
    """
    s = dict(state)
    t = action.type
    if t == "heat":
        dq = float(action.params.get("dq", 0.0))
        s["V"] = float(s["V"])  # 等容
        s["T"] = float(s["T"]) + dq / (float(s["n"]) * _CV)
    elif t == "move":
        s["V"] = float(action.params.get("v", s["V"]))
    s["p"] = _R * float(s["T"]) * float(s["n"]) / float(s["V"])
    return s


class IdealGasWorldModel(WorldModel):
    """世界模型: predict 与执行器共用同一解析前向真值; infer_inverse 给可逆动作精确逆."""

    def predict(
        self, state_before: dict[str, Any], action: PhysicalAction
    ) -> dict[str, Any]:
        return forward(state_before, action)

    def infer_inverse(
        self, state_before: dict[str, Any], action: PhysicalAction
    ) -> PhysicalAction | None:
        if action.type == "heat":
            return PhysicalAction("heat", {"dq": -float(action.params.get("dq", 0.0))})
        if action.type == "move":
            return PhysicalAction("move", {"v": float(state_before["V"])})
        return None


class ThermoExecutor(SensorModelExecutor):
    """理想气体执行器 — gauge 型传感器 (压力表/温度计), systematic+sigma 施于 p/T."""

    def __init__(
        self,
        fail_on: set[str] | None = None,
        *,
        initial: dict[str, Any] | None = None,
        error_model=None,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            fail_on=fail_on,
            initial=initial or dict(_INITIAL),
            error_model=error_model,
            seed=seed,
        )

    # 前向真值: 与 world model 共用同一解析源 (单一事实来源).
    def _forward(self, state: dict[str, Any], action: PhysicalAction) -> dict[str, Any]:
        return forward(state, action)

    # gauge 传感器无"守恒 dec/inc"语义 → 返回空; 实际偏置由 _apply_bias 每通道加.
    def _sensor_keys(self, action_type: str) -> tuple[str, str]:
        del action_type
        return ("", "")

    def _noise_eligible(self, action: PhysicalAction) -> bool:
        del action
        return True  # 传感器永远在读, 注入 gauge 偏置

    def _apply_bias(
        self,
        state: dict[str, Any],
        action_type: str,
        magnitude: float,
    ) -> dict[str, Any]:
        """gauge 读数偏置: 同一幅值同时抬高压强表和温度计的读数 (每通道 +=).
        覆写默认的守恒 dec/inc 语义 — 真实仪器就是"偏置叠加在各自读数上"."""
        del action_type
        view = dict(state)
        for key in ("p", "T"):
            if isinstance(view.get(key), int | float):
                view[key] = float(view.get(key, 0.0)) + magnitude
        return view


# ═══════════════════════════════════════════════════════════════════
# M3 生命周期接线 (microduck "releases swapped not patched"):
# 该执行器的"策略/参数"以**自包含制品**交付, 走 BehaviorLifecycle 安装,
# 用**物理健康门控**验证 (EOS 恒成立 + 状态物理合法), 不合格自动回滚.
# ═══════════════════════════════════════════════════════════════════

# 该工具制品契约(参数槽)版本 — 运行时据此握手 (microduck model_api).
THERMO_CONTRACT_VERSION = 1


def build_thermo_artifact(
    version: int,
    *,
    cv: float = _CV,
    initial: dict[str, Any] | None = None,
    systematic: float = 0.0,
    sigma: float = 0.0,
) -> BehaviorArtifact:
    """把热力学执行器参数打包成自包含制品 (config 烘焙, 部署侧不依赖外部上下文)."""
    return BehaviorArtifact(
        name="ideal_gas_chamber",
        version=version,
        contract_version=THERMO_CONTRACT_VERSION,
        config={
            "cv": cv,
            "initial": dict(initial or _INITIAL),
            "error_model": {"systematic": systematic, "sigma": sigma},
        },
    )


def executor_from_artifact(artifact: BehaviorArtifact) -> ThermoExecutor:
    """由制品 config 实例化执行器 (归一化/参数已烘焙进制品, 不靠运行时补)."""
    cfg = artifact.config
    em = ErrorModel(
        systematic=float(cfg["error_model"]["systematic"]),
        sigma=float(cfg["error_model"]["sigma"]),
    )
    return ThermoExecutor(
        initial=dict(cfg["initial"]),
        error_model=em,
        seed=0,
    )


def thermo_health_check(artifact: BehaviorArtifact) -> bool:
    """物理健康门控: 用制品配置实例化并跑一次可逆探查, 校验 EOS 恒成立 + 状态合法.

    任一状态量为非正, 或任一步违反 pV = nRT (超容差), 即判不健康 → 触发回滚.
    """
    ex = executor_from_artifact(artifact)
    s = ex.observe()
    ok: bool = all(s.get(k, 0.0) > 0 for k in ("p", "V", "T", "n"))
    for action in (
        PhysicalAction("heat", {"dq": 100.0}),
        PhysicalAction("move", {"v": float(s["V"]) * 2}),
    ):
        ex.execute(action)
        s = ex.observe()
        if not (s["p"] > 0 and s["V"] > 0 and s["T"] > 0):
            return False
        lhs = s["p"] * s["V"]
        rhs = _R * s["T"] * s["n"]
        if abs(lhs - rhs) > 1e-6 * max(lhs, rhs):
            return False
    return bool(ok)


def install_thermo(
    lifecycle: BehaviorLifecycle,
    artifact: BehaviorArtifact,
) -> InstallResult:
    """把热力学制品装进生命周期, 用该制品自身的物理健康检查做健康门控."""
    return lifecycle.install(
        artifact, health_check=lambda _v: thermo_health_check(artifact)
    )
