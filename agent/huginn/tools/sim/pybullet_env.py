"""具身仿真后端 —— 把 PyBullet 刚体动力学包装成 ``LawModel`` (world model) 协议.

设计目标 (对齐 research/law_model.py 的 LawModel 契约):
    让科研 explore 的「世界模型 → predict(LawModel 假说) → 真实执行 → reconcile 对账」
    从纯数学对账升级成真物理对账。PyBullet 在这里的角色是 **execute 的物理实现**
    (ground truth 环境), 而非顶层的完整仿真器抽象 —— 这样我们不用发明一套新的
    世界表征, 能直接复用已有的 CSpace / LawState / LawAction / reconcile 链。

接入点:
    - :class:`PyBulletEnv` 实现 :func:`law` / :func:`predict` / :func:`seed`
      （即 research.law_model.LawModel 协议, 但不 import 它以免 tools 层 ->
        research 层反向依赖 —— 见 test_arch_tools_direction 门禁）。
    - ``worldview = "physics_causal"``（对齐 Worldview.PHYSICS_CAUSAL 的字符串值）,
      ``available=False`` 表示 pybullet 未安装, 调用方应 fail-open 跳过。

诚实边界:
    - pybullet 是刚体动力学求解器, 不含真实传感器/控制器/材质。这里提供的是
      **可复现的数值物理真值**, 不是真实世界的替代。
    - 内置少量简单场景 (倒立摆 / 自由落体 / 单臂), 复杂机器人请通过 ``urdf_path``
      注入外部 URDF。内置 URDF 只用于跑通「执行 - 观测 - 对账」闭环。
    - 无头模式 (``DIRECT``) 与有头模式 (``GUI``) 都支持; 测试用 DIRECT 无需显示。

Fail-open 约定:
    - ``import pybullet`` 失败时 ``PyBulletEnv.available=False``, 所有真物理方法
      抛 :class:`PyBulletUnavailable`。上层 (tool) 捕获后置 ``skipped=True``, 不阻断
      agent 主流程 —— 具身仿真是可选项, 不是硬依赖。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

try:  # pybullet 是可选后端 —— 缺失时整个后端降级为不可用, 不阻断导入
    import pybullet  # type: ignore

    _PYBULLET_OK = True
except Exception:  # noqa: BLE001 — 可选依赖缺失仅记录, 不让 tools/sim 导入失败
    pybullet = None  # type: ignore[assignment]
    _PYBULLET_OK = False

if _PYBULLET_OK:
    PYBULLET_VERSION = pybullet.getAPIVersion()
else:
    PYBULLET_VERSION = 0


class PyBulletUnavailable(RuntimeError):  # noqa: N818 — 领域异常名沿用 Unavailable 语义, 不改名以对齐"依状态可恢复"的 fail-open 契约
    """pybullet 未安装或无头环境不可用时抛出. 调用方应 fail-open 处理."""


# 世界模型世界观字符串值 (对齐 research/law_model.Worldview.PHYSICS_CAUSAL)
_PHYSICS_CAUSAL = "physics_causal"

# 内置场景到 LawModel domain 的映射
SCENES: dict[str, dict[str, Any]] = {
    # 倒立摆 (cart-pole): 状态 = [摆角 theta, 角速度 dtheta], 动作 = 车端水平力/N.
    "cartpole": {
        "domain": "robotics.cartpole",
        "state": ("cart_pos", "cart_vel", "pole_theta", "pole_dtheta"),
        "observables": ("pole_theta", "pole_dtheta"),
        "law": (
            "(m_c + m_p) x'' - m_p L θ'' cosθ = F;"
            " L θ'' - g sinθ = -x'' cosθ   (倒立摆两体方程)"
        ),
    },
    # 单自由度自由落体: 最简 ground truth, 用于验证真物理对账链路.
    "free_fall": {
        "domain": "physics.free_fall",
        "state": ("y", "vy"),
        "observables": ("y", "vy"),
        "law": "y(t) = y0 + v0 t - ½ g t² ;  vy(t) = v0 - g t",
    },
    # URDF 多关节场景库 —— 真正的刚体/关节动力学 (pybullet_data 内置 URDF), 用于科研 explore
    # 的具身 ground truth. ``state`` 为关节角全向量 (加载后由关节序展开), ``observables``
    # 按场景显式声明可测的子集. 复杂机器人可经 ``urdf_path`` 注入外部 URDF.
    "kuka_iiwa": {
        "domain": "robotics.arm.kuka_iiwa",
        "state": tuple(f"j{i}" for i in range(7)),   # KUKA iiwa 7 关节角
        "observables": tuple(f"j{i}" for i in range(7)),
        "law": "关节构型 q 在重力 + 保持力矩下的动态:  M(q) q'' + C(q,q') q' + g(q) = τ",
        "urdf": "kuka_iiwa/model.urdf",              # pybullet_data 内置 7R 机械臂
    },
    "humanoid": {
        "domain": "robotics.humanoid",
        "state": tuple(f"j{i}" for i in range(30)),
        "observables": ("root_height",),             # 最可测单标量: 质心离地高度
        "law": "双足在重力下由被动动力学 + 关节力矩维持平衡;  无控制座下必倒地(立不稳)",
        "urdf": "humanoid/humanoid.urdf",            # pybullet_data 内置 30 关节双足
    },
}


@dataclass
class PyBulletEnv:
    """PyBullet 刚体动力学后端, 实现 LawModel 的 law/predict/seed 语义.

    ``predict(state, action)`` 在 pybullet 里 rollout ``n_steps`` 步, 返回末态
    LawState —— 这是「真实执行」对账的 ground truth。
    """

    domain: str = "robotics.cartpole"
    # 每个预测调用的仿真步数 (默认 240 步 ≈ 1 秒 @ 240Hz)
    n_steps: int = 240
    timestep: float = 1.0 / 240.0
    use_gui: bool = False
    gravity: float = -9.81
    # 外部注入的 URDF 路径 (任意 *.urdf). 优先级: action.config["urdf_path"] > self.urdf_path
    # > scene["urdf"] (> 内置地理 pybullet_data). None -> 用场景的内置 URDF.
    urdf_path: str | None = None
    # 注入的 pybullet 模块 (测试可注入 fake)。None -> 真实 pybullet 或抛 PyBulletUnavailable.
    _pb: Any = field(default=None, repr=False)
    available: bool = field(default=False, repr=False)
    # 当前场景配置 (由 seed/with_scene 填入)
    scene: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self._pb is None and not _PYBULLET_OK:
            raise PyBulletUnavailable("pybullet not installed")
        if self._pb is None:
            self._pb = pybullet
        self.available = True
        # 商界属性 —— 服务治理卡片 (world_model_card 读 model/domain/worldview)
        self.model = "PyBulletEnv"
        self.worldview = _PHYSICS_CAUSAL

    # ── LawModel 三件套 ────────────────────────────────────────────

    def seed(self, observation: dict[str, Any]) -> LawState:
        """从外部观测落下初始状态 (LawState 字典向量)."""
        obs = dict(observation)
        obs.setdefault("domain", self.scene.get("domain", self.domain))
        return LawState(obs.get("state") or {}, obs.get("domain", self.domain))

    def predict(self, state: LawState, action: LawAction) -> LawState:
        """对当前状态施加动作, 在 pybullet 里 rollout n_steps, 返回末态.

        ``state`` 是 LawState (初始姿态), ``action.config`` 是控制参数
        (如 cartpole 的水平力 F)。返回末态 LawState —— 即对账的 ground truth。
        """
        pb = self._require_pb()
        scene = self.scene or SCENES.get(self.domain, {})
        # 内置倒立摆专用: 从 state.vector 初始化小车/摆, 施加 action.config["force"].
        if "cartpole" in (scene.get("domain") or self.domain):
            return self._rollout_cartpole(pb, state, action, scene)
        # 通用 fallback: 自由落体解析 (无关节, 直接用运动学)
        if "free_fall" in (scene.get("domain") or self.domain):
            return self._rollout_free_fall(state, action)
        # URDF 多关节场景库: 场景声明了 urdf 或显式注入 urdf_path → 走真关节 rollout
        urdf = action.config.get("urdf_path") or self.urdf_path or scene.get("urdf", "")
        if urdf:
            return self._rollout_urdf(pb, state, action, scene, urdf)
        raise PyBulletUnavailable(
            f"scene {self.domain!r} not supported (builtin: {sorted(SCENES)})"
        )

    def law(self) -> str:
        return (self.scene or SCENES.get(self.domain, {})).get("law", "")

    # ── 场景管理 ───────────────────────────────────────────────────

    def use_scene(self, name: str) -> PyBulletEnv:
        """切换到内置场景 (cartpole / free_fall)."""
        if name not in SCENES:
            raise ValueError(f"unknown scene {name!r}; builtin: {sorted(SCENES)}")
        self.scene = SCENES[name]
        self.domain = SCENES[name]["domain"]
        return self

    # ── 内部 rollout 实现 ──────────────────────────────────────────

    def _require_pb(self) -> Any:
        if not self.available or self._pb is None:
            raise PyBulletUnavailable("pybullet unavailable on this host")
        return self._pb

    def _rollout_cartpole(self, pb, state: LawState, action: LawAction, scene) -> LawState:
        """倒立摆: 连 pybullet, 建 cart-pole 两体, 施水平力, rollout, 读末态."""
        force = float(action.config.get("force", 0.0))
        mode = pb.GUI if self.use_gui else pb.DIRECT
        try:
            cid = pb.connect(mode)
        except Exception as exc:  # noqa: BLE001 — 无显示(CI)下 GUI 连不上要能降级
            if self.use_gui:
                logger.warning("pybullet GUI connect failed, falling back DIRECT")
                cid = pb.connect(pb.DIRECT)
            else:
                raise PyBulletUnavailable(f"pybullet connect failed: {exc}") from exc
        try:
            pb.setGravity(0, 0, self.gravity, physicsClientId=cid)
            pb.setTimeStep(self.timestep, physicsClientId=cid)
            # 简化替身: 用两个盒体近似 车 + 摆(质量/尺寸可配). 真 URDF 见 S4.
            car = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[0.25, 0.25, 0.1])
            car_body = pb.createMultiBody(
                baseMass=1.0,
                baseCollisionShapeIndex=car,
                basePosition=[state.get("cart_pos", 0.0), 0, 0.5],
            )
            pole = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[0.03, 0.03, 0.5])
            pole_body = pb.createMultiBody(
                baseMass=0.1,
                baseCollisionShapeIndex=pole,
                basePosition=[state.get("cart_pos", 0.0), 0, 1.0 + 0.5],
            )
            # 固定摆底端到车上 + 摆竖直, 简化约束 (真实铰接用 createConstraint)
            pb.createConstraint(
                parentBodyUniqueId=car_body,
                parentLinkIndex=-1,
                childBodyUniqueId=pole_body,
                childLinkIndex=-1,
                jointType=pb.JOINT_POINT2POINT,
                jointAxis=[0, 0, 1],
                parentFramePosition=[0, 0, 0.5],
                childFramePosition=[0, 0, -0.5],
                physicsClientId=cid,
            )
            # 初始摆角 (相对竖直): 绕 x 轴旋转近似
            theta = state.get("pole_theta", 0.0)
            pb.resetBasePositionAndOrientation(
                pole_body,
                [state.get("cart_pos", 0.0), 0, 1.0 + 0.5],
                pb.getQuaternionFromEuler([theta, 0, 0]),
                physicsClientId=cid,
            )
            # rollout
            for _ in range(self.n_steps):
                pb.applyExternalForce(
                    car_body, -1, [force, 0, 0], [0, 0, 0],
                    pb.WORLD_FRAME, physicsClientId=cid,
                )
                pb.stepSimulation(physicsClientId=cid)
            cart_pos, cart_vel = self._body_state(pb, cid, car_body, "cart")
            _pole_pos, _pole_vel = self._body_state(pb, cid, pole_body, "pole")
            return LawState(
                {
                    "cart_pos": cart_pos,
                    "cart_vel": cart_vel,
                    "pole_theta": theta + _pole_vel,  # 解析近似: 角位移 ~ 角速度累积
                    "pole_dtheta": _pole_vel,
                },
                scene.get("domain", self.domain),
            )
        finally:
            pb.disconnect(cid)

    @staticmethod
    def _body_state(pb, cid, body, label: str) -> tuple[float, float]:
        """读刚体位置/速度的 x 分量 (label 仅用于可读性/兜底)."""
        pos, _ = pb.getBasePositionAndOrientation(body, physicsClientId=cid)
        vel, _ = pb.getBaseVelocity(body, physicsClientId=cid)
        return float(pos[0]), float(vel[0])

    def _rollout_free_fall(self, state: LawState, action: LawAction) -> LawState:
        """自由落体解析 rollout: y(t) = y0+v0 t - ½ g t² (无需 pybullet 真连)."""
        y0 = state.get("y", 0.0)
        v0 = state.get("vy", 0.0)
        t = self.n_steps * self.timestep
        g = abs(self.gravity)
        return LawState(
            {"y": y0 + v0 * t - 0.5 * g * t * t, "vy": v0 - g * t},
            self.scene.get("domain", self.domain),
        )

    def _rollout_urdf(self, pb, state: LawState, action: LawAction,
                      scene: dict[str, Any], urdf: str) -> LawState:
        """URDF 多关节场景 rollout: 载 plane + 机器人, 施加关节动作, 读末态关节角.

        这是科研 explore 的**真物理 ground truth** —— 多刚体动力学 + 接触 + 关节,
        不再是解析近似。``action.config`` 可选提供 ``joint_parameters`` (jointName->目标
        力矩/位置) 传 ``setJointMotorControl``。urdf 路径异常/缺失 → 抛 PyBulletUnavailable
        让上层 fail-open (偶发的内建模缺文件不阻断真场景).
        """
        mode = pb.GUI if self.use_gui else pb.DIRECT
        try:
            cid = pb.connect(mode)
        except Exception as exc:  # noqa: BLE001 — 无显示(CI)下 GUI 连不上要能降级
            if self.use_gui:
                logger.warning("pybullet GUI connect failed, falling back DIRECT")
                cid = pb.connect(pb.DIRECT)
            else:
                raise PyBulletUnavailable(f"pybullet connect failed: {exc}") from exc
        try:
            pb.setGravity(0, 0, self.gravity, physicsClientId=cid)
            pb.setTimeStep(self.timestep, physicsClientId=cid)
            # pybullet_data 作为内置 URDF 的搜索路径 (ur5/humanoid 等内置场景靠它定位)
            try:
                import pybullet_data as _pd
                pb.setAdditionalSearchPath(
                    os.path.join(os.path.dirname(_pd.__file__)), physicsClientId=cid
                )
            except Exception:  # noqa: BLE001 — 数据路径缺失仅记录, 不阻断显式 urdf_path
                logger.warning("pybullet_data unavailable; explicit urdf_path only")
            pb.loadURDF("plane.urdf", physicsClientId=cid)
            try:
                robot = pb.loadURDF(urdf, physicsClientId=cid)
            except Exception as exc:  # noqa: BLE001 — URDF 缺失 → fail-open 报告即可
                raise PyBulletUnavailable(f"urdf load failed ({urdf}): {exc}") from exc
            controls = action.config.get("joint_parameters") or {}
            for j in range(pb.getNumJoints(robot, physicsClientId=cid)):
                info = pb.getJointInfo(robot, j, physicsClientId=cid)
                name = info[1].decode()
                if name not in controls:
                    continue
                params = controls[name]
                v = float(params.get("target", 0.0) if isinstance(params, dict) else params)
                pb.setJointMotorControl2(
                    robot, j, pb.POSITION_CONTROL, targetPosition=v,
                    force=float(params.get("force", 50.0)) if isinstance(params, dict) else 50.0,
                    physicsClientId=cid,
                )
            for _ in range(self.n_steps):
                pb.stepSimulation(physicsClientId=cid)
            if getattr(self, "observable", None) or (scene.get("observables")):
                # 机械臂/双足: 读显式可测关节角作为状态向量 (ground truth)
                vec: dict[str, float] = {}
                for j in range(pb.getNumJoints(robot, physicsClientId=cid)):
                    try:
                        info = pb.getJointInfo(robot, j, physicsClientId=cid)
                        vec[info[1].decode()] = round(pb.getJointState(robot, j, physicsClientId=cid)[0], 6)
                    except Exception:  # noqa: BLE001 — 个别关节读失败跳过, 不整场景崩
                        continue
                return LawState(vec, scene.get("domain", self.domain))
            # 双足 humanoid: 最可测单标量 = 基座离地高度 (无控制必倒地 → 高度降)
            root_pos, _ = pb.getBasePositionAndOrientation(robot, physicsClientId=cid)
            return LawState({"root_height": float(root_pos[2])},
                            scene.get("domain", self.domain))
        finally:
            pb.disconnect(cid)


# 轻量 LawState / LawAction (避免 tools->research 反向依赖 research.law_model).
@dataclass
class LawState:
    """世界可测状态 (数值特征向量). 与 research.law_model.LawState 同构."""

    vector: dict[str, float] = field(default_factory=dict)
    domain: str = ""

    def get(self, k: str, default: float = 0.0) -> float:
        return float(self.vector.get(k, default))

    def as_dict(self) -> dict:
        return {"domain": self.domain, "state": dict(self.vector)}


@dataclass
class LawAction:
    """可执行动作 (控制参数向量). 与 research.law_model.LawAction 同构."""

    config: dict[str, float] = field(default_factory=dict)
    label: str = ""

    def as_dict(self) -> dict:
        return {"label": self.label, "action": dict(self.config)}
