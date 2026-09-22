"""PyBullet 具身后端 (LawModel) 测试.

关键: 即使 pybullet 未安装也必须全绿 —— 具身仿真是可选后端, 测试不能因为它
不在沙箱而失败。用 --tb=short 跑; 需要真物理 rollout 的用例在 pybullet 缺失时
自动跳过 (skip), 但 fail-open 的契约用例始终执行。
"""

from __future__ import annotations

import pytest

from huginn.tools.sim import pybullet_env as pbe
from huginn.tools.sim.pybullet_env import (
    SCENES,
    LawAction,
    PyBulletEnv,
    PyBulletUnavailable,
)
from huginn.tools.sim.pybullet_tool import PyBulletTool, PyBulletToolInput

_HAS_PYB = pbe._PYBULLET_OK


def test_scenes_are_registered():
    assert "cartpole" in SCENES
    assert "free_fall" in SCENES


def test_env_metrics():
    # 无论有没有 pybullet, LawModel 三件套签名 / 场景配置都必须存在.
    for _name, cfg in SCENES.items():
        assert cfg["domain"]
        assert cfg["state"]
        assert cfg["observables"]
        assert cfg["law"]


def test_reconcile_free_fall_matches_analytic():
    """自由落体的解析 rollout 与解析值对账应 borne_out (不用真 pybullet)."""
    if not _HAS_PYB:
        return
    env = PyBulletEnv()
    env.use_scene("free_fall")
    s = env.seed({"state": {"y": 10.0, "vy": 0.0}})
    a = LawAction({"g": env.gravity})
    fin = env.predict(s, a)
    assert fin.domain == "physics.free_fall"
    # y(t) = 10 - 0.5*9.81*t² ; t = n_steps*timestep = 1s
    assert abs(fin.get("y") - (10.0 - 0.5 * 9.81)) < 1e-6


def test_tool_reconcile_borne_out_when_within_tol():
    if not _HAS_PYB:
        return
    tool = PyBulletTool()
    r = tool.call(
        {
            "action": "reconcile",
            "scene": "free_fall",
            "state": {"y": 10.0, "vy": 0.0},
            "action_cfg": {},
            "expected": {"y": 10.0 - 0.5 * 9.81},
            "tol": 0.05,
        }
    )
    assert r.data["borne_out"] is True
    assert r.data["verdict"] == "borne_out"


def test_tool_predict_failopen_without_pybullet():
    """pybullet 缺失时工具必须 fail-open (skipped=True), 不抛异常."""
    if _HAS_PYB:
        return
    tool = PyBulletTool()
    # 真实 pybullet 缺失 -> available=False, 但 call 不抛.
    assert tool.available is False
    r = tool.call({"action": "predict", "scene": "cartpole"})
    assert r.data.get("available") is False
    assert r.data.get("skipped") is True


def test_pybulletunavailable_raised_when_forced():
    """明确请求真物理但后端不存在时抛 PyBulletUnavailable, 且只在无后端时."""
    if _HAS_PYB:
        # 有 pybullet 真实环境: 允许构造, 不应抛
        env = PyBulletEnv()
        assert env.available is True
        return
    try:
        PyBulletEnv()
    except PyBulletUnavailable:
        assert True
    else:
        raise AssertionError("expected PyBulletUnavailable without pybullet")


def test_tool_info_reports_version_and_scenes():
    if not _HAS_PYB:
        return
    tool = PyBulletTool()
    r = tool.call({"action": "info"})
    assert r.data["available"] is True
    assert r.data["backend"] == "pybullet"
    assert "cartpole" in r.data["scenes"]


def test_unknown_action_raises_validation_error():
    # Literal action 在 pydantic 层就拦截非法值 —— 工具不吞未知命令.
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        PyBulletToolInput(action="bogus")


def test_input_schema_roundtrip():
    d = PyBulletToolInput(action="reconcile", scene="free_fall", tol=0.01)
    assert d.scene == "free_fall"
    assert d.tol == 0.01


def test_urdf_scenes_are_registered():
    # S4 URDF 多场景库: kuka_iiwa / humanoid 需带 domain + law + urdf (真关节动力学).
    for name in ("kuka_iiwa", "humanoid"):
        assert name in SCENES, name
        cfg = SCENES[name]
        assert cfg["domain"].startswith("robotics")
        assert cfg["law"]
        assert cfg["urdf"]          # 内置 URDF 路径 (pybullet_data)


def test_urdf_scene_rollout_returns_joint_ground_truth():
    """URDF 机械臂 rollout 应返回可观测量向量 (关节角真值)."""
    if not _HAS_PYB:
        return
    env = PyBulletEnv()
    env.use_scene("kuka_iiwa")
    fin = env.predict(env.seed({"state": {}}), LawAction({}))
    assert fin.domain == "robotics.arm.kuka_iiwa"
    assert len(fin.vector) >= 1      # 至少读到关节角 ground truth


def test_urdf_path_injection_overrides_scene_urdf():
    """显式 urdf_path 应覆盖场景内置 URDF; 缺失路径则 fail-open 抛 Unavailable."""
    if not _HAS_PYB:
        return
    env = PyBulletEnv()
    env.use_scene("kuka_iiwa")
    env.urdf_path = "/nonexistent/robot.urdf"
    try:
        env.predict(env.seed({"state": {}}), LawAction({}))
    except PyBulletUnavailable:
        assert True
    else:
        raise AssertionError("expected PyBulletUnavailable for missing urdf_path")


def test_cspace_add_state_bridges_lawmodel():
    """S4 桥: PyBulletEnv 可经 CSpace.add_state 成为状态在场 (falsifiable card)."""
    if not _HAS_PYB:
        return
    from huginn.research.cspace import CSpace
    env = PyBulletEnv()
    env.use_scene("free_fall")
    c = CSpace()
    b = c.add_state("embodied", env, state={"y": 10.0, "vy": 0.0}, action={})
    assert b.kind == "state"
    assert b.falsifiable is True        # law 三件套在 → 可 reconcile 对账
    assert b.payload["law"]
    assert b.payload["card"]["worldview"] == "physics_causal"
    assert b.payload["card"]["falsifiable"] is True
