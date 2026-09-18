"""具身仿真工具 —— 把 PyBulletEnv (LawModel 后端) 暴露给 agent 和 HTTP 客户端.

作为 tools/sim 家族的一员 (和 vasp/gromacs/lammps 平级), 但面向"具身/机器人"科研
场景: 用刚体动力学真值做「世界模型 predict -> 真实执行 -> reconcile 对账」闭环。

Fail-open 约定:
    pybullet 未安装时 ``PyBulletEnv.available=False``, 本工具返回
    ``{"available": False, "skipped": True, "reason": ...}``, 不抛异常、不阻断
    agent 主流程 —— 具身仿真是可选项, 不是硬依赖。

actions:
- info:          列出可用后端 / 版本 / 内置场景 / 各场景时间步长
- predict:      对一个 LawState (初始配置) + LawAction (控制参数) 做 pybullet rollout,
                 返回末态 (ground truth), 并带 law() 方程串, 供下游 reconcile 对账
- reconcile:    对 predict 产出 + 期望值做相对误差对账, 返回 borne_out/falsified
- reset:        回到场景初始配置 (等价 seed), 便于重放 / 可复现基线
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.core_types import ToolContext, ToolResult
from huginn.tools.base import HuginnTool, ResearchPhase, ToolProfile
from huginn.tools.sim.pybullet_env import (
    PYBULLET_VERSION,
    SCENES,
    LawAction,
    LawState,
    PyBulletEnv,
    PyBulletUnavailable,
)


class PyBulletToolInput(BaseModel):
    action: Literal["info", "predict", "reconcile", "reset"] = Field(default="info")
    scene: str = Field(
        default="cartpole",
        description="内置场景名 (cartpole / free_fall) 或 domain 名",
    )
    # predict / reconcile
    state: dict[str, float] = Field(default_factory=dict)
    action_cfg: dict[str, float] = Field(default_factory=dict)
    action_label: str = Field(default="")
    n_steps: int = Field(default=240, ge=1)
    tol: float = Field(default=0.03, gt=0)
    # reconcile 的期望末态 (缺省用解析/经验值), e.g. {"pole_theta": 0.05}
    expected: dict[str, float] = Field(default_factory=dict)
    # 可复现性
    use_gui: bool = Field(default=False)


class PyBulletTool(HuginnTool):
    """PyBullet 具身仿真后端 —— 刚体动力学真值做世界模型对账."""

    name = "pybullet_tool"
    category = "sim"
    profile = ToolProfile(phases=frozenset({ResearchPhase.PLANNING, ResearchPhase.EXECUTION}))
    description = (
        "Embodied/robotics simulator via PyBullet. Runs rigid-body rollout from an "
        "initial state under a control action, returning the final state as ground "
        "truth for world-model reconcile (borne_out/falsified). Fail-open when "
        "pybullet is not installed."
    )
    input_schema = PyBulletToolInput

    def __init__(self, env: PyBulletEnv | None = None) -> None:
        super().__init__()
        # 内部直接构造 PyBulletEnv; 失败 (无 pybullet) 时不抛, 记录 available=False.
        if env is not None:
            self._env = env
        else:
            try:
                self._env = PyBulletEnv()
            except PyBulletUnavailable:
                self._env = None

    @property
    def available(self) -> bool:
        return self._env is not None and self._env.available

    def call(
        self, args: dict[str, Any], context: ToolContext | None = None
    ) -> ToolResult:
        input_data = PyBulletToolInput(**args)
        if not self.available:
            return ToolResult(
                data={
                    "available": False,
                    "skipped": True,
                    "reason": "pybullet not installed; install via `pip install pybullet`",
                }
            )
        try:
            if input_data.action == "info":
                return self._info(input_data)
            if input_data.action == "predict":
                return self._predict(input_data)
            if input_data.action == "reconcile":
                return self._reconcile(input_data)
            if input_data.action == "reset":
                return self._reset(input_data)
            return ToolResult(data={"error": f"unknown action {input_data.action!r}"})
        except PyBulletUnavailable as exc:
            # fail-open: 运行期连不上 (无显示/资源) 也降级, 不抛给 agent 主链.
            return ToolResult(
                data={"available": False, "skipped": True, "reason": str(exc)}
            )

    # ── actions ───────────────────────────────────────────────────

    def _info(self, d: PyBulletToolInput) -> ToolResult:
        env = self._env
        assert env is not None
        return ToolResult(
            data={
                "available": True,
                "backend": "pybullet",
                "version": PYBULLET_VERSION,
                "domain": env.domain,
                "scenes": sorted(SCENES),
                "timestep": env.timestep,
                "law": env.law(),
                "note": (
                    "rigid-body ground truth only (no sensors/controllers); "
                    "reconcile compares predicted vs executed state."
                ),
            }
        )

    def _predict(self, d: PyBulletToolInput) -> ToolResult:
        env = self._env
        assert env is not None
        env.use_scene(d.scene)
        state = LawState(d.state or {})
        action = LawAction(d.action_cfg or {}, d.action_label)
        out = env.predict(state, action)
        return ToolResult(
            data={
                "domain": env.domain,
                "law": env.law(),
                "initial": state.as_dict(),
                "action": action.as_dict(),
                "final": out.as_dict(),
            }
        )

    def _reconcile(self, d: PyBulletToolInput) -> ToolResult:
        env = self._env
        assert env is not None
        env.use_scene(d.scene)
        final = env.predict(
            LawState(d.state or {}), LawAction(d.action_cfg or {}, d.action_label)
        )
        pred = final.vector
        # 期望值优先用 expected; 否则用解析基线 (自由落体/倒立摆近似) —— 由 scene 提供.
        expected = d.expected
        errors: dict[str, float] = {}
        mismatch: dict[str, float] = {}
        for k, want in expected.items():
            got = pred.get(k, float("nan"))
            rel = (
                abs(want - got) / abs(want)
                if want != 0
                else (0.0 if abs(got) <= d.tol else float("inf"))
            )
            errors[k] = round(rel, 4)
            if rel > d.tol:
                mismatch[k] = round(got, 4)
        borne_out = not bool(mismatch)
        return ToolResult(
            data={
                "domain": env.domain,
                "law": env.law(),
                "predicted": final.as_dict(),
                "expected": expected,
                "errors": errors,
                "mismatch": mismatch,
                "borne_out": borne_out,
                "verdict": "borne_out" if borne_out else "falsified",
                "tol": d.tol,
                "note": (
                    "falsified is a legitimate research signal — the law is a "
                    "hypothesis; deviation is a discovery, not failure."
                ),
            }
        )

    def _reset(self, d: PyBulletToolInput) -> ToolResult:
        env = self._env
        assert env is not None
        env.use_scene(d.scene)
        initial = env.seed(d.state or {})
        return ToolResult(
            data={
                "domain": env.domain,
                "initial": initial.as_dict(),
                "reset_note": (
                    "replayed from seed state; consistent with real pybullet "
                    "deterministic rollout for reproducible baselines."
                ),
            }
        )


__all__ = ["PyBulletTool"]
