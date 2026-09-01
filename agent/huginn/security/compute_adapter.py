"""通用外部计算工具适配器 — 把 Huginn 真实计算工具 (VASP/DFT/MD/CFD/FEM…) 接进执行器.

真实工具是**长时外部进程/远程作业**, 不是内存规则表 — 因此加一层薄适配, 让
外部工具也能复用 ``SensorModelExecutor`` 的 确认/传感器/标定/制品生命周期/安全闸门.
本模块 = microduck "执行器 = 前向真值 + 传感器视图" 在"真实计算工具"上的接缝:

- ``JobSpec``       : 一次外部调用的规格 (command/cwd/env/timeout).
- ``run_job``       : 实际调用 (subprocess, 带超时; 远程/GPU 作业后续替换该实现).
- ``ComputationalToolAdapter`` (Protocol): ``build_job(action, workdir)`` 生成调用,
  ``parse_output(stdout, rc)`` 把输出解析成 state dict. **这是接真实工具的只改点**.
- ``ExternalComputeExecutor`` : 把 adapter 的"建作业→运行→解析"接进 ``_forward``,
  从而如同一个执行器使用.
- ``ShellComputeTool`` (占位): 用本机 python 子进程算一个确定性物理量 (E=n·Cv·T),
  证明"外部进程→解析→state"链路纯软件即可端到端验证 (CI 可跑). 世界模型用同一
  解析公式 (快), 执行器走真子进程, 二者必须一致 — 这就是"世界模型 == 仪表"契约.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from huginn.security.actuator_model import ErrorModel, SensorModelExecutor
from huginn.security.behavior_lifecycle import (
    BehaviorArtifact,
    BehaviorLifecycle,
    InstallResult,
)
from huginn.security.world_model import PhysicalAction, WorldModel

# 理想气体参数 (用能量标量 E = n·Cv·T 作"可观测"), 单原子 Cv = 3/2 R.
_R = 8.31446261815324
_CV = 1.5 * _R


class ToolInvocationError(Exception):
    """外部工具调用/解析失败."""


@dataclass(frozen=True)
class JobSpec:
    """一次外部工具调用: 命令 + 工作目录 + 超时 + 环境."""

    command: tuple[str, ...]
    cwd: str | None = None
    timeout: float = 30.0
    env: dict[str, str] | None = None


def run_job(job: JobSpec) -> subprocess.CompletedProcess:
    """同步执行外部工具 (带超时). 远程/GPU 作业未来替换本实现即可."""
    try:
        return subprocess.run(
            list(job.command),
            cwd=job.cwd,
            env=job.env,
            capture_output=True,
            text=True,
            timeout=job.timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        raise ToolInvocationError(f"tool timeout after {job.timeout}s: {e.cmd}") from e
    except OSError as e:
        raise ToolInvocationError(f"tool spawn failed: {e}") from e


class ComputationalToolAdapter(Protocol):
    """真实工具的接缝: 生成调用 + 把输出解析成 state. 接 VASP 只需实现这两个方法."""

    def build_job(self, action: PhysicalAction, workdir: Path) -> JobSpec: ...

    def parse_output(self, stdout: str, returncode: int) -> dict[str, Any]: ...


class ExternalComputeExecutor(SensorModelExecutor):
    """把外部计算工具接进执行器骨架: ``_forward`` = 建作业→运行→解析成 state."""

    def __init__(
        self,
        adapter: ComputationalToolAdapter,
        *,
        workdir: str | Path | None = None,
        initial: dict[str, Any] | None = None,
        error_model: ErrorModel | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(initial=initial or {}, error_model=error_model, seed=seed)
        self.adapter = adapter
        self._workdir = workdir

    def _forward(self, state: dict[str, Any], action: PhysicalAction) -> dict[str, Any]:
        workdir = (
            Path(self._workdir)
            if self._workdir
            else Path(tempfile.mkdtemp(prefix="ext_tool_"))
        )
        job = self.adapter.build_job(action, workdir)
        proc = run_job(job)
        parsed = self.adapter.parse_output(proc.stdout, proc.returncode)
        out = dict(state)
        out.update(parsed)  # 外部工具产生的是"新 state" (如能量/力), 与演化式工具不同.
        return out

    # 外部工具无守恒 dec/inc; 传感器为对实测量的 gauge 偏置.
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
        if isinstance(view.get("energy"), int | float):
            view["energy"] = float(view.get("energy", 0.0)) + magnitude
        return view


# ── 占位外部工具: 本机 python 子进程 (CI 可跑) ──────────────────
class ShellComputeTool:
    """占位: 用本机 python 子进程计算 E = n·Cv·T. 证明"外部进程→解析"链路可用."""

    def build_job(self, action: PhysicalAction, workdir: Path) -> JobSpec:
        n = float(action.params.get("n", 1.0))
        T = float(action.params.get("T", 300.0))
        expr = f"{n}*{_CV}*{T}"
        script = "import json;" f"print(json.dumps({{'energy': {expr}}}))"
        return JobSpec(
            command=(sys.executable, "-c", script), cwd=str(workdir), timeout=15.0
        )

    def parse_output(self, stdout: str, returncode: int) -> dict[str, Any]:
        if returncode != 0:
            raise ToolInvocationError(f"external tool rc={returncode}")
        return json.loads(stdout.strip().splitlines()[-1])


class ShellComputeWorldModel(WorldModel):
    """世界模型: 用同一解析公式 (EOS) 快速预演; executor 走真子进程. 二者须一致.

    这里界定了关键契约: 对计算类工具,"世界模型是快代理, 执行器是真实计算",
    像素上一致 (view-consistency) 才是可信的 sim 运行.
    """

    def predict(
        self, state_before: dict[str, Any], action: PhysicalAction
    ) -> dict[str, Any]:
        n = float(action.params.get("n", 1.0))
        T = float(action.params.get("T", 300.0))
        return {"energy": n * _CV * T}

    def infer_inverse(
        self, state_before: dict[str, Any], action: PhysicalAction
    ) -> PhysicalAction | None:
        return None  # 外部作业一般不可逆 (同 microduck 的 walk/recover).


# ── 制品接缝 (同形) ─────────────────────────────────────────────
SHELL_COMPUTE_CONTRACT_VERSION = 1


def build_shell_compute_artifact(
    version: int,
    *,
    systematic: float = 0.0,
    sigma: float = 0.0,
) -> BehaviorArtifact:
    return BehaviorArtifact(
        name="external_shell_compute",
        version=version,
        contract_version=SHELL_COMPUTE_CONTRACT_VERSION,
        config={"error_model": {"systematic": systematic, "sigma": sigma}},
    )


def shell_executor_from_artifact(artifact: BehaviorArtifact) -> ExternalComputeExecutor:
    cfg = artifact.config
    em = ErrorModel(
        systematic=float(cfg["error_model"]["systematic"]),
        sigma=float(cfg["error_model"]["sigma"]),
    )
    return ExternalComputeExecutor(
        ShellComputeTool(), initial={"energy": 0.0}, error_model=em, seed=0
    )


def shell_health_check(artifact: BehaviorArtifact) -> bool:
    """健康门控: 真跑一次外部子进程, 校验返回码 0 且能量有限为正."""
    try:
        ex = shell_executor_from_artifact(artifact)
        ex.execute(PhysicalAction("shell_compute", {"n": 1.0, "T": 300.0}))
    except ToolInvocationError:
        return False
    e = ex.observe().get("energy", 0.0)
    return isinstance(e, int | float) and e > 0


def install_shell_compute(
    lifecycle: BehaviorLifecycle,
    artifact: BehaviorArtifact,
) -> InstallResult:
    return lifecycle.install(
        artifact, health_check=lambda _v: shell_health_check(artifact)
    )
