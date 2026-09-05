"""通用外部计算工具适配器 — 把 Huginn 真实计算工具接进执行器.

真实工具是**长时外部进程/远程作业**, 不是内存规则表 — 因此加一层薄适配, 让
外部工具也能复用 ``SensorModelExecutor`` 的 确认/传感器/标定/制品生命周期/安全闸门.
本模块 = microduck "执行器 = 前向真值 + 传感器视图" 在"真实计算工具"上的接缝.

支持的计算形态 (三种, 统一经 ``JobBackend.run(spec) -> JobResult``)::

  - **本地进程** (:class:`LocalJobBackend`) : ``subprocess`` 一次运行, 带超时.
  - **远程/长时 HPC 作业** (:class:`RemoteHpcJobBackend`) : 提交 → 轮询 → 取回,
    编排在骨架里, 具体调度器 (Slurm/PBS/GPU) 只要实现 :class:`HpcTransport`.
  - **常驻服务/API** (:class:`HttpJobBackend`) : REST/gRPC 调用 HTTP 端点.

``ComputationalToolAdapter`` 是**接真实工具的只改点** — 无论哪种形态都只写
``build_job(action, workdir)`` 出 spec + ``parse_output(text, code)`` / ``parse_observation``
把结果解析成 state. 收敛/演化型工具可额外实现 ``parse_observation -> ParsedObservation``,
经 ``max_iterations`` 做迭代重试 (:class:`ConvergencePendingError` 标记未收敛挂起).

占位 (CI 可跑): ``ShellComputeTool`` (本地进程, E=n·Cv·T)、:class:`RelaxComputeTool`
(收敛型, 模拟 DFT 结构弛豫迭代)、HTTP 后端 (用注入 caller 测流程, 默认 urllib 真调)。
原则不变: 世界模型 (快代理) 与真实后端 (真计算) 必须 view-consistent.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
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


# ── 三形态计算: 作业规格 + 后端抽象 ─────────────────────────────
@dataclass(frozen=True)
class HttpJob:
    """常驻服务/API 调用规格: 方法 + URL + 头 + 载荷 + 超时."""

    method: str = "POST"
    url: str = ""
    headers: dict[str, str] | None = None
    payload: str | None = None
    timeout: float = 30.0


@dataclass(frozen=True)
class RemoteJobSpec:
    """远程/长时 HPC 作业规格: 命令 + 运行目录 + 超时 + 资源申请."""

    command: tuple[str, ...] = ()
    cwd: str | None = None
    timeout: float = 300.0
    env: dict[str, str] | None = None
    partition: str = ""
    nodes: int = 1
    gpus: int = 0
    # 调度器提交命令模板 (如 "sbatch"), 默认关; 真 Slurm/PBS 时替换.
    submit_prefix: tuple[str, ...] = ()


@dataclass(frozen=True)
class JobResult:
    """一次计算执行的结果: 文本 + 退出码 + 是否成功."""

    text: str = ""
    code: int = 0
    ok: bool = True

    @classmethod
    def from_proc(cls, p: subprocess.CompletedProcess) -> JobResult:
        return cls(text=p.stdout or "", code=p.returncode, ok=p.returncode == 0)


class JobBackend(Protocol):
    """统一执行形态: 本地 / 远程 HPC / HTTP 三后端都实现 ``run -> JobResult``.

    返回**文本 + 退出码**, 语义交给 adapter 的 ``parse_output``/``parse_observation``,
    骨架本身不理解单个作业内容 — 因此接新工具仍是"写一个 adapter", 换形态仅换后端.
    """

    def run(self, spec: Any, *, timeout: float) -> JobResult: ...


class LocalJobBackend:
    """本地进程后端: subprocess 一次运行 (现有 ``run_job`` 语义)."""

    name = "local"
    # M3 后端访问门: 标记计算形态, 供"设备私有化"判定某形态是否把数据投递远端.
    backend_kind = "local"

    def run(self, spec: Any, *, timeout: float) -> JobResult:
        if not isinstance(spec, JobSpec):
            raise ToolInvocationError(f"local backend needs JobSpec, got {type(spec).__name__}")
        return JobResult.from_proc(run_job(spec))


class HpcTransport(Protocol):
    """远程作业的调度器接缝: 提交 / 轮询 / 取回 / 取消. Slurm/PBS/GPU 各实现一只."""

    def submit(self, spec: RemoteJobSpec) -> str: ...
    def poll(self, job_id: str) -> str: ...  # PENDING / RUNNING / DONE / FAILED
    def fetch(self, job_id: str) -> str: ...  # 最终 stdout
    def cancel(self, job_id: str) -> None: ...


class RemoteHpcJobBackend:
    """远程/长时 HPC 后端: 提交 → 轮询直到 DONE → 取回 stdout.

    一次 ``run`` 对应一次完整远程作业 (这正是不在本地同步死等的关键 — 编排在骨架,
    调度器只暴露四件套). 轮询超时抛 :class:`ToolInvocationError`.
    """

    name = "remote_hpc"
    # M3 后端访问门: 远程作业会提交到远端集群 → 设备私有化下禁用.
    backend_kind = "remote_hpc"

    def __init__(
        self,
        transport: HpcTransport | None = None,
        *,
        poll_interval: float = 0.05,
    ) -> None:
        self.transport = transport
        self._poll_interval = poll_interval

    def run(self, spec: Any, *, timeout: float) -> JobResult:
        if not isinstance(spec, RemoteJobSpec):
            raise ToolInvocationError(
                f"remote_hpc backend needs RemoteJobSpec, got {type(spec).__name__}"
            )
        if self.transport is None:
            raise ToolInvocationError(
                "remote_hpc backend needs an HpcTransport (Slurm/PBS/GPU) — not configured"
            )
        jid = self.transport.submit(spec)
        deadline = time.monotonic() + timeout
        while True:
            status = self.transport.poll(jid)
            if status == "DONE":
                return JobResult(text=self.transport.fetch(jid), code=0, ok=True)
            if status == "FAILED":
                return JobResult(text=self.transport.fetch(jid), code=1, ok=False)
            if time.monotonic() > deadline:
                self.transport.cancel(jid)
                raise ToolInvocationError(f"remote job {jid} timed out after {timeout}s")
            time.sleep(self._poll_interval)


class HttpJobBackend:
    """常驻服务/API 后端: 调 HTTP 端点, 响应体当 stdout.

    默认用 stdlib ``urllib`` 真调; 测试可注入 ``caller`` (同步 ``(req, timeout) -> JobResult``)
    验证流程而不起真服务. 真实 gRPC 后端可实现同一 ``caller`` 契约.
    """

    name = "http"
    # M3 后端访问门: HTTP 调用可能打到远端服务 → 设备私有化下需判定.
    backend_kind = "http"

    def __init__(
        self,
        caller: Any | None = None,
    ) -> None:
        self._caller = caller

    def run(self, spec: Any, *, timeout: float) -> JobResult:
        if not isinstance(spec, HttpJob):
            raise ToolInvocationError(f"http backend needs HttpJob, got {type(spec).__name__}")
        if self._caller is not None:
            return self._caller(spec)
        return _urllib_call(spec)


# M3 后端访问门: "设备私有化" (local_only) 下仍允许的计算形态.
_DEVICE_PRIVACY_OK_KINDS = frozenset({"local", "device"})


def backend_allows_local_only(kind: str | None) -> bool:
    """设备私有化 (execution_privacy=local_only) 下某计算形态是否可用.

    local / device 在设备端就地算 → 允许; remote_hpc / http 会把作业/数据投递远端
    → 禁止. unknown (None/未标注) 保守允许, 避免因元数据缺位误伤.
    """
    if kind is None:
        return True
    return kind in _DEVICE_PRIVACY_OK_KINDS


def _urllib_call(req: HttpJob) -> JobResult:
    """stdlib urllib 真调: 一次 HTTP 请求并读响应体."""
    data = req.payload.encode("utf-8") if req.payload else None
    r = urllib.request.Request(
        req.url, data=data, headers=req.headers or {}, method=req.method
    )
    try:
        with urllib.request.urlopen(r, timeout=req.timeout) as resp:
            return JobResult(text=resp.read().decode("utf-8", "replace"), code=resp.status, ok=True)
    except urllib.error.HTTPError as e:
        return JobResult(text=e.read().decode("utf-8", "replace"), code=e.code, ok=False)
    except OSError as e:
        raise ToolInvocationError(f"http call failed: {e}") from e


# ── 收敛/演化语义 ───────────────────────────────────────────────
class ConvergencePendingError(Exception):
    """收敛/演化型计算在最大迭代次数内未收敛 → 标记"任务挂起未完成"."""


@dataclass
class ParsedObservation:
    """adapter 产出: 状态 + 是否收敛 + 迭代号.

    ``converged=False`` 时 ``ExternalComputeExecutor`` 会继续迭代 (同动作续算),
    直到 ``max_iterations``; 超限抛 :class:`ConvergencePendingError` (不改 state).
    """

    values: dict[str, Any]
    converged: bool = True
    iteration: int = 0


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
        backend: JobBackend | None = None,
        max_iterations: int = 5,
    ) -> None:
        super().__init__(initial=initial or {}, error_model=error_model, seed=seed)
        self.adapter = adapter
        self._workdir = workdir
        # 后端形态: 默认本地进程; 换远程 HPC/常驻 API 只换 backend (adapter 不动).
        self.backend = backend if backend is not None else LocalJobBackend()
        self.max_iterations = max_iterations

    def _forward(self, state: dict[str, Any], action: PhysicalAction) -> dict[str, Any]:
        workdir = (
            Path(self._workdir)
            if self._workdir
            else Path(tempfile.mkdtemp(prefix="ext_tool_"))
        )
        out = dict(state)
        # 收敛/演化型工具: adapter 实现可选 parse_observation → 迭代直到收敛或超限.
        if hasattr(self.adapter, "parse_observation"):
            for _it in range(1, self.max_iterations + 1):
                spec = self.adapter.build_job(action, workdir)
                res = self.backend.run(spec, timeout=_spec_timeout(spec))
                obs = self.adapter.parse_observation(res)  # type: ignore[attr-defined]
                out.update(obs.values)
                if obs.converged:
                    return out
                # 未收敛: 同动作续算 (真实 DFT 弛豫即反复算同一构型直到能量收敛).
            raise ConvergencePendingError(
                f"{type(self.adapter).__name__} not converged within {self.max_iterations} iters"
            )
        # 仪表/一次性型: 一次调用, parse_output 返回 flat state (向后兼容).
        spec = self.adapter.build_job(action, workdir)
        res = self.backend.run(spec, timeout=_spec_timeout(spec))
        parsed = self.adapter.parse_output(res.text, res.code)
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


def _spec_timeout(spec: Any) -> float:
    """取 job spec 的超时 (三形态都有 timeout 字段). 后端用它作为本轮运行上限."""
    return float(getattr(spec, "timeout", 30.0))


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


# ── 占位 (形态二): 收敛型 DFT 弛豫 (本地 python 子进程, CI 可跑) ──
class RelaxComputeTool:
    """收敛型占位: 模拟 DFT 结构弛豫 — 每个迭代输出能量, 量差 < 容差才收敛.

    真实 VASP 弛豫是"反复算同一构型直到能量/力收敛"; 这里用子进程逐帧输出
    ``{"energy", "iteration", "converged"}``, 由 executor 的 ``max_iterations`` 驱动
    迭代. adapter 仅这里的 ``build_job`` + ``parse_observation`` 会被骨架调用.
    """

    def build_job(self, action: PhysicalAction, workdir: Path) -> JobSpec:
        n = float(action.params.get("n", 1.0))
        T = float(action.params.get("T", 300.0))
        tol = float(action.params.get("tol", 0.01))
        steps = int(action.params.get("steps", 2))
        state_file = str(workdir / ".relax_state.json")
        script = _relax_script(n, T, tol, steps=steps, state_file=state_file)
        return JobSpec(
            command=(sys.executable, "-c", script), cwd=str(workdir), timeout=15.0
        )

    def parse_observation(self, res: JobResult) -> ParsedObservation:
        # 校验工具退出码; 收敛工具以"能量量差 ≤ tol"判定.
        if not res.ok:
            raise ToolInvocationError(f"relax tool rc={res.code}")
        data = _parse_text_json(res.text)
        energy = float(data["energy"])
        iteration = int(data.get("iteration", 0))
        target = float(data.get("target", 0.0))
        converged = abs(energy - target) <= float(data.get("tol", 0.01))
        return ParsedObservation(
            values={"energy": energy},
            converged=converged,
            iteration=iteration,
        )


class RelaxShellComputeWorldModel(WorldModel):
    """收敛型世界模型: 解析地已知弛豫终态能量 (目标), 作慢而准的"真值"参考."""

    def predict(
        self, state_before: dict[str, Any], action: PhysicalAction
    ) -> dict[str, Any]:
        n = float(action.params.get("n", 1.0))
        T = float(action.params.get("T", 300.0))
        return {"energy": n * _CV * T}

    def infer_inverse(
        self, state_before: dict[str, Any], action: PhysicalAction
    ) -> PhysicalAction | None:
        return None


def _relax_script(
    n: float,
    T: float,
    tol: float,
    *,
    steps: int,
    state_file: str,
) -> str:
    """生成收敛型子进程脚本 (多行字符串): 用 workdir 持久化迭代进度并续算.

    与真实 DFT 弛豫一致: 每次子进程从上一次 checkpoint (``.relax_state.json``) 续算,
    每 ``steps`` 步向目标能量收缩 ``target`` 一次, 量差 ≤ tol 即收敛. 末帧打印 JSON.
    """
    target = n * _CV * T
    lines = [
        "import json, os",
        f"target={target!r}; tol={tol!r}; steps={int(steps)!r}",
        f"state_file={json.dumps(state_file)}",
        "try:",
        "    st=json.load(open(state_file))",
        "    cur=float(st.get('energy',0.0)); it=int(st.get('iteration',0))",
        "except (OSError, ValueError, json.JSONDecodeError):",
        "    cur=0.0; it=0",
        "frame=None",
        "for i in range(1, steps+1):",
        "    it += 1",
        "    cur = cur + (target-cur)*0.6",
        "    frame={'energy': cur, 'iteration': it, 'converged': abs(cur-target)<=tol,",
        "           'target': target, 'tol': tol}",
        "    if frame['converged']:",
        "        break",
        "os.makedirs(os.path.dirname(state_file), exist_ok=True)",
        "json.dump({'energy': cur, 'iteration': it}, open(state_file,'w'))",
        "print(json.dumps(frame))",
    ]
    return "\n".join(lines)


# ── 占位 (形态三): 常驻 HTTP API (本地 http.server, CI 可跑) ────
class HttpRelaxTool:
    """HTTP 形态占位: 一次 GET 带参数, 服务端返回能量/收敛 JSON (本地 http.server).

    与 :class:`RelaxComputeTool` 同一能量语义, 但走 ``HttpJobBackend``(urllib) 而不是
    子进程 — 证明"换后端形态、adapter 只改 build_job 出 HttpJob" 的接缝.
    """

    def build_job(self, action: PhysicalAction, workdir: Path) -> HttpJob:
        del workdir
        n = float(action.params.get("n", 1.0))
        T = float(action.params.get("T", 300.0))
        base = str(action.params.get("base_url", "http://127.0.0.1:0"))
        return HttpJob(
            method="GET",
            url=f"{base}/relax?n={n}&T={T}",
            timeout=float(action.params.get("timeout", 5.0)),
        )

    def parse_observation(self, res: JobResult) -> ParsedObservation:
        if not res.ok:
            raise ToolInvocationError(f"http tool rc={res.code}: {res.text}")
        data = _parse_text_json(res.text)
        energy = float(data["energy"])
        target = float(data.get("target", 0.0))
        tol = float(data.get("tol", 0.01))
        return ParsedObservation(
            values={"energy": energy},
            converged=abs(energy - target) <= tol,
            iteration=int(data.get("iteration", 0)),
        )

    def parse_output(self, text: str, returncode: int) -> dict[str, Any]:
        if returncode != 0:
            raise ToolInvocationError(f"http tool rc={returncode}")
        return _parse_text_json(text)


def _parse_text_json(text: str) -> dict[str, Any]:
    """解析工具输出的末行 JSON (容忍脚本额外 stdout)."""
    try:
        return json.loads(text.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError, ValueError) as e:
        raise ToolInvocationError(f"unparseable tool output: {e}") from e


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
