"""通用外部计算工具适配器 + 占位测试.

证明"真实计算工具(外部进程/远程作业)"能以同一执行器骨架接入: 建作业→运行
→解析成 state; 世界模型(快代理)与真实后端(真计算)必须一致 (view-consistency),
是"世界模型 == 仪表"契约的软件兑现. 覆盖三形态计算: 本地进程 / 远程 HPC /
常驻 HTTP API, 以及收敛(弛豫)语义.
"""

from __future__ import annotations

import pytest

from huginn.security.behavior_lifecycle import BehaviorLifecycle
from huginn.security.compute_adapter import (
    ConvergencePendingError,
    ExternalComputeExecutor,
    HttpJob,
    HttpJobBackend,
    JobResult,
    RemoteHpcJobBackend,
    RemoteJobSpec,
    ShellComputeTool,
    ShellComputeWorldModel,
    ToolInvocationError,
    build_shell_compute_artifact,
    shell_executor_from_artifact,
)
from huginn.security.physics_schema import matches_state
from huginn.security.tool_registry import (
    install_tool,
    make_components,
    registered_tools,
)
from huginn.security.workspace import PhysicalWorkspace
from huginn.security.world_model import PhysicalAction

_CV = 1.5 * 8.31446261815324


def test_external_executor_runs_real_subprocess():
    """外部执行器真跑子进程: 解析出能量 = n·Cv·T."""
    ex = shell_executor_from_artifact(build_shell_compute_artifact(1))
    ex.execute(PhysicalAction("shell_compute", {"n": 2.0, "T": 300.0}))
    assert ex.observe()["energy"] == pytest.approx(2.0 * _CV * 300.0)


def test_world_model_matches_external_executor():
    """快代理(世界模型) == 真实子进程(执行器) → view-consistency 契约成立."""
    ex = shell_executor_from_artifact(build_shell_compute_artifact(1))
    wm = ShellComputeWorldModel()
    a = PhysicalAction("shell_compute", {"n": 1.0, "T": 400.0})
    pred = wm.predict({}, a)
    ex.execute(a)
    assert matches_state(pred, ex.observe(), tolerance=1e-9)


def test_external_tool_wired_into_workspace_and_registry():
    """外部工具经同界面解析并跑进同一 PhysicalWorkspace (确认闭环), 且已注册."""
    assert "external_shell_compute" in registered_tools()
    exe, wm = make_components("external_shell_compute", build_shell_compute_artifact(1))
    assert isinstance(exe, ExternalComputeExecutor)
    wa = PhysicalWorkspace(wm, exe)
    wa.execute(PhysicalAction("shell_compute", {"n": 1.0, "T": 300.0}), preflight=True)
    assert wa.state["energy"] == pytest.approx(_CV * 300.0)


def test_external_tool_install_health_gate(tmp_path):
    """经 registry 安装: 外部子进程健康门控通过并成为 current."""
    lc = BehaviorLifecycle(tmp_path)
    r = install_tool(lc, "external_shell_compute", build_shell_compute_artifact(1))
    assert r.healthy and lc.current_version() == 1


def test_tool_failure_is_surface():
    """外部工具返回非 0 → ToolInvocationError (健康门控/执行层可捕获)."""
    tool = ShellComputeTool()
    with pytest.raises(ToolInvocationError):
        tool.parse_output("", 1)


# ── 形态二: 收敛型 (DFT 弛豫, 多轮续算直到收敛) ─────────────────
def test_convergent_relax_executor_converges_within_iters(tmp_path):
    """收敛型工具: executor 逐轮续算 (workdir checkpoint), 量差 ≤ tol 即收敛."""
    from huginn.security.compute_adapter import RelaxComputeTool

    ex = ExternalComputeExecutor(
        RelaxComputeTool(), workdir=tmp_path, initial={"energy": 0.0}, max_iterations=20
    )
    a = PhysicalAction("relax", {"n": 1.0, "T": 300.0, "tol": 1e-3})
    ex.execute(a)
    e = ex.observe()["energy"]
    # 收敛后能量逼近解析终态 n·Cv·T
    assert abs(e - (_CV * 300.0)) <= 1e-3


def test_convergent_relax_exceeds_max_iters_raises_pending(tmp_path):
    """收敛型工具超限: 未收敛抛 ConvergencePendingError (改标记挂起, 不硬给错误量)."""
    from huginn.security.compute_adapter import RelaxComputeTool

    ex = ExternalComputeExecutor(
        RelaxComputeTool(),
        workdir=tmp_path,
        initial={"energy": 0.0},
        max_iterations=1,  # 仅一轮子进程, steps=2 不足以达到 tol=1e-9
    )
    a = PhysicalAction("relax", {"n": 1.0, "T": 300.0, "tol": 1e-9, "steps": 2})
    with pytest.raises(ConvergencePendingError):
        ex.execute(a)


def test_relax_world_model_matches_converged_executor(tmp_path):
    """收敛型: 世界模型(解析终态) == 收敛后真实执行(hard 校验)."""
    from huginn.security.compute_adapter import (
        RelaxComputeTool,
        RelaxShellComputeWorldModel,
    )

    ex = ExternalComputeExecutor(
        RelaxComputeTool(), workdir=tmp_path, initial={"energy": 0.0}, max_iterations=20
    )
    wm = RelaxShellComputeWorldModel()
    a = PhysicalAction("relax", {"n": 1.0, "T": 300.0, "tol": 1e-3})
    ex.execute(a)
    pred = wm.predict({}, a)
    assert matches_state(pred, ex.observe(), tolerance=1e-3)


# ── 形态三: 远程 HPC (提交→轮询→取回) ───────────────────────────
class _FakeHpcTransport:
    """假 HPC 调度器: 提交即进队列, 一次轮询后 DONE, 返回缓存 stdout."""

    def __init__(self) -> None:
        self.out = "fake-hpc-result"

    def submit(self, spec: RemoteJobSpec) -> str:
        return "job-0"

    def poll(self, job_id: str) -> str:
        return "PENDING" if job_id == "__first__" else "DONE"

    def fetch(self, job_id: str) -> str:
        return self.out

    def cancel(self, job_id: str) -> None:
        pass


def test_remote_hpc_backend_submit_poll_fetch():
    """远程 HPC 后端: 提交 → 轮询 → 取回 stdout (假 transport 验证编排)."""
    spec = RemoteJobSpec(command=("vasp",))
    be = RemoteHpcJobBackend(_FakeHpcTransport())
    res = be.run(spec, timeout=5.0)
    assert res.ok and res.text == "fake-hpc-result"


def test_remote_hpc_requires_transport():
    """远程 HPC 后端无 transport → ToolInvocationError (不静默死等)."""
    be = RemoteHpcJobBackend(None)
    with pytest.raises(ToolInvocationError):
        be.run(RemoteJobSpec(command=("vasp",)), timeout=5.0)


def test_remote_hpc_times_out_and_cancels():
    """远程作业卡死 → 超时后取消并抛 ToolInvocationError."""

    class _Slow:
        def submit(self, s): return "j"
        def poll(self, j): return "RUNNING"
        def fetch(self, j): return ""
        def cancel(self, j): self.cancelled = True

    t = _Slow()
    be = RemoteHpcJobBackend(t, poll_interval=0.0)
    with pytest.raises(ToolInvocationError):
        be.run(RemoteJobSpec(command=("vasp",)), timeout=0.01)
    assert t.cancelled is True


# ── 形态四(3): 常驻 HTTP API ────────────────────────────────────
def test_http_backend_with_injected_caller():
    """HTTP 后端: 注入 caller 走真实 spec→JobResult 编排, 不起服务即可测."""
    calls: list[HttpJob] = []

    def caller(req: HttpJob) -> JobResult:
        calls.append(req)
        return JobResult(text=req.url, code=200, ok=True)

    be = HttpJobBackend(caller=caller)
    res = be.run(
        HttpJob(method="GET", url="http://x/relax?n=1&T=300"), timeout=5.0
    )
    assert res.ok and res.text == "http://x/relax?n=1&T=300"
    assert calls[0].method == "GET"


def test_http_backend_rejects_wrong_spec():
    be = HttpJobBackend()
    with pytest.raises(ToolInvocationError):
        be.run(JobResult(text="x"), timeout=1.0)


def test_http_relax_tool_wired_through_http_backend():
    """HTTP 形态端到端: HttpRelaxTool adapter 经 HttpJobBackend(注入 caller) 进执行器."""
    from huginn.security.compute_adapter import HttpRelaxTool

    captured: dict[str, str] = {}

    def caller(req: HttpJob) -> JobResult:
        captured["url"] = req.url
        target = _CV * 300.0
        body = (f'{{"energy": {target}, "iteration": 1, "converged": true, '
                f'"target": {target}, "tol": 0.01}}')
        return JobResult(text=body, code=200, ok=True)

    ex = ExternalComputeExecutor(
        HttpRelaxTool(),
        backend=HttpJobBackend(caller=caller),
        initial={"energy": 0.0},
    )
    ex.execute(PhysicalAction("relax", {"n": 1.0, "T": 300.0, "base_url": "http://mp"}))
    assert captured["url"].startswith("http://mp/relax")
    assert ex.observe()["energy"] == pytest.approx(_CV * 300.0)


def test_local_backend_is_default_and_runs_subprocess():
    """LocalJobBackend 是默认后端 (向后兼容), 真跑子进程解析."""
    tool = ShellComputeTool()
    ex = ExternalComputeExecutor(tool, initial={"energy": 0.0})
    ex.execute(PhysicalAction("shell_compute", {"n": 1.0, "T": 300.0}))
    assert ex.observe()["energy"] == pytest.approx(_CV * 300.0)
