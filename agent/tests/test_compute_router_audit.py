"""M1 分流审计单测 — 数据驱动阈值 + 路由决策写 audit.

覆盖:
- 默认路由行为复刻 (DFT/MD 100 原子 / 1h, QC 50 原子, 未登记 generic 恒 local,
  user_preference 覆盖);
- 数据驱动覆盖 (register_tool / 构造 tool_specs 换阈值);
- 路由决策落 append-only audit (compute_route 事件含 tool/target/reason).
"""

from __future__ import annotations

import pytest

from huginn.execution.compute_policy import ComputePolicy
from huginn.execution.compute_router import ComputeRouter, ComputeToolSpec
from huginn.execution.orchestrator import ExecutionOrchestrator
from huginn.feature_flags import FeatureFlags
from huginn.security.audit import AuditLogger


def test_default_dft_md_thresholds_preserved():
    r = ComputeRouter()
    assert r.route("vasp_tool", "relax", {"n_atoms": 200}).target == "hpc"
    assert r.route("lammps_tool", "run", {"n_atoms": 10}).target == "local"
    # 墙钟 > 1h → hpc
    assert r.route("vasp_tool", "scf", {"walltime": "2h"}).target == "hpc"
    # 原子/墙钟都未超 → local, 文案复刻
    d = r.route("vasp_tool", "scf", {"n_atoms": 10})
    assert d.target == "local"
    assert d.reason == "below DFT/MD HPC thresholds"


def test_default_qc_thresholds_preserved():
    r = ComputeRouter()
    assert r.route("gaussian_tool", "opt", {"n_atoms": 60}).target == "hpc"
    assert r.route("orca_tool", "sp", {"n_atoms": 10}).target == "local"
    assert r.route("gaussian_tool", "opt", {"n_atoms": 10}).reason == "below QC HPC threshold"


def test_unknown_tool_defaults_local():
    assert ComputeRouter().route("some_unknown_tool", "x", {}).target == "local"


def test_user_preference_overrides():
    r = ComputeRouter()
    assert r.route("gaussian_tool", "opt", {"n_atoms": 10, "execution_backend": "hpc"}).target == "hpc"
    assert r.route("vasp_tool", "relax", {"execution_backend": "remote"}).target == "hpc"
    assert r.route("vasp_tool", "relax", {"execution_backend": "local"}).target == "local"


def test_register_tool_overrides_threshold():
    r = ComputeRouter()
    r.register_tool("custom_tool", ComputeToolSpec(scaling="dft_md", atom_threshold=50))
    assert r.route("custom_tool", "run", {"n_atoms": 60}).target == "hpc"
    assert r.route("custom_tool", "run", {"n_atoms": 10}).target == "local"


def test_constructor_tool_specs_injected():
    specs = {"vasp_tool": ComputeToolSpec(scaling="dft_md", atom_threshold=500)}
    r = ComputeRouter(tool_specs=specs)
    assert r.route("vasp_tool", "relax", {"n_atoms": 400}).target == "local"
    assert r.route("vasp_tool", "relax", {"n_atoms": 600}).target == "hpc"
    # 未覆盖工具仍走默认
    assert r.route("gaussian_tool", "opt", {"n_atoms": 60}).target == "hpc"


def test_route_audit_event_written(tmp_path, monkeypatch):
    """_audit_compute_route 落一条 compute_route 审计事件 (含 target/reason)."""
    log = AuditLogger(tmp_path / "audit.jsonl")
    monkeypatch.setattr("huginn.security.audit.get_audit_logger", lambda: log)

    orch = ExecutionOrchestrator(working_dir=str(tmp_path))
    orch._audit_compute_route(
        "s1", "vasp_tool", "relax", {"n_atoms": 200}, "hpc", "n_atoms=200 > 100"
    )
    events = log.query(event_type="compute_route")
    assert len(events) == 1
    rec = events[0]
    assert rec["details"]["tool"] == "vasp_tool"
    assert rec["details"]["target"] == "hpc"
    assert rec["details"]["reason"].startswith("n_atoms=")
    # 参数只存哈希不落明文
    assert rec["input_hash"]


@pytest.mark.asyncio
async def test_orchestrator_run_writes_route_audit(tmp_path, monkeypatch):
    """run() 走完整 stage 后, 路由决策已进 audit."""
    log = AuditLogger(tmp_path / "audit.jsonl")
    monkeypatch.setattr("huginn.security.audit.get_audit_logger", lambda: log)

    async def _tool(action: str, **params):  # noqa: ANN001
        return {"ok": True, "action": action}

    orch = ExecutionOrchestrator(
        working_dir=str(tmp_path), tool_registry={"vasp_tool": _tool}
    )
    rec = await orch.run(
        [{"id": "s1", "tool": "vasp_tool", "action": "relax", "params": {"n_atoms": 200}}]
    )
    assert rec.stage_results[0].execution_target == "hpc"
    events = log.query(event_type="compute_route")
    assert any(e["details"]["tool"] == "vasp_tool" for e in events)


# ── M2: 细粒度权限 + 预算 ─────────────────────────────────────────

def _enable_compute_policy(monkeypatch) -> FeatureFlags:
    """构造一个开了 compute_policy 的 FeatureFlags 并顶掉 shared 单例."""
    ff = FeatureFlags()
    ff.toggle("compute_policy", True)
    monkeypatch.setattr("huginn.feature_flags.FeatureFlags.shared", lambda: ff)
    return ff


def test_policy_requires_approval_for_hpc_nonelevated():
    p = ComputePolicy()
    v = p.enforce("gaussian_tool", "hpc", "alice", scaling="qc")
    assert v.allowed is True
    assert v.requires_approval is True
    assert "approval" in v.reason


def test_policy_elevated_actor_skips_gate():
    p = ComputePolicy(elevated_actors=["root"])
    v = p.enforce("gaussian_tool", "hpc", "root", scaling="qc")
    assert v.requires_approval is False
    assert v.allowed is True


def test_policy_heavy_quota_exhausted():
    p = ComputePolicy(max_heavy_per_window=2)
    for _ in range(2):
        v = p.enforce("vasp_tool", "local", "alice", scaling="dft_md")
        assert v.allowed is True
    v = p.enforce("vasp_tool", "local", "alice", scaling="dft_md")
    assert v.allowed is False
    assert "budget" in v.reason


def test_policy_quota_per_actor_independent():
    p = ComputePolicy(max_heavy_per_window=1)
    assert p.enforce("vasp_tool", "local", "a", scaling="dft_md").allowed
    assert p.enforce("vasp_tool", "local", "b", scaling="dft_md").allowed
    assert p.enforce("vasp_tool", "local", "a", scaling="dft_md").allowed is False


def test_policy_nonheavy_generic_not_budgeted():
    p = ComputePolicy(max_heavy_per_window=1)
    # generic 不计 heavy → 可无限
    for _ in range(3):
        assert p.enforce("thermo_tool", "local", "alice", scaling="generic").allowed


@pytest.mark.asyncio
async def test_orchestrator_policy_blocks_nonelevated_hpc(tmp_path, monkeypatch):
    """开 compute_policy 后, 非提升 actor 路由到 hpc 的调用被拦截."""
    _enable_compute_policy(monkeypatch)
    log = AuditLogger(tmp_path / "audit.jsonl")
    monkeypatch.setattr("huginn.security.audit.get_audit_logger", lambda: log)

    async def _tool(action: str, **params):  # noqa: ANN001
        return {"ok": True}

    orch = ExecutionOrchestrator(
        working_dir=str(tmp_path), tool_registry={"vasp_tool": _tool}
    )
    rec = await orch.run(
        [{"id": "s1", "tool": "vasp_tool", "action": "relax", "params": {"n_atoms": 500}}]
    )
    sr = rec.stage_results[0]
    assert sr.success is False
    assert "approval" in (sr.error_message or "")
    # 决策已审计
    assert log.query(event_type="compute_policy")


@pytest.mark.asyncio
async def test_orchestrator_policy_off_still_runs(tmp_path, monkeypatch):
    """compute_policy 默认关 ⇒ 不拦截, 照常执行."""
    log = AuditLogger(tmp_path / "audit.jsonl")
    monkeypatch.setattr("huginn.security.audit.get_audit_logger", lambda: log)

    async def _tool(action: str, **params):  # noqa: ANN001
        return {"ok": True}

    orch = ExecutionOrchestrator(
        working_dir=str(tmp_path), tool_registry={"vasp_tool": _tool}
    )
    rec = await orch.run(
        [{"id": "s1", "tool": "vasp_tool", "action": "relax", "params": {"n_atoms": 500}}]
    )
    assert rec.stage_results[0].success is True


# ── M3: 设备私有化 (compute local_only) ───────────────────────────

def test_router_local_only_overrides_heuristic_hpc():
    r = ComputeRouter()
    # 默认 → hpc
    assert r.route("vasp_tool", "relax", {"n_atoms": 500}).target == "hpc"
    # local_only → 强制本地
    d = r.route("vasp_tool", "relax", {"n_atoms": 500}, local_only=True)
    assert d.target == "local"
    assert "local_only" in d.reason


def test_router_local_only_overrides_user_backend():
    r = ComputeRouter()
    assert r.route("gaussian_tool", "opt", {"execution_backend": "hpc"}).target == "hpc"
    d = r.route("gaussian_tool", "opt", {"execution_backend": "hpc"}, local_only=True)
    assert d.target == "local"
    assert "overridden" in d.reason
    # 本地偏好 + local_only → 仍本地
    assert r.route("vasp_tool", "relax", {"execution_backend": "local"}, local_only=True).target == "local"


def test_router_device_is_first_class_target():
    r = ComputeRouter()
    assert r.route("vasp_tool", "relax", {"execution_backend": "device"}).target == "device"


def test_backend_access_gate_local_only():
    from huginn.security.compute_adapter import (
        HttpJobBackend,
        LocalJobBackend,
        RemoteHpcJobBackend,
        backend_allows_local_only,
    )
    assert LocalJobBackend.backend_kind == "local"
    assert RemoteHpcJobBackend.backend_kind == "remote_hpc"
    assert HttpJobBackend.backend_kind == "http"
    # local/device 允许, remote/hpc 禁止, unknown 保守允许
    assert backend_allows_local_only("local") is True
    assert backend_allows_local_only("device") is True
    assert backend_allows_local_only("remote_hpc") is False
    assert backend_allows_local_only("http") is False
    assert backend_allows_local_only(None) is True


@pytest.mark.asyncio
async def test_orchestrator_local_only_does_not_block_normal_task(tmp_path):
    """P1(修正版): local_only=True 也不妨碍普通任务 — 不敏感数据照常投 hpc 并成功."""
    async def _tool(action: str, **params):  # noqa: ANN001
        return {"ok": True}

    orch = ExecutionOrchestrator(
        working_dir=str(tmp_path),
        tool_registry={"vasp_tool": _tool},
        local_only=True,
    )
    rec = await orch.run(
        [{"id": "s1", "tool": "vasp_tool", "action": "relax", "params": {"n_atoms": 500}}]
    )
    sr = rec.stage_results[0]
    assert sr.success is True
    # 不敏感 → 不被私有化拦截, 仍按路由本领地/hpc
    assert sr.execution_target == "hpc"


@pytest.mark.asyncio
async def test_orchestrator_sensitive_call_forced_local(tmp_path, monkeypatch):
    """P1(修正版): 只有敏感数据 (ephemeral 路径) 才强制本地, 且落审计."""
    log = AuditLogger(tmp_path / "audit.jsonl")
    monkeypatch.setattr("huginn.security.audit.get_audit_logger", lambda: log)

    async def _tool(action: str, **params):  # noqa: ANN001
        return {"ok": True}

    orch = ExecutionOrchestrator(
        working_dir=str(tmp_path), tool_registry={"vasp_tool": _tool}
    )
    rec = await orch.run(
        [{
            "id": "s1",
            "tool": "vasp_tool",
            "action": "relax",
            "params": {"path": "/root/private.key", "n_atoms": 500},
        }]
    )
    sr = rec.stage_results[0]
    # 敏感数据路由本会 hpc, 但被强制本地
    assert sr.execution_target == "local"
    assert "privacy" in (sr.route_reason or "")
    # 覆盖已审计
    events = log.query(event_type="privacy_enforce")
    assert any(e["details"]["tool"] == "vasp_tool" for e in events)


# ── P1: 逐调用敏感判定单元 ─────────────────────────────────────

def test_decide_privacy_defaults_to_allow():
    from huginn.execution.privacy_decision import decide_privacy
    from huginn.privacy_guard import PrivacyGuard

    pg = PrivacyGuard()
    d = decide_privacy("vasp relax 500 atoms quick run", privacy=pg)
    assert d.sensitive is False
    assert d.signal is None


def test_decide_privacy_ephemeral_is_sensitive():
    from huginn.execution.privacy_decision import decide_privacy
    from huginn.privacy_guard import PrivacyGuard

    pg = PrivacyGuard()
    d = decide_privacy("relax with /home/user/api_key = sk-secret123", privacy=pg)
    assert d.sensitive is True
    assert d.signal == "ephemeral"


def test_decide_privacy_local_only_tag_is_sensitive():
    from huginn.execution.privacy_decision import decide_privacy
    from huginn.privacy_guard import PrivacyGuard

    pg = PrivacyGuard()
    pg.tag_local_only("/secret/dir")
    d = decide_privacy("process /secret/dir/results.json", privacy=pg)
    assert d.sensitive is True
    assert d.signal == "local_only_tag"


def test_decide_privacy_conservative_temp_only_if_requested():
    from huginn.execution.privacy_decision import decide_privacy
    from huginn.privacy_guard import PrivacyGuard

    pg = PrivacyGuard()
    content = "POSCAR\nSi\n1.0\nDirect\n0 0 0\n"  # 结构 → temporary
    # 默认: 结构原始数据不阻断, 允许
    assert decide_privacy(content, privacy=pg).sensitive is False
    # 私密设备档: temporary 也留本地
    d = decide_privacy(content, privacy=pg, conservative_temporary=True)
    assert d.sensitive is True
    assert d.signal == "private_device_temporary"


# ── 用户自由度: 显式覆盖出口 ─────────────────────────────────

def test_decide_privacy_override_allow_wins():
    from huginn.execution.privacy_decision import decide_privacy
    from huginn.privacy_guard import PrivacyGuard

    pg = PrivacyGuard()
    # 敏感数据 (ephemeral) 默认 → 本地; 用户 override "allow" → 放行
    content = "vasp /home/user/api_key = sk-secret123"
    assert decide_privacy(content, privacy=pg).sensitive is True
    d = decide_privacy(content, privacy=pg, override="allow")
    assert d.sensitive is False
    assert d.signal == "user_override_allow"


def test_decide_privacy_override_local_wins():
    from huginn.execution.privacy_decision import decide_privacy
    from huginn.privacy_guard import PrivacyGuard

    pg = PrivacyGuard()
    # 不敏感普通数据默认放行; 用户 override "local" → 强制本地
    assert decide_privacy("vasp 500 atoms", privacy=pg).sensitive is False
    d = decide_privacy("vasp 500 atoms", privacy=pg, override="local")
    assert d.sensitive is True
    assert d.signal == "user_override_local"


@pytest.mark.asyncio
async def test_orchestrator_user_override_allow_sends_remote(tmp_path, monkeypatch):
    """用户显式 allow → 即便敏感数据也照常投 hpc (自由度), 且覆盖被审计."""
    log = AuditLogger(tmp_path / "audit.jsonl")
    monkeypatch.setattr("huginn.security.audit.get_audit_logger", lambda: log)

    async def _tool(action: str, **params):  # noqa: ANN001
        return {"ok": True}

    orch = ExecutionOrchestrator(
        working_dir=str(tmp_path), tool_registry={"vasp_tool": _tool}
    )
    rec = await orch.run(
        [{
            "id": "s1",
            "tool": "vasp_tool",
            "action": "relax",
            # 敏感路径, 但用户显式 privacy_override="allow"
            "params": {"path": "/root/private.key", "n_atoms": 500, "privacy_override": "allow"},
        }]
    )
    sr = rec.stage_results[0]
    assert sr.success is True
    assert sr.execution_target == "hpc"
    # 覆盖已审计 (signal=user_override_allow)
    events = log.query(event_type="privacy_enforce")
    assert any("user_override_allow" in (e["details"].get("signal") or "") for e in events)
