"""I5/§10: PersonaKernel 安全/授权边界接入准入（deny/allow 清单）."""
from __future__ import annotations

from types import SimpleNamespace

from huginn.core_types import RiskLevel
from huginn.normative.admission import AdmissionRequest, admit
from huginn.normative.ledger import NormativeLedger
from huginn.persona_kernel import (
    PersonaKernel,
    PersonaKernelStore,
    parse_kernel_boundaries,
    sync_from_persona,
)

TENANT = "t1"
SESSION = "s1"


def _kernel() -> PersonaKernel:
    return PersonaKernel(
        persona_id="p1",
        role="scientist",
        safety_constraints=(
            "deny_action:bash_tool",
            "deny_resource:/etc/*",
        ),
        authorization_boundaries=(
            "allow_action:write",
            "allow_action:file_read_tool",
            "allow_resource:workspace/*",
        ),
    )


def test_kernel_boundary_parse_and_store(tmp_path):
    boundaries = parse_kernel_boundaries(_kernel())
    assert boundaries["denied_actions"] == ["bash_tool"]
    assert boundaries["denied_resources"] == ["/etc/*"]
    assert boundaries["allowed_actions"] == ["write", "file_read_tool"]
    assert boundaries["allowed_resources"] == ["workspace/*"]

    store = PersonaKernelStore(tmp_path / "kernels.db")
    store.register(_kernel(), tenant_id="t1")
    assert store.get("p1", tenant_id="t1").role == "scientist"
    assert store.boundaries_for("p1", tenant_id="t1") == boundaries
    assert store.boundaries_for("p1", tenant_id="t2") == {}, "跨租户不可见"


def test_admission_enforces_kernel_boundaries(tmp_path):
    ledger = NormativeLedger(tmp_path / "norm.db")
    view = ledger.project(TENANT, SESSION)
    boundaries = parse_kernel_boundaries(_kernel())

    # 允许清单外动作 → deny
    denied_action = AdmissionRequest(
        tenant_id=TENANT, session_id=SESSION, action_name="bash_tool",
        risk=RiskLevel.HIGH,
    )
    assert admit(denied_action, view, persona_boundaries=boundaries).mode == "deny"

    # 允许动作但资源越界 → deny
    denied_resource = AdmissionRequest(
        tenant_id=TENANT, session_id=SESSION, action_name="write",
        resources=("/etc/passwd",), risk=RiskLevel.MEDIUM,
    )
    assert admit(
        denied_resource, view, persona_boundaries=boundaries
    ).mode == "deny"

    # 允许动作 + 允许资源 → 通过边界，进入授权判定（无授权 → ask）
    inside = AdmissionRequest(
        tenant_id=TENANT, session_id=SESSION, action_name="write",
        resources=("workspace/a.txt",), risk=RiskLevel.MEDIUM,
    )
    decision = admit(inside, view, persona_boundaries=boundaries)
    assert decision.mode == "ask", decision


def test_shadow_tool_gate_reads_kernel(monkeypatch, tmp_path):
    """生产路径：evaluate_tool_call 接收 PersonaKernel 边界并 deny."""
    import huginn.normative.shadow as shadow

    ledger = NormativeLedger(tmp_path / "norm.db")
    monkeypatch.setattr(shadow, "_ledger", ledger)
    shadow._view_cache.clear()
    monkeypatch.setenv("HUGINN_NORMATIVE_SHADOW", "1")

    decision = shadow.evaluate_tool_call(
        tenant_id=TENANT,
        session_id=SESSION,
        tool_name="bash_tool",
        persona_boundaries=parse_kernel_boundaries(_kernel()),
        trust="full",
    )
    assert decision is not None
    assert decision.mode == "deny"
    assert "persona" in " ".join(decision.reasons)


def test_sync_from_persona_registers_kernel(tmp_path, monkeypatch):
    import huginn.persona_kernel as kernel_mod

    store = PersonaKernelStore(tmp_path / "kernels.db")
    monkeypatch.setattr(kernel_mod, "get_persona_kernel_store", lambda: store)
    persona = SimpleNamespace(
        name="dft_expert",
        variables={
            "persona_kernel": {
                "role": "dft expert",
                "safety_constraints": ["deny_action:bash_tool"],
                "authorization_boundaries": ["allow_resource:workspace/*"],
            }
        },
    )
    synced = sync_from_persona(persona, tenant_id="t1")
    assert synced is not None
    assert synced.persona_id == "dft_expert"
    assert store.boundaries_for("dft_expert", tenant_id="t1")["denied_actions"] == [
        "bash_tool"
    ]
