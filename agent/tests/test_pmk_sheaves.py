"""PMK 双 sheaf 不变量：跨 worldview 不粘合、冲突必须可寻址."""
from __future__ import annotations

from huginn.research.world_model_binding import cross_glue_labels
from huginn.runtime.task_lifecycle import _check_pmk_axes


def test_worldview_mismatch():
    """I49: 跨 worldview 的差异不是矛盾，不粘合、不阻断."""
    wm = "world_model=x worldview=physics_causal domain=d contract=v1"
    obstructions = cross_glue_labels(
        wm, {"kb": "obs [worldview=latent_transition domain=d contract=v1]"}
    )
    assert obstructions[0]["kind"] == "worldview_mismatch"

    axes = _check_pmk_axes(
        {
            "world_model": (
                "world_model=x worldview=physics_causal domain=d contract=v1"
            ),
            "kb": "obs [worldview=latent_transition domain=d contract=v1]",
        }
    )
    assert not axes.blocked
    assert "worldview_mismatch" in axes.advisory


def test_addressable_conflict():
    """I6: 认识论冲突必须引用 ≥2 个可寻址节点；纯文本只作 advisory."""
    from huginn.research.belief_ledger import BeliefLedger, BeliefRecord

    ledger = BeliefLedger(":memory:")
    first = BeliefRecord(
        belief_id="b1", tenant_id="t1", subject="band_gap", claim="3.4 eV",
        provenance=("doi:10.1/a",), valid_from="2026-01-01T00:00:00+00:00",
    )
    second = BeliefRecord(
        belief_id="b2", tenant_id="t1", subject="band_gap", claim="3.2 eV",
        provenance=("doi:10.1/b",), valid_from="2026-02-01T00:00:00+00:00",
    )
    ledger.append(first)
    ledger.append(second)
    obstruction = ledger.obstruction("t1", "band_gap")
    assert obstruction is not None
    assert len(obstruction.refs) >= 2, "冲突必须可寻址到 ≥2 个节点"
    assert not obstruction.blocked, "认识论冲突只 advisory"
