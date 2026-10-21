"""§20.7/I41–I43: 经验账的世界模型对账标签与权威上限."""
from __future__ import annotations

from huginn.research.experience_archive import ExecutableExperience


def _exp(**overrides) -> ExecutableExperience:
    data = {
        "goal": "band gap",
        "dimension": "electronic",
        "config": {"cutoff": 400},
        "objectives": {"gap": 1.2},
        "world_model_id": "law.exoplanet",
        "worldview": "physics_causal",
        "law": "T_eq law",
        "contract_version": "v1",
    }
    data.update(overrides)
    return ExecutableExperience(**data)


def test_evidential_authority_ladder():
    prediction_only = _exp(reconcile_status="insufficient")
    assert prediction_only.evidential_authority() <= 0.4

    reconciled = _exp(reconcile_status="borne_out")
    assert reconciled.evidential_authority() == 0.6

    replicated = _exp(
        reconcile_status="borne_out", ground_truth_source="tool_run:42"
    )
    assert replicated.evidential_authority() == 0.9

    falsified = _exp(reconcile_status="falsified")
    assert falsified.evidential_authority() <= 0.1

    non_world = ExecutableExperience(
        goal="g", dimension="d", config={}, objectives={"x": 1.0}
    )
    assert non_world.evidential_authority() <= 0.4


def test_labels_roundtrip():
    exp = _exp(
        reconcile_status="borne_out",
        ground_truth_source="pybullet_run:7",
        prediction_error=0.01,
    )
    payload = exp.to_dict()
    assert payload["world_model_id"] == "law.exoplanet"
    assert payload["reconcile_status"] == "borne_out"
    restored = ExecutableExperience.from_dict(payload)
    assert restored.worldview == "physics_causal"
    assert restored.contract_version == "v1"
    assert restored.prediction_error == 0.01
    assert restored.ground_truth_source == "pybullet_run:7"
    assert restored.evidential_authority() == 0.9
