"""T1: contract.json 的强类型加载、字段校验与字节指纹."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from huginn.task_contract import (
    CONTRACT_SCHEMA,
    ContractError,
    LoopLogSpec,
    RequiredKey,
    TaskContract,
    check_required_keys,
    load_contract,
)


def _base(**over: object) -> dict:
    doc: dict = {
        "schema": CONTRACT_SCHEMA,
        "required_files": ["analysis.py", "results.json", "loop_log.jsonl", "report.md"],
        "required_keys": [
            {"key": "q1_delta_ea", "type": "number"},
            {"key": "q1_n_papers", "type": "integer"},
            {"key": "hypotheses", "type": "object"},
        ],
        "hypotheses": {"labels": ["P1", "P2"], "prediction_only": ["P2"]},
        "loop_log": {
            "path": "loop_log.jsonl",
            "lines": 50,
            "fields": ["loop_id", "status", "delta_ea"],
        },
        "static": {"forbidden_imports": ["pandas"], "seed": 20261029},
    }
    doc.update(over)
    return doc


def _write(tmp_path: Path, doc: object, name: str = "contract.json") -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def test_load_minimal_contract(tmp_path: Path) -> None:
    p = _write(tmp_path, _base())
    c = load_contract(p)
    assert isinstance(c, TaskContract)
    assert c.schema == CONTRACT_SCHEMA
    assert c.required_files == ("analysis.py", "results.json", "loop_log.jsonl", "report.md")
    assert c.required_keys[0] == RequiredKey("q1_delta_ea", "number", "results.json")
    assert c.hypotheses is not None
    assert c.hypotheses.labels == ("P1", "P2")
    assert c.hypotheses.prediction_only == frozenset({"P2"})
    assert isinstance(c.loop_log, LoopLogSpec)
    assert (c.loop_log.path, c.loop_log.lines) == ("loop_log.jsonl", 50)
    assert c.loop_log.fields == ("loop_id", "status", "delta_ea")
    assert c.static is not None
    assert c.static.forbidden_imports == ("pandas",)
    assert c.static.seed == 20261029
    assert c.source == str(p)


def test_sha256_is_byte_fingerprint(tmp_path: Path) -> None:
    p = _write(tmp_path, _base())
    c = load_contract(p)
    assert c.sha256 == hashlib.sha256(p.read_bytes()).hexdigest()

    # 只改一个字符 (甚至只是缩进) 指纹就要变 —— 门校验的是同一份字节.
    p.write_text(p.read_text(encoding="utf-8").replace("20261029", "20261039"), encoding="utf-8")
    c2 = load_contract(p)
    assert c2.sha256 != c.sha256
    assert c2.static is not None
    assert c2.static.seed == 20261039


def test_missing_file_raises_contract_error(tmp_path: Path) -> None:
    with pytest.raises(ContractError):
        load_contract(tmp_path / "contract.json")


def test_invalid_json_raises_contract_error(tmp_path: Path) -> None:
    p = tmp_path / "contract.json"
    p.write_text("{not json", encoding="utf-8")
    with pytest.raises(ContractError):
        load_contract(p)


@pytest.mark.parametrize(
    "field",
    ["schema", "required_files", "required_keys", "hypotheses", "loop_log", "static"],
)
def test_missing_top_level_field_raises(tmp_path: Path, field: str) -> None:
    doc = _base()
    doc.pop(field)
    with pytest.raises(ContractError, match=field):
        load_contract(_write(tmp_path, doc))


def test_wrong_schema_version_raises(tmp_path: Path) -> None:
    with pytest.raises(ContractError, match="schema"):
        load_contract(_write(tmp_path, _base(schema="huginn.task-contract/v0")))


def test_unknown_top_level_field_raises(tmp_path: Path) -> None:
    with pytest.raises(ContractError, match="未知字段"):
        load_contract(_write(tmp_path, _base(surprise=True)))


def test_required_files_must_be_relative(tmp_path: Path) -> None:
    with pytest.raises(ContractError, match="相对路径"):
        load_contract(_write(tmp_path, _base(required_files=["/tmp/escape.md"])))


def test_required_keys_bad_type_rejected(tmp_path: Path) -> None:
    doc = _base(required_keys=[{"key": "q1", "type": "float"}])
    with pytest.raises(ContractError, match="type"):
        load_contract(_write(tmp_path, doc))


def test_prediction_only_must_be_subset(tmp_path: Path) -> None:
    doc = _base(hypotheses={"labels": ["P1"], "prediction_only": ["P2"]})
    with pytest.raises(ContractError, match="prediction_only"):
        load_contract(_write(tmp_path, doc))


def test_loop_log_negative_lines_rejected(tmp_path: Path) -> None:
    doc = _base(loop_log={"path": "loop_log.jsonl", "lines": -1, "fields": []})
    with pytest.raises(ContractError, match="lines"):
        load_contract(_write(tmp_path, doc))


def test_null_sections_mean_not_applicable(tmp_path: Path) -> None:
    doc = _base(hypotheses=None, loop_log=None, static=None)
    c = load_contract(_write(tmp_path, doc))
    assert c.hypotheses is None
    assert c.loop_log is None
    assert c.static is None


def test_check_required_keys_type_and_missing() -> None:
    keys = (
        RequiredKey("q1_delta_ea", "number"),
        RequiredKey("q1_n_papers", "integer"),
        RequiredKey("q1_ci_low", "number"),
        RequiredKey("hypotheses", "object"),
    )
    flat = {"q1_delta_ea": "12.5", "q1_n_papers": 3, "q1_ci_low": True}
    problems = check_required_keys(flat, keys)
    assert any("q1_delta_ea" in p and "number" in p for p in problems)
    assert not any("q1_n_papers" in p for p in problems)
    # bool 是 int 的子类, 但不能当 number/integer 用.
    assert any("q1_ci_low" in p for p in problems)
    assert any("hypotheses" in p and "missing" in p for p in problems)


# ===== T3: 启动校验 (driver 起跑前) =====


def test_driver_refuses_to_start_without_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """无 contract.json 且未显式放行 → 拒绝启动 (不猜验收口径)."""
    from huginn.run_cognitive_driver import ensure_contract

    monkeypatch.delenv("HUGINN_ALLOW_UNCONTRACTED", raising=False)
    with pytest.raises(ContractError, match="HUGINN_ALLOW_UNCONTRACTED"):
        ensure_contract(tmp_path)


def test_driver_allows_uncontracted_only_with_explicit_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from huginn.run_cognitive_driver import ensure_contract

    monkeypatch.setenv("HUGINN_ALLOW_UNCONTRACTED", "1")
    assert ensure_contract(tmp_path) is None


def test_driver_loads_and_validates_contract_on_startup(tmp_path: Path) -> None:
    from huginn.run_cognitive_driver import ensure_contract

    p = _write(tmp_path, _base())
    c = ensure_contract(tmp_path)
    assert c is not None and c.sha256 == hashlib.sha256(p.read_bytes()).hexdigest()


def test_driver_surfaces_broken_contract_instead_of_running(tmp_path: Path) -> None:
    from huginn.run_cognitive_driver import ensure_contract

    _write(tmp_path, _base(schema="huginn.task-contract/v0"))
    with pytest.raises(ContractError, match="schema"):
        ensure_contract(tmp_path)
