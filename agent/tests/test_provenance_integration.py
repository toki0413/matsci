"""Provenance 接进 autoloop engine 的接线测试.

覆盖 ProvenanceRecord / ProvenanceLogger 的核心语义, 以及 engine._execute
把每次 tool call 接进 provenance record 的接线点 (run() 落盘 + tool_chain).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from huginn.autoloop.engine import AutoloopEngine
from huginn.provenance import (
    ProvenanceLogger,
    ProvenanceRecord,
    capture,
)


# ── 1. ProvenanceRecord 构造 + add_snapshot ────────────────────────────────


class TestProvenanceRecordSnapshot:
    def test_create_and_add_snapshot(self):
        rec = ProvenanceRecord(run_id="r1", objective="optimize bandgap")
        snap = capture("vasp_tool", {"encut": 520, "kpar": 4})
        rec.add_snapshot(snap)

        assert rec.run_id == "r1"
        assert rec.objective == "optimize bandgap"
        assert len(rec.tool_chain) == 1
        entry = rec.tool_chain[0]
        assert entry["tool_name"] == "vasp_tool"
        assert entry["input_params"]["encut"] == 520
        # add_snapshot 落进 tool_chain 的是 dict 快照, 不是 ProvenanceSnapshot 对象
        assert isinstance(entry, dict)

    def test_add_snapshot_is_decoupled_from_source(self):
        rec = ProvenanceRecord(run_id="r1")
        snap = capture("qe_tool", {"ecutwfc": 50})
        rec.add_snapshot(snap)
        # 后续改原 snapshot 不应影响已落进 tool_chain 的快照
        snap.input_params["ecutwfc"] = 999
        assert rec.tool_chain[0]["input_params"]["ecutwfc"] == 50


# ── 2. ProvenanceLogger 写 + read_run ──────────────────────────────────────


class TestProvenanceLoggerRoundtrip:
    def test_write_and_read_run(self, tmp_path):
        logger = ProvenanceLogger(path=tmp_path / "prov.jsonl")

        rec = ProvenanceRecord(
            run_id="run_42",
            objective="LJ 团簇能量优化",
            tags=["md", "cluster"],
        )
        rec.add_snapshot(capture("lammps_tool", {"pair_style": "lj/cut"}))
        logger.log(rec)

        loaded = logger.read_run("run_42")
        assert len(loaded) == 1
        assert loaded[0].run_id == "run_42"
        assert loaded[0].objective == "LJ 团簇能量优化"
        assert len(loaded[0].tool_chain) == 1
        assert loaded[0].tool_chain[0]["tool_name"] == "lammps_tool"

    def test_read_run_isolates_other_runs(self, tmp_path):
        logger = ProvenanceLogger(path=tmp_path / "prov.jsonl")
        logger.log(ProvenanceRecord(run_id="a", objective="A"))
        logger.log(ProvenanceRecord(run_id="b", objective="B"))
        logger.log(ProvenanceRecord(run_id="a", objective="A2"))

        a_records = logger.read_run("a")
        assert [r.objective for r in a_records] == ["A", "A2"]


# ── 3. tool_chain 序列化 ────────────────────────────────────────────────────


class TestToolChainSerialization:
    def test_tool_chain_roundtrips_through_json(self):
        rec = ProvenanceRecord(run_id="r1", objective="serialize me")
        rec.add_snapshot(capture("vasp_tool", {"encut": 520}))
        rec.add_snapshot(capture("qe_tool", {"ecutwfc": 50}))

        # to_dict → JSON → from_dict 必须无损保留 tool_chain 顺序和内容
        blob = json.dumps(rec.to_dict(), ensure_ascii=False, default=str)
        restored = ProvenanceRecord.from_dict(json.loads(blob))

        assert len(restored.tool_chain) == 2
        assert [s["tool_name"] for s in restored.tool_chain] == ["vasp_tool", "qe_tool"]
        assert restored.tool_chain[0]["input_params"]["encut"] == 520

    def test_tool_chain_entries_are_plain_dicts(self):
        rec = ProvenanceRecord(run_id="r1")
        rec.add_snapshot(capture("vasp_tool", {"encut": 520}))
        # add_snapshot 落的是 dict, 整条 chain 都该是可 JSON 序列化的 dict
        assert all(isinstance(s, dict) for s in rec.tool_chain)
        json.dumps(rec.to_dict(), default=str)  # 不抛就算过


# ── 4. 空记录处理 ────────────────────────────────────────────────────────────


class TestEmptyRecord:
    def test_empty_record_logs_and_reads_back(self, tmp_path):
        logger = ProvenanceLogger(path=tmp_path / "prov.jsonl")
        logger.log(ProvenanceRecord(run_id="empty_run"))

        loaded = logger.read_run("empty_run")
        assert len(loaded) == 1
        assert loaded[0].tool_chain == []
        assert loaded[0].inputs == {}
        assert loaded[0].outputs == {}

    def test_empty_record_dict_has_all_keys(self):
        d = ProvenanceRecord(run_id="e").to_dict()
        for key in ("run_id", "objective", "inputs", "outputs",
                    "tool_chain", "timestamps", "dois", "tags"):
            assert key in d
        assert d["tool_chain"] == []


# ── 5. 多次 snapshot 保序 ───────────────────────────────────────────────────


class TestSnapshotOrdering:
    def test_multiple_snapshots_preserve_insertion_order(self):
        rec = ProvenanceRecord(run_id="r1")
        names = ["vasp_tool", "qe_tool", "lammps_tool", "gp_tool"]
        for i, name in enumerate(names):
            rec.add_snapshot(capture(name, {"step": i}))

        assert [s["tool_name"] for s in rec.tool_chain] == names
        # input_params 的 step 也得跟顺序对齐
        assert [s["input_params"]["step"] for s in rec.tool_chain] == [0, 1, 2, 3]

    def test_order_survives_jsonl_roundtrip(self, tmp_path):
        logger = ProvenanceLogger(path=tmp_path / "prov.jsonl")
        rec = ProvenanceRecord(run_id="ordered")
        for name in ["a_tool", "b_tool", "c_tool"]:
            rec.add_snapshot(capture(name, {"k": name}))
        logger.log(rec)

        loaded = logger.read_run("ordered")[0]
        assert [s["tool_name"] for s in loaded.tool_chain] == ["a_tool", "b_tool", "c_tool"]


# ── 6. engine 接线: run() 落盘 provenance, _execute 记 tool call ─────────────


class _DummyTracker:
    """Minimal ProgressTracker stand-in — 只吃调用, 不做事."""

    def start_task(self, *a, **kw) -> None: ...
    def update(self, *a, **kw) -> None: ...
    def complete(self, *a, **kw) -> None: ...
    def fail(self, *a, **kw) -> None: ...


@pytest.fixture
def engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AutoloopEngine:
    """建一个所有重子组件都 stub 掉的 engine, _execute 留真实好测接线."""
    monkeypatch.setattr("huginn.autoloop.engine.get_model", lambda settings: MagicMock())
    monkeypatch.setattr("huginn.autoloop.engine.MemoryManager", lambda *a, **kw: MagicMock())
    monkeypatch.setattr("huginn.autoloop.engine.ProjectKnowledgeGraph", lambda *a, **kw: MagicMock())
    monkeypatch.setattr("huginn.autoloop.engine.BenchmarkRunner", lambda *a, **kw: MagicMock())
    monkeypatch.setattr("huginn.autoloop.engine.CoderRunner", lambda *a, **kw: MagicMock())
    monkeypatch.setattr(
        "huginn.agents.speculator.on_turn_start",
        lambda *a, **kw: {"hint": "", "predictions": []},
        raising=False,
    )
    eng = AutoloopEngine(workspace=tmp_path)
    eng.progress_tracker = _DummyTracker()
    return eng


def _stub_phases_keep_execute(engine: AutoloopEngine) -> None:
    """stub 掉 phase 方法, 但 _execute 留真实 (只 stub 它的子执行器)."""
    engine._perceive = lambda: {"changed_files": ["x.py"], "timestamp": "t"}  # type: ignore[assignment]
    engine._hypothesize = AsyncMock(return_value="test hypothesis")  # type: ignore[assignment]
    engine._plan = AsyncMock(return_value={"mode": "coder", "description": "do x"})  # type: ignore[assignment]
    # _execute 不 stub, 它会走真实 dispatch → _execute_coder
    engine._execute_coder = AsyncMock(return_value={"mode": "coder", "status": "ok"})  # type: ignore[assignment]
    engine._validate = AsyncMock(return_value={"tests_passed": True})  # type: ignore[assignment]
    engine._learn = AsyncMock(return_value=None)  # type: ignore[assignment]
    engine._report = AsyncMock(return_value=str(engine.workspace / "report.md"))  # type: ignore[assignment]


class TestEngineProvenanceWiring:
    def test_record_provenance_appends_snapshot(self, engine: AutoloopEngine):
        # 直接验接线点: setup record 后, _record_provenance 得往里加一条快照
        rec = ProvenanceRecord(run_id="direct", objective="wire check")
        engine._provenance_record = rec
        engine._record_provenance("coder", {"mode": "coder", "description": "x"}, {"ok": True})

        assert len(rec.tool_chain) == 1
        assert rec.tool_chain[0]["tool_name"] == "coder"
        assert rec.tool_chain[0]["input_params"]["mode"] == "coder"

    def test_record_provenance_noop_without_record(self, engine: AutoloopEngine):
        # 没建 record (单测里直接调 _execute) 不能炸, 静默跳过
        engine._provenance_record = None
        engine._record_provenance("coder", {"x": 1}, None)  # 不抛就算过

    def test_run_persists_provenance_with_tool_chain(self, engine: AutoloopEngine):
        _stub_phases_keep_execute(engine)
        result = asyncio.run(
            engine.run(objective="o", max_iterations=1, progressive_budget=False)
        )

        # run() 结束得带上 provenance_path, 且文件真落盘了
        assert result.provenance_path is not None
        prov_path = Path(result.provenance_path)
        assert prov_path.exists()

        # JSONL 能读回这条 run, tool_chain 里记了 coder 这步
        logger = ProvenanceLogger(path=prov_path)
        records = logger.read_run(result.run_id)
        assert len(records) == 1
        tool_names = [s["tool_name"] for s in records[0].tool_chain]
        assert "coder" in tool_names
        assert records[0].objective == "o"
