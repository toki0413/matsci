"""Provenance 测试 — 归并原 test_provenance.py / test_provenance_base.py /
test_provenance_integration.py 三个文件, 覆盖 provenance 三层:

  1. 数据层: ProvenanceRecord + ProvenanceLogger + export_crate
     (record 构造 / snapshot 追加 / JSONL 追加读写 / run 过滤 /
      file hash / ROCrate 导出结构 / 损坏行容错)
  2. 捕获层: HuginnTool.call 自动捕获 (ProvenanceSnapshot + contextvar collector,
     含 legacy tool 遮蔽与 feature flag 关闭)
  3. 接线层: autoloop engine._execute 把每次 tool call 接进 provenance record
     (run() 落盘 + tool_chain)
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from huginn.autoloop.engine import AutoloopEngine
from huginn.core_types import ToolResult
from huginn.provenance import (
    ProvenanceLogger,
    ProvenanceRecord,
    ProvenanceSnapshot,
    capture,
    capture_run_inputs,
    export_crate,
)
from huginn.tools.base import (
    HuginnTool,
    get_provenance_collector,
    set_provenance_collector,
)

_skip_ci_run_cognitive = os.environ.get("HUGINN_CI", "").lower() in ("1", "true", "yes")

# ═══════════════════════════════════════════════════════════════════════════
# 1. 数据层 — ProvenanceRecord / ProvenanceLogger / export_crate
# ═══════════════════════════════════════════════════════════════════════════

# ── ProvenanceRecord 基本语义 ───────────────────────────────────────────────


class TestProvenanceRecord:
    def test_default_empty(self):
        r = ProvenanceRecord(run_id="r1")
        assert r.run_id == "r1"
        assert r.objective == ""
        assert r.inputs == {}
        assert r.outputs == {}
        assert r.tool_chain == []
        assert r.dois == []

    def test_add_snapshot_appends(self):
        r = ProvenanceRecord(run_id="r1")
        snap = capture("vasp_tool", {"encut": 520})
        r.add_snapshot(snap)
        assert len(r.tool_chain) == 1
        assert r.tool_chain[0]["tool_name"] == "vasp_tool"

    def test_to_dict_roundtrip(self):
        r = ProvenanceRecord(
            run_id="r1",
            objective="optimize bandgap",
            inputs={"params": {"encut": 520}},
            outputs={"bandgap": 1.2},
            dois=["10.1234/abc"],
            tags=["dft", "bandgap"],
        )
        d = r.to_dict()
        assert d["run_id"] == "r1"
        assert d["objective"] == "optimize bandgap"
        r2 = ProvenanceRecord.from_dict(d)
        assert r2.run_id == r.run_id
        assert r2.objective == r.objective
        assert r2.dois == r.dois

    def test_from_dict_ignores_unknown_keys(self):
        d = {"run_id": "r1", "bogus": 123, "objective": "x"}
        r = ProvenanceRecord.from_dict(d)
        assert r.run_id == "r1"
        assert r.objective == "x"
        assert not hasattr(r, "bogus")


# ── capture_run_inputs ──────────────────────────────────────────────────────


class TestCaptureRunInputs:
    def test_params_only(self):
        inputs = capture_run_inputs(params={"encut": 520, "kpar": 4})
        assert inputs["params"]["encut"] == 520
        assert inputs["files"] == {}

    def test_file_hash(self, tmp_path):
        f = tmp_path / "POSCAR"
        f.write_text("test content", encoding="utf-8")
        inputs = capture_run_inputs(files=[str(f)], params={"x": 1})
        assert "POSCAR" in inputs["files"]
        assert len(inputs["files"]["POSCAR"]) == 12  # sha256 前 12 位

    def test_missing_file_empty_hash(self):
        inputs = capture_run_inputs(files=["/nonexistent/file.txt"])
        assert inputs["files"]["file.txt"] == ""


# ── ProvenanceLogger ────────────────────────────────────────────────────────


class TestProvenanceLogger:
    def test_log_creates_file(self, tmp_path):
        path = tmp_path / "prov.jsonl"
        logger = ProvenanceLogger(path=str(path))
        logger.log(ProvenanceRecord(run_id="r1", objective="test"))
        assert path.exists()
        line = path.read_text(encoding="utf-8").strip()
        d = json.loads(line)
        assert d["run_id"] == "r1"

    def test_log_appends_multiple(self, tmp_path):
        path = tmp_path / "prov.jsonl"
        logger = ProvenanceLogger(path=str(path))
        logger.log(ProvenanceRecord(run_id="r1"))
        logger.log(ProvenanceRecord(run_id="r2"))
        logger.log(ProvenanceRecord(run_id="r3"))
        records = logger.read_all()
        assert len(records) == 3
        assert [r.run_id for r in records] == ["r1", "r2", "r3"]

    def test_read_all_empty_when_no_file(self, tmp_path):
        logger = ProvenanceLogger(path=str(tmp_path / "nope.jsonl"))
        assert logger.read_all() == []

    def test_read_run_filters_by_id(self, tmp_path):
        path = tmp_path / "prov.jsonl"
        logger = ProvenanceLogger(path=str(path))
        logger.log(ProvenanceRecord(run_id="r1", objective="first"))
        logger.log(ProvenanceRecord(run_id="r2", objective="second"))
        logger.log(ProvenanceRecord(run_id="r1", objective="third"))
        r1 = logger.read_run("r1")
        assert len(r1) == 2
        assert all(r.run_id == "r1" for r in r1)
        assert r1[0].objective == "first"
        assert r1[1].objective == "third"

    def test_corrupt_line_skipped(self, tmp_path):
        path = tmp_path / "prov.jsonl"
        path.write_text(
            json.dumps({"run_id": "r1"}) + "\n"
            + "THIS IS NOT JSON\n"
            + json.dumps({"run_id": "r2"}) + "\n",
            encoding="utf-8",
        )
        logger = ProvenanceLogger(path=str(path))
        records = logger.read_all()
        assert len(records) == 2
        assert {r.run_id for r in records} == {"r1", "r2"}

    def test_creates_parent_dir(self, tmp_path):
        path = tmp_path / "nested" / "deep" / "prov.jsonl"
        logger = ProvenanceLogger(path=str(path))
        logger.log(ProvenanceRecord(run_id="r1"))
        assert path.exists()

    def test_default_path_uses_cache_dir(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HUGINN_CACHE_DIR", str(tmp_path))
        logger = ProvenanceLogger()
        assert logger.path == tmp_path / "provenance.jsonl"

    def test_default_path_fallback(self, monkeypatch, tmp_path):
        monkeypatch.delenv("HUGINN_CACHE_DIR", raising=False)
        logger = ProvenanceLogger()
        assert logger.path == Path(".huginn") / "provenance.jsonl"


# ── export_crate ────────────────────────────────────────────────────────────


class TestExportCrate:
    def test_crate_has_context_and_graph(self):
        r = ProvenanceRecord(run_id="r1", objective="test run")
        crate = export_crate(r)
        assert "@context" in crate
        assert "ro/crate" in crate["@context"]
        assert "@graph" in crate
        assert isinstance(crate["@graph"], list)

    def test_crate_root_entity(self):
        r = ProvenanceRecord(
            run_id="r1",
            objective="optimize bandgap",
            timestamps={"start": "2026-07-01T10:00:00Z", "end": "2026-07-01T11:00:00Z"},
        )
        crate = export_crate(r)
        root = crate["@graph"][0]
        assert root["@id"] == "run:r1"
        assert "CreateAction" in root["@type"]
        assert root["name"] == "optimize bandgap"
        assert root["startTime"] == "2026-07-01T10:00:00Z"

    def test_crate_tool_entities(self):
        r = ProvenanceRecord(run_id="r1")
        snap1 = capture("vasp_tool", {"encut": 520})
        snap2 = capture("gp_tool", {"X": [[1.0]]})
        r.add_snapshot(snap1)
        r.add_snapshot(snap2)
        crate = export_crate(r)
        tool_entities = [e for e in crate["@graph"] if e.get("@type") == "SoftwareApplication"]
        names = {e["name"] for e in tool_entities}
        assert "vasp_tool" in names
        assert "gp_tool" in names

    def test_crate_input_file_entities(self, tmp_path):
        f = tmp_path / "POSCAR"
        f.write_text("structure", encoding="utf-8")
        inputs = capture_run_inputs(files=[str(f)])
        r = ProvenanceRecord(run_id="r1", inputs=inputs)
        crate = export_crate(r)
        file_entities = [e for e in crate["@graph"] if e.get("@type") == "File"]
        assert any(e["name"] == "POSCAR" for e in file_entities)

    def test_crate_output_entities(self):
        r = ProvenanceRecord(
            run_id="r1",
            outputs={"bandgap": 1.23, "formation_energy": -0.5},
        )
        crate = export_crate(r)
        out_entities = [e for e in crate["@graph"] if e.get("@type") == "PropertyValue"]
        names = {e["name"] for e in out_entities}
        assert "bandgap" in names
        assert "formation_energy" in names

    def test_crate_doi_entities(self):
        r = ProvenanceRecord(
            run_id="r1",
            dois=["10.1234/abc", "10.5678/def"],
        )
        crate = export_crate(r)
        doi_entities = [e for e in crate["@graph"] if e.get("@type") == "ScholarlyArticle"]
        assert len(doi_entities) == 2
        ids = {e["@id"] for e in doi_entities}
        assert "10.1234/abc" in ids

    def test_crate_empty_record(self):
        r = ProvenanceRecord(run_id="empty")
        crate = export_crate(r)
        # 只有 root entity, 没别的
        assert len(crate["@graph"]) == 1
        assert crate["@graph"][0]["@id"] == "run:empty"


# ── 集成: snapshot → record → logger → crate 全链路 ─────────────────────────


class TestEndToEnd:
    def test_full_chain(self, tmp_path):
        path = tmp_path / "prov.jsonl"
        logger = ProvenanceLogger(path=str(path))

        record = ProvenanceRecord(
            run_id="run_001",
            objective="LJ 团簇能量优化",
            inputs=capture_run_inputs(params={"epsilon": 1.0, "sigma": 1.0}),
            timestamps={"start": "2026-07-01T00:00:00Z", "end": "2026-07-01T01:00:00Z"},
            dois=["10.1234/lj13"],
            tags=["md", "cluster"],
        )
        # 模拟两次 tool call
        record.add_snapshot(capture("lammps_tool", {"pair_style": "lj/cut"}))
        record.add_snapshot(capture("gp_tool", {"length_scale": 1.0}))

        record.outputs = {"energy": -3.0, "sigma": 0.1}
        logger.log(record)

        # 读回来
        loaded = logger.read_run("run_001")
        assert len(loaded) == 1
        r = loaded[0]
        assert r.objective == "LJ 团簇能量优化"
        assert len(r.tool_chain) == 2
        assert r.tool_chain[0]["tool_name"] == "lammps_tool"
        assert r.tool_chain[1]["tool_name"] == "gp_tool"
        assert r.outputs["energy"] == -3.0

        # 导 crate
        crate = export_crate(r)
        root = crate["@graph"][0]
        assert root["name"] == "LJ 团簇能量优化"
        tools = [e for e in crate["@graph"] if e.get("@type") == "SoftwareApplication"]
        assert len(tools) == 2


class TestExportCrateUniqueToolIds:
    """同名工具多次调用时, root instrument 引用和 tool entity 的 @id 必须一致."""

    def test_duplicate_tool_ids_consistent(self):
        record = ProvenanceRecord(run_id="run_dup", objective="重复工具测试")
        record.add_snapshot(capture("vasp_tool", {"encut": 520}))
        record.add_snapshot(capture("vasp_tool", {"encut": 600}))
        record.add_snapshot(capture("qe_tool", {"ecutwfc": 50}))

        crate = export_crate(record)
        root = crate["@graph"][0]
        instrument_ids = [ref["@id"] for ref in root["instrument"]]
        tool_entities = [e for e in crate["@graph"] if e.get("@type") == "SoftwareApplication"]
        tool_entity_ids = [e["@id"] for e in tool_entities]

        # 三个 tool entity, @id 全唯一
        assert len(tool_entity_ids) == 3
        assert len(set(tool_entity_ids)) == 3
        # root instrument 引用必须和 tool entity @id 完全对齐 (顺序也一致)
        assert instrument_ids == tool_entity_ids
        # 第二个 vasp 调用应该有 _1 后缀, 不能两条都指向 tool:vasp_tool
        assert instrument_ids[0] == "tool:vasp_tool"
        assert instrument_ids[1] == "tool:vasp_tool_1"
        assert instrument_ids[2] == "tool:qe_tool"


# ═══════════════════════════════════════════════════════════════════════════
# 2. 捕获层 — HuginnTool.call 自动 provenance 捕获
# ═══════════════════════════════════════════════════════════════════════════

# A minimal tool that opts into the new _execute override point, so it inherits
# the base-class call() wrapper (and thus automatic provenance capture).
class _EchoTool(HuginnTool):
    name = "echo_tool"
    version = "2.3"

    async def _execute(self, args, context):  # noqa: ANN001 - test stub
        return ToolResult(data={"echoed": args.get("x", 0)}, success=True)


# A legacy-style tool that overrides call() directly, shadowing the wrapper.
class _LegacyTool(HuginnTool):
    name = "legacy_tool"

    async def call(self, args, context):  # noqa: ANN001 - test stub
        # does its own thing, including no automatic snapshot
        return ToolResult(data={"legacy": True}, success=True)


@pytest.fixture(autouse=True)
def _reset_collector():
    """The provenance collector is a context var that would otherwise leak
    between tests. Reset it before and after each test for isolation."""
    set_provenance_collector(None)
    yield
    set_provenance_collector(None)


def _run(coro):
    """asyncio.run copies the caller's context into the task, so a collector
    set before run() is visible inside the awaited call()."""
    return asyncio.run(coro)


# ── ProvenanceSnapshot construction ────────────────────────────────────────


class TestProvenanceSnapshot:
    def test_creation_and_fields(self):
        snap = ProvenanceSnapshot(
            timestamp="2026-07-06T00:00:00+00:00",
            tool_name="t",
            tool_version="1.0",
            input_params={"a": 1},
            output_hash="abc123def456",
        )
        assert snap.tool_name == "t"
        assert snap.tool_version == "1.0"
        assert snap.input_params == {"a": 1}
        assert snap.output_hash == "abc123def456"

    def test_to_dict_roundtrip(self):
        snap = ProvenanceSnapshot(
            timestamp="ts",
            tool_name="t",
            tool_version="1.0",
            input_params={"encut": 520},
            output_hash="deadbeef",
        )
        d = snap.to_dict()
        assert d["tool_name"] == "t"
        assert d["input_params"] == {"encut": 520}
        assert d["output_hash"] == "deadbeef"


# ── collector helpers ───────────────────────────────────────────────────────


class TestCollectorHelpers:
    def test_default_is_none(self):
        assert get_provenance_collector() is None

    def test_set_and_get_roundtrip(self):
        col: list = []
        set_provenance_collector(col)
        assert get_provenance_collector() is col
        set_provenance_collector(None)
        assert get_provenance_collector() is None


# ── HuginnTool.call provenance capture ─────────────────────────────────────


class TestCallCapture:
    def test_creates_snapshot_when_collector_set(self):
        collector: list = []
        set_provenance_collector(collector)

        tool = _EchoTool()
        result = _run(tool.call({"x": 5}, None))

        assert result.success
        assert len(collector) == 1
        snap = collector[0]
        assert isinstance(snap, ProvenanceSnapshot)
        assert snap.tool_name == "echo_tool"
        assert snap.tool_version == "2.3"

    def test_no_snapshot_when_collector_none(self):
        set_provenance_collector(None)
        tool = _EchoTool()
        # _capture_provenance must short-circuit and return None
        assert tool._capture_provenance({"x": 1}, ToolResult(data={}, success=True)) is None
        assert get_provenance_collector() is None
        # and a real call must not raise just because there's nowhere to store
        result = _run(tool.call({"x": 1}, None))
        assert result.success

    def test_snapshot_captures_input_params_and_output_hash(self):
        collector: list = []
        set_provenance_collector(collector)
        tool = _EchoTool()
        _run(tool.call({"x": 7, "y": 3}, None))

        snap = collector[0]
        assert snap.input_params == {"x": 7, "y": 3}
        assert snap.output_hash
        assert len(snap.output_hash) == 16
        assert snap.timestamp  # non-empty

    def test_same_output_produces_stable_hash(self):
        collector_a: list = []
        collector_b: list = []
        tool = _EchoTool()

        set_provenance_collector(collector_a)
        _run(tool.call({"x": 1}, None))
        set_provenance_collector(collector_b)
        _run(tool.call({"x": 1}, None))

        assert collector_a[0].output_hash == collector_b[0].output_hash

    def test_legacy_tool_overriding_call_is_not_double_captured(self):
        # _LegacyTool overrides call() directly, shadowing the wrapper.
        # It must still work, and the wrapper must not inject a snapshot.
        collector: list = []
        set_provenance_collector(collector)
        tool = _LegacyTool()
        result = _run(tool.call({"x": 1}, None))

        assert result.success
        assert result.data == {"legacy": True}
        assert collector == []  # wrapper was shadowed -> no auto snapshot

    def test_no_snapshot_when_provenance_flag_disabled(self, monkeypatch):
        from huginn.feature_flags import FeatureFlags

        monkeypatch.setattr(
            FeatureFlags,
            "is_enabled",
            lambda self, feature: feature != "provenance",
        )
        collector: list = []
        set_provenance_collector(collector)
        tool = _EchoTool()
        _run(tool.call({"x": 1}, None))

        assert collector == []  # flag off -> capture skipped


# ═══════════════════════════════════════════════════════════════════════════
# 3. 接线层 — autoloop engine.provenance 接线
# ═══════════════════════════════════════════════════════════════════════════

# ── ProvenanceRecord 构造 + add_snapshot ────────────────────────────────────


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


# ── ProvenanceLogger 写 + read_run ──────────────────────────────────────────


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


# ── tool_chain 序列化 ────────────────────────────────────────────────────────


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


# ── 空记录处理 ──────────────────────────────────────────────────────────────


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


# ── 多次 snapshot 保序 ──────────────────────────────────────────────────────


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


# ── engine 接线: run() 落盘 provenance, _execute 记 tool call ───────────────


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
    # ponytail: KB 冷启动跑 ONNX embedding > 120s, KG 写 ~/.huginn 污染 home
    monkeypatch.setattr("huginn.autoloop.engine.AutoloopEngine._get_kb", lambda self: None)
    monkeypatch.setattr("huginn.autoloop.conjecture.get_kg", lambda *a, **kw: None)
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

    @pytest.mark.skipif(_skip_ci_run_cognitive, reason="run_cognitive hangs on CI asyncio.run")
    def test_run_persists_provenance_with_tool_chain(self, engine: AutoloopEngine):
        _stub_phases_keep_execute(engine)
        # v10: run_cognitive 1-action-per-iter, max_iter=3 到 execute (hyp→plan→exec)
        result = asyncio.run(
            engine.run_cognitive(objective="o", max_iterations=3, progressive_budget=False)
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