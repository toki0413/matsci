"""M4 FAIR provenance 测试 — ProvenanceRecord + ProvenanceLogger + export_crate.

覆盖: record 构造 / snapshot 追加 / JSONL 追加读写 / run 过滤 /
      file hash / ROCrate 导出结构 / 损坏行容错.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from huginn.provenance import (
    ProvenanceLogger,
    ProvenanceRecord,
    ProvenanceSnapshot,
    capture,
    capture_run_inputs,
    export_crate,
)


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

