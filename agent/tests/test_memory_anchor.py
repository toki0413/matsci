"""落地锚的单测: 结构化解析 / 覆盖率 / 召回先验 / stats 可观测."""

from __future__ import annotations

from pathlib import Path

from huginn.memory.anchor import Anchor, anchored_ratio, has_anchor, parse_anchor
from huginn.memory.longterm import LongTermMemory
from huginn.memory.manager import MemoryManager


def test_parse_anchor_source_prefixes():
    assert parse_anchor({"source": "session:s1/tool:bash"}) == Anchor("session", "session:s1/tool:bash")
    assert parse_anchor({"source": "run:r45"}) == Anchor("run", "run:r45")
    assert parse_anchor({"source": "tool:grep"}) == Anchor("tool", "tool:grep")
    assert parse_anchor({"source": "distiller:kid_1"}) == Anchor("distiller", "distiller:kid_1")
    assert parse_anchor({"source": "exec:abc123"}) == Anchor("exec", "exec:abc123")
    assert parse_anchor({"source": "measure:1.23eV"}) == Anchor("measure", "measure:1.23eV")
    assert parse_anchor({"source": "cite:10.1000/x"}).kind == "cite"
    assert parse_anchor({"source": "doi:10.1000/x"}).kind == "cite"


def test_parse_anchor_run_id_and_absent():
    # 结构化 run_id 兜底为 run 锚
    assert parse_anchor({"source": "", "run_id": "r7"}) == Anchor("run", "r7")
    # 空前缀 (如 "run:") 不算锚
    assert parse_anchor({"source": "run:"}) is None
    # 自由文本 source / 空 source → 无锚 (宁缺勿猜)
    assert parse_anchor({"source": "S7_self_modify"}) is None
    assert parse_anchor({"source": "user said so"}) is None
    assert parse_anchor({"source": ""}) is None
    assert parse_anchor({}) is None


def test_has_anchor_and_ratio():
    rows = [
        {"source": "run:r1"},
        {"source": "session:s/tool:t"},
        {"source": ""},
        {},
    ]
    assert has_anchor(rows[0]) and has_anchor(rows[1])
    assert not has_anchor(rows[2]) and not has_anchor(rows[3])
    assert anchored_ratio(rows) == 0.5
    # 空集 → 0.0 (无信号, 不臆造)
    assert anchored_ratio([]) == 0.0


def test_grounding_prior_is_stable():
    ranked = [
        {"id": "a", "source": ""},
        {"id": "b", "source": "run:r1"},
        {"id": "c", "source": ""},
        {"id": "d", "source": "session:s/tool:t"},
    ]
    out = LongTermMemory._grounding_prior(ranked)
    # 有锚前移, 同锚态内部相对次序不变 (stable)
    assert [r["id"] for r in out] == ["b", "d", "a", "c"]
    # 全窗无锚 → 严格 no-op
    plain = [{"id": "x", "source": ""}, {"id": "y"}]
    assert LongTermMemory._grounding_prior(plain) == plain


def test_anchored_ratio_from_db(tmp_path: Path):
    mem = LongTermMemory(db_path=tmp_path / "memory.db", enable_semantic=False)
    mem.store(content="grounded fact", source="run:r1")
    mem.store(content="loose fact", source="")
    assert mem.anchored_ratio() == 0.5
    mgr = MemoryManager(longterm=mem)
    assert mgr.stats()["anchored_ratio"] == 0.5


def test_retrieve_prefers_anchored(tmp_path: Path):
    mem = LongTermMemory(db_path=tmp_path / "memory.db", enable_semantic=False)
    loose = mem.store(content="alpha beta", source="", tier="long")
    grounded = mem.store(content="alpha gamma", source="run:r9", tier="long")
    rows = mem.retrieve(query="alpha", top_k=2)
    assert [r["id"] for r in rows] == [grounded, loose]
