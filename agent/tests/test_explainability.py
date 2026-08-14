"""Explainer 统一解释入口测试 (第 3 项改进).

验证:
- explain 把 audit 事件 + provenance 产出整合成时间线/依赖边/关键发现.
- 无 provider / 无匹配时返回空结构.
- 时间戳归一化 (ISO 字符串 / float).
- 与真实 AuditLogger + ProvenanceRegistry 的集成.
"""

from __future__ import annotations

from pathlib import Path

from huginn.explainability import Explainer, ExplainStep


class FakeAudit:
    def __init__(self, events):
        self._events = events

    def query(self, *, action=None, actor=None, tool=None, limit=1000):
        out = []
        for e in self._events:
            if action and action not in e.get("action", ""):
                continue
            if actor and e.get("actor") != actor:
                continue
            if tool and (e.get("details") or {}).get("tool") != tool:
                continue
            out.append(e)
        return out[:limit]


class FakeProvenance:
    def __init__(self, entries, search_hits=None):
        self._entries = entries
        self._search_hits = search_hits

    def search(self, q):
        return self._search_hits if self._search_hits is not None else list(self._entries)

    def recent(self, n=10):
        return list(self._entries)[:n]


def test_explain_integrates_audit_and_provenance() -> None:
    audit = FakeAudit([
        {"timestamp": "2026-08-14T10:00:00Z", "actor": "agent", "action": "vasp_relaxation",
         "details": {"tool": "vasp_tool"}},
        {"timestamp": "2026-08-14T10:05:00Z", "actor": "agent", "action": "analyze_band",
         "details": {"tool": "visualize_tool"}},
    ])
    prov = FakeProvenance([
        {"file_path": "OUTCAR", "produced_by": "vasp_tool", "produced_at": 100.0,
         "input_files": ["POSCAR"], "key_properties": {"energy": -10.5}},
    ])
    exp = Explainer(audit=audit, provenance=prov).explain("vasp")
    # 时间线含审计事件 + 产物
    kinds = {s.kind for s in exp.steps}
    assert "audit" in kinds and "artifact" in kinds
    # 依赖边: POSCAR -> OUTCAR
    assert exp.edges == [{"from_file": "POSCAR", "to_file": "OUTCAR", "produced_by": "vasp_tool"}]
    # 关键发现聚合
    assert exp.key_findings == {"energy": -10.5}


def test_explain_timeline_sorted() -> None:
    audit = FakeAudit([
        {"timestamp": "2026-08-14T10:05:00Z", "actor": "agent", "action": "b"},
        {"timestamp": "2026-08-14T10:00:00Z", "actor": "agent", "action": "a"},
    ])
    exp = Explainer(audit=audit, provenance=None).explain()
    acts = [s.action for s in exp.timeline()]
    assert acts == ["a", "b"], "时间线应按时间升序"


def test_explain_no_provider_returns_empty() -> None:
    exp = Explainer(audit=None, provenance=None).explain("anything")
    assert exp.steps == []
    assert exp.artifacts == []
    assert exp.edges == []
    assert exp.key_findings == {}


def test_explain_filters_by_actor() -> None:
    audit = FakeAudit([
        {"timestamp": 1.0, "actor": "user", "action": "mkdir"},
        {"timestamp": 2.0, "actor": "agent", "action": "mkdir"},
    ])
    exp = Explainer(audit=audit, provenance=None).explain(actor="agent")
    assert [s.actor for s in exp.steps] == ["agent"]


def test_timestamp_normalization() -> None:
    from huginn.explainability import _to_ts

    assert isinstance(_to_ts(123.0), float)
    assert _to_ts(123.0) == 123.0
    assert _to_ts("garbage") == 0.0
    iso = _to_ts("1970-01-01T00:00:01Z")
    assert iso > 0.0


def test_recent_uses_provenance_when_no_goal() -> None:
    prov = FakeProvenance([
        {"file_path": "OUTCAR", "produced_by": "vasp_tool", "produced_at": 1.0,
         "input_files": [], "key_properties": {}},
    ])
    exp = Explainer(audit=None, provenance=prov).explain()
    assert exp.artifacts and exp.artifacts[0]["file_path"] == "OUTCAR"


def test_integration_with_real_audit_and_provenance(tmp_path: Path) -> None:
    from huginn.provenance.registry import ProvenanceEntry, _ProvenanceStore
    from huginn.security.audit import AuditLogger

    audit = AuditLogger(str(tmp_path / "audit.jsonl"))
    audit.log("tool_call", "agent", "vasp_relaxation", details={"tool": "vasp_tool"})

    store = _ProvenanceStore(str(tmp_path / "prov.db"))
    store.save(ProvenanceEntry(
        file_path=str(tmp_path / "OUTCAR"),
        produced_by="vasp_tool",
        produced_at=1.0,
        input_files=[str(tmp_path / "POSCAR")],
        key_properties={"energy": -10.5},
    ))

    exp = Explainer(audit=audit, provenance=store).explain("vasp")
    assert any(s.kind == "audit" for s in exp.steps)
    assert any(s.kind == "artifact" for s in exp.steps)
    assert exp.key_findings.get("energy") == -10.5