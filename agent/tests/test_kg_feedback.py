"""kg_feedback: 验证结果 → KG 回写的落库语义.

brooks-sweep R1 修复 (抽取 _write_result_constraints 降嵌套) 的行为护栏:
只写 error / (未通过且 warning) 的 check, 重复节点不重建, 关联节点不存在不凭空造.
"""

from __future__ import annotations

import pytest

from huginn.kg.entities import EntityType, Relation
from huginn.validation import kg_feedback


class _FakeKG:
    """最小 KG 替身: 只记录 add_node / add_edge / save 的调用."""

    def __init__(self, existing: set[str] | None = None) -> None:
        self.nodes: dict[str, dict] = {}
        self.edges: list[tuple[str, str, object]] = []
        self.saved = 0
        for node_id in existing or set():
            self.nodes[node_id] = {}

    def has_node(self, node_id: str) -> bool:
        return node_id in self.nodes

    def add_node(self, node_id: str, **kwargs) -> None:
        self.nodes[node_id] = kwargs

    def add_edge(self, src: str, dst: str, relation=None, properties=None) -> None:
        self.edges.append((src, dst, relation))

    def save(self) -> None:
        self.saved += 1


@pytest.fixture
def fake_kg(monkeypatch):
    kg = _FakeKG(existing={f"{EntityType.COMPOUND}:Si", f"{EntityType.TOOL}:vasp_tool"})
    monkeypatch.setattr(kg_feedback, "_get_project_kg", lambda: kg)
    return kg


def test_writes_only_error_and_failed_warning_checks(fake_kg):
    results = [
        {
            "name": "physics",
            "passed": False,
            "checks": [
                {"severity": "error", "message": "energy must be negative"},
                {"severity": "warning", "message": "lattice constant off by 3%"},
                {"severity": "info", "message": "just a note"},
            ],
        },
        {
            "name": "passing",
            "passed": True,
            "checks": [{"severity": "warning", "message": "ignored when passed"}],
        },
    ]

    created = kg_feedback.write_validation_to_kg(results, material="Si", tool_name="vasp_tool")

    assert created == 2, "只有 error 与未通过时的 warning 应落库"
    assert fake_kg.saved == 1, "有新增才 save 一次"
    assert len(fake_kg.edges) == 4, "两个约束各连 material 与 tool 各一条边"
    assert {(src, dst, rel) for src, dst, rel in fake_kg.edges} == {
        (f"{EntityType.FACT}:physics:energy_must_be_negative", f"{EntityType.COMPOUND}:Si", Relation.VALIDATES),
        (f"{EntityType.FACT}:physics:energy_must_be_negative", f"{EntityType.TOOL}:vasp_tool", Relation.RELATED_TO),
        (f"{EntityType.FACT}:physics:lattice_constant_off_by_3%", f"{EntityType.COMPOUND}:Si", Relation.VALIDATES),
        (f"{EntityType.FACT}:physics:lattice_constant_off_by_3%", f"{EntityType.TOOL}:vasp_tool", Relation.RELATED_TO),
    }


def test_existing_constraint_node_is_not_duplicated(fake_kg):
    # 预置已存在的约束节点: 重复回写不应重建, 也不 save
    node_id = f"{EntityType.FACT}:physics:energy_must_be_negative"
    fake_kg.nodes[node_id] = {}

    created = kg_feedback.write_validation_to_kg(
        [{"name": "physics", "passed": False,
          "checks": [{"severity": "error", "message": "energy must be negative"}]}],
        material="Si",
    )

    assert created == 0
    assert fake_kg.saved == 0


def test_absent_material_and_tool_skip_edges(fake_kg):
    fake_kg.nodes.pop(f"{EntityType.COMPOUND}:Si")
    fake_kg.nodes.pop(f"{EntityType.TOOL}:vasp_tool")

    created = kg_feedback.write_validation_to_kg(
        [{"name": "physics", "passed": False,
          "checks": [{"severity": "error", "message": "energy must be negative"}]}],
        material="Si",
        tool_name="vasp_tool",
    )

    assert created == 1
    assert fake_kg.edges == [], "关联节点不存在时不应凭空造边"


def test_no_project_kg_returns_zero(monkeypatch):
    monkeypatch.setattr(kg_feedback, "_get_project_kg", lambda: None)

    assert kg_feedback.write_validation_to_kg(
        [{"name": "x", "passed": False, "checks": [{"severity": "error", "message": "m"}]}]
    ) == 0
