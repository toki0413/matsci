"""Tests for the refactor engine."""

from __future__ import annotations

from pathlib import Path

from huginn.coder.refactor_engine import PlannedEdit, RefactorEngine


class _FakeModel:
    """LangChain-compatible fake that returns a JSON edit plan."""

    def __init__(self, plan_json: str) -> None:
        self._plan = plan_json

    def invoke(self, messages: list[object]) -> object:
        class _Response:
            content = self._plan

        return _Response()


def test_refactor_plan_and_apply(tmp_path: Path):
    (tmp_path / "math.py").write_text(
        "def add(a, b):\n    return a + b\n", encoding="utf-8"
    )
    (tmp_path / "main.py").write_text(
        "from math import add\n\nprint(add(1, 2))\n", encoding="utf-8"
    )

    plan_json = (
        '{"edits": ['
        '{"path": "math.py", "old_string": "def add(a, b):\\n    return a + b", '
        '"new_string": "def plus(a, b):\\n    return a + b"},'
        '{"path": "main.py", "old_string": "from math import add\\n\\nprint(add(1, 2))", '
        '"new_string": "from math import plus\\n\\nprint(plus(1, 2))"}'
        "]}"
    )

    engine = RefactorEngine(root=tmp_path, model=_FakeModel(plan_json))
    plan = engine.plan("rename add to plus")

    assert len(plan) == 2
    assert all(isinstance(e, PlannedEdit) for e in plan)

    result = engine.apply(plan, dry_run=True)
    assert result["dry_run"] is True
    assert result["applied"] == 2
    assert result["errors"] == []
    assert "def plus(a, b)" in result["diff"]

    # Files should be unchanged after dry run.
    assert "def add" in (tmp_path / "math.py").read_text(encoding="utf-8")

    apply_result = engine.apply(plan, dry_run=False)
    assert apply_result["applied"] == 2
    assert "def plus" in (tmp_path / "math.py").read_text(encoding="utf-8")
    assert "from math import plus" in (tmp_path / "main.py").read_text(encoding="utf-8")

    # Rollback restores originals.
    engine.rollback(apply_result["snapshots"])
    assert "def add" in (tmp_path / "math.py").read_text(encoding="utf-8")
    assert "from math import add" in (tmp_path / "main.py").read_text(encoding="utf-8")


def test_refactor_apply_reports_missing_old_string(tmp_path: Path):
    (tmp_path / "mod.py").write_text("x = 1\n", encoding="utf-8")

    plan_json = (
        '{"edits": ['
        '{"path": "mod.py", "old_string": "not present", "new_string": "y = 2"}'
        "]}"
    )

    engine = RefactorEngine(root=tmp_path, model=_FakeModel(plan_json))
    plan = engine.plan("do something")
    result = engine.apply(plan, dry_run=False)

    assert result["applied"] == 0
    assert len(result["errors"]) == 1
    assert "not found" in result["errors"][0]
