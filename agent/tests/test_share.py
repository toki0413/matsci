"""ShareManager 统一分享总线 — 五类资产清单/导出/再导入."""
from __future__ import annotations

import pytest

from huginn.share import ShareManager


def test_list_counts_all_kinds():
    counts = ShareManager.list()
    for k in ("capability", "workflow", "persona", "skill", "lean"):
        assert k in counts
        assert counts[k]["available"] > 0
    assert counts["total"] == sum(
        counts[k]["available"] for k in ("capability", "workflow", "persona", "skill", "lean")
    )


def test_export_structure():
    b = ShareManager.export_bundle(kinds=("workflow", "lean"))
    assert b["spec"] == "huginn-share 1"
    assert b["kinds_included"] == ["workflow", "lean"]
    assert b["count"] == len(b["items"])
    kinds = {i["kind"] for i in b["items"]}
    assert {"workflow", "lean"} <= kinds


def test_import_bundle_restores_workflow(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    b = ShareManager.export_bundle(kinds=("workflow", "lean"))
    res = ShareManager.import_bundle(b)
    assert res["ok"]
    assert res["errors"] == []
    imported_names = {i.split(":", 1)[1] for i in res["imported"]}
    assert "standard_dft_workflow" in imported_names


def test_import_rejects_foreign_spec():
    with pytest.raises(ValueError):
        ShareManager.import_bundle({"spec": "other", "items": []})


def test_manifest_counts():
    m = ShareManager.manifest(per_kind_limit=4)
    assert m["total"] >= 5
    assert set(m["kinds"]) == {"capability", "workflow", "persona", "skill", "lean"}
    assert len(m["kinds"]["workflow"]) >= 1