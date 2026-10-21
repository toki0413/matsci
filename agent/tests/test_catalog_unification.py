"""P1 统一目录契约门 —— 只读检索跨四类注册表, 不新增注册表.

红蓝双轨:
  蓝队: 一次检索跨 tool/capability/skill/workflow; kind 过滤; 去重; 上限;
        catalog_search 返回调用提示; workflow_tool 可按名渲染已注册工作流.
  红队: 只读(不启停/不注册); 未知 kind 空结果; 未知工作流明确报错而非静默.
"""

from __future__ import annotations

import asyncio

import pytest

from huginn.catalog.models import CatalogEntry
from huginn.catalog.search import collect_entries, search_entries, summarize
from huginn.tools.catalog_tool import CatalogSearchInput, CatalogSearchTool


def _entry(kind: str, name: str, desc: str = "", origin: str = "builtin") -> CatalogEntry:
    return CatalogEntry(
        id=f"{kind}:{name}", kind=kind, name=name, origin=origin,
        registered_names=[name], meta={"description": desc, "category": "test"},
    )


def _patch_collectors(monkeypatch):
    """注入确定性四类目录, 隔离真实注册表."""
    from huginn.catalog import search as search_mod

    fake = {
        "tool": [_entry("tool", "hodge_circulation", "高阶拓扑环流诊断")],
        "capability": [_entry("capability", "research_pipeline", "文献研究复合能力")],
        "skill": [_entry("skill", "standard_dft", "标准 DFT 工作流技能")],
        "workflow": [_entry("workflow", "aimd_template", "从头算分子动力学模板")],
    }
    monkeypatch.setattr(search_mod, "_kind_collectors", lambda handles: fake)


def test_blue_search_spans_all_four_kinds(monkeypatch):
    _patch_collectors(monkeypatch)
    entries = search_entries(query="", limit=100)
    assert summarize(entries) == {"tool": 1, "capability": 1, "skill": 1, "workflow": 1}


def test_blue_search_matches_description_and_kind_filter(monkeypatch):
    _patch_collectors(monkeypatch)
    assert [e.name for e in search_entries("拓扑")] == ["hodge_circulation"]
    assert [e.name for e in search_entries(kind="skill,workflow")] == [
        "standard_dft", "aimd_template"]
    assert search_entries(kind="capability")[0].name == "research_pipeline"


def test_red_unknown_kind_is_empty_not_guessed(monkeypatch):
    _patch_collectors(monkeypatch)
    assert search_entries(kind="nonexistent") == []


def test_blue_limit_and_dedupe(monkeypatch):
    from huginn.catalog import search as search_mod

    dup = [_entry("tool", "a"), _entry("tool", "a"), _entry("tool", "b")]
    monkeypatch.setattr(search_mod, "_kind_collectors", lambda handles: {"tool": dup})
    assert [e.name for e in collect_entries()] == ["a", "b"]
    assert len(search_entries(limit=1)) == 1


def test_blue_catalog_tool_returns_items_with_call_hint(monkeypatch):
    _patch_collectors(monkeypatch)
    tool = CatalogSearchTool()
    res = asyncio.run(tool.call(CatalogSearchInput(action="list", limit=10)))
    assert res.success
    assert res.data["n_items"] == 4
    hints = {i["kind"]: i["call_hint"] for i in res.data["items"]}
    assert "capability_tool" in hints["capability"]
    assert "workflow_tool" in hints["workflow"]
    assert res.data["counts"]["skill"] == 1


def test_red_catalog_tool_is_read_only():
    tool = CatalogSearchTool()
    assert tool.read_only is True
    assert tool.destructive is False


def test_blue_workflow_render_returns_executable_stages(monkeypatch):
    """命名工作流可按名渲染成可执行 stage 对象 (修通 workflow 孤岛)."""
    from huginn.workflows.registry import WorkflowPackage, WorkflowRegistry

    pkg = WorkflowPackage(
        name="wf_render_probe", kind="stages",
        stages=[{"id": "s1", "name": "probe", "tool": "echo_tool", "tool_input": {}}],
    )
    monkeypatch.setitem(WorkflowRegistry._packages, "wf_render_probe", pkg)
    stages = WorkflowRegistry.render("wf_render_probe")
    assert [s.tool for s in stages] == ["echo_tool"]
    # 序列化视图与执行视图同源
    assert WorkflowRegistry.render_stages("wf_render_probe")[0]["tool"] == "echo_tool"


def test_red_unknown_workflow_raises():
    from huginn.workflows.registry import WorkflowRegistry

    with pytest.raises(ValueError):
        WorkflowRegistry.render("no_such_workflow_xyz")
