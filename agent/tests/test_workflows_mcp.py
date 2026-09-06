"""workflows-mcp 码头 — 三只只读工具: manifest / view / render."""
from __future__ import annotations

import asyncio
import json

from huginn.workflows.mcp_export import WorkflowMCPBackend, list_json


def _run(coro):
    return asyncio.run(coro)


def test_mcp_tools_names():
    b = WorkflowMCPBackend()
    names = {t.name for t in b.mcp_tools()}
    assert {"workflow_manifest", "view_workflow", "render_workflow"} == names


def test_manifest_tool():
    b = WorkflowMCPBackend()
    r = _run(b.call("workflow_manifest", {}))
    assert r["success"]
    assert len(r["data"]) >= 10
    assert r["data"][0]["kind"] in ("template", "script", "stages")


def test_view_workflow_exports_package():
    b = WorkflowMCPBackend()
    r = _run(b.call("view_workflow", {"name": "standard_dft_workflow"}))
    assert r["success"]
    assert r["data"]["spec"] == "huginn/workflow 1"
    assert r["data"]["kind"] == "template"


def test_render_workflow_stages():
    b = WorkflowMCPBackend()
    r = _run(b.call("render_workflow", {
        "name": "aimd_workflow", "params": {"structure_path": "POSCAR"}}))
    assert r["success"]
    assert r["data"]["n_stages"] >= 1
    assert r["data"]["stages"][0]["tool"]  # stage 骨架含 tool


def test_unknown_workflow_returns_code():
    b = WorkflowMCPBackend()
    r = _run(b.call("view_workflow", {"name": "does-not-exist"}))
    assert not r["success"]
    assert r["code"] == "not_found"


def test_list_json_is_stdout_ready():
    data = json.loads(list_json())
    assert isinstance(data, list)
    assert data[0]["name"]