"""WorkflowRegistry 集装箱层 — 注册/导出/再导入/模板渲染."""
from __future__ import annotations

import pytest

from huginn.workflows.registry import WorkflowRegistry, WorkflowPackage


@pytest.fixture(autouse=True)
def _reset():
    WorkflowRegistry._packages = {}
    WorkflowRegistry._template_factories = {}
    yield
    WorkflowRegistry._packages = {}
    WorkflowRegistry._template_factories = {}


def _make_script() -> dict:
    return {
        "id": "wf-test-1", "objective": "sweep", "max_concurrent": 4,
        "subtasks": [
            {"id": "s1", "tool": "sympy", "args": {}},
            {"id": "s2", "tool": "bash", "args": {}},
        ],
    }


def test_register_script_and_manifest():
    WorkflowRegistry.register_script("sweep", _make_script(), "扫参", "materials")
    m = WorkflowRegistry.manifest()
    assert any(x["name"] == "sweep" and x["kind"] == "script"
               and x["n_subtasks"] == 2 for x in m)


def test_export_import_roundtrip():
    WorkflowRegistry.register_script("sweep", _make_script())
    data = WorkflowRegistry.export("sweep")
    assert data["spec"] == "huginn/workflow 1"
    # 模拟外地实例: 清空后 import 还原
    WorkflowRegistry._packages = {}
    restored = WorkflowRegistry.import_dict(data)
    assert restored == "sweep"
    assert WorkflowRegistry.get("sweep").kind == "script"
    assert WorkflowRegistry.get("sweep").script["subtasks"][0]["id"] == "s1"


def test_import_rejects_foreign_spec():
    with pytest.raises(ValueError):
        WorkflowRegistry.import_dict({"spec": "other", "name": "x"})


def test_unknown_kind_rejected():
    with pytest.raises(ValueError):
        WorkflowRegistry.register_package(WorkflowPackage(name="x", kind="nope"))


def test_builtin_templates_registered():
    names = WorkflowRegistry.register_builtin_templates()
    assert len(names) >= 10  # templates.py 里 *_workflow 系列
    by_name = {x["name"]: x for x in WorkflowRegistry.manifest()}
    assert "standard_dft_workflow" in by_name
    # 参数 schema 从工厂签名提取
    assert "structure_path" in by_name["standard_dft_workflow"]["params"]


def test_export_excludes_runtime_state():
    """注册一个 stages 工作流, 确保骨架导出不含运行时状态字段."""
    from huginn.workflows.stages import ComputationalStage, RetryPolicy, ValidationRule

    stage = ComputationalStage(
        id="relax", name="Relax", tool="vasp_tool",
        tool_input={"action": "relax"},
        validation=ValidationRule(check="convergence"),
        retry_policy=RetryPolicy(max_retries=2),
    )
    WorkflowRegistry.register_stages("dft-demo", [stage])
    data = WorkflowRegistry.export("dft-demo")
    s0 = data["stages"][0]
    assert s0["id"] == "relax"
    assert s0["tool"] == "vasp_tool"
    for bad in ("status", "result", "attempts", "started_at", "completed_at"):
        assert bad not in s0
    assert s0["validation"]["check"] == "convergence"