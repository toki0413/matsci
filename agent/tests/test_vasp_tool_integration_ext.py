"""sim/vasp_tool.py 集成路径补测 — 覆盖 __init__/_find_vasp/estimate_cost/
validate_input/call(mock + 缺文件)/poll_job/wait_job/eos/_parse_vasprun_quick/
_read_incar_params/_modify_incar/_uq_hint/_structure_file_hint.

配合 test_vasp_rust_ext.py (Rust _parse_outcar 桥接), 把 sim/vasp_tool.py
覆盖率从 40% 提升到 90%+.
"""

from __future__ import annotations

import asyncio
import shutil
import time

import pytest

from huginn.tools.sim import executable_resolver as er
from huginn.tools.sim import vasp_tool as vt

# anyio 只作用于 async 测试, 同步测试不受影响
pytestmark = pytest.mark.anyio


def _tool(**kw):
    return vt.VaspTool(**kw)


def _args(**kw):
    base = {"action": "relax", "working_dir": "."}
    base.update(kw)
    return vt.VaspToolInput(**base)


# ── __init__ / _find_vasp ────────────────────────────────────────────────


def test_find_vasp_env_match(monkeypatch, tmp_path):
    exe = tmp_path / "vasp_exe"
    exe.write_text("#!/bin/sh\n")
    monkeypatch.setenv("VASP_EXECUTABLE", str(exe))
    assert _tool()._find_vasp() == str(exe)


def test_find_vasp_env_not_exists(monkeypatch):
    monkeypatch.setenv("VASP_EXECUTABLE", "/nonexistent/vasp")
    # env 路径不存在 → 落到 PATH 查找 → 返回 None
    assert _tool()._find_vasp() is None


def test_find_vasp_via_path(monkeypatch, tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    exe = bindir / "vasp"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    monkeypatch.setattr(shutil, "which", lambda name: str(exe) if name == "vasp" else None)
    assert _tool()._find_vasp() == str(exe)


def test_find_vasp_which_raises(monkeypatch):
    def boom(name):
        raise OSError("boom")

    monkeypatch.setattr(shutil, "which", boom)
    assert _tool()._find_vasp() is None


# ── estimate_cost ─────────────────────────────────────────────────────────


def test_estimate_cost_poll_wait_none():
    tool = _tool()
    assert tool.estimate_cost(_args(action="poll_job", job_id="x")) is None
    assert tool.estimate_cost(_args(action="wait_job", job_id="x")) is None


def test_estimate_cost_compute():
    tool = _tool()
    c = tool.estimate_cost(_args(walltime_hours=3))
    assert c["cpu_hours"] == 12
    assert c["walltime_hours"] == 3


# ── validate_input ────────────────────────────────────────────────────────


async def _val(tool, args, context):
    return await tool.validate_input(args, context)


def _ctx():
    return type("C", (), {"workspace": "."})()


async def test_validate_poll_wait_pass():
    tool = _tool()
    r = await _val(tool, _args(action="poll_job", job_id="x"), _ctx())
    assert r.result is True


async def test_validate_eos_missing_dir():
    tool = _tool()
    r = await _val(tool, _args(action="eos", working_dir="/no/such/dir"), _ctx())
    assert r.result is False
    assert r.error_code == 404


async def test_validate_missing_workdir(tmp_path):
    tool = _tool()
    r = await _val(tool, _args(action="relax", working_dir="/no/dir"), _ctx())
    assert r.result is False
    assert r.error_code == 404


async def test_validate_missing_poscar(tmp_path):
    d = tmp_path / "wd"
    d.mkdir()
    tool = _tool()
    r = await _val(tool, _args(action="relax", working_dir=str(d)), _ctx())
    assert r.result is False
    assert r.error_code == 404


async def test_validate_success(tmp_path):
    d = tmp_path / "wd"
    d.mkdir()
    (d / "POSCAR").write_text("POSCAR\n", encoding="utf-8")
    tool = _tool()
    r = await _val(tool, _args(action="relax", working_dir=str(d)), _ctx())
    assert r.result is True


# ── call(): 缺目录 / 缺 POSCAR / mock ────────────────────────────────────


async def test_call_workdir_not_found(tmp_path):
    tool = _tool()
    res = await tool.call(_args(action="relax", working_dir="/no/dir"), _ctx())
    assert res.success is False
    assert "Working directory not found" in res.error


async def test_call_poscar_not_found(tmp_path):
    d = tmp_path / "wd"
    d.mkdir()
    tool = _tool()
    res = await tool.call(_args(action="relax", working_dir=str(d)), _ctx())
    assert res.success is False
    assert "POSCAR not found" in res.error


async def test_call_mock_result(monkeypatch, tmp_path):
    d = tmp_path / "wd"
    d.mkdir()
    (d / "POSCAR").write_text("cell\n", encoding="utf-8")
    tool = _tool(vasp_executable=None)
    monkeypatch.setattr(er, "resolve_executable", lambda name: None)
    res = await tool.call(_args(action="relax", working_dir=str(d)), _ctx())
    assert res.success is True
    assert res.metadata.get("mock") is True
    assert "mock" in res.data["status"]
    assert res.data["uq_hint"]["tool"] == "gp_tool"


async def test_call_mock_with_incar_override(tmp_path):
    d = tmp_path / "wd"
    d.mkdir()
    (d / "POSCAR").write_text("cell\n", encoding="utf-8")
    (d / "INCAR").write_text("ENCUT = 400\n", encoding="utf-8")
    tool = _tool(vasp_executable=None)
    res = await tool.call(
        _args(action="relax", working_dir=str(d), incar_overrides={"ENCUT": 600}), _ctx()
    )
    assert res.success is True
    # INCAR 被改写
    assert "ENCUT = 600" in (d / "INCAR").read_text(encoding="utf-8")


# ── poll_job / wait_job ──────────────────────────────────────────────────


async def test_poll_unknown_job():
    tool = _tool()
    res = await tool.call(_args(action="poll_job", working_dir=".", job_id="nope"), _ctx())
    assert res.success is False
    assert "Unknown job_id" in res.error


async def test_poll_known_job(tmp_path):
    tool = _tool()
    job_id = "job1"
    vt.VaspTool._async_jobs[job_id] = {
        "status": "done",
        "started_at": time.time() - 5,
        "result": {"energy": -10.0},
        "error": None,
        "compute_action": "relax",
    }
    try:
        res = await tool.call(_args(action="poll_job", working_dir=".", job_id=job_id), _ctx())
        assert res.success is True
        assert res.data["status"] == "done"
        assert res.data["progress"] == 100
        assert res.data["partial_result"]["energy"] == -10.0
    finally:
        vt.VaspTool._async_jobs.pop(job_id, None)


async def test_wait_unknown_job():
    tool = _tool()
    res = await tool.call(_args(action="wait_job", working_dir=".", job_id="nope"), _ctx())
    assert res.success is False


async def test_wait_task_none(tmp_path):
    tool = _tool()
    job_id = "j2"
    vt.VaspTool._async_jobs[job_id] = {
        "status": "running", "started_at": time.time(), "result": None,
        "error": None, "compute_action": "relax", "task": None,
    }
    try:
        res = await tool.call(_args(action="wait_job", working_dir=".", job_id=job_id), _ctx())
        assert res.success is True
    finally:
        vt.VaspTool._async_jobs.pop(job_id, None)


async def test_wait_task_done(tmp_path):
    tool = _tool()
    job_id = "j3"
    vt.VaspTool._async_jobs[job_id] = {
        "status": "failed", "started_at": time.time(), "result": None,
        "error": "boom", "compute_action": "relax",
        "task": asyncio.sleep(0),
    }
    try:
        res = await tool.call(_args(action="wait_job", working_dir=".", job_id=job_id), _ctx())
        assert res.success is True
        assert res.data["status"] == "failed"
    finally:
        vt.VaspTool._async_jobs.pop(job_id, None)


# ── _parse_vasprun_quick ─────────────────────────────────────────────────


def _install_fake_pymatgen(monkeypatch, vasprun_class):
    """注入 fake pymatgen.io.vasp 模块, 让 `from pymatgen.io.vasp import Vasprun` 命中."""
    import sys as _sys
    import types as _types

    pymatgen = _types.ModuleType("pymatgen")
    io_mod = _types.ModuleType("pymatgen.io")
    vasp_mod = _types.ModuleType("pymatgen.io.vasp")
    vasp_mod.Vasprun = vasprun_class
    io_mod.vasp = vasp_mod
    pymatgen.io = io_mod
    monkeypatch.setitem(_sys.modules, "pymatgen", pymatgen)
    monkeypatch.setitem(_sys.modules, "pymatgen.io", io_mod)
    monkeypatch.setitem(_sys.modules, "pymatgen.io.vasp", vasp_mod)


def test_parse_vasprun_pymatgen(monkeypatch, tmp_path):
    p = tmp_path / "vasprun.xml"
    p.write_text("<vasprun><calculation></calculation></vasprun>", encoding="utf-8")

    class _VR:
        def __init__(self, path):
            pass

        efermi = 5.0
        eigenvalue_band_properties = (1.2, 3.0, 1.8)

    _install_fake_pymatgen(monkeypatch, _VR)
    tool = _tool()
    r = tool._parse_vasprun_quick(p)
    assert r["parse_source"] == "pymatgen_vasprun"
    assert r["band_gap"] == pytest.approx(1.2)
    assert r["efermi"] == pytest.approx(5.0)


def test_parse_vasprun_elementtree(monkeypatch, tmp_path):
    p = tmp_path / "vasprun.xml"
    p.write_text(
        "<vasprun><calculation><energy><i name='e_wo_entrp'>-10.5</i></energy>"
        "<varray name='forces'><v>1 2 3</v></varray></calculation>"
        "<kpoints><varray name='kpointlist'><v>0 0 0</v></varray></kpoints>"
        "</vasprun>",
        encoding="utf-8",
    )
    # Vasprun 构造抛异常 → 落 ElementTree
    class _BadVR:
        def __init__(self, path):
            raise RuntimeError("pymatgen parse fail")

    _install_fake_pymatgen(monkeypatch, _BadVR)
    tool = _tool()
    r = tool._parse_vasprun_quick(p)
    assert r["energy_vasprun"] == pytest.approx(-10.5)
    assert r["kpoint_count"] == 1
    assert r["forces_vasprun"] == [[1.0, 2.0, 3.0]]


def test_parse_vasprun_bad_xml(tmp_path):
    p = tmp_path / "vasprun.xml"
    p.write_text("not <xml", encoding="utf-8")
    tool = _tool()
    r = tool._parse_vasprun_quick(p)
    assert "parse_error" in r


# ── _read_incar_params / _modify_incar ───────────────────────────────────


def test_read_incar_params(tmp_path, monkeypatch):
    d = tmp_path / "wd"
    d.mkdir()
    (d / "INCAR").write_text(
        "# comment\nENCUT = 520\nISMEAR = 1\nLREAL = .TRUE.\n", encoding="utf-8"
    )
    tool = _tool()
    params = tool._read_incar_params(d)
    assert params["ENCUT"] == 520
    assert params["ISMEAR"] == 1
    assert "LREAL" in params


def test_read_incar_empty(tmp_path):
    tool = _tool()
    assert tool._read_incar_params(tmp_path) == {}


def test_modify_incar_override_and_add(tmp_path):
    incar = tmp_path / "INCAR"
    incar.write_text("ENCUT = 400\n# keep\n", encoding="utf-8")
    tool = _tool()
    tool._modify_incar(incar, {"ENCUT": 600, "ISMEAR": 1})
    content = incar.read_text(encoding="utf-8")
    assert "ENCUT = 600" in content
    assert "ISMEAR = 1" in content  # 新增
    assert "# keep" in content


# ── _uq_hint / _structure_file_hint ──────────────────────────────────────


def test_uq_hint():
    tool = _tool()
    h = tool._uq_hint()
    assert h["tool"] == "gp_tool"
    assert "X" in h["data_mapping"]


def test_structure_file_hint(tmp_path):
    tool = _tool()
    h = tool._structure_file_hint(tmp_path)
    assert h["type"] == "vasp_optimized_structure"
    assert h["exists"] is False
    assert "xrd_sim_tool" in "\n".join(h["downstream_tools"])


# ── _is_float ────────────────────────────────────────────────────────────


def test_is_float():
    tool = _tool()
    assert tool._is_float("3.14") is True
    assert tool._is_float("abc") is False
