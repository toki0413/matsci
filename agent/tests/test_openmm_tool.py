"""openmm_tool.py 集成路径补测 — 覆盖 is_read_only/is_destructive、call 分派、
_energy_minimize(pdb 缺失/import 缺失/成功/异常)、_md_run(同上)、_analyze
(traj 缺失/import 缺失/pdb 缺失/rmsd·energy·temperature·rg·未知)、分析器、
_resolve_file、_nonbonded_method、_constraints、_make_simulation、_parse_md_log.

openmm 未安装 → ImportError 分支真实命中; 成功路径用 fake openmm 模块覆盖.
把 openmm_tool.py 覆盖率从 22% 提升到 90%+.
"""

from __future__ import annotations

import builtins
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from huginn.tools.sim.openmm_tool import OpenMMTool, OpenMMToolInput

pytestmark = pytest.mark.anyio


# ── fake openmm 模块 ─────────────────────────────────────────────────────


class _FakePlatform:
    @staticmethod
    def getPlatformByName(name):
        return "CPU"


class _FakeState:
    def __init__(self, pe=10.0, positions=None):
        self._pe = pe
        self._positions = positions or []

    def getPotentialEnergy(self):
        return self

    def getPositions(self):
        return self._positions

    def value_in_unit(self, unit):
        return self._pe


class _FakeContext:
    def __init__(self):
        self.positions = []

    def setPositions(self, p):
        self.positions = p

    def getState(self, **kw):
        return _FakeState(pe=10.0, positions=self.positions)


class _FakeSimulation:
    def __init__(self, *a, **k):
        self.context = _FakeContext()
        self.reporters = []
        self._steps = 0

    def minimizeEnergy(self, maxIterations=None):
        pass

    def step(self, n):
        self._steps += n


class _FakeUnitVal:
    def __init__(self, v):
        self.v = v

    def value_in_unit(self, u):
        return self.v


def _install_fake_openmm(monkeypatch):
    openmm = types.ModuleType("openmm")
    openmm.LangevinMiddleIntegrator = lambda *a, **k: object()
    openmm.MonteCarloBarostat = lambda *a, **k: object()
    openmm.Platform = _FakePlatform
    # unit 值需支持与浮点相乘/相除 (如 300.0*kelvin, 1.0/picoseconds)
    class _Unit:
        def __mul__(self, other):
            return other

        def __rmul__(self, other):
            return other

        def __truediv__(self, other):
            return other

        def __rtruediv__(self, other):
            return other

    unit = types.ModuleType("openmm.unit")
    for name in [
        "kelvin", "kilojoules_per_mole", "nanometers", "picoseconds",
        "atmospheres",
    ]:
        setattr(unit, name, _Unit())
    openmm.unit = unit

    app = types.ModuleType("openmm.app")

    class _FakePositions(list):
        pass

    class _PDBFile:
        def __init__(self, *a, **k):
            self.topology = object()
            # positions 长度需与 dcd 帧数一致, 供 rmsd/rg 分析使用
            self.positions = [
                _FakeUnitVal(np.array([0.0, 0.0, 0.0])),
                _FakeUnitVal(np.array([0.0, 0.0, 0.0])),
            ]

        @staticmethod
        def writeFile(topology, positions, f):
            f.write("fake pdb")

    class _Modeller:
        def __init__(self, topology, positions):
            self.topology = topology
            self.positions = positions

        def addHydrogens(self, *a, **k):
            self.positions = []

        def addSolvent(self, *a, **k):
            self.positions = []

    class _FakeSystem:
        def addForce(self, *a, **k):
            pass

    class _ForceField:
        def __init__(self, *a, **k):
            pass

        def createSystem(self, topology, **kw):
            return _FakeSystem()

    class _DCDReporter:
        def __init__(self, *a, **k):
            pass

    class _StateDataReporter:
        def __init__(self, *a, **k):
            pass

    class _DCDFile:
        def __init__(self, *a, **k):
            pass

        @staticmethod
        def open(*a, **k):
            return _DCDFile()

        def getNumFramesPerFile(self):
            return 2

        def readPositions(self, i):
            # returns a list of unit-vector objects
            return [_FakeUnitVal(np.array([0.0, 0.0, 0.0])), _FakeUnitVal(np.array([1.0, 0.0, 0.0]))]

    class _PME:
        pass

    class _CutoffNonPeriodic:
        pass

    class _NoCutoff:
        pass

    class _AllBonds:
        pass

    class _HBonds:
        pass

    app.PDBFile = _PDBFile
    app.Modeller = _Modeller
    app.ForceField = _ForceField
    app.DCDReporter = _DCDReporter
    app.StateDataReporter = _StateDataReporter
    app.DCDFile = _DCDFile
    app.Simulation = _FakeSimulation
    app.PME = _PME
    app.CutoffNonPeriodic = _CutoffNonPeriodic
    app.NoCutoff = _NoCutoff
    app.AllBonds = _AllBonds
    app.HBonds = _HBonds
    openmm.app = app

    monkeypatch.setitem(sys.modules, "openmm", openmm)
    monkeypatch.setitem(sys.modules, "openmm.unit", unit)
    monkeypatch.setitem(sys.modules, "openmm.app", app)


def _tool():
    return OpenMMTool()


def _args(**kw):
    """Return a raw dict for tool.call() (which parses it into OpenMMToolInput)."""
    base = {"action": "energy_minimize", "working_dir": "."}
    base.update(kw)
    return base


def _inp(**kw):
    """Return an OpenMMToolInput for direct method calls."""
    return OpenMMToolInput(**_args(**kw))


# ── 基础方法 ─────────────────────────────────────────────────────────────


def test_is_read_only():
    assert _tool().is_read_only(_inp(action="analyze")) is True
    assert _tool().is_read_only(_inp(action="md_run")) is False


def test_is_destructive():
    assert _tool().is_destructive(_inp(action="md_run")) is True
    assert _tool().is_destructive(_inp(action="energy_minimize")) is True
    assert _tool().is_destructive(_inp(action="analyze")) is False


def test_call_unknown_action():
    # pydantic Literal 约束 action, 非法值应在解析时被拒绝
    with pytest.raises(ValidationError):
        _tool().call({"action": "bogus"})


# ── _energy_minimize ─────────────────────────────────────────────────────


def test_energy_minimize_pdb_missing(tmp_path):
    res = _tool().call(_args(working_dir=str(tmp_path), pdb_file="missing.pdb"))
    assert res.success is False
    assert "PDB file not found" in res.error


def test_energy_minimize_import_missing(monkeypatch, tmp_path):
    # 确保 openmm 不存在
    monkeypatch.setitem(sys.modules, "openmm", None)
    pdb = tmp_path / "x.pdb"
    pdb.write_text("fake", encoding="utf-8")
    res = _tool().call(
        _args(working_dir=str(tmp_path), pdb_file=str(pdb), action="energy_minimize")
    )
    assert res.success is True
    assert res.data["status"] == "skipped"


def test_energy_minimize_success(monkeypatch, tmp_path):
    _install_fake_openmm(monkeypatch)
    pdb = tmp_path / "x.pdb"
    pdb.write_text("fake", encoding="utf-8")
    # 物理审计: 用不抛异常的 auditor
    auditor_mod = types.ModuleType("huginn.execution.physics_auditor")

    class _Audit:
        def to_dict(self):
            return {"has_errors": False, "findings": []}

    class _Auditor:
        def audit(self, *a, **k):
            return _Audit()

    auditor_mod.PhysicsAuditor = _Auditor
    monkeypatch.setitem(sys.modules, "huginn.execution.physics_auditor", auditor_mod)
    res = _tool().call(
        _args(working_dir=str(tmp_path), pdb_file=str(pdb), action="energy_minimize")
    )
    assert res.success is True
    assert res.data["action"] == "energy_minimize"
    assert res.data["converged"] is False
    assert (tmp_path / "minimized.pdb").exists()


def test_energy_minimize_success_max_iterations_zero(monkeypatch, tmp_path):
    _install_fake_openmm(monkeypatch)
    pdb = tmp_path / "x.pdb"
    pdb.write_text("fake", encoding="utf-8")
    res = _tool().call(
        _args(
            working_dir=str(tmp_path), pdb_file=str(pdb),
            action="energy_minimize", max_iterations=0,
        )
    )
    assert res.success is True


def test_energy_minimize_solvent_implicit(monkeypatch, tmp_path):
    _install_fake_openmm(monkeypatch)
    pdb = tmp_path / "x.pdb"
    pdb.write_text("fake", encoding="utf-8")
    res = _tool().call(
        _args(
            working_dir=str(tmp_path), pdb_file=str(pdb),
            action="energy_minimize", solvent="implicit",
        )
    )
    assert res.success is True


def test_energy_minimize_exception(monkeypatch, tmp_path):
    _install_fake_openmm(monkeypatch)
    pdb = tmp_path / "x.pdb"
    pdb.write_text("fake", encoding="utf-8")
    openmm_app = sys.modules["openmm.app"]

    class _BoomPDBFile:
        def __init__(self, *a, **k):
            raise RuntimeError("pdb boom")

    openmm_app.PDBFile = _BoomPDBFile
    res = _tool().call(
        _args(working_dir=str(tmp_path), pdb_file=str(pdb), action="energy_minimize")
    )
    assert res.success is False
    assert "OpenMM minimization failed" in res.error


def test_energy_minimize_audit_exception(monkeypatch, tmp_path):
    """物理审计抛异常不应阻断结果返回."""
    _install_fake_openmm(monkeypatch)
    pdb = tmp_path / "x.pdb"
    pdb.write_text("fake", encoding="utf-8")
    auditor_mod = types.ModuleType("huginn.execution.physics_auditor")

    class _BoomAuditor:
        def audit(self, *a, **k):
            raise RuntimeError("audit boom")

    auditor_mod.PhysicsAuditor = _BoomAuditor
    monkeypatch.setitem(sys.modules, "huginn.execution.physics_auditor", auditor_mod)
    res = _tool().call(
        _args(working_dir=str(tmp_path), pdb_file=str(pdb), action="energy_minimize")
    )
    assert res.success is True
    assert res.data["action"] == "energy_minimize"


# ── _md_run ──────────────────────────────────────────────────────────────


def test_md_run_pdb_missing(tmp_path):
    res = _tool().call(_args(working_dir=str(tmp_path), pdb_file="missing.pdb", action="md_run"))
    assert res.success is False
    assert "PDB file not found" in res.error


def test_md_run_import_missing(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "openmm", None)
    pdb = tmp_path / "x.pdb"
    pdb.write_text("fake", encoding="utf-8")
    res = _tool().call(
        _args(working_dir=str(tmp_path), pdb_file=str(pdb), action="md_run")
    )
    assert res.success is True
    assert res.data["status"] == "skipped"


def test_md_run_success(monkeypatch, tmp_path):
    _install_fake_openmm(monkeypatch)
    pdb = tmp_path / "x.pdb"
    pdb.write_text("fake", encoding="utf-8")
    auditor_mod = types.ModuleType("huginn.execution.physics_auditor")

    class _Audit:
        def to_dict(self):
            return {"has_errors": False, "findings": []}

    class _Auditor:
        def audit(self, *a, **k):
            return _Audit()

    auditor_mod.PhysicsAuditor = _Auditor
    monkeypatch.setitem(sys.modules, "huginn.execution.physics_auditor", auditor_mod)
    # fake DCDReporter 不写文件, 预创建以便断言输出存在
    (tmp_path / "trajectory.dcd").write_bytes(b"")
    res = _tool().call(
        _args(working_dir=str(tmp_path), pdb_file=str(pdb), action="md_run")
    )
    assert res.success is True
    assert res.data["action"] == "md_run"
    assert (tmp_path / "final.pdb").exists()
    assert (tmp_path / "trajectory.dcd").exists()


def test_md_run_nvt_no_equil(monkeypatch, tmp_path):
    _install_fake_openmm(monkeypatch)
    pdb = tmp_path / "x.pdb"
    pdb.write_text("fake", encoding="utf-8")
    res = _tool().call(
        _args(
            working_dir=str(tmp_path), pdb_file=str(pdb), action="md_run",
            ensemble="nvt", equilibration_steps=0,
        )
    )
    assert res.success is True


def test_md_run_audit_exception(monkeypatch, tmp_path):
    """md_run 物理审计抛异常不应阻断结果返回."""
    _install_fake_openmm(monkeypatch)
    pdb = tmp_path / "x.pdb"
    pdb.write_text("fake", encoding="utf-8")
    auditor_mod = types.ModuleType("huginn.execution.physics_auditor")

    class _BoomAuditor:
        def audit(self, *a, **k):
            raise RuntimeError("audit boom")

    auditor_mod.PhysicsAuditor = _BoomAuditor
    monkeypatch.setitem(sys.modules, "huginn.execution.physics_auditor", auditor_mod)
    res = _tool().call(
        _args(working_dir=str(tmp_path), pdb_file=str(pdb), action="md_run")
    )
    assert res.success is True
    assert res.data["action"] == "md_run"


def test_md_run_exception(monkeypatch, tmp_path):
    _install_fake_openmm(monkeypatch)
    pdb = tmp_path / "x.pdb"
    pdb.write_text("fake", encoding="utf-8")
    openmm_app = sys.modules["openmm.app"]

    class _BoomPDBFile:
        def __init__(self, *a, **k):
            raise RuntimeError("md boom")

    openmm_app.PDBFile = _BoomPDBFile
    res = _tool().call(
        _args(working_dir=str(tmp_path), pdb_file=str(pdb), action="md_run")
    )
    assert res.success is False
    assert "OpenMM MD failed" in res.error


# ── _analyze ─────────────────────────────────────────────────────────────


def test_analyze_traj_missing(tmp_path):
    res = _tool().call(
        _args(working_dir=str(tmp_path), action="analyze", trajectory_file="missing.dcd")
    )
    assert res.success is False
    assert "Trajectory file not found" in res.error


def test_analyze_import_missing(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "openmm", None)
    traj = tmp_path / "x.dcd"
    traj.write_bytes(b"fake")
    res = _tool().call(
        _args(working_dir=str(tmp_path), action="analyze", trajectory_file=str(traj))
    )
    assert res.success is True
    assert res.data["status"] == "skipped"


def test_analyze_pdb_missing(monkeypatch, tmp_path):
    _install_fake_openmm(monkeypatch)
    traj = tmp_path / "x.dcd"
    traj.write_bytes(b"fake")
    res = _tool().call(
        _args(working_dir=str(tmp_path), action="analyze", trajectory_file=str(traj))
    )
    assert res.success is False
    assert "PDB file needed" in res.error


def test_analyze_rmsd(monkeypatch, tmp_path):
    _install_fake_openmm(monkeypatch)
    traj = tmp_path / "x.dcd"
    traj.write_bytes(b"fake")
    pdb = tmp_path / "ref.pdb"
    pdb.write_text("fake", encoding="utf-8")
    res = _tool().call(
        _args(
            working_dir=str(tmp_path), action="analyze",
            trajectory_file=str(traj), pdb_file=str(pdb), analysis_type="rmsd",
        )
    )
    assert res.success is True
    assert res.data["analysis_type"] == "rmsd"
    assert res.data["n_frames"] == 2


def test_analyze_energy(monkeypatch, tmp_path):
    _install_fake_openmm(monkeypatch)
    traj = tmp_path / "x.dcd"
    traj.write_bytes(b"fake")
    pdb = tmp_path / "ref.pdb"
    pdb.write_text("fake", encoding="utf-8")
    # 写 md_log.csv
    (tmp_path / "md_log.csv").write_text(
        "Step,Potential,Kinetic,Temperature,Volume\n"
        "0,-10.0,1.0,300.0,1.0\n"
        "10,-11.0,1.1,301.0,1.1\n",
        encoding="utf-8",
    )
    res = _tool().call(
        _args(
            working_dir=str(tmp_path), action="analyze",
            trajectory_file=str(traj), pdb_file=str(pdb), analysis_type="energy",
        )
    )
    assert res.success is True
    assert res.data["analysis_type"] == "energy"


def test_analyze_energy_no_log(monkeypatch, tmp_path):
    _install_fake_openmm(monkeypatch)
    traj = tmp_path / "x.dcd"
    traj.write_bytes(b"fake")
    pdb = tmp_path / "ref.pdb"
    pdb.write_text("fake", encoding="utf-8")
    res = _tool().call(
        _args(
            working_dir=str(tmp_path), action="analyze",
            trajectory_file=str(traj), pdb_file=str(pdb), analysis_type="energy",
        )
    )
    assert res.success is False
    assert "md_log.csv not found" in res.error


def test_analyze_temperature(monkeypatch, tmp_path):
    _install_fake_openmm(monkeypatch)
    traj = tmp_path / "x.dcd"
    traj.write_bytes(b"fake")
    pdb = tmp_path / "ref.pdb"
    pdb.write_text("fake", encoding="utf-8")
    (tmp_path / "md_log.csv").write_text(
        "Step,Potential,Kinetic,Temperature,Volume\n"
        "0,-10.0,1.0,300.0,1.0\n"
        "10,-11.0,1.1,302.0,1.1\n",
        encoding="utf-8",
    )
    res = _tool().call(
        _args(
            working_dir=str(tmp_path), action="analyze",
            trajectory_file=str(traj), pdb_file=str(pdb), analysis_type="temperature",
        )
    )
    assert res.success is True
    assert res.data["mean_temperature"] == 301.0


def test_analyze_radius_gyration(monkeypatch, tmp_path):
    _install_fake_openmm(monkeypatch)
    traj = tmp_path / "x.dcd"
    traj.write_bytes(b"fake")
    pdb = tmp_path / "ref.pdb"
    pdb.write_text("fake", encoding="utf-8")
    res = _tool().call(
        _args(
            working_dir=str(tmp_path), action="analyze",
            trajectory_file=str(traj), pdb_file=str(pdb), analysis_type="radius_gyration",
        )
    )
    assert res.success is True
    assert res.data["analysis_type"] == "radius_gyration"


def test_analyze_unknown_type(monkeypatch, tmp_path):
    _install_fake_openmm(monkeypatch)
    traj = tmp_path / "x.dcd"
    traj.write_bytes(b"fake")
    pdb = tmp_path / "ref.pdb"
    pdb.write_text("fake", encoding="utf-8")
    res = _tool().call(
        _args(
            working_dir=str(tmp_path), action="analyze",
            trajectory_file=str(traj), pdb_file=str(pdb), analysis_type="rmsd",
        )
    )
    # 触发未知类型: 直接构造 (analysis_type 是 Literal, 用 model_construct 绕过校验)
    bad = OpenMMToolInput.model_construct(
        action="analyze", working_dir=str(tmp_path),
        trajectory_file=str(traj), pdb_file=str(pdb), analysis_type="bogus",
    )
    res2 = _tool()._analyze(bad, Path(tmp_path))
    assert res2.success is False
    assert "Unknown analysis type" in res2.error


def test_analyze_exception(monkeypatch, tmp_path):
    _install_fake_openmm(monkeypatch)
    traj = tmp_path / "x.dcd"
    traj.write_bytes(b"fake")
    pdb = tmp_path / "ref.pdb"
    pdb.write_text("fake", encoding="utf-8")
    openmm_app = sys.modules["openmm.app"]

    class _BoomPDBFile:
        def __init__(self, *a, **k):
            raise RuntimeError("analyze boom")

    openmm_app.PDBFile = _BoomPDBFile
    res = _tool().call(
        _args(
            working_dir=str(tmp_path), action="analyze",
            trajectory_file=str(traj), pdb_file=str(pdb), analysis_type="rmsd",
        )
    )
    assert res.success is False
    assert "Analysis failed" in res.error


# ── 分析器 ───────────────────────────────────────────────────────────────


def test_analyze_rmsd_method(monkeypatch, tmp_path):
    _install_fake_openmm(monkeypatch)
    openmm_app = sys.modules["openmm.app"]
    dcd = openmm_app.DCDFile()
    # pdb.positions 需要与 dcd 帧长度一致 (2)
    pdb = types.SimpleNamespace(
        positions=[
            _FakeUnitVal(np.array([0.0, 0.0, 0.0])),
            _FakeUnitVal(np.array([0.0, 0.0, 0.0])),
        ]
    )
    res = _tool()._analyze_rmsd(dcd, pdb)
    assert res.success is True
    assert len(res.data["rmsd_values_angstrom"]) == 2


def test_analyze_rg_method(monkeypatch, tmp_path):
    _install_fake_openmm(monkeypatch)
    openmm_app = sys.modules["openmm.app"]
    dcd = openmm_app.DCDFile()
    pdb = types.SimpleNamespace(positions=[])
    res = _tool()._analyze_rg(dcd, pdb)
    assert res.success is True
    assert len(res.data["rg_values_nm"]) == 2


def test_analyze_temperature_method_no_log(tmp_path):
    res = _tool()._analyze_temperature(tmp_path / "nope.dcd")
    assert res.success is False


# ── helpers ──────────────────────────────────────────────────────────────


def test_resolve_file():
    tool = _tool()
    assert tool._resolve_file(None, Path(".")) is None
    assert tool._resolve_file("", Path(".")) is None
    # 不存在的相对路径
    assert tool._resolve_file("missing.pdb", Path(".")) is None


def test_nonbonded_method(monkeypatch):
    _install_fake_openmm(monkeypatch)
    assert OpenMMTool._nonbonded_method(_inp(solvent="explicit")) is sys.modules["openmm.app"].PME
    assert OpenMMTool._nonbonded_method(_inp(solvent="implicit")) is sys.modules["openmm.app"].NoCutoff
    assert OpenMMTool._nonbonded_method(_inp(solvent="vacuum")) is sys.modules["openmm.app"].CutoffNonPeriodic


def test_constraints(monkeypatch):
    _install_fake_openmm(monkeypatch)
    assert OpenMMTool._constraints(_inp(solvent="vacuum")) is sys.modules["openmm.app"].AllBonds
    assert OpenMMTool._constraints(_inp(solvent="explicit")) is sys.modules["openmm.app"].HBonds


def test_make_simulation(monkeypatch):
    _install_fake_openmm(monkeypatch)
    modeller = types.SimpleNamespace(topology=object())
    sim = OpenMMTool._make_simulation(modeller, object(), object(), "CPU")
    assert sim is not None


def test_parse_md_log(tmp_path):
    log = tmp_path / "md_log.csv"
    log.write_text(
        "Step,Potential,Kinetic,Temperature,Volume\n"
        "0,-10.0,1.0,300.0,1.0\n"
        "10,-11.0,1.1,302.0,1.1\n"
        "20,-12.0,1.2,304.0,1.2\n",
        encoding="utf-8",
    )
    parsed = OpenMMTool._parse_md_log(log)
    assert parsed["n_data_points"] == 3
    assert parsed["temperatures"] == [300.0, 302.0, 304.0]
    assert parsed["volumes"] == [1.0, 1.1, 1.2]


def test_parse_md_log_missing(tmp_path):
    assert OpenMMTool._parse_md_log(tmp_path / "nope.csv") == {}


def test_parse_md_log_bad_rows(tmp_path):
    log = tmp_path / "md_log.csv"
    log.write_text(
        "Step,Potential,Kinetic,Temperature,Volume\n"
        "0,-10.0,1.0,300.0\n"  # 长度 4, 有 volume 列
        "bad,row\n"
        "10,-11.0,1.1,302.0,1.1\n",
        encoding="utf-8",
    )
    parsed = OpenMMTool._parse_md_log(log)
    assert parsed["n_data_points"] == 2


def test_parse_md_log_short_rows(tmp_path):
    log = tmp_path / "md_log.csv"
    log.write_text(
        "Step,Potential,Kinetic,Temperature,Volume\n"
        "0\n"
        "10,-11.0,1.1,302.0,1.1\n",
        encoding="utf-8",
    )
    parsed = OpenMMTool._parse_md_log(log)
    assert parsed["n_data_points"] == 1


def test_parse_md_log_read_exception(monkeypatch, tmp_path):
    """读取日志抛异常时, 异常被记录并返回空列表结构 (不阻断)."""
    log = tmp_path / "md_log.csv"
    log.write_text("x", encoding="utf-8")

    real_open = builtins.open

    def _boom_open(*a, **k):
        if a:
            target = str(a[0])
            if target.endswith("md_log.csv"):
                raise OSError("boom")
        return real_open(*a, **k)

    monkeypatch.setattr(builtins, "open", _boom_open)
    parsed = OpenMMTool._parse_md_log(log)
    assert parsed["n_data_points"] == 0
    assert parsed["steps"] == []
