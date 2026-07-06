"""Tests for DFT auto-healing: VaspTool retry loop + DiagnoseTool recommend_fixes."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from huginn.tools.diagnose_tool import DiagnoseInput, DiagnoseTool
from huginn.tools.sim.vasp_tool import VaspTool, VaspToolInput


def _sandbox_result(returncode: int, stderr: str = "", stdout: str = ""):
    return SimpleNamespace(returncode=returncode, stderr=stderr, stdout=stdout)


def _make_workdir(tmp_path: Path, incar_content: str = "ALGO = Fast\nNELM = 60\n") -> Path:
    (tmp_path / "POSCAR").write_text("dummy poscar\n", encoding="utf-8")
    (tmp_path / "INCAR").write_text(incar_content, encoding="utf-8")
    return tmp_path


class TestVaspAutoHeal:
    def test_retries_after_zbrent_error_and_succeeds(self, tmp_path, monkeypatch):
        work_dir = _make_workdir(tmp_path)
        tool = VaspTool(vasp_executable="fake-vasp")

        # 第一次失败 (ZBRENT), 第二次成功
        calls = {"n": 0}

        def fake_run(cmd, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return _sandbox_result(1, stderr="ZBRENT: fatal error in bracketing")
            # 模拟 VASP 收敛后写出 OUTCAR
            (work_dir / "OUTCAR").write_text(
                "reached required accuracy\n"
                "free  energy   TOTEN  =  -10.0\n",
                encoding="utf-8",
            )
            return _sandbox_result(0, stderr="", stdout="reached required accuracy")

        monkeypatch.setattr(tool.sandbox, "run", fake_run)

        args = VaspToolInput(
            action="scf", working_dir=str(work_dir), max_auto_retries=2
        )
        result = asyncio.run(tool.call(args, MagicMock()))

        assert result.success is True
        assert calls["n"] == 2
        # INCAR 被改了
        incar_text = (work_dir / "INCAR").read_text(encoding="utf-8")
        assert "Normal" in incar_text  # ALGO=Normal
        # autoheal 日志记了
        assert "autoheal_attempts" in result.data
        assert len(result.data["autoheal_attempts"]) == 1
        assert result.data["autoheal_attempts"][0]["fixes_applied"]["ALGO"] == "Normal"

    def test_no_retry_when_no_rule_matches(self, tmp_path, monkeypatch):
        work_dir = _make_workdir(tmp_path)
        tool = VaspTool(vasp_executable="fake-vasp")

        def fake_run(cmd, **kwargs):
            return _sandbox_result(1, stderr="some unknown gibberish error xyz123")

        monkeypatch.setattr(tool.sandbox, "run", fake_run)

        args = VaspToolInput(
            action="scf", working_dir=str(work_dir), max_auto_retries=3
        )
        result = asyncio.run(tool.call(args, MagicMock()))

        assert result.success is False
        # 没命中规则, 只跑了一次
        # autoheal 不应被记录 (没修动)
        assert "autoheal_attempts" not in result.data or len(result.data.get("autoheal_attempts", [])) == 0

    def test_max_retries_exhausted_returns_failure(self, tmp_path, monkeypatch):
        work_dir = _make_workdir(tmp_path)
        tool = VaspTool(vasp_executable="fake-vasp")

        def fake_run(cmd, **kwargs):
            return _sandbox_result(1, stderr="EDDDAV: diagonalization failed")

        monkeypatch.setattr(tool.sandbox, "run", fake_run)

        args = VaspToolInput(
            action="scf", working_dir=str(work_dir), max_auto_retries=1
        )
        result = asyncio.run(tool.call(args, MagicMock()))

        assert result.success is False
        # 应该跑了 2 次 (1 初次 + 1 重试)
        # autoheal 记了 1 次 (重试那次)
        assert "autoheal_attempts" in result.data
        assert len(result.data["autoheal_attempts"]) == 1

    def test_max_auto_retries_zero_disables_healing(self, tmp_path, monkeypatch):
        work_dir = _make_workdir(tmp_path)
        tool = VaspTool(vasp_executable="fake-vasp")

        calls = {"n": 0}

        def fake_run(cmd, **kwargs):
            calls["n"] += 1
            return _sandbox_result(1, stderr="ZBRENT: fatal error")

        monkeypatch.setattr(tool.sandbox, "run", fake_run)

        args = VaspToolInput(
            action="scf", working_dir=str(work_dir), max_auto_retries=0
        )
        result = asyncio.run(tool.call(args, MagicMock()))

        assert result.success is False
        assert calls["n"] == 1  # 只跑一次, 不重试
        assert "autoheal_attempts" not in result.data

    def test_read_incar_params_parses_numbers(self, tmp_path):
        work_dir = _make_workdir(tmp_path, "ALGO = Fast\nNELM = 60\nENCUT = 520.0\nISMEAR = 0\n")
        tool = VaspTool(vasp_executable="fake-vasp")
        params = tool._read_incar_params(work_dir)
        assert params["ALGO"] == "Fast"
        assert params["NELM"] == 60
        assert params["ENCUT"] == 520.0
        assert params["ISMEAR"] == 0


class TestDiagnoseRecommendFixes:
    def test_recommend_fixes_matches_zbrent(self):
        tool = DiagnoseTool()
        args = DiagnoseInput(
            action="recommend_fixes",
            error_message="ZBRENT: fatal error in bracketing",
            software="vasp",
            current_params={"ALGO": "Fast"},
        )
        result = asyncio.run(tool.call(args, MagicMock()))

        assert result.success is True
        assert result.data["matched"] is True
        assert result.data["recommended_fixes"]["ALGO"] == "Normal"
        assert result.data["reasoning"] is not None

    def test_recommend_fixes_no_match_returns_empty(self):
        tool = DiagnoseTool()
        args = DiagnoseInput(
            action="recommend_fixes",
            error_message="completely unknown error pattern xyz",
            software="vasp",
        )
        result = asyncio.run(tool.call(args, MagicMock()))

        assert result.success is True
        assert result.data["matched"] is False
        assert result.data["recommended_fixes"] == {}

    def test_recommend_fixes_software_name_normalized(self):
        # 传 "vasp" 应该被补成 "vasp_tool" 匹配规则
        tool = DiagnoseTool()
        args = DiagnoseInput(
            action="recommend_fixes",
            error_message="EDDDAV diagonalization error",
            software="vasp",
            current_params={},
        )
        result = asyncio.run(tool.call(args, MagicMock()))
        assert result.data["matched"] is True
        assert "ALGO" in result.data["recommended_fixes"]

    def test_diagnose_action_still_works(self):
        # 原 diagnose 路径不受影响
        tool = DiagnoseTool()
        args = DiagnoseInput(
            action="diagnose",
            error_message="SCF convergence failure",
            software="vasp",
            calculation_type="DFT",
        )
        result = asyncio.run(tool.call(args, MagicMock()))
        assert result.success is True
        assert "general_advice" in result.data
