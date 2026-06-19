"""Smoke tests for modules that previously had zero coverage.

These tests just exercise basic construction and happy-path methods to guard
against import regressions and simple runtime errors.
"""

from __future__ import annotations

from pathlib import Path

import pytest


class TestEvolutionSmoke:
    def test_execution_logger(self, tmp_path: Path):
        from huginn.evolution.logger import ExecutionLogger

        logger = ExecutionLogger(persist_dir=str(tmp_path))
        logger.log_tool_call(
            session_id="s1",
            tool_name="vasp_tool",
            tool_input={"args": {}},
            result={"energy": -1.0},
        )
        logger.log_conversation(
            session_id="s1",
            user_message="hello",
            agent_response="hi",
        )
        path = logger.export_for_evolution()
        assert Path(path).exists()

    def test_evolution_engine_initialization(self, tmp_path: Path):
        from huginn.evolution.engine import EvolutionEngine
        from huginn.evolution.logger import ExecutionLogger

        logger = ExecutionLogger(persist_dir=str(tmp_path))
        engine = EvolutionEngine(logger=logger)
        assert str(engine.rules_path).endswith("evolution_rules.json")

    def test_skill_library(self, tmp_path: Path):
        from huginn.evolution.skill_evolver import Skill, SkillLibrary

        lib = SkillLibrary(str(tmp_path / "skills.json"))
        skill = Skill(
            skill_id="test_skill",
            name="test_skill",
            description="A test skill",
            domain="test",
            trigger_patterns=["test"],
            workflow=[],
        )
        lib.add(skill)
        assert lib.get("test_skill") is not None
        assert lib.find_by_trigger("test query")[0].skill_id == "test_skill"


class TestPhysicsValidationSmoke:
    def test_validate_dft_result(self):
        from huginn.validation.physics import PhysicsValidator

        validator = PhysicsValidator()
        result = {
            "total_energy": -10.0,
            "forces": [[0.01, 0.0, 0.0]],
            "band_gap": 1.0,
            "volume": 10.0,
            "magnetic_moments": {"Fe": 2.1},
        }
        checks = validator.validate_dft_result(result)
        assert checks
        assert all(hasattr(c, "passed") for c in checks)

    def test_validate_md_result(self):
        from huginn.validation.physics import PhysicsValidator

        validator = PhysicsValidator()
        result = {
            "energies": [1.0, 1.001, 1.002],
            "temperatures": [300, 301, 299],
            "atom_count": 10,
            "density": 2.3,
        }
        checks = validator.validate_md_result(result)
        assert checks


class TestMCPClientSmoke:
    async def test_manager_initial_state(self):
        from huginn.mcp_client import MCPClientManager

        mgr = MCPClientManager()
        assert mgr.list_tools() == []
        assert mgr.get_tool_info("unknown") is None


class TestDiagnosticsSmoke:
    def test_convergence_diagnosis(self):
        from huginn.diagnostics.convergence import ConvergenceDiagnostician

        doc = ConvergenceDiagnostician()
        report = doc.diagnose("vasp", "EDDDAV error in the log")
        assert report is not None
        assert "电子步不收敛" in report.problem
        fixes = doc.suggest_auto_fix(report)
        assert fixes is not None
        assert "ALGO" in fixes


class TestMechanicsSmoke:
    def test_import_and_basic_functions(self):
        from huginn import mechanics

        assert mechanics.__name__ == "huginn.mechanics"
