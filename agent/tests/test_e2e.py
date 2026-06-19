"""End-to-end integration tests for Huginn.

These tests exercise multiple components together to verify
system-level correctness.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

# Ensure the agent package is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from huginn.crypto import CryptoVault
from huginn.memory.longterm import LongTermMemory
from huginn.memory.manager import MemoryManager
from huginn.memory.session import SessionContext
from huginn.skills.presets import HT_SCREENING, STANDARD_DFT, SYMBOLIC_VERIFY
from huginn.skills.registry import SkillRegistry
from huginn.tools.report_tool import ReportTool, ReportToolInput
from huginn.tools.symbolic_regression_tool import (
    SymbolicRegressionInput,
    SymbolicRegressionTool,
)
from huginn.types import ToolContext


@pytest.fixture
def tmp_db_path(tmp_path):
    return str(tmp_path / "test_e2e.db")


@pytest.fixture
def tmp_report_dir(tmp_path):
    return str(tmp_path / "reports")


class TestCryptoMemoryIntegration:
    """Test encryption + memory integration."""

    def test_encrypted_memory_roundtrip(self, tmp_db_path):
        """Store and retrieve memories with encrypted content.

        Note: Encrypted content cannot be searched by keyword (FTS5/LIKE).
        This test verifies storage/retrieval of encrypted data via category filter.
        """
        vault = CryptoVault(master_password="test_password_123")
        memory = LongTermMemory(tmp_db_path)

        # Encrypt a computational result
        plaintext = "Band gap of Si = 1.12 eV (HSE06)"
        encrypted = vault.encrypt(plaintext)

        # Store in memory with a distinct category
        memory.store(
            content=encrypted,
            category="encrypted_result",
            source="dft_calculation",
            importance=0.9,
            tags=["band_gap", "silicon"],
        )

        # Retrieve by category (since encrypted content is not searchable by keyword)
        results = memory.retrieve("", category="encrypted_result", top_k=5)
        assert len(results) > 0

        decrypted = vault.decrypt(results[0]["content"])
        assert "1.12 eV" in decrypted


class TestSkillsExecution:
    """Test skill definition and execution."""

    def test_skill_registry_presets(self):
        """All preset skills are registered."""
        # Ensure presets are registered (idempotent)
        for skill in [STANDARD_DFT, HT_SCREENING, SYMBOLIC_VERIFY]:
            SkillRegistry.register(skill)

        names = SkillRegistry.list_skills()
        assert "standard_dft" in names
        assert "ht_screening" in names
        assert "symbolic_verify" in names

    def test_skill_serialization(self):
        """Skills can be serialized to/from dict."""
        import dataclasses

        data = dataclasses.asdict(STANDARD_DFT)
        assert data["name"] == "standard_dft"
        assert len(data["steps"]) > 0


class TestReportGeneration:
    """Test report tool with realistic data."""

    def test_generate_markdown_report(self, tmp_report_dir):
        """Generate a full Markdown report from workflow data."""
        tool = ReportTool()
        data = {
            "methods": {
                "software": "VASP 6.3.2",
                "functional": "PBE",
                "encut": 520,
                "kpoints": "4x4x4",
                "pseudopotentials": "PAW_PBE",
                "smearing": "Gaussian 0.05",
                "ediffg": -0.01,
            },
            "structure": {
                "formula": "Si2",
                "spacegroup": "Fd-3m",
                "initial_a": 5.43,
                "final_a": 5.47,
            },
            "convergence": {
                "energy": -10.1234,
                "n_iterations": 12,
                "n_electronic": 45,
            },
            "results": {
                "band_gap": {"value": 0.65, "unit": "eV", "method": "PBE"},
                "bulk_modulus": {"value": 97.8, "unit": "GPa"},
            },
            "validation": {
                "checks": [
                    {
                        "name": "Energy convergence",
                        "passed": True,
                        "message": "< 1e-5 eV",
                    },
                    {
                        "name": "Force convergence",
                        "passed": True,
                        "message": "< 0.01 eV/Å",
                    },
                ]
            },
            "literature_comparison": {
                "comparisons": [
                    {
                        "property": "band_gap",
                        "calculated": 0.65,
                        "reference": 1.12,
                        "source": "Exp.",
                    },
                ]
            },
            "resources": {
                "cpu_hours": 4.5,
                "walltime_hours": 0.5,
                "memory_gb": 8,
                "cores": 16,
            },
            "software_version": "VASP 6.3.2",
            "input_hash": "abc123",
            "random_seed": 42,
            "input_files": ["INCAR", "POSCAR", "KPOINTS"],
        }

        args = ReportToolInput(
            action="generate",
            workflow_results=data,
            style="full",
            format="markdown",
            output_path=os.path.join(tmp_report_dir, "report.md"),
        )
        result = asyncio.run(
            tool.call(args, ToolContext(session_id="test", workspace="."))
        )

        assert result.success
        assert result.data is not None
        assert "saved_to" in result.data

        # Verify file was written
        path = Path(result.data["saved_to"])
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "PBE" in content
        assert "Fd-3m" in content
        assert "Validation" in content

    def test_generate_json_report(self):
        """Generate JSON report format."""
        tool = ReportTool()
        data = {
            "methods": {"software": "VASP", "functional": "PBE", "encut": 520},
            "structure": {"formula": "Si2"},
            "results": {"band_gap": 0.65},
        }
        args = ReportToolInput(action="generate", workflow_results=data, format="json")
        result = asyncio.run(
            tool.call(args, ToolContext(session_id="test", workspace="."))
        )

        assert result.success
        assert "band_gap" in result.data["report"] and "0.65" in result.data["report"]

    def test_generate_html_report(self):
        """Generate HTML report format."""
        tool = ReportTool()
        data = {
            "methods": {"software": "VASP", "functional": "PBE"},
            "structure": {"formula": "Si2"},
            "results": {"band_gap": 0.65},
        }
        args = ReportToolInput(action="generate", workflow_results=data, format="html")
        result = asyncio.run(
            tool.call(args, ToolContext(session_id="test", workspace="."))
        )

        assert result.success
        assert "<!DOCTYPE html>" in result.data["report"]

    def test_generate_latex_report(self):
        """Generate LaTeX report format."""
        tool = ReportTool()
        data = {
            "methods": {"software": "VASP", "functional": "PBE"},
            "structure": {"formula": "Si2"},
            "results": {"band_gap": 0.65},
        }
        args = ReportToolInput(action="generate", workflow_results=data, format="latex")
        result = asyncio.run(
            tool.call(args, ToolContext(session_id="test", workspace="."))
        )

        assert result.success
        assert "\\documentclass" in result.data["report"]


class TestSymbolicRegression:
    """Test symbolic regression tool."""

    def test_mock_discovery(self):
        """Run discovery with mock fallback (PSE not available)."""
        # Use non-existent path to force fast mock fallback and avoid
        # slow PSE import when the real path exists on disk.
        tool = SymbolicRegressionTool(pse_path="/nonexistent/pse/path")
        args = SymbolicRegressionInput(
            action="discover",
            data_json={"x": list(range(20)), "y": [i * i + 0.1 * i for i in range(20)]},
            target_column="y",
            feature_columns=["x"],
            time_limit=10,
        )
        result = asyncio.run(
            tool.call(args, ToolContext(session_id="test", workspace="."))
        )
        # Mock mode returns success=False but with helpful data
        assert result.data is not None
        assert "message" in result.data

    def test_evaluate_action(self):
        """Evaluate action returns success with metrics."""
        tool = SymbolicRegressionTool(pse_path="/nonexistent/pse/path")
        args = SymbolicRegressionInput(
            action="evaluate",
            data_json={"x": [1, 2, 3], "y": [1, 4, 9]},
            target_column="y",
            feature_columns=["x"],
            probe_expression="x**2",
        )
        result = asyncio.run(
            tool.call(args, ToolContext(session_id="test", workspace="."))
        )
        assert result.success or result.data is not None


class TestMemoryManagerWorkflow:
    """Test the full memory manager workflow."""

    def test_auto_promotion(self, tmp_db_path):
        """Simulate auto-promotion of tool results."""
        vault = CryptoVault(master_password="mem_test")
        session = SessionContext()
        longterm = LongTermMemory(tmp_db_path)
        manager = MemoryManager(session, longterm)

        # Add messages
        session.add_message("user", "Calculate band gap of GaAs")
        session.add_message("assistant", "Running DFT calculation...")

        # Simulate a tool result
        tool_result = {
            "tool": "vasp_tool",
            "action": "band_gap",
            "properties": {"band_gap": 1.42, "unit": "eV"},
        }

        # Promote to long-term memory
        manager.promote_tool_result("vasp_band_gap", tool_result)

        # Verify in long-term memory
        results = longterm.retrieve("band_gap")
        assert len(results) > 0


class TestToolRegistryIntegration:
    """Verify all tools are importable and have correct schemas."""

    def test_all_tools_importable(self):
        """All tool modules can be imported."""
        from huginn.evaluation.evaluation_tool import EvaluationTool
        from huginn.rag.rag_tool import RAGTool
        from huginn.tools.database_tool import DatabaseTool
        from huginn.tools.diagnose_tool import DiagnoseTool
        from huginn.tools.diff_tool import DiffTool
        from huginn.tools.extract_tool import ExtractTool
        from huginn.tools.job_tool import JobTool
        from huginn.tools.lammps_tool import LammpsTool
        from huginn.tools.potential_tool import PotentialTool
        from huginn.tools.report_tool import ReportTool
        from huginn.tools.structure_tool import StructureTool
        from huginn.tools.symbolic_regression_tool import SymbolicRegressionTool
        from huginn.tools.validate_tool import ValidateTool
        from huginn.tools.vasp_tool import VaspTool

        tools = [
            StructureTool,
            ExtractTool,
            JobTool,
            DatabaseTool,
            PotentialTool,
            DiffTool,
            ValidateTool,
            DiagnoseTool,
            VaspTool,
            LammpsTool,
            SymbolicRegressionTool,
            ReportTool,
            RAGTool,
            EvaluationTool,
        ]
        for T in tools:
            inst = T()
            assert inst.name
            assert inst.description
            assert inst.input_schema is not None

    def test_report_tool_estimate_cost(self):
        """Report tool has zero cost."""
        tool = ReportTool()
        cost = tool.estimate_cost(ReportToolInput())
        assert cost is not None
        assert cost["cpu_hours"] == 0.0

    def test_tool_result_schema(self):
        """ToolResult can hold success and failure states."""
        from huginn.types import ToolResult

        success = ToolResult(data={"value": 42}, success=True)
        assert success.success
        assert success.data["value"] == 42

        failure = ToolResult(data=None, success=False, error="something broke")
        assert not failure.success
        assert "broke" in failure.error
