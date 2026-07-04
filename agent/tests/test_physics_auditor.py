"""Tests for PhysicsAuditor — checks that computational results are physically plausible.

The auditor is pure dict-in/dict-out logic, so no mocking of tools or files
is needed. Each test feeds a hand-crafted parsed dict and asserts on the
returned AuditReport.
"""

from __future__ import annotations

import pytest

from huginn.execution.physics_auditor import (
    AuditReport,
    PhysicsAuditor,
    PhysicsFinding,
)


# ── shared test fixtures ────────────────────────────────────────────────────

# A clean VASP result: negative energy per atom, modest gap, converged,
# small magnetic moments, positive volume. Should produce zero findings.
VALID_VASP_PARSED = {
    "energy": -150.5,  # ~-10 eV/atom for 15 atoms
    "converged": True,
    "band_gap": 1.2,
    "magnetic_moments": [0.5, -0.3],
    "volume": 250.0,
    "nelm": 60,
    "ispin": 2,
}

# XYZ-style structure string: first line is the atom count (15).
VALID_VASP_PARAMS = {
    "structure": "15\ncomment\nC 0 0 0\n" + "H 0 0 1\n" * 14,
    "action": "scf",
}


@pytest.fixture
def auditor() -> PhysicsAuditor:
    return PhysicsAuditor()


# ── VASP / DFT checks ───────────────────────────────────────────────────────


class TestVaspAudit:
    def test_no_findings_for_valid_result(self, auditor):
        report = auditor.audit("vasp_tool", "scf", VALID_VASP_PARSED, VALID_VASP_PARAMS)
        assert report.findings == []
        assert not report.has_errors
        assert not report.has_warnings

    def test_positive_energy_error(self, auditor):
        # +50 eV over 15 atoms -> +3.33 eV/atom, material is unbound
        parsed = {**VALID_VASP_PARSED, "energy": 50.0}
        report = auditor.audit("vasp_tool", "scf", parsed, VALID_VASP_PARAMS)

        errors = [f for f in report.findings if f.severity == "error"]
        assert len(errors) == 1
        assert errors[0].field == "energy"
        assert "positive" in errors[0].message.lower()
        assert report.has_errors

    def test_negative_bandgap_error(self, auditor):
        parsed = {**VALID_VASP_PARSED, "band_gap": -0.5}
        report = auditor.audit("vasp_tool", "scf", parsed, VALID_VASP_PARAMS)

        errors = [f for f in report.findings if f.severity == "error"]
        assert len(errors) == 1
        assert errors[0].field == "band_gap"
        assert "negative" in errors[0].message.lower()

    def test_not_converged_error(self, auditor):
        parsed = {**VALID_VASP_PARSED, "converged": False}
        report = auditor.audit("vasp_tool", "relax", parsed, VALID_VASP_PARAMS)

        errors = [f for f in report.findings if f.severity == "error"]
        assert any(f.field == "converged" for f in errors)
        assert report.has_errors

    def test_large_magnetic_moment_warning(self, auditor):
        # 20 μB is way past the d-element range
        parsed = {**VALID_VASP_PARSED, "magnetic_moments": [20.0, -0.3]}
        report = auditor.audit("vasp_tool", "scf", parsed, VALID_VASP_PARAMS)

        warnings = [f for f in report.findings if f.severity == "warning"]
        assert len(warnings) == 1
        assert warnings[0].field == "magnetic_moments"
        assert report.has_warnings
        assert not report.has_errors

    def test_high_nelm_info(self, auditor):
        # NELM=300 with converged=True -> only an info-level note
        parsed = {**VALID_VASP_PARSED, "nelm": 300}
        report = auditor.audit("vasp_tool", "scf", parsed, VALID_VASP_PARAMS)

        infos = [f for f in report.findings if f.severity == "info"]
        assert len(infos) == 1
        assert infos[0].field == "nelm"
        assert not report.has_errors
        assert not report.has_warnings

    def test_isif_mismatch_info(self, auditor):
        # ISIF=2 with action=relax means cell shape/volume were never touched
        params = {**VALID_VASP_PARAMS, "isif": 2}
        report = auditor.audit("vasp_tool", "relax", VALID_VASP_PARSED, params)

        infos = [f for f in report.findings if f.severity == "info"]
        assert any(f.field == "isif" and f.category == "parameter_mismatch" for f in infos)

    def test_non_positive_volume_error(self, auditor):
        parsed = {**VALID_VASP_PARSED, "volume": -10.0}
        report = auditor.audit("vasp_tool", "scf", parsed, VALID_VASP_PARAMS)

        errors = [f for f in report.findings if f.severity == "error"]
        assert any(f.field == "volume" for f in errors)


# ── LAMMPS / MD checks ──────────────────────────────────────────────────────


class TestLammpsAudit:
    def test_lammps_high_temp_warning(self, auditor):
        data = {
            "thermo_data": {"temp": [300.0, 5000.0, 15000.0]},
            "final_energy": -100.0,
        }
        report = auditor.audit("lammps_tool", "run", data, {})

        warnings = [f for f in report.findings if f.severity == "warning"]
        assert any(f.field == "temp" and f.category == "unphysical_value" for f in warnings)
        assert report.has_warnings

    def test_lammps_energy_drift_warning(self, auditor):
        # First half steady at -100, second half drifts to -90 -> ~10% drift
        energies = [-100.0] * 10 + [-100.0, -98.0, -96.0, -94.0, -92.0,
                                    -90.0, -90.0, -90.0, -90.0, -90.0]
        data = {
            "thermo_data": {"toteng": energies},
            "final_energy": energies[-1],
        }
        report = auditor.audit("lammps_tool", "run", data, {})

        warnings = [f for f in report.findings if f.severity == "warning"]
        assert any(f.field == "toteng" and f.category == "thermodynamic_violation" for f in warnings)

    def test_lammps_no_final_energy_warning(self, auditor):
        # Empty thermo + missing final energy for a run action
        data = {"thermo_data": {}, "final_energy": None}
        report = auditor.audit("lammps_tool", "run", data, {})

        warnings = [f for f in report.findings if f.severity == "warning"]
        assert any(f.field == "final_energy" for f in warnings)
        assert report.has_warnings

    def test_lammps_temperature_spike_warning(self, auditor):
        # 25 steps: baseline ~300 K, then sudden jump to 2000 K in last 5
        baseline = [300.0] * 20
        spike = [2000.0] * 5
        data = {
            "thermo_data": {"temp": baseline + spike},
            "final_energy": -100.0,
        }
        report = auditor.audit("lammps_tool", "run", data, {})

        spikes = [f for f in report.findings if f.category == "thermodynamic_violation"
                  and f.field == "temp"]
        assert len(spikes) == 1
        assert "spike" in spikes[0].message.lower()

    def test_lammps_valid_result_no_findings(self, auditor):
        # Stable NVT-ish run: temp around 300 K, steady energy
        data = {
            "thermo_data": {
                "temp": [300.0] * 25,
                "toteng": [-100.0] * 25,
                "press": [1.0] * 25,
            },
            "final_energy": -100.0,
        }
        report = auditor.audit("lammps_tool", "run", data, {})
        assert report.findings == []
        assert not report.has_errors
        assert not report.has_warnings


# ── report structure & edge cases ───────────────────────────────────────────


class TestAuditReport:
    def test_audit_report_has_errors_property(self, auditor):
        # Positive energy -> error, large mag moment -> warning
        parsed = {
            "energy": 50.0,
            "converged": True,
            "magnetic_moments": [20.0],
            "ispin": 2,
        }
        report = auditor.audit("vasp_tool", "scf", parsed, VALID_VASP_PARAMS)

        assert report.has_errors
        assert report.has_warnings
        severities = {f.severity for f in report.findings}
        assert "error" in severities
        assert "warning" in severities

    def test_unknown_tool_skipped(self, auditor):
        # Unknown tool name should not crash and should yield no findings
        report = auditor.audit("unknown_tool", "whatever", {"energy": 999}, {})

        assert report.findings == []
        assert not report.has_errors
        assert not report.has_warnings
        assert report.tool_name == "unknown_tool"

    def test_to_dict_serialization(self, auditor):
        parsed = {**VALID_VASP_PARSED, "energy": 50.0}
        report = auditor.audit("vasp_tool", "scf", parsed, VALID_VASP_PARAMS)

        d = report.to_dict()
        assert d["tool_name"] == "vasp_tool"
        assert d["action"] == "scf"
        assert isinstance(d["findings"], list)
        assert len(d["findings"]) >= 1
        assert d["has_errors"] is True

        # Each finding dict has the expected keys
        first = d["findings"][0]
        for key in ("severity", "category", "message", "field", "value", "expected_range"):
            assert key in first

    def test_finding_to_dict_roundtrip(self):
        finding = PhysicsFinding(
            severity="error",
            category="unphysical_value",
            message="bad energy",
            field="energy",
            value=42.0,
            expected_range="-200 to 0",
        )
        d = finding.to_dict()
        assert d["severity"] == "error"
        assert d["category"] == "unphysical_value"
        assert d["value"] == 42.0
        assert d["expected_range"] == "-200 to 0"

    def test_empty_report_properties(self):
        # A fresh report with no findings should report False on both flags
        report = AuditReport(tool_name="vasp_tool", action="scf")
        assert not report.has_errors
        assert not report.has_warnings
        d = report.to_dict()
        assert d["findings"] == []
        assert d["has_errors"] is False

    def test_qe_tool_uses_vasp_checks(self, auditor):
        # QE/CP2K share the VASP energy/convergence semantics
        report = auditor.audit("qe_tool", "scf", VALID_VASP_PARSED, VALID_VASP_PARAMS)
        assert report.findings == []

        bad_parsed = {**VALID_VASP_PARSED, "energy": 50.0}
        report = auditor.audit("qe_tool", "scf", bad_parsed, VALID_VASP_PARAMS)
        assert report.has_errors

    def test_none_input_params_does_not_crash(self, auditor):
        # input_params defaults to {} when None is passed
        report = auditor.audit("vasp_tool", "scf", VALID_VASP_PARSED, None)
        # No n_atoms derivable from empty params -> energy check skipped
        assert not report.has_errors


# ── integration smoke test: auditor wired into tool result path ─────────────


class TestToolIntegration:
    """Smoke-test that the auditor import path used inside the tools resolves.

    The real VaspTool/LammpsTool call sites import PhysicsAuditor lazily inside
    a try/except. We just verify the import works so the integration is wired
    up correctly without needing a full VASP/LAMMPS run.
    """

    def test_import_from_execution_package(self):
        from huginn.execution.physics_auditor import PhysicsAuditor as PA

        assert PA is PhysicsAuditor
