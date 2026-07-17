"""Regression tests for audit_20260717/14 P1-3 (DEM script), P1-4 (MSD unwrap),
and P1-5 (SCF convergence criterion).

Each P1 fix gets one focused test class. The tests are minimal — they verify
the specific bug identified in the audit no longer reproduces, plus one
near-edge case. They are NOT exhaustive.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from huginn.execution.physics_auditor import PhysicsAuditor
from huginn.tools.sim.lammps_tool import LammpsTool, LammpsToolInput
from huginn.tools.sim.vasp_tool import VaspTool


# ── P1-3: DEM script generator ──────────────────────────────────────────────


class TestDemScriptSyntax:
    """Verify the generated LAMMPS DEM script is syntactically valid per
    https://docs.lammps.org/pair_granular.html (audit_20260717/14 P1-3)."""

    @pytest.fixture
    def script(self) -> str:
        args = LammpsToolInput(
            action="dem_packing",
            dem_radius=5.0,
            dem_density=2500.0,
            dem_youngs=1e8,
            dem_poisson=0.3,
            dem_friction=0.5,
            dem_restitution=0.8,
            dem_n_steps=10000,
        )
        return LammpsTool._generate_dem_input_script(args)

    def test_set_after_create_atoms(self, script: str):
        """`set type 1 diameter ...` must come AFTER create_atoms — LAMMPS set
        only acts on existing atoms, so calling it before create_atoms is a
        silent no-op (the original bug, audit P1-3a)."""
        i_create = script.index("create_atoms")
        i_set = script.index("set             type 1 diameter")
        assert i_set > i_create, "set diameter must come after create_atoms"

    def test_pair_coeff_uses_official_syntax(self, script: str):
        """pair_coeff must use the official LAMMPS granular syntax:
        `hertz/material E v gamma_n tangential mindlin_rescale G_t mu_t x_t
        damping tsuji gamma_n`. The old script had `damping_coeff` (臆造),
        `tsudi` (typo for tsuji), and illegal `rolling/twisting` positions."""
        # locate the pair_coeff block (single line + continuations)
        idx = script.index("pair_coeff")
        block = []
        for line in script[idx:].splitlines():
            block.append(line)
            if not line.rstrip().endswith("\\"):
                break
        block_text = " ".join(b.strip().rstrip("\\") for b in block)

        # tangential must come before mindlin_rescale (前置)
        assert "tangential" in block_text
        assert "mindlin" in block_text
        i_tan = block_text.index("tangential")
        i_mind = block_text.index("mindlin")
        assert i_tan < i_mind, "tangential keyword must precede mindlin model"

        # damping must be the tsuji model, not the bogus `damping_coeff`
        assert "damping tsuji" in block_text
        assert "damping_coeff" not in block_text, "damping_coeff is not a real LAMMPS keyword"
        assert "tsudi" not in block_text, "tsudi is a typo for tsuji"

    def test_timestep_derived_from_rayleigh(self, script: str):
        """timestep must be derived from Rayleigh contact time
        t_R = π·r·sqrt(ρ/G) / (0.1631ν + 0.8766), not the old hardcoded 1e-6/1e-7."""
        # parse timestep value from script
        ts_line = next(
            line for line in script.splitlines() if line.strip().startswith("timestep")
        )
        ts_val = float(ts_line.split()[1])

        # recompute expected Rayleigh-derived dt
        r, rho, E, nu = 5.0, 2500.0, 1e8, 0.3
        G = E / (2 * (1 + nu))
        rayleigh_t = math.pi * r * math.sqrt(rho / G) / (0.1631 * nu + 0.8766)
        expected_dt = 0.1 * rayleigh_t
        assert ts_val == pytest.approx(expected_dt, rel=1e-3)

        # sanity: not the old hardcoded values
        assert ts_val != pytest.approx(1e-6, rel=1e-3)
        assert ts_val != pytest.approx(1e-7, rel=1e-3)


# ── P1-4: MSD unwrap ────────────────────────────────────────────────────────


def _frame(timestep: int, atoms: list[dict], box=None) -> dict:
    return {
        "timestep": timestep,
        "atoms": atoms,
        "box": box or [[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]],
    }


class TestMsdUnwrap:
    """Verify _compute_msd handles xu/yu/zu, ix/iy/iz, and wrapped-only
    incremental MIC unwrap correctly (audit_20260717/14 P1-4)."""

    @pytest.fixture
    def tool(self) -> LammpsTool:
        return LammpsTool()

    def test_unwrapped_coords_direct_diff(self, tool: LammpsTool):
        """When xu/yu/zu are available, MSD = mean((xu[t]-xu[0])²).
        Particle drifts 1 unit per frame in x; MSD should be 1, 4, 9."""
        frames = [
            _frame(0, [{"x": 0.0, "y": 0.0, "z": 0.0,
                        "xu": 0.0, "yu": 0.0, "zu": 0.0}]),
            _frame(1, [{"x": 1.0, "y": 0.0, "z": 0.0,
                        "xu": 1.0, "yu": 0.0, "zu": 0.0}]),
            _frame(2, [{"x": 2.0, "y": 0.0, "z": 0.0,
                        "xu": 2.0, "yu": 0.0, "zu": 0.0}]),
            _frame(3, [{"x": 3.0, "y": 0.0, "z": 0.0,
                        "xu": 3.0, "yu": 0.0, "zu": 0.0}]),
        ]
        msd = tool._compute_msd(frames)
        assert msd is not None
        assert [m["msd"] for m in msd] == pytest.approx([1.0, 4.0, 9.0])

    def test_image_flag_reconstruction(self, tool: LammpsTool):
        """ix/iy/iz image flags reconstruct unwrapped: xu = x + ix*Lx.
        Box L=10, particle at x=1 with ix=2 → xu=21, diff from x=0=0."""
        frames = [
            _frame(0, [{"x": 0.0, "y": 0.0, "z": 0.0,
                        "ix": 0, "iy": 0, "iz": 0}]),
            _frame(1, [{"x": 1.0, "y": 0.0, "z": 0.0,
                        "ix": 2, "iy": 0, "iz": 0}]),
        ]
        msd = tool._compute_msd(frames)
        assert msd is not None
        # xu = 1 + 2*10 = 21, diff = 21, MSD = 21² = 441
        assert msd[0]["msd"] == pytest.approx(441.0)

    def test_wrapped_incremental_unwrap_no_jump(self, tool: LammpsTool):
        """Wrapped-only path: particle crosses boundary each frame by +0.5.
        Without incremental unwrap, diff from frame[0] would jump by ~L each
        wrap. With incremental MIC unwrap, total displacement accumulates."""
        L = 10.0
        # particle moves +0.5 each frame, wrapping at L
        positions = [(i * 0.5) % L for i in range(5)]
        frames = [_frame(i, [{"x": x, "y": 0.0, "z": 0.0}])
                  for i, x in enumerate(positions)]
        msd = tool._compute_msd(frames)
        assert msd is not None
        # at frame i, unwrapped position is 0.5*i, MSD = (0.5*i)²
        for i, m in enumerate(msd, start=1):
            assert m["msd"] == pytest.approx((0.5 * i) ** 2, abs=1e-9), \
                f"frame {i}: expected {(0.5*i)**2}, got {m['msd']}"

    def test_wrapped_saturation_warning(self, tool: LammpsTool):
        """When per-frame displacement exceeds L/2 (so MIC unwrap misses a
        wrap event), a saturation warning must be surfaced via _msd_warnings."""
        L = 10.0
        # particle moves 6.0 per frame (more than L/2=5.0) — MIC will see -4.0
        # instead of +6.0, missing the wrap. Saturation guard must fire.
        frames = [
            _frame(0, [{"x": 0.0, "y": 0.0, "z": 0.0}]),
            _frame(1, [{"x": 6.0, "y": 0.0, "z": 0.0}]),
        ]
        tool._compute_msd(frames)
        warnings = getattr(tool, "_msd_warnings", [])
        assert any("may have been missed" in w for w in warnings), \
            f"expected saturation warning, got: {warnings}"

    def test_wrapped_no_warning_for_small_displacement(self, tool: LammpsTool):
        """No saturation warning when all per-frame displacements are < L/2."""
        frames = [
            _frame(0, [{"x": 0.0, "y": 0.0, "z": 0.0}]),
            _frame(1, [{"x": 1.0, "y": 0.0, "z": 0.0}]),
        ]
        tool._compute_msd(frames)
        # the "wrapped-only" notice is allowed, but no "missed a wrapping event"
        warnings = getattr(tool, "_msd_warnings", [])
        assert not any("missed a wrapping event" in w for w in warnings)


# ── P1-5: SCF/band/dos convergence criterion ────────────────────────────────


class TestScfConvergenceCriterion:
    """Verify _parse_outcar uses action-aware convergence (audit P1-5):
    - relax: ionic marker "reached required accuracy"
    - scf: electronic "EDIFF is reached"
    - band/dos: non-SCC, any output = converged"""

    @pytest.fixture
    def tool(self) -> VaspTool:
        return VaspTool(vasp_executable="fake")

    def test_scf_with_ediff_is_converged(self, tool: VaspTool, tmp_path: Path):
        """SCF OUTCAR with "EDIFF is reached" must be converged=True,
        even without the ionic marker "reached required accuracy"."""
        outcar = tmp_path / "OUTCAR"
        outcar.write_text(
            " vasp.5.4.1\n"
            " free  energy   TOTEN  =       -10.5 eV\n"
            "   NELM =    60\n"
            "   EDIFF is reached\n"
            " E-fermi :   2.5\n",
            encoding="utf-8",
        )
        result = tool._parse_outcar_python(outcar, action="scf")
        assert result["converged"] is True, \
            "SCF with EDIFF reached must be converged"

    def test_scf_nelm_hit_not_converged(self, tool: VaspTool, tmp_path: Path):
        """SCF OUTCAR without "EDIFF is reached" (NELM hit) must be
        converged=False."""
        outcar = tmp_path / "OUTCAR"
        outcar.write_text(
            " vasp.5.4.1\n"
            " free  energy   TOTEN  =       -10.5 eV\n"
            "   NELM =    60\n"
            " E-fermi :   2.5\n",
            encoding="utf-8",
        )
        result = tool._parse_outcar_python(outcar, action="scf")
        assert result["converged"] is False, \
            "SCF without EDIFF reached (NELM hit) must NOT be converged"

    def test_relax_uses_ionic_marker(self, tool: VaspTool, tmp_path: Path):
        """relax OUTCAR uses "reached required accuracy" (ionic marker).
        Without it, relaxed=False even if "EDIFF is reached" is present."""
        outcar = tmp_path / "OUTCAR"
        outcar.write_text(
            " vasp.5.4.1\n"
            " free  energy   TOTEN  =       -10.5 eV\n"
            "   NELM =    60\n"
            "   EDIFF is reached\n"
            " E-fermi :   2.5\n",
            encoding="utf-8",
        )
        result = tool._parse_outcar_python(outcar, action="relax")
        # EDIFF reached alone does NOT mark relax as converged — needs ionic
        assert result["converged"] is False, \
            "relax must require ionic marker, not just EDIFF"

    def test_band_non_scc_treats_output_as_converged(
        self, tool: VaspTool, tmp_path: Path
    ):
        """band/dos are non-self-consistent (ICHARG=11), no SCF loop runs —
        any output (E-fermi or energy line) = converged=True."""
        outcar = tmp_path / "OUTCAR"
        outcar.write_text(
            " vasp.5.4.1\n"
            " free  energy   TOTEN  =       -10.5 eV\n"
            " E-fermi :   2.5\n",
            encoding="utf-8",
        )
        for action in ("band", "dos"):
            result = tool._parse_outcar_python(outcar, action=action)
            assert result["converged"] is True, \
                f"{action} non-SCC with output must be converged=True"

    def test_auditor_demotes_scf_nonconverged_to_warning(
        self, auditor: PhysicsAuditor
    ):
        """physics_auditor must NOT flag scf/band/dos converged=False as
        'error' (which triggers AutoFixLoop 2× retry). Demote to 'warning'."""
        parsed = {
            "energy": -150.5,
            "converged": False,
            "band_gap": 1.2,
            "magnetic_moments": [0.5],
            "volume": 250.0,
            "nelm": 60,
            "ispin": 2,
        }
        params = {
            "structure": "15\ncomment\nC 0 0 0\n" + "H 0 0 1\n" * 14,
            "action": "scf",
        }
        report = auditor.audit("vasp_tool", "scf", parsed, params)
        errors = [f for f in report.findings if f.severity == "error"
                  and f.field == "converged"]
        warnings = [f for f in report.findings if f.severity == "warning"
                    and f.field == "converged"]
        assert not errors, "scf converged=False must NOT be severity=error"
        assert len(warnings) == 1, \
            "scf converged=False should be severity=warning (1 finding)"

    def test_auditor_keeps_relax_nonconverged_as_error(
        self, auditor: PhysicsAuditor
    ):
        """relax converged=False stays as 'error' — that's a real failure
        of ionic relaxation, AutoFixLoop should retry."""
        parsed = {
            "energy": -150.5,
            "converged": False,
            "band_gap": 1.2,
            "magnetic_moments": [0.5],
            "volume": 250.0,
            "nelm": 60,
            "ispin": 2,
        }
        params = {
            "structure": "15\ncomment\nC 0 0 0\n" + "H 0 0 1\n" * 14,
            "action": "relax",
        }
        report = auditor.audit("vasp_tool", "relax", parsed, params)
        errors = [f for f in report.findings if f.severity == "error"
                  and f.field == "converged"]
        assert len(errors) == 1, \
            "relax converged=False must be severity=error"


@pytest.fixture
def auditor() -> PhysicsAuditor:
    return PhysicsAuditor()
