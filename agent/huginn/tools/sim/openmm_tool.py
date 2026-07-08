"""OpenMM biomolecular MD tool — Python-native molecular dynamics.

OpenMM is a GPU-accelerated MD engine with a Python API, ideal for
biomolecular simulations (proteins, nucleic acids, ligands). Handles
energy minimization, NVT/NPT equilibration, production runs, and basic
trajectory analysis. Falls back gracefully when openmm is not installed.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.phases import ResearchPhase
from huginn.security import SandboxError, SandboxExecutor
from huginn.tools.base import HuginnTool, ToolProfile
from huginn.types import ToolContext, ToolResult

logger = logging.getLogger(__name__)


class OpenMMToolInput(BaseModel):
    action: Literal["energy_minimize", "md_run", "analyze"] = Field(...)
    working_dir: str | None = Field(default=None)

    # Structure input — PDB or topology files
    pdb_file: str | None = Field(default=None, description="Input .pdb structure file")
    topology_file: str | None = Field(
        default=None, description="Topology file (prmtop/top/gro) if not embedded in PDB"
    )

    # Force field selection
    forcefield: Literal["amber14-all", "amber14/protein.ff14SB", "charmm36", "amber99sbildn.xml", "amoeba2013.xml"] = Field(
        default="amber14-all", description="Force field XML name"
    )
    water_model: Literal["tip3p", "tip3pfb", "tip4pew", "spce"] = Field(
        default="tip3p", description="Water model for explicit solvent"
    )
    solvent: Literal["explicit", "implicit", "vacuum"] = Field(
        default="explicit", description="Solvent type"
    )

    # MD run params
    temperature: float = Field(default=300.0, gt=0, description="Temperature (K)")
    pressure: float = Field(default=1.0, description="Pressure (atm) — only for NPT")
    timestep_fs: float = Field(default=2.0, gt=0, description="Integration timestep (fs)")
    n_steps: int = Field(default=5000, ge=1, description="Number of MD steps")
    equilibration_steps: int = Field(default=500, ge=0, description="Equilibration steps before production")
    ensemble: Literal["nvt", "npt"] = Field(default="npt", description="Thermodynamic ensemble")

    # Minimize params
    max_iterations: int = Field(default=1000, ge=0, description="Max minimization iterations (0 = until converged)")

    # Output
    output_pdb: str | None = Field(default=None, description="Output PDB after run")
    trajectory_dcd: str | None = Field(default=None, description="Trajectory output .dcd file")
    report_interval: int = Field(default=500, ge=1, description="Steps between state reports")

    # Analysis
    trajectory_file: str | None = Field(default=None, description="Trajectory file for analysis")
    analysis_type: Literal["rmsd", "energy", "temperature", "radius_gyration"] = Field(
        default="rmsd", description="Analysis type"
    )


class OpenMMTool(HuginnTool):
    """Run OpenMM biomolecular molecular dynamics simulations."""

    name = "openmm_tool"
    category = "sim"
    profile = ToolProfile(
        cost_tier="heavy",
        phases=frozenset({ResearchPhase.EXECUTION}),
        constraint_scope="md",
        light_alternatives=("symbolic_math_tool", "numerical_tool"),
    )
    description = (
        "OpenMM biomolecular MD: energy minimization, NVT/NPT molecular "
        "dynamics with AMBER/CHARMM force fields. Supports explicit/implicit "
        "solvent and vacuum. Includes trajectory analysis (RMSD, energy, "
        "temperature, radius of gyration)."
    )
    input_schema = OpenMMToolInput

    def __init__(self, sandbox: SandboxExecutor | None = None):
        super().__init__()
        self.sandbox = sandbox or SandboxExecutor()

    def is_read_only(self, args: OpenMMToolInput) -> bool:
        return args.action == "analyze"

    def is_destructive(self, args: OpenMMToolInput) -> bool:
        return args.action in ("energy_minimize", "md_run")

    def call(
        self, args: dict[str, Any], context: ToolContext | None = None
    ) -> ToolResult:
        input_data = OpenMMToolInput(**args)
        work_dir = (
            Path(input_data.working_dir) if input_data.working_dir else Path.cwd()
        )
        work_dir.mkdir(parents=True, exist_ok=True)

        if input_data.action == "energy_minimize":
            return self._energy_minimize(input_data, work_dir)
        if input_data.action == "md_run":
            return self._md_run(input_data, work_dir)
        if input_data.action == "analyze":
            return self._analyze(input_data, work_dir)
        return ToolResult(data=None, success=False, error=f"Unknown action: {input_data.action}")

    # ── energy_minimize ──────────────────────────────────────

    def _energy_minimize(self, inp: OpenMMToolInput, work_dir: Path) -> ToolResult:
        pdb = self._resolve_file(inp.pdb_file, work_dir)
        if pdb is None:
            return ToolResult(data=None, success=False, error=f"PDB file not found: {inp.pdb_file}")

        try:
            from openmm.app import ForceField, Modeller, PDBFile
            from openmm import LangevinMiddleIntegrator, Platform
            from openmm.unit import kelvin, picoseconds, nanometers, kilojoules_per_mole
        except ImportError:
            return ToolResult(
                data={"action": "energy_minimize", "status": "skipped", "pdb_file": str(pdb)},
                success=True,
                error="openmm not installed. Install with: conda install -c conda-forge openmm",
            )

        try:
            pdb_struct = PDBFile(str(pdb))
            forcefield = ForceField(inp.forcefield, f"{inp.water_model}.xml")

            modeller = Modeller(pdb_struct.topology, pdb_struct.positions)
            if inp.solvent == "explicit":
                modeller.addHydrogens(forcefield, pH=7.0)
                modeller.addSolvent(forcefield, model=inp.water_model, padding=1.0*nanometers)

            system = forcefield.createSystem(modeller.topology,
                                              nonbondedMethod=self._nonbonded_method(inp),
                                              nonbondedCutoff=1.0*nanometers,
                                              constraints=self._constraints(inp))

            integrator = LangevinMiddleIntegrator(inp.temperature*kelvin, 1.0/picoseconds, inp.timestep_fs*0.001*picoseconds)
            platform = Platform.getPlatformByName("CPU")  # ponytail: GPU if available, but CPU is safer default
            sim = self._make_simulation(modeller, system, integrator, platform)
            sim.context.setPositions(modeller.positions)

            # Report initial energy
            state0 = sim.context.getState(getEnergy=True)
            e0 = state0.getPotentialEnergy().value_in_unit(kilojoules_per_mole)

            # Minimize
            if inp.max_iterations > 0:
                sim.minimizeEnergy(maxIterations=inp.max_iterations)
            else:
                sim.minimizeEnergy()

            state = sim.context.getState(getEnergy=True, getPositions=True)
            ef = state.getPotentialEnergy().value_in_unit(kilojoules_per_mole)

            out_pdb = Path(inp.output_pdb) if inp.output_pdb else work_dir / "minimized.pdb"
            with open(out_pdb, "w") as f:
                PDBFile.writeFile(modeller.topology, state.getPositions(), f)

            data = {
                "action": "energy_minimize",
                "forcefield": inp.forcefield,
                "initial_energy_kj_mol": round(e0, 2),
                "final_energy_kj_mol": round(ef, 2),
                "energy_change_kj_mol": round(e0 - ef, 2),
                "converged": abs(e0 - ef) > 0.01,
                "output_pdb": str(out_pdb),
                "message": "Energy minimization complete.",
            }

            # Physics audit — energy should decrease, not increase
            try:
                from huginn.execution.physics_auditor import PhysicsAuditor
                auditor = PhysicsAuditor()
                audit = auditor.audit("openmm_tool", "energy_minimize", data, inp.model_dump())
                data["physics_audit"] = audit.to_dict()
            except Exception:
                logger.debug("audit failure can't block result delivery", exc_info=True)

            return ToolResult(data=data)
        except Exception as exc:
            return ToolResult(data=None, success=False, error=f"OpenMM minimization failed: {exc}")

    # ── md_run ───────────────────────────────────────────────

    def _md_run(self, inp: OpenMMToolInput, work_dir: Path) -> ToolResult:
        pdb = self._resolve_file(inp.pdb_file, work_dir)
        if pdb is None:
            return ToolResult(data=None, success=False, error=f"PDB file not found: {inp.pdb_file}")

        try:
            from openmm.app import ForceField, Modeller, PDBFile, DCDReporter, StateDataReporter
            from openmm import LangevinMiddleIntegrator, MonteCarloBarostat, Platform
            from openmm.unit import kelvin, picoseconds, nanometers, atmospheres, nanometers
        except ImportError:
            return ToolResult(
                data={"action": "md_run", "status": "skipped", "pdb_file": str(pdb)},
                success=True,
                error="openmm not installed. Install with: conda install -c conda-forge openmm",
            )

        try:
            pdb_struct = PDBFile(str(pdb))
            forcefield = ForceField(inp.forcefield, f"{inp.water_model}.xml")

            modeller = Modeller(pdb_struct.topology, pdb_struct.positions)
            modeller.addHydrogens(forcefield, pH=7.0)
            if inp.solvent == "explicit":
                modeller.addSolvent(forcefield, model=inp.water_model, padding=1.0*nanometers)

            system = forcefield.createSystem(modeller.topology,
                                              nonbondedMethod=self._nonbonded_method(inp),
                                              nonbondedCutoff=1.0*nanometers,
                                              constraints=self._constraints(inp))

            if inp.ensemble == "npt":
                system.addForce(MonteCarloBarostat(inp.pressure*atmospheres, inp.temperature*kelvin))

            integrator = LangevinMiddleIntegrator(
                inp.temperature*kelvin, 1.0/picoseconds, inp.timestep_fs*0.001*picoseconds
            )
            platform = Platform.getPlatformByName("CPU")
            sim = self._make_simulation(modeller, system, integrator, platform)

            # Reporters
            traj_file = Path(inp.trajectory_dcd) if inp.trajectory_dcd else work_dir / "trajectory.dcd"
            log_file = work_dir / "md_log.csv"
            sim.reporters.append(DCDReporter(str(traj_file), inp.report_interval))
            sim.reporters.append(StateDataReporter(
                str(log_file), inp.report_interval,
                step=True, potentialEnergy=True, kineticEnergy=True,
                temperature=True, volume=True, speed=True,
            ))

            # Minimize + equilibrate
            sim.minimizeEnergy(maxIterations=100)
            if inp.equilibration_steps > 0:
                sim.step(inp.equilibration_steps)

            # Production
            sim.step(inp.n_steps)

            # Final state
            state = sim.context.getState(getEnergy=True, getPositions=True)
            from openmm.unit import kilojoules_per_mole
            final_pe = state.getPotentialEnergy().value_in_unit(kilojoules_per_mole)

            out_pdb = Path(inp.output_pdb) if inp.output_pdb else work_dir / "final.pdb"
            with open(out_pdb, "w") as f:
                PDBFile.writeFile(modeller.topology, state.getPositions(), f)

            # Parse the log for time series
            time_series = self._parse_md_log(log_file)

            data = {
                "action": "md_run",
                "forcefield": inp.forcefield,
                "ensemble": inp.ensemble,
                "temperature_k": inp.temperature,
                "pressure_atm": inp.pressure,
                "timestep_fs": inp.timestep_fs,
                "n_steps": inp.n_steps,
                "equilibration_steps": inp.equilibration_steps,
                "final_energy_kj_mol": round(final_pe, 2),
                "output_pdb": str(out_pdb),
                "trajectory_dcd": str(traj_file),
                "md_log": str(log_file),
                "time_series": time_series,
                "message": f"MD run complete: {inp.n_steps} steps at {inp.temperature}K.",
            }

            try:
                from huginn.execution.physics_auditor import PhysicsAuditor
                auditor = PhysicsAuditor()
                audit = auditor.audit("openmm_tool", "md_run", data, inp.model_dump())
                data["physics_audit"] = audit.to_dict()
            except Exception:
                logger.debug("audit failure can't block result delivery", exc_info=True)

            return ToolResult(data=data)
        except Exception as exc:
            return ToolResult(data=None, success=False, error=f"OpenMM MD failed: {exc}")

    # ── analyze ──────────────────────────────────────────────

    def _analyze(self, inp: OpenMMToolInput, work_dir: Path) -> ToolResult:
        traj = self._resolve_file(inp.trajectory_file, work_dir)
        if traj is None:
            return ToolResult(data=None, success=False, error=f"Trajectory file not found: {inp.trajectory_file}")

        try:
            from openmm.app import PDBFile, DCDFile
            from openmm.unit import kilojoules_per_mole, nanometers
        except ImportError:
            return ToolResult(
                data={"action": "analyze", "status": "skipped"},
                success=True,
                error="openmm not installed.",
            )

        try:
            pdb_file = self._resolve_file(inp.pdb_file, work_dir)
            if pdb_file is None:
                return ToolResult(data=None, success=False, error="PDB file needed as topology reference for analysis")

            pdb = PDBFile(str(pdb_file))
            dcd = DCDFile.open(str(traj))

            if inp.analysis_type == "rmsd":
                return self._analyze_rmsd(dcd, pdb)
            if inp.analysis_type == "energy":
                return self._analyze_energy(traj)
            if inp.analysis_type == "temperature":
                return self._analyze_temperature(traj)
            if inp.analysis_type == "radius_gyration":
                return self._analyze_rg(dcd, pdb)
            return ToolResult(data=None, success=False, error=f"Unknown analysis type: {inp.analysis_type}")
        except Exception as exc:
            return ToolResult(data=None, success=False, error=f"Analysis failed: {exc}")

    def _analyze_rmsd(self, dcd, pdb) -> ToolResult:
        import numpy as np

        ref_pos = pdb.positions
        n_frames = dcd.getNumFramesPerFile()
        rmsds = []
        for i in range(n_frames):
            frame = dcd.readPositions(i)
            diff = np.array([(frame[j].value_in_unit(nanometers) - ref_pos[j].value_in_unit(nanometers)) for j in range(len(ref_pos))])
            rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1))) * 10  # nm → Å
            rmsds.append(round(float(rmsd), 3))

        return ToolResult(data={
            "analysis_type": "rmsd",
            "n_frames": len(rmsds),
            "rmsd_values_angstrom": rmsds,
            "mean_rmsd": round(float(np.mean(rmsds)), 3) if rmsds else None,
            "max_rmsd": round(float(np.max(rmsds)), 3) if rmsds else None,
            "message": "RMSD trajectory analysis complete.",
        })

    def _analyze_energy(self, traj_path: Path) -> ToolResult:
        """Parse energy from the MD log CSV."""
        log_path = traj_path.parent / "md_log.csv"
        if not log_path.exists():
            return ToolResult(data=None, success=False, error="md_log.csv not found alongside trajectory")
        energies = self._parse_md_log(log_path)
        return ToolResult(data={
            "analysis_type": "energy",
            **energies,
            "message": "Energy trajectory analysis complete.",
        })

    def _analyze_temperature(self, traj_path: Path) -> ToolResult:
        log_path = traj_path.parent / "md_log.csv"
        if not log_path.exists():
            return ToolResult(data=None, success=False, error="md_log.csv not found")
        series = self._parse_md_log(log_path)
        temps = series.get("temperatures", [])
        return ToolResult(data={
            "analysis_type": "temperature",
            "temperatures": temps,
            "mean_temperature": round(sum(temps) / len(temps), 2) if temps else None,
            "temperature_std": round((sum((t - sum(temps)/len(temps))**2 for t in temps) / len(temps))**0.5, 2) if temps else None,
            "message": "Temperature trajectory analysis complete.",
        })

    def _analyze_rg(self, dcd, pdb) -> ToolResult:
        import numpy as np

        ref_pos = pdb.positions
        n_frames = dcd.getNumFramesPerFile()
        rgs = []
        for i in range(n_frames):
            frame = dcd.readPositions(i)
            coords = np.array([p.value_in_unit(nanometers) for p in frame])
            center = coords.mean(axis=0)
            rg = np.sqrt(np.mean(np.sum((coords - center)**2, axis=1)))
            rgs.append(round(float(rg), 3))

        return ToolResult(data={
            "analysis_type": "radius_gyration",
            "n_frames": len(rgs),
            "rg_values_nm": rgs,
            "mean_rg": round(float(np.mean(rgs)), 3) if rgs else None,
            "message": "Radius of gyration analysis complete.",
        })

    # ── helpers ──────────────────────────────────────────────

    def _resolve_file(self, path_str: str | None, work_dir: Path) -> Path | None:
        if not path_str:
            return None
        p = Path(path_str)
        if not p.is_absolute():
            p = work_dir / p
        return p if p.exists() else None

    @staticmethod
    def _nonbonded_method(inp: OpenMMToolInput):
        from openmm.app import NoCutoff, PME, CutoffNonPeriodic
        if inp.solvent == "explicit":
            return PME
        if inp.solvent == "implicit":
            return NoCutoff
        return CutoffNonPeriodic

    @staticmethod
    def _constraints(inp: OpenMMToolInput):
        from openmm.app import HBonds, AllBonds
        if inp.solvent == "vacuum":
            return AllBonds
        return HBonds

    @staticmethod
    def _make_simulation(modeller, system, integrator, platform):
        from openmm.app import Simulation
        return Simulation(modeller.topology, system, integrator, platform)

    @staticmethod
    def _parse_md_log(log_path: Path) -> dict[str, Any]:
        """Parse OpenMM StateDataReporter CSV output."""
        import csv

        steps, pes, kes, temps, vols = [], [], [], [], []
        if not log_path.exists():
            return {}
        try:
            with open(log_path, newline="") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                for row in reader:
                    if len(row) < 2:
                        continue
                    try:
                        steps.append(int(row[0]))
                        pes.append(float(row[1]))
                        kes.append(float(row[2]))
                        temps.append(float(row[3]))
                        if len(row) > 4:
                            vols.append(float(row[4]))
                    except (ValueError, IndexError):
                        continue
        except Exception:
            pass

        return {
            "steps": steps,
            "potential_energies": pes,
            "kinetic_energies": kes,
            "temperatures": temps,
            "volumes": vols,
            "n_data_points": len(steps),
        }
