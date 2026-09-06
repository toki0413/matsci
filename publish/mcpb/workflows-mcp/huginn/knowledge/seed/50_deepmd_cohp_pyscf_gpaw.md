# DeepMD, LOBSTER/COHP, PySCF, GPAW — Quick Reference

These four codes fill gaps not covered by VASP/QE/CP2K seeds:
- **DeepMD-kit**: train neural-network potentials for LAMMPS-scale MD
- **LOBSTER**: extract COHP/ICOHP/COOP bonding analysis from plane-wave DFT
- **PySCF**: Pythonic all-electron quantum chemistry (molecules + periodic)
- **GPAW**: PAW DFT with real-space / LCAO / PW modes, Python API

All four are open-source and pip-installable (or conda). Use them when
the mainline code (VASP/QE) is awkward or impossible.

---

## DeepMD-kit (NNP training → LAMMPS)

### When to use
- LAMMPS MD > 1 ns for a system where classical force fields fail
- Active learning across temperatures / compositions / phases
- Coupling with `dpdata` for DFT data management

### Minimum input (Se_2 descriptor, typical for alloys)

`water_se_a.json`:
```json
{
  "model": {
    "descriptor": {
      "type": "se_e2_a",
      "sel": [46, 92],
      "rcut_smth": 0.50,
      "rcut": 6.00,
      "neuron": [25, 50, 100],
      "resnet_dt": false,
      "axis_neuron": 16,
      "seed": 1
    },
    "fitting_net": {
      "neuron": [240, 240, 240],
      "resnet_dt": true,
      "seed": 1
    }
  },
  "learning_rate": { "type": "exp", "start_lr": 0.001, "stop_lr": 3.51e-8, "decay_steps": 5000 },
  "loss": { "type": "ener", "start_pref_e": 0.02, "limit_pref_e": 1, "start_pref_f": 1000, "limit_pref_f": 1, "start_pref_v": 0, "limit_pref_v": 0 },
  "training": { "training_data": { "systems": ["data/"], "batch_size": "auto" }, "numb_steps": 1000000, "disp_file": "lcurve.out", "disp_freq": 100, "save_freq": 1000 }
}
```

### CLI
```bash
dp train water_se_a.json          # train
dp freeze -o graph.pb             # extract frozen model
dp compress -i graph.pb -o graph-compress.pb   # optional, smaller model
dp test -m graph.pb -s data_test -d results    # RMSE on test set
```

### LAMMPS coupling
```lammps
pair_style deepmd graph.pb 10      # 10 = neighbor list refresh cutoff (Å)
pair_coeff * *                     # DeepMD reads type map from model
```

### `dpdata` for VASP → DeepMD conversion
```python
import dpdata
ds = dpdata.LabeledSystem("OUTCAR", fmt="vasp/outcar")
ds.to("deepmd/npy", "data/")       # one frame per subdir
```

### Pitfalls
- `sel` must cover max neighbors within `rcut`; undersized `sel` silently drops pairs
- Energies in eV, coordinates in Å — DeepMD uses LAMMPS units, not atomic units
- DFT data must be at the SAME k-point grid / ENCUT as production; mismatched Ecut breaks transferability
- Active learning (`dp run_param` + `model_devi`) flags high-uncertainty frames for re-DFT

---

## LOBSTER (COHP analysis)

### When to use
- Quantify bond strength (ICOHP < 0 means bonding, > 0 antibonding)
- Distinguish ionic vs covalent contributions
- Validate geometry against electronic structure (e.g. is this "bond" real?)

### Theory recap
- COHP (Crystal Orbital Hamilton Population) projects the band energy
  (Hamiltonian matrix element) onto atom pairs — gives energy-resolved
  bonding/antibonding character.
- ICOHP = integral of COHP up to E_F; summed over pairs → bond strength indicator.
- COOP is the analogous overlap-population version (older, less used now).

### Workflow
1. Run VASP SCF with these INCAR tags (mandatory):
   ```
   LWAVE = .TRUE.
   LCHARG = .FALSE.
   ISMEAR = 0          # must be Gauss/Methfessel for projection
   NSW = 0             # single point (or use the converged WAVECAR)
   PREC = Accurate
   ```
2. Provide `lobsterin`:
   ```
   COHPstartEnergy -15.0
   COHPendEnergy 5.0
   cohpGenerator from 0.1 to 3.0 orbitalwise
   basisSet pbeVaspFit2015   # match your POTCAR (PBE 54)
   saveCohpToFile all
   saveIcohpLayer all
   ```
3. Run: `lobster` (reads POSCAR/POTCAR/WAVECAR)
4. Outputs: `COHPCAR.lobster`, `ICOHPLIST.lobster`, `DOSCAR.lobster`, `lobsterout`

### Python post-processing (`lobsterpy`)
```python
from lobsterpy.cohp.analyze import Analysis
an = Analysis(POSCAR="POSCAR", COHPCAR="COHPCAR.lobster",
              ICOHPLIST="ICOHPLIST.lobster")
an.get_cohp_plot_dict()              # plot data
print(an.get_summary_dict())         # ICOHP per pair + bonding summary
```

### Pitfalls
- `basisSet` MUST match the POTCAR generation (PBE 52 vs 54); wrong basis → garbage COHP
- Plane-wave PAW projection is only reliable when basis functions are compact
  (LOBSTER uses Slater-type analytic basis); LOBSTER warns if projection % < 90%
- Surface calculations need enough vacuum (>12 Å) to avoid spurious image COHP
- For magnetic systems, run spin-polarized VASP, LOBSTER reads ISPIN=2

---

## PySCF (Pythonic all-electron QC)

### When to use
- Molecules needing all-electron accuracy (heavy atoms, core spectroscopy)
- Periodic systems where plane-wave PAW is insufficient (e.g. core-level shifts)
- Custom Hamiltonians or post-HF methods (CCSD(T), CASSCF)
- Embedding (QM/MM) with fine-grained Python control

### Modes
- `pyscf.gto` — molecular Gaussian basis
- `pyscf.pbc` — periodic (crystal) calculations with Gaussian basis + PW dual

### Molecular example (HF → MP2)
```python
from pyscf import gto, scf, mp
mol = gto.M(atom='H 0 0 0; F 0 0 1.1', basis='cc-pVTZ', verbose=4)
mf = scf.RHF(mol).run()
mp2 = mp.MP2(mf).run()
print(f"MP2 energy: {mp2.e_tot} Ha")
```

### Periodic example (diamond, Γ-point)
```python
from pyscf.pbc import gto as pgto, scf as pscf
cell = pgto.Cell()
cell.atom = 'C 0 0 0; C 0.25 0.25 0.25'
cell.a = '''3.57 0 0  0 3.57 0  0 0 3.57'''
cell.basis = 'gth-szv-milliam'
cell.pseudo = 'gth-pbe'
cell.build()
mf = pscf.KRHF(cell, kpts=cell.make_kpts([4,4,4])).run()
```

### Strengths
- All-electron basis (cc-pVnZ, def2, ANO-RCC); no pseudopotential ambiguity
- PySCF-AGP2 / PySCF-DMRG for strong correlation
- PySCF-DF (density fitting) makes MP2/CCSD affordable for ~30-atom cells

### Pitfalls
- No `EDIFF`-style auto-convergence; set `conv_tol=1e-9` explicitly for SCF
- Periodic BCC/FCC cells need explicit k-mesh; Γ-only often lies
- Memory blows up for CCSD on > 50 electrons; use frozen-core + local correlation

---

## GPAW (PAW DFT, real-space / LCAO / PW)

### When to use
- Need a Python-scriptable DFT workflow (no INCAR file)
- TDDFT, GW, BSE without VASP license
- Large cells where PAW + real-space grid scales linearly with N
- Quick prototyping of new functionals / pseudopotentials

### Install
```bash
pip install gpaw                # includes libPAW, libxc
gpaw install-data ~/gpaw-data   # download PAW datasets (one-time)
```

### Molecular example (H₂O)
```python
from ase import Atoms
from ase.optimize import BFGS
from gpaw import GPAW
h2o = Atoms('H2O', positions=[[0,0,-0.7],[0,0,0.7],[0,0,0]], cell=[8,8,8])
h2o.center()
h2o.calc = GPAW(mode='lcao', xc='PBE', basis='dzp', txt='h2o.txt')
BFGS(h2o).run(fmax=0.02)
print(h2o.get_potential_energy())
```

### Bulk example (Si band structure)
```python
from ase.build import bulk
from gpaw import GPAW
si = bulk('Si', 'diamond', a=5.43)
si.calc = GPAW(mode='pw', xc='PBE', kpts=[4,4,4], txt='si.txt')
si.get_potential_energy()
# Band structure along standard path
bs = si.cell.bandpath('LGXG', npoints=50)
si.calc = GPAW(mode='pw', xc='PBE', kpts={'path': bs.path, 'npoints': 50}, txt='si_bs.txt')
si.get_potential_energy()
bs.write('si_bs.json')   # plot with ase.dft.band_structure
```

### Modes summary

| Mode | Best for | Scaling | Note |
|---|---|---|---|
| `lcao` | Molecules, large aperiodic | O(N) | basis set basis; fewer k-points needed |
| `pw`   | Bulk, accurate | O(N log N) | uses FFT, similar to QE |
| `fd`   | Real-space grid, FD | O(N) | multigrid; great for large/defective cells |

### Strengths
- Native ASE integration — no format conversion friction
- TDDFT (LCAO/TDDFT), GW, BSE via `gpaw.response`
- PAW datasets are versioned and pip-installable (no POTCAR licensing issue)

### Pitfalls
- LCAO mode needs more k-points than PW for convergence of metals
- `mode='fd'` needs careful `h` grid spacing (0.15–0.20 Å for transition metals)
- Spin-orbit coupling requires `spinors=True` and special PAW setups
- Bader/ELF analysis uses ASE tools (`ase.utils.bader`), not built-in

---

## Cross-tool selection guide

| Need | Tool of choice | Alternative |
|---|---|---|
| Train NNP for >1 ns MD | DeepMD-kit | MACE (also Python, GPU) |
| Bond strength from existing VASP | LOBSTER + `lobsterpy` | `pymatgen.electronic_structure.cohp` |
| All-electron molecular QC | PySCF | NWChem / ORCA (seed 08) |
| Periodic all-electron / hybrid | PySCF.pbc | CRYSTAL (commercial) |
| Python-scriptable DFT | GPAW | ABINIT / Quantum ESPRESSO Python API |
| Core-level shifts | PySCF (ΔSCF all-electron) | VASP ICORELEVEL (PAW limited) |
| TDDFT / GW / BSE without license | GPAW.response | BerkeleyGW (standalone) |

## Sources
- DeepMD-kit: https://docs.deepmodeling.com/projects/deepmd/ (LGPL-3.0-or-later, citation: Wang et al., Comput. Phys. Commun. 228, 178 (2018))
- LOBSTER: https://www.cohp.de (academic free, citation: Dronskowski & Blochl, J. Phys. Chem. 97, 8617 (1993); Maintz et al., Angew. Chem. Int. Ed. 55, 2 (2016))
- lobsterpy: https://github.com/JaGeo/lobsterpy (MIT, Müller et al., J. Open Source Softw. 8, 5294 (2023))
- PySCF: https://pyscf.org (Apache-2.0, Sun et al., J. Chem. Phys. 153, 024109 (2020); Wiley Comput. Mol. Sci. 8, e1340 (2018))
- GPAW: https://gitlab.com/gpaw/gpaw (GPL-3.0-or-later, citation: Enkovaara et al., J. Phys.: Condens. Matter 22, 253202 (2010))
