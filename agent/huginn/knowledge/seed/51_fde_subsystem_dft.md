# Frozen Density Embedding (FDE) — Subsystem DFT Quick Reference

FDE lets you split a large system into an **active** fragment (high-level
treatment) and a **frozen** environment (lower-level density only), with
the two coupled through a non-additive embedding functional. It is the
canonical "multiscale by energy-functional decomposition" method, cleaner
than cluster models (no dangling bonds) and cleaner than QM/MM (no link
atoms). For huginn this matters whenever slab/adsorption calculations
exceed ~200 atoms or when mixing methods (PBE environment + hybrid/MP2
active) is desired.

---

## 1. Theoretical foundation

### Total energy decomposition

For a system partitioned into fragments A (active) and B (frozen environment):

```
E[ρ_A, ρ_B] = T_s[ρ_A] + T_s[ρ_B]                    # noninteracting kinetic energies
            + V_ext[ρ_A + ρ_B]                        # external (nuclei) potential
            + J[ρ_A + ρ_B]                            # Hartree
            + E_xc[ρ_A + ρ_B]                         # exchange-correlation
            + E_nad_T[ρ_A, ρ_B]                       # non-additive kinetic
            + E_nad_xc[ρ_A, ρ_B]                      # non-additive XC
```

where `E_nad_X[ρ_A, ρ_B] = E_X[ρ_A + ρ_B] - E_X[ρ_A] - E_X[ρ_B]`.

### Kohn-Sham equations for the active fragment

The active orbitals satisfy a modified KS equation with an embedding potential:

```
[-½∇² + V_eff^A(r)] φ_i^A(r) = ε_i^A φ_i^A(r)

V_eff^A(r) = V_ext^A(r) + V_H[ρ_A + ρ_B](r) + V_xc[ρ_A + ρ_B](r)
           + V_nad_T[ρ_A, ρ_B](r) + V_nad_xc[ρ_A, ρ_B](r)
```

The last two terms are the FDE signature — they are absent in supermolecular
KS-DFT and carry all information about the fragment partition.

### Original reference

Wesolowski & Warshel, *J. Phys. Chem.* **97**, 8050 (1993).
The non-additive kinetic energy `E_nad_T` is the only approximation beyond
standard KS-DFT; common approximations include:

| Functional | Form | Accuracy |
|---|---|---|
| `TF` (Thomas-Fermi) | `E_nad_T ≈ C_F ∫ [(ρ_A+ρ_B)^(5/3) - ρ_A^(5/3) - ρ_B^(5/3)] dr` | Poor for bonds |
| `vW` (von Weizsäcker) | Gradient correction to TF | Better for covalent |
| `LLP` (Lee-Lee-Parr) | `c_LLP ∫ ρ^(5/3) (1 - ½(∇ρ)²/(ρ^(8/3)))` | Standard choice |
| `PW91k` | Perdew-Wang 91 kinetic form | Most common in production |
| `PBEk` | PBE-style kinetic | Better for H-bonds |

### Freeze-and-thaw (polarization)

If both fragments are allowed to relax, FDE becomes **freeze-and-thaw**
(cyclic optimization):
1. Freeze B, optimize A
2. Freeze A, optimize B
3. Repeat until both densities self-consistent

Cost: ~2× single-side FDE, gives true subsystem DFT minimum.

---

## 2. When to use FDE (and when NOT to)

### Use FDE when

- **Active site in a large slab** — adsorbate + first-surface-shell active,
  rest of slab frozen. Saves 5–10× vs full slab relaxation.
- **Solvation by explicit molecules** — solute active, first-shell waters
  frozen at PCM/DFT density. More accurate than pure PCM for H-bonding.
- **Defect in a perfect crystal** — defect + nearest neighbors active,
  rest frozen at bulk DFT density.
- **Mixed-level multiscale** — environment at LDA/GGA, active at hybrid
  (HSE06) or post-HF (MP2, CCSD). FDE provides the coupling energy
  consistently.
- **Conformational sampling of a local motif** — freeze the framework,
  scan only the active torsions.

### Do NOT use FDE when

- Charge transfer between A and B is significant (>0.1 e) — FDE assumes
  densities add, no orbital mixing across fragments.
- Strong covalent bonds cross the A/B boundary — `E_nad_T` fails there.
- You need excitation spectra that delocalize across the whole system.
- The system is small enough that supermolecular DFT is affordable.

---

## 3. Software stack

| Code | FDE mode | Strength | License |
|---|---|---|---|
| **PySCF** + `pyfde` | Molecular + periodic (via `pyscf.pbc`) | Pythonic, all-electron, post-HF active region | Apache-2.0 |
| **ADF** (BAND) | `BAND` FDE module | Mature, parallel, full doc | Commercial (free academic) |
| **CP2K** | Quickstep FDE | Mixed Gaussian/PW, periodic | GPL-2.0 |
| **Molcas** | OpenMolcas FDE | Multiconfigurational active (CASPT2 active) | LGPL-3.0 |
| **PyFrag 2019** | ADF wrapper | Analysis & visualization of FDE results | Open |
| **QE-FDE** | Quantum ESPRESSO fork | Periodic plane-wave FDE | GPL (research only) |

### PySCF `pyfde` minimal example

```python
from pyscf import gto, scf
from pyfde import FDE

# Active: H2O molecule (high-level: MP2)
mol_A = gto.M(atom='O 0 0 0; H 0 0 0.96; H 0.91 0 -0.24', basis='cc-pVTZ')

# Environment: another H2O at frozen density (from prior DFT)
mol_B = gto.M(atom='O 3 0 0; H 3 0 0.96; H 3.91 0 -0.24', basis='cc-pVTZ')
mf_B = scf.RHF(mol_B).run()             # frozen density

# FDE: solve A in the embedding potential of B
fde = FDE(mol_A, mol_B, rho_B=mf_B.make_rdm1())
mf_A = fde.kernel()                     # returns embedded KS solver

# Upgrade active to MP2 (post-HF on embedded orbitals)
from pyscf import mp
mp2 = mp.MP2(mf_A).run()
print(f"FDE-MP2 energy: {mp2.e_tot} Ha")
```

### ADF / BAND FDE input (key block)

```
FDE
  # Frozen fragment from a previous calculation
  Fragments
    B  file=water_B.t21  type=FDE  subcharge=0.0  relax=0
  End
  # Kinetic-energy functional (see table above)
  KinEnergy  PW91k
  # Allow charge transfer? (default no)
  AllowChargeTransfer  No
  # Use freeze-and-thaw? (default no, single-side FDE)
  FreezeAndThaw  No
End
```

---

## 4. Practical pitfalls

1. **`E_nad_T` is the dominant error source** — TF/vW badly underestimate
   kinetic non-additivity for covalent bonds. Use `PW91k` or `PBEk` as default.
2. **Frozen density must come from a compatible method** — if ρ_B is from
   LDA but the active region uses B3LYP, the XC non-additive term is
   inconsistent. Either (a) use the same XC functional for both, or (b)
   document the inconsistency.
3. **Frozen density is geometry-specific** — moving the environment atoms
   invalidates ρ_B; either recompute or use freeze-and-thaw.
4. **Basis set superposition error (BSSE)** — FDE does not automatically
   correct BSSE; use Boys-Bernardi counterpoise on the active fragment.
5. **Charge transfer is forbidden by construction** — if you suspect CT,
   check the embedded density vs the isolated density. Differences > 0.05 e
   in the active region suggest the partition is unphysical.
6. **PBC + FDE is finicky** — most plane-wave FDE implementations need
   isolated fragments in large cells; use `pyscf.pbc` or ADF-BAND for true
   periodic FDE.
7. **Frozen density of a metal is ill-defined** — metal densities fluctuate
   with k-mesh/Smearing; FDE for metals often requires freeze-and-thaw to
   converge.

---

## 5. How huginn should use FDE

### Trigger pattern (auto-suggest FDE when)

- User mentions "adsorption on a slab" + slab atoms > 100
- User wants post-HF (MP2/CCSD) on a periodic system
- User wants solvation by explicit waters + solute at high level
- User mentions "defect in crystal" + supercell > 200 atoms

### Workflow template

1. Compute ρ_B (environment) at GGA level, save density (`CHGCAR`, `.wfn`,
   or PySCF `make_rdm1()`).
2. Build active fragment (adsorbate + first-shell atoms, defect + nearest
   neighbors).
3. Choose kinetic functional (`PW91k` default; `PBEk` for H-bonds).
4. Run single-side FDE first; check charge transfer & density change.
5. If CT < 0.05 e and density stable, escalate active region to
   hybrid/MP2/CCSD on embedded orbitals.
6. If CT > 0.1 e or density unstable, switch to freeze-and-thaw or
   abandon FDE in favor of full supermolecular DFT.

### Huginn-aware choice (vs other multiscale methods)

| Need | Method | Why |
|---|---|---|
| Covalent bond crosses boundary | **QM/MM** (link atoms) | FDE's E_nad_T fails for covalent cuts |
| Ionic + ionic | **FDE** | No covalent cut, density additivity holds |
| Surface + adsorbate | **FDE** (active = adsorbate+1st layer) | Avoids cluster model artifacts |
| Solvent + solute (H-bonding) | **FDE** with `PBEk` | Better than PCM for directed H-bonds |
| Metal surface | **QM/MM** or full DFT | Metal FDE is finicky (see pitfall 7) |
| Defect + bulk | **FDE** (active = defect+NN) | Avoids supercell periodic images |
| Reactive dynamics | **QM/MM** | FDE freeze breaks along reaction coordinate |

---

## 6. Cross-references inside this seed library

- **Seed 02 / 03 / 04**: VASP/QE/CP2K — FDE is *not* native to VASP; use
  CP2K Quickstep or QE-FDE fork
- **Seed 27 / 45-48**: catalysis — for slabs > 100 atoms, consider FDE on
  adsorbate + first-surface-shell
- **Seed 50**: PySCF — use `pyfde` for Python-driven FDE, post-HF on
  embedded orbitals
- **Seed 11**: MLPs — FDE is the conceptual opposite of NNP training
  (FDE partitions by fragment, NNP interpolates globally)

## Sources

- Wesolowski & Warshel, *J. Phys. Chem.* **97**, 8050 (1993) — original FDE
- Wesolowski et al., *J. Chem. Phys.* **143**, 134109 (2015) — FDE perspective
- Jacob & Neugebauer, *WIREs Comput. Mol. Sci.* **4**, 325 (2014) — subsystem DFT review
- PySCF FDE: https://pyscf.org/user/fde.html (Sun et al.)
- ADF FDE: https://www.scm.com/doc/TaskMenu/Frozen_Density_Embedding.html
- CP2K FDE: https://manual.cp2k.org/trunk/methods/fde.html
- Hapuarachchi et al., *J. Chem. Theory Comput.* **17**, 6597 (2021) — `PW91k` benchmark
