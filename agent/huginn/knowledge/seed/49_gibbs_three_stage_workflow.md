# Gibbs Free Energy Three-Stage Workflow

> Standard computational thermochemistry pipeline: geo_opt → freq → gibbs_energy.
> Applies to molecules, adsorbates, surfaces, and bulk. The three-stage split
> ensures convergence quality and saves compute (freq is expensive, only run
> on a converged geometry).

## Why three stages

| Stage | Purpose | Output |
|---|---|---|
| 1. geo_opt | Find local minimum structure | Relaxed positions + E_DFT |
| 2. freq | Compute Hessian (2nd derivatives) at minimum | Vibrational frequencies, ZPE, TS |
| 3. gibbs | Combine E_DFT + ZPE + TS + thermal corrections | G(T, p) |

Skipping freq (using only E_DFT) is OK for screening but wrong for quantitative
thermodynamics. Skipping geo_opt (running freq on un-relaxed structure) gives
imaginary frequencies and useless ZPE.

## Stage 1: Geometry optimization

### Convergence criteria (force-based, not energy-based)

Use force convergence, not energy convergence. Forces detect unconverged
directions that energy doesn't.

VASP: `EDIFFG = -0.02` (eV/Å, max force on any atom < 0.02 eV/Å)
- Tighter: `EDIFFG = -0.01` for high-precision thermochemistry
- Looser: `EDIFFG = -0.05` for screening (acceptable if ZPE error budget is 0.05 eV)

QE: `conv_thr = 1.0d-8` (energy), `tstress = .true.`, `tprnfor = .true.`,
then check `bfgs` converged with force threshold `press_conv_thr` + `etot_conv_thr`.

CP2K: `MAX_FORCE 0.0003` a.u. ≈ 0.015 eV/Å; `RMS_FORCE 0.00015` a.u.
`MAX_DR 0.0015` a.u. ≈ 0.08 Å; `RMS_DR 0.001` a.u.

Gaussian: `Opt=(CalcFC,Tight)` or `Opt=(CalcFC,VeryTight)` for thermo.

### Energy convergence (SCF)

VASP: `EDIFF = 1e-7` (tighter than default 1e-4) — required for accurate forces.
QE: `conv_thr = 1.0d-8` (default 1e-6 is too loose for freq).
Gaussian: `SCF=(Conver=8)` or `SCF=(Conver=10)` for very tight.

### K-points and basis

- Molecules / Γ-only: single k-point (Γ) with large box (≥ 15 Å vacuum).
- Surface slabs: k-grid in-plane only (e.g. 4×4×1 for (2×2) slab).
- Bulk: full k-grid, converge to 1 meV/atom.

For freq, the same k-grid as geo_opt is usually fine. Tightening k-grid for freq
is expensive and rarely necessary.

## Stage 2: Frequency calculation

### Key parameter: which atoms to free

- **Molecule**: all atoms free. Box ≥ 15 Å to avoid image-image coupling.
- **Adsorbate on slab**: adsorbate atoms free, slab fully constrained
  (cheap, robust, "frozen slab" approximation).
- **Surface reaction path**: adsorbate + top 1-2 layers free, deeper layers fixed.
- **Bulk phonon**: all atoms free (but use DFPT, not finite-difference, for speed).

Freezing the slab reduces the Hessian from (3N × 3N) to (3n × 3n) where n is
the adsorbate atom count. For a 50-atom slab + 5-atom adsorbate, this is 15 × 15
instead of 150 × 150 — ~100× cheaper.

### Frequency settings (VASP)

```
IBRION = 5                  # finite differences, second derivative
POTIM = 0.015               # displacement step, Å (0.01-0.02 is safe)
NFREE = 2                   # central differences (2-point) or 4-point
PREC = Accurate             # required for freq
LREAL = .FALSE.             # reciprocal-space projectors (real-space fails for freq)
ADDGRID = .TRUE.            # finer grid for augmentation
NELM = 200                  # max SCF iterations (longer than default)
NCORE = 1                   # parallelization: 1 core per band-group (other modes corrupt freq)
```

For Γ-only freq: `KPOINTS` with single Γ point; use `vasp_gam` binary (faster).

### Frequency settings (QE)

```
&control
  calculation = 'scf'
  tstress = .true.
  tprnfor = .true.
/
&phonon
  ldisp = .true.            # dispersion (bulk) or .false. (Γ-only)
  nq1 = 1, nq2 = 1, nq3 = 1 # Γ only for slab/molecule
  fildyn = 'matdyn'
/
```

Or use `ph.x` separately after SCF.

### Frequency settings (Gaussian)

```
# B3LYP/def2TZVP Freq=(CalcFC,ScaleFC=0.99)
```
`CalcFC` analytically computes F-C constants (much faster than finite-difference
for molecules with hundreds of atoms).

### Low-frequency cutoff

Low-frequency modes (< 50 cm⁻¹) are often numerical noise, not real vibrations.
For adsorbates, freeze soft modes:

- Replace any ω_i < 50 cm⁻¹ with ω_i = 50 cm⁻¹ before computing ZPE and TS
  (standard practice in catalysis DFT community)
- This prevents divergent entropies (TS_vib ∝ ln(ω), diverges at ω→0)
- Reference: Isegren et al., J. Chem. Phys. 153, 124116 (2020)

For Gibbs free energy of adsorption, this is the single largest correction.

### Output parsing

Key outputs:
- `freq.dat` (VASP OUTCAR): list of frequencies in cm⁻¹
- Imaginary frequencies (negative in VASP; "f/i" in Gaussian): indicates saddle
  point, not minimum. Re-optimize with different starting geometry.
- ZPE = 0.5 × Σ_i hν_i (sum over all real modes)
- S_vib(T) = Σ_i [hν_i/(kT × (1 - e^{-hν_i/kT})) - k ln(1 - e^{-hν_i/kT})]
- For gases, also get S_rot, S_trans from molecular geometry + symmetry.

## Stage 3: Gibbs free energy

### General formula

$$G(T, p) = E_{DFT} + ZPE + \Delta H_{thermal}(T) - T S(T) + \Delta G_{pressure}(p)$$

Where:
- E_DFT from stage 1 (electronic energy)
- ZPE from stage 2 (zero-point vibrational energy)
- ΔH_thermal(T) = thermal correction to enthalpy (vib + rot + trans)
- T·S(T) = entropy term (vib + rot + trans)
- ΔG_pressure(p) = kT ln(p/p°), gas only (for standard pressure, = 0)

### Gas-phase molecule (ideal gas approximation)

$$G_{gas}(T, p) = E_{DFT} + ZPE + \Delta H_{thermal}(T) - T [S_{trans} + S_{rot} + S_{vib}] + kT \ln(p/p°)$$

For ideal gas at 298.15 K, 1 atm:
- S_trans = R[ln(V/N·(2πmkT/h²)^{3/2}) + 5/2]  (Sackur-Tetrode)
- S_rot = R[ln(8π²I_A·I_B·I_C·(kT)³/h⁶ / σ) + 3/2]  (nonlinear molecule)
  or R[ln(8π²IkT/σh²) + 1]  (linear)
- σ = symmetry number (e.g. H₂O = 2, CH₄ = 12, CO₂ = 2)

Common gas-phase values at 298.15 K, 1 atm:
| Gas | ZPE (eV) | TS (eV) | G_corr (= ZPE - TS) (eV) |
|---|---|---|---|
| H₂ | 0.273 | 0.403 | -0.130 |
| O₂ | 0.098 | 0.633 | -0.535 |
| N₂ | 0.149 | 0.591 | -0.442 |
| H₂O (g) | 0.560 | 0.670 | -0.110 |
| CO₂ | 0.310 | 0.680 | -0.370 |
| CH₄ | 1.115 | 0.660 | +0.455 |
| NH₃ | 0.900 | 0.580 | +0.320 |
| CO | 0.135 | 0.585 | -0.450 |

These are reference values; cite NIST-JANAF or ATcT when using in publication.

### Adsorbed species (harmonic approximation)

$$G_{ads}(T) = E_{DFT}(ads) + ZPE_{vib} + \Delta H_{vib}(T) - T S_{vib}(T)$$

(no translation or rotation; all modes treated as vibrations)

For adsorbed H on metal: usually 1 stretch mode (~1500–2000 cm⁻¹) + frustrated
translations/rotations (50–300 cm⁻¹). The frustrated modes are the ones affected
by the 50 cm⁻¹ cutoff.

### Bulk solid

Use phonon DOS from DFPT (QE `ph.x`) or finite-difference (VASP `IBRION=5` + supercell):

$$G_{bulk}(T) = E_{DFT} + \int_0^\infty d\omega \, g(\omega) \left[ \frac{\hbar\omega}{2} + kT \ln\left(1 - e^{-\hbar\omega/kT}\right) \right]$$

where g(ω) is the phonon density of states.

Tools: `phonopy` (VASP/CP2K), `phono3py` (anharmonic), `thermo_pw` (QE).

### Solvation correction

For species in solution, add ΔG_solv:
$$G_{solvated} = G_{gas} + \Delta G_{solv}$$

Methods:
- **Implicit (continuum)**: VASPsol, PCM, COSMO, SMD. ~0.1–0.5 eV for typical molecules.
- **Explicit**: include first solvation shell in DFT. More accurate, much more expensive.
- **Cluster-continuum**: explicit 1-3 waters + implicit bulk.

For catalysis ΔG (CHE), implicit solvation is usually sufficient.

## Common pitfalls

- **Imaginary frequencies**: if any ω_i < 0, structure is at a saddle point, not
  minimum. Re-optimize with symmetry broken (e.g. perturb atoms by 0.05 Å) and re-run freq.
- **Frozen-slab artifact**: if slab is too thin (≤ 2 layers), freq modes couple to
  slab deformation. Use ≥ 4 layers, freeze bottom 2.
- **Γ-only for surface**: yes, this is OK. Bulk phonons need full q-grid; surface
  phonons at Γ-only are sufficient for ZPE.
- **Pressure correction missing**: gas-phase G must include kT ln(p/p°) if you
  compare across pressures. Often hidden in tabulated values — check units.
- **Reference state confusion**: G°(298 K, 1 bar) vs G°(298 K, 1 atm) differ by
  ~3 J/mol·K × 298 K ≈ 0.001 eV — small but non-zero.
- **H₂ reference**: many catalysis workflows hard-code G(H₂) = E_DFT(H₂) - 0.13 eV
  (the ZPE - TS correction at 298 K). This is fine if you use 1 atm, 298 K. Update
  for other T/p.

## Recommended workflow template

```python
# Pseudo-code for huginn to follow
def gibbs_three_stage(species, structure, mode="adsorbed"):
    # Stage 1: geo_opt
    geo_opt_input = make_geo_opt_input(
        structure,
        ediffg=-0.02,    # 0.02 eV/Å force threshold
        ediff=1e-7,      # tight SCF
        prec="Accurate",
        ibrion=2,        # CG
        nsw=200,
    )
    relaxed = run_vasp(geo_opt_input)
    E_dft = relaxed.energy

    # Stage 2: freq
    freq_input = make_freq_input(
        relaxed.structure,
        ibrion=5,        # finite diff
        potim=0.015,
        nfree=2,
        ncore=1,         # critical
        lreal=False,     # critical
        kpoints="gamma" if mode == "adsorbed" else "auto",
        freeze_slab=(mode == "adsorbed"),
    )
    freq_result = run_vasp(freq_input)
    frequencies = freq_result.frequencies

    # Apply 50 cm^-1 cutoff (adsorbed only)
    if mode == "adsorbed":
        frequencies = [max(f, 50) for f in frequencies]

    # Stage 3: gibbs
    ZPE = 0.5 * sum(h * nu for nu in frequencies)
    S_vib = compute_s_vib(frequencies, T=298.15)
    G = E_dft + ZPE - T * S_vib
    if mode == "gas":
        G += gas_phase_corrections(relaxed.structure, T=298.15, p=1.0)
    return G
```

## For huginn

When user asks for "free energy" / "Gibbs" / "thermodynamics":
1. **Always** go through the three-stage pipeline. Don't skip freq.
2. Default: `EDIFFG = -0.02`, `EDIFF = 1e-7`, `IBRION = 5`, `POTIM = 0.015`,
   `NCORE = 1`, `LREAL = .FALSE.`, Γ-only for slabs/adsorbates.
3. For adsorbate, freeze slab (lower 2 layers). For molecule, free all atoms.
4. Apply 50 cm⁻¹ cutoff on low modes for adsorbate.
5. For gas-phase corrections, use the table above or NIST-JANAF.
6. Report: E_DFT, ZPE, TS, G(T, p) with the source of corrections.
7. Sanity check: imaginary frequencies → re-optimize. G_adsorbate typically
   within 0.3 eV of E_DFT for adsorbed species at 298 K.

## Sources

- Nørskov et al., J. Phys. Chem. B 108, 17886 (2004) — CHE method
- Isegren et al., J. Chem. Phys. 153, 124116 (2020) — low-freq cutoff justification
- NIST-JANAF Thermochemical Tables (public domain) — gas-phase references
- ATcT (Active Thermochemical Tables, Argonne) — high-accuracy gas-phase refs
