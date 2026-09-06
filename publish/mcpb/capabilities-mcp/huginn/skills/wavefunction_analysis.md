# Skill: Wavefunction Analysis

## Description
Perform advanced wavefunction analysis on molecular and periodic systems using quantum chemistry tools. Covers electron density analysis, orbital analysis, reactivity descriptors, and interaction visualization.

## Trigger Conditions
- User asks about "wavefunction analysis", "electron density", "molecular orbital"
- User mentions Multiwfn, Gaussian output analysis, ORCA post-processing
- User wants to predict reactivity sites, analyze aromaticity, or visualize weak interactions
- User asks about charge analysis (Mulliken, Hirshfeld, RESP, ADCH, NPA)

## Core Methods Reference

### 1. Reactivity Descriptors (Conceptual DFT)
- **Fukui function**: f+, f-, f0 for predicting electrophilic/nucleophilic/radical sites
  - f- significant positive → electrophilic attack site
  - f+ significant positive → nucleophilic attack site (less reliable)
  - Always use proper finite-difference with N/N±1 states, NOT orbital-freezing approximation
- **Dual descriptor**: Δf = f+ - f-
  - Positive → nucleophilic site; Negative → electrophilic site
  - More reliable than Fukui function alone, especially for nucleophilic sites
- **Condensed Fukui / Dual descriptor**: Atomic-level quantitative comparison
  - Use Hirshfeld charges for best accuracy in condensed forms
- **Local softness**: s± = S × f± — enables cross-molecule comparison
- **Electrophilicity / Nucleophilicity indices**: ω, N — global reactivity measures

### 2. Weak Interaction Analysis
- **IGMH (Independent Gradient Model based on Hirshfeld partition)**:
  - Strictly based on wavefunction (more rigorous than IGM)
  - Excellent for visualizing non-covalent interactions
  - δg_inter isosurface reveals interaction regions
  - Use grid-screening for large systems to reduce cost dramatically
- **NCI (Non-Covalent Interaction)**:
  - Based on electron density and its derivatives
  - RDG (reduced density gradient) isosurface colored by sign(λ₂)ρ
  - Positive sign(λ₂)ρ → repulsive; Near zero → van der Waals; Negative → attractive
- **mIGM**: Geometry-only fast approximation (no wavefunction needed)
  - Good for quick inspection; IGMH preferred when wavefunction available
- **amIGM**: Time-averaged mIGM for MD trajectories

### 3. Aromaticity Analysis
- **NICS (Nucleus-Independent Chemical Shift)**:
  - NICS(0): At ring center — affected by σ framework
  - NICS(1): 1 Å above ring — better π-electron indicator
  - NICS_ZZ: Only ZZ component of shielding tensor
  - Negative NICS → aromatic; Positive → antiaromatic
- **ICSS (Isosurface of Chemical Shielding)**:
  - 3D visualization of shielding/deshielding regions
  - ICSS_ZZ isosurface most informative for π systems
- **Magnetic induced current**: Ring current direction and strength
- **HOMA, Bird index**: Geometry-based aromaticity measures

### 4. Charge Analysis
- **Hirshfeld charge**: Fast, good for reactivity prediction, but basis-set sensitive
- **ADCH charge**: Corrects Hirshfeld dipole, better for ESP reproduction
- **RESP charge**: Standard for force field development (especially AMBER)
  - Requires two-stage fitting with hyperbolic restraints
- **NPA (Natural Population Analysis)**: Based on NBO, chemically intuitive
- **CM5 charge**: Fast correction to Hirshfeld for solvation free energies
- **Mulliken charge**: Never use for quantitative analysis (basis-set dependent, unphysical)

### 5. Excited State Analysis
- **Hole-electron analysis**: Comprehensive excited state characterization
  - Requires reference-state orbitals and CI coefficients
  - Gaussian: add IOp(9/40=4) to print all coefficients > 0.0001
  - ORCA: use TPrint keyword in %cis or %tddft block
- **NTO (Natural Transition Orbital)**: Simplifies excitation to dominant hole-electron pair
  - Fails if excitation cannot be described by single pair
- **Transition density / transition dipole moment density**: Real-space visualization

### 6. Electrostatic Potential (ESP) Analysis
- **ESP on molecular surface**: Colored vdW surface reveals reactive regions
  - Negative ESP → electrophilic attack site
  - Positive ESP → nucleophilic attack site
- **ALIE (Average Local Ionization Energy)**: Measures electron binding strength
  - Minima on vdW surface → sites vulnerable to electrophilic/radical attack
- **LEAE (Local Electron Attachment Energy)**: Measures electron affinity spatially
  - Negative LEAE → nucleophilic attack site (Lewis acidic regions)

## Software Workflows

### Multiwfn Workflow
1. Prepare wavefunction file: .fch (Gaussian), .molden (ORCA), .wfn, .wfx
2. Launch Multiwfn, load file
3. Select main function (e.g., 3 for ESP, 4 for orbital, 20 for weak interaction)
4. Follow sub-function menus; most analyses are interactive
5. Export cube files for visualization in VMD/GaussView

### Gaussian → Multiwfn Pipeline
```bash
# Step 1: Gaussian calculation
g16 input.gjf

# Step 2: Convert checkpoint to fch format
formchk input.chk input.fch

# Step 3: Multiwfn analysis
Multiwfn input.fch

# Step 4: Visualize in VMD
# Load structure + cube file, use "Volume" representation
```

### ORCA → Multiwfn Pipeline
```bash
# Step 1: ORCA calculation
orca input.inp > input.out

# Step 2: Convert to molden
orca_2mkl input -molden

# Step 3: Multiwfn analysis
Multiwfn input.molden
```

## Validation Checks
- **Physical reasonableness**: Charges should sum to system charge; ESP extrema should be chemically sensible
- **Convergence sensitivity**: Test with different grid spacings (0.1–0.25 Bohr)
- **Basis set effect**: Minimal basis → unphysical; 6-31G* usually sufficient for visualization; TZVP better for quantitative descriptors
- **Comparison**: Cross-check Fukui predictions with dual descriptor; validate with experimental or high-level theoretical data when possible

## Common Pitfalls
- Using orbital-freezing approximation for Fukui function (bad practice,审稿人会批评)
- Ignoring degeneracy in frontier orbitals (use orbital-weighted or degeneracy-corrected formulas)
- Applying gas-phase descriptors to solvent-sensitive reactions without caution
- Confusing "electrophilic reaction site" with "electrophilic atom" (opposite meanings in conventional usage)
