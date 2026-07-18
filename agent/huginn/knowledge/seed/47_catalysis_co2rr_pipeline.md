# CO₂RR Catalysis Pipeline (CO₂ Reduction Reaction)

CO₂RR converts CO₂ + H⁺ + e⁻ into C1/C2/C3 products:
- 2e⁻: CO or HCOOH
- 4e⁻: HCHO or C₂H₄ (via C-C coupling)
- 6e⁻: CH₃OH
- 8e⁻: CH₄
- 12e⁻: C₂H₄ (ethanol / ethylene route)

Competes with HER (same potential window); selectivity is the main challenge.

## Reaction mechanism (C1 pathway)

Acid, on Cu-based catalysts (typical):

1. CO₂ + * + H⁺ + e⁻ → *COOH            (ΔG₁, protonation of O)
2. *COOH → *CO + H₂O                     (ΔG₂, spontaneous)
3. *CO + H⁺ + e⁻ → *CHO or *COH          (ΔG₃, selectivity-determining)
4a. *CHO → ... → CH₄ (8e⁻ total)
4b. *CO + *CO → *OCCO → ... → C₂H₄ (C-C coupling, 12e⁻)

C1 vs C2 branching at step 3 determines product selectivity.

## Key descriptors

### CO binding energy (ΔG_*CO)
- ΔG_*CO < -0.6 eV: CO poisons surface (strong binding, no further reaction)
- ΔG_*CO ≈ -0.3 to -0.5 eV: C1/C2 products accessible (Cu sweet spot)
- ΔG_*CO > -0.2 eV: CO desorbs → only HCOOH or H₂ (HER dominates)

### COOH binding energy (ΔG_*COOH)
- Coupled to ΔG_*CO via scaling: ΔG_*COOH ≈ ΔG_*CO + 0.5 eV (approximate)
- Hard to tune independently

### CHO vs COH branching
- *CHO preferred on weak-CO-binding surfaces (Au, Ag)
- *COH preferred on strong-CO-binding surfaces (Cu, Pt)
- This determines downstream C1 vs C2 selectivity

## Limiting potential

$$U_L = -\frac{\max_i \Delta G_i}{n_i e}$$

where n_i is the electron count for step i. The most uphill step sets U_L.
η_CO₂RR = |U_L - E°_eq|, where E°_eq depends on product (CO: -0.11 V, CH₄: +0.17 V,
C₂H₄: +0.08 V vs RHE at pH 0).

## Scaling relations

| Pair | Slope | Intercept (eV) | Notes |
|---|---|---|---|
| ΔG_*COOH vs ΔG_*CO | ~0.5 | ~1.4 | universal on metals |
| ΔG_*O vs ΔG_*CO | ~1 | ~0.5 | varies by surface |
| ΔG_*CHO vs ΔG_*CO | ~1 | ~0.4 | C1 branching |
| ΔG_*COH vs ΔG_*CO | ~0.5 | ~0.8 | C1 branching |

Breaking scaling (e.g. via single-atom catalysts, doping) is the main design lever.

## C2 selectivity (C-C coupling)

On Cu (unique among metals), *CO coverage is high enough that *CO + *CO dimerization
is feasible. The dimer *OCCO then protonates to *OCCHO, leading to:
- C₂H₄ (ethylene, 12e⁻)
- C₂H₅OH (ethanol, 12e⁻)
- CH₃COOH (acetic acid, 8e⁻)

C2 branching happens on the surface via *OCCHO, *OCCOH intermediates.

## DFT computation

Use CHE, but with extra care because CO₂RR has multiple intermediates:

For each intermediate (*COOH, *CO, *CHO, *COH, *OCCO, *OCCHO, ...):
$$G_i = E_{DFT}(i) + ZPE_i - T S_i$$

Use experimental references for gas-phase products (CO, CH₄, C₂H₄, ...).
For solvation: implicit solvation (VASPsol) is often essential — CO₂RR involves
charged intermediates (*COOH, *COH) whose stabilization by water is significant.

Approximate ZPE/TS (eV, 298 K):
| Species | ZPE | TS |
|---|---|---|
| *CO | 0.12 | 0.05 |
| *COOH | 0.36 | 0.07 |
| *CHO | 0.28 | 0.05 |
| *COH | 0.30 | 0.05 |
| *OCCO | 0.24 | 0.08 |

## Recommended workflow

1. **Build slab** for catalyst surface (Cu(111), Cu(100), Cu(211) for stepped).
2. **Relax clean slab**, then relax + each intermediate (*COOH, *CO, *CHO, *COH).
3. **Test multiple sites** (atop, bridge, hollow) for each intermediate.
4. **Frequency** (Γ-only) for adsorbed species to get ZPE.
5. **Solvation** (VASPsol recommended for CO₂RR; charged intermediates).
6. **Compute ΔG_i** for each step; identify PDS.
7. **Compute U_L** and selectivity:
   - Compare ΔG_*CHO vs ΔG_*COH (C1 branching)
   - Check *CO coverage (C2 coupling needs > 1/4 ML on Cu)
8. **Volcano plot** vs. ΔG_*CO for series; peak near ΔG_*CO ≈ -0.4 eV (Cu).

## Common pitfalls

- **CO₂ binding is weak**: ΔG_*CO₂ ≈ 0 eV on most surfaces. Don't waste compute
  relaxing *CO₂; instead start from *COOH.
- **Proton source**: DFT can't easily model explicit proton transfer. CHE assumes
  PCET (proton-coupled electron transfer). For non-PCET steps, use grand-canonical DFT.
- **Coverage**: C2 selectivity needs *CO coverage > 1/4 ML. Run at relevant
  coverage; don't extrapolate from 1/16 ML.
- **Cu surface orientation**: Cu(100) favors C2; Cu(111) favors C1; Cu(211)
  (stepped) has different selectivity. Report surface in figure caption.
- **Solvent & electrolyte**: implicit solvation shifts ΔG by ~0.1–0.3 eV. Explicit
  cations (K⁺, Cs⁺) at outer Helmholtz plane can stabilize *COOH via field effects —
  this is not captured by VASPsol alone.
- **C-C coupling barrier**:CHE gives thermodynamic limiting potential, not kinetics.
  *CO + *CO → *OCCO has a 0.4–0.8 eV barrier on Cu. Use NEB if selectivity
  hinges on kinetics.

## Interpretation table

| ΔG_*CO (eV) | η_CO₂RR (V) | Main product | Example |
|---|---|---|---|
| < -0.6 | > 0.6 | CO poisoning (HER dominates) | Pt, Rh |
| -0.5 to -0.3 | 0.3–0.5 | **C2 (C₂H₄, EtOH)** | Cu(100), Cu(111) |
| -0.3 to -0.1 | 0.2–0.4 | **CO** (2e⁻ product) | Au, Ag |
| -0.1 to +0.1 | 0.4–0.6 | HCOOH (2e⁻) | Sn, Bi, Pb |
| > +0.1 | > 0.6 | HER dominates | W, Mo |

## For huginn

When user mentions CO₂RR / CO₂ reduction / electrochemical CO₂:
1. Identify target product (CO, HCOOH, CH₄, C₂H₄, ...).
2. Compute ΔG_*CO, ΔG_*COOH, ΔG_*CHO, ΔG_*COH.
3. Identify PDS for each product pathway.
4. Compute U_L; compare η to Cu reference (-0.4 eV ΔG_*CO, η ≈ 0.4 V).
5. Check C1 vs C2 branching via *CHO vs *COH.
6. If C2 is target, check *CO coverage (need > 1/4 ML).
7. Note: kinetic barriers (NEB) for C-C coupling often override thermodynamic prediction.
8. Report solvation model, coverage, surface orientation in figure caption.
