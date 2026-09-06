# NRR Catalysis Pipeline (Nitrogen Reduction Reaction)

NRR converts N₂ + 6H⁺ + 6e⁻ → 2NH₃ (acid) or N₂ + 6H₂O + 6e⁻ → 2NH₃ + 6OH⁻ (base).

NRR is **extremely difficult** — industrial Haber-Bosch runs at 400-500°C and 150-300 atm.
Electrochemical NRR is far less efficient than HER; Faradaic efficiency is usually < 10%.
The N≡N triple bond (945 kJ/mol) is the root cause.

## Reaction mechanism

Three broad mechanisms:

### 1. Dissociative (Haber-Bosch-like)
N₂ + 2* → 2 *N          (rate-limiting N≡N cleavage)
*N + H⁺ + e⁻ → *NH
*NH + H⁺ + e⁻ → *NH₂
*NH₂ + H⁺ + e⁻ → *NH₃ + *
... (repeat for second *N)
Total: N₂ + 6H⁺ + 6e⁻ → 2NH₃.

Dissociative pathway requires very high temperatures / pressures (Haber-Bosch).
Not electrochemically favorable under ambient conditions.

### 2. Associative (Distal pathway, most common in electrochemistry)
N₂ + * → *N₂               (weak adsorption, often rate-limiting)
*N₂ + H⁺ + e⁻ → *N₂H       (PDS for most catalysts)
*N₂H + H⁺ + e⁻ → *NHNH
*NHNH + H⁺ + e⁻ → *NHNH₂
*NHNH₂ + H⁺ + e⁻ → *NH₂NH₂
*NH₂NH₂ + H⁺ + e⁻ → *NH₂ + NH₃   (first NH₃ released)
*NH₂ + 2H⁺ + 2e⁻ → NH₃ + *
Total: N₂ + 6H⁺ + 6e⁻ → 2NH₃.

Distal means the far N gets protonated first; proximal N stays on surface.

### 3. Associative (Alternating pathway)
N₂ + * → *N₂
*N₂ + H⁺ + e⁻ → *N₂H
*N₂H + H⁺ + e⁻ → *NHNH
*NHNH + H⁺ + e⁻ → *NHNH₂
*NHNH₂ + H⁺ + e⁻ → *NH₂NH₂
*NH₂NH₂ + H⁺ + e⁻ → *NH₃NH₃ (protonated hydrazine)
*NH₃NH₃ → 2NH₃ + *
Both N atoms protonated alternately; releases NH₃ simultaneously.

## Key descriptors

### N₂ binding energy (ΔG_*N₂)
- Most surfaces bind N₂ very weakly (physisorption, ΔG_*N₂ > 0 eV)
- Strong-binding surfaces (Fe, Mo) cleave N₂ → dissociative pathway
- Weak-binding surfaces (Au, Ag) can't activate N₂ → high overpotential
- Li-mediated NRR (Li + N₂ → Li₃N → NH₃) bypasses this via chemical activation

### *N₂H formation energy (ΔG_*N₂H)
- ΔG_*N₂H is the universal NRR descriptor for associative pathway
- ΔG_*N₂H > 0.5 eV: rate-limiting, high overpotential (most catalysts)
- ΔG_*N₂H ≈ 0 eV: ideal (no known ambient catalyst achieves this)
- Scaling: ΔG_*N₂H ≈ 0.5 × ΔG_*N + const, hard to break

### NHₓ binding energies
- ΔG_*NH, ΔG_*NH₂, ΔG_*NH₃ scale with ΔG_*N (slope ≈ 1, different intercepts)
- ΔG_*NH₃ < -0.4 eV: NH₃ poisons surface (can't desorb)
- ΔG_*NH₃ > -0.2 eV: NH₃ desorbs (good for Faradaic efficiency)

## Limiting potential

$$U_L = -\frac{\max_i \Delta G_i}{n_i e}$$

For associative distal on most metals, PDS is *N₂H formation:
$$\eta_{NRR} \approx \frac{\Delta G_{*N_2H}}{e}$$

For *N₂H ≈ 0.5 eV → η ≈ 0.5 V (typical for Fe, Mo).
For *N₂H ≈ 1.0 eV → η ≈ 1.0 V (typical for Au, Ag).

NH₃ equilibrium potential: E° = +0.092 V vs RHE (pH 0, 298 K).

## Competing HER

HER and NRR share potential window. HER is kinetically favored (2e⁻, fast).
On most surfaces, HER current density is 100× NRR current density.

Strategies to suppress HER:
- **Lithium-mediated**: Li + N₂ → Li₃N at non-aqueous electrolyte, then protonate
  to NH₃. HER suppressed because no aqueous H⁺ until last step.
- **High overpotential + proton-blocking layer**: e.g., MOF pores that filter H⁺
- **Surface poisoning**: O or OH blocks HER active sites
- **Hydrophobic electrolyte**: organic solvent, ionic liquid

## DFT computation

Use CHE for thermodynamic PDS:

For each intermediate (*N₂, *N₂H, *NHNH, *NHNH₂, *NH₂NH₂, *NH₂, *NH₃, *N):
$$G_i = E_{DFT}(i) + ZPE_i - T S_i$$

Use experimental N₂, NH₃ references (gas phase at 298 K):
- G_N₂ = E_DFT(N₂) + 0.15 eV (ZPE - TS correction)
- G_NH₃ = E_DFT(NH₃) + 0.10 eV (approximate)

For the 6-step distal pathway:
- ΔG₁ = G(*N₂H) - G(*) - G(N₂) - (G(H⁺) + G(e⁻))
- ΔG₂ = G(*NHNH) - G(*N₂H) - (G(H⁺) + G(e⁻))
- ...

Using CHE: G(H⁺) + G(e⁻) = 0.5 × G(H₂) at U = 0 V vs RHE.

Approximate ZPE/TS (eV, 298 K):
| Species | ZPE | TS |
|---|---|---|
| *N₂ | 0.16 | 0.05 |
| *N₂H | 0.31 | 0.06 |
| *NHNH | 0.45 | 0.07 |
| *NH₂NH₂ | 0.62 | 0.08 |
| *NH₂ | 0.34 | 0.05 |
| *NH₃ | 0.49 | 0.07 |
| N₂ (g) | 0.15 | 0.59 |
| NH₃ (g) | 0.90 | 0.95 |

## Recommended workflow

1. **Build slab** for catalyst surface (Fe(111), Mo(110), Ru(0001), single-atom catalysts).
2. **N₂ adsorption**: test end-on (η¹) vs side-on (η²) configurations.
3. **Relax with each intermediate** (*N₂, *N₂H, *NHNH, ..., *NH₃, *N).
4. **Frequency** (Γ-only) for adsorbed species.
5. **Compute ΔG_i** for each step via CHE.
6. **Identify PDS** (usually *N₂H formation for associative).
7. **Check HER competition**: compare η_NRR vs η_HER on same surface.
8. **Volcano plot** vs. ΔG_*N₂H; peak near ΔG_*N₂H ≈ 0 eV (no known catalyst at peak).

## Common pitfalls

- **N₂ adsorption is too weak**: on most surfaces, *N₂ doesn't bind at all in DFT.
  Report ΔG_*N₂ > 0 eV; this is physical, not a bug.
- **Spin state**: many NRR-active surfaces (Fe, Co, Mo) are magnetic. Run spin-polarized.
- **Solvent**: implicit solvation (VASPsol) essential for NRR — N₂H is polar.
- **DFT functional**: PBE often under-binds N₂H; RPBE/SCAN may be better. Use
  consistent functional across catalyst series.
- **Coverage**: *N₂H coverage affects subsequent step. Run at 1/4 ML or 1/9 ML.
- **NH₃ detection artifacts**: experimental NRR papers often report NH₃ from
  contamination (lab air, N in catalyst support). Always check blank runs.
- **Dissociative vs associative**: don't conflate. Fe/Mo at high T → dissociative.
  Ambient electrochemistry → associative (distal or alternating).

## Interpretation table

| ΔG_*N₂H (eV) | η_NRR (V) | NRR activity | Example |
|---|---|---|---|
| < 0.3 | < 0.3 | **excellent** (theoretical) | none known |
| 0.3–0.6 | 0.3–0.6 | good | Fe(111) in some studies, Ru surfaces |
| 0.6–1.0 | 0.6–1.0 | weak | Mo, W surfaces |
| > 1.0 | > 1.0 | HER dominates entirely | Au, Ag, Cu |

## For huginn

When user mentions NRR / nitrogen reduction / ammonia synthesis / electrochemical NH₃:
1. Identify mechanism (dissociative vs associative-distal vs associative-alternating).
2. Compute ΔG_*N₂ (often > 0 eV — note in report).
3. Compute ΔG_*N₂H, ΔG_*NHNH, ..., ΔG_*NH₃, ΔG_*N.
4. Identify PDS (usually *N₂H formation).
5. Compute η_NRR; compare to η_HER on same surface (HER often wins).
6. If user wants Faradaic efficiency, must include kinetic model (not just CHE).
7. Consider Li-mediated NRR as alternative pathway if aqueous NRR fails.
8. Note: ambient electrochemical NRR is a hot research area with many artifacts.
   Be skeptical of reports claiming FE > 30% at low overpotential.
