# HER Catalysis Pipeline (Hydrogen Evolution Reaction)

HER is the cathodic half-reaction of water splitting: 2H⁺ + 2e⁻ → H₂.
In acid: 2H⁺ + 2e⁻ → H₂. In base: 2H₂O + 2e⁻ → H₂ + 2OH⁻.

## Reaction mechanism (Volmer-Heyrovsky / Volmer-Tafel)

Three elementary steps:

1. **Volmer** (proton adsorption): H⁺ + e⁻ + * → *H
2. **Heyrovsky** (electrochemical desorption): *H + H⁺ + e⁻ → H₂ + *
3. **Tafel** (recombination): 2 *H → H₂ + 2 *

Two pathways: Volmer-Heyrovsky or Volmer-Tafel. Rate-limiting step determines
Tafel slope (30 / 40 / 120 mV/dec depending on step + coverage).

## Sabatier principle and ΔG\*H descriptor

- Sabatier: optimal catalyst binds *H neither too weakly nor too strongly.
- **ΔG\*H** (free energy of H adsorption) is the universal HER descriptor:
  $$\Delta G_{*H} = G_{*H} - G_* - \frac{1}{2} G_{H_2}$$
- Ideal value: **ΔG\*H ≈ 0 eV** (zero overpotential).
- Tolerance: |ΔG\*H| < 0.2 eV is "active", |ΔG\*H| < 0.1 eV is "excellent".

Reference points:
- Pt(111): ΔG\*H ≈ -0.09 eV (slightly too strong, but best in practice)
- MoS₂ edge: ΔG\*H ≈ +0.08 eV (close to optimal, DFT)
- Ni(111): ΔG\*H ≈ -0.27 eV (too strong)
- Cu(111): ΔG\*H ≈ +0.34 eV (too weak)

## Overpotential

$$\eta_{HER} = \frac{|\Delta G_{*H}|}{e}$$

(at pH = 0; pH-corrected below.) Units: V. Smaller is better.

## pH correction (Nernst equation)

For H⁺/H₂ couple at non-zero pH:
$$E_{H^+/H_2}(pH) = E^0_{H^+/H_2} - \frac{k_B T \ln 10}{e} \cdot pH$$

At 298.15 K: $\frac{k_B T \ln 10}{e} = 0.0592$ V per pH unit (acidic → basic).

For HER overpotential at pH:
$$\eta_{HER}(pH) = \frac{|\Delta G_{*H} - \Delta G_{pH\,shift}|}{e}$$

where $\Delta G_{pH\,shift} = k_B T \ln 10 \cdot pH \approx 0.0592 \cdot pH$ eV
at 298.15 K (accounts for proton chemical potential shift).

For neutral pH (pH=7): shift ≈ 0.414 eV.
For pH=14: shift ≈ 0.829 eV.

## DFT computation of ΔG\*H

Compute free energies via the computational hydrogen electrode (CHE, Nørskov 2004):

$$G_{*H} = E_{DFT}(*H) + ZPE_{*H} - T S_{*H}$$
$$G_* = E_{DFT}(*) + ZPE_* - T S_*$$
$$\frac{1}{2} G_{H_2} = \frac{1}{2}\left(E_{DFT}(H_2) + ZPE_{H_2} - T S_{H_2}\right)$$

Approximations (good for screening):
- ZPE(*H) ≈ 0.20 eV (single H-S stretch mode ≈ 0.16–0.24 eV; tabulate per site)
- S(*H) ≈ 0 (adsorbed, low entropy)
- For gas-phase H₂ at 298 K: ZPE = 0.27 eV, TS = 0.40 eV → G_H₂ = E_DFT - 0.13 eV

Combined: **ΔG\*H ≈ ΔE_DFT + 0.20 - 0.5 × (-0.13) = ΔE_DFT + 0.265 eV**
(approximate, "rule of thumb" for screening).

## Recommended workflow

1. **Build clean slab** for the surface of interest (e.g. 4-layer slab, bottom 2 fixed).
2. **Relax clean slab** + **relax slab + H at top site** (try top/bridge/hollow).
3. **Frequency calc** for adsorbed H only (freeze slab; Γ-only vasp_gam OK).
4. **Apply ZPE/TS corrections** (use freq result for *H, gas-phase H₂ values from NIST).
5. **Compute ΔG\*H** and **η_HER**.
6. **Volcano plot** vs. ΔG\*H for series of catalysts.

## Common pitfalls

- **Site sampling**: H adsorbs at top/bridge/hollow differently on each surface.
  Test all; report the lowest-energy.
- **Coverage**: at high coverage, ΔG\*H shifts (H-H repulsion). Use 1/4 ML or 1/8 ML
  for low-coverage ΔG\*H consistent with Sabatier ideal.
- **Solvent**: implicit solvation (VASPsol) shifts ΔG\*H by ~0.05–0.1 eV. Optional
  for screening; required for quantitative comparison.
- **Functional**: PBE commonly used; RPBE gives weaker binding (~+0.1 eV). Consistent
  functional matters for ranking.
- **D3/vdW**: vdW correction shifts ΔG\*H by ~0.05 eV on metals, larger on 2D materials.

## Interpretation table

| ΔG\*H (eV) | η_HER (V) | HER activity | Example |
|---|---|---|---|
| < -0.4 | > 0.4 | very weak (H poisoned) | Au, Cu |
| -0.4 to -0.2 | 0.2–0.4 | weak | Ag, WC |
| -0.2 to +0.2 | < 0.2 | **good** (Sabatier peak) | Pt, MoS₂ edge, Ni₂P |
| +0.2 to +0.4 | 0.2–0.4 | weak | Ni(111), Co |
| > +0.4 | > 0.4 | very weak (H won't adsorb) | TiO₂, graphene |

## For huginn

When user mentions HER / hydrogen evolution / water splitting:
1. Compute ΔG\*H via CHE.
2. Report η_HER at pH=0 and pH=7 (default pHs).
3. Compare to Pt(111) = -0.09 eV reference.
4. Generate volcano plot if a series is given.
5. Always note functional + coverage + solvation in the figure caption.
