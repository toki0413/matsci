# OER Catalysis Pipeline (Oxygen Evolution Reaction)

OER is the anodic half-reaction of water splitting: 2H₂O → O₂ + 4H⁺ + 4e⁻
(acid) or 4OH⁻ → O₂ + 2H₂O + 4e⁻ (base).

OER is the bottleneck of water splitting — it's a 4-electron process with
slow kinetics, while HER is 2-electron. Most OER catalysts need η > 0.3 V.

## Reaction mechanism (4-electron, adsorbates evolving)

Acid, generic M-based catalyst:

1. H₂O + * → *OH + H⁺ + e⁻           (ΔG₁)
2. *OH → *O + H⁺ + e⁻                 (ΔG₂)
3. H₂O + *O → *OOH + H⁺ + e⁻         (ΔG₃)
4. *OOH → O₂ + * + H⁺ + e⁻            (ΔG₄)

Sum: 2H₂O → O₂ + 4H⁺ + 4e⁻. Thermodynamic minimum: 1.23 V (at pH 0, 298 K).

The *OOH intermediate is the hardest to stabilize without over-binding *OH.
Scaling relation: ΔG_*OOH ≈ ΔG_*OH + 3.2 eV (±0.2 eV across catalysts).
This scaling sets the **theoretical minimum OER overpotential ≈ 0.37 V**.

## Descriptors and overpotential

$$\eta_{OER} = \frac{\max(\Delta G_1, \Delta G_2, \Delta G_3, \Delta G_4) - 1.23\,\text{eV}}{e}$$

The maximum-ΔG step is the **potential-determining step (PDS)**.

Ideal catalyst: ΔG₁ = ΔG₂ = ΔG₃ = ΔG₄ = 1.23 eV (zero overpotential).
Practical catalyst: max(ΔGᵢ) > 1.23 eV by at least ~0.37 V (scaling limit).

## Scaling relations (key ones)

- ΔG_*OOH vs ΔG_*OH: linear, slope ≈ 1, intercept ≈ 3.2 eV (universal)
- ΔG_*O vs ΔG_*OH: linear, slope ≈ 1, intercept ≈ 1.6 eV (varies more)
- These come from the bond-order conservation (H adsorbs similarly on *O and *OH)

**Breaking scaling relations** is a major research goal:
- Different adsorption sites for *OH vs *OOH
- Single-atom catalysts with different coordination
- Doping / strain to decouple the two

## pH correction

$$\Delta G_i(pH) = \Delta G_i(pH=0) - k_B T \ln 10 \cdot pH$$

(Each step releases 1 H⁺, so the free energy shifts by the proton chemical potential.)
At 298 K: -0.0592 eV per pH unit, per step.

But the overall OER potential also shifts by the same amount — the overpotential
**η_OER is pH-independent** at equilibrium (Nernst shift cancels out).

## DFT computation

Free energy of each adsorbate via CHE (Nørskov 2004, Man 2011):

$$G_{*OH} = E_{DFT}(*OH) + ZPE_{*OH} - T S_{*OH}$$
$$G_{*O} = E_{DFT}(*O) + ZPE_{*O} - T S_{*O}$$
$$G_{*OOH} = E_{DFT}(*OOH) + ZPE_{*OOH} - T S_{*OOH}$$
$$G_{O_2} = 4.92\,\text{eV} + 2 G_{H_2O} - 2 G_{H_2}$$

(Don't compute O₂ from DFT directly — triplet state, large error. Use experimental
O₂ free energy via water splitting equilibrium: ΔG_rxn = 4.92 eV at 298 K, pH 0.)

Approximate ZPE/TS corrections (in eV, 298 K):
| Species | ZPE | TS |
|---|---|---|
| *OH | 0.35 | 0.07 |
| *O | 0.05 | 0.02 |
| *OOH | 0.46 | 0.11 |
| H₂O (g) | 0.56 | 0.67 |
| H₂ (g) | 0.27 | 0.40 |

Net per-step correction: ~0.1–0.2 eV, not negligible.

## Recommended workflow

1. **Build slab** for catalyst surface (e.g. RuO₂(110), IrO₂(110), NiFe-LDH).
2. **Relax clean slab**, then relax slab + *OH, slab + *O, slab + *OOH at each
   relevant site (top, bridge, hollow, atop-metal).
3. **Frequency** (Γ-only) for each adsorbate to get ZPE (or use tabulated values).
4. **Apply ZPE/TS corrections** + use experimental G_O₂ = 4.92 eV.
5. **Compute ΔG₁–ΔG₄**, find PDS, compute η_OER.
6. **Volcano plot** vs. ΔG_*O (or ΔG_*OH) for catalyst series.

## Common pitfalls

- **O₂ reference**: never compute O₂ free energy from DFT. Use 4.92 eV.
- **Site dependence**: *OH, *O, *OOH prefer different sites on oxides. Test all.
- **Coverage / lateral repulsion**: at high *O coverage, ΔG shifts by 0.1–0.3 eV.
  Use 1/4 ML or 1/6 ML for screening.
- **Solvent**: implicit solvation (VASPsol) shifts OER ΔG by ~0.1–0.2 eV, often
  non-negligible. Use for quantitative comparison.
- **Hubbard U**: DFT+U on oxide d-states changes ΔG by ~0.2 eV. Use U = 3.3 eV
  for Ti 3d, 4.0 eV for Fe 3d, etc. (per Materials Project / OQMD conventions).
- **Spin state**: many OER catalysts are magnetic (Fe, Co, Ni, Mn). Spin-polarize
  and report the converged magnetic moment.

## Interpretation table

| η_OER (V) | OER activity | Example |
|---|---|---|
| < 0.3 | **excellent** (at scaling limit) | RuO₂(110), IrO₂(110) |
| 0.3–0.5 | good | NiFe-LDH, CoPi |
| 0.5–0.7 | moderate | Ni(OH)₂, Co₃O₄ |
| > 0.7 | weak | Fe₂O₃, TiO₂ |

Note: RuO₂/IrO₂ are the best OER catalysts but Ru/Ir are scarce. Mixed
Ni-Fe oxyhydroxides are the practical choice for alkaline electrolyzers.

## For huginn

When user mentions OER / oxygen evolution / water oxidation:
1. Compute ΔG_*OH, ΔG_*O, ΔG_*OOH via CHE.
2. Identify PDS (max-ΔG step).
3. Report η_OER; compare to RuO₂(110) = ~0.37 V.
4. Check scaling: is ΔG_*OOH - ΔG_*OH ≈ 3.2 eV? If far off, look for artifacts.
5. Volcano plot vs. ΔG_*O for series; peak at ΔG_*O ≈ 1.6 eV (Man 2011).
6. Note U-value, solvation, coverage in figure caption.
