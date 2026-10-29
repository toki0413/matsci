# Analysis Report: Catalyst Effects on MgH2 Activation Energy

## Executive Summary

This analysis investigates the effect of catalysts on the activation energy (Ea) of MgH2 desorption using a dataset of desorption-only measurements from multiple research papers. Five hypotheses were tested:

### H1: Within-paper paired analysis
- Paired samples: 15 from 12 papers
- Fixed effects mean ΔEa: -56.49 kJ/mol (95% CI: [-74.51, -41.96])
- Random effects (DL) mean ΔEa: -48.56 kJ/mol
- Between-study heterogeneity (τ²): 0.00
- Paired t-test: t = -7.25, p = 0.000004
- **Conclusion**: SUPPORTED - Catalysts significantly reduce activation energy.

### H2: Eta-squared for catalyst family
- η² = 0.4721
- Permutation p-value = 0.0010
- **Conclusion**: SUPPORTED - Catalyst family explains a significant proportion of variance.

### H3: Method comparison
- 1 material-method pairs with both Kissinger and isothermal_Arrhenius estimates
  - MgH2-10 wt% H-V2O5: Kissinger=113.80, Isothermal=84.55, Δ=29.25

### H4: LOSO cross-validation
- 19 papers in cross-validation
- RMSE: 39.33 kJ/mol
- MAE: 29.38 kJ/mol
- R²: -0.1560

### H5: Extrapolation prediction
- Training samples: 41
- Predicted Ea for novel catalyzed_MgH2 (Kissinger): 94.71 kJ/mol

## Methodology

1. **Data preprocessing**: Ea values were parsed from strings (e.g., '109 ± 1' → 109). Absorption rows and one questionable entry were excluded.
2. **H1**: Within-paper paired differences (catalyzed – pure) were computed. DerSimonian-Laird random-effects model estimated pooled mean with between-study heterogeneity. Cluster bootstrap (50 loops, resampling papers with replacement) provided confidence intervals.
3. **H2**: Eta-squared from one-way ANOVA of Ea by catalyst family. Permutation test (1000 permutations) assessed significance by shuffling group labels.
4. **H3**: Materials with both Kissinger and isothermal_Arrhenius estimates within the same study were compared.
5. **H4**: Leave-one-study-out cross-validation with ridge regression using one-hot encoded features for material class, catalyst family, and method.
6. **H5**: Ridge regression model fitted on all data and used to predict Ea for a hypothetical catalyzed_MgH2 with a novel catalyst family.
7. **Bootstrap**: 50 cluster bootstrap loops were performed, resampling papers with replacement and recomputing all hypotheses.

## Data Summary

- Desorption-only rows: 41
- Material classes: alloy, catalyzed_MgH2, complex_hydride, pure_MgH2
- Catalyst families: metal_nanoparticle, mxene_carbon, none, other, oxide
- Methods: Kissinger, isothermal_Arrhenius
- Number of papers: 19
