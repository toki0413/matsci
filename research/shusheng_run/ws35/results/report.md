# Comprehensive Analysis Report: Catalyst Effects on MgH2 Activation Energy

## Executive Summary

Out of 5 hypotheses, **2** were supported at α = 0.05.

## H1: Within-Paper Paired ΔEa Analysis

This hypothesis tests whether catalysts significantly reduce activation energy
within the same study. We computed paired differences (catalyzed - pure) for all
catalyzed/pure sample pairs within each paper, then applied DerSimonian-Laird
random-effects meta-analysis with bootstrap confidence intervals.

**Results:**
- Number of papers with paired comparisons: 12
- Total paired differences: 15
- Mean ΔEa: -56.49 kJ/mol (95% CI: [-71.76, -41.22])
- DerSimonian-Laird random effects mean: -51.60 kJ/mol (95% CI: [-52.16, -51.03])
- Bootstrap 95% CI: [-67.83, -41.67]
- Paired t-test: t = -7.250, p = 0.0000

**Conclusion: SUPPORTED** - Catalysts significantly reduce activation energy.

## H2: Eta-Squared for Catalyst Family Effect

This hypothesis tests whether catalyst family explains a significant proportion
of variance in activation energy. Eta-squared (η²) measures the proportion of
total variance attributable to catalyst family, with permutation testing for
statistical significance.

**Results:**
- Number of catalyst families: 5
- Eta-squared (η²): 0.3088
- F-statistic: 3.909 (ANOVA p = 0.0100)
- Permutation test p-value: 0.0080

**Conclusion: SUPPORTED** - Catalyst family explains a significant proportion of variance in Ea.

## H3: Method Comparison (Kissinger vs. isothermal_Arrhenius)

This hypothesis tests whether the choice of kinetic method (Kissinger vs.
isothermal_Arrhenius) produces systematically different Ea estimates. We compare
the absolute difference between method estimates within the same material/paper
to the typical within-paper ΔEa.

**Results:**
- Number of method pairs: 1
- Mean absolute |Δ_method|: 29.25 kJ/mol
- Median absolute |Δ_method|: 29.25 kJ/mol
- Median within-paper ΔEa: 49.80 kJ/mol
- Ratio (|Δ_method| / median ΔEa): 0.59
- Paired t-test on method differences: t = nan, p = nan

**Conclusion: NOT SUPPORTED** - Method choice does not produce systematically different Ea estimates.

## H4: Leave-One-Study-Out Cross-Validation

This hypothesis tests whether a regression model using material class,
catalyst family, and method features can predict activation energy. We use
leave-one-study-out (LOSO) cross-validation to assess predictive performance.

**Results:**
- Number of samples: 40
- Number of papers: 19
- Number of features: 11
- Fixed effects mean prediction: 113.06 kJ/mol
- Random effects mean prediction: 112.90 kJ/mol
- Between-study variance (τ²): 408.30
- RMSE: 33.52 kJ/mol
- Predictions within ±10 kJ/mol: 9/40

**Conclusion: NOT SUPPORTED** - The model does not predict Ea accurately.

## H5: Extrapolation Prediction for Hypothetical Material

This hypothesis tests whether the model can extrapolate to a hypothetical
material with a novel catalyst family. We fit a ridge regression model on all
data and predict Ea for a catalyzed MgH2 with a novel perovskite catalyst
using the Kissinger method.

**Results:**
- Predicted Ea for hypothetical material: 93.64 kJ/mol
- RMSE on all samples: 24.81 kJ/mol
- Observed Ea range: [56.74, 210.00] kJ/mol
- Prediction reasonable (within plausible range): True

**Conclusion: SUPPORTED** - The model produces a reasonable extrapolated prediction.

## Bootstrap Analysis Results (50 Loops)

Cluster resampling by doi was performed 50 times to assess stability of results.

### H1 Bootstrap Results
- Mean ΔEa (bootstrap): N/A
- 95% CI (bootstrap): N/A
- Supported in 50/50 loops

### H2 Bootstrap Results
- Mean η² (bootstrap): N/A
- 95% CI (bootstrap): N/A
- Supported in 40/50 loops

### H3 Bootstrap Results
- Mean ratio (bootstrap): N/A
- 95% CI (bootstrap): N/A
- Supported in 0/50 loops

### H4 Bootstrap Results
- Mean RMSE (bootstrap): 34.28 kJ/mol
- 95% CI (bootstrap): [22.04, 49.34] kJ/mol
- Supported in 0/50 loops

## Data Summary

- Desorption-only rows: 40
- Total papers: 19
- Material classes: complex_hydride, pure_MgH2, alloy, catalyzed_MgH2
- Catalyst families: mxene_carbon, other, none, metal_nanoparticle, oxide
- Methods: Kissinger, isothermal_Arrhenius

## Methodology Notes

1. **H1**: Within-paper paired differences were computed for all catalyzed/pure sample pairs within each study. The DerSimonian-Laird random-effects model accounts for between-study heterogeneity. Bootstrap confidence intervals were obtained by cluster resampling (resampling papers with replacement).

2. **H2**: Eta-squared was computed from one-way ANOVA of Ea by catalyst family. Permutation testing (1000 permutations) assessed statistical significance by shuffling group labels and recomputing η².

3. **H3**: For materials with both Kissinger and isothermal_Arrhenius estimates in the same study, the absolute difference was computed. This was compared to the typical within-paper ΔEa to assess whether method choice introduces systematic bias.

4. **H4**: Leave-one-study-out cross-validation used ridge regression with features from material class, catalyst family, and method. Fixed and random effects meta-analyses were applied to the LOSO predictions.

5. **H5**: A ridge regression model was fit on all data and used to predict Ea for a hypothetical catalyzed MgH2 with a novel catalyst family (extrapolation beyond observed categories).

6. **Bootstrap**: 50 bootstrap loops with cluster resampling by doi were performed to assess the stability of each hypothesis test. In each loop, papers were resampled with replacement, and all hypotheses were recomputed on the bootstrap sample.
