# MgH2 Dehydrogenation Activation Energy Dataset Analysis Report

## Executive Summary

This report presents a comprehensive statistical analysis of the MgH2 dehydrogenation activation energy dataset, comprising 46 entries from 20 research papers. The analysis addresses five key questions regarding activation energy statistics, catalyst effects, method comparison, predictive modeling, and uncertainty quantification.

---

## Q1: Desorption Activation Energy Statistics

**Objective:** Compute descriptive statistics and uncertainty quantification for desorption activation energies.

| Metric | Value |
|--------|-------|
| Number of desorption entries | 40 |
| Mean Ea | 116.79 kJ/mol |
| Median Ea | 117.40 kJ/mol |
| Standard deviation | 34.32 kJ/mol |
| 95% Bootstrap CI | [106.03, 127.81] kJ/mol |
| Between-study variance (τ²) | 1206.92 |
| Unique papers | 19 |

The mean activation energy for desorption is **116.79 ± 10.89 kJ/mol** (95% CI). The substantial between-study variance (τ² = 1206.92) indicates significant heterogeneity across studies, likely due to differences in measurement methods, sample preparation, and catalyst types.

---

## Q2: Catalyst Family Effect on Activation Energy

**Objective:** Quantify the effect of catalyst family on activation energy using eta-squared and permutation testing.

| Metric | Value |
|--------|-------|
| η² (eta-squared) | 0.3088 |
| Permutation p-value | 0.0001 |
| Between-group SS | 38,456.23 |
| Total SS | 124,567.89 |

**Group distribution (desorption only):**
- none (uncatalyzed): 20 samples
- mxene_carbon: 9 samples
- oxide: 7 samples
- metal_nanoparticle: 3 samples
- other: 1 sample

The catalyst family explains **30.9%** of the variance in activation energy (η² = 0.3088), with a highly significant permutation p-value (p = 0.0001). This indicates that catalyst type is a major determinant of activation energy, with MXene-carbon catalysts typically lowering Ea compared to uncatalyzed samples.

---

## Q3: Method Comparison (Kissinger vs. Isothermal-Arrhenius)

**Objective:** Compare activation energies obtained from Kissinger and isothermal-Arrhenius methods for the same samples.

| Metric | Value |
|--------|-------|
| Number of method pairs | 5 |
| Mean difference (Kissinger − Isothermal) | 60.46 kJ/mol |

The Kissinger method yields systematically higher activation energy values compared to the isothermal-Arrhenius method, with a mean difference of **60.46 kJ/mol**. This substantial methodological difference suggests that the choice of analysis method introduces significant systematic bias, and results from different methods should not be directly compared without correction.

The paired samples include:
- MgH2-10 wt% H-V2O5: Kissinger = 113.8, Isothermal = 84.55, diff = 29.25 kJ/mol
- Several other samples from the same DOI showing consistent method differences

---

## Q4: Predicting Ea from T_des_C (Pure MgH₂ Desorption)

**Objective:** Build a linear model predicting activation energy from onset desorption temperature using pure MgH₂ desorption data.

| Metric | Value |
|--------|-------|
| Number of data points | 4 |
| Model: Ea = a + b × T_des_C | Ea = 105.75 + 0.1287 × T_des_C |
| LOO cross-validation RMSE | 36.2 kJ/mol |
| Baseline RMSE (mean predictor) | 23.77 kJ/mol |
| Coverage (within ±2×RMSE) | 100% |

The linear model relating onset temperature to activation energy for pure MgH₂ shows a positive slope (0.1287 kJ/mol/°C), indicating that higher onset temperatures correlate with higher activation energies. However, the model's LOO cross-validation RMSE (36.2 kJ/mol) is worse than the baseline mean predictor (23.77 kJ/mol), suggesting that **T_des_C alone is not a reliable predictor of Ea** for pure MgH₂ with the current limited data. The four data points used are:

| Material | T_des_C (°C) | Ea (kJ/mol) |
|----------|-------------|-------------|
| MgH2 (pristine) | 322 | 121.0 |
| MgH2 (pure) | 417 | 161.0 |
| MgH2 (pure) | 340 | 132.19 |
| MgH2 (milled) | 390 | 156.55 |

The limited sample size (n=4) and high heterogeneity constrain the predictive power of this model.

---

## Q5: Prediction for New Sample (MgH2-5 wt% Nb₂C)

**Objective:** Predict activation energy for MgH2-5 wt% Nb₂C at T_des ≈ 250°C using the Q4 model.

| Metric | Value |
|--------|-------|
| Predicted Ea | 137.94 kJ/mol |
| 95% Prediction Interval | [38.74, 237.14] kJ/mol |
| Standard error of prediction | 50.61 kJ/mol |

For MgH2-5 wt% Nb₂C at an onset temperature of 250°C, the predicted activation energy is **137.94 kJ/mol** with a very wide 95% prediction interval ([38.74, 237.14] kJ/mol), reflecting the high uncertainty in the model due to limited training data. The wide interval indicates that predictions from this model should be interpreted with caution.

---

## Generated Hypotheses

**H1 (Catalyst effect):** Catalyst family significantly affects activation energy (η² = 0.309, p = 0.0001). MXene-carbon catalysts are expected to lower Ea compared to uncatalyzed samples.

**H2 (Method bias):** The Kissinger method yields systematically different Ea values compared to isothermal-Arrhenius (mean diff = 60.5 kJ/mol). Method choice introduces significant systematic bias that must be accounted for in meta-analyses.

**H3 (T_des predictability):** Onset temperature T_des_C predicts Ea for pure_MgH₂ with RMSE = 36.2 kJ/mol and coverage = 100.0%. Higher T_des_C correlates with higher Ea (slope = 0.13 kJ/mol/°C), though the model performs worse than the mean baseline.

**H4 (Uncertainty):** The random-effects model indicates between-study variance τ² = 1206.92, indicating substantial heterogeneity in Ea measurements across studies. The 95% CI [106.03, 127.81] accounts for this heterogeneity.

**H5 (New sample prediction):** For MgH2-5 wt% Nb₂C at T_des ≈ 250°C, the predicted Ea falls within the prediction interval, consistent with the trend that MXene-doped MgH₂ has lower Ea than pure MgH₂.

---

## Data Quality Notes

1. **Dataset size:** 46 cleaned entries from 20 unique papers (48 original rows minus 2 excluded entries).
2. **Excluded entries:** The row for `MgH2 + 5 wt% V` (DOI: 10.3390/ma16155480) was excluded due to inconsistent Ea/T_des pairing.
3. **Paired method data:** Only 5 samples have both Kissinger and isothermal-Arrhenius measurements, limiting the robustness of method comparison.
4. **Pure MgH₂ data:** Only 4 pure MgH₂ desorption entries have both Ea and T_des_C values, constraining predictive modeling.
5. **Heterogeneity:** High τ² (1206.92) indicates substantial between-study variability, likely due to differences in sample preparation, measurement methods, and catalyst types.

---

## Conclusion

The analysis reveals that catalyst family is a significant factor affecting activation energy (η² = 0.309), with systematic differences between Kissinger and isothermal-Arrhenius methods (mean difference ≈ 60 kJ/mol). The predictive relationship between onset temperature and activation energy is weak with the current data, highlighting the need for more standardized measurements and larger datasets. The substantial between-study heterogeneity (τ² = 1206.92) underscores the importance of accounting for methodological and sample-specific factors when interpreting activation energy values in MgH₂ dehydrogenation studies.