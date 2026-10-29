# Catalyst Effect Analysis Report

## Summary of Hypothesis Tests

### H1: Within-paper paired ΔEa Analysis
- Number of paired samples: 15
- Number of papers: 12
- Mean ΔEa (fixed effects): -56.49 kJ/mol
- Random effects mean ΔEa: -51.58 kJ/mol
- Between-study variance (τ²): 444.69
- 95% Cluster bootstrap CI: [-70.15, -40.44] kJ/mol

### H2: η² for Catalyst Family with Permutation Test
- Observed η²: 0.3088
- Permutation p-value: 0.0000

### H3: Method Comparison (Kissinger vs Isothermal Arrhenius)
- Number of comparison pairs: 1
- Mean difference (Iso - Kissinger): -29.25 kJ/mol
- Paired t-test: t=nan, p=nan

### H4: Leave-One-Subject-Out Cross-Validation
- Number of papers: 19
- RMSE: 34.74 kJ/mol
- MAE: 25.49 kJ/mol
- R²: -0.0250

### H5: Out-of-Sample Prediction (catalyzed_MgH2, other, isothermal_Arrhenius)
- Training samples: 40
- No matching rows found; predicting for hypothetical
- Predicted Ea: 98.81 kJ/mol

## Data Summary
- Desorption-only rows from multiple papers
- Material classes: catalyzed_MgH2, pure_MgH2, alloy, complex_hydride
- Catalyst families: mxene_carbon, oxide, metal_nanoparticle, other, none
- Methods: Kissinger, isothermal_Arrhenius