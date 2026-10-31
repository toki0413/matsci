# Arrhenius Parameter Analysis Report

## Data Summary
- Number of data points: 75
- Bootstrap iterations: 50
- Random seed: 42

## Full Dataset Fit Results

### Log-Log Linear Regression
- Pre-exponential factor A: 3.636907e-01
- Radius scaling exponent p: 0.999873
- Activation energy Ea: 109890.899225 J/mol (109.89 kJ/mol)

### Nonlinear Least Squares
- Pre-exponential factor A: 3.966378e-01
- Radius scaling exponent p: 0.987632
- Activation energy Ea: 108869.240380 J/mol (108.87 kJ/mol)

## Bootstrap Statistics (50 iterations)

### Log-Log Regression
- A: 3.671603e-01 ± 2.051156e-02
- p: 1.000458 ± 0.002456
- Ea: 109889.837841 ± 151.712408 J/mol

### Nonlinear Least Squares
- A: 3.916380e-01 ± 4.402592e-02
- p: 0.988435 ± 0.010535
- Ea: 108998.287269 ± 739.173797 J/mol

## Model Comparison (AIC/BIC)

| Model | RSS | AIC | BIC | Parameters |
|-------|-----|-----|-----|------------|
| Arrhenius (free p) | 7.906831e+08 | 1218.818981 | 1225.771445 | 3 |
| Fick (p=2 fixed) | 2.561815e+11 | 1650.374783 | 1655.009760 | 2 |

- AIC difference (Arrhenius - Fick): -431.555802
- BIC difference (Arrhenius - Fick): -429.238314

**Interpretation:**
- Negative AIC/BIC difference favors the Arrhenius model (lower AIC/BIC is better).
- Positive difference favors the simpler Fick model.
- The magnitude indicates the strength of evidence.

## Prediction at T=650K, r=2×10⁻⁶m

### Log-Log Regression
- Predicted t50: 494.023026 s
- Bootstrap mean: 494.063656 s
- Bootstrap std: 2.006722 s

### Nonlinear Least Squares
- Predicted t50: 523.679628 s
- Bootstrap mean: 520.928328 s
- Bootstrap std: 19.897907 s

## Conclusion
The Arrhenius model with free radius scaling exponent p provides a flexible
description of the temperature and radius dependence of t50. The bootstrap
analysis quantifies the uncertainty in the parameter estimates. Model comparison
via AIC/BIC helps determine whether the simpler Fick model (p=2) is sufficient
or if the additional parameter p significantly improves the fit.
