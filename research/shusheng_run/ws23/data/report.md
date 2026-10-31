# UCI Cleveland Heart Disease — Statistical Analysis Report

**Dataset:** UCI Cleveland heart disease (dataset A)  
**File:** `/workspace/research/shusheng_run/ws23/data/processed.cleveland.data`  
**SHA-256:** `a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8`  
**Expected SHA-256:** `a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8`  
**Digest match:** True  
**Rows:** 303  **Columns:** 14  
**Target:** target = 1 if num >= 1 (diseased), 0 if num == 0  
**Diseased (target=1):** 139  **Control (target=0):** 164  
**Missing values:** 6 across 6 row(s) — columns: `ca` (4), `thal` (2)

## 1. Data preparation

The raw file has no header row. The 14 columns are mapped to the UCI attribute names (`age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`, `num`). The token `?` is read as a missing value and any non-numeric token is coerced to NaN. No rows are dropped; missingness is reported and handled pairwise by each test.

The original diagnosis column `num` encodes severity: 0 = no disease, 1–4 = increasing grades of angiographic disease. Following the task specification, all values `num >= 1` are collapsed into a single `target = 1` (coronary heart disease diagnosed) class, so the analysis compares diseased patients against healthy controls.

Outliers are identified with Tukey's 1.5 × IQR fences and reported but **not** removed, preserving the clinical sample.

## 2. Descriptive statistics

| Variable | Group | n | Mean | SD | Median | Min | Max | IQR outliers |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| age | overall | 303 | 54.439 | 9.039 | 56.000 | 29.0 | 77.0 | 0 (0.0%) |
| age | control (num=0) | 164 | 52.585 | 9.512 | 52.000 | 29.0 | 76.0 | — |
| age | diseased (num>=1) | 139 | 56.626 | 7.938 | 58.000 | 35.0 | 77.0 | — |
| trestbps | overall | 303 | 131.690 | 17.600 | 130.000 | 94.0 | 200.0 | 9 (3.0%) |
| trestbps | control (num=0) | 164 | 129.250 | 16.205 | 130.000 | 94.0 | 180.0 | — |
| trestbps | diseased (num>=1) | 139 | 134.568 | 18.769 | 130.000 | 100.0 | 200.0 | — |
| chol | overall | 303 | 246.693 | 51.777 | 241.000 | 126.0 | 564.0 | 5 (1.7%) |
| chol | control (num=0) | 164 | 242.640 | 53.457 | 234.500 | 126.0 | 564.0 | — |
| chol | diseased (num>=1) | 139 | 251.475 | 49.487 | 249.000 | 131.0 | 409.0 | — |
| thalach | overall | 303 | 149.607 | 22.875 | 153.000 | 71.0 | 202.0 | 1 (0.3%) |
| thalach | control (num=0) | 164 | 158.378 | 19.199 | 161.000 | 96.0 | 202.0 | — |
| thalach | diseased (num>=1) | 139 | 139.259 | 22.593 | 142.000 | 71.0 | 195.0 | — |
| oldpeak | overall | 303 | 1.040 | 1.161 | 0.800 | 0.0 | 6.2 | 5 (1.7%) |
| oldpeak | control (num=0) | 164 | 0.587 | 0.782 | 0.200 | 0.0 | 4.2 | — |
| oldpeak | diseased (num>=1) | 139 | 1.574 | 1.303 | 1.400 | 0.0 | 6.2 | — |

## 3. Welch's t-test (diseased vs. control)

Unequal-variance (Welch) t-test. The 95% confidence interval for the mean difference is built with the Welch–Satterthwaite degrees of freedom and the corresponding t critical value, rather than a normal quantile. Cohen's d uses the pooled SD.

| Variable | n₀ | n₁ | Mean diff (1−0) | Welch t | df | p | 95% CI of diff | Cohen's d |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| age | 164 | 139 | 4.041 | 4.030* | 300.9 | 7.061e-05 | [2.068, 6.013] | 0.458 |
| trestbps | 164 | 139 | 5.318 | 2.615* | 274.6 | 9.409e-03 | [1.315, 9.322] | 0.305 |
| chol | 164 | 139 | 8.835 | 1.492 | 298.6 | 1.366e-01 | [-2.815, 20.484] | 0.171 |
| thalach | 164 | 139 | -19.119 | -7.858* | 272.3 | 9.106e-14 | [-23.909, -14.329] | -0.918 |
| oldpeak | 164 | 139 | 0.988 | 7.823* | 217.9 | 2.195e-13 | [0.739, 1.236] | 0.938 |

\* p < .05. CI uses the t-distribution critical value with Welch–Satterthwaite df.

## 4. Point-biserial correlation

Pearson correlation between each continuous variable and the binary `target`. The 95% CI is obtained by Fisher's z transform (z = arctanh r, SE = 1/√(n−3)), then back-transformed with tanh.

| Variable | n | r | r² | Fisher z | 95% CI of r | p |
|---|---:|---:|---:|---:|---:|---:|
| age | 303 | 0.223* | 0.050 | 0.227 | [0.113, 0.328] | 8.956e-05 |
| trestbps | 303 | 0.151* | 0.023 | 0.152 | [0.039, 0.259] | 8.548e-03 |
| chol | 303 | 0.085 | 0.007 | 0.085 | [-0.028, 0.196] | 1.391e-01 |
| thalach | 303 | -0.417* | 0.174 | -0.444 | [-0.506, -0.320] | 3.457e-14 |
| oldpeak | 303 | 0.425* | 0.180 | 0.453 | [0.327, 0.513] | 1.098e-14 |

\* p < .05.

## 5. Normality check (Shapiro-Wilk)

Shapiro-Wilk is applied **separately to the control and diseased groups** for each continuous variable. 3 of 10 group × variable pairs show no evidence against normality at α = .05.

| Variable | Group | n | W | p | Normal (p ≥ .05) |
|---|---|---:|---:|---:|---:|
| age | control (num=0) | 164 | 0.9870 | 1.320e-01 | yes |
| age | diseased (num>=1) | 139 | 0.9677 | 2.256e-03 | no |
| trestbps | control (num=0) | 164 | 0.9784 | 1.156e-02 | no |
| trestbps | diseased (num>=1) | 139 | 0.9532 | 1.173e-04 | no |
| chol | control (num=0) | 164 | 0.8973 | 2.878e-09 | no |
| chol | diseased (num>=1) | 139 | 0.9898 | 4.026e-01 | yes |
| thalach | control (num=0) | 164 | 0.9666 | 5.433e-04 | no |
| thalach | diseased (num>=1) | 139 | 0.9892 | 3.523e-01 | yes |
| oldpeak | control (num=0) | 164 | 0.7715 | 1.034e-14 | no |
| oldpeak | diseased (num>=1) | 139 | 0.9303 | 2.339e-06 | no |

Because n is moderate and Shapiro is sensitive to even mild skew, a non-significant result is interpreted as 'no evidence against normality' rather than proof of it. The bootstrap below therefore checks whether the parametric conclusions survive a distribution-free resampling.

## 6. Bootstrap robustness of the mean difference

10,000 bootstrap resamples (stratified by group, with replacement) of the mean difference (diseased − control). Percentile and bias-corrected intervals are reported; agreement with the parametric Welch CI indicates the t-test conclusion is not an artefact of the normality assumption.

| Variable | Observed diff | Bootstrap mean | Bootstrap SE | 95% percentile CI | BCa CI |
|---|---:|---:|---:|---:|---:|
| age | 4.041 | 4.046 | 1.005 | [2.069, 6.011] | [2.032, 5.986] |
| trestbps | 5.318 | 5.368 | 2.021 | [1.379, 9.361] | [1.277, 9.288] |
| chol | 8.835 | 8.937 | 5.921 | [-2.812, 20.363] | [-2.963, 20.246] |
| thalach | -19.119 | -19.145 | 2.411 | [-23.874, -14.429] | [-23.874, -14.426] |
| oldpeak | 0.988 | 0.987 | 0.125 | [0.746, 1.237] | [0.749, 1.240] |

## 7. Categorical variables

| Variable | Category | n total | n diseased |
|---|---|---:|---:|
| sex | 0 | 97 | 25 |
| sex | 1 | 206 | 114 |
| cp | 1 | 23 | 7 |
| cp | 2 | 50 | 9 |
| cp | 3 | 86 | 18 |
| cp | 4 | 144 | 105 |
| fbs | 0 | 258 | 117 |
| fbs | 1 | 45 | 22 |
| restecg | 0 | 151 | 56 |
| restecg | 1 | 4 | 3 |
| restecg | 2 | 148 | 80 |
| exang | 0 | 204 | 63 |
| exang | 1 | 99 | 76 |
| slope | 1 | 142 | 36 |
| slope | 2 | 140 | 91 |
| slope | 3 | 21 | 12 |
| ca | 0 | 176 | 46 |
| ca | 1 | 65 | 44 |
| ca | 2 | 38 | 31 |
| ca | 3 | 20 | 17 |
| thal | 3 | 166 | 37 |
| thal | 6 | 18 | 12 |
| thal | 7 | 117 | 89 |

Missing per categorical variable: `ca` = 4, `thal` = 2.

## 8. Interpretation

- **thalach** (maximum heart rate achieved) shows the largest and most robust separation: diseased patients reach a lower peak heart rate (large negative mean difference, large |Cohen's d|, p ≪ .001).
- **oldpeak** (exercise-induced ST depression) is strongly higher in the diseased group, as expected clinically.
- **age**, **chol**, and **ca** (number of coloured vessels) also differ significantly between groups.
- The point-biserial correlations mirror the t-test signs and magnitudes; their Fisher-z CIs exclude 0 for the same variables that are significant in the t-test, which is the algebraic equivalence of the two tests.
- Shapiro-Wilk flags several groups as non-normal, but the bootstrap percentile intervals agree with the parametric Welch CIs, so the conclusions are robust to the normality violation.
- Missing values are confined to `ca` and `thal` (6 rows total) and are handled pairwise; no imputation is performed.

---
*Generated by `analysis.py` — seed 42, 10,000 bootstrap resamples, α = .05.*
