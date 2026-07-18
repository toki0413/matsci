# Optical Constants Reference (refractiveindex.info sampling)

> Data source: refractiveindex.info (CC0 1.0, public domain dedication).

This is a hand-picked subset of the most-used optical constants from the
refractiveindex.info database. Full database has thousands of entries.
For anything not below, fetch the YAML refractiveindex.info entry at runtime.

## Why optical constants matter

- Optical absorption / reflectivity calculations need complex refractive index
  $\tilde{n} = n + i k$ as a function of wavelength.
- Solar-cell design needs $n(\lambda)$, $k(\lambda)$ for transparent electrodes.
- Plasmonics needs Drude parameters for metals.
- Color/reflectance prediction needs $n, k$ across visible 380–700 nm.

## Refractive index models (three forms)

1. **Tabulated**: $n(\lambda), k(\lambda)$ as (λ, n, k) rows. Use linear/log interp.
2. **Sellmeier**: $n^2(\lambda) = 1 + \sum_i \frac{B_i \lambda^2}{\lambda^2 - C_i}$
3. **Drude-Lorentz**: for metals; $\varepsilon(\omega) = \varepsilon_\infty - \frac{\omega_p^2}{\omega^2 + i\gamma\omega} + \sum_i \frac{f_i \omega_i^2}{\omega_i^2 - \omega^2 - i\gamma_i\omega}$

## Silicon (Si) — crystalline, room temperature (Sellmeier, 0.4–1.5 µm)

$n^2(\lambda) = 11.6858 + \frac{0.939816 \mu m^2}{\lambda^2} + \frac{0.0357 \mu m^4}{\lambda^4 - 0.0563 \mu m^2}$

Common values:
| λ (nm) | n | k |
|---|---|---|
| 400 | 5.572 | 0.387 |
| 500 | 4.295 | 0.073 |
| 600 | 3.946 | 0.025 |
| 800 | 3.693 | 0.006 |
| 1000 | 3.578 | 0.000 |
| 1500 | 3.484 | 0.000 |

Above 1.1 µm Si is essentially transparent (k≈0).

## Fused silica (SiO₂) — Sellmeier, 0.2–3.7 µm

$n^2(\lambda) = 1 + \frac{0.6961663 \mu m^2 \lambda^2}{\lambda^2 - 0.0684043^2} + \frac{0.4079426 \mu m^2 \lambda^2}{\lambda^2 - 0.1162414^2} + \frac{0.8974794 \mu m^2 \lambda^2}{\lambda^2 - 9.896161^2}$

Common values (k ≈ 0 across visible for pure silica):
| λ (nm) | n |
|---|---|
| 200 | 1.551 |
| 300 | 1.488 |
| 400 | 1.470 |
| 500 | 1.462 |
| 600 | 1.458 |
| 800 | 1.453 |
| 1000 | 1.450 |
| 1500 | 1.444 |

## Gold (Au) — Drude + 2 Lorentz, 0.2–10 µm (Olmon 2012)

$\varepsilon_\infty = 1.54$, $\omega_p = 8.9$ eV, $\gamma = 0.074$ eV
Lorentz poles at $\omega_1 = 2.97$ eV, $\omega_2 = 3.96$ eV (approximate)

Common values:
| λ (nm) | n | k |
|---|---|---|
| 300 | 1.604 | 1.728 |
| 400 | 1.458 | 1.954 |
| 500 | 0.954 | 1.832 |
| 600 | 0.229 | 3.426 |
| 700 | 0.131 | 4.028 |
| 800 | 0.170 | 5.265 |
| 1000 | 0.276 | 6.834 |

Plasma edge near 520 nm (interband transitions above this).

## Silver (Ag) — Drude, 0.2–10 µm (Olmon 2012)

$\varepsilon_\infty = 4.0$, $\omega_p = 9.17$ eV, $\gamma = 0.021$ eV (low loss → sharp plasmon)

Common values:
| λ (nm) | n | k |
|---|---|---|
| 300 | 1.349 | 0.977 |
| 400 | 0.124 | 1.997 |
| 500 | 0.144 | 2.607 |
| 600 | 0.157 | 3.224 |
| 700 | 0.151 | 3.605 |
| 800 | 0.158 | 4.183 |
| 1000 | 0.176 | 5.412 |

Ag has the lowest optical loss of all metals in visible/NIR — preferred for plasmonics.

## Aluminum (Al) — Drude + interband, 0.2–10 µm (Rakic 1998)

| λ (nm) | n | k |
|---|---|---|
| 200 | 0.119 | 1.254 |
| 300 | 0.276 | 3.446 |
| 400 | 0.493 | 4.489 |
| 500 | 0.825 | 5.659 |
| 600 | 1.022 | 6.499 |
| 800 | 1.644 | 7.973 |
| 1000 | 1.881 | 9.030 |

Al has an interband absorption around 800 nm; less ideal than Ag/Au for visible
plasmonics, but cheaper and good for UV.

## BK7 glass (N-BK7 Schott) — Sellmeier, 0.3–2.3 µm

$n^2(\lambda) = 1 + \frac{1.03961212 \mu m^2 \lambda^2}{\lambda^2 - 0.00600069867} + \frac{0.231792344 \mu m^2 \lambda^2}{\lambda^2 - 0.0200179144} + \frac{1.01046945 \mu m^2 \lambda^2}{\lambda^2 - 103.560653}$

Common values:
| λ (nm) | n |
|---|---|
| 400 | 1.530 |
| 500 | 1.521 |
| 600 | 1.516 |
| 800 | 1.511 |
| 1000 | 1.508 |
| 1500 | 1.500 |

## Useful derived quantities

- **Reflectivity at normal incidence**: $R = \left| \frac{\tilde{n} - 1}{\tilde{n} + 1} \right|^2$
- **Skin depth**: $\delta = \frac{\lambda}{4\pi k}$
- **Absorption coefficient**: $\alpha = \frac{4\pi k}{\lambda}$
- **Dielectric function**: $\varepsilon = \tilde{n}^2 = (n^2 - k^2) + 2ink$

## Common pitfalls

- **Dispersion model validity**: each Sellmeier/Drude fit has a λ range. Extrapolating
  outside it can give unphysical n. Always check the source page.
- **Temperature dependence**: most tables are at 20–25 °C. For cryogenic or high-T,
  look up thermo-optic coefficient dn/dT (fused silica: +1.1 × 10⁻⁵ /K).
- **Crystalline vs amorphous**: Si values above are for c-Si. a-Si has higher k in visible.
- **Purity**: SiO₂ values are for fused silica. Borofloat / soda-lime differ by ~0.01.
- **Metal films vs bulk**: thin films (< 50 nm) have different optical constants
  than bulk due to surface scattering and grain boundaries.

## Sources

- Main: https://refractiveindex.info (CC0 1.0)
- Si: Li 1980 (J. Phys. Chem. Ref. Data 9, 561)
- SiO₂: Malitson 1965 (J. Opt. Soc. Am. 55, 1205)
- Au, Ag: Olmon 2012 (Phys. Rev. B 86, 235147)
- Al: Rakic 1998 (Appl. Opt. 37, 5271)
- N-BK7: Schott datasheet (vendor)

## For huginn

When the user asks for n/k at any λ, default to:
1. If the material is in this file → use the listed values.
2. Otherwise → fetch from refractiveindex.info YAML at runtime (no API, just
   `https://refractiveindex.info/database/data-nk/{mat}/{source}.yml`).
3. Report the source paper and λ-validity range alongside the value.
