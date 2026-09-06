# Computational Thermodynamics

## Statistical Mechanics Foundations
- **Partition function (Z)**: Encodes all thermodynamic properties.
- **Helmholtz free energy**: `F = −k_B T ln Z`.
- **Entropy and heat capacity**: Derived from phonon density of states or configurational counting.

## Methods
- **Phonon free energy**: Harmonic or quasiharmonic approximation from DFT phonons.
- **Cluster expansion + Monte Carlo**: Configurational entropy and phase stability in alloys.
- **CALPHAD**: Empirical Gibbs-energy models fitted to experimental and computed data.
- **Ab-initio thermodynamics**: Calculate formation energies and chemical potentials as a function of T and p.

## Common Calculations
- **Thermal expansion**: Quasiharmonic approximation minimizing `F(V, T)`.
- **Heat capacity**: `C_V = Σ k_B (ℏω / k_BT)² exp(ℏω/kBT) / (exp(ℏω/kBT) − 1)²`.
- **Finite-temperature phase diagrams**: Combine vibrational and configurational free energies.

## Pitfalls
- Anharmonicity near phase transitions can break the quasiharmonic approximation.
- Small supercells give inaccurate low-frequency phonon sampling.
- Convergence with k-point/grid density is essential for free-energy differences.
