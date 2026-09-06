# Magnetic Materials

## Magnetic Order
- **Ferromagnetism**: Parallel alignment of spins; finite net magnetization below Tc.
- **Antiferromagnetism**: Antiparallel neighboring spins; zero net magnetization.
- **Ferrimagnetism**: Antiparallel but unequal moments; net magnetization.

## Energy Scales
- **Exchange interaction (J)**: Determines magnetic ordering temperature.
- **Magnetic anisotropy energy (MAE)**: Energy barrier for magnetization rotation; MAE = E_hard − E_easy (per unit volume).
- **Magnetocrystalline anisotropy**: Originates from spin–orbit coupling; often requires non-collinear or SOC calculations.

## Computational Methods
- **Collinear DFT + U**: Captures localized moments in transition-metal oxides.
- **Non-collinear DFT + SOC**: Needed for spin–orbit-driven properties (MAE, Dzyaloshinskii–Moriya).
- **Heisenberg exchange from DFT**: Map total energies of spin configurations to J_ij.
- **Monte Carlo**: Use extracted J_ij to estimate Curie/Neel temperatures.

## Practical Notes
- Check magnetic ground state carefully; small energy differences between spin configurations are common.
- Hubbard U is often essential for localized d/f electrons.
- Compare total energies per formula unit and include spin–orbit coupling for anisotropy.

## Tools
- `VASP` (ISPIN, LSORBIT, MAGMOM), `Quantum ESPRESSO` (nspin=2, lspinorb)
- `TB2J` for exchange parameters
- `Spirit`, `Vampire` for spin dynamics / Monte Carlo
