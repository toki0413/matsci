# Vibrational Spectroscopy: IR and Raman

## Infrared (IR) Spectroscopy
- Measures absorption of infrared radiation by vibrational transitions.
- Active modes require a change in dipole moment.
- Useful for identifying functional groups, adsorbates, and oxidation states.

## Raman Spectroscopy
- Measures inelastic scattering of monochromatic light (usually laser).
- Active modes require a change in polarizability.
- Complementary to IR; symmetric vibrations often appear strongly in Raman.

## Phonon Calculations
- DFT phonon frequencies at Γ point correspond to IR/Raman active modes (within harmonic approximation).
- Intensities require derivatives of the dielectric tensor (IR) or Raman tensors.
- Tools: `Phonopy`, `VASP` (DFPT), `Quantum ESPRESSO`, `Abinit`.

## Interpretation Tips
- Broad bands often indicate disorder, defects, or strong anharmonicity.
- Isotope shifts confirm element participation in a mode.
- Compare computed frequencies with experiment; typical scaling factors 0.95–1.0 for DFT phonons.

## Common Pitfalls
- Wrong band assignment due to coincidental overlap.
- Ignoring anharmonic effects at high temperature or for soft modes.
- Surface-enhanced Raman (SERS) can shift bands and change selection rules.
