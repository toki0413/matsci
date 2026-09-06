# X-Ray Diffraction (XRD) for Materials

## Bragg's Law
Constructive interference occurs when:

`n λ = 2 d sin θ`

- `λ`: X-ray wavelength (Cu Kα ≈ 1.5418 Å)
- `d`: spacing between lattice planes
- `θ`: diffraction angle

## Powder Diffraction
- A polycrystalline sample produces rings that project as peaks at characteristic 2θ angles.
- Peak positions identify the phase; peak widths relate to crystallite size and microstrain.
- Scherrer equation: `D = K λ / (β cos θ)`, where `β` is the integral breadth (radians) and `K ≈ 0.9`.

## Rietveld Refinement
- Fits the whole diffraction pattern to a structural model.
- Refinable parameters: lattice parameters, atomic positions, site occupancies, thermal parameters, microstructure (size/strain), and preferred orientation.
- Goodness-of-fit indicators: R_wp, R_exp, χ² (should approach 1).

## Common Uses
- Phase identification (match to ICDD/PDF database).
- Quantitative phase analysis with internal standards.
- In-situ temperature/pressure studies of phase transitions.

## Best Practices
- Grind sample to ~10 µm to reduce preferred orientation and microabsorption.
- Use appropriate scan step size and counting time for the desired resolution/statistics.
- Compare simulated patterns from DFT-relaxed structures with experiment.
