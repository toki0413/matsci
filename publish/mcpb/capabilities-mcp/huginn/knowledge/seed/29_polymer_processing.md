# Polymer Processing and Rheology

## Common Processing Methods
- **Extrusion**: Continuous shaping via screw-driven melt flow; key parameters are temperature profile, screw speed, and die design.
- **Injection molding**: Cyclic filling, packing, and cooling; influenced by melt temperature, mold temperature, and holding pressure.
- **Compression molding**: Suitable for thermosets and thick parts; curing kinetics dominate cycle time.
- **Spinning**: Fiber production from solution or melt; draw ratio controls orientation and modulus.
- **Additive manufacturing**: Fused filament fabrication (FFF) and vat photopolymerization; layer height and print speed affect resolution and warpage.

## Rheological Concepts
- **Shear viscosity (η)**: Resistance to flow; typically shear-thinning for polymer melts.
- **Relaxation time (τ)**: Time for polymer chains to reconfigure; affects melt strength and die swell.
- **Cox–Merz rule**: Complex viscosity from oscillatory shear approximates steady shear viscosity.

## Simulation Approaches
- **Mold filling simulation**: Solve non-Newtonian flow with heat transfer and crystallization.
- **Coarse-grained MD / DPD**: Capture mesoscale morphology during processing.
- **Calibrate viscosity models (Carreau–Yasuda, Cross) against rheometry data.**

## Quality Metrics
- Warpage, shrinkage, residual stress, weld-line strength, surface finish.
