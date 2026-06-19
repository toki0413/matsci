# Topological Data Analysis for Materials

Topological data analysis (TDA) captures shape and connectivity information that is robust to noise and continuous deformations. For atomic structures, TDA complements geometric descriptors by encoding ring, cavity, and connectivity statistics.

## Persistent Homology

Persistent homology tracks how topological features—connected components, loops, voids—appear and disappear as a filtration parameter grows.

- **0-dimensional features** = connected components.
- **1-dimensional features** = loops or rings.
- **2-dimensional features** = voids or cavities.

A **persistence diagram** plots the birth (filtration value where a feature appears) versus death (where it disappears). Features with long lifetimes (birth ≪ death) are considered topologically significant.

## Betti Numbers

Betti numbers count the number of independent topological features:

- β₀ = number of connected components.
- β₁ = number of 1-D cycles / rings.
- β₂ = number of 2-D voids / cavities.

For a perfect crystalline lattice with periodic boundary conditions, β₀ = 1 and higher Betti numbers depend on the chosen filtration and atomic packing.

## Filtrations for Atomic Structures

Common filtrations used in materials science:

1. **Vietoris–Rips (Rips) filtration** — connects points within a distance threshold; good for point-cloud data such as atomic coordinates.
2. **Alpha complex filtration** — built from the Delaunay triangulation; computationally cheaper and geometrically faithful in 3D.
3. **Cubical filtration** — voxel-based, useful for electron-density grids or microscopy images.

## Mapper Algorithm

Mapper converts high-dimensional data into a simplified graph by:

1. Applying a lens/filter function (e.g., PCA coordinate, density).
2. Binning the filtered values.
3. Clustering points inside each bin.
4. Connecting clusters that share points.

Mapper is useful for visualizing energy landscapes, phase spaces, and configuration spaces.

## Applications

- Classifying amorphous vs. crystalline phases via persistent homology of RDF/ADF fingerprints.
- Detecting pore channels and cages in zeolites and metal-organic frameworks.
- Quantifying defect topology around dislocations and grain boundaries.
- Building structure–property maps where persistence diagrams serve as input features for ML models.

## Practical Notes

- Choose the filtration radius to match physically relevant length scales (e.g., first coordination shell).
- Normalize diagrams before machine learning using persistence images or persistence landscapes.
- Pair TDA descriptors with composition and SOAP features for richer structure representations.
