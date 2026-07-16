# Built-in Knowledge Base Seeds

This directory contains small reference documents that are automatically loaded
into the Huginn RAG knowledge base on first use.

## Contents

| File | Topic |
|------|-------|
| `00_physical_constants.md` | Physical constants and unit conversions |
| `01_dft_best_practices.md` | DFT setup checklist |
| `02_vasp_quick_reference.md` | VASP tags and troubleshooting |
| `03_quantum_espresso_quick_reference.md` | Quantum ESPRESSO input and settings |
| `04_cp2k_quick_reference.md` | CP2K basis sets and MD settings |
| `05_lammps_quick_reference.md` | LAMMPS units, fixes, potentials |
| `06_abaqus_quick_reference.md` | Abaqus elements and analysis types |
| `07_openfoam_quick_reference.md` | OpenFOAM solvers and dictionaries |
| `08_orca_quantum_chemistry.md` | ORCA methods and basis sets |
| `09_materials_databases.md` | Common materials databases |
| `10_geometry_validation.md` | 3D structure validation and invariants |
| `11_machine_learning_potentials.md` | ML potentials and active learning |
| `12_hpc_job_submission.md` | Slurm/PBS job scripts and resource hints |
| `13_molecular_dynamics_best_practices.md` | MD ensembles, thermostats, sampling |
| `14_phase_diagrams.md` | Convex hull, chemical potential, finite-T |
| `15_defect_chemistry.md` | Point-defect formation and transition levels |
| `16_electronic_structure_analysis.md` | DOS, bands, charge analysis |
| `17_workflow_automation_tips.md` | Reproducibility, convergence, HPC tips |
| `18_x_ray_diffraction.md` | XRD, Bragg's law, Rietveld refinement |
| `19_thermodynamics_databases.md` | CALPHAD and materials property databases |
| `20_scanning_probe_microscopy.md` | AFM, STM, KPFM, MFM, PFM |
| `21_crystallography_basics.md` | Lattices, Miller indices, space groups |
| `22_spectroscopy_ir_raman.md` | IR/Raman and phonon calculations |
| `23_electrochemistry.md` | Electrode potentials, Pourbaix, batteries |
| `24_polymer_simulation.md` | Force fields, coarse-graining, Tg, Rg |
| `25_magnetic_materials.md` | Magnetic order, exchange, anisotropy |
| `26_battery_interfaces.md` | SEI, CEI, interfacial stability |
| `27_catalysis_descriptors.md` | Adsorption energies, scaling, volcanoes |
| `28_topology_in_materials.md` | Topological insulators, Weyl/Dirac, TDA |
| `29_polymer_processing.md` | Extrusion, molding, rheology |
| `30_optoelectronic_materials.md` | LEDs, solar cells, GW/BSE |
| `31_biomaterials.md` | Biocompatibility, scaffolds, implants |
| `32_mechanical_properties.md` | Elasticity, strength, DFT prediction |
| `33_computational_thermodynamics.md` | Phonons, free energy, phase diagrams |
| `38_benchmark_evaluation_lessons.md` | Benchmark evaluation lessons: PaperBench/MLE-bench/SAB/HLE + noise-as-feature epistemology |

## Updating seeds

Seeds are identified by a content hash. If you edit a seed file, existing
knowledge-base entries will not be replaced automatically unless you run:

```bash
huginn seed-knowledge --force
```

Adding new `.md` files to this directory will cause them to be loaded the next
time the knowledge base is initialized.
