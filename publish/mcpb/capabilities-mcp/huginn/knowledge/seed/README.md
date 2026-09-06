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
| `34_topological_data_analysis.md` | TDA, persistent homology, ring/cavity descriptors |
| `35_geometric_invariants.md` | Coordinate-free structural invariants, ML descriptors |
| `36_multi_modal_visualization.md` | Plot/structure/text visualization for simulation output |
| `37_scientific_discovery_benchmarking.md` | RCBench cross-domain lessons: phased protocol, anti-over-engineering |
| `38_benchmark_evaluation_lessons.md` | Benchmark evaluation lessons: PaperBench/MLE-bench/SAB/HLE + noise-as-feature epistemology |
| `39_orchestrator_unification_lessons.md` | Orchestrator 4-layer architecture + min_calls patch history (v5 R17 reverted) |
| `40_security_hardening_and_bench_activation.md` | v5 spec: 25 additions (G20-G44) + 8 subtractions (R14-R21) + 5 milestones (M1-M5) |
| `41_ai4science_onescience.md` | OneScience AI4Science toolkit: install/domains, GPU & DCU platforms |
| `42_optimade_federation.md` | OPTIMADE REST API: one client, many materials databases |
| `43_jarvis_databases.md` | JARVIS (NIST) DFT/FF/TB/ML/ChemNLP infrastructure |
| `44_optical_constants_reference.md` | refractiveindex.info optical constants subset |
| `45_catalysis_her_pipeline.md` | HER: Volmer-Heyrovsky/Tafel, volcano, scaling relations |
| `46_catalysis_oer_pipeline.md` | OER: 4-electron bottleneck, adsorbate scaling, stability |
| `47_catalysis_co2rr_pipeline.md` | CO₂RR: C1/C2/C3 products, C-C coupling, selectivity |
| `48_catalysis_nrr_pipeline.md` | NRR: N₂ activation, Faradaic efficiency limits |
| `49_gibbs_three_stage_workflow.md` | geo_opt → freq → gibbs_energy thermochemistry pipeline |
| `50_deepmd_cohp_pyscf_gpaw.md` | DeepMD-kit / LOBSTER-COHP / PySCF / GPAW quick reference |
| `51_fde_subsystem_dft.md` | Frozen density embedding (FDE) subsystem DFT |
| `52_nature_review_strategies.md` | Nature-style peer review & author response taxonomy |

## Updating seeds

Seeds are identified by a content hash. If you edit a seed file, existing
knowledge-base entries will not be replaced automatically unless you run:

```bash
huginn seed-knowledge --force
```

Adding new `.md` files to this directory will cause them to be loaded the next
time the knowledge base is initialized.
