# Materials Databases Reference

Curated list of commonly used materials databases and when to use them.

## Crystallographic / computed structures

- **Materials Project** (materialsproject.org)
  - DFT-computed structures, energies, band structures, elastic tensors.
  - API via `pymatgen` or `mp-api`.
- **AFLOW** (aflowlib.org)
  - High-throughput DFT data; useful for prototypes and convex hulls.
- **OQMD** (oqmd.org)
  - Open quantum materials database; formation energies and structures.
- **NOMAD** (nomad-lab.eu)
  - Repository for shared computational materials data.

## Experimental / ICSD structures

- **ICSD** (Inorganic Crystal Structure Database)
  - Authoritative experimental crystal structures; often requires subscription.
- **COD** (Crystallography Open Database)
  - Free open crystallographic structures.

## Small molecules

- **PubChem** (pubchem.ncbi.nlm.nih.gov)
  - Chemical structures, properties, synonyms.
- **ChemSpider** (chemspider.com)
  - Aggregated chemical data.
- **ZINC** (zinc.docking.org)
  - Purchasable compounds for docking and screening.

## Interatomic potentials / force fields

- **OpenKIM** (openkim.org)
  - Verified interatomic potentials with reproducible tests.
- **NIST Interatomic Potentials** (www.ctcms.nist.gov/potentials)
  - EAM/MEAM potentials for metals.
- **Interatomic Potentials Repository** (pure.mpg.de)
  - Various potentials for atomistic simulations.

## Best practices

- Always cite the database and the calculation methodology.
- Verify downloaded structures before production runs (check oxidation, magnetism, supercell).
- Use ICSD/COD as starting geometries; relax with DFT/MD before property calculations.
