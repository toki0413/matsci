# OPTIMADE Federation Query Protocol

> Data source: OPTIMADE specification (CC-BY 4.0, optimade.org) + provider registry.

OPTIMADE (Open Databases Integration for Materials Design) is a REST API spec
that lets one client query many materials databases uniformly. Instead of
learning MP/OQMD/AFLOW/NOMAD/JARVIS APIs separately, learn OPTIMADE once.

## Why OPTIMADE matters

- One query syntax, many providers
- Standardized fields (`chemical_formula_descriptive`, `nelements`, `band_gap`...)
- Filter grammar: `?filter=nelements=2 AND elements HAS "Si"`
- Pagination: `page_limit` + `page_offset` or `links` cursor
- Rate limits per provider (typically 10-30 req/s)

## Query endpoints

| Endpoint | Returns |
|---|---|
| `/info` | Provider metadata, available fields, schemas |
| `/structures` | Material structures (CIF-like) |
| `/materials` | Material properties (some providers alias to /structures) |
| `/references` | Literature references |
| `/links` | Related resources / other OPTIMADE providers |

## Filter grammar (compact)

- Comparison: `nelements=3`, `band_gap>=0.5`
- Logical: `AND`, `OR`, `NOT`
- List: `elements HAS "Si"`, `elements HAS ALL "Si","O"`, `elements HAS ONLY "Si","O"`
- String: `chemical_formula_descriptive CONTAINS "Fe2O3"`
- Parentheses for grouping

URL-encode the filter for HTTP. Example:
```
/structures?filter=nelements=2 AND elements HAS "Si"&page_limit=50
```

## Standard fields (always available)

| Field | Type | Notes |
|---|---|---|
| `id` | string | Provider-specific material ID |
| `task_id` | string | Alias of id in some providers |
| `chemical_formula_descriptive` | string | "Fe2O3" |
| `chemical_formula_reduced` | string | "Fe2O3" |
| `chemical_formula_anonymous` | string | "A2B3" |
| `elements` | list[str] | ["Fe", "O"] |
| `nelements` | int | 2 |
| `lattice_vectors` | list[list[float]] | Rows = lattice vectors, Å |
| `cartesian_site_positions` | list[list[float]] | Å |
| `species_at_sites` | list[str] | Per-atom species |
| `structure_features` | list[str] | ["disorder"] etc. |

Provider-specific fields live under prefixes: `_mp_chemsys`, `_oqmd_band_gap`,
`_aflow_protostructure`. Query `/info` to see what's exposed.

## Major providers (verified 2025)

| Provider | Base URL | Coverage | Notes |
|---|---|---|---|
| Materials Project | https://optimade.materialsproject.org | 150k+ | MP legacy + new |
| OQMD | https://oqmd.org/optimade | 1M+ | Formations, convex hull |
| AFLOW | https://api.aflow.org/optimade | 3.5M+ | Largest, automated |
| NOMAD | https://nomad-lab.eu/optimade/index | 19.4M calc entries | VASP/QE/CP2K raw |
| JARVIS | https://jarvis.nist.gov/optimade | 80k+ | NIST, public domain |
| Materials Cloud | https://optimade.materialscloud.org | curated | AiiDA provenance |
| tc-database | https://tcdc.physics.uoc.gr/optimade | topological | Bi2Se3 family |
|odbx (Open Database of Xtals) | https://optimade.odbx.science | 10M+ | Aggregated |
| 2D structures | https://optimade.mpds.io | thin films | MPDS-led |

Use `GET /links` on any provider to discover others in the federation.

## Python client (lazy — pip installable)

```python
from optimade.adapters import Structure
import requests

r = requests.get(
    "https://optimade.materialsproject.org/structures",
    params={"filter": 'elements HAS ALL "Si","O" AND nelements=2', "page_limit": 20},
    headers={"Accept": "application/vnd.optimade+json"},
)
structs = [Structure(s) for s in r.json()["data"]]
```

The `optimade-python-tools` package gives full client + server reference impl.
No API key for most providers; MP requires a free API key for some endpoints.

## Common pitfalls

- **Field availability varies**: not every provider exposes `band_gap`. Check `/info`.
- **Pagination cursor vs offset**: prefer `links[next]` cursor when offered.
- **Lattice units**: spec says Å, but always verify via `/info` if provider is unfamiliar.
- **Fractional vs Cartesian**: spec uses `cartesian_site_positions`; convert if you need fractional.
- **Species vs elements**: `species_at_sites` can have oxidation state (`Fe3+`); `elements` is bare.

## When to use OPTIMADE vs direct API

- **OPTIMADE**: discovery / screening / cross-database comparison / simple structure fetch
- **Direct provider API**: complex queries (e.g. MP's `summary` endpoint with many
  computed properties), bulk downloads (e.g. NOMAD's raw-file API), authenticated writes

For huginn, default to OPTIMADE for structure/property lookup; fall back to direct
API only when OPTIMADE lacks the field or for bulk/raw data.

## Sources

- Spec: https://optimade.org/ (CC-BY 4.0)
- Provider registry: https://providers.optimade.org
- Client: https://github.com/Materials-Consortia/optimade-python-tools (MIT)
- Tutorial: https://optimade.org/odr/tutorial
