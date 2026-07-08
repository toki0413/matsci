# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- 5xx retry with 1s backoff in desktop API client for remote backend deployment
- Version bump script (`scripts/bump_version.ps1`) to sync version across all config files
- Tag-driven stable release workflow (`.github/workflows/release.yml`)

### Changed
- Desktop CI builds remain as prerelease (`desktop-ci-N` tag)
- Stable releases now triggered by `v*` tags (e.g. `v0.2.0`)

## [0.1.0] - 2026-07-08

### Added
- LangGraph ReAct agent with 7-phase autoloop (perceive → hypothesize → plan → execute → validate → learn → report)
- 40+ simulation tools (VASP, QE, CP2K, Gaussian, ORCA, LAMMPS, Gromacs, Abaqus, Comsol, Elmer, FEniCS, OpenFOAM, RDKit, OpenMM, AutoDock Vina, etc.)
- Tool execution hooks (PRE/POST_TOOL_USE) with 15 built-in science hooks
- Event-driven simulation pipeline with 14 workflow rules
- Scientific workflow DAG visualization (Mermaid)
- Compression-aware intelligent prefetching
- Provenance registry with JSONL snapshots
- Agent trajectory logging
- Tauri desktop app with React 18 + TypeScript + Tailwind
- WebSocket streaming chat with plan confirmation
- 9 tool panels (evolve, benchmark, explore, coder, execute, workflow, diagnose, hpc, team)
- Credential management panel
- 37 science-skills for biomedical/chemical/materials databases
- CI test suites: API contract (1227 cases), security (63), chaos (26), a11y (21), performance (17)
- GitHub Actions CI: Python 3.10-3.13, Integration tests, Desktop build, Stress test
- Desktop CI prerelease builds with MSI/EXE/wheel artifacts
- Executable resolver with user-guided path selection (local/HPC/mock)
- Transolver++ PDE solver integration
