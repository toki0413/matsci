# Huginn

An intelligent, LLM-driven agent system for computational materials science. Automates DFT calculations, molecular dynamics simulations, symbolic regression discovery, document retrieval, and autonomous exploration of material design spaces.

## Features

- **DFT Automation**: INCAR generation, relaxation, static, DOS, and band structure calculations via VASP
- **Molecular Dynamics**: Melt-quench, NPT/NVT simulations via LAMMPS with RDF and structure analysis
- **Symbolic Regression**: Discover analytical formulas from data via PSE/PSRN (Nature Computational Science)
- **Intelligent RAG**: Document retrieval with ChromaDB embeddings, keyword fallback, and encrypted storage
- **Exploration Engine**: Autonomous multi-objective optimization with LLM-driven branch generation
- **Memory System**: Three-tier memory (session, long-term SQLite+FTS5, auto-promotion)
- **Skills Framework**: 12 declarative material science workflows
- **MCP Integration**: Connect to Materials Project, NIST databases, and mathematical analysis tools
- **Report Generation**: Auto-generate Markdown/LaTeX/HTML reports from simulation results
- **Security**: AES-128 encryption at rest with per-item salt and memory-only keys
- **Desktop App**: Tauri v2 + React 18 frontend (work in progress)
- **Coder Mode**: Autonomous code editing with read/write/edit, shell, git, and code execution tools

## Quick Start

### Installation

```bash
# Clone the repository
cd matsci-agent/agent

# Install dependencies
pip install -e .
```

### Run the Agent

```bash
# CLI mode
python -m huginn.cli "Calculate the band gap of Si"

# API server
python -m huginn.server

# MCP servers (in separate terminals)
python servers/mat-db-mcp/server.py
python servers/math-anything-mcp/server.py
```

### Run Tests

```bash
pytest tests/ -x -v
```

## Coder Mode

Run an autonomous coding session (Codex-like) that can read, edit, write,
execute shell commands, inspect git state, and run Python snippets:

```bash
# One-shot task
huginn coder "Add a docstring to huginn/tools/code_tool.py"

# Interactive mode
huginn coder

# Auto-approve destructive actions (use with caution)
huginn coder "Refactor the CLI" --auto-approve
```

Coder tools: `file_read_tool`, `file_write_tool`, `file_edit_tool`,
`bash_tool`, `git_tool`, `code_tool`.

## Architecture

See [docs/architecture.md](docs/architecture.md) for detailed system design.

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   CLI/API   │     │ Desktop App │     │   MCPs      │
└──────┬──────┘     └──────┬──────┘     └──────┬──────┘
       │                   │                    │
       └───────────────────┼────────────────────┘
                           │
                    ┌──────▼──────┐
                    │ HuginnAgent │
                    └──────┬──────┘
                           │
       ┌──────────┬────────┼────────┬──────────┐
       ▼          ▼        ▼        ▼          ▼
   ┌───────┐  ┌───────┐ ┌──────┐ ┌──────┐  ┌───────┐
   │Memory │  │Skills │ │ Tools│ │RAG   │  │Explore│
   └───────┘  └───────┘ └──────┘ └──────┘  └───────┘
```

## Tools

| Tool | Description |
|------|-------------|
| `vasp_tool` | DFT calculations (relaxation, static, DOS, band structure) |
| `lammps_tool` | MD simulations (melt-quench, NPT/NVT, RDF) |
| `symbolic_regression_tool` | Formula discovery via PSE/PSRN |
| `rag_manager` | Document retrieval and Q&A |
| `report_tool` | Auto-generate computational reports |
| `hpc_tool` | Slurm job submission and monitoring |
| `database_tool` | SQLite database operations |
| `diff_tool` | File diff and comparison |
| `mcp_client` | External MCP server tools (Materials Project, math analysis) |
| `openfoam_tool` | CFD case setup, meshing, solving, and log parsing |
| `comsol_tool` | COMSOL Multiphysics model execution and result export |
| `abaqus_tool` | ABAQUS FEA execution via Python scripting or MCP |
| `packing_tool` | Particle/molecule packing with XYZ/PDB/LAMMPS output |
| `qe_tool` | Quantum ESPRESSO DFT input generation and execution |
| `cp2k_tool` | CP2K input generation and execution |
| `code_tool` | Generic Python execution for analysis and scripting |
| `file_read_tool` | Read project files (read-only) |
| `file_write_tool` | Create or overwrite project files |
| `file_edit_tool` | Make precise string replacements |
| `bash_tool` | Execute shell commands |
| `git_tool` | Inspect git status, diff, and history |

## Skills

12 preset skills in `huginn/skills/presets.py`:

1. `standard_dft` — Standard DFT relaxation + static
2. `aimd` — Ab initio molecular dynamics
3. `defect_calculation` — Point defect formation energy
4. `surface_calculation` — Surface energy and slab models
5. `lammps_melt_quench` — Melt-quench glass generation
6. `ml_potential_training` — Train ML interatomic potentials
7. `band_gap_analysis` — Band gap with different functionals
8. `elastic_constants` — Elastic constant calculation
9. `phonon_calculation` — Phonon DOS and dispersion
10. `convergence_diagnosis` — Automatic convergence troubleshooting
11. `high_throughput_screening` — Batch property screening
12. `symbolic_regression_discovery` — Discover analytical relationships

## Memory System

The agent automatically remembers important computational results:

- **Session memory**: Current conversation context (auto-compacted at >100 messages)
- **Long-term memory**: SQLite + FTS5 full-text search with importance scoring
- **Auto-promotion**: Successful tool results are automatically saved to long-term memory

## Security

- **EncryptedVectorStore**: Document text encrypted with AES-128-CBC + HMAC-SHA256
- **CryptoVault**: PBKDF2 key derivation with per-item random salt
- **Memory-only keys**: Decryption keys never written to disk
- **EncryptedDatabase**: Transparent SQLite file encryption

## Development

### Project Structure

```
agent/
├── huginn/           # Core package
│   ├── agent.py            # Main agent
│   ├── crypto.py           # Encryption utilities
│   ├── database.py         # Database layer
│   ├── hpc.py              # HPC integration
│   ├── memory/             # Memory system
│   ├── skills/             # Skills framework
│   ├── tools/              # Tool implementations
│   ├── rag/                # RAG system
│   ├── mcp_integration/    # MCP client
│   └── exploration/        # Exploration engine
├── servers/                # MCP servers
│   ├── mat-db-mcp/         # Materials database MCP
│   └── math-anything-mcp/  # Math analysis MCP
├── desktop/                # Tauri desktop app
├── tests/                  # Test suite
└── docs/                   # Documentation
```

### Adding a New Tool

1. Create a class inheriting from `HuginnTool` in `huginn/tools/`
2. Define `name`, `description`, and `input_schema` (Pydantic model)
3. Implement `call()` method returning `ToolResult`
4. Register in `server.py` and `cli.py`
5. Add tests in `tests/`

### Adding a New Skill

1. Define a `SkillDefinition` in `huginn/skills/presets.py`
2. Add steps referencing tool names and parameters
3. The skill is automatically available via `agent.list_skills()`

## Citation

If you use Huginn in your research, please cite:

```bibtex
@software{huginn,
  title = {Huginn: An LLM-Driven Agent for Computational Materials Science},
  year = {2025},
}
```

## License

MIT License
