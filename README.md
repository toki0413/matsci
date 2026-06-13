# MatSci-Agent

An intelligent, LLM-driven agent system for computational materials science with **formally verified mathematics**. Automates DFT calculations, molecular dynamics, symbolic regression, and autonomous exploration — while using Lean 4 to formally verify tensor algebra, finite element methods, numerical linear algebra, DFT theory, thermodynamics, and probability.

---

## Highlights

- **6-Phase Verified Math**: Tensor algebra → FEM weak forms → Numerical LA → DFT → Thermodynamics → Probability, all with Lean 4 formal proofs
- **Multi-Provider LLM Support**: OpenAI, Anthropic, DeepSeek, Google GenAI, OpenRouter, NVIDIA, Ollama, **vLLM**, **LM Studio**, and any OpenAI-compatible local endpoint
- **Symbolic ↔ Formal Bridge**: SymPy expressions are automatically translated to Lean `Float` definitions and type-checked
- **Config Persistence**: Save/load TOML or JSON config files; CLI flags override config files which override environment variables
- **Rust Accelerators**: Optional `matsci-ext` PyO3 extension accelerates LAMMPS/VASP parsing and MSD/RDF analysis
- **FEA/CFD Tools**: Native `comsol_tool`, `openfoam_tool`, `abaqus_tool` (script generator + MCP bridge), LAMMPS and VASP execution tools
- **Packing/Visualization**: Native `packing_tool` for molecule/particle packing with 3D preview; feeds OpenFOAM, COMSOL, and Abaqus
- **Generic Code Execution**: Native `code_tool` lets the LLM write and run Python for custom analysis, visualization, UQ, GP, Bayesian optimization, and post-processing
- **Coder Mode**: `matsci coder` provides an autonomous Codex-like editing loop with read/write/edit, shell, git, and permission checks
- **UQ/GP Skills**: Declarative skills for Monte Carlo propagation, sensitivity/Sobol analysis, GP prediction, and Bayesian calibration
- **Open-Source DFT**: Native `qe_tool` (Quantum ESPRESSO) and `cp2k_tool`
- **Retry & Resilience**: Automatic exponential backoff on rate limits and transient API errors

---

## Quick Start

### 1. Installation

**Python backend (required)**

```bash
cd matsci-agent/agent
pip install -e .
# Optional: for config file support
pip install toml
```

**Rust CLI frontend (recommended)**

```bash
cd matsci-agent/cli
cargo build --release
```

On Windows with the GNU toolchain, ensure MinGW's `dlltool.exe` is on `PATH`:

```powershell
$env:PATH = "C:\mingw64\mingw64\bin;$env:PATH"
cargo build --release
```

The resulting binary is at `cli/target/release/matsci` (or `matsci.exe` on Windows). Add it to your `PATH` or copy it to a directory already on `PATH`.

**Rust performance accelerators (optional)**

```bash
cd matsci-agent/pyext
maturin build --release
pip install target/wheels/matsci_ext-*.whl
```

Currently accelerates:
- LAMMPS dump/trajectory parsing, MSD, and RDF computation (`LammpsTool`)
- VASP OUTCAR streaming parse (`VaspTool`)
- General NumPy-array MSD/RDF (`matsci_ext.compute_msd`, `matsci_ext.compute_rdf`)

`LammpsTool` and `VaspTool` automatically use the Rust backend when installed and fall back to pure Python otherwise.

On Windows with the GNU toolchain, ensure MinGW's `dlltool.exe` is on `PATH` (same as the CLI build above).

### Optional: Abaqus Integration

If you have the `.abaqus-mcp` server installed in your home directory, `matsci chat` will auto-discover and connect it:

```bash
# Default location: ~/.abaqus-mcp/mcp_server.py
# Override via environment variable:
export ABAQUS_MCP_SERVER_PATH=/path/to/mcp_server.py

# Or in matsci.toml:
abaqus_mcp_server = "/path/to/mcp_server.py"
```

Requires Abaqus/CAE to be running with the `.abaqus-mcp` plugin loaded and its loop started. Available tools include `execute_script`, `submit_job`, `get_odb_info`, and `get_viewport_image`.

### Optional: COMSOL Integration

`matsci-agent` includes a `comsol_tool` that generates COMSOL Java scripts and runs them via the COMSOL CLI when available:

```bash
# Requires comsol or comsolmph on PATH, or set:
export COMSOL_EXECUTABLE=/path/to/comsol

# In matsci chat, the agent can generate + run models:
# comsol_tool action=run physics=solid_mechanics ...
```

If COMSOL is not installed, the tool exports the generated script so you can run it manually.

### Optional: OpenFOAM Integration

`matsci-agent` includes an `openfoam_tool` that generates complete OpenFOAM case directories and runs `blockMesh` + the chosen solver when OpenFOAM is installed:

```bash
# Requires blockMesh and the solver (e.g. icoFoam, simpleFoam) on PATH, or set:
export OPENFOAM_DIR=/path/to/openfoam/version

# In matsci chat, the agent can generate + run cases:
# openfoam_tool action=run solver=icoFoam case_name=pipe_flow ...
```

If OpenFOAM is not installed, the tool exports the full case (`system/`, `constant/`, `0/`) so you can run it manually.

### Optional: Packing and Visualization

`matsci-agent` includes a `packing_tool` for packmol-style molecular packing and particle-filled composite systems:

```bash
# Pack 20 water molecules into a 20 Å box
# packing_tool action=pack mode=molecules ...

# Pack spherical SiO2 inclusions into a matrix for FEA/OpenFOAM multiphase modeling
# packing_tool action=pack mode=particles ...

# Preview any XYZ/PDB file
# packing_tool action=preview structure_file=system.xyz ...
```

Supported inputs:
- XYZ files
- SMILES strings (requires RDKit)
- Molecule names like `water`, `methane` (uses ASE if available, otherwise built-in placeholders)
- Particle JSON like `{"shape": "sphere", "radius": 2.0, "n_points": 50, "symbol": "Si"}`

Supported outputs: `xyz`, `pdb`, `lammps-data`, plus a PNG 3D preview. If Packmol is installed, the tool can also generate/run a native `packmol.inp`.

The packing result can be consumed directly by downstream tools:
- `openfoam_tool action=set_fields` → `system/setFieldsDict` + `0/alpha.*` for VOF/multiphase; runs `setFields` if OpenFOAM is installed, otherwise exports files.
- `comsol_tool action=import_packing` → Java script that creates spheres at particle centres; runs `comsol batch` if COMSOL is installed, otherwise exports the script.
- `abaqus_tool action=import_packing` → Python script that adds reference points or spherical inclusions; runs `abaqus cae noGUI=...` if Abaqus is installed, otherwise exports the script. The script can also be sent to the existing Abaqus MCP server.

### Optional: Generic Code Execution

`matsci-agent` includes a `code_tool` that lets the LLM generate and execute Python code inside a sandboxed subprocess. This is the preferred way to do ad-hoc analysis instead of adding a dedicated tool for every numeric task:

```bash
# Run Python code and capture a result variable
# code_tool action=execute code="import numpy as np; result = np.sin(np.pi/2)" result_variable=result

# Generate a plot and auto-detect output files
# code_tool action=execute code="... plt.savefig('uplot.png') ..."
```

Use cases:
- Custom uncertainty quantification and Sobol sensitivity analysis.
- Gaussian process surrogate modeling and Bayesian optimization.
- Post-processing simulation outputs (parse logs, compute MSD/RDF, plot stress-strain).
- Prototyping new analysis steps before deciding whether to promote them to core tools.

The tool runs with a timeout, captures stdout/stderr, and collects saved PNG/CSV/JSON/XYZ/data files. Reusable UQ/GP/Bayesian-optimization logic is still available as library modules (`matsci_agent.tools.uq_tool`, `matsci_agent.tools.gp_tool`) for the LLM to import when needed.

### Optional: Quantum ESPRESSO and CP2K

`matsci-agent` includes native tools for open-source DFT codes:

```bash
# Quantum ESPRESSO
export QE_EXECUTABLE=/path/to/pw.x
# CP2K
export CP2K_EXECUTABLE=/path/to/cp2k.popt
```

Both tools generate input files, run the code when available, and parse total energy / forces / stress / convergence. If the executable is not found, they export the input file for manual execution.

### 2. Configure an LLM Provider

**Option A — Environment variables**
```bash
export MATSCI_PROVIDER=openai
export MATSCI_MODEL=gpt-4o
export OPENAI_API_KEY=sk-...
```

**Option B — Config file**
```bash
matsci configure   # interactive wizard
# or write manually:
cat > matsci.toml << 'EOF'
provider = "openai"
model = "gpt-4o"
api_key = "sk-..."
EOF
matsci chat --config matsci.toml
```

**Option C — CLI flags**
```bash
matsci chat --provider openai --model gpt-4o
matsci chat --provider ollama --ollama-url http://localhost:11434
matsci chat --provider vllm --base-url http://localhost:8000/v1 --model llama-3-8b
```

### 3. Start Chatting

```bash
matsci chat
```

### 5. Autonomous Coder Mode

```bash
# One-shot task with interactive approval for destructive actions
matsci coder "Refactor the CLI argument parsing into a separate module"

# Auto-approve all destructive actions (use with caution)
matsci coder "Update all docstrings in matsci_agent/tools" --auto-approve
```

### 6. Run the Test Suite

```bash
cd agent
pytest tests/ -x -q          # Python agent tests (192 tests)
cd lean/MatSciLean
lake build MatSciLean         # Lean formalization build

# Rust CLI smoke tests
cd ../../cli
./target/release/matsci --version
./target/release/matsci tools
```

---

## LLM Provider Reference

| Provider | Needs API Key | Default Model | Notes |
|----------|---------------|---------------|-------|
| `openai` | Yes (unless local) | `gpt-4o` | Also works with local OpenAI-compatible endpoints |
| `anthropic` | Yes | `claude-3-5-sonnet-20241022` | |
| `deepseek` | Yes | `deepseek-chat` | |
| `google-genai` | Yes | `gemini-2.5-pro` | |
| `openrouter` | Yes | `anthropic/claude-sonnet-4` | Unified API for many models |
| `nvidia` | Yes | `meta/llama-3.1-405b-instruct` | NVIDIA AI Endpoints |
| `ollama` | No | `qwen2.5:14b` | Local inference; set `--ollama-url` |
| `vllm` | No | *required* | OpenAI-compatible; set `--base-url` |
| `local` | No | *required* | Alias for any OpenAI-compatible local server |

**Local endpoints** (vLLM, LM Studio, TGI, etc.) do **not** require a real API key — a dummy key is sent automatically when `--base-url` points to `localhost`, `127.*`, `::1`, or `0.0.0.0`.

---

## Lean Formalization

All symbolic math results can be formally verified in Lean 4.

### Build

```bash
cd agent/lean/MatSciLean
lake build MatSciLean
```

**Speeding up rebuilds:** After the first successful build, `.olean` cache files are generated. Use `lake build --no-build` to skip recompilation of unchanged modules, or share the `build/` directory across machines.

### Modules

| Module | Content |
|--------|---------|
| `TensorAlgebra.lean` | Index notation, contraction, metric transforms |
| `FiniteElement.lean` | Weak forms: linear elasticity, heat conduction, element assembly |
| `NumericalLinearAlgebra.lean` | LU, Cholesky, Jacobi, Conjugate Gradient, error matrices |
| `DFT.lean` | Free electron gas, tight-binding, LDA exchange-correlation |
| `Thermodynamics.lean` | Equations of state, free energy, Clausius-Clapeyron, partition functions |
| `Probability.lean` | Normal distribution, GP kernels, MC sampling, Bayesian update |

### Verification Workflow

1. Use the `symbolic_math_tool` to derive an expression (e.g., tensor calculus, FEM weak form).
2. The agent automatically routes the result to `lean_tool` with `auto_verify_action` set to the matching verifier.
3. Lean code is generated and type-checked via `lake build`.

---

## Configuration File Format

TOML (`matsci.toml`):

```toml
provider = "openai"
model = "gpt-4o"
api_key = "sk-..."
base_url = ""
ollama_host = "http://localhost:11434"
workspace = "."
auto_approve = false
enable_exploration = true
max_parallel_branches = 5
```

JSON (`matsci.json`) is also accepted. Load with `matsci chat --config matsci.toml`.

Save a running config:
```python
from matsci_agent.config import MatSciConfig
cfg = MatSciConfig.from_env()
cfg.save("matsci.toml")
```

---

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   CLI/API   │     │ Desktop App │     │   MCPs      │
└──────┬──────┘     └──────┬──────┘     └──────┬──────┘
       │                   │                    │
       └───────────────────┼────────────────────┘
                           │
                    ┌──────▼──────┐
                    │ MatSciAgent │
                    └──────┬──────┘
                           │
       ┌──────────┬────────┼────────┬──────────┐
       ▼          ▼        ▼        ▼          ▼
   ┌───────┐  ┌───────┐ ┌──────┐ ┌──────┐  ┌───────┐
   │Memory │  │Skills │ │ Tools│ │RAG   │  │Lean   │
   └───────┘  └───────┘ └──────┘ └──────┘  └───┬───┘
                                                 │
                                          ┌──────▼──────┐
                                          │ MatSciLean  │
                                          │ (Lean 4)    │
                                          └─────────────┘
```

---

## Project Structure

```
matsci-agent/
├── agent/
│   ├── matsci_agent/       # Core Python package
│   │   ├── agent.py        # LLM provider factory + retry
│   │   ├── config.py       # Config dataclass with save/load
│   │   ├── cli.py          # CLI entry point
│   │   ├── tools/          # 20+ core tools (VASP, LAMMPS, OpenFOAM, Packing, Abaqus, Code, Lean, ...)
│   │   ├── skills/         # 19 declarative workflows
│   │   ├── memory/         # Session + long-term + FTS5
│   │   └── rag/            # ChromaDB + encrypted storage
│   ├── lean/MatSciLean/    # Lean 4 formalization library
│   ├── servers/            # MCP servers (mat-db, math-anything)
│   ├── tests/              # 192 pytest tests
│   └── docs/               # Architecture docs
├── desktop/                # Tauri v2 + React 18 frontend
└── skills/                 # Shared skill definitions
```

---

## Development

### Adding a New Tool

1. Create a class inheriting from `MatSciTool` in `matsci_agent/tools/`
2. Define `name`, `description`, and `input_schema` (Pydantic model)
3. Implement `call()` returning `ToolResult`
4. Register in `cli.py`
5. Add tests in `tests/`

### Adding a New Lean Module

1. Create `lean/MatSciLean/<Module>.lean`
2. Add `import MatSciLean.<Module>` to `MatSciLean.lean`
3. Add `verify_<module>` to `auto_pipeline.py`
4. Register the workflow template in `workflows/templates.py`
5. Add a skill preset in `skills/presets.py`
6. Add Python tests in `tests/`

---

## License

MIT License
