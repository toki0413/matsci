# Huginn Architecture

## Overview

Huginn is a modular, LLM-driven agent system for computational materials science. It supports DFT calculations (VASP), molecular dynamics (LAMMPS), symbolic regression, RAG-based document retrieval, encrypted data management, and automated exploration workflows.

## System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     User Interface                            │
│  CLI (cli.py)  │  Desktop App (Tauri+React)  │  API Server  │
├──────────────────────────────────────────────────────────────┤
│                     Agent Layer                               │
│              HuginnAgent (LangGraph)                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │   Memory     │  │    Skills    │  │ Exploration  │       │
│  │  (3-tier)    │  │ (Declarative)│  │   Engine     │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
├──────────────────────────────────────────────────────────────┤
│                     Tool Layer                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │     DFT      │  │    MD/ML     │  │     RAG      │       │
│  │  (vasp_tool) │  │ (lammps_tool)│  │(rag_manager) │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │   Symbolic   │  │    MCP       │  │    Report    │       │
│  │  Regression  │  │  Integration │  │   Generator  │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
├──────────────────────────────────────────────────────────────┤
│                     Infrastructure                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │    Crypto    │  │    HPC       │  │   Database   │       │
│  │  (Vault)     │  │  (Slurm/SSH) │  │ (Chroma/FTS5)│       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└──────────────────────────────────────────────────────────────┘
```

## Module Descriptions

### 1. Agent Layer (`huginn/agent.py`)

The core `HuginnAgent` is built on LangGraph and manages the reasoning loop:

- **State management**: Tracks messages, tool calls, and reasoning traces
- **Memory integration**: Automatically promotes important tool results to long-term memory
- **Skill execution**: Declarative workflow execution via `execute_skill()`
- **Exploration**: Integrates with `ExplorationEngine` for autonomous discovery

### 2. Memory System (`huginn/memory/`)

Three-tier memory architecture:

| Tier | Class | Purpose | Backend |
|------|-------|---------|---------|
| Session | `SessionContext` | Current conversation | In-memory |
| Long-term | `LongTermMemory` | Persistent facts | SQLite + FTS5 |
| Manager | `MemoryManager` | Coordination | Python |

Key features:
- Automatic message compaction (summarization when >100 messages)
- Importance scoring (0-1) for memory retrieval
- Vector semantic search (optional, via ChromaDB)
- Category and tag-based filtering

### 3. Skills System (`huginn/skills/`)

Declarative workflow definition and execution:

```python
skill = SkillDefinition(
    name="standard_dft",
    description="Run a standard DFT relaxation",
    steps=[
        {"tool": "vasp_relaxation", "params": {"encut": 520}},
        {"tool": "vasp_static", "params": {"encut": 520}},
    ]
)
```

12 preset skills covering: DFT, AIMD, defects, surfaces, LAMMPS, ML potentials, phonons, elastic constants, convergence diagnosis, high-throughput screening, and symbolic regression.

### 4. Tool Layer (`huginn/tools/`)

| Tool | Purpose | Key Features |
|------|---------|--------------|
| `vasp_tool` | DFT calculations | INCAR parsing, relaxation, static, DOS, band structure |
| `lammps_tool` | Molecular dynamics | Melt-quench, NPT/NVT, RDF analysis |
| `symbolic_regression_tool` | Formula discovery | PSE/PSRN integration, Pareto frontier |
| `rag_manager` | Document retrieval | ChromaDB + keyword fallback, encrypted storage |
| `report_tool` | Report generation | Markdown/LaTeX/HTML/JSON output |

### 5. RAG System (`huginn/rag/`)

- **VectorStore**: ChromaDB-based with `all-MiniLM-L6-v2` embeddings
- **EncryptedVectorStore**: Document encryption at rest (AES-128-CBC + HMAC)
- **File parsing**: PDF, CSV, JSON, TXT with smart chunking
- **Keyword fallback**: BM25-style search when vector search fails

### 6. Crypto Module (`huginn/crypto.py`)

- **CryptoVault**: Fernet encryption (AES-128-CBC + HMAC-SHA256) with PBKDF2
- **KeyManager**: Password-protected master key file
- **EncryptedDatabase**: Transparent SQLite encryption at rest
- **EncryptedConfig**: Encrypted JSON configuration files

Security guarantees:
- Per-item random salt
- Keys never persisted to disk (only encrypted key blobs)
- Memory-only key storage

### 7. MCP Integration (`huginn/mcp_integration/`)

Connects to external MCP servers:
- **mat-db-mcp**: Materials Project, NIST interatomic potentials, property search
- **math-anything-mcp**: Math extraction, dimensional analysis, expression normalization

Architecture:
- `MCPClientManager`: Async stdio-based server connections
- `MCPAdapter`: Wraps MCP tools as LangChain `StructuredTool`

### 8. Exploration Engine (`huginn/exploration/`)

Autonomous discovery via LLM-driven branch generation:

```
Objective → Generate Branches → Evaluate → Pareto Prune → Backtrack if needed
```

Components:
- `ExplorationEngine`: Main loop with iteration control
- `ExplorationOrchestrator`: Async parallel execution with semaphore
- `Backtracker`: Failure diagnosis and recovery strategies

### 9. Infrastructure

**HPC Integration** (`huginn/hpc.py`):
- Slurm job submission and monitoring
- SSH-based remote execution
- Job queue querying

**Database** (`huginn/database.py`):
- SQLite with FTS5 full-text search
- Encrypted database wrapper

## Data Flow

```
User Query
    ↓
HuginnAgent.chat()
    ↓
Build prompt with memory injection
    ↓
LLM reasoning → tool selection
    ↓
Tool execution (local/MCP/HPC)
    ↓
Result storage (session + long-term memory)
    ↓
Response generation
```

For exploration workflows:

```
Objective
    ↓
ExplorationEngine.generate_branches()
    ↓
ExplorationOrchestrator.execute_branches()
    ↓
Pareto frontier update
    ↓
Backtracker (if failures)
    ↓
Return best solutions
```

## Design Principles

1. **Graceful degradation**: Every component has a mock/fallback mode for development without full infrastructure
2. **Security by default**: Encryption at rest, memory-only keys, per-item salt
3. **Modularity**: Each component can be used independently
4. **Type safety**: Pydantic models for all inputs/outputs
5. **Testability**: 44+ unit tests covering all major components

## Configuration

Key configuration files:
- `pyproject.toml`: Python dependencies and project metadata
- `config.yaml` (optional): Agent, LLM, and tool configuration
- `tauri.conf.json`: Desktop app configuration

## Development Guidelines

- Add tests for new features in `tests/`
- Use `ToolResult` for all tool return values
- Prefer `async` for I/O-bound operations
- Follow existing type annotation style
- Update `docs/` for architectural changes
