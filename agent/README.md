# Huginn (agent package `huginn-agent`)

An LLM-driven agent system for **general scientific research**. It started in
computational materials science and has since generalized into a multi-domain
research automation agent: DFT / molecular dynamics / CFD-FEA simulation,
symbolic regression, causal analysis (TDA/SINDy), document retrieval, autonomous
design-space exploration, and multi-agent research collaboration — all with Lean 4
formal verification of the underlying math.

> 定位演进：**并非"材料科学专用"**。材料/仿真只是能力集之一；整系统重心是
> 通用科研自动化（研究项目、多智能体 team、知识蒸馏、因果/结构分析、MCP 生态、
> 远程/HPC 调度）。材料域能力见下方 Features 的 Simulation 条目。

> 本 README 是 `agent/` 子包文档。**全项目文档导航见 [docs/INDEX.md](docs/INDEX.md)**，
> 根级项目总览见 [../README.md](../README.md)。

## Features

- **DFT Automation**: INCAR generation, relaxation, static, DOS, and band structure calculations via VASP
- **Molecular Dynamics**: Melt-quench, NPT/NVT simulations via LAMMPS with RDF and structure analysis
- **Symbolic Regression**: Discover analytical formulas from data via PSE/PSRN (Nature Computational Science)
- **Intelligent RAG**: Document retrieval with ChromaDB embeddings, keyword fallback, and encrypted storage
- **Exploration Engine**: Autonomous multi-objective optimization with LLM-driven branch generation
- **Memory System**: Three-tier memory (session, long-term SQLite+FTS5, auto-promotion) + knowledge distillation
- **Skills Framework**: Declarative material science workflows
- **MCP Integration**: Connect to Materials Project, NIST databases, and mathematical analysis tools
- **Report Generation**: Auto-generate Markdown/LaTeX/HTML reports from simulation results
- **Security**: AES-128 encryption at rest with per-item salt and memory-only keys; fail-closed tool metadata
- **Desktop App**: Tauri v2 + React 18 frontend (work in progress)
- **Coder Mode**: Autonomous code editing with read/write/edit, shell, git, and code execution tools
- **Multi-Agent**: Orchestrator, sub-agents, swarm/team collaboration
- **Causal & Autoloop**: Causal graph modeling, autonomous exploration loop, self-evolution
- **Physical-world access**: Experiment protocol orchestration via `PhysicalWorkspace`
  (time-reversible + spatially-composable + perception-confirmed), see `huginn/security/`
- **Unified explainability**: `explain()` facade in `huginn/explainability.py` assembles
  audit/provenance/event observations into an end-to-end explanation timeline
- **Unified gating**: `CoEffectRegistry` drives activation and aggregates decisions for all
  gates, see `huginn/security/gate.py`

## Key Concepts

两条正交控制轴（互不干扰，可独立设置）：

| 概念 | 控制什么 | 取值 | 环境变量 |
|---|---|---|---|
| **极简模式 ModelTier** | 认知编排开销（phase / plan / 纪律 / compaction / 外部思考） | `full` / `balanced` / `minimal` | `HUGINN_MODEL_TIER` |
| **思考强度 ThinkingIntensity** | 模型推理深度（provider reasoning budget） | `low` / `medium` / `high` / `max` | `HUGINN_THINKING` |

- 极简模式越"minimal"，越信任模型、跳过 phase/plan 门控；安全层始终保留。
- 思考强度映射到各 provider 的推理预算（如 Anthropic: 4096 / 16000 / 32000 / 64000）。
- 契约自动生成：`python -m huginn.cli.config_audit --<domain> --out docs/<domain>-contract.md`。

## Quick Start

### Installation

```bash
# Clone the repository
cd agent

# Install dependencies
pip install -e .
```

### Run the Agent

```bash
# CLI mode
python -m huginn.cli "Calculate the band gap of Si"

# API server
python -m huginn.server

# MCP servers: configured in the repo-root .mcp.json, registered internally at
# startup (huginn/lifespan.py) via huginn/tools/mcp_adapter.py — no manual
# `python servers/...` launch needed.
```

`huginn` exposes many subcommands via `huginn/cli/commands/`, e.g. `chat`,
`coder`, `serve`, `workflow`, `autoloop`, `swarm`, `hpc`, `explore`,
`scheduler`, `bench`, `diagnose`, `kg`, `sessions`, `scheduler`, `replay`,
`refactor`, `skills`, `tools`, plus `version`/`configure`. Run `huginn --help`
for the full list.

### Run Tests

```bash
pytest tests/ -x -v
```

## Coder Mode

Run an autonomous coding session (Codex-like) that can read, edit, write,
execute shell commands, inspect git state, and run Python snippets:

```bash
# One-shot task
huginn-agent coder "Add a docstring to huginn/tools/code_tool.py"

# Interactive mode
huginn-agent coder

# Auto-approve destructive actions (use with caution)
huginn-agent coder "Refactor the CLI" --auto-approve
```

Coder tools: `file_read_tool`, `file_write_tool`, `file_edit_tool`,
`bash_tool`, `git_tool`, `code_tool`.

## Architecture

See [docs/tech-spec.md](docs/tech-spec.md) (current factual record) and
[docs/architecture.md](docs/architecture.md) for detailed system design.

```
┌────────────┐   ┌────────────┐   ┌────────────┐
│ CLI (click)│   │ API Server │   │ Desktop App│
│ huginn.cli │   │ FastAPI +  │   │ Tauri+React│
└─────┬──────┘   │   WS/SSE   │   └─────┬──────┘
      │          └─────┬──────┘         │
      └────────────────┼────────────────┘
                       ▼
              ┌─────────────────┐
              │  Agent 层       │
              │ agent/ agents/  │  orchestrator, subagent, swarm,
              │                 │  speculator, loop_detector ...
              └────────┬────────┘
                       │
    ┌─────────┬────────┼────────┬──────────┬──────────┐
    ▼         ▼        ▼        ▼          ▼          ▼
  memory   evolution  tools   knowledge    kg       causality
 (3-tier)  (distill) (150+)  Cloud(KB)   graph      (SCM)
```

Real entry points: `huginn-agent` CLI (console script → `huginn.cli:main`) and
`python -m huginn.server` (FastAPI + WebSocket). All shared state lives in
`huginn/server_core.py`, lifecycle in `huginn/lifespan.py`, routes in
`huginn/routes/`.

> **单网关（Single Gateway）**：`huginn.server` 是唯一业务网关，外部消费者
> （CLI / 桌面 / 脚本）一律作为 HTTP/WS API 客户端，不直接 `import huginn.*`
> 业务模块。由 `tests/test_arch_single_gateway.py` 在 CI 强制。详见
> [../docs/architecture/decisions/0001-single-gateway.md](../docs/architecture/decisions/0001-single-gateway.md)。

## Tools

150+ built-in tools are registered through `huginn/tools/__init__.py`
(`_CORE_MODULES` ~50 lightweight + `_OPTIONAL_MODULES` heavy simulation/science).
Representative tools by category:

| Category | Tools |
|----------|-------|
| Coder / file | `bash_tool`, `code_tool`, `file_read/write/edit_tool`, `multi_edit_tool`, `glob_tool`, `grep_tool`, `git_tool`, `github_tool`, `diff_tool`, `eval_tool`, `validate_tool`, `diagnose_tool` |
| Sci / DFT | `vasp_tool`, `qe_tool`, `cp2k_tool`, `gaussian_tool`, `orca_tool`, `structure_tool`, `symmetry_tool`, `xrd_sim_tool` |
| Simulation | `lammps_tool`, `gromacs_tool`, `openmm_tool`, `openfoam_tool`, `comsol_tool`, `abaqus_tool`, `fenics_tool`, `elmer_tool`, `packing_tool`, `fep_tool`, `enhanced_sampling_tool` |
| Symbolic / math | `symbolic_regression_tool`, `symbolic_math_tool`, `discrete_smt/group/oeis/additive_tool`, `numerical_tool`, `unit_tool`, `autodiff_tool`, `lean_tool`, `bourbaki_tool`, `tensor_algebra` |
| Data / retrieval | `database_tool`, `report_tool`, `extract_tool`, `tool_search_tool`, `agentic_search_tool`, `web_search_tool`, `literature_tool`, `materials_database_tool`, `experimental_data_tool` |
| Memory / meta | `remember_tool`, `recall_tool`, `recall_context_tool`, `self_observe_tool`, `todo_tool`, `notebook_tool`, `scheduler_tool`, `plan_store_tool`, `prospective_tool` |
| Multi-agent | `subagent_tool`, `orchestrate_tool`, `review_committee_tool`, `skills_tool`, `workflow_tool` |
| Vision / ML | `visualize_tool`, `vision_describe_tool`, `image_analysis_tool`, `image_design_tool`, `model3d_tool`, `ml_potential_tool`, `active_learning_tool`, `interpretable_ml_tool`, `gnn_tool`, `vae_tool`, `transformer_tool` |

Full authoritative list of registered tool classes lives in
`huginn/tools/__init__.py::_CORE_MODULES` / `_OPTIONAL_MODULES`.

## Skills

Declarative workflows under `huginn/skills/` (e.g. `band_structure.md`,
`structure_relaxation.md`, `wavefunction_analysis.md`) plus `presets.py`:

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

Three-tier memory (`huginn/memory/`):

- **Session memory**: Current conversation context; model `reasoning_content`
  (COT) is persisted to `session.reasoning_trace` via `add_reasoning` for
  downstream distillation.
- **Long-term memory**: SQLite + FTS5 full-text search with importance scoring.
- **Auto-promotion / distillation**: Successful tool results auto-save to
  long-term memory; `huginn/evolution/knowledge_distiller.py` distills
  execution logs into knowledge whose `verification_status` is promoted to
  `confirmed` on verified successful use, then auto-ingests into the KB.

## Security

- Unified error envelope (`huginn_error_response` with `request_id`); every
  API endpoint guarded by `require_api_key`.
- AES-128-CBC + HMAC-SHA256 encryption at rest with per-item salt; decryption
  keys are memory-only and never written to disk.
- Fail-closed tool metadata (`is_read_only` / `is_destructive` /
  `requires_confirmation`).

## Development

### Project Structure

```
agent/
├── huginn/                 # Core package
│   ├── cli/                # click CLI (main.py + commands/)
│   ├── server.py           # FastAPI + WebSocket entry
│   ├── server_core.py      # shared app state
│   ├── lifespan.py         # startup/shutdown lifecycle
│   ├── routes/             # HTTP/WS route handlers (v1 + root compat)
│   ├── agent/  agents/     # agent loop, orchestrator, subagent, swarm
│   ├── tools/              # 150+ tool implementations + registry
│   ├── memory/             # session / long-term / manager
│   ├── evolution/          # knowledge distiller + evolution manager
│   ├── knowledge/  kg/     # knowledge base, auto-ingest, knowledge graph
│   ├── causal/  autoloop/  # causal modeling, autonomous exploration loop
│   ├── metacog/  runtime/  # meta-cognition, task lifecycle
│   ├── api/                # API layer (context/event/filter)
│   └── security/           # auth, middleware
├── tests/                  # pytest suite (conftest with isolation guards)
└── docs/                   # INDEX.md (导航), tech-spec.md (现状), architecture.md + 契约文档
```

### Adding a New Tool

1. Create a class inheriting from `HuginnTool` in `huginn/tools/`
2. Define `name`, `description`, and `input_schema` (Pydantic model)
3. Implement `_execute()`; `call()` returns `ToolResult`
4. Add the `(module, ClassName)` tuple to `_CORE_MODULES` or `_OPTIONAL_MODULES`
   in `huginn/tools/__init__.py`
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
