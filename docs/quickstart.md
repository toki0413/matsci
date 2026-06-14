# Huginn Quick Start

This guide walks you through running Huginn locally — from setup to your first formally verified FEM weak form derivation.

---

## 1. Installation

```bash
# Clone
cd matsci-agent/agent

# Python dependencies
pip install -e .

# Optional: for config file support
pip install toml

# Lean 4 (required for formal verification)
# Install via elan: https://github.com/leanprover/elan
# Verify:
lake --version
```

---

## 2. Configure Your LLM

Huginn supports **9 providers** including local endpoints. For a fully offline setup:

### Option A: Ollama (recommended for beginners)

```bash
# Install Ollama: https://ollama.com
ollama pull qwen2.5:14b

# Configure
huginn configure
# Provider: ollama
# Model: qwen2.5:14b
# Ollama host: http://localhost:11434
```

### Option B: vLLM / LM Studio

```bash
huginn chat --provider vllm \
  --base-url http://localhost:8000/v1 \
  --model llama-3.1-8b
```

Local endpoints **do not require a real API key** — a dummy key is sent automatically.

---

## 3. Run the Pipeline Demo (No LLM Required)

To verify that symbolic derivation → Lean formalization works on your machine:

```bash
cd agent
PYTHONIOENCODING=utf-8 python demo_comprehensive.py
```

Expected output:

```
============================================================
Huginn Comprehensive Pipeline Demo
============================================================

--- Phase 1: FEM Weak Forms ---
  [Heat Conduction] PASS (0.51s)
  [Linear Elasticity] PASS (0.68s)
  [Bar Element] PASS (0.52s)

--- Phase 2: Tensor Algebra ---
  [Tensor] PASS

--- Phase 3: Numerical Linear Algebra ---
  [LA] PASS

============================================================
Results: 5/5 pipelines passed
============================================================
```

This demonstrates:
- **SymPy** derives weak forms from strong forms
- **Lean 4** compiles the symbolic expressions into verified `Float` definitions
- The entire bridge from calculus to type-checked proof is automated

---

## 4. Interactive Chat

```bash
huginn chat
```

Try asking:

```
> Derive the weak form for 1D heat conduction and verify it in Lean
```

The agent will:
1. Call `symbolic_math_tool` with `action=weak_form`, `target=heat_conduction`
2. Receive the bilinear form `k*ux*vx` and linear functional `f*v`
3. Automatically route to `lean_tool` with `auto_verify_action=fem`
4. Generate Lean code, compile with `lake build`, and report success

---

## 5. Architecture at a Glance

```
User Input
    │
    ▼
┌─────────────────┐     ┌─────────────────┐
│  SymbolicMath   │────▶│    LeanTool     │
│  (SymPy)        │     │  (Lean 4 + Lake)│
└─────────────────┘     └─────────────────┘
        │                       │
        ▼                       ▼
   weak_form terms         type-checked
   bilinear_form           Float definitions
   linear_functional
```

**Key insight:** The agent does not "trust" the LLM's math. Every symbolic result is translated into Lean 4 and must pass the type checker before being presented to the user.

---

## 6. Troubleshooting

| Issue | Fix |
|-------|-----|
| `UnicodeEncodeError` on Windows | Run with `PYTHONIOENCODING=utf-8` |
| `lake` not found | Install Lean 4 via `elan` |
| Lean build timeout | First build is slow; keep `build/` for caching |
| VASP/LAMMPS not found | Tools fall back to mock mode automatically |
| API key errors for local models | Ensure `--base-url` points to `localhost` or `127.*` |

---

## 7. Next Steps

- **Explore workflows**: `huginn/workflows/templates.py` contains 12 preset pipelines
- **Add a Lean module**: See `agent/lean/HuginnLean/README.md`
- **Run the test suite**: `pytest tests/ -x -q`
- **Read the threat model**: `docs/threat_model.md`
