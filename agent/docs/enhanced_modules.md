# Enhanced Modules — Usage Guide

This document covers the modules added or enhanced in the latest round of
improvements, with practical usage examples for each.

## Table of Contents

- [Security: Safe Math Eval](#security-safe-math-eval)
- [NumericalTool](#numericaltool)
- [UnitTool](#unittool)
- [SymmetryTool](#symmetrytool)
- [DescriptorTool](#descriptortool)
- [AutoDiffTool](#autodifftool)
- [UQTool](#uqtool)
- [GPTool](#gptool)
- [Context Management](#context-management)
- [Token Counting](#token-counting)
- [Telemetry Memory Tracking](#telemetry-memory-tracking)
- [Conversation Branch Tree](#conversation-branch-tree)
- [High-Throughput Workflow](#high-throughput-workflow)

---

## Security: Safe Math Eval

Replaces raw `eval()` with an AST-walking evaluator that only permits
mathematical operations and a whitelist of numpy functions.

```python
from huginn.security.math_eval import safe_math_eval

# Basic arithmetic
safe_math_eval("1 + 2 * 3")  # -> 7

# Numpy functions (whitelisted)
safe_math_eval("np.sin(np.pi / 2)", {})  # -> 1.0

# With variables
safe_math_eval("x**2 + 2*x + 1", {"x": 3})  # -> 16

# Subscripts (for ODE/optimization expressions)
safe_math_eval("X[0]**2 + X[1]**2", {"X": [3, 4]})  # -> 25

# Rejected: __import__, open(), attribute access, lambda
safe_math_eval("__import__('os').system('ls')")  # raises SafeEvalError
```

## NumericalTool

Unified interface to scipy/numpy solvers. All expressions are evaluated
through `safe_math_eval` — no raw `eval()`.

```python
from huginn.tools.numerical_tool import NumericalTool

tool = NumericalTool()

# Root finding
result = await tool.call({"action": "root", "func": "x**2 - 4", "x0": 1.0})
# -> root ≈ 2.0

# ODE integration (exponential decay)
result = await tool.call({
    "action": "ode",
    "func": "-y[0]",
    "t_span": [0.0, 5.0],
    "y0": [1.0],
    "t_eval": [0, 1, 2, 3, 4, 5],
})

# Constrained optimization (new)
result = await tool.call({
    "action": "constrained_minimize",
    "func": "(X[0]-3)**2 + (X[1]+1)**2",
    "x0": [0.0, 0.0],
    "bounds": [[0, 5], [-5, 0]],
})

# SVD (new)
result = await tool.call({
    "action": "svd",
    "A": [[1, 2], [3, 4], [5, 6]],
})

# Matrix exponential (new)
result = await tool.call({
    "action": "matrix_exp",
    "A": [[0, 1], [-1, 0]],  # rotation generator
})
```

## UnitTool

Unit conversion and dimension checking with pint (fallback registry if
pint is unavailable).

```python
from huginn.tools.unit_tool import UnitTool

tool = UnitTool()

# Basic conversion
result = await tool.call({
    "action": "convert",
    "value": 1.0,
    "from_unit": "eV",
    "to_unit": "J",
})

# Infer dimension from expression (new)
result = await tool.call({
    "action": "infer_dimension",
    "expression": "m * a",
    "variables": {"m": "kg", "a": "m/s**2"},
})
# -> result_dimension: "[length] [mass] / [time]^2"

# Unit-aware arithmetic (new)
result = await tool.call({
    "action": "unit_arithmetic",
    "operation": "multiply",
    "value1": 2.0, "unit1": "N",
    "value2": 3.0, "unit2": "m",
})
# -> result_unit: "N*m"

# Natural unit conversion (new)
result = await tool.call({
    "action": "natural_units",
    "value": 1.0,
    "from_system": "atomic",
    "to_system": "si",
    "quantity": "energy",
})
# -> ~4.36e-18 J (1 Hartree)
```

## SymmetryTool

Crystal symmetry analysis via pymatgen/spglib.

```python
from huginn.tools.symmetry_tool import SymmetryTool

tool = SymmetryTool()

# Basic analysis
result = await tool.call({
    "action": "analyze",
    "structure": {
        "lattice": {"a": 2.87, "b": 2.87, "c": 2.87,
                     "alpha": 90, "beta": 90, "gamma": 90},
        "sites": [{"element": "Fe", "abc": [0, 0, 0]}],
    },
})

# Subgroup analysis (new)
result = await tool.call({
    "action": "subgroups",
    "structure": {...},
    "index": 2,
})

# Wyckoff position splitting (new)
result = await tool.call({
    "action": "wyckoff_split",
    "structure": {...},
    "subgroup_number": 123,
})

# Magnetic site detection (new)
result = await tool.call({
    "action": "magnetic",
    "structure": {...},
})
# -> [{element: "Fe", moment: 2.2, ...}]
```

## DescriptorTool

Materials descriptors for ML pipelines. New actions: `matminer`,
`mbtr`, `acsf`, `coulomb_matrix`.

```python
from huginn.tools.descriptor_tool import DescriptorTool

tool = DescriptorTool()

# Composition descriptors (existing)
result = await tool.call({"action": "composition", "formula": "Fe2O3"})

# matminer element properties (new, requires matminer)
result = await tool.call({"action": "matminer", "formula": "Fe2O3"})

# MBTR descriptor (new, requires dscribe)
result = await tool.call({
    "action": "mbtr",
    "structure": {...},
    "k": 2,
    "n_grid": 100,
})

# ACSF descriptor (new, requires dscribe)
result = await tool.call({
    "action": "acsf",
    "structure": {...},
    "rcut": 6.0,
})

# Coulomb matrix (new, requires dscribe)
result = await tool.call({
    "action": "coulomb_matrix",
    "structure": {...},
    "n_atoms_max": 50,
})
```

## AutoDiffTool

Automatic differentiation with JAX (finite difference fallback).

```python
from huginn.tools.autodiff_tool import AutoDiffTool

tool = AutoDiffTool()

# Gradient (JAX or finite difference)
result = await tool.call({
    "action": "gradient",
    "function": "birch_murnaghan",
    "variables": {"V": [100.0], "E0": [-10.0]},
    "function_params": {"B0": 100, "B0p": 4, "V0": 110},
})

# Jacobian (now uses JAX jacfwd, new)
result = await tool.call({
    "action": "jacobian",
    "function": "custom",
    "variables": {"x0": [1.0], "x1": [2.0]},
})

# Optimization with L-BFGS (new)
result = await tool.call({
    "action": "optimize",
    "function": "morse",
    "variables": {"r": [2.0]},
    "function_params": {"D": 1.0, "a": 1.5, "re": 1.0},
    "method": "lbfgs",
    "bounds": [[0.5, 5.0]],
})

# SLSQP with constraints (new)
result = await tool.call({
    "action": "optimize",
    "function": "custom",
    "variables": {"x0": [1.0], "x1": [1.0]},
    "method": "slsqp",
    "bounds": [[0, 10], [0, 10]],
    "constraints": [{"type": "ineq", "fun": "X[0] + X[1] - 1"}],
})
```

## UQTool

Uncertainty quantification. New actions: `pce`, `morris`.

```python
from huginn.tools.uq_tool import UQTool

tool = UQTool()

# Monte Carlo propagation
result = tool.call({
    "action": "monte_carlo",
    "func": "a + b",
    "inputs": [
        {"name": "a", "dist": "normal", "params": [1.0, 0.1]},
        {"name": "b", "dist": "normal", "params": [2.0, 0.2]},
    ],
    "n_samples": 10000,
})

# Polynomial Chaos Expansion (new)
result = tool.call({
    "action": "pce",
    "func": "a + b",
    "inputs": [
        {"name": "a", "dist_type": "uniform", "params": [0, 1]},
        {"name": "b", "dist_type": "uniform", "params": [0, 1]},
    ],
    "order": 3,
})

# Morris elementary effects (new)
result = tool.call({
    "action": "morris",
    "func": "x0 + 2*x1",
    "inputs": [
        {"name": "x0", "lower": 0, "upper": 1},
        {"name": "x1", "lower": 0, "upper": 1},
    ],
    "r": 10,
})
# -> mu_star ≈ [1, 2]
```

## GPTool

Gaussian process regression and Bayesian optimization. New: Matérn
kernels, UCB/PI acquisition functions.

```python
from huginn.tools.gp_tool import GPTool

tool = GPTool()

# Fit with Matérn 3/2 kernel (new)
result = tool.call({
    "action": "fit",
    "X": [[0], [1], [2], [3]],
    "y": [0, 1, 4, 9],
    "kernel": "matern32",
})

# Suggest with UCB acquisition (new)
result = tool.call({
    "action": "suggest",
    "X_train": [[0], [1], [2], [3]],
    "y_train": [0, 1, 4, 9],
    "X_candidates": [[0.5], [1.5], [2.5]],
    "acquisition": "ucb",
    "kappa": 2.0,
})
```

## Context Management

### compact_messages (O(n) optimization)

Previously O(n²) — recalculated total tokens after each pop. Now
pre-computes per-message tokens and subtracts in a single pass.

```python
from huginn.utils.context import compact_messages

messages = [...]  # large message list
budget = 8000
compact = compact_messages(messages, budget, keep_last_n=4)
```

### summarize_compact_messages (summary length cap)

Summaries are now capped at 2000 tokens. When the cap is exceeded, the
summarizer re-compresses the combined summary automatically.

```python
from huginn.utils.context import summarize_compact_messages

compacted, summary = await summarize_compact_messages(
    messages,
    budget=8000,
    keep_last_n=4,
    summarizer=my_async_summarizer,
    existing_summary=previous_summary,
)
```

## Token Counting

Model-aware tokenizer selection (new):

```python
from huginn.utils.tokens import count_tokens, count_message_tokens

# Backward compatible (cl100k_base)
count_tokens("hello world")  # -> 2

# Model-specific encoding
count_tokens("hello world", model_name="gpt-4o")  # -> 3 (o200k_base)
count_tokens("你好世界", model_name="gpt-4o")  # CJK-aware

# Message-level
count_message_tokens("hello", model_name="o1")  # -> 5 (1 + 4 overhead)
```

Supported encodings:
- `o200k_base`: GPT-4o, o1, o3 series
- `cl100k_base`: GPT-4, GPT-3.5, Claude, Kimi, Deepseek (default)

## Telemetry Memory Tracking

Telemetry spans now record process RSS at start and end:

```python
from huginn.telemetry import TelemetryCollector

collector = TelemetryCollector()

with collector.span("heavy_computation") as span:
    # ... do work ...
    pass

# Span now has memory_start_mb, memory_end_mb, memory_peak_mb
print(span.memory_peak_mb)

# Snapshot current memory
snapshot = collector.memory_snapshot()
# -> {"rss_mb": 256.3, "traced_current_mb": ..., "traced_peak_mb": ...}

# Summary includes memory stats
summary = collector.summary()
# -> {"by_name": {"heavy_computation": {
#       "count": 1, "avg_duration_ms": ...,
#       "avg_memory_delta_mb": 12.5, "max_memory_peak_mb": 268.8
# }}}
```

## Conversation Branch Tree

Fork and backtrack conversations for multi-hypothesis exploration:

```python
# In HuginnAgent:
agent.fork_conversation()       # Fork from current position
agent.switch_branch(node_id)    # Switch to a different branch
agent.conversation_branches()   # List all branches

# ToolMessage metadata is now preserved in the tree, so tool call
# chains are correctly reconstructed when replaying history.
```

## High-Throughput Workflow

Parameter sweeps over any registered tool:

```python
from huginn.tools.high_throughput_tool import HighThroughputTool

tool = HighThroughputTool()

# Grid sweep
result = await tool.call({
    "tool_name": "vasp_tool",
    "space_type": "grid",
    "parameter_space": {
        "encut": [400, 500, 600],
        "kspacing": [0.03, 0.04],
    },
    "base_input": {"structure": {...}},
    "max_parallel": 4,
})

# Latin hypercube sampling
result = await tool.call({
    "tool_name": "lammps_tool",
    "space_type": "lhs",
    "parameter_space": {
        "temperature": [300, 1500],
        "pressure": [0, 100000],
    },
    "n_samples": 20,
    "base_input": {...},
})
```

## EvidenceFusionTool (Dempster-Shafer)

Multi-source evidence fusion using Dempster-Shafer theory. Combines
uncertain evidence from DFT, MD, experiments, and literature.

```python
from huginn.tools.evidence_fusion_tool import EvidenceFusionTool

tool = EvidenceFusionTool()

# Combine evidence from multiple simulation methods
result = await tool.call({
    "action": "combine",
    "evidence": [
        {"hypotheses": ["stable"], "mass": 0.7, "source": "DFT"},
        {"hypotheses": ["stable", "metastable"], "mass": 0.5, "source": "MD"},
        {"hypotheses": ["unstable"], "mass": 0.3, "source": "experiment"},
    ],
})
# -> combined mass, belief, plausibility, conflict level

# Pignistic transform for decision-making
result = await tool.call({
    "action": "pignistic",
    "evidence": [
        {"hypotheses": ["stable"], "mass": 0.6},
        {"hypotheses": ["stable", "metastable"], "mass": 0.3},
    ],
})
# -> BetP(stable) ≈ 0.75, BetP(metastable) ≈ 0.15, BetP(unstable) ≈ 0.10

# Weighted combination (sources with different reliability)
result = await tool.call({
    "action": "weighted_combine",
    "evidence": [
        {"hypotheses": ["stable"], "mass": 0.8, "source": "DFT", "weight": 0.9},
        {"hypotheses": ["stable"], "mass": 0.5, "source": "ML", "weight": 0.6},
    ],
})
```

## TDATool (Persistent Homology)

Topological data analysis for energy landscapes, crystal structures, and
point clouds. Uses ripser/gudhi with scipy fallback.

```python
from huginn.tools.tda_tool import TDATool

tool = TDATool()

# Persistence diagram from point cloud
import numpy as np
theta = np.linspace(0, 2*np.pi, 20)
circle = [[np.cos(t), np.sin(t)] for t in theta]
result = await tool.call({
    "action": "persistence_diagram",
    "point_cloud": circle,
    "max_dim": 1,
})
# -> H1 has a persistent feature (the circle's hole)

# Energy landscape topology
result = await tool.call({
    "action": "energy_landscape_topology",
    "energies": [-5.0, -4.8, -3.0, -2.9, 0.5],
    "structures": [[0,0,0], [0.1,0,0], [5,0,0], [5.1,0,0], [10,0,0]],
})
# -> n_basins=2, n_pathways=0, basin_sizes=[2,2,1]

# Crystal structure topology
result = await tool.call({
    "action": "structure_topology",
    "structure": {
        "lattice": {"a": 3.0, "b": 3.0, "c": 3.0,
                     "alpha": 90, "beta": 90, "gamma": 90},
        "sites": [
            {"element": "Si", "abc": [0, 0, 0]},
            {"element": "Si", "abc": [0.25, 0.25, 0.25]},
        ],
    },
    "radii": [1.0, 1.5, 2.0, 2.5, 3.0],
})
# -> Betti numbers at each radius reveal connectivity evolution

# Bottleneck distance between two diagrams
result = await tool.call({
    "action": "bottleneck_distance",
    "diagram": [{"dim": 0, "birth": 0.0, "death": 1.5}],
    "diagram2": [{"dim": 0, "birth": 0.0, "death": 1.3}],
})
```

## Information Geometry (GPTool)

Natural gradient optimization, Fisher information for experimental design,
and KL divergence between GP posteriors.

```python
from huginn.tools.gp_tool import GPTool

tool = GPTool()

# Natural gradient hyperparameter optimization
result = tool.call({
    "action": "natural_gradient",
    "X": [[0], [0.5], [1], [1.5], [2]],
    "y": [0, 0.25, 1, 2.25, 4],
    "n_steps": 20,
    "lr": 0.001,
})
# -> optimized (sigma_f, length_scale, sigma_n), log-lik trajectory

# Fisher information for experimental design
result = tool.call({
    "action": "fisher_information",
    "X_train": [[0], [1], [2]],
    "y_train": [0, 1, 4],
    "X_candidates": [[0.5], [1.5], [3]],
})
# -> D-optimal, A-optimal, E-optimal criteria, per-candidate contribution

# KL divergence between two GP posteriors
result = tool.call({
    "action": "kl_divergence",
    "X": [[0], [1], [2], [3]],
    "y1": [0, 1, 4, 9],
    "y2": [0, 1.2, 3.8, 9.1],
})
# -> mean_kl, max_kl, per_point_kl
```

## IBP Nonparametric Feature Discovery (DescriptorTool)

Indian Buffet Process for discovering unknown number of latent features
in materials data.

```python
from huginn.tools.descriptor_tool import DescriptorTool

tool = DescriptorTool()

# Discover latent features in composition-property data
result = await tool.call({
    "action": "ibp",
    "data": [
        [1.2, 0.5, 3.1, 0.8, 2.0],  # sample 1
        [0.8, 0.3, 2.9, 0.6, 1.8],  # sample 2
        # ... more samples
    ],
    "alpha": 1.0,
    "n_iterations": 100,
    "n_init_features": 5,
    "beta": 0.5,
    "seed": 42,
})
# -> n_features, Z (binary assignment), W (weights),
#    feature_importance, feature_interpretation (top-3 observed features per latent)
```
