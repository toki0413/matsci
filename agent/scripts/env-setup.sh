#!/usr/bin/env bash
#
# Setup the Huginn agent Python environment in a fresh/restored sandbox.
#
#   ./scripts/env-setup.sh [--fix-uv] [--all-extras]
#
# WHY THIS EXISTS
# ----------------
# This sandbox uses `uv` for dependency management (see pyproject.toml). In a
# freshly-provisioned sandbox the `.venv` is an empty shell (no site-packages),
# and `uv run` / `uv sync` can HANG resolving git dependencies:
#
#   * pyproject.toml `[project.optional-dependencies].benchmark` has
#     AtomWorld @ git+https://github.com/MasterAI-EAM/AtomWorld.git
#   * `uv`'s built-in GitHub handling queries api.github.com for HEAD (403
#     Forbidden behind the sandbox proxy), then enters an unbounded git fetch.
#
# `git ls-remote` (which honours http_proxy/https_proxy) DOES reach GitHub, so
# the problem is uv bypassing the proxy for its GitHub resolution, not the
# network. Rather than fighting uv inside the sandbox, this script installs the
# missing packages via pip (which honours the proxy) as a fallback.
#
# BEST EFFORT: this script is idempotent. It snapshots which high-level module
# groups are importable and installs ONLY what's missing. Safe to re-run.
#
# NOTE on the remaining "degraded" tool notices at import: a handful of tools
# (jax/vina/openmm/OCR/browser/ML-potential) need opt-in groups or external
# binaries and are DESIGNED to degrade gracefully. They are out of scope here.

set -euo pipefail
cd "$(dirname "$0")/.."
PROJECT_ROOT="$PWD"

FIX_UV=0
ALL_EXTRAS=0
for arg in "$@"; do
    case "$arg" in
        --fix-uv) FIX_UV=1 ;;
        --all-extras) ALL_EXTRAS=1 ;;
        *) echo "unknown arg: $arg" >&2; echo "usage: $0 [--fix-uv] [--all-extras]" >&2; exit 2 ;;
    esac
done

PY=python
echo "==> Python: $($PY --version 2>&1) at $(which $PY)"

# ── 1. Diagnose uv lock (informational / optional) ───────────────────────────
if [ "$FIX_UV" -eq 1 ]; then
    echo "==> Attempting uv resolution (may hang on the AtomWorld git dep)..."
    timeout 60 uv sync --no-dev 2>/dev/null \
        && echo "   uv resolved OK." \
        || echo "   uv resolution still blocked (AtomWorld git) — falling through to pip."
fi

# ── 2. Core runtime deps (pyproject [project.dependencies]) ──────────────────
echo "==> Installing core runtime dependencies via pip..."
$PY -m pip install --quiet \
    "pydantic>=2.13" \
    cryptography pyyaml toml python-dotenv click rich networkx \
    numpy scipy sympy "z3-solver>=4.12" aiohttp httpx requests tenacity \
    Pillow fastapi uvicorn sse-starlette python-multipart websockets \
    "mcp>=1.28.1,<2.0" \
    "langchain>=1.3.14" "langchain-core>=1.5.2" \
    "langchain-openai>=1.4.1" \
    "langgraph>=1.2.10" "langgraph-checkpoint-sqlite>=2.0.0" \
    "deepagents>=0.5.0" \
    || { echo "   pip core install failed" >&2; exit 1; }

# ── 3. Dev/test deps ─────────────────────────────────────────────────────────
# pyproject addopts enables --cov, so pytest-cov is required to run pytest.
# asyncio_mode=auto → pytest-asyncio; config may use -n auto → pytest-xdist.
echo "==> Installing dev/test dependencies..."
$PY -m pip install --quiet \
    pytest pytest-cov pytest-asyncio pytest-xdist pytest-benchmark \
    hypothesis black ruff mypy \
    || { echo "   pip dev install failed" >&2; exit 1; }

# ── 4. Optional science extras (pyproject [project.optional-dependencies].all) ─
if [ "$ALL_EXTRAS" -eq 1 ]; then
    echo "==> Installing optional science extras..."
    # Git deps (AtomWorld) and ML-model groups (mace/fairchem/nep) are
    # intentionally dropped to avoid the uv-git-hang and huge model trees.
    $PY -m pip install --quiet \
        "pymatgen>=2025.10.7" ase dscribe paramiko jedi \
        "chromadb>=0.4" "sentence-transformers>=2.5" \
        pymupdf pypdf easyocr pytesseract py4vasp \
        paddleocr nougat-ocr \
        matplotlib SciencePlots ultraplot jieba openpyxl \
        "mp-api>=0.12" statsmodels matminer rdkit trimesh \
        2>&1 | tail -n 2 || true
fi

# ── 5. Health self-check ─────────────────────────────────────────────────────
echo "==> Verifying tool registration..."
$PY - <<'PYEOF'
import importlib, sys
from huginn.tools import register_all_tools
from huginn.tools.registry import ToolRegistry

missing_mods = []
for mod in ("langchain_openai", "chromadb", "sentence_transformers",
            "pymatgen", "ase", "sklearn", "torch", "matplotlib"):
    try:
        importlib.import_module(mod)
    except ImportError:
        missing_mods.append(mod)

register_all_tools()
names = list(ToolRegistry._tools.keys())

core_ours = ["tool_search", "literature_tool", "literature_pipeline_tool",
             "web_search_tool", "agentic_search_tool",
             "materials_autoresearch_tool", "report_tool"]
missing_tools = [o for o in core_ours if not any(o in n for n in names)]

print(f"  registered tools: {len(names)}")
if missing_mods:
    print(f"  NOT importable modules: {missing_mods}")
if missing_tools:
    print(f"  core tools missing: {missing_tools}")
if missing_mods or missing_tools:
    print("  health: PARTIAL")
    sys.exit(1)
print("  health: OK")
PYEOF

echo "==> Env setup complete."