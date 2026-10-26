#!/usr/bin/env python3
"""Build Huginn as a single executable .exe using a clean temporary venv.

This avoids analyzing the full 245-package dev environment, which causes
PyInstaller to hang. Only core dependencies are installed, cutting analysis
time from 10+ min to ~1 min.

Usage:
    uv run python scripts/build_exe_clean.py
    # Output: dist/huginn-agent.exe
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DIST_DIR = PROJECT_ROOT / "dist"
BUILD_DIR = PROJECT_ROOT / "build"

CORE_DEPS = [
    "pydantic>=2.0",
    "langchain>=0.3.0",
    "langchain-core>=0.3.0",
    "langgraph>=0.2.0",
    "deepagents>=0.5.0",
    "click>=8.0",
    "rich>=13.0",
    "networkx>=3.0",
    "numpy>=1.24",
    "aiohttp>=3.9",
    "python-dotenv>=1.0",
    "cryptography>=42.0",
    "pyyaml>=6.0",
    "mcp>=1.0",  # MCP SDK
    "fastapi>=0.110",  # Server
    "uvicorn>=0.25",  # Server
    "websockets>=12.0",  # Server
    "httpx>=0.27",  # HTTP client (used by many tools)
    "psutil>=5.9",  # System info
    "pyinstaller>=6.0",  # Self-build
]

EXCLUDES = [
    # Dev / test tools
    "pytest",
    "_pytest",
    "coverage",
    "mypy",
    "black",
    "ruff",
    "pre_commit",
    "pip_audit",
    "memory_profiler",
    # Heavy optional sci-libs (not used by core coder/chat/serve)
    "scipy",
    "matplotlib",
    "matplotlib.pyplot",
    "sklearn",
    "sklearn.cluster",
    "sklearn.metrics",
    "sklearn.linear_model",
    "sklearn.neighbors",
    "seaborn",
    "plotly",
    "bokeh",
    # Material science optional (not in core deps)
    "pymatgen",
    "ase",
    "dscribe",
    "paramiko",
    # RAG optional (not in core deps)
    "chromadb",
    "sentence_transformers",
    "pymupdf",
    "PyPDF2",
    "easyocr",
    "pytesseract",
    # HPC / container optional
    "apptainer",
    "docker",
    "paramiko",
    # Unused heavy libs
    "torch",
    "tensorflow",
    "jax",
    "jaxlib",
    "flax",
    "optax",
    "transformers",
    "accelerate",
    "datasets",
    # Unused in core
    "pillow",
    "PIL",
    "PIL.Image",
    "PIL.ImageFilter",
    "gi",
    "gobject",
    "cairo",
    "gtk",
    "wx",
    "PyQt5",
    "PyQt6",
    "PySide2",
    "PySide6",
]

_HIDDEN_IMPORTS = [
    "--hidden-import=huginn.tools.bash_tool",
    "--hidden-import=huginn.tools.code_tool",
    "--hidden-import=huginn.tools.file_edit_tool",
    "--hidden-import=huginn.tools.file_read_tool",
    "--hidden-import=huginn.tools.file_write_tool",
    "--hidden-import=huginn.tools.git_tool",
    "--hidden-import=huginn.tools.bourbaki_tool",
    "--hidden-import=huginn.coder.checkpoint",
    "--hidden-import=huginn.coder.test_runner",
    "--hidden-import=huginn.mcp_client",
    "--hidden-import=huginn.crypto",
    "--hidden-import=huginn.privacy.scanner",
    "--hidden-import=huginn.rag.encrypted_rag",
    "--hidden-import=huginn.server",
]


def clean():
    for d in (DIST_DIR, BUILD_DIR, PROJECT_ROOT / "huginn-agent.spec"):
        if d.exists():
            if d.is_dir():
                shutil.rmtree(d)
            else:
                d.unlink()
            print(f"Cleaned {d}")


def _create_venv() -> tuple[Path, Path, Path]:
    """Create the isolated build venv; returns (temp_dir, pip_exe, python_exe)."""
    temp_dir = Path(tempfile.mkdtemp(prefix="huginn_build_"))
    venv_dir = temp_dir / "venv"
    print(f"Creating temporary venv: {venv_dir}")

    subprocess.run(
        [sys.executable, "-m", "venv", str(venv_dir)],
        check=True,
    )

    pip = venv_dir / "Scripts" / "pip.exe"
    python = venv_dir / "Scripts" / "python.exe"
    return temp_dir, pip, python


def _install_deps(pip: Path, python: Path) -> None:
    """Install core deps with uv (10x faster than pip), then huginn editable."""
    print("Installing core dependencies with uv (fast)...")
    uv_cmd = shutil.which("uv")
    if uv_cmd:
        subprocess.run(
            [uv_cmd, "pip", "install", "--python", str(python), "--quiet"] + CORE_DEPS,
            check=True,
        )
    else:
        subprocess.run(
            [str(pip), "install", "--quiet"] + CORE_DEPS,
            check=True,
        )

    print("Installing huginn in temp venv...")
    if uv_cmd:
        subprocess.run(
            [
                uv_cmd,
                "pip",
                "install",
                "--python",
                str(python),
                "-e",
                str(PROJECT_ROOT),
            ],
            check=True,
        )
    else:
        subprocess.run(
            [str(pip), "install", "-e", str(PROJECT_ROOT)],
            check=True,
        )


def _build_cmd(python: Path, output_name: str) -> list[str]:
    """Assemble the PyInstaller argv for the clean venv."""
    assets_src = PROJECT_ROOT / "huginn" / "assets"
    assets_dst = "huginn/assets"

    cmd = [
        str(python),
        "-m",
        "PyInstaller",
        "--onedir",  # Directory mode: fast build, fast startup
        "--name",
        output_name,
        "--clean",
        "--noconfirm",
        "--noupx",  # Skip UPX compression (saves time)
        *(
            f"--add-data={assets_src}{os.pathsep}{assets_dst}".split()
            if assets_src.exists()
            else []
        ),
        *sum([["--exclude-module", e] for e in EXCLUDES], []),
        *_HIDDEN_IMPORTS,
        "--collect-all=huginn",
        str(PROJECT_ROOT / "scripts" / "entry.py"),
    ]

    return [c for c in cmd if c]


def _report_output(output_name: str) -> None:
    """Verify output (onedir mode: exe is inside dist/huginn-agent/)."""
    exe_dir = DIST_DIR / output_name
    exe_path = exe_dir / f"{output_name}.exe"
    if exe_path.exists():
        dir_size = sum(f.stat().st_size for f in exe_dir.rglob("*") if f.is_file())
        dir_size_mb = dir_size / (1024 * 1024)
        print(f"\n{'=' * 50}")
        print(f"OK Built: {exe_path}")
        print(f"  Total size: {dir_size_mb:.1f} MB (directory)")
        print("\nUsage:")
        print(f"  .\\dist\\{output_name}\\{output_name}.exe --help")
        print(f'  .\\dist\\{output_name}\\{output_name}.exe coder "fix bug"')
        print(f"{'=' * 50}")
    else:
        print(f"\n⚠ Expected output not found: {exe_path}")
        sys.exit(1)


def build():
    temp_dir, pip, python = _create_venv()
    _install_deps(pip, python)

    output_name = "huginn-agent"
    cmd = _build_cmd(python, output_name)

    print("\nRunning PyInstaller with clean venv...")
    print(f"Command: {' '.join(cmd[:10])} ...")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print("Build failed!")
        sys.exit(1)

    _report_output(output_name)

    print(f"\nCleaning temp venv: {temp_dir}")
    shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    clean()
    build()
