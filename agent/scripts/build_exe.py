#!/usr/bin/env python3
"""Build Huginn as a single executable .exe for Windows.

Usage:
    uv run python scripts/build_exe.py
    # Output: dist/huginn-agent.exe

Inspired by Claude Code and Qoder distribution model.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DIST_DIR = PROJECT_ROOT / "dist"
BUILD_DIR = PROJECT_ROOT / "build"


def clean():
    """Remove previous build artifacts."""
    for d in (DIST_DIR, BUILD_DIR):
        if d.exists():
            shutil.rmtree(d)
            print(f"Cleaned {d}")


def build():
    """Run PyInstaller to build the single executable."""
    entry_point = "huginn.cli:main"
    output_name = "huginn-agent"

    # Assets to bundle
    assets_src = PROJECT_ROOT / "huginn" / "assets"
    assets_dst = "huginn/assets"

    cmd = [
        sys.executable,
        "-m", "PyInstaller",
        "--onedir",                # Directory mode for Tauri resources
        "--name", output_name,
        # "--clean",                 # Skip clean to avoid locked files
        "--noconfirm",
        # Add data files
        *(f"--add-data={assets_src}{os.pathsep}{assets_dst}".split() if assets_src.exists() else []),
        # Hidden imports for dynamic loading
        "--hidden-import=huginn.tools.bash_tool",
        "--hidden-import=huginn.tools.code_tool",
        "--hidden-import=huginn.tools.file_edit_tool",
        "--hidden-import=huginn.tools.file_read_tool",
        "--hidden-import=huginn.tools.file_write_tool",
        "--hidden-import=huginn.tools.git_tool",
        "--hidden-import=huginn.tools.bourbaki_tool",
        "--hidden-import=huginn.tools.diff_tool",
        "--hidden-import=huginn.tools.validate_tool",
        "--hidden-import=huginn.tools.diagnose_tool",
        "--hidden-import=huginn.tools.extract_tool",
        "--hidden-import=huginn.tools.job_tool",
        "--hidden-import=huginn.tools.database_tool",
        "--hidden-import=huginn.tools.potential_tool",
        "--hidden-import=huginn.tools.structure_tool",
        "--hidden-import=huginn.tools.report_tool",
        "--hidden-import=huginn.tools.orchestrate_tool",
        "--hidden-import=huginn.tools.memory_tool",
        "--hidden-import=huginn.tools.lean_tool",
        "--hidden-import=huginn.tools.skill_tool",
        "--hidden-import=huginn.tools.evidence_fusion_tool",
        "--hidden-import=huginn.tools.tda_tool",
        "--hidden-import=huginn.tools.unit_tool",
        "--hidden-import=huginn.tools.numerical_tool",
        "--hidden-import=huginn.tools.high_throughput_tool",
        "--hidden-import=huginn.tools.symmetry_tool",
        "--hidden-import=huginn.tools.gp_tool",
        "--hidden-import=huginn.tools.uq_tool",
        "--hidden-import=huginn.tools.descriptor_tool",
        "--hidden-import=huginn.tools.autodiff_tool",
        "--hidden-import=huginn.tools.symbolic_regression_tool",
        "--hidden-import=huginn.tools.symbolic_math_tool",
        "--hidden-import=huginn.tools.visualize_tool",
        "--hidden-import=huginn.tools.active_learning_tool",
        "--hidden-import=huginn.tools.ml_potential_tool",
        "--hidden-import=huginn.tools.characterization_tool",
        "--hidden-import=huginn.tools.experimental_data_tool",
        "--hidden-import=huginn.tools.materials_database_tool",
        "--hidden-import=huginn.security.safe_eval",
        "--hidden-import=huginn.security.math_eval",
        "--hidden-import=huginn.skills.presets",
        "--hidden-import=huginn.workflows.high_throughput",
        "--hidden-import=huginn.utils.units",
        "--hidden-import=huginn.utils.numerical",
        "--hidden-import=huginn.utils.conversation_tree",
        "--hidden-import=huginn.utils.prompt_cache",
        "--hidden-import=huginn.utils.tokens",
        "--hidden-import=huginn.coder.checkpoint",
        "--hidden-import=huginn.coder.test_runner",
        "--hidden-import=huginn.mcp_client",
        "--hidden-import=huginn.crypto",
        "--hidden-import=huginn.privacy.scanner",
        "--hidden-import=huginn.rag.encrypted_rag",
        "--hidden-import=huginn.server",
        "--hidden-import=langchain",
        "--hidden-import=langchain_core",
        "--hidden-import=langgraph",
        "--hidden-import=click",
        "--hidden-import=rich",
        "--hidden-import=cryptography",
        "--hidden-import=aiohttp",
        "--hidden-import=pydantic",
        "--hidden-import=numpy",
        "--hidden-import=networkx",
        "--hidden-import=yaml",
        "--hidden-import=dotenv",
        # Collect all huginn subpackages
        "--collect-all=huginn",
        # Exclude heavy optional deps to reduce size
        "--exclude-module=pytest",
        "--exclude-module=pytest_asyncio",
        "--exclude-module=pytest_benchmark",
        "--exclude-module=pytest_cov",
        "--exclude-module=black",
        "--exclude-module=ruff",
        "--exclude-module=mypy",
        "--exclude-module=pre_commit",
        "--exclude-module=pip_audit",
        "--exclude-module=memory_profiler",
        # Entry script
        str(PROJECT_ROOT / "scripts" / "entry.py"),
    ]

    print("Building Huginn executable...")
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print("Build failed!")
        sys.exit(1)

    exe_path = DIST_DIR / f"{output_name}.exe"
    if exe_path.exists():
        size_mb = exe_path.stat().st_size / (1024 * 1024)
        print(f"\n✓ Built: {exe_path}")
        print(f"  Size: {size_mb:.1f} MB")
        print(f"\nUsage:")
        print(f"  .\\dist\\{output_name}.exe --help")
        print(f"  .\\dist\\{output_name}.exe coder \"fix bug\"")
    else:
        print(f"\n⚠ Expected output not found: {exe_path}")
        sys.exit(1)


if __name__ == "__main__":
    # Skip clean to avoid locked files from previous runs
    # clean()
    build()
