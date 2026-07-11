"""PyInstaller build script for Huginn Agent complete toolkit exe.

Usage:
    python build_exe.py

Output:
    dist/huginn-agent/ directory containing huginn-agent.exe
"""

import subprocess
import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).parent.resolve()
    entry = str(project_root / "huginn" / "cli" / "main.py")
    dist_dir = str(project_root / "dist")
    build_dir = str(project_root / "build")

    # Packages with dynamic imports that need full collection
    collect_all = [
        "langchain", "langchain_core", "langgraph", "deepagents",
        "click", "rich", "pydantic", "pydantic_core",
        "numpy", "networkx", "aiohttp", "yaml", "dotenv",
        "cryptography", "mcp",
    ]

    # Optional scientific packages
    for pkg in ["pymatgen", "ase", "matplotlib", "sympy", "scipy"]:
        try:
            __import__(pkg)
            collect_all.append(pkg)
        except ImportError:
            pass

    # Hidden imports for huginn internal modules
    hidden = [
        "huginn.cli.main", "huginn.cli.context", "huginn.cli.availability",
        "huginn.cli.commands", "huginn.cli.lazy_loader", "huginn.cli.slash_commands",
        "huginn.cli.custom_commands", "huginn.cli.input_parser", "huginn.cli.design_system",
        "huginn.tools.base", "huginn.tools",
        "huginn.autoloop.engine", "huginn.autoloop.phase_gate",
        "huginn.autoloop.budget", "huginn.autoloop.campaign",
        "huginn.autoloop.dynamic_workflow", "huginn.autoloop.goal_scheduler",
        "huginn.autoloop.hypothesis_loop", "huginn.autoloop.plan_store",
        "huginn.autoloop.red_team",
        "huginn.tools.symbolic_math.tool", "huginn.tools.symbolic_math.pde",
        "huginn.tools.symbolic_math.variational", "huginn.tools.symbolic_math.diffgeo",
        "huginn.tools.symbolic_math.algebra", "huginn.tools.symbolic_math.calculus",
        "huginn.tools.symbolic_math.tensor", "huginn.tools.symbolic_math.fem",
        "huginn.tools.symbolic_math.physics", "huginn.tools.symbolic_math._parsers",
        "huginn.tools.sci.multi_fidelity_tool", "huginn.tools.wetlab_rpc_tool",
        "huginn.persistence.campaign", "huginn.scheduling.scheduler",
    ]

    # Auto-discover all tool modules
    tools_dir = project_root / "huginn" / "tools"
    for py_file in tools_dir.rglob("*.py"):
        if py_file.name.startswith("__"):
            continue
        rel = py_file.relative_to(project_root).with_suffix("")
        hidden.append(str(rel).replace("\\", ".").replace("/", "."))

    excludes = [
        "tkinter", "unittest", "test", "tests", "pytest", "hypothesis",
        "IPython", "jupyter", "notebook",
    ]

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name", "huginn-agent",
        "--noconfirm",
        "--clean",
        "--distpath", dist_dir,
        "--workpath", build_dir,
        "--specpath", str(project_root / "build"),
        "--paths", str(project_root),
        "--console",
        # onedir mode (faster startup than onefile for large deps)
    ]

    for pkg in collect_all:
        cmd.extend(["--collect-all", pkg])

    for mod in hidden:
        cmd.extend(["--hidden-import", mod])

    for exc in excludes:
        cmd.extend(["--exclude-module", exc])

    cmd.append(entry)

    print("Building huginn-agent.exe...")
    print(f"Entry: {entry}")
    print(f"Collect-all: {len(collect_all)} packages")
    print(f"Hidden imports: {len(hidden)} modules")
    print(f"Excludes: {len(excludes)} modules")
    print()

    result = subprocess.run(cmd, cwd=str(project_root))
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
