"""Bourbaki / Lean 4 environment installer and manager.

Handles automatic detection, download, and installation of the Lean 4
toolchain (via elan) so that Bourbaki formal verification becomes
actually runnable rather than always falling back to 'sorry' mode.

Usage:
    from huginn.bourbaki_env import LeanEnvironment
    env = LeanEnvironment()
    if env.ensure():
        result = env.run_check("conservation_of_mass")
    else:
        print("Bourbaki unavailable — install skipped or failed")
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Any


LEAN_INSTALL_DIR = Path(__file__).parent.parent / "lean"
ELAN_URL = "https://raw.githubusercontent.com/leanprover/elan/master/elan-init.ps1"


class LeanEnvironment:
    """Manages Lean 4 installation and project execution."""

    def __init__(self) -> None:
        self.install_dir = LEAN_INSTALL_DIR.resolve()
        self._lake_path: Path | None = None
        self._detected = False

    def _find_lake(self) -> Path | None:
        """Find lake executable (system or local)."""
        # Check system PATH first
        if shutil.which("lake"):
            return Path(shutil.which("lake"))
        # Check elan default location (Windows/macOS/Linux)
        elan_home = Path.home() / ".elan" / "bin"
        if elan_home.exists():
            lake = elan_home / "lake.exe"  # Windows
            if lake.exists():
                return lake
            lake = elan_home / "lake"  # Unix
            if lake.exists():
                return lake
        # Check local install
        local = self.install_dir / "bin" / "lake.exe"
        if local.exists():
            return local
        return None

    def detect(self) -> bool:
        """Detect whether Lean 4 is available."""
        # Ensure elan PATH is set
        elan_bin = Path.home() / ".elan" / "bin"
        if elan_bin.exists() and str(elan_bin) not in os.environ.get("PATH", ""):
            os.environ["PATH"] = f"{elan_bin}{os.pathsep}{os.environ.get('PATH', '')}"
        path = self._find_lake()
        if path:
            self._lake_path = path
            self._detected = True
            return True
        return False

    def ensure(self) -> bool:
        """Ensure Lean 4 is available, installing if necessary."""
        if self.detect():
            return True
        return self._install()

    def _install(self) -> bool:
        """Download and install elan + Lean 4."""
        print("[Bourbaki] Lean 4 not detected. Attempting auto-install...")
        system = platform.system()

        if system == "Windows":
            return self._install_windows()
        elif system == "Linux":
            return self._install_linux()
        elif system == "Darwin":
            return self._install_macos()
        else:
            print(f"[Bourbaki] Unsupported platform: {system}")
            return False

    def _install_windows(self) -> bool:
        """Install elan on Windows via PowerShell script."""
        try:
            # Download elan-init.ps1
            ps1_path = self.install_dir / "elan-init.ps1"
            self.install_dir.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(ELAN_URL, ps1_path)

            # Run with PowerShell
            result = subprocess.run(
                ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(ps1_path), "-y"],
                capture_output=True, text=True, timeout=300,
            )
            if result.returncode != 0:
                print(f"[Bourbaki] elan install failed: {result.stderr}")
                return False

            # elan installs to ~/.elan/bin, add to PATH for this process
            elan_bin = Path.home() / ".elan" / "bin"
            os.environ["PATH"] = f"{elan_bin}{os.pathsep}{os.environ.get('PATH', '')}"

            return self.detect()
        except Exception as e:
            print(f"[Bourbaki] Installation error: {e}")
            return False

    def _install_linux(self) -> bool:
        """Install elan on Linux via downloaded script (no shell=True)."""
        try:
            import urllib.request
            url = "https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh"
            with urllib.request.urlopen(url, timeout=60) as resp:
                script = resp.read().decode()
            result = subprocess.run(
                ["sh", "-s", "-y"],
                input=script, capture_output=True, text=True, timeout=300,
            )
            if result.returncode != 0:
                print(f"[Bourbaki] elan install failed: {result.stderr}")
                return False
            elan_bin = Path.home() / ".elan" / "bin"
            os.environ["PATH"] = f"{elan_bin}{os.pathsep}{os.environ.get('PATH', '')}"
            return self.detect()
        except Exception as e:
            print(f"[Bourbaki] Installation error: {e}")
            return False

    def _install_macos(self) -> bool:
        """Install elan on macOS."""
        return self._install_linux()  # Same script works on macOS

    def init_project(self) -> bool:
        """Initialize the Huginn Lean 4 project if not present."""
        if not self._lake_path:
            return False
        project_dir = self.install_dir / "project"
        if (project_dir / "lakefile.toml").exists():
            return True
        project_dir.mkdir(parents=True, exist_ok=True)
        lakefile = project_dir / "lakefile.toml"
        lakefile.write_text(
            '''name = "huginn"
version = "0.1.0"
defaultTargets = ["Huginn"]

[[lean_lib]]
name = "Huginn"
''',
            encoding="utf-8",
        )
        # Create basic structure
        (project_dir / "Huginn").mkdir(exist_ok=True)
        (project_dir / "Huginn" / "Basic.lean").write_text(
            '''import Mathlib

namespace Huginn

/-- A material system has a state space and an evolution function. -/
class MaterialSystem (α : Type) where
  state_space : Type
  evolution : α → α

/-- A conservation law asserts an invariant under evolution. -/
class ConservationLaw (M : MaterialSystem α) where
  invariant : α → ℝ
  preserved : ∀ s, invariant (M.evolution s) = invariant s

end Huginn
''',
            encoding="utf-8",
        )
        # Run lake build to fetch dependencies
        result = subprocess.run(
            [str(self._lake_path), "build"],
            cwd=project_dir, capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            print(f"[Bourbaki] lake build failed: {result.stderr[:500]}")
            return False
        return True

    def run_check(self, theorem_name: str, lean_code: str | None = None) -> dict[str, Any]:
        """Run a formal verification check."""
        if not self._lake_path:
            return {"success": False, "error": "Lean not available"}
        project_dir = self.install_dir / "project"
        if not self.init_project():
            return {"success": False, "error": "Project initialization failed"}

        # Write theorem to temporary file
        check_file = project_dir / "Huginn" / "Check.lean"
        check_file.write_text(
            f"import Huginn.Basic\n\n{lean_code or f'-- Check: {theorem_name}'}\n",
            encoding="utf-8",
        )

        result = subprocess.run(
            [str(self._lake_path), "build"],
            cwd=project_dir, capture_output=True, text=True, timeout=120,
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "theorem": theorem_name,
        }
