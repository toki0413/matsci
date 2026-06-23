"""math-anything compatibility stubs — analysis and visualization endpoints."""

from __future__ import annotations

import os
import tempfile
from typing import Any

import numpy as np
from fastapi import APIRouter

router = APIRouter(tags=["compat"])


@router.get("/firewall/status")
async def firewall_status() -> dict[str, Any]:
    return {"enabled": False}


@router.post("/sandbox/execute")
async def sandbox_execute(params: dict[str, Any]) -> dict[str, Any]:
    """Execute Python code in a sandbox."""
    code = params.get("code", "")
    timeout = params.get("timeout_seconds", 10)

    # Pre-validate code against restricted execution policy
    try:
        from huginn.security import RestrictedPythonError, validate_code

        validate_code(code)
    except RestrictedPythonError as e:
        return {"success": False, "error": f"Policy violation: {e}"}

    try:
        import subprocess

        from huginn.security import SandboxConfig, SandboxExecutor

        # Write code to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()
            tmp_path = f.name

        sandbox = SandboxExecutor(
            SandboxConfig(
                allowed_executables={"python", "python3"},
                default_timeout=min(float(timeout), 300.0),
                max_timeout=300.0,
                max_output_bytes=10 * 1024 * 1024,
            )
        )
        sb_result = sandbox.run(
            ["python", tmp_path],
            timeout=min(float(timeout), 300.0),
        )
        result = sb_result

        os.unlink(tmp_path)

        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"Execution timed out after {timeout}s"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ── Analysis endpoints ──────────────────────────────────────────────


@router.post("/analyze/symmetry")
async def analyze_symmetry(params: dict[str, Any]) -> dict[str, Any]:
    """Analyze crystal symmetry using spglib."""
    lattice = params.get("lattice")
    positions = params.get("positions")
    numbers = params.get("numbers")
    symprec = params.get("symprec", 1e-5)

    if not all([lattice, positions, numbers]):
        return {"error": "lattice, positions, and numbers are required"}

    try:
        import spglib

        cell = (lattice, positions, numbers)
        spacegroup = spglib.get_spacegroup(cell, symprec=symprec)
        symmetry_dataset = spglib.get_symmetry_dataset(cell, symprec=symprec)

        result: dict[str, Any] = {
            "spacegroup": spacegroup,
            "international_symbol": symmetry_dataset.international if symmetry_dataset else None,
            "hall_symbol": symmetry_dataset.hall if symmetry_dataset else None,
            "number": symmetry_dataset.number if symmetry_dataset else None,
            "pointgroup": symmetry_dataset.pointgroup if symmetry_dataset else None,
            "crystal_system": _crystal_system(symmetry_dataset.number) if symmetry_dataset else None,
        }

        if symmetry_dataset:
            result["equivalent_atoms"] = list(map(int, symmetry_dataset.equivalent_atoms))
            result["rotations_count"] = len(symmetry_dataset.rotations)

        return result

    except ImportError:
        # Fallback: basic symmetry detection from lattice parameters
        lat = np.array(lattice)
        norms = np.linalg.norm(lat, axis=1)
        angles = []
        for i in range(3):
            for j in range(i + 1, 3):
                cos_angle = np.dot(lat[i], lat[j]) / (norms[i] * norms[j] + 1e-12)
                angles.append(float(np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))))

        system = _guess_crystal_system(norms, angles)
        return {
            "spacegroup": "unknown (spglib not installed)",
            "crystal_system": system,
            "lattice_norms": norms.tolist(),
            "lattice_angles_deg": angles,
            "note": "Install spglib for full symmetry analysis: pip install spglib",
        }


def _crystal_system(spg_number: int) -> str:
    if spg_number <= 2:
        return "triclinic"
    if spg_number <= 15:
        return "monoclinic"
    if spg_number <= 74:
        return "orthorhombic"
    if spg_number <= 142:
        return "tetragonal"
    if spg_number <= 167:
        return "trigonal"
    if spg_number <= 194:
        return "hexagonal"
    return "cubic"


def _guess_crystal_system(norms: np.ndarray, angles: list[float]) -> str:
    a, b, c = norms
    alpha, beta, gamma = angles
    tol = 0.01
    angle_tol = 1.0

    equal_ab = abs(a - b) < tol
    equal_bc = abs(b - c) < tol
    equal_ac = abs(a - c) < tol
    all_equal = equal_ab and equal_bc
    all_90 = all(abs(ang - 90.0) < angle_tol for ang in angles)

    if all_equal and all_90:
        return "cubic"
    if equal_ab and not equal_bc and all_90:
        return "tetragonal"
    if all_90:
        return "orthorhombic"
    if abs(alpha - 90) < angle_tol and abs(beta - 90) < angle_tol and abs(gamma - 120) < angle_tol:
        return "hexagonal"
    return "triclinic"


@router.post("/analyze/spectral")
async def analyze_spectral(params: dict[str, Any]) -> dict[str, Any]:
    """Spectral analysis (FFT) on time-series or signal data."""
    data = params.get("data")
    sample_rate = params.get("sample_rate", 1.0)

    if data is None:
        return {"error": "data array is required"}

    signal = np.array(data, dtype=np.float64)
    n = len(signal)

    # FFT
    fft_vals = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    magnitudes = np.abs(fft_vals) / n

    # Find peaks (top 5)
    peak_indices = np.argsort(magnitudes)[-5:][::-1]
    peaks = []
    for idx in peak_indices:
        if magnitudes[idx] > 0.01 * np.max(magnitudes):
            peaks.append({
                "frequency": float(freqs[idx]),
                "magnitude": float(magnitudes[idx]),
            })

    return {
        "n_samples": n,
        "sample_rate": sample_rate,
        "frequency_range": [float(freqs[0]), float(freqs[-1])],
        "dominant_frequency": float(freqs[np.argmax(magnitudes[1:]) + 1]) if n > 1 else 0.0,
        "peaks": peaks,
        "total_power": float(np.sum(magnitudes ** 2)),
        "mean": float(np.mean(signal)),
        "std": float(np.std(signal)),
    }


@router.post("/analyze/dynamics")
async def analyze_dynamics(params: dict[str, Any]) -> dict[str, Any]:
    """Basic molecular dynamics analysis from trajectory data."""
    positions = params.get("positions")  # list of frames, each a list of [x,y,z]
    timestep = params.get("timestep", 1.0)  # fs
    masses = params.get("masses")  # atomic masses

    if positions is None:
        return {"error": "positions (list of trajectory frames) is required"}

    frames = np.array(positions)
    n_frames, n_atoms, n_dim = frames.shape

    result: dict[str, Any] = {
        "n_frames": n_frames,
        "n_atoms": n_atoms,
        "timestep": timestep,
        "total_time": (n_frames - 1) * timestep,
    }

    # MSD (Mean Squared Displacement)
    if n_frames > 1:
        displacements = frames - frames[0]
        msd = np.mean(np.sum(displacements ** 2, axis=2), axis=1)
        result["msd"] = msd.tolist()
        # Diffusion coefficient estimate (Einstein relation: MSD = 6Dt in 3D)
        times = np.arange(n_frames) * timestep
        if n_frames > 2:
            slope = np.polyfit(times[1:], msd[1:], 1)[0]
            result["diffusion_coefficient_estimate"] = float(slope / 6.0)

    # RMSD between consecutive frames
    if n_frames > 1:
        rmsd = []
        for i in range(1, n_frames):
            d = np.sqrt(np.mean(np.sum((frames[i] - frames[i - 1]) ** 2, axis=1)))
            rmsd.append(float(d))
        result["rmsd_consecutive_mean"] = float(np.mean(rmsd))
        result["rmsd_consecutive_max"] = float(np.max(rmsd))

    # Kinetic energy estimate if masses provided
    if masses and n_frames > 1:
        masses_arr = np.array(masses)
        velocities = np.diff(frames, axis=0) / timestep
        ke_frames = []
        for v in velocities:
            ke = 0.5 * np.sum(masses_arr[:, None] * v ** 2)
            ke_frames.append(float(ke))
        result["kinetic_energy_mean"] = float(np.mean(ke_frames))
        result["kinetic_energy_std"] = float(np.std(ke_frames))

    return result


@router.post("/analyze/tda")
async def analyze_tda(params: dict[str, Any]) -> dict[str, Any]:
    """Topological Data Analysis — persistence diagram computation."""
    data = params.get("data")  # point cloud: list of [x, y, z, ...]
    max_dimension = params.get("max_dimension", 1)
    n_bins = params.get("n_bins", 50)

    if data is None:
        return {"error": "data (point cloud) is required"}

    points = np.array(data, dtype=np.float64)
    n_points, n_dim = points.shape

    try:
        import gudhi

        rips = gudhi.RipsComplex(points, max_dimension=max_dimension + 1)
        simplex_tree = rips.create_simplex_tree(max_alpha_square=0.0)
        persistence = simplex_tree.persistence(min_persistence=0.0)
        diagram = []
        for dim, (birth, death) in persistence:
            diagram.append({"dimension": dim, "birth": float(birth), "death": float(death)})
        return {
            "n_points": n_points,
            "embedding_dimension": n_dim,
            "persistence_diagram": diagram,
            "engine": "gudhi",
        }
    except ImportError:
        pass

    # Fallback: pairwise distance-based heuristic
    from scipy.spatial.distance import pdist

    distances = pdist(points)
    return {
        "n_points": n_points,
        "embedding_dimension": n_dim,
        "pairwise_distances": {
            "min": float(np.min(distances)),
            "max": float(np.max(distances)),
            "mean": float(np.mean(distances)),
            "std": float(np.std(distances)),
        },
        "note": "Install gudhi for full TDA: pip install gudhi",
        "engine": "scipy_fallback",
    }


@router.post("/analyze/sindy")
async def analyze_sindy(params: dict[str, Any]) -> dict[str, Any]:
    """SINDy — Sparse Identification of Nonlinear Dynamics."""
    data = params.get("data")  # time-series: list of state vectors [x(t)]
    t = params.get("time")  # time points
    threshold = params.get("threshold", 0.01)
    library = params.get("library", "polynomial")  # polynomial or fourier

    if data is None:
        return {"error": "data (time-series of state vectors) is required"}

    X = np.array(data, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n_samples, n_vars = X.shape

    if t is not None:
        dt = np.diff(np.array(t))
        mean_dt = float(np.mean(dt))
    else:
        mean_dt = 1.0

    # Numerical differentiation (central differences)
    dX = np.gradient(X, mean_dt, axis=0)

    # Build library
    if library == "polynomial":
        # Polynomial basis: 1, x, x^2, xy, ...
        Theta = np.column_stack([
            np.ones(n_samples),  # constant
            X,  # linear
        ])
        feature_names = ["1"] + [f"x{i}" for i in range(n_vars)]
        # Quadratic terms
        for i in range(n_vars):
            Theta = np.column_stack([Theta, X[:, i: i + 1] ** 2])
            feature_names.append(f"x{i}^2")
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                Theta = np.column_stack([Theta, (X[:, i] * X[:, j]).reshape(-1, 1)])
                feature_names.append(f"x{i}*x{j}")
    else:
        # Fourier basis
        Theta = np.column_stack([np.ones(n_samples), X])
        feature_names = ["1"] + [f"x{i}" for i in range(n_vars)]
        for i in range(n_vars):
            Theta = np.column_stack([Theta, np.sin(X[:, i: i + 1])])
            feature_names.append(f"sin(x{i})")
            Theta = np.column_stack([Theta, np.cos(X[:, i: i + 1])])
            feature_names.append(f"cos(x{i})")

    # Sparse regression: iterative thresholded least squares
    n_features = Theta.shape[1]
    Xi = np.zeros((n_features, n_vars))
    for k in range(n_vars):
        xi = np.linalg.lstsq(Theta, dX[:, k], rcond=None)[0]
        # Iterative thresholding
        for _ in range(10):
            small = np.abs(xi) < threshold
            xi[small] = 0
            if np.any(~small):
                xi[~small] = np.linalg.lstsq(Theta[:, ~small], dX[:, k], rcond=None)[0]
        Xi[:, k] = xi

    # Extract discovered equations
    equations = []
    for k in range(n_vars):
        terms = []
        for j in range(n_features):
            if abs(Xi[j, k]) > 1e-10:
                coeff = float(Xi[j, k])
                terms.append(f"{coeff:+.4f}*{feature_names[j]}")
        equations.append(" + ".join(terms) if terms else "0")

    return {
        "n_samples": n_samples,
        "n_variables": n_vars,
        "library": library,
        "n_features": n_features,
        "discovered_equations": equations,
        "feature_names": feature_names,
        "threshold": threshold,
        "engine": "numpy_sindy",
    }


# ── Visualization endpoints ─────────────────────────────────────────


@router.post("/viz/dos")
async def viz_dos(params: dict[str, Any]) -> dict[str, Any]:
    """Density of States visualization data."""
    energies = params.get("energies")
    dos = params.get("dos")
    fermi_level = params.get("fermi_level", 0.0)

    if energies is None or dos is None:
        return {
            "fallback": True,
            "html": "<div>DOS visualization requires energies and dos arrays</div>",
            "hint": "Provide energies (list of eV values) and dos (list of states/eV)",
        }

    e = np.array(energies)
    d = np.array(dos)
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    total_states = float(_trapz(d, e)) if _trapz else 0.0
    html = (
        f"<div class='dos-plot'>"
        f"<h3>Density of States</h3>"
        f"<p>Total states: {total_states:.2f} | Fermi level: {fermi_level} eV</p>"
        f"<p>Data points: {len(e)} | Range: [{e.min():.2f}, {e.max():.2f}] eV</p>"
        f"<pre>{_ascii_plot(e, d, 'DOS')}</pre>"
        f"</div>"
    )
    return {
        "fallback": False,
        "html": html,
        "total_states": total_states,
        "energy_range": [float(e.min()), float(e.max())],
        "fermi_level": fermi_level,
    }


@router.post("/viz/phase")
async def viz_phase(params: dict[str, Any]) -> dict[str, Any]:
    """Phase portrait visualization."""
    x_data = params.get("x")
    y_data = params.get("y")

    if x_data is None or y_data is None:
        return {
            "fallback": True,
            "html": "<div>Phase portrait requires x and y trajectory arrays</div>",
        }

    x = np.array(x_data)
    y = np.array(y_data)
    html = (
        f"<div class='phase-plot'>"
        f"<h3>Phase Portrait</h3>"
        f"<p>Trajectory points: {len(x)}</p>"
        f"<p>X range: [{x.min():.3f}, {x.max():.3f}]</p>"
        f"<p>Y range: [{y.min():.3f}, {y.max():.3f}]</p>"
        f"<pre>{_ascii_phase(x, y)}</pre>"
        f"</div>"
    )
    return {"fallback": False, "html": html, "n_points": len(x)}


@router.post("/viz/persistence")
async def viz_persistence(params: dict[str, Any]) -> dict[str, Any]:
    """Persistence diagram visualization for TDA."""
    diagram = params.get("diagram")  # list of {dimension, birth, death}

    if diagram is None:
        return {
            "fallback": True,
            "html": "<div>Persistence diagram requires diagram data from /analyze/tda</div>",
        }

    points = [
        (p.get("dimension", 0), p.get("birth", 0), p.get("death", 0))
        for p in diagram
    ]
    by_dim: dict[int, list] = {}
    for dim, b, d in points:
        by_dim.setdefault(dim, []).append((b, d))

    html = "<div class='persistence-diagram'><h3>Persistence Diagram</h3>"
    for dim in sorted(by_dim):
        pts = by_dim[dim]
        lifetimes = [d - b for b, d in pts]
        html += (
            f"<p>H{dim}: {len(pts)} features | "
            f"max lifetime: {max(lifetimes):.4f} | "
            f"mean lifetime: {np.mean(lifetimes):.4f}</p>"
        )
    html += "</div>"
    return {"fallback": False, "html": html, "n_features": len(points)}


@router.post("/viz/sindy")
async def viz_sindy(params: dict[str, Any]) -> dict[str, Any]:
    """SINDy results visualization."""
    equations = params.get("equations")

    if equations is None:
        return {
            "fallback": True,
            "html": "<div>SINDy visualization requires discovered equations from /analyze/sindy</div>",
        }

    html = "<div class='sindy-results'><h3>Discovered Equations</h3><ol>"
    for i, eq in enumerate(equations):
        html += f"<li>dx<sub>{i}</sub>/dt = {eq}</li>"
    html += "</ol></div>"
    return {"fallback": False, "html": html, "n_equations": len(equations)}


# ── Helpers ─────────────────────────────────────────────────────────


def _ascii_plot(x: np.ndarray, y: np.ndarray, label: str, width: int = 60, height: int = 15) -> str:
    """Generate a simple ASCII line plot."""
    if len(x) == 0:
        return "(empty data)"
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())
    if y_max == y_min:
        y_max = y_min + 1.0

    # Downsample to width
    indices = np.linspace(0, len(x) - 1, width, dtype=int)
    x_s = x[indices]
    y_s = y[indices]

    grid = [[" " for _ in range(width)] for _ in range(height)]
    for col, val in enumerate(y_s):
        row = int((val - y_min) / (y_max - y_min) * (height - 1))
        row = max(0, min(height - 1, row))
        grid[height - 1 - row][col] = "*"

    lines = [f"  {label}"]
    for i, row in enumerate(grid):
        y_label = y_max - i * (y_max - y_min) / (height - 1)
        lines.append(f"{y_label:8.2f} |{''.join(row)}")
    lines.append(f"         +{'─' * width}")
    lines.append(f"          {x_min:.2f}{' ' * (width - 10)}{x_max:.2f}")
    return "\n".join(lines)


def _ascii_phase(x: np.ndarray, y: np.ndarray, width: int = 40, height: int = 20) -> str:
    """Generate a simple ASCII phase portrait."""
    if len(x) == 0:
        return "(empty data)"
    grid = [[" " for _ in range(width)] for _ in range(height)]
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())
    x_range = x_max - x_min or 1.0
    y_range = y_max - y_min or 1.0

    for xi, yi in zip(x, y):
        col = int((xi - x_min) / x_range * (width - 1))
        row = int((yi - y_min) / y_range * (height - 1))
        col = max(0, min(width - 1, col))
        row = max(0, min(height - 1, row))
        grid[height - 1 - row][col] = "."

    return "\n".join("".join(r) for r in grid)
