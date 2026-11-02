"""Visualization helpers for evolution/benchmark/exploration reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_json(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Report not found: {target}")
    return json.loads(target.read_text(encoding="utf-8"))


def _save_or_show(fig, output_path: str | Path | None) -> Path | None:
    import matplotlib.pyplot as plt

    if output_path:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return target
    plt.close(fig)
    return None


def _benchmark_status_counts(results: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"passed": 0, "failed": 0, "skipped": 0}
    for r in results:
        if r.get("skipped"):
            counts["skipped"] += 1
        elif r.get("passed"):
            counts["passed"] += 1
        else:
            counts["failed"] += 1
    return counts


def _benchmark_category_breakdown(
    results: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, int]], dict[str, float]]:
    categories: dict[str, dict[str, int]] = {}
    task_times: dict[str, float] = {}
    for r in results:
        cat = r.get("category", "unknown")
        bucket = categories.setdefault(cat, {"passed": 0, "failed": 0, "skipped": 0})
        if r.get("skipped"):
            bucket["skipped"] += 1
        elif r.get("passed"):
            bucket["passed"] += 1
        else:
            bucket["failed"] += 1
        task_times[r.get("task_id", "?")] = r.get("exec_time_seconds", 0.0)
    return categories, task_times


def _render_benchmark_pie(plt, counts: dict[str, int], output_path) -> Path | None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(
        counts.values(),
        labels=counts.keys(),
        autopct="%1.1f%%",
        colors=["green", "red", "gray"],
    )
    ax.set_title("Benchmark Overall Results")
    return _save_or_show(fig, output_path)


def _render_benchmark_bars(
    plt,
    categories: dict[str, dict[str, int]],
    task_times: dict[str, float],
    output_path,
) -> Path | None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Pass/fail by category
    ax = axes[0]
    cats = list(categories.keys())
    passed = [categories[c]["passed"] for c in cats]
    failed = [categories[c]["failed"] for c in cats]
    skipped = [categories[c]["skipped"] for c in cats]
    x = range(len(cats))
    width = 0.25
    ax.bar([i - width for i in x], passed, width, label="passed", color="green")
    ax.bar(x, failed, width, label="failed", color="red")
    ax.bar([i + width for i in x], skipped, width, label="skipped", color="gray")
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=30, ha="right")
    ax.set_title("Benchmark Results by Category")
    ax.set_ylabel("Count")
    ax.legend()

    # Execution time by task
    ax = axes[1]
    ax.barh(list(task_times.keys()), list(task_times.values()), color="steelblue")
    ax.set_xlabel("Execution time (s)")
    ax.set_title("Task Execution Time")

    fig.tight_layout()
    return _save_or_show(fig, output_path)


def plot_benchmark_report(
    report: dict[str, Any],
    output_path: str | Path | None = None,
    plot_type: str = "bar",
) -> Path | None:
    """Visualize a benchmark report.

    Args:
        report: Benchmark report dict.
        output_path: Where to save the figure.
        plot_type: "bar" for category breakdown + task times, "pie" for overall summary.

    Returns the saved plot path, or None when no output path is given.
    """
    import matplotlib.pyplot as plt

    results = report.get("results", [])
    if not results:
        raise ValueError("No results in benchmark report")

    if plot_type == "pie":
        counts = _benchmark_status_counts(results)
        return _render_benchmark_pie(plt, counts, output_path)

    categories, task_times = _benchmark_category_breakdown(results)
    return _render_benchmark_bars(plt, categories, task_times, output_path)


def _evolution_confidences(
    failure_rules: list[dict[str, Any]],
    success_skills: list[dict[str, Any]],
    prompt_patches: list[dict[str, Any]],
) -> list[float]:
    confidences = []
    confidences.extend([r.get("confidence", 0.0) for r in failure_rules])
    confidences.extend([s.get("extraction_confidence", 0.0) for s in success_skills])
    confidences.extend([p.get("confidence", 0.0) for p in prompt_patches])
    return confidences


def _render_confidence_histogram(
    plt, confidences: list[float], output_path
) -> Path | None:
    fig, ax = plt.subplots(figsize=(8, 5))
    if confidences:
        ax.hist(
            confidences,
            bins=10,
            range=(0, 1),
            color="mediumpurple",
            edgecolor="black",
        )
    ax.set_title("Confidence Distribution")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    return _save_or_show(fig, output_path)


def _render_evolution_counts(
    ax,
    failure_rules: list[dict[str, Any]],
    success_skills: list[dict[str, Any]],
    prompt_patches: list[dict[str, Any]],
) -> None:
    # Counts by type
    labels = ["Failure rules", "Success skills", "Prompt patches"]
    values = [len(failure_rules), len(success_skills), len(prompt_patches)]
    colors = ["coral", "seagreen", "royalblue"]
    ax.bar(labels, values, color=colors)
    ax.set_title("Evolution Outputs Count")
    ax.set_ylabel("Count")


def _render_evolution_confidence_panel(ax, confidences: list[float]) -> None:
    # Confidence distribution
    if confidences:
        ax.hist(
            confidences, bins=10, range=(0, 1), color="mediumpurple", edgecolor="black"
        )
    ax.set_title("Confidence Distribution")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Frequency")


def _render_evolution_summary(
    plt,
    failure_rules: list[dict[str, Any]],
    success_skills: list[dict[str, Any]],
    prompt_patches: list[dict[str, Any]],
    output_path,
) -> Path | None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    _render_evolution_counts(axes[0], failure_rules, success_skills, prompt_patches)
    confidences = _evolution_confidences(failure_rules, success_skills, prompt_patches)
    _render_evolution_confidence_panel(axes[1], confidences)
    fig.tight_layout()
    return _save_or_show(fig, output_path)


def plot_evolution_report(
    report: dict[str, Any],
    output_path: str | Path | None = None,
    plot_type: str = "summary",
) -> Path | None:
    """Visualize an evolution report.

    Args:
        report: Evolution report dict.
        output_path: Where to save the figure.
        plot_type: "summary" (counts + confidence) or "confidence" (histogram only).
    """
    import matplotlib.pyplot as plt

    failure_rules = report.get("failure_rules", [])
    success_skills = report.get("success_skills", [])
    prompt_patches = report.get("prompt_patches", [])

    if plot_type == "confidence":
        confidences = _evolution_confidences(
            failure_rules, success_skills, prompt_patches
        )
        return _render_confidence_histogram(plt, confidences, output_path)

    return _render_evolution_summary(
        plt, failure_rules, success_skills, prompt_patches, output_path
    )


def _normalize(values: list[float]) -> list[float]:
    """Min-max normalize a list of floats to [0, 1]."""
    lo, hi = min(values), max(values)
    if hi == lo:
        return [0.5 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def _exploration_objectives(
    result: dict[str, Any],
) -> tuple[list[str], list[dict[str, float]]]:
    """Return objective names and a list of objective dicts from the Pareto front."""
    front = result.get("pareto_front", []) or []
    if not front:
        raise ValueError("No Pareto front in exploration result")
    names = list(front[0].get("objectives", {}).keys())
    objectives = [b.get("objectives", {}) for b in front]
    return names, objectives


def _resolve_exploration_plot_type(names: list[str], plot_type: str) -> str:
    if plot_type != "auto":
        return plot_type
    if len(names) == 2:
        return "2d"
    if len(names) == 3:
        return "3d"
    if len(names) > 3:
        return "parallel"
    return "bar"


def _render_pareto_2d(
    plt,
    names: list[str],
    objectives: list[dict[str, float]],
    best: dict[str, Any] | None,
    output_path,
) -> Path | None:
    if len(names) != 2:
        raise ValueError("2D plot requires exactly 2 objectives")
    x_name, y_name = names
    xs = [o[x_name] for o in objectives]
    ys = [o[y_name] for o in objectives]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(xs, ys, color="darkorange", s=100, edgecolors="black")
    if best and "objectives" in best:
        bx, by = best["objectives"][x_name], best["objectives"][y_name]
        ax.scatter([bx], [by], color="red", s=150, marker="*", label="best branch")
        ax.legend()
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title("Pareto Front")
    fig.tight_layout()
    return _save_or_show(fig, output_path)


def _render_pareto_3d(
    plt,
    names: list[str],
    objectives: list[dict[str, float]],
    best: dict[str, Any] | None,
    output_path,
) -> Path | None:
    if len(names) != 3:
        raise ValueError("3D plot requires exactly 3 objectives")
    x_name, y_name, z_name = names
    xs = [o[x_name] for o in objectives]
    ys = [o[y_name] for o in objectives]
    zs = [o[z_name] for o in objectives]
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xs, ys, zs, color="darkorange", s=100, edgecolors="black")
    if best and "objectives" in best:
        bobj = best["objectives"]
        ax.scatter(
            [bobj[x_name]],
            [bobj[y_name]],
            [bobj[z_name]],
            color="red",
            s=150,
            marker="*",
            label="best branch",
        )
        ax.legend()
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_zlabel(z_name)
    ax.set_title("Pareto Front (3D)")
    fig.tight_layout()
    return _save_or_show(fig, output_path)


def _render_parallel_coordinates(
    plt,
    names: list[str],
    objectives: list[dict[str, float]],
    best: dict[str, Any] | None,
    output_path,
) -> Path | None:
    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.5), 6))
    x = range(len(names))
    for obj in objectives:
        vals = [obj.get(n, 0.0) for n in names]
        norm_vals = _normalize(vals)
        ax.plot(x, norm_vals, alpha=0.6, color="steelblue")
    if best and "objectives" in best:
        vals = [best["objectives"].get(n, 0.0) for n in names]
        norm_vals = _normalize(vals)
        ax.plot(x, norm_vals, color="red", linewidth=2, marker="o", label="best branch")
        ax.legend()
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Normalized objective value")
    ax.set_title("Pareto Front Parallel Coordinates")
    fig.tight_layout()
    return _save_or_show(fig, output_path)


def _render_radar(
    plt,
    names: list[str],
    best: dict[str, Any] | None,
    output_path,
) -> Path | None:
    if not best or "objectives" not in best:
        raise ValueError("Radar plot requires a best_branch with objectives")
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})
    objs = best["objectives"]
    values = [objs.get(n, 0.0) for n in names]
    # Normalize to [0, 1] for readability
    norm_values = _normalize(values)
    angles = [i / len(names) * 2 * 3.14159 for i in range(len(names))]
    angles += angles[:1]
    norm_values += norm_values[:1]
    ax.plot(angles, norm_values, "o-", color="teal", linewidth=2)
    ax.fill(angles, norm_values, alpha=0.25, color="teal")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1)
    ax.set_title("Best Branch Objectives (Radar)")
    fig.tight_layout()
    return _save_or_show(fig, output_path)


def _render_best_branch_bar(
    plt,
    best: dict[str, Any] | None,
    output_path,
) -> Path | None:
    # Fallback bar chart
    fig, ax = plt.subplots(figsize=(8, 6))
    if best and "objectives" in best:
        objs = best["objectives"]
        ax.bar(objs.keys(), objs.values(), color="teal")
        ax.set_title("Best Branch Objectives")
        ax.set_ylabel("Value")
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    else:
        ax.text(0.5, 0.5, "No best branch objectives", ha="center", va="center")
    fig.tight_layout()
    return _save_or_show(fig, output_path)


def plot_exploration_result(
    result: dict[str, Any],
    output_path: str | Path | None = None,
    plot_type: str = "auto",
) -> Path | None:
    """Visualize an exploration result.

    Args:
        result: Exploration result dict.
        output_path: Where to save the figure.
        plot_type: One of "auto", "2d", "3d", "parallel", "radar".
            "auto" picks 2d/3d/parallel based on objective count.
    """
    import matplotlib.pyplot as plt

    names, objectives = _exploration_objectives(result)
    best = result.get("best_branch")
    plot_type = _resolve_exploration_plot_type(names, plot_type)

    if plot_type == "2d":
        return _render_pareto_2d(plt, names, objectives, best, output_path)
    if plot_type == "3d":
        return _render_pareto_3d(plt, names, objectives, best, output_path)
    if plot_type == "parallel":
        return _render_parallel_coordinates(plt, names, objectives, best, output_path)
    if plot_type == "radar":
        return _render_radar(plt, names, best, output_path)
    return _render_best_branch_bar(plt, best, output_path)


def _convergence_series(history: list[dict[str, Any]]) -> dict[str, list]:
    steps = list(range(1, len(history) + 1))
    total_rules = [h.get("total_rules", 0) for h in history]
    total_skills = [h.get("total_skills", 0) for h in history]
    avg_conf = [h.get("avg_confidence", 0.0) for h in history]
    new_rules = [
        h.get("new_failure_rules", 0) + h.get("new_prompt_patches", 0) for h in history
    ]
    new_skills = [h.get("new_success_skills", 0) for h in history]
    return {
        "steps": steps,
        "total_rules": total_rules,
        "total_skills": total_skills,
        "avg_conf": avg_conf,
        "new_rules": new_rules,
        "new_skills": new_skills,
    }


def _render_convergence_totals(ax, series: dict[str, list]) -> None:
    steps = series["steps"]

    # Cumulative totals
    ax.plot(steps, series["total_rules"], "o-", label="Total rules", color="coral")
    ax.plot(steps, series["total_skills"], "s-", label="Total skills", color="seagreen")
    ax.set_ylabel("Count")
    ax.set_title("Evolution Convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)


def _render_convergence_cycle_bars(ax, series: dict[str, list]) -> None:
    steps = series["steps"]

    # New items per cycle + avg confidence
    ax.bar(
        [s - 0.15 for s in steps],
        series["new_rules"],
        width=0.3,
        label="New rules",
        color="coral",
    )
    ax.bar(
        [s + 0.15 for s in steps],
        series["new_skills"],
        width=0.3,
        label="New skills",
        color="seagreen",
    )
    ax.set_xlabel("Evolution cycle")
    ax.set_ylabel("New items")
    ax2 = ax.twinx()
    ax2.plot(steps, series["avg_conf"], "^-", color="royalblue", label="Avg confidence")
    ax2.set_ylabel("Avg confidence", color="royalblue")
    ax2.tick_params(axis="y", labelcolor="royalblue")
    ax2.set_ylim(0, 1)
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax.grid(True, alpha=0.3)


def plot_evolution_convergence(
    history: list[dict[str, Any]],
    output_path: str | Path | None = None,
) -> Path | None:
    """Visualize evolution convergence over repeated cycles.

    ``history`` is a list of dicts with keys:
    ``total_rules``, ``total_skills``, ``avg_confidence``,
    ``new_failure_rules``, ``new_success_skills``, ``new_prompt_patches``.
    """
    import matplotlib.pyplot as plt

    if not history:
        raise ValueError("Empty evolution history")

    series = _convergence_series(history)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    _render_convergence_totals(axes[0], series)
    _render_convergence_cycle_bars(axes[1], series)

    fig.tight_layout()
    return _save_or_show(fig, output_path)


def plot_from_file(
    kind: str,
    path: str | Path,
    output_path: str | Path | None = None,
    plot_type: str | None = None,
) -> Path | None:
    """Load a report file and render the appropriate visualization."""
    data = _load_json(path)
    if kind == "bench":
        return plot_benchmark_report(data, output_path, plot_type=plot_type or "bar")
    if kind == "evolution":
        if plot_type == "convergence":
            if not isinstance(data, list):
                raise ValueError(
                    "Convergence plot requires an evolution history JSON list"
                )
            return plot_evolution_convergence(data, output_path)
        return plot_evolution_report(
            data, output_path, plot_type=plot_type or "summary"
        )
    if kind == "explore":
        return plot_exploration_result(data, output_path, plot_type=plot_type or "auto")
    raise ValueError(f"Unknown visualization kind: {kind}")


# ── Materials-science plots (Phase 4c) ────────────────────────────


def apply_arial_style() -> None:
    """统一 Arial 字体 + 20pt 加粗. 用户硬性要求, 所有 materials 图都用这套."""
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 20
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["figure.titlesize"] = 22
    plt.rcParams["figure.titleweight"] = "bold"


def _plot_band_curves(ax, bands_data: list[dict[str, Any]]) -> None:
    for band in bands_data:
        kpoints = band.get("kpoints", [])
        energies = band.get("energies", [])
        if kpoints and energies:
            ax.plot(kpoints, energies, "-", color="steelblue", linewidth=1.5)


def _band_tick_positions(bands_data: list[dict[str, Any]], n_ticks: int) -> list[float]:
    return (
        [
            i * (len(bands_data[0].get("kpoints", [1])) - 1) / (n_ticks - 1)
            for i in range(n_ticks)
        ]
        if bands_data
        else []
    )


def _draw_band_high_symmetry(
    ax, bands_data: list[dict[str, Any]], kpath: list[str]
) -> None:
    # 高对称点竖线 (假设 kpoints 范围 0..len(kpath)-1, 等间距)
    n_ticks = len(kpath)
    if n_ticks <= 1:
        return
    tick_positions = _band_tick_positions(bands_data, n_ticks)
    for x in tick_positions:
        ax.axvline(x=x, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(kpath)


def plot_band_structure(
    bands_data: list[dict[str, Any]],
    kpath: list[str],
    fermi: float = 0.0,
    output_path: str | Path | None = None,
) -> Path | None:
    """画能带结构. e-vs-k 折线, 高对称点竖线标注.

    bands_data: 每个 dict 含 "kpoints" (list[float]) + "energies" (list[float])
    kpath: 高对称点标签列表, 如 ["Γ", "X", "M", "Γ"]
    fermi: 费米能级 (eV), 画虚线
    """
    import matplotlib.pyplot as plt

    apply_arial_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    _plot_band_curves(ax, bands_data)

    # 费米能级虚线
    ax.axhline(
        y=fermi, color="red", linestyle="--", linewidth=1.5, label=f"E_f = {fermi:.2f}"
    )

    _draw_band_high_symmetry(ax, bands_data, kpath)

    ax.set_xlabel("Wave Vector")
    ax.set_ylabel("Energy (eV)")
    ax.set_title("Band Structure")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save_or_show(fig, output_path)


def plot_dos(
    dos_data: dict[str, Any],
    energy: list[float] | None = None,
    fermi: float = 0.0,
    output_path: str | Path | None = None,
) -> Path | None:
    """画态密度 (DOS). 主 DOS 曲线 + 可选投影 DOS.

    dos_data: {"total": [floats], "orbital_s": [floats], "orbital_p": [floats], ...}
    energy: 能量轴 (eV), 不给就用 range
    fermi: 费米能级, 画虚线
    """
    import matplotlib.pyplot as plt

    apply_arial_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    total = dos_data.get("total", [])
    if not total:
        raise ValueError("dos_data must contain 'total' key with list of floats")
    if energy is None:
        energy = list(range(len(total)))

    ax.plot(total, energy, "-", color="steelblue", linewidth=2, label="Total DOS")

    # 投影 DOS
    for key, vals in dos_data.items():
        if key == "total" or not isinstance(vals, list):
            continue
        if len(vals) == len(energy):
            ax.plot(vals, energy, "-", linewidth=1.2, label=key)

    ax.axhline(
        y=fermi, color="red", linestyle="--", linewidth=1.5, label=f"E_f = {fermi:.2f}"
    )
    ax.set_xlabel("DOS (states/eV)")
    ax.set_ylabel("Energy (eV)")
    ax.set_title("Density of States")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save_or_show(fig, output_path)


def plot_phonon_dispersion(
    branches: list[dict[str, Any]],
    qpath: list[str] | None = None,
    output_path: str | Path | None = None,
) -> Path | None:
    """画声子色散. 每条 branch 一条曲线.

    branches: 每个 dict 含 "qpoints" (list[float]) + "frequencies" (list[float])
    qpath: 高对称点标签
    """
    import matplotlib.pyplot as plt

    apply_arial_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    for _i, branch in enumerate(branches):
        qpoints = branch.get("qpoints", [])
        freqs = branch.get("frequencies", [])
        if qpoints and freqs:
            ax.plot(qpoints, freqs, "-", color="steelblue", linewidth=1.2)

    # 零频线 (声学声子参考)
    ax.axhline(y=0, color="gray", linestyle=":", linewidth=0.8)

    if qpath and len(qpath) > 1 and branches:
        n_q = len(branches[0].get("qpoints", [1]))
        tick_positions = [i * (n_q - 1) / (len(qpath) - 1) for i in range(len(qpath))]
        for x in tick_positions:
            ax.axvline(x=x, color="gray", linestyle=":", linewidth=0.8)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(qpath)

    ax.set_xlabel("Wave Vector")
    ax.set_ylabel("Frequency (THz)")
    ax.set_title("Phonon Dispersion")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save_or_show(fig, output_path)


def _plot_atoms(ax, species: list[str], coords: list[list[float]]) -> None:
    # 原子颜色映射 (常见元素)
    color_map = {
        "Si": "goldenrod",
        "O": "red",
        "C": "black",
        "N": "blue",
        "H": "lightgray",
        "Fe": "darkred",
        "Cu": "orange",
        "Al": "gray",
        "Ti": "lightblue",
        "Ga": "green",
        "As": "purple",
        "Na": "violet",
        "Cl": "green",
        "Mg": "darkgreen",
        "Ca": "darkorange",
    }

    # 画原子
    for i, (sp, coord) in enumerate(zip(species, coords)):
        color = color_map.get(sp, "steelblue")
        ax.scatter(
            coord[0],
            coord[1],
            coord[2],
            c=color,
            s=300,
            edgecolors="black",
            linewidths=1.5,
            label=sp if i == 0 else "",
        )


def _plot_bonds(ax, coords: list[list[float]], bonds: list[list[int]]) -> None:
    # 画键 (如果给了 bonds)
    for bond in bonds:
        if len(bond) >= 2 and bond[0] < len(coords) and bond[1] < len(coords):
            c1 = coords[bond[0]]
            c2 = coords[bond[1]]
            ax.plot([c1[0], c2[0]], [c1[1], c2[1]], [c1[2], c2[2]], "k-", linewidth=2)


def _dedupe_legend(ax) -> None:
    # 去重 legend
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        unique = [
            (h, lb)
            for i, (h, lb) in enumerate(zip(handles, labels))
            if lb not in labels[:i]
        ]
        ax.legend([u[0] for u in unique], [u[1] for u in unique])


def plot_structure_3d(
    structure: dict[str, Any],
    output_path: str | Path | None = None,
) -> Path | None:
    """画晶体结构 3D 球棍模型.

    structure: {"lattice": [[a1],[a2],[a3]], "species": ["Si","Si"],
                "coords": [[x,y,z], ...], "bonds": [[i,j], ...] (可选)}
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    apply_arial_style()
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    species = structure.get("species", [])
    coords = structure.get("coords", [])
    bonds = structure.get("bonds", [])

    _plot_atoms(ax, species, coords)
    _plot_bonds(ax, coords, bonds)

    ax.set_xlabel("x (Å)", fontweight="bold")
    ax.set_ylabel("y (Å)", fontweight="bold")
    ax.set_zlabel("z (Å)", fontweight="bold")
    ax.set_title("Crystal Structure")
    _dedupe_legend(ax)

    fig.tight_layout()
    return _save_or_show(fig, output_path)
