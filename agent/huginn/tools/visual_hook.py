"""Auto-render numerical tool results as charts for VLM analysis.

When a tool returns structured numerical data (energy values, spectra, etc.),
this hook generates a quick matplotlib chart and attaches it as base64 to
the tool result. The agent loop can then pass it to a VLM for visual analysis.
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Tool names that produce numerical data suitable for visualization
_VISUALIZABLE_PATTERNS = {
    "thermo_tool": "phase_diagram",     # formation energies -> scatter
    "band_structure": "line_plot",       # band structure -> line plot
    "dos": "line_plot",                  # density of states -> line plot
    "phonon": "line_plot",               # phonon dispersion -> line plot
    "mechanical_tool": "stress_strain",  # stress-strain curve
    "benchmark": "bar_chart",            # benchmark results -> bar chart
    "evolution": "convergence",          # evolution convergence
}

# Max base64 image size to avoid context bloat (256KB)
_MAX_IMAGE_BYTES = 256 * 1024


def should_visualize(tool_name: str, output: dict[str, Any]) -> bool:
    """Check if a tool's output is suitable for auto-visualization."""
    if not output or not output.get("result"):
        return False
    result = output["result"]
    if not isinstance(result, dict):
        return False

    for pattern in _VISUALIZABLE_PATTERNS:
        if pattern in tool_name.lower():
            return True

    for key in ("energies", "bands", "dos", "frequencies", "stress", "strain", "scores"):
        if key in result and isinstance(result[key], (list, dict)):
            return True

    return False


def render_tool_output(tool_name: str, output: dict[str, Any]) -> str | None:
    """Generate a quick chart from tool output, return as base64 string.

    Returns None if rendering fails or data is unsuitable.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.debug("matplotlib not available, skipping visualization")
        return None

    result = output.get("result", {})
    if not isinstance(result, dict):
        return None

    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    plotted = False

    # Band structure / DOS / phonon -> line plot
    for key in ("bands", "dos", "frequencies"):
        data = result.get(key)
        if isinstance(data, list) and data:
            if isinstance(data[0], list):
                for line in data[:20]:  # cap at 20 lines
                    if isinstance(line, list):
                        ax.plot(line, linewidth=0.8)
            else:
                ax.plot(data, linewidth=1.2)
            ax.set_xlabel("k-path" if key == "bands" else key)
            ax.set_ylabel(key)
            plotted = True
            break

    # Energies -> scatter/bar
    if not plotted:
        energies = result.get("energies") or result.get("energy")
        if isinstance(energies, list) and energies:
            labels = result.get("labels", [f"#{i}" for i in range(len(energies))])
            ax.bar(labels[:20], energies[:20])
            ax.set_ylabel("Energy (eV)")
            plotted = True
        elif isinstance(energies, (int, float)):
            ax.text(0.5, 0.5, f"E = {energies:.4f} eV", ha="center", va="center", fontsize=16)
            plotted = True

    # Stress-strain
    if not plotted:
        stress = result.get("stress")
        strain = result.get("strain")
        if isinstance(stress, list) and isinstance(strain, list):
            ax.plot(strain, stress, "b-o", markersize=4)
            ax.set_xlabel("Strain")
            ax.set_ylabel("Stress (GPa)")
            plotted = True

    # Scores -> bar chart
    if not plotted:
        scores = result.get("scores")
        if isinstance(scores, dict):
            names = list(scores.keys())[:10]
            values = [scores[n] for n in names]
            ax.barh(names, values)
            ax.set_xlabel("Score")
            plotted = True

    if not plotted:
        plt.close(fig)
        return None

    ax.set_title(f"{tool_name} result", fontsize=12, fontweight="bold")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)

    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    if len(b64) > _MAX_IMAGE_BYTES:
        return None  # too large for context

    return b64


def enrich_with_visual(tool_name: str, output: dict[str, Any]) -> dict[str, Any]:
    """Post-process tool output: add _visual_base64 field if renderable."""
    if not should_visualize(tool_name, output):
        return output

    b64 = render_tool_output(tool_name, output)
    if b64:
        output["_visual_base64"] = b64
        output["_visual_hint"] = (
            "A chart has been auto-generated from this tool's numerical output. "
            "Even without vision capability, visualize the data shape in your reasoning: "
            "describe trends, identify peaks/anomalies, and compare against expected patterns. "
            # ponytail: Mirage effect — text LLMs have latent visual reasoning
            # (Mirage: The Illusion of Visual Understanding, arXiv:2603.21687).
            # We exploit it by prompting the LLM to "imagine" the chart.
        )

    return output
