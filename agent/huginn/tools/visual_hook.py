"""Auto-render numerical tool results as charts for VLM analysis.

When a tool returns structured numerical data (energy values, spectra, etc.),
this hook generates a quick matplotlib chart and attaches it as base64 to
the tool result. The agent loop can then pass it to a VLM for visual analysis.

Also extracts structured "visual primitives" — numerical deictic pointers
(peak positions, trends, anomalies) — that give text-only LLMs concrete
coordinates to reason about, inspired by Thinking with Visual Primitives'
"point while it reasons" principle.
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


def extract_visual_primitives(tool_name: str, output: dict[str, Any]) -> str:
    """Extract numerical deictic pointers from tool output.

    Inspired by Thinking with Visual Primitives: instead of vague "imagine
    the chart", give the LLM concrete coordinates — peak positions, trends,
    anomaly indices — that anchor its reasoning to the data shape.

    Also draws on 3D Primitives are a Spatial Language: structured primitives
    (here numerical, not geometric) serve as a bridge between text reasoning
    and visual understanding. This is the text-only path of our Mirage
    strategy — we don't need the LLM to see the image, we give it the
    primitives that describe the image.
    """
    result = output.get("result", {})
    if not isinstance(result, dict):
        return ""

    lines: list[str] = []

    # 1D numeric lists: bands, dos, frequencies, energies, stress, strain
    for key in ("bands", "dos", "frequencies", "energies", "stress", "strain"):
        data = result.get(key)
        if not isinstance(data, list) or not data:
            continue
        # Handle nested lists (e.g. bands = [[...], [...]]) — flatten to summary
        if isinstance(data[0], list):
            sub_lines = []
            for bi, band in enumerate(data[:5]):
                try:
                    nums = [float(v) for v in band if isinstance(v, (int, float))]
                except (ValueError, TypeError):
                    continue
                if not nums:
                    continue
                pk = max(range(len(nums)), key=lambda i: nums[i])
                mn = min(range(len(nums)), key=lambda i: nums[i])
                sub_lines.append(f"  band{bi}: peak=idx {pk}({nums[pk]:.4f}), min=idx {mn}({nums[mn]:.4f})")
            if sub_lines:
                lines.append(f"[{key}] {len(data)} bands:\n" + "\n".join(sub_lines))
            continue
        try:
            nums = [float(v) for v in data if isinstance(v, (int, float))]
        except (ValueError, TypeError):
            continue
        if not nums:
            continue

        n = len(nums)
        peak_idx = max(range(n), key=lambda i: nums[i])
        min_idx = min(range(n), key=lambda i: nums[i])
        mean_v = sum(nums) / n
        std_v = (sum((x - mean_v) ** 2 for x in nums) / n) ** 0.5

        # trend: compare first half mean vs second half mean
        mid = n // 2
        if mid > 0:
            first_half = sum(nums[:mid]) / mid
            second_half = sum(nums[mid:]) / (n - mid)
            if second_half > first_half * 1.05:
                trend = "increasing"
            elif second_half < first_half * 0.95:
                trend = "decreasing"
            else:
                trend = "approximately flat"
        else:
            trend = "unknown"

        # anomalies: points > 2 std from mean
        anomalies = []
        if std_v > 0:
            for i, v in enumerate(nums):
                if abs(v - mean_v) > 2 * std_v:
                    anomalies.append(f"idx {i}={v:.4f}")
        anomalies_str = ", ".join(anomalies[:5]) if anomalies else "none"

        lines.append(
            f"[{key}] n={n}, peak=idx {peak_idx} ({nums[peak_idx]:.4f}), "
            f"min=idx {min_idx} ({nums[min_idx]:.4f}), "
            f"mean={mean_v:.4f}, std={std_v:.4f}, "
            f"trend={trend}, anomalies={anomalies_str}"
        )

    # Dict scores: top/bottom items
    scores = result.get("scores")
    if isinstance(scores, dict) and scores:
        try:
            items = sorted(scores.items(), key=lambda kv: float(kv[1]), reverse=True)
            top = ", ".join(f"{k}={v:.3f}" for k, v in items[:3])
            bot = ", ".join(f"{k}={v:.3f}" for k, v in items[-2:])
            lines.append(f"[scores] top3: {top}; bottom: {bot}")
        except (ValueError, TypeError):
            pass

    if not lines:
        return ""

    return "\n".join(lines)


def enrich_with_visual(tool_name: str, output: dict[str, Any]) -> dict[str, Any]:
    """Post-process tool output: add _visual_base64 + _visual_primitives."""
    if not should_visualize(tool_name, output):
        return output

    b64 = render_tool_output(tool_name, output)
    primitives = extract_visual_primitives(tool_name, output)

    if b64:
        output["_visual_base64"] = b64

    if primitives:
        # Structured visual primitives as deictic pointers for text LLM reasoning.
        # Thinking with Visual Primitives: "point while it reasons" — give
        # coordinates, not vague descriptions. 3D Primitives: primitives as a
        # spatial language bridging text and visual. Mirage effect: text LLMs
        # have latent visual reasoning; these primitives activate it without
        # requiring actual image input.
        # When a VLM is available, _visual_base64 provides the actual image;
        # when not, _visual_primitives gives the LLM enough structure to
        # "visualize" the data shape through coordinates alone.
        output["_visual_hint"] = (
            "Visual primitives extracted from tool output (use as deictic pointers):\n"
            f"{primitives}\n"
            "Reason about the data shape using these coordinates: "
            "where are peaks/valleys, what does the trend imply, "
            "do anomalies suggest physics or noise?"
        )

    return output
