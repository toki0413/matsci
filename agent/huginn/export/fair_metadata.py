"""FAIR data principles — make agent outputs Findable, Accessible,
Interoperable, Reusable.

Generates schema.org/Dataset JSON-LD metadata for research outputs,
following DataCite and schema.org standards.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Default license for agent-generated datasets. CC-BY-4.0 is the most
# permissive standard license that still requires attribution, which is
# what most funders expect for open research data.
_DEFAULT_LICENSE = "https://creativecommons.org/licenses/by/4.0/"

_DEFAULT_CREATOR = "HuginnAgent"

# Maps common result-dict keys to schema.org PropertyValue labels so the
# variableMeasured list is useful without the caller doing extra work.
_RESULT_KEY_LABELS: dict[str, str] = {
    "energy": "Total Energy (eV)",
    "band_gap": "Band Gap (eV)",
    "volume": "Volume (Å³)",
    "bulk_modulus": "Bulk Modulus (GPa)",
    "magnetization": "Magnetization (μB)",
    "lattice_a": "Lattice constant a (Å)",
    "lattice_b": "Lattice constant b (Å)",
    "lattice_c": "Lattice constant c (Å)",
    "force": "Force (eV/Å)",
    "stress": "Stress (GPa)",
    "convergence": "Convergence",
    "tests_passed": "Tests Passed",
}


def _extract_variables(results: Any) -> list[dict[str, Any]]:
    """Pull measurable properties out of *results* into PropertyValue dicts.

    Handles dict-of-dicts, list-of-dicts, and flat dict shapes. Unknown
    keys are still included with their raw key as the label — better to
    over-report than to drop a variable the researcher needs.
    """
    variables: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _add(key: str, value: Any) -> None:
        if key in seen or value is None:
            return
        seen.add(key)
        label = _RESULT_KEY_LABELS.get(key, key)
        entry: dict[str, Any] = {
            "@type": "PropertyValue",
            "name": label,
        }
        if isinstance(value, (int, float)):
            entry["value"] = value
        elif isinstance(value, str):
            entry["value"] = value
        elif isinstance(value, dict):
            # nested result dict — flatten one level
            for k2, v2 in value.items():
                if isinstance(v2, (int, float, str)):
                    _add(f"{key}.{k2}", v2)
            return  # already added sub-keys
        variables.append(entry)

    if isinstance(results, dict):
        for k, v in results.items():
            _add(k, v)
    elif isinstance(results, list):
        for item in results:
            if isinstance(item, dict):
                for k, v in item.items():
                    _add(k, v)

    return variables


def generate_dataset_metadata(
    run_id: str,
    objective: str,
    results: Any,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a schema.org/Dataset JSON-LD dict for a research run.

    Args:
        run_id: Unique run identifier (e.g. ``loop_abc123``).
        objective: The research objective / hypothesis tested.
        results: Result data — a dict, list of dicts, or anything
            ``_extract_variables`` can walk. Values become
            ``variableMeasured`` entries.
        provenance: Optional provenance info (tool chain, timestamps,
            trajectory path). Mapped into ``wasGeneratedBy``.

    Returns:
        A dict ready to serialize as JSON-LD. The ``@context`` is
        schema.org so any JSON-LD parser can interpret it.
    """
    now = datetime.now(timezone.utc).isoformat()

    # ── provenance → wasGeneratedBy ──
    prov = provenance or {}
    was_generated_by: dict[str, Any] = {
        "@type": "ResearchAction",
        "name": f"HuginnAgent run {run_id}",
        "description": objective,
    }
    if prov.get("trajectory_path"):
        was_generated_by["instrument"] = prov["trajectory_path"]
    if prov.get("provenance_path"):
        was_generated_by["object"] = prov["provenance_path"]
    if prov.get("start_time"):
        was_generated_by["startTime"] = prov["start_time"]
    if prov.get("end_time"):
        was_generated_by["endTime"] = prov["end_time"]

    # ── distribution (where to find the output) ──
    distribution: list[dict[str, Any]] = []
    report_path = prov.get("report_path")
    if report_path:
        distribution.append({
            "@type": "DataDownload",
            "encodingFormat": "text/markdown",
            "contentUrl": f"file://{report_path}",
            "name": "Research Report",
        })
    if prov.get("trajectory_path"):
        distribution.append({
            "@type": "DataDownload",
            "encodingFormat": "application/json",
            "contentUrl": f"file://{prov['trajectory_path']}",
            "name": "Execution Trajectory",
        })

    metadata: dict[str, Any] = {
        "@context": "https://schema.org/",
        "@type": "Dataset",
        "name": f"HuginnAgent Research Output: {run_id}",
        "description": objective,
        "creator": {
            "@type": "Organization",
            "name": _DEFAULT_CREATOR,
        },
        "dateCreated": now,
        "license": _DEFAULT_LICENSE,
        "keywords": ["materials science", "computational", "agent-generated"],
        "isAccessibleForFree": True,
        "variableMeasured": _extract_variables(results),
        "measurementTechnique": prov.get(
            "measurement_technique",
            "Automated computational workflow via HuginnAgent",
        ),
        "wasGeneratedBy": was_generated_by,
    }
    if distribution:
        metadata["distribution"] = distribution

    return metadata


def write_fair_jsonld(metadata: dict[str, Any], output_path: str | Path) -> Path:
    """Write a JSON-LD metadata dict to *output_path* as UTF-8 JSON.

    Returns the resolved ``Path`` for convenience.
    """
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return p


def generate_citation(metadata: dict[str, Any]) -> str:
    """Return a BibTeX citation string for a dataset metadata dict.

    Uses the run_id from ``wasGeneratedBy.name`` as the cite key, falling
    back to ``name`` if the structured field is missing.
    """
    # Pull a cite key from the metadata.
    wgb = metadata.get("wasGeneratedBy", {})
    name_field = wgb.get("name", "") if isinstance(wgb, dict) else ""
    # "HuginnAgent run loop_abc123" → "huginn_loop_abc123"
    parts = name_field.replace(":", "").split()
    if len(parts) >= 2:
        cite_key = "_".join(
            p.lower() for p in parts if p.lower() not in ("run",)
        )
    else:
        cite_key = "huginn_dataset"

    title = metadata.get("name", "Untitled Dataset")
    creator = metadata.get("creator", {})
    if isinstance(creator, dict):
        author = creator.get("name", _DEFAULT_CREATOR)
    else:
        author = str(creator)
    year = (metadata.get("dateCreated", "")[:4]) or str(
        datetime.now(timezone.utc).year
    )
    url = ""
    dist = metadata.get("distribution", [])
    if dist and isinstance(dist, list) and isinstance(dist[0], dict):
        url = dist[0].get("contentUrl", "")
    license_url = metadata.get("license", _DEFAULT_LICENSE)

    lines = [
        f"@dataset{{{cite_key},",
        f"  title       = {{{title}}},",
        f"  author      = {{{author}}},",
        f"  year        = {{{year}}},",
        f"  license     = {{{license_url}}},",
    ]
    if url:
        lines.append(f"  url         = {{{url}}},")
    lines.append("}")
    return "\n".join(lines)
