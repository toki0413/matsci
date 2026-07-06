"""Tests for FAIR metadata generation (schema.org/Dataset JSON-LD).

Covers metadata generation, required fields, variableMeasured extraction,
BibTeX citation, and file writing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from huginn.export.fair_metadata import (
    generate_citation,
    generate_dataset_metadata,
    write_fair_jsonld,
)


# ── generate_dataset_metadata ────────────────────────────────────


class TestGenerateDatasetMetadata:
    def test_returns_valid_json_ld(self):
        md = generate_dataset_metadata(
            run_id="loop_test01",
            objective="Optimize C-S-H defect kinetics",
            results={"energy": -42.5, "band_gap": 1.2},
        )
        assert md["@context"] == "https://schema.org/"
        assert md["@type"] == "Dataset"

    def test_has_required_fields(self):
        md = generate_dataset_metadata(
            run_id="loop_test02",
            objective="DFT band structure calculation",
            results={},
        )
        for field in ("name", "description", "creator", "license", "dateCreated"):
            assert field in md, f"missing required field: {field}"

    def test_creator_is_organization(self):
        md = generate_dataset_metadata("r1", "test", {})
        creator = md["creator"]
        assert isinstance(creator, dict)
        assert creator["@type"] == "Organization"
        assert "name" in creator

    def test_license_is_url(self):
        md = generate_dataset_metadata("r1", "test", {})
        assert md["license"].startswith("https://")

    def test_run_id_in_name(self):
        md = generate_dataset_metadata("loop_abc123", "test", {})
        assert "loop_abc123" in md["name"]

    def test_objective_in_description(self):
        md = generate_dataset_metadata("r1", "My research objective", {})
        assert md["description"] == "My research objective"


# ── variableMeasured ─────────────────────────────────────────────


class TestVariableMeasured:
    def test_variables_extracted_from_dict(self):
        md = generate_dataset_metadata(
            "r1", "test", {"energy": -10.5, "band_gap": 0.8}
        )
        var_names = [v["name"] for v in md["variableMeasured"]]
        assert any("Energy" in n for n in var_names)
        assert any("Band Gap" in n for n in var_names)

    def test_variables_extracted_from_list(self):
        md = generate_dataset_metadata(
            "r1", "test", [{"energy": -5.0}, {"volume": 100.0}]
        )
        var_names = [v["name"] for v in md["variableMeasured"]]
        assert any("Energy" in n for n in var_names)
        assert any("Volume" in n for n in var_names)

    def test_empty_results_gives_empty_variables(self):
        md = generate_dataset_metadata("r1", "test", {})
        assert md["variableMeasured"] == []

    def test_nested_dict_flattened(self):
        md = generate_dataset_metadata(
            "r1", "test", {"convergence": {"energy": 1e-6, "force": 1e-4}}
        )
        var_names = [v["name"] for v in md["variableMeasured"]]
        assert any("energy" in n.lower() for n in var_names)
        assert any("force" in n.lower() for n in var_names)


# ── generate_citation ─────────────────────────────────────────────


class TestGenerateCitation:
    def test_returns_bibtex_format(self):
        md = generate_dataset_metadata("loop_x1", "test objective", {"energy": -1.0})
        cite = generate_citation(md)
        assert cite.startswith("@dataset{")
        assert cite.endswith("}")
        assert "title" in cite
        assert "author" in cite
        assert "year" in cite

    def test_citation_contains_title(self):
        md = generate_dataset_metadata("loop_x2", "My Research", {})
        cite = generate_citation(md)
        assert "My Research" in cite or "loop_x2" in cite

    def test_citation_contains_license(self):
        md = generate_dataset_metadata("loop_x3", "test", {})
        cite = generate_citation(md)
        assert "creativecommons" in cite


# ── write_fair_jsonld ────────────────────────────────────────────


class TestWriteFairJsonld:
    def test_creates_file(self, tmp_path):
        md = generate_dataset_metadata("loop_w1", "test", {"energy": -1.0})
        out = tmp_path / "loop_w1_dataset.jsonld"
        result = write_fair_jsonld(md, out)
        assert result == out
        assert out.exists()

    def test_file_is_valid_json(self, tmp_path):
        md = generate_dataset_metadata("loop_w2", "test", {})
        out = tmp_path / "loop_w2_dataset.jsonld"
        write_fair_jsonld(md, out)
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["@type"] == "Dataset"
        assert data["name"] == "HuginnAgent Research Output: loop_w2"

    def test_creates_parent_dirs(self, tmp_path):
        md = generate_dataset_metadata("loop_w3", "test", {})
        out = tmp_path / "nested" / "dir" / "loop_w3_dataset.jsonld"
        write_fair_jsonld(md, out)
        assert out.exists()

    def test_with_provenance(self, tmp_path):
        md = generate_dataset_metadata(
            "loop_w4",
            "test with provenance",
            {"energy": -5.0},
            provenance={
                "report_path": "/tmp/report.md",
                "trajectory_path": "/tmp/traj.json",
                "provenance_path": "/tmp/prov.jsonl",
                "start_time": "2026-01-01T10:00:00",
                "end_time": "2026-01-01T11:00:00",
            },
        )
        assert "distribution" in md
        assert len(md["distribution"]) == 2
        assert md["wasGeneratedBy"]["startTime"] == "2026-01-01T10:00:00"
