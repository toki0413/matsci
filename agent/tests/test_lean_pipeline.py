"""End-to-end tests for the Python → Lean 4 stability verification pipeline."""

import pytest
from pathlib import Path

from huginn.lean.pipeline import StabilityPipeline


LEAN_PROJECT = Path(__file__).parent.parent / "lean" / "HuginnLean"


class TestStabilityPipeline:
    @pytest.fixture(scope="class")
    def pipe(self):
        if not (LEAN_PROJECT / "lakefile.toml").exists():
            pytest.skip("HuginnLean project not found")
        return StabilityPipeline(LEAN_PROJECT)

    def test_cubic_iron(self, pipe):
        result = pipe.verify_cubic({"C11": 230.0, "C12": 135.0, "C44": 117.0})
        assert result.success, result.stderr
        assert "true" in result.stdout
        # Zener ratio for iron: A = 2*117/(230-135) ≈ 2.46
        lines = result.stdout.strip().splitlines()
        assert len(lines) >= 4
        zener = float(lines[2].strip())
        assert 2.4 < zener < 2.5
        au = float(lines[3].strip())
        assert au > 0.5  # iron is anisotropic

    def test_cubic_unstable(self, pipe):
        result = pipe.verify_cubic({"C11": 100.0, "C12": 150.0, "C44": 50.0})
        assert result.success, result.stderr
        assert "false" in result.stdout

    def test_hexagonal_zinc(self, pipe):
        result = pipe.verify_hexagonal({
            "C11": 165.0, "C12": 31.0, "C13": 50.0,
            "C33": 61.0, "C44": 39.0, "C66": 67.0,
        })
        assert result.success, result.stderr
        assert "true" in result.stdout
        # Check that moduli and AU are present in output
        lines = result.stdout.strip().splitlines()
        assert len(lines) >= 6
        # Kv ≈ 72.6, Kr ≈ 64.1 for zinc
        kv = float(lines[1].strip())
        kr = float(lines[2].strip())
        assert 70.0 < kv < 75.0
        assert 60.0 < kr < 70.0
        au = float(lines[5].strip())
        assert au > 0.0  # zinc is anisotropic

    def test_orthorhombic_olivine(self, pipe):
        result = pipe.verify_orthorhombic({
            "C11": 324.0, "C22": 197.0, "C33": 235.0,
            "C44": 64.0,  "C55": 78.0,  "C66": 79.0,
            "C12": 59.0,  "C13": 79.0,  "C23": 78.0,
        })
        assert result.success, result.stderr
        assert "true" in result.stdout
        lines = result.stdout.strip().splitlines()
        assert len(lines) >= 6
        au = float(lines[5].strip())
        assert au > 0.0  # olivine is anisotropic

    def test_dispatcher_cubic(self, pipe):
        result = pipe.verify("cubic", {"C11": 230.0, "C12": 135.0, "C44": 117.0})
        assert result.success
        assert "true" in result.stdout

    def test_dispatcher_invalid_system(self, pipe):
        with pytest.raises(ValueError):
            pipe.verify("monoclinic", {})
