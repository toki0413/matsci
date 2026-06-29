"""Experimental data tool — parse common materials characterization data.

Supports XRD pattern files (two-column text/CSV) and basic image metadata
for SEM/TEM images. Read-only and safe to auto-execute.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool
from huginn.types import ToolContext, ToolResult


class ExperimentalDataInput(BaseModel):
    action: Literal["xrd_load", "xrd_peaks", "xrd_d_spacing", "image_metadata"] = Field(
        ..., description="Experimental data action"
    )
    file_path: str | None = Field(
        default=None, description="Path to data file (required for load actions)"
    )
    two_theta: list[float] | None = Field(
        default=None, description="2θ values in degrees"
    )
    intensity: list[float] | None = Field(
        default=None, description="XRD intensity values"
    )
    wavelength: float | None = Field(
        default=1.5406, description="X-ray wavelength in Å (Cu Kα default)"
    )
    prominence: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Minimum relative prominence for peak detection",
    )
    max_peaks: int = Field(
        default=20, ge=1, le=200, description="Maximum peaks to report"
    )


class ExperimentalDataOutput(BaseModel):
    action: str
    count: int
    data: dict[str, Any]
    warnings: list[str] = []


class ExperimentalDataTool(HuginnTool):
    """Parse and analyze experimental materials characterization data."""

    name = "experimental_data_tool"
    category = "materials"
    description = (
        "Load XRD patterns, detect peaks, compute d-spacings, "
        "and read SEM/TEM image metadata."
    )
    input_schema = ExperimentalDataInput
    output_schema = ExperimentalDataOutput
    read_only = True

    def is_read_only(self, args: ExperimentalDataInput) -> bool:
        return True

    async def call(
        self, args: ExperimentalDataInput, context: ToolContext
    ) -> ToolResult:
        try:
            if args.action == "xrd_load":
                data, warnings = self._xrd_load(args)
            elif args.action == "xrd_peaks":
                data, warnings = self._xrd_peaks(args)
            elif args.action == "xrd_d_spacing":
                data, warnings = self._xrd_d_spacing(args)
            elif args.action == "image_metadata":
                data, warnings = self._image_metadata(args)
            else:  # pragma: no cover
                raise ValueError(f"Unknown action: {args.action}")

            output = ExperimentalDataOutput(
                action=args.action,
                count=data.get("count", 0),
                data=data,
                warnings=warnings,
            )
            return ToolResult(data=output.model_dump(exclude_none=True))
        except Exception as exc:  # pragma: no cover
            return ToolResult(data=None, success=False, error=str(exc))

    def _xrd_load(
        self, args: ExperimentalDataInput
    ) -> tuple[dict[str, Any], list[str]]:
        if args.file_path is None:
            raise ValueError("file_path is required for xrd_load")
        path = Path(args.file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        two_theta, intensity = _load_xy_data(path)
        if len(two_theta) == 0:
            raise ValueError("No numeric data found in file")

        return {
            "count": len(two_theta),
            "two_theta": two_theta.tolist(),
            "intensity": intensity.tolist(),
            "two_theta_range": [float(two_theta.min()), float(two_theta.max())],
            "intensity_max": float(intensity.max()),
        }, []

    def _xrd_peaks(
        self, args: ExperimentalDataInput
    ) -> tuple[dict[str, Any], list[str]]:
        two_theta, intensity = self._ensure_xy(args)
        peaks = _find_peaks(
            intensity, prominence=args.prominence, max_peaks=args.max_peaks
        )
        peak_list = []
        for idx, height in peaks:
            fwhm = _estimate_fwhm(intensity, idx)
            peak_list.append(
                {
                    "index": int(idx),
                    "two_theta": float(two_theta[idx]),
                    "intensity": float(height),
                    "fwhm_points": int(fwhm),
                }
            )
        return {"count": len(peak_list), "peaks": peak_list}, []

    def _xrd_d_spacing(
        self, args: ExperimentalDataInput
    ) -> tuple[dict[str, Any], list[str]]:
        two_theta, intensity = self._ensure_xy(args)
        peaks = _find_peaks(
            intensity, prominence=args.prominence, max_peaks=args.max_peaks
        )
        wavelength = args.wavelength or 1.5406
        results = []
        for idx, height in peaks:
            theta_rad = np.radians(two_theta[idx] / 2.0)
            d = wavelength / (2.0 * np.sin(theta_rad))
            results.append(
                {
                    "two_theta": float(two_theta[idx]),
                    "intensity": float(height),
                    "d_spacing_A": float(d),
                }
            )
        return {
            "count": len(results),
            "d_spacings": results,
            "wavelength_A": wavelength,
        }, []

    def _image_metadata(
        self, args: ExperimentalDataInput
    ) -> tuple[dict[str, Any], list[str]]:
        if args.file_path is None:
            raise ValueError("file_path is required for image_metadata")
        path = Path(args.file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        warnings: list[str] = []
        metadata: dict[str, Any] = {
            "file": str(path),
            "size_bytes": path.stat().st_size,
        }

        try:
            from PIL import Image

            with Image.open(path) as img:
                metadata["format"] = img.format
                metadata["mode"] = img.mode
                metadata["width"] = img.width
                metadata["height"] = img.height
                metadata["dpi"] = img.info.get("dpi")
        except ImportError:
            warnings.append("PIL not installed; only basic file metadata available")
        except Exception as exc:
            warnings.append(f"Could not read image metadata: {exc}")

        return {"count": 1, "metadata": metadata}, warnings

    def _ensure_xy(self, args: ExperimentalDataInput) -> tuple[np.ndarray, np.ndarray]:
        if args.two_theta is not None and args.intensity is not None:
            return np.asarray(args.two_theta, dtype=float), np.asarray(
                args.intensity, dtype=float
            )
        if args.file_path:
            return _load_xy_data(Path(args.file_path))
        raise ValueError("Provide either file_path or two_theta + intensity")


def _load_xy_data(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load two-column numeric data from text/csv."""
    rows: list[tuple[float, float]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "!", ";", "//", "/*")):
            continue
        parts = line.replace(",", " ").split()
        if len(parts) < 2:
            continue
        try:
            x = float(parts[0])
            y = float(parts[1])
        except ValueError:
            continue
        rows.append((x, y))
    if not rows:
        return np.array([]), np.array([])
    arr = np.asarray(rows, dtype=float)
    return arr[:, 0], arr[:, 1]


def _find_peaks(
    intensity: np.ndarray, prominence: float = 0.05, max_peaks: int = 20
) -> list[tuple[int, float]]:
    """Find local maxima above a relative prominence threshold.

    Returns list of (index, normalized_intensity) sorted by height.
    """
    if intensity.size < 3:
        return []
    baseline = np.percentile(intensity, 10)
    top = np.max(intensity)
    if top <= baseline:
        return []
    norm = (intensity - baseline) / (top - baseline)
    threshold = prominence
    peaks = []
    for i in range(1, len(norm) - 1):
        if norm[i] > threshold and norm[i] >= norm[i - 1] and norm[i] > norm[i + 1]:
            peaks.append((i, float(norm[i])))
    peaks.sort(key=lambda p: p[1], reverse=True)
    return peaks[:max_peaks]


def _estimate_fwhm(intensity: np.ndarray, idx: int) -> int:
    """Estimate full width at half maximum in data points."""
    if idx <= 0 or idx >= len(intensity) - 1:
        return 0
    half = intensity[idx] / 2.0
    left = idx
    while left > 0 and intensity[left] > half:
        left -= 1
    right = idx
    while right < len(intensity) - 1 and intensity[right] > half:
        right += 1
    return right - left
