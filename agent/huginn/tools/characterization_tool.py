"""Characterization data analysis tool.

Parses common experimental outputs (XRD, Raman/IR, PDF, TEM, XPS) and extracts
actionable structural/chemical information. Heavy dependencies are imported
lazily.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool
from huginn.types import ToolContext, ToolResult


class CharacterizationInput(BaseModel):
    action: Literal[
        "xrd_peak_detect",
        "spectroscopy_peak_fit",
        "pdf_fit",
        "tem_image_analysis",
        "xps_peak_fit",
    ] = Field(...)
    data_path: str = Field(..., description="Path to CSV/JSON data file")
    output_path: str | None = Field(
        default=None, description="Optional path to save annotated results"
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Action-specific parameters (threshold, distance, etc.)",
    )


class CharacterizationTool(HuginnTool):
    """Analyze XRD, spectroscopy, PDF, microscopy, and XPS data."""

    name = "characterization_tool"
    category = "cv"
    description = (
        "Detect peaks in XRD/spectroscopy data, fit PDFs, and analyze "
        "microscopy / XPS data."
    )
    input_schema = CharacterizationInput
    read_only = True

    def is_read_only(self, args: CharacterizationInput) -> bool:
        return True

    async def call(
        self, args: CharacterizationInput, context: ToolContext
    ) -> ToolResult:
        if not Path(args.data_path).exists():
            return ToolResult(
                data=None,
                success=False,
                error=f"Data file not found: {args.data_path}",
            )

        try:
            if args.action == "xrd_peak_detect":
                return self._xrd_peak_detect(args)
            if args.action == "spectroscopy_peak_fit":
                return self._spectroscopy_peak_fit(args)
            if args.action == "pdf_fit":
                return self._pdf_fit(args)
            if args.action == "tem_image_analysis":
                return self._tem_image_analysis(args)
            if args.action == "xps_peak_fit":
                return self._xps_peak_fit(args)
        except Exception as exc:
            return ToolResult(data=None, success=False, error=str(exc))

        return ToolResult(
            data=None, success=False, error=f"Unknown action: {args.action}"
        )

    def _read_xy(self, path: str) -> tuple[np.ndarray, np.ndarray]:
        suffix = Path(path).suffix.lower()
        if suffix == ".json":
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            x = np.asarray(
                data.get("x") or data.get("2theta") or data.get("wavenumber"),
                dtype=float,
            )
            y = np.asarray(data.get("y") or data.get("intensity"), dtype=float)
            return x, y

        rows = []
        with Path(path).open("r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if not row:
                    continue
                try:
                    rows.append([float(row[0]), float(row[1])])
                except ValueError:
                    continue
        arr = np.asarray(rows, dtype=float)
        return arr[:, 0], arr[:, 1]

    def _xrd_peak_detect(self, args: CharacterizationInput) -> ToolResult:
        x, y = self._read_xy(args.data_path)
        threshold = float(args.parameters.get("threshold", np.max(y) * 0.05))
        distance = int(args.parameters.get("distance", 5))

        try:
            from scipy.signal import find_peaks

            peaks, props = find_peaks(y, height=threshold, distance=distance)
        except ImportError as exc:
            raise RuntimeError(
                "XRD peak detection requires scipy. Install: pip install scipy"
            ) from exc

        results = []
        for idx in peaks:
            results.append(
                {
                    "2theta": float(x[idx]),
                    "intensity": float(y[idx]),
                }
            )

        results.sort(key=lambda p: p["intensity"], reverse=True)
        data = {"peaks": results, "n_peaks": len(results)}
        self._maybe_save(args.output_path, data)
        return ToolResult(data=data)

    def _spectroscopy_peak_fit(self, args: CharacterizationInput) -> ToolResult:
        x, y = self._read_xy(args.data_path)
        threshold = float(args.parameters.get("threshold", np.max(y) * 0.05))
        distance = int(args.parameters.get("distance", 5))

        try:
            from scipy.signal import find_peaks

            peaks, props = find_peaks(y, height=threshold, distance=distance)
        except ImportError as exc:
            raise RuntimeError(
                "Spectroscopy peak fitting requires scipy. Install: pip install scipy"
            ) from exc

        results = []
        for idx in peaks:
            # Estimate FWHM using interpolated half-max.
            half = float(y[idx]) / 2.0
            left = idx
            while left > 0 and y[left] > half:
                left -= 1
            right = idx
            while right < len(y) - 1 and y[right] > half:
                right += 1
            fwhm = float(x[min(len(x) - 1, right)] - x[left])
            results.append(
                {
                    "position": float(x[idx]),
                    "intensity": float(y[idx]),
                    "fwhm": fwhm,
                }
            )

        results.sort(key=lambda p: p["intensity"], reverse=True)
        data = {"peaks": results, "n_peaks": len(results)}
        self._maybe_save(args.output_path, data)
        return ToolResult(data=data)

    def _pdf_fit(self, args: CharacterizationInput) -> ToolResult:
        # Minimal real-space Gaussian envelope fit for PDF-like data.
        x, y = self._read_xy(args.data_path)
        n_peaks = int(args.parameters.get("n_peaks", 3))
        try:
            from scipy.optimize import curve_fit
            from scipy.signal import find_peaks
        except ImportError as exc:
            raise RuntimeError(
                "PDF fitting requires scipy. Install: pip install scipy"
            ) from exc

        def _pdf_envelope(r, *params):
            out = np.zeros_like(r)
            for i in range(0, len(params), 4):
                a, mu, sigma, damp = params[i : i + 4]
                out += a * np.exp(-damp * r) * np.exp(-0.5 * ((r - mu) / sigma) ** 2)
            return out

        peaks, _ = find_peaks(y, distance=5)
        chosen = sorted(peaks, key=lambda i: y[i], reverse=True)[:n_peaks]
        p0 = []
        for idx in chosen:
            p0.extend([float(y[idx]), float(x[idx]), 0.1, 0.01])

        bounds = (
            [0.0] * len(p0),
            [np.inf, max(x), max(x) - min(x), 1.0] * (len(p0) // 4),
        )
        try:
            popt, _ = curve_fit(_pdf_envelope, x, y, p0=p0, bounds=bounds, maxfev=10000)
        except Exception as exc:
            return ToolResult(
                data=None,
                success=False,
                error=f"PDF envelope fit failed: {exc}",
            )

        fit_peaks = []
        for i in range(0, len(popt), 4):
            a, mu, sigma, damp = popt[i : i + 4]
            fit_peaks.append(
                {
                    "r": float(mu),
                    "amplitude": float(a),
                    "sigma": float(sigma),
                    "damping": float(damp),
                    "fwhm": float(2.355 * sigma),
                }
            )
        data = {"peaks": fit_peaks, "n_peaks": len(fit_peaks)}
        self._maybe_save(args.output_path, data)
        return ToolResult(data=data)

    def _tem_image_analysis(self, args: CharacterizationInput) -> ToolResult:
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError(
                "TEM image analysis requires Pillow. Install: pip install Pillow"
            ) from exc

        img = Image.open(args.data_path).convert("L")
        arr = np.asarray(img, dtype=float)

        try:
            from scipy.fft import fft2, fftshift
            from scipy.signal import find_peaks
        except ImportError as exc:
            raise RuntimeError(
                "TEM image analysis requires scipy. Install: pip install scipy"
            ) from exc

        fft = fftshift(fft2(arr - arr.mean()))
        power = np.abs(fft) ** 2

        # Radial average of the power spectrum.
        cy, cx = np.array(power.shape) // 2
        y, x = np.indices(power.shape)
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)
        radial = np.bincount(r.ravel(), power.ravel()) / np.bincount(r.ravel())
        radial[:1] = 0  # suppress DC

        peaks, _ = find_peaks(
            radial,
            height=float(args.parameters.get("height", np.max(radial) * 0.1)),
            distance=int(args.parameters.get("distance", 3)),
        )

        # Convert pixel frequencies to real-space d-spacings using pixel_size.
        pixel_size = float(args.parameters.get("pixel_size", 1.0))
        d_spacings = []
        for freq in peaks:
            if freq > 0:
                d_spacings.append(float(pixel_size / freq))

        data = {
            "image_shape": arr.shape,
            "dominant_frequencies": [int(f) for f in peaks[:5]],
            "d_spacings": d_spacings[:5],
            "pixel_size": pixel_size,
        }
        self._maybe_save(args.output_path, data)
        return ToolResult(data=data)

    def _xps_peak_fit(self, args: CharacterizationInput) -> ToolResult:
        x, y = self._read_xy(args.data_path)
        n_peaks = int(args.parameters.get("n_peaks", 2))
        try:
            from scipy.optimize import curve_fit
            from scipy.signal import find_peaks
        except ImportError as exc:
            raise RuntimeError(
                "XPS peak fitting requires scipy. Install: pip install scipy"
            ) from exc

        def _gaussian(x, a, mu, sigma, bg):
            return bg + a * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

        def _multi_gaussian(x, *params):
            bg = params[-1]
            out = np.full_like(x, bg, dtype=float)
            for i in range(0, len(params) - 1, 3):
                a, mu, sigma = params[i : i + 3]
                out += a * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
            return out

        peaks, _ = find_peaks(y, distance=5)
        chosen = sorted(peaks, key=lambda i: y[i], reverse=True)[:n_peaks]
        p0 = []
        for idx in chosen:
            p0.extend([float(y[idx]), float(x[idx]), 0.5])
        p0.append(np.min(y))

        try:
            popt, _ = curve_fit(_multi_gaussian, x, y, p0=p0, maxfev=20000)
        except Exception as exc:
            return ToolResult(
                data=None,
                success=False,
                error=f"XPS peak fit failed: {exc}",
            )

        fit_peaks = []
        for i in range(0, len(popt) - 1, 3):
            a, mu, sigma = popt[i : i + 3]
            fit_peaks.append(
                {
                    "binding_energy": float(mu),
                    "amplitude": float(a),
                    "sigma": float(sigma),
                    "fwhm": float(2.355 * sigma),
                    "area": float(a * sigma * np.sqrt(2 * np.pi)),
                }
            )
        data = {
            "peaks": sorted(fit_peaks, key=lambda p: p["binding_energy"]),
            "n_peaks": len(fit_peaks),
            "background": float(popt[-1]),
        }
        self._maybe_save(args.output_path, data)
        return ToolResult(data=data)

    def _maybe_save(self, output_path: str | None, data: dict[str, Any]) -> None:
        if output_path:
            Path(output_path).write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
