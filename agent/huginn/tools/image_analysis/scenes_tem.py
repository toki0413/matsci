"""TEM 晶格分析 — FFT 径向功率谱 + d-spacing 晶面匹配."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from huginn.types import ToolResult
from huginn.tools.image_analysis._utils import load_gray

if TYPE_CHECKING:
    from huginn.tools.image_analysis.tool import ImageAnalysisInput

logger = logging.getLogger(__name__)


# 常见晶面 d-spacing 参考表 (nm), 用于 TEM FFT 峰匹配
# 数值取常见金属/半导体, 仅做粗略指认, 真要做标定还得知道晶格常数
_D_SPACING_TABLE: dict[str, float] = {
    # FCC (Au/Ag/Cu/Pt/Al/Ni)
    "FCC(111)": 0.235, "FCC(200)": 0.204, "FCC(220)": 0.144, "FCC(311)": 0.123,
    # BCC (Fe/W/Cr/Mo)
    "BCC(110)": 0.203, "BCC(200)": 0.143, "BCC(211)": 0.117,
    # HCP (Ti/Mg/Zn/Zr)
    "HCP(100)": 0.247, "HCP(002)": 0.236, "HCP(101)": 0.209,
    # 石墨/石墨烯
    "Graphite(002)": 0.335, "Graphite(100)": 0.213, "Graphite(110)": 0.123,
    # 硅 (金刚石立方)
    "Si(111)": 0.314, "Si(220)": 0.192, "Si(311)": 0.164,
    # TiO2 锐钛矿常见面
    "Anatase(101)": 0.352, "Anatase(200)": 0.189,
}


def tem_lattice(args: "ImageAnalysisInput") -> ToolResult:
    arr = load_gray(args.image_path)
    pixel_size = float(args.parameters.get("pixel_size_nm", 1.0))
    fft_threshold = args.parameters.get("fft_threshold", None)

    try:
        from scipy.fft import fft2, fftshift
        from scipy.signal import find_peaks
    except ImportError as exc:
        return ToolResult(
            data=None,
            success=False,
            error=f"TEM 晶格分析需要 scipy.fft 和 scipy.signal, 请安装 scipy ({exc})",
        )

    # 减均值后做 FFT, 抑制 DC
    fft = fftshift(fft2(arr - arr.mean()))
    power = np.abs(fft) ** 2

    # 径向平均功率谱
    cy, cx = np.array(power.shape) // 2
    y, x = np.indices(power.shape)
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)
    r_flat = r.ravel()
    radial = (
        np.bincount(r_flat, power.ravel())
        / np.maximum(np.bincount(r_flat), 1)
    )
    radial[:1] = 0.0  # 去 DC

    if fft_threshold is None:
        thr = float(np.max(radial) * 0.1)
    else:
        thr = float(fft_threshold)

    peaks, _ = find_peaks(radial, height=thr, distance=3)

    # 像素频率 → d-spacing
    # 径向距离 r 对应频率 f = r / N (cycles/pixel), 周期 = 1/f = N/r px
    # d = 周期 * pixel_size
    N = float(max(arr.shape))
    d_entries: list[dict[str, Any]] = []
    for r_val in peaks:
        if r_val <= 0:
            continue
        d_nm = pixel_size * N / float(r_val)
        entry: dict[str, Any] = {
            "freq_px": int(r_val),
            "d_nm": float(d_nm),
        }
        # 在参考表里找最近的晶面
        best_plane = None
        best_dev = float("inf")
        for plane, d_ref in _D_SPACING_TABLE.items():
            if d_ref <= 0:
                continue
            dev = abs(d_nm - d_ref) / d_ref
            if dev < best_dev and dev < 0.10:  # 10% 容差
                best_dev = dev
                best_plane = plane
        entry["matched_plane"] = best_plane
        entry["match_deviation_pct"] = float(best_dev * 100) if best_plane else None
        d_entries.append(entry)

    d_entries.sort(key=lambda e: e["freq_px"])

    # 2D FFT 局部峰值 (找最强的几个峰, 用于判断晶格方向)
    peak_2d: list[dict[str, int]] = []
    try:
        local_max = power > thr
        # 简单取前若干个最亮的非中心点
        ys, xs = np.where(local_max)
        if len(ys) > 0:
            vals = power[ys, xs]
            order = np.argsort(vals)[::-1][:10]
            for i in order:
                peak_2d.append({"y": int(ys[i]), "x": int(xs[i]), "power": float(vals[i])})
    except Exception:
        pass

    if d_entries:
        summary = (
            f"TEM 晶格分析: 检测到 {len(peaks)} 个径向 FFT 峰, "
            f"最强 d={d_entries[0]['d_nm']:.3f} nm"
            + (f" → {d_entries[0]['matched_plane']}" if d_entries[0].get("matched_plane") else "")
        )
    else:
        summary = "TEM 晶格分析: 未检测到明显 FFT 峰, 检查 fft_threshold 或图片质量"

    data = {
        "summary": summary,
        "measurements": {
            "image_shape": [int(arr.shape[0]), int(arr.shape[1])],
            "pixel_size_nm": pixel_size,
            "fft_threshold": thr,
            "n_peaks": int(len(peaks)),
            "d_spacings": d_entries[:10],
            "fft_peak_2d": peak_2d,
        },
        "histogram": radial[1:201].astype(float).tolist(),
    }
    return ToolResult(data=data)
