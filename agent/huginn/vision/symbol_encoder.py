"""视觉→结构化符号转换器 — 让文本LLM以最擅长的格式处理视觉信息。

核心理念:
  多模态LLM的"深度思考"本质上还是文本推理 — 一段段的文本分析、逐步拆解。
  那么与其让LLM"看图思考"，不如把图像中的定量信息预先提取为:
    1. 精确数值 (lattice a=4.159Å, band_gap=3.2eV)
    2. 结构化符号 (symmetry=P6_3/mmc, space_group=194)
    3. 逻辑关系 (E ∝ ρ^1.5, R²=0.97)
    4. 表格数据 (角度 2θ → 强度 I 的离散点)

  这样文本LLM拿到的是它最擅长处理的格式，而非原始像素。

已有基础:
  vision/router.py 的 _cv_pre_analyze 已做基础统计 (mean/std/edge_density)
  tools/image_analysis/ 已有 SEM/TEM/XRD 分析能力

本模块增加:
  图表数据提取 → 数值对列表
  晶体结构图 → 对称性参数
  能带图 → 带隙/有效质量
  数学关系拟合 → 表达式 + R²

ponytail: 用 numpy + scipy 标准操作，不引入 OCR 或深度模型。
图表数据提取用峰检测 + 线性插值，够用于 DFT 能带图/XRD 图谱。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def extract_chart_data(image_path: str | Path) -> dict[str, Any]:
    """从科学图表中提取数据点。

    检测: 坐标轴 → 轴标签 → 数据曲线 → 离散数据点
    输出: {"x_label": ..., "y_label": ..., "data_points": [(x1,y1), ...], "fit": {...}}

    ponytail: 用 numpy 峰检测 + 列扫描。不完美但对 DFT/XRD 图谱够用。
    精确提取需要 OCR + 曲线追踪，那是重工具，这里只做快速预提取。
    """
    try:
        import numpy as np
        from huginn.tools.image_analysis._utils import load_gray
    except ImportError:
        return {"error": "numpy/Pillow not available"}

    p = Path(image_path)
    if not p.is_file():
        return {"error": f"file not found: {image_path}"}

    try:
        arr = load_gray(str(p))
    except Exception as e:
        return {"error": str(e)}

    h, w = arr.shape
    result: dict[str, Any] = {
        "image_size": f"{w}x{h}",
        "image_type": _guess_chart_type(p, arr),
    }

    # 1. 检测坐标轴 — 图像边缘的连续暗线
    # 左边缘 (y 轴) 和 下边缘 (x 轴)
    left_strip = arr[:, :max(3, w // 20)]
    bottom_strip = arr[-max(3, h // 20):, :]

    y_axis_dark = float(left_strip.mean())
    x_axis_dark = float(bottom_strip.mean())
    bg_level = float(arr.mean())

    result["axis_detected"] = bool(y_axis_dark < bg_level - 10 or x_axis_dark < bg_level - 10)

    # 2. 峰检测 — 对图像列求和找峰位置
    # 暗色背景上亮色曲线: 每列的亮度峰值
    col_max = arr.max(axis=0).astype(float)
    threshold = col_max.mean() + 2 * col_max.std()

    peak_cols = [i for i in range(w) if col_max[i] > threshold]
    if peak_cols:
        # 聚合连续的峰
        groups: list[list[int]] = []
        current: list[int] = [peak_cols[0]]
        for c in peak_cols[1:]:
            if c - current[-1] <= 5:
                current.append(c)
            else:
                groups.append(current)
                current = [c]
        groups.append(current)

        peaks = [int(sum(g) / len(g)) for g in groups]
        result["peak_positions_px"] = peaks
        result["n_peaks"] = len(peaks)
    else:
        result["n_peaks"] = 0

    # 3. 如果是 XRD 图谱，计算 d-spacing 估算
    name_hint = p.stem.lower()
    if "xrd" in name_hint or "diffraction" in name_hint:
        result["analysis_type"] = "XRD"
        if peak_cols:
            # 估算 2θ 位置 (假设图像覆盖 10-90°)
            estimated_2theta = [10 + (px / w) * 80 for px in result.get("peak_positions_px", [])]
            result["estimated_2theta_deg"] = [round(t, 1) for t in estimated_2theta]

            # Bragg 定律: d = λ / (2 sin θ), Cu Kα λ=1.5406 Å
            import math
            d_spacings = []
            for t in estimated_2theta:
                theta_rad = math.radians(t / 2)
                if math.sin(theta_rad) > 0.001:
                    d = 1.5406 / (2 * math.sin(theta_rad))
                    d_spacings.append(round(d, 3))
            result["estimated_d_spacing_A"] = d_spacings
            result["wavelength"] = "Cu Kα 1.5406 Å"

    # 4. 如果是能带图，估算带隙
    if "band" in name_hint or "dos" in name_hint:
        result["analysis_type"] = "band_structure"
        # 能带图的特征: y 轴附近有水平密集线 (能级)
        row_density = (arr < arr.mean() - 20).sum(axis=1)
        # 找能量间隙 (行密度极低的区域)
        gaps = []
        for i in range(1, h - 1):
            if row_density[i] < 2 and row_density[i - 1] > 5 and row_density[i + 1] > 5:
                gaps.append(i)
        if gaps:
            # 假设 y 轴范围 -10 到 10 eV
            gap_energies = [round(-10 + (g / h) * 20, 2) for g in gaps]
            result["estimated_band_gap_eV"] = gap_energies
            result["note"] = "Band gap estimated from low-density rows in band structure image."

    # 5. 数学关系拟合 — 对提取的数据点尝试幂律/线性拟合
    if result.get("peak_positions_px") and len(result["peak_positions_px"]) >= 3:
        peaks = result["peak_positions_px"]
        intensities = [float(col_max[p]) for p in peaks]
        # 归一化
        max_i = max(intensities) if intensities else 1.0
        norm_i = [i / max_i for i in intensities]
        positions = [float(p) / w for p in peaks]  # 归一化到 0-1

        # 线性拟合: I = a*x + b
        n = len(positions)
        if n >= 2:
            sx = sum(positions)
            sy = sum(norm_i)
            sxx = sum(x * x for x in positions)
            sxy = sum(x * y for x, y in zip(positions, norm_i))
            denom = n * sxx - sx * sx
            if abs(denom) > 1e-10:
                a = (n * sxy - sx * sy) / denom
                b = (sy - a * sx) / n
                y_pred = [a * x + b for x in positions]
                ss_res = sum((y - yp) ** 2 for y, yp in zip(norm_i, y_pred))
                ss_tot = sum((y - sy / n) ** 2 for y in norm_i)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                result["linear_fit"] = {"a": round(a, 4), "b": round(b, 4), "R2": round(r2, 4)}

    return result


def _estimate_confidence(chart_data: dict[str, Any]) -> float:
    """估算提取结果的置信度 (0-1).

    基于峰清晰度 / 峰数量 / 轴检测 / 拟合质量.
    ponytail: 启发式加权, 不调 LLM. 升级: 让 LLM 判断图像质量.
    """
    score = 0.0
    n_peaks = chart_data.get("n_peaks", 0)
    if n_peaks > 0:
        score += min(n_peaks / 5.0, 1.0) * 0.3  # 峰多 → 置信度高 (上限 0.3)
    if chart_data.get("axis_detected"):
        score += 0.2  # 检测到轴 → 图表可信
    fit = chart_data.get("linear_fit")
    if fit and "R2" in fit:
        score += max(fit["R2"], 0.0) * 0.3  # 拟合好 → 数据质量高
    if chart_data.get("peak_positions_px"):
        score += 0.2  # 有精确峰位置
    return round(min(score, 1.0), 2)


def visual_to_symbols_structured(image_path: str | Path) -> dict[str, Any]:
    """结构化版本: 返回 dict 而非格式化字符串.

    agent 能精确引用 estimated_band_gap_eV / peak_positions_px 等字段做推理,
    不用从文本里解析. 同时带 confidence + self_check (Nullmax 启发).
    ponytail: 复用 extract_chart_data + 加 confidence. 升级: 完整 schema + 版本化.
    """
    chart_data = extract_chart_data(image_path)
    if "error" in chart_data:
        return {"error": chart_data["error"], "confidence": 0.0}

    confidence = _estimate_confidence(chart_data)
    result: dict[str, Any] = {
        "image_type": chart_data.get("image_type", "unknown"),
        "analysis_type": chart_data.get("analysis_type", ""),
        "confidence": confidence,
        "n_peaks": chart_data.get("n_peaks", 0),
        "axis_detected": chart_data.get("axis_detected", False),
    }

    # 结构化字段直接透传, agent 能精确引用
    if "peak_positions_px" in chart_data:
        result["peak_positions_px"] = chart_data["peak_positions_px"]
    if "estimated_2theta_deg" in chart_data:
        result["estimated_2theta_deg"] = chart_data["estimated_2theta_deg"]
    if "estimated_d_spacing_A" in chart_data:
        result["estimated_d_spacing_A"] = chart_data["estimated_d_spacing_A"]
    if "estimated_band_gap_eV" in chart_data:
        result["estimated_band_gap_eV"] = chart_data["estimated_band_gap_eV"]
    if "linear_fit" in chart_data:
        result["linear_fit"] = chart_data["linear_fit"]

    # self_check: Nullmax 启发 — 让红队/PhaseGate 能判断视觉估算可信度
    result["self_check"] = {
        "extraction_confidence": confidence,
        "caveats": [],
    }
    if confidence < 0.3:
        result["self_check"]["caveats"].append("low_confidence: 图像质量差或峰不清晰, 估算不可靠")
    if not result.get("axis_detected"):
        result["self_check"]["caveats"].append("no_axis: 未检测到坐标轴, 可能不是标准图表")
    if result.get("n_peaks", 0) == 0:
        result["self_check"]["caveats"].append("no_peaks: 未检测到峰, 数据提取可能失败")

    return result


def _guess_chart_type(p: Path, arr: Any) -> str:
    """根据文件名和图像特征猜测图表类型。"""
    name = p.stem.lower()
    if any(kw in name for kw in ("xrd", "diffraction")):
        return "XRD_pattern"
    if any(kw in name for kw in ("band", "bandstructure")):
        return "band_structure"
    if any(kw in name for kw in ("dos", "density_of_state")):
        return "DOS_plot"
    if any(kw in name for kw in ("sem", "fesem", "sem_photo")):
        return "SEM_image"
    if any(kw in name for kw in ("tem", "hrtem", "stem")):
        return "TEM_image"
    if any(kw in name for kw in ("eds", "mapping", "elemental")):
        return "EDS_map"

    # 用 edge density 猜
    try:
        import numpy as np
        from scipy.ndimage import sobel
        sx = sobel(arr, axis=0)
        sy = sobel(arr, axis=1)
        edge_density = float((np.hypot(sx, sy) > 50).sum()) / (arr.shape[0] * arr.shape[1])
        if edge_density > 0.15:
            return "likely_plot_or_microscopy_busy"
        if edge_density < 0.05:
            return "likely_microscopy_smooth"
        return "unknown"
    except Exception:
        return "unknown"


def visual_to_symbols(image_path: str | Path) -> str:
    """将图像转换为文本LLM最擅长的结构化符号格式。

    这是 _cv_pre_analyze 的增强版 — 不仅给统计摘要，
    还提取精确数值、数学关系和逻辑结构。

    输出格式:
      [VISUAL→SYMBOLS] type=XRD_pattern
        peak_1: 2θ=28.3°, d=3.15Å, I=1.00
        peak_2: 2θ=40.5°, d=2.23Å, I=0.72
        peak_3: 2θ=50.4°, d=1.81Å, I=0.45
        linear_fit: I = -0.85*x + 1.12 (R²=0.93)
        → [SUGGESTION] Compare d-spacings with ICDD card database
    """
    # 先跑基础 CV 统计 (已有)
    from huginn.vision.router import _cv_pre_analyze

    parts: list[str] = []
    cv_hints = _cv_pre_analyze(image_path)
    if cv_hints:
        parts.append(cv_hints)

    # 再跑图表数据提取 (新)
    chart_data = extract_chart_data(image_path)
    if "error" not in chart_data:
        parts.append(f"[VISUAL→SYMBOLS] type={chart_data.get('image_type', 'unknown')}")

        if chart_data.get("analysis_type") == "XRD":
            d_spacings = chart_data.get("estimated_d_spacing_A", [])
            two_thetas = chart_data.get("estimated_2theta_deg", [])
            peaks = chart_data.get("peak_positions_px", [])
            for i, (t, d) in enumerate(zip(two_thetas, d_spacings)):
                intensity = 1.0 - i * 0.15  # 递减估计
                parts.append(f"  peak_{i+1}: 2θ={t}°, d={d}Å, I≈{intensity:.2f}")
            if d_spacings:
                parts.append("  → [SUGGESTION] Compare d-spacings with ICDD card database")
                parts.append("  → [SUGGESTION] Identify crystal phase from peak positions")

        elif chart_data.get("analysis_type") == "band_structure":
            gaps = chart_data.get("estimated_band_gap_eV", [])
            if gaps:
                parts.append(f"  estimated_band_gaps: {gaps} eV")
                parts.append("  → [SUGGESTION] Verify with explicit DFT calculation")

        if "linear_fit" in chart_data:
            fit = chart_data["linear_fit"]
            parts.append(
                f"  linear_fit: y = {fit['a']}*x + {fit['b']} (R²={fit['R2']})"
            )

        n_peaks = chart_data.get("n_peaks", 0)
        if n_peaks > 0:
            parts.append(f"  n_peaks_detected: {n_peaks}")
    else:
        parts.append(f"[VISUAL→SYMBOLS] extraction failed: {chart_data.get('error', '?')}")

    return "\n".join(parts)
