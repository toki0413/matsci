"""论文图表数据反提取 — 颜色匹配找曲线 + 像素→数据坐标反变换."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from huginn.types import ToolResult
from huginn.tools.image_analysis._utils import load_rgb, parse_color

if TYPE_CHECKING:
    from huginn.tools.image_analysis.tool import ImageAnalysisInput

logger = logging.getLogger(__name__)


def plot_extract(args: "ImageAnalysisInput") -> ToolResult:
    rgb = load_rgb(args.image_path)
    H, W, _ = rgb.shape

    required = ["x_min", "x_max", "y_min", "y_max"]
    for k in required:
        if k not in args.parameters:
            return ToolResult(
                data=None,
                success=False,
                error=f"plot_extract 需要 {required} 四个轴范围参数",
            )

    x_min = float(args.parameters["x_min"])
    x_max = float(args.parameters["x_max"])
    y_min = float(args.parameters["y_min"])
    y_max = float(args.parameters["y_max"])
    x_axis_type = args.parameters.get("x_axis_type", "linear")
    y_axis_type = args.parameters.get("y_axis_type", "linear")
    curve_color = args.parameters.get("curve_color", "blue")
    color_tol = float(args.parameters.get("color_tolerance", 30.0))

    # 坐标轴像素范围, 不给就默认留 10% 边距
    axis_box = args.parameters.get("axis_box", None)
    if axis_box and len(axis_box) == 4:
        x_left, y_top, x_right, y_bottom = [int(v) for v in axis_box]
    else:
        x_left = int(W * 0.10)
        x_right = int(W * 0.95)
        y_top = int(H * 0.10)
        y_bottom = int(H * 0.90)

    # 颜色匹配找曲线像素
    target = parse_color(curve_color)
    dist = np.sqrt(((rgb - target) ** 2).sum(axis=2))
    curve_mask = dist < color_tol

    # 限制在 axis box 内
    box_mask = np.zeros_like(curve_mask)
    box_mask[y_top : y_bottom + 1, x_left : x_right + 1] = True
    curve_mask = curve_mask & box_mask

    # 每个 x 列取 y 均值作为一个数据点
    points: list[list[float]] = []
    for px in range(x_left, x_right + 1):
        ys = np.where(curve_mask[:, px])[0]
        if len(ys) == 0:
            continue
        py = float(ys.mean())
        fx = (px - x_left) / max(x_right - x_left, 1)
        fy = (py - y_top) / max(y_bottom - y_top, 1)
        # 图像 y 向下, 所以 y_top 对应 y_max, y_bottom 对应 y_min
        if x_axis_type == "log":
            lx_min = np.log10(max(x_min, 1e-12))
            lx_max = np.log10(max(x_max, 1e-12))
            dx = 10.0 ** (lx_min + fx * (lx_max - lx_min))
        else:
            dx = x_min + fx * (x_max - x_min)
        if y_axis_type == "log":
            ly_min = np.log10(max(y_min, 1e-12))
            ly_max = np.log10(max(y_max, 1e-12))
            dy = 10.0 ** (ly_max - fy * (ly_max - ly_min))
        else:
            dy = y_max - fy * (y_max - y_min)
        points.append([float(dx), float(dy)])

    if not points:
        data = {
            "summary": "未提取到数据点, 检查 curve_color / axis_box / color_tolerance",
            "measurements": {
                "n_points": 0,
                "curve_color_rgb": [int(target[0]), int(target[1]), int(target[2])],
            },
        }
        return ToolResult(data=data)

    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])
    summary = (
        f"图表数据提取: 提取到 {len(points)} 个数据点, "
        f"x 范围 [{xs.min():.4g}, {xs.max():.4g}], "
        f"y 范围 [{ys.min():.4g}, {ys.max():.4g}]"
    )

    data = {
        "summary": summary,
        "measurements": {
            "image_shape": [int(H), int(W)],
            "n_points": int(len(points)),
            "x_range": [float(xs.min()), float(xs.max())],
            "y_range": [float(ys.min()), float(ys.max())],
            "x_axis_type": x_axis_type,
            "y_axis_type": y_axis_type,
            "axis_box_px": [x_left, y_top, x_right, y_bottom],
            "curve_color_rgb": [int(target[0]), int(target[1]), int(target[2])],
            "color_tolerance": color_tol,
        },
        "points": points,
    }
    return ToolResult(data=data)
