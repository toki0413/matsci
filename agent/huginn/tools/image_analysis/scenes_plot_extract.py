"""论文图表数据反提取 — 颜色匹配找曲线 + 像素→数据坐标反变换.

支持:
  - 单曲线 (指定 curve_color) / 多曲线 (curve_color="auto" 自动 k-means 取主色)
  - 轴范围自动 OCR 检测 (pytesseract), 找不到仍可手动传
  - 可选 plotdigitizer 集成, 装了就优先用它做更精确的数字化
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from huginn.core_types import ToolResult
from huginn.tools.image_analysis._utils import auto_detect_colors, load_rgb, parse_color

if TYPE_CHECKING:
    from huginn.tools.image_analysis.tool import ImageAnalysisInput

logger = logging.getLogger(__name__)

# plotdigitizer 是可选依赖, 装了就走它那条更精确的路径
try:
    import plotdigitizer as _pd  # noqa: F401

    _PLOTDIGITIZER_AVAILABLE = True
except ImportError:
    logger.debug("best-effort op failed", exc_info=True)
    _PLOTDIGITIZER_AVAILABLE = False

# 匹配刻度数字: 整数 / 小数 / 科学计数法 (1e-5, -2.3E+4)
_TICK_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")


@dataclass
class _PlotOptions:
    """一次 plot_extract 调用的曲线 / 坐标轴参数."""

    curve_color: Any
    color_tol: float
    x_axis_type: str
    y_axis_type: str


def _plot_options(args: ImageAnalysisInput) -> _PlotOptions:
    """从 parameters 读取曲线色 / 容差 / 坐标轴类型, 缺省用原默认值."""
    return _PlotOptions(
        curve_color=args.parameters.get("curve_color", "blue"),
        color_tol=float(args.parameters.get("color_tolerance", 30.0)),
        x_axis_type=args.parameters.get("x_axis_type", "linear"),
        y_axis_type=args.parameters.get("y_axis_type", "linear"),
    )


def _is_finite_number(s: str) -> bool:
    """正则可能抓到空串或非数, 这里兜一层."""
    try:
        v = float(s)
    except (ValueError, TypeError):
        logger.debug("best-effort op failed", exc_info=True)
        return False
    return bool(np.isfinite(v))


def _resolve_axis_box(
    axis_box: list | tuple | None, H: int, W: int
) -> tuple[int, int, int, int]:
    """axis_box 给了就取整, 不给就按 10% 边距默认.

    返回 (x_left, y_top, x_right, y_bottom).
    """
    if axis_box and len(axis_box) == 4:
        x_left, y_top, x_right, y_bottom = (int(v) for v in axis_box)
        return x_left, y_top, x_right, y_bottom

    x_left = int(W * 0.10)
    x_right = int(W * 0.95)
    y_top = int(H * 0.10)
    y_bottom = int(H * 0.90)
    return x_left, y_top, x_right, y_bottom


def _tick_strips(
    rgb: np.ndarray, x_left: int, y_top: int, x_right: int, y_bottom: int
) -> tuple[np.ndarray, np.ndarray]:
    """切出 y 轴 (轴线左侧) 和 x 轴 (轴线下方) 的刻度文字条带."""
    H, W, _ = rgb.shape
    box_w = max(x_right - x_left, 1)
    box_h = max(y_bottom - y_top, 1)

    # y 轴刻度标签一般在轴线左侧的一小条里
    strip_w = max(int(box_w * 0.10), 5)
    y_x0 = max(0, x_left - strip_w)
    y_x1 = min(W, x_left + 2)
    y_strip = rgb[y_top : y_bottom + 1, y_x0:y_x1]

    # x 轴刻度标签一般在轴线下方的一小条里
    strip_h = max(int(box_h * 0.10), 5)
    x_y0 = max(0, y_bottom - 2)
    x_y1 = min(H, y_bottom + strip_h)
    x_strip = rgb[x_y0:x_y1, x_left : x_right + 1]
    return y_strip, x_strip


def _ocr_tick_numbers(
    strip: np.ndarray, pytesseract: Any, image_cls: Any, axis: str
) -> list[float]:
    """对一条刻度条带做 OCR 并抽数字; OCR 可执行文件缺失时返回空列表."""
    try:
        text = pytesseract.image_to_string(image_cls.fromarray(strip.astype(np.uint8)))
        return [float(m) for m in _TICK_NUM_RE.findall(text) if _is_finite_number(m)]
    except Exception as exc:  # OCR 经常因为 tesseract 没装可执行文件而炸
        logger.debug("%s 轴 OCR 失败: %s", axis, exc)
        return []


def _auto_detect_axes(
    rgb: np.ndarray, axis_box: list | tuple | None
) -> tuple[float, float, float, float] | None:
    """OCR 坐标轴边缘的刻度文字, 反推轴范围.

    在 axis_box 的左边缘 (y 轴标签) 和下边缘 (x 轴标签) 各取约 10% 宽度的条带
    做 OCR, 用正则抽数字 (含科学计数法). 两个轴各自至少凑够 3 个数字才用
    min/max 拟合范围, 否则返回 None, 让调用方继续走手动传参的老路子.
    """
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        logger.debug("pytesseract 或 Pillow 没装, 跳过 OCR 轴检测")
        return None

    H, W, _ = rgb.shape
    x_left, y_top, x_right, y_bottom = _resolve_axis_box(axis_box, H, W)
    y_strip, x_strip = _tick_strips(rgb, x_left, y_top, x_right, y_bottom)

    y_nums = _ocr_tick_numbers(y_strip, pytesseract, Image, "y")
    x_nums = _ocr_tick_numbers(x_strip, pytesseract, Image, "x")

    x_range = (min(x_nums), max(x_nums)) if len(x_nums) >= 3 else None
    y_range = (min(y_nums), max(y_nums)) if len(y_nums) >= 3 else None
    if x_range is None or y_range is None:
        return None
    return (x_range[0], x_range[1], y_range[0], y_range[1])


def _pixel_to_data(
    px: int,
    py: float,
    x_left: int,
    y_top: int,
    x_right: int,
    y_bottom: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    x_axis_type: str,
    y_axis_type: str,
) -> tuple[float, float]:
    """单个像素坐标 → 数据坐标. 图像 y 向下, y_top 对应 y_max."""
    fx = (px - x_left) / max(x_right - x_left, 1)
    fy = (py - y_top) / max(y_bottom - y_top, 1)
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
    return float(dx), float(dy)


def _extract_single_curve(
    rgb: np.ndarray,
    target: np.ndarray,
    color_tol: float,
    x_left: int,
    y_top: int,
    x_right: int,
    y_bottom: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    x_axis_type: str,
    y_axis_type: str,
) -> list[list[float]]:
    """对一个目标颜色做颜色匹配, 再按 x 列取 y 均值还原成数据点."""
    dist = np.sqrt(((rgb - target) ** 2).sum(axis=2))
    curve_mask = dist < color_tol

    # 限制在 axis box 内, 边框外的不算
    box_mask = np.zeros_like(curve_mask)
    box_mask[y_top : y_bottom + 1, x_left : x_right + 1] = True
    curve_mask = curve_mask & box_mask

    points: list[list[float]] = []
    for px in range(x_left, x_right + 1):
        ys = np.where(curve_mask[:, px])[0]
        if len(ys) == 0:
            continue
        py = float(ys.mean())
        dx, dy = _pixel_to_data(
            px,
            py,
            x_left,
            y_top,
            x_right,
            y_bottom,
            x_min,
            x_max,
            y_min,
            y_max,
            x_axis_type,
            y_axis_type,
        )
        points.append([dx, dy])
    return points


def _plotdigitizer_points(
    image_path: str,
    ranges: tuple[float, float, float, float],
    box: tuple[int, int, int, int],
) -> list[list[float]]:
    """按 plotdigitizer 的新/老 API 分别取点, 新 API 不可用就退回老 API."""
    import plotdigitizer as pd  # 延迟 import, 避免模块加载时就强依赖

    x_min, x_max, y_min, y_max = ranges
    x_left, y_top, x_right, y_bottom = box

    # plotdigitizer 一般需要图像里 2~3 个标定点来标定坐标系.
    # 这里用 axis_box 的两个角点 + 对应数据值做近似标定,
    # 不同版本 API 略有差异, 包一层 try 兜底.
    points: list[list[float]] = []
    try:
        # 较新的 API: 传 image + 标定点列表
        fig = pd.Figure(image_path)
        # 左下角 (x_min, y_min), 右上角 (x_max, y_max) 标定
        fig.set_scale(
            (x_left, y_bottom, x_min, y_min),
            (x_right, y_bottom, x_max, y_min),
            (x_left, y_top, x_min, y_max),
        )
        pts = fig.digitize()
        for p in pts:
            points.append([float(p[0]), float(p[1])])
    except (AttributeError, TypeError):
        # 老 API: 函数式调用
        pts = pd.digitize(
            image_path,
            [(x_left, y_bottom, x_min, y_min), (x_right, y_bottom, x_max, y_min)],
        )
        for p in pts:
            points.append([float(p[0]), float(p[1])])
    return points


def _plotdigitizer_result(points: list[list[float]], curve_color: Any) -> ToolResult:
    """把 plotdigitizer 提取到的点打包成统一 ToolResult."""
    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])
    summary = (
        f"图表数据提取 (plotdigitizer): 提取到 {len(points)} 个数据点, "
        f"x 范围 [{xs.min():.4g}, {xs.max():.4g}], "
        f"y 范围 [{ys.min():.4g}, {ys.max():.4g}]"
    )
    target = parse_color(curve_color) if curve_color != "auto" else None
    data: dict[str, Any] = {
        "summary": summary,
        "measurements": {
            "n_points": int(len(points)),
            "x_range": [float(xs.min()), float(xs.max())],
            "y_range": [float(ys.min()), float(ys.max())],
            "method": "plotdigitizer",
            "curve_color_rgb": (
                [int(target[0]), int(target[1]), int(target[2])]
                if target is not None
                else None
            ),
        },
        "points": points,
        "curves": [
            {
                "color": (
                    [int(target[0]), int(target[1]), int(target[2])]
                    if target is not None
                    else None
                ),
                "n_points": len(points),
                "points": points,
            }
        ],
    }
    return ToolResult(data=data)


def _try_plotdigitizer(
    image_path: str,
    ranges: tuple[float, float, float, float],
    box: tuple[int, int, int, int],
    options: _PlotOptions,
) -> ToolResult | None:
    """用 plotdigitizer 做精确数字化. 失败返回 None, 调用方走兜底逻辑."""
    if not _PLOTDIGITIZER_AVAILABLE:
        return None
    try:
        points = _plotdigitizer_points(image_path, ranges, box)
        if not points:
            return None
        return _plotdigitizer_result(points, options.curve_color)
    except Exception as exc:
        logger.debug("plotdigitizer 提取失败, 回退到内置方法: %s", exc)
        return None


def _colorful_pixels(flat: np.ndarray) -> np.ndarray:
    """剔掉近白 / 近黑的背景像素, 只留彩色像素."""
    # 背景判定: 三通道都 >240 视为近白, 都 <15 视为近黑
    is_white = (flat[:, 0] > 240) & (flat[:, 1] > 240) & (flat[:, 2] > 240)
    is_black = (flat[:, 0] < 15) & (flat[:, 1] < 15) & (flat[:, 2] < 15)
    return flat[~(is_white | is_black)]


def _sample_pixels(colorful: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """最多抽 5000 个像素做聚类, 样本不足就全用."""
    n_samples = min(5000, len(colorful))
    if n_samples < len(colorful):
        idx = rng.choice(len(colorful), n_samples, replace=False)
        return colorful[idx]
    return colorful


def _distinct_colors(sample: np.ndarray) -> list[np.ndarray]:
    """样本比 k 还少时, 直接取所有互不相同的颜色."""
    uniq: list[np.ndarray] = []
    for c in sample:
        if not any(np.linalg.norm(c - u) < 5 for u in uniq):
            uniq.append(c)
    return uniq


def _kmeans_centers(sample: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """简版 k-means: 随机初始化 k 个中心, 迭代 20 轮或收敛即停."""
    centers = sample[rng.choice(len(sample), k, replace=False)].astype(float)
    for _ in range(20):
        dists = ((sample[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = dists.argmin(axis=1)
        new_centers = np.array(
            [
                sample[labels == i].mean(axis=0) if (labels == i).any() else centers[i]
                for i in range(k)
            ]
        )
        if np.allclose(centers, new_centers, atol=1.0):
            break
        centers = new_centers
    return centers


def _filter_centers(centers: np.ndarray) -> list[np.ndarray]:
    """滤掉残存的近白/近黑中心和因聚类塌缩产生的近似重复色."""
    candidates: list[np.ndarray] = []
    for c in centers:
        if float(c.sum()) > 720 or float(c.sum()) < 45:
            continue
        if any(np.linalg.norm(c - s) < 5 for s in candidates):
            continue
        candidates.append(c)
    return candidates


def _detect_curve_colors(rgb: np.ndarray, k: int = 5) -> list[np.ndarray]:
    """curve_color="auto" 时: 简版 k-means 找曲线主色.

    先把近白 / 近黑的背景像素从聚类样本里剔掉 (plot 图背景通常占绝大比例,
    不剔的话 k 个中心会被白色 shades 占满, 真正的曲线色反而被并掉).
    剔完再跑和 scenes_eds / auto_detect_colors 一样的 k-means 迭代.
    """
    flat = rgb.reshape(-1, 3).astype(float)
    colorful = _colorful_pixels(flat)

    # 彩色像素太少 (基本是纯背景图) 就回退到全图聚类
    if len(colorful) < k * 10:
        colorful = flat

    rng = np.random.default_rng(42)
    sample = _sample_pixels(colorful, rng)

    # 样本比 k 还少就直接退化为取所有不同色
    if len(sample) < k:
        return (
            _distinct_colors(sample) or list(auto_detect_colors(rgb, k=k).values())[:1]
        )

    centers = _kmeans_centers(sample, k, rng)
    candidates = _filter_centers(centers)

    # 全被滤掉的极端情况, 至少返回一个中心, 别让上层空转
    if not candidates:
        candidates = list(centers) or list(auto_detect_colors(rgb, k=k).values())[:1]
    return candidates


def _resolve_axis_ranges(
    args: ImageAnalysisInput, rgb: np.ndarray, axis_box: list | tuple | None
) -> tuple[float, float, float, float] | ToolResult:
    """轴范围: 优先用户给的, 没给就 OCR 自动检测; 都失败返回错误 ToolResult."""
    required = ["x_min", "x_max", "y_min", "y_max"]
    has_all_axis = all(k in args.parameters for k in required)

    if has_all_axis:
        return (
            float(args.parameters["x_min"]),
            float(args.parameters["x_max"]),
            float(args.parameters["y_min"]),
            float(args.parameters["y_max"]),
        )

    detected = _auto_detect_axes(rgb, axis_box)
    if detected is None:
        return ToolResult(
            data=None,
            success=False,
            error=(
                "plot_extract 缺少轴范围参数 (x_min/x_max/y_min/y_max), "
                "且自动 OCR 检测失败. 请手动提供轴范围, 或安装 "
                "pytesseract + tesseract-ocr 以启用自动检测."
            ),
        )
    x_min, x_max, y_min, y_max = detected
    logger.info(
        "OCR 自动检测轴范围: x[%.4g, %.4g] y[%.4g, %.4g]", x_min, x_max, y_min, y_max
    )
    return detected


def _single_curve_empty_result(target: np.ndarray) -> ToolResult:
    """没提取到任何数据点时的空结果."""
    data = {
        "summary": "未提取到数据点, 检查 curve_color / axis_box / color_tolerance",
        "measurements": {
            "n_points": 0,
            "curve_color_rgb": [int(target[0]), int(target[1]), int(target[2])],
        },
        "curves": [],
    }
    return ToolResult(data=data)


def _single_curve_result(
    points: list[list[float]],
    target: np.ndarray,
    image_hw: tuple[int, int],
    axis_box_px: tuple[int, int, int, int],
    options: _PlotOptions,
) -> ToolResult:
    """单曲线提取结果的打包."""
    if not points:
        return _single_curve_empty_result(target)

    H, W = image_hw
    x_axis_type, y_axis_type = options.x_axis_type, options.y_axis_type
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
            "axis_box_px": list(axis_box_px),
            "curve_color_rgb": [int(target[0]), int(target[1]), int(target[2])],
            "color_tolerance": options.color_tol,
        },
        # 兼容老调用方: 单曲线时 points 仍是平铺的 [[x,y], ...]
        "points": points,
        # 同时给一份 curves 结构, 跟多曲线输出对齐
        "curves": [
            {
                "color": [int(target[0]), int(target[1]), int(target[2])],
                "n_points": len(points),
                "points": points,
            }
        ],
    }
    return ToolResult(data=data)


def _extract_single_result(
    rgb: np.ndarray,
    box: tuple[int, int, int, int],
    ranges: tuple[float, float, float, float],
    options: _PlotOptions,
    image_hw: tuple[int, int],
) -> ToolResult:
    """单曲线: 颜色匹配提点后再打包结果."""
    x_left, y_top, x_right, y_bottom = box
    x_min, x_max, y_min, y_max = ranges
    target = parse_color(options.curve_color)
    points = _extract_single_curve(
        rgb,
        target,
        options.color_tol,
        x_left,
        y_top,
        x_right,
        y_bottom,
        x_min,
        x_max,
        y_min,
        y_max,
        options.x_axis_type,
        options.y_axis_type,
    )
    return _single_curve_result(points, target, image_hw, box, options)


def plot_extract(args: ImageAnalysisInput) -> ToolResult:
    rgb = load_rgb(args.image_path)
    H, W, _ = rgb.shape

    # 坐标轴像素范围, 不给就默认留 10% 边距
    axis_box = args.parameters.get("axis_box", None)
    box = _resolve_axis_box(axis_box, H, W)

    ranges = _resolve_axis_ranges(args, rgb, axis_box)
    if isinstance(ranges, ToolResult):
        return ranges

    options = _plot_options(args)

    # 装了 plotdigitizer 且轴范围已知, 优先走它 (失败自动回退)
    if _PLOTDIGITIZER_AVAILABLE and options.curve_color != "auto":
        pd_result = _try_plotdigitizer(args.image_path, ranges, box, options)
        if pd_result is not None:
            return pd_result

    # 多曲线: curve_color="auto" 时 k-means 找主色, 每个颜色提一条
    if isinstance(options.curve_color, str) and options.curve_color.lower() == "auto":
        return _extract_multi_curve(rgb, box, ranges, options, (H, W))

    # 单曲线: 原有逻辑
    return _extract_single_result(rgb, box, ranges, options, (H, W))


def _extract_curves_for_colors(
    rgb: np.ndarray,
    colors: list[np.ndarray],
    box: tuple[int, int, int, int],
    ranges: tuple[float, float, float, float],
    options: _PlotOptions,
) -> list[dict[str, Any]]:
    """每个主色各提一条曲线; 提不到点的颜色直接跳过."""
    x_left, y_top, x_right, y_bottom = box
    x_min, x_max, y_min, y_max = ranges

    curves: list[dict[str, Any]] = []
    for idx, color in enumerate(colors):
        pts = _extract_single_curve(
            rgb,
            color,
            options.color_tol,
            x_left,
            y_top,
            x_right,
            y_bottom,
            x_min,
            x_max,
            y_min,
            y_max,
            options.x_axis_type,
            options.y_axis_type,
        )
        if not pts:
            continue
        curves.append(
            {
                "color": [int(color[0]), int(color[1]), int(color[2])],
                "color_index": idx,
                "n_points": len(pts),
                "points": pts,
            }
        )
    return curves


def _multi_curve_empty_result(
    image_hw: tuple[int, int], n_detected_colors: int
) -> ToolResult:
    """一条曲线都没提出来时的兜底结果."""
    H, W = image_hw
    data = {
        "summary": "多曲线自动检测: 未提取到任何曲线, 检查 axis_box / color_tolerance",
        "measurements": {
            "image_shape": [int(H), int(W)],
            "n_curves": 0,
            "n_detected_colors": n_detected_colors,
        },
        "curves": [],
    }
    return ToolResult(data=data)


def _multi_curve_result(
    curves: list[dict[str, Any]],
    colors: list[np.ndarray],
    box: tuple[int, int, int, int],
    ranges: tuple[float, float, float, float],
    options: _PlotOptions,
    image_hw: tuple[int, int],
) -> ToolResult:
    """多曲线提取结果的打包."""
    H, W = image_hw
    x_min, x_max, y_min, y_max = ranges
    x_axis_type, y_axis_type = options.x_axis_type, options.y_axis_type
    x_left, y_top, x_right, y_bottom = box

    total_pts = sum(c["n_points"] for c in curves)
    # 取点最多的那条作为"主曲线", 方便只读 points 的老逻辑
    primary = max(curves, key=lambda c: c["n_points"])
    summary = (
        f"多曲线提取: 检测到 {len(colors)} 个主色, 成功提取 {len(curves)} 条曲线, "
        f"共 {total_pts} 个数据点"
    )
    data = {
        "summary": summary,
        "measurements": {
            "image_shape": [int(H), int(W)],
            "n_curves": len(curves),
            "n_detected_colors": len(colors),
            "n_points_total": total_pts,
            "x_range": [x_min, x_max],
            "y_range": [y_min, y_max],
            "x_axis_type": x_axis_type,
            "y_axis_type": y_axis_type,
            "axis_box_px": [x_left, y_top, x_right, y_bottom],
            "color_tolerance": options.color_tol,
        },
        # 平铺主曲线点, 给只认 points 的调用方留个口子
        "points": primary["points"],
        "curves": curves,
    }
    return ToolResult(data=data)


def _extract_multi_curve(
    rgb: np.ndarray,
    box: tuple[int, int, int, int],
    ranges: tuple[float, float, float, float],
    options: _PlotOptions,
    image_hw: tuple[int, int],
) -> ToolResult:
    """curve_color="auto" 的分支: k-means 主色 → 每个颜色提一条曲线."""
    colors = _detect_curve_colors(rgb, k=5)
    curves = _extract_curves_for_colors(rgb, colors, box, ranges, options)
    if not curves:
        return _multi_curve_empty_result(image_hw, len(colors))
    return _multi_curve_result(curves, colors, box, ranges, options, image_hw)
