"""SEM 形貌分析 — 衬度统计 / 粗糙度 / 暗区分割 / 边缘密度."""
from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import numpy as np

from huginn.types import ToolResult
from huginn.tools.image_analysis._utils import load_gray

if TYPE_CHECKING:
    from huginn.tools.image_analysis.tool import ImageAnalysisInput

logger = logging.getLogger(__name__)


def sem_analysis(args: "ImageAnalysisInput") -> ToolResult:
    arr = load_gray(args.image_path)
    pixel_size = float(args.parameters.get("pixel_size_nm", 1.0))
    contrast_threshold = args.parameters.get("contrast_threshold", None)

    mean_i = float(arr.mean())
    std_i = float(arr.std())
    p5, p50, p95 = np.percentile(arr, [5, 50, 95])

    # 表面粗糙度: 用 uniform_filter 算局部均值, 残差的 RMS
    roughness_rms = std_i
    try:
        from scipy.ndimage import uniform_filter

        local_mean = uniform_filter(arr, size=15)
        residual = arr - local_mean
        roughness_rms = float(np.sqrt((residual ** 2).mean()))
    except ImportError:
        logger.debug("scipy 不可用, 粗糙度退化为全局 std")

    # 阈值分割找暗区 (颗粒/孔洞)
    if contrast_threshold is None:
        thr = mean_i - 0.5 * std_i
    else:
        thr = float(contrast_threshold)
    binary = arr < thr

    # 连通域统计
    areas: list[float] = []
    n_regions = 0
    try:
        from scipy.ndimage import label as nd_label

        labeled, n_regions = nd_label(binary)
        counts = np.bincount(labeled.ravel())
        counts[0] = 0
        areas = counts[counts > 0].astype(float).tolist()
    except ImportError:
        n_regions = int(binary.sum())
        logger.debug("scipy 不可用, 仅返回暗像素总数")

    # 用 cv2 跑一遍边缘统计作为可选增强
    edge_density = 0.0
    try:
        import cv2

        edges = cv2.Canny(arr.astype(np.uint8), 50, 150)
        edge_density = float(edges.sum() / 255 / edges.size)
    except ImportError:
        # numpy Sobel 兜底
        try:
            from scipy.ndimage import sobel

            gx = sobel(arr, axis=1)
            gy = sobel(arr, axis=0)
            grad = np.sqrt(gx ** 2 + gy ** 2)
            edge_density = float((grad > grad.mean() + 2 * grad.std()).mean())
        except ImportError:
            pass

    # 直方图 (32 bins)
    hist, _ = np.histogram(arr, bins=32, range=(0, 255))

    mean_area = float(np.mean(areas)) if areas else 0.0
    summary = (
        f"SEM 图像 {arr.shape[0]}x{arr.shape[1]} px, 平均衬度 {mean_i:.1f}, "
        f"RMS 粗糙度 {roughness_rms:.2f}, 检测到 {n_regions} 个暗区, "
        f"平均面积 {mean_area:.1f} px²"
    )

    data = {
        "summary": summary,
        "measurements": {
            "image_shape": [int(arr.shape[0]), int(arr.shape[1])],
            "pixel_size_nm": pixel_size,
            "contrast_mean": mean_i,
            "contrast_std": std_i,
            "contrast_p5": float(p5),
            "contrast_p50": float(p50),
            "contrast_p95": float(p95),
            "surface_roughness_rms": roughness_rms,
            "edge_density": edge_density,
            "threshold_used": float(thr),
            "n_dark_regions": int(n_regions),
            "dark_region_areas_px2": [float(a) for a in areas[:200]],
            "mean_area_px2": mean_area,
            "mean_area_nm2": mean_area * pixel_size ** 2,
        },
        "histogram": hist.astype(float).tolist(),
    }
    return ToolResult(data=data)


def generate_verification_code(image_path: str, analysis_result: dict) -> str:
    """根据 SEM 分析结果声称的测量值, 生成独立验证代码.

    SWE-Vision 范式: 不信 VLM 的 "看图说话", 而是生成 Python 代码做
    结构化测量来交叉验证. 代码用 PIL 重新加载图像, 用 numpy/scipy
    独立测量衬度和暗区, 然后断言 measured vs claimed 在容差内.

    生成的代码在沙箱里跑, 失败返回错误不阻塞主流程.
    """
    m = analysis_result.get("measurements", {})
    claimed = {
        "image_shape": m.get("image_shape"),
        "contrast_mean": m.get("contrast_mean"),
        "contrast_std": m.get("contrast_std"),
        "n_dark_regions": m.get("n_dark_regions"),
        "mean_area_px2": m.get("mean_area_px2"),
        "threshold_used": m.get("threshold_used"),
    }
    claimed_json = json.dumps(claimed, default=str)
    safe_path = image_path.replace("\\", "/")

    lines = [
        "import json",
        "from PIL import Image",
        "import numpy as np",
        "",
        f"IMAGE_PATH = {safe_path!r}",
        f"CLAIMED = {claimed_json}",
        "",
        "img = Image.open(IMAGE_PATH).convert('L')",
        "arr = np.asarray(img, dtype=float)",
        "",
        "measured = {}",
        "measured['image_shape'] = [int(arr.shape[0]), int(arr.shape[1])]",
        "measured['contrast_mean'] = float(arr.mean())",
        "measured['contrast_std'] = float(arr.std())",
        "",
        "# 用同样的阈值重新分割暗区, skimage 优先, scipy 兜底",
        "thr = CLAIMED.get('threshold_used')",
        "if thr is None:",
        "    thr = arr.mean() - 0.5 * arr.std()",
        "binary = arr < float(thr)",
        "",
        "try:",
        "    from skimage.measure import label as sk_label",
        "    labeled, n = sk_label(binary)",
        "    counts = np.bincount(labeled.ravel())",
        "    counts[0] = 0",
        "    areas = counts[counts > 0].astype(float)",
        "except ImportError:",
        "    from scipy.ndimage import label as nd_label",
        "    labeled, n = nd_label(binary)",
        "    counts = np.bincount(labeled.ravel())",
        "    counts[0] = 0",
        "    areas = counts[counts > 0].astype(float)",
        "",
        "measured['n_dark_regions'] = int(n)",
        "measured['mean_area_px2'] = float(areas.mean()) if len(areas) > 0 else 0.0",
        "",
        "# 逐项对比 measured vs claimed, 15% 容差",
        "checks = []",
        "for key in ['contrast_mean', 'contrast_std', 'n_dark_regions', 'mean_area_px2']:",
        "    mv = measured.get(key)",
        "    cv = CLAIMED.get(key)",
        "    if mv is None or cv is None:",
        "        continue",
        "    rel = abs(mv - cv) / (abs(cv) + 1e-9)",
        "    ok = rel < 0.15",
        "    checks.append({'key': key, 'measured': mv, 'claimed': cv,",
        "                   'rel_diff': rel, 'match': ok})",
        "",
        "# 图像尺寸必须完全一致",
        "if CLAIMED.get('image_shape') is not None:",
        "    shape_ok = measured['image_shape'] == CLAIMED['image_shape']",
        "    checks.append({'key': 'image_shape', 'measured': measured['image_shape'],",
        "                   'claimed': CLAIMED['image_shape'], 'match': shape_ok})",
        "",
        "all_match = all(c.get('match', False) for c in checks)",
        "",
        "# 断言: 核心测量值必须在容差内, 不满足就 AssertionError",
        "assert all_match, 'Verification failed: ' + json.dumps(checks, default=str)",
        "",
        "result = {'all_match': all_match, 'checks': checks,",
        "          'measured': measured, 'claimed': CLAIMED}",
        "print('__VERIFY_RESULT__:' + json.dumps(result, default=str))",
    ]
    return "\n".join(lines)
