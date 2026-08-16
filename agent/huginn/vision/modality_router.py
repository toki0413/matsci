"""自研多模态模态路由模块 —— 动态分辨率 + 模态识别 + 工具建议.

本模块为 huginn 自研实现, 不属于阿里 QwenLM 的 Qwen-MM-Plugins 官方插件,
也未包含其 Skill / MCP Server 架构。仅借鉴其 "动态分辨率" 这一通用概念:

  1. 动态分辨率预处理 (dynamic resolution)
     大图等比缩放到目标 patch 尺寸, 小图保留原分辨率 (细文字/缩略图鲁棒性).
     思路与 Qwen-VL 的 patch-based 视觉编码类似: 超出最大 patch 数时降采样,
     否则保留细节.

  2. 模态识别与路由 (modality routing)
     判断图像属于 显微图(SEM/TEM/EDS)/图表(plot)/文档/照片 哪种模态,
     把路由决策 + 推荐 image_analysis_tool action 一起注入文本通道,
     让 agent 知道该调哪个分析工具.

  3. 多工具集成建议
     每个模态映射到 image_analysis_tool 的 action (sem_analysis/tem_lattice/
     eds_mapping/particle_stats/defect_detect/plot_extract/deplot_chart 等),
     agent 可直接按建议调用.

best-effort 设计: 所有函数在依赖缺失/读取失败时安全降级, 不阻塞主链路.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── 动态分辨率常量 ─────────────────────────────────────────────────
# 以 patch 数为视觉 token 上限的通用约定: 默认 cap 在 ~1MP 附近.
# 超出 max_pixel_cap 的图等比降采样到 cap, 少于此的图保留原尺寸 (保细节).
DEFAULT_MAX_PIXEL_CAP = 1_200_000  # ≈ 1.2MP
DEFAULT_MAX_SIDE = 2048           # 最长边硬上限, 防止极端宽高比
DEFAULT_MIN_SIDE = 32             # 最短边下限, 防止缩到不可读

# image_analysis_tool 支持的 action (见 tools/image_analysis/tool.py)
_KNOWN_ACTIONS = {
    "sem_analysis", "tem_lattice", "eds_mapping", "particle_stats",
    "defect_detect", "phase_field", "plot_extract", "deplot_chart",
}

# 文件名强信号 → 模态 (覆盖 _cv_pre_analyze 的 image_type_guess)
_FILENAME_SIGNALS: list[tuple[str, str, str]] = [
    (r"\b(sem|fesem|sem_photo)\b", "SEM", "sem_analysis"),
    (r"\b(tem|hrtem|stem)\b", "TEM", "tem_lattice"),
    (r"\b(eds|mapping)\b", "EDS", "eds_mapping"),
    (r"\b(particle|psd|pore)\b", "PARTICLE", "particle_stats"),
    (r"\b(defect|crack|void)\b", "DEFECT", "defect_detect"),
    (r"\b(phase.?field|grain)\b", "PHASE_FIELD", "phase_field"),
    (r"\b(xrd|diffraction|plot|curve|chart|figure)\b", "PLOT", "plot_extract"),
]


# ── 1. 动态分辨率预处理 ────────────────────────────────────────────


def recommend_resolution(
    width: int,
    height: int,
    max_pixel_cap: int = DEFAULT_MAX_PIXEL_CAP,
    max_side: int = DEFAULT_MAX_SIDE,
    min_side: int = DEFAULT_MIN_SIDE,
) -> tuple[int, int]:
    """给定原始宽高, 返回推荐的目标宽高 (等比缩放).

    规则:
      - 总像素超出 max_pixel_cap → 等比缩到 cap 内
      - 任一边超过 max_side → 等比缩到 max_side 内
      - 过小 (< min_side) 不放大 (保留细节优先)
    完全不需要缩放时原样返回 (w, h).

    纯函数, 便于单测. 输入非法 (<=0) 时返回 (0, 0) 表示无法处理.
    """
    if width <= 0 or height <= 0:
        return 0, 0
    w, h = int(width), int(height)
    scale = 1.0
    # 像素 cap
    if w * h > max_pixel_cap:
        scale = min(scale, (max_pixel_cap / (w * h)) ** 0.5)
    # 最长边 cap
    if max(w, h) > max_side:
        scale = min(scale, max_side / max(w, h))
    if scale >= 1.0:
        return w, h
    nw = max(int(w * scale), min_side)
    nh = max(int(h * scale), min_side)
    return nw, nh


def dynamic_resolution_hint(
    image_path: str | Path | bytes,
    max_pixel_cap: int = DEFAULT_MAX_PIXEL_CAP,
) -> str | None:
    """对图片算动态分辨率, 返回一行机读提示 (是否需要降采样).

    best-effort: 非文件/读取失败返回 None, 不阻塞.
    """
    if isinstance(image_path, (bytes, bytearray)):
        return None
    p = Path(image_path)
    if not p.is_file():
        return None
    try:
        from PIL import Image

        with Image.open(p) as im:
            w, h = im.size
        nw, nh = recommend_resolution(w, h, max_pixel_cap=max_pixel_cap)
        if (nw, nh) == (w, h):
            return f"[MM-Router] resolution {w}x{h}=ok (no resize)"
        ratio = (nw * nh) / (w * h)
        return (
            f"[MM-Router] resolution {w}x{h}->{nw}x{nh} "
            f"(resize to {ratio:.0%}, keep fine text under {max_pixel_cap}px)"
        )
    except Exception:
        logger.debug("dynamic_resolution_hint failed, non-fatal", exc_info=True)
    return None


# ── 2. 模态识别与路由 ──────────────────────────────────────────────


def detect_modality(image_path: str | Path | bytes) -> dict[str, Any]:
    """识别图像模态, 返回路由决策 dict.

    返回:
      {
        "modality": "SEM" | "TEM" | "EDS" | "PARTICLE" | "DEFECT" |
                     "PHASE_FIELD" | "PLOT" | "DOCUMENT" | "PHOTO" | "UNKNOWN",
        "action": str | None,       # 推荐 image_analysis_tool action
        "confidence": "high"|"medium"|"low",
        "reasons": [str, ...],
      }

    判定顺序: 文件名强信号 > 图像统计特征 (边缘密度/颜色/尺寸) > 默认.
    """
    result: dict[str, Any] = {
        "modality": "UNKNOWN",
        "action": None,
        "confidence": "low",
        "reasons": [],
    }
    if isinstance(image_path, (bytes, bytearray)):
        result["reasons"].append("raw bytes, no path/filename signal")
        return result

    p = Path(image_path)
    name_hint = (p.stem or "").lower()
    ext = p.suffix.lower()
    # 下划线/连字符/空格都视为分隔符, 让 \b 词边界能命中 sem_photo / eds_mapping
    # 这类连写文件名 (否则 \b 在 '_' 处失效).
    name_hint_norm = re.sub(r"[\s_\-.,]+", " ", name_hint).strip()

    # 1. 文件名强信号
    for pattern, modality, action in _FILENAME_SIGNALS:
        if re.search(pattern, name_hint_norm):
            result["modality"] = modality
            result["action"] = action
            result["confidence"] = "high"
            result["reasons"].append(f"filename signal: {pattern}")
            return result

    if not p.is_file():
        result["reasons"].append(f"file not found: {image_path}")
        return result

    # 2. 图像统计特征
    try:
        from huginn.tools.image_analysis._utils import load_gray, load_rgb

        gray = load_gray(str(p))
        std_i = float(gray.std())
        edge_density = _edge_density(gray)
        # 图表: 单色/低饱和背景 + 高边缘密度 (坐标轴/曲线)
        try:
            rgb = load_rgb(str(p))
            s = _saturation(rgb)
        except Exception:
            s = 0.0

        if edge_density >= 0:
            # 高边缘 + 低饱和 → 图表
            if edge_density > 0.12 and s < 0.35:
                result["modality"] = "PLOT"
                result["action"] = "plot_extract"
                result["confidence"] = "medium"
                result["reasons"].append(
                    f"low-sat({s:.2f}) + busy edges({edge_density:.3f}) -> plot"
                )
                return result
            # 高边缘 + 高饱和 → 彩色显微/EDS 图
            if edge_density > 0.12 and s >= 0.35:
                result["modality"] = "EDS"
                result["action"] = "eds_mapping"
                result["confidence"] = "medium"
                result["reasons"].append(
                    f"high-sat({s:.2f}) + busy edges({edge_density:.3f}) -> EDS"
                )
                return result
            # 低边缘 + 低对比 → 显微图 (SEM/TEM 平滑区)
            if edge_density < 0.05 and std_i < 40:
                result["modality"] = "SEM"
                result["action"] = "sem_analysis"
                result["confidence"] = "medium"
                result["reasons"].append(
                    f"smooth({edge_density:.3f}) + low-contrast({std_i:.0f}) -> SEM"
                )
                return result
        result["reasons"].append(
            f"no strong signal (edge={edge_density:.3f}, sat={s:.2f})"
        )
    except Exception:
        logger.debug("detect_modality stats failed, non-fatal", exc_info=True)
        result["reasons"].append("image stats unavailable")

    # 3. 扩展名兜底
    if ext in (".csv", ".txt"):
        result["modality"] = "DOCUMENT"
        result["reasons"].append("extension is tabular/text")
    return result


def _edge_density(gray: Any) -> float:
    """近似边缘密度 (scipy.sobel), 失败返回 -1."""
    try:
        import numpy as np
        from scipy.ndimage import sobel

        sx = sobel(gray, axis=0)
        sy = sobel(gray, axis=1)
        total = gray.shape[0] * gray.shape[1]
        if total <= 0:
            return -1.0
        return float((np.hypot(sx, sy) > 50).sum()) / total
    except Exception:
        return -1.0


def _saturation(rgb: Any) -> float:
    """图像平均饱和度 (用 scipy 不可用时的 numpy 最小实现)."""
    try:
        import numpy as np

        arr = np.asarray(rgb, dtype=float)
        if arr.ndim == 2:
            return 0.0
        mx = arr.max(axis=2)
        mn = arr.min(axis=2)
        denom = mx.copy()
        denom[denom == 0] = 1.0
        sat = (mx - mn) / denom
        return float(sat.mean())
    except Exception:
        return 0.0


def modality_routing_hint(
    image_path: str | Path | bytes,
    max_pixel_cap: int = DEFAULT_MAX_PIXEL_CAP,
) -> str | None:
    """综合动态分辨率 + 模态路由, 返回一段机读提示注入文本通道.

    best-effort: 读取失败返回 None. 供 build_cv_context 调用.
    """
    p = Path(image_path) if not isinstance(image_path, (bytes, bytearray)) else None
    if p is not None and not p.is_file():
        return None

    lines: list[str] = ["[MM-Router]"]
    res = dynamic_resolution_hint(image_path, max_pixel_cap)
    if res:
        lines.append(res)
    det = detect_modality(image_path)
    if det["action"] in _KNOWN_ACTIONS:
        lines.append(
            f"modality={det['modality']} (conf={det['confidence']}) "
            f"-> recommend image_analysis_tool action={det['action']}"
        )
    else:
        lines.append(
            f"modality={det['modality']} (conf={det['confidence']}) "
            f"-> no dedicated action, use semantic description"
        )
    return "\n".join(lines)


__all__ = [
    "DEFAULT_MAX_PIXEL_CAP",
    "recommend_resolution",
    "dynamic_resolution_hint",
    "detect_modality",
    "modality_routing_hint",
]
