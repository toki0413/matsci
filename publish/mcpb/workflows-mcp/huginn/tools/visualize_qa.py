"""渲染式图表 QA 原语 — 确定性图表质量门禁 (无 LLM, numpy-only).

对应 WorldClaw render-based refinement / Qwen-MM 视觉反馈闭环的
「生成 → 诊断渲染 → 质量检测」第一步. 对 visualize_tool 产出的 PNG 做
确定性健康检查, 输出 verdict ∈ {pass, fix_needed, fail} + 结构化 flags.

设计约束 (ponytail):
- 纯函数, 不依赖 agent/LLM 状态, 方便单测与在批次 C 精修闭环里复用.
- 只依赖 numpy + Pillow (PIL 缺失或图片损坏时优雅降级为 verdict=error).
- 单个检查失败不阻断整体 (如 scipy 缺失时 cluttered 使用 numpy 梯度).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

VERDICT_PASS = "pass"
VERDICT_FIX = "fix_needed"
VERDICT_FAIL = "fail"
VERDICT_ERROR = "error"

# flag → 机读修正指令 (供批次 C 精修闭环拼接 directive)
FLAG_REMEDIES: dict[str, str] = {
    "blank": "figure has no visible content; check data/colormap/axes",
    "clipped": "content touches image edges; set bbox_inches='tight' or add margins",
    "extremely_elongated": "aspect ratio outside sane bounds; check figure size / aspect",
    "cluttered": "high edge density in plot interior; reduce label/text overlap",
}

# 与背景色差超过该阈值视为"内容"像素 (0-255)
_CONTENT_DELTA = 20.0


def _load_gray_resized(
    path: str | Path, max_dim: int = 2048
) -> tuple[np.ndarray, dict[str, Any]]:
    """加载灰度图 + 动态分辨率预处理.

    超大图等比缩到 max_dim 以控制计算量; 小图保持原样 (细文字不失真).
    返回 (灰度数组, 元信息).
    """
    from PIL import Image

    img = Image.open(path).convert("L")
    w, h = img.size
    meta: dict[str, Any] = {"orig_width": w, "orig_height": h, "downscaled": False}
    if max_dim and max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        img = img.resize(
            (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
        )
        meta["downscaled"] = True
    arr = np.asarray(img, dtype=float)
    meta["width"], meta["height"] = arr.shape[1], arr.shape[0]
    return arr, meta


def _background_level(gray: np.ndarray) -> float:
    """背景亮度 = 四角小块像素的中值.

    图表四角几乎总是背景 (白底≈255, 深色底≈低值). 比直方图 argmax 稳健:
    当墨色内容占主导时, argmax 会把前景当背景, 导致 blank/clipped 误判.
    """
    h, w = gray.shape
    cw = max(1, w // 10)
    ch = max(1, h // 10)
    corners = np.concatenate(
        [
            gray[:ch, :cw].ravel(),
            gray[:ch, -cw:].ravel(),
            gray[-ch:, :cw].ravel(),
            gray[-ch:, -cw:].ravel(),
        ]
    )
    return float(np.median(corners))


def _content_ratio(gray: np.ndarray) -> float:
    """非背景像素占比."""
    bg = _background_level(gray)
    return float((np.abs(gray - bg) > _CONTENT_DELTA).mean())


def check_blank(gray: np.ndarray) -> tuple[str | None, dict[str, Any]]:
    """整图无可见内容 → blank. std 过小或内容像素占比过低."""
    std = float(gray.std())
    non_bg = _content_ratio(gray)
    flag = "blank" if (std < 3.0 or non_bg < 0.01) else None
    return flag, {"std": std, "non_bg_ratio": non_bg, "bg_level": _background_level(gray)}


def check_clipped(
    gray: np.ndarray, margin_frac: float = 0.02, threshold: float = 0.005
) -> tuple[str | None, dict[str, Any]]:
    """内容贴边 → clipped. 边缘带内内容像素占比过高表示被裁切."""
    h, w = gray.shape
    mw = max(1, int(round(w * margin_frac)))
    mh = max(1, int(round(h * margin_frac)))
    bg = _background_level(gray)
    bands = [
        gray[:mh, :].ravel(),
        gray[-mh:, :].ravel(),
        gray[:, :mw].ravel(),
        gray[:, -mw:].ravel(),
    ]
    edge = np.concatenate(bands)
    content = float((np.abs(edge - bg) > _CONTENT_DELTA).mean())
    flag = "clipped" if content > threshold else None
    return flag, {"edge_content_ratio": content}


def check_aspect(
    gray: np.ndarray, min_ratio: float = 0.3, max_ratio: float = 3.3
) -> tuple[str | None, dict[str, Any]]:
    """极端纵横比 → 可能被拉伸/压缩失真."""
    h, w = gray.shape
    ratio = w / h if h > 0 else 0.0
    flag = (
        "extremely_elongated" if (ratio < min_ratio or ratio > max_ratio) else None
    )
    return flag, {"width": w, "height": h, "aspect": float(ratio)}


def check_cluttered(
    gray: np.ndarray, edge_thresh: float = 0.25
) -> tuple[str | None, dict[str, Any]]:
    """内部区域边缘密度过高 → cluttered (标签/文字重叠启发)."""
    h, w = gray.shape
    if h < 8 or w < 8:
        return None, {"inner_edge_density": 0.0, "note": "too small"}
    inner = gray[int(h * 0.08) : int(h * 0.92), int(w * 0.08) : int(w * 0.92)]
    try:
        from scipy.ndimage import sobel

        sx = sobel(inner, axis=0)
        sy = sobel(inner, axis=1)
    except Exception:  # scipy 缺失 → numpy 梯度兜底
        sx = np.gradient(inner, axis=0)
        sy = np.gradient(inner, axis=1)
    edge = float((np.hypot(sx, sy) > 50).mean())
    flag = "cluttered" if edge > edge_thresh else None
    return flag, {"inner_edge_density": edge}


def qa_figure(
    image_path: str | Path,
    max_dim: int = 2048,
    margin_frac: float = 0.02,
    edge_thresh: float = 0.25,
) -> dict[str, Any]:
    """对一张图表 PNG 做确定性 QA.

    返回 dict: {verdict, flags, remedies, metrics, image_path}.
    verdict 判定: blank/clipped 直接 fail; 其余问题 fix_needed; 无 flag pass;
    文件缺失/读取失败 → error.
    """
    p = Path(image_path)
    result: dict[str, Any] = {
        "image_path": str(p),
        "verdict": VERDICT_PASS,
        "flags": [],
        "remedies": [],
        "metrics": {},
    }
    if not p.is_file():
        result["verdict"] = VERDICT_ERROR
        result["error"] = f"file not found: {image_path}"
        return result
    try:
        gray, meta = _load_gray_resized(p, max_dim=max_dim)
    except Exception as exc:  # Pillow 缺失 / 非图片文件
        logger.debug("visualize_qa: image load failed", exc_info=True)
        result["verdict"] = VERDICT_ERROR
        result["error"] = f"image load failed: {exc}"
        return result

    result["metrics"]["image"] = meta
    for fn in (check_blank, check_clipped, check_aspect, check_cluttered):
        try:
            flag, m = fn(gray)
        except Exception as exc:  # 单个检查失败不阻断整体
            logger.debug("visualize_qa: check %s failed", fn.__name__, exc_info=True)
            result["metrics"][fn.__name__] = {"error": str(exc)}
            continue
        result["metrics"][fn.__name__] = m
        if flag:
            result["flags"].append(flag)
            result["remedies"].append(FLAG_REMEDIES[flag])

    if not result["flags"]:
        result["verdict"] = VERDICT_PASS
    elif any(f in ("blank", "clipped") for f in result["flags"]):
        result["verdict"] = VERDICT_FAIL
    else:
        result["verdict"] = VERDICT_FIX
    return result


def qa_directive(result: dict[str, Any]) -> str:
    """把 QA 结果序列化为机读修正指令 (供精修闭环消费).

    verdict=pass 返回空串; 否则拼接每条 flag 的 remedy.
    """
    if result.get("verdict") == VERDICT_PASS:
        return ""
    if result.get("verdict") == VERDICT_ERROR:
        return f"visual QA error: {result.get('error', 'unknown')}"
    parts = result.get("remedies") or []
    return " | ".join(parts) if parts else "visual QA flagged; inspect figure"
