"""图像 I/O + 颜色解析 + Otsu 阈值 — scenes_*.py 共用的零件.

PIL 必装 (Pillow), cv2/skimage 都是可选, 这里不引入它们.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def load_gray(path: str) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("需要 Pillow: pip install Pillow") from exc
    return np.asarray(Image.open(path).convert("L"), dtype=float)


def load_rgb(path: str) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("需要 Pillow: pip install Pillow") from exc
    return np.asarray(Image.open(path).convert("RGB"), dtype=float)


def parse_color(color: Any) -> np.ndarray:
    """把颜色字符串/列表统一成 (R,G,B) float 数组."""
    if isinstance(color, (list, tuple)) and len(color) >= 3:
        return np.array(color[:3], dtype=float)
    if not isinstance(color, str):
        return np.array([128.0, 128.0, 128.0])

    s = color.strip().lower()
    named = {
        "red": [255, 0, 0], "green": [0, 200, 0], "blue": [0, 0, 255],
        "yellow": [255, 215, 0], "cyan": [0, 200, 200], "magenta": [255, 0, 255],
        "white": [255, 255, 255], "black": [0, 0, 0],
        "orange": [255, 140, 0], "purple": [160, 32, 240],
        "pink": [255, 105, 180], "brown": [139, 69, 19],
    }
    if s in named:
        return np.array(named[s], dtype=float)
    if s.startswith("#"):
        hex_str = s[1:]
        if len(hex_str) == 6:
            r = int(hex_str[0:2], 16)
            g = int(hex_str[2:4], 16)
            b = int(hex_str[4:6], 16)
            return np.array([r, g, b], dtype=float)
    # rgb(255,0,0) 形式
    if s.startswith("rgb"):
        inside = s[s.find("(") + 1 : s.find(")")]
        parts = [float(x.strip()) for x in inside.split(",")]
        if len(parts) >= 3:
            return np.array(parts[:3], dtype=float)
    return np.array([128.0, 128.0, 128.0])


def auto_detect_colors(rgb: np.ndarray, k: int = 5) -> dict[str, np.ndarray]:
    """没给 element_colors 时的兜底: 采样后跑一轮简版 k-means."""
    flat = rgb.reshape(-1, 3).astype(float)
    n_samples = min(5000, flat.shape[0])
    rng = np.random.default_rng(42)
    idx = rng.choice(flat.shape[0], n_samples, replace=False)
    sample = flat[idx]

    centers = sample[rng.choice(n_samples, k, replace=False)].astype(float)
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

    # 按亮度排序起名
    order = centers.sum(axis=1).argsort()
    names = [f"phase_{i + 1}" for i in range(k)]
    return {names[i]: centers[order[i]] for i in range(k)}


def otsu_numpy(arr: np.ndarray) -> float:
    """纯 numpy Otsu, skimage 不可用时的兜底."""
    hist, edges = np.histogram(arr.ravel(), bins=256, range=(0, 255))
    hist = hist.astype(float)
    total = hist.sum()
    if total == 0:
        return 127.5
    prob = hist / total
    cum_sum = np.cumsum(prob)
    cum_mean = np.cumsum(np.arange(256) * prob)
    global_mean = cum_mean[-1]
    denom = cum_sum * (1.0 - cum_sum) + 1e-12
    between = (global_mean * cum_sum - cum_mean) ** 2 / denom
    return float(edges[np.argmax(between)])
