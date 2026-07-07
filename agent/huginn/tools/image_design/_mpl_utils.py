"""matplotlib 共享工具: 字体配置 / 存图 / 图像加载 / 颜色解析 / 参数取值.

所有函数从原 ImageDesignTool 方法转来, 去掉 self.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .tool import ImageDesignInput

logger = logging.getLogger(__name__)


def setup_matplotlib() -> None:
    """统一 Arial 20pt bold, 用户硬性要求."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    # 显式注册 Arial Bold, 不然 findfont 会回退到 weight 400
    for candidate in [
        r"C:\Windows\Fonts\arialbd.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
    ]:
        if Path(candidate).exists():
            try:
                font_manager.fontManager.addfont(candidate)
            except Exception:
                logger.debug("addfont failed", exc_info=True)

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 20
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["axes.titleweight"] = "bold"


def save_figure(fig, output_path: str) -> None:
    """保存 figure, 自动建父目录, 位图 300 dpi."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    suffix = out.suffix.lower()
    dpi = 300 if suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff") else None
    fig.savefig(str(out), dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    import matplotlib.pyplot as plt

    plt.close(fig)


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
    """颜色字符串/列表统一成 (R,G,B) float, 给 overlay 用."""
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
    }
    if s in named:
        return np.array(named[s], dtype=float)
    if s.startswith("#") and len(s) == 7:
        return np.array(
            [int(s[1:3], 16), int(s[3:5], 16), int(s[5:7], 16)], dtype=float
        )
    return np.array([128.0, 128.0, 128.0])


def get_param(args: ImageDesignInput, key: str, default: Any = None) -> Any:
    """data 字段优先, 其次 parameters."""
    if args.data and key in args.data:
        return args.data[key]
    return args.parameters.get(key, default)
