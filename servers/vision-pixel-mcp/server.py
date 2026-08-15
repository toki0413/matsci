"""Vision Pixel MCP Server — 通用像素级视觉工具.

移植自 dsh-vision-router 的像素闭环能力, 用 PIL/numpy 纯 Python 实现,
无 Node/sharp/tesseract/chrome 依赖. 提供与 huginn 已有 image_analysis_tool
(材料科学 SEM/TEM/EDS 分析) 互补的通用像素操作:

  - vision_crop                按像素框裁剪放大
  - vision_colors              主色提取 (hex + 占比)
  - vision_pixel_diff          逐像素对比: 差异率 + 最差区块
  - vision_extract_foreground  边界洪泛抠图 (纯色背景 → 透明 PNG)
  - vision_trace               简易 SVG 矢量化 (色块轮廓)
  - vision_describe            基础看图问答 (返回尺寸/通道/主色摘要)

启动: python server.py [--transport stdio|sse]
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool, Resource

try:
    from PIL import Image
    import numpy as np
    _HAS_NP = True
except Exception:  # pragma: no cover
    _HAS_NP = False

app = Server("vision-pixel-mcp")

MAX_SIDE = 2000  # 最长边上限, 防止极端宽高比


def _load_image(path: str) -> "Image.Image":
    """按魔数读取图片, 无扩展名也能识别. 最长边超限自动等比降采样."""
    img = Image.open(path)
    img.load()
    w, h = img.size
    scale = min(1.0, MAX_SIDE / max(w, h))
    if scale < 1.0:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    return img


def _encode_png(img: "Image.Image") -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _to_rgba(path: str) -> tuple["Image.Image", "np.ndarray"]:
    img = _load_image(path)
    arr = np.asarray(img.convert("RGBA"), dtype=np.uint8)
    return img, arr


# ---------------------------------------------------------------------------
# 工具实现
# ---------------------------------------------------------------------------


def vision_crop(image_path: str, region: str) -> dict[str, Any]:
    """按像素框 x1,y1,x2,y2 裁剪放大. region 如 "100,200,400,500"."""
    img = _load_image(image_path)
    try:
        parts = [int(p.strip()) for p in region.split(",")]
        if len(parts) != 4:
            raise ValueError
        x1, y1, x2, y2 = parts
    except Exception:
        return {"error": f"invalid region '{region}', expected x1,y1,x2,y2"}
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.width, x2), min(img.height, y2)
    if x2 <= x1 or y2 <= y1:
        return {"error": "empty crop region after clamping"}
    crop = img.crop((x1, y1, x2, y2))
    return {
        "region": [x1, y1, x2, y2],
        "size": [crop.width, crop.height],
        "png_base64": _encode_png(crop),
    }


def vision_colors(image_path: str, top: int = 8) -> dict[str, Any]:
    """主色提取: 量化后返回 hex + 占比."""
    img = _load_image(image_path).convert("RGB")
    small = img.resize((max(1, img.width // 8), max(1, img.height // 8)), Image.BOX)
    colors = Counter(small.getdata())
    total = sum(colors.values())
    out = []
    for (r, g, b), cnt in colors.most_common(top):
        out.append({"hex": f"#{r:02x}{g:02x}{b:02x}", "rgb": [r, g, b],
                    "ratio": round(cnt / total, 4)})
    return {"colors": out, "image_size": [img.width, img.height]}


def vision_pixel_diff(original: str, rebuilt: str, threshold: int = 16) -> dict[str, Any]:
    """逐像素对比两张图: 差异率 + 最差 8x8 网格区域. 需 numpy."""
    if not _HAS_NP:
        return {"error": "numpy required"}
    a = _load_image(original).convert("RGB")
    b = _load_image(rebuilt).convert("RGB")
    # 尺寸对齐: 缩放到较小者
    w = min(a.width, b.width)
    h = min(a.height, b.height)
    a = a.resize((w, h))
    b = b.resize((w, h))
    arr_a = np.asarray(a, dtype=np.int16)
    arr_b = np.asarray(b, dtype=np.int16)
    diff = np.abs(arr_a - arr_b).max(axis=2)
    total_px = diff.size
    differ = diff > threshold
    diff_count = int(differ.sum())
    diff_ratio = diff_count / total_px

    # 最差 8x8 网格
    gh = min(8, h // 8) or 1
    gw = min(8, w // 8) or 1
    best = None
    for gy in range(0, h - gh + 1, gh):
        for gx in range(0, w - gw + 1, gw):
            block = differ[gy:gy + gh, gx:gx + gw]
            score = float(block.mean())
            if best is None or score > best[0]:
                best = (score, [gx, gy, gx + gw, gy + gh])
    return {
        "diff_ratio": round(diff_ratio, 4),
        "diff_pixels": diff_count,
        "total_pixels": total_px,
        "threshold": threshold,
        "worst_region": best[1] if best else None,
        "worst_density": round(best[0], 4) if best else None,
    }


def vision_extract_foreground(image_path: str, bg_tolerance: int = 24) -> dict[str, Any]:
    """边界洪泛抠图: 从图片四边遇到的相似颜色作为背景, 其余变透明. 需 numpy."""
    if not _HAS_NP:
        return {"error": "numpy required"}
    img = _load_image(image_path).convert("RGBA")
    arr = np.asarray(img, dtype=np.uint8).copy()
    h, w = arr.shape[:2]
    visited = np.zeros((h, w), dtype=bool)
    stack = []
    # 四边作为种子
    for x in range(w):
        for y in (0, h - 1):
            stack.append((y, x))
    for y in range(h):
        for x in (0, w - 1):
            stack.append((y, x))
    bg = arr[0, 0, :3].astype(np.int16)
    while stack:
        y, x = stack.pop()
        if y < 0 or y >= h or x < 0 or x >= w or visited[y, x]:
            continue
        visited[y, x] = True
        px = arr[y, x, :3].astype(np.int16)
        if int(np.abs(px - bg).max()) > bg_tolerance:
            continue
        arr[y, x, 3] = 0  # 透明
        stack.append((y - 1, x))
        stack.append((y + 1, x))
        stack.append((y, x - 1))
        stack.append((y, x + 1))
    out = Image.fromarray(arr, mode="RGBA")
    return {
        "bg_tolerance": bg_tolerance,
        "removed_transparent": int(visited.sum()),
        "png_base64": _encode_png(out),
    }


def vision_trace(image_path: str, steps: int = 4) -> dict[str, Any]:
    """简易 SVG 矢量化: 颜色量化后, 为每个色块生成一个矩形轮廓."""
    img = _load_image(image_path).convert("RGB")
    n = max(2, min(16, steps))
    q = img.quantize(colors=n, method=Image.MEDIANCUT).convert("RGB")
    arr = np.asarray(q, dtype=np.uint8)
    h, w = arr.shape[:2]
    rects = []
    seen = set()
    for step in range(n):
        # 采样像素找该色块 bbox
        max_rect = None
        max_area = 0
        for y in range(0, h, 4):
            for x in range(0, w, 4):
                key = tuple(arr[y, x])
                if key in seen:
                    continue
                # 计算该颜色的 bbox
                mask = (arr == key).all(axis=2)
                ys, xs = np.where(mask)
                if len(ys) == 0:
                    continue
                area = int(mask.sum())
                if area > max_area:
                    max_area = area
                    max_rect = (key, int(xs.min()), int(ys.min()),
                                int(xs.max()), int(ys.max()))
        if max_rect is None:
            break
        key, x0, y0, x1, y1 = max_rect
        seen.add(key)
        rects.append({
            "fill": f"rgb({key[0]},{key[1]},{key[2]})",
            "bbox": [x0, y0, x1, y1],
            "area_px": max_area,
        })
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">\n'
        '<rect width="100%" height="100%" fill="white"/>\n'
    )
    for r in rects:
        x0, y0 = r["bbox"][0], r["bbox"][1]
        wd, ht = r["bbox"][2] - x0, r["bbox"][3] - y0
        svg += f'  <rect x="{x0}" y="{y0}" width="{wd}" height="{ht}" fill="{r["fill"]}" opacity="0.9"/>\n'
    svg += "</svg>"
    return {
        "n_regions": len(rects),
        "regions": rects,
        "svg": svg,
    }


def vision_describe(image_path: str) -> dict[str, Any]:
    """基础看图摘要: 尺寸/模式/主色. 深度语义描述走 huginn 的 image_analysis_tool."""
    img = _load_image(image_path)
    colors = vision_colors(image_path, top=5)
    return {
        "image_size": [img.width, img.height],
        "mode": img.mode,
        "top_colors": colors["colors"],
        "note": "需要深度语义时请改用 huginn 的 image_analysis_tool (SEM/TEM/EDS 等)",
    }


# ---------------------------------------------------------------------------
# MCP 工具注册
# ---------------------------------------------------------------------------

TOOLS: dict[str, dict[str, Any]] = {
    "vision_crop": {
        "description": ("按像素框 x1,y1,x2,y2 裁剪放大图片. "
                        "返回裁剪后的 PNG (base64) 与区域坐标."),
        "inputSchema": {"type": "object",
                        "properties": {
                            "image_path": {"type": "string", "description": "图片路径"},
                            "region": {"type": "string", "description": "x1,y1,x2,y2"},
                        },
                        "required": ["image_path", "region"]},
        "fn": vision_crop,
    },
    "vision_colors": {
        "description": "提取图片主色 (hex + 占比).",
        "inputSchema": {"type": "object",
                        "properties": {
                            "image_path": {"type": "string"},
                            "top": {"type": "integer", "default": 8},
                        },
                        "required": ["image_path"]},
        "fn": vision_colors,
    },
    "vision_pixel_diff": {
        "description": "逐像素对比两张图, 返回差异率与最差区域.",
        "inputSchema": {"type": "object",
                        "properties": {
                            "original": {"type": "string"},
                            "rebuilt": {"type": "string"},
                            "threshold": {"type": "integer", "default": 16},
                        },
                        "required": ["original", "rebuilt"]},
        "fn": vision_pixel_diff,
    },
    "vision_extract_foreground": {
        "description": "边界洪泛抠图: 纯色背景变透明, 返回透明 PNG.",
        "inputSchema": {"type": "object",
                        "properties": {
                            "image_path": {"type": "string"},
                            "bg_tolerance": {"type": "integer", "default": 24},
                        },
                        "required": ["image_path"]},
        "fn": vision_extract_foreground,
    },
    "vision_trace": {
        "description": "简易 SVG 矢量化: 颜色量化后输出色块矩形轮廓.",
        "inputSchema": {"type": "object",
                        "properties": {
                            "image_path": {"type": "string"},
                            "steps": {"type": "integer", "default": 4},
                        },
                        "required": ["image_path"]},
        "fn": vision_trace,
    },
    "vision_describe": {
        "description": "基础看图摘要 (尺寸/模式/主色). 深度语义走 image_analysis_tool.",
        "inputSchema": {"type": "object",
                        "properties": {"image_path": {"type": "string"}},
                        "required": ["image_path"]},
        "fn": vision_describe,
    },
}


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(name=name, description=meta["description"],
             inputSchema=meta["inputSchema"])
        for name, meta in TOOLS.items()
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    meta = TOOLS.get(name)
    if meta is None:
        raise ValueError(f"unknown tool: {name}")
    try:
        result = meta["fn"](**arguments)
    except Exception as e:  # noqa: BLE001
        result = {"error": f"{type(e).__name__}: {e}"}
    return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]


async def main(transport: str) -> None:
    async with stdio_server() as (read, write):
        await app.run(
            read, write,
            InitializationOptions(
                server_name="vision-pixel-mcp",
                server_version="0.1.0",
                capabilities=app.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--transport", default="stdio", choices=["stdio", "sse"])
    args = ap.parse_args()
    asyncio.run(main(args.transport))