"""DesignAtom 工具: 把设计任务抽象成原子, 像乐高积木一样自由组合.

设计思路 (参考 frontend-design / Graphic Design / famou-result-visualization):
- 没有原生多模态也能做生成式设计: LLM 生代码 → 渲染 → 工具反馈
- 把设计任务拆成 4 类原子:
    Layout:   布局 (grid/flex/hero/two-column/dashboard/sidebar)
    Style:    风格 (palette/typography/spacing/theme)
    Geometry: 几何 (shape/size/position/rotation)
    DataViz:  数据可视化 (bar/line/scatter/heatmap/contour)
- 每个原子有标准参数 schema + render() 方法返回代码片段
- compose action 把多个原子按组合规则拼成完整设计

actions:
- list_atoms:        列出所有可用原子类型和参数 schema
- render_atom:       渲染单个原子为代码片段 (HTML/CSS/SVG/Python)
- compose:           组合多个原子生成完整设计代码
- preview:           生成可预览的 HTML 或 Python 文件内容
"""

from __future__ import annotations

import json
import logging
from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool
from huginn.types import ToolContext, ToolResult, ValidationResult

logger = logging.getLogger(__name__)


# ---------- 原子定义 ----------

# 每个原子: {name, category, params: {key: {type, default, description}}}
_ATOM_REGISTRY: dict[str, dict[str, Any]] = {
    # Layout atoms
    "layout.grid": {
        "category": "layout",
        "description": "CSS Grid 布局, 行列网格",
        "params": {
            "columns": {"type": "int", "default": 3, "description": "列数"},
            "rows": {"type": "int", "default": 2, "description": "行数"},
            "gap": {"type": "str", "default": "16px", "description": "间距"},
        },
    },
    "layout.flex": {
        "category": "layout",
        "description": "Flexbox 布局",
        "params": {
            "direction": {"type": "str", "default": "row",
                          "description": "row|column|row-reverse|column-reverse"},
            "justify": {"type": "str", "default": "flex-start",
                        "description": "主轴对齐"},
            "align": {"type": "str", "default": "stretch",
                      "description": "交叉轴对齐"},
            "gap": {"type": "str", "default": "12px", "description": "间距"},
        },
    },
    "layout.hero": {
        "category": "layout",
        "description": "Hero 区块: 大标题 + 副标题 + CTA",
        "params": {
            "title": {"type": "str", "default": "Title", "description": "主标题"},
            "subtitle": {"type": "str", "default": "", "description": "副标题"},
            "cta": {"type": "str", "default": "Get Started", "description": "按钮文字"},
            "align": {"type": "str", "default": "center",
                      "description": "left|center|right"},
        },
    },
    "layout.two_column": {
        "category": "layout",
        "description": "两栏布局: 主内容 + 侧栏",
        "params": {
            "main_ratio": {"type": "float", "default": 0.7,
                           "description": "主栏占比 0-1"},
            "gap": {"type": "str", "default": "24px", "description": "栏间距"},
        },
    },
    "layout.dashboard": {
        "category": "layout",
        "description": "仪表板布局: 顶栏 + 侧边栏 + 内容区",
        "params": {
            "sidebar_width": {"type": "str", "default": "240px",
                              "description": "侧边栏宽度"},
            "header_height": {"type": "str", "default": "64px",
                              "description": "顶栏高度"},
        },
    },
    # Style atoms
    "style.palette": {
        "category": "style",
        "description": "配色方案",
        "params": {
            "primary": {"type": "str", "default": "#2563eb", "description": "主色"},
            "secondary": {"type": "str", "default": "#64748b",
                          "description": "次色"},
            "accent": {"type": "str", "default": "#f59e0b",
                       "description": "强调色"},
            "bg": {"type": "str", "default": "#ffffff", "description": "背景色"},
            "text": {"type": "str", "default": "#0f172a", "description": "文字色"},
        },
    },
    "style.typography": {
        "category": "style",
        "description": "字体排版",
        "params": {
            "font_family": {"type": "str", "default": "Arial, sans-serif",
                            "description": "字体族"},
            "base_size": {"type": "str", "default": "16px",
                          "description": "基础字号"},
            "heading_scale": {"type": "float", "default": 1.25,
                              "description": "标题放大比例"},
            "line_height": {"type": "float", "default": 1.6,
                            "description": "行高"},
        },
    },
    "style.spacing": {
        "category": "style",
        "description": "间距系统",
        "params": {
            "unit": {"type": "str", "default": "8px",
                     "description": "基础间距单位"},
            "scale": {"type": "str", "default": "major-third",
                      "description": "比例尺: major-third|minor-third|perfect-fourth"},
        },
    },
    "style.theme": {
        "category": "style",
        "description": "主题 (亮/暗)",
        "params": {
            "mode": {"type": "str", "default": "light",
                     "description": "light|dark"},
            "radius": {"type": "str", "default": "8px",
                       "description": "圆角"},
            "shadow": {"type": "str", "default": "0 1px 3px rgba(0,0,0,0.1)",
                       "description": "阴影"},
        },
    },
    # Geometry atoms
    "geometry.shape": {
        "category": "geometry",
        "description": "形状 (SVG)",
        "params": {
            "kind": {"type": "str", "default": "rect",
                     "description": "rect|circle|ellipse|polygon|path"},
            "width": {"type": "str", "default": "100",
                      "description": "宽度(SVG单位)"},
            "height": {"type": "str", "default": "100",
                       "description": "高度(SVG单位)"},
            "fill": {"type": "str", "default": "#2563eb",
                     "description": "填充色"},
            "stroke": {"type": "str", "default": "none",
                       "description": "描边色"},
        },
    },
    "geometry.position": {
        "category": "geometry",
        "description": "定位 (CSS)",
        "params": {
            "position": {"type": "str", "default": "relative",
                         "description": "static|relative|absolute|fixed|sticky"},
            "top": {"type": "str", "default": "auto"},
            "right": {"type": "str", "default": "auto"},
            "bottom": {"type": "str", "default": "auto"},
            "left": {"type": "str", "default": "auto"},
        },
    },
    # DataViz atoms
    "dataviz.bar": {
        "category": "dataviz",
        "description": "柱状图 (matplotlib)",
        "params": {
            "title": {"type": "str", "default": "Bar Chart",
                      "description": "图表标题"},
            "x_label": {"type": "str", "default": "X"},
            "y_label": {"type": "str", "default": "Y"},
            "color": {"type": "str", "default": "#2563eb",
                      "description": "柱色"},
            "data": {"type": "list", "default": [],
                     "description": "数据, [(label, value), ...]"},
        },
    },
    "dataviz.line": {
        "category": "dataviz",
        "description": "折线图 (matplotlib)",
        "params": {
            "title": {"type": "str", "default": "Line Chart"},
            "x_label": {"type": "str", "default": "X"},
            "y_label": {"type": "str", "default": "Y"},
            "color": {"type": "str", "default": "#2563eb"},
            "marker": {"type": "str", "default": "o",
                       "description": "点标记: o|s|^|none"},
            "data": {"type": "list", "default": [],
                     "description": "[(x, y), ...]"},
        },
    },
    "dataviz.scatter": {
        "category": "dataviz",
        "description": "散点图 (matplotlib)",
        "params": {
            "title": {"type": "str", "default": "Scatter"},
            "x_label": {"type": "str", "default": "X"},
            "y_label": {"type": "str", "default": "Y"},
            "color": {"type": "str", "default": "#2563eb"},
            "size": {"type": "int", "default": 30, "description": "点大小"},
            "data": {"type": "list", "default": [], "description": "[(x, y), ...]"},
        },
    },
    "dataviz.heatmap": {
        "category": "dataviz",
        "description": "热力图 (matplotlib)",
        "params": {
            "title": {"type": "str", "default": "Heatmap"},
            "cmap": {"type": "str", "default": "viridis",
                     "description": "色彩映射"},
            "data": {"type": "list", "default": [],
                     "description": "2D 数组 [[...], ...]"},
        },
    },
    "dataviz.contour": {
        "category": "dataviz",
        "description": "等值线图 (matplotlib)",
        "params": {
            "title": {"type": "str", "default": "Contour"},
            "x_label": {"type": "str", "default": "X"},
            "y_label": {"type": "str", "default": "Y"},
            "levels": {"type": "int", "default": 10, "description": "等值线数"},
            "data": {"type": "dict", "default": {},
                     "description": "{X: [...], Y: [...], Z: [[...], ...]}"},
        },
    },
}


# ---------- 原子渲染器 ----------

def _render_layout_grid(p: dict[str, Any]) -> str:
    cols = p.get("columns", 3)
    rows = p.get("rows", 2)
    gap = p.get("gap", "16px")
    return (
        f".grid {{\n"
        f"  display: grid;\n"
        f"  grid-template-columns: repeat({cols}, 1fr);\n"
        f"  grid-template-rows: repeat({rows}, 1fr);\n"
        f"  gap: {gap};\n"
        f"}}\n"
    )


def _render_layout_flex(p: dict[str, Any]) -> str:
    return (
        f".flex {{\n"
        f"  display: flex;\n"
        f"  flex-direction: {p.get('direction', 'row')};\n"
        f"  justify-content: {p.get('justify', 'flex-start')};\n"
        f"  align-items: {p.get('align', 'stretch')};\n"
        f"  gap: {p.get('gap', '12px')};\n"
        f"}}\n"
    )


def _render_layout_hero(p: dict[str, Any]) -> str:
    title = p.get("title", "Title")
    subtitle = p.get("subtitle", "")
    cta = p.get("cta", "Get Started")
    align = p.get("align", "center")
    sub_html = f"\n    <p class=\"hero-sub\">{subtitle}</p>" if subtitle else ""
    return (
        f"<section class=\"hero\" style=\"text-align: {align}; padding: 64px 24px;\">\n"
        f"  <h1 class=\"hero-title\">{title}</h1>{sub_html}\n"
        f"  <button class=\"hero-cta\">{cta}</button>\n"
        f"</section>\n"
    )


def _render_layout_two_column(p: dict[str, Any]) -> str:
    ratio = float(p.get("main_ratio", 0.7))
    main_pct = int(ratio * 100)
    side_pct = 100 - main_pct
    gap = p.get("gap", "24px")
    return (
        f".two-col {{\n"
        f"  display: grid;\n"
        f"  grid-template-columns: {main_pct}fr {side_pct}fr;\n"
        f"  gap: {gap};\n"
        f"}}\n"
    )


def _render_layout_dashboard(p: dict[str, Any]) -> str:
    sw = p.get("sidebar_width", "240px")
    hh = p.get("header_height", "64px")
    return (
        f".dashboard {{\n"
        f"  display: grid;\n"
        f"  grid-template-columns: {sw} 1fr;\n"
        f"  grid-template-rows: {hh} 1fr;\n"
        f"  grid-template-areas: \"header header\" \"sidebar main\";\n"
        f"  height: 100vh;\n"
        f"}}\n"
        f".dashboard-header {{ grid-area: header; }}\n"
        f".dashboard-sidebar {{ grid-area: sidebar; }}\n"
        f".dashboard-main {{ grid-area: main; padding: 16px; overflow: auto; }}\n"
    )


def _render_style_palette(p: dict[str, Any]) -> str:
    return (
        f":root {{\n"
        f"  --color-primary: {p.get('primary', '#2563eb')};\n"
        f"  --color-secondary: {p.get('secondary', '#64748b')};\n"
        f"  --color-accent: {p.get('accent', '#f59e0b')};\n"
        f"  --color-bg: {p.get('bg', '#ffffff')};\n"
        f"  --color-text: {p.get('text', '#0f172a')};\n"
        f"}}\n"
    )


def _render_style_typography(p: dict[str, Any]) -> str:
    fam = p.get("font_family", "Arial, sans-serif")
    base = p.get("base_size", "16px")
    scale = float(p.get("heading_scale", 1.25))
    lh = float(p.get("line_height", 1.6))
    h1 = float(base.replace("px", "")) * (scale ** 3)
    h2 = float(base.replace("px", "")) * (scale ** 2)
    h3 = float(base.replace("px", "")) * scale
    return (
        f":root {{\n"
        f"  --font-family: {fam};\n"
        f"  --font-base: {base};\n"
        f"  --line-height: {lh};\n"
        f"  --font-h1: {h1:.1f}px;\n"
        f"  --font-h2: {h2:.1f}px;\n"
        f"  --font-h3: {h3:.1f}px;\n"
        f"}}\n"
        f"body {{ font-family: var(--font-family); font-size: var(--font-base); "
        f"line-height: var(--line-height); }}\n"
        f"h1 {{ font-size: var(--font-h1); }}\n"
        f"h2 {{ font-size: var(--font-h2); }}\n"
        f"h3 {{ font-size: var(--font-h3); }}\n"
    )


def _render_style_spacing(p: dict[str, Any]) -> str:
    unit = p.get("unit", "8px")
    unit_val = float(unit.replace("px", ""))
    scale_name = p.get("scale", "major-third")
    ratios = {
        "major-third": 1.25,
        "minor-third": 1.2,
        "perfect-fourth": 1.333,
    }
    r = ratios.get(scale_name, 1.25)
    sizes = [unit_val * (r ** i) for i in range(6)]
    css = ":root {\n"
    for i, s in enumerate(sizes):
        css += f"  --space-{i}: {s:.2f}px;\n"
    css += "}\n"
    return css


def _render_style_theme(p: dict[str, Any]) -> str:
    mode = p.get("mode", "light")
    radius = p.get("radius", "8px")
    shadow = p.get("shadow", "0 1px 3px rgba(0,0,0,0.1)")
    if mode == "dark":
        return (
            f":root {{\n"
            f"  --bg: #0f172a;\n"
            f"  --text: #f1f5f9;\n"
            f"  --radius: {radius};\n"
            f"  --shadow: {shadow};\n"
            f"}}\n"
        )
    return (
        f":root {{\n"
        f"  --bg: #ffffff;\n"
        f"  --text: #0f172a;\n"
        f"  --radius: {radius};\n"
        f"  --shadow: {shadow};\n"
        f"}}\n"
    )


def _render_geometry_shape(p: dict[str, Any]) -> str:
    kind = p.get("kind", "rect")
    w = p.get("width", "100")
    h = p.get("height", "100")
    fill = p.get("fill", "#2563eb")
    stroke = p.get("stroke", "none")
    if kind == "rect":
        return (
            f"<svg width=\"{w}\" height=\"{h}\">\n"
            f"  <rect width=\"{w}\" height=\"{h}\" fill=\"{fill}\" stroke=\"{stroke}\"/>\n"
            f"</svg>\n"
        )
    if kind == "circle":
        r = str(float(w) / 2)
        cx = str(float(w) / 2)
        cy = str(float(h) / 2)
        return (
            f"<svg width=\"{w}\" height=\"{h}\">\n"
            f"  <circle cx=\"{cx}\" cy=\"{cy}\" r=\"{r}\" fill=\"{fill}\" stroke=\"{stroke}\"/>\n"
            f"</svg>\n"
        )
    if kind == "ellipse":
        rx = str(float(w) / 2)
        ry = str(float(h) / 2)
        cx = str(float(w) / 2)
        cy = str(float(h) / 2)
        return (
            f"<svg width=\"{w}\" height=\"{h}\">\n"
            f"  <ellipse cx=\"{cx}\" cy=\"{cy}\" rx=\"{rx}\" ry=\"{ry}\" fill=\"{fill}\" stroke=\"{stroke}\"/>\n"
            f"</svg>\n"
        )
    # polygon / path: 让 agent 自己给完整 path d
    return (
        f"<svg width=\"{w}\" height=\"{h}\">\n"
        f"  <!-- kind={kind} 需要额外 path 数据 -->\n"
        f"</svg>\n"
    )


def _render_geometry_position(p: dict[str, Any]) -> str:
    pos = p.get("position", "relative")
    top = p.get("top", "auto")
    right = p.get("right", "auto")
    bottom = p.get("bottom", "auto")
    left = p.get("left", "auto")
    return (
        f".positioned {{\n"
        f"  position: {pos};\n"
        f"  top: {top};\n"
        f"  right: {right};\n"
        f"  bottom: {bottom};\n"
        f"  left: {left};\n"
        f"}}\n"
    )


def _render_dataviz_bar(p: dict[str, Any]) -> str:
    title = p.get("title", "Bar Chart")
    x_label = p.get("x_label", "X")
    y_label = p.get("y_label", "Y")
    color = p.get("color", "#2563eb")
    data = p.get("data", [])
    data_repr = json.dumps(data, ensure_ascii=False)
    return (
        f"import matplotlib.pyplot as plt\n"
        f"import json\n\n"
        f"data = {data_repr}\n"
        f"labels = [d[0] for d in data]\n"
        f"values = [d[1] for d in data]\n\n"
        f"fig, ax = plt.subplots(figsize=(8, 5))\n"
        f"ax.bar(labels, values, color='{color}')\n"
        f"ax.set_title('{title}', fontsize=20, fontweight='bold', fontfamily='Arial')\n"
        f"ax.set_xlabel('{x_label}', fontsize=14, fontfamily='Arial')\n"
        f"ax.set_ylabel('{y_label}', fontsize=14, fontfamily='Arial')\n"
        f"for item in (ax.get_xticklabels() + ax.get_yticklabels()):\n"
        f"    item.set_fontsize(12)\n"
        f"    item.set_fontfamily('Arial')\n"
        f"plt.tight_layout()\n"
        f"plt.savefig('bar.png', dpi=150)\n"
        f"plt.show()\n"
    )


def _render_dataviz_line(p: dict[str, Any]) -> str:
    title = p.get("title", "Line Chart")
    x_label = p.get("x_label", "X")
    y_label = p.get("y_label", "Y")
    color = p.get("color", "#2563eb")
    marker = p.get("marker", "o")
    data = p.get("data", [])
    data_repr = json.dumps(data, ensure_ascii=False)
    return (
        f"import matplotlib.pyplot as plt\n\n"
        f"data = {data_repr}\n"
        f"xs = [d[0] for d in data]\n"
        f"ys = [d[1] for d in data]\n\n"
        f"fig, ax = plt.subplots(figsize=(8, 5))\n"
        f"ax.plot(xs, ys, color='{color}', marker='{marker}', linewidth=2)\n"
        f"ax.set_title('{title}', fontsize=20, fontweight='bold', fontfamily='Arial')\n"
        f"ax.set_xlabel('{x_label}', fontsize=14, fontfamily='Arial')\n"
        f"ax.set_ylabel('{y_label}', fontsize=14, fontfamily='Arial')\n"
        f"for item in (ax.get_xticklabels() + ax.get_yticklabels()):\n"
        f"    item.set_fontsize(12)\n"
        f"    item.set_fontfamily('Arial')\n"
        f"plt.tight_layout()\n"
        f"plt.savefig('line.png', dpi=150)\n"
        f"plt.show()\n"
    )


def _render_dataviz_scatter(p: dict[str, Any]) -> str:
    title = p.get("title", "Scatter")
    x_label = p.get("x_label", "X")
    y_label = p.get("y_label", "Y")
    color = p.get("color", "#2563eb")
    size = int(p.get("size", 30))
    data = p.get("data", [])
    data_repr = json.dumps(data, ensure_ascii=False)
    return (
        f"import matplotlib.pyplot as plt\n\n"
        f"data = {data_repr}\n"
        f"xs = [d[0] for d in data]\n"
        f"ys = [d[1] for d in data]\n\n"
        f"fig, ax = plt.subplots(figsize=(8, 5))\n"
        f"ax.scatter(xs, ys, color='{color}', s={size})\n"
        f"ax.set_title('{title}', fontsize=20, fontweight='bold', fontfamily='Arial')\n"
        f"ax.set_xlabel('{x_label}', fontsize=14, fontfamily='Arial')\n"
        f"ax.set_ylabel('{y_label}', fontsize=14, fontfamily='Arial')\n"
        f"for item in (ax.get_xticklabels() + ax.get_yticklabels()):\n"
        f"    item.set_fontsize(12)\n"
        f"    item.set_fontfamily('Arial')\n"
        f"plt.tight_layout()\n"
        f"plt.savefig('scatter.png', dpi=150)\n"
        f"plt.show()\n"
    )


def _render_dataviz_heatmap(p: dict[str, Any]) -> str:
    title = p.get("title", "Heatmap")
    cmap = p.get("cmap", "viridis")
    data = p.get("data", [])
    data_repr = json.dumps(data, ensure_ascii=False)
    return (
        f"import matplotlib.pyplot as plt\n"
        f"import numpy as np\n\n"
        f"data = np.array({data_repr})\n\n"
        f"fig, ax = plt.subplots(figsize=(8, 6))\n"
        f"im = ax.imshow(data, cmap='{cmap}', aspect='auto')\n"
        f"ax.set_title('{title}', fontsize=20, fontweight='bold', fontfamily='Arial')\n"
        f"cbar = fig.colorbar(im, ax=ax)\n"
        f"cbar.ax.tick_params(labelsize=12)\n"
        f"for item in (ax.get_xticklabels() + ax.get_yticklabels()):\n"
        f"    item.set_fontsize(12)\n"
        f"    item.set_fontfamily('Arial')\n"
        f"plt.tight_layout()\n"
        f"plt.savefig('heatmap.png', dpi=150)\n"
        f"plt.show()\n"
    )


def _render_dataviz_contour(p: dict[str, Any]) -> str:
    title = p.get("title", "Contour")
    x_label = p.get("x_label", "X")
    y_label = p.get("y_label", "Y")
    levels = int(p.get("levels", 10))
    data = p.get("data", {})
    data_repr = json.dumps(data, ensure_ascii=False)
    return (
        f"import matplotlib.pyplot as plt\n"
        f"import numpy as np\n\n"
        f"raw = {data_repr}\n"
        f"X = np.array(raw.get('X', []))\n"
        f"Y = np.array(raw.get('Y', []))\n"
        f"Z = np.array(raw.get('Z', []))\n\n"
        f"fig, ax = plt.subplots(figsize=(8, 6))\n"
        f"cs = ax.contour(X, Y, Z, {levels})\n"
        f"ax.clabel(cs, inline=True, fontsize=10)\n"
        f"ax.set_title('{title}', fontsize=20, fontweight='bold', fontfamily='Arial')\n"
        f"ax.set_xlabel('{x_label}', fontsize=14, fontfamily='Arial')\n"
        f"ax.set_ylabel('{y_label}', fontsize=14, fontfamily='Arial')\n"
        f"for item in (ax.get_xticklabels() + ax.get_yticklabels()):\n"
        f"    item.set_fontsize(12)\n"
        f"    item.set_fontfamily('Arial')\n"
        f"plt.tight_layout()\n"
        f"plt.savefig('contour.png', dpi=150)\n"
        f"plt.show()\n"
    )


_RENDERERS: dict[str, Any] = {
    "layout.grid": _render_layout_grid,
    "layout.flex": _render_layout_flex,
    "layout.hero": _render_layout_hero,
    "layout.two_column": _render_layout_two_column,
    "layout.dashboard": _render_layout_dashboard,
    "style.palette": _render_style_palette,
    "style.typography": _render_style_typography,
    "style.spacing": _render_style_spacing,
    "style.theme": _render_style_theme,
    "geometry.shape": _render_geometry_shape,
    "geometry.position": _render_geometry_position,
    "dataviz.bar": _render_dataviz_bar,
    "dataviz.line": _render_dataviz_line,
    "dataviz.scatter": _render_dataviz_scatter,
    "dataviz.heatmap": _render_dataviz_heatmap,
    "dataviz.contour": _render_dataviz_contour,
}


class DesignAtomInput(BaseModel):
    action: Literal["list_atoms", "render_atom", "compose", "preview"] = Field(...)
    # render_atom 时必填
    atom_name: str | None = Field(
        default=None,
        description="原子名, 如 layout.grid / style.palette / dataviz.bar",
    )
    params: dict[str, Any] | None = Field(
        default=None, description="原子参数"
    )
    # compose / preview 时必填
    atoms: list[dict[str, Any]] | None = Field(
        default=None,
        description="原子列表, 每项 {atom_name, params}",
    )
    # preview 时可选
    output_format: Literal["html", "python", "auto"] = Field(
        default="auto", description="输出格式"
    )


class DesignAtomTool(HuginnTool):
    """DesignAtom: 设计任务原子化, 像乐高积木一样组合."""

    name = "design_atom_tool"
    category = "design"
    description = (
        "Decompose design tasks into atoms (Layout/Style/Geometry/DataViz) "
        "and compose them like Lego. Actions: list_atoms (show all atom "
        "types and params), render_atom (render one atom to code), "
        "compose (combine multiple atoms), preview (generate full HTML or "
        "Python file content)."
    )
    input_schema = DesignAtomInput

    def is_read_only(self, args: DesignAtomInput) -> bool:
        return True

    async def validate_input(
        self, args: DesignAtomInput, context: ToolContext | None = None
    ) -> ValidationResult:
        if args.action == "render_atom":
            if not args.atom_name:
                return ValidationResult(
                    result=False, message="render_atom 需要 atom_name"
                )
            if args.atom_name not in _ATOM_REGISTRY:
                return ValidationResult(
                    result=False,
                    message=f"未知原子: {args.atom_name}. 可用: {list(_ATOM_REGISTRY.keys())}",
                )
        if args.action in ("compose", "preview"):
            if not args.atoms:
                return ValidationResult(
                    result=False, message=f"{args.action} 需要 atoms 列表"
                )
            for a in args.atoms:
                if "atom_name" not in a:
                    return ValidationResult(
                        result=False,
                        message="atoms 每项必须有 atom_name",
                    )
                if a["atom_name"] not in _ATOM_REGISTRY:
                    return ValidationResult(
                        result=False,
                        message=f"未知原子: {a['atom_name']}",
                    )
        return ValidationResult(result=True)

    async def call(
        self, args: dict[str, Any], context: ToolContext | None = None
    ) -> ToolResult:
        input_data = DesignAtomInput(**args)

        try:
            if input_data.action == "list_atoms":
                return ToolResult(
                    data={
                        "atoms": {
                            name: {
                                "category": info["category"],
                                "description": info["description"],
                                "params": info["params"],
                            }
                            for name, info in _ATOM_REGISTRY.items()
                        },
                        "categories": sorted({
                            info["category"]
                            for info in _ATOM_REGISTRY.values()
                        }),
                        "total": len(_ATOM_REGISTRY),
                    },
                    success=True,
                )

            if input_data.action == "render_atom":
                code = self._render_one(
                    input_data.atom_name or "",
                    input_data.params or {},
                )
                return ToolResult(
                    data={
                        "atom_name": input_data.atom_name,
                        "category": _ATOM_REGISTRY[input_data.atom_name]["category"],
                        "code": code,
                    },
                    success=True,
                )

            if input_data.action == "compose":
                snippets = []
                for a in input_data.atoms or []:
                    name = a["atom_name"]
                    params = a.get("params", {})
                    snippets.append({
                        "atom_name": name,
                        "category": _ATOM_REGISTRY[name]["category"],
                        "code": self._render_one(name, params),
                    })
                return ToolResult(
                    data={"snippets": snippets, "count": len(snippets)},
                    success=True,
                )

            if input_data.action == "preview":
                content = self._build_preview(
                    input_data.atoms or [],
                    input_data.output_format,
                )
                return ToolResult(
                    data={
                        "content": content,
                        "format": input_data.output_format
                        if input_data.output_format != "auto"
                        else self._detect_format(input_data.atoms or []),
                    },
                    success=True,
                )

            return ToolResult(
                data=None,
                success=False,
                error=f"未知 action: {input_data.action}",
            )

        except Exception as e:
            logger.warning("design_atom_tool failed: %s", e, exc_info=True)
            return ToolResult(data=None, success=False, error=str(e))

    def _render_one(self, name: str, params: dict[str, Any]) -> str:
        renderer = _RENDERERS.get(name)
        if not renderer:
            return f"<!-- 未知原子: {name} -->\n"
        # 用注册表的默认值兜底
        defaults = {
            k: v["default"]
            for k, v in _ATOM_REGISTRY[name]["params"].items()
        }
        merged = {**defaults, **(params or {})}
        return renderer(merged)

    def _detect_format(self, atoms: list[dict[str, Any]]) -> str:
        # 有 dataviz.* 就走 python, 否则 html
        for a in atoms:
            if a["atom_name"].startswith("dataviz."):
                return "python"
        return "html"

    def _build_preview(
        self,
        atoms: list[dict[str, Any]],
        output_format: str,
    ) -> str:
        fmt = output_format
        if fmt == "auto":
            fmt = self._detect_format(atoms)

        snippets = [
            self._render_one(a["atom_name"], a.get("params", {}))
            for a in atoms
        ]

        if fmt == "python":
            # 拼接所有 dataviz 代码块, 加注释分隔
            parts = [
                "# DesignAtom generated Python preview",
                "# Arial 20pt+ bold 按用户规范",
                "",
            ]
            for a, s in zip(atoms, snippets):
                parts.append(f"# === {a['atom_name']} ===")
                parts.append(s)
                parts.append("")
            return "\n".join(parts)

        # html: 包一个完整 HTML 文档, CSS 进 <style>, HTML body 进 body
        css_parts = []
        html_parts = []
        svg_parts = []
        for a, s in zip(atoms, snippets):
            cat = _ATOM_REGISTRY[a["atom_name"]]["category"]
            if cat in ("style",):
                css_parts.append(f"/* {a['atom_name']} */\n{s}")
            elif cat == "layout" and "<" in s:
                # hero 这种直接出 HTML
                html_parts.append(f"<!-- {a['atom_name']} -->\n{s}")
            elif cat == "layout":
                css_parts.append(f"/* {a['atom_name']} */\n{s}")
            elif cat == "geometry":
                svg_parts.append(f"<!-- {a['atom_name']} -->\n{s}")
            else:
                # dataviz 在 html 模式下退化成占位
                html_parts.append(
                    f"<!-- {a['atom_name']}: 需要 python 输出, 见 compose -->\n"
                )
        css_block = "\n".join(css_parts) if css_parts else "/* no css atoms */"
        html_block = "\n".join(html_parts) if html_parts else ""
        svg_block = "\n".join(svg_parts) if svg_parts else ""
        return (
            "<!DOCTYPE html>\n<html lang=\"zh-CN\">\n<head>\n"
            "<meta charset=\"UTF-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n"
            "<title>DesignAtom Preview</title>\n"
            "<style>\n"
            f"{css_block}\n"
            "</style>\n</head>\n<body>\n"
            f"{html_block}\n"
            f"{svg_block}\n"
            "</body>\n</html>\n"
        )
