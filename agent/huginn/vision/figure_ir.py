"""统一图表 IR + render — MLIR 启发, 偷形不偷魂.

MLIR 的 dialect/lowering 框架太重, 但"统一中间表示 + 多后端渲染"的形可以偷.
这里只做 dict + render 函数对: spec → IR → 选后端 → PNG/HTML.

IR dict 结构:
    {
        "chart_type": "line"|"bar"|"scatter"|"heatmap",
        "title": "...",
        "axes": {"x": "k-path", "y": "Energy (eV)"},
        "series": [{"label": "Si", "data": [...], "color": "#1f77b4"}],
        "style": "science"|"ieee"|"nature",
    }

后端:
    scienceplots: SciencePlots + matplotlib, 出刊级静态图 (默认)
    ultraplot: UltraPlot, 复杂多面板
    flint: 生成 Vega-Lite JSON, 交互式 HTML

样式规范固化在 lowering 层 (Arial 20pt 加粗, science/ieee 主题), 用户只声明数据
和想要的"形", 不重复写样式.

ponytail: 先实现 SciencePlots 后端, ultraplot/flint 留接口. 升级: dialect 框架.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def to_ir(
    data: dict[str, Any],
    chart_type: str = "line",
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    style: str = "science",
    series: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """把 spec 编译成统一 IR dict.

    Args:
        data: 原始数据, key 是 series 名, value 是数据点列表
              (1D: [y1, y2, ...] 或 2D: [[x1,y1], [x2,y2], ...])
        chart_type: line | bar | scatter | heatmap
        title / x_label / y_label: 图表标注
        style: science | ieee | nature (SciencePlots 主题)
        series: 可选, 显式指定 series 顺序 / 颜色 / 标签

    Returns:
        统一 IR dict, 可传给 render()
    """
    if series is None:
        series = []
        for label, values in data.items():
            series.append({"label": label, "data": values})

    return {
        "chart_type": chart_type,
        "title": title,
        "axes": {"x": x_label, "y": y_label},
        "series": series,
        "style": style,
    }


def render(
    ir: dict[str, Any],
    backend: str = "scienceplots",
    output_path: str | Path | None = None,
    **kwargs: Any,
) -> str:
    """渲染 IR dict 到图像文件, 返回路径.

    Args:
        ir: to_ir() 产出的 IR dict
        backend: scienceplots | ultraplot | flint
        output_path: 输出路径. None 则放 ~/.huginn/figures/ 下
        **kwargs: 透传给后端 (dpi / figsize 等)

    Returns:
        输出文件路径
    """
    if backend == "scienceplots":
        return _render_scienceplots(ir, output_path, **kwargs)
    elif backend == "ultraplot":
        return _render_ultraplot(ir, output_path, **kwargs)
    elif backend == "flint":
        return _render_flint(ir, output_path, **kwargs)
    else:
        logger.warning("unknown backend %s, fallback to scienceplots", backend)
        return _render_scienceplots(ir, output_path, **kwargs)


def _resolve_output_path(output_path: str | Path | None, ext: str) -> Path:
    """确定输出路径, 默认放 ~/.huginn/figures/."""
    if output_path is not None:
        p = Path(output_path)
    else:
        try:
            from huginn.utils.runtime import get_runtime_home
            fig_dir = get_runtime_home() / "figures"
        except Exception:
            fig_dir = Path.home() / ".huginn" / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        p = fig_dir / f"figure_{ts}.{ext}"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _apply_style(style: str) -> list[str]:
    """返回 SciencePlots 主题列表, 固化样式规范.

    用户 user_profile 要求: Arial 20pt 加粗, science/ieee 主题.
    ponytail: 固化在 lowering 层, 用户不重复写. 升级: 用户可覆盖.
    """
    styles = ["science"]
    if style == "ieee":
        styles.append("ieee")
    elif style == "nature":
        styles.append("nature")
    return styles


def _render_scienceplots(
    ir: dict[str, Any], output_path: str | Path | None, **kwargs: Any
) -> str:
    """SciencePlots + matplotlib 后端 — 出刊级静态图.

    样式规范 (用户 user_profile 要求):
    - 字体: Arial 20pt 加粗
    - 主题: SciencePlots science / ieee
    ponytail: 用 plt.style.context, 不全局改 rcParams. 升级: 自定义 rcParams.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise RuntimeError("matplotlib not available, install: pip install matplotlib")

    try:
        import scienceplots  # noqa: F401
    except ImportError:
        logger.debug("SciencePlots not available, using default matplotlib style")

    styles = _apply_style(ir.get("style", "science"))
    figsize = kwargs.get("figsize", (6, 4))
    dpi = kwargs.get("dpi", 150)

    with plt.style.context(styles):
        # 固化样式: Arial 20pt 加粗
        plt.rcParams["font.family"] = "Arial"
        plt.rcParams["font.size"] = 20
        plt.rcParams["font.weight"] = "bold"
        plt.rcParams["axes.labelweight"] = "bold"
        plt.rcParams["axes.titleweight"] = "bold"
        # SciencePlots science 主题默认 usetex=True, 需要完整 LaTeX 环境.
        # TinyTeX 常缺 type1cm.sty, 关掉用 matplotlib 内置渲染.
        # ponytail: 关 usetex, 字体降级到 DejaVu Sans (Arial 找不到时). 升级: 装完整 LaTeX.
        plt.rcParams["text.usetex"] = False

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        chart_type = ir.get("chart_type", "line")
        series_list = ir.get("series", [])
        axes = ir.get("axes", {})

        for s in series_list:
            label = s.get("label", "")
            data = s.get("data", [])
            color = s.get("color")
            if not data:
                continue
            if chart_type == "line":
                if isinstance(data[0], (list, tuple)) and len(data[0]) >= 2:
                    xs = [p[0] for p in data]
                    ys = [p[1] for p in data]
                    ax.plot(xs, ys, label=label, color=color, linewidth=2)
                else:
                    ax.plot(data, label=label, color=color, linewidth=2)
            elif chart_type == "bar":
                if isinstance(data[0], (list, tuple)) and len(data[0]) >= 2:
                    labels = [str(p[0]) for p in data]
                    values = [p[1] for p in data]
                    ax.bar(labels, values, label=label, color=color)
                else:
                    ax.bar(range(len(data)), data, label=label, color=color)
            elif chart_type == "scatter":
                if isinstance(data[0], (list, tuple)) and len(data[0]) >= 2:
                    xs = [p[0] for p in data]
                    ys = [p[1] for p in data]
                    ax.scatter(xs, ys, label=label, color=color, s=50)
                else:
                    ax.scatter(range(len(data)), data, label=label, color=color, s=50)
            elif chart_type == "heatmap":
                try:
                    import numpy as np
                    arr = np.array(data)
                    im = ax.imshow(arr, cmap=color or "viridis", aspect="auto")
                    fig.colorbar(im, ax=ax, label=label)
                except Exception:
                    logger.debug("heatmap render failed", exc_info=True)

        if ir.get("title"):
            ax.set_title(ir["title"])
        if axes.get("x"):
            ax.set_xlabel(axes["x"])
        if axes.get("y"):
            ax.set_ylabel(axes["y"])
        if len(series_list) > 1:
            ax.legend()

        fig.tight_layout()
        out = _resolve_output_path(output_path, "png")
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return str(out)


def _render_ultraplot(
    ir: dict[str, Any], output_path: str | Path | None, **kwargs: Any
) -> str:
    """UltraPlot 后端 — 复杂多面板图.

    ponytail: stub, 先检测 ultraplot 装了没. 升级: 多面板布局.
    """
    try:
        import ultraplot as uplt  # noqa: F401
    except ImportError:
        logger.warning("ultraplot not available, fallback to scienceplots")
        return _render_scienceplots(ir, output_path, **kwargs)

    # ultraplot API 类似 matplotlib, 复用 scienceplots 逻辑
    # 多面板布局是 ultraplot 的强项, 但当前 IR 没有多面板 spec
    # ponytail: 先退化成单面板, 等用户需要多面板时再扩展 IR
    return _render_scienceplots(ir, output_path, **kwargs)


def _render_flint(
    ir: dict[str, Any], output_path: str | Path | None, **kwargs: Any
) -> str:
    """Flint 后端 — 生成 Vega-Lite JSON, 交互式 HTML.

    ponytail: 只生成 Vega-Lite spec JSON 文件, 不调 Flint MCP (MCP 调用留给 agent).
    升级: 直接调 flint-chart-mcp 渲染.
    """
    chart_type = ir.get("chart_type", "line")
    series_list = ir.get("series", [])
    axes = ir.get("axes", {})

    # IR → Vega-Lite spec lowering
    vl_type = {"line": "line", "bar": "bar", "scatter": "point", "heatmap": "rect"}.get(
        chart_type, "line"
    )

    # 把所有 series 合并成长格式 data
    values: list[dict[str, Any]] = []
    for s in series_list:
        label = s.get("label", "series")
        data = s.get("data", [])
        for i, v in enumerate(data):
            if isinstance(v, (list, tuple)) and len(v) >= 2:
                values.append({"x": v[0], "y": v[1], "series": label})
            else:
                values.append({"x": i, "y": v, "series": label})

    vl_spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "description": ir.get("title", ""),
        "data": {"values": values},
        "mark": vl_type,
        "encoding": {
            "x": {"field": "x", "type": "quantitative", "title": axes.get("x", "")},
            "y": {"field": "y", "type": "quantitative", "title": axes.get("y", "")},
            "color": {"field": "series", "type": "nominal"} if len(series_list) > 1 else {},
        },
    }

    out = _resolve_output_path(output_path, "json")
    Path(out).write_text(json.dumps(vl_spec, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(out)


def ir_to_structured(ir: dict[str, Any]) -> dict[str, Any]:
    """IR → 结构化描述 (Nullmax 启发: 多模态输出必须可被解释).

    让 agent 能精确引用 chart_type / series / axes 等字段做推理,
    不用从渲染后的图像里重新提取.
    """
    return {
        "chart_type": ir.get("chart_type", "unknown"),
        "title": ir.get("title", ""),
        "axes": ir.get("axes", {}),
        "n_series": len(ir.get("series", [])),
        "series_labels": [s.get("label", "") for s in ir.get("series", [])],
        "style": ir.get("style", "science"),
    }
