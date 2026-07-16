"""Plot tool — 统一画图, Arial 20pt+ 加粗.

治 ζ_plot: agent 每次重写 matplotlib 样式代码, 浪费 5-10 calls.
统一工具: loss curve / scatter / histogram / heatmap, 自动保存 outputs/.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool
from huginn.types import ToolContext, ToolResult


class PlotToolInput(BaseModel):
    action: Literal["loss_curve", "scatter", "histogram", "heatmap", "line"] = Field(
        description="Type of plot to generate"
    )
    data: str = Field(
        description="JSON-encoded data. Format depends on action: "
        "loss_curve: {\"losses\":[...], \"title\":\"...\"}; "
        "scatter: {\"x\":[...], \"y\":[...], \"xlabel\":\"...\", \"ylabel\":\"...\"}; "
        "histogram: {\"values\":[...], \"bins\":30, \"label\":\"...\"}; "
        "heatmap: {\"matrix\":[[...]], \"xlabel\":\"...\", \"ylabel\":\"...\"}; "
        "line: {\"x\":[...], \"y\":[...], \"xlabel\":\"...\", \"ylabel\":\"...\"}"
    )
    output_path: str = Field(
        default="outputs/plot.png",
        description="Output file path (relative to workspace)"
    )
    title: str = Field(default="", description="Plot title")


class PlotTool(HuginnTool):
    """Generate plots with Arial 20pt+ bold font."""

    name = "plot_tool"
    category = "analysis"
    description = (
        "Generate publication-quality plots (loss curves, scatter, histogram, heatmap). "
        "Uses Arial 20pt+ bold font. Saves to outputs/. "
        "Input data as JSON string."
    )
    destructive = False
    input_schema = PlotToolInput

    async def call(
        self, args: dict[str, Any], context: ToolContext | None = None
    ) -> ToolResult:
        input_data = PlotToolInput(**args)

        try:
            import matplotlib
            matplotlib.use("Agg")  # 无头模式
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError as e:
            return ToolResult(
                data=None, success=False,
                error=f"matplotlib not available: {e}. Run: pip install matplotlib numpy",
            )

        # Arial 20pt+ 加粗
        plt.rcParams.update({
            "font.family": "Arial",
            "font.size": 20,
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "figure.figsize": (10, 7),
            "figure.dpi": 150,
        })

        try:
            data = json.loads(input_data.data)
        except json.JSONDecodeError as e:
            return ToolResult(data=None, success=False, error=f"Invalid JSON data: {e}")

        fig, ax = plt.subplots()

        if input_data.action == "loss_curve":
            losses = data.get("losses", [])
            ax.plot(losses, linewidth=2.5, color="#2563eb")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Loss")
            if input_data.title:
                ax.set_title(input_data.title)
            ax.grid(True, alpha=0.3)

        elif input_data.action == "scatter":
            x, y = data.get("x", []), data.get("y", [])
            ax.scatter(x, y, s=50, alpha=0.7, color="#2563eb")
            ax.set_xlabel(data.get("xlabel", "X"))
            ax.set_ylabel(data.get("ylabel", "Y"))
            if input_data.title:
                ax.set_title(input_data.title)
            ax.grid(True, alpha=0.3)

        elif input_data.action == "histogram":
            values = data.get("values", [])
            bins = data.get("bins", 30)
            ax.hist(values, bins=bins, color="#2563eb", alpha=0.8, edgecolor="black")
            ax.set_xlabel(data.get("label", "Value"))
            ax.set_ylabel("Count")
            if input_data.title:
                ax.set_title(input_data.title)
            ax.grid(True, alpha=0.3)

        elif input_data.action == "heatmap":
            matrix = np.array(data.get("matrix", []))
            im = ax.imshow(matrix, cmap="viridis", aspect="auto")
            fig.colorbar(im, ax=ax)
            ax.set_xlabel(data.get("xlabel", "X"))
            ax.set_ylabel(data.get("ylabel", "Y"))
            if input_data.title:
                ax.set_title(input_data.title)

        elif input_data.action == "line":
            x, y = data.get("x", []), data.get("y", [])
            ax.plot(x, y, linewidth=2.5, color="#2563eb")
            ax.set_xlabel(data.get("xlabel", "X"))
            ax.set_ylabel(data.get("ylabel", "Y"))
            if input_data.title:
                ax.set_title(input_data.title)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存
        out_path = Path(input_data.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close(fig)

        return ToolResult(
            data={
                "path": str(out_path),
                "size": out_path.stat().st_size,
                "action": input_data.action,
                "font": "Arial 20pt bold",
            },
            success=True,
            side_effects=[str(out_path)],
        )


if __name__ == "__main__":
    # self-check: 画一个 loss curve 验证字体
    import asyncio
    import tempfile

    async def _test():
        tool = PlotTool()
        with tempfile.TemporaryDirectory() as tmp:
            data = json.dumps({"losses": [2.0, 1.5, 1.0, 0.7, 0.5, 0.3]})
            result = await tool.call({
                "action": "loss_curve",
                "data": data,
                "output_path": f"{tmp}/test.png",
                "title": "Test Loss",
            })
            print(result.data)
            assert Path(f"{tmp}/test.png").exists()
            print("[plot_tool] self-check OK")

    asyncio.run(_test())
