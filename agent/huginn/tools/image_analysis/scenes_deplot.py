"""图表转表格 — 用 Google DePlot 把 chart 图片转成结构化表格文本.

依赖可选: transformers + torch. 没装就优雅降级, 返回提示信息而不是报错.
模块导入时只用 find_spec 探测依赖是否在, 真正的重 import 放到调用时,
避免拖慢 image_analysis 包的整体加载.
"""
from __future__ import annotations

import importlib.util
import logging
import threading
from typing import TYPE_CHECKING

from huginn.types import ToolResult

if TYPE_CHECKING:
    from huginn.tools.image_analysis.tool import ImageAnalysisInput

logger = logging.getLogger(__name__)

_DEPLOT_MODEL_ID = "google/deplot"

# 只探测是否装了 transformers / torch, 不真正 import (那俩 import 很重)
_HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None
_HAS_TORCH = importlib.util.find_spec("torch") is not None
_DEPLOT_AVAILABLE = _HAS_TRANSFORMERS and _HAS_TORCH

# processor / model 加载很慢, 加载一次后缓存复用.
# threading.Lock 保护防止并发重复加载.
_deplot_cache: dict = {"processor": None, "model": None}
_deplot_lock = threading.Lock()


def _to_markdown_table(text: str) -> str:
    """把 DePlot 输出的原始文本尽量规整成 Markdown 表格.

    DePlot 的输出格式不固定, 一般是一行表头 + 若干行数据, 列之间用
    tab / 连续空格 / 竖线分隔. 这里按行切, 再按分隔符切列, 拼成 MD 语法.
    """
    if not text or not text.strip():
        return ""
    lines = [ln.rstrip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return ""

    rows: list[list[str]] = []
    for ln in lines:
        # 优先按 tab 切, 再按 3+ 连续空格切, 最后按竖线切
        if "\t" in ln:
            cells = [c.strip() for c in ln.split("\t")]
        elif " | " in ln:
            cells = [c.strip() for c in ln.split("|")]
        else:
            # 3 个以上连续空格当列分隔
            import re
            cells = [c.strip() for c in re.split(r"\s{3,}", ln)]
        cells = [c for c in cells if c != ""]
        if cells:
            rows.append(cells)

    if not rows:
        return text.strip()

    # 对齐列数到最宽的一行
    n_cols = max(len(r) for r in rows)
    rows = [r + [""] * (n_cols - len(r)) for r in rows]

    md_lines = []
    md_lines.append("| " + " | ".join(rows[0]) + " |")
    md_lines.append("| " + " | ".join(["---"] * n_cols) + " |")
    for r in rows[1:]:
        md_lines.append("| " + " | ".join(r) + " |")
    return "\n".join(md_lines)


def _load_deplot():
    """延迟加载 DePlot 的 processor + model, 加载过就缓存."""
    # Fast path: check without lock
    if _deplot_cache["processor"] is not None and _deplot_cache["model"] is not None:
        return _deplot_cache["processor"], _deplot_cache["model"]

    with _deplot_lock:
        # Double-check after acquiring lock
        if _deplot_cache["processor"] is not None and _deplot_cache["model"] is not None:
            return _deplot_cache["processor"], _deplot_cache["model"]

        import torch  # noqa: F401  torch 要先 import, transformers 推理依赖它
        from transformers import AutoProcessor, AutoModelForVision2Seq

        logger.info("首次加载 DePlot 模型 %s, 会慢一点", _DEPLOT_MODEL_ID)
        processor = AutoProcessor.from_pretrained(_DEPLOT_MODEL_ID)
        model = AutoModelForVision2Seq.from_pretrained(_DEPLOT_MODEL_ID)
        _deplot_cache["processor"] = processor
        _deplot_cache["model"] = model
        return processor, model


def deplot_chart(args: "ImageAnalysisInput") -> ToolResult:
    """把图表图片转成结构化表格 (Markdown)."""
    image_path = args.image_path

    # 没装 transformers / torch, 直接告诉用户怎么装, 不报错
    if not _DEPLOT_AVAILABLE:
        missing = []
        if not _HAS_TORCH:
            missing.append("torch")
        if not _HAS_TRANSFORMERS:
            missing.append("transformers")
        data = {
            "summary": (
                f"DePlot 不可用: 缺少依赖 ({', '.join(missing)}). "
                f"安装后即可使用: pip install {' '.join(missing)}"
            ),
            "available": False,
            "missing_dependencies": missing,
            "model_id": _DEPLOT_MODEL_ID,
            "install_hint": f"pip install {' '.join(missing)}",
        }
        return ToolResult(data=data)

    # 真正加载模型 + 推理, 任何异常都降级成可读的提示
    try:
        from PIL import Image

        processor, model = _load_deplot()
        image = Image.open(image_path).convert("RGB")

        max_tokens = int(args.parameters.get("max_new_tokens", 512))
        inputs = processor(images=image, return_tensors="pt")
        generated_ids = model.generate(**inputs, max_new_tokens=max_tokens)
        raw_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        markdown_table = _to_markdown_table(raw_text)
        # 行列数粗估一下, 方便上层判断
        n_rows = max(raw_text.count("\n") + 1, 1)
        n_cols = 0
        for ln in raw_text.splitlines():
            if "\t" in ln:
                n_cols = max(n_cols, len(ln.split("\t")))

        summary = (
            f"DePlot 图表转表格: 成功转换, 约 {n_rows} 行 "
            f"(Markdown 表格已生成)"
        )
        data = {
            "summary": summary,
            "available": True,
            "model_id": _DEPLOT_MODEL_ID,
            "raw_text": raw_text.strip(),
            "markdown_table": markdown_table,
            "estimated_rows": n_rows,
            "estimated_columns": n_cols or (markdown_table.count("---") if markdown_table else 0),
        }
        return ToolResult(data=data)
    except Exception as exc:
        logger.warning("DePlot 推理失败: %s", exc, exc_info=True)
        data = {
            "summary": f"DePlot 推理失败: {exc}",
            "available": True,
            "model_id": _DEPLOT_MODEL_ID,
            "error": str(exc),
            "install_hint": (
                "首次使用会从 HuggingFace 下载 google/deplot 模型, "
                "确保网络可达且有足够磁盘空间."
            ),
        }
        return ToolResult(data=data)
