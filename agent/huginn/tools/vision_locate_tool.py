"""VisionLocate — 图像目标定位工具 (ground/detect/crop).

沉淀自 agent-vision-toolkit 的 ground/detect/crop 思路: 让 agent 在截图/图表/
显微图上定位目标, 拿回原始像素坐标的边界框. 面向 GUI 自动化 / 元素清单 /
裁剪复用场景.

实现是纯编排层, 后端复用 vision.local_decoder.decode_image — 同一套
云端/本地视觉解码, 零新依赖. 把"只输出 JSON"的指令当 question 传给
多模态模型, 然后解析成结构化 bbox. 解析失败/解码不可用时返回 success=False,
让 agent 知道没拿到坐标而不是瞎猜.

接入点:
  - ToolRegistry 注册为 "vision_locate" (huginn/tools/__init__.py _OPTIONAL_MODULES)
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.core_types import ToolContext, ToolResult, ValidationResult
from huginn.tools.base import HuginnTool

logger = logging.getLogger(__name__)


# ── 纯解析函数 (无 IO, 可单测) ──────────────────────────────────────

def _extract_json(text: str) -> Any | None:
    """从视觉模型回复里抽出第一个 JSON 对象/数组.

    容忍 markdown 代码围栏和前后杂文. 找不到返回 None.
    """
    if not text:
        return None
    cleaned = text.strip()
    # 去 ```json ... ``` 围栏
    cleaned = re.sub(r"```(?:json)?\s*", "", cleaned).strip()
    # 定位第一个 { 或 [ 到末尾, 尝试解析; 逐段收紧
    for start_ch, _end_ch in (("{", "}"), ("[", "]")):
        start = cleaned.find(start_ch)
        if start == -1:
            continue
        for end in range(len(cleaned), start, -1):
            try:
                return json.loads(cleaned[start:end])
            except (json.JSONDecodeError, ValueError):
                continue
    return None


def _as_int(value: Any) -> int | None:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def _clamp_box(box: dict[str, Any], width: int, height: int) -> dict[str, int] | None:
    """把 {x1,y1,x2,y2} 夹紧到图片范围, 并归一化 min/max 顺序."""
    x1, y1 = _as_int(box.get("x1")), _as_int(box.get("y1"))
    x2, y2 = _as_int(box.get("x2")), _as_int(box.get("y2"))
    if None in (x1, y1, x2, y2):
        return None
    lo_x, hi_x = sorted((x1, x2))
    lo_y, hi_y = sorted((y1, y2))
    return {
        "x1": max(0, min(lo_x, width)),
        "y1": max(0, min(lo_y, height)),
        "x2": max(0, min(hi_x, width)),
        "y2": max(0, min(hi_y, height)),
    }


def _parse_grounding(text: str, width: int, height: int) -> dict[str, int] | None:
    """解析 ground 结果 → 单框 {x1,y1,x2,y2}."""
    obj = _extract_json(text)
    if not isinstance(obj, dict):
        return None
    # 模型有时把元素数组或 {elements:[...]} 一起返回, 取第一个框
    elements = obj.get("elements")
    if isinstance(elements, list) and elements:
        obj = elements[0] if isinstance(elements[0], dict) else {}
    return _clamp_box(obj, width, height)


def _parse_detect(text: str, width: int, height: int) -> list[dict[str, Any]]:
    """解析 detect 结果 → 元素清单 [{text, x1,y1,x2,y2}, ...]."""
    obj = _extract_json(text)
    if isinstance(obj, dict):
        obj = obj.get("elements")
    if not isinstance(obj, list):
        return []
    out: list[dict[str, Any]] = []
    for item in obj:
        if not isinstance(item, dict):
            continue
        box = _clamp_box(item, width, height)
        if box is None:
            continue
        out.append({
            "text": str(item.get("text", "") or ""),
            **box,
        })
    return out


# ── 提示词 ──────────────────────────────────────────────────────────

_GROUND_PROMPT = (
    '请定位图片中"{target}"的边界框。只输出一个 JSON 对象, 格式: '
    '{{"x1": 左, "y1": 上, "x2": 右, "y2": 下}}, 单位是像素, 相对图片原始尺寸。'
    "不要输出其他文字或解释。"
)

_DETECT_PROMPT = (
    '请列出图片中所有与"{target}"相关的元素。只输出 JSON: '
    '{{"elements": [{{"text": "可见文本或元素描述", "x1": 左, "y1": 上, '
    '"x2": 右, "y2": 下}}]}}, 单位是像素, 相对图片原始尺寸。'
    "不要输出其他文字。"
)


# ── 工具 ────────────────────────────────────────────────────────────

class VisionLocateInput(BaseModel):
    image_path: str = Field(..., description="图像文件路径")
    target: str = Field(
        ..., description="要定位的目标, 如 '发送按钮' / '主峰' / '标题栏'"
    )
    action: Literal["ground", "detect", "crop"] = Field(
        default="ground",
        description=(
            "ground: 定位单个目标返回一个 bbox; "
            "detect: 列出所有相关元素带各自 bbox; "
            "crop: 按定位框裁剪出目标区域存成新文件"
        ),
    )
    output_path: str | None = Field(
        default=None,
        description="action=crop 时的输出文件路径 (如 send-button.png)",
    )


class VisionLocateTool(HuginnTool):
    """在图片上定位目标, 返回像素坐标边界框.

    复用本地/云端视觉解码器读图, 让文本模型获得"目标在哪"的确定信息,
    供 GUI 自动化 / 元素清单 / 裁剪复用使用. 返回原始像素坐标.
    """

    name = "vision_locate"
    category = "cv"
    description = (
        "Locate a target in an image and return its pixel bounding box "
        "(ground), list all matching elements (detect), or crop the located "
        "region to a file (crop). Uses the vision decoder — works with "
        "text-only models. Returns {x1,y1,x2,y2} in original pixel coordinates."
    )
    input_schema = VisionLocateInput
    read_only = True  # ground/detect 只读; crop 会写文件, is_read_only 里细分

    def is_read_only(self, args: VisionLocateInput) -> bool:
        action = getattr(args, "action", "ground")
        return action != "crop"

    async def validate_input(
        self, args: dict[str, Any], context: ToolContext | None = None
    ) -> ValidationResult:
        input_data = args if isinstance(args, VisionLocateInput) else VisionLocateInput(**args)
        if not Path(input_data.image_path).exists():
            return ValidationResult(
                result=False, message=f"图片不存在: {input_data.image_path}"
            )
        if input_data.action == "crop" and not input_data.output_path:
            return ValidationResult(
                result=False, message="action=crop 时必须提供 output_path"
            )
        return ValidationResult(result=True)

    async def call(
        self, args: dict[str, Any], context: ToolContext
    ) -> ToolResult:
        input_data = args if isinstance(args, VisionLocateInput) else VisionLocateInput(**args)
        try:
            from huginn.vision.local_decoder import decode_image

            path = Path(input_data.image_path)
            # 拿图片尺寸用于坐标夹紧
            try:
                from PIL import Image
                with Image.open(path) as img:
                    width, height = img.size
            except Exception as exc:
                return ToolResult(
                    data=None, success=False,
                    error=f"读取图片尺寸失败: {exc}",
                )

            if input_data.action == "detect":
                raw = decode_image(
                    path, question=_DETECT_PROMPT.format(target=input_data.target)
                )
                if not raw:
                    return ToolResult(
                        data=None, success=False,
                        error="视觉解码不可用或未返回内容 (检查 HUGINN_VISION_PROVIDER / 本地 ollama 视觉模型)",
                    )
                elements = _parse_detect(raw, width, height)
                if not elements:
                    return ToolResult(
                        data={"raw": raw[:400]}, success=False,
                        error="未能从模型回复解析出元素清单",
                    )
                return ToolResult(data={"elements": elements, "size": [width, height]}, success=True)

            # ground 与 crop 共用同一套"定位单框"逻辑
            raw = decode_image(
                path, question=_GROUND_PROMPT.format(target=input_data.target)
            )
            if not raw:
                return ToolResult(
                    data=None, success=False,
                    error="视觉解码不可用或未返回内容 (检查 HUGINN_VISION_PROVIDER / 本地 ollama 视觉模型)",
                )
            box = _parse_grounding(raw, width, height)
            if box is None:
                return ToolResult(
                    data={"raw": raw[:400]}, success=False,
                    error="未能从模型回复解析出边界框",
                )

            if input_data.action == "crop":
                try:
                    from PIL import Image as _PILImage
                    with _PILImage.open(path) as img:
                        img.convert("RGB").crop(
                            (box["x1"], box["y1"], box["x2"], box["y2"])
                        ).save(input_data.output_path)
                except Exception as exc:
                    return ToolResult(
                        data={"box": box}, success=False,
                        error=f"裁剪保存失败: {exc}",
                    )
                return ToolResult(
                    data={"box": box, "output_path": input_data.output_path},
                    success=True,
                )

            return ToolResult(data={"box": box, "size": [width, height]}, success=True)
        except Exception as exc:
            logger.warning("vision_locate failed: %s", exc, exc_info=True)
            return ToolResult(data=None, success=False, error=str(exc))


# ── self-check (assert-based, 无框架) ────────────────────────

def _selfcheck() -> None:
    """校验解析层的纯函数: JSON 抽取 / 夹紧 / ground 与 detect 解析."""
    # 1. _extract_json 容忍 markdown 围栏 + 前后杂文
    assert _extract_json('```json\n{"x1": 1, "y1": 2, "x2": 3, "y2": 4}\n```') == {
        "x1": 1, "y1": 2, "x2": 3, "y2": 4,
    }
    assert _extract_json('结果如下 {"a": 1} 完毕') == {"a": 1}
    assert _extract_json("没有 JSON") is None

    # 2. _clamp_box: 越界夹紧 + 归一化 min/max 顺序
    box = _clamp_box({"x1": 50, "y1": 80, "x2": 20, "y2": 30}, 100, 100)
    assert box == {"x1": 20, "y1": 30, "x2": 50, "y2": 80}
    assert _clamp_box({"x1": -5, "y1": 0, "x2": 999, "y2": 999}, 100, 100) == {
        "x1": 0, "y1": 0, "x2": 100, "y2": 100,
    }
    # 非法字段 → None
    assert _clamp_box({"x1": "a", "y1": 0, "x2": 1, "y2": 2}, 100, 100) is None

    # 3. _parse_grounding: 直接对象 / 包在 elements 里都能解
    g1 = _parse_grounding('{"x1":10,"y1":20,"x2":30,"y2":40}', 200, 200)
    assert g1 == {"x1": 10, "y1": 20, "x2": 30, "y2": 40}
    g2 = _parse_grounding('{"elements":[{"x1":10,"y1":20,"x2":30,"y2":40}]}', 200, 200)
    assert g2 == {"x1": 10, "y1": 20, "x2": 30, "y2": 40}
    assert _parse_grounding("无法定位", 200, 200) is None

    # 4. _parse_detect: 多元素清单, 非法条目跳过
    det = _parse_detect(
        '{"elements":[{"text":"按钮A","x1":1,"y1":2,"x2":3,"y2":4},'
        '{"text":"按钮B","x1":5,"y1":6,"x2":7,"y2":8}]}',
        100, 100,
    )
    assert len(det) == 2
    assert det[0]["text"] == "按钮A"
    assert det[1]["y2"] == 8
    assert _parse_detect("{}", 100, 100) == []

    # 5. 工具元数据
    tool = VisionLocateTool()
    assert tool.name == "vision_locate"
    assert tool.category == "cv"
    assert tool.is_read_only(VisionLocateInput(image_path="x", target="y")) is True
    assert tool.is_read_only(
        VisionLocateInput(image_path="x", target="y", action="crop", output_path="o.png")
    ) is False

    print("all self-checks passed (vision_locate parsing OK)")


if __name__ == "__main__":
    _selfcheck()
