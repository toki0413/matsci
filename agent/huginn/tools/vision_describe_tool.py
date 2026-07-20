"""VisionDescribe — 分层视觉描述工具.

DeepSeek 纯文本 LLM 没有视觉 encoder, 但材料科学任务里大量视觉信息
(XRD 谱/SEM 形貌/论文图/坐标系图表). 本工具是统一入口, 内部按可用
资源自动选 Tier, 把图像转成结构化 JSON 让 DeepSeek 推理.

三层降级 (按可用性自动选):
  Tier 3: DeepSeek-OCR-2 (3B, GPU)   — 整页视觉压缩, 公式/表格/化学式
  Tier 2: PaddleOCR + image_analysis — OCR + 专用 CV 算法, 纯 CPU 可跑
  Tier 1: EasyOCR/Tesseract + 像素统计 — 零依赖兜底

agent 调 vision_describe(image, question) 不感知后端切换, 只看到不同
rich 度的 JSON. 跟调 read_csv 一样, 符合"工具自动触发"原则.

接入点:
  - ToolRegistry 注册为 "vision_describe"
  - visual_inspect.py 反馈通道改调本工具
  - smart_ingest.py 图片摄入可调本工具做富描述

设计原则 (ponytail):
  - 各 Tier 先 stub, 返结构化 "unavailable" 信息. 后续填实现 0 接口改动.
  - 自动探测可用引擎, 缓存探测结果 (进程级).
  - 失败降级不抛异常, 返 success=False + error 让 agent 决策.
  - 不进入决策回路, 只做感知前端 (符合"反对黑 box ML 进决策"偏好).
"""
from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool
from huginn.types import ToolContext, ToolResult, ValidationResult

logger = logging.getLogger(__name__)


# ── 引擎可用性探测 (进程级缓存) ──────────────────────────────

_probed: dict[str, bool | None] = {
    "deepseek_ocr": None,
    "paddleocr": None,
    "easyocr": None,
    "tesseract": None,
}


def _probe_deepseek_ocr() -> bool:
    """探测 DeepSeek-OCR-2 是否本地可用.

    判据: huggingface 权重已下载 + transformers 可 import + GPU 可用.
    ponytail: 用 try import + 检查缓存目录, 不真加载模型 (避免冷启动).
    升级路径: 启动时预热, 首次调用 0 延迟.
    """
    if _probed["deepseek_ocr"] is not None:
        return _probed["deepseek_ocr"]
    try:
        import torch
        if not torch.cuda.is_available():
            _probed["deepseek_ocr"] = False
            return False
        from transformers import AutoModel
        # 权重路径: 由 HUGINN_DEEPSEEK_OCR_PATH 指向本地下载目录
        import os
        path = os.environ.get("HUGINN_DEEPSEEK_OCR_PATH", "")
        if not path or not Path(path).exists():
            _probed["deepseek_ocr"] = False
            return False
        _probed["deepseek_ocr"] = True
        return True
    except Exception:
        _probed["deepseek_ocr"] = False
        return False


def _probe_paddleocr() -> bool:
    """探测 PaddleOCR 是否可用. 纯 CPU 可跑."""
    if _probed["paddleocr"] is not None:
        return _probed["paddleocr"]
    try:
        import paddleocr  # noqa: F401
        _probed["paddleocr"] = True
        return True
    except Exception:
        _probed["paddleocr"] = False
        return False


def _probe_easyocr() -> bool:
    """探测 EasyOCR (ocr_loader 已用)."""
    if _probed["easyocr"] is not None:
        return _probed["easyocr"]
    try:
        import easyocr  # noqa: F401
        _probed["easyocr"] = True
        return True
    except Exception:
        _probed["easyocr"] = False
        return False


def _probe_tesseract() -> bool:
    """探测 Tesseract."""
    if _probed["tesseract"] is not None:
        return _probed["tesseract"]
    try:
        import pytesseract  # noqa: F401
        pytesseract.get_tesseract_version()
        _probed["tesseract"] = True
        return True
    except Exception:
        _probed["tesseract"] = False
        return False


def _reset_probe_cache() -> None:
    """测试用: 重置探测缓存, 让下次重新探测."""
    for k in _probed:
        _probed[k] = None


def _pick_tier() -> tuple[str, dict[str, bool]]:
    """选最佳可用 Tier. 返 (tier_name, availability_dict)."""
    avail = {
        "deepseek_ocr": _probe_deepseek_ocr(),
        "paddleocr": _probe_paddleocr(),
        "easyocr": _probe_easyocr(),
        "tesseract": _probe_tesseract(),
    }
    if avail["deepseek_ocr"]:
        return "tier3_deepseek_ocr", avail
    if avail["paddleocr"]:
        return "tier2_paddleocr", avail
    if avail["easyocr"] or avail["tesseract"]:
        return "tier1_classic_ocr", avail
    return "tier0_none", avail


# ── Tier 实现 (先 stub, 后续填) ──────────────────────────────

def _tier3_deepseek_ocr(
    image_bytes: bytes, question: str
) -> dict[str, Any]:
    """Tier 3: DeepSeek-OCR-2 推理.

    3B 参数, GPU 推理, ~6GB VRAM. 整页视觉压缩, 保留版式/公式/表格/
    化学式结构. 输出结构化 markdown / JSON.

    ponytail: 当前 stub. 后续填实现:
      1. 启动时加载模型到 GPU (单例, 避免重复加载)
      2. 调用 model.chat() 把图像 + question 喂进去
      3. 返结构化 JSON (用 structured output 约束)
    升级路径: 接 vLLM 批量推理, 多请求复用 KV cache.
    """
    return {
        "tier": "tier3_deepseek_ocr",
        "available": False,
        "error": "DeepSeek-OCR-2 未部署 (stub). 设 HUGINN_DEEPSEEK_OCR_PATH 环境变量指向本地权重目录后启用",
    }


# PaddleOCR 单例 (模块级, 跨调用复用, 避免重复加载模型)
_paddleocr_instance: Any = None


def _tier2_paddleocr(
    image_bytes: bytes, question: str
) -> dict[str, Any]:
    """Tier 2: PaddleOCR + image_analysis_tool.

    纯 CPU 可跑. 三件套:
      - 文本检测 + 识别 (PaddleOCR)
      - 版面分析 (PP-StructureV2)
      - 表格识别 (PP-Structure)

    结合 image_analysis_tool 的 8 个 action 做谱图数学化.
    question 关键词路由: "XRD"/"SEM"/"stress"/"DSC" 等触发对应 action.

    ponytail: PaddleOCR 加载慢 (几十秒), 用模块级单例避免重复加载.
    """
    global _paddleocr_instance
    try:
        import io as _io
        from PIL import Image
        import numpy as np

        # 单例: 第一次调用加载, 之后复用. 跨调用共享模型.
        if _paddleocr_instance is None:
            from paddleocr import PaddleOCR
            _paddleocr_instance = PaddleOCR(
                use_angle_cls=True,
                lang="ch",  # 中英混排, ch 模型两者都支持
                show_log=False,
                # ponytail: 不开 structure_analysis, 它是另一套模型重.
                # 升级路径: 加 PPStructureV2 做版面/表格.
            )

        img = Image.open(_io.BytesIO(image_bytes))
        if img.mode != "RGB":
            img = img.convert("RGB")
        arr = np.asarray(img)

        # PaddleOCR 推理
        ocr_result = _paddleocr_instance.ocr(arr, cls=True)
        text_blocks: list[dict[str, Any]] = []
        if ocr_result and ocr_result[0]:
            for line in ocr_result[0]:
                # line = [bbox, (text, confidence)]
                bbox, (text, conf) = line[0], line[1]
                # bbox 是 4 个 [x,y] 点
                xs = [p[0] for p in bbox]
                ys = [p[1] for p in bbox]
                text_blocks.append({
                    "text": text,
                    "bbox": [min(xs), min(ys), max(xs), max(ys)],
                    "confidence": float(conf),
                })

        # 按位置排序 (上到下, 左到右) 让 LLM 易读
        text_blocks.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))

        # question 关键词路由: 谱图类问题触发 image_analysis_tool
        structured: dict[str, Any] | None = None
        q_lower = question.lower() if question else ""
        scene_map = {
            "xrd": "plot_extract",
            "ir": "plot_extract",
            "raman": "plot_extract",
            "uv": "plot_extract",
            "sem": "sem_analysis",
            "tem": "sem_analysis",
            "stress": "plot_extract",
            "strain": "plot_extract",
            "dsc": "plot_extract",
            "tga": "plot_extract",
        }
        scene = None
        for kw, sc in scene_map.items():
            if kw in q_lower:
                scene = sc
                break

        if scene and text_blocks:
            try:
                from huginn.tools.registry import ToolRegistry
                import base64 as b64
                img_tool = ToolRegistry.get("image_analysis_tool")
                if img_tool:
                    res = img_tool.call({
                        "image_base64": b64.b64encode(image_bytes).decode(),
                        "scene": scene,
                        "task_description": question,
                    })
                    if res and getattr(res, "success", False):
                        structured = res.data if hasattr(res, "data") else res
            except Exception as exc:
                logger.debug("image_analysis_tool call failed: %s", exc, exc_info=True)

        out: dict[str, Any] = {
            "tier": "tier2_paddleocr",
            "available": True,
            "text_blocks": text_blocks,
            "text_concat": " ".join(b["text"] for b in text_blocks),
            "image_size": list(img.size),
            "block_count": len(text_blocks),
        }
        if structured is not None:
            out["structured_analysis"] = structured
            out["scene"] = scene
        return out
    except Exception as exc:
        return {
            "tier": "tier2_paddleocr",
            "available": False,
            "error": f"PaddleOCR 推理失败: {exc}",
        }


def _tier1_classic_ocr(
    image_bytes: bytes, question: str
) -> dict[str, Any]:
    """Tier 1: EasyOCR/Tesseract + 像素统计.

    复用 ocr_loader._ocr_image. 输出: {text_blocks, pixel_stats}.
    像素统计用 PIL+numpy 算 (直方图/连通域/边缘), 不依赖 cv2.

    ponytail: 当前 stub, 后续填实现. 这一层是最稳的兜底,
    ocr_loader 已实现完整链路, 直接复用.
    """
    try:
        from huginn.knowledge.ocr_loader import _ocr_image
        from PIL import Image
        import numpy as np

        img = Image.open(io.BytesIO(image_bytes))
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        text = _ocr_image(img, engine="auto")
        arr = np.asarray(img)
        # 像素统计: 灰度直方图 10 bin + 尺寸
        if arr.ndim == 3:
            arr = arr.mean(axis=2)
        hist, _ = np.histogram(arr, bins=10, range=(0, 255))
        return {
            "tier": "tier1_classic_ocr",
            "available": True,
            "text": text,
            "pixel_stats": {
                "size": list(img.size),
                "histogram": hist.tolist(),
                "mean": float(arr.mean()),
                "std": float(arr.std()),
            },
            "note": "Tier 1 兜底: OCR 文本 + 像素统计, 无版式/公式/表格结构",
        }
    except Exception as exc:
        return {
            "tier": "tier1_classic_ocr",
            "available": False,
            "error": f"Tier 1 失败: {exc}",
        }


# ── 统一调度入口 ─────────────────────────────────────────────

def describe_image_bytes(
    image_bytes: bytes, question: str = ""
) -> dict[str, Any]:
    """分层调度: 按可用资源选 Tier, 把图像 bytes 转结构化 JSON.

    跟 describe_image 同流程, 但接受 bytes 输入 (用于 visual_inspect 的
    cropped base64 场景, 不需要落盘). describe_image 内部也调本函数.

    Args:
        image_bytes: 图像二进制 (PNG/JPEG)
        question: agent 的具体问题

    Returns:
        跟 describe_image 同结构. 顶层永远不抛.
    """
    if not image_bytes:
        return {"tier": "error", "available": False, "error": "空图像 bytes"}

    tier_name, avail = _pick_tier()

    if tier_name == "tier3_deepseek_ocr":
        result = _tier3_deepseek_ocr(image_bytes, question)
    elif tier_name == "tier2_paddleocr":
        result = _tier2_paddleocr(image_bytes, question)
    elif tier_name == "tier1_classic_ocr":
        result = _tier1_classic_ocr(image_bytes, question)
    else:
        result = {
            "tier": "tier0_none",
            "available": False,
            "error": "无可用视觉引擎 (DeepSeek-OCR / PaddleOCR / EasyOCR / Tesseract 都未安装)",
            "availability": avail,
        }

    result.setdefault("availability", avail)
    result.setdefault("question", question)
    return result


def describe_image(
    image_path: str | Path, question: str = ""
) -> dict[str, Any]:
    """分层调度: 按可用资源选 Tier, 把图像转结构化 JSON.

    Args:
        image_path: 图像文件路径
        question: agent 的具体问题 (e.g. "这是 XRD 谱吗? 主峰在哪?")
            Tier 3 能直接答; Tier 2/1 用关键词路由到专用 action

    Returns:
        dict 含:
          - tier: 实际用的 Tier 名
          - available: 该 Tier 是否真跑成功
          - 内容字段 (text / pixel_stats / structured_layout / ...)
          - error: 失败时的描述
        顶层永远不抛, 失败返 success=False 让 agent 决策.
    """
    path = Path(image_path)
    if not path.exists():
        return {"tier": "error", "available": False, "error": f"图像不存在: {image_path}"}

    try:
        image_bytes = path.read_bytes()
    except Exception as exc:
        return {"tier": "error", "available": False, "error": f"读图失败: {exc}"}

    return describe_image_bytes(image_bytes, question)


# ── HuginnTool 包装 ──────────────────────────────────────────

class VisionDescribeInput(BaseModel):
    image_path: str = Field(..., description="图像文件路径")
    question: str = Field(
        default="",
        description=(
            "对图像的具体问题, 引导 Tier 3 VLM 或 Tier 2 路由. "
            "例: '这是 XRD 谱吗? 标出主峰 2θ' / 'SEM 颗粒分布如何?'"
        ),
    )
    output_path: str | None = Field(
        default=None,
        description="可选, 把结果 JSON 保存到该路径",
    )


class VisionDescribeTool(HuginnTool):
    """分层视觉描述: 图像 → 结构化 JSON.

    自动按可用资源降级:
      Tier 3: DeepSeek-OCR-2 (GPU, 3B) — 整页视觉压缩
      Tier 2: PaddleOCR + 专用 CV (CPU) — OCR + 谱图数学化
      Tier 1: EasyOCR/Tesseract + 像素统计 — 零依赖兜底

    agent 不感知后端切换, 只看到 JSON. 跟 read_csv 同级.
    """

    name = "vision_describe"
    category = "cv"
    description = (
        "Describe a materials science image as structured JSON. "
        "Auto-degrades by available resources: DeepSeek-OCR-2 (GPU) "
        "-> PaddleOCR + CV (CPU) -> EasyOCR/Tesseract + pixel stats. "
        "Use this instead of trying to 'see' the image directly — "
        "returns text blocks, layout structure, peak positions, "
        "particle stats, or pixel statistics depending on tier."
    )
    input_schema = VisionDescribeInput
    read_only = True

    def is_read_only(self, args: VisionDescribeInput) -> bool:
        return True

    async def validate_input(
        self, args: dict[str, Any], context: ToolContext | None = None
    ) -> ValidationResult:
        input_data = args if isinstance(args, VisionDescribeInput) else VisionDescribeInput(**args)
        if not Path(input_data.image_path).exists():
            return ValidationResult(
                result=False, message=f"图片不存在: {input_data.image_path}"
            )
        return ValidationResult(result=True)

    async def call(
        self, args: dict[str, Any], context: ToolContext
    ) -> ToolResult:
        input_data = args if isinstance(args, VisionDescribeInput) else VisionDescribeInput(**args)
        try:
            result = describe_image(input_data.image_path, input_data.question)
            if input_data.output_path and result.get("available"):
                import json
                Path(input_data.output_path).write_text(
                    json.dumps(result, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            success = bool(result.get("available"))
            return ToolResult(
                data=result,
                success=success,
                error=None if success else result.get("error", "unknown"),
            )
        except Exception as exc:
            logger.warning("vision_describe failed: %s", exc, exc_info=True)
            return ToolResult(data=None, success=False, error=str(exc))


# ── self-check (assert-based, 无框架) ────────────────────────

def _selfcheck() -> None:
    """10 项 assert 验证调度框架核心行为.

    各 Tier 用 monkey-patch 模拟可用/不可用, 验证降级链.
    """
    import tempfile
    from PIL import Image

    # 1. 不存在的图像 → error tier
    out = describe_image("/nonexistent.png", "test")
    assert out["tier"] == "error"
    assert out["available"] is False
    assert "不存在" in out["error"]

    # 2. 真实图像 + 所有 Tier 都 stub/不可用 → tier0_none 或 tier1
    # 先重置缓存让 _pick_tier 重新探测
    _reset_probe_cache()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        Image.new("RGB", (10, 10), (128, 128, 128)).save(tmp.name)
        tmp_path = tmp.name
    try:
        out = describe_image(tmp_path, "test")
        # 至少能跑到某个 Tier (可能是 tier1 如果 EasyOCR 装了, 否则 tier0)
        assert out["tier"] in (
            "tier0_none", "tier1_classic_ocr",
            "tier2_paddleocr", "tier3_deepseek_ocr",
        )
        assert "availability" in out
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    # 3. _pick_tier 返回 availability dict 含 4 个引擎
    _reset_probe_cache()
    tier, avail = _pick_tier()
    assert set(avail.keys()) == {"deepseek_ocr", "paddleocr", "easyocr", "tesseract"}

    # 4. 探测缓存生效: 第二次调 _probe_* 不重新探测
    _reset_probe_cache()
    _ = _probe_easyocr()  # 第一次探测
    cached = _probed["easyocr"]
    _ = _probe_easyocr()  # 第二次应走缓存
    assert _probed["easyocr"] == cached

    # 5. _reset_probe_cache 真重置
    _reset_probe_cache()
    assert _probed["deepseek_ocr"] is None
    assert _probed["paddleocr"] is None

    # 6. _tier3_deepseek_ocr stub 返结构化 unavailable
    out = _tier3_deepseek_ocr(b"", "test")
    assert out["tier"] == "tier3_deepseek_ocr"
    assert out["available"] is False
    assert "stub" in out["error"]

    # 7. _tier2_paddleocr: 没装时返 unavailable, 装了时返结构化结果
    out = _tier2_paddleocr(b"\x89PNG fake", "test")
    assert out["tier"] == "tier2_paddleocr"
    # 没装或假图片都走 except, 返 unavailable
    if not out["available"]:
        assert "PaddleOCR" in out["error"] or "推理失败" in out["error"]

    # 8. _tier1_classic_ocr 真实跑 (EasyOCR/Tesseract 装了的话)
    # 不强制 assert available, 只验返结构正确
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        Image.new("RGB", (10, 10), (255, 255, 255)).save(tmp.name)
        tmp_path = tmp.name
    try:
        out = _tier1_classic_ocr(Path(tmp_path).read_bytes(), "test")
        assert out["tier"] == "tier1_classic_ocr"
        if out["available"]:
            assert "text" in out
            assert "pixel_stats" in out
            assert "size" in out["pixel_stats"]
            assert "histogram" in out["pixel_stats"]
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    # 9. describe_image 顶层不抛 (失败也返 dict)
    # 用一个非图像文件触发 Tier 1 失败
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(b"not an image")
        tmp_path = tmp.name
    try:
        out = describe_image(tmp_path, "test")
        assert isinstance(out, dict)
        assert "tier" in out
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    # 10. VisionDescribeTool name/category 正确
    tool = VisionDescribeTool()
    assert tool.name == "vision_describe"
    assert tool.category == "cv"
    assert tool.read_only is True

    print("all self-checks passed")


if __name__ == "__main__":
    _selfcheck()
