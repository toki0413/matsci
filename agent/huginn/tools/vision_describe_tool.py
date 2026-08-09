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
from typing import Any

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

        import numpy as np
        from PIL import Image

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
                import base64 as b64

                from huginn.tools.registry import ToolRegistry
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
        import numpy as np
        from PIL import Image

        from huginn.knowledge.ocr_loader import _ocr_image

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

def _run_tier(image_bytes: bytes, question: str) -> dict[str, Any]:
    """单次 Tier 调度. 不含 consistency 检查.

    从原 describe_image_bytes 抽出, 保持逻辑不变. describe_image_bytes
    在这上面加 A1 Hallu 检测层.
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


# A1: 两次 describe 结果比较的关键字段.
# text_blocks 数量差 >20% 或 structured_analysis 计数类字段不一致 → inconsistent.
# ponytail: 字段级硬比较, 不上语义匹配. ceiling: 同义不同字会误判.
_COMPARE_FIELDS = (
    "n_curves", "n_points", "n_particles", "n_elements",
    "n_defects", "n_phases", "n_fft_spots",
)


def _compare_describe_results(r1: dict[str, Any], r2: dict[str, Any]) -> dict[str, Any]:
    """比较两次 describe 结果, 返回 {inconsistent, note}.

    三级比较:
      1. text_blocks 数量差异 >20% → inconsistent
      2. structured_analysis 计数字段不一致 → inconsistent
      3. pixel_stats mean 差异 >5% → inconsistent
    """
    notes: list[str] = []
    inconsistent = False

    # 1. text_blocks 数量
    n1 = len(r1.get("text_blocks") or [])
    n2 = len(r2.get("text_blocks") or [])
    if n1 > 0 or n2 > 0:
        diff_ratio = abs(n1 - n2) / max(n1, n2, 1)
        if diff_ratio > 0.2:
            inconsistent = True
            notes.append(f"text_blocks: {n1} vs {n2} ({diff_ratio:.0%})")

    # 2. structured_analysis 计数字段
    s1 = r1.get("structured_analysis") or {}
    s2 = r2.get("structured_analysis") or {}
    for key in _COMPARE_FIELDS:
        v1, v2 = s1.get(key), s2.get(key)
        if v1 is not None and v2 is not None:
            try:
                if abs(float(v1) - float(v2)) > 0.5:
                    inconsistent = True
                    notes.append(f"{key}: {v1} vs {v2}")
            except (TypeError, ValueError):
                if v1 != v2:
                    inconsistent = True
                    notes.append(f"{key}: {v1} vs {v2}")

    # 3. pixel_stats mean (Tier 1 兜底路径)
    ps1 = (r1.get("pixel_stats") or {}).get("mean")
    ps2 = (r2.get("pixel_stats") or {}).get("mean")
    if ps1 is not None and ps2 is not None and abs(ps1 - ps2) / max(abs(ps1), abs(ps2), 1.0) > 0.05:
            inconsistent = True
            notes.append(f"pixel mean: {ps1:.1f} vs {ps2:.1f}")

    return {
        "inconsistent": inconsistent,
        "note": "; ".join(notes) if notes else "consistent",
    }


def describe_image_bytes(
    image_bytes: bytes, question: str = "", check_consistency: bool = False
) -> dict[str, Any]:
    """分层调度: 按可用资源选 Tier, 把图像 bytes 转结构化 JSON.

    跟 describe_image 同流程, 但接受 bytes 输入 (用于 visual_inspect 的
    cropped base64 场景, 不需要落盘). describe_image 内部也调本函数.

    Args:
        image_bytes: 图像二进制 (PNG/JPEG)
        question: agent 的具体问题
        check_consistency: A1 Hallu 检测 — 开启时同图调两次 (第二次 question
            加尾空格扰动), 比较结果标 low_confidence. 默认 False 保持现有行为.
            ponytail: 尾空格是最小扰动, 不改图像. ceiling: 文本级比较不等于
            语义级一致. 升级: 接 LLM judge 做语义比较.

    Returns:
        跟 describe_image 同结构. 顶层永远不抛.
        check_consistency=True 时额外含 low_confidence + consistency_note.
    """
    result = _run_tier(image_bytes, question)

    if not check_consistency:
        return result

    # A1: Hallu 检测 — 同图同 question 加尾空格, 比较两次结果.
    # PerceptionBench 实证: 同题问两次答案不同的概率很高, 不一致 → 瞎蒙.
    result2 = _run_tier(image_bytes, question + " ")
    diff = _compare_describe_results(result, result2)
    result["low_confidence"] = diff["inconsistent"]
    result["consistency_note"] = diff["note"]
    return result


def describe_image(
    image_path: str | Path, question: str = "", check_consistency: bool = False
) -> dict[str, Any]:
    """分层调度: 按可用资源选 Tier, 把图像转结构化 JSON.

    Args:
        image_path: 图像文件路径
        question: agent 的具体问题 (e.g. "这是 XRD 谱吗? 主峰在哪?")
            Tier 3 能直接答; Tier 2/1 用关键词路由到专用 action
        check_consistency: A1 Hallu 检测, 透传给 describe_image_bytes

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

    return describe_image_bytes(image_bytes, question, check_consistency=check_consistency)


# ── S3: 多图序列 (in-situ / 原位 / 反应序列) ─────────────────────────────


# 帧间一致性阈值 — 超过任一阈值标 low_inter_frame_consistency
# ponytail: 经验阈值. 升级路径: 按图像类型 (XRD/SEM/TEM) 自适应阈值.
_INTER_FRAME_HIST_CORR_THRESHOLD = 0.8   # 相邻帧直方图相关 < 0.8 → 不一致
_INTER_FRAME_TEXT_DELTA_THRESHOLD = 0.5   # text_blocks 数量变化 > 50% → 不一致
_INTER_FRAME_FIELD_DELTA_THRESHOLD = 0.5  # structured_analysis 字段数变化 > 50% → 不一致


def _hist_correlation(h1: list[float], h2: list[float]) -> float:
    """两个直方图的 Pearson 相关 — 越接近 1 越相似.

    ponytail: numpy 不可用时降级到纯 Python 实现.
    """
    if not h1 or not h2 or len(h1) != len(h2):
        return 0.0
    n = len(h1)
    s1 = sum(h1)
    s2 = sum(h2)
    if s1 == 0 or s2 == 0:
        return 0.0
    m1 = [v / s1 for v in h1]
    m2 = [v / s2 for v in h2]
    mean1 = sum(m1) / n
    mean2 = sum(m2) / n
    num = sum((m1[i] - mean1) * (m2[i] - mean2) for i in range(n))
    den1 = sum((m1[i] - mean1) ** 2 for i in range(n)) ** 0.5
    den2 = sum((m2[i] - mean2) ** 2 for i in range(n)) ** 0.5
    if den1 == 0 or den2 == 0:
        return 0.0
    return num / (den1 * den2)


def describe_image_sequence(
    image_paths: list[str], question: str = "",
    check_consistency: bool = True,
) -> dict[str, Any]:
    """S3: 多图序列视觉描述 + 帧间一致性检测.

    对每帧单独 describe_image (复用单图路径), 然后做相邻帧一致性检测:
      - pixel_stats.histogram 相关 (峰值漂移代理)
      - text_blocks 数量变化 (内容变化代理)
      - structured_analysis 字段数变化 (结构化信息变化代理)
    任一指标超阈值标 low_inter_frame_consistency=True + inconsistency_reasons.

    与 S1 时空表征的耦合: 原位 XRD 帧间峰位漂移可视化 ≈ F(q,t) 衰减
    (G(r,t) 的空间傅立叶变换). 让 agent 能跨"视觉帧"和"MD 模拟时序"
    做同一物理量的交叉验证.

    ponytail: 复用单图 describe, 不重新实现. 帧间一致性 3 指标够用,
    升级路径: 加 LLM 跨帧问答一致性.
    """
    if not image_paths:
        return {"tier": "error", "available": False, "error": "image_paths 为空"}

    # 单帧走原单图路径 (保留一致性检测语义)
    if len(image_paths) == 1:
        return describe_image(image_paths[0], question, check_consistency=check_consistency)

    # 多帧: 每帧单独 describe
    per_frame: list[dict[str, Any]] = []
    for i, p in enumerate(image_paths):
        r = describe_image(p, question, check_consistency=False)  # 帧内不再开 A1, 跨帧才是重点
        r["frame_index"] = i
        per_frame.append(r)

    # 帧间一致性: 相邻两两比较
    inter_frame_issues: list[str] = []
    pair_results: list[dict] = []
    for i in range(1, len(per_frame)):
        prev = per_frame[i - 1]
        curr = per_frame[i]
        issues: list[str] = []

        # 1. histogram 相关
        prev_hist = ((prev.get("pixel_stats") or {}).get("histogram") or [])
        curr_hist = ((curr.get("pixel_stats") or {}).get("histogram") or [])
        if prev_hist and curr_hist:
            corr = _hist_correlation(prev_hist, curr_hist)
            if corr < _INTER_FRAME_HIST_CORR_THRESHOLD:
                issues.append(f"hist_corr={corr:.3f}<{_INTER_FRAME_HIST_CORR_THRESHOLD}")

        # 2. text_blocks 数量变化
        prev_n = len(prev.get("text_blocks") or prev.get("text") or [])
        curr_n = len(curr.get("text_blocks") or curr.get("text") or [])
        if prev_n > 0:
            delta = abs(curr_n - prev_n) / prev_n
            if delta > _INTER_FRAME_TEXT_DELTA_THRESHOLD:
                issues.append(f"text_n_delta={delta:.3f}>{_INTER_FRAME_TEXT_DELTA_THRESHOLD}")

        # 3. structured_analysis 字段数变化
        prev_sa = prev.get("structured_analysis") or {}
        curr_sa = curr.get("structured_analysis") or {}
        if isinstance(prev_sa, dict) and isinstance(curr_sa, dict) and prev_sa:
            prev_nf = len(prev_sa)
            curr_nf = len(curr_sa)
            delta_f = abs(curr_nf - prev_nf) / prev_nf
            if delta_f > _INTER_FRAME_FIELD_DELTA_THRESHOLD:
                issues.append(f"field_n_delta={delta_f:.3f}>{_INTER_FRAME_FIELD_DELTA_THRESHOLD}")

        pair_results.append({
            "frame_pair": (i - 1, i),
            "issues": issues,
            "ok": len(issues) == 0,
        })
        if issues:
            inter_frame_issues.append(f"pair({i-1},{i}): " + ", ".join(issues))

    return {
        "tier": "multi_frame_sequence",
        "available": True,
        "n_frames": len(per_frame),
        "per_frame": per_frame,
        "inter_frame_consistency": {
            "pairs": pair_results,
            "issues": inter_frame_issues,
            "low_inter_frame_consistency": len(inter_frame_issues) > 0,
            "inconsistency_reasons": inter_frame_issues,
        },
        # 视觉-时空耦合提示: 让 agent 知道这跟 F(q,t) 衰减对偶
        "physical_coupling_hint": (
            "原位 XRD 帧间峰位漂移可视化 ≈ F(q,t) 中间散射函数衰减 — "
            "若同时有 lammps _physical_timeseries F_q_t 数据, 可交叉验证."
        ),
    }


# ── M5: 结构化 captioning ─────────────────────────────────────────────────


# 8 scene action 跟 caption 模板的映射. 每个 action 一段 caption 模板,
# 用 .format(**structured_dict) 填关键字段. 字段缺失时模板自动跳过.
# ponytail: 模板拼接, 不调 LLM (节省成本 + 避免循环依赖). 升级: 接 LLM 汇总.
_SCENE_CAPTION_TEMPLATES: dict[str, str] = {
    "plot_extract": (
        "[Plot] {n_curves} curve(s), x-axis='{x_axis_type}', y-axis='{y_axis_type}'. "
        "Extracted data points: {n_points}."
    ),
    "sem_analysis": (
        "[SEM] {n_particles} particles detected, mean size={mean_size_nm} nm, "
        "coverage={coverage_percent}%. Contrast: {contrast_level}."
    ),
    "tem_lattice": (
        "[TEM] Lattice fringes d-spacing={d_spacing_nm} nm, FFT spots: {n_fft_spots}. "
        "Crystallographic orientation: {orientation}."
    ),
    "eds_mapping": (
        "[EDS] {n_elements} elements mapped: {elements_list}. "
        "Spatial distribution: {distribution_pattern}."
    ),
    "particle_stats": (
        "[Particles] Count={n_particles}, size distribution: "
        "min={min_size}/mean={mean_size}/max={max_size} nm, polydispersity={polydispersity}."
    ),
    "defect_detect": (
        "[Defects] {n_defects} {defect_type} defects detected, "
        "density={defect_density}. Sensitivity={sensitivity}."
    ),
    "phase_field": (
        "[Phase Field] {n_phases} phases, volume fractions: {volume_fractions}. "
        "Interface fraction={interface_percent}%."
    ),
    "deplot_chart": (
        "[Chart] Captioned: {chart_caption}. "
        "Data series identified: {n_series}."
    ),
}


def _format_caption(scene: str, structured: dict[str, Any]) -> str:
    """把 image_analysis_tool 的 structured_analysis 渲染成一段 caption.

    字段缺失时, 模板里 {field} 留空 — LLM 看到空字段也能理解是 'unknown'.
    """
    template = _SCENE_CAPTION_TEMPLATES.get(scene)
    if not template:
        return f"[{scene}] analysis completed."
    try:
        # 安全 format: 缺字段不抛
        from string import Formatter
        fields_needed = [
            f for _, f, _, _ in Formatter().parse(template) if f is not None
        ]
        safe_kwargs = {f: structured.get(f, "unknown") for f in fields_needed}
        return template.format(**safe_kwargs)
    except (KeyError, IndexError, ValueError):
        return f"[{scene}] analysis completed."


def caption_image_bytes(
    image_bytes: bytes,
    question: str = "",
    max_scenes: int = 3,
) -> dict[str, Any]:
    """M5: 结构化 captioning — 调 image_analysis_tool 8 scene action, 汇总 caption.

    流程:
      1. 先调 describe_image_bytes 拿基础结构化结果 (Tier 3/2/1)
      2. 如果已有 structured_analysis (Tier 2 question 路由命中), 直接 caption
      3. 否则启发式跑 top-N scene action (默认 plot_extract + sem_analysis,
         因为这两类覆盖材料科学图像最常见的谱图 + 形貌)
      4. 把每个 scene 的 structured_analysis 渲染成 caption 段
      5. 拼接成完整 caption, 让 text-only LLM 通过 caption 理解图像内容

    替代真 VLM: 不需要视觉 encoder, 用专用 CV 算法 + 模板拼接.
    ponytail: 模板汇总, 不调 LLM 汇总 (避免成本 + 循环依赖).
    升级路径: 接 LLM 把多段 caption 重写成一段自然语言.

    Args:
        image_bytes: 图像二进制
        question: 可选, 影响 scene 路由 (含 'XRD'/'SEM' 等关键词时优先对应 scene)
        max_scenes: 最多跑几个 scene action (避免开销)

    Returns:
        dict 含:
          - caption: 汇总 caption 文本 (核心输出)
          - scene_captions: 各 scene 单独 caption (列表)
          - scenes_tried: 跑了哪些 scene
          - base_tier: describe_image_bytes 选的 Tier
          - available: 是否拿到任何结构化结果
    """
    base = describe_image_bytes(image_bytes, question)
    base_tier = base.get("tier", "unknown")
    base_available = bool(base.get("available"))

    scene_captions: list[str] = []
    scenes_tried: list[str] = []

    # 1. 如果 base 已经有 structured_analysis (Tier 2 question 路由命中), 直接用
    existing_struct = base.get("structured_analysis")
    existing_scene = base.get("scene")
    if isinstance(existing_struct, dict) and existing_scene:
        cap = _format_caption(existing_scene, existing_struct)
        scene_captions.append(cap)
        scenes_tried.append(existing_scene)

    # 2. 启发式补 scene: 如果还不够 max_scenes, 跑默认 2 个最常见 scene
    # ponytail: plot_extract + sem_analysis 覆盖最常见的谱图 + 形貌.
    # 升级路径: 用 pixel_stats / 直方图自动判图像类型再选 scene.
    default_scenes = ["plot_extract", "sem_analysis"]
    q_lower = question.lower() if question else ""
    # question 含特定关键词时把对应 scene 提到前面
    if any(k in q_lower for k in ("xrd", "ir", "raman", "uv", "stress", "strain", "dsc", "tga")):
        default_scenes = ["plot_extract"] + [s for s in default_scenes if s != "plot_extract"]
    elif any(k in q_lower for k in ("sem", "tem", "particle", "morphology")):
        default_scenes = ["sem_analysis"] + [s for s in default_scenes if s != "sem_analysis"]

    for scene in default_scenes:
        if len(scenes_tried) >= max_scenes:
            break
        if scene in scenes_tried:
            continue
        try:
            # scene 函数接收 ImageAnalysisInput(image_path, action, parameters)
            # 落盘 image_bytes 到 tmp 文件 — scene 函数需要真实文件路径读图.
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp.write(image_bytes)
                tmp_path = tmp.name
            try:
                from huginn.tools.image_analysis.tool import ImageAnalysisInput
                input_data = ImageAnalysisInput(
                    image_path=tmp_path,
                    action=scene,  # type: ignore[arg-type]
                    parameters={"task_description": question or f"Run {scene} analysis"},
                )
                # lazy import 对应 scene 模块, 直接调 sync 函数 (绕开 async ToolRegistry)
                mod_path = f"huginn.tools.image_analysis.scenes_{scene.split('_')[0]}"
                if scene == "plot_extract":
                    mod_path = "huginn.tools.image_analysis.scenes_plot_extract"
                elif scene == "deplot_chart":
                    mod_path = "huginn.tools.image_analysis.scenes_deplot"
                elif scene == "particle_stats":
                    mod_path = "huginn.tools.image_analysis.scenes_particles"
                elif scene == "defect_detect":
                    mod_path = "huginn.tools.image_analysis.scenes_defect"
                elif scene == "phase_field":
                    mod_path = "huginn.tools.image_analysis.scenes_phase_field"
                import importlib
                mod = importlib.import_module(mod_path)
                fn = getattr(mod, scene)
                res = fn(input_data)
                if res and getattr(res, "success", False):
                    structured = res.data if hasattr(res, "data") else res
                    if isinstance(structured, dict):
                        cap = _format_caption(scene, structured)
                        scene_captions.append(cap)
                        scenes_tried.append(scene)
            finally:
                Path(tmp_path).unlink(missing_ok=True)
        except Exception as exc:
            logger.debug("caption scene %s failed: %s", scene, exc, exc_info=True)

    # 3. Tier 0 兜底: 如果啥 scene 都没跑出来, 用 pixel_stats 生成最简 caption
    if not scene_captions and base_available:
        ps = base.get("pixel_stats") or {}
        if ps:
            size = ps.get("size", [0, 0])
            mean = ps.get("mean", 0)
            std = ps.get("std", 0)
            scene_captions.append(
                f"[Pixel] Image size={size}, gray mean={mean:.1f}, std={std:.1f}. "
                "No specialized scene analysis available."
            )
            scenes_tried.append("pixel_stats")

    caption_text = "\n".join(scene_captions) if scene_captions else ""

    return {
        "caption": caption_text,
        "scene_captions": scene_captions,
        "scenes_tried": scenes_tried,
        "base_tier": base_tier,
        "available": bool(scene_captions),
        "base_result": base,
    }


# ── HuginnTool 包装 ──────────────────────────────────────────

class VisionDescribeInput(BaseModel):
    image_path: str = Field(..., description="图像文件路径")
    image_paths: list[str] = Field(
        default_factory=list,
        description=(
            "S3: 多图序列 (in-situ XRD / 原位 TEM / 反应序列). "
            "非空时走 describe_image_sequence 多图路径, image_path 被忽略. "
            "对每帧单独 describe, 再做帧间一致性检测 — "
            "相邻帧的 pixel_stats 直方图相关 + text_blocks 数量变化 + structured_analysis 字段数变化, "
            "任一指标超阈值标 low_inter_frame_consistency. "
            "原位 XRD 帧间峰位漂移可视化 ≈ F(q,t) 衰减 (与 lammps _physical_timeseries 时空对偶)."
        ),
    )
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
    check_consistency: bool = Field(
        default=False,
        description=(
            "A1 Hallu 检测 — 开启时同图调两次, 比较结果标 low_confidence. "
            "用于检测 MLLM 是否瞎蒙 (PerceptionBench 启发)."
        ),
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
            # S3: image_paths 非空走多图序列路径 (in-situ XRD / 原位 TEM)
            if input_data.image_paths:
                result = describe_image_sequence(
                    input_data.image_paths, input_data.question,
                    check_consistency=True,  # 多图默认开跨帧一致性
                )
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

            # P1-4: 量化类图像 (XRD/SEM/TEM/谱/峰/颗粒) 自动开启 A1 Hallu 检测.
            # 之前 check_consistency 默认 False, 生产路径永不触发 — agent 瞎蒙无感知.
            check_consistency = input_data.check_consistency
            if not check_consistency:
                q_lower = (input_data.question or "").lower()
                if any(kw in q_lower for kw in (
                    "xrd", "sem", "tem", "谱", "峰", "颗粒", "分布", "晶格", "d-spacing",
                    "fft", "particle", "lattice", "diffraction",
                )):
                    check_consistency = True

            result = describe_image(
                input_data.image_path, input_data.question,
                check_consistency=check_consistency,
            )
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

    # 11. A1: check_consistency 默认 False, 行为不变 (向后兼容)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        Image.new("RGB", (10, 10), (128, 128, 128)).save(tmp.name)
        tmp_path = tmp.name
    try:
        out_default = describe_image(tmp_path, "test")
        assert "low_confidence" not in out_default, "default should not add low_confidence"
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    # 12. A1: check_consistency=True 时返回 low_confidence 字段
    _reset_probe_cache()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        Image.new("RGB", (10, 10), (128, 128, 128)).save(tmp.name)
        tmp_path = tmp.name
    try:
        out_consistency = describe_image(tmp_path, "test", check_consistency=True)
        assert "low_confidence" in out_consistency, "check_consistency=True must add low_confidence"
        assert "consistency_note" in out_consistency, "check_consistency=True must add consistency_note"
        # 同图同引擎两次调用, 结果应该一致 → low_confidence=False
        # (除非 Tier 不稳定, 那 exactly 是 Hallu 检测要抓的)
        assert isinstance(out_consistency["low_confidence"], bool)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    # 13. A1: _compare_describe_results 三级比较
    # 13a. text_blocks 数量差异 >20% → inconsistent
    r1 = {"text_blocks": [1, 2, 3, 4, 5]}
    r2 = {"text_blocks": [1, 2]}
    diff = _compare_describe_results(r1, r2)
    assert diff["inconsistent"] is True, f"5 vs 2 text_blocks should be inconsistent: {diff}"
    assert "text_blocks" in diff["note"]

    # 13b. structured_analysis 计数字段不一致 → inconsistent
    r1 = {"structured_analysis": {"n_particles": 10}}
    r2 = {"structured_analysis": {"n_particles": 5}}
    diff = _compare_describe_results(r1, r2)
    assert diff["inconsistent"] is True, f"n_particles 10 vs 5 should be inconsistent: {diff}"
    assert "n_particles" in diff["note"]

    # 13c. pixel_stats mean 差异 >5% → inconsistent
    r1 = {"pixel_stats": {"mean": 100.0}}
    r2 = {"pixel_stats": {"mean": 50.0}}
    diff = _compare_describe_results(r1, r2)
    assert diff["inconsistent"] is True, f"mean 100 vs 50 should be inconsistent: {diff}"

    # 13d. 一致结果 → inconsistent=False
    r1 = {"text_blocks": [1, 2, 3], "structured_analysis": {"n_particles": 5}, "pixel_stats": {"mean": 100.0}}
    r2 = {"text_blocks": [1, 2, 3], "structured_analysis": {"n_particles": 5}, "pixel_stats": {"mean": 100.0}}
    diff = _compare_describe_results(r1, r2)
    assert diff["inconsistent"] is False, f"identical results should be consistent: {diff}"
    assert diff["note"] == "consistent"

    print("all self-checks passed (A1 Hallu detection OK)")


if __name__ == "__main__":
    _selfcheck()
