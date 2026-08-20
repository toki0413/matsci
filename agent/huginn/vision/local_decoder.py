"""本地视觉解码器 —— 文本模型看图的常驻快路径。

把跨 agent 的视觉委托下沉成一次本地 multimodel LLM 调用 (如 ollama 上的
qwen2.5-vl), 省掉整个 agent 往返的编排/提示/prompt 开销. 没有可用模型时
decode 返回 None, 调用方回落原有跨 agent 委托, 零额外依赖.

实现只扫 stdlib (urllib), 不打新依赖. 可用性检测带 TTL 常驻缓存, 避免
每个图像 turn 都打一遍 /api/tags.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
import urllib.request
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 本地视觉解码模型的命名规则; 用户可能只拉了 qwen2.5-vl, 前缀匹配取最大版本.
_DEFAULT_MODEL_PREFIX = os.environ.get(
    "HUGINN_LOCAL_VISION_MODEL", "qwen2.5-vl"
).lower()
# 可用性缓存 TTL: 2s 内不重复探测 ollama, 常驻不刷屏. 探测失败提前失效.
_AVAIL_TTL = float(os.environ.get("HUGINN_LOCAL_VISION_TTL", "2.0"))
# 单次解码超时, 防止本地模型抽风把文本请求拖死.
_HTTP_TIMEOUT = float(os.environ.get("HUGINN_LOCAL_VISION_TIMEOUT", "20"))


def _ollama_host() -> str:
    return os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")


# 模块级常驻缓存: (host, model) -> (available_at_ts, model_name|None)
_CACHE: dict[tuple[str, str], tuple[float, str | None]] = {}


def _list_models(host: str) -> list[str]:
    """GET /api/tags, 失败返回空列表."""
    try:
        req = urllib.request.Request(f"{host}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            data = json.loads(resp.read().decode("utf-8") or "{}")
        return [m.get("name", "") for m in data.get("models", [])]
    except Exception:
        logger.debug("ollama tags lookup failed", exc_info=True)
        return []


def available() -> bool:
    """常住可用性探测: 返回是否找到本地视觉模型 (带 TTL 缓存)."""
    host = _ollama_host()
    key = (host, _DEFAULT_MODEL_PREFIX)
    now = time.monotonic()
    cached = _CACHE.get(key)
    if cached is not None and now - cached[0] < _AVAIL_TTL:
        return cached[1] is not None
    found = _find_model(host)
    _CACHE[key] = (now, found)
    return found is not None


def _find_model(host: str) -> str | None:
    """按前缀匹配 /api/tags 里最匹配的本地视觉模型名."""
    prefix = _DEFAULT_MODEL_PREFIX
    names = _list_models(host)
    exact = next((n for n in names if n.lower() == prefix), None)
    if exact:
        return exact
    best = None
    for n in names:
        if n.lower().startswith(prefix):
            if best is None or len(n) > len(best):
                best = n
    return best


def _encode_image(image_path: str | Path | bytes) -> str | None:
    """把图片转成 ollama 需要的 base64 字符串 (裸 bytes 不适用时返回 None)."""
    if isinstance(image_path, (bytes, bytearray)):
        return base64.b64encode(bytes(image_path)).decode("ascii")
    try:
        return base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
    except Exception:
        logger.debug("image encode failed", exc_info=True)
        return None


def decode_image(image_path: str | Path | bytes, question: str | None = None) -> str | None:
    """常驻本地视觉解码: 一张图 → 一段文本描述. 失败/无模型返回 None.

    *question* 给模型的指令, 默认要材料科学相关要点. 在一次请求里完成,
    不掉 agent, 也不动提示堆的其余部分.
    """
    if not available():
        return None
    host = _ollama_host()
    model = _find_model(host)
    if not model:
        _CACHE.pop((host, _DEFAULT_MODEL_PREFIX), None)
        return None
    b64 = _encode_image(image_path)
    if not b64:
        return None

    prompt = question or (
        "请用中文简要描述这张图片最明显的视觉内容, 重点标注与材料科学相关的特征, "
        "例如显微结构、晶体形貌、谱图峰位或形貌异常."
    )
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [b64],
            }
        ],
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    try:
        req = urllib.request.Request(
            f"{host}/api/chat", data=data, method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as resp:
            body = json.loads(resp.read().decode("utf-8") or "{}")
        text = (body.get("message") or {}).get("content", "")
        return text.strip() or None
    except Exception:
        logger.debug("local vision decode failed", exc_info=True)
        return None


def clear_cache() -> None:
    """测试/故障恢复用: 清掉常驻可用性缓存."""
    _CACHE.clear()


__all__ = ["available", "decode_image", "clear_cache"]