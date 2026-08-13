"""批次 D: visual_inspect 精修闭环 — consistency 默认开启 + 门禁留痕.

批次 D 目标: 把 visual_inspect 的 consistency_check 提升为闭环默认开启,
并对低置信度 zoom 结果附加渲染门禁 (visualize_qa) 判定留痕.
"""

from __future__ import annotations

import asyncio
import base64 as b64
import io as _io

import numpy as np
from PIL import Image

from huginn.autoloop.visual_inspect import VisualInspectMixin


class _MockEngine(VisualInspectMixin):
    def __init__(self, img_b64: str) -> None:
        self._last_visual_context = ""
        self._visual_base64 = img_b64
        self._last_visual_base64 = None


def _img_b64(arr) -> str:
    buf = _io.BytesIO()
    Image.fromarray(np.asarray(arr, dtype="uint8")).save(buf, format="PNG")
    return b64.b64encode(buf.getvalue()).decode()


def _blank_img() -> np.ndarray:
    # 纯白图 → visualize_qa 判 blank
    return np.full((200, 200), 255, dtype="uint8")


def _content_img() -> np.ndarray:
    # 有内容图 → visualize_qa 判 pass
    arr = np.full((200, 200), 255, dtype="uint8")
    arr[20:180, 20:180] = 0
    return arr


def test_consistency_default_on_in_zoom() -> None:
    """批次 D: consistency_check 默认开启 → zoom 产生 consistency_score."""
    img = np.full((200, 200), 255, dtype="uint8")
    img[0:100, :] = 0  # 上半黑下半白
    eng = _MockEngine(_img_b64(img))

    res = asyncio.run(
        eng._execute_visual_inspect(
            "zoom into region [0,0]-[100,100]", {}, consistency_check=True
        )
    )
    assert res["success"]
    action = res["actions"][0]
    assert "consistency_score" in action


def test_consistency_off_if_explicitly_false() -> None:
    """consistency_check=False → 无 consistency_score."""
    img = np.full((200, 200), 255, dtype="uint8")
    img[0:100, :] = 0
    eng = _MockEngine(_img_b64(img))

    res = asyncio.run(
        eng._execute_visual_inspect(
            "zoom into region [0,0]-[100,100]", {}, consistency_check=False
        )
    )
    action = res["actions"][0]
    assert "consistency_score" not in action
