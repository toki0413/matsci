"""Mental imagery — LLM "想象" 图像 → sketch (合成) → verify (连通域) loop.

借鉴 Mirage (arXiv:2506.17218, MIT, 2025-06): text LLM 通过描述 + 验证形成
"心理意象", 不依赖真 VLM. 跟 visual_inspect.chain 集成 — agent 可以调
"chain: mental_imagery: sketch lattice cubic 4Å; verify lattice 4Å"
串联想象-验证, 形成跟 OpenThinkIMG 风格一致的视觉推理回路.

设计:
  sketch(spec) -> image_bytes: numpy 合成图, 支持 lattice / particles / spectrum
    3 种模板. ponytail: 只支持预设模板, 不上 text-to-image 模型.
  verify(image_bytes, expected) -> dict: 用 extract_box_primitives (M6) 检查
    连通域数量是否符合 expected, 形成真 sketch→verify 闭环.

ceiling:
  - sketch 只支持 3 种模板 (lattice/particles/spectrum), 不任意图像
  - verify 只检查连通域数量, 不检查形状/语义
升级路径:
  - sketch: 接 Stable Diffusion / DALL-E 真生成任意图像
  - verify: 接 LLM 判断图像是否符合 spec (semantic check)
"""
from __future__ import annotations

import io
import logging
import re
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ── sketch: spec → 合成图 bytes ────────────────────────────────────────────


def sketch(spec: str) -> bytes:
    """根据 spec 描述合成图像 bytes (PNG).

    支持的 spec 关键词 (任选其一):
      - "lattice <a>Å [cubic|hex]": 晶格条纹, a 是周期 (Å → 像素, 1Å=10px)
      - "particles <n>": n 个圆形粒子 (黑底白图反白, 黑粒子在白底)
      - "spectrum <n_peaks>": n_peaks 个高斯峰 (1D 曲线 tiled 成 2D 图)

    ponytail: 3 种预设模板, 关键词正则匹配. 不上 text-to-image 模型.
    ceiling: 不支持任意描述 (e.g. "一张猫的图").
    升级路径: 接 Stable Diffusion 真生成.

    Args:
        spec: 自然语言描述 (e.g. "lattice 4Å cubic", "particles 10")

    Returns:
        PNG 图像 bytes
    """
    spec_lower = spec.lower()
    img_arr: np.ndarray | None = None

    if "lattice" in spec_lower:
        m = re.search(r"(\d+\.?\d*)\s*(?:å|a|angstrom)", spec_lower)
        a_angstrom = float(m.group(1)) if m else 4.0
        period_px = max(5, int(a_angstrom * 10))
        size = 200
        x = np.arange(size)
        XX, YY = np.meshgrid(x, x)
        if "hex" in spec_lower:
            # 六方: 60° 旋转叠加两个正弦
            angle = np.pi / 3
            XXr = XX * np.cos(angle) + YY * np.sin(angle)
            YYr = -XX * np.sin(angle) + YY * np.cos(angle)
            grating = (
                128
                + 80 * np.sin(2 * np.pi * XXr / period_px)
                + 80 * np.sin(2 * np.pi * YYr / period_px)
            )
        else:
            # cubic: 两个正弦叠加
            grating = (
                128
                + 80 * np.sin(2 * np.pi * XX / period_px)
                + 80 * np.sin(2 * np.pi * YY / period_px)
            )
        img_arr = np.clip(grating, 0, 255).astype(np.uint8)

    elif "particle" in spec_lower:
        m = re.search(r"(\d+)\s*particles?", spec_lower)
        n = int(m.group(1)) if m else 10
        size = 200
        img_arr = np.full((size, size), 255, dtype=np.uint8)
        rng = np.random.default_rng(42)
        for _ in range(n):
            cx = int(rng.integers(20, size - 20))
            cy = int(rng.integers(20, size - 20))
            r = int(rng.integers(5, 15))
            # numpy 矢量画圆 (比 putpixel 双循环快)
            ys, xs = np.ogrid[:size, :size]
            mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= r * r
            img_arr[mask] = 0

    elif "spectrum" in spec_lower or "spectra" in spec_lower:
        m = re.search(r"(\d+)\s*peaks?", spec_lower)
        n_peaks = int(m.group(1)) if m else 3
        size = 200
        x = np.linspace(0, 10, size)
        y = np.zeros(size)
        rng = np.random.default_rng(42)
        for _ in range(n_peaks):
            mu = float(rng.uniform(1, 9))
            sigma = float(rng.uniform(0.1, 0.3))
            amp = float(rng.uniform(0.5, 1.0))
            y += amp * np.exp(-((x - mu) / sigma) ** 2)
        y_2d = np.clip(y * 200, 0, 255).astype(np.uint8)
        # 1D 曲线 tiled 成 2D 图 (高度 50)
        img_arr = np.tile(y_2d, (50, 1))

    if img_arr is None:
        # 默认: 白图 (spec 不匹配任何模板)
        img_arr = np.full((100, 100), 255, dtype=np.uint8)

    try:
        from PIL import Image
        buf = io.BytesIO()
        Image.fromarray(img_arr).save(buf, format="PNG")
        return buf.getvalue()
    except ImportError:
        # PIL 不可用时返回空 bytes (verify 会处理)
        return b""


# ── verify: image_bytes + expected → 检查结果 ─────────────────────────────


def verify(image_bytes: bytes, expected: dict[str, Any]) -> dict[str, Any]:
    """用 extract_box_primitives (M6) 检查 image_bytes 是否符合 expected.

    真正的 sketch→verify 闭环: sketch 出 N 个粒子, verify 检查是不是真有
    N 个连通域. 不调 LLM, 不依赖 OCR — 直接用 CV 算子验证.

    Args:
        image_bytes: sketch 输出的图像 bytes
        expected: 期望 dict, 支持:
            - {"kind": "particles", "n": 10}: 期望 N 个连通域
            - {"kind": "lattice"}: 期望有连通域 (不强制数量)
            - {"kind": "spectrum", "n_peaks": 3}: 期望 1D 曲线 (退化, 不强制)
            - 任意 dict: 至少检查 sketch 不空白

    Returns:
        dict: verified (bool), n_regions_detected, expected, raw_primitives
    """
    if not image_bytes:
        return {"verified": False, "error": "empty image bytes", "expected": expected}

    try:
        from huginn.tools.visual_hook import extract_box_primitives, parse_box_primitive
    except ImportError:
        return {"verified": False, "error": "visual_hook not available", "expected": expected}

    primitives = extract_box_primitives(image_bytes, threshold=128, max_boxes=50)
    boxes = parse_box_primitive(primitives)
    # 排除 overall bbox (label="overall"), 只数 region N
    region_boxes = [b for b in boxes if b.get("label") != "overall" and b.get("label", "").startswith("region")]
    n_detected = len(region_boxes)

    kind = expected.get("kind", "")
    verified = False
    note = ""

    if kind == "particles":
        n_expected = int(expected.get("n", 0))
        # 容差: 检测到的粒子数在 expected ± 50% 内算通过
        # (合成时 rng 可能因重叠连成一片)
        tol = max(2, n_expected // 2)
        verified = abs(n_detected - n_expected) <= tol
        note = f"particles expected={n_expected}, detected={n_detected}, tol=±{tol}"

    elif kind == "lattice":
        # lattice 期望有连通域 (低阈值区的暗条纹)
        verified = n_detected >= 1 or len(boxes) >= 1
        note = f"lattice detected={n_detected} regions (>=1 expected)"

    elif kind == "spectrum":
        # spectrum 是 1D tiled, 暗区域是峰位 — 至少有 1 个连通域
        verified = len(boxes) >= 1
        note = f"spectrum detected={len(boxes)} boxes (>=1 expected)"

    else:
        # 任意 expected: 至少检查 sketch 不全白 (有内容)
        verified = len(boxes) >= 1
        note = f"generic check: {len(boxes)} boxes detected"

    return {
        "verified": verified,
        "n_regions_detected": n_detected,
        "n_boxes_total": len(boxes),
        "expected": expected,
        "note": note,
        "raw_primitives": primitives[:300] if primitives else "",
    }


# ── mental_imagery_loop: sketch → verify 闭环 ─────────────────────────────


def mental_imagery_loop(spec: str, max_iter: int = 1) -> dict[str, Any]:
    """sketch → verify 一次循环. max_iter>1 时迭代调整 (目前只支持 1 次).

    流程:
      1. sketch(spec) → image_bytes
      2. 解析 spec 拿 expected (e.g. "particles 10" → {"kind": "particles", "n": 10})
      3. verify(image_bytes, expected) → result
      4. 返回 sketch + verify 结果

    ponytail: max_iter 默认 1 (sketch 一次, verify 一次, 不迭代).
    升级路径: verify 失败时调 LLM 改 spec 重 sketch (真闭环).
    """
    img_bytes = sketch(spec)

    # 解析 spec → expected
    spec_lower = spec.lower()
    expected: dict[str, Any] = {}
    if "particle" in spec_lower:
        m = re.search(r"(\d+)\s*particles?", spec_lower)
        n = int(m.group(1)) if m else 10
        expected = {"kind": "particles", "n": n}
    elif "lattice" in spec_lower:
        expected = {"kind": "lattice"}
    elif "spectrum" in spec_lower or "spectra" in spec_lower:
        m = re.search(r"(\d+)\s*peaks?", spec_lower)
        n = int(m.group(1)) if m else 3
        expected = {"kind": "spectrum", "n_peaks": n}
    else:
        expected = {"kind": "unknown"}

    verify_result = verify(img_bytes, expected)

    return {
        "spec": spec,
        "sketch_image_bytes": img_bytes,
        "expected": expected,
        "verify": verify_result,
        "iter_count": 1,
        "loop_completed": verify_result.get("verified", False),
    }


# ── selfcheck ──────────────────────────────────────────────────────────────


def _selfcheck() -> None:
    """L10 selfcheck: sketch → verify 闭环 (3 种模板 × 各 1 场景)."""
    # 1. sketch particles 10 → verify 检测到 ~10 个连通域
    out1 = mental_imagery_loop("particles 10")
    assert out1["expected"] == {"kind": "particles", "n": 10}, out1["expected"]
    assert out1["sketch_image_bytes"], "sketch returned empty bytes"
    v1 = out1["verify"]
    assert "verified" in v1, v1
    n1 = v1["n_regions_detected"]
    assert 5 <= n1 <= 20, f"expected ~10 particles, detected {n1}: {v1['note']}"
    print(f"1. particles 10 → verified={v1['verified']}, detected={n1}: {v1['note']}")

    # 2. sketch lattice 4Å cubic → verify 检测到至少 1 个连通域
    out2 = mental_imagery_loop("lattice 4Å cubic")
    assert out2["expected"] == {"kind": "lattice"}, out2["expected"]
    v2 = out2["verify"]
    assert v2["verified"], f"lattice verify failed: {v2['note']}"
    print(f"2. lattice 4Å cubic → verified={v2['verified']}: {v2['note']}")

    # 3. sketch spectrum 3 peaks → verify 至少 1 个 box
    out3 = mental_imagery_loop("spectrum 3 peaks")
    assert out3["expected"] == {"kind": "spectrum", "n_peaks": 3}, out3["expected"]
    v3 = out3["verify"]
    assert v3["verified"], f"spectrum verify failed: {v3['note']}"
    print(f"3. spectrum 3 peaks → verified={v3['verified']}: {v3['note']}")

    # 4. 未知 spec → 默认白图, verify 失败 (无连通域)
    out4 = mental_imagery_loop("a cat sitting on a chair")
    assert out4["expected"] == {"kind": "unknown"}, out4["expected"]
    v4 = out4["verify"]
    # 白图无连通域 → verified=False (这是预期的 ceiling)
    print(f"4. unknown spec → verified={v4['verified']}: {v4['note']}")

    # 5. sketch 单独调用 + verify 单独调用 (解耦)
    img = sketch("particles 5")
    assert img, "sketch standalone returned empty"
    v5 = verify(img, {"kind": "particles", "n": 5})
    assert "verified" in v5
    print(f"5. standalone sketch+verify → detected={v5['n_regions_detected']}")

    print("L10 ALL CHECKS PASSED")


if __name__ == "__main__":
    _selfcheck()
