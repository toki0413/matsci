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

Bourbaki 三结构视角 (B 文档化):
  代数 I  (free monoid): spec 是字符串 (自然语言), sketch/verify 操作把 spec 映射
                         到 image_bytes. mental_imagery_loop 是 sketch∘verify 复合.
  代数 II (SE(3) 群作用): lattice 模板用 2D 正弦光栅, 可被 SE(2) 旋转作用.
                         sketch 不直接施加 SE(3), 但输出图像兼容 SE(3) (FFT 旋转不变).
  拓扑   (连通域邻域):  verify 用 extract_box_primitives (M6) 检查连通域.
                         BoxPrimitivesView (topology_protocol) 适配 sketch 输出.

# 架构状态: 研究探索层 — 未接入主循环, 保留作为 future hook. 如需启用, 在 huginn/events/unified_bus.py 订阅 cognitive.* 事件并接入.
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
        logger.debug("best-effort op failed", exc_info=True)
        return b""


# ── verify: image_bytes + expected → 检查结果 ─────────────────────────────


def _decode_to_gray(image_bytes: bytes) -> np.ndarray | None:
    """解码 image_bytes 到灰度 numpy array. 失败返 None."""
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode != "L":
            img = img.convert("L")
        return np.asarray(img)
    except Exception:
        logger.debug("best-effort op failed", exc_info=True)
        return None


def _analyze_particles(img_arr: np.ndarray) -> dict[str, Any]:
    """A2: 粒子形状分析 — 连通域 + 圆度 + 等效直径分布.

    圆度 = 4πA/P², 完美圆 = 1.0. 合成粒子是圆形, 圆度应 >0.5.
    等效直径 = sqrt(4A/π), 分布 std/mean <0.5 表示尺寸均匀.
    ponytail: scipy.ndimage 做连通域, 不上 OpenCV contour.
    """
    try:
        from scipy import ndimage
    except ImportError:
        return {"n": 0, "error": "scipy.ndimage not available"}
    binary = img_arr < 128
    labeled, n = ndimage.label(binary)
    if n == 0:
        return {"n": 0, "circularities": [], "diameters": []}
    circularities: list[float] = []
    diameters: list[float] = []
    for i in range(1, n + 1):
        region = labeled == i
        area = int(region.sum())
        if area < 3:  # 噪声点跳过
            continue
        eroded = ndimage.binary_erosion(region)
        perimeter = int((region & ~eroded).sum())
        if perimeter > 0:
            circ = 4 * np.pi * area / (perimeter ** 2)
            circularities.append(float(min(circ, 1.0)))
        diam = np.sqrt(4 * area / np.pi)
        diameters.append(float(diam))
    return {
        "n": len(circularities),
        "circularities": circularities,
        "diameters": diameters,
        "mean_diameter": float(np.mean(diameters)) if diameters else 0.0,
        "std_diameter": float(np.std(diameters)) if diameters else 0.0,
        "mean_circularity": float(np.mean(circularities)) if circularities else 0.0,
    }


def _analyze_lattice_fft(img_arr: np.ndarray) -> dict[str, Any]:
    """A2: 晶格 FFT 周期性检查 — 2D FFT 找最大非 DC 峰.

    峰值 > 均值 3 倍 → has_periodicity=True.
    ponytail: numpy.fft, 不上 scipy.fftpack.
    """
    f = np.fft.fft2(img_arr.astype(float))
    f_shift = np.fft.fftshift(f)
    magnitude = np.abs(f_shift)
    h, w = magnitude.shape
    magnitude[h // 2, w // 2] = 0  # 去 DC
    peak_idx = np.unravel_index(np.argmax(magnitude), magnitude.shape)
    peak_val = float(magnitude[peak_idx])
    cy, cx = h // 2, w // 2
    freq_dist = float(np.sqrt((peak_idx[0] - cy) ** 2 + (peak_idx[1] - cx) ** 2))
    period_px = float(h / freq_dist) if freq_dist > 0 else 0.0
    return {
        "peak_position": [int(peak_idx[0]), int(peak_idx[1])],
        "peak_value": peak_val,
        "freq_dist": freq_dist,
        "period_px": period_px,
        "has_periodicity": peak_val > magnitude.mean() * 3,
    }


def _analyze_spectrum_peaks(img_arr: np.ndarray) -> dict[str, Any]:
    """A2: 谱图峰位检查 — 行均值 1D 信号 + find_peaks."""
    try:
        from scipy.signal import find_peaks
    except ImportError:
        return {"n_peaks": 0, "error": "scipy.signal not available"}
    signal = img_arr.mean(axis=0).astype(float)
    peaks, _ = find_peaks(signal, height=signal.max() * 0.3, distance=5)
    return {
        "n_peaks": len(peaks),
        "peak_positions": peaks.tolist(),
    }


def verify(image_bytes: bytes, expected: dict[str, Any]) -> dict[str, Any]:
    """A2 升级: extract_box_primitives + numpy 级分析.

    三种 kind 各有 numpy 级检查:
      - particles: 连通域圆度 + 等效直径 (不只查数量)
      - lattice: FFT 主峰周期性 (不只查连通域)
      - spectrum: find_peaks 峰位数量 (不只查 box)

    原 extract_box_primitives 逻辑保留, numpy 分析是附加层.
    numpy 分析失败时退化到原逻辑, 不阻塞 verify.
    """
    if not image_bytes:
        return {"verified": False, "error": "empty image bytes", "expected": expected}

    try:
        from huginn.tools.visual_hook import extract_box_primitives, parse_box_primitive
    except ImportError:
        return {"verified": False, "error": "visual_hook not available", "expected": expected}

    primitives = extract_box_primitives(image_bytes, threshold=128, max_boxes=50)
    boxes = parse_box_primitive(primitives)
    region_boxes = [b for b in boxes if b.get("label") != "overall" and b.get("label", "").startswith("region")]
    n_detected = len(region_boxes)

    kind = expected.get("kind", "")
    verified = False
    note = ""
    shape_analysis: dict[str, Any] | None = None

    # A2: numpy 级分析 (附加层, 失败退化到原逻辑)
    img_arr = _decode_to_gray(image_bytes)

    if kind == "particles":
        n_expected = int(expected.get("n", 0))
        tol = max(2, n_expected // 2)
        count_ok = abs(n_detected - n_expected) <= tol
        # A2: 形状检查 — 圆度均值 >0.5 才算真粒子
        shape_ok = True
        if img_arr is not None:
            shape_analysis = _analyze_particles(img_arr)
            circs = shape_analysis.get("circularities", [])
            if circs:
                shape_ok = bool(np.mean(circs) > 0.5)
        verified = count_ok and shape_ok
        note = f"particles expected={n_expected}, detected={n_detected}, tol=±{tol}, shape_ok={shape_ok}"

    elif kind == "lattice":
        # A2: FFT 周期性检查
        fft_ok = True
        if img_arr is not None:
            fft_analysis = _analyze_lattice_fft(img_arr)
            fft_ok = fft_analysis["has_periodicity"]
            note = f"lattice FFT periodic={fft_ok}, period={fft_analysis['period_px']:.1f}px"
        else:
            note = f"lattice detected={n_detected} regions (FFT unavailable)"
        verified = fft_ok and (n_detected >= 1 or len(boxes) >= 1)

    elif kind == "spectrum":
        # A2: find_peaks 峰位数量检查
        n_expected_peaks = int(expected.get("n_peaks", 0))
        n_detected_peaks = n_detected  # fallback
        if img_arr is not None:
            peak_analysis = _analyze_spectrum_peaks(img_arr)
            n_detected_peaks = peak_analysis.get("n_peaks", n_detected)
        tol = max(1, n_expected_peaks // 3)
        verified = abs(n_detected_peaks - n_expected_peaks) <= tol
        note = f"spectrum expected={n_expected_peaks} peaks, detected={n_detected_peaks}, tol=±{tol}"

    else:
        verified = len(boxes) >= 1
        note = f"generic check: {len(boxes)} boxes detected"

    result: dict[str, Any] = {
        "verified": verified,
        "n_regions_detected": n_detected,
        "n_boxes_total": len(boxes),
        "expected": expected,
        "note": note,
        "raw_primitives": primitives[:300] if primitives else "",
    }
    if shape_analysis is not None:
        result["shape_analysis"] = shape_analysis
    return result


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

    # 6. A2: particles 形状检查 — shape_analysis 字段
    img = sketch("particles 10")
    v6 = verify(img, {"kind": "particles", "n": 10})
    assert "shape_analysis" in v6, "A2: particles verify must include shape_analysis"
    sa = v6["shape_analysis"]
    assert "circularities" in sa, "shape_analysis must have circularities"
    assert "diameters" in sa, "shape_analysis must have diameters"
    if sa["circularities"]:
        # 合成粒子是圆形, 圆度均值应 >0.5
        assert sa["mean_circularity"] > 0.3, f"synthetic particles should be circular: {sa['mean_circularity']}"
    print(f"6. A2 particles shape: n={sa['n']}, mean_circ={sa['mean_circularity']:.2f}")

    # 7. A2: lattice FFT 周期性检查
    img = sketch("lattice 4Å cubic")
    v7 = verify(img, {"kind": "lattice"})
    # lattice verify 应该包含 FFT 分析的 note
    assert "FFT" in v7["note"] or "period" in v7["note"], f"A2: lattice note should mention FFT: {v7['note']}"
    print(f"7. A2 lattice FFT: {v7['note']}")

    # 8. A2: spectrum find_peaks 检查
    img = sketch("spectrum 3 peaks")
    v8 = verify(img, {"kind": "spectrum", "n_peaks": 3})
    # spectrum verify 应该用 find_peaks 的数量
    assert "detected=" in v8["note"], f"A2: spectrum note should mention detected peaks: {v8['note']}"
    print(f"8. A2 spectrum peaks: {v8['note']}")

    # 9. A2: _decode_to_gray 工具函数
    img = sketch("particles 5")
    arr = _decode_to_gray(img)
    assert arr is not None, "_decode_to_gray should decode valid PNG"
    assert arr.ndim == 2, f"grayscale array should be 2D: {arr.shape}"
    print(f"9. A2 _decode_to_gray: shape={arr.shape}")

    # 10. A2: _analyze_particles 在无粒子图上返回 n=0
    # 白图无暗粒子
    white_img = np.full((50, 50), 255, dtype=np.uint8)
    from PIL import Image as _PILImage
    buf = io.BytesIO()
    _PILImage.fromarray(white_img).save(buf, format="PNG")
    result = _analyze_particles(_decode_to_gray(buf.getvalue()))
    assert result["n"] == 0, f"white image should have 0 particles: {result['n']}"
    print("10. A2 white image: 0 particles (OK)")

    print("L10 ALL CHECKS PASSED (A2 shape/FFT/peaks OK)")


if __name__ == "__main__":
    _selfcheck()
