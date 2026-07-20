"""传统 CV 算子图像对比 — vision LLM 不可用时的客观相似度评分.

agent 工作流和 RCBench 评分共用同一套算子, 打通"生成图-评分"断层.
四算子等权: SSIM (结构) + HSV histogram correlation (配色) + HOG cosine
(形状/纹理) + ORB keypoint match ratio (关键点).

Source: 用户要求 vision LLM 之前 CV 一样做图像评分, 工作流必须包含 CV.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional


def cv_image_similarity(target_path: Path, generated_path: Path) -> dict:
    """对比 target 和 generated, 返回 {score, details, avg}.

    score ∈ [0, 100], sigmoid 拉伸让中段敏感. 单算子失败跳过, 全失败 → None.
    """
    import numpy as np
    try:
        import cv2
    except ImportError:
        return {"score": None, "details": "cv2 not installed"}
    try:
        from skimage.metrics import structural_similarity as _ssim
    except ImportError:
        _ssim = None

    try:
        tg = cv2.imread(str(target_path))
        gg = cv2.imread(str(generated_path))
        if tg is None or gg is None:
            return {"score": None, "details": "imread failed"}
        h_t, w_t = tg.shape[:2]
        gg_r = cv2.resize(gg, (w_t, h_t), interpolation=cv2.INTER_AREA)
        tg_gray = cv2.cvtColor(tg, cv2.COLOR_BGR2GRAY)
        gg_gray = cv2.cvtColor(gg_r, cv2.COLOR_BGR2GRAY)

        scores: dict[str, float] = {}

        # 1. SSIM — 结构相似度, 抓 layout/edge
        if _ssim is not None:
            try:
                s = _ssim(tg_gray, gg_gray, data_range=255)
                scores["ssim"] = max(0.0, float(s))
            except Exception:
                pass

        # 2. HSV histogram correlation — 配色一致性
        try:
            tg_hsv = cv2.cvtColor(tg, cv2.COLOR_BGR2HSV)
            gg_hsv = cv2.cvtColor(gg_r, cv2.COLOR_BGR2HSV)
            corr_sum = 0.0
            for ch in range(3):
                h_t_ = cv2.calcHist([tg_hsv], [ch], None, [64], [0, 256])
                h_g_ = cv2.calcHist([gg_hsv], [ch], None, [64], [0, 256])
                cv2.normalize(h_t_, h_t_, 0, 1, cv2.NORM_MINMAX)
                cv2.normalize(h_g_, h_g_, 0, 1, cv2.NORM_MINMAX)
                corr_sum += float(cv2.compareHist(h_t_, h_g_, cv2.HISTCMP_CORREL))
            scores["histogram"] = max(0.0, corr_sum / 3.0)
        except Exception:
            pass

        # 3. HOG cosine — 梯度方向直方图, 抓形状/纹理
        try:
            win_size = (64, 128)
            tg_rz = cv2.resize(tg_gray, win_size)
            gg_rz = cv2.resize(gg_gray, win_size)
            hog = cv2.HOGDescriptor(win_size, (16, 16), (8, 8), (8, 8), 9)
            h_t_ = hog.compute(tg_rz).flatten()
            h_g_ = hog.compute(gg_rz).flatten()
            if h_t_.size and h_g_.size:
                cos = float(np.dot(h_t_, h_g_) / (np.linalg.norm(h_t_) * np.linalg.norm(h_g_) + 1e-9))
                scores["hog"] = max(0.0, cos)
        except Exception:
            pass

        # 4. ORB keypoint match ratio — 关键点匹配
        try:
            orb = cv2.ORB_create(nfeatures=500)
            kp1, des1 = orb.detectAndCompute(tg_gray, None)
            kp2, des2 = orb.detectAndCompute(gg_gray, None)
            if des1 is not None and des2 is not None and len(des1) > 0 and len(des2) > 0:
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(des1, des2)
                ratio = len(matches) / max(len(kp1), len(kp2))
                scores["orb"] = min(1.0, ratio)
        except Exception:
            pass

        if not scores:
            return {"score": None, "details": "all CV operators failed"}

        avg = sum(scores.values()) / len(scores)
        # sigmoid 拉伸: 0.3→~17, 0.5→~62, 0.7→~90. 视觉感知中段更敏感.
        # ponytail: 等权平均 + 固定 sigmoid 参数. 天花板: 不同 criterion
        #   对算子敏感度不同 (折线图对 HOG 敏感, colorbar 对 histogram 敏感),
        #   升级路径: 按 criterion keywords 动态加权.
        score = int(100.0 / (1.0 + float(np.exp(-8.0 * (avg - 0.4)))))
        return {"score": score, "details": scores, "avg": float(avg)}
    except Exception as _e:
        return {"score": None, "details": f"cv compare error: {_e}"}


def cv_best_match(target_path: Path, generated_images: list[Path]) -> dict:
    """target 对所有 generated 图算相似度, 取最高. 返回 {score, best_path, details}."""
    best: dict = {"score": None, "best_path": None, "details": []}
    for img in generated_images:
        if not img.exists():
            continue
        r = cv_image_similarity(target_path, img)
        best["details"].append({"img": str(img), **r})
        if r.get("score") is not None:
            if best["score"] is None or r["score"] > best["score"]:
                best["score"] = r["score"]
                best["best_path"] = str(img)
    return best


# === self-check ===
def _self_check() -> int:
    """assert-based demo: 验证 CV 算子在合成图上能跑通."""
    import tempfile
    import numpy as np
    try:
        import cv2
    except ImportError:
        print("[cv_compare] cv2 not installed, skip self-check")
        return 0

    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        # 合成两张相似图 (同一底色 + 不同矩形位置)
        a = np.full((100, 100, 3), 200, dtype=np.uint8)
        cv2.rectangle(a, (20, 20), (60, 60), (50, 50, 50), -1)
        cv2.imwrite(str(tmp_p / "a.png"), a)
        b = a.copy()
        cv2.rectangle(b, (25, 25), (65, 65), (50, 50, 50), -1)  # 略偏移
        cv2.imwrite(str(tmp_p / "b.png"), b)
        c = np.full((100, 100, 3), 10, dtype=np.uint8)  # 完全不同
        cv2.imwrite(str(tmp_p / "c.png"), c)

        r_ab = cv_image_similarity(tmp_p / "a.png", tmp_p / "b.png")
        r_ac = cv_image_similarity(tmp_p / "a.png", tmp_p / "c.png")
        assert r_ab["score"] is not None, f"相似图应能算分: {r_ab}"
        assert r_ac["score"] is not None, f"不同图应能算分: {r_ac}"
        # 相似图分数应高于不同图
        assert r_ab["score"] > r_ac["score"], (
            f"相似图分数应更高: ab={r_ab['score']} ac={r_ac['score']}"
        )

        # best_match
        bm = cv_best_match(tmp_p / "a.png", [tmp_p / "b.png", tmp_p / "c.png"])
        assert bm["score"] == r_ab["score"], f"best_match 应选 b: {bm}"
        assert "b.png" in (bm["best_path"] or ""), f"best_path 应是 b: {bm}"

    print(f"[cv_compare] self-check OK (similar>{r_ab['score']}, diff={r_ac['score']})")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_self_check())
