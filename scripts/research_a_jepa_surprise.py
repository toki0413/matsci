"""research_a_jepa_surprise —— 用 JEPA 连续 surprise 当终态, 测分形吸引域 (方案A 研究).

背景: 方案①b 用 completion_auditor 布尔指纹当"吸引域"得到 β≈0 (布尔门限不是连续
吸引域, f(ε) 饱和不衰减)。研究 A 换用 engine_reflect 的**连续** `_compute_surprise`
(预测文本 vs 实际文本的语义差异, 弱机回落 Jaccard, 纯字符串零 LLM)。

关键区别:
  布尔分类器: 离散跳变, 深度扰动即饱和 → 测不出分形
  连续 surprise: 连续值 [0,1], 可扫扰动 ε 看是否随 ε 平滑/跳变 → 合法测分形边界

方法(A 具体化):
  1. 固定实际文本 actual (真实结果如 "stress is 0.05 MPa PASS")
  2. 预测文本在一条"连续扰动轨迹"上偏移: 从完全正确的预测, 逐渐换词/删词, 制造
     越来越大的语义偏离, 参数化为 t∈[0,1]
  3. 对每个 t 算 surprise(t) = antipode(pred(t), actual), 得曲线 s(t)
  4. 看 s(t) 是否处处平滑 (分形特征: 在极小区间内的微小扰动导致 surprise 突变)
     → 用"区间扫描": 对固定 m, 检查连续 t 处 surprise 的跳变度 max |s(t)-s(t+δ)|
     → 若随 δ 缩小, 跳变不衰减 → 存在分形边界

零依赖: 复刻 _compute_surprise(遵守 Jaccard 回落), 不 import 完整 engine
(避免句子编码器拉下载)。这是"用真 JEPA 逻辑、假数据"的研究路径, 后续换 ST 即可。
"""
from __future__ import annotations

import re
import statistics


# ── 从 engine_reflect 复刻(遵守其回落路径)的 surprise 计算 ───────
_STOP1 = {"the","a","an","is","are","was","were","be","to","of","in","on",
          "at","for","and","or","not","this","that","it","with","from","by",
          "as","will","can","may"}
_STOP2 = _STOP1 | {"energy","result","value","system","model","data","using",
                   "shown","show","also","which","has","have","had","been",
                   "more","than"}

_WORDS = re.compile(r"[a-zA-Z_]\w{2,}")

def _kw(text, stop, min_len=3):
    return {w for w in _WORDS.findall(text.lower()) if w not in stop and len(w) >= min_len}

def _bg(text):
    w = _WORDS.findall(text.lower())
    return {f"{w[i]}_{w[i+1]}" for i in range(len(w)-1)}

def _jac(a, b):
    if not a and not b:
        return 0.0
    u = a | b
    return 1.0 - len(a & b) / len(u) if u else 0.0


# ── 连续语义 surprise: ST 可用→cosine, 否则回落 Jaccard ──────────
import os as _os

_ST_SINGLETON = [None]  # 缓存 ST 实例(避免逐点重建)

def _get_st():
    """优先用独立 ST 单例(本脚本自建缓存); 未装/失败则 None → 回落 Jaccard."""
    if _ST_SINGLETON[0] is not None:
        return _ST_SINGLETON[0]
    try:
        from sentence_transformers import SentenceTransformer
        model = _os.environ.get("HUGINN_ST_MODEL", "paraphrase-multilingual-mpnet-base-v2").strip()
        _ST_SINGLETON[0] = SentenceTransformer(model)
        return _ST_SINGLETON[0]
    except Exception:
        _ST_SINGLETON[0] = None
        return None

def _embed(encoder, text):
    try:
        import numpy as np
        v = encoder.encode([text], normalize_embeddings=True)[0]
        return np.asarray(v, dtype=np.float32)
    except Exception:
        return None

def _cosine_distance(a, b) -> float:
    import numpy as np
    va = np.asarray(a, dtype=np.float32).ravel()
    vb = np.asarray(b, dtype=np.float32).ravel()
    if va.size == 0 or vb.size == 0:
        return 0.0
    na = va / (np.linalg.norm(va) + 1e-12)
    nb = vb / (np.linalg.norm(vb) + 1e-12)
    return float(max(0.0, min(1.0, 1.0 - float(np.dot(na, nb)))))

def _embed_batch(encoder, texts):
    """批量 encode 所有扰动文本 → (N, dim) 归一化向量阵."""
    import numpy as np
    vs = encoder.encode(list(texts), normalize_embeddings=True,
                        batch_size=64, show_progress_bar=False)
    return np.asarray(vs, dtype=np.float32)


def surprise(prediction: str, actual: str, use_embedding: bool = False) -> dict:
    """copy engine Jaccard 回落路径; use_embedding=True 时优先语义 cosine."""
    if use_embedding:
        enc = _get_st()
        va, vb = (_embed(enc, prediction) if enc else None), \
                 (_embed(enc, actual) if enc else None)
        if va is not None and vb is not None:
            d = _cosine_distance(va, vb)
            return {"point": d, "mean": d, "worst": d, "semantic": True}
    # 回落 Jaccard (同 engine)
    p1, a1 = _kw(prediction,_STOP1), _kw(actual,_STOP1)
    p2, a2 = _kw(prediction,_STOP2), _kw(actual,_STOP2)
    p3, a3 = _kw(prediction,_STOP1,min_len=5), _kw(actual,_STOP1,min_len=5)
    pb, ab = _bg(prediction), _bg(actual)
    est = [_jac(p1,a1), _jac(p2,a2), _jac(p3,a3),
           _jac(pb,ab) if pb or ab else 0.0]
    return {"point": est[0], "mean": statistics.mean(est), "worst": max(est), "semantic": False}


# ── 扰动轨迹: 预测文本沿语义偏离方向连续偏移 ───────────────────
# actual: 正确答案。预测从"几乎正确"逐渐换/删/扰词, 制造连续语义偏离。
_ACTUAL = "the applied tensile stress is 0.05 MPa which is below the 250 MPa yield, so PASS"
_TOKENS = _ACTUAL.split()

def predict_at(t: float) -> str:
    """把 actual 的 token 按 t∈[0,1] 替换成扰动词, t 越大偏离越多.

    用确定性 token 替换(基于 t 选取要换的位置), 制造连续偏离轨迹, 可复现。
    """
    n = len(_TOKENS)
    nt = int(t * (n - 3))          # 被改的 token 数, t=0 不改, t=1 几乎全改
    rng = __import__("random").Random(7)
    idx = set(rng.sample(range(n), nt)) if nt else set()
    out = []
    # 扰动词: 同域但错误的值/词
    distract = {"0.05":"9.99","MPa":"kPa","250":"5","PASS":"FAIL",
                "stress":"strain","below":"above","applied":"removed",
                "tensile":"shear"}
    for i, w in enumerate(_TOKENS):
        if i in idx:
            out.append(distract.get(w, w[::-1]))  # 反转 = 混叠扰动
        else:
            out.append(w)
    return " ".join(out)


def scan_surprise_curve(t_grid: int = 400, use_embedding: bool = False) -> dict:
    """扫 t 得 surprise 曲线, 测跳变度随分辨率缩小是否衰减."""
    ts = [i / t_grid for i in range(t_grid + 1)]
    svals = [surprise(predict_at(t), _ACTUAL, use_embedding=use_embedding)["mean"] for t in ts]
    max_jump = max(abs(svals[i] - svals[i-1]) for i in range(1, len(svals)))
    coarse = [surprise(predict_at(i/100), _ACTUAL, use_embedding=use_embedding)["mean"] for i in range(101)]
    coarse_jump = max(abs(coarse[i]-coarse[i-1]) for i in range(1, len(coarse)))
    return {
        "n_points": len(svals),
        "max_jump_fine": max_jump,
        "max_jump_coarse": coarse_jump,
        "jump_ratio": coarse_jump / max_jump if max_jump else float("inf"),
        "s_range": (min(svals), max(svals)),
    }


# ── 不确定性指数 β 估计 (Chern–Markus): 二维参数空间 P_cross(δ) ~ δ^β ──
# 之前只有 max_jump(2 个分辨率), 不能定形分/平滑。这里做真正的多尺度标度律:
#   1. 二维扰动参数 (u,v), 在 [0,1]² 网格上生成预测文本 → 语义嵌入
#   2. 定义两个自洽终态吸引域: CA=正确PASS文本, CB=FAIL域代表文本
#   3. 归属 g = argmin( dist(pred,CA), dist(pred,CB) )  (无任意阈值)
#   4. 对网格间距 δ 采样两点, 统计落入不同吸引域比例 P_cross(δ)
#   5. 拟合 log P_cross ~ β·log δ → β<1 分形边界, β≈1 平滑边界
# 对照(校验估计器): 光滑对照 g=u<0.5 应出 β≈1; 噪声对照 g=random 应出 β≈0。

# 两个自洽终态吸引域中心文本
_CA = _ACTUAL  # PASS 域(正确解
_CB = "the applied shear stress is 9.99 MPa which is above the 5 MPa yield, so FAIL"

# 近A域扰动词(把语义推向偏离但不翻域) / 远BI域扰动词(推向 FAIL 域)
_NEAR_POOL = {"tensile", "0.05", "below"}      # ← 近 A 的错值
_NEAR_DIST = {"tensile": "shear", "0.05": "0.06", "below": "just_below"}
_FAR_POOL = {"PASS", "0.05", "250", "below", "MPa"}  # ← 翻到 FAIL，推向 CB
_FAR_DIST = {"PASS": "FAIL", "0.05": "9.99", "250": "5", "below": "above", "MPa": "GPa"}


def _disturb(indices, rng, dist, text_tokens, mask=None):
    """把 text_tokens 中 indices 命中的 token 换成 dist 扰动."""
    out = list(text_tokens)
    for i in indices:
        if mask is not None and not mask[i]:
            continue
        w = out[i]
        out[i] = dist.get(w, w + "_x")
    return out


def gen_pred_text(u: float, v: float, seed: int = 11):
    """二维扰动: u→近A域替换量, v→远B域替换量. 输出一条扰动预测文本."""
    toks = _CA.split()
    rng = __import__("random").Random(seed)
    near_i = [i for i, w in enumerate(toks) if w in _NEAR_POOL]
    far_i = [i for i, w in enumerate(toks) if w in _FAR_POOL]
    # v 越高越向 B 翻; 真实语义上部分阈值位翻越.CA RNG 越大扰动越难预测
    kn = int(u * len(near_i))
    kf = int(v * len(far_i))
    rng.shuffle(near_i)
    rng.shuffle(far_i)
    toks = _disturb(near_i[:kn], rng, _NEAR_DIST, toks)
    toks = _disturb(far_i[:kf], rng, _FAR_DIST, toks)
    return " ".join(toks)


def _semantic_labels(grid_n: int) -> list[list[str]]:
    """返回 [0,1]² 网格上逐点的扰动预测文本矩阵 (行序 v, 列序 u)."""
    return [[gen_pred_text(u, v) for v in _u01(grid_n)] for u in _u01(grid_n)]


def _u01(n):
    return [i / (n - 1) if n > 1 else 0.5 for i in range(n)]


def assign_labels_flat(e, ca_enc, cb_enc, grid_n):
    """按最小语义距把每条预测指派到 CA 或 CB 吸引域, 返回 (grid_n,grid_n) 标签阵."""
    import numpy as np
    dA = 1.0 - np.dot(e, ca_enc)  # e:(N,d) 已归一; ca_enc:(d,)
    dB = 1.0 - np.dot(e, cb_enc)
    g = np.where(dA <= dB, 0, 1).reshape(grid_n, grid_n)
    return g


def _p_cross(g, grid_n, delta_frac, rng, samples=8000):
    """在网格上采样相距δ处的点对, 返回被不同吸引域的占比 P_cross(δ)."""
    d = max(1, int(round(delta_frac * (grid_n - 1))))
    steps = [(-d, 0), (d, 0), (0, -d), (0, d), (d, d), (-d, d), (d, -d), (-d, -d)]
    cnt = tot = 0
    for _ in range(samples):
        i = int(rng.integers(0, grid_n))
        j = int(rng.integers(0, grid_n))
        di, dj = rng.choice(steps)
        i2, j2 = i + di, j + dj
        if 0 <= i2 < grid_n and 0 <= j2 < grid_n:
            tot += 1
            if g[i, j] != g[i2, j2]:
                cnt += 1
    return cnt / tot if tot else 0.0


def _fit_beta(deltas, pcs) -> dict:
    """log P_cross ~ β log δ 线性拟合(忽略 P=0/1 饱和端点)."""
    import math
    import numpy as np
    pts = [(math.log(d), math.log(max(p, 1e-9))) for d, p in zip(deltas, pcs)
           if 0.0 < p < 1.0]
    if len(pts) < 3:
        return {"beta": float("nan"), "r2": float("nan"), "n_used": len(pts)}
    xs = [x for x, _ in pts]
    ys = [y for _, y in pts]
    slope, intercept = np.polyfit(xs, ys, 1)
    yhat = [slope * x + intercept for x in xs]
    ss_res = sum((y - yh) ** 2 for y, yh in zip(ys, yhat))
    ss_tot = sum((y - sum(ys) / len(ys)) ** 2 for y in ys)
    r2 = 1.0 - ss_res / ss_tot if ss_tot else float("nan")
    return {"beta": float(slope), "r2": float(r2), "n_used": len(pts)}


DELTAS = [0.02, 0.04, 0.06, 0.08, 0.12, 0.16, 0.25, 0.33]


def estimate_beta(grid_n: int = 110, seed: int = 3) -> dict:
    """跑完整 β 估计: 真标签 + 光滑对照 + 噪声对照."""
    import numpy as np
    import random
    enc = _get_st()
    if enc is None:
        return {"error": "需要 --embedding(ST 不可用)"}
    texts_2d = _semantic_labels(grid_n)
    flat = [t for row in texts_2d for t in row]
    e = _embed_batch(enc, flat)
    ca_v = _embed_batch(enc, [_CA])[0]
    cb_v = _embed_batch(enc, [_CB])[0]
    g_true = assign_labels_flat(e, ca_v, cb_v, grid_n)

    # 光滑对照: 垂直边界 → β≈1
    g_smooth = np.asarray([[0 if u < 0.5 else 1 for _ in _u01(grid_n)]
                            for u in _u01(grid_n)], dtype=int)
    # 噪声对照: 随机标签 → β≈0 (单 rng 递增, 不要每行重置 seed)
    rng_noise = random.Random(seed)
    g_noise = np.asarray(
        [[rng_noise.randint(0, 1) for _ in _u01(grid_n)] for _ in _u01(grid_n)],
        dtype=int,
    )

    out = {}
    for name, gg in [("真实吸引域", g_true), ("光滑对照", g_smooth), ("噪声对照", g_noise)]:
        rng = np.random.default_rng(seed)
        pcs = [_p_cross(gg, grid_n, d, rng) for d in DELTAS]
        fit = _fit_beta(DELTAS, pcs)
        out[name] = {"deltas": DELTAS, "p_cross": pcs, **fit}
    return {"grid_n": grid_n, **out}


if __name__ == "__main__":
    import sys
    use_emb = "--embedding" in sys.argv

    # ── β 估计(不确定性指数): 多尺度跨边界概率标度律 ──
    if "--beta" in sys.argv:
        if not use_emb:
            print("β 估计需要 --embedding(ST 语义), 否则退化为词集离散无意义。")
            sys.exit(1)
        print("=" * 66)
        print("不确定性指数 β 估计: P_cross(δ) ~ δ^β   (Chern–Markus)")
        print("=" * 66)
        res = estimate_beta()
        if "error" in res:
            print("错误:", res["error"])
            sys.exit(2)
        print(f"二维扰动网格 {res['grid_n']}×{res['grid_n']}  两点抽样 8000/δ\n")
        for name, r in res.items():
            if name == "grid_n":
                continue
            n_used = r.get("n_used", 0)
            sign = ("分形边界" if r["beta"] < 0.85 else
                    "平滑(β≈1)" if r["beta"] > 0.95 else "模糊")
            print(f"  {name:<8}  β={r['beta']:+.3f}  R²={r['r2']:.3f}  "
                  f"(拟合点 {n_used}/{len(DELTAS)})  → {sign}")
        b = res.get("真实吸引域", {}).get("beta")
        print("-" * 66)
        if b is not None and b < 0.85:
            print(f"  β={b:+.3f} < 1 → JEPA语义失败边界呈分形, 假说获得定量支持")
        else:
            print(f"  β={b if b is not None else float('nan'):+.3f} ≈ 1 → 边界平滑, 假说未获支持(非分形)")
        sys.exit(0)

    mode = "语义cosine" if use_emb else "Jaccard回落"
    print("=" * 66)
    print(f"研究 A: JEPA surprise 当吸引域 —— 模式: {mode}")
    print("=" * 66)
    # 单点 sanity
    print("sanity 预测=实际:", round(surprise(_ACTUAL, _ACTUAL, use_embedding=use_emb)["mean"], 3))
    print("sanity 完全偏离:", round(surprise("totally different answer yes nothing", _ACTUAL, use_embedding=use_emb)["mean"], 3))

    r = scan_surprise_curve(use_embedding=use_emb)
    print(f"\n扫描 surprise(t): {r['n_points']} 点")
    print(f"  surprise 范围: {r['s_range'][0]:.3f} ~ {r['s_range'][1]:.3f}")
    print(f"  细分辨率 max_jump = {r['max_jump_fine']:.3f}")
    print(f"  粗分辨率 max_jump = {r['max_jump_coarse']:.3f}")
    label = r["max_jump_fine"]>0 and r["jump_ratio"]>0.8
    print(f"  jump_ratio(fine≈coarse? 不随分辨率衰减) = {r['jump_ratio']:.2f}")
    if use_emb:
        if r["max_jump_fine"]>0.05:
            print(f"  → {mode}: 连续域仍有非零跳变且不衰减 → 倾向于存在 canvas 边界(分形信号)")
        else:
            print(f"  → {mode}: 连续域细分辨率跳变≈0 → 边界光滑, 非分形")
    else:
        print(f"  → {mode}: 平台跳变 = Jaccard 词集离散伪影, 非真分形 (需 --embedding)")