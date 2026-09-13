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


if __name__ == "__main__":
    import sys
    use_emb = "--embedding" in sys.argv
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