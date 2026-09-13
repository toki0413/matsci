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

def surprise(prediction: str, actual: str) -> dict:
    """复刻 _compute_surprise_robust 的 Jaccard 回落: 1 - jaccard = 语义距离."""
    p1, a1 = _kw(prediction,_STOP1), _kw(actual,_STOP1)
    p2, a2 = _kw(prediction,_STOP2), _kw(actual,_STOP2)
    p3, a3 = _kw(prediction,_STOP1,min_len=5), _kw(actual,_STOP1,min_len=5)
    pb, ab = _bg(prediction), _bg(actual)
    est = [_jac(p1,a1), _jac(p2,a2), _jac(p3,a3),
           _jac(pb,ab) if pb or ab else 0.0]
    return {"point": est[0], "mean": statistics.mean(est), "worst": max(est)}


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


def scan_surprise_curve(t_grid: int = 400) -> dict:
    """扫 t 得 surprise 曲线, 测跳变度随分辨率缩小是否衰减."""
    ts = [i / t_grid for i in range(t_grid + 1)]
    svals = [surprise(predict_at(t), _ACTUAL)["mean"] for t in ts]
    # 最大相邻跳变 (delta = 1/t_grid)
    max_jump = max(abs(svals[i] - svals[i-1]) for i in range(1, len(svals)))
    # 用更粗分辨率重扫, 若 max_jump 不随分辨率下降而下降 → 存在不连续跳变(分形信号)
    coarse = [surprise(predict_at(i/100), _ACTUAL)["mean"] for i in range(101)]
    coarse_jump = max(abs(coarse[i]-coarse[i-1]) for i in range(1, len(coarse)))
    return {
        "n_points": len(svals),
        "max_jump_fine": max_jump,    # 细分辨率
        "max_jump_coarse": coarse_jump,  # 粗分辨率
        "jump_ratio": coarse_jump / max_jump if max_jump else float("inf"),
        "s_range": (min(svals), max(svals)),
    }


if __name__ == "__main__":
    print("=" * 66)
    print("研究 A: JEPA 连续 surprise 当吸引域 —— 测边界是否分形")
    print("零 LLM: 复刻 _compute_surprise 的 Jaccard 回落路径")
    print("=" * 66)
    # 单点 sanity: 预测=实际 → surprise≈0
    print("sanity 预测=实际:", round(surprise(_ACTUAL, _ACTUAL)["mean"], 3))
    print("sanity 预测=完全偏离:", round(surprise("totally different answer yes", _ACTUAL)["mean"], 3))

    r = scan_surprise_curve()
    print(f"\n扫描 surprise(t): {r['n_points']} 点")
    print(f"  surprise 范围: {r['s_range'][0]:.3f} ~ {r['s_range'][1]:.3f}")
    print(f"  细分辨率 max_jump = {r['max_jump_fine']:.3f}")
    print(f"  粗分辨率 max_jump = {r['max_jump_coarse']:.3f}")
    # 关键判据: fine 分辨率 max_jump 明显 > 0, 说明"极微小扰动也导致 surprise 突变"
    if r["max_jump_fine"] > 0.05:
        print("  → 判据: 细分辨率仍有非零跳变 → 存在尖锐边界(分形信号), 需进一步估 β")
    else:
        print("  → 判据: 细分辨率跳变≈0 → 边界光滑, 非分形")
    print("  注: 这测的是'预测扰动 → surprise 跳变', 是真 agent 能采样的连续指纹。")