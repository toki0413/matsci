"""方案① 机制验证: 结构化计划槽是否让 span predictor 更贴近 actual.

对照设计 (干净 A/B, 隔离"槽信息"这一个变量):
  - predictor 固定: 直接用已落盘 /root/.huginn/models/jepa_span_predictor.json
    (对 134 对原始 prediction 训练, 不含槽)。
  - 同一批方法级目标 (curated, givens 在 plan 时确实可知), 测试预测文本两种版本:
      A) 原始 prediction
      B) 原始 prediction + PLAN_SLOTS 块 (仅输入参量 + 公式, 不含答案)
  - 分别算 span surprise (逐预测 span 到最近真实 span 的掩码均值距离)。
  - 若 B 均值 < A 均值, 说明把"plan 时已知的输入/公式"显式追加到预测侧,
    predictor 能据此更贴近 actual —— 机制成立, 运行时采集才有价值。

诚实边界:
  - 这些方法级 prediction 本身已含不少数字 (runtime 文本很富), 故这里 slots
    是**把 givens 规范化**而非"补全空缺数字"; 增益偏保守是预期。
  - slots 只含 givens/公式; detect_leak 确认任何槽不等于 actual 末位答案码。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.train_jepa_predictor import load_pairs, load_encoder
from scripts.train_jepa_span_predictor import _spans, _dist
from huginn.jepa_slots import append_slots, detect_leak

MODEL = Path("/root/.huginn/models/jepa_span_predictor.json")

# 方法级目标的 plan-时已知 givens + 公式 (人工核过 actual, 勿拍脑袋)
CURATED: dict[str, tuple[list[dict], str]] = {
    "spring_period": ([{"label": "m", "value": 0.5, "unit": "kg"},
                       {"label": "k", "value": 8.0, "unit": "N/m"}], "T=2*pi*sqrt(m/k)"),
    "rc_time_constant": ([{"label": "R", "value": 1000, "unit": "ohm"},
                          {"label": "C", "value": 0.001, "unit": "F"}], "tau=R*C"),
    "freefall_distance": ([{"label": "g", "value": 9.81, "unit": "m/s2"},
                           {"label": "t", "value": 2.0, "unit": "s"}], "d=0.5*g*t^2"),
    "sound_wavelength": ([{"label": "v", "value": 343, "unit": "m/s"},
                          {"label": "f", "value": 440, "unit": "Hz"}], "lambda=v/f"),
    "thermal_expansion": ([{"label": "alpha", "value": 1.2e-05, "unit": "1/K"},
                           {"label": "L0", "value": 2.0, "unit": "m"},
                           {"label": "dT", "value": 50.0, "unit": "K"}], "dL=alpha*L0*dT"),
    "root_mean_square_identity": ([{"label": "a", "value": 3.0},
                                   {"label": "b", "value": 4.0}], "RMS=sqrt((a^2+b^2)/2)"),
    "newton_raphson_sqrt2": ([{"label": "x0", "value": 1.5}], "x_{n+1}=(x_n+2/x_n)/2"),
    "projectile_max_range_angle": ([{"label": "v0", "value": 10, "unit": "m/s"},
                                    {"label": "g", "value": 9.81, "unit": "m/s2"}],
                                   "R(theta)=v0^2*sin(2theta)/g"),
    "newton_2d_system_solve": ([{"label": "target", "value": 3.0}], "min f(x) -> x=target"),
}


def load_model() -> dict:
    data = json.loads(MODEL.read_text(encoding="utf-8"))
    return {"weights": {k: np.asarray(v, dtype=np.float64) for k, v in data["weights"].items()},
            "dim": int(data["dim"])}


def main() -> None:
    enc = load_encoder()
    pairs = load_pairs(Path("/root/.huginn/corpus/jepa_pairs.jsonl"))
    by_obj: dict[str, list[dict]] = {}
    for p in pairs:
        by_obj.setdefault(p.get("objective", ""), []).append(p)
    pack = load_model()

    def vec_of(spans):
        # 逐 span 编码+归一, 与训练端一致
        if not spans:
            return np.zeros((0, pack["dim"]), dtype=np.float64)
        vs = enc.encode(spans, normalize_embeddings=True)
        return np.asarray(vs, dtype=np.float64)

    def surprise(pred_spans, act_spans):
        P = vec_of(pred_spans)
        A = vec_of(act_spans)
        if P.shape[0] == 0 or A.shape[0] == 0:
            return None
        H = np.tanh(P @ pack["weights"]["W1"] + pack["weights"]["b1"])
        fwd = H @ pack["weights"]["W2"] + pack["weights"]["b2"]
        d = [min(_dist(fwd[k], A[j]) for j in range(A.shape[0])) for k in range(P.shape[0])]
        return float(np.mean(d)) if d else None

    orig_vals, slot_vals = [], []
    leaked = 0
    rows = []
    for obj, (inputs, formula) in CURATED.items():
        for p in by_obj.get(obj, []):
            pred = p["prediction"]
            act = p["actual"]
            if detect_leak(inputs, act):
                leaked += 1
                continue
            act_spans = _spans(act)
            s_orig = surprise(_spans(pred), act_spans)
            slotted = append_slots(pred, inputs, formula)
            s_slot = surprise(_spans(slotted), act_spans)
            if s_orig is None or s_slot is None:
                continue
            orig_vals.append(s_orig)
            slot_vals.append(s_slot)
            rows.append((obj, s_orig, s_slot))
        else:
            pass

    if not orig_vals:
        print("无可用配对 (可能全部漏检). 退出.")
        return
    o = float(np.mean(orig_vals)); s = float(np.mean(slot_vals))
    wins = sum(1 for r in rows if r[2] < r[1] - 1e-6)
    print(f"方法级 panel={len(orig_vals)} 对; 泄漏丢弃={leaked} 对")
    print(f"[A] 原始找槽: mean surprise = {o:.4f}")
    print(f"[B] PLAN_SLOTS: mean surprise = {s:.4f}")
    print(f"delta = {s - o:+.4f}; 逐对 slot<orig: {wins}/{len(rows)}")
    for obj, a, b in rows:
        print(f"  {obj:30s} A={a:.4f}  B={b:.4f}  d={b-a:+.4f}")


if __name__ == "__main__":
    main()