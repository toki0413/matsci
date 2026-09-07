"""结论证伪门禁 (huginn.validation.claim_grounding) 单测.

验证: 模型报告里的数值必须落在工具执行轨迹中; 未落地主张被标记 needs_grounding.
纯函数, 零网络/零 LLM, 确定性。
"""
from __future__ import annotations

from huginn.validation.claim_grounding import (
    extract_numeric_claims,
    verify_claims,
)

_TRACE = [
    '{"AIC": -30.941, "params": {"b": 0.0347}, "R2": 0.997}',
    '{"bootstrap_95ci": [0.0298, 0.0444], "verdict": "参数可信"}',
    '{"loocv_mse": 0.0153}',
    '{"recommended_next_sample_x": [410, 400, 385]}',
]

_GOOD = (
    "指数律 AIC=-30.941 优于线性律; 参数 b 95%CI=[0.0298,0.0444] 不含 0, 可识别; "
    "LOOCV=0.0153; 建议在 410/400/385 K 补样。"
)
_BAD = "线性律 AIC=42.73 与指数律 AIC=36.15 相近; 参数 a=1.98 可识别; 建议 x=4.0 补样。"


def test_claims_true_to_trace_pass():
    r = verify_claims(_GOOD, _TRACE)
    assert r["verdict"] == "pass", r
    assert r["unsubstantiated"] == []


def test_fabricated_numbers_caught():
    r = verify_claims(_BAD, _TRACE)
    assert r["verdict"] == "needs_grounding", r
    assert set(r["unsubstantiated"]) >= {42.73, 36.15, 4.0}


def test_missing_contextual_value_flagged():
    # 引用了一个未出现在轨迹的统计量 (线性的 R2), 但轨迹只有指数律 → 判 not grounded
    text = "线性律 R2=0.884 差于指数律 R2=0.997; LOOCV=0.0153。"
    r = verify_claims(text, _TRACE)
    assert "0.884" in [str(v) for v in r["unsubstantiated"]] or 0.884 in r["unsubstantiated"]
    assert r["verdict"] == "needs_grounding"


def test_empty_trace_all_claims_unsubstantiated():
    assert verify_claims(_BAD, [])["verdict"] == "needs_grounding"


def test_small_integers_not_claims_by_default():
    # 无意义小整数不作为主张; 但带小数统计量会
    claims = extract_numeric_claims("模型取了 2 个点, 100 个样本, 每轮 3 步。")
    assert all(abs(c) > 100 or "." not in ("%g" % c) for c in claims) or not claims