"""声明门禁统一化单测 — 锁住「单一实现」契约.

背景: 曾有一段时期每个 demo 各自 bootstrap 一份 claim-gate (``_load_gate``),
导致同一套证伪逻辑被复制多份, 改一处漏几处。统一化方案把唯一实现收敛到
产品模块 ``huginn.research.grounding_verifier()``。

本测试把统一契约锁住:
  1. ``grounding_verifier()`` 是产品公开的单一实现, 且行为等价于
     ``verify_claims(..., allow_derived=True)``(同一底层逻辑)。
  2. 所有 examples/*.py 只从统一入口取门禁, 不得自带 ``_load_gate`` 式加载。
  3. 行为语义: 落地放行 / 杜撰拦截 / 轨迹真值推导放行。

为避免触发 ``huginn.validation`` 包级的重依赖 import 链(生产环境里由
langchain 等承担), 这里加载 ``claim_grounding.py`` 的**单文件**方式与
``grounding_verifier`` 的兜底路径一致: 直接 spec 加载该模块文件, 不经过包
__init__。零网络/零 LLM, 确定性。
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

from huginn.research import grounding_verifier

_ROOT = Path(__file__).resolve().parents[1]
_EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def _load_standalone_verify_claims():
    """与 grounding_verifier 兜底一致的『单文件』加载, 绕开包级重依赖."""
    src = _ROOT / "huginn/validation/claim_grounding.py"
    spec = importlib.util.spec_from_file_location("_cg", str(src))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.verify_claims


# 含"轨迹真值 + 由真值推导的衍生量 + 杜撰数值"的报告, 用于锁定语义.
_TRACE = [
    '{"hot_spot_offset_deg": 12.3, "day_night_delta_T_K": 421.5, "equatorial_jet_ms": 0.62}',
    '{"AIC": -30.941, "R2": 0.997}',
]
_GROUNDED = (
    "昼夜温差 ΔT=421.5 K, 热点东移 12.3°, 喷射 0.62 m/s; "
    "比值 421.5/0.62≈679.8。"
)


def test_product_single_source_is_callable() -> None:
    assert callable(grounding_verifier())


def test_single_source_matches_verify_claims_allow_derived() -> None:
    """统一实现必须与 verify_claims(allow_derived=True) 同一语义."""
    a = grounding_verifier()(_GROUNDED, _TRACE)
    b = _load_standalone_verify_claims()(_GROUNDED, _TRACE, allow_derived=True)
    assert a["verdict"] == b["verdict"] == "pass"
    assert a == b  # 键与数值完全一致 → 确认就是同一实现


def test_single_source_rejects_fabricated_numbers() -> None:
    fake = "昼夜温差 ΔT=999.0 K, 杜撰 AIC=88.88, 热点东移 777.0°。"
    r = grounding_verifier()(fake, _TRACE)
    assert r["verdict"] == "needs_grounding"
    assert set(r["unsubstantiated"]) >= {999.0, 88.88, 777.0}


def test_single_source_allows_trace_derived_values() -> None:
    """由轨迹真值 +/−/×/÷ 推出的衍生量应放行(仍可证伪, 非杜撰)."""
    r = grounding_verifier()("喷射 0.62 m/s; 量纲比值: ΔT/喷射 ≈ 679.8。", _TRACE)
    assert r["verdict"] == "pass", r
    assert any(abs(v - 679.8) < 0.02 for v in r["derived"])


# ── 统一化契约: examples 不得自带门禁加载 ─────────────────────────────
def test_examples_import_unified_gate_only() -> None:
    """所有 examples/*.py 只从 huginn.research 取门禁; 一旦有人回退到
    `_load_gate` 式各自 bootstrap, 此测试立即红。"""
    targets = [
        "ai4s_hotjupiter_demo.py",
        "ai4s_numerics_demo.py",
        "ai4s_internlm_demo.py",
        "ai4s_realdata_demo.py",
    ]
    missing: list[str] = []
    relapsed: list[str] = []
    for fn in targets:
        src = (_EXAMPLES / fn).read_text(encoding="utf-8")
        if "from huginn.research import grounding_verifier" not in src:
            missing.append(fn)
        for bad in ("def _load_gate", "huginn.validation.claim_grounding",
                    "spec_from_file_location"):
            if bad in src:
                relapsed.append(f"{fn}:{bad}")
    assert not missing, f"缺统一门禁导入: {missing}"
    assert not relapsed, f"检测到旧式就地加载: {relapsed}"