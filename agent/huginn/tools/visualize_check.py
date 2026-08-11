"""图↔数据一致性校验 — 渲染式验证闭环的数据维度 (批次 B).

对应 WorldClaw render-based refinement「检测对象是否放对位置」在图表域的对应物:
图必须反映源数据. 本模块提供:
  1. extract_figure_numeric      — 反提取图上数值快照 (峰位/强度/带隙等)
  2. check_figure_vs_expected    — 图上数值 vs 源数据/figure_ir 容差比对 → numeric_drift
  3. check_figure_duplicate      — 用 ImageIndex 反查图, 检测近重复/冲突图
  4. consistency_verdict         — 组合成整体 verdict

所有校验接收可注入的 extractor / index, 便于确定性单测 (不依赖真实编码器).
纯函数, 不依赖 agent/LLM 状态, 供批次 C 精修闭环消费.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

VERDICT_PASS = "pass"
VERDICT_FIX = "fix_needed"
VERDICT_FAIL = "fail"
VERDICT_ERROR = "error"

# extractor 签名: (image_path: str) -> dict  (来自 symbol_encoder.extract_chart_data 风格)
Extractor = Callable[[str], dict[str, Any]]


def _default_extractor(image_path: str) -> dict[str, Any]:
    """默认提取器: 复用 symbol_encoder.extract_chart_data."""
    from huginn.vision.symbol_encoder import extract_chart_data

    return extract_chart_data(image_path)


def extract_figure_numeric(
    image_path: str | Path, extractor: Extractor | None = None
) -> dict[str, Any]:
    """反提取图上数值快照.

    泛型扫描: 收集 dict 中所有标量数值与纯数值列表字段 (峰位/强度/带隙/2θ…).
    提取失败或 extractor 返回 dict 带 "error" 时返回空 dict (不抛异常).
    """
    ex = extractor or _default_extractor
    try:
        data = ex(str(image_path))
    except Exception:
        logger.debug("visualize_check: extract failed", exc_info=True)
        return {}
    if not isinstance(data, dict) or data.get("error"):
        return {}
    out: dict[str, Any] = {}
    for k, v in data.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[k] = float(v)
        elif (
            isinstance(v, (list, tuple))
            and v
            and all(
                isinstance(x, (int, float)) and not isinstance(x, bool) for x in v
            )
        ):
            out[k] = [float(x) for x in v]
    return out


def check_figure_vs_expected(
    image_path: str | Path,
    expected: dict[str, Any],
    tolerance_pct: float = 10.0,
    extractor: Extractor | None = None,
) -> dict[str, Any]:
    """把图上反提取的数值与 expected 比对, 超容差 → numeric_drift flag.

    expected: {key: float} 或 {key: [floats]} (来自 figure_ir / 源数据).
    图上没提取到某 key 时跳过 (不判 drift, 避免误报).
    返回 {flags, extracted, verdict_ok}.
    """
    extracted = extract_figure_numeric(image_path, extractor=extractor)
    flags: list[dict[str, Any]] = []
    tol = tolerance_pct / 100.0
    for key, exp in expected.items():
        act = extracted.get(key)
        if act is None:
            continue
        if isinstance(exp, (list, tuple)):
            if not isinstance(act, list) or len(exp) != len(act):
                continue
            pairs = list(zip(exp, act))
        else:
            first = act[0] if isinstance(act, list) and act else act
            if first is None:
                continue
            pairs = [(float(exp), first)]
        for ex, ac in pairs:
            ex = float(ex)
            ac = float(ac)
            if abs(ex) < 1e-12:
                continue
            dev = abs(ac - ex) / abs(ex)
            if dev > tol:
                flags.append(
                    {
                        "key": key,
                        "expected": round(ex, 4),
                        "extracted": round(ac, 4),
                        "deviation_pct": round(dev * 100, 1),
                    }
                )
    return {"flags": flags, "extracted": extracted, "verdict_ok": not flags}


def check_figure_duplicate(
    image_path: str | Path,
    index: Any | None,
    threshold: float = 0.92,
    top_k: int = 3,
) -> dict[str, Any]:
    """用 ImageIndex 反查图, 检测近重复图 (同 fig 重复/冲突).

    index 需提供 search(query=..., top_k=...) → entries 带 "similarity" 与 "path".
    index 为 None 或搜索失败 → 保守返回 duplicate=False.
    """
    if index is None:
        return {"duplicate": False, "matches": [], "note": "no index"}
    try:
        results = index.search(query=image_path, top_k=top_k)
    except Exception:
        logger.debug("visualize_check: index search failed", exc_info=True)
        return {"duplicate": False, "matches": [], "note": "search failed"}
    if not results:
        return {"duplicate": False, "matches": [], "note": "no results"}

    matches: list[dict[str, Any]] = []
    for r in results:
        path = r.get("path")
        sim = r.get("similarity")
        if path is None or sim is None:
            continue
        if str(path) == str(image_path):
            continue  # 排除自身
        matches.append({"path": str(path), "similarity": float(sim)})

    dup_matches = [m for m in matches if m["similarity"] >= threshold]
    return {
        "duplicate": bool(dup_matches),
        "matches": matches,
        "duplicate_paths": [m["path"] for m in dup_matches],
    }


def consistency_verdict(
    image_path: str | Path,
    expected: dict[str, Any],
    index: Any | None = None,
    tolerance_pct: float = 10.0,
    extractor: Extractor | None = None,
    dup_threshold: float = 0.92,
) -> dict[str, Any]:
    """组合数据维度一致性 verdict (批次 C 门禁入口).

    判定: 有重复图 → fail; 只有 numeric_drift → fix_needed; 全过 → pass;
    expected 为空且无 index → error (无校验依据).
    """
    if not expected and index is None:
        return {
            "verdict": VERDICT_ERROR,
            "flags": [],
            "error": "no expected values and no index provided",
            "numeric_drift": {"flags": []},
            "duplicate": {"duplicate": False, "matches": []},
        }

    drift = check_figure_vs_expected(
        image_path, expected, tolerance_pct=tolerance_pct, extractor=extractor
    )
    dup = check_figure_duplicate(image_path, index, threshold=dup_threshold)

    flags: list[str] = [f"numeric_drift:{f['key']}" for f in drift["flags"]]
    if dup["duplicate"]:
        flags.append("duplicate_figure")

    if not flags:
        verdict = VERDICT_PASS
    elif dup["duplicate"]:
        verdict = VERDICT_FAIL
    else:
        verdict = VERDICT_FIX

    return {
        "verdict": verdict,
        "flags": flags,
        "numeric_drift": drift,
        "duplicate": dup,
    }
