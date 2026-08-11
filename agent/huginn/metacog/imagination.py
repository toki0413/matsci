"""Imagination — 在 hypothesis manifold 上做 structure-preserving transformation.

不是 pattern completion, 是 extrapolation: 把 hypothesis 的 Bourbaki mother
structure (algebraic / topological / order) 做一次 group action / 拓扑变换 /
偏序变换, 生成一个原 manifold 上不存在的 hypothesis. 跟 interpolation 的区别
靠 fisher distance > sigma 来保证.

失败一律返回 None, 不阻塞主循环. 升级路径:
  - LLM transform     -> Lean 4 形式化 hypothesis + verified structure-preserving map
  - falsifiability    -> Lean 4 tactic 给可证伪 witness
  - fisher distance   -> 真 Fisher info metric (parametric hypothesis)
"""
from __future__ import annotations

# 直接跑脚本时把 agent/ 加到 sys.path (被 import 时不执行, rcb_runner 已设好)
if __name__ == "__main__":
    import sys as _sys
    from pathlib import Path as _Path
    _agent_root = str(_Path(__file__).resolve().parents[2])
    if _agent_root not in _sys.path:
        _sys.path.insert(0, _agent_root)

import json
import logging
import math
import time
from pathlib import Path
from typing import Any

from huginn.metacog.hypothesis_manifold import Hypothesis, HypothesisManifold

logger = logging.getLogger(__name__)

# Bourbaki 三族 — mother structures 的工程映射
_TRANSFORM_TYPES = ("algebraic", "topological", "order")

# interpolation 阈值: 新 h 跟所有现有 h fisher distance 都要 > sigma 才算 extrapolation
_DEFAULT_SIGMA = 0.5

# stagnation: posterior best h_id 连续 N 轮不变 -> 触发 imagination
_DEFAULT_STAGNATION_N = 3


# ---------- LLM prompt 构造 ----------

def _build_transform_prompt(h: Hypothesis, transform_type: str) -> tuple[str, str]:
    """构造 Bourbaki 变换 prompt. 返回 (system, user)."""
    if transform_type == "algebraic":
        sys_text = (
            "You apply a Bourbaki algebraic structure transformation to a scientific "
            "hypothesis: act on its symmetry structure by a group action. "
            "Examples: spin-rotation symmetry -> gauge symmetry; "
            "time-reversal symmetry -> particle-hole symmetry; "
            "discrete C2 rotation -> continuous SO(2). "
            "The new hypothesis must remain falsifiable (have a testable prediction)."
        )
        instruction = (
            "Identify the symmetry structure in the hypothesis, then apply one "
            "non-trivial group action to produce a new hypothesis."
        )
    elif transform_type == "topological":
        sys_text = (
            "You apply a Bourbaki topological structure transformation to a scientific "
            "hypothesis: change its neighborhood assumption. "
            "Examples: nearest-neighbor hopping -> next-nearest-neighbor hopping; "
            "local correlation -> long-range correlation; "
            "short-range interaction -> long-range interaction. "
            "The new hypothesis must remain falsifiable."
        )
        instruction = (
            "Identify the neighborhood/range assumption in the hypothesis, then "
            "flip it (local->long-range or long-range->local) to produce a new hypothesis."
        )
    elif transform_type == "order":
        sys_text = (
            "You apply a Bourbaki order structure transformation to a scientific "
            "hypothesis: change its ordering principle. "
            "Examples: energy minimization -> entropy maximization; "
            "ground-state ordering -> excited-state ordering; "
            "thermodynamic ordering -> kinetic ordering. "
            "The new hypothesis must remain falsifiable."
        )
        instruction = (
            "Identify the ordering principle in the hypothesis, then replace it "
            "with a different ordering to produce a new hypothesis."
        )
    else:
        raise ValueError(f"unknown transform_type: {transform_type}")

    sys_text += (
        "\n\nReturn STRICT JSON:\n"
        '{"new_description": <string>, "new_predictions": {<name>: <number>, ...}, '
        '"new_n_params": <int>, "transform_reason": <short string>}\n'
        "new_predictions must include at least one observable that differs numerically "
        "from the original. new_n_params is the effective parameter count of the new "
        "hypothesis (>=1). Return JSON only, no prose."
    )

    usr = (
        f"Original hypothesis:\n"
        f"  description: {h.description}\n"
        f"  predictions: {json.dumps(h.predictions, ensure_ascii=False)}\n"
        f"  n_params: {h.n_params}\n"
        f"\n{instruction}\n"
    )
    return sys_text, usr


def _build_falsifiability_prompt(h: Hypothesis) -> tuple[str, str]:
    """构造可证伪性检查 prompt. 返回 (system, user)."""
    sys_text = (
        "You are a falsifiability auditor for scientific hypotheses. "
        "A hypothesis is falsifiable iff it predicts at least one observable "
        "whose measurement could refute it. "
        'Return STRICT JSON: {"falsifiable": <bool>, "observable": <string>, "reason": <string>}. '
        "Return JSON only, no prose."
    )
    usr = (
        f"Hypothesis:\n"
        f"  description: {h.description}\n"
        f"  predictions: {json.dumps(h.predictions, ensure_ascii=False)}\n"
        f"\nIs this hypothesis falsifiable? Name the observable that refutes it.\n"
    )
    return sys_text, usr


# ---------- LLM 调用 + JSON 解析 ----------

def _call_llm_sync(model: Any, sys_text: str, usr_text: str) -> str:
    """sync LLM call. 失败抛异常, 由调用方降级. 跟 llm_likelihood 同款."""
    from huginn.metacog.step_evaluator import _build_messages, _resp_to_text
    messages = _build_messages(sys_text, usr_text)
    if hasattr(model, "invoke"):
        return _resp_to_text(model.invoke(messages))
    raise ValueError("model has no sync invoke; ainvoke-only models not supported")


def _parse_first_json(text: str) -> dict | None:
    """从 text 抓第一个平衡的 {...} JSON 对象 (支持嵌套). 失败返回 None.

    ponytail: 手写平衡括号匹配, 不引 json5/retrying. LLM 偶尔在 JSON 外加
    markdown fence / 前后文本, 这里只找第一个平衡的 { ... }.
    """
    if not text:
        return None
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    logger.debug("best-effort op failed", exc_info=True)
                    return None
    return None


def _parse_transform_response(text: str) -> Hypothesis | None:
    """解析 LLM 变换输出 -> Hypothesis. 失败返回 None."""
    obj = _parse_first_json(text)
    if obj is None:
        return None
    desc = obj.get("new_description")
    if not isinstance(desc, str) or not desc.strip():
        return None
    preds = obj.get("new_predictions")
    if not isinstance(preds, dict) or not preds:
        return None
    try:
        preds_clean = {str(k): float(v) for k, v in preds.items()}
    except (TypeError, ValueError):
        logger.debug("best-effort op failed", exc_info=True)
        return None
    n_params = obj.get("new_n_params", 1)
    try:
        n_params = max(1, int(n_params))
    except (TypeError, ValueError):
        logger.debug("best-effort op failed", exc_info=True)
        n_params = 1
    # h_id 用毫秒时间戳, 同迭代内冲突概率极低
    return Hypothesis(
        h_id=f"h_imagine_{int(time.time() * 1000) % 10**9}",
        description=desc.strip(),
        predictions=preds_clean,
        n_params=n_params,
    )


def _parse_falsifiability_response(text: str) -> bool:
    """解析 LLM 可证伪性输出 -> bool. 解析失败保守返回 False (拒绝)."""
    obj = _parse_first_json(text)
    if obj is None:
        return False
    val = obj.get("falsifiable")
    if isinstance(val, bool):
        return val
    return bool(isinstance(val, str) and val.lower() in ("true", "yes", "1"))


# ---------- 公开 API ----------

def check_falsifiability(h: Hypothesis, model: Any = None) -> bool:
    """LLM 检查 hypothesis 是否可证伪. 失败保守返回 False (拒绝).

    升级路径: Lean 4 tactic 形式化 hypothesis + 可证伪 witness.
    无 model 时退化到 "有 prediction 就算可证伪" (prototype 不阻塞).
    """
    if model is None:
        return bool(h.predictions)
    try:
        sys_text, usr_text = _build_falsifiability_prompt(h)
        text = _call_llm_sync(model, sys_text, usr_text)
        return _parse_falsifiability_response(text)
    except Exception as e:
        logger.warning("falsifiability_check_fallback: reason=%s, h=%s", e, h.h_id)
        return False


def _fisher_distance_between(h_i: Hypothesis, h_j: Hypothesis) -> float:
    """两个 Hypothesis 之间的 fisher distance. 跟 manifold.fisher_distance 同公式.

    ponytail: 重复公式避免把 new_h 临时塞进 manifold 改 state. 升级路径:
    manifold.fisher_distance 加一个 Hypothesis 重载 (不靠 h_id 查).
    """
    common = set(h_i.predictions) & set(h_j.predictions)
    if not common:
        return float("inf")
    return math.sqrt(sum((h_i.predictions[k] - h_j.predictions[k]) ** 2 for k in common))


def check_interpolation(
    new_h: Hypothesis,
    manifold: HypothesisManifold,
    sigma: float = _DEFAULT_SIGMA,
) -> bool:
    """检查 new_h 是否是 interpolation (跟某个现有 h 太近).

    Returns True 如果是 interpolation (该拒绝). False 如果是 extrapolation (可接受).
    新 h 跟所有现有 h 的 fisher distance 都 > sigma 才算 extrapolation.
    ponytail: fisher distance 是 prediction-disagreement proxy, 真 linear
    combination 检测需要参数化 hypothesis 空间. 升级路径: 真 Fisher metric.
    """
    if manifold is None:
        return False  # 没 manifold 没法判, 放行
    for existing_h in manifold._hyp.values():
        d = _fisher_distance_between(new_h, existing_h)
        if d <= sigma:
            return True
    return False


def imagine(
    hypothesis: Hypothesis,
    transform_type: str,
    model: Any = None,
) -> Hypothesis | None:
    """在 hypothesis 上做 Bourbaki structure-preserving transformation.

    transform_type in {"algebraic", "topological", "order"}.
    LLM 变换 + 可证伪性检查. 任何失败 (LLM 错误 / JSON 解析 / 不可证伪) 都返回 None.
    """
    if transform_type not in _TRANSFORM_TYPES:
        logger.warning("imagine: unknown transform_type=%s", transform_type)
        return None
    if model is None:
        logger.debug("imagine: no model, cannot do LLM transform")
        return None
    try:
        sys_text, usr_text = _build_transform_prompt(hypothesis, transform_type)
        text = _call_llm_sync(model, sys_text, usr_text)
        new_h = _parse_transform_response(text)
        if new_h is None:
            logger.debug("imagine: transform response unparseable")
            return None
        # 可证伪性检查 — 不可证伪 -> 拒绝
        if not check_falsifiability(new_h, model):
            logger.debug("imagine: new hypothesis not falsifiable, rejected")
            return None
        return new_h
    except Exception as e:
        logger.warning("imagine_fallback: reason=%s, transform=%s", e, transform_type)
        return None


def _write_imagination_log(
    log_path: Path | None,
    *,
    parent_h_id: str | None,
    new_h_id: str | None,
    transform_type: str,
    interpolation_check: str,
    falsifiability_check: str,
    accepted: bool,
) -> None:
    """写一条 entry 到 .huginn/imagination_log.jsonl. 失败静默."""
    if log_path is None:
        return
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": time.time(),
            "parent_h_id": parent_h_id,
            "new_h_id": new_h_id,
            "transform_type": transform_type,
            "interpolation_check": interpolation_check,
            "falsifiability_check": falsifiability_check,
            "accepted": accepted,
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        logger.debug("imagination log write failed", exc_info=True)


def imagine_with_checks(
    hypothesis: Hypothesis,
    transform_type: str,
    manifold: HypothesisManifold,
    *,
    model: Any = None,
    sigma: float = _DEFAULT_SIGMA,
    log_path: Path | None = None,
) -> Hypothesis | None:
    """imagine 完整流水线: LLM 变换 -> 可证伪性 -> interpolation 检查 -> 写 log.

    rcb_runner 主循环调这个. 失败任一步都返回 None, 不阻塞.
    imagine() 内部已做 falsifiability, 这里只加 interpolation + log.
    """
    parent_h_id = hypothesis.h_id
    new_h = imagine(hypothesis, transform_type, model=model)
    if new_h is None:
        _write_imagination_log(
            log_path,
            parent_h_id=parent_h_id,
            new_h_id=None,
            transform_type=transform_type,
            interpolation_check="skipped",
            falsifiability_check="failed",
            accepted=False,
        )
        return None

    is_interpolation = check_interpolation(new_h, manifold, sigma=sigma)
    accepted = not is_interpolation

    _write_imagination_log(
        log_path,
        parent_h_id=parent_h_id,
        new_h_id=new_h.h_id,
        transform_type=transform_type,
        interpolation_check="rejected" if is_interpolation else "passed",
        falsifiability_check="passed",
        accepted=accepted,
    )

    if is_interpolation:
        logger.debug("imagine_with_checks: interpolation rejected (sigma=%s)", sigma)
        return None
    return new_h


def detect_stagnation(best_h_id_history: list[str], N: int = _DEFAULT_STAGNATION_N) -> bool:
    """检测 posterior best h_id 是否连续 N 轮不变.

    history 按时间顺序 (最新在尾). 连续 N 个相同 -> stagnation.
    ponytail: 简单窗口比较 O(N). 升级路径: posterior entropy drop 检测.
    """
    if len(best_h_id_history) < N:
        return False
    recent = best_h_id_history[-N:]
    return all(h_id == recent[0] for h_id in recent)


def _build_blind_spot_prompt(h: Hypothesis, blind_spot: Any) -> tuple[str, str]:
    """构造 blind-spot bypass prompt. 返回 (system, user).

    ponytail: 跟 _build_transform_prompt 同款拼法, 换内容. 升级路径:
    LLM 先推断最相关的 Bourbaki 变换族, 再调对应 _build_transform_prompt.
    """
    skill = getattr(blind_spot, "skill", "unknown") or "unknown"
    why_blind = getattr(blind_spot, "why_blind", "") or ""
    workaround = getattr(blind_spot, "possible_workaround", "") or ""

    sys_text = (
        "You apply a structure-preserving transformation to a scientific "
        "hypothesis, guided by a known blind spot of the agent. "
        f"The agent cannot reliably use a particular tool/approach ({skill}): "
        f"{why_blind}. "
        "Generate a new hypothesis that AVOIDS relying on this blind spot by "
        "shifting the mathematical structure of the original hypothesis "
        "(not just by swapping the tool). "
        "Bourbaki mother structure shifts to consider: "
        "(1) algebraic: change symmetry group; "
        "(2) topological: change neighborhood/locality assumption; "
        "(3) order: change ordering principle (energy -> entropy, etc)."
    )
    if workaround:
        sys_text += f" Suggested workaround direction: {workaround}."
    sys_text += (
        "\n\nReturn STRICT JSON:\n"
        '{"new_description": <string>, "new_predictions": {<name>: <number>, ...}, '
        '"new_n_params": <int>, "transform_reason": <short string>}\n'
        "new_predictions must include at least one observable that differs "
        "numerically from the original. new_n_params >= 1. "
        "Return JSON only, no prose."
    )

    usr = (
        f"Original hypothesis:\n"
        f"  description: {h.description}\n"
        f"  predictions: {json.dumps(h.predictions, ensure_ascii=False)}\n"
        f"  n_params: {h.n_params}\n"
        f"\nBlind spot to bypass: {skill}\n"
        f"  why_blind: {why_blind}\n"
        f"  possible_workaround: {workaround}\n"
        f"\nGenerate a new hypothesis that bypasses this blind spot by "
        f"shifting the mathematical structure (not just by avoiding the tool)."
    )
    return sys_text, usr


def imagine_from_blind_spot(
    blind_spot: Any,
    manifold: HypothesisManifold,
    model: Any = None,
    *,
    log_path: Path | None = None,
    sigma: float = _DEFAULT_SIGMA,
) -> Hypothesis | None:
    """从 self-model 盲点触发 imagination.

    选 manifold 中第一个 hypothesis 作 parent (manifold._hyp 是 dict, 按
    insertion order), 让 LLM 生成绕过 blind_spot 的新 h. 失败返回 None, 不阻塞.

    ponytail: 跟 imagine_with_checks 同款 pipeline (LLM transform → 可证伪性
    → interpolation → log), 换 prompt. parent 选第一个而非 best posterior:
    blind spot bypass 不依赖 best h, 任意 h 都可作种子. 升级路径: LLM 推断
    最相关的 parent h (跟 blind spot 语义最匹配的).
    """
    if blind_spot is None or manifold is None or not manifold._hyp:
        return None
    if model is None:
        logger.debug("imagine_from_blind_spot: no model")
        return None
    if not getattr(blind_spot, "skill", ""):
        return None

    parent_h = next(iter(manifold._hyp.values()))
    try:
        sys_text, usr_text = _build_blind_spot_prompt(parent_h, blind_spot)
        text = _call_llm_sync(model, sys_text, usr_text)
        new_h = _parse_transform_response(text)
        if new_h is None:
            logger.debug("imagine_from_blind_spot: response unparseable")
            _write_imagination_log(
                log_path,
                parent_h_id=parent_h.h_id,
                new_h_id=None,
                transform_type="blind_spot_bypass",
                interpolation_check="skipped",
                falsifiability_check="failed",
                accepted=False,
            )
            return None

        if not check_falsifiability(new_h, model):
            logger.debug("imagine_from_blind_spot: not falsifiable")
            _write_imagination_log(
                log_path,
                parent_h_id=parent_h.h_id,
                new_h_id=new_h.h_id,
                transform_type="blind_spot_bypass",
                interpolation_check="skipped",
                falsifiability_check="failed",
                accepted=False,
            )
            return None

        is_interp = check_interpolation(new_h, manifold, sigma=sigma)
        accepted = not is_interp
        _write_imagination_log(
            log_path,
            parent_h_id=parent_h.h_id,
            new_h_id=new_h.h_id,
            transform_type="blind_spot_bypass",
            interpolation_check="rejected" if is_interp else "passed",
            falsifiability_check="passed",
            accepted=accepted,
        )
        if is_interp:
            logger.debug("imagine_from_blind_spot: interpolation rejected")
            return None
        return new_h
    except Exception as e:
        logger.warning("imagine_from_blind_spot_fallback: %s", e)
        return None


# ---------- Self-check ----------

class _SeqMock:
    """Mock LLM: 按顺序返回预设 response. 用完最后一个重复最后一个."""
    def __init__(self, *responses: str):
        self.responses = list(responses)
        self.idx = 0
        self.call_count = 0

    def invoke(self, messages):
        self.call_count += 1
        r = self.responses[min(self.idx, len(self.responses) - 1)]
        self.idx += 1
        return r


class _ErrMock:
    """Mock LLM: 永远抛异常."""
    def __init__(self, error: Exception):
        self.error = error
        self.call_count = 0

    def invoke(self, messages):
        self.call_count += 1
        raise self.error


def _selfcheck() -> None:
    """Assert-based demo: mock LLM 验证三族变换 / interpolation 拒绝 / 可证伪性拒绝 / 失败降级."""
    import tempfile

    # 原始 hypothesis — 掺杂半导体的电导率模型
    h0 = Hypothesis(
        h_id="h_base",
        description="Doped semiconductor: carrier density n, conductivity sigma = n e mu, mu bounded by ionized impurity scattering",
        predictions={"conductivity": 1.0, "mobility": 100.0},
        n_params=2,
    )

    falsifiability_resp = '{"falsifiable": true, "observable": "conductivity", "reason": "measurable"}'

    # Case 1: 三族变换都能生成新 hypothesis
    transforms = {
        "algebraic": '{"new_description": "Gauge-coupled carrier field: conductivity governed by U(1) gauge symmetry, sigma = n e^2 tau / m", "new_predictions": {"conductivity": 1.5, "mobility": 150.0}, "new_n_params": 3, "transform_reason": "spin-rotation -> gauge"}',
        "topological": '{"new_description": "Long-range hopping transport: next-nearest-neighbor hopping dominates, sigma proportional to t2^2", "new_predictions": {"conductivity": 2.0, "mobility": 80.0}, "new_n_params": 2, "transform_reason": "NN -> NNN hopping"}',
        "order": '{"new_description": "Entropy-maximized transport: conductivity set by entropy maximization, not energy minimization", "new_predictions": {"conductivity": 0.8, "mobility": 120.0}, "new_n_params": 2, "transform_reason": "energy -> entropy ordering"}',
    }
    for t_type, resp in transforms.items():
        # imagine 内部调 2 次 LLM: transform + falsifiability
        mock = _SeqMock(resp, falsifiability_resp)
        new_h = imagine(h0, t_type, model=mock)
        assert new_h is not None, f"case1 {t_type}: imagine returned None"
        assert new_h.h_id != h0.h_id, f"case1 {t_type}: new h_id should differ"
        assert new_h.description != h0.description, f"case1 {t_type}: description unchanged"
        assert new_h.predictions, f"case1 {t_type}: empty predictions"
        assert new_h.n_params >= 1, f"case1 {t_type}: n_params < 1"
        print(f"[CHECK] case1 {t_type}: new h_id={new_h.h_id}, predictions={new_h.predictions}")

    # Case 2: interpolation 检查 — 拒绝太近, 接受够远, 空 manifold 放行
    m = HypothesisManifold()
    m.add(Hypothesis("h_near", "near", predictions={"conductivity": 1.05, "mobility": 100.5}, n_params=2))
    # 远: distance sqrt(4^2 + 99.5^2) >> 0.5
    h_far = Hypothesis("h_far", "far", predictions={"conductivity": 5.0, "mobility": 200.0}, n_params=2)
    assert not check_interpolation(h_far, m, sigma=0.5), "case2a: h_far should be extrapolation"
    # 近: distance sqrt(0^2 + 0.1^2) = 0.1 < 0.5
    h_close = Hypothesis("h_close", "close", predictions={"conductivity": 1.05, "mobility": 100.4}, n_params=2)
    assert check_interpolation(h_close, m, sigma=0.5), "case2b: h_close should be interpolation"
    # 空 manifold
    m_empty = HypothesisManifold()
    assert not check_interpolation(h_far, m_empty), "case2c: empty manifold = extrapolation"
    # 无 common keys -> distance inf -> extrapolation
    h_diff_keys = Hypothesis("h_diff", "diff keys", predictions={"band_gap": 1.2}, n_params=1)
    assert not check_interpolation(h_diff_keys, m, sigma=0.5), "case2d: no common keys = extrapolation"
    print("[CHECK] case2: interpolation check (far ok / close reject / empty ok / no-common-keys ok)")

    # Case 3: 可证伪性检查 — 拒绝不可证伪
    mock_not_falsifiable = _SeqMock('{"falsifiable": false, "reason": "no testable prediction"}')
    h_unfalsifiable = Hypothesis("h_unf", "vague claim with no prediction", predictions={}, n_params=0)
    assert not check_falsifiability(h_unfalsifiable, model=mock_not_falsifiable), "case3a: should reject"
    # 无 model + 有 predictions -> 通过 (prototype fallback)
    h_with_pred = Hypothesis("h_pred", "has prediction", predictions={"x": 1.0}, n_params=1)
    assert check_falsifiability(h_with_pred, model=None), "case3b: no model + has pred = pass"
    # 无 model + 无 predictions -> 拒绝
    assert not check_falsifiability(h_unfalsifiable, model=None), "case3c: no model + no pred = reject"
    print("[CHECK] case3: falsifiability (reject unfalsifiable / no-model fallback)")

    # Case 4: 失败降级 — LLM 异常 / JSON 解析失败 / 无 model / 未知 type -> None
    assert imagine(h0, "algebraic", model=_ErrMock(RuntimeError("llm down"))) is None, "case4a: llm error -> None"
    assert imagine(h0, "algebraic", model=_SeqMock("not json at all")) is None, "case4b: bad json -> None"
    assert imagine(h0, "algebraic", model=None) is None, "case4c: no model -> None"
    assert imagine(h0, "unknown_type", model=_SeqMock("{}")) is None, "case4d: unknown type -> None"
    # transform response 缺字段 -> None
    assert imagine(h0, "algebraic", model=_SeqMock('{"new_description": ""}')) is None, "case4e: empty desc -> None"
    print("[CHECK] case4: failure fallback returns None (5 paths)")

    # Case 5: imagine_with_checks 完整流水线 — 通过 + 拒绝 + 写 log
    with tempfile.TemporaryDirectory() as td:
        log_path = Path(td) / "imagination_log.jsonl"
        m2 = HypothesisManifold()
        m2.add(h0)

        # 5a: 通过路径 — 新 h 跟 h0 距离够远
        ok_resp = '{"new_description": "totally new mechanism with gauge field", "new_predictions": {"conductivity": 10.0, "mobility": 500.0}, "new_n_params": 3, "transform_reason": "test"}'
        mock_ok = _SeqMock(ok_resp, falsifiability_resp)
        new_h = imagine_with_checks(h0, "algebraic", m2, model=mock_ok, sigma=0.5, log_path=log_path)
        assert new_h is not None, "case5a: imagine_with_checks should succeed"
        lines = log_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1, f"case5a: log should have 1 entry, got {len(lines)}"
        entry = json.loads(lines[0])
        assert entry["accepted"] is True, f"case5a: accepted should be True, got {entry}"
        assert entry["transform_type"] == "algebraic"
        assert entry["parent_h_id"] == "h_base"
        assert entry["interpolation_check"] == "passed"
        assert entry["falsifiability_check"] == "passed"
        print(f"[CHECK] case5a: full pipeline ok, log accepted={entry['accepted']}")

        # 5b: 拒绝路径 — interpolation (新 h 跟 h0 太近)
        close_resp = '{"new_description": "close to original", "new_predictions": {"conductivity": 1.01, "mobility": 100.1}, "new_n_params": 2, "transform_reason": "test"}'
        mock_close = _SeqMock(close_resp, falsifiability_resp)
        new_h_close = imagine_with_checks(h0, "topological", m2, model=mock_close, sigma=0.5, log_path=log_path)
        assert new_h_close is None, "case5b: interpolation should be rejected"
        lines = log_path.read_text(encoding="utf-8").strip().split("\n")
        last_entry = json.loads(lines[-1])
        assert last_entry["accepted"] is False
        assert last_entry["interpolation_check"] == "rejected"
        assert last_entry["falsifiability_check"] == "passed"
        print(f"[CHECK] case5b: interpolation rejected, log accepted={last_entry['accepted']}")

        # 5c: imagine 返回 None (LLM 失败) -> log 写 falsifiability_check=failed
        mock_fail = _ErrMock(RuntimeError("llm down"))
        new_h_fail = imagine_with_checks(h0, "order", m2, model=mock_fail, sigma=0.5, log_path=log_path)
        assert new_h_fail is None, "case5c: llm failure should return None"
        lines = log_path.read_text(encoding="utf-8").strip().split("\n")
        last_entry = json.loads(lines[-1])
        assert last_entry["accepted"] is False
        assert last_entry["falsifiability_check"] == "failed"
        assert last_entry["interpolation_check"] == "skipped"
        print("[CHECK] case5c: llm failure logged, falsifiability_check=failed")

        # 5d: log_path=None -> 不写文件, 不抛异常
        new_h_nolog = imagine_with_checks(h0, "algebraic", m2, model=_SeqMock(ok_resp, falsifiability_resp), sigma=0.5, log_path=None)
        assert new_h_nolog is not None, "case5d: log_path=None should still work"

    print("[CHECK] case5: imagine_with_checks full pipeline (4 sub-cases)")

    # Case 6: detect_stagnation
    assert not detect_stagnation(["h1"], N=3), "case6a: < N -> not stagnation"
    assert not detect_stagnation(["h1", "h2", "h1"], N=3), "case6b: not all same -> not stagnation"
    assert detect_stagnation(["h1", "h1", "h1"], N=3), "case6c: all same -> stagnation"
    assert detect_stagnation(["h2", "h1", "h1", "h1"], N=3), "case6d: last 3 same -> stagnation"
    assert not detect_stagnation([], N=3), "case6e: empty -> not stagnation"
    # 自定义 N
    assert detect_stagnation(["h1", "h1"], N=2), "case6f: N=2 all same -> stagnation"
    print("[CHECK] case6: detect_stagnation (6 sub-cases)")

    # Case 7: imagine_from_blind_spot — 失败降级路径
    from huginn.metacog.blind_spot_mapper import BlindSpot
    _bs = BlindSpot(
        skill="vasp", why_blind="no VASP license",
        possible_workaround="use QE or open-source DFT", priority="high",
    )
    # 7a: blind_spot=None / manifold=None / 无 model / 空 manifold → None
    assert imagine_from_blind_spot(None, m2, model=None) is None, "case7a: None blind_spot"
    assert imagine_from_blind_spot(_bs, None, model=_SeqMock("{}")) is None, "case7a: None manifold"
    assert imagine_from_blind_spot(_bs, m2, model=None) is None, "case7a: no model"
    _m_empty = HypothesisManifold()
    assert imagine_from_blind_spot(_bs, _m_empty, model=_SeqMock("{}")) is None, "case7a: empty manifold"
    _bs_no_skill = BlindSpot(skill="", why_blind="empty skill")
    assert imagine_from_blind_spot(_bs_no_skill, m2, model=_SeqMock("{}")) is None, "case7a: empty skill"
    print("[CHECK] case7a: failure degradation (None/empty/no-model/empty-skill)")

    # 7b: LLM 失败 → None
    assert imagine_from_blind_spot(
        _bs, m2, model=_ErrMock(RuntimeError("llm down"))) is None, "case7b: llm error"
    # 7c: LLM 返回非 JSON → None
    assert imagine_from_blind_spot(
        _bs, m2, model=_SeqMock("not json at all")) is None, "case7c: bad json"
    print("[CHECK] case7b/c: llm failure / bad json -> None")

    # 7d: 成功路径 — manifold 已有 h0, 新 h 距离够远, 通过 falsifiability
    with tempfile.TemporaryDirectory() as td:
        _log_path = Path(td) / "imagination_log.jsonl"
        _m_bs = HypothesisManifold()
        _m_bs.add(h0)
        _ok_resp = (
            '{"new_description": "QE-based DFT surrogate with different basis set", '
            '"new_predictions": {"conductivity": 8.0, "mobility": 400.0}, '
            '"new_n_params": 3, "transform_reason": "vasp -> QE"}'
        )
        _mock_ok = _SeqMock(_ok_resp, falsifiability_resp)
        _new_h = imagine_from_blind_spot(
            _bs, _m_bs, model=_mock_ok, log_path=_log_path, sigma=0.5)
        assert _new_h is not None, "case7d: should succeed"
        assert _new_h.h_id != h0.h_id, "case7d: new h_id differs"
        # log 写了一条 accepted=True
        _lines = _log_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(_lines) == 1, f"case7d: log should have 1 entry, got {len(_lines)}"
        _entry = json.loads(_lines[0])
        assert _entry["accepted"] is True, "case7d: accepted should be True"
        assert _entry["transform_type"] == "blind_spot_bypass"
        assert _entry["parent_h_id"] == "h_base"
        assert _entry["interpolation_check"] == "passed"
        assert _entry["falsifiability_check"] == "passed"
        print(f"[CHECK] case7d: blind_spot bypass ok, log accepted={_entry['accepted']}")

        # 7e: interpolation 拒绝 — 新 h 跟 h0 太近
        _close_resp = (
            '{"new_description": "tiny tweak", '
            '"new_predictions": {"conductivity": 1.01, "mobility": 100.1}, '
            '"new_n_params": 2, "transform_reason": "test"}'
        )
        _mock_close = _SeqMock(_close_resp, falsifiability_resp)
        _new_h_close = imagine_from_blind_spot(
            _bs, _m_bs, model=_mock_close, log_path=_log_path, sigma=0.5)
        assert _new_h_close is None, "case7e: interpolation should be rejected"
        _lines = _log_path.read_text(encoding="utf-8").strip().split("\n")
        _last = json.loads(_lines[-1])
        assert _last["accepted"] is False
        assert _last["interpolation_check"] == "rejected"
        print(f"[CHECK] case7e: interpolation rejected, log accepted={_last['accepted']}")

        # 7f: falsifiability 拒绝
        _mock_unf = _SeqMock(_ok_resp, '{"falsifiable": false, "reason": "no test"}')
        _new_h_unf = imagine_from_blind_spot(
            _bs, _m_bs, model=_mock_unf, log_path=_log_path, sigma=0.5)
        assert _new_h_unf is None, "case7f: should reject unfalsifiable"
        _lines = _log_path.read_text(encoding="utf-8").strip().split("\n")
        _last = json.loads(_lines[-1])
        assert _last["falsifiability_check"] == "failed"
        print("[CHECK] case7f: unfalsifiable rejected")

        # 7g: log_path=None 不抛
        _new_h_nolog = imagine_from_blind_spot(
            _bs, _m_bs, model=_SeqMock(_ok_resp, falsifiability_resp), log_path=None)
        assert _new_h_nolog is not None, "case7g: log_path=None should still work"
        print("[CHECK] case7g: log_path=None ok")

    print("OK imagination self-check passed (7 cases)")


if __name__ == "__main__":
    _selfcheck()
