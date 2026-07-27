"""LLM-as-Likelihood — 用 LLM 评估 P(O|h) 替代 Gaussian.

Gaussian likelihood 只看数值预测误差, LLM-as-judge 能评估 hypothesis 对
observation 的语义解释力, 是 abduction 的关键升级路径.

失败一律降级到 HypothesisManifold._gaussian_log_likelihood, 不阻塞主循环.
env HUGINN_LLM_LIKELIHOOD=1 启用 (v14 HUGINN_DARWIN_LLM_EVAL=1 兼容映射),
默认关闭. HUGINN_LLM_LIKELIHOOD_INTERVAL (默认 5) 控制 LLM 调用频率.
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
import os
import re
from typing import Any, Callable

from huginn.metacog.hypothesis_manifold import (
    Hypothesis,
    HypothesisManifold,
    Observation,
)

logger = logging.getLogger(__name__)

# Gaussian 5σ 外 ≈ -12.5, 留余量到 -50; 上限 0 (完美 fit)
_LOG_LIK_MIN = -50.0
_LOG_LIK_MAX = 0.0


def is_llm_likelihood_enabled() -> bool:
    """env HUGINN_LLM_LIKELIHOOD=1 启用. v14 HUGINN_DARWIN_LLM_EVAL=1 兼容映射."""
    val = os.environ.get("HUGINN_LLM_LIKELIHOOD", "0").lower()
    if val in ("1", "true", "yes"):
        return True
    # v14 废弃 env 兼容: 老用户设了 HUGINN_DARWIN_LLM_EVAL=1 自动启用
    legacy = os.environ.get("HUGINN_DARWIN_LLM_EVAL", "0").lower()
    return legacy in ("1", "true", "yes")


def get_llm_likelihood_interval() -> int:
    try:
        return max(1, int(os.environ.get("HUGINN_LLM_LIKELIHOOD_INTERVAL", "5")))
    except (TypeError, ValueError):
        return 5


def _build_llm_lik_prompt(
    h: Hypothesis,
    o: Observation,
    task_ctx: str = "",
) -> tuple[str, str]:
    """构造 LLM 评估 prompt. 返回 (system, user)."""
    sys_text = (
        "You are a Bayesian likelihood estimator for scientific hypotheses. "
        "Given hypothesis H and observation O, estimate log P(O|H) — the "
        "log-likelihood of observing O under H.\n"
        f"Return STRICT JSON: {{\"log_lik\": <float>, \"reason\": <short string>}}. "
        f"log_lik range: [{_LOG_LIK_MIN}, {_LOG_LIK_MAX}]. "
        "0 = O exactly as H predicts; -50 = O essentially impossible under H."
    )
    usr = (
        f"Hypothesis: {h.description}\n"
        f"H predicts: {h.predictions}\n"
        f"Observation: {o.name} = {o.value} (sigma={o.sigma})\n"
    )
    if task_ctx:
        usr += f"\nTask context (truncated):\n{task_ctx[:800]}\n"
    usr += (
        "\nEstimate log P(O|H). Consider: does H actually predict O's name? "
        "Is |O - pred| within sigma? Is H's mechanism explanatory or ad hoc?\n"
        "Return JSON only, no prose."
    )
    return sys_text, usr


def _parse_log_lik_json(text: str) -> tuple[float | None, str]:
    """解析 {\"log_lik\": float, \"reason\": str}. 失败返回 (None, reason)."""
    if not text:
        return None, "empty response"
    # LLM 偶尔在 JSON 外加 markdown fence / 前后文本, 找第一个 {...}
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if not m:
        return None, "no JSON object found"
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        return None, f"json parse error: {e}"
    val = obj.get("log_lik")
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        return None, f"log_lik not numeric: {type(val).__name__}"
    val = float(val)
    if val < _LOG_LIK_MIN or val > _LOG_LIK_MAX:
        return None, f"log_lik out of range [{_LOG_LIK_MIN},{_LOG_LIK_MAX}]: {val}"
    reason = str(obj.get("reason", ""))[:200]
    return val, reason


class LLMLikelihood:
    """LLM-as-judge likelihood, 注入 HypothesisManifold 替代 Gaussian.

    用法:
        llm_lik = LLMLikelihood(model, task_ctx="...", interval=5)
        manifold = HypothesisManifold(likelihood_log=llm_lik.log_lik)
        # rcb_runner 主循环每轮 set iter_n
        llm_lik.iter_n = current_iter

    iter_n % interval == 0 时调 LLM, 否则用 Gaussian. 任何失败 (超时 /
    JSON parse error / API error / log_lik 越界) 都降级到 Gaussian.
    """

    def __init__(
        self,
        model: Any,
        *,
        task_ctx: str = "",
        interval: int = 5,
        gaussian_fallback: Callable[[Hypothesis, Observation], float] | None = None,
    ):
        self._model = model
        self._task_ctx = task_ctx or ""
        self._interval = max(1, int(interval))
        self._gaussian = gaussian_fallback or HypothesisManifold._gaussian_log_likelihood
        # rcb_runner 主循环每轮 set, 决定本轮是否调 LLM
        self.iter_n = 0

    def log_lik(self, hypothesis: Hypothesis, observation: Observation) -> float:
        if self._model is None or self.iter_n % self._interval != 0:
            return self._gaussian(hypothesis, observation)
        try:
            sys_text, usr_text = _build_llm_lik_prompt(
                hypothesis, observation, self._task_ctx,
            )
            text = self._call_llm_sync(sys_text, usr_text)
            val, reason = _parse_log_lik_json(text)
            if val is None:
                logger.warning(
                    "llm_likelihood_fallback: reason=%s, hypothesis=%s, observation=%s",
                    reason, hypothesis.h_id, observation.name,
                )
                return self._gaussian(hypothesis, observation)
            logger.debug(
                "llm_likelihood: h=%s o=%s log_lik=%.3f reason=%s",
                hypothesis.h_id, observation.name, val, reason,
            )
            return val
        except Exception as e:
            logger.warning(
                "llm_likelihood_fallback: reason=%s, hypothesis=%s, observation=%s",
                e, hypothesis.h_id, observation.name,
            )
            return self._gaussian(hypothesis, observation)

    def _call_llm_sync(self, sys_text: str, usr_text: str) -> str:
        # ponytail: sync invoke only. ainvoke-only model 走降级 (Gaussian).
        # 不在 sync 函数里 asyncio.run — 跟 rcb 主循环 event loop 冲突.
        # 升级路径: 整条 log_posterior → abductive_inference 链路 async 化.
        from huginn.metacog.step_evaluator import _build_messages, _resp_to_text
        messages = _build_messages(sys_text, usr_text)
        model = self._model
        if hasattr(model, "invoke"):
            return _resp_to_text(model.invoke(messages))
        raise ValueError(
            "model has no sync invoke; ainvoke-only models not supported in sync log_lik path"
        )


# ---------- Self-check ----------

def _selfcheck() -> None:
    """Assert-based demo: 验证 LLMLikelihood 的 4 条路径 + interval 控制 + env 兼容 + 注入."""
    import math
    import tempfile  # noqa: F401  # 保留 import 习惯, 后续扩展用

    class _MockModel:
        """Mock LLM client. invoke() 返回固定文本或抛异常."""
        def __init__(self, response: str = "", error: Exception | None = None):
            self.response = response
            self.error = error
            self.call_count = 0

        def invoke(self, messages):
            self.call_count += 1
            if self.error is not None:
                raise self.error
            return self.response

    h = Hypothesis(
        h_id="h_test",
        description="Test hypothesis: predicts x=1.0",
        predictions={"x": 1.0},
        n_params=1,
    )
    o = Observation("x", 1.05, sigma=0.1)
    gaussian_val = HypothesisManifold._gaussian_log_likelihood(h, o)
    # z = 0.05/0.1 = 0.5, -0.5*0.25 = -0.125 (浮点有 epsilon, 用 isclose)
    assert math.isclose(gaussian_val, -0.125, abs_tol=1e-9), f"setup: gaussian_val={gaussian_val}"

    # Case 1: 正常路径 — LLM 返回合法 JSON → 返回 log_lik
    mock_ok = _MockModel(response='{"log_lik": -0.3, "reason": "close fit"}')
    llm1 = LLMLikelihood(mock_ok, interval=1)
    llm1.iter_n = 0
    val1 = llm1.log_lik(h, o)
    assert val1 == -0.3, f"case1: expect -0.3, got {val1}"
    assert mock_ok.call_count == 1, f"case1: LLM called once, got {mock_ok.call_count}"
    print(f"[CHECK] case1 ok: log_lik={val1}, LLM called")

    # Case 2: JSON 解析失败 → 降级到 Gaussian
    mock_bad = _MockModel(response="not json at all")
    llm2 = LLMLikelihood(mock_bad, interval=1)
    llm2.iter_n = 0
    val2 = llm2.log_lik(h, o)
    assert val2 == gaussian_val, f"case2: expect Gaussian {gaussian_val}, got {val2}"
    assert mock_bad.call_count == 1, "case2: LLM called once before fallback"
    print(f"[CHECK] case2 ok: JSON parse fail -> Gaussian {val2}")

    # Case 3: log_lik 越界 → 降级到 Gaussian (两个方向都测)
    mock_hi = _MockModel(response='{"log_lik": 5.0, "reason": "too high"}')
    llm3 = LLMLikelihood(mock_hi, interval=1)
    llm3.iter_n = 0
    assert llm3.log_lik(h, o) == gaussian_val, "case3a: above 0 -> Gaussian"

    mock_lo = _MockModel(response='{"log_lik": -100.0, "reason": "too low"}')
    llm3b = LLMLikelihood(mock_lo, interval=1)
    llm3b.iter_n = 0
    assert llm3b.log_lik(h, o) == gaussian_val, "case3b: below -50 -> Gaussian"
    print(f"[CHECK] case3 ok: out-of-range (both sides) -> Gaussian")

    # Case 4: LLM 异常 → 降级到 Gaussian
    mock_err = _MockModel(error=RuntimeError("simulated timeout"))
    llm4 = LLMLikelihood(mock_err, interval=1)
    llm4.iter_n = 0
    val4 = llm4.log_lik(h, o)
    assert val4 == gaussian_val, f"case4: expect Gaussian, got {val4}"
    print(f"[CHECK] case4 ok: LLM exception -> Gaussian {val4}")

    # Case 5: interval=5 — iter 0/5/10 调 LLM, iter 1/2/3/4/6/7/8/9 用 Gaussian
    mock_int = _MockModel(response='{"log_lik": -0.2, "reason": "interval test"}')
    llm5 = LLMLikelihood(mock_int, interval=5)
    llm_val = -0.2
    for it in range(11):
        llm5.iter_n = it
        v = llm5.log_lik(h, o)
        if it % 5 == 0:
            assert v == llm_val, f"iter {it}: expect LLM {llm_val}, got {v}"
        else:
            assert v == gaussian_val, f"iter {it}: expect Gaussian {gaussian_val}, got {v}"
    assert mock_int.call_count == 3, (
        f"interval=5: LLM called 3 times (iter 0,5,10), got {mock_int.call_count}"
    )
    print(f"[CHECK] case5 ok: interval=5, LLM called {mock_int.call_count} times in 11 iters")

    # Case 6: model=None → 一律 Gaussian (不调 LLM)
    llm6 = LLMLikelihood(None, interval=1)
    llm6.iter_n = 0
    assert llm6.log_lik(h, o) == gaussian_val, "case6: model=None -> Gaussian"
    print(f"[CHECK] case6 ok: model=None -> Gaussian")

    # Case 7: env 兼容映射 (v14 HUGINN_DARWIN_LLM_EVAL -> v15 HUGINN_LLM_LIKELIHOOD)
    _saved_lik = os.environ.get("HUGINN_LLM_LIKELIHOOD")
    _saved_darwin = os.environ.get("HUGINN_DARWIN_LLM_EVAL")
    try:
        os.environ["HUGINN_LLM_LIKELIHOOD"] = "0"
        os.environ["HUGINN_DARWIN_LLM_EVAL"] = "1"
        assert is_llm_likelihood_enabled(), "v14 env=1 should map to enabled"

        os.environ["HUGINN_DARWIN_LLM_EVAL"] = "0"
        os.environ["HUGINN_LLM_LIKELIHOOD"] = "1"
        assert is_llm_likelihood_enabled(), "v15 env=1 should enable"

        os.environ["HUGINN_LLM_LIKELIHOOD"] = "0"
        assert not is_llm_likelihood_enabled(), "both 0 -> disabled"

        # interval 读取
        os.environ["HUGINN_LLM_LIKELIHOOD_INTERVAL"] = "3"
        assert get_llm_likelihood_interval() == 3, "interval=3"
        os.environ.pop("HUGINN_LLM_LIKELIHOOD_INTERVAL", None)
        assert get_llm_likelihood_interval() == 5, "default interval=5"
    finally:
        for k, v in [
            ("HUGINN_LLM_LIKELIHOOD", _saved_lik),
            ("HUGINN_DARWIN_LLM_EVAL", _saved_darwin),
        ]:
            if v is not None:
                os.environ[k] = v
            else:
                os.environ.pop(k, None)
    print(f"[CHECK] case7 ok: env compat mapping + interval read")

    # Case 8: 注入到 HypothesisManifold 验证接口兼容 + interval gate
    mock_inj = _MockModel(response='{"log_lik": -0.1, "reason": "inject test"}')
    llm8 = LLMLikelihood(mock_inj, interval=5)  # interval=5 才能测 gate
    m = HypothesisManifold(likelihood_log=llm8.log_lik)
    m.add(h)
    llm8.iter_n = 0  # 0%5==0 → 调 LLM
    post = m.log_posterior([o])
    assert "h_test" in post, f"manifold injection: posterior keys={post}"
    assert mock_inj.call_count == 1, f"iter 0 should call LLM once, got {mock_inj.call_count}"
    # iter_n=1 + interval=5 → Gaussian, 不调 LLM
    llm8.iter_n = 1
    pre_calls = mock_inj.call_count
    m.log_posterior([o])
    assert mock_inj.call_count == pre_calls, "iter_n=1 interval=5 should not call LLM"
    print(f"[CHECK] case8 ok: injected into HypothesisManifold, posterior={post}")

    print("OK llm_likelihood self-check passed (8 cases)")


if __name__ == "__main__":
    _selfcheck()
