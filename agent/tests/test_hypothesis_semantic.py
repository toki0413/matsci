"""P1#1: hypothesis_semantic 语义判定 — 优雅降级 + LLM 路径测试.

覆盖:
- 关 flag (默认) 时 classify_* 走关键词 fallback, 行为与改造前一致.
- 开 flag + 有 model provider 时走 LLM 语义判定, 大小写/标点宽容.
- LLM 输出非法标签 / 调用异常 → 降级回 fallback.
- 三条路径 (dimension/family/failure) 各自独立验证.
"""

from __future__ import annotations

import pytest

from huginn.autoloop import hypothesis_semantic as hs
from huginn.feature_flags import FeatureFlags


@pytest.fixture(autouse=True)
def _reset_state():
    """每个用例前清空 provider + 关 flag, 避免用例间污染."""
    hs.set_model_provider(None)
    FeatureFlags.shared().disable("hypothesis_llm_semantic")
    yield


# ── 1. 关 flag / 无 model → 关键词 fallback (向后兼容) ─────────────────────
def test_dimension_keyword_fallback():
    assert hs.classify_dimension("") == ""
    assert hs.classify_dimension("如果掺杂增加, 带隙减小") == "composition"
    assert hs.classify_dimension("温度调控载流子迁移") == "temperature"
    # 无关键词命中 → 空 (原行为)
    assert hs.classify_dimension("随便一段无关键词的话") == ""


def test_family_keyword_fallback():
    assert hs.classify_family("") == "dft-direct"  # 默认族
    assert hs.classify_family("用 DFT+VASP 计算") == "dft-direct"
    assert hs.classify_family("用 machine learning potential 拟合") == "ml-potential"
    assert hs.classify_family("用 symbolic regression 提取解析式") == "symbolic-regression"


def test_failure_keyword_fallback_default():
    assert hs.classify_failure("") == "hypothesis_error"
    assert hs.classify_failure("结果与假设相反") == "hypothesis_error"


# ── 2. 开 flag + 有 model → LLM 语义判定 ────────────────────────────────────
class _FakeModel:
    """按 prompt 内容返回对应标签的 fake provider."""

    def __init__(self, dim="temperature", fam="symbolic-regression", fail="tool_error"):
        self._dim, self._fam, self._fail = dim, fam, fail

    def invoke(self, prompt: str):
        result = _Resp(self._dim if "维度" in prompt else self._fam if "方法族" in prompt else self._fail)
        return result


class _Resp:
    def __init__(self, content: str):
        self.content = content


def test_llm_classify_dimension():
    FeatureFlags.shared().enable("hypothesis_llm_semantic")
    hs.set_model_provider(lambda: _FakeModel(dim="temperature"))
    assert hs.classify_dimension("样本在高温下发生相变") == "temperature"


def test_llm_classify_family():
    FeatureFlags.shared().enable("hypothesis_llm_semantic")
    hs.set_model_provider(lambda: _FakeModel(fam="symbolic-regression"))
    assert hs.classify_family("用 symbolic regression 提取解析式") == "symbolic-regression"


def test_llm_classify_failure():
    FeatureFlags.shared().enable("hypothesis_llm_semantic")
    hs.set_model_provider(lambda: _FakeModel(fail="tool_error"))
    assert hs.classify_failure("内存溢出导致崩溃") == "tool_error"


def test_llm_label_case_and_punctuation_tolerant():
    FeatureFlags.shared().enable("hypothesis_llm_semantic")

    class CapModel:
        def invoke(self, prompt: str):
            return _Resp("  HYPOTHESIS_ERROR.  ")

    hs.set_model_provider(lambda: CapModel())
    assert hs.classify_failure("结果相反") == "hypothesis_error"


# ── 3. 非法标签 / 异常 → 降级回 fallback ────────────────────────────────────
def test_llm_invalid_label_falls_back():
    FeatureFlags.shared().enable("hypothesis_llm_semantic")

    class BadModel:
        def invoke(self, prompt: str):
            return _Resp("我不知道")

    hs.set_model_provider(lambda: BadModel())
    assert hs.classify_dimension("随便一段话") == ""
    assert hs.classify_failure("随便") == "hypothesis_error"


def test_llm_invoke_exception_falls_back():
    FeatureFlags.shared().enable("hypothesis_llm_semantic")

    class BoomModel:
        def invoke(self, prompt: str):
            raise RuntimeError("llm down")

    hs.set_model_provider(lambda: BoomModel())
    assert hs.classify_dimension("如果掺杂增加, 带隙减小") == "composition"


def test_flag_off_never_calls_llm():
    """关 flag 时即便有 provider 也不调模型, 走 fallback."""

    class ShouldNotCall:
        def invoke(self, prompt: str):
            raise AssertionError("flag off 时不应调用 LLM")

    hs.set_model_provider(lambda: ShouldNotCall())
    assert hs.classify_dimension("如果掺杂增加, 带隙减小") == "composition"
