"""SciExplorer 炼化自检: 大数组摘要 + 守恒律检查 + prompt 规则.

最小可运行检查 (ponytail):
1. _jsonify 大数组返回 _array_summary 而非全量 list
2. _jsonify 长 list 返回 _list_summary
3. _check_conservation_laws 标记能量守恒违规
4. _check_conservation_laws 干净假设通过
5. system prompt 含定性行为优先 + 守恒律规则
"""
from __future__ import annotations


def _test_large_array_summary():
    """大 numpy 数组走摘要, 不走 tolist."""
    import numpy as np
    from huginn.types import _jsonify

    # 小数组照常 tolist
    small = np.array([1, 2, 3])
    out = _jsonify(small)
    assert isinstance(out, list), "小数组应返回 list"

    # 大数组给摘要
    big = np.arange(500)
    out = _jsonify(big)
    assert isinstance(out, dict), "大数组应返回摘要 dict"
    assert out.get("_array_summary") is True
    assert out["shape"] == [500]
    assert len(out["sample"]) <= 20


def _test_large_list_summary():
    """长 list 也截断."""
    from huginn.types import _jsonify

    short = [1, 2, 3]
    assert _jsonify(short) == [1, 2, 3]

    long_list = list(range(500))
    out = _jsonify(long_list)
    assert isinstance(out, dict)
    assert out.get("_list_summary") is True
    assert out["length"] == 500
    assert len(out["sample"]) <= 20


def _test_conservation_energy_violation():
    """能量守恒违规被标记."""
    from huginn.tools.hypothesis_generator_tool import HypothesisGeneratorTool

    h = {
        "hypothesis_id": "H1",
        "statement": "该反应器能创造能量, 实现永动机",
        "testable_prediction": "效率大于100%",
    }
    flag = HypothesisGeneratorTool._check_conservation_laws(h)
    assert not flag["passed"], "永动机假设应被标记"
    assert "energy_conservation_suspect" in flag["flags"]


def _test_conservation_clean_pass():
    """干净假设通过."""
    from huginn.tools.hypothesis_generator_tool import HypothesisGeneratorTool

    h = {
        "hypothesis_id": "H2",
        "statement": "GaN 掺 Mg 后空穴浓度随退火温度升高而增大",
        "testable_prediction": "退火温度 700-900C 时空穴浓度单调递增",
    }
    flag = HypothesisGeneratorTool._check_conservation_laws(h)
    assert flag["passed"], "正常假设不应触发守恒律警告"
    assert flag["flags"] == []


def _test_prompt_has_qualitative_and_conservation():
    """system prompt 含定性行为优先 + 守恒律规则."""
    from huginn.tools.hypothesis_generator_tool import _HYPOTHESIS_SYSTEM_PROMPT

    assert "定性行为优先" in _HYPOTHESIS_SYSTEM_PROMPT, "prompt 缺定性行为优先规则"
    assert "守恒律" in _HYPOTHESIS_SYSTEM_PROMPT, "prompt 缺守恒律规则"


if __name__ == "__main__":
    _test_large_array_summary()
    print("[1/5] 大数组摘要 OK")
    _test_large_list_summary()
    print("[2/5] 长 list 截断 OK")
    _test_conservation_energy_violation()
    print("[3/5] 能量守恒违规检测 OK")
    _test_conservation_clean_pass()
    print("[4/5] 干净假设通过 OK")
    _test_prompt_has_qualitative_and_conservation()
    print("[5/5] prompt 规则就位 OK")
    print("\nAll SciExplorer distill self-checks passed.")
