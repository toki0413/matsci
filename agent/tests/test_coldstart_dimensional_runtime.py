"""S3 运行时接线: compile_domain_guards 里跑量纲预检(单位+跨量恒等), verify_domain_ready 暴露.

新域加载时会跑一次量纲预检 —— 单位不合法或派生关系量纲不自洽的契约, 会被 compile 标
dimensional.ok=False、verify_domain_ready 的 dimensional_ready=False, 不必等数值阶段才暴露.
"""
from __future__ import annotations

from huginn.research.coldstart_guards import compile_domain_guards, verify_domain_ready, _dimensional_precheck


def test_runtime_compile_runs_dimensional_precheck_live():
    """ecology 契约的 frequency=1/period_est 跨量恒等被 compile 检过(非只活在测试)."""
    g = compile_domain_guards("ecology_dynamics")
    dim = g["dimensional"]
    assert dim["ok"] is True
    assert dim["derived"]["frequency"]["ok"] is True
    assert dim["derived"]["frequency"]["expected"] == "T-1"     # 1/time
    assert dim["quantities"]["period_est"]["valid"] is True
    assert dim["quantities"]["period_est"]["dimension_signature"] == "T1"
    # 有契约的域 → 预检启用
    assert verify_domain_ready(g)["dimensional_ready"] is True


def test_runtime_compile_flags_garbage_unit():
    """单位不可解析的契约 → 量纲预检 ok=False(非叠加到 ready; 独立 dimensional_ready)."""
    bad = _dimensional_precheck({"quantities": {"speed": {"unit": "secods"}}})
    assert bad["ok"] is False
    assert bad["quantities"]["speed"]["valid"] is False
    g = compile_domain_guards("ecology_dynamics")
    assert verify_domain_ready(g)["dimensional_ready"] is True      # 正常域不受影响


def test_runtime_domain_without_contract_is_unrestricted():
    """无 scientific_contract 的域(如 rigidity): 量纲预检 ok, dimensional_ready True, 不设限."""
    g = compile_domain_guards("rigidity")
    assert g["dimensional"]["ok"] is True
    assert verify_domain_ready(g)["dimensional_ready"] is True