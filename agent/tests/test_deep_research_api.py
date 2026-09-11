"""深研 HTTP 端点 (ADR-0001 迁移目标) 测试.

覆盖:
  - 受控数学求值器: 合法表达式通过, 危险/越权节点被拒 (不 exec).
  - POST /research/run_program: 确定性深研, 纯 JSON 请求往返, 产出完整工件.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from huginn.routes.deep_research import _safe_eval_expr, router

# ── 受控求值器 ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "expr,params,expected",
    [
        ("a + b", {"a": 1.0, "b": 2.0}, 3.0),
        ("2*a + b", {"a": 1.5, "b": 2.0}, 5.0),
        ("sqrt(a) + sin(b)", {"a": 4.0, "b": 0.0}, 2.0),
        ("abs(a - b)", {"a": 1.0, "b": 5.0}, 4.0),
        ("max(a, b) / 2", {"a": 3.0, "b": 1.0}, 1.5),
    ],
)
def test_safe_eval_expr_valid(expr, params, expected):
    assert _safe_eval_expr(expr, params) == expected


def test_safe_eval_expr_rejects_dangerous():
    # 试图访问属性 / 下标 / import / 调用 —— 应抛 ValueError, 绝不能算出结果
    for bad in ("__import__('os')", "a.__class__", "=[1,2]", "lambda: 1", "a()"):
        with pytest.raises(ValueError):
            _safe_eval_expr(bad, {"a": 1.0})
    with pytest.raises(ValueError):
        _safe_eval_expr("unknown_var", {"a": 1.0})


def test_safe_eval_expr_rejects_empty():
    with pytest.raises(ValueError):
        _safe_eval_expr("", {})
    with pytest.raises(ValueError):
        _safe_eval_expr("   ", {})


# ── HTTP 端点 ───────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_run_program_returns_outcome(client):
    resp = client.post(
        "/research/run_program",
        json={
            "goal": "find optimal a,b",
            "objectives_config": {"score": "maximize"},
            "experiments": [
                {"name": "e0", "hypothesis": "baseline",
                 "objectives": {"score": "a + b"}, "params": {"a": 1.0, "b": 2.0}},
                {"name": "e1", "hypothesis": "candidate",
                 "objectives": {"score": "2*a + b"}, "params": {"a": 1.5, "b": 2.0}},
            ],
            "max_iterations": 8,
            "min_iterations": 2,
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["explored"] >= 1
    assert isinstance(data["pareto_front"], list)
    assert "report" in data
    # 确定性综合: verdict 是声明门禁结果 (pass / needs_grounding 等), 非空即可
    assert isinstance(data["verdict"], str) and data["verdict"]
    # 确定性综合无需 client → report_source 非 llm
    assert data.get("report_source") in ("deterministic",)


def test_run_program_validation(client):
    # 未知字段 → 400
    r = client.post("/research/run_program", json={"nope": 1})
    assert r.status_code == 400
    # 缺 goal → 400
    r = client.post("/research/run_program", json={"experiments": []})
    assert r.status_code == 400
    # 危险表达式 → 前置校验 400 (绝不静默跑空)
    r = client.post(
        "/research/run_program",
        json={
            "goal": "g",
            "objectives_config": {"score": "maximize"},
            "experiments": [
                {"name": "e0", "hypothesis": "h",
                 "objectives": {"score": "__import__('os')"},
                 "params": {"a": 1.0}},
            ],
        },
    )
    assert r.status_code == 400
    # 未知变量 → 400
    r = client.post(
        "/research/run_program",
        json={
            "goal": "g",
            "objectives_config": {"score": "maximize"},
            "experiments": [
                {"name": "e0", "hypothesis": "h",
                 "objectives": {"score": "a + b"},
                 "params": {"a": 1.0}},
            ],
        },
    )
    assert r.status_code == 400
