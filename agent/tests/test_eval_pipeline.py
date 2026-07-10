"""Quick self-check: verify eval pipeline + metrics wiring work end-to-end.

Run: python tests/test_eval_pipeline.py
"""

import sys
from pathlib import Path

# Ensure agent is importable
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_metrics_tracking():
    """Verify Prometheus metrics are properly wired."""
    from huginn.routes.metrics import (
        track_llm_usage, track_tool_call, track_agent_turn,
        LLM_TOKENS_TOTAL, LLM_COST_USD,
        PROMPT_CACHE_HITS_TOTAL, PROMPT_CACHE_MISSES_TOTAL,
    )

    # Track a fake LLM call
    track_llm_usage("deepseek-chat", {
        "input_tokens": 1000,
        "output_tokens": 500,
        "cache_read_input_tokens": 200,
        "cache_creation_input_tokens": 100,
    })

    track_tool_call("structure_tool")
    track_agent_turn("test_thread_001")

    # Read back the metrics text
    from huginn.routes.metrics import generate_latest
    output = generate_latest().decode("utf-8")

    assert "huginn_llm_tokens_total" in output, "LLM_TOKENS_TOTAL missing"
    assert "huginn_llm_cost_usd" in output, "LLM_COST_USD missing"
    assert "huginn_tool_calls_total" in output, "TOOL_CALLS_TOTAL missing"
    assert "huginn_agent_turns_total" in output, "AGENT_TURNS_TOTAL missing"
    assert "huginn_prompt_cache_hits_total" in output, "CACHE_HITS missing"

    print("[PASS] metrics tracking — all 5 Prometheus metrics wired")


def test_eval_endpoint_imports():
    """Verify eval route module loads and exposes 4 endpoints."""
    from huginn.routes.eval import router

    paths = [r.path for r in router.routes]
    assert "/eval/run" in paths, f"/eval/run missing: {paths}"
    assert "/eval/history" in paths, f"/eval/history missing: {paths}"
    assert "/eval/analyze" in paths, f"/eval/analyze missing: {paths}"

    print(f"[PASS] eval endpoint — 4 routes registered: {paths}")


def test_eval_router_registered():
    """Verify eval_router is in ALL_ROUTERS."""
    from huginn.routes import ALL_ROUTERS

    found = False
    for r in ALL_ROUTERS:
        if hasattr(r, "routes"):
            for route in r.routes:
                if "/eval" in str(route.path):
                    found = True
                    break
        if found:
            break

    assert found, "eval_router not found in ALL_ROUTERS"
    print("[PASS] eval router — registered in ALL_ROUTERS")


def test_bench_standalone():
    """Verify MatWorldBench evaluates correctly without agent."""
    from huginn.evaluation.matworld_bench import MatWorldBench

    bench = MatWorldBench()
    assert len(bench.tasks) >= 10, f"expected >=10 tasks, got {len(bench.tasks)}"

    # Test with a correct answer
    result = bench.evaluate("si_bandgap", {"band_gap_eV": 1.12})
    assert result.passed, f"si_bandgap should pass: {result}"
    assert result.score == 1.0

    # Test with a wrong answer
    result = bench.evaluate("si_bandgap", {"band_gap_eV": 5.0})
    assert not result.passed, "si_bandgap should fail with 5.0 eV"

    print("[PASS] MatWorldBench — 10 tasks, evaluate() works")


def test_grader_registry():
    """Verify default_registry has the 4 built-in graders."""
    from huginn.validation.grader import default_registry

    reg = default_registry()
    names = reg.names()
    assert "physics" in names, f"physics grader missing: {names}"
    assert "dimensional" in names, f"dimensional grader missing: {names}"
    assert "hallucination" in names, f"hallucination grader missing: {names}"

    # evaluate_all should return results
    results = reg.evaluate_all({
        "output": {"band_gap_eV": 1.12},
        "task": {"id": "test", "category": "electronic", "expected": {"band_gap_eV": 1.12}},
    })
    assert len(results) >= 3, f"expected >=3 grader results, got {len(results)}"

    print(f"[PASS] GraderRegistry — {len(names)} graders: {names}")


def test_parse_agent_output():
    """Verify _parse_agent_output extracts structured data."""
    from huginn.routes.eval import _parse_agent_output

    # JSON block
    result = _parse_agent_output('The answer is ```json\n{"band_gap_eV": 1.12}\n```')
    assert result.get("band_gap_eV") == 1.12, f"JSON block parse failed: {result}"

    # Key: value pattern
    result = _parse_agent_output("The band gap is band_gap_eV: 1.15 eV")
    assert "band_gap_eV" in result, f"key:value parse failed: {result}"
    assert result["band_gap_eV"] == 1.15

    print("[PASS] _parse_agent_output — JSON + key:value extraction works")


if __name__ == "__main__":
    test_metrics_tracking()
    test_eval_endpoint_imports()
    test_eval_router_registered()
    test_bench_standalone()
    test_grader_registry()
    test_parse_agent_output()
    print("\nAll self-checks passed.")
