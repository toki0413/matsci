"""ClawBench-style benchmark runner.

Runs materials_science_research_cases against a WS-based agent,
computing pass^3, pass@3, and FinalScore metrics.
Requires a running backend.

Usage:
    python tests/test_clawbench_runner.py [--trials 3]
"""

import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass
from types import SimpleNamespace
from websockets.asyncio.client import connect

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from huginn.benchmark.core import BenchmarkSuite

WS_URL = "ws://127.0.0.1:8000/ws/agent?token=dev"
MAX_SIZE = 2**24
RECV_TIMEOUT = 300  # 5 min per message (DeepSeek reasoner is slow)


@dataclass
class WSAgent:
    """Minimal agent that chats via WebSocket — compatible with BenchmarkSuite."""

    async def chat(self, message: str, thread_id: str = "bench"):
        """Async generator yielding state dicts, matching LangGraph interface."""
        full_text = []

        try:
            async with connect(
                WS_URL,
                open_timeout=60,
                ping_interval=None,
                ping_timeout=None,
                max_size=MAX_SIZE,
            ) as ws:
                await ws.send(json.dumps({
                    "type": "user_input",
                    "content": message,
                    "thread_id": thread_id,
                    "persona": "research",
                }))

                while True:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=RECV_TIMEOUT)
                    except asyncio.TimeoutError:
                        break

                    data = json.loads(raw)
                    msg_type = data.get("type", "")

                    if msg_type == "text_delta":
                        full_text.append(data.get("text", ""))
                    elif msg_type in ("done", "message_complete"):
                        break
                    elif msg_type == "error":
                        full_text.append(f"[ERROR: {data.get('error', '')}]")
                        break
                    elif msg_type == "ping":
                        await ws.send(json.dumps({"type": "pong"}))
                    elif msg_type == "plan_proposed":
                        plan_id = data.get("plan_id", "")
                        if plan_id:
                            await ws.send(json.dumps({
                                "type": "plan_confirm",
                                "plan_id": plan_id,
                                "approved": True,
                                "thread_id": thread_id,
                            }))
                    elif msg_type == "approval_request":
                        req_id = data.get("id", data.get("request_id", ""))
                        if req_id:
                            await ws.send(json.dumps({
                                "type": "approval_response",
                                "request_id": req_id,
                                "approved": True,
                                "thread_id": thread_id,
                            }))
                    elif msg_type == "clarification":
                        q_id = data.get("question_id", "")
                        await ws.send(json.dumps({
                            "type": "clarification_response",
                            "question_id": q_id,
                            "answer": "Please use your best judgment.",
                            "thread_id": thread_id,
                        }))
        except Exception as e:
            full_text.append(f"[WS ERROR: {type(e).__name__}: {e}]")

        yield {"messages": [SimpleNamespace(content="".join(full_text))]}


async def run_benchmark(trials: int = 3):
    suite = BenchmarkSuite(name="clawbench-matsci")
    suite.materials_science_research_cases()

    print("=" * 60)
    print(f"ClawBench Materials Science Benchmark")
    print(f"Cases: {len(suite.cases)} | Trials: {trials}")
    print("=" * 60)

    agent = WSAgent()
    start = time.time()
    result = await suite.run_multi_trial(agent, trials=trials, thread_id="clawbench")
    elapsed = time.time() - start

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"FinalScore: {result.final_score:.2f}")
    print(f"  S: {result.avg_score:.4f}")
    print(f"  pass^3: {result.pass_all_rate:.4f}")
    print(f"  pass@3: {result.pass_any_rate:.4f}")
    print(f"  coverage: {result.coverage:.4f}")
    print(f"Avg latency: {result.avg_latency_ms:.0f}ms")

    print(f"\n{'Case':<40} {'Score':>6} {'pass^3':>7} {'pass@3':>7}")
    print("-" * 62)
    for cr in result.case_results:
        t = cr.task[:38] + ".." if len(cr.task) > 40 else cr.task
        print(f"  {t:<38} {cr.avg_score:>6.2f} {'Y' if cr.pass_all else 'N':>7} {'Y' if cr.pass_any else 'N':>7}")

    report = {
        "final_score": result.final_score,
        "avg_score": result.avg_score,
        "pass_all_rate": result.pass_all_rate,
        "pass_any_rate": result.pass_any_rate,
        "coverage": result.coverage,
        "avg_latency_ms": result.avg_latency_ms,
        "total_time_s": elapsed,
        "trials": result.trials,
        "cases": [
            {"case_id": cr.case_id, "task": cr.task, "category": cr.category,
             "avg_score": cr.avg_score, "max_score": cr.max_score,
             "pass_all": cr.pass_all, "pass_any": cr.pass_any}
            for cr in result.case_results
        ],
    }
    out = os.path.join(os.path.dirname(__file__), "clawbench_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nResults: {out}")


if __name__ == "__main__":
    trials = 3
    if "--trials" in sys.argv:
        idx = sys.argv.index("--trials")
        if idx + 1 < len(sys.argv):
            trials = int(sys.argv[idx + 1])
    asyncio.run(run_benchmark(trials=trials))
