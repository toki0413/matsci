"""Huginn → CausalGame adapter runner.

Runs the real Huginn agent (HuginnAgent + cognitive engine) against a CausalGame
scenario by wrapping the CausalGame `CanyonClient` as first-class Huginn tools.

This is NOT a back-challenge prompt: the agent gets generic controlled-experiment,
confounder/selection-bias/noise-aware methodology, and must discover the hidden
causal mechanism itself through `deploy` experiments before a one-shot `submit`.

Usage:
    python -m huginn.cli.causal_runner --experiment antenna_trap \
        [--base-url http://localhost:8000] [--model deepseek-chat] \
        [--max-tool-calls 40]

Env:
    HUGINN_PROVIDER / HUGINN_MODEL / DEEPSEEK_API_KEY  (model + key)
    CAUSALGAME_DIR  (path to the CausalGame repo that contains agent/client.py;
                     defaults to /tmp/CausalGame)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

# CausalGame's CanyonClient lives in the CausalGame repo — put it on the path.
_CAUSAL_DIR = os.environ.get("CAUSALGAME_DIR", "/tmp/CausalGame")
if str(_CAUSAL_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(_CAUSAL_DIR))

DEFAULT_BASE_URL = "http://localhost:8000"


# ── Tool factory: wrap one CanyonClient session as Huginn tools ─────────────
def _make_tools(client) -> list:
    """Build first-class tools bound to a single CausalGame session."""

    @tool("causal_status", description=(
        "Get the current mission status: stage (1 or 2), drones_remaining, "
        "deployments_remaining, and the victory_threshold survival rate. "
        "Call this at the start to know your budget."))
    def causal_status() -> str:
        return json.dumps(client.get_status(), ensure_ascii=False)

    @tool("causal_action_space", description=(
        "Get the action space: the design parameters (component DEF values with "
        "min/max/default), equipment options, and budget constraints (e.g. "
        "total_def_budget). Call this to learn what you can control."))
    def causal_action_space() -> str:
        return json.dumps(client.get_action_space(), ensure_ascii=False)

    @tool("causal_history", description=(
        "Get the historical flight records for every past deployment: survival "
        "status, hit_count, plus per-flight environment data. Use this to compare "
        "survival across designs and detect confounders / selection bias / noise."))
    def causal_history() -> str:
        return json.dumps(client.get_history(), ensure_ascii=False)

    @tool("causal_deploy", description=(
        "Deploy drones for an experiment. Arguments: design = dict of "
        "{component_name: def_value} (use the names/values from causal_action_space); "
        "count = number of drones (default 1); equipment = optional dict, e.g. "
        "{'enhancement_module': 'signal_filter'}. Returns per-drone survival status. "
        "Run CONTROLLED experiments: change one variable at a time. "
        "If the response says the deployment budget is exhausted, STOP deploying and "
        "immediately call causal_submit with your best design."))
    def causal_deploy(design: dict, count: int = 1, equipment: dict | None = None) -> str:
        try:
            res = client.deploy_drone(design=design, count=count, equipment=equipment)
        except Exception as e:
            # HTTPError (e.g. 400 when deployment budget is exhausted) must NOT
            # crash the agent loop — return a structured hint so the agent can
            # pivot to causal_submit instead of dying mid-turn.
            return json.dumps({
                "error": str(e),
                "hint": ("Deployment rejected. If the deployment budget is now exhausted "
                         "(deployments_remaining == 0), do NOT keep deploying. Call "
                         "causal_submit immediately with the best design you have discovered."),
            }, ensure_ascii=False)
        return json.dumps(res, ensure_ascii=False)

    @tool("causal_submit", description=(
        "ONE-SHOT final submission of your design for Stage-2 evaluation on a "
        "1000-drone fleet. Can only be called ONCE. Arguments: design = dict of "
        "{component_name: def_value}, equipment = optional dict. Returns survival_rate, "
        "victory, final_score. Call ONLY when you are confident in the causal mechanism."))
    def causal_submit(design: dict, equipment: dict | None = None) -> str:
        try:
            res = client.submit_final_design(design=design, equipment=equipment)
        except Exception as e:
            return json.dumps({"error": str(e)}, ensure_ascii=False)
        return json.dumps(res, ensure_ascii=False)

    return [causal_status, causal_action_space, causal_history,
            causal_deploy, causal_submit]


def _system_prompt() -> str:
    return (
        "You are a drone designer in a research simulator. Your goal is to DISCOVER the "
        "hidden causal mechanism behind drone survival through controlled experiments, "
        "then submit the best design.\n"
        "Budget: at most 10 deployments and 200 drones in Stage 1.\n"
        "Methodology:\n"
        "1. Read causal_action_space to learn the controllable design parameters and budget.\n"
        "2. Run CONTROLLED experiments with causal_deploy: vary ONE parameter at a time and "
        "   compare survival rates. Do not change many things at once.\n"
        "3. After each batch, read causal_history: the per-flight environment data may reveal "
        "   hidden confounders, survivorship censoring, or measurement noise that a naive "
        "   average would misinterpret.\n"
        "4. Form a causal (not merely correlational) hypothesis about which components/settings "
        "   drive survival. Consider that your observations may be biased by which drones "
        "   survived to be observed, confounded by hidden environment variables, or corrupted "
        "   by noise.\n"
        "5. When confident, call causal_submit ONCE with your best design, then give a short "
        "   final summary of your causal reasoning and the returned evaluation.\n"
        "Be honest about uncertainty; do not overfit surface correlations."
    )


def _extract_final(messages: list) -> tuple[str, dict | None]:
    """Return (last AI text, submit result dict if submitted)."""
    ai_text = ""
    submit_result = None
    for m in reversed(messages):
        if isinstance(m, ToolMessage):
            name = getattr(m, "name", "")
            if name == "causal_submit":
                try:
                    submit_result = json.loads(m.content)
                except Exception:
                    submit_result = {"raw": str(m.content)}
                    # prefer the last non-empty submit result
        elif isinstance(m, AIMessage) and getattr(m, "content", ""):
            if not ai_text:
                ai_text = m.content
    # prefer the earliest-looking final AI text; keep last non-empty
    for m in messages:
        if isinstance(m, AIMessage) and getattr(m, "content", ""):
            ai_text = m.content
    return ai_text, submit_result


async def run(
    experiment: str,
    base_url: str = DEFAULT_BASE_URL,
    model_ref: str | None = None,
    max_tool_calls: int = 40,
    thinking: str = "high",
) -> int:
    from huginn.config import HuginnConfig
    from huginn.models.registry import ModelRegistry

    os.environ.setdefault("HUGINN_THINKING", thinking)
    # benchmark 场景 (无人工 subprocess): 与 rcb_runner 对齐, 走 CSM 子集模式 —
    # CSM transition 依旧执行 (含 S3/S6), 但不触发不必要的 context compaction.
    os.environ.setdefault("HUGINN_CSM_SUBSET_MODE", "1")
    cfg = HuginnConfig.from_env()
    registry = ModelRegistry.from_config(cfg)
    alias = registry.default_alias()
    model = registry.resolve(alias) if alias else registry.resolve(model_ref or "deepseek/deepseek-chat")

    # Bind a CausalGame session for this experiment.
    from agent.client import CanyonClient  # now on sys.path
    client = CanyonClient(
        base_url=base_url,
        model_name=getattr(model, "name", None) or model_ref or "huginn",
        agent_name="huginn",
        experiment=experiment,
        execution_mode="hybrid",
    )
    print(f"[causal] experiment={experiment} session={client._session_id} model={model_ref or alias}", flush=True)

    ws = Path(os.environ.get("HUGINN_WORKSPACE", "/workspace")) / "causalgame_ws" / experiment
    ws.mkdir(parents=True, exist_ok=True)

    from huginn.agent import HuginnAgent
    agent = HuginnAgent(
        model=model,
        system_prompt=_system_prompt(),
        workspace=ws,
        max_tool_output_tokens=cfg.max_tool_output_tokens,
        context_budget_tokens=cfg.context_budget_tokens,
        max_tool_calls=max_tool_calls,
        auto_approve=True,
    )
    for t in _make_tools(client):
        agent.register_tool(t)

    msg = (
        "Begin the CausalGame mission for scenario '" + experiment + "'. "
        "Read the action space, run controlled experiments to uncover the hidden causal "
        "mechanism of survival, then submit your final design once."
    )
    thread_id = f"causal_{experiment}"
    print("\n=== Huginn agent running ===\n", flush=True)
    try:
        async for chunk in agent.chat(msg, thread_id=thread_id):
            messages = chunk.get("messages", [])
            if not messages:
                continue
            for c in chunk.get("streaming", []):
                if isinstance(c, str):
                    print(c, end="", flush=True)
    except Exception as e:
        print(f"\n[ERROR] {e}", flush=True)

    # Fallback: if the agent exhausted its deployment budget but never submitted,
    # force a final submit turn so a discovered best design is not thrown away.
    try:
        st = client.get_status()
        if not st.get("game_over") and st.get("deployments_remaining", 0) <= 0:
            print("\n[budget exhausted, forcing submit turn]\n", flush=True)
            async for chunk in agent.chat(
                "Your deployment budget is now exhausted. You must immediately call "
                "causal_submit with the best design you have discovered. Do not deploy again.",
                thread_id=thread_id,
            ):
                for c in chunk.get("streaming", []):
                    if isinstance(c, str):
                        print(c, end="", flush=True)
    except Exception as e:
        print(f"\n[force-submit error] {e}", flush=True)

    # Report final outcome.
    try:
        st = client.get_status()
        final = st.get("final_evaluation") or st.get("final_result")
        print("\n=== final status ===\n", flush=True)
        print(json.dumps({
            "game_over": st.get("game_over"),
            "stage": st.get("stage"),
            "deployments_used": st.get("deployments_used"),
            "drones_used": st.get("drones_used"),
            "final_evaluation": final,
        }, ensure_ascii=False, indent=2), flush=True)
    except Exception as e:
        print(f"\n[final status error] {e}", flush=True)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Huginn CausalGame adapter runner")
    parser.add_argument("--experiment", required=True, help="CausalGame scenario name")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=None, help="provider/model ref, e.g. deepseek/deepseek-chat")
    parser.add_argument("--max-tool-calls", type=int, default=40)
    parser.add_argument("--thinking", default=os.environ.get("HUGINN_THINKING", "high"))
    args = parser.parse_args()
    asyncio.run(run(
        args.experiment, args.base_url,
        model_ref=args.model, max_tool_calls=args.max_tool_calls,
        thinking=args.thinking,
    ))


if __name__ == "__main__":
    main()
