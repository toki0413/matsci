"""端到端验证 BranchIncubator: 真实 dispatch + rollout value, 看 use 路径是否通电."""
import asyncio
import os
import sys

sys.path.insert(0, "/workspace/agent")

os.environ.setdefault("HUGINN_PROVIDER", "internlm")
os.environ.setdefault("HUGINN_MODEL", "intern-s2-preview")

TASK = (
    "Propose one concrete, testable hypothesis for: small feedforward networks as a "
    "probe of solution-space rigidity. Define N_c(w) = minimal hidden width for zero "
    "violation (max abs error <= 1e-3) on held-out constraint points. One sentence only."
)


async def main() -> None:
    from huginn.metacog.branch_incubator import BranchIncubator
    from huginn.runtime.step_verifier import (
        StepVerifierHook,
        make_branch_value_fn,
        make_default_llm_chat_fn,
    )
    from huginn.server_core import get_agent_factory

    factory = get_agent_factory()

    depth = int(os.environ.get("BRANCH_DEPTH", "2"))
    value_fn = make_branch_value_fn(StepVerifierHook(make_default_llm_chat_fn()))

    inc = BranchIncubator()
    results = await inc.run_round(
        task=TASK,
        agent_factory=factory,
        n_branches=3,
        round_idx=0,
        total_rounds=10,
        depth=depth,
        width=2,
        value_fn=value_fn,
    )
    print(f"depth={depth} returned={len(results)}")
    for r in results:
        hyp = (r.hypothesis or "")[:90]
        print(
            f"  fam={r.family_id:22s} success={r.success} "
            f"value={r.value} tokens={r.tokens_used} "
            f"parent={'Y' if r.parent_agent_id else 'N'} err={r.error} hyp={hyp!r}"
        )
    ok = [r for r in results if r.success and r.hypothesis]
    valued = [r for r in ok if r.value is not None]
    print(f"ok={len(ok)} valued={len(valued)}")


if __name__ == "__main__":
    asyncio.run(main())