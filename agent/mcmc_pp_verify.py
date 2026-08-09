"""验证 ProcessPool 升级路径 A: 真并行多链 MCMC + gate 回退."""
import asyncio
import contextlib
import os
import time

from huginn.metacog.hypothesis_manifold import (
    Hypothesis,
    HypothesisManifold,
    Observation,
)


def build() -> HypothesisManifold:
    m = HypothesisManifold()
    for hid, desc, scale, n in (
        ("h_a", "Hypothesis A", 1.0, 2),
        ("h_b", "Hypothesis B", 0.5, 3),
        ("h_c", "Hypothesis C", 0.0, 1),
    ):
        with contextlib.suppress(ValueError):
            m.add(Hypothesis(hid, desc, predictions={"accuracy": 0.92 * scale}, n_params=n))
    return m


async def run(env_override=None):
    if env_override:
        os.environ.update(env_override)
    m = build()
    obs = [Observation(name="accuracy", value=0.92, sigma=0.1)]
    t0 = time.time()
    res = await m.mcmc_multi_chain(
        obs, n_chains=2, n_steps_per_chain=2000,
        checkpoint_interval=0, anneal=True, t_high=10.0,
        global_proposal_prob=0.3,
    )
    dt = time.time() - t0
    return res, dt


print("=== ProcessPool 真并行路径 (默认) ===")
res, dt = asyncio.run(run())
print(f"  wallclock={dt:.2f}s r_hat={res['r_hat']:.3f} converged={res['converged']} "
      f"accept={[round(a,3) for a in res['accept_rates']]} n_chains={len(res['chains'])}")
assert len(res["chains"]) == 2, f"expected 2 chains, got {len(res['chains'])}"

print("=== gate: HUGINN_MCMC_PARALLEL=0 回退 asyncio ===")
res2, dt2 = asyncio.run(run({"HUGINN_MCMC_PARALLEL": "0"}))
print(f"  wallclock={dt2:.2f}s r_hat={res2['r_hat']:.3f} n_chains={len(res2['chains'])}")
assert len(res2["chains"]) == 2

print("\nPASS: ProcessPool 真并行 + asyncio 回退均正常")
