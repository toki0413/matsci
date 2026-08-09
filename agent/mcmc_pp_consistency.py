"""验证 ProcessPool 与 asyncio 两条路径结果一致性 + R̂ 计算."""
import asyncio
import os

from huginn.metacog.hypothesis_manifold import (
    HypothesisManifold, Hypothesis, Observation,
)


def build() -> HypothesisManifold:
    m = HypothesisManifold()
    for hid, desc, scale, n in (
        ("h_a", "Hypothesis A", 1.0, 2),
        ("h_b", "Hypothesis B", 0.5, 3),
        ("h_c", "Hypothesis C", 0.0, 1),
    ):
        try:
            m.add(Hypothesis(hid, desc, predictions={"accuracy": 0.92 * scale}, n_params=n))
        except ValueError:
            pass
    return m


obs = [Observation(name="accuracy", value=0.92, sigma=0.1)]


async def run_pp():
    os.environ["HUGINN_MCMC_PARALLEL"] = "1"
    m = build()
    return await m.mcmc_multi_chain(
        obs, n_chains=4, n_steps_per_chain=20000,
        checkpoint_interval=0, anneal=True, t_high=10.0, global_proposal_prob=0.3,
    )


async def run_async_path():
    os.environ["HUGINN_MCMC_PARALLEL"] = "0"
    m = build()
    return await m.mcmc_multi_chain(
        obs, n_chains=4, n_steps_per_chain=20000,
        checkpoint_interval=0, anneal=True, t_high=10.0, global_proposal_prob=0.3,
    )


print("=== ProcessPool (n_steps=20000, 4 链) ===")
pp = asyncio.run(run_pp())
print(f"  r_hat={pp['r_hat']:.3f} converged={pp['converged']} "
      f"accept={[round(a,3) for a in pp['accept_rates']]}")

print("=== asyncio (n_steps=20000, 4 链) ===")
al = asyncio.run(run_async_path())
print(f"  r_hat={al['r_hat']:.3f} converged={al['converged']} "
      f"accept={[round(a,3) for a in al['accept_rates']]}")

# 一致性: 相同种子 + 相同逻辑, 接受率应高度接近
assert len(pp["accept_rates"]) == len(al["accept_rates"]) == 4
for i, (a, b) in enumerate(zip(sorted(pp["accept_rates"]), sorted(al["accept_rates"]))):
    assert abs(a - b) < 0.02, f"chain {i} mismatch: pp={a} asyncio={b}"
print("\nPASS: ProcessPool 与 asyncio 结果一致, R̂ 可计算")