"""验证: 温度退火 + 全局 proposal 混合是否提升 MCMC 探索性 (接受率)."""
import asyncio
import random

from huginn.metacog.hypothesis_manifold import (
    HypothesisManifold, Hypothesis, Observation,
)


def build() -> HypothesisManifold:
    m = HypothesisManifold()
    targets = {"accuracy": 0.92}
    for h in (
        Hypothesis(h_id="h_paper_repro", description="Paper results reproducible",
                   predictions=targets, n_params=2),
        Hypothesis(h_id="h_partial_repro", description="Partial reproduction",
                   predictions={k: v * 0.5 for k, v in targets.items()}, n_params=3),
        Hypothesis(h_id="h_null_baseline", description="Null/baseline",
                   predictions={k: 0.0 for k in targets}, n_params=1),
    ):
        try:
            m.add(h)
        except ValueError:
            pass
    return m


obs = [Observation(name="accuracy", value=0.92, sigma=0.1)]

# 1. 原行为 (无全局混合, 无退火): 从 h_paper (MAP) 出发
m = build()
rng = random.Random(42)
curr, cached = "h_paper_repro", None
acc = 0
for _ in range(2000):
    prev = curr
    curr, cached = m.mcmc_step(obs, curr, rng=rng, cached_log_p_current=cached)
    if curr != prev:
        acc += 1
print(f"[原行为] MAP出发 2000步 accept_rate={acc/2000:.4f}")

# 2. 全局混合 (无退火): global_proposal_prob=0.3
m = build()
rng = random.Random(42)
curr, cached = "h_paper_repro", None
acc = 0
for _ in range(2000):
    prev = curr
    curr, cached = m.mcmc_step(obs, curr, rng=rng, cached_log_p_current=cached,
                               global_proposal_prob=0.3)
    if curr != prev:
        acc += 1
print(f"[全局混合] MAP出发 2000步 accept_rate={acc/2000:.4f}")

# 3. 全局混合 + 退火: 高温初启探索
m = build()
rng = random.Random(42)
curr, cached = "h_paper_repro", None
acc = 0
n = 2000
for step in range(1, n + 1):
    prev = curr
    T = 10.0 * (1.0 / 10.0) ** (step / n)  # 退火
    curr, cached = m.mcmc_step(obs, curr, rng=rng, cached_log_p_current=cached,
                               temperature=T, global_proposal_prob=0.3)
    if curr != prev:
        acc += 1
print(f"[全局+退火] MAP出发 {n}步 accept_rate={acc/n:.4f}")


# 4. 多链 (mcmc_multi_chain) 端到端: 验证新参数接线 + 接受率
async def multi():
    m = build()
    res = await m.mcmc_multi_chain(
        obs, n_chains=2, n_steps_per_chain=500,
        checkpoint_interval=0, anneal=True, t_high=10.0,
        global_proposal_prob=0.3,
    )
    return res

res = asyncio.run(multi())
print(f"[multi] accept_rates={[round(r,3) for r in res['accept_rates']]} "
      f"r_hat={res['r_hat']:.3f} converged={res['converged']}")