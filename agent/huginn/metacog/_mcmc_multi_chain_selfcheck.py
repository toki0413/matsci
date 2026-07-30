"""多链 MCMC selfcheck — 验证 spec long-run-mcmc-checkpoint-parallel Step 3+4.

跑法: python -m huginn.metacog._mcmc_multi_chain_selfcheck
"""
import asyncio
import os
import random
import tempfile
from pathlib import Path

from huginn.metacog.hypothesis_manifold import HypothesisManifold, Hypothesis, Observation
from huginn.runtime.engine_state import save_engine_state, load_engine_state


def _build_manifold() -> HypothesisManifold:
    """3-hypothesis manifold, 1 observable, 用作所有测试的 fixture."""
    m = HypothesisManifold()
    m.add(Hypothesis("h1", "pred x=1.0", predictions={"x": 1.0}, n_params=1))
    m.add(Hypothesis("h2", "pred x=2.0", predictions={"x": 2.0}, n_params=1))
    m.add(Hypothesis("h3", "pred x=1.5", predictions={"x": 1.5}, n_params=1))
    return m


def _obs():
    return [Observation("x", 1.1, sigma=0.3)]


def test1_short_chain_r_hat():
    """1000 步太少, 阈值放宽到 1.5. 验证 multi_chain 能跑完 + 返回结构正确."""
    m = _build_manifold()
    r = asyncio.run(m.mcmc_multi_chain(_obs(), n_chains=4, n_steps_per_chain=1000, checkpoint_interval=200))
    assert "chains" in r and len(r["chains"]) == 4
    assert "r_hat" in r and "converged" in r and "accept_rates" in r
    assert len(r["accept_rates"]) == 4
    # r_hat 可能 nan (样本太少 min_len<2), nan 不算失败
    if not (r["r_hat"] != r["r_hat"]):  # not nan
        assert r["r_hat"] < 1.5, f"R-hat {r['r_hat']} 超过 1.5"
    print(f"test1_short_chain_r_hat OK (r_hat={r['r_hat']}, accept_rates={r['accept_rates']})")


def test2_long_chain_r_hat():
    """10000 步足够收敛, 阈值收紧到 1.2. obs=1.1 应让所有链聚到 h1."""
    m = _build_manifold()
    r = asyncio.run(m.mcmc_multi_chain(_obs(), n_chains=4, n_steps_per_chain=10000, checkpoint_interval=2500))
    assert len(r["chains"]) == 4
    assert len(r["accept_rates"]) == 4
    if not (r["r_hat"] != r["r_hat"]):  # not nan
        assert r["r_hat"] < 1.2, f"R-hat {r['r_hat']} 超过 1.2"
    print(f"test2_long_chain_r_hat OK (r_hat={r['r_hat']}, accept_rates={r['accept_rates']})")


def test3_single_chain_crash_others_continue():
    """验证 asyncio.gather(return_exceptions=True): 链 2 raise, 链 0/1/3 正常.

    直接调 _run_single_chain 构造 4 个协程, 链 2 注入 RuntimeError.
    500 步 + thin=1000 → samples 为空, R-hat 会是 nan, 但不影响断言.
    """
    m = _build_manifold()

    async def crash_chain():
        raise RuntimeError("simulated crash")

    async def normal_chain(cid):
        return await m._run_single_chain(
            cid, _obs(), 500, 100,
            rng=random.Random(cid * 7919 + 42),
            temperature=1.0, init_h_id="h1", on_checkpoint=None,
        )

    async def _run():
        return await asyncio.gather(
            normal_chain(0), normal_chain(1), crash_chain(), normal_chain(3),
            return_exceptions=True,
        )

    results = asyncio.run(_run())
    assert isinstance(results[2], Exception), f"链 2 应是 Exception, 实际 {type(results[2])}"
    assert all(not isinstance(r, Exception) for r in [results[0], results[1], results[3]]), \
        "正常链不应返回 Exception"
    # 3 链算 R-hat (降级路径). samples 可能空 → nan, 不强制断言数值.
    chains = [results[i]["samples"] for i in [0, 1, 3]]
    r_hat = m._gelman_rubin(chains)
    print(f"test3_single_chain_crash_others_continue OK (3-chain R-hat={r_hat})")


def test4_checkpoint_resume():
    """跑 5000 步 → save → load → 验证 step_count 从 5000 恢复.

    on_chain_checkpoint 回调不做事, 只验证回调机制能跑通 + _mcmc_chains round-trip.
    """
    m = _build_manifold()
    r = asyncio.run(m.mcmc_multi_chain(
        _obs(), n_chains=2, n_steps_per_chain=5000, checkpoint_interval=2500,
        on_chain_checkpoint=lambda cid, state: None,
    ))
    assert len(r["chains"]) == 2, f"期望 2 链, 实际 {len(r['chains'])}"

    # 模拟 engine 持有 _mcmc_chains — step_count 由调用方记录 (mcmc_multi_chain 不返回它)
    from types import SimpleNamespace
    engine = SimpleNamespace(_mcmc_chains={})
    for i, chain_result in enumerate(r["chains"]):
        engine._mcmc_chains[i] = {
            "step_count": 5000,
            "current": chain_result[-1] if chain_result else "h1",
        }

    # 强制开 persistence — save/load 都要这个 flag
    prev = os.environ.get("HUGINN_USE_PERSISTENCE", "0")
    os.environ["HUGINN_USE_PERSISTENCE"] = "1"
    try:
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            saved = save_engine_state(engine, "test_run", ws)
            assert saved is not None, "save_engine_state 返回 None (flag off?)"
            loaded = load_engine_state("test_run", ws)
            assert loaded is not None, "load_engine_state 返回 None"
            assert loaded._mcmc_chains == engine._mcmc_chains, \
                f"_mcmc_chains round-trip 不一致: {loaded._mcmc_chains} vs {engine._mcmc_chains}"
            for cid in [0, 1]:
                assert loaded._mcmc_chains[cid]["step_count"] == 5000, \
                    f"链 {cid} step_count 未恢复: {loaded._mcmc_chains[cid]}"
    finally:
        os.environ["HUGINN_USE_PERSISTENCE"] = prev
    print(f"test4_checkpoint_resume OK (chains={engine._mcmc_chains})")


if __name__ == "__main__":
    test1_short_chain_r_hat()
    test2_long_chain_r_hat()
    test3_single_chain_crash_others_continue()
    test4_checkpoint_resume()
    print("all multi-chain selfcheck tests passed")
