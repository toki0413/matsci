"""Phase E 压力测试 — 工业级交付验证.

测试维度:
1. 数值稳定性: 极端输入 (NaN, inf, 零方差, 完全冲突)
2. 大规模数据: 大样本 MCMC, 高维 DOE
3. 重复执行: 确定性验证 (seed 复现)
4. 边界条件: 空列表, 单元素, 退化输入
5. 并发安全: 多次实例化 + 交叉调用
6. 协议鲁棒性: 注入尝试, 超长字符串, 嵌套 dict
"""

from __future__ import annotations

import asyncio
import math
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor

import numpy as np

# 确保能 import
sys.path.insert(0, ".")

from huginn.autoloop.phase_gate import (
    DempsterShaferCombiner,
    MathEvidenceChecker,
    PhaseGateHook,
    PhaseGate,
)
from huginn.tools.sci.multi_fidelity_tool import MultiFidelityInput, MultiFidelityTool
from huginn.tools.wetlab_rpc_tool import (
    PROTOCOLS,
    WetlabInput,
    WetlabRpcTool,
    _parse_protocol_result,
    _validate_protocol_params,
)


def _ok(name: str, condition: bool, detail: str = "") -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    return condition


# ── 1. 数值稳定性 ─────────────────────────────────────────────────


def stress_dempster_shafer_extreme():
    """DS 组合的极端输入."""
    print("\n=== 1. Dempster-Shafer 数值稳定性 ===")
    all_pass = True

    # 完全冲突: m1=(1,0,0), m2=(0,1,0) → K=1, 应返回全 fail
    try:
        result = DempsterShaferCombiner.combine_pair((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        all_pass &= _ok(
            "完全冲突返回全 fail",
            result == (0.0, 1.0, 0.0),
            f"got {result}",
        )
    except Exception as e:
        all_pass &= _ok("完全冲突不崩溃", False, str(e))

    # 零不确定性完全一致: m1=(0.5,0.5,0), m2=(0.5,0.5,0)
    try:
        result = DempsterShaferCombiner.combine_pair((0.5, 0.5, 0.0), (0.5, 0.5, 0.0))
        total = sum(result)
        all_pass &= _ok(
            "零不确定性一致组合归一化",
            abs(total - 1.0) < 1e-9,
            f"sum={total}",
        )
    except Exception as e:
        all_pass &= _ok("零不确定性组合不崩溃", False, str(e))

    # 接近零的值: m1=(1e-15, 1e-15, 1-2e-15)
    try:
        m1 = (1e-15, 1e-15, 1.0 - 2e-15)
        m2 = (1e-15, 1e-15, 1.0 - 2e-15)
        result = DempsterShaferCombiner.combine_pair(m1, m2)
        total = sum(result)
        all_pass &= _ok(
            "极小值组合归一化",
            abs(total - 1.0) < 1e-6 or result == (0.0, 1.0, 0.0),
            f"sum={total}, result={result}",
        )
    except Exception as e:
        all_pass &= _ok("极小值组合不崩溃", False, str(e))

    # 大规模组合: 100 个证据源
    try:
        masses = [(0.6, 0.1, 0.3)] * 100
        result = DempsterShaferCombiner.combine(masses)
        total = sum(result)
        all_pass &= _ok(
            "100源组合归一化",
            abs(total - 1.0) < 1e-6,
            f"sum={total}, result={result}",
        )
    except Exception as e:
        all_pass &= _ok("100源组合不崩溃", False, str(e))

    return all_pass


def stress_math_checker_edge_cases():
    """MathEvidenceChecker 边界条件."""
    print("\n=== 2. MathEvidenceChecker 边界条件 ===")
    all_pass = True
    checker = MathEvidenceChecker()

    # 空证据
    try:
        passed, feedback, details = checker({})
        all_pass &= _ok(
            "空证据放行",
            passed is True,
            f"passed={passed}, feedback={feedback}",
        )
    except Exception as e:
        all_pass &= _ok("空证据不崩溃", False, str(e))

    # 所有证据都 None
    try:
        passed, feedback, details = checker({
            "conservation_law": None,
            "dimensional_consistent": None,
            "pde_classification": None,
        })
        all_pass &= _ok(
            "None 证据放行",
            passed is True,
            f"passed={passed}",
        )
    except Exception as e:
        all_pass &= _ok("None 证据不崩溃", False, str(e))

    # 混合 bool / dict / None
    try:
        passed, feedback, details = checker({
            "conservation_law": True,
            "dimensional_consistent": {"passed": False, "detail": "unit mismatch"},
            "pde_classification": None,
            "sobol_top_features": True,
            "constraint_check": False,
        })
        all_pass &= _ok(
            "混合类型证据不崩溃",
            isinstance(passed, bool),
            f"passed={passed}",
        )
    except Exception as e:
        all_pass &= _ok("混合类型证据不崩溃", False, str(e))

    # 异常值: 字符串、数字
    try:
        passed, feedback, details = checker({
            "conservation_law": "yes",
            "dimensional_consistent": 42,
        })
        all_pass &= _ok(
            "异常类型证据降级",
            isinstance(passed, bool),
            f"passed={passed}",
        )
    except Exception as e:
        all_pass &= _ok("异常类型证据不崩溃", False, str(e))

    return all_pass


# ── 2. 大规模数据 ─────────────────────────────────────────────────


async def stress_bayesian_calibrate_large():
    """大规模 MCMC 采样."""
    print("\n=== 3. Bayesian Calibrate 大规模 ===")
    all_pass = True
    tool = MultiFidelityTool()

    # 大样本: 5000 MCMC 采样, 20 个 HF 点
    try:
        np.random.seed(42)
        X_lf = np.random.rand(50, 3).tolist()
        y_lf = [sum(x) for x in X_lf]
        X_hf = np.random.rand(20, 3).tolist()
        y_hf = [2 * sum(x) + 0.5 for x in X_hf]

        result = await tool.call({
            "action": "bayesian_calibrate",
            "X_lf": X_lf,
            "y_lf": y_lf,
            "X_hf": X_hf,
            "y_hf": y_hf,
            "theta_prior_low": [0.0, 0.0, 0.0],
            "theta_prior_high": [1.0, 1.0, 1.0],
            "n_mcmc_samples": 5000,
            "n_burnin": 1000,
            "proposal_std": 0.05,
            "sigma_n": 0.1,
            "seed": 42,
        })
        all_pass &= _ok(
            "5000样本 MCMC 完成",
            result.success,
            f"acceptance_rate={result.data.get('acceptance_rate') if result.success else result.error}",
        )
        if result.success:
            samples = np.array(result.data["posterior_samples"])
            all_pass &= _ok(
                "后验样本数 = 4000",
                len(samples) == 4000,
                f"got {len(samples)}",
            )
            # 后验均值在合理范围
            post_mean = np.array(result.data["posterior_mean"])
            all_pass &= _ok(
                "后验均值在 [0,1] 内",
                np.all(post_mean >= -0.1) and np.all(post_mean <= 1.1),
                f"mean={post_mean}",
            )
    except Exception as e:
        all_pass &= _ok("大规模 MCMC 不崩溃", False, str(e))

    return all_pass


async def stress_nested_doe_high_dim():
    """高维 DOE."""
    print("\n=== 4. Nested DOE 高维 ===")
    all_pass = True
    tool = MultiFidelityTool()

    # 10 维, 100 LF 点, 20 HF 点
    try:
        result = await tool.call({
            "action": "nested_doe",
            "n_hf": 20,
            "n_lf": 100,
            "dim": 10,
            "bounds_low": [0.0] * 10,
            "bounds_high": [1.0] * 10,
            "seed": 42,
        })
        all_pass &= _ok(
            "10维 DOE 完成",
            result.success,
            f"error={result.error}" if not result.success else "",
        )
        if result.success:
            X_lf = np.array(result.data["X_lf"])
            X_hf = np.array(result.data["X_hf"])
            all_pass &= _ok("LF shape=(100,10)", X_lf.shape == (100, 10), f"got {X_lf.shape}")
            all_pass &= _ok("HF shape=(20,10)", X_hf.shape == (20, 10), f"got {X_hf.shape}")
            # 嵌套性: HF ⊂ LF
            hf_set = {tuple(x) for x in X_hf}
            lf_set = {tuple(x) for x in X_lf}
            all_pass &= _ok("HF ⊂ LF", hf_set.issubset(lf_set), f"missing {len(hf_set - lf_set)}")
    except Exception as e:
        all_pass &= _ok("高维 DOE 不崩溃", False, str(e))

    return all_pass


async def stress_variance_reduction_large():
    """大样本方差缩减."""
    print("\n=== 5. Variance Reduction 大样本 ===")
    all_pass = True
    tool = MultiFidelityTool()

    # 10000 样本
    try:
        np.random.seed(42)
        n = 10000
        y_lf = np.random.randn(n).tolist()
        # HF = LF + small noise (高相关)
        y_hf = (np.array(y_lf) + 0.1 * np.random.randn(n)).tolist()

        result = await tool.call({
            "action": "variance_reduction",
            "y_hf_samples": y_hf,
            "y_lf_samples": y_lf,
        })
        all_pass &= _ok(
            "10000样本完成",
            result.success,
            f"error={result.error}" if not result.success else "",
        )
        if result.success:
            ratio = result.data["reduction_ratio"]
            all_pass &= _ok(
                "高相关 → 显著缩减",
                ratio > 0.5,
                f"reduction_ratio={ratio:.4f}",
            )
    except Exception as e:
        all_pass &= _ok("大样本不崩溃", False, str(e))

    return all_pass


# ── 3. 协议鲁棒性 / 注入测试 ─────────────────────────────────────


async def stress_protocol_injection():
    """协议参数注入测试."""
    print("\n=== 6. 协议注入鲁棒性 ===")
    all_pass = True
    tool = WetlabRpcTool()

    # 超长字符串参数
    try:
        result = await tool.call(WetlabInput(
            action="submit_protocol",
            protocol="XRD",
            params={
                "scan_range": [10, 80],
                "step_size": 0.02,
                "dwell_time": 1.0,
                "wavelength": 1.5406,
            },
            sample={
                "sample_id": "A" * 10000,
                "composition": "TiO2",
                "preparation_method": "sol-gel",
            },
        ), context=None)
        all_pass &= _ok("超长 sample_id 不崩溃", result.success, f"error={result.error}" if not result.success else "")
    except Exception as e:
        all_pass &= _ok("超长 sample_id 不崩溃", False, str(e))

    # 嵌套 dict 作为参数值
    try:
        result = await tool.call(WetlabInput(
            action="submit_protocol",
            protocol="XRD",
            params={
                "scan_range": [10, 80],
                "step_size": 0.02,
                "dwell_time": 1.0,
                "extra_nested": {"key": {"nested": "value"}},
            },
            sample={
                "sample_id": "S001",
                "composition": "TiO2",
                "preparation_method": "sol-gel",
            },
        ), context=None)
        # 未知参数被忽略, 不报错
        all_pass &= _ok("嵌套 dict 参数被忽略", result.success, f"error={result.error}" if not result.success else "")
    except Exception as e:
        all_pass &= _ok("嵌套 dict 参数不崩溃", False, str(e))

    # NaN 参数值
    try:
        result = await tool.call(WetlabInput(
            action="submit_protocol",
            protocol="XRD",
            params={
                "scan_range": [10, 80],
                "step_size": float("nan"),
                "dwell_time": 1.0,
            },
            sample={
                "sample_id": "S001",
                "composition": "TiO2",
                "preparation_method": "sol-gel",
            },
        ), context=None)
        # NaN 应该被范围检查拦截 (nan < 0.005 is False, nan > 1.0 is False, 所以可能放行)
        # 关键是不崩溃
        all_pass &= _ok("NaN 参数不崩溃", isinstance(result.success, bool), f"success={result.success}")
    except Exception as e:
        all_pass &= _ok("NaN 参数不崩溃", False, str(e))

    # inf 参数值
    try:
        result = await tool.call(WetlabInput(
            action="submit_protocol",
            protocol="XRD",
            params={
                "scan_range": [10, 80],
                "step_size": float("inf"),
                "dwell_time": 1.0,
            },
            sample={
                "sample_id": "S001",
                "composition": "TiO2",
                "preparation_method": "sol-gel",
            },
        ), context=None)
        all_pass &= _ok("inf 参数被范围检查拦截", not result.success, f"success={result.success}")
    except Exception as e:
        all_pass &= _ok("inf 参数不崩溃", False, str(e))

    # 负数 scan_range
    try:
        result = await tool.call(WetlabInput(
            action="submit_protocol",
            protocol="XRD",
            params={
                "scan_range": [-10, 80],
                "step_size": 0.02,
                "dwell_time": 1.0,
            },
            sample={
                "sample_id": "S001",
                "composition": "TiO2",
                "preparation_method": "sol-gel",
            },
        ), context=None)
        # scan_range 是 list, 不做数值范围检查, 但不应该崩溃
        all_pass &= _ok("负数 scan_range 不崩溃", isinstance(result.success, bool))
    except Exception as e:
        allok = _ok("负数 scan_range 不崩溃", False, str(e))
        all_pass &= allok

    # parse_result 注入: raw_result 包含超长 dict
    try:
        big_raw = {f"key_{i}": i for i in range(1000)}
        big_raw["peaks"] = [{"two_theta": 25.3, "intensity": 100}]
        big_raw["crystallite_size_nm"] = 32.5
        big_raw["phase_ids"] = ["anatase"]
        result = await tool.call(WetlabInput(
            action="parse_result",
            protocol="XRD",
            raw_result=big_raw,
        ), context=None)
        all_pass &= _ok("超大 raw_result 解析", result.success, f"error={result.error}" if not result.success else "")
    except Exception as e:
        all_pass &= _ok("超大 raw_result 不崩溃", False, str(e))

    return all_pass


# ── 4. 并发安全 ─────────────────────────────────────────────────


async def stress_concurrent_access():
    """多实例并发调用."""
    print("\n=== 7. 并发安全 ===")
    all_pass = True

    # 多个 tool 实例并发
    tools = [MultiFidelityTool() for _ in range(5)]

    async def register_and_fit(tool: MultiFidelityTool, idx: int):
        await tool.call({
            "action": "register_source",
            "name": f"src_{idx}",
            "level": 0,
            "cost": 1.0,
            "X": [[0.1 * idx], [0.3 * idx], [0.5 * idx]],
            "y": [0.1 * idx, 0.3 * idx, 0.5 * idx],
        })
        return await tool.call({"action": "fit_surrogate"})

    try:
        results = await asyncio.gather(*[
            register_and_fit(t, i) for i, t in enumerate(tools)
        ])
        all_ok = all(r.success for r in results)
        all_pass &= _ok("5并发实例独立工作", all_ok, f"successes={sum(r.success for r in results)}/5")
    except Exception as e:
        all_pass &= _ok("并发不崩溃", False, str(e))

    # 同一实例的并发调用 (应该安全, 因为 call 是 async)
    shared_tool = MultiFidelityTool()
    try:
        tasks = []
        for i in range(10):
            tasks.append(shared_tool.call({
                "action": "register_source",
                "name": f"concurrent_{i}",
                "level": 0,
                "cost": 1.0,
                "X": [[float(i)]],
                "y": [float(i)],
            }))
        results = await asyncio.gather(*tasks)
        all_ok = all(r.success for r in results)
        all_pass &= _ok("10并发注册到同一实例", all_ok, f"successes={sum(r.success for r in results)}/10")
    except Exception as e:
        all_pass &= _ok("同实例并发不崩溃", False, str(e))

    return all_pass


# ── 5. 确定性验证 ─────────────────────────────────────────────────


async def stress_determinism():
    """同 seed 重复执行结果一致."""
    print("\n=== 8. 确定性验证 ===")
    all_pass = True

    # nested_doe seed 复现
    tool1 = MultiFidelityTool()
    tool2 = MultiFidelityTool()
    try:
        r1 = await tool1.call({
            "action": "nested_doe",
            "n_hf": 10, "n_lf": 30, "dim": 3,
            "bounds_low": [0, 0, 0], "bounds_high": [1, 1, 1],
            "seed": 12345,
        })
        r2 = await tool2.call({
            "action": "nested_doe",
            "n_hf": 10, "n_lf": 30, "dim": 3,
            "bounds_low": [0, 0, 0], "bounds_high": [1, 1, 1],
            "seed": 12345,
        })
        same = r1.data["X_lf"] == r2.data["X_lf"]
        all_pass &= _ok("同 seed DOE 复现", same, "")
    except Exception as e:
        all_pass &= _ok("seed 复现不崩溃", False, str(e))

    # bayesian_calibrate seed 复现
    tool3 = MultiFidelityTool()
    tool4 = MultiFidelityTool()
    try:
        calib_args = {
            "action": "bayesian_calibrate",
            "X_lf": [[0.1], [0.5], [0.9]],
            "y_lf": [0.1, 0.5, 0.9],
            "X_hf": [[0.3], [0.7]],
            "y_hf": [0.6, 1.4],
            "theta_prior_low": [0.0],
            "theta_prior_high": [1.0],
            "n_mcmc_samples": 500,
            "n_burnin": 100,
            "proposal_std": 0.1,
            "sigma_n": 0.1,
            "seed": 999,
        }
        r1 = await tool3.call(dict(calib_args))
        r2 = await tool4.call(dict(calib_args))
        s1 = np.array(r1.data["posterior_samples"])
        s2 = np.array(r2.data["posterior_samples"])
        same = np.allclose(s1, s2)
        all_pass &= _ok("同 seed MCMC 复现", same, f"max_diff={np.max(np.abs(s1-s2)):.2e}" if not same else "")
    except Exception as e:
        all_pass &= _ok("MCMC seed 复现不崩溃", False, str(e))

    return all_pass


# ── 6. 退化输入 ─────────────────────────────────────────────────


async def stress_degenerate_inputs():
    """退化输入: 最小数据, 单点, 零方差."""
    print("\n=== 9. 退化输入 ===")
    all_pass = True

    # 单点 HF + 单点 LF
    tool = MultiFidelityTool()
    try:
        result = await tool.call({
            "action": "bayesian_calibrate",
            "X_lf": [[0.5]],
            "y_lf": [0.5],
            "X_hf": [[0.5]],
            "y_hf": [1.0],
            "theta_prior_low": [0.0],
            "theta_prior_high": [1.0],
            "n_mcmc_samples": 200,
            "n_burnin": 50,
            "seed": 42,
        })
        all_pass &= _ok("单点数据不崩溃", result.success, f"error={result.error}" if not result.success else "")
    except Exception as e:
        all_pass &= _ok("单点数据不崩溃", False, str(e))

    # 零方差 HF
    tool2 = MultiFidelityTool()
    try:
        result = await tool2.call({
            "action": "variance_reduction",
            "y_hf_samples": [1.0] * 100,
            "y_lf_samples": [1.0] * 100,
        })
        all_pass &= _ok("零方差数据不崩溃", result.success, f"error={result.error}" if not result.success else "")
        if result.success:
            # 零方差 → var=0, 不应出 NaN
            vr = result.data["variance_reduced"]
            all_pass &= _ok("零方差结果无 NaN", not math.isnan(vr), f"variance_reduced={vr}")
    except Exception as e:
        all_pass &= _ok("零方差不崩溃", False, str(e))

    # variance_reduction: HF 全正, LF 全负 (负相关)
    tool3 = MultiFidelityTool()
    try:
        result = await tool3.call({
            "action": "variance_reduction",
            "y_hf_samples": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y_lf_samples": [-1.0, -2.0, -3.0, -4.0, -5.0],
        })
        all_pass &= _ok("负相关数据不崩溃", result.success, f"error={result.error}" if not result.success else "")
    except Exception as e:
        all_pass &= _ok("负相关不崩溃", False, str(e))

    return all_pass


# ── 主入口 ───────────────────────────────────────────────────────


async def main():
    print("=" * 60)
    print("Phase E 压力测试 — 工业级交付验证")
    print("=" * 60)

    results = []

    # 同步测试
    results.append(stress_dempster_shafer_extreme())
    results.append(stress_math_checker_edge_cases())

    # 异步测试
    results.append(await stress_bayesian_calibrate_large())
    results.append(await stress_nested_doe_high_dim())
    results.append(await stress_variance_reduction_large())
    results.append(await stress_protocol_injection())
    results.append(await stress_concurrent_access())
    results.append(await stress_determinism())
    results.append(await stress_degenerate_inputs())

    # 汇总
    print("\n" + "=" * 60)
    total = len(results)
    passed = sum(results)
    print(f"总计: {passed}/{total} 组通过")
    if passed == total:
        print("✅ 全部压力测试通过")
    else:
        print(f"❌ {total - passed} 组失败")
    print("=" * 60)

    return 0 if passed == total else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
