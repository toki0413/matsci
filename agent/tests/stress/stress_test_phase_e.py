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
from pathlib import Path

import numpy as np

# 确保能 import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from huginn.autoloop.phase_gate import (
    DempsterShaferCombiner,
    MathEvidenceChecker,
)
from huginn.tools.sci.multi_fidelity_tool import MultiFidelityTool
from huginn.tools.wetlab_rpc_tool import (
    WetlabInput,
    WetlabRpcTool,
)


def _ok(name: str, condition: bool, detail: str = "") -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    return condition


# ── 1. 数值稳定性 ─────────────────────────────────────────────────


def _ds_full_conflict() -> bool:
    """完全冲突: m1=(1,0,0), m2=(0,1,0) → K=1, 应返回全 fail."""
    try:
        result = DempsterShaferCombiner.combine_pair((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        return _ok("完全冲突返回全 fail", result == (0.0, 1.0, 0.0), f"got {result}")
    except Exception as e:
        return _ok("完全冲突不崩溃", False, str(e))


def _ds_zero_uncertainty() -> bool:
    """零不确定性完全一致: m1=(0.5,0.5,0), m2=(0.5,0.5,0)."""
    try:
        result = DempsterShaferCombiner.combine_pair((0.5, 0.5, 0.0), (0.5, 0.5, 0.0))
        total = sum(result)
        return _ok("零不确定性一致组合归一化", abs(total - 1.0) < 1e-9, f"sum={total}")
    except Exception as e:
        return _ok("零不确定性组合不崩溃", False, str(e))


def _ds_tiny_values() -> bool:
    """接近零的值: m1=(1e-15, 1e-15, 1-2e-15)."""
    try:
        m1 = (1e-15, 1e-15, 1.0 - 2e-15)
        m2 = (1e-15, 1e-15, 1.0 - 2e-15)
        result = DempsterShaferCombiner.combine_pair(m1, m2)
        total = sum(result)
        return _ok(
            "极小值组合归一化",
            abs(total - 1.0) < 1e-6 or result == (0.0, 1.0, 0.0),
            f"sum={total}, result={result}",
        )
    except Exception as e:
        return _ok("极小值组合不崩溃", False, str(e))


def _ds_hundred_sources() -> bool:
    """大规模组合: 100 个证据源."""
    try:
        masses = [(0.6, 0.1, 0.3)] * 100
        result = DempsterShaferCombiner.combine(masses)
        total = sum(result)
        return _ok(
            "100源组合归一化",
            abs(total - 1.0) < 1e-6,
            f"sum={total}, result={result}",
        )
    except Exception as e:
        return _ok("100源组合不崩溃", False, str(e))


def stress_dempster_shafer_extreme():
    """DS 组合的极端输入."""
    print("\n=== 1. Dempster-Shafer 数值稳定性 ===")
    checks = (
        _ds_full_conflict,
        _ds_zero_uncertainty,
        _ds_tiny_values,
        _ds_hundred_sources,
    )
    results = [check() for check in checks]
    return all(results)


def _mc_empty(checker) -> bool:
    """空证据应放行."""
    try:
        passed, feedback, _details = checker({})
        return _ok(
            "空证据放行", passed is True, f"passed={passed}, feedback={feedback}"
        )
    except Exception as e:
        return _ok("空证据不崩溃", False, str(e))


def _mc_all_none(checker) -> bool:
    """所有证据都 None 应放行."""
    try:
        passed, _feedback, _details = checker(
            {
                "conservation_law": None,
                "dimensional_consistent": None,
                "pde_classification": None,
            }
        )
        return _ok("None 证据放行", passed is True, f"passed={passed}")
    except Exception as e:
        return _ok("None 证据不崩溃", False, str(e))


def _mc_mixed_types(checker) -> bool:
    """混合 bool / dict / None 不崩溃."""
    try:
        passed, _feedback, _details = checker(
            {
                "conservation_law": True,
                "dimensional_consistent": {"passed": False, "detail": "unit mismatch"},
                "pde_classification": None,
                "sobol_top_features": True,
                "constraint_check": False,
            }
        )
        return _ok("混合类型证据不崩溃", isinstance(passed, bool), f"passed={passed}")
    except Exception as e:
        return _ok("混合类型证据不崩溃", False, str(e))


def _mc_odd_types(checker) -> bool:
    """异常值 (字符串、数字) 应降级而非崩溃."""
    try:
        passed, _feedback, _details = checker(
            {
                "conservation_law": "yes",
                "dimensional_consistent": 42,
            }
        )
        return _ok("异常类型证据降级", isinstance(passed, bool), f"passed={passed}")
    except Exception as e:
        return _ok("异常类型证据不崩溃", False, str(e))


def stress_math_checker_edge_cases():
    """MathEvidenceChecker 边界条件."""
    print("\n=== 2. MathEvidenceChecker 边界条件 ===")
    checker = MathEvidenceChecker()
    checks = (_mc_empty, _mc_all_none, _mc_mixed_types, _mc_odd_types)
    results = [check(checker) for check in checks]
    return all(results)


# ── 2. 大规模数据 ─────────────────────────────────────────────────


def _large_mcmc_args() -> dict:
    """5000 采样 / 20 个 HF 点的 MCMC 参数."""
    np.random.seed(42)
    X_lf = np.random.rand(50, 3).tolist()
    y_lf = [sum(x) for x in X_lf]
    X_hf = np.random.rand(20, 3).tolist()
    y_hf = [2 * sum(x) + 0.5 for x in X_hf]
    return {
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
    }


def _large_mcmc_checks(result) -> bool:
    """后验样本数与均值的合理性检查."""
    samples = np.array(result.data["posterior_samples"])
    post_mean = np.array(result.data["posterior_mean"])
    checks = (
        _ok("后验样本数 = 4000", len(samples) == 4000, f"got {len(samples)}"),
        _ok(
            "后验均值在 [0,1] 内",
            np.all(post_mean >= -0.1) and np.all(post_mean <= 1.1),
            f"mean={post_mean}",
        ),
    )
    return all(checks)


async def stress_bayesian_calibrate_large():
    """大规模 MCMC 采样."""
    print("\n=== 3. Bayesian Calibrate 大规模 ===")
    tool = MultiFidelityTool()
    try:
        result = await tool.call(_large_mcmc_args())
        ok = _ok(
            "5000样本 MCMC 完成",
            result.success,
            f"acceptance_rate={result.data.get('acceptance_rate') if result.success else result.error}",
        )
        if not result.success:
            return ok
        return ok and _large_mcmc_checks(result)
    except Exception as e:
        return _ok("大规模 MCMC 不崩溃", False, str(e))


async def stress_nested_doe_high_dim():
    """高维 DOE."""
    print("\n=== 4. Nested DOE 高维 ===")
    all_pass = True
    tool = MultiFidelityTool()

    # 10 维, 100 LF 点, 20 HF 点
    try:
        result = await tool.call(
            {
                "action": "nested_doe",
                "n_hf": 20,
                "n_lf": 100,
                "dim": 10,
                "bounds_low": [0.0] * 10,
                "bounds_high": [1.0] * 10,
                "seed": 42,
            }
        )
        all_pass &= _ok(
            "10维 DOE 完成",
            result.success,
            f"error={result.error}" if not result.success else "",
        )
        if result.success:
            X_lf = np.array(result.data["X_lf"])
            X_hf = np.array(result.data["X_hf"])
            all_pass &= _ok(
                "LF shape=(100,10)", X_lf.shape == (100, 10), f"got {X_lf.shape}"
            )
            all_pass &= _ok(
                "HF shape=(20,10)", X_hf.shape == (20, 10), f"got {X_hf.shape}"
            )
            # 嵌套性: HF ⊂ LF
            hf_set = {tuple(x) for x in X_hf}
            lf_set = {tuple(x) for x in X_lf}
            all_pass &= _ok(
                "HF ⊂ LF", hf_set.issubset(lf_set), f"missing {len(hf_set - lf_set)}"
            )
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

        result = await tool.call(
            {
                "action": "variance_reduction",
                "y_hf_samples": y_hf,
                "y_lf_samples": y_lf,
            }
        )
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


_XRD_BASE_PARAMS = {"scan_range": [10, 80], "step_size": 0.02, "dwell_time": 1.0}


def _err_detail(result) -> str:
    """失败时的 error 文本 (成功时为空)."""
    return f"error={result.error}" if not result.success else ""


def _submit_xrd(params: dict, sample_id: str = "S001") -> WetlabInput:
    """构造一份 XRD 提交协议输入."""
    return WetlabInput(
        action="submit_protocol",
        protocol="XRD",
        params=params,
        sample={
            "sample_id": sample_id,
            "composition": "TiO2",
            "preparation_method": "sol-gel",
        },
    )


async def _inj_long_sample_id(tool) -> bool:
    """超长字符串参数 (sample_id 10000 字符)."""
    try:
        result = await tool.call(
            _submit_xrd(
                {**_XRD_BASE_PARAMS, "wavelength": 1.5406}, sample_id="A" * 10000
            ),
            context=None,
        )
        return _ok("超长 sample_id 不崩溃", result.success, _err_detail(result))
    except Exception as e:
        return _ok("超长 sample_id 不崩溃", False, str(e))


async def _inj_nested_param(tool) -> bool:
    """嵌套 dict 作为参数值."""
    try:
        result = await tool.call(
            _submit_xrd(
                {**_XRD_BASE_PARAMS, "extra_nested": {"key": {"nested": "value"}}}
            ),
            context=None,
        )
        # 未知参数被忽略, 不报错
        return _ok("嵌套 dict 参数被忽略", result.success, _err_detail(result))
    except Exception as e:
        return _ok("嵌套 dict 参数不崩溃", False, str(e))


async def _inj_nan_param(tool) -> bool:
    """NaN 参数值: 关键是不崩溃."""
    try:
        result = await tool.call(
            _submit_xrd({**_XRD_BASE_PARAMS, "step_size": float("nan")}),
            context=None,
        )
        # NaN 应该被范围检查拦截 (nan < 0.005 is False, nan > 1.0 is False, 所以可能放行)
        return _ok(
            "NaN 参数不崩溃",
            isinstance(result.success, bool),
            f"success={result.success}",
        )
    except Exception as e:
        return _ok("NaN 参数不崩溃", False, str(e))


async def _inj_inf_param(tool) -> bool:
    """inf 参数值应被范围检查拦截."""
    try:
        result = await tool.call(
            _submit_xrd({**_XRD_BASE_PARAMS, "step_size": float("inf")}),
            context=None,
        )
        return _ok(
            "inf 参数被范围检查拦截", not result.success, f"success={result.success}"
        )
    except Exception as e:
        return _ok("inf 参数不崩溃", False, str(e))


async def _inj_negative_scan(tool) -> bool:
    """负数 scan_range: list 不做数值范围检查, 但不应该崩溃."""
    try:
        result = await tool.call(
            _submit_xrd({**_XRD_BASE_PARAMS, "scan_range": [-10, 80]}),
            context=None,
        )
        return _ok("负数 scan_range 不崩溃", isinstance(result.success, bool))
    except Exception as e:
        return _ok("负数 scan_range 不崩溃", False, str(e))


async def _inj_big_raw_result(tool) -> bool:
    """parse_result 注入: raw_result 包含超长 dict."""
    try:
        big_raw = {f"key_{i}": i for i in range(1000)}
        big_raw["peaks"] = [{"two_theta": 25.3, "intensity": 100}]
        big_raw["crystallite_size_nm"] = 32.5
        big_raw["phase_ids"] = ["anatase"]
        result = await tool.call(
            WetlabInput(action="parse_result", protocol="XRD", raw_result=big_raw),
            context=None,
        )
        return _ok("超大 raw_result 解析", result.success, _err_detail(result))
    except Exception as e:
        return _ok("超大 raw_result 不崩溃", False, str(e))


async def stress_protocol_injection():
    """协议参数注入测试."""
    print("\n=== 6. 协议注入鲁棒性 ===")
    tool = WetlabRpcTool()
    checks = (
        _inj_long_sample_id,
        _inj_nested_param,
        _inj_nan_param,
        _inj_inf_param,
        _inj_negative_scan,
        _inj_big_raw_result,
    )
    results = [await check(tool) for check in checks]
    return all(results)


# ── 4. 并发安全 ─────────────────────────────────────────────────


def _register_args(name: str, x: list[list[float]], y: list[float]) -> dict:
    """构造一次 register_source 调用参数."""
    return {
        "action": "register_source",
        "name": name,
        "level": 0,
        "cost": 1.0,
        "X": x,
        "y": y,
    }


async def _register_and_fit(tool: MultiFidelityTool, idx: int):
    """注册数据源后拟合代理模型."""
    await tool.call(
        _register_args(
            f"src_{idx}",
            [[0.1 * idx], [0.3 * idx], [0.5 * idx]],
            [0.1 * idx, 0.3 * idx, 0.5 * idx],
        )
    )
    return await tool.call({"action": "fit_surrogate"})


async def _concurrent_separate_instances() -> bool:
    """5 个独立实例并发."""
    tools = [MultiFidelityTool() for _ in range(5)]
    try:
        results = await asyncio.gather(
            *[_register_and_fit(t, i) for i, t in enumerate(tools)]
        )
        return _ok(
            "5并发实例独立工作",
            all(r.success for r in results),
            f"successes={sum(r.success for r in results)}/5",
        )
    except Exception as e:
        return _ok("并发不崩溃", False, str(e))


async def _concurrent_shared_instance() -> bool:
    """同一实例的并发调用 (应该安全, 因为 call 是 async)."""
    shared_tool = MultiFidelityTool()
    try:
        tasks = [
            shared_tool.call(
                _register_args(f"concurrent_{i}", [[float(i)]], [float(i)])
            )
            for i in range(10)
        ]
        results = await asyncio.gather(*tasks)
        return _ok(
            "10并发注册到同一实例",
            all(r.success for r in results),
            f"successes={sum(r.success for r in results)}/10",
        )
    except Exception as e:
        return _ok("同实例并发不崩溃", False, str(e))


async def stress_concurrent_access():
    """多实例并发调用."""
    print("\n=== 7. 并发安全 ===")
    checks = (_concurrent_separate_instances, _concurrent_shared_instance)
    results = [await check() for check in checks]
    return all(results)


# ── 5. 确定性验证 ─────────────────────────────────────────────────


_DET_DOE_ARGS = {
    "action": "nested_doe",
    "n_hf": 10,
    "n_lf": 30,
    "dim": 3,
    "bounds_low": [0, 0, 0],
    "bounds_high": [1, 1, 1],
    "seed": 12345,
}

_DET_CALIB_ARGS = {
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


def _repro_detail(same: bool, s1, s2) -> str:
    """复现失败时的最大偏差文本."""
    return "" if same else f"max_diff={np.max(np.abs(s1 - s2)):.2e}"


async def _det_doe_repro() -> bool:
    """nested_doe 同 seed 复现."""
    try:
        r1 = await MultiFidelityTool().call(dict(_DET_DOE_ARGS))
        r2 = await MultiFidelityTool().call(dict(_DET_DOE_ARGS))
        same = r1.data["X_lf"] == r2.data["X_lf"]
        return _ok("同 seed DOE 复现", same, "")
    except Exception as e:
        return _ok("seed 复现不崩溃", False, str(e))


async def _det_mcmc_repro() -> bool:
    """bayesian_calibrate 同 seed 复现."""
    try:
        r1 = await MultiFidelityTool().call(dict(_DET_CALIB_ARGS))
        r2 = await MultiFidelityTool().call(dict(_DET_CALIB_ARGS))
        s1 = np.array(r1.data["posterior_samples"])
        s2 = np.array(r2.data["posterior_samples"])
        same = np.allclose(s1, s2)
        return _ok("同 seed MCMC 复现", same, _repro_detail(same, s1, s2))
    except Exception as e:
        return _ok("MCMC seed 复现不崩溃", False, str(e))


async def stress_determinism():
    """同 seed 重复执行结果一致."""
    print("\n=== 8. 确定性验证 ===")
    checks = (_det_doe_repro, _det_mcmc_repro)
    results = [await check() for check in checks]
    return all(results)


# ── 6. 退化输入 ─────────────────────────────────────────────────

_DEG_SINGLE_POINT_ARGS = {
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
}


def _variance_reduction_args(y_hf: list[float], y_lf: list[float]) -> dict:
    """构造一次 variance_reduction 调用参数."""
    return {"action": "variance_reduction", "y_hf_samples": y_hf, "y_lf_samples": y_lf}


async def _deg_single_point() -> bool:
    """单点 HF + 单点 LF."""
    try:
        result = await MultiFidelityTool().call(dict(_DEG_SINGLE_POINT_ARGS))
        return _ok("单点数据不崩溃", result.success, _err_detail(result))
    except Exception as e:
        return _ok("单点数据不崩溃", False, str(e))


async def _deg_zero_variance() -> bool:
    """零方差 HF: var=0, 不应出 NaN."""
    try:
        result = await MultiFidelityTool().call(
            _variance_reduction_args([1.0] * 100, [1.0] * 100)
        )
        ok = _ok("零方差数据不崩溃", result.success, _err_detail(result))
        if not result.success:
            return ok
        vr = result.data["variance_reduced"]
        return ok and _ok(
            "零方差结果无 NaN", not math.isnan(vr), f"variance_reduced={vr}"
        )
    except Exception as e:
        return _ok("零方差不崩溃", False, str(e))


async def _deg_negative_correlation() -> bool:
    """HF 全正, LF 全负 (负相关)."""
    try:
        result = await MultiFidelityTool().call(
            _variance_reduction_args(
                [1.0, 2.0, 3.0, 4.0, 5.0], [-1.0, -2.0, -3.0, -4.0, -5.0]
            )
        )
        return _ok("负相关数据不崩溃", result.success, _err_detail(result))
    except Exception as e:
        return _ok("负相关不崩溃", False, str(e))


async def stress_degenerate_inputs():
    """退化输入: 最小数据, 单点, 零方差."""
    print("\n=== 9. 退化输入 ===")
    checks = (_deg_single_point, _deg_zero_variance, _deg_negative_correlation)
    results = [await check() for check in checks]
    return all(results)


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
