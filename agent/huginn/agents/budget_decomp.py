"""素数预算分解 — 算术基本定理给 subagent 三参数选可行配置.

治 spec 天花板 "max_depth / parallel / per_subagent 三参数无原则".
N = depth × parallel × per_subagent, 算术基本定理 N = Π p_i^a_i 唯一分解,
所有合法 (d, p, n) 配置 = 把素因子分成 3 组的有序分拆.

530 = 2 × 5 × 53. 素数选择原则:
  depth 用小素数 (2, 3)      — 失控风险随深度指数增长
  parallel 用中素数 (5, 7)   — LLM API 限速约束
  per_subagent 用大素数 (53+) — 单 agent 任务复杂度

不做 (YAGNI):
  - 整数规划求最优配置 — 素数分解 + 启发式足够
  - 动态预算 (跑一步重新分解) — spec 升级路径, 不在本次范围

天花板: 假设 N 需全用. 实际 agent 可能 200 步就完成, 升级: 剩余预算重新分解.
"""
from __future__ import annotations

from itertools import product
from typing import Any

from sympy import factorint, isprime


def budget_configurations(
    n: int,
    *,
    max_depth: int = 3,
    max_parallel: int = 10,
    min_per: int = 20,
) -> list[dict[str, int]]:
    """枚举 n 的所有合法 (depth, parallel, per_subagent) 素数配置.

    合法 = depth × parallel × per_subagent == n, 且三个数都由 n 的素因子乘出.
    max_depth / max_parallel / min_per 过滤失控配置.

    ponytail: sympy.factorint 一次分解, 后续纯组合枚举, 无重复因子.
    """
    if n < 1:
        return []
    factors = factorint(n)  # {p: exp}
    primes = []
    for p, e in factors.items():
        primes.extend([p] * e)
    # primes 是 n 的所有素因子 (带重复), 长度 = Ω(n) (素因子总数, 计重数)
    # 把 primes 分成 3 组 (有序: depth, parallel, per), 每组乘积 = 该参数
    seen: set[tuple[int, int, int]] = set()
    out: list[dict[str, int]] = []
    # 三组分配 = 两个切分点, 用笛卡尔积枚举每个素数去哪组
    for assign in product((0, 1, 2), repeat=len(primes)):
        d_parts, p_parts, n_parts = [], [], []
        for prime, group in zip(primes, assign):
            (d_parts if group == 0 else p_parts if group == 1 else n_parts).append(prime)
        d = 1
        for x in d_parts:
            d *= x
        p = 1
        for x in p_parts:
            p *= x
        per = 1
        for x in n_parts:
            per *= x
        key = (d, p, per)
        if key in seen:
            continue
        seen.add(key)
        if d > max_depth:
            continue
        if p > max_parallel:
            continue
        if per < min_per:
            continue
        out.append({"depth": d, "parallel": p, "per_subagent": per})
    # 排序: 优先每个参数都是单个素因子 (不合并), 然后 parallel 大, per 大.
    # 530 = 2×5×53 唯一分解 → (2,5,53) 三个素数各给一个参数, 合并=0 排第一.
    # spec 原则: depth 用小素数 / parallel 中素数 / per 大素数.
    out.sort(key=lambda c: (
        sum(1 for v in c.values() if not isprime(v) and v != 1),  # 合并次数
        -c["parallel"],
        -c["per_subagent"],
    ))
    return out


def recommend(n: int) -> dict[str, int]:
    """给 n 选推荐配置: depth≤2 优先, parallel 尽量大, per≥20.

    ponytail: 530 → {depth:2, parallel:5, per:53}. 没有合法配置时返回
    退化配置 {depth:1, parallel:1, per:n}.
    """
    configs = budget_configurations(n)
    if not configs:
        return {"depth": 1, "parallel": 1, "per_subagent": n}
    return configs[0]


def config_cost(config: dict[str, int], n_total: int) -> dict[str, Any]:
    """算单配置的指标: 上限 tool_calls, 是否推荐区.

    上限 = n_total × parallel × per_subagent × (depth 层展开系数).
    depth=1: 上限 = n_total × parallel × per
    depth=2: 上限 = n_total × parallel × per × per (每 sub 再派 per 个 sub-sub)
    ponytail: depth≥3 不算 (理论上限巨大但调试不可行, spec 已 cap max_depth=3).
    """
    d = config["depth"]
    p = config["parallel"]
    per = config["per_subagent"]
    if d == 1:
        ceiling = n_total * p * per
    elif d == 2:
        ceiling = n_total * p * per * per
    else:
        ceiling = float("inf")
    return {
        "ceiling": ceiling,
        "depth_risk": "low" if d <= 1 else "medium" if d == 2 else "high",
        "wallclock_factor": 1.0 / p,  # parallel 个并行 → wall-clock 1/p (理想)
    }


# ── selfcheck ──────────────────────────────────────────────

if __name__ == "__main__":
    # 1. 530 = 2 × 5 × 53, 含 {depth:2, parallel:5, per:53}
    cfgs = budget_configurations(530)
    keys = {(c["depth"], c["parallel"], c["per_subagent"]) for c in cfgs}
    assert (2, 5, 53) in keys, f"530 应含 (2,5,53), got {keys}"
    assert (1, 10, 53) in keys, f"530 应含 (1,10,53), got {keys}"
    assert (1, 2, 265) in keys, f"530 应含 (1,2,265), got {keys}"
    print(f"[ok] 530 的合法配置 ({len(cfgs)} 个): {cfgs[:3]}...")

    # 2. 不含非法: 4 不是 530 因子
    assert all(c["depth"] != 4 for c in cfgs), "4 不应是 530 的 depth 因子"
    assert all(c["per_subagent"] != 7 for c in cfgs), "7 不应是 530 的 per 因子"
    print("[ok] 不含非法因子")

    # 3. recommend(530) = {depth:2, parallel:5, per:53}
    r = recommend(530)
    assert r == {"depth": 2, "parallel": 5, "per_subagent": 53}, r
    print(f"[ok] recommend(530) = {r}")

    # 4. config_cost 上限测算
    cost = config_cost(r, 530)
    assert cost["ceiling"] == 530 * 5 * 53 * 53, cost
    assert cost["depth_risk"] == "medium"
    assert abs(cost["wallclock_factor"] - 0.2) < 1e-9
    print(f"[ok] config_cost(530 recommend) ceiling={cost['ceiling']:,}")

    # 5. 素数 n (e.g. 53) 只能 {depth:1, parallel:1, per:53}
    cfgs_prime = budget_configurations(53)
    assert len(cfgs_prime) == 1, cfgs_prime
    assert cfgs_prime[0] == {"depth": 1, "parallel": 1, "per_subagent": 53}
    print(f"[ok] 素数 53 退化为单配置: {cfgs_prime[0]}")

    # 6. min_per 过滤: 530 with min_per=200 只留 per>=200 的
    cfgs_high = budget_configurations(530, min_per=200)
    assert all(c["per_subagent"] >= 200 for c in cfgs_high), cfgs_high
    print(f"[ok] min_per=200 过滤后剩 {len(cfgs_high)} 个配置")

    print("[budget_decomp] self-check OK (6/6)")
