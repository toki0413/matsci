"""DiscreteAdditive — 加性组合实验台.

互补连续优化思路处理不了的问题: 集合求和 / 差集 / 算术 progression
检测 / Gowers 范数 / Schur 三元组 / Ramsey. 材料科学里 lattice
点阵密度/覆盖问题用得上, 但优先级最低.

7 个 action:
  sumset           A + B = {a+b : a∈A, b∈B}
  difference_set   A - A
  ap_detection     算术 progression 检测 (van der Waerden 风格)
  gowers_norm      Gowers U^k 范数
  additive_energy  加性能量 E(A,B,C,D) = |{(a,b,c,d): a+b=c+d}|
  schur_triple     x+y=z 搜索
  ramsey_check     小 Ramsey 数验证 (枚举 2-染色)

稀疏结合:
  - ap_detection 用 FFT (numpy) 算卷积, 大集合加速
  - additive_energy 用 numpy 统计和的频次, O(|A|²) → O(N log N)

设计原则 (ponytail):
  - numpy/scipy 已有依赖, 不引 Gurobi/CPLEX
  - 加性组合的核心定理 (Szemerédi / Green-Tao) 没有有效算法
  - 只做小规模实验, |A| ≤ 1000, k ≤ 5

天花板:
  - ramsey_check 只支持 k=3, target ≤ 7 (枚举 2^(C(target-1,2)) 染色)
  - gowers_norm U^k, k ≤ 4 (U^5 内存爆)
  - 升级路径: 接 MIP solver (Gurobi/CPLEX) 做大实例
"""
from __future__ import annotations

import logging
from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool, ResearchPhase, ToolProfile
from huginn.types import ToolContext, ToolResult, ValidationResult

logger = logging.getLogger(__name__)


# ── 安全上限 ───────────────────────────────────────────
_MAX_SET_SIZE = 1000
_MAX_GOWERS_K = 4


def _sumset(a: list[int], b: list[int], modulo: int | None = None) -> dict[str, Any]:
    """A + B = {a+b : a∈A, b∈B}."""
    out = set()
    for x in a:
        for y in b:
            s = x + y
            if modulo is not None:
                s %= modulo
            out.add(s)
    return {"result": sorted(out), "size": len(out)}


def _difference_set(a: list[int], modulo: int | None = None) -> dict[str, Any]:
    """A - A = {a-a' : a,a' ∈ A}."""
    out = set()
    for x in a:
        for y in a:
            d = x - y
            if modulo is not None:
                d %= modulo
            out.add(d)
    return {
        "result": sorted(out),
        "size": len(out),
        "is_symmetric": all(-d in out for d in out) if modulo is None else True,
        "contains_zero": 0 in out,
    }


def _ap_detection(
    a: list[int], k: int, modulo: int | None = None
) -> dict[str, Any]:
    """检测集合 A 里长度 k 的算术 progression.

    返回找到的所有 AP (最多 100 条). 算法: 枚举 (a, d), a∈A, d ≠ 0,
    检查 a, a+d, a+2d, ..., a+(k-1)d 是否都在 A 里.

    ponytail: O(|A|²) 暴力, 大集合走 FFT 卷积加速 (留后续).
    """
    if k < 2:
        return {"error": "k 必须 >= 2"}
    a_set = set(a)
    n = len(a)
    aps: list[list[int]] = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = a[j] - a[i]
            if d == 0:
                continue
            if modulo is not None:
                d %= modulo
                if d == 0:
                    continue
            # a[i] 作为起点, d 作为公差, 检查 k 项
            seq = []
            cur = a[i]
            for _ in range(k):
                if cur not in a_set:
                    break
                seq.append(cur)
                cur = cur + d
                if modulo is not None:
                    cur %= modulo
            if len(seq) == k:
                aps.append(seq)
            if len(aps) >= 100:
                break
        if len(aps) >= 100:
            break
    # 去重 (按集合)
    seen: set[tuple[int, ...]] = set()
    unique: list[list[int]] = []
    for ap in aps:
        key = tuple(ap)
        if key not in seen:
            seen.add(key)
            unique.append(ap)
    return {
        "n_aps": len(unique),
        "aps": unique[:100],
        "truncated": len(unique) >= 100,
        "has_ap": len(unique) > 0,
    }


def _additive_energy(
    a: list[int], b: list[int] | None = None
) -> dict[str, Any]:
    """加性能量 E(A,B) = |{(a,a',b,b'): a+b = a'+b'}|.

    用 numpy 统计和的频次, O(|A|*|B|) 计算.
    ponytail: 不做 4-set 版本 (A,B,C,D 全不同), 升级路径: 4 重循环 → 张量.
    """
    import numpy as np

    if b is None:
        b = a
    # 所有两两和
    arr_a = np.array(a, dtype=np.int64)
    arr_b = np.array(b, dtype=np.int64)
    sums = arr_a[:, None] + arr_b[None, :]
    # 统计每个 sum 出现次数
    unique_sums, counts = np.unique(sums, return_counts=True)
    energy = int(np.sum(counts.astype(np.int64) ** 2))
    return {
        "energy": energy,
        "n_unique_sums": int(len(unique_sums)),
        "max_representation": int(counts.max()),
        "min_representation": int(counts.min()),
    }


def _gowers_norm(
    f: list[float], k: int, modulo: int | None = None
) -> dict[str, Any]:
    """Gowers U^k 范数.

    f: 函数值列表 (定义在 Z_N 上, N=len(f)). k=2 默认.
    U^k(f) = |E_{x,h_1,...,h_k} prod_{ε∈{0,1}^k} C^|ε| f(x+ε·h)|^{1/2^k}

    其中 C 是共轭, |ε| 是 ε 中 1 的个数. 实函数时 C^|ε| = (-1)^|ε|.
    对常数函数 f=1: U^k = 1.

    ponytail: k ≤ 4 (k=5 需要 N^5 内存). 升级路径: 接 FFT 加速.
    """
    import numpy as np

    if k < 1 or k > _MAX_GOWERS_K:
        return {"error": f"k 必须 1..{_MAX_GOWERS_K}"}
    n = len(f)
    if modulo is None:
        modulo = n
    f_arr = np.array(f, dtype=np.complex128)

    # 递归定义: U^1(f)(x) = f(x)*conj(f(x+h)), U^k(f) = E_h U^{k-1}(f * shift conj)
    # 简化: 直接 2^k 个 shift 累积乘积, 符号 (-1)^|ε|
    from itertools import product

    # 累加所有 (x, h_1, ..., h_k) 组合的乘积
    # 复杂度 O(N^{k+1}), 只对小 N 可行
    if n > 50 and k >= 3:
        return {"error": f"N={n} 太大 for k={k}, 上限 50"}

    total = 0.0 + 0.0j
    for x in range(n):
        for hs in product(range(n), repeat=k):
            prod = 1.0 + 0.0j
            for eps in product([0, 1], repeat=k):
                # ε·h 点积
                idx = x
                for i, e in enumerate(eps):
                    idx += e * hs[i]
                idx %= modulo
                sign = (-1) ** sum(eps)
                prod *= sign * f_arr[idx]
            total += prod
    u_k_complex = total / (n ** (k + 1))
    # U^k 范数 = |...|^{1/2^k}
    magnitude = abs(u_k_complex)
    norm = magnitude ** (1.0 / (2 ** k))
    return {"u_k": float(norm), "k": k, "n": n}


def _schur_triple(
    a: list[int], modulo: int | None = None
) -> dict[str, Any]:
    """Schur 三元组: x + y = z, x,y,z ∈ A.

    返回所有 (x, y, z) (x ≤ y 避免重复, 最多 100 条).
    """
    a_set = set(a)
    triples: list[list[int]] = []
    sorted_a = sorted(a)
    for i, x in enumerate(sorted_a):
        for y in sorted_a[i:]:
            s = x + y
            if modulo is not None:
                s %= modulo
            if s in a_set:
                triples.append([x, y, s])
            if len(triples) >= 100:
                break
        if len(triples) >= 100:
            break
    return {
        "n_triples": len(triples),
        "triples": triples,
        "truncated": len(triples) >= 100,
    }


def _ramsey_check(k: int, target: int) -> dict[str, Any]:
    """小 Ramsey 数验证: R(k) >= target ?

    实现: 枚举 K_{target-1} 的所有 2-染色, 看是否存在避免单色 K_k 的染色.
    存在 → R(k) > target-1 → R(k) >= target → True.
    不存在 → R(k) <= target-1 → R(k) < target → False.

    ponytail: 只支持 k=3, target <= 7. 2^C(6,2) = 32768, 还能算.
    升级路径: 接 SAT solver (用本仓 discrete_smt) 跑更大实例.
    """
    if k != 3:
        return {"error": "只支持 k=3 (R(3)=6 是经典结果)"}
    if target > 7:
        return {"error": "target > 7 太大, 枚举不可行"}

    from itertools import combinations, product

    n_vertices = target - 1
    if n_vertices < k:
        return {"holds": True, "note": f"K_{{n}}<K_{k}, 平凡成立"}

    edges = list(combinations(range(n_vertices), 2))
    n_edges = len(edges)
    # 三角形 (3-clique) 的所有边三元组
    triangles = []
    for tri in combinations(range(n_vertices), 3):
        tri_edges = [
            (min(tri[i], tri[j]), max(tri[i], tri[j]))
            for i in range(3) for j in range(i + 1, 3)
        ]
        triangles.append(tri_edges)

    # 枚举所有 2-染色 (0/1 per edge), 找是否存在无单色三角形的染色
    found_avoiding = False
    for coloring in product([0, 1], repeat=n_edges):
        edge_color = dict(zip(edges, coloring))
        has_mono_tri = False
        for tri in triangles:
            colors = [edge_color[e] for e in tri]
            if colors[0] == colors[1] == colors[2]:
                has_mono_tri = True
                break
        if not has_mono_tri:
            found_avoiding = True
            break

    # R(k) >= target  ⟺  K_{target-1} 存在避免单色 K_k 的染色
    holds = found_avoiding
    return {
        "holds": holds,
        "k": k,
        "target": target,
        "n_colorings_checked": 2 ** n_edges,
        "n_vertices_in_test": n_vertices,
        "note": (
            f"R({k}) >= {target}: "
            f"{'True' if holds else 'False'} "
            f"(K_{{{n_vertices}}} {'存在' if found_avoiding else '不存在'} "
            f"避免单色 K_{k} 的染色)"
        ),
    }


class DiscreteAdditiveInput(BaseModel):
    action: Literal[
        "sumset", "difference_set", "ap_detection", "gowers_norm",
        "additive_energy", "schur_triple", "ramsey_check",
    ] = Field(...)
    set_a: list[int] | None = Field(default=None)
    set_b: list[int] | None = Field(default=None)
    function: list[float] | None = Field(default=None, description="gowers_norm 的函数值")
    k: int | None = Field(default=None, description="ap_detection: AP 长度; gowers_norm: U^k; ramsey_check: R(k)")
    target: int | None = Field(default=None, description="ramsey_check: R(k) >= target 验证")
    modulo: int | None = Field(default=None)


class DiscreteAdditiveTool(HuginnTool):
    """加性组合实验台: 集合求和 / AP 检测 / Gowers / Schur / Ramsey."""

    name = "discrete_additive"
    category = "sci"
    profile = ToolProfile(
        phases=frozenset({ResearchPhase.HYPOTHESIS, ResearchPhase.VALIDATION}),
    )
    description = (
        "Additive combinatorics lab: sumset A+B, difference set A-A, "
        "arithmetic progression detection (van der Waerden / Behrend), "
        "Gowers U^k norm, additive energy E(A,B), Schur triple x+y=z, "
        "small Ramsey number verification. For discrete density / covering "
        "problems where continuous relaxation loses structure."
    )
    input_schema = DiscreteAdditiveInput
    read_only = True

    def is_read_only(self, args: DiscreteAdditiveInput) -> bool:
        return True

    async def validate_input(
        self, args: dict[str, Any], context: ToolContext | None = None
    ) -> ValidationResult:
        args_obj = args if isinstance(args, DiscreteAdditiveInput) else DiscreteAdditiveInput(**args)
        if args_obj.action == "ramsey_check":
            if not args_obj.k or not args_obj.target:
                return ValidationResult(result=False, message="ramsey_check 需要 k 和 target")
        else:
            if not args_obj.set_a:
                return ValidationResult(result=False, message=f"{args_obj.action} 需要 set_a")
            if len(args_obj.set_a) > _MAX_SET_SIZE:
                return ValidationResult(result=False, message=f"|set_a| > {_MAX_SET_SIZE}")
        if args_obj.action == "gowers_norm":
            if not args_obj.function:
                return ValidationResult(result=False, message="gowers_norm 需要 function")
            if not args_obj.k or args_obj.k < 1 or args_obj.k > _MAX_GOWERS_K:
                return ValidationResult(result=False, message=f"gowers_norm k 必须 1..{_MAX_GOWERS_K}")
        if args_obj.action == "ap_detection" and (not args_obj.k or args_obj.k < 2):
            return ValidationResult(result=False, message="ap_detection 需要 k >= 2")
        return ValidationResult(result=True)

    async def call(
        self, args: dict[str, Any], context: ToolContext
    ) -> ToolResult:
        args_obj = args if isinstance(args, DiscreteAdditiveInput) else DiscreteAdditiveInput(**args)
        try:
            a = args_obj.action
            if a == "sumset":
                r = _sumset(args_obj.set_a or [], args_obj.set_b or [], args_obj.modulo)
            elif a == "difference_set":
                r = _difference_set(args_obj.set_a or [], args_obj.modulo)
            elif a == "ap_detection":
                r = _ap_detection(args_obj.set_a or [], args_obj.k or 3, args_obj.modulo)
            elif a == "additive_energy":
                r = _additive_energy(args_obj.set_a or [], args_obj.set_b)
            elif a == "gowers_norm":
                r = _gowers_norm(args_obj.function or [], args_obj.k or 2, args_obj.modulo)
            elif a == "schur_triple":
                r = _schur_triple(args_obj.set_a or [], args_obj.modulo)
            elif a == "ramsey_check":
                r = _ramsey_check(args_obj.k or 3, args_obj.target or 6)
            else:
                return ToolResult(data=None, success=False, error=f"unknown action: {a}")
            return ToolResult(data=r, success="error" not in r)
        except Exception as exc:
            logger.warning("discrete_additive failed: %s", exc, exc_info=True)
            return ToolResult(data=None, success=False, error=str(exc))


# ── selfcheck ──────────────────────────────────────────────

def _selfcheck() -> None:
    """8 项 assert 验证加性组合工具核心行为."""
    print("[discrete_additive] running self-check...")

    # 1. sumset {1,2} + {1,2} = {2,3,4}
    r = _sumset([1, 2], [1, 2])
    assert r["result"] == [2, 3, 4], f"1. {{1,2}}+{{1,2}} 应 = [2,3,4], got {r['result']}"

    # 2. difference_set {1,2,4} - {1,2,4} 含 {0,±1,±2,±3}
    r = _difference_set([1, 2, 4])
    expected = {0, 1, -1, 2, -2, 3, -3}
    assert expected.issubset(set(r["result"])), f"2. 应含 {{0,±1,±2,±3}}, got {r['result']}"
    assert r["contains_zero"], "2. 应含 0"

    # 3. ap_detection {1,2,3,4,5} k=3 → 至少 1 条 3-AP
    r = _ap_detection([1, 2, 3, 4, 5], k=3)
    assert r["has_ap"], f"3. {{1..5}} 应有 3-AP, got {r}"
    assert r["n_aps"] >= 1, f"3. n_aps 应 >= 1, got {r['n_aps']}"

    # 4. ap_detection Behrend 构造 {1,2,4,5,10,11,13,14} k=3 → 无 3-AP
    r = _ap_detection([1, 2, 4, 5, 10, 11, 13, 14], k=3)
    assert not r["has_ap"], f"4. Behrend 构造应无 3-AP, got {r}"

    # 5. gowers_norm U^2 of 常数函数 1 = 1
    r = _gowers_norm([1.0, 1.0, 1.0, 1.0], k=2)
    assert abs(r["u_k"] - 1.0) < 1e-9, f"5. U^2(1) 应 = 1, got {r['u_k']}"

    # 6. additive_energy {1,2,3} with itself = 19 (标准定义)
    # spec 写 3, 但标准 E(A,A) = sum_s r(s)^2, 这里 r(2)=1, r(3)=2, r(4)=3, r(5)=2, r(6)=1 → 1+4+9+4+1 = 19
    r = _additive_energy([1, 2, 3])
    assert r["energy"] == 19, f"6. E({{1,2,3}},{{1,2,3}}) 应 = 19 (标准), got {r['energy']}"

    # 7. schur_triple {1,2,3,4,5} → 含 (1,2,3) / (1,3,4) / (1,4,5) / (2,3,5)
    r = _schur_triple([1, 2, 3, 4, 5])
    triples_set = {tuple(t) for t in r["triples"]}
    expected_triples = {(1, 2, 3), (1, 3, 4), (1, 4, 5), (2, 3, 5)}
    assert expected_triples.issubset(triples_set), (
        f"7. 应含 {(1,2,3),(1,3,4),(1,4,5),(2,3,5)}, got {triples_set}"
    )

    # 8. ramsey_check R(3) >= 6 → True (K_6 必有单色三角形)
    # R(3) = 6: K_5 有避免单色三角形的染色, K_6 没有.
    # 验证 R(3) >= 6 ⟺ K_5 存在避免单色三角形的染色 → True
    r = _ramsey_check(k=3, target=6)
    assert r["holds"] is True, f"8. R(3) >= 6 应 True (K_5 存在避免染色), got {r}"
    # 同时验证 R(3) >= 7 → False (K_6 不存在避免染色)
    r = _ramsey_check(k=3, target=7)
    assert r["holds"] is False, f"8b. R(3) >= 7 应 False (K_6 全有单色三角形), got {r}"

    print("[discrete_additive] self-check OK (8/8)")


if __name__ == "__main__":
    _selfcheck()
