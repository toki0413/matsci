"""DiscreteGroup — 有限群 + 有限域计算.

互补 SymmetryTool (晶体学空间群, 230 种, 连续表示) — 本工具处理抽象
有限群 (任意阶), 用 sympy.combinatorics 的置换群表示.

11 个 action: from_generators / permutation_group / cyclic / dihedral /
symmetric / alternating / analyze / verify_subgroup / verify_homomorphism /
group_action_orbits / finite_field.

稀疏结合: analyze 输出 Cayley 图稀疏性 (|V|=|G|, |E|=|G|*|S|),
group_action_orbits 用 scipy.sparse.csgraph.connected_components.

ponytail: sympy.combinatorics 优先, 不引 GAP/SageMath.
大群 (|G| > 10^4) 走 LLM 跨域类比. 升级路径: Schreier-Sims + 表示论 (sage).
"""
from __future__ import annotations

import logging
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool, ResearchPhase, ToolProfile
from huginn.types import ToolContext, ToolResult, ValidationResult

logger = logging.getLogger(__name__)

# 进程级群对象缓存
_GROUP_CACHE: dict[str, Any] = {}


def _cache_group(group: Any) -> str:
    handle = f"g_{uuid.uuid4().hex[:8]}"
    _GROUP_CACHE[handle] = group
    return handle


def _get_group(handle: str) -> Any | None:
    return _GROUP_CACHE.get(handle)


def _parse_perm(s: str) -> Any:
    """(1 2 3) 或 (1 2)(3 4) → sympy Permutation."""
    from sympy.combinatorics import Permutation
    s = s.strip()
    if not s.startswith("("):
        return Permutation([])
    cycles = []
    i = 0
    while i < len(s):
        if s[i] == "(":
            j = s.index(")", i)
            cyc = [int(x) - 1 for x in s[i + 1:j].split()]
            if cyc:
                cycles.append(cyc)
            i = j + 1
        else:
            i += 1
    return Permutation(cycles) if cycles else Permutation([])


def _parse_perm_list(specs: list[str]) -> list[Any]:
    return [_parse_perm(s) for s in specs]


def _from_generators(gen_strs: list[str]) -> Any:
    from sympy.combinatorics import PermutationGroup
    return PermutationGroup(_parse_perm_list(gen_strs))


def _cyclic_group(n: int) -> Any:
    from sympy.combinatorics import Permutation, PermutationGroup
    # 单 n-cycle: (0 1 2 ... n-1), 不是 n 个 transposition 的乘积
    return PermutationGroup([Permutation(list(range(1, n)) + [0])])


def _dihedral_group(n: int) -> Any:
    from sympy.combinatorics import Permutation, PermutationGroup
    # r = 单 n-cycle (0 1 ... n-1)
    r = Permutation(list(range(1, n)) + [0])
    # s: i ↔ n-1-i (反射), 单点 + transposition
    s_pairs = [(i, n - 1 - i) for i in range(n // 2)]
    s = Permutation(s_pairs) if s_pairs else Permutation([])
    return PermutationGroup([r, s])


def _symmetric_group(n: int) -> Any:
    from sympy.combinatorics import Permutation, PermutationGroup
    if n <= 1:
        return PermutationGroup([Permutation([])])
    # S_n = <(0 1), (0 1 2 ... n-1)>
    gen1 = Permutation([1, 0] + list(range(2, n)))  # (0 1)
    gen2 = Permutation(list(range(1, n)) + [0])  # n-cycle
    return PermutationGroup([gen1, gen2])


def _alternating_group(n: int) -> Any:
    from sympy.combinatorics import Permutation, PermutationGroup
    if n <= 2:
        return PermutationGroup([Permutation([])])
    # A_n = <(0 1 2), (0 1 2 ... n) if n odd, (1 2 ... n) if n even>
    gen1 = Permutation([(0, 1, 2)])
    if n % 2 == 1:
        gen2 = Permutation(list(range(1, n)) + [0])  # n-cycle
    else:
        # (1 2 ... n-1), 单 (n-1)-cycle 作用于 1..n-1
        perm_arr = list(range(n))
        for i in range(1, n - 1):
            perm_arr[i] = i + 1
        perm_arr[n - 1] = 1
        gen2 = Permutation(perm_arr)
    return PermutationGroup([gen1, gen2])


def _analyze_group(group: Any) -> dict[str, Any]:
    """群结构分析 + Cayley 图稀疏性."""
    n = int(group.order())
    out: dict[str, Any] = {
        "order": n,
        "is_abelian": bool(group.is_abelian),
        "is_cyclic": bool(group.is_cyclic),
        "is_solvable": bool(group.is_solvable),
        "is_nilpotent": bool(group.is_nilpotent),
        "generators": [str(g) for g in group.generators],
        "identity": str(group.identity),
    }
    if n <= 1000:
        try:
            out["center_order"] = int(group.center().order())
        except Exception:
            out["center_order"] = None
    else:
        out["center_order"] = None
        out["center_skipped"] = "|G| > 1000"
    if n <= 200:
        try:
            out["n_conjugacy_classes"] = len(group.conjugacy_classes())
        except Exception:
            out["n_conjugacy_classes"] = None
    else:
        out["n_conjugacy_classes"] = None
        out["cc_skipped"] = "|G| > 200"
    if n <= 500:
        try:
            from sympy import factorint
            factors = factorint(n)
            sylow_info = {}
            for p in factors:
                try:
                    syl = group.sylow_subgroup(p)
                    sylow_info[p] = {"order": int(syl.order())}
                except Exception as e:
                    sylow_info[p] = {"error": str(e)}
            out["sylow"] = sylow_info
        except Exception:
            out["sylow"] = None
    else:
        out["sylow"] = None
        out["sylow_skipped"] = "|G| > 500"
    n_gens = len(group.generators)
    out["cayley_graph"] = {
        "n_nodes": n,
        "n_edges_directed": n * n_gens,
        "avg_degree": (2 * n * n_gens) / n if n > 0 else 0,
        "density": (n * n_gens) / (n * (n - 1)) if n > 1 else 0.0,
        "is_sparse": n * n_gens < n * (n - 1) / 2 if n > 1 else True,
    }
    return out


def _verify_subgroup(H: Any, G: Any) -> dict[str, Any]:
    """H ≤ G: H ⊆ G + 阶整除."""
    try:
        h_elems = set(H.elements)
        g_elems = set(G.elements)
    except Exception:
        return {"is_subgroup": None, "skipped": "群太大不能枚举"}
    if not h_elems.issubset(g_elems):
        return {"is_subgroup": False, "reason": "H 有元素不在 G 里"}
    if G.order() % H.order() != 0:
        return {"is_subgroup": False, "reason": f"|H|={H.order()} 不整除 |G|={G.order()}"}
    return {"is_subgroup": True, "index": int(G.order() // H.order())}


def _verify_homomorphism(
    G: Any, H: Any, gen_images: list[str]
) -> dict[str, Any]:
    """弱同态验证: 生成元阶保持 + 像在 H 里."""
    if len(gen_images) != len(G.generators):
        return {"is_homomorphism": False, "reason": "生成元像数量不对"}
    try:
        images = _parse_perm_list(gen_images)
    except Exception as e:
        return {"is_homomorphism": False, "reason": f"解析失败: {e}"}
    from sympy.combinatorics import Permutation
    h_degree = H.degree
    for g, im in zip(G.generators, images):
        # 把 im 升到 H 的 degree (恒等 () size 0 → size H.degree)
        if im.size < h_degree:
            im = Permutation(list(im.array_form) + list(range(im.size, h_degree)))
        ord_g = g.order()
        try:
            ord_im = im.order()
        except Exception:
            ord_im = 0
        if ord_im and ord_g % ord_im != 0:
            return {"is_homomorphism": False, "reason": f"阶不保持: |g|={ord_g}, |φ(g)|={ord_im}"}
        try:
            if not H.contains(im):
                return {"is_homomorphism": False, "reason": f"φ(g)={im} 不在 H 里"}
        except Exception:
            pass
    return {"is_homomorphism": True, "note": "弱验证: 阶保持 + 像在 H"}


def _group_action_orbits(group: Any, n_points: int) -> dict[str, Any]:
    """group 作用于 {0,...,n_points-1} 的轨道, 用 scipy.sparse.csgraph."""
    try:
        import numpy as np
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components
    except ImportError:
        return {"error": "scipy required"}
    gens = list(group.generators)
    rows, cols = [], []
    for i in range(n_points):
        rows.append(i); cols.append(i)
        for g in gens:
            j = g(i) if g(i) < n_points else i
            rows.append(i); cols.append(j)
    data = [1] * len(rows)
    A = csr_matrix((data, (rows, cols)), shape=(n_points, n_points))
    n_orbits, labels = connected_components(A, directed=False)
    return {
        "n_orbits": int(n_orbits),
        "orbit_sizes": sorted(np.bincount(labels).tolist(), reverse=True),
        "is_transitive": n_orbits == 1,
    }


def _int_to_poly(x: int, p: int, k: int) -> list[int]:
    coeffs = []
    for _ in range(k):
        coeffs.append(x % p)
        x //= p
    return coeffs


def _poly_to_int(coeffs: list[int], p: int) -> int:
    return sum(c * (p ** i) for i, c in enumerate(coeffs))


def _finite_field_op(
    p: int, k: int, operation: str, elements: list[int]
) -> dict[str, Any]:
    """GF(p^k) 运算."""
    if k == 1:
        if operation == "add":
            return {"result": (elements[0] + elements[1]) % p}
        if operation == "mul":
            return {"result": (elements[0] * elements[1]) % p}
        if operation == "invert":
            return {"result": pow(elements[0], p - 2, p)}
        return {"error": f"unknown op: {operation}"}
    from sympy.polys.galoistools import (
        gf_mul, gf_add, gf_rem, gf_irreducible_p,
    )
    from sympy.polys.domains import ZZ

    def find_irreducible(p: int, k: int) -> list[int]:
        # 从 c=1 开始, 跳过 c=0 (会生成 x^k 退化多项式)
        for c in range(1, p ** k):
            coeffs = []
            tmp = c
            for _ in range(k):
                coeffs.append(tmp % p)
                tmp //= p
            poly = coeffs + [1]
            try:
                if gf_irreducible_p(poly, p, ZZ):
                    return poly
            except Exception:
                continue
        return None

    mod_poly = find_irreducible(p, k)
    if mod_poly is None:
        return {"error": f"找不到 GF({p}^{k}) 的不可约多项式"}
    if operation == "add":
        a = _int_to_poly(elements[0], p, k)
        b = _int_to_poly(elements[1], p, k)
        res = gf_add(a, b, p, ZZ)
        return {"result": _poly_to_int(res, p), "mod_poly": mod_poly}
    if operation == "mul":
        a = _int_to_poly(elements[0], p, k)
        b = _int_to_poly(elements[1], p, k)
        prod = gf_mul(a, b, p, ZZ)
        res = gf_rem(prod, mod_poly, p, ZZ)
        return {"result": _poly_to_int(res, p), "mod_poly": mod_poly}
    if operation == "invert":
        # ponytail: GF(p^k) 求逆用暴力枚举 (元素数少), 不依赖 gf_extgcd (sympy 没导出)
        a = _int_to_poly(elements[0], p, k)
        # 找 b 使 a*b ≡ 1 mod mod_poly
        for b_int in range(1, p ** k):
            b = _int_to_poly(b_int, p, k)
            prod = gf_mul(a, b, p, ZZ)
            res = gf_rem(prod, mod_poly, p, ZZ)
            if _poly_to_int(res, p) == 1:
                return {"result": b_int, "mod_poly": mod_poly}
        return {"error": "不可逆 (元素 0)"}
    return {"error": f"unknown op: {operation}"}


class DiscreteGroupInput(BaseModel):
    action: Literal[
        "from_generators", "permutation_group", "cyclic", "dihedral",
        "symmetric", "alternating", "analyze", "verify_subgroup",
        "verify_homomorphism", "group_action_orbits", "finite_field",
    ] = Field(...)
    generators: list[str] | None = Field(default=None)
    permutations: list[str] | None = Field(default=None)
    n: int | None = Field(default=None)
    group_handle: str | None = Field(default=None)
    subgroup_handle: str | None = Field(default=None)
    gen_images: list[str] | None = Field(default=None)
    n_points: int | None = Field(default=None)
    p: int | None = Field(default=None)
    k: int = Field(default=1)
    operation: str | None = Field(default=None)
    elements: list[int] | None = Field(default=None)


class DiscreteGroupTool(HuginnTool):
    """有限群 + 有限域计算 (互补 SymmetryTool 的晶体学连续群)."""

    name = "discrete_group"
    category = "sci"
    profile = ToolProfile(
        phases=frozenset({ResearchPhase.HYPOTHESIS, ResearchPhase.VALIDATION}),
    )
    description = (
        "Finite group and finite field computation. Construct cyclic, "
        "dihedral, symmetric, alternating groups or groups from generators. "
        "Analyze order, center, conjugacy classes, Sylow subgroups, Cayley "
        "graph sparsity. Verify subgroup/homomorphism. Compute group action "
        "orbits via scipy sparse csgraph. GF(p^k) arithmetic."
    )
    input_schema = DiscreteGroupInput
    read_only = True

    def is_read_only(self, args: DiscreteGroupInput) -> bool:
        return True

    async def validate_input(
        self, args: dict[str, Any], context: ToolContext | None = None
    ) -> ValidationResult:
        args_obj = args if isinstance(args, DiscreteGroupInput) else DiscreteGroupInput(**args)
        if args_obj.action in ("cyclic", "dihedral", "symmetric", "alternating"):
            if not args_obj.n or args_obj.n < 1:
                return ValidationResult(result=False, message=f"{args_obj.action} 需要 n >= 1")
        if args_obj.action == "finite_field":
            if not args_obj.p or args_obj.p < 2:
                return ValidationResult(result=False, message="finite_field 需要 p >= 2")
        return ValidationResult(result=True)

    async def call(
        self, args: dict[str, Any], context: ToolContext
    ) -> ToolResult:
        args_obj = args if isinstance(args, DiscreteGroupInput) else DiscreteGroupInput(**args)
        try:
            a = args_obj.action
            if a == "from_generators":
                if not args_obj.generators:
                    return ToolResult(data=None, success=False, error="需要 generators")
                g = _from_generators(args_obj.generators)
                return ToolResult(data={"handle": _cache_group(g), "order": int(g.order())})
            if a == "permutation_group":
                if not args_obj.permutations:
                    return ToolResult(data=None, success=False, error="需要 permutations")
                perms = _parse_perm_list(args_obj.permutations)
                g = _from_generators([str(p) for p in perms])
                return ToolResult(data={"handle": _cache_group(g), "order": int(g.order())})
            if a == "cyclic":
                g = _cyclic_group(args_obj.n or 1)
                return ToolResult(data={"handle": _cache_group(g), "order": int(g.order())})
            if a == "dihedral":
                g = _dihedral_group(args_obj.n or 1)
                return ToolResult(data={"handle": _cache_group(g), "order": int(g.order())})
            if a == "symmetric":
                g = _symmetric_group(args_obj.n or 1)
                return ToolResult(data={"handle": _cache_group(g), "order": int(g.order())})
            if a == "alternating":
                g = _alternating_group(args_obj.n or 1)
                return ToolResult(data={"handle": _cache_group(g), "order": int(g.order())})
            if a == "analyze":
                if not args_obj.group_handle:
                    return ToolResult(data=None, success=False, error="需要 group_handle")
                g = _get_group(args_obj.group_handle)
                if g is None:
                    return ToolResult(data=None, success=False, error="handle 不存在")
                return ToolResult(data=_analyze_group(g))
            if a == "verify_subgroup":
                if not (args_obj.group_handle and args_obj.subgroup_handle):
                    return ToolResult(data=None, success=False, error="需要 group_handle + subgroup_handle")
                G = _get_group(args_obj.group_handle)
                H = _get_group(args_obj.subgroup_handle)
                if G is None or H is None:
                    return ToolResult(data=None, success=False, error="handle 不存在")
                return ToolResult(data=_verify_subgroup(H, G))
            if a == "verify_homomorphism":
                if not (args_obj.group_handle and args_obj.subgroup_handle and args_obj.gen_images):
                    return ToolResult(data=None, success=False, error="需要 group_handle + subgroup_handle + gen_images")
                G = _get_group(args_obj.group_handle)
                H = _get_group(args_obj.subgroup_handle)
                if G is None or H is None:
                    return ToolResult(data=None, success=False, error="handle 不存在")
                return ToolResult(data=_verify_homomorphism(G, H, args_obj.gen_images))
            if a == "group_action_orbits":
                if not (args_obj.group_handle and args_obj.n_points):
                    return ToolResult(data=None, success=False, error="需要 group_handle + n_points")
                g = _get_group(args_obj.group_handle)
                if g is None:
                    return ToolResult(data=None, success=False, error="handle 不存在")
                return ToolResult(data=_group_action_orbits(g, args_obj.n_points))
            if a == "finite_field":
                if not (args_obj.p and args_obj.operation and args_obj.elements):
                    return ToolResult(data=None, success=False, error="需要 p + operation + elements")
                return ToolResult(data=_finite_field_op(args_obj.p, args_obj.k, args_obj.operation, args_obj.elements))
            return ToolResult(data=None, success=False, error=f"unknown action: {a}")
        except Exception as exc:
            logger.warning("discrete_group failed: %s", exc, exc_info=True)
            return ToolResult(data=None, success=False, error=str(exc))


def _selfcheck() -> None:
    """12 项 assert 验证有限群 + 有限域核心行为."""
    print("[discrete_group] running self-check...")

    # 1. C_6 阶 6, Abel
    g = _cyclic_group(6)
    a = _analyze_group(g)
    assert a["order"] == 6, f"1. C_6 order 6, got {a['order']}"
    assert a["is_abelian"] is True, f"1. C_6 abelian"

    # 2. S_4 阶 24, 非 Abel
    g = _symmetric_group(4)
    a = _analyze_group(g)
    assert a["order"] == 24, f"2. S_4 order 24, got {a['order']}"
    assert a["is_abelian"] is False, f"2. S_4 not abelian"

    # 3. D_4 阶 8, 中心阶 2
    g = _dihedral_group(4)
    a = _analyze_group(g)
    assert a["order"] == 8, f"3. D_4 order 8, got {a['order']}"
    assert a.get("center_order") == 2, f"3. D_4 center 2, got {a.get('center_order')}"

    # 4. A_4 阶 12, 4 共轭类
    g = _alternating_group(4)
    a = _analyze_group(g)
    assert a["order"] == 12, f"4. A_4 order 12, got {a['order']}"
    assert a.get("n_conjugacy_classes") == 4, f"4. A_4 4 cc, got {a.get('n_conjugacy_classes')}"

    # 5. from_generators <(1 2 3), (1 2)> = S_3
    g = _from_generators(["(1 2 3)", "(1 2)"])
    a = _analyze_group(g)
    assert a["order"] == 6, f"5. <(1 2 3), (1 2)> order 6, got {a['order']}"

    # 6. sylow field 存在
    g = _symmetric_group(3)
    a = _analyze_group(g)
    assert "sylow" in a, "6. sylow field missing"

    # 7. C_3 ≤ S_3
    from sympy.combinatorics import Permutation, PermutationGroup
    H3 = PermutationGroup([Permutation([(0, 1, 2)])])
    G3 = PermutationGroup([Permutation([1, 0, 2]), Permutation([(0, 1, 2)])])
    r = _verify_subgroup(H3, G3)
    assert r["is_subgroup"] is True, f"7. C_3 ≤ S_3, got {r}"

    # 8. D_4 不是 S_3 子群 (阶不整除)
    G = _symmetric_group(3)
    H = _dihedral_group(4)
    r = _verify_subgroup(H, G)
    assert r["is_subgroup"] is False, f"8. D_4 not ≤ S_3, got {r}"

    # 9. sign: S_3 → C_2 是同态
    G = _symmetric_group(3)
    H = _symmetric_group(2)
    r = _verify_homomorphism(G, H, ["(1 2)", "()"])
    assert r["is_homomorphism"] is True, f"9. sign homomorphism, got {r}"

    # 10. S_3 作用于 {0,1,2} 是传递的
    g = _symmetric_group(3)
    r = _group_action_orbits(g, 3)
    assert r["n_orbits"] == 1, f"10. S_3 transitive, got {r}"
    assert r["is_transitive"] is True

    # 11. GF(5): 2+3=0, 2*3=1
    r = _finite_field_op(5, 1, "add", [2, 3])
    assert r["result"] == 0, f"11. 2+3 in GF(5) = 0, got {r}"
    r = _finite_field_op(5, 1, "mul", [2, 3])
    assert r["result"] == 1, f"11. 2*3 in GF(5) = 1, got {r}"

    # 12. GF(2^2) 找得到不可约多项式
    r = _finite_field_op(2, 2, "mul", [3, 3])
    assert r.get("mod_poly") is not None, f"12. GF(2^2) mod_poly, got {r}"
    assert "result" in r, f"12. GF(2^2) mul result, got {r}"

    print("[discrete_group] self-check OK (12/12)")


if __name__ == "__main__":
    _selfcheck()
