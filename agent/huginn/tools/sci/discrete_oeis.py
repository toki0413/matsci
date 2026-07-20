"""DiscreteOEIS — OEIS 序列反查 + 公式匹配.

互补人类"先猜形式再验证"的偏置. 给定前几项, 反查是哪个 OEIS 序列;
给定公式, 反查是否有已知序列匹配.

4 个 action:
  lookup           前缀匹配 (给前几项, 找 OEIS)
  lookup_formula   公式反查 (给公式, 找匹配序列)
  describe         取元数据 (给 A 号, 取 name/refs)
  related          相关序列 (同 keywords)

稀疏结合:
  内置 50+ 序列, 内存索引, O(1) 查询. 大规模查询走 OEIS online
  (留后续, 当前纯本地).

设计原则 (ponytail):
  - 内置常用序列足够覆盖材料科学场景 (晶格计数 / 配位 / 斐波那契)
  - 不引 oeis 包, 不走网络 (本地字典)
  - 公式反查用 sympy 简化后字符串比较 (不解析 AST)
  - 升级路径: 接 OEIS REST API + 全量本地索引 (500MB)

天花板:
  - 内置序列数 < 100, 不可能覆盖所有 OEIS (40 万+)
  - 公式反查只做规范化字符串匹配, 不做语义等价
  - 升级路径: 接 sympy 序列生成器 + 在线 OEIS 搜索
"""
from __future__ import annotations

import logging
from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool, ResearchPhase, ToolProfile
from huginn.types import ToolContext, ToolResult, ValidationResult

logger = logging.getLogger(__name__)


# ── 内置 OEIS 序列 (前 16 项, 够前缀匹配) ──────────────────
# ponytail: 只内置材料科学/组合数学常用的 50+ 条, 不追求覆盖.
# 升级路径: 接 OEIS JSON API 拉全量.
_BUILTIN_SEQUENCES: list[dict[str, Any]] = [
    {
        "a_number": "A000045",
        "name": "Fibonacci numbers",
        "formula": "F(n) = F(n-1) + F(n-2), F(0)=0, F(1)=1",
        "keywords": ["nonn", "core", "easy", "nice"],
        "terms": [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610],
    },
    {
        "a_number": "A000040",
        "name": "Prime numbers",
        "formula": "p_n = n-th prime",
        "keywords": ["nonn", "core", "hard", "nice"],
        "terms": [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53],
    },
    {
        "a_number": "A000079",
        "name": "Powers of 2",
        "formula": "a(n) = 2^n",
        "keywords": ["nonn", "core", "easy"],
        "terms": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768],
    },
    {
        "a_number": "A000142",
        "name": "Factorial numbers",
        "formula": "a(n) = n!",
        "keywords": ["nonn", "core", "easy"],
        "terms": [1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800, 39916800, 479001600, 6227020800, 87178291200, 1307674368000],
    },
    {
        "a_number": "A000217",
        "name": "Triangular numbers",
        "formula": "a(n) = n*(n+1)/2",
        "keywords": ["nonn", "core", "easy", "nice"],
        "terms": [0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105, 120],
    },
    {
        "a_number": "A000290",
        "name": "Squares",
        "formula": "a(n) = n^2",
        "keywords": ["nonn", "core", "easy"],
        "terms": [0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225],
    },
    {
        "a_number": "A000244",
        "name": "Powers of 3",
        "formula": "a(n) = 3^n",
        "keywords": ["nonn", "core", "easy"],
        "terms": [1, 3, 9, 27, 81, 243, 729, 2187, 6561, 19683, 59049, 177147, 531441, 1594323, 4782969, 14348907],
    },
    {
        "a_number": "A001399",
        "name": "Number of partitions of n into at most 3 parts",
        "formula": "round((n+3)^2/12)",
        "keywords": ["nonn", "easy"],
        "terms": [1, 1, 2, 3, 4, 5, 7, 8, 10, 12, 14, 16, 19, 21, 24, 27],
    },
    {
        "a_number": "A000108",
        "name": "Catalan numbers",
        "formula": "C(n) = binomial(2n,n)/(n+1)",
        "keywords": ["nonn", "core", "easy", "nice"],
        "terms": [1, 1, 2, 5, 14, 42, 132, 429, 1430, 4862, 16796, 58786, 208012, 742900, 2674440, 9694845],
    },
    {
        "a_number": "A001006",
        "name": "Motzkin numbers",
        "formula": "M(n) = M(n-1) + sum_{k=0..n-2} M(k)*M(n-2-k)",
        "keywords": ["nonn", "core", "nice"],
        "terms": [1, 1, 2, 4, 9, 21, 51, 127, 323, 835, 2188, 5798, 15511, 41835, 113634, 310572],
    },
    {
        "a_number": "A001169",
        "name": "Number of near-perfect matchings in K_{2n}",
        "formula": "a(n) = (2n-1)!!",
        "keywords": ["nonn"],
        "terms": [1, 1, 3, 15, 105, 945, 10395, 135135, 2027025, 34459425, 654729075, 13749310575, 316234143225, 7905853580625, 213458046676875, 6190283353629375],
    },
    {
        "a_number": "A001220",
        "name": "Wieferich primes",
        "formula": "primes p such that 2^(p-1) ≡ 1 mod p^2",
        "keywords": ["nonn", "hard", "bref"],
        "terms": [1093, 3511],
    },
    {
        "a_number": "A001221",
        "name": "omega(n): number of distinct prime factors",
        "formula": "a(n) = number of distinct primes dividing n",
        "keywords": ["nonn", "core", "easy"],
        "terms": [0, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 2, 1, 2, 2, 1],
    },
    {
        "a_number": "A001222",
        "name": "bigomega(n): number of prime factors with multiplicity",
        "formula": "a(n) = Omega(n)",
        "keywords": ["nonn", "core", "easy"],
        "terms": [0, 1, 1, 2, 1, 2, 1, 3, 2, 2, 1, 3, 1, 2, 2, 4],
    },
    {
        "a_number": "A001359",
        "name": "Lesser of twin primes",
        "formula": "p such that p+2 is also prime",
        "keywords": ["nonn", "nice"],
        "terms": [3, 5, 11, 17, 29, 41, 59, 71, 101, 107, 137, 149, 179, 191, 197, 227],
    },
    {
        "a_number": "A002110",
        "name": "Primorial numbers (products of first n primes)",
        "formula": "a(n) = product_{k=1..n} prime(k)",
        "keywords": ["nonn", "core", "easy"],
        "terms": [1, 2, 6, 30, 210, 2310, 30030, 510510, 9699690, 223092870, 6469693230, 200560490130, 7420738134810, 304250263527210, 13082761331670030, 614889782588491410],
    },
    {
        "a_number": "A003415",
        "name": "n-th derivative of n",
        "formula": "a(n) = n * sum_{p|n} 1/p",
        "keywords": ["nonn"],
        "terms": [0, 1, 1, 4, 1, 5, 1, 12, 6, 7, 1, 16, 1, 9, 8, 32],
    },
    {
        "a_number": "A005408",
        "name": "Odd numbers",
        "formula": "a(n) = 2n+1",
        "keywords": ["nonn", "core", "easy"],
        "terms": [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31],
    },
    {
        "a_number": "A005843",
        "name": "Even numbers",
        "formula": "a(n) = 2n",
        "keywords": ["nonn", "core", "easy"],
        "terms": [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30],
    },
    {
        "a_number": "A006530",
        "name": "Largest prime factor of n",
        "formula": "a(n) = max prime dividing n",
        "keywords": ["nonn", "easy"],
        "terms": [1, 2, 3, 2, 5, 3, 7, 2, 3, 5, 11, 3, 13, 7, 5, 2],
    },
    {
        "a_number": "A010060",
        "name": "Thue-Morse sequence",
        "formula": "a(n) = parity of number of 1s in binary n",
        "keywords": ["nonn", "core", "easy", "nice"],
        "terms": [0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0],
    },
    {
        "a_number": "A001220",
        "name": "Wieferich primes",
        "formula": "2^(p-1) ≡ 1 mod p^2",
        "keywords": ["nonn", "hard"],
        "terms": [1093, 3511],
    },
    {
        "a_number": "A000005",
        "name": "d(n): number of divisors of n",
        "formula": "a(n) = tau(n) = number of divisors",
        "keywords": ["nonn", "core", "easy"],
        "terms": [1, 2, 2, 3, 2, 4, 2, 4, 3, 4, 2, 6, 2, 4, 4, 5],
    },
    {
        "a_number": "A000010",
        "name": "Euler totient function phi(n)",
        "formula": "a(n) = |{k: 1<=k<=n, gcd(k,n)=1}|",
        "keywords": ["nonn", "core", "easy", "nice"],
        "terms": [1, 1, 2, 2, 4, 2, 6, 4, 6, 4, 10, 4, 12, 6, 8, 8],
    },
    {
        "a_number": "A000203",
        "name": "sigma(n): sum of divisors of n",
        "formula": "a(n) = sum_{d|n} d",
        "keywords": ["nonn", "core", "easy"],
        "terms": [1, 3, 4, 7, 6, 12, 8, 15, 13, 18, 12, 28, 14, 24, 24, 31],
    },
    {
        "a_number": "A001008",
        "name": "Numerators of harmonic numbers",
        "formula": "a(n) = numerator of sum_{k=1..n} 1/k",
        "keywords": ["nonn", "frac", "nice"],
        "terms": [1, 3, 11, 25, 137, 49, 363, 761, 7129, 7381, 83711, 86021, 1145993, 1171733, 1195757, 2436559],
    },
    {
        "a_number": "A000129",
        "name": "Pell numbers",
        "formula": "a(n) = 2a(n-1) + a(n-2)",
        "keywords": ["nonn", "easy", "nice"],
        "terms": [0, 1, 2, 5, 12, 29, 70, 169, 408, 985, 2378, 5741, 13860, 33461, 80782, 195025],
    },
    {
        "a_number": "A001109",
        "name": "a(n)^2 is triangular",
        "formula": "a(n) = 6a(n-1) - a(n-2) + 2, a(0)=0, a(1)=1",
        "keywords": ["nonn", "easy"],
        "terms": [0, 1, 8, 49, 288, 1681, 9800, 57121, 332928, 1940449, 11309768, 65918161, 384199200, 2239277041, 13051463048, 76069501249],
    },
    {
        "a_number": "A001597",
        "name": "Perfect powers: m^k where m > 0 and k > 1",
        "formula": "a(n) = m^k for some m > 0, k > 1",
        "keywords": ["nonn", "easy"],
        "terms": [1, 4, 8, 9, 16, 25, 27, 32, 36, 49, 64, 81, 100, 121, 125, 144],
    },
    {
        "a_number": "A001620",
        "name": "Decimal expansion of Euler's constant gamma",
        "formula": "gamma = limit (H_n - log n)",
        "keywords": ["cons", "nonn"],
        "terms": [5, 7, 7, 2, 1, 5, 6, 6, 4, 9, 0, 1, 5, 3, 2, 8],
    },
    {
        "a_number": "A001622",
        "name": "Decimal expansion of golden ratio phi",
        "formula": "phi = (1+sqrt(5))/2",
        "keywords": ["cons", "nonn"],
        "terms": [1, 6, 1, 8, 0, 3, 3, 9, 8, 8, 7, 4, 9, 8, 9, 4],
    },
    {
        "a_number": "A001113",
        "name": "Decimal expansion of e",
        "formula": "e = sum_{k>=0} 1/k!",
        "keywords": ["cons", "nonn"],
        "terms": [2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4, 5, 9, 0, 4, 5],
    },
    {
        "a_number": "A000796",
        "name": "Decimal expansion of Pi",
        "formula": "pi = 4*atan(1)",
        "keywords": ["cons", "nonn", "core"],
        "terms": [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3],
    },
    {
        "a_number": "A002064",
        "name": "C_4n lattice colorings",
        "formula": "a(n) = (n^4 + 2n^3 + 11n^2 + 10n) / 4",
        "keywords": ["nonn", "easy"],
        "terms": [0, 1, 6, 18, 40, 75, 126, 196, 288, 405, 550, 726, 936, 1183, 1470, 1800],
    },
    {
        "a_number": "A001399",
        "name": "Number of ways of writing n as sum of 3 positive squares",
        "formula": "a(n) = |{(x,y,z): x^2+y^2+z^2=n, x,y,z>=0}|",
        "keywords": ["nonn"],
        "terms": [1, 1, 2, 3, 4, 5, 7, 8, 10, 12, 14, 16, 19, 21, 24, 27],
    },
    {
        "a_number": "A002144",
        "name": "Primes of form 4k+1",
        "formula": "p ≡ 1 (mod 4)",
        "keywords": ["nonn", "easy"],
        "terms": [5, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97, 101, 109, 113, 137, 149],
    },
    {
        "a_number": "A002145",
        "name": "Primes of form 4k+3",
        "formula": "p ≡ 3 (mod 4)",
        "keywords": ["nonn", "easy"],
        "terms": [3, 7, 11, 19, 23, 31, 43, 47, 59, 67, 71, 79, 83, 103, 107, 127],
    },
    {
        "a_number": "A002193",
        "name": "Decimal expansion of sqrt(2)",
        "formula": "sqrt(2)",
        "keywords": ["cons", "nonn"],
        "terms": [1, 4, 1, 4, 2, 1, 3, 5, 6, 2, 3, 7, 3, 0, 9, 5],
    },
    {
        "a_number": "A003136",
        "name": "Loeschian numbers: x^2 + xy + y^2",
        "formula": "n = x^2 + xy + y^2 for some x,y",
        "keywords": ["nonn", "nice"],
        "terms": [0, 1, 3, 4, 7, 9, 12, 13, 16, 19, 21, 25, 27, 28, 31, 36],
    },
    {
        "a_number": "A005117",
        "name": "Squarefree numbers",
        "formula": "n not divisible by p^2 for any prime p",
        "keywords": ["nonn", "easy"],
        "terms": [1, 2, 3, 5, 6, 7, 10, 11, 13, 14, 15, 17, 19, 21, 22, 23],
    },
    {
        "a_number": "A008683",
        "name": "Mertens function",
        "formula": "M(n) = sum_{k=1..n} mu(k)",
        "keywords": ["sign", "core", "nice"],
        "terms": [1, 0, -1, -1, -2, -1, -2, -2, -2, -1, -2, -2, -3, -2, -1, -2],
    },
    {
        "a_number": "A000004",
        "name": "Zero sequence",
        "formula": "a(n) = 0",
        "keywords": ["nonn", "core", "easy"],
        "terms": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    },
    {
        "a_number": "A000012",
        "name": "All 1 sequence",
        "formula": "a(n) = 1",
        "keywords": ["nonn", "core", "easy"],
        "terms": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    },
    {
        "a_number": "A000027",
        "name": "Natural numbers",
        "formula": "a(n) = n",
        "keywords": ["nonn", "core", "easy"],
        "terms": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    },
    {
        "a_number": "A001477",
        "name": "Nonnegative integers",
        "formula": "a(n) = n",
        "keywords": ["nonn", "core", "easy"],
        "terms": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    },
    {
        "a_number": "A001478",
        "name": "Negative integers",
        "formula": "a(n) = -n",
        "keywords": ["sign", "easy"],
        "terms": [0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15],
    },
    {
        "a_number": "A001489",
        "name": "Negative of n",
        "formula": "a(n) = -n",
        "keywords": ["sign", "easy"],
        "terms": [0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15],
    },
    {
        "a_number": "A008587",
        "name": "Multiples of 5",
        "formula": "a(n) = 5n",
        "keywords": ["nonn", "easy"],
        "terms": [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75],
    },
    {
        "a_number": "A008589",
        "name": "Multiples of 7",
        "formula": "a(n) = 7n",
        "keywords": ["nonn", "easy"],
        "terms": [0, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84, 91, 98, 105],
    },
    {
        "a_number": "A000930",
        "name": "Narayana's cows sequence",
        "formula": "a(n) = a(n-1) + a(n-3)",
        "keywords": ["nonn", "easy", "nice"],
        "terms": [1, 1, 1, 2, 3, 4, 6, 9, 13, 19, 28, 41, 60, 88, 129, 189],
    },
    {
        "a_number": "A000931",
        "name": "Padovan sequence",
        "formula": "a(n) = a(n-2) + a(n-3)",
        "keywords": ["nonn", "easy", "nice"],
        "terms": [1, 0, 0, 1, 0, 1, 1, 1, 2, 2, 3, 4, 5, 7, 9, 12],
    },
]


# 索引: A 号 → 元数据
_BY_A: dict[str, dict[str, Any]] = {s["a_number"]: s for s in _BUILTIN_SEQUENCES}


def _prefix_match(
    query: list[int], terms: list[int], min_match: int = 4
) -> bool:
    """query 是否为 terms 的前缀 (至少 min_match 项)."""
    if len(query) < min_match or len(terms) < len(query):
        return False
    return terms[: len(query)] == query


def _formula_to_sympy_str(formula: str) -> str:
    """用 sympy 简化公式后转字符串, 做规范化比较.

    ponytail: 简化后字符串比较, 不解析 AST.
    天花板: 不做语义等价 (e.g. n^2+2n vs (n+1)^2-1).
    升级路径: 用 sympy.expand/simplify + 结构 hash.
    """
    try:
        from sympy import sympify

        # 只取等号右边或整式
        s = formula.split("=", 1)[-1].strip()
        expr = sympify(s)
        return str(expr.expand())
    except Exception:
        # 不能 sympify 的 (如 "p_n = n-th prime") 直接返回原文
        return formula.strip()


def _lookup(sequence: list[int], max_results: int = 10) -> list[dict[str, Any]]:
    """前缀匹配查序列."""
    out = []
    for s in _BUILTIN_SEQUENCES:
        if _prefix_match(sequence, s["terms"]):
            out.append({
                "a_number": s["a_number"],
                "name": s["name"],
                "formula": s["formula"],
                "matched_terms": s["terms"][: len(sequence)],
            })
            if len(out) >= max_results:
                break
    return out


def _lookup_formula(
    formula: str, max_results: int = 10
) -> list[dict[str, Any]]:
    """公式反查: sympy 简化后字符串比较."""
    target = _formula_to_sympy_str(formula)
    out = []
    for s in _BUILTIN_SEQUENCES:
        if _formula_to_sympy_str(s["formula"]) == target:
            out.append({
                "a_number": s["a_number"],
                "name": s["name"],
                "formula": s["formula"],
            })
            if len(out) >= max_results:
                break
    return out


def _describe(a_number: str) -> dict[str, Any] | None:
    """取元数据."""
    s = _BY_A.get(a_number)
    if not s:
        return None
    return {
        "a_number": s["a_number"],
        "name": s["name"],
        "formula": s["formula"],
        "keywords": s["keywords"],
        "first_terms": s["terms"],
    }


def _related(a_number: str, max_results: int = 10) -> list[dict[str, Any]]:
    """同 keywords 的相关序列."""
    s = _BY_A.get(a_number)
    if not s:
        return []
    target_kw = set(s["keywords"])
    out = []
    for other in _BUILTIN_SEQUENCES:
        if other["a_number"] == a_number:
            continue
        common = target_kw & set(other["keywords"])
        if common:
            out.append({
                "a_number": other["a_number"],
                "name": other["name"],
                "common_keywords": sorted(common),
            })
    # 按 common keywords 数量降序
    out.sort(key=lambda x: -len(x["common_keywords"]))
    return out[:max_results]


class DiscreteOEISInput(BaseModel):
    action: Literal["lookup", "lookup_formula", "describe", "related"] = Field(...)
    sequence: list[int] | None = Field(default=None, description="前几项整数序列")
    formula: str | None = Field(default=None, description="公式字符串")
    a_number: str | None = Field(default=None, description="OEIS A 号, 如 A000045")
    max_results: int = Field(default=10, description="最多返回数")


class DiscreteOEISTool(HuginnTool):
    """OEIS 序列反查 + 公式匹配."""

    name = "discrete_oeis"
    category = "sci"
    profile = ToolProfile(
        phases=frozenset({ResearchPhase.HYPOTHESIS, ResearchPhase.VALIDATION}),
    )
    description = (
        "OEIS sequence reverse lookup. Given first few terms, find which "
        "OEIS sequence it is. Given a formula, find matching sequences. "
        "Built-in 50+ common sequences (Fibonacci, Catalan, primes, "
        "triangular, squares, etc). Complements human 'guess-then-verify' "
        "bias by automated enumeration."
    )
    input_schema = DiscreteOEISInput
    read_only = True

    def is_read_only(self, args: DiscreteOEISInput) -> bool:
        return True

    async def validate_input(
        self, args: dict[str, Any], context: ToolContext | None = None
    ) -> ValidationResult:
        args_obj = args if isinstance(args, DiscreteOEISInput) else DiscreteOEISInput(**args)
        if args_obj.action in ("lookup",) and not args_obj.sequence:
            return ValidationResult(result=False, message="lookup 需要 sequence")
        if args_obj.action == "lookup" and len(args_obj.sequence or []) < 4:
            return ValidationResult(result=False, message="lookup 至少需要 4 项")
        if args_obj.action == "lookup_formula" and not args_obj.formula:
            return ValidationResult(result=False, message="lookup_formula 需要 formula")
        if args_obj.action in ("describe", "related") and not args_obj.a_number:
            return ValidationResult(result=False, message=f"{args_obj.action} 需要 a_number")
        return ValidationResult(result=True)

    async def call(
        self, args: dict[str, Any], context: ToolContext
    ) -> ToolResult:
        args_obj = args if isinstance(args, DiscreteOEISInput) else DiscreteOEISInput(**args)
        try:
            a = args_obj.action
            if a == "lookup":
                r = {"matches": _lookup(args_obj.sequence or [], args_obj.max_results)}
            elif a == "lookup_formula":
                r = {"matches": _lookup_formula(args_obj.formula or "", args_obj.max_results)}
            elif a == "describe":
                d = _describe(args_obj.a_number or "")
                if d is None:
                    return ToolResult(
                        data=None, success=False,
                        error=f"未知 A 号: {args_obj.a_number}",
                    )
                r = d
            elif a == "related":
                r = {"related": _related(args_obj.a_number or "", args_obj.max_results)}
            else:
                return ToolResult(data=None, success=False, error=f"unknown action: {a}")
            return ToolResult(data=r, success=True)
        except Exception as exc:
            logger.warning("discrete_oeis failed: %s", exc, exc_info=True)
            return ToolResult(data=None, success=False, error=str(exc))


# ── selfcheck ──────────────────────────────────────────────

def _selfcheck() -> None:
    """8 项 assert 验证 OEIS 反查核心行为."""
    print("[discrete_oeis] running self-check...")

    # 1. Fibonacci 前缀匹配
    r = _lookup([0, 1, 1, 2, 3, 5, 8, 13])
    assert r, "1. Fibonacci 应该匹配到"
    assert any(m["a_number"] == "A000045" for m in r), f"1. 应含 A000045, got {r}"

    # 2. Catalan 前缀匹配
    r = _lookup([1, 1, 2, 5, 14, 42])
    assert any(m["a_number"] == "A000108" for m in r), f"2. 应含 A000108, got {r}"

    # 3. 太短不查 (min 4)
    r = _lookup([1, 1, 2])
    assert not r, f"3. 3 项应不匹配, got {r}"

    # 4. describe A000045
    r = _describe("A000045")
    assert r is not None, "4. A000045 应在库"
    assert r["name"] == "Fibonacci numbers", f"4. name 错: {r['name']}"

    # 5. describe 不存在的 A 号
    r = _describe("A999999")
    assert r is None, "5. A999999 应不存在"

    # 6. related A000045 应有同 keywords (core/easy/nice) 的序列
    r = _related("A000045")
    assert r, "6. A000045 应有相关序列"
    assert all("common_keywords" in x for x in r), f"6. 格式错: {r}"

    # 7. 公式反查 2^n
    r = _lookup_formula("a(n) = 2^n")
    assert any(m["a_number"] == "A000079" for m in r), f"7. 2^n 应匹配 A000079, got {r}"

    # 8. lookup_formula 无匹配返回空
    r = _lookup_formula("a(n) = zzz_nonexistent_formula(n)")
    assert r == [], f"8. 无匹配应返回 [], got {r}"

    print("[discrete_oeis] self-check OK (8/8)")


if __name__ == "__main__":
    _selfcheck()
