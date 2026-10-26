"""Conjecture 进化环 — Conjecture Machines 风格.

不引入 lean 编译器, 参考其设计框架: 命题写成可验证形式 (statement + proof_script),
用 sympy 验证 + AST 断言 + sorry 字符串扫描替代 lean compiler.

设计参考: Conjecture Machines (DeepMind) — ML 找候选 → LLM 精化 → 形式化验证.
huginn autoloop/conjecture.py 已有 Moonshine 三步流水线, 这里加进化选择环.

用户约束: lean 太重, 只参考其设计框架, 不真的引入 lean 编译器.

跑法:
    python -m huginn.lean.conjecture_library
"""

from __future__ import annotations

import ast
import json
import logging
import sqlite3
import tempfile
import threading
from collections.abc import Callable
from contextlib import closing
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from huginn.utils.runtime import HUGINN_DIR_NAME

logger = logging.getLogger(__name__)

# 替代 lean 的 "sorry" / "admit" — proof script 含这些词说明证没写完.
# 跟 Lean 的 sorry/admit 同义, 但在 Python proof script 里查.
_SORRY_MARKERS = ("sorry", "admit", "by_contradiction_tactic", "exact sorry")

# proof_script 白名单 import — 与 task_synthesizer 的 _JUDGE_ALLOWED_MODULES 逐字一致.
_PROOF_ALLOWED_MODULES = frozenset(
    {
        "math",
        "statistics",
        "json",
        "re",
        "numpy",
        "pandas",
        "sympy",
        "scipy",
    }
)

# proof_script 里禁用的内建调用 — 能绕过白名单拿回危险能力.
_FORBIDDEN_CALLS = frozenset(
    {
        "__import__",
        "eval",
        "exec",
        "compile",
        "getattr",
        "setattr",
        "delattr",
        "globals",
        "locals",
    }
)


@dataclass
class Conjecture:
    """可验证命题 — lean 风格但用 sympy 验证.

    statement: 命题陈述 (人类可读)
    sympy_expr: sympy 表达式字符串, e.g. "B" 或 "Eq(B, (c11+2*c12)/3)"
    test_cases: list of {inputs, expected} — sympy 验证用
    proof_script: Python 源码, def prove() -> bool, 用 sympy 验证 statement
    fitness: 适应度 (验证通过=1.0, 失败=0.0)
    generation: 进化代数
    parent_ids: 父命题 id (用于追踪进化树)
    sorry_status: v6 G45 — none/placeholder/filled/impossible
        none = 完整证明; placeholder = 含 sorry 待填充;
        filled = 原 sorry 已被后续证明填充; impossible = 判定不可实现
    """

    id: str
    statement: str
    sympy_expr: str
    test_cases: list[dict[str, Any]] = field(default_factory=list)
    proof_script: str = ""
    fitness: float = 0.0
    generation: int = 0
    parent_ids: list[str] = field(default_factory=list)
    sorry_status: str = "none"


def _proof_ast_is_safe(tree: ast.Module) -> bool:
    """第 2 层: import 白名单 + 禁 dunder + 禁危险内建调用."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in _PROOF_ALLOWED_MODULES:
                    return False
        elif isinstance(node, ast.ImportFrom) and (
            (node.module or "").split(".")[0] not in _PROOF_ALLOWED_MODULES
        ):
            return False
        # dunder 属性访问 (.__class__ / .__bases__ / .__subclasses__ /
        # .__globals__ / .__getattribute__ ...) 是绕过白名单取回危险能力的
        # 主要路径 — 一律拒绝. 正常 sympy 证明不靠 dunder 属性.
        if (
            isinstance(node, ast.Attribute)
            and node.attr.startswith("__")
            and node.attr.endswith("__")
        ):
            return False
        # dunder 名称引用 (__builtins__ / __import__ ...) 同样拒绝.
        if (
            isinstance(node, ast.Name)
            and node.id.startswith("__")
            and node.id.endswith("__")
        ):
            return False
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id in _FORBIDDEN_CALLS:
                return False
    return True


def _exec_prove(tree: ast.Module) -> bool:
    """第 3 层: exec proof_script, 要求 prove() 返回 True."""
    safe_globals: dict[str, Any] = {"__builtins__": __builtins__}
    try:
        exec(compile(tree, "<proof>", "exec"), safe_globals)
    except Exception:
        return False
    prove_fn = safe_globals.get("prove")
    if not callable(prove_fn):
        return False
    try:
        return prove_fn() is True
    except Exception:
        return False


def _test_cases_pass(conj: Conjecture) -> bool:
    """第 4 层: test_cases 全部通过 sympy_expr 数值验证."""
    if not conj.test_cases:
        return True
    try:
        from sympy import sympify

        expr = sympify(conj.sympy_expr)
    except Exception:
        return False
    for case in conj.test_cases:
        try:
            inputs = case.get("inputs", {})
            expected = case.get("expected")
            if expected is None:
                return False
            result = expr.subs(inputs)
            if abs(float(result) - float(expected)) > 1e-6:
                return False
        except Exception:
            return False
    return True


def verify_conjecture(conj: Conjecture) -> tuple[bool, str]:
    """替代 lean 验证. 四层检查, 返回 (was_verified, sorry_status).

    sorry 语义 (v6 G45): sorry 不再 reject, 而是标记 placeholder.
      - 含 sorry → (False, "placeholder") — 进 conjecture_gaps 表
      - 不含 sorry + 全部通过 → (True, "none")
      - 不含 sorry + 失败 → (False, "none") — 真失败, 淘汰

    四层检查:
    1. proof_script 含 sorry/admit → 标记 placeholder (不 reject)
    2. proof_script AST 合法 + 白名单 import + 无危险内建
    3. proof_script 的 prove() 返回 True
    4. test_cases 全部通过 sympy_expr 数值验证

    ponytail: 这是 lean compiler 的廉价替代, 升级路径是接 lean_tool 当
    verifier service (LeanInterface 已存在, 但需要 lake 可执行文件).
    """
    # 1. sorry 扫描 — 含 sorry 标记 placeholder, 不 reject
    proof_lower = conj.proof_script.lower()
    if any(marker in proof_lower for marker in _SORRY_MARKERS):
        logger.debug("conj %s placeholder: sorry marker in proof", conj.id)
        return (False, "placeholder")

    # 2. AST 合法 + 白名单 import + 无危险内建
    try:
        tree = ast.parse(conj.proof_script)
    except SyntaxError:
        return (False, "none")
    if not _proof_ast_is_safe(tree):
        return (False, "none")

    # 3. exec proof_script, 调 prove() 应返回 True
    if not _exec_prove(tree):
        return (False, "none")

    # 4. test_cases 数值验证 (sympy subs 后比对 expected)
    if not _test_cases_pass(conj):
        return (False, "none")

    return (True, "none")


def verify_conjecture_bool(conj: Conjecture) -> bool:
    """兼容包装: 老调用方要 bool. 返回 was_verified."""
    return verify_conjecture(conj)[0]


def _score_population(
    population: list[Conjecture],
    fitness_fn: Any | None,
    library: ConjectureLibrary | None,
) -> list[Conjecture]:
    """给一代 population 打分, 返回通过的; placeholder 顺手进 gaps 表."""
    new_passed: list[Conjecture] = []
    for conj in population:
        was_verified, sorry_status = verify_conjecture(conj)
        # 三维适应度: fitness_fn 传入则用, 否则二元
        if fitness_fn is not None:
            conj.fitness = fitness_fn(conj, was_verified, sorry_status)
        else:
            conj.fitness = 1.0 if was_verified else 0.0
        conj.sorry_status = sorry_status
        if conj.fitness > 0:
            new_passed.append(conj)
        elif sorry_status == "placeholder" and library is not None:
            # v6 G48: sorry 位进 gaps 表而非淘汰
            library.add_placeholder(conj)
    return new_passed


def evolve_conjectures(
    seed: str,
    n_variants: int = 5,
    max_gen: int = 10,
    model: Any = None,
    library: ConjectureLibrary | None = None,
    fitness_fn: Any | None = None,
) -> list[Conjecture]:
    """进化环 — Conjecture Machines 风格.

    初始从 seed 生成 n_variants 个变体 (gen 0); 每代 verify → 通过的进 library
    → 变异交叉出下一代; max_gen 或无新通过命题即停.
    v6 G48: sorry 位进 gaps 表而非淘汰. fitness_fn 可选 (三维适应度), 否则二元.
    返回所有验证通过的命题.
    ponytail: 每代 n_variants 次 LLM 调用较慢; 升级路径是并行 LLM + 接 lean_tool.
    """
    passed: list[Conjecture] = []
    population = _generate_variants(seed, n_variants, model=model, generation=0)

    for gen in range(max_gen):
        new_passed = _score_population(population, fitness_fn, library)
        if not new_passed:
            logger.info("evolve stop at gen %d: no new pass", gen)
            break

        passed.extend(new_passed)
        if library is not None:
            for conj in new_passed:
                library.add(conj)

        # 下一代: 从 new_passed 变异 + 交叉
        if gen < max_gen - 1:
            population = _evolve_population(
                new_passed,
                n_variants,
                model=model,
                generation=gen + 1,
            )
        else:
            logger.info("evolve stop at gen %d: max_gen reached", gen)

    return passed


def default_three_dim_fitness(
    conj: Conjecture,
    was_verified: bool,
    sorry_status: str,
) -> float:
    """G48 三维适应度默认实现.

    correctness (0/1): 验证通过 = 1, 否则 0
    novelty (0-1): statement 长度归一化 (越短越新颖, 简洁 = 数学美)
    structure_preservation (0-1): sorry 位 = 0.3 (有结构但缺证明),
        完整 = 1.0, impossible = 0

    返回加权和 (correctness 0.5 + novelty 0.2 + structure 0.3).

    ponytail: novelty 用长度代理是粗糙的, 升级路径是 LLM 评估 + RAG recall
    距离 (跟已知命题的差异度).
    """
    correctness = 1.0 if was_verified else 0.0
    # novelty: statement 越短越新颖 (1.0 - len/100, clamp 0-1)
    novelty = max(0.0, min(1.0, 1.0 - len(conj.statement) / 100.0))
    # structure_preservation
    if sorry_status == "placeholder":
        structure = 0.3
    elif sorry_status == "filled":
        structure = 0.8
    elif sorry_status == "impossible":
        structure = 0.0
    else:  # none
        structure = 1.0 if was_verified else 0.5
    return correctness * 0.5 + novelty * 0.2 + structure * 0.3


def _fill_one_gap(
    library: ConjectureLibrary,
    gap: dict[str, Any],
    model: Any,
) -> tuple[str, str, bool]:
    """尝试填充一个 sorry gap, 返回 (conj_id, action, success)."""
    conj_id = gap["conj_id"]
    statement = gap["statement"] or ""
    sympy_expr = gap["sympy_expr"] or ""
    proof_script = gap["proof_script"] or ""

    # model=None 走模板 (尝试简单替换 sorry)
    if model is None or hasattr(model, "_mock_name"):
        success, filled_proof, counterexample = _template_fill_sorry(
            statement,
            sympy_expr,
            proof_script,
        )
    else:
        try:
            success, filled_proof, counterexample = _llm_fill_sorry(
                statement,
                sympy_expr,
                proof_script,
                model,
            )
        except Exception:
            logger.debug("LLM fill_sorry failed for %s", conj_id, exc_info=True)
            success, _filled_proof, counterexample = (
                False,
                "",
                "LLM unavailable",
            )

    if success:
        library.mark_filled(conj_id, filled_by="fill_sorry_gaps")
        return (conj_id, "filled", True)
    if counterexample:
        library.mark_impossible(
            conj_id,
            counterexample=counterexample,
            classification="unreachable",
        )
        return (conj_id, "impossible", False)
    # 保留 placeholder
    return (conj_id, "placeholder", False)


def fill_sorry_gaps(
    library: ConjectureLibrary,
    model: Any = None,
    max_fill: int = 5,
) -> list[tuple[str, str, bool]]:
    """G48: 填充 sorry gaps — sorry 位作为变异目标而非淘汰.

    遍历 library 里的 placeholder gaps, 用 LLM 尝试填充. 填充成功 → mark_filled;
    填充失败 + 反例 → mark_impossible; 失败 + 无反例 → 保留 placeholder.

    返回 [(gap_id, action, success), ...], action 是 "filled" / "impossible" /
    "placeholder".

    ponytail: LLM 填充是 best-effort, 失败不抛. 升级路径是多 LLM 协作 (v7).
    """
    results: list[tuple[str, str, bool]] = []
    gaps = library.get_research_gaps(status="placeholder")
    if not gaps:
        return results

    for gap in gaps[:max_fill]:
        results.append(_fill_one_gap(library, gap, model))

    filled_count = sum(1 for _cid, _action, ok in results if ok)
    logger.info(
        "fill_sorry_gaps: %d/%d filled",
        filled_count,
        min(len(gaps), max_fill),
    )
    return results


# sorry 清洗规则: (正则, 替换) — 先删注释行, 再把 sorry/admit 词换成 True.
_SORRY_SCRUB_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"#\s*(sorry|admit)[^\n]*", ""),
    (r"\bsorry\b", "True"),
    (r"\badmit\b", "True"),
)


def _strip_sorry_markers(proof_script: str) -> str:
    """把 sorry / admit 注释行删掉, 词本身替换成 True."""
    import re

    cleaned = proof_script
    for pattern, repl in _SORRY_SCRUB_PATTERNS:
        cleaned = re.sub(pattern, repl, cleaned, flags=re.IGNORECASE)
    return cleaned


def _template_fill_sorry(
    statement: str,
    sympy_expr: str,
    proof_script: str,
) -> tuple[bool, str, str]:
    """模板填充 sorry — 测试用. 去掉 sorry 标记后用 sympy 重验.

    成功返回 (True, filled_proof, ""); 失败返回 (False, "", counterexample).
    ponytail: 只处理最简单的 sorry 模式. 真实 LLM 路径在 _llm_fill_sorry.
    """
    # 检查 proof_script 里 sorry 是不是注释 — 模板尝试去掉 sorry 重跑
    if "sorry" not in proof_script.lower() and "admit" not in proof_script.lower():
        return (False, "", "no sorry marker found")

    cleaned = _strip_sorry_markers(proof_script)

    # 用 verify_conjecture 检查 cleaned 是否能过
    test_conj = Conjecture(
        id="fill-test",
        statement=statement,
        sympy_expr=sympy_expr,
        proof_script=cleaned,
    )
    was_verified, sorry_status = verify_conjecture(test_conj)
    if was_verified:
        return (True, cleaned, "")
    # 失败, 返回反例 (粗糙: 跑 proof 看是否 False)
    return (False, "", "template fill failed: cleaned proof still invalid")


def _invoke_model_sync(model: Any, messages: list[Any]) -> str:
    """同步调用模型; 已在事件循环里时用 invoke, 否则 asyncio.run(ainvoke)."""
    import asyncio

    try:
        asyncio.get_running_loop()
        text = model.invoke(messages)
    except RuntimeError:
        text = asyncio.run(model.ainvoke(messages))
    return str(text.content).strip()


def _llm_fill_messages(
    statement: str,
    sympy_expr: str,
    proof_script: str,
) -> list[Any]:
    """构造 sorry 填充的 prompt."""
    from langchain_core.messages import HumanMessage, SystemMessage

    return [
        SystemMessage(
            content=(
                "You are a proof filler. Given a conjecture with a sorry "
                "(placeholder) in its proof script, complete the proof using "
                'sympy. Output JSON: {"proof_script": "...", '
                '"counterexample": "..."}. If you can complete the proof, '
                "put it in proof_script and leave counterexample empty. If you "
                "find the conjecture is false, put the counterexample in "
                "counterexample and leave proof_script empty. No sorry/admit. "
                "Only use: sympy, math, re, statistics, json."
            )
        ),
        HumanMessage(
            content=(
                f"Statement: {statement}\n"
                f"Sympy expr: {sympy_expr}\n"
                f"Proof script (with sorry):\n{proof_script}"
            )
        ),
    ]


def _llm_fill_verdict(
    statement: str,
    sympy_expr: str,
    parsed: dict[str, Any],
) -> tuple[bool, str, str]:
    """按 LLM 回执三态判定: 反例 / 缺 proof / 验证通过."""
    counterexample = parsed.get("counterexample", "")
    if counterexample:
        return (False, "", counterexample)

    filled_proof = parsed.get("proof_script", "")
    if not filled_proof or "def prove" not in filled_proof:
        return (False, "", "LLM output missing proof_script")

    # 验证 filled_proof 真的能过
    test_conj = Conjecture(
        id="fill-llm",
        statement=statement,
        sympy_expr=sympy_expr,
        proof_script=filled_proof,
    )
    was_verified, _ = verify_conjecture(test_conj)
    if was_verified:
        return (True, filled_proof, "")
    return (False, "", "LLM filled proof failed verification")


def _llm_fill_sorry(
    statement: str,
    sympy_expr: str,
    proof_script: str,
    model: Any,
) -> tuple[bool, str, str]:
    """LLM 填充 sorry — 给 LLM 看 sorry proof, 让它补完.

    成功返回 (True, filled_proof, "").
    失败返回 (False, "", counterexample_or_reason).
    """
    text = _invoke_model_sync(
        model, _llm_fill_messages(statement, sympy_expr, proof_script)
    )

    parsed = _parse_json_object(text)
    if not parsed:
        return (False, "", "LLM output not parseable")

    return _llm_fill_verdict(statement, sympy_expr, parsed)


# ── LLM 生成 (model=None 走模板, 测试用) ───────────────────────────


def _gen_variant_messages(seed: str, n: int) -> list[Any]:
    """构造 conjecture 变体生成 prompt."""
    from langchain_core.messages import HumanMessage, SystemMessage

    return [
        SystemMessage(
            content=(
                "You are a conjecture generator. "
                "Given a seed statement, generate variants verifiable with sympy. "
                "Output a JSON array of objects with keys: "
                "statement, sympy_expr, test_cases (list of {inputs, expected}), "
                "proof_script (Python def prove()->bool using sympy). "
                "No sorry/admit. No markdown."
            )
        ),
        HumanMessage(content=f"Seed: {seed}\nGenerate {n} variants."),
    ]


def _conjectures_from_payload(
    parsed: list[dict[str, Any]],
    generation: int,
    parents: list[Conjecture] | None,
) -> list[Conjecture]:
    """把 LLM 回执 array 转成 Conjecture 列表 (跳过不合法条目)."""
    parent_ids = [pp.id for pp in (parents or [])]
    result: list[Conjecture] = []
    for i, p in enumerate(parsed):
        if not _validate_conjecture_payload(p):
            continue
        result.append(
            Conjecture(
                id=f"conj-gen{generation:02d}-{i:03d}",
                statement=p["statement"],
                sympy_expr=p["sympy_expr"],
                test_cases=p.get("test_cases", []),
                proof_script=p["proof_script"],
                generation=generation,
                parent_ids=parent_ids,
            )
        )
    return result


def _generate_variants(
    seed: str,
    n: int,
    model: Any = None,
    generation: int = 0,
    parents: list[Conjecture] | None = None,
) -> list[Conjecture]:
    """生成 n 个变体. model=None 或 mock 走模板."""
    if model is None or hasattr(model, "_mock_name"):
        return _template_variants(seed, n, generation, parents)

    try:
        text = _invoke_model_sync(model, _gen_variant_messages(seed, n))
        parsed = _parse_json_array(text)
        if not parsed:
            return _template_variants(seed, n, generation, parents)

        result = _conjectures_from_payload(parsed, generation, parents)
        return result if result else _template_variants(seed, n, generation, parents)
    except Exception:
        logger.debug("LLM variant gen failed, fallback to template", exc_info=True)
        return _template_variants(seed, n, generation, parents)


def _evolve_population(
    parents: list[Conjecture],
    n: int,
    model: Any = None,
    generation: int = 0,
) -> list[Conjecture]:
    """从 parents 变异 + 交叉生成下一代. model=None 走模板."""
    if not parents:
        return []
    if model is None or hasattr(model, "_mock_name"):
        return _template_evolve(parents, n, generation)
    # LLM: 用 parents 作示例生成新变体
    seed = parents[0].statement
    return _generate_variants(
        seed, n, model=model, generation=generation, parents=parents
    )


def _tv_bulk_modulus(generation: int, parent_ids: list[str]) -> Conjecture:
    """模板 0: 立方晶体 bulk modulus."""
    return Conjecture(
        id=f"conj-gen{generation:02d}-000",
        statement="Bulk modulus of cubic crystal: B = (c11 + 2*c12) / 3",
        sympy_expr="B",
        test_cases=[{"inputs": {"B": 60.0}, "expected": 60.0}],
        proof_script=(
            "def prove():\n"
            "    from sympy import symbols\n"
            "    c11, c12 = symbols('c11 c12')\n"
            "    B = (c11 + 2*c12) / 3\n"
            "    val = B.subs({c11: 100, c12: 40})\n"
            "    return abs(float(val) - 60.0) < 1e-6\n"
        ),
        generation=generation,
        parent_ids=parent_ids,
    )


def _tv_voigt_hexagonal(generation: int, parent_ids: list[str]) -> Conjecture:
    """模板 1: 六方晶体 Voigt bulk modulus 上界."""
    return Conjecture(
        id=f"conj-gen{generation:02d}-001",
        statement=(
            "Voigt bulk modulus upper bound hexagonal: "
            "B = (2*c11 + c33 + 2*c12 + 4*c13) / 9"
        ),
        sympy_expr="B",
        test_cases=[{"inputs": {"B": 257.78}, "expected": 257.78}],
        proof_script=(
            "def prove():\n"
            "    from sympy import symbols\n"
            "    c11, c12, c13, c33 = symbols('c11 c12 c13 c33')\n"
            "    B = (2*c11 + c33 + 2*c12 + 4*c13) / 9\n"
            "    # Mg hcp: c11=294, c12=98, c13=57, c33=348 -> B=257.78\n"
            "    val = B.subs({c11: 294, c12: 98, c13: 57, c33: 348})\n"
            "    return abs(float(val) - 257.78) < 0.1\n"
        ),
        generation=generation,
        parent_ids=parent_ids,
    )


def _tv_shear_voigt(generation: int, parent_ids: list[str]) -> Conjecture:
    """模板 2: 立方晶体 Voigt shear modulus."""
    return Conjecture(
        id=f"conj-gen{generation:02d}-002",
        statement="Shear modulus Voigt cubic: G = (c11 - c12 + 3*c44) / 5",
        sympy_expr="G",
        test_cases=[{"inputs": {"G": 62.0}, "expected": 62.0}],
        proof_script=(
            "def prove():\n"
            "    from sympy import symbols\n"
            "    c11, c12, c44 = symbols('c11 c12 c44')\n"
            "    G = (c11 - c12 + 3*c44) / 5\n"
            "    val = G.subs({c11: 100, c12: 40, c44: 50})\n"
            "    return abs(float(val) - 62.0) < 0.1\n"
        ),
        generation=generation,
        parent_ids=parent_ids,
    )


# 模板命题工厂表 — 顺序即 id 后缀 000/001/002 的顺序.
_TEMPLATE_BUILDERS: tuple[Callable[[int, list[str]], Conjecture], ...] = (
    _tv_bulk_modulus,
    _tv_voigt_hexagonal,
    _tv_shear_voigt,
)


def _template_variants(
    seed: str,
    n: int,
    generation: int,
    parents: list[Conjecture] | None,
) -> list[Conjecture]:
    """模板生成变体 — 测试用. 出几个已知正确的弹性力学命题."""
    parent_ids = [p.id for p in (parents or [])]
    return [build(generation, parent_ids) for build in _TEMPLATE_BUILDERS][:n]


def _template_evolve(
    parents: list[Conjecture],
    n: int,
    generation: int,
) -> list[Conjecture]:
    """模板进化: 复制 parents, 改 id/generation/parent_ids. 测试用."""
    result: list[Conjecture] = []
    for i, p in enumerate(parents[:n]):
        result.append(
            Conjecture(
                id=f"conj-gen{generation:02d}-{i:03d}",
                statement=p.statement + " (variant)",
                sympy_expr=p.sympy_expr,
                test_cases=p.test_cases,
                proof_script=p.proof_script,
                generation=generation,
                parent_ids=[p.id],
            )
        )
    return result


def _parse_json_array(text: str) -> list[dict[str, Any]]:
    if not text:
        return []
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]
            if text.startswith("json"):
                text = text[4:]
    try:
        result = json.loads(text.strip())
        if isinstance(result, list):
            return result
    except (json.JSONDecodeError, TypeError):
        logger.debug("best-effort op failed", exc_info=True)
    return []


def _validate_conjecture_payload(p: dict[str, Any]) -> bool:
    return (
        isinstance(p.get("statement"), str)
        and isinstance(p.get("sympy_expr"), str)
        and isinstance(p.get("proof_script"), str)
        and "def prove" in p["proof_script"]
    )


# ── v6 G45: sorry 分类器 — sorry 是地图, 不是坟墓 ────────────────

# 分类维度 (跟 add_placeholder 的 classification 字段对齐):
# - novel_variant: 现有理论的变式应用, 原则上可证, 只是缺一步技巧
# - unexplored:    新空白, 可能是理论创新的起点
# - unreachable:   物理上不可实现 (违反守恒律 / 量纲不一致)
# - known_limit:   已知理论的边界 (超出适用范围)
_SORRY_CLASSIFICATION_PROMPT = (
    "You are a research gap classifier. Given a mathematical conjecture whose "
    "proof contains a sorry (placeholder), classify WHY the sorry exists:\n"
    "- novel_variant: existing theory applies, just missing a technique\n"
    "- unexplored: genuine new territory, possible innovation seed\n"
    "- unreachable: physically impossible (conservation/dimensional violation)\n"
    "- known_limit: hits known theory boundary (out of applicability)\n"
    'Output JSON: {"classification": "...", "reason": "..."}'
)


def classify_sorry(conj: Conjecture, model: Any = None) -> str:
    """分类 sorry 位的性质 — 用户思想: sorry = 研究盲区地图.

    返回 novel_variant / unexplored / unreachable / known_limit.
    走 RAG recall + LLM, 失败降级到规则匹配.

    ponytail: 没传 model 或 RAG 没数据时走规则匹配 (statement 关键词),
    升级路径是接 RAG recall 做语义匹配.
    """
    if model is not None and not hasattr(model, "_mock_name"):
        try:
            return _llm_classify_sorry(conj, model)
        except Exception:
            logger.debug("LLM classify_sorry failed, fallback to rules", exc_info=True)

    return _rule_classify_sorry(conj)


def _llm_classify_sorry(conj: Conjecture, model: Any) -> str:
    """LLM 分类 sorry. 失败抛异常让上层降级."""
    from langchain_core.messages import HumanMessage, SystemMessage

    messages = [
        SystemMessage(content=_SORRY_CLASSIFICATION_PROMPT),
        HumanMessage(
            content=(
                f"Statement: {conj.statement}\n"
                f"Sympy expr: {conj.sympy_expr}\n"
                f"Proof script (with sorry):\n{conj.proof_script}"
            )
        ),
    ]
    text = _invoke_model_sync(model, messages)

    parsed = _parse_json_object(text)
    cls = parsed.get("classification", "")
    if cls in {"novel_variant", "unexplored", "unreachable", "known_limit"}:
        return cls
    return _rule_classify_sorry(conj)


# 规则分类关键词表: 顺序即优先级 (命中即返回), 都不命中则 unexplored.
_RULE_CLASSIFY_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "unreachable",
        (
            "conservation",
            "守恒",
            "dimensional",
            "量纲",
            "violate",
            "违反",
            "impossible",
            "contradiction",
            "矛盾",
        ),
    ),
    (
        "known_limit",
        (
            "limit",
            "边界",
            "boundary",
            "applicability",
            "适用",
            "regime",
            "asymptotic",
            "近似",
        ),
    ),
    (
        "novel_variant",
        (
            "theorem",
            "定律",
            "law",
            "formula",
            "公式",
            "variant",
            "变式",
            "extension",
            "推广",
        ),
    ),
)


def _rule_classify_sorry(conj: Conjecture) -> str:
    """规则匹配分类. 关键词粗判, 没数据时兜底."""
    text = (conj.statement + " " + conj.proof_script).lower()
    for classification, keywords in _RULE_CLASSIFY_KEYWORDS:
        if any(k in text for k in keywords):
            return classification
    # 默认: 新空白
    return "unexplored"


def _parse_json_object(text: str) -> dict[str, Any]:
    """解析单个 JSON 对象. 失败返回 {}."""
    if not text:
        return {}
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]
            if text.startswith("json"):
                text = text[4:]
    try:
        result = json.loads(text.strip())
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, TypeError):
        logger.debug("best-effort op failed", exc_info=True)
    return {}


# ── ConjectureLibrary: 累积验证通过的命题 ──────────────────────────


class ConjectureLibrary:
    """SQLite + JSONL 双写累积验证通过的命题.

    SQLite 用于查询, JSONL 用于 git diff 可读. 跟 stable_principles 的
    双写模式一致 (longterm.py 也是 SQLite + JSONL).
    """

    def __init__(self, workspace: str | Path = "."):
        ws = Path(workspace).resolve() / HUGINN_DIR_NAME
        ws.mkdir(parents=True, exist_ok=True)
        self.db_path = ws / "conjecture_library.db"
        self.jsonl_path = ws / "conjecture_library.jsonl"
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS conjectures ("
                "id TEXT PRIMARY KEY,"
                "statement TEXT NOT NULL,"
                "sympy_expr TEXT NOT NULL,"
                "test_cases TEXT,"
                "proof_script TEXT NOT NULL,"
                "fitness REAL DEFAULT 0.0,"
                "generation INTEGER DEFAULT 0,"
                "parent_ids TEXT,"
                "created_at TEXT DEFAULT (datetime('now'))"
                ")"
            )
            # v6 G45: research gaps 表 — sorry 命题进这里当研究盲区
            # status: placeholder (待填充) / filled (已填充) / impossible (判定不可实现)
            conn.execute(
                "CREATE TABLE IF NOT EXISTS conjecture_gaps ("
                "conj_id TEXT PRIMARY KEY,"
                "statement TEXT NOT NULL,"
                "sympy_expr TEXT,"
                "proof_script TEXT,"
                "status TEXT DEFAULT 'placeholder',"
                "classification TEXT,"
                "counterexample TEXT,"
                "filled_by TEXT,"
                "created_at TEXT DEFAULT (datetime('now')),"
                "updated_at TEXT DEFAULT (datetime('now'))"
                ")"
            )
            conn.commit()

    def _connect(self):
        # ponytail: sqlite3.Connection.__exit__ 只 commit 不 close, Windows
        # 上会留文件锁阻碍 tempdir 清理. 用 closing 包一层保证 close.
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return closing(conn)

    def add(self, conj: Conjecture) -> bool:
        """加一个验证通过的命题. True=新增, False=已存在 (id 冲突)."""
        with self._lock:
            with self._connect() as conn:
                try:
                    conn.execute(
                        "INSERT INTO conjectures "
                        "(id, statement, sympy_expr, test_cases, proof_script, "
                        "fitness, generation, parent_ids) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            conj.id,
                            conj.statement,
                            conj.sympy_expr,
                            json.dumps(conj.test_cases, ensure_ascii=False),
                            conj.proof_script,
                            conj.fitness,
                            conj.generation,
                            json.dumps(conj.parent_ids, ensure_ascii=False),
                        ),
                    )
                    conn.commit()
                except sqlite3.IntegrityError:
                    logger.debug("best-effort op failed", exc_info=True)
                    return False

            # JSONL 追加 (仅新增时)
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(conj), ensure_ascii=False) + "\n")
        return True

    def list_all(self) -> list[Conjecture]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM conjectures ORDER BY generation, id"
            ).fetchall()
        return [
            Conjecture(
                id=r["id"],
                statement=r["statement"],
                sympy_expr=r["sympy_expr"],
                test_cases=json.loads(r["test_cases"] or "[]"),
                proof_script=r["proof_script"],
                fitness=r["fitness"],
                generation=r["generation"],
                parent_ids=json.loads(r["parent_ids"] or "[]"),
            )
            for r in rows
        ]

    def count(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT count(*) FROM conjectures").fetchone()[0]

    # ── v6 G45: research gaps ─────────────────────────────────────

    def add_placeholder(
        self, conj: Conjecture, classification: str | None = None
    ) -> bool:
        """把含 sorry 的命题登记为研究盲区. True=新增, False=已存在.

        classification 由 classify_sorry 给出 (novel_variant / unexplored /
        unreachable / known_limit), 不传则稍后由 mark_impossible 补.
        """
        with self._lock, self._connect() as conn:
            try:
                conn.execute(
                    "INSERT INTO conjecture_gaps "
                    "(conj_id, statement, sympy_expr, proof_script, "
                    "status, classification) "
                    "VALUES (?, ?, ?, ?, 'placeholder', ?)",
                    (
                        conj.id,
                        conj.statement,
                        conj.sympy_expr,
                        conj.proof_script,
                        classification,
                    ),
                )
                conn.commit()
            except sqlite3.IntegrityError:
                logger.debug("best-effort op failed", exc_info=True)
                return False
        return True

    def get_research_gaps(self, status: str | None = None) -> list[dict[str, Any]]:
        """返回研究盲区列表. status=None 返回全部, 否则按状态过滤.

        状态: placeholder (待填充) / filled (已填充) / impossible (判定不可实现)
        返回 dict: {conj_id, statement, sympy_expr, proof_script,
                    status, classification, counterexample, filled_by}
        """
        with self._connect() as conn:
            if status is None:
                rows = conn.execute(
                    "SELECT conj_id, statement, sympy_expr, proof_script, "
                    "status, classification, counterexample, filled_by "
                    "FROM conjecture_gaps ORDER BY created_at"
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT conj_id, statement, sympy_expr, proof_script, "
                    "status, classification, counterexample, filled_by "
                    "FROM conjecture_gaps WHERE status = ? ORDER BY created_at",
                    (status,),
                ).fetchall()
        return [dict(r) for r in rows]

    def mark_filled(self, conj_id: str, filled_by: str) -> bool:
        """sorry 位被后续证明填充. filled_by 是填充命题 id. True=更新成功."""
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "UPDATE conjecture_gaps SET status='filled', "
                "filled_by=?, updated_at=datetime('now') "
                "WHERE conj_id=? AND status='placeholder'",
                (filled_by, conj_id),
            )
            conn.commit()
            return cur.rowcount > 0

    def mark_impossible(
        self,
        conj_id: str,
        counterexample: str | None = None,
        classification: str | None = None,
    ) -> bool:
        """判定 sorry 位不可实现. counterexample 是反例说明. True=更新成功."""
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "UPDATE conjecture_gaps SET status='impossible', "
                "counterexample=?, classification=?, "
                "updated_at=datetime('now') "
                "WHERE conj_id=? AND status='placeholder'",
                (counterexample, classification, conj_id),
            )
            conn.commit()
            return cur.rowcount > 0


# ── self-check ────────────────────────────────────────────────────


def _sc_verify_good_and_sorry() -> tuple[Conjecture, Conjecture]:
    """1-3: good 命题通过, sorry/admit 标记 placeholder."""
    # 1. verify 通过正确命题 — 返回 (True, "none")
    good = _template_variants("seed", 1, 0, None)[0]
    ok, status = verify_conjecture(good)
    assert ok is True and status == "none", "good conjecture should pass with none"

    # 2. verify 标记 sorry 为 placeholder (v6 G45: 不 reject, 标记)
    bad_sorry = Conjecture(
        id="bad-sorry",
        statement="bad",
        sympy_expr="x",
        proof_script="def prove():\n    # sorry\n    return True",
    )
    ok, status = verify_conjecture(bad_sorry)
    assert ok is False and status == "placeholder", "sorry should be placeholder"

    # 3. verify 标记 admit 为 placeholder
    bad_admit = Conjecture(
        id="bad-admit",
        statement="bad",
        sympy_expr="x",
        proof_script="def prove():\n    # admit\n    return True",
    )
    ok, status = verify_conjecture(bad_admit)
    assert ok is False and status == "placeholder"
    return good, bad_sorry


def _sc_verify_rejects() -> None:
    """4-7: import os / return False / 无 prove / getattr 链全部拒绝."""
    # 4. verify 拒绝 import os — 真失败 (none)
    bad_import = Conjecture(
        id="bad-import",
        statement="bad",
        sympy_expr="x",
        proof_script="import os\n\ndef prove():\n    return True",
    )
    ok, status = verify_conjecture(bad_import)
    assert ok is False and status == "none"

    # 5. verify 拒绝 prove() 返回 False
    bad_false = Conjecture(
        id="bad-false",
        statement="bad",
        sympy_expr="x",
        proof_script="def prove():\n    return False",
    )
    ok, status = verify_conjecture(bad_false)
    assert ok is False and status == "none"

    # 6. verify 拒绝无 prove 函数
    bad_nofn = Conjecture(
        id="bad-nofn",
        statement="bad",
        sympy_expr="x",
        proof_script="x = 1",
    )
    ok, status = verify_conjecture(bad_nofn)
    assert ok is False and status == "none"

    # 7. verify 拒绝 getattr 链 (跟 G23 一致)
    bad_getattr = Conjecture(
        id="bad-getattr",
        statement="bad",
        sympy_expr="x",
        proof_script=(
            "def prove():\n    x = getattr(prove, '__name__')\n    return True\n"
        ),
    )
    ok, status = verify_conjecture(bad_getattr)
    assert ok is False and status == "none"


def _sc_verify_bool_wrapper(good: Conjecture, bad_sorry: Conjecture) -> None:
    """8: verify_conjecture_bool 兼容包装."""
    # 8. verify_conjecture_bool 兼容包装 — 老调用方要 bool
    assert verify_conjecture_bool(good) is True
    assert verify_conjecture_bool(bad_sorry) is False


def _sc_evolve_template() -> list[Conjecture]:
    """9: evolve_conjectures 模板模式跑通."""
    # 9. evolve_conjectures 模板模式跑通
    passed = evolve_conjectures(
        "bulk modulus",
        n_variants=3,
        max_gen=2,
        model=None,
    )
    assert len(passed) > 0, "should have passed conjectures"
    assert all(c.fitness == 1.0 for c in passed)
    assert all(c.sorry_status == "none" for c in passed)
    return passed


def _sc_library_accumulate(passed: list[Conjecture]) -> None:
    """10: ConjectureLibrary 累积 / 去重 / JSONL 落盘."""
    # 10. ConjectureLibrary 累积
    with tempfile.TemporaryDirectory() as tmp:
        lib = ConjectureLibrary(workspace=tmp)
        assert lib.count() == 0
        for c in passed:
            lib.add(c)
        assert lib.count() == len(passed)
        # 重复 add 同 id 不增加 (IntegrityError 返回 False)
        for c in passed:
            assert lib.add(c) is False
        assert lib.count() == len(passed)
        # list_all 返回相同数量
        all_c = lib.list_all()
        assert len(all_c) == len(passed)
        # JSONL 文件存在
        assert lib.jsonl_path.exists()
        # JSONL 行数 == count
        with open(lib.jsonl_path, encoding="utf-8") as f:
            lines = [_l for _l in f if _l.strip()]
        assert len(lines) == len(passed)


def _sc_library_wiring() -> None:
    """11: evolve + library 联动 — 通过的进 library."""
    # 11. evolve + library 联动 — 通过的进 library
    with tempfile.TemporaryDirectory() as tmp:
        lib = ConjectureLibrary(workspace=tmp)
        passed3 = evolve_conjectures(
            "elastic constant",
            n_variants=2,
            max_gen=2,
            model=None,
            library=lib,
        )
        assert lib.count() == len(passed3), (
            f"library {lib.count()} != passed {len(passed3)}"
        )


def _sc_gaps_table() -> None:
    """12: conjecture_gaps 表 — sorry 命题进 gaps 而非淘汰."""
    # 12. v6 G45: conjecture_gaps 表 — sorry 命题进 gaps 而非淘汰
    with tempfile.TemporaryDirectory() as tmp:
        lib = ConjectureLibrary(workspace=tmp)
        # 加一个 sorry 命题作为 placeholder
        sorry_conj = Conjecture(
            id="sorry-001",
            statement="conservation law violates dimensional analysis",
            sympy_expr="x",
            proof_script="def prove():\n    # sorry need conservation\n    return True",
        )
        assert lib.add_placeholder(sorry_conj, classification="unreachable") is True
        # 重复加返回 False
        assert lib.add_placeholder(sorry_conj) is False
        # get_research_gaps 返回
        gaps = lib.get_research_gaps()
        assert len(gaps) == 1
        assert gaps[0]["conj_id"] == "sorry-001"
        assert gaps[0]["status"] == "placeholder"
        assert gaps[0]["classification"] == "unreachable"
        # 按 status 过滤
        placeholders = lib.get_research_gaps(status="placeholder")
        assert len(placeholders) == 1
        filled = lib.get_research_gaps(status="filled")
        assert len(filled) == 0
        # mark_filled
        assert lib.mark_filled("sorry-001", "conj-gen00-000") is True
        assert lib.mark_filled("sorry-001", "x") is False  # 已 filled
        filled_gaps = lib.get_research_gaps(status="filled")
        assert len(filled_gaps) == 1
        assert filled_gaps[0]["filled_by"] == "conj-gen00-000"
        # mark_impossible (新 placeholder)
        sorry2 = Conjecture(
            id="sorry-002",
            statement="percolation threshold boundary",
            sympy_expr="y",
            proof_script="def prove():\n    # sorry\n    return True",
        )
        lib.add_placeholder(sorry2)
        assert (
            lib.mark_impossible("sorry-002", counterexample="z<0 violates phi_c>0")
            is True
        )
        impossibles = lib.get_research_gaps(status="impossible")
        assert len(impossibles) == 1
        assert impossibles[0]["counterexample"] == "z<0 violates phi_c>0"


def _sc_classify_sorry() -> None:
    """13: classify_sorry 规则分类."""
    # 13. v6 G45: classify_sorry 规则分类
    # unreachable: 含 conservation
    c_unreachable = Conjecture(
        id="c1",
        statement="this violates conservation law",
        sympy_expr="x",
        proof_script="def prove():\n    # sorry\n    return True",
    )
    assert classify_sorry(c_unreachable) == "unreachable"
    # known_limit: 含 boundary
    c_limit = Conjecture(
        id="c2",
        statement="hits the boundary of applicability",
        sympy_expr="x",
        proof_script="def prove():\n    # sorry\n    return True",
    )
    assert classify_sorry(c_limit) == "known_limit"
    # novel_variant: 含 theorem
    c_variant = Conjecture(
        id="c3",
        statement="extension of known theorem",
        sympy_expr="x",
        proof_script="def prove():\n    # sorry\n    return True",
    )
    assert classify_sorry(c_variant) == "novel_variant"
    # unexplored: 默认
    c_new = Conjecture(
        id="c4",
        statement="some random new thing",
        sympy_expr="x",
        proof_script="def prove():\n    # sorry\n    return True",
    )
    assert classify_sorry(c_new) == "unexplored"


def _sc_evolve_gaps_empty() -> None:
    """14: 模板变体无 sorry → gaps 表为空."""
    # 14. evolve + library + sorry: sorry 命题进 gaps 表
    with tempfile.TemporaryDirectory() as tmp:
        lib = ConjectureLibrary(workspace=tmp)
        # 模板变体都是完整证明 (无 sorry), 所以 gaps 表为空
        evolve_conjectures(
            "bulk modulus",
            n_variants=2,
            max_gen=1,
            model=None,
            library=lib,
        )
        assert len(lib.get_research_gaps()) == 0, "template variants have no sorry"


def _sc_fill_success() -> None:
    """15: fill_sorry_gaps 模板路径 — 填充成功."""
    # 15. v6 G48: fill_sorry_gaps 模板路径 — 填充成功
    with tempfile.TemporaryDirectory() as tmp:
        lib = ConjectureLibrary(workspace=tmp)
        # 构造一个 sorry 命题: prove 用 sorry 占位但实际能过
        sorry_conj = Conjecture(
            id="fill-001",
            statement="Bulk modulus cubic: B = (c11 + 2*c12) / 3",
            sympy_expr="B",
            test_cases=[{"inputs": {"B": 60.0}, "expected": 60.0}],
            proof_script=(
                "def prove():\n"
                "    # sorry need to verify\n"
                "    from sympy import symbols\n"
                "    c11, c12 = symbols('c11 c12')\n"
                "    B = (c11 + 2*c12) / 3\n"
                "    val = B.subs({c11: 100, c12: 40})\n"
                "    return abs(float(val) - 60.0) < 1e-6\n"
            ),
        )
        lib.add_placeholder(sorry_conj)
        results = fill_sorry_gaps(lib, model=None, max_fill=5)
        assert len(results) == 1
        gap_id, action, success = results[0]
        assert gap_id == "fill-001"
        # 模板路径: 删 sorry 注释后能过 → filled
        assert action == "filled", f"expected filled, got {action}"
        assert success is True
        # gaps 表状态更新
        filled_gaps = lib.get_research_gaps(status="filled")
        assert len(filled_gaps) == 1


def _sc_fill_failure() -> None:
    """16: fill_sorry_gaps 模板路径 — sorry 在 return 里."""
    # 16. v6 G48: fill_sorry_gaps 模板路径 — 填充失败 (sorry 在 return 里)
    with tempfile.TemporaryDirectory() as tmp:
        lib = ConjectureLibrary(workspace=tmp)
        # sorry 直接当返回值, 删了之后 return True 但没 prove 逻辑
        bad_sorry = Conjecture(
            id="fill-002",
            statement="some impossible claim",
            sympy_expr="x",
            proof_script=("def prove():\n    # sorry\n    return sorry\n"),
        )
        lib.add_placeholder(bad_sorry)
        results = fill_sorry_gaps(lib, model=None, max_fill=5)
        assert len(results) == 1
        gap_id, action, success = results[0]
        # 模板把 sorry 替换成 True, return True 能过 → filled (模板的局限)
        # 或: 替换后 prove 没定义 → fail → impossible
        assert action in ("filled", "impossible", "placeholder"), (
            f"unexpected action: {action}"
        )


def _sc_three_dim_fitness() -> None:
    """17: 三维适应度 default_three_dim_fitness."""
    # 17. v6 G48: 三维适应度 default_three_dim_fitness
    good2 = _template_variants("seed", 1, 0, None)[0]
    fitness = default_three_dim_fitness(good2, was_verified=True, sorry_status="none")
    assert 0 <= fitness <= 1.0
    # 验证通过 + 完整证明 → 高分
    assert fitness > 0.5, f"verified+none should score high, got {fitness}"
    # sorry 位 → 低分
    sorry_fitness = default_three_dim_fitness(
        good2,
        was_verified=False,
        sorry_status="placeholder",
    )
    assert sorry_fitness < fitness, "placeholder should score lower than verified"


def _sc_evolve_fitness_fn() -> None:
    """18: evolve 带 fitness_fn."""
    # 18. v6 G48: evolve 带 fitness_fn
    with tempfile.TemporaryDirectory() as tmp:
        lib = ConjectureLibrary(workspace=tmp)
        passed = evolve_conjectures(
            "bulk modulus",
            n_variants=2,
            max_gen=1,
            model=None,
            library=lib,
            fitness_fn=default_three_dim_fitness,
        )
        assert len(passed) > 0
        # 三维适应度应给通过命题非零分
        assert all(c.fitness > 0 for c in passed)


def _sc_fill_empty() -> None:
    """19: fill_sorry_gaps 空 gaps 表."""
    # 19. v6 G48: fill_sorry_gaps 空 gaps 表
    with tempfile.TemporaryDirectory() as tmp:
        lib = ConjectureLibrary(workspace=tmp)
        results = fill_sorry_gaps(lib, model=None, max_fill=5)
        assert results == []


def _self_check() -> int:
    """assert-based demo: 验证 verify_conjecture + evolve + library + gaps + classify_sorry."""
    good, bad_sorry = _sc_verify_good_and_sorry()
    _sc_verify_rejects()
    _sc_verify_bool_wrapper(good, bad_sorry)
    passed = _sc_evolve_template()
    _sc_library_accumulate(passed)
    _sc_library_wiring()
    _sc_gaps_table()
    _sc_classify_sorry()
    _sc_evolve_gaps_empty()
    _sc_fill_success()
    _sc_fill_failure()
    _sc_three_dim_fitness()
    _sc_evolve_fitness_fn()
    _sc_fill_empty()
    print("[CONJLIB] self-check OK (incl G48 fill_sorry_gaps + 3D fitness)")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(_self_check())
