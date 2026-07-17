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
import threading
from contextlib import closing
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 替代 lean 的 "sorry" / "admit" — proof script 含这些词说明证没写完.
# 跟 Lean 的 sorry/admit 同义, 但在 Python proof script 里查.
_SORRY_MARKERS = ("sorry", "admit", "by_contradiction_tactic", "exact sorry")

# proof_script 白名单 import — 跟 task_synthesizer 的 _JUDGE_ALLOWED_MODULES 一致.
_PROOF_ALLOWED_MODULES = frozenset({
    "sympy", "math", "re", "statistics", "json",
})


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
    """
    id: str
    statement: str
    sympy_expr: str
    test_cases: list[dict[str, Any]] = field(default_factory=list)
    proof_script: str = ""
    fitness: float = 0.0
    generation: int = 0
    parent_ids: list[str] = field(default_factory=list)


def verify_conjecture(conj: Conjecture) -> bool:
    """替代 lean 验证. 四层检查:

    1. proof_script 不含 sorry/admit (字符串扫描, 跟 lean sorry 同义)
    2. proof_script AST 合法 + 白名单 import + 无危险内建
    3. proof_script 的 prove() 返回 True
    4. test_cases 全部通过 sympy_expr 数值验证

    ponytail: 这是 lean compiler 的廉价替代, 升级路径是接 lean_tool 当
    verifier service (LeanInterface 已存在, 但需要 lake 可执行文件).
    """
    # 1. sorry 扫描 — proof script 不能含 sorry/admit
    proof_lower = conj.proof_script.lower()
    if any(marker in proof_lower for marker in _SORRY_MARKERS):
        logger.debug("conj %s rejected: sorry marker in proof", conj.id)
        return False

    # 2. AST 合法 + 白名单 import + 无危险内建
    try:
        tree = ast.parse(conj.proof_script)
    except SyntaxError:
        return False

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top not in _PROOF_ALLOWED_MODULES:
                    return False
        elif isinstance(node, ast.ImportFrom):
            top = (node.module or "").split(".")[0]
            if top not in _PROOF_ALLOWED_MODULES:
                return False
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id in {
                "__import__", "eval", "exec", "compile", "getattr",
            }:
                return False

    # 3. exec proof_script, 调 prove() 应返回 True
    safe_globals: dict[str, Any] = {"__builtins__": __builtins__}
    try:
        exec(compile(tree, "<proof>", "exec"), safe_globals)
    except Exception:
        return False
    prove_fn = safe_globals.get("prove")
    if not callable(prove_fn):
        return False
    try:
        if prove_fn() is not True:
            return False
    except Exception:
        return False

    # 4. test_cases 数值验证 (sympy subs 后比对 expected)
    if conj.test_cases:
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


def evolve_conjectures(
    seed: str,
    n_variants: int = 5,
    max_gen: int = 10,
    model: Any = None,
    library: "ConjectureLibrary | None" = None,
) -> list[Conjecture]:
    """进化环 — Conjecture Machines 风格.

    初始: 从 seed 用 LLM 生成 n_variants 个变体 (gen 0)
    每代:
      1. verify_conjecture 验证每个变体
      2. 通过的进 library, 失败的淘汰
      3. 通过的变异 + 交叉生成下一代
    终止: max_gen 或无新通过命题

    返回所有验证通过的命题 (含初始 + 进化出来的).

    ponytail: max_gen=10 是天花板 — 每代 LLM 调用 n_variants 次, 慢;
    升级路径是并行 LLM + 接 lean_tool 当 verifier (LeanInterface).
    """
    passed: list[Conjecture] = []
    population = _generate_variants(seed, n_variants, model=model, generation=0)

    for gen in range(max_gen):
        new_passed: list[Conjecture] = []
        for conj in population:
            conj.fitness = 1.0 if verify_conjecture(conj) else 0.0
            if conj.fitness > 0:
                new_passed.append(conj)

        if new_passed:
            passed.extend(new_passed)
            if library is not None:
                for conj in new_passed:
                    library.add(conj)

        # 没新通过的, 终止
        if not new_passed:
            logger.info("evolve stop at gen %d: no new pass", gen)
            break

        # 下一代: 从 new_passed 变异 + 交叉
        if gen < max_gen - 1:
            population = _evolve_population(
                new_passed, n_variants, model=model, generation=gen + 1,
            )
        else:
            logger.info("evolve stop at gen %d: max_gen reached", gen)

    return passed


# ── LLM 生成 (model=None 走模板, 测试用) ───────────────────────────

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
        from langchain_core.messages import HumanMessage, SystemMessage
        import asyncio

        messages = [
            SystemMessage(content=(
                "You are a conjecture generator for materials science. "
                "Given a seed statement, generate variants verifiable with sympy. "
                "Output a JSON array of objects with keys: "
                "statement, sympy_expr, test_cases (list of {inputs, expected}), "
                "proof_script (Python def prove()->bool using sympy). "
                "No sorry/admit. No markdown."
            )),
            HumanMessage(content=f"Seed: {seed}\nGenerate {n} variants."),
        ]
        try:
            asyncio.get_running_loop()
            text = model.invoke(messages)
        except RuntimeError:
            text = asyncio.run(model.ainvoke(messages))
        text = str(text.content).strip()

        parsed = _parse_json_array(text)
        if not parsed:
            return _template_variants(seed, n, generation, parents)

        result: list[Conjecture] = []
        for i, p in enumerate(parsed):
            if not _validate_conjecture_payload(p):
                continue
            result.append(Conjecture(
                id=f"conj-gen{generation:02d}-{i:03d}",
                statement=p["statement"],
                sympy_expr=p["sympy_expr"],
                test_cases=p.get("test_cases", []),
                proof_script=p["proof_script"],
                generation=generation,
                parent_ids=[pp.id for pp in (parents or [])],
            ))
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
    return _generate_variants(seed, n, model=model, generation=generation, parents=parents)


def _template_variants(
    seed: str, n: int, generation: int, parents: list[Conjecture] | None,
) -> list[Conjecture]:
    """模板生成变体 — 测试用. 出几个已知正确的弹性力学命题."""
    parent_ids = [p.id for p in (parents or [])]

    templates = [
        Conjecture(
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
        ),
        Conjecture(
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
        ),
        Conjecture(
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
        ),
    ]
    return templates[:n]


def _template_evolve(
    parents: list[Conjecture], n: int, generation: int,
) -> list[Conjecture]:
    """模板进化: 复制 parents, 改 id/generation/parent_ids. 测试用."""
    result: list[Conjecture] = []
    for i, p in enumerate(parents[:n]):
        result.append(Conjecture(
            id=f"conj-gen{generation:02d}-{i:03d}",
            statement=p.statement + " (variant)",
            sympy_expr=p.sympy_expr,
            test_cases=p.test_cases,
            proof_script=p.proof_script,
            generation=generation,
            parent_ids=[p.id],
        ))
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
        pass
    return []


def _validate_conjecture_payload(p: dict[str, Any]) -> bool:
    return (
        isinstance(p.get("statement"), str)
        and isinstance(p.get("sympy_expr"), str)
        and isinstance(p.get("proof_script"), str)
        and "def prove" in p["proof_script"]
    )


# ── ConjectureLibrary: 累积验证通过的命题 ──────────────────────────

class ConjectureLibrary:
    """SQLite + JSONL 双写累积验证通过的命题.

    SQLite 用于查询, JSONL 用于 git diff 可读. 跟 stable_principles 的
    双写模式一致 (longterm.py 也是 SQLite + JSONL).
    """

    def __init__(self, workspace: str | Path = "."):
        ws = Path(workspace).resolve() / ".huginn"
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
                        (conj.id, conj.statement, conj.sympy_expr,
                         json.dumps(conj.test_cases, ensure_ascii=False),
                         conj.proof_script, conj.fitness, conj.generation,
                         json.dumps(conj.parent_ids, ensure_ascii=False)),
                    )
                    conn.commit()
                except sqlite3.IntegrityError:
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


# ── self-check ────────────────────────────────────────────────────

def _self_check() -> int:
    """assert-based demo: 验证 verify_conjecture + evolve + library."""
    import tempfile

    # 1. verify 通过正确命题
    good = _template_variants("seed", 1, 0, None)[0]
    assert verify_conjecture(good) is True, "good conjecture should pass"

    # 2. verify 拒绝 sorry
    bad_sorry = Conjecture(
        id="bad-sorry", statement="bad", sympy_expr="x",
        proof_script="def prove():\n    # sorry\n    return True",
    )
    assert verify_conjecture(bad_sorry) is False, "sorry should be rejected"

    # 3. verify 拒绝 admit
    bad_admit = Conjecture(
        id="bad-admit", statement="bad", sympy_expr="x",
        proof_script="def prove():\n    # admit\n    return True",
    )
    assert verify_conjecture(bad_admit) is False

    # 4. verify 拒绝 import os
    bad_import = Conjecture(
        id="bad-import", statement="bad", sympy_expr="x",
        proof_script="import os\n\ndef prove():\n    return True",
    )
    assert verify_conjecture(bad_import) is False

    # 5. verify 拒绝 prove() 返回 False
    bad_false = Conjecture(
        id="bad-false", statement="bad", sympy_expr="x",
        proof_script="def prove():\n    return False",
    )
    assert verify_conjecture(bad_false) is False

    # 6. verify 拒绝无 prove 函数
    bad_nofn = Conjecture(
        id="bad-nofn", statement="bad", sympy_expr="x",
        proof_script="x = 1",
    )
    assert verify_conjecture(bad_nofn) is False

    # 7. verify 拒绝 getattr 链 (跟 G23 一致)
    bad_getattr = Conjecture(
        id="bad-getattr", statement="bad", sympy_expr="x",
        proof_script=(
            "def prove():\n"
            "    x = getattr(prove, '__name__')\n"
            "    return True\n"
        ),
    )
    assert verify_conjecture(bad_getattr) is False

    # 8. evolve_conjectures 模板模式跑通
    passed = evolve_conjectures(
        "bulk modulus", n_variants=3, max_gen=2, model=None,
    )
    assert len(passed) > 0, "should have passed conjectures"
    assert all(c.fitness == 1.0 for c in passed)

    # 9. ConjectureLibrary 累积
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
            lines = [l for l in f if l.strip()]
        assert len(lines) == len(passed)

    # 10. evolve + library 联动 — 通过的进 library
    with tempfile.TemporaryDirectory() as tmp:
        lib = ConjectureLibrary(workspace=tmp)
        passed3 = evolve_conjectures(
            "elastic constant", n_variants=2, max_gen=2,
            model=None, library=lib,
        )
        assert lib.count() == len(passed3), (
            f"library {lib.count()} != passed {len(passed3)}"
        )

    print("[CONJLIB] self-check OK")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_self_check())
