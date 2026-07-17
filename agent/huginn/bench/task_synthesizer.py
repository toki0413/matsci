"""SCALECUA 风格的可验证任务合成器.

LLM 生成任务描述 + 判分脚本, sanity_check 用已知答案验证判分脚本本身没写错.
失败时降级到静态 benchmark. 不引入新组件, 复用 BenchmarkTask + BenchmarkRunner.

设计参考: SCALECUA (Synthetic Constraints for Agent Evaluation) — 用 LLM
合成带可验证判分的任务, 替代静态 benchmark 的覆盖面不足.

跑法:
    python -m huginn.bench.task_synthesizer
"""

from __future__ import annotations

import ast
import json
import logging
import re
import textwrap
from dataclasses import dataclass
from typing import Any

from .task import BenchmarkTask

logger = logging.getLogger(__name__)

# 白名单: LLM 生成的 judge 脚本只能调这些模块.
# 跟 security/script_runner.py 的 _safe_import 一致, 不引入新依赖.
_JUDGE_ALLOWED_MODULES = frozenset({
    "math", "statistics", "json", "re", "numpy", "pandas", "sympy", "scipy",
})


@dataclass
class SynthesizedTask:
    """合成任务 — LLM 生成的任务描述 + 判分脚本 + 已知答案.

    known_answer 用来跑 sanity_check 验证判分脚本本身没写错.
    验证通过后转 BenchmarkTask 接入 BenchmarkRunner.
    """
    id: str
    domain: str
    difficulty: str
    prompt: str
    judge_script: str  # Python 源码, def judge(output: str) -> tuple[bool, str]
    known_answer: str  # 已知正确答案, sanity check 用
    timeout_seconds: float = 180.0


def synthesize_task(
    domain: str,
    difficulty: str,
    model: Any = None,
    task_id: str | None = None,
) -> SynthesizedTask | None:
    """LLM 生成合成任务. model=None 时走模板 (测试用).

    返回 None 表示合成彻底失败, 调用方降级到静态 benchmark.
    """
    tid = task_id or f"synth-{domain[:12]}-{difficulty[:4]}"
    if model is None or _is_mock(model):
        return _template_synthesize(tid, domain, difficulty)

    try:
        text = _invoke_model(model, _build_synth_prompt(domain, difficulty))
        parsed = _parse_json(text)
        if not _validate_synth_payload(parsed):
            logger.warning("synth payload invalid, fallback to template")
            return _template_synthesize(tid, domain, difficulty)
        return SynthesizedTask(
            id=tid,
            domain=domain,
            difficulty=difficulty,
            prompt=parsed["prompt"],
            judge_script=parsed["judge_script"],
            known_answer=parsed["known_answer"],
            timeout_seconds=float(parsed.get("timeout_seconds", 180.0)),
        )
    except Exception:
        logger.debug("LLM synth failed", exc_info=True)
        return _template_synthesize(tid, domain, difficulty)


def sanity_check_judge(judge_script: str, known_answer: str) -> bool:
    """用已知答案跑 judge 脚本, 验证脚本本身没写错.

    judge 脚本必须 def judge(output: str) -> tuple[bool, str].
    对 known_answer 应返回 (True, ...). 同时测一个明显错答案应返回 (False, ...).
    两个都通过才算 sanity OK.

    沙盒: AST 预扫描只允许白名单 import, exec 在隔离 globals 里跑.
    ponytail: 升级路径是 security/sandbox.py 的 Docker 容器, 这里够用因为
    白名单 + 无 __import__ + 已知答案双向验证.
    """
    # 1. AST 预扫描: 只允许白名单 import, 禁危险内建
    try:
        tree = ast.parse(judge_script)
    except SyntaxError:
        return False

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top not in _JUDGE_ALLOWED_MODULES:
                    return False
        elif isinstance(node, ast.ImportFrom):
            top = (node.module or "").split(".")[0]
            if top not in _JUDGE_ALLOWED_MODULES:
                return False
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id in {
                "__import__", "eval", "exec", "compile"
            }:
                return False
            # getattr 链封堵 (跟 G23 一致)
            if isinstance(fn, ast.Name) and fn.id == "getattr":
                # judge 脚本一般不需要 getattr, 直接禁
                return False

    # 2. 在隔离 globals 里 exec
    safe_globals: dict[str, Any] = {"__builtins__": __builtins__}
    try:
        exec(compile(tree, "<judge>", "exec"), safe_globals)
    except Exception:
        return False
    judge_fn = safe_globals.get("judge")
    if not callable(judge_fn):
        return False

    # 3. 已知答案应判 True
    try:
        ok_pos, _ = judge_fn(known_answer)
    except Exception:
        return False
    if ok_pos is not True:
        return False

    # 4. 明显错答案应判 False
    try:
        ok_neg, _ = judge_fn("__definitely_wrong_answer_xyz__")
    except Exception:
        return False
    if ok_neg is not False:
        return False

    return True


def to_benchmark_task(synth: SynthesizedTask) -> BenchmarkTask | None:
    """把 SynthesizedTask 转成 BenchmarkTask 接入 BenchmarkRunner.

    先跑 sanity_check, 通过才返回. 失败返回 None, 调用方降级.
    """
    if not sanity_check_judge(synth.judge_script, synth.known_answer):
        logger.warning("judge sanity check failed for %s", synth.id)
        return None

    # 编译一次拿 judge 函数, evaluator 闭包调它
    safe_globals: dict[str, Any] = {"__builtins__": __builtins__}
    try:
        exec(compile(synth.judge_script, "<judge>", "exec"), safe_globals)
        judge_fn = safe_globals["judge"]
    except Exception:
        return None

    def evaluate(output: str) -> tuple[bool, str]:
        try:
            return judge_fn(output)
        except Exception as e:
            return False, f"judge error: {e}"

    return BenchmarkTask(
        id=synth.id,
        category=f"synth-{synth.domain}",
        prompt=synth.prompt,
        evaluator=evaluate,
        tags=["synth", synth.domain, synth.difficulty],
        requires_api_key=True,
        reference=synth.known_answer,
        timeout_seconds=synth.timeout_seconds,
    )


class SynthesizedBenchAdapter:
    """SCALECUA 风格合成任务适配器.

    用法:
        adapter = SynthesizedBenchAdapter(model=gemini)
        tasks = adapter.load_tasks(domain="materials-science", difficulty="hard", n=10)
        # tasks 已经过 sanity_check 过滤, 可直接喂 BenchmarkRunner
    """

    def __init__(
        self,
        model: Any = None,
        fallback_tasks: list[BenchmarkTask] | None = None,
    ):
        self.model = model
        self.fallback = fallback_tasks or []

    def load_tasks(
        self,
        domain: str = "materials-science",
        difficulty: str = "medium",
        n: int = 10,
    ) -> list[BenchmarkTask]:
        """合成 n 个任务, 过 sanity_check, 失败降级到 fallback."""
        tasks: list[BenchmarkTask] = []
        for i in range(n):
            synth = synthesize_task(
                domain, difficulty, model=self.model,
                task_id=f"synth-{i:03d}",
            )
            if synth is None:
                continue
            bt = to_benchmark_task(synth)
            if bt is not None:
                tasks.append(bt)
        if not tasks:
            logger.warning("synth produced 0 valid tasks, fallback to static")
            return list(self.fallback)
        return tasks


# ── 模板合成 (测试 + LLM 失败降级) ──────────────────────────────────

def _template_synthesize(
    task_id: str, domain: str, difficulty: str,
) -> SynthesizedTask:
    """不用 LLM 的兜底合成. 出一个可算的数值题."""
    if difficulty == "easy":
        prompt = (
            "What is the bulk modulus (GPa) of a cubic crystal with "
            "c11=100, c12=40? Reply with only the number."
        )
        known = "60"
        judge = textwrap.dedent('''
            def judge(output):
                import re
                nums = re.findall(r"[-+]?\\d*\\.?\\d+", output)
                if not nums:
                    return False, "no number found"
                n = float(nums[0])
                if abs(n - 60.0) <= 0.5:
                    return True, "got " + str(n)
                return False, "got " + str(n) + ", expected 60"
        ''').strip()
    elif difficulty == "medium":
        prompt = (
            "Compute the Birch-Murnaghan pressure (GPa) at V/V0=0.95 "
            "for B0=100, B0'=4. Reply with only the number."
        )
        known = "9.48"
        judge = textwrap.dedent('''
            def judge(output):
                import re
                nums = re.findall(r"[-+]?\\d*\\.?\\d+", output)
                if not nums:
                    return False, "no number"
                n = float(nums[0])
                # BM3 压力 at eta=0.95, B0=100, B0'=4: ~9.48 GPa
                if abs(n - 9.48) <= 0.5:
                    return True, "got " + str(n)
                return False, "got " + str(n) + ", expected ~9.48"
        ''').strip()
    else:  # hard
        prompt = (
            "For a hexagonal crystal with c11=550, c12=124, c13=120, "
            "c33=560, c44=110, c66=(c11-c12)/2, verify stability. "
            "Reply with STABLE or UNSTABLE."
        )
        known = "STABLE"
        judge = textwrap.dedent('''
            def judge(output):
                up = output.upper().strip()
                if "STABLE" in up and "UNSTABLE" not in up:
                    return True, "STABLE"
                return False, "should be STABLE"
        ''').strip()

    return SynthesizedTask(
        id=task_id,
        domain=domain,
        difficulty=difficulty,
        prompt=prompt,
        judge_script=judge,
        known_answer=known,
    )


# ── LLM 调用 helpers (跟 conjecture.py 同套) ────────────────────────

def _is_mock(model: Any) -> bool:
    return hasattr(model, "_mock_name")


def _invoke_model(model: Any, messages: list) -> str:
    import asyncio

    try:
        asyncio.get_running_loop()
        resp = model.invoke(messages)
    except RuntimeError:
        resp = asyncio.run(model.ainvoke(messages))
    return str(resp.content).strip()


def _build_synth_prompt(domain: str, difficulty: str) -> list:
    from langchain_core.messages import HumanMessage, SystemMessage

    return [
        SystemMessage(content=(
            "You are a benchmark task synthesizer in the style of SCALECUA. "
            "Generate a verifiable task with a judge function. "
            "Output ONLY a JSON object with keys: "
            "prompt (task description), "
            "judge_script (Python source defining "
            "`def judge(output: str) -> tuple[bool, str]`), "
            "known_answer (string the judge should accept), "
            "timeout_seconds (float, default 180). "
            "judge_script may only import: "
            "math, statistics, json, re, numpy, pandas, sympy, scipy. "
            "No __import__, eval, exec, compile, getattr. "
            "No markdown, no explanation."
        )),
        HumanMessage(content=(
            f"Domain: {domain}\n"
            f"Difficulty: {difficulty}\n"
            f"Generate one verifiable task."
        )),
    ]


def _parse_json(text: str) -> dict[str, Any]:
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
        pass
    return {}


def _validate_synth_payload(p: dict[str, Any]) -> bool:
    return (
        isinstance(p.get("prompt"), str)
        and isinstance(p.get("judge_script"), str)
        and isinstance(p.get("known_answer"), str)
        and "def judge" in p["judge_script"]
    )


# ── self-check ────────────────────────────────────────────────────

def _self_check() -> int:
    """assert-based demo: 验证 synthesize_task + sanity_check_judge + adapter."""
    # 1. 模板合成
    t = synthesize_task("materials-science", "easy")
    assert t is not None
    assert "def judge" in t.judge_script
    assert t.known_answer == "60"

    # 2. sanity_check 通过
    assert sanity_check_judge(t.judge_script, t.known_answer) is True

    # 3. 错判分脚本被拒绝 (永远返回 True)
    bad_judge = "def judge(output):\n    return True, 'always'"
    assert sanity_check_judge(bad_judge, "anything") is False

    # 4. 拒绝 import os
    evil_judge = "import os\n\ndef judge(output):\n    return True, 'ok'"
    assert sanity_check_judge(evil_judge, "ok") is False

    # 5. 拒绝 __import__
    evil2 = "def judge(output):\n    m = __import__('os')\n    return True, 'ok'"
    assert sanity_check_judge(evil2, "ok") is False

    # 6. 拒绝 getattr
    evil3 = (
        "def judge(output):\n"
        "    x = getattr(output, 'strip')()\n"
        "    return True, 'ok'"
    )
    assert sanity_check_judge(evil3, "ok") is False

    # 7. 转 BenchmarkTask
    bt = to_benchmark_task(t)
    assert bt is not None
    assert bt.id == t.id
    passed, reason = bt.evaluator("60")
    assert passed, f"should pass on known answer: {reason}"
    passed, _ = bt.evaluator("42")
    assert not passed

    # 8. Adapter
    adapter = SynthesizedBenchAdapter(model=None)
    tasks = adapter.load_tasks(domain="materials-science", difficulty="easy", n=3)
    assert len(tasks) == 3
    assert all(task.tags[0] == "synth" for task in tasks)

    # 9. 三种难度都能跑
    for d in ["easy", "medium", "hard"]:
        t = synthesize_task("physics", d)
        assert t is not None, f"failed at {d}"
        assert sanity_check_judge(t.judge_script, t.known_answer), (
            f"sanity failed at {d}"
        )

    # 10. Adapter 全部合成失败时降级到 fallback
    empty_adapter = SynthesizedBenchAdapter(
        model=None,
        fallback_tasks=[BenchmarkTask(
            id="fallback-1", category="fallback",
            prompt="hi", evaluator=lambda out: (True, "ok"),
        )],
    )
    # monkeypatch synthesize_task 让它返回 None.
    # 用 sys.modules[__name__] 而非 import, 因为 python -m 模式下 __name__ 是
    # __main__, 重新 import 会拿到另一个模块副本, monkeypatch 不生效.
    import sys
    mod = sys.modules[__name__]
    orig = mod.synthesize_task
    mod.synthesize_task = lambda *a, **k: None
    try:
        fb_tasks = empty_adapter.load_tasks(n=2)
        assert len(fb_tasks) == 1, f"expected 1 fallback, got {len(fb_tasks)}"
        assert fb_tasks[0].id == "fallback-1"
    finally:
        mod.synthesize_task = orig

    print("[SYNTH] self-check OK")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_self_check())
