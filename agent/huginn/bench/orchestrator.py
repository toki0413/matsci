"""通用 benchmark 编排器.

抽象 paperbench 的 while 循环 + 三档分流 + phase-aware budget, 让 5 个
benchmark 适配器共用同一套控制流. 治 4 个架构断层:
  - paperbench 独有 while+三档分流 → 抽象到 Orchestrator, 5 适配器共用
  - PhaseManager ↔ budget 不通 → budget_override 通道
  - 其他 4 适配器无兜底 → Orchestrator 三档分流统一兜底
  - subagent 没用 → TOOL_FILTER 含 subagent_tool (适配器层配)

用法:
    orch = BenchmarkOrchestrator(
        agent=agent, workspace=workspace,
        deliverable_spec=PAPERBENCH_DELIVERABLES,
        max_total_calls=530, timeout=14400, tag="PB",
    )
    final = await orch.run(initial_prompt)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# OpenAI basicagent 风格的 continue message.
# LLM 不调工具时注入, 治 ζ_stop (agent 在 "Now let me write X" 后无 tool_call 被判结束).
CONTINUE_MSG = (
    "Please proceed to the next step using your best judgement. "
    "If you believe you are finished, double check your work to continue "
    "to refine and improve your submission."
)


# ── Deliverable spec ──────────────────────────────────────────

@dataclass
class DeliverableSpec:
    """benchmark 交付物定义: 一组 (名称, glob 模式) 检查项.

    空集 = 完成. ponytail: 不用 YAML plan + ProgressTracker, set 检查够用.
    """
    checks: list[tuple[str, str]]  # (描述, 相对 workspace 的 glob 模式)

    def missing(self, workspace: Path) -> set[str]:
        """返回缺失的 deliverable 描述集, 空集 = 全齐."""
        ws = Path(workspace)
        missing: set[str] = set()
        for name, pattern in self.checks:
            # glob 模式如 "submission/*.py" 或 "submission/reproduce.sh"
            matched = list(ws.glob(pattern))
            if not matched:
                missing.add(name)
        return missing


# 预定义的 benchmark deliverable specs
PAPERBENCH_DELIVERABLES = DeliverableSpec(checks=[
    ("submission/reproduce.sh",   "submission/reproduce.sh"),
    ("submission/*.py",            "submission/*.py"),
    ("submission/outputs/*.json", "submission/outputs/*.json"),
])

RCB_DELIVERABLES = DeliverableSpec(checks=[
    ("report/report.md",     "report/report.md"),
    ("report/images/*.png",  "report/images/*.png"),
])

MLE_DELIVERABLES = DeliverableSpec(checks=[
    ("submission/submission.csv", "submission/submission.csv"),
    ("submission/*.py",           "submission/*.py"),
])

SAB_DELIVERABLES = DeliverableSpec(checks=[
    ("pred_*.py",  "pred_*.py"),
    ("pred_*.txt", "pred_*.txt"),
])

# HLE 无 deliverable 检查 (单题问答), Orchestrator 退化为单次 chat
HLE_DELIVERABLES = DeliverableSpec(checks=[])


# ── 三档分流 prompt ───────────────────────────────────────────

def _triage_prompt(missing: set[str]) -> str:
    """advisory: 提示缺失交付物, 让 LLM 自己决定最小可行实现."""
    return (
        "Submission incomplete. Missing deliverables:\n"
        + "".join(f"  - {m}\n" for m in sorted(missing))
        + "\nYou decide the minimal viable version of each missing file. "
        "Prefer REAL stubs (runnable code that produces actual output, even if "
        "results are weak) over dummy output. If a full implementation is "
        "infeasible in remaining budget, write a working skeleton and document "
        "the gap honestly in report.md."
    )


def _execution_prompt() -> str:
    """闭环纠错: 代码齐全但无 outputs 时, 注入执行+修复指令."""
    return (
        "CRITICAL: You wrote code but NEVER executed it. No output files exist. "
        "Execution = 35% of your grade. Do this NOW:\n"
        "1. Run `bash submission/reproduce.sh` or `python submission/train.py`\n"
        "2. If it fails, read stderr, FIX the error in the .py, re-run\n"
        "3. Repeat until output files exist (up to 10 attempts)\n"
        "4. Save metrics to outputs/metrics.json\n\n"
        "Do NOT write new features. EXECUTE and FIX what you have."
    )


# ── Orchestrator ──────────────────────────────────────────────

class BenchmarkOrchestrator:
    """通用 benchmark 编排器: while 循环 + 三档分流 + phase-aware budget.

    数据流:
        PhaseManager.transition(target)
            └─ proposed_budget = PHASE_BUDGETS[target]
        Orchestrator.run()
            └─ 读 phase_manager.proposed_budget
            └─ chat(budget_override=proposed_budget)
    """

    def __init__(
        self,
        agent: Any,
        workspace: Path | str,
        deliverable_spec: DeliverableSpec,
        max_total_calls: int = 530,
        timeout: int = 3600,
        tag: str = "BENCH",
    ) -> None:
        self.agent = agent
        self.workspace = Path(workspace)
        self.deliverable_spec = deliverable_spec
        self.max_total_calls = max_total_calls
        self.timeout = timeout
        self.tag = tag
        # sanity gate 上次结果. None=deliverable 未齐未检查; dict=已检查 (passed/fail).
        # run() 读 _last_sanity 决定注入 fix_prompt 还是 CONTINUE_MSG.
        self._last_sanity: Any = None

    def _log(self, msg: str) -> None:
        print(f"[{self.tag}] {msg}", flush=True)

    def _is_done(self, calls: int) -> bool:
        """机械式完成判据: (deliverable 全齐 AND sanity gate 过) OR 超 max_total_calls.

        R17: 删 50% budget 下限补丁. 原补丁强制 agent 至少用一半 budget,
        是 SCALECUA task_synthesizer 上线前的临时占位 — 现难度由 task_synthesizer
        按 (paper, complexity_tier) 合成时给定, 不再用 budget 下限强迫消耗.

        [断层6] 修复: deliverable 全齐不等于真完成. pinn 跑 0.117s 假训练,
        final_loss 16 位精度重复也能写出 outputs/*.json. 加 sanity_gate 兜底,
        fail 时 _is_done 返回 False, run() 注入 fix_prompt 继续.
        """
        if calls >= self.max_total_calls:
            return True
        missing = self.deliverable_spec.missing(self.workspace)
        if missing:
            self._last_sanity = None  # deliverable 不齐, 重置避免 stale
            return False
        # deliverable 全齐, 跑 sanity gate. 结果缓存到 self._last_sanity.
        from huginn.runtime.sanity_gate import check_sanity
        self._last_sanity = check_sanity(self.workspace)
        return bool(self._last_sanity["passed"])

    def _get_budget_override(self) -> Any:
        """从 agent 的 phase_manager 读 proposed_budget, 打通 phase→budget.

        phase_budgets 通道对所有 mode 生效. harness 的 max_total_calls
        会在外层截断, budget_override 只影响 agent 单轮 recursion_limit,
        不会让 agent 跑超过 harness 上限. (原 mode 守卫是 phase_budgets
        死配置的根因 — EXECUTION=300 等预算从未生效, 删掉让通道真正通.)
        """
        pm = getattr(self.agent, "_phase_manager", None)
        if pm is None:
            return None
        return getattr(pm, "proposed_budget", None)

    async def run(self, initial_prompt: str) -> str:
        """主循环: while + 三档分流 + sanity gate + budget_override."""
        final = ""
        tool_count = 0
        turn = 0

        try:
            async with asyncio.timeout(self.timeout):
                current_msg = initial_prompt
                while not self._is_done(tool_count):
                    turn += 1
                    made_tool_call = False
                    budget = self._get_budget_override()
                    # [断层8] tool_calls 不可观测: 每轮注入累计 budget hint,
                    # 让 agent 自救 (低预算时优先做高价值动作). ponytail: 一行拼接.
                    budget_hint = (
                        f"\n\n[budget] tool_calls: {tool_count}/{self.max_total_calls} "
                        f"({self.max_total_calls - tool_count} remaining)"
                    )
                    async for chunk in self.agent.chat(current_msg + budget_hint, budget_override=budget):
                        msgs = chunk.get("messages", []) if isinstance(chunk, dict) else []
                        if msgs:
                            last = msgs[-1]
                            content = getattr(last, "content", "")
                            if content:
                                final = str(content)
                            msg_type = getattr(last, "type", "")
                            if msg_type == "tool":
                                tool_count += 1
                                made_tool_call = True
                                tool_name = getattr(last, "name", "unknown")
                                self._log(f"tool #{tool_count}: {tool_name}")
                            elif msg_type == "ai" and content:
                                if getattr(last, "tool_calls", None):
                                    made_tool_call = True
                                preview = content[:200].replace("\n", " ")
                                self._log(f"AI: {preview}...")

                    # 四档分流: sanity_fail > triage > continue
                    # sanity gate fail 时 _is_done 返回 False, 优先注入 fix_prompt
                    if self._last_sanity and not self._last_sanity["passed"]:
                        current_msg = self._last_sanity["fix_prompt"]
                        self._log(f"Sanity FAIL: {self._last_sanity['reason']}")
                    elif not made_tool_call and not self._is_done(tool_count):
                        missing = self.deliverable_spec.missing(self.workspace)
                        if not missing:
                            # 全齐但 agent 自停 → 继续优化
                            current_msg = CONTINUE_MSG
                        elif self._has_code_no_output(missing):
                            current_msg = _execution_prompt()
                            self._log("Triage: code ready, no outputs -> execution loop")
                        else:
                            current_msg = _triage_prompt(missing)
                            self._log(f"Triage: missing {len(missing)} -> minimal skeleton")
                    else:
                        current_msg = CONTINUE_MSG
        except asyncio.TimeoutError:
            final = f"[TIMEOUT after {self.timeout}s]"

        self._log(f"Agent finished. Tool calls: {tool_count}, turns: {turn}")
        return final

    def _has_code_no_output(self, missing: set[str]) -> bool:
        """检查是否"代码齐全但无 outputs"模式."""
        sub = self.workspace / "submission"
        has_code = (sub / "reproduce.sh").exists() and bool(list(sub.glob("*.py")))
        has_output_missing = any("outputs" in m for m in missing)
        return has_code and has_output_missing


# ── self-check ────────────────────────────────────────────────

def _self_check() -> int:
    """assert-based demo: 验证 DeliverableSpec / _triage_prompt / _execution_prompt."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp)

        # 场景 1: 空 workspace -> 缺 3 件
        m = PAPERBENCH_DELIVERABLES.missing(ws)
        assert m == {"submission/reproduce.sh", "submission/*.py", "submission/outputs/*.json"}, m
        assert "Missing deliverables" in _triage_prompt(m)

        # 场景 2: 部分 (reproduce.sh + .py, 无 outputs) -> 缺 1 件
        sub = ws / "submission"; sub.mkdir()
        (sub / "reproduce.sh").write_text("python train.py")
        (sub / "train.py").write_text("print('hi')")
        m = PAPERBENCH_DELIVERABLES.missing(ws)
        assert m == {"submission/outputs/*.json"}, m
        assert "NEVER executed" in _execution_prompt()

        # 场景 3: 全齐 -> 空集
        (sub / "outputs").mkdir()
        (sub / "outputs" / "loss.json").write_text('{"loss":[1.0]}')
        m = PAPERBENCH_DELIVERABLES.missing(ws)
        assert m == set(), m

        # 场景 4: HLE 无 deliverable -> 永远空集 (Orchestrator 退化为单次 chat)
        assert HLE_DELIVERABLES.missing(ws) == set()

        # 场景 5: [断层6] sanity gate 集成 — deliverable 全齐但 sanity fail
        # 验证 _is_done 返回 False, _last_sanity 缓存 fix_prompt 可读
        import json as _json
        out = sub / "outputs"
        # 写两个 final_loss 16 位精度重复的 json (sanity_gate test 3 同款)
        (out / "r1.json").write_text(_json.dumps({"final_loss": 3.306041717529297, "training_time": 100.0}))
        (out / "r2.json").write_text(_json.dumps({"final_loss": 3.306041717529297, "training_time": 200.0}))
        orch = BenchmarkOrchestrator(
            agent=None, workspace=ws,
            deliverable_spec=PAPERBENCH_DELIVERABLES,
            max_total_calls=530, tag="TEST",
        )
        assert orch._is_done(0) is False, "sanity fail 时 _is_done 应返回 False"
        assert orch._last_sanity is not None
        assert orch._last_sanity["passed"] is False
        assert "float_dedup" in orch._last_sanity["reason"], orch._last_sanity["reason"]
        assert "SANITY GATE FAIL" in orch._last_sanity["fix_prompt"]

        # 场景 6: sanity pass (真实 loss + 训练时间 + 单调曲线)
        (out / "r1.json").write_text(_json.dumps({
            "final_loss": 0.5, "training_time": 100.0, "loss_curve": [1.0, 0.8, 0.6, 0.5],
        }))
        (out / "r2.json").write_text(_json.dumps({
            "final_loss": 0.3, "training_time": 200.0, "loss_curve": [1.0, 0.7, 0.5, 0.3],
        }))
        orch2 = BenchmarkOrchestrator(
            agent=None, workspace=ws,
            deliverable_spec=PAPERBENCH_DELIVERABLES,
            max_total_calls=530, tag="TEST",
        )
        assert orch2._is_done(0) is True, "sanity pass 时 _is_done 应返回 True"
        assert orch2._last_sanity["passed"] is True

    print("[ORCH] self-check OK")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_self_check())
