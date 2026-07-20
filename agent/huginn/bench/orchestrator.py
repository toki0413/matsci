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

    def _log(self, msg: str) -> None:
        print(f"[{self.tag}] {msg}", flush=True)

    def _is_done(self, calls: int) -> bool:
        """机械式完成判据: deliverable 全齐 OR 超 max_total_calls.

        R17: 删 50% budget 下限补丁. 原补丁强制 agent 至少用一半 budget,
        是 SCALECUA task_synthesizer 上线前的临时占位 — 现难度由 task_synthesizer
        按 (paper, complexity_tier) 合成时给定, 不再用 budget 下限强迫消耗.
        """
        if calls >= self.max_total_calls:
            return True
        # deliverable 全齐即完成
        if not self.deliverable_spec.missing(self.workspace):
            return True
        return False

    def _get_budget_override(self) -> Any:
        """从 agent 的 phase_manager 读 proposed_budget, 打通 phase→budget.

        只在 research mode 时传 budget_override. chat mode (默认) 用 agent
        构造时的 max_tool_calls, 避免 OPEN phase 的 500 calls 覆盖适配器的
        max_total_calls 设置.
        """
        pm = getattr(self.agent, "_phase_manager", None)
        if pm is None:
            return None
        mode = getattr(self.agent, "_mode", "chat")
        if mode != "research":
            return None
        return getattr(pm, "proposed_budget", None)

    async def run(self, initial_prompt: str) -> str:
        """主循环: while + 三档分流 + budget_override."""
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
                    async for chunk in self.agent.chat(current_msg, budget_override=budget):
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

                    # 三档分流: agent 无 tool_call 且未完成时注入明确指令
                    if not made_tool_call and not self._is_done(tool_count):
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

    print("[ORCH] self-check OK")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_self_check())
