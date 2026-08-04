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
    # P0-3: 删 pred_*.txt — SAB 任务从未要求, 之前强制 agent 造无意义文件.
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
        # P1-C8: 可观测性字段, run() 结束后外部适配器读取写 meta
        self.tool_calls_used: int = 0
        self.turns_used: int = 0
        # C8 补: context overflow / compaction / crash 追踪
        self.context_overflow_count: int = 0
        self.compaction_count: int = 0
        self.crash_traceback: str | None = None
        # C2: checkpoint 尺寸上限 (100MB, roadmap 完成判据 5). run() 结束后填,
        # 适配器读 meta 时一并落盘. VACUUM 触发标志记录到 vacuum_triggered.
        self.checkpoint_size_mb: float = 0.0
        self.vacuum_triggered: bool = False

    def _log(self, msg: str) -> None:
        print(f"[{self.tag}] {msg}", flush=True)

    def _enforce_checkpoint_size_limit(self) -> None:
        """C2: benchmark 结束后检查 checkpoint 文件大小, 超 100MB 触发 VACUUM.

        SQLite 删除数据后文件不会自动缩小 (RemoveMessage 只删 row, 页不回收),
        所以 _trim_checkpointer_messages 的真修剪在文件层面看不出效果. VACUUM
        重建数据库文件, 是唯一能让文件真正缩小的手段.

        ponytail: 只在 run() 结束后跑 VACUUM — 运行中锁表会卡 graph.
          ceiling: VACUUM 需要额外磁盘空间临时存放新文件, 磁盘满会失败.
          升级路径: 改用 auto_vacuum = INCREMENTAL, 在线增量回收.
        """
        import os as _os
        _path = _os.environ.get("HUGINN_CHECKPOINTER_PATH")
        if not _path or not _os.path.exists(_path):
            return
        _size = _os.path.getsize(_path)
        self.checkpoint_size_mb = round(_size / (1024 * 1024), 2)
        if self.checkpoint_size_mb <= 100:
            return
        self._log(
            f"Checkpoint {self.checkpoint_size_mb}MB > 100MB limit, VACUUM..."
        )
        _ckpt = getattr(self.agent, "checkpointer", None)
        _conn = getattr(_ckpt, "conn", None) or getattr(_ckpt, "_conn", None)
        if _conn is None:
            self._log("VACUUM skipped: no sqlite conn on checkpointer")
            return
        try:
            _conn.execute("VACUUM")
            self.vacuum_triggered = True
            self.checkpoint_size_mb = round(
                _os.path.getsize(_path) / (1024 * 1024), 2
            )
            self._log(f"VACUUM done: {self.checkpoint_size_mb}MB")
        except Exception as _e:
            self._log(f"VACUUM failed: {_e}")

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
                        # C8: 检测 compaction 事件 (chunk 里带 compaction 标记)
                        if isinstance(chunk, dict) and chunk.get("compaction_triggered"):
                            self.compaction_count += 1

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
        except Exception as exc:
            # C8: 捕获 crash traceback + 检测 context overflow
            import traceback as _tb
            self.crash_traceback = _tb.format_exc()
            try:
                from huginn.agent.streaming import _is_context_overflow
                if _is_context_overflow(exc):
                    self.context_overflow_count += 1
                    final = f"[CONTEXT_OVERFLOW: {exc}]"
                    self._log(f"Context overflow detected: {exc}")
                else:
                    final = f"[CRASH: {type(exc).__name__}: {exc}]"
                    self._log(f"Crash: {self.crash_traceback[-500:]}")
            except Exception:
                final = f"[CRASH: {type(exc).__name__}: {exc}]"
                self._log(f"Crash: {self.crash_traceback[-500:]}")

        self._log(f"Agent finished. Tool calls: {tool_count}, turns: {turn}")
        # P1-C8: 暴露可观测性字段给外部适配器写 meta
        self.tool_calls_used = tool_count
        self.turns_used = turn
        # C2: run 结束后检查 checkpoint 尺寸, 超 100MB VACUUM (roadmap 判据 5)
        try:
            self._enforce_checkpoint_size_limit()
        except Exception as _e:
            self._log(f"checkpoint size enforcement failed: {_e}")
        return final

    def _has_code_no_output(self, missing: set[str]) -> bool:
        """检查是否"代码齐全但无 outputs"模式 — 通用判定, 不绑定目录结构.

        之前硬编码 submission/reproduce.sh + *.py, SAB/RCB 形态永远匹配不到.
        改为按 DeliverableSpec 自身的 pattern 分类: 代码类 (.py/.sh) 全齐 +
        输出类 (.json/.png/.csv/report.md/outputs/) 缺失 = 执行闭环断裂.
        """
        checks = self.deliverable_spec.checks
        code_pats = [p for _, p in checks if p.endswith((".py", ".sh"))]
        output_pats = [
            p for _, p in checks
            if any(k in p for k in ("outputs", ".json", ".png", ".csv", "report.md"))
        ]
        if not code_pats or not output_pats:
            return False
        ws = Path(self.workspace)
        has_code = any(list(ws.glob(p)) for p in code_pats)
        has_output_missing = any(
            not list(ws.glob(p)) for p in output_pats
        )
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


def _c2_self_check() -> int:
    """C2 self-check: 验证 checkpoint 尺寸上限 + VACUUM 触发逻辑.

    覆盖:
    1. checkpoint_size_mb / vacuum_triggered 字段存在且初值为 0/False
    2. 小文件 (<100MB) 不触发 VACUUM
    3. 大文件 (>100MB) 且有 conn 时触发 VACUUM
    4. 无 conn 时优雅降级 (不崩, 只 log)

    ponytail: 临时 SQLite 文件模拟 checkpointer. ceiling: 不验真 VACUUM
      在 SqliteSaver 上的副作用, 只验调用路径.
    """
    import gc
    import os
    import sqlite3
    import tempfile

    # 假 agent: orchestrator 从 agent.checkpointer 拿 checkpointer, 再从
    # checkpointer.conn 拿 sqlite 连接. 两层 getattr, 测试时用一个对象兼任.
    class _FakeAgent:
        def __init__(self, conn):
            # checkpointer 指向自己, conn 直接暴露 — 模拟 SqliteSaver 的结构
            self.checkpointer = self
            self.conn = conn

    # 1. 字段初值 (无 conn)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        orch = BenchmarkOrchestrator(
            agent=_FakeAgent(None), workspace=tmp,
            deliverable_spec=HLE_DELIVERABLES, tag="C2",
        )
        assert orch.checkpoint_size_mb == 0.0
        assert orch.vacuum_triggered is False
        print("[CHECK C2.1] fields initialized OK")

        # 2. 小文件 (<100MB) 不触发 VACUUM
        small = os.path.join(tmp, "small.sqlite")
        with open(small, "wb") as f:
            f.write(b"\x00" * (1024 * 1024))  # 1MB
        os.environ["HUGINN_CHECKPOINTER_PATH"] = small
        orch._enforce_checkpoint_size_limit()
        assert orch.checkpoint_size_mb >= 1.0, orch.checkpoint_size_mb
        assert orch.checkpoint_size_mb < 100, orch.checkpoint_size_mb
        assert orch.vacuum_triggered is False
        print(f"[CHECK C2.2] small file ({orch.checkpoint_size_mb}MB) no VACUUM OK")

    # 3. 大文件 (>100MB) 触发 VACUUM — 独立临时目录避免 Windows 文件锁
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        big = os.path.join(tmp, "big.sqlite")
        conn = sqlite3.connect(big)
        conn.execute("CREATE TABLE t (x BLOB)")
        # 写 101MB 数据 (1 行 1MB × 101 行)
        chunk = b"x" * (1024 * 1024)
        for _ in range(101):
            conn.execute("INSERT INTO t VALUES (?)", (chunk,))
        conn.commit()
        _size_before = os.path.getsize(big) / (1024 * 1024)
        assert _size_before > 100, f"setup failed: {_size_before}MB"

        orch2 = BenchmarkOrchestrator(
            agent=_FakeAgent(conn), workspace=tmp,
            deliverable_spec=HLE_DELIVERABLES, tag="C2",
        )
        os.environ["HUGINN_CHECKPOINTER_PATH"] = big
        orch2._enforce_checkpoint_size_limit()
        assert orch2.vacuum_triggered is True, "大文件应触发 VACUUM"
        print(
            f"[CHECK C2.3] big file ({_size_before:.1f}MB -> "
            f"{orch2.checkpoint_size_mb}MB) VACUUM triggered OK"
        )
        conn.close()
        gc.collect()  # 释放 Windows 文件句柄

    # 4. 无 conn 优雅降级
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        big2 = os.path.join(tmp, "big2.sqlite")
        with open(big2, "wb") as f:
            f.write(b"\x00" * (101 * 1024 * 1024))  # 101MB 假文件
        os.environ["HUGINN_CHECKPOINTER_PATH"] = big2
        orch3 = BenchmarkOrchestrator(
            agent=_FakeAgent(None), workspace=tmp,
            deliverable_spec=HLE_DELIVERABLES, tag="C2",
        )
        orch3._enforce_checkpoint_size_limit()
        assert orch3.checkpoint_size_mb > 100
        assert orch3.vacuum_triggered is False
        print("[CHECK C2.4] no-conn graceful degradation OK")

    # 清环境变量
    os.environ.pop("HUGINN_CHECKPOINTER_PATH", None)
    print("[CHECK C2] ALL ASSERTS PASSED")
    return 0


if __name__ == "__main__":
    import sys
    if "--self-check-c2" in sys.argv:
        sys.exit(_c2_self_check())
    sys.exit(_self_check())
