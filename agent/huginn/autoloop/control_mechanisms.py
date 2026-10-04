"""控制面机制登记表 —— "控制面预算"的单一权威声明 (控制面的声明层).

为什么需要: ``control_surface_audit.md`` §3 立了原则 —— 新增硬机制必须先过两问,
且每个 ``campaign.control_trace`` 的触发率要可统计, "长期 0 触发或长期误杀 →
删或降". 但这条原则此前**只写在 markdown 里**, 没有代码可强制: 新机制能静默
新增 (改一处 emit 即生效, 无人察觉), 死机制无人回收, 机制总数没有上界. 这正是
"系统太高熵"的结构性来源之一 —— 与配置面 (env_schema) 的病症同型.

本模块把控制面收成**一处声明**: 每个 trace ``name`` 在此登记一次
(类别 / 效果 / 说明). ``tests/test_control_mechanism_governance.py`` 强制:

1. 代码里发射的每个 trace name 必须已在此登记 (拦截静默新增机制);
2. 每条登记必须仍有发射点 (拦截死机制);
3. 机制总数 ``<= CONTROL_MECHANISM_BUDGET`` (只减不增, 逼控制面收敛);
4. ``effect == "stop"`` 只允许出现在 ``STOP_ALLOWLIST`` 里 (把 §3 的
   "科学硬终止只留挂钟与目标达成, 外加一类资源熔断" 从散文变成硬约束).

类别 (见 ``control_surface_audit.md`` §2):
    A 终止类(能结束 run)  B 强制/引导类  C 诚实/证据门(允许硬)  D 观测类
效果 (它当下**实际**改变什么; 不是它曾经的形态):
    stop        结束 run (科学出口或资源熔断)
    block       拦住某个阶段/完成信号, 不改终止
    force       替下一次动作定向 (强制转向)
    advisory    只写 trace + 提示, 不改主流程
    observation 只记录

改本模块后跑 ``pytest tests/test_control_mechanism_governance.py``; 新增机制必须
先登记, 且先过 §3 两问 —— 不允许再出现未经声明的 emit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "Mechanism",
    "MECHANISMS",
    "CONTROL_MECHANISM_BUDGET",
    "STOP_ALLOWLIST",
    "EFFECTS",
    "CATEGORIES",
    "scan_emitted_names",
    "summary",
]


@dataclass(frozen=True)
class Mechanism:
    """单个控制面机制的权威声明."""

    category: str  # A|B|C|D
    effect: str    # stop|block|force|advisory|observation
    note: str


CATEGORIES: frozenset[str] = frozenset({"A", "B", "C", "D"})
EFFECTS: frozenset[str] = frozenset({"stop", "block", "force", "advisory", "observation"})

# 只有这两类可以结束 run (§3): 目标达成 (goal_judge) 与资源熔断 (failure_budget).
# 挂钟预算耗尽不作为 trace 机制登记 (它是循环边界, 不在此面). 新增 stop 必须先改
# 这里, 且必须能回答 §3 两问 —— 默认禁止.
STOP_ALLOWLIST: frozenset[str] = frozenset({"goal_judge", "failure_budget"})


# 单一权威登记表 (按名字排序).
MECHANISMS: dict[str, Mechanism] = {
    # ---- A 终止类 (曾能结束 run; 除两个出口外均已降级) -----------------
    "convergence_advisory": Mechanism(
        "A", "advisory",
        "A1/A2/A3 合流: surprise/exec/rename 三种收敛提示, kind 区分来源",
    ),
    "darwin_stagnation": Mechanism(
        "A", "advisory", "A3 达尔文棘轮连续低增益 → 提示 + trace (原硬停已降)",
    ),
    "belief_convergence": Mechanism(
        "A", "advisory", "A4 belief σ² 收敛 → 提示 + trace (原硬停已降)",
    ),
    "decider_stop": Mechanism(
        "A", "advisory", "A5 LLM decider 想收结: 长程拦截并显式提示, 不替它停",
    ),
    "execute_budget_gate": Mechanism(
        "A", "block", "A6 档位预算门拒绝 execute (仅短程; 长程整段收掉)",
    ),
    "failure_budget": Mechanism(
        "A", "stop", "A8 同类失败连续触顶 → 资源熔断停止 (非科学出口)",
    ),
    "goal_judge": Mechanism(
        "A", "stop", "出口: 独立 GoalJudge 判目标达成 → 允许收结",
    ),
    # ---- B 强制/引导类 ------------------------------------------------
    "stall_as_action": Mechanism(
        "B", "force", "D3 长程停滞且预算未尽 → 强制转向 (默认关)",
    ),
    "pivot_directive": Mechanism(
        "B", "advisory", "B1 越阈值/等价族打转 → 注入强制变异令 (提示层)",
    ),
    "effort_floor": Mechanism(
        "B", "advisory", "B4 努力度下限提示",
    ),
    "curiosity_hint": Mechanism(
        "B", "advisory", "B7 自模型预测不准的簇喂给 hypothesize (默认关)",
    ),
    # ---- C 诚实/证据门 (允许硬) ---------------------------------------
    "goal_acceptance": Mechanism(
        "C", "block", "验收门·证据: 完成声明无有限执行证据 → 拦, 不终止",
    ),
    "goal_metacog_audit": Mechanism(
        "C", "block", "验收门·完成度审计 (推导链/复现证据/证据强度启发式)",
    ),
    "goal_skeptic": Mechanism(
        "C", "block", "验收门·对抗审查: 独立 LLM 拿证据证伪声明, 不过则拦",
    ),
    "report_citation": Mechanism(
        "C", "advisory", "C2 报告 Results 数值无法溯源执行台账 → trace + 标注",
    ),
    "report_discrimination": Mechanism(
        "C", "advisory", "报告判别性纪律: 无对照/分离 → trace + 标注",
    ),
    "report_decisive": Mechanism(
        "C", "advisory", "报告决定性纪律: 只给软语气不给硬口径 → trace + 标注",
    ),
    # ---- D 观测类 -----------------------------------------------------
    "collab_blind_reconstruct": Mechanism(
        "D", "observation", "协作·盲重建 (受控独立观察者, 差分传感器)",
    ),
    "collab_branch_incubator": Mechanism(
        "D", "observation", "协作·N 路隔离探索孵化",
    ),
    "collab_failure_inverter": Mechanism(
        "D", "observation", "协作·失败反推 (含成功路径观测)",
    ),
    "code_lab_timeout": Mechanism(
        "D", "observation", "证据·算力: 沙箱超时饿死的修复尝试计数",
    ),
    "hypothesis_status_writeback": Mechanism(
        "D", "observation", "假设状态回写 (support/refute 证据落图)",
    ),
}


# 控制面预算 (only-shrink): 当前实测机制数. 收敛后下调 —— 新增机制须先过 §3 两问
# 并给出"删掉哪一条"的理由, 不允许悄悄顶上界. 治理测试断言实际数 <= 此值.
CONTROL_MECHANISM_BUDGET = 22


# 发射点的三种形态 (与 _control_trace / _emit_control_trace / engine_observe
# 的直发 campaign.control_trace 对齐). 单一实现, 供治理测试与审计共用.
_TRACE_CALL = re.compile(
    r"_control_trace\(\s*(?:name\s*=\s*)?[\"']([a-z_]+)[\"']"
)
# 直发: "campaign.control_trace" 事件里 name 是字面量 (engine_observe 的写法).
_DIRECT_EMIT = re.compile(
    r"\"campaign\.control_trace\".{0,400}?\"name\"\s*:\s*\"([a-z_]+)\"", re.S
)


def scan_emitted_names(root: Path | None = None) -> dict[str, list[str]]:
    """静态扫描 ``huginn/`` 下所有对照控制面 trace name 的发射点.

    返回 ``{name: [file:line, ...]}``. 只认字面量 name —— 动态名 (变量) 抓不到,
    那属于本模块的盲区, 需人工确认; 现有发射点全为字面量.
    """
    root = root or Path(__file__).resolve().parents[1]  # huginn/
    found: dict[str, list[str]] = {}
    for py in root.rglob("*.py"):
        if "__pycache__" in str(py) or py.resolve() == Path(__file__).resolve():
            continue
        try:
            text = py.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for rx in (_TRACE_CALL, _DIRECT_EMIT):
            for m in rx.finditer(text):
                loc = f"{py.relative_to(root).as_posix()}:{text[: m.start()].count(chr(10)) + 1}"
                found.setdefault(m.group(1), []).append(loc)
    return found


def summary() -> dict[str, object]:
    """控制面现状摘要 (供审计/报告复用)."""
    emitted = scan_emitted_names()
    by_cat: dict[str, int] = {}
    for m in MECHANISMS.values():
        by_cat[m.category] = by_cat.get(m.category, 0) + 1
    return {
        "declared": len(MECHANISMS),
        "budget": CONTROL_MECHANISM_BUDGET,
        "headroom": CONTROL_MECHANISM_BUDGET - len(MECHANISMS),
        "by_category": dict(sorted(by_cat.items())),
        "undeclared_emitted": sorted(set(emitted) - set(MECHANISMS)),
        "dead_declared": sorted(set(MECHANISMS) - set(emitted)),
    }