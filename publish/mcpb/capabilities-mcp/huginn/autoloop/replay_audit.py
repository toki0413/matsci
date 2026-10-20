"""离线重放审计 — 不跑循环, 用记录轨迹一次性枚举"无进展但未出口"的空转路径.

问题: 长程自主循环的空转症状一直靠"跑一轮(~1h) → 暴露一个症状 → 补一个补丁"
来发现, 又慢又贵, 于是退化成"跑一次修一次". 本模块把历史 run 的**记录轨迹**
(episodic shard + 假设图 + run.log) 重放一遍, 用**真实守卫函数**逐轮判定"有没有
进展", 并核对每个无进展轮是否**可观测** (advisory trace 落盘; A2 前则核对是否触发
可终止出口). 秒级、确定性、不调 LLM.

判据 (进展不变量): 每一轮必须满足
    progress(轮) = ∃ { 假设图新增节点, 实质且非换名的新假设, 新执行指纹 }
无进展的轮必须伴随一个**可终止出口** (收敛/结题/停机); 仅"提示/重定向/reset"
这类软动作**不算出口** —— 这正是 run47/run49 无限打转的缺口.

A2 更新 (控制面审计 §3/§A2): 收敛/换名债务/结题类硬终止已**降级为 advisory**
(只提示 + trace, 不再 `should_stop`), 线上终止出口只保留**挂钟预算**与**目标达成**.
故"无进展⇒必须终止"不再是线上不变量. 本工具据此改口径: 出口体检只核对
"无进展轮是否**可观测**(advisory trace 是否落盘)", 换名债务重放给的是 **A2 前的
反事实** (若仍终止会在第几次), 不再当作"必然终止"的证明.

用法:
    PYTHONPATH=/workspace/agent python -m huginn.autoloop.replay_audit <run_dir> [...]
    # 加 --llm 启用**语义重放** (对假设图判, 抓同义改写; 需可用 model, 每节点一次调用)
    PYTHONPATH=/workspace/agent python -m huginn.autoloop.replay_audit --llm <run_dir> [...]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from huginn.autoloop.hypothesis_loop import _is_substantive_statement, _statement_key
from huginn.metacog.equivalence_auditor import EquivalenceAuditor

# 可终止出口 (soft 动作不算出口): 键 = run.log 里出现的标记
TERMINAL_MARKERS = ("exec convergence", "exec-convergence", "rename debt",
                    "goal completed", "conclude")
# 软动作 (会改变方向但不终止, 单独统计)
SOFT_MARKERS = ("renamed-reduction", "counterexample hunt", "repeat execution")
# 与 cognitive_loop._RENAME_DEBT_LIMIT 同步 (同环境变量/同默认), 供离线重放给出
# "A2 前该轨迹会在第几次换名越界"的**反事实** (A2 后越界只写 advisory trace, 不终止).
_DEBT_LIMIT = int(os.environ.get("HUGINN_RENAME_DEBT_LIMIT", "8"))


def _iso_epoch(s: str) -> float | None:
    try:
        return datetime.strptime(s[:19], "%Y-%m-%dT%H:%M:%S").replace(
            tzinfo=UTC).timestamp()
    except Exception:  # — 原因: 时间戳格式不可解析, 视为无时间 (该节点不参与排序)
        return None


# ── 轨迹加载 ──────────────────────────────────────────────────────────

def load_episodic(run_dir: str) -> list[dict]:
    """读所有 episodic shard, 按时间排序返回 [{ts, action, statement, ...}]."""
    rows: list[dict] = []
    for f in sorted(Path(run_dir).glob(".huginn/memory/episodic/*/shard_*.jsonl")):
        for line in f.read_text(encoding="utf-8", errors="ignore").splitlines():
            try:
                d = json.loads(line)
            except Exception:  # — 原因: 坏 jsonl 行跳过, 不让单行污染整份轨迹
                continue
            e = d.get("entry") or {}
            if not e:
                continue
            rows.append({"ts": float(d.get("ts") or 0.0), **e})
    rows.sort(key=lambda r: r["ts"])
    return rows


def cycles_from_episodic(rows: list[dict]) -> list[dict]:
    """按 hypothesize 切轮: 每轮 = 一段 hypothesize→...→learn."""
    cycles: list[dict] = []
    cur: dict | None = None
    for r in rows:
        if r.get("action") == "hypothesize" or cur is None:
            if cur:
                cycles.append(cur)
            cur = {"ts": r["ts"], "statement": "", "actions": [], "surprise": []}
        cur["actions"].append(r.get("action"))
        if r.get("hypothesis") and not cur["statement"]:
            cur["statement"] = r["hypothesis"]
        if r.get("surprise") is not None:
            cur["surprise"].append(r["surprise"])
    if cur:
        cycles.append(cur)
    return cycles


def load_graph_nodes(run_dir: str) -> list[tuple[float | None, str]]:
    nodes: list[tuple[float | None, str]] = []
    for f in Path(run_dir).glob(".huginn/hypothesis_graph_*.json"):
        try:
            d = json.loads(f.read_text(encoding="utf-8", errors="ignore"))
        except Exception:  # — 原因: 图文件损坏/半写跳过, 缺图不影响其余轨迹重放
            continue
        for n in (d.get("nodes") if isinstance(d, dict) else d) or []:
            ca = str(n.get("created_at") or "")
            nodes.append((_iso_epoch(ca) if ca else None, n.get("statement") or ""))
    return nodes


def surprises_from_episodic(rows: list[dict]) -> Counter:
    """统一 surprise 分布 —— 与路由/记忆/展示同源 (``signals.routing_surprise``).

    run.log 里出现的 ``surprise=`` **全是自由文本**: 计划描述里的
    ``[auto-routed: surprise=1.00]`` 前缀、LLM 复述的 "surprise=1.0" 等, 不是信号.
    拿它统计会把"计划里字面复现的值"误报成"路由信号死"(run65 实测: run.log 字面
    全 1.0, 但 episodic 的秩归一信号有 4 个不同值 ⇒ 路由**未**退化). episodic
    快照的 ``surprise`` 就是 ``routing_surprise``(v31 统一), 是同一事实源.
    """
    c: Counter = Counter()
    for r in rows:
        v = r.get("surprise")
        if v is None:
            continue
        c[round(float(v), 6)] += 1
    return c


def exec_evidence_from_episodic(rows: list[dict]) -> dict:
    """从 episodic 轨迹提取**真实执行证据** (不受调试开关影响).

    run.log 的 ``[code-lab-run]`` 行只在 ``HUGINN_EXEC_ROUTE_DEBUG`` 打开时输出
    (见 :meth:`engine_act._run_code_lab`) ⇒ 正常 run 里 ``scan_runlog`` 恒得
    ``execs=0``, 误报"执行=0". 但 episodic 每轮的 ``execute`` 动作**总是**落盘,
    且带 ``exec_ok`` (= ``execution_result is not None``), 是不依赖开关的权威源.
    故以它为回退: 没抓到调试行时用这里的计数, 而非把"没开调试"当成"没执行".
    """
    execs = 0
    exec_ok = 0
    for r in rows:
        if r.get("action") != "execute" or "exec_ok" not in r:
            continue
        execs += 1
        if r.get("exec_ok"):
            exec_ok += 1
    return {"execs": execs, "exec_ok": exec_ok}


def input_stats_from_episodic(rows: list[dict]) -> dict:
    """从 episodic 取"输入长度/执行输出规模" —— 不依赖 run.log 的调试开关.

    run.log 的 ``[code-lab-author] prompt_len=`` / ``[exec-route] obj_len=`` /
    ``[code-lab-run] nobj=`` 三行都只在 ``HUGINN_EXEC_ROUTE_DEBUG`` 下输出; 不打开时
    ``scan_runlog`` 得空 Counter ⇒ "输入冻结 / 执行输出恒同"判定**静默失明**
    (run80-84 实测: 关掉开关, 三项全空, 判词整段不触发). episodic 快照的
    ``prompt_len`` / ``obj_len`` / ``nobj`` 是结构化权威源 (每轮必然落盘), 作回退.

    注意 ``prompt_len`` / ``nobj`` 为 ``None``(本轮未调作者 / 非 execute 轮)时不计入:
    记录它们是**本轮真值**, 不是残值; 计 0 会把"没调"混进"调了多长"消掉区分度.
    """
    pl: Counter = Counter()
    ol: Counter = Counter()
    nb: Counter = Counter()
    for r in rows:
        v = r.get("prompt_len")
        if v:
            pl[int(v)] += 1
        v = r.get("obj_len")
        if v is not None:
            ol[int(v)] += 1
        v = r.get("nobj")
        if v is not None:
            nb[int(v)] += 1
    return {"prompt_lens": pl, "obj_lens": ol, "nobj": nb}


def scan_runlog(run_dir: str) -> dict:
    """把 run.log 折成计数/序列: 用于出口体检与"输入冻结"检测.

    ⚠ ``prompt_len`` / ``obj_len`` / ``nobj`` 三项的来源行都在 ``HUGINN_EXEC_ROUTE_DEBUG``
    调试开关下 —— 这里抓到就抓到, 抓不到由 :func:`input_stats_from_episodic` 从
    episodic 结构化字段回退 (见 :func:`audit`), 以免关掉开关就静默失明.
    """
    p = Path(run_dir) / "run.log"
    ev = {"execs": 0, "exec_ok": 0, "nobj": Counter(), "prompt_lens": Counter(),
          "surprises": Counter(), "obj_lens": Counter(), "repeat_streaks": [],
          "rename": [], "hunt": 0, "plan_fail": 0, "terminal": 0, "soft": 0,
          "traces": Counter()}
    if not p.exists():
        return ev
    for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = re.search(r"\[code-lab-author\].*prompt_len=(\d+)", ln)
        if m:
            ev["prompt_lens"][int(m.group(1))] += 1
        m = re.search(r"\[code-lab-run\].*success=(\w+) nobj=(\d+)", ln)
        if m:
            ev["execs"] += 1
            ev["exec_ok"] += 1 if m.group(1) == "True" else 0
            ev["nobj"][int(m.group(2))] += 1
        # 注意: 不解析 run.log 的 `surprise=`. 那些全是自由文本(计划描述前缀 /
        # LLM 复述), 不是信号 —— 会误报"路由信号死". surprise 改由 episodic 的
        # 统一秩信号取 (surprises_from_episodic), 见 audit().
        # 锚定到行内真实字段位 (`obj_len=%d is_exp_desc=%s`): 该行 desc[:80] 是 repr 的
        # **自由文本**, 未锚定的 `obj_len=(\d+)` 会被描述里字面出现的 "obj_len=N" 假命中.
        m = re.search(r"obj_len=(\d+) is_exp_desc=", ln)
        if m:
            ev["obj_lens"][int(m.group(1))] += 1
        m = re.search(r"repeat execution detected \(streak=(\d+)\)", ln)
        if m:
            ev["repeat_streaks"].append(int(m.group(1)))
        m = re.search(r"renamed-reduction (\d+)× consecutive:\s*(\S+)", ln)
        if m:
            ev["rename"].append((int(m.group(1)), m.group(2)))
        if "counterexample hunt triggered" in ln:
            ev["hunt"] += 1
        if "plan_check failed" in ln:
            ev["plan_fail"] += 1
        # A2 后 "rename debt / exec convergence" 等命中 TERMINAL_MARKERS 的行**多带**
        # "advisory only ... no stop" —— 那是**提示**不是终止. 不排除会把 advisory
        # 误计成终止出口, 让出口体检假阳性 (run65 实测: 该前缀行存在但 terminal 语义为 0).
        if any(t in ln for t in TERMINAL_MARKERS) and "advisory only" not in ln:
            ev["terminal"] += 1
        if any(s in ln for s in SOFT_MARKERS):
            ev["soft"] += 1
        # 控制面 trace 落盘计数: A2 后"无进展是否**可观测**"就靠这个核对.
        m = re.search(r"control_trace name=(\S+)", ln)
        if m:
            ev["traces"][m.group(1)] += 1
    return ev


# ── 真实守卫重放 ──────────────────────────────────────────────────────

def replay_rename_streak(statements: list[str]) -> dict:
    """用真实 `EquivalenceAuditor._is_equivalent` 重放换名归约链.

    严格复刻 hypothesis_loop 的计数语义: 只在"判为换名"时 +1; 非换名**不重置**;
    streak>=5 时执行"阻断+重定向"**并把 streak 归零** (L2298-2307).
    输出用于证明这是闭环: 每 5 次换名 → 1 次 hunt + 1 次 escalate+reset, 无终止.
    """
    ea = EquivalenceAuditor()
    prior: list[str] = []
    streak = 0
    events: list[tuple[int, str]] = []
    rejected = Counter()
    for i, s in enumerate(statements):
        s = (s or "").strip()
        if not s:
            continue
        if not _is_substantive_statement(s):
            rejected["empty_shell"] += 1
            continue
        if _statement_key(s) in {_statement_key(p) for p in prior}:
            rejected["exact_dup"] += 1
        if any(p and ea._is_equivalent(s, p) for p in prior):
            streak += 1
            if streak == 3:
                events.append((i, "hunt"))
            elif streak >= 5:
                events.append((i, "escalate_reset"))
                streak = 0
        prior.append(s)
    return {"events": events, "violations": len(events),
            "hunts": sum(1 for _, k in events if k == "hunt"),
            "escalate_resets": sum(1 for _, k in events if k == "escalate_reset"),
            "rejected": dict(rejected)}


def cluster_statements(stmts: list[str]) -> dict:
    """对一组假设陈述做**等价聚类** (真实 `_is_equivalent`, difflib>0.8).

    "假设图增长"只在节点数上看起来像进展; 若 158 个节点其实只聚成个位数簇,
    说明图在膨胀而信息没增长 —— 这是可复现的无进展证据.
    """
    ea = EquivalenceAuditor()
    reps: list[str] = []
    for s in stmts:
        s = (s or "").strip()
        if not _is_substantive_statement(s):
            continue
        if not any(ea._is_equivalent(s, r) for r in reps):
            reps.append(s)
    return {"nodes": len([s for s in stmts if (s or "").strip()]),
            "clusters": len(reps),
            "redundancy": (1.0 - len(reps) / max(1, len([s for s in stmts if (s or '').strip()])))}


def replay_graph_based_audit(
    statements: list[str],
    auditor: EquivalenceAuditor | None = None,
    max_nodes: int | None = None,
) -> dict:
    """离线重放 v12"图基冗余审计": 冗余基线从 objective 换成**假设图**.

    旧码 audit(candidate, original_problem=objective) 把每条候选假设都拿去和"待解的
    问题"比 ⇒ 机制不同 (代数秩 / 流形维数 / Rademacher 复杂度) 也被判换名, 债务虚高
    → 过早终止, 误杀有效探索. 新码只和"已入图的旧假设"比 ⇒ 只有真换名重提才计入.
    本函数在**同一批**记录节点上按新判据重放, 输出被新判据判为换名的次数;
    与 run.log 记录 (旧判据) 的换名次数对照, 即可量化"有多少次本来有效的探索曾被误判".

    参数 auditor: 传入带真实 model 的 EquivalenceAuditor (`--llm`) 可启用**语义判**
    (抓同义改写); 默认无 model ⇒ 仅字面等价 (difflib), 结果是真 flag 数的**下界**.
    参数 max_nodes: 只重放前 N 个节点 (`--llm` 每节点一次调用, 便于小范围试跑);
    None ⇒ 全量. 结果含 nodes (采样数) / nodes_total (全量), 避免把采样当全量.
    """
    ea = auditor if auditor is not None else EquivalenceAuditor()
    semantic = bool(auditor is not None and getattr(auditor, "_model", None) is not None)
    total = sum(1 for s in statements if _is_substantive_statement((s or "").strip()))
    capped = statements[:max_nodes] if max_nodes else statements
    prior: list[str] = []
    flags = 0
    for i, s in enumerate(capped):
        s = (s or "").strip()
        if not _is_substantive_statement(s):
            continue
        if semantic and i and i % 20 == 0:
            print(f"  ... 语义重放 {i}/{len(capped)} 节点", file=sys.stderr, flush=True)
        v = ea.audit_hypothesis_against_graph(s, prior)
        if v.is_equivalent_renaming:
            flags += 1
        prior.append(s)
    return {
        "nodes": len(prior),
        "nodes_total": total,
        "max_nodes": max_nodes,
        "graph_based_rename_flags": flags,
        "flag_rate": (flags / len(prior)) if prior else 0.0,
        "semantic_llm": semantic,
    }


def rename_oscillation(recorded: list[tuple[int, str]]) -> dict:
    """用 run.log 记录的 rename 事件判定"闭环振荡".

    记录形如 [(3,'trigger'), (5,'escalate'), (3,'trigger'), (5,'escalate'), ...].
    escalate 与 hunt 交替且都 >0 ⇒ 每 5 次换名只做"阻断+重定向+streak 归零",
    没有任何终止动作, 可无限循环.
    """
    hunts = sum(1 for _, k in recorded if k.startswith("trigger"))
    esc = sum(1 for _, k in recorded if k.startswith("escalate"))
    streak_seq = [s for s, _ in recorded]
    return {"events": len(recorded), "hunts": hunts, "escalations": esc,
            "streak_seq": streak_seq,
            "closed_loop": hunts > 0 and esc > 0 and abs(hunts - esc) <= 1}


def replay_rename_debt(recorded: list[tuple[int, str]]) -> dict:
    """离线重放 v11"换名债务": 给出 **A2 前**该轨迹会在第几次换名越界.

    A2 已把该出口降级为 advisory (cognitive_loop.py: "已降级为提示 + trace,
    不再自动终止 run") —— 越界只写一条 `rename_debt` trace, 不 `should_stop`.
    故 `terminal_at_rename_occurrence` / `terminates` 都是**反事实** (terminates
    = "若沿用 A2 前语义会在第几次终止"), `stops_live` 恒为 False. run65 实测
    可终止出口=0, 与 A2 一致.

    为何用 run.log 的**记录事件**而非字面 `_is_equivalent`: 线上 `_rename_streak`
    由 recall+LLM 等价审计驱动, 是**语义级**的 (本工具已证: 字面聚类抓不到换名),
    故字面重放会系统性低计. 记录事件 (streak, kind) 才是线上真实触发序列 ——
    由它反推"换名发生次数", 再套债务语义 (单调; 3/5 阶梯自复位不复位债务),
    即可确定性地算出越界点.
    """
    debt = 0
    max_debt = 0
    occ = 0          # 累计换名次数
    prev = 0         # 上一事件记录的 streak
    terminal_at_occ: int | None = None
    for s, kind in recorded:
        delta = s - prev if s > prev else s   # escalate 后 streak 归零 ⇒ 新周期从 0 起
        occ += delta
        debt += delta
        max_debt = max(max_debt, debt)
        if debt >= _DEBT_LIMIT and terminal_at_occ is None:
            terminal_at_occ = occ
        prev = 0 if kind.startswith("escalate") else s
    total_occ = occ
    return {"limit": _DEBT_LIMIT, "max_debt": max_debt,
            "rename_occurrences": total_occ,
            "terminal_at_rename_occurrence": terminal_at_occ,
            "terminates": terminal_at_occ is not None,
            # A2: 线上该出口只提示不终止 ⇒ 上面 terminal_at_* 是反事实, 这里是事实.
            "stops_live": False}


# ── 审计 ─────────────────────────────────────────────────────────────

def _build_llm_auditor() -> EquivalenceAuditor:
    """构造带真实 model 的审计器, 供 `--llm` 做**语义重放**. 无 config/依赖时抛错."""
    from huginn.llm import get_model

    return EquivalenceAuditor(model=get_model(temperature=0.0))


def audit(
    run_dir: str,
    auditor: EquivalenceAuditor | None = None,
    max_nodes: int | None = None,
) -> dict:
    rows = load_episodic(run_dir)
    cycles = cycles_from_episodic(rows)
    nodes = load_graph_nodes(run_dir)
    log = scan_runlog(run_dir)
    # 执行计数回退: run.log 的 [code-lab-run] 只在 HUGINN_EXEC_ROUTE_DEBUG 下输出,
    # 正常 run 抓不到 ⇒ 用 episodic 的 execute 动作计数, 免把"没开调试"误报成"没执行".
    if log["execs"] == 0:
        epi = exec_evidence_from_episodic(rows)
        log["execs"], log["exec_ok"] = epi["execs"], epi["exec_ok"]
    # 输入长度/执行规模回退: 同 execs 的坑 —— 那三行的来源全在 HUGINN_EXEC_ROUTE_DEBUG
    # 下, 关掉开关 scan_runlog 得空 ⇒ "输入冻结/执行输出恒同"静默失明. 空则取 episodic.
    _epi_in = input_stats_from_episodic(rows)
    for _k in ("prompt_lens", "obj_lens", "nobj"):
        if not log[_k]:
            log[_k] = _epi_in[_k]

    # 每轮"新增假设节点"数: 按 graph.created_at 落进 [cycle.ts, next.ts)
    node_ts = sorted(t for t, _ in nodes if t is not None)
    new_nodes: list[int] = []
    for i, c in enumerate(cycles):
        lo, hi = c["ts"], (cycles[i + 1]["ts"] if i + 1 < len(cycles) else float("inf"))
        new_nodes.append(sum(1 for t in node_ts if lo <= t < hi))

    rename = replay_rename_streak([c["statement"] for c in cycles])
    # 假设图等价聚类: 节点数 vs 真正不同的命题簇 (可复现的"膨胀无信息")
    graph_clusters = cluster_statements([s for _, s in nodes])
    # v12 判据验证: 同一批节点用"图基冗余审计"重放 (旧码对 objective 判)
    graph_audit = replay_graph_based_audit(
        [s for _, s in nodes], auditor=auditor, max_nodes=max_nodes)
    # 记录在案的换名事件 → 是否闭环振荡 (线上走 recall+LLM, 离线只能用记录值)
    osc = rename_oscillation(log["rename"])
    # v11 出口验证: 在新"换名债务"不变量下, 这条旧轨迹是否**必然终止**
    debt_replay = replay_rename_debt(log["rename"])

    # surprise 分布一律取**统一信号** (episodic 的 routing_surprise 秩归一),
    # 不用 run.log 的自由文本 `surprise=` (见 surprises_from_episodic).
    log["surprises"] = surprises_from_episodic(rows)

    # 结构通道是否**被激活过**: 有活跃 map 时 StructureDescriptor.encode 必产出
    # 非零 (已自检: 最小晶体胞得 [1.5,90,0,1.43,15.6,...]); 全 0 ⇒ 该 run 从未
    # 建结构 map (非结构域目标如 ML/数学属正常), **不是**编码器坏. 旧判据把"全 0"
    # 当"编码器死", 对每个 ML/数学 run 都误报 (cognitive_checks
    # ``_snapshot_structure_desc`` 已注明此坑).
    struct_vecs = [r.get("structure_desc") for r in rows if r.get("structure_desc")]
    struct_unexercised = bool(struct_vecs) and all(
        not any(float(v) != 0.0 for v in vec) for vec in struct_vecs)

    n_cycles = len(cycles)
    zero_progress_cycles = [i for i in range(n_cycles) if new_nodes[i] == 0]
    # "恒定才叫死"三处统一口径: 恰好 1 个取值 **且** 样本 >=2 —— 单样本轨迹
    # (只调过一次作者 / 只有一轮)的 1 个值不足以证明冻结, 否则短 run 一律假阳性.
    frozen_prompt = (len(log["prompt_lens"]) == 1 and sum(log["prompt_lens"].values()) >= 2)
    saturated_surprise = (
        len(log["surprises"]) == 1 and sum(log["surprises"].values()) >= 2)
    frozen_goal = (len(log["obj_lens"]) == 1 and sum(log["obj_lens"].values()) >= 2)

    return {
        "run_dir": run_dir,
        "cycles": n_cycles,
        "execs": log["execs"],
        "new_nodes_total": sum(new_nodes),
        "new_nodes_per_cycle": new_nodes,
        "zero_progress_cycles": zero_progress_cycles,
        "graph_clusters": graph_clusters,
        "graph_based_audit": graph_audit,
        "rename": rename,
        "rename_oscillation": osc,
        "rename_debt_replay": debt_replay,
        "exits": {
            "terminal": log["terminal"],
            "soft": log["soft"],
            "hunt": log["hunt"],
            "plan_fail": log["plan_fail"],
            "repeat_streaks": log["repeat_streaks"],
            # A2 后终止出口被降级, 判据从"是否终止"改为"无进展是否可观测" ⇒
            # 落盘的 control_trace 计数就是这条判据的实测证据.
            "control_traces": dict(log["traces"]),
        },
        "input_frozen": {
            "prompt_len_values": dict(log["prompt_lens"]),
            "obj_len_values": dict(log["obj_lens"]),
            "surprise_values": dict(log["surprises"]),
            "prompt_frozen": frozen_prompt,
            "goal_frozen": frozen_goal,
            "surprise_saturated": saturated_surprise,
        },
        "structure_channel_unexercised": struct_unexercised,
        "nobj_distribution": dict(log["nobj"]),
    }


def _verdict(a: dict) -> list[str]:
    out = []
    n, z = a["cycles"], len(a["zero_progress_cycles"])
    rn = a["rename"]
    osc = a.get("rename_oscillation") or {}
    gc = a.get("graph_clusters") or {}
    ex = a["exits"]
    if n:
        out.append(f"轮数={n} 执行={a['execs']} 假设图新增节点={a['new_nodes_total']}")
        out.append(f"零进展轮={z}/{n}")
    if osc.get("events"):
        out.append(
            "换名归约%s: 记录 %d 次 (hunt=%d, escalate+reset=%d), streak 序列=%s"
            % ("**闭环振荡**" if osc.get("closed_loop") else "",
               osc["events"], osc["hunts"], osc["escalations"], osc["streak_seq"][:12]))
    if osc.get("closed_loop"):
        out.append(
            "  ⇒ 每 5 次换名只做'阻断+重定向'并把 streak **归零**"
            "([hypothesis_loop.py:2310]), 无终止动作, 可无限循环")
    dr = a.get("rename_debt_replay") or {}
    if dr.get("terminates"):
        out.append(
            "v11 换名债务重放 (A2 前**反事实**): 共 %d 次换名, 单调债务在 limit=%d 处越界 →"
            " 若沿用 A2 前语义, 第 %d 次换名即触发 'conclude+stop' (旧码 3/5 阶梯自复位,"
            " 可无限打转). **A2 已把该出口降级为 advisory**(只提示+trace, 不 should_stop),"
            " 故线上**不终止** —— 本行不是'必然终止'的证明"
            % (dr["rename_occurrences"], dr["limit"],
               dr["terminal_at_rename_occurrence"]))
    elif dr.get("max_debt", 0) > 0:
        out.append(
            "v11 换名债务重放 (A2 前反事实): 峰值 %d < limit=%d, 未形成持续换名闭环"
            % (dr["max_debt"], dr["limit"]))
    if rn["violations"]:
        out.append(
            "离线重放(仅最近 %d 轮)也见换名闭环: hunt=%d, escalate+reset=%d; "
            "实质守卫拒绝=%s" % (n, rn["hunts"], rn["escalate_resets"], rn["rejected"] or "{}"))
    if gc.get("nodes"):
        out.append(
            "假设图 %d 节点, 字面等价聚类 %d 簇 (字面冗余率 %.0f%%): 换名是**语义级**的,"
            " 字面守卫抓不到 ⇒ 只有 recall+LLM 的等价审计能抓, 而它只重定向不终止"
            % (gc["nodes"], gc["clusters"], 100 * gc["redundancy"]))
    ga = a.get("graph_based_audit") or {}
    if ga.get("nodes"):
        _mode = "语义+字面" if ga.get("semantic_llm") else "仅字面(真值的下界)"
        _sampled = bool(
            ga.get("max_nodes") and ga["nodes"] < ga.get("nodes_total", ga["nodes"]))
        if _sampled:
            # 采样模式下不得拿"全量旧换名数"减"采样新 flag 数": 分母不一致, 差集无意义.
            _mode += f", 采样 {ga['nodes']}/{ga['nodes_total']} 节点"
            out.append(
                "v12 判据验证 [%s]: 图基审计(对假设图判)标 %d 个换名 (%.0f%%); "
                "采样模式 ⇒ 不做与旧码全量换名数的差集对比 (避免把采样当全量)"
                % (_mode, ga["graph_based_rename_flags"], 100 * ga["flag_rate"]))
        else:
            # 旧码"换名次数"有两种口径, 差别不小: (a) run.log **记录事件** = 实际打出的
            # streak 触发数; (b) 离线按 delta 反推的**换名发生次数** (把 streak 自复位前的
            # 递增也算上). 只报一个会把"差集"伪装成单值, 故给区间.
            _ev = osc.get("events", 0)
            _dr = dr.get("rename_occurrences", 0)
            _d_ev = max(0, _ev - ga["graph_based_rename_flags"])
            _d_dr = max(0, _dr - ga["graph_based_rename_flags"])
            out.append(
                "v12 判据验证 [%s]: 图基审计(对假设图判)在 %d 个节点上标 %d 个换名 (%.0f%%). "
                "旧码(对 objective 判)基线两口径: run.log **记录事件** %d 次 → 差集 %d; "
                "离线反推**换名发生** %d 次 → 差集 %d. 差集即被误判的机制不同假设"
                "(代数秩/流形维数/Rademacher 等); 给区间而非单值, 是因为'旧码真正记为换名几次'"
                "本身取决于口径"
                % (_mode, ga["nodes"], ga["graph_based_rename_flags"],
                   100 * ga["flag_rate"], _ev, _d_ev, _dr, _d_dr))
    if ex["terminal"] == 0:
        # A2 后终止出口只保留挂钟/目标达成: 判据从"是否终止"改为"无进展是否**可观测**"。
        # 落盘 control_trace 计数就是这条判据的实测证据 —— 故**只要有任何 trace 就报**,
        # 不能再挂在 `soft>0` 下: 零进展但无软动作的轮 (如 run80 D-slice 只发
        # branch_slice_skip) 会被整段吞掉, 证据只在 --json 里可见 = 判词失效。
        _tr = ex.get("control_traces") or {}
        _tr_s = ", ".join(
            f"{k}×{v}" for k, v in sorted(_tr.items(), key=lambda kv: -kv[1]))
        if ex["soft"] > 0:
            out.append(
                "出口体检 (A2 语义): 软动作 %d 次, 可终止出口 0 次 —— A2 后终止出口只保留"
                "**挂钟预算/目标达成**, 收敛·换名债务·结题类均已降级为 advisory, 故此**不是**"
                "不变量违规; 改判「无进展是否**可观测**」" % ex["soft"])
        if _tr_s:
            out.append("  ⇒ 落盘 control_trace: " + _tr_s)
        elif ex["soft"] > 0 or z:
            out.append(
                "  ⇒ 落盘 control_trace: **无** —— 有无进展轮却无任何 trace = 真·不可观测")
    ifr = a["input_frozen"]
    if ifr["prompt_frozen"]:
        out.append(f"输入冻结: prompt_len 恒定 {list(ifr['prompt_len_values'])} → 同一问题反复问")
    elif ifr["goal_frozen"] and not ifr["prompt_len_values"]:
        # obj_len 恒定**只在没有作者 prompt 采样时**才算冻结证据: 研究目标本就恒定,
        # 作者每轮真实输入是 prompt_len (v11 反冻结后随迭代变化 ⇒ 非冻结). 否则会
        # 把健康 run 误报为"输入冻结"(run50 实测: prompt_len 变动, obj_len 恒 509).
        out.append(f"输入冻结: obj_len 恒定 {list(ifr['obj_len_values'])} → 计划描述不变")
    if ifr["surprise_saturated"]:
        out.append(
            f"路由信号死: 统一 surprise(秩归一) 恒定 {list(ifr['surprise_values'])}"
            " → 路由退化为恒同一条")
    if a["structure_channel_unexercised"]:
        out.append(
            "结构通道未激活: structure_desc 全 0 (该 run 未建结构 map) —— 非结构域"
            "目标(ML/数学)属正常; 编码器自检可产出非零 ⇒ **不是**'编码器坏/通道无信息'")
    if ex["repeat_streaks"]:
        out.append(f"重复执行 streak 峰值={max(ex['repeat_streaks'])} (最长 {len(ex['repeat_streaks'])} 次命中)")
    if len(a["nobj_distribution"]) == 1 and a["execs"]:
        out.append(f"执行输出恒同: nobj 恒为 {list(a['nobj_distribution'])} → 执行层零新信息")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="离线重放审计: 枚举无进展但未出口的空转路径")
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--json", action="store_true", help="额外输出原始审计 JSON")
    ap.add_argument(
        "--llm", action="store_true",
        help="对每个假设节点启用 LLM 语义判 (与假设图比, 抓同义改写); 需可用 model."
             " 不传时仅字面判, 结果是真 flag 数的下界. 注意: 每节点一次 LLM 调用.",
    )
    ap.add_argument(
        "--max-nodes", type=int, default=0,
        help="配合 --llm: 每个 run 只语义重放前 N 个节点 (0=全量), 便于小范围试跑.",
    )
    args = ap.parse_args(argv)
    max_nodes = args.max_nodes or None

    auditor: EquivalenceAuditor | None = None
    if args.llm:
        try:
            auditor = _build_llm_auditor()
            print("--llm: 已加载 model, 启用语义重放")
        except Exception as e:  # 无 config / 缺依赖 → 明确告知并退化
            print(f"! --llm 不可用 ({type(e).__name__}: {e}); 退化为仅字面重放")
            auditor = None

    for rd in args.run_dirs:
        a = audit(rd, auditor=auditor, max_nodes=max_nodes)
        print(f"\n===== {rd} =====")
        for line in _verdict(a):
            print("  •", line)
        if args.json:
            print(json.dumps(a, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
