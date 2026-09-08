"""决策门控 —— 以 Transformer 固有缺陷为探针的"查漏补缺"治理面 (M4).

三条缺口的唯一命题: 我们一直在给 agent"看得更多"的能力(检索、长上下文、多工具),
却低估了"看得更多"自带的三种失真 —— 锚定、稀释、把看过当成用过. 本模块把三种
失真各自渲染成一个显式、可证伪、可入 harness 审计面的门控原语:

  漏A · plan 修订门 (锚定):     新证据到达后, 显式复盘初始计划是否仍成立,
       而不默认"上下文会自然覆盖旧计划".  → :func:`revision_gate`
  漏B · 决策先导摘要 + 相关度门控 (稀释): 长工具输出先进治理 —— 先提炼先导摘要
       再进决策上下文, 低相关项不注入决策, 防止 softmax 式"均匀灌注无关信息".
       → :func:`distill_tool_output` / :func:`select_decision_context`
  漏C · "得分≠使用" grounding 审计 (看过≠用过, Jain & Wallace):
       高价值(存活/高分)项的数值是否真写进最终报告, 而不只是被发现/检索到.
       → :func:`grounding_audit`

对齐心智模型: 动态上下文聚合(每个决策点重算该关注什么, 漏A) + IO 优先(在注入决策
前于入口处净化, 漏B) + 注意力权重≠解释(高权重≠真用了, 漏C). 三条都把"再看一次"
从"信息无损全灌"改造成"进决策前先过一遍显式净化和对账".
"""
from __future__ import annotations

import re
from typing import Any


# ── 漏A · plan 修订门(锚定反制) ───────────────────────────────────────────
def _first_scalar(node: Any) -> float | None:
    """从任意结构里取第一个可量标量(与 program / law_model 同语义), 不伪造."""
    if isinstance(node, bool):
        return None
    if isinstance(node, (int, float)):
        return float(node)
    if isinstance(node, dict):
        for k in ("T_eq_K", "T", "score", "value", "y", "actual", "S_Wm2"):
            if k in node:
                v = _first_scalar(node[k])
                if v is not None:
                    return v
        for v in node.values():
            s = _first_scalar(v)
            if s is not None:
                return s
    elif isinstance(node, (list, tuple)):
        for v in node:
            s = _first_scalar(v)
            if s is not None:
                return s
    return None


def revision_gate(plan_dict: dict | None, cache: dict | None, goal: str) -> dict[str, Any]:
    """漏A · plan 修订门: 用执行后证据(→cache)复盘初始计划是否仍成立.

    plan_dict: :class:`ResearchPlan.to_dict()` 的产物(含 `goal`/`experiments` 名称表).
    cache:    name -> {summary, objectives, predicted, success} 的真实执行实录.

    判定(全部可证伪):
      - **falsified**: 计划里某实验带 `predicted` 且与真实 `summary` 偏差超容差(>3%)
        → 切入点假设被证伪, 计划应重启该方向.
      - **stale**:     计划里某实验从未进入 cache(=未被执行/被剪枝前就消失)
        → 计划的方向留白, 需要补跑或明确弃用.

    返回 {reviewed, falsified, stale, revised, verdict}, 其中 revised=True 即
    "该计划已过期, 需要显式重规划" —— 而不是假装上下文会自动覆盖旧计划(锚定反制).
    """
    reviewed = sorted(set(plan_dict.get("experiments", []) or []))
    cache = cache or {}

    # 对账容差: 与 world model reconcile 一致 (相对误差 <= 3% 视为成立)
    _TOL = 0.03

    falsified: list[str] = []
    for name in reviewed:
        res = cache.get(name) or {}
        try:
            pred = _first_scalar(res.get("predicted"))
            actual = _first_scalar(res.get("summary"))
        except Exception:  # noqa: BLE001 — 取值异常按无数值处理(不伪造)
            pred = actual = None
        if pred is not None and actual is not None:
            err = abs(pred - actual) / (abs(actual) or 1.0)
            if err > _TOL:
                falsified.append(name)

    stale = [n for n in reviewed if n not in cache]
    revised = bool(falsified or stale)
    return {
        "goal": goal,
        "reviewed": reviewed,
        "falsified": falsified,
        "stale": stale,
        "revised": revised,
        "verdict": "plan_holds" if not revised else "replan_needed",
    }


# ── 漏B · 决策先导摘要 + 相关度门控(稀释反制) ─────────────────────────────
def distill_tool_output(
    name: str,
    raw: str,
    *,
    goal: str = "",
    max_chars: int = 4000,
    drop_below: float = 0.0,
) -> dict[str, Any]:
    """漏B · 把一块工具回调改造成"决策先导摘要 + 相关度门控"后的注入决策格式.

    - `relevance` 0..1: 粗略相关度 = 目标关键词在输出中的命中和占总单词比重.
      仅作门控信号, 绝不把"相关度高"冒充解释(见漏C).
    - `front`: 先导摘要 —— 长度 > max_chars 时截断保留首段(长输出进决策前的净化),
      否则原样. 让 LLM 决策上下文先看到"要点", 不被冗长回调均匀灌注稀释.
    - `inject`: 是否允许注入决策上下文. `relevance < drop_below` 且 drop_below>0
      时拒之 → 真正"低相关项不注入"的门控; 默认 drop_below=0 = 只净化不丢, 兼容旧行为.

    注意: 原始 `raw` 必须完整留在 trace/门禁证据里 —— 本门控只裁剪"注入决策"那个
    视角的文本, 绝不裁剪可证伪证据本身.
    """
    raw = raw if isinstance(raw, str) else str(raw)
    kw = [w.strip().lower() for w in re.split(r"[^0-9a-zA-Z_\u4e00-\u9fff]+", goal or "")
          if len(w.strip()) > 1]
    low = raw.lower()
    hits = sum(1 for k in kw if k in low) if kw else 0
    n_terms = len(kw) or 1
    relevance = min(1.0, hits / n_terms)

    if len(raw) > max_chars:
        front = raw[:max_chars] + f"\n…[先导摘要在 {len(raw)} 字符, 完整证据见 trace]"
    else:
        front = raw

    inject = True if drop_below <= 0 or drop_below == 0 else relevance >= drop_below
    return {
        "tool": name,
        "relevance": round(relevance, 3),
        "front": front,
        "inject": bool(inject),
        "original_len": len(raw),
        "summary_len": len(front),
    }


def select_decision_context(
    results: list[tuple[str, str]],
    goal: str,
    *,
    max_chars: int = 4000,
    drop_below: float = 0.0,
) -> dict[str, Any]:
    """漏B · 批量门控: 一批工具回调 → 注入决策的净化上下文 + 门控日志.

    returns {context, gates:[{relevance, inject, ...}], dropped:[name]}
    context 保留原始可证伪项(注入视角), dropped 记录被相关度门控拦下的项名.
    """
    ctx: list[str] = []
    gates: list[dict] = []
    dropped: list[str] = []
    for name, raw in results:
        g = distill_tool_output(name, raw, goal=goal, max_chars=max_chars,
                                drop_below=drop_below)
        gates.append(g)
        if g["inject"]:
            ctx.append(g["front"])
        else:
            dropped.append(name)
    return {
        "context": "\n\n".join(ctx),
        "gates": gates,
        "dropped": dropped,
    }


# ── 漏C · "得分≠使用" grounding 审计(看过≠用过) ───────────────────────────
def grounding_audit(report: str, pareto_front: list[dict], /) -> dict[str, Any]:
    """漏C · 审计高价值实验的数值是否真写进最终报告(而非只被发现).

    pareto_front: 存活假说 [{name, hypothesis, objectives}] —— "高注意力/高分"一侧.
    report:       最终报告文本 —— "实际输出"一侧.

    Jain & Wallace 的镜像: 高权重(这里是高 objectives)≠ 真的被用于产出结论. 逐个存活
    项取其 objectives 的第一个标量, 看它是否以字符串形式出现在 report 里:
      - 全部出现 → verdict=proper_use: 高分被真实消费(记忆中的"用过"有当下落).
      - 有缺失  → verdict=score_without_use: 该高分项被检索到/存活, 却没进结论
                  — 提醒"看过≠用过"或结论对高价值证据存在 lipstick 式表面用.
    可证伪: 换一个不含这些数值的报告, verdict 立即翻转.
    """
    report = report or ""
    used: list[str] = []
    unused: list[str] = []
    for it in pareto_front or []:
        name = it.get("name") if isinstance(it, dict) else str(it)
        if not isinstance(it, dict):
            continue
        scalar = _first_scalar(it.get("objectives"))
        if scalar is None:
            # 无数值指标: 无法核对"用过", 记为未核对(诚实, 不硬判)
            unused.append(f"{name}:no-scalar")
            continue
        # 数值文本有多种合法写法(42 / 42.0 / 4.2e1): 全部核对, 免误报"未用".
        forms = {str(scalar)}
        if float(scalar).is_integer():
            forms.add(str(int(scalar)))
        if not any(f in report for f in forms):
            unused.append(name)
        else:
            used.append(name)
    return {
        "checked": used + unused,
        "used": used,
        "unused": unused,
        "verdict": ("proper_use" if not unused else "score_without_use"),
    }


# ── 缺陷三映射: 团队/假说视角分离度(防多头塌缩成单头) ────────────────────
def role_separation(
    survivors: list[dict], *, worldview_key: str = "worldview"
) -> dict[str, Any]:
    """度量存活假说在"视角/角色"上的分歧度 —— 防御"共识孤岛".

    背景 (缺陷三): Scientist×N 并行跑, 但没有机制保证 N 个头给出分歧; Pareto 剪枝
    还会主动淘汰少数派 → 幸存集可能全是同一视角, 注意力头塌缩成单头. 本函数给每个
    存活假说取一个"视角标签"并测分离度:

      - 标签: 显式 worldview / role / source 之一优先; 否则 "∅"(未标记).
      - separation = 去重标签数 / 存活数 (0..1)
      - verdict 三态(诚实, 不硬判):
          * unlabeled           : 全部未打标签 —— 无显式视角分化证据, 记 unobserved;
          * collapsed_single_view: 有标签但只剩单一视角(或分离度 < 0.5) —— 塌缩雷达;
          * diverse             : 去重视角 >= 2 且占比过半 —— 视角分化健康.

    可证伪: 换一批标签或阈值, 判定立即翻转.
    """
    labels: list[str] = []
    untagged = 0
    for it in survivors or []:
        if not isinstance(it, dict):
            continue
        lab = it.get(worldview_key) or it.get("role") or it.get("source")
        if lab in (None, "", "∅", "unknown"):
            untagged += 1
            labels.append("∅")
        else:
            labels.append(str(lab))
    n = len(labels)
    if n == 0:
        return {"checked": 0, "separation": 0.0, "untagged": 0,
                "labels": [], "verdict": "unlabeled"}
    distinct = len({l for l in labels if l != "∅"})
    separation = distinct / n
    if untagged == n:
        verdict = "unlabeled"
    elif distinct >= 2 and separation >= 0.5:
        verdict = "diverse"
    else:
        verdict = "collapsed_single_view"
    return {
        "checked": n,
        "separation": round(separation, 3),
        "untagged": untagged,
        "labels": sorted(set(labels)),
        "verdict": verdict,
    }