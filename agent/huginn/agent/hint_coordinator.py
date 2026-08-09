"""Huginn hint coordinator — Hodge 正交分解把 14 个 hint 降到 5.

v14 Phase 1 Task 5: Step 2 每轮原本无仲裁叠加 14 hint, context 被占满
且可能冲突. 这里按 spec §"hint 分类"把 14 hint 分到 gradient/curl/
harmonic/retrieval/deprecated 五族, 合并出 3 个 prompt 块 + 1 个由
调用方拼的 Meta-Trace top-k 块.

ponytail: 不引新依赖, 不上策略模式, 单文件单类. 14 hint 的分类是设计
意图启发 (spec §诚实边界 4), 不是数学推导.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


# v15 Phase 2 Task 4: posterior-guided hint 构造.
# ponytail: 单函数, 不引新依赖, 不上 token 化. 失败一律返回空串不阻塞主循环.
def _build_posterior_guided_hint(
    manifold: Any,
    observations: list,
    history_entries: list[dict] | None = None,
    *,
    max_chars: int = 1500,
    mcmc_current: str | None = None,
) -> str:
    """构造 posterior-guided hint 段: 核心 hint + 探索 hint (+ MCMC 驻留站点).

    核心 hint: abductive_inference 选出的 best hypothesis (description + predictions
    + posterior lift). 探索 hint: propose_next_exploration 返回的 max info gain
    hypothesis. MCMC hint: mcmc_current (MCMC 采样链当前驻留的假设) — 这是 MCMC
    的"动态采样路径", 可能停留在 argmax 之外的次优但值得关注的站点, 补充 argmax
    的贪婪盲区. 三段拼一起, 优先保留核心 > 探索 > MCMC.

    token 控制: 总 char ≤ max_chars, 超了按 核心 > 探索 > MCMC 顺序截断.
    ponytail: 4 char ≈ 1 token, 用 char count 代理. 升级路径: tiktoken 精确计数.

    失败降级: manifold=None / 空 hypothesis / 空 observations → 返回 "".
    history_entries 当前未用 (boost 在 coordinate() 里算), 留 signature 给后续扩展.
    mcmc_current 不在 manifold 时 (已废弃/不存在) 静默跳过, 不报错.
    """
    if manifold is None:
        return ""
    try:
        hyp_count = len(manifold._hyp)
    except Exception:
        return ""
    if hyp_count == 0 or not observations:
        return ""

    parts: list[str] = []
    best_h_id: str | None = None

    # 核心 hint: argmax_h P(h|O)
    try:
        best_h = manifold.abductive_inference(observations)
        if best_h is not None:
            best_h_id = best_h.h_id
            log_post_dict = manifold.log_posterior(observations)
            log_post = log_post_dict.get(best_h.h_id, 0.0)
            log_prior = best_h.log_prior()
            lift = log_post - log_prior
            core_lines = [
                "[posterior core hint]",
                f"best_hypothesis: {best_h.h_id}",
                f"description: {best_h.description}",
            ]
            if best_h.predictions:
                pred_str = ", ".join(
                    f"{k}={v:.4g}"
                    for k, v in list(best_h.predictions.items())[:5]
                )
                core_lines.append(f"predictions: {pred_str}")
            core_lines.append(
                f"posterior_lift: {lift:.3f} "
                f"(log_post={log_post:.3f}, log_prior={log_prior:.3f})"
            )
            parts.append("\n".join(core_lines))
    except Exception:
        logger.debug("posterior core hint failed", exc_info=True)

    # 探索 hint: max Fisher distance to current best (info gain 代理)
    try:
        next_h = manifold.propose_next_exploration(observations)
        if next_h is not None and (not parts or next_h.h_id != best_h.h_id):
            explore_lines = [
                "[posterior explore hint]",
                f"next_exploration: {next_h.h_id}",
                f"description: {next_h.description}",
            ]
            if next_h.predictions:
                pred_str = ", ".join(
                    f"{k}={v:.4g}"
                    for k, v in list(next_h.predictions.items())[:5]
                )
                explore_lines.append(f"predictions: {pred_str}")
            parts.append("\n".join(explore_lines))
    except Exception:
        logger.debug("posterior explore hint failed", exc_info=True)

    # MCMC hint: MCMC 采样链当前驻留的假设 (动态采样路径, 补充 argmax 贪婪盲区).
    # mcmc_current 不在 manifold 时静默跳过. 优先级最低 — 截断时最先被裁.
    if mcmc_current:
        try:
            _m_h = manifold._hyp.get(mcmc_current)
            if _m_h is not None:
                m_lines = [
                    "[posterior mcmc hint]",
                    f"mcmc_current_hypothesis: {mcmc_current}",
                    f"description: {_m_h.description}",
                ]
                if _m_h.predictions:
                    _mpred = ", ".join(
                        f"{k}={v:.4g}"
                        for k, v in list(_m_h.predictions.items())[:5]
                    )
                    m_lines.append(f"predictions: {_mpred}")
                # 若 MCMC 驻留站点就是 argmax (核心 hint), 标出来避免冗余重复.
                if mcmc_current == best_h_id:
                    m_lines.append("note: same as best_hypothesis (MCMC converged here)")
                # #3: least-visited 假设 = 未探索方向, 引导生成多样性.
                try:
                    _lv = manifold.mcmc_least_visited(k=3)
                    if _lv:
                        m_lines.append(
                            "least_visited (underexplored directions): "
                            + ", ".join(_lv)
                        )
                except Exception:
                    pass
                parts.append("\n".join(m_lines))
        except Exception:
            logger.debug("posterior mcmc hint failed", exc_info=True)

    if not parts:
        return ""

    out = "\n\n".join(parts)
    if len(out) <= max_chars:
        return out
    # 超了: 优先保留 core, explore 用剩余预算
    if len(parts) >= 2:
        core = parts[0]
        remain = max_chars - len(core) - 2  # 2 for "\n\n"
        if remain > 0:
            out = core + "\n\n" + parts[1][:remain]
        else:
            out = core[:max_chars]
    else:
        out = parts[0][:max_chars]
    return out


# E2-1: 通用观测抽取 — 让假设流形对任何 benchmark 都可用.
# 从自由文本里抓 "metric = value" 喂给 manifold 做后验/溯因. 不调 LLM.
# 白名单过滤避免误抓年份/版本号; 失败/无文本返回空 list, 不阻塞.
_METRIC_WHITELIST = frozenset({
    "mae", "rmse", "mse", "r2", "r\u00b2", "r3", "accuracy",
    "precision", "recall", "f1", "auc", "pearson", "spearman",
    "loss", "error", "score", "bias",
})
_NUMERIC_PAIR_RE = re.compile(
    r"\b([A-Za-z][A-Za-z0-9_\u00b2\u00b3]{1,15})\s*(?:=|:|of|\u2248|~|is)\s*"
    r"([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)"
)


def extract_numeric_targets(text: str) -> dict[str, float]:
    """从文本抓 'metric = value' 模式, 返回 {metric: value}.

    白名单 + 数值量级过滤, 避免误抓年份/版本号. 无文本返回空 dict.
    """
    if not text:
        return {}
    targets: dict[str, float] = {}
    for m in _NUMERIC_PAIR_RE.finditer(text):
        name = m.group(1).lower()
        try:
            val = float(m.group(2))
        except ValueError:
            continue
        if name not in _METRIC_WHITELIST:
            continue
        if abs(val) > 1e6:
            continue
        targets[name] = val
    return targets


def extract_observations(text: str) -> list:
    """从文本解析 Observation 列表, 喂给假设流形.

    sigma=0.1 — 归一化度量常见测量噪声水平, 太宽会让 BIC prior 主导
    likelihood. ponytail: 升级路径是接 LLM-as-likelihood 给 per-obs sigma.
    """
    from huginn.metacog.hypothesis_manifold import Observation

    obs: list = []
    seen: set[str] = set()
    for m in _NUMERIC_PAIR_RE.finditer(text):
        name = m.group(1).lower()
        try:
            val = float(m.group(2))
        except ValueError:
            continue
        if name not in _METRIC_WHITELIST or name in seen:
            continue
        if abs(val) > 1e6:
            continue
        seen.add(name)
        obs.append(Observation(name=name, value=val, sigma=0.1))
    return obs


# 14 hint 五族分类 (spec §"hint 分类"). retrieval 族不在这处理, 由
# build_meta_trace_text 按 darwin_score top-k 回灌. deprecated 族这里
# 也不直接收, scan_text 例外 — 调用方仍传进来我们合进 step2_prompt.
HINT_FAMILIES: dict[str, tuple[str, ...]] = {
    "gradient": ("step2_prompt", "iter_prompt", "fcm_winner"),
    "curl": ("compass", "step_eval", "drift_info"),
    "harmonic": ("imagination", "meta_agent"),
    "retrieval": ("kb_chunks", "prospective", "target_chain", "episode_history"),
    "deprecated": ("scan_hint", "fcm_winner_reminder"),
}


class HintCoordinator:
    """对 14 hint 做 Hodge 正交分解, 输出 ≤3 个合并 prompt 块.

    coordinate() 返回 (prompt, events):
      - prompt: 合并后的字符串, 直接发给 LLM
      - events: trace 事件列表 (如 "cochain_conflict:..."), 调用方写进 meta_trace.jsonl

    ponytail: 类本身无状态, 不持有 workspace/llm 句柄. 升级路径: Phase 3
    cross_task_prior 需要读 CrossTaskStore, 那时再加 __init__ 参数.
    """

    # 冲突检测关键词 — gradient (FCM winner) 跟 harmonic (imagination) 对端词.
    # ponytail: keyword 匹配, 不上 embedding. 天花板: 同义改写漏检,
    # 升级路径跟 supported_ratio 共用 TF-IDF + cosine 基础设施.
    _GRADIENT_CONFLICT_MARKERS = ("按选定方案执行", "按 fcm winner")
    _HARMONIC_CONFLICT_MARKERS = ("换数学结构", "imagination")

    def coordinate(
        self,
        iter_n: int,
        csm_state: str | None,
        beta: tuple[int, int],
        last_verdict: str | None,
        fcm_winner: str | None,
        scan_text: str | None,
        step2_prompt: str,
        iter_prompt: str | None,
        compass: str | None,
        step_eval: str | None,
        drift_info: str | None,
        imagination: str | None,
        meta_agent: str | None,
        cross_task_prior: list[dict] | None = None,
        manifold: Any = None,
        posterior_hint: str | None = None,
    ) -> tuple[str, list[str]]:
        """合并 14 hint 为 3 个 prompt 块, 返回 (prompt, events).

        块结构 (spec §"hint 正交叠加"):
          [gradient block]    — step2_prompt + iter_prompt + fcm_winner
          [progress audit]    — curl 族, last_verdict != "pass" 时合并
          [topology probe]    — harmonic 族, β_1>0 或 stagnation 时合并

        Meta-Trace top-k 块由调用方用 build_meta_trace_text 拼, 这块不管.

        cross_task_prior (v14 Task 15): 同 domain 历史 high darwin entry,
        用来 boost 当前 hint 中关键词重叠 >0.5 的块. 调用方必须传同 domain
        的 entry, 跨 domain 隔离在 CrossTaskStore.query_high_darwin(domain=...)
        层实现 — 这里不重复过滤.

        v15 Task 4 升级: manifold 可用时, boost 权重从 keyword overlap 改为
        posterior lift (log_posterior - log_prior). entry 缺 v15 字段时降级
        到 keyword overlap. posterior_hint 是预构造的核心+探索 hint 段,
        注入到 blocks 最前面 (最高优先级), 不参与 boost 重排.

        ponytail: 调用方一次拿全 3 块, 自己决定是否再拼 Meta-Trace.
        """
        events: list[str] = []
        blocks: list[str] = []

        # --- gradient 族 (必需) ---
        # iter_prompt 是 iter>0 的 "continue execution" 文本, 跟 step2_prompt
        # 在 benchmark runner 里互斥 (iter 0 直接复用 step2_prompt). 取非空的那个.
        _grad = iter_prompt if (iter_prompt and iter_prompt != step2_prompt) else (step2_prompt or "")
        # scan_hint deprecated: 合并进 step2_prompt (spec §deprecated)
        if scan_text:
            _grad = (_grad + "\n\n" + scan_text).strip() if _grad else scan_text
        # fcm_winner reminder deprecated: spec "iter>0 不重复", 只 iter 0 注入
        if fcm_winner and iter_n == 0:
            _fcw = fcm_winner[:1200]
            _grad = (
                _grad + "\n\n## Selected Execution Plan (FCM winner)\n" + _fcw
            ).strip() if _grad else f"## Selected Execution Plan (FCM winner)\n{_fcw}"
        if _grad:
            blocks.append("[gradient block]\n" + _grad)

        # --- curl 族 (条件触发): verdict != pass 时合并 progress audit ---
        # 首轮 verdict=None 不触发; pass 不触发; 其他 (fix_needed / blocked) 触发
        _verdict_triggers = last_verdict is not None and last_verdict.lower() != "pass"
        _curl_parts = [p for p in (compass, step_eval, drift_info) if p]
        if _curl_parts and _verdict_triggers:
            blocks.append("[progress audit]\n" + "\n\n".join(_curl_parts))

        # --- harmonic 族 (条件触发): β_1>0 或 stagnation 时合并 topology probe ---
        _beta_1 = beta[1] if beta and len(beta) >= 2 else 0
        # stagnation 代理: spec 签名没单独参数, 复用 csm_state / last_verdict
        # 任一含 "stagnation" 关键词视为停滞信号. 调用方按需传.
        _stagnation = any(
            s and "stagnation" in s.lower() for s in (csm_state, last_verdict)
        )
        _harm_parts = [p for p in (imagination, meta_agent) if p]
        _harm_trigger = (_beta_1 > 0 or _stagnation) and bool(_harm_parts)

        # 冲突仲裁 (spec §"冲突仲裁"): gradient (FCM winner) 跟 harmonic (imagination)
        # 冲突时按族优先级保留 gradient, harmonic 块标注 "deferred", 事件入 list
        if _harm_trigger:
            _probe = "\n\n".join(_harm_parts)
            if fcm_winner and self._detect_conflict(fcm_winner, imagination):
                _probe = "FCM winner active, imagination deferred\n\n" + _probe
                events.append("cochain_conflict:fcm_vs_imagination")
            blocks.append("[topology probe]\n" + _probe)

        # --- v14 Task 15 / v15 Task 4: 跨 task prior boost ---
        # 选择条件不变 (v14 keyword overlap > 0.5), boost marker 升级:
        # entry 有 hypothesis_id + log_posterior 且 manifold 能查到时, marker
        # 用 posterior lift (log_posterior - log_prior) 替代 darwin_score.
        # ponytail: lift=log_lik_sum, Gaussian 下 ≤0, 越接近 0 说明数据越支持
        # 该 hypothesis. 用 lift 当 weight 是相对比较 (越高的 entry 越值得借势),
        # 不要求 lift>0. 失败降级: 无 manifold / entry 缺 v15 字段 → v14 darwin.
        if cross_task_prior:
            _boosted: list[str] = []
            _rest: list[str] = []
            for _blk in blocks:
                # 剥掉 "[gradient block]\n" header, 只用正文算重叠
                _hint_text = _blk.split("\n", 1)[1] if "\n" in _blk else _blk
                _best_overlap = 0.0
                _best_darwin = 0.0
                _best_lift = 0.0
                _best_entry_v15 = False
                for _entry in cross_task_prior:
                    _att = _entry.get("attempted") or ""
                    _ov = self._keyword_overlap(_hint_text, _att)
                    if _ov <= _best_overlap:
                        continue
                    # 找到更高 overlap 的 entry, 更新所有字段
                    _best_overlap = _ov
                    _best_darwin = float(_entry.get("darwin_score") or 0.0)
                    _best_entry_v15 = False
                    _h_id = _entry.get("hypothesis_id")
                    _log_post = float(_entry.get("log_posterior", 0.0) or 0.0)
                    if (manifold is not None and _h_id
                            and _log_post != 0.0):
                        try:
                            _h = manifold._hyp.get(_h_id)
                            if _h is not None:
                                _best_lift = _log_post - _h.log_prior()
                                _best_entry_v15 = True
                        except Exception:
                            logger.debug("manifold hypothesis lookup failed", exc_info=True)
                if _best_overlap > 0.5:
                    if _best_entry_v15:
                        _boosted.append(
                            f"[prior validated: lift={_best_lift:.2f}]\n" + _blk
                        )
                        events.append(
                            f"posterior_lift_boost:lift={_best_lift:.2f},"
                            f"overlap={_best_overlap:.2f}"
                        )
                    else:
                        _boosted.append(
                            f"[prior validated: darwin={_best_darwin:.2f}]\n"
                            + _blk
                        )
                        events.append(
                            f"cross_task_prior_boost:overlap={_best_overlap:.2f}"
                        )
                else:
                    _rest.append(_blk)
            blocks = _boosted + _rest

        # --- v15 Task 4: posterior-guided hint 注入 (最高优先级, 不参与 boost) ---
        # 核心 hint (abduction best) + 探索 hint (next exploration) 拼好的串,
        # 直接插到 blocks 最前面. 失败降级 (None/空) 跳过, 不影响 v14 行为.
        if posterior_hint:
            blocks.insert(0, posterior_hint)

        return "\n\n".join(blocks), events

    def _detect_conflict(self, hint_a: str | None, hint_b: str | None) -> bool:
        """简单关键词检测: gradient 跟 harmonic 冲突.

        ponytail: keyword 匹配, 不上语义相似度. 同义改写漏检是已知天花板.
        """
        if not hint_a or not hint_b:
            return False
        _a = hint_a.lower()
        _b = hint_b.lower()
        _grad_hit = any(m.lower() in _a for m in self._GRADIENT_CONFLICT_MARKERS)
        _harm_hit = any(m.lower() in _b for m in self._HARMONIC_CONFLICT_MARKERS)
        return _grad_hit and _harm_hit

    @staticmethod
    def _keyword_overlap(hint_text: str, prior_attempted: str) -> float:
        """简单关键词重叠率 (Jaccard index).

        ponytail: 用 set 交集, 不上 embedding. 天花板: 同义改写漏检
        (如 "superradiance" ↔ "bosonic mode"), 升级路径 TF-IDF + cosine.
        """
        _hint_words = set(hint_text.lower().split())
        _prior_words = set(prior_attempted.lower().split())
        if not _hint_words or not _prior_words:
            return 0.0
        return len(_hint_words & _prior_words) / len(_hint_words | _prior_words)


# === self-check ===
if __name__ == "__main__":
    # 场景 1 (spec §"冲突仲裁"): FCM winner="按选定方案执行" + imagination="换数学结构家族"
    # 断言: prompt 含 "FCM winner active, imagination deferred" + 事件被返回
    _hc = HintCoordinator()
    _prompt, _events = _hc.coordinate(
        iter_n=0,
        csm_state="execute",
        beta=(1, 1),  # β_1>0 触发 topology probe
        last_verdict="fix_needed",  # verdict != pass 触发 curl
        fcm_winner="按选定方案执行: 用 PDE 求解器",
        scan_text=None,
        step2_prompt="Now execute the task following your methodology checklist.",
        iter_prompt=None,
        compass="coverage=60%, missing band gap",
        step_eval="gap_severity=0.4",
        drift_info="drift=0.2",
        imagination="换数学结构家族: PDE ↔ variational",
        meta_agent=None,
    )
    assert "FCM winner active, imagination deferred" in _prompt, \
        f"conflict annotation missing:\n{_prompt}"
    assert "cochain_conflict:fcm_vs_imagination" in _events, \
        f"conflict event missing: {_events}"
    assert "[gradient block]" in _prompt
    assert "[progress audit]" in _prompt
    assert "[topology probe]" in _prompt

    # 场景 2: verdict=pass + β_1=0 → curl/harmonic 都不触发
    _p2, _e2 = _hc.coordinate(
        iter_n=1,
        csm_state="execute",
        beta=(1, 0),
        last_verdict="pass",
        fcm_winner=None,
        scan_text=None,
        step2_prompt="execute",
        iter_prompt="Continue execution. Iteration 2/4.",
        compass="coverage=100%",
        step_eval=None,
        drift_info=None,
        imagination="换数学结构",
        meta_agent=None,
    )
    assert "[gradient block]" in _p2, "gradient always present"
    assert "[progress audit]" not in _p2, "pass verdict should skip curl"
    assert "[topology probe]" not in _p2, "β_1=0 no stagnation should skip harmonic"
    assert _e2 == [], "no events when no conflict"

    # 场景 3: _detect_conflict 单元覆盖
    assert _hc._detect_conflict("按选定方案执行", "换数学结构家族") is True
    assert _hc._detect_conflict("按 FCM winner 跑", "imagination probe") is True
    assert _hc._detect_conflict("plan A", "imagination probe") is False
    assert _hc._detect_conflict("按选定方案执行", "normal hint") is False
    assert _hc._detect_conflict(None, "imagination") is False
    assert _hc._detect_conflict("按选定方案执行", None) is False

    # 场景 4: stagnation 触发 harmonic (即使 β_1=0)
    _p4, _ = _hc.coordinate(
        iter_n=2,
        csm_state="stagnation_detected",
        beta=(1, 0),
        last_verdict="fix_needed",
        fcm_winner=None,
        scan_text=None,
        step2_prompt="execute",
        iter_prompt=None,
        compass=None,
        step_eval=None,
        drift_info=None,
        imagination="imagination probe",
        meta_agent="meta_agent review",
    )
    assert "[topology probe]" in _p4, "stagnation should trigger harmonic"

    print("HintCoordinator self-check passed")
    print(f"  conflict prompt length: {len(_prompt)} chars")
    print(f"  events: {_events}")

    # === v14 Task 15: 跨 task darwin prior 影响 ===
    # case 1: prior attempted 跟 progress audit 内容重叠 >0.5 → boost 到 gradient 前面.
    # 用 Jaccard >0.5 选文本: hint="compute superradiance extraction" + prior="compute superradiance"
    # 交集={compute, superradiance}=2, 并集={compute, superradiance, extraction}=3 → 0.667.
    _prior1 = [
        {"attempted": "compute superradiance", "darwin_score": 0.9, "domain": "astronomy"}
    ]
    _p15a, _e15a = _hc.coordinate(
        iter_n=2,
        csm_state="execute",
        beta=(1, 1),  # β_1>0 触发 topology probe
        last_verdict="fix_needed",  # 触发 curl
        fcm_winner=None,
        scan_text=None,
        step2_prompt="execute methodology checklist",  # 不命中 prior
        iter_prompt=None,
        compass="compute superradiance extraction",  # 命中 prior → boost
        step_eval=None,
        drift_info=None,
        imagination="imagination probe different topic",  # 不命中 prior
        meta_agent=None,
        cross_task_prior=_prior1,
    )
    assert "[prior validated: darwin=0.90]" in _p15a, \
        f"case1 marker missing:\n{_p15a}"
    # boost 后 [progress audit] 应在 [gradient block] 之前
    assert _p15a.index("[prior validated: darwin=0.90]") < _p15a.index("[gradient block]"), \
        f"case1 boost order wrong:\n{_p15a}"
    assert any(e.startswith("cross_task_prior_boost:") for e in _e15a), \
        f"case1 boost event missing: {_e15a}"
    print(f"[CHECK v14 Task 15] case1 boost OK (events={_e15a})")

    # case 2: 跨 domain 隔离 — 调用方传空 list (CrossTaskStore 已按 domain 过滤掉)
    _p15b, _e15b = _hc.coordinate(
        iter_n=0,
        csm_state="execute",
        beta=(1, 1),
        last_verdict="fix_needed",
        fcm_winner=None,
        scan_text=None,
        step2_prompt="compute superradiance extraction",
        iter_prompt=None,
        compass="coverage check",
        step_eval=None,
        drift_info=None,
        imagination="imagination probe",
        meta_agent=None,
        cross_task_prior=[],  # 空 list = 本 domain 无历史 high darwin entry
    )
    assert "[prior validated" not in _p15b, \
        f"case2 should not boost when prior empty:\n{_p15b}"
    assert _e15b == [] or not any(
        e.startswith("cross_task_prior_boost:") for e in _e15b
    ), f"case2 should not emit boost event: {_e15b}"
    print("[CHECK v14 Task 15] case2 cross-domain isolation OK")

    # case 3: 关键词重叠 <0.5 不 boost
    # hint="compute orbital elements" + prior="compute quantum tunneling"
    # 交集={compute}=1, 并集={compute, orbital, elements, quantum, tunneling}=5 → 0.2.
    _prior3 = [
        {"attempted": "compute quantum tunneling", "darwin_score": 0.9, "domain": "astronomy"}
    ]
    _p15c, _e15c = _hc.coordinate(
        iter_n=0,
        csm_state="execute",
        beta=(1, 0),
        last_verdict=None,
        fcm_winner=None,
        scan_text=None,
        step2_prompt="compute orbital elements",  # 跟 prior 重叠 <0.5
        iter_prompt=None,
        compass=None,
        step_eval=None,
        drift_info=None,
        imagination=None,
        meta_agent=None,
        cross_task_prior=_prior3,
    )
    assert "[prior validated" not in _p15c, \
        f"case3 should not boost when overlap <0.5:\n{_p15c}"
    assert not any(
        e.startswith("cross_task_prior_boost:") for e in _e15c
    ), f"case3 should not emit boost event: {_e15c}"
    print("[CHECK v14 Task 15] case3 low overlap no-boost OK")

    print("v14 Task 15 self-check PASSED")

    # === v15 Phase 2 Task 4: posterior-guided hint ===
    # 用真 HypothesisManifold 验证: hint 非空 + ≤1500 chars + 失败降级 + coordinate 注入.
    from huginn.metacog.hypothesis_manifold import (
        Hypothesis,
        HypothesisManifold,
        Observation,
    )

    _m = HypothesisManifold()
    _m.add(Hypothesis(
        h_id="h_paper_repro",
        description="Paper results reproducible: metrics match claimed values",
        predictions={"mae": 0.5, "r2": 0.85, "accuracy": 0.90},
        n_params=2,
    ))
    _m.add(Hypothesis(
        h_id="h_partial_repro",
        description="Partial reproduction: metrics at 50% of claimed values",
        predictions={"mae": 0.25, "r2": 0.425, "accuracy": 0.45},
        n_params=3,
    ))
    _m.add(Hypothesis(
        h_id="h_null_baseline",
        description="Null/baseline result: no signal, default metrics",
        predictions={"mae": 0.0, "r2": 0.0, "accuracy": 0.0},
        n_params=1,
    ))
    _obs = [
        Observation("mae", 0.48, sigma=0.1),
        Observation("r2", 0.83, sigma=0.1),
        Observation("accuracy", 0.88, sigma=0.1),
    ]

    # case 1: 正常路径 — hint 非空, 含 core + explore, ≤1500 chars
    _hint = _build_posterior_guided_hint(_m, _obs)
    assert _hint, f"hint should not be empty with valid manifold+obs:\n{_hint}"
    assert "[posterior core hint]" in _hint, f"missing core hint:\n{_hint}"
    assert "best_hypothesis:" in _hint, f"missing best_hypothesis:\n{_hint}"
    assert "posterior_lift:" in _hint, f"missing lift:\n{_hint}"
    assert "[posterior explore hint]" in _hint, f"missing explore hint:\n{_hint}"
    assert len(_hint) <= 1500, (
        f"hint {len(_hint)} chars > 1500:\n{_hint}")
    print(f"[CHECK v15 Task 4] case1 hint OK ({len(_hint)} chars)")

    # case 2: 失败降级 — manifold=None → 空串
    assert _build_posterior_guided_hint(None, _obs) == "", \
        "manifold=None should return ''"
    # 空 observations → 空串
    assert _build_posterior_guided_hint(_m, []) == "", \
        "empty obs should return ''"
    # 空 manifold (无 hypothesis) → 空串
    _m_empty = HypothesisManifold()
    assert _build_posterior_guided_hint(_m_empty, _obs) == "", \
        "empty manifold should return ''"
    print("[CHECK v15 Task 4] case2 failure degradation OK")

    # case 3: coordinate 注入 — posterior_hint 出现在 prompt 最前面
    _ph = _build_posterior_guided_hint(_m, _obs)
    _p4, _e4 = _hc.coordinate(
        iter_n=1,
        csm_state="S4_CONSTRUCT",
        beta=(1, 0),
        last_verdict="pass",
        fcm_winner=None,
        scan_text=None,
        step2_prompt="execute methodology checklist",
        iter_prompt="Continue execution. Iteration 2/4.",
        compass=None,
        step_eval=None,
        drift_info=None,
        imagination=None,
        meta_agent=None,
        posterior_hint=_ph,
    )
    assert "[posterior core hint]" in _p4, (
        f"posterior_hint not injected:\n{_p4}")
    # posterior_hint 应在 [gradient block] 之前 (最高优先级)
    assert _p4.index("[posterior core hint]") < _p4.index("[gradient block]"), \
        f"posterior_hint should be before gradient block:\n{_p4}"
    print("[CHECK v15 Task 4] case3 coordinate injection OK")

    # case 4: posterior lift boost — entry 有 hypothesis_id + log_posterior, manifold 能查到
    # 选 h_paper_repro (best), 给它一个高 log_posterior 的历史 entry, 验证 boost marker
    _best_h = _m.abductive_inference(_obs)
    _log_post = _m.log_posterior(_obs).get(_best_h.h_id, 0.0)
    _prior_entries_v15 = [
        {
            "attempted": "compute band gap via DFT",
            "darwin_score": 0.9,
            "hypothesis_id": _best_h.h_id,
            "log_posterior": _log_post,
            "domain": "materials",
        }
    ]
    _p4b, _e4b = _hc.coordinate(
        iter_n=1,
        csm_state="S4_CONSTRUCT",
        beta=(1, 1),
        last_verdict="fix_needed",
        fcm_winner=None,
        scan_text=None,
        step2_prompt="execute methodology checklist",
        iter_prompt=None,
        compass="compute band gap via DFT extraction",
        step_eval=None,
        drift_info=None,
        imagination=None,
        meta_agent=None,
        cross_task_prior=_prior_entries_v15,
        manifold=_m,
    )
    assert any(e.startswith("posterior_lift_boost:") for e in _e4b), (
        f"posterior_lift_boost event missing: {_e4b}")
    assert "[prior validated: lift=" in _p4b, (
        f"lift marker missing:\n{_p4b}")
    print(f"[CHECK v15 Task 4] case4 posterior lift boost OK (events={_e4b})")

    # case 5: 失败降级到 v14 keyword overlap — entry 无 hypothesis_id (v14 entry)
    # 应走 keyword overlap 路径, emit cross_task_prior_boost (不是 posterior_lift_boost)
    _prior_entries_v14 = [
        {"attempted": "compute band gap", "darwin_score": 0.9, "domain": "materials"}
    ]
    _p4c, _e4c = _hc.coordinate(
        iter_n=1,
        csm_state="S4_CONSTRUCT",
        beta=(1, 1),
        last_verdict="fix_needed",
        fcm_winner=None,
        scan_text=None,
        step2_prompt="execute methodology checklist",
        iter_prompt=None,
        compass="compute band gap extraction",
        step_eval=None,
        drift_info=None,
        imagination=None,
        meta_agent=None,
        cross_task_prior=_prior_entries_v14,
        manifold=_m,  # manifold 可用但 entry 无 v15 字段 → 降级
    )
    # 应走 keyword overlap (无 v15 字段)
    assert not any(e.startswith("posterior_lift_boost:") for e in _e4c), (
        f"v14 entry should not trigger posterior_lift: {_e4c}")
    print("[CHECK v15 Task 4] case5 v14 fallback OK")

    # case 6: token 控制 — 极长 description 时 explore 段被截, 总长 ≤1500
    _m_long = HypothesisManifold()
    _m_long.add(Hypothesis(
        h_id="h_long_core",
        description="X" * 800,  # 极长 description
        predictions={"mae": 0.5},
        n_params=2,
    ))
    _m_long.add(Hypothesis(
        h_id="h_long_explore",
        description="Y" * 800,
        predictions={"mae": 0.0},
        n_params=1,
    ))
    _hint_long = _build_posterior_guided_hint(
        _m_long, [Observation("mae", 0.5, sigma=0.1)])
    assert len(_hint_long) <= 1500, (
        f"long hint {len(_hint_long)} > 1500:\n{_hint_long}")
    # 核心段必须保留
    assert "[posterior core hint]" in _hint_long, (
        f"core hint must survive truncation:\n{_hint_long}")
    print(
        f"[CHECK v15 Task 4] case6 token control OK "
        f"(truncated to {len(_hint_long)} chars)")

    print("v15 Task 4 self-check PASSED")
