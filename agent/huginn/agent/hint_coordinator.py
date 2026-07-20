"""Huginn hint coordinator — Hodge 正交分解把 14 个 hint 降到 5.

v14 Phase 1 Task 5: Step 2 每轮原本无仲裁叠加 14 hint, context 被占满
且可能冲突. 这里按 spec §"hint 分类"把 14 hint 分到 gradient/curl/
harmonic/retrieval/deprecated 五族, 合并出 3 个 prompt 块 + 1 个由
调用方拼的 Meta-Trace top-k 块.

ponytail: 不引新依赖, 不上策略模式, 单文件单类. 14 hint 的分类是设计
意图启发 (spec §诚实边界 4), 不是数学推导.
"""

from __future__ import annotations


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

        ponytail: 调用方一次拿全 3 块, 自己决定是否再拼 Meta-Trace.
        """
        events: list[str] = []
        blocks: list[str] = []

        # --- gradient 族 (必需) ---
        # iter_prompt 是 iter>0 的 "continue execution" 文本, 跟 step2_prompt
        # 在 rcb_runner 里互斥 (iter 0 直接复用 step2_prompt). 取非空的那个.
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

        # --- v14 Task 15: 跨 task darwin prior boost ---
        # 对每个 block, 找 cross_task_prior 中关键词重叠最高的 entry, >0.5 时
        # 加 [prior validated: darwin=X.XX] 标记并移到 blocks 前面 (boost 优先级).
        # gradient 族因构建顺序天然在前, 多块同时命中时仍保持原相对顺序.
        if cross_task_prior:
            _boosted: list[str] = []
            _rest: list[str] = []
            for _blk in blocks:
                # 剥掉 "[gradient block]\n" header, 只用正文算重叠
                _hint_text = _blk.split("\n", 1)[1] if "\n" in _blk else _blk
                _best_overlap = 0.0
                _best_darwin = 0.0
                for _entry in cross_task_prior:
                    _att = _entry.get("attempted") or ""
                    _ov = self._keyword_overlap(_hint_text, _att)
                    if _ov > _best_overlap:
                        _best_overlap = _ov
                        _best_darwin = float(_entry.get("darwin_score") or 0.0)
                if _best_overlap > 0.5:
                    _boosted.append(
                        f"[prior validated: darwin={_best_darwin:.2f}]\n" + _blk
                    )
                    events.append(
                        f"cross_task_prior_boost:overlap={_best_overlap:.2f}"
                    )
                else:
                    _rest.append(_blk)
            blocks = _boosted + _rest

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
