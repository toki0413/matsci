"""认知环路的无副作用纯函数辅助模块.

从 ``cognitive_loop.py`` (198KB) 拆出的段 1: 只含纯函数 / 常量, 不触碰
engine 状态、I/O 之外的共享副作用. 这些函数被 autoloop 各引擎与 RCB
(rcb_step2) 共用, 拆出后 cognitive_loop 通过 import 再导出, 对外契约不变
(``from huginn.autoloop.cognitive_loop import darwin_ratchet_check`` 仍可用).

拆出原则:
- 只搬"给定参数 → 返回结果"的纯函数, 无 self / 无 engine 状态读写.
- 文件 I/O / 记忆 / 指标等有副作用者留在 cognitive_loop (段 2+ 再处理).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _extract_tests_passed(validation: Any) -> bool:
    """从 validation 结果里抽 tests_passed 布尔, 给 validate→learn 门用.

    validation 形状不固定 (dict / str / None), 抽不出明确失败就默认 True,
    避免门控把现有 happy path 误阻断. 只有明确说 fail / passed=False 才拦.
    """
    if isinstance(validation, dict):
        for key in ("tests_passed", "passed", "success", "ok"):
            if key in validation:
                return bool(validation[key])
        return True
    if isinstance(validation, str):
        low = validation.lower()
        return "fail" not in low
    # None 或其它: 没有明确失败信号, 放行
    return True


def _derive_light_on_track(action: str, cog: dict) -> str:
    """Loop: 每轮轻量 on_track 派生 — 不限 validate, 用机械信号.

    返回 "true" / "false" / "unsure", 供连续偏移累积检测. 不调 LLM
    (轻量), 单轮无产出的硬 redirect 由 reflect_fn 现有逻辑处理, 这里
    只补一个跨轮累积信号给 decider 预警 "有产出但连续无实质进展".

    - observe:  有 observation 上下文 → true, 否则 unsure (observe 很难判失败)
    - hypothesize: 有 hypothesis → true, 无 → false
    - plan:     有 plan → true, 无 → false
    - execute:  有 execution_result → true, 无 → false
    - validate: tests_passed → true, 否则 false (复用 _extract_tests_passed)
    - learn/pivot/skip/stop: unsure (非产出型 action)
    """
    if action == "observe":
        return "true" if cog.get("context") else "unsure"
    if action == "hypothesize":
        return "true" if cog.get("hypothesis") else "false"
    if action == "plan":
        return "true" if cog.get("plan") else "false"
    if action == "execute":
        return "true" if cog.get("execution_result") is not None else "false"
    if action == "validate":
        return "true" if _extract_tests_passed(cog.get("validation")) else "false"
    return "unsure"


def _snapshot_structure_desc(cog: dict) -> list[float]:
    """从 cog 提取结构描述符, 缺失填 16 维全 0.

    episodic 快照里带一份结构向量, 供后续按空间检索. 当前 cog 不存
    StructureCognitiveMap, 所以现在基本回退全 0; 将来 cog 挂上 cmap 后
    这段就是入口, 不用改调用方.
    """
    try:
        from huginn.metacog.structure_descriptor import StructureDescriptor
        cmap = cog.get("structure_cognitive_map") or cog.get("cmap") or cog.get("structure")
        if cmap is None:
            return [0.0] * 16
        vec = StructureDescriptor().encode(cmap)
        return [float(x) for x in vec]
    except Exception:
        return [0.0] * 16


# 已知 _validate dict 字段 (autoloop engine.py _validate 方法返回):
#   tests_passed: bool
#   constraints_satisfied: bool
#   benchmarks: dict (key: benchmark_name, val: {metric: value})
#   r_phys / physics_validation / physics_validation_error
#   thinking_collapse / thinking_collapse_error
#   test_output / math_validation / math_validation_error / math_evidence_error
#   generative_verify / generative_verify_error
#   reviewer_critique / reviewer_critique_error
#   grader_scores / grader_reward / grader_error
#   eval_summary / prediction_error
#   effort_floor_passed / effort_floor_deficits / failure_kind
#   emergent_complexity / literature_comparison / ...
_VALIDATION_ERROR_KEYS: tuple[str, ...] = (
    "thinking_collapse",
    "physics_validation_error",
    "math_validation_error",
    "math_evidence_error",
    "generative_verify_error",
    "reviewer_critique_error",
    "grader_error",
    "prediction_error",
    "test_output",
    "effort_floor_deficits",
)


def _validation_to_step_eval_fields(
    validation: dict,
    tests_ok: bool,
    execution_result: Any,
    *,
    step_id: int,
) -> dict:
    """P0.2: _validate dict → StepEvaluation 兼容字段映射.

    之前 reflect_fn 硬取 summary/result/errors 三个不存在的字段, 导致
    _step_eval.attempted/found/deviation 全空, PMK/drift/metrics 吃空数据.
    这里集中映射, 测试能直接验.

    attempted: execution_result 的 description (validation 里没有).
    found: tests_passed + benchmarks 关键指标摘要.
    deviation: 失败时收集所有 *_error / thinking_collapse / effort_floor_deficits.
    evidence_quality: tests_ok=high, 否则 low; 有 *_error 降级.
    structure_check: tests_ok=passed, 否则 failed.

    ponytail: dict 字段映射, 不上 schema 库. 升级路径: pydantic model.
    """
    # attempted: validation 里没有 attempted, 从 execution_result 拼
    # G6: PMK 的 K 查询看不到 visual_primitives → 加 visual 摘要让 KB 能召回视觉经验.
    _attempted = ""
    if isinstance(execution_result, dict):
        _attempted = str(
            execution_result.get("description")
            or execution_result.get("summary")
            or execution_result.get("result_type")
            or ""
        )[:200]
    # G6: 附 visual_primitives 摘要 (前 150 字), 让 PMK 的 K 查询能看到视觉内容.
    # ponytail: 直接拼字符串, 不上 schema. 升级路径: StepEvaluation 加 visual_context 字段.
    _vis = validation.get("visual_primitives") if isinstance(validation, dict) else None
    if _vis and isinstance(_vis, str):
        _attempted = (f"{_attempted} | Visual: {_vis[:150]}").strip(" |")

    # found: tests_passed 状态 + benchmarks 关键指标
    _found_parts: list[str] = []
    if tests_ok:
        _found_parts.append("tests_passed=True")
    else:
        _found_parts.append("tests_passed=False")
    _benches = validation.get("benchmarks") or {}
    if isinstance(_benches, dict):
        for _bn, _bv in list(_benches.items())[:3]:
            if isinstance(_bv, dict):
                _metric = _bv.get("metric") or _bv.get("value") or ""
                if _metric:
                    _found_parts.append(f"{_bn}={_metric}")
            else:
                _found_parts.append(f"{_bn}={_bv}")
    _found = "; ".join(_found_parts)[:200]

    # deviation: 失败时收集错误信号
    _dev_parts: list[str] = []
    if not tests_ok:
        for _k in _VALIDATION_ERROR_KEYS:
            _v = validation.get(_k)
            if _v:
                _dev_parts.append(f"{_k}: {str(_v)[:100]}")
    _deviation = "; ".join(_dev_parts)[:300]

    # evidence_quality: tests_ok=high, 否则 low
    _eq = "high" if tests_ok else "low"
    if _dev_parts:
        _eq = "low"

    return {
        "step_id": step_id,
        "attempted": _attempted,
        "found": _found,
        "on_track": "true" if tests_ok else "false",
        "evidence_quality": _eq,
        "deviation": _deviation,
        "structure_check": "passed" if tests_ok else "failed",
        "pmk_feedback": "",
        "tool_call_health": None,
        "target_chain_ref": None,
    }


def darwin_ratchet_check(
    darwin_score: float,
    supported_ratio: float,
    stagnation_count: int,
    stagnation_limit: int = 3,
    hypothesis_graph: Any = None,
    speculator_hint: str | None = None,
) -> tuple[str, dict]:
    """RCB 可直接调用的 darwin ratchet 检查. 返回 (action, state_dict).

    action ∈ {"continue", "pivot", "counterexample", "stop"}
    - continue:      stagnation 未达 limit
    - pivot:         stagnation >= limit, 方法问题, 换 hypothesis
    - counterexample:stagnation >= limit, 证据指向 hypothesis 错
    - stop:          stagnation >= limit, 调节无效

    RCB 路径不传 hypothesis_graph 时, 退化为只看 darwin_score / supported_ratio.
    ponytail: 无 _last_failure_mode / _consecutive_failures 这些 engine 状态,
    用 supported_ratio + darwin_score 阈值做粗分类. ceiling: 真 stall 归因要看
    failure trace, 升级时让 RCB 把 failure trace 传进来.
    """
    if stagnation_count < stagnation_limit:
        return "continue", {
            "stagnation_count": stagnation_count,
            "darwin_score": darwin_score,
            "supported_ratio": supported_ratio,
            "speculator_hint": speculator_hint,
        }

    # stagnation 达 limit → 分类 stall (无 history, 用 snapshot 阈值)
    if darwin_score < 2.0:
        action = "stop"
    elif supported_ratio < 0.3:
        action = "counterexample"
    else:
        action = "pivot"

    new_stagnation = stagnation_count
    new_hint = speculator_hint
    if action == "pivot":
        new_stagnation = 0  # 给 pivot 后的新假设重新累积
    elif action == "counterexample":
        new_stagnation = 0
        # 注入 counterexample hint, 调用方下轮 hypothesize 会读到
        new_hint = (speculator_hint or "") + (
            "\n[Darwin ratchet] stagnation 达 limit 且 supported_ratio 偏低, "
            "当前 hypothesis 可能错. Hunt for a counterexample: 构造一个具体 "
            "scenario / 参数集让该 hypothesis 失败, 若找到则 refute 并 pivot."
        )

    return action, {
        "stagnation_count": new_stagnation,
        "darwin_score": darwin_score,
        "supported_ratio": supported_ratio,
        "speculator_hint": new_hint,
    }


def classify_stall(
    darwin_score_history: list[float],
    supported_ratio_history: list[float],
    hypothesis_graph: Any = None,
) -> str:
    """RCB 可直接调用的 stagnation 分类. 返回 stall_type.

    stall_type ∈ {"pivot", "counterexample", "stop"}
    - pivot:          idea pool 枯竭, score 平台, 换 hypothesis
    - counterexample: supported_ratio 下滑或低位, 找反例
    - stop:           score 持续低位, 调节无效

    无 hypothesis_graph 时, 用 score history 形状分类.
    ponytail: 只看 last-3 形状, 不做完整 trend analysis. ceiling: 6+ 点的
    Bayesian change-point detection.
    """
    n = len(darwin_score_history)
    if n == 0:
        return "pivot"  # 没历史, 默认换 hypothesis

    last_score = darwin_score_history[-1]
    # 1. 持续低位 → stop
    if n >= 3 and all(s < 2.0 for s in darwin_score_history[-3:]):
        return "stop"
    if last_score < 1.5:
        return "stop"

    # 2. supported_ratio 下滑或低位 → counterexample
    if supported_ratio_history:
        last_ratio = supported_ratio_history[-1]
        if last_ratio < 0.3:
            return "counterexample"
        # 下滑检测: 最近 2 个点都比前 2 个点低
        if len(supported_ratio_history) >= 4:
            recent = supported_ratio_history[-2:]
            prev = supported_ratio_history[-4:-2]
            if all(r < p for r, p in zip(recent, prev)):
                return "counterexample"

    # 3. score 平台 → pivot (idea pool 枯竭, 换方向)
    if n >= 3:
        last3 = darwin_score_history[-3:]
        if max(last3) - min(last3) < 0.5:
            return "pivot"

    # 4. 默认 → pivot (stagnation 已触发, 换 hypothesis 是合理默认)
    return "pivot"


def metacog_check_completion(
    report_md: str,
    outputs_dir: str | Path,
    hypothesis_graph: Any = None,
    derivation_chain: list[str] | None = None,
) -> dict:
    """RCB 可直接调用的完成度检查. 返回 4 层审计结果.

    返回 {
        "derivation_chain_coverage": float,  # 0.0-1.0
        "topology_health": float,           # 0.0-1.0
        "repro_evidence": float,            # 0.0-1.0
        "evidence_strength": float,         # 0.0-1.0
        "passed": bool,                     # 所有层 >= 0.5 才 True
        "block_reasons": list[str],
    }

    ponytail: 4 层用文件 / 字符串启发式算, 不调 LLM judge. ceiling: 真 audit
    调 CompletionAuditor + EquivalenceAuditor. RCB 路径无 engine state,
    退化为 report / outputs_dir / graph 三源启发式.
    """
    reasons: list[str] = []

    # L1: derivation_chain_coverage — chain 中有多少 step 在 report 里被提到
    if derivation_chain:
        mentioned = sum(
            1 for step in derivation_chain
            if step and step[:50] in report_md
        )
        cov = mentioned / len(derivation_chain)
    else:
        # 没 chain 传进来, 退化为: report 里有没有 derivation / derive 字样
        _low = report_md.lower()
        cov = 1.0 if ("derivation" in _low or "derive" in _low) else 0.3
    if cov < 0.5:
        reasons.append(f"derivation_chain_coverage={cov:.2f} < 0.5")

    # L2: topology_health — hypothesis_graph 连通分量数 / 节点数
    if hypothesis_graph is not None:
        try:
            n_nodes = len(hypothesis_graph.all_nodes())
            n_comp = hypothesis_graph.component_count()
            # 0 节点 → 0; 否则 [0,1] 标准化, 越多分量越健康
            topo = (n_comp / n_nodes) if n_nodes > 0 else 0.0
        except Exception:
            logger.debug("best-effort op failed", exc_info=True)
            topo = 0.5  # 不阻断, advisory
    else:
        topo = 1.0  # 没 graph, 默认不阻断
    if topo < 0.5:
        reasons.append(f"topology_health={topo:.2f} < 0.5")

    # L3: repro_evidence — outputs_dir 里有文件 (.png/.csv/.json/.txt/.parquet)
    try:
        out_path = Path(outputs_dir)
        if out_path.exists() and out_path.is_dir():
            files = list(out_path.rglob("*"))
            evidence_files = [
                f for f in files
                if f.is_file() and f.suffix.lower() in
                {".png", ".csv", ".json", ".txt", ".parquet", ".pdf", ".jpg"}
            ]
            # 5+ 文件 → 1.0; 线性插值
            repro = min(1.0, len(evidence_files) / 5.0)
        else:
            repro = 0.0
    except Exception:
        logger.debug("best-effort op failed", exc_info=True)
        repro = 0.5
    if repro < 0.5:
        reasons.append(f"repro_evidence={repro:.2f} < 0.5")

    # L4: evidence_strength — report 里引用 / ref / citation 数
    try:
        # ponytail: 字符串计数, 不上正则. ceiling: 真 citation parser.
        _low = report_md.lower()
        cite_hits = (
            _low.count("[") + _low.count("ref:")
            + _low.count("cite") + _low.count("see ")
        )
        # 10+ 引用 → 1.0
        strength = min(1.0, cite_hits / 10.0)
    except Exception:
        logger.debug("best-effort op failed", exc_info=True)
        strength = 0.5
    if strength < 0.5:
        reasons.append(f"evidence_strength={strength:.2f} < 0.5")

    passed = (cov >= 0.5 and topo >= 0.5
              and repro >= 0.5 and strength >= 0.5)
    return {
        "derivation_chain_coverage": cov,
        "topology_health": topo,
        "repro_evidence": repro,
        "evidence_strength": strength,
        "passed": passed,
        "block_reasons": reasons,
    }


def metacog_check_topology_collapse(
    hypothesis_graph: Any = None,
    hypothesis_list: list | None = None,
) -> dict:
    """RCB 可直接调用的拓扑坍缩检测. 返回检测结果.

    返回 {
        "collapsed": bool,
        "diversity_score": float,  # 0.0-1.0, 越低越坍缩
        "reason": str | None,
    }

    无 hypothesis_graph 时, 用 hypothesis_list (List[Hypothesis]) 计算.
    ponytail: 多样性用 unique statements / total 算. ceiling: 真 diversity
    要看 statement 语义嵌入的 cosine 距离, 这里只看字面重复.
    """
    # 路径 1: hypothesis_graph 优先
    if hypothesis_graph is not None:
        try:
            n_nodes = len(hypothesis_graph.all_nodes())
            n_comp = hypothesis_graph.component_count()
            if n_nodes == 0:
                return {"collapsed": True, "diversity_score": 0.0,
                        "reason": "hypothesis_graph 为空 (无节点)"}
            # ponytail: 用 component_count / n_nodes 当 diversity 代理.
            # ceiling: 用 statement 语义嵌入算真 diversity.
            diversity = n_comp / n_nodes
            collapsed = hypothesis_graph.is_collapsed(min_components=2)
            reason = None
            if collapsed:
                reason = (f"连通分量 {n_comp} < 2, 搜索空间坍缩 "
                          f"(n_nodes={n_nodes})")
            return {
                "collapsed": collapsed,
                "diversity_score": diversity,
                "reason": reason,
            }
        except Exception as exc:
            return {"collapsed": False, "diversity_score": 0.5,
                    "reason": f"hypothesis_graph 检测异常: {exc}"}

    # 路径 2: hypothesis_list (List[Hypothesis] or List[dict])
    if hypothesis_list is not None:
        statements = []
        for h in hypothesis_list:
            stmt = getattr(h, "statement", None)
            if stmt is None and isinstance(h, dict):
                stmt = h.get("statement", "")
            if stmt:
                statements.append(stmt)
        if not statements:
            return {"collapsed": True, "diversity_score": 0.0,
                    "reason": "hypothesis_list 为空或无 statement"}
        unique = len(set(statements))
        diversity = unique / len(statements)
        collapsed = diversity < 0.5
        reason = None
        if collapsed:
            reason = (f"unique statements {unique}/{len(statements)} "
                      f"< 0.5, 假设同质化")
        return {
            "collapsed": collapsed,
            "diversity_score": diversity,
            "reason": reason,
        }

    # 路径 3: 无 graph 无 list → 不坍缩 (兜底)
    return {"collapsed": False, "diversity_score": 1.0, "reason": None}


def metacog_check_selection_bias(
    observations: list | None = None,
    hint: str | None = None,
) -> dict:
    """RCB / autoloop 可直接调用的选择偏差检测. 返回检测结果.

    返回 {
        "biased": bool,
        "bias_type": str,   # closed_outcome / single_group / unbalanced_group / ""
        "n": int,
        "reason": str | None,
        "hint": str,        # 命中时给 agent 的可读提示
    }

    喂一组观察记录 (每条可带 outcome / group 字段), 检测样本是否系统性缺了
    一类 (幸存者偏差 / 隐藏子群体). 无 observations 或不足以判定时返回不命中.

    ponytail: 委托 metacog.selection_bias.detect_selection_bias, 纯启发式.
    hint 传入时 (调用方已持有独立 hint) 直接透出, 否则用 detect 给的内置 hint.
    """
    if not observations:
        return {"biased": False, "bias_type": "", "n": 0,
                "reason": None, "hint": hint or ""}
    try:
        from huginn.metacog.selection_bias import detect_selection_bias
        v = detect_selection_bias(observations)
        return {
            "biased": v.biased,
            "bias_type": v.bias_type,
            "n": v.n,
            "reason": v.hint if v.biased else None,
            "hint": hint or v.hint,
        }
    except Exception as exc:
        return {"biased": False, "bias_type": "", "n": 0,
                "reason": f"selection bias 检测异常: {exc}", "hint": hint or ""}