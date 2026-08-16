"""RCB 认知/证据/假设流 — 依赖 rcb_utils."""
from __future__ import annotations

import contextlib
import json
import logging
import types
from pathlib import Path
from typing import Any

from huginn.cli.rcb_utils import (
    _METRIC_WHITELIST,
    _MODEL_VERSION,
    _NUMERIC_PAIR_RE,
    _extract_numeric_targets,
    _infer_domain,
    _load_manifold,
    _make_simplex_id,
    _save_manifold,
)
from huginn.utils.runtime import HUGINN_DIR_NAME

logger = logging.getLogger(__name__)


def _write_cognitive_evidence(
    ws,
    iter_n: int,
    *,
    entry: dict | None = None,
    pmk_state: dict | None = None,
    hypo_manifold=None,
    heat_engine=None,
    bandit_controller=None,
    completion_audit: dict | None = None,
    mcmc_info: dict | None = None,
    anomaly_info: dict | None = None,
    is_final: bool = False,
) -> None:
    """P0-C: 把认知层证据追加写入 ws/cognitive_evidence.md."""
    try:
        _ce_path = ws / "cognitive_evidence.md"
        _lines = []
        _tag = "FINAL" if is_final else f"iter_{iter_n + 1}"
        _lines.append(f"\n## Cognitive Evidence [{_tag}]\n")

        if entry:
            _darwin = float(entry.get("darwin_score", 0.5))
            _sup = float(entry.get("supported_ratio", 0.0))
            _lines.append(
                f"- darwin_score: {_darwin:.3f}  supported_ratio: {_sup:.3f}  "
                f"on_track: {entry.get('on_track', 'N/A')}"
            )
            _attempted = (entry.get("attempted") or "")[:150]
            if _attempted:
                _lines.append(f"- attempted: {_attempted}")

        if pmk_state:
            _lines.append(f"- PMK persona: {(pmk_state.get('persona') or '')[:100]}")
            _lines.append(f"- PMK memory: {(pmk_state.get('memory') or '')[:100]}")
            _lines.append(f"- PMK kb: {(pmk_state.get('kb') or '')[:100]}")

        if mcmc_info:
            _lines.append(
                f"- MCMC: accepted={mcmc_info.get('accepted', 'N/A')} "
                f"current_h_id={mcmc_info.get('h_id', 'N/A')} "
                f"llh={mcmc_info.get('llh', 'N/A')}"
            )

        if heat_engine is not None:
            try:
                _hk = getattr(heat_engine, "kinematics", None)
                if _hk:
                    _lines.append(
                        f"- heat: Re_cog={_hk.get('Re_cog', 0):.3f} "
                        f"T_hot={_hk.get('T_hot', 0):.3f} "
                        f"belief_entropy={_hk.get('belief_entropy', 0):.3f}"
                    )
            except Exception:
                logger.debug("heat engine evidence snapshot skipped", exc_info=True)

        if bandit_controller is not None:
            try:
                _vl = getattr(bandit_controller, "verified_lessons", None)
                if _vl:
                    _n_lessons = len(_vl)
                    _lines.append(f"- verified_lessons: {_n_lessons} patterns tracked")
            except Exception:
                logger.debug("verified_lessons evidence snapshot skipped", exc_info=True)

        if completion_audit:
            _lines.append(
                f"- completion_audit: passed={completion_audit.get('passed', 'N/A')} "
                f"gaps={completion_audit.get('n_gaps', 'N/A')} "
                f"coverage={completion_audit.get('coverage', 'N/A')}"
            )

        if anomaly_info:
            _lines.append(
                f"- isomorphic_anomaly: {len(anomaly_info.get('pairs', []))} pair(s), "
                f"{len(anomaly_info.get('generated', []))} new hypothesis"
            )

        if is_final:
            _lines.append("\n[final cognitive evidence snapshot]")

        with _ce_path.open("a", encoding="utf-8") as _f:
            _f.write("\n".join(_lines) + "\n")
    except Exception as _ce_e:
        print(f"[cognitive_evidence write skipped: {_ce_e}]", flush=True)


async def _trigger_anomaly_hypothesis(
    anomaly_pairs: list[tuple[str, str]], model,
) -> list[str]:
    """调 AutoloopEngine.trigger_isomorphic_anomaly_hypothesis 接通死代码."""
    if not anomaly_pairs:
        return []
    try:
        from huginn.autoloop.engine import AutoloopEngine

        async def _stub_hypothesize(ctx):
            if model is None:
                return None
            _prompt = ctx.get("summary", "")
            try:
                if hasattr(model, "chat"):
                    _resp = await model.chat(_prompt)
                elif hasattr(model, "ainvoke"):
                    _resp = await model.ainvoke(_prompt)
                else:
                    return None
            except Exception:
                logger.debug("best-effort op failed", exc_info=True)
                return None
            _txt = _resp if isinstance(_resp, str) else str(
                getattr(_resp, "content", _resp))
            return _txt.strip() or None

        _stub = types.SimpleNamespace(
            _hypothesize=_stub_hypothesize,
            _last_hypothesis=None,
            _last_raw_hypothesis=None,
        )
        return await AutoloopEngine.trigger_isomorphic_anomaly_hypothesis(
            _stub, anomaly_pairs)
    except Exception as _e:
        print(f"[anomaly] trigger failed: {_e}", flush=True)
        return []


def _init_hypothesis_manifold(
    *,
    ws: Path,
    task_id: str,
    checklist: str,
    instructions: Any,
    scan_text: str = "",
    model: Any = None,
    task_ctx: str = "",
):
    """v15 Phase 2 Task 3.1: 初始化 HypothesisManifold, 跨轮持久化."""
    from huginn.metacog.hypothesis_manifold import Hypothesis, HypothesisManifold

    path = ws / HUGINN_DIR_NAME / "hypothesis_manifold.jsonl"

    loaded = _load_manifold(path)
    if loaded is not None:
        return loaded

    text_pool = " ".join(filter(None, [
        checklist or "", str(instructions or ""), scan_text or "",
    ]))
    targets = _extract_numeric_targets(text_pool)

    manifold = HypothesisManifold()
    h_paper = Hypothesis(
        h_id="h_paper_repro",
        description="Paper results reproducible: metrics match claimed values",
        predictions=dict(targets),
        n_params=2,
    )
    h_partial = Hypothesis(
        h_id="h_partial_repro",
        description="Partial reproduction: metrics at 50% of claimed values",
        predictions={k: v * 0.5 for k, v in targets.items()},
        n_params=3,
    )
    h_null = Hypothesis(
        h_id="h_null_baseline",
        description="Null/baseline result: no signal, default metrics",
        predictions=dict.fromkeys(targets, 0.0),
        n_params=1,
    )
    for h in (h_paper, h_partial, h_null):
        with contextlib.suppress(ValueError):
            manifold.add(h)

    _save_manifold(manifold, path)
    return manifold


def _collect_observations(
    *,
    step_result: str,
    report_text: str = "",
    checklist: str = "",
) -> list:
    """v15 Phase 2 Task 3.2: 从 step_result / report_text 抓数值作为 observations."""
    from huginn.metacog.hypothesis_manifold import Observation

    observations: list = []
    text = f"{step_result or ''}\n{report_text or ''}"
    if not text.strip():
        return observations

    seen: set[str] = set()
    for m in _NUMERIC_PAIR_RE.finditer(text):
        name = m.group(1).lower()
        try:
            val = float(m.group(2))
        except ValueError:
            logger.debug("best-effort op failed", exc_info=True)
            continue
        if name not in _METRIC_WHITELIST:
            continue
        if abs(val) > 1e6:
            continue
        if name in seen:
            continue
        seen.add(name)
        observations.append(Observation(name=name, value=val, sigma=0.1))
    return observations


def _compute_v15_fields(manifold, observations) -> tuple[str | None, float, float]:
    """算 (hypothesis_id, log_posterior, fisher_info) 给 entry 用."""
    if manifold is None or not observations:
        return None, 0.0, 0.0
    try:
        best_h = manifold.abductive_inference(observations)
        if best_h is None:
            return None, 0.0, 0.0
        log_post_dict = manifold.log_posterior(observations)
        log_post = log_post_dict.get(best_h.h_id, 0.0)
        fisher_sum = 0.0
        fisher_n = 0
        for other_id in manifold._hyp:
            if other_id == best_h.h_id:
                continue
            fisher_sum += manifold.fisher_distance(best_h.h_id, other_id)
            fisher_n += 1
        fisher_info = fisher_sum / max(fisher_n, 1)
        return best_h.h_id, log_post, fisher_info
    except Exception:
        return None, 0.0, 0.0


def _record_abduction(
    manifold,
    observations,
    *,
    trace_path,
    task_id: str,
    iteration: int,
    ts: float,
) -> None:
    """v15 Phase 2 Task 3.3: 在 meta-trace 写一条 abduction entry."""
    if manifold is None or not observations:
        return
    try:
        best_h_id, log_post, fisher_info = _compute_v15_fields(
            manifold, observations)
        if best_h_id is None:
            return
        abd_entry = {
            "iteration": iteration,
            "ts": ts,
            "role": "abductive_inference",
            "attempted": f"abductive inference over {len(observations)} observations",
            "found": f"best_h={best_h_id} log_posterior={log_post:.3f}",
            "evidence": [
                f"obs_{i+1}: {o.name}={o.value:.4g}"
                for i, o in enumerate(observations[:5])
            ],
            "limitations": [],
            "artifacts": [],
            "next_hint": f"prior boost for {best_h_id}",
            "darwin_score": 0.0,
            "supported_ratio": 0.0,
            "simplex_id": _make_simplex_id(
                task_id, iteration, "abductive_inference"),
            "cochain_type": "gradient",
            "domain": _infer_domain(task_id),
            "task_id": task_id,
            "model_version": _MODEL_VERSION,
            "hypothesis_id": best_h_id,
            "log_posterior": log_post,
            "fisher_info": fisher_info,
            "imagination_parent": None,
        }
        with open(trace_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(abd_entry, ensure_ascii=False) + "\n")
    except Exception:
        logger.debug("best-effort op failed", exc_info=True)


def _append_observations_log(observations, path: Path, *, iteration: int) -> None:
    """observations 持久化到 .huginn/observations.jsonl, 跨轮累积."""
    if not observations:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            for o in observations:
                f.write(json.dumps({
                    "type": "observation",
                    "iteration": iteration,
                    "name": o.name,
                    "value": o.value,
                    "sigma": o.sigma,
                }, ensure_ascii=False) + "\n")
    except Exception:
        logger.debug("observations save skipped", exc_info=True)
