"""投机执行 —— 意图层 + 工具层.

用户历史是最懂用户的"廉价分布 q(x)". 这里做两件事:
  1. 意图层: 记录最近 N 轮 scenario 命中, 新对话开头预测 top-3 意图
  2. 工具层: 意图命中后预热对应工具的缓存条目 (prefetch)

让 LLM 只验证不确定的部分, 廉价先验交给历史 + 关键词.

不做上下文层 (RAG 预检索) 和 token 层 (speculative decoding).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


def _default_history_path() -> Path:
    """历史持久化路径, 跟 skill_evolver / logger 一样放 ~/.huginn/."""
    override = os.environ.get("HUGINN_SPECULATOR_HISTORY")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".huginn" / "speculator_history.json"


@dataclass
class Prediction:
    """单条意图预测结果."""

    scenario_name: str
    score: float
    recommended_tools: list[str]
    confidence: float


class IntentSpeculator:
    """意图层投机执行器.

    基于用户历史 scenario 命中记录, 预测下一轮最可能的意图,
    并预热对应工具的缓存条目, 让 LLM 只验证不确定的部分.

    用法::

        s = IntentSpeculator()
        s.record("dft_structure_optimization", "relax Si", ["vasp_tool"], True)
        preds = s.predict("relax GaN")  # top-3 预测
        s.prefetch(preds)               # 预热 top-1 工具缓存
    """

    _singleton_lock = threading.Lock()
    _singleton: IntentSpeculator | None = None

    def __init__(
        self,
        history_path: Path | str | None = None,
        max_history: int = 20,
    ) -> None:
        self._history_path = Path(history_path) if history_path else _default_history_path()
        self._max_history = max_history
        self._lock = threading.RLock()
        # 每条: timestamp / scenario_name / matched_query / tools_used / hit
        self._history: list[dict[str, Any]] = []
        # 每次 predict() 返回的条目数, 给 stats 算平均预测数
        self._prediction_counts: list[int] = []
        self._load()

    @classmethod
    def shared(cls) -> IntentSpeculator:
        """进程级单例, 避免每个调用方各读一遍历史文件."""
        with cls._singleton_lock:
            if cls._singleton is None:
                cls._singleton = cls()
            return cls._singleton

    # ------------------------------------------------------------------ 持久化

    def _load(self) -> None:
        if not self._history_path.exists():
            return
        try:
            raw = json.loads(self._history_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                self._history = list(raw.get("history", []))
                self._prediction_counts = list(raw.get("prediction_counts", []))
            elif isinstance(raw, list):
                # 老格式: 直接是 history 数组
                self._history = list(raw)
        except Exception as exc:
            logger.warning("speculator history load failed: %s", exc)
            self._history = []

    def _save(self) -> None:
        try:
            self._history_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "history": self._history[-self._max_history:],
                "prediction_counts": self._prediction_counts[-self._max_history:],
            }
            self._history_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("speculator history save failed: %s", exc)

    # ------------------------------------------------------------------ 公开 API

    def predict(self, query: str | None = None) -> list[Prediction]:
        """预测 top-3 意图. confidence < 0.4 的不返回.

        - 传了 query: 先用关键词表做廉价匹配 (不调 LLM), 再跟历史频率加权
        - 不传 query: 纯靠历史频率预测
        """
        with self._lock:
            hist = list(self._history)

        # ---- 历史频率加权 ----
        # 最近 5 轮权重 0.5, 6-20 轮 0.3, 更早 0.2
        hist_scores: dict[str, float] = {}
        total_weight = 0.0
        n = len(hist)
        for i, rec in enumerate(hist):
            # i=0 最老, i=n-1 最新; 越新权重越大
            age = n - 1 - i
            if age < 5:
                w = 0.5
            elif age < 20:
                w = 0.3
            else:
                w = 0.2
            total_weight += w
            name = rec.get("scenario_name", "")
            if name:
                hist_scores[name] = hist_scores.get(name, 0.0) + w

        # 归一化到 [0, 1]
        if total_weight > 0:
            for k in hist_scores:
                hist_scores[k] /= total_weight

        # ---- 关键词廉价匹配 ----
        query_match: tuple[str, float] | None = None
        if query:
            query_match = self._keyword_match(query)

        # ---- 合并分数: query 权重 0.6, 历史 0.4 ----
        all_scores: dict[str, float] = {}
        if query_match:
            matched_name, qscore = query_match
            for name, hscore in hist_scores.items():
                if name == matched_name:
                    all_scores[name] = 0.6 * qscore + 0.4 * hscore
                else:
                    all_scores[name] = 0.4 * hscore
            # 匹配到的场景即使没历史也要加上
            if matched_name not in all_scores:
                all_scores[matched_name] = 0.6 * qscore
        else:
            all_scores = dict(hist_scores)

        # ---- 排序取 top-3, 过滤低置信度 ----
        ranked = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)[:3]

        preds: list[Prediction] = []
        for name, score in ranked:
            if score < 0.4:
                continue
            tools = self._bundle_tools(name)
            preds.append(
                Prediction(
                    scenario_name=name,
                    score=round(score, 4),
                    recommended_tools=tools,
                    confidence=round(score, 4),
                )
            )

        # 记录本次预测条目数, 给 stats 算平均
        with self._lock:
            self._prediction_counts.append(len(preds))
            if len(self._prediction_counts) > self._max_history:
                self._prediction_counts = self._prediction_counts[-self._max_history:]
            self._save()

        return preds

    def record(
        self,
        scenario_name: str,
        query: str,
        tools_used: list[str],
        hit: bool,
    ) -> None:
        """记录一轮 scenario 命中, 更新历史. 每次 scenario 命中后调.

        hit 表示本次预测是否命中 (上层判断后回填).
        """
        with self._lock:
            self._history.append(
                {
                    "timestamp": time.time(),
                    "scenario_name": scenario_name,
                    "matched_query": query,
                    "tools_used": list(tools_used),
                    "hit": bool(hit),
                }
            )
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]
            self._save()

    def prefetch(
        self,
        predictions: list[Prediction],
        cache: Any | None = None,
    ) -> dict:
        """对 top-1 预测的工具 bundle 预热缓存.

        只对幂等无副作用的轻量工具 (structure_tool / materials_database_tool /
        symbolic_math_tool) 做, 重型仿真工具绝不 prefetch.

        返回 {"prefetched": [...], "skipped": [...], "errors": [...]}.
        """
        from huginn.tools.tool_cache import PREFETCH_SAFE_TOOLS, ToolCache

        if cache is None:
            cache = ToolCache.shared()

        result: dict[str, list] = {"prefetched": [], "skipped": [], "errors": []}
        if not predictions:
            return result

        # 只对 top-1 预测做 prefetch, 避免预热一堆用不上的
        top = predictions[0]
        safe_tools = [t for t in top.recommended_tools if t in PREFETCH_SAFE_TOOLS]

        for tool_name in safe_tools:
            common_inputs = self._common_inputs_for(tool_name)
            if not common_inputs:
                result["skipped"].append(tool_name)
                continue
            try:
                n = cache.prefetch(tool_name, common_inputs, runner=self._run_tool_sync)
                if n > 0:
                    result["prefetched"].append({"tool": tool_name, "count": n})
                else:
                    result["skipped"].append(tool_name)
            except Exception as exc:
                result["errors"].append({"tool": tool_name, "error": str(exc)})

        return result

    def stats(self) -> dict:
        """返回最近 N 轮的命中率、最常命中的 scenario、平均预测数."""
        with self._lock:
            hist = list(self._history)
            pred_counts = list(self._prediction_counts)

        if not hist:
            return {
                "hit_rate": 0.0,
                "total_records": 0,
                "most_common_scenario": None,
                "scenario_distribution": {},
                "avg_predictions": 0.0,
            }

        hits = sum(1 for r in hist if r.get("hit"))
        hit_rate = hits / len(hist)

        scenario_counts = Counter(
            r.get("scenario_name", "") for r in hist if r.get("hit")
        )
        most_common = scenario_counts.most_common(1)[0][0] if scenario_counts else None

        avg_preds = sum(pred_counts) / len(pred_counts) if pred_counts else 0.0

        return {
            "hit_rate": round(hit_rate, 4),
            "total_records": len(hist),
            "most_common_scenario": most_common,
            "scenario_distribution": dict(scenario_counts),
            "avg_predictions": round(avg_preds, 2),
        }

    # ------------------------------------------------------------------ 内部

    @staticmethod
    def _keyword_match(query: str) -> tuple[str, float] | None:
        """用 scenario_tool 的关键词表做廉价匹配, 不调 LLM.

        命中返回 (scenario_type, 0.7), 没命中返回 None.
        """
        from huginn.tools.scenario_tool import _KEYWORD_FALLBACK

        lowered = query.lower()
        for keywords, scenario_type in _KEYWORD_FALLBACK:
            for kw in keywords:
                # 中文关键词直接 in 匹配; 英文关键词用小写匹配
                if any(ord(c) > 127 for c in kw):
                    if kw in query:
                        return scenario_type, 0.7
                else:
                    if kw in lowered:
                        return scenario_type, 0.7
        return None

    @staticmethod
    def _bundle_tools(scenario_name: str) -> list[str]:
        """从 scenario_tool 的 bundle 表拿推荐工具列表."""
        from huginn.tools.scenario_tool import SCENARIO_TOOL_BUNDLES

        bundle = SCENARIO_TOOL_BUNDLES.get(scenario_name, {})
        return list(bundle.get("recommended_tools", []))

    @staticmethod
    def _common_inputs_for(tool_name: str) -> list[dict]:
        """每个安全工具的常见输入, 给 prefetch 用.

        只列幂等无副作用的读操作. 重型仿真工具不在这里.
        """
        if tool_name == "structure_tool":
            # 常见结构 analyze, file_path 传化学式, structure_tool
            # 内部会先查 local_structure_db 命中直接返回
            return [
                {"action": "analyze", "file_path": "Si"},
                {"action": "analyze", "file_path": "Cu"},
                {"action": "analyze", "file_path": "Fe"},
                {"action": "analyze", "file_path": "GaN"},
            ]
        if tool_name == "materials_database_tool":
            # Si / Cu 的 summary 和 structure 是最高频查询
            return [
                {"action": "mp_summary", "query": "Si", "limit": 1},
                {"action": "mp_summary", "query": "Cu", "limit": 1},
                {"action": "mp_structure", "query": "mp-149", "limit": 1},
                {"action": "mp_structure", "query": "mp-13", "limit": 1},
            ]
        if tool_name == "symbolic_math_tool":
            # 常见量纲分析和表达式化简
            return [
                {
                    "action": "dimensional_analysis",
                    "target": "check_equation",
                    "expression": "210 GPa = 500 MPa / 0.001",
                },
                {"action": "simplify", "expression": "E**2 - p**2*c**2"},
            ]
        return []

    def _run_tool_sync(self, tool_name: str, inp: dict) -> dict | None:
        """实际跑工具拿结果, 给 ToolCache.prefetch 当 runner.

        工具 call() 是 async, 这里同步跑. 如果已经在 event loop 里
        (比如 Jupyter), asyncio.run 会挂, 那种情况下用线程池兜底.
        """
        try:
            tool, args = self._instantiate_tool(tool_name, inp)
        except Exception as exc:
            logger.debug("prefetch instantiate %s failed: %s", tool_name, exc)
            return None

        try:
            from huginn.types import ToolContext

            ctx = ToolContext(
                session_id=f"prefetch_{tool_name}",
                workspace=str(Path.cwd()),
                config=None,
            )
            try:
                # 已经在 running loop 里 → 用线程跑, 避免 asyncio.run 报错
                asyncio.get_running_loop()
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    result = pool.submit(asyncio.run, tool.call(args, ctx)).result(
                        timeout=30
                    )
            except RuntimeError:
                # 没有 running loop, 直接 asyncio.run
                result = asyncio.run(tool.call(args, ctx))

            if hasattr(result, "data") and result.data is not None:
                return result.data if isinstance(result.data, dict) else {"data": result.data}
            return None
        except Exception as exc:
            logger.debug("prefetch run %s failed: %s", tool_name, exc)
            return None

    @staticmethod
    def _instantiate_tool(tool_name: str, inp: dict):
        """根据工具名实例化工具和参数."""
        if tool_name == "structure_tool":
            from huginn.tools.structure_tool import StructureTool, StructureToolInput

            return StructureTool(), StructureToolInput(**inp)
        if tool_name == "materials_database_tool":
            from huginn.tools.materials_database_tool import (
                MaterialsDatabaseInput,
                MaterialsDatabaseTool,
            )

            return MaterialsDatabaseTool(), MaterialsDatabaseInput(**inp)
        if tool_name == "symbolic_math_tool":
            from huginn.tools.symbolic_math_tool import (
                SymbolicMathInput,
                SymbolicMathTool,
            )

            return SymbolicMathTool(), SymbolicMathInput(**inp)
        raise ValueError(f"unknown tool for prefetch: {tool_name}")


def on_turn_start(user_msg: str, cache: Any | None = None) -> dict:
    """每轮对话开始时的投机执行钩子.

    给 engine 在 turn 开始时调, 拿 top-3 意图 + 预热工具缓存.
    返回::

        {
            "predictions": [...],     # top-3 Prediction dict
            "prefetch_result": {...}, # prefetch 跑了哪些工具
            "hint": str,              # 给 LLM 看的提示文字
        }

    失败不抛异常, 返回空结果. 预测只是 hint, LLM 可以无视.
    """
    # flag 关掉时直接返回空 hint, 不预测不预热
    try:
        from huginn.feature_flags import FeatureFlags
        if not FeatureFlags.shared().is_enabled("speculator"):
            return {
                "predictions": [],
                "prefetch_result": {"prefetched": [], "skipped": [], "errors": []},
                "hint": "",
            }
    except Exception:
        # flag 层挂了不能带挂业务, 继续走原逻辑
        pass

    speculator = IntentSpeculator.shared()
    try:
        preds = speculator.predict(user_msg)
    except Exception as exc:
        logger.debug("speculator predict failed: %s", exc)
        preds = []

    prefetch_result: dict = {"prefetched": [], "skipped": [], "errors": []}
    # top-1 confidence > 0.6 才预热, 避免低质量预测浪费算力
    if preds and preds[0].confidence > 0.6:
        try:
            prefetch_result = speculator.prefetch(preds, cache=cache)
        except Exception as exc:
            logger.debug("speculator prefetch failed: %s", exc)

    hint = ""
    if preds:
        top = preds[0]
        hint = (
            f"基于历史, 下一步可能要做 {top.scenario_name} "
            f"(confidence={top.confidence:.2f})"
        )
        prefetched = prefetch_result.get("prefetched", [])
        if prefetched:
            tools = [p["tool"] for p in prefetched]
            hint += f", 已预热工具: {', '.join(tools)}"

    return {
        "predictions": [
            {
                "scenario_name": p.scenario_name,
                "score": p.score,
                "recommended_tools": p.recommended_tools,
                "confidence": p.confidence,
            }
            for p in preds
        ],
        "prefetch_result": prefetch_result,
        "hint": hint,
    }
