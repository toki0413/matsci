"""EnginePerceiveMixin — AutoloopEngine 的 perceive 阶段方法族.

从 engine.py 拆出 (P3 slim-down 续). 包含 perceive 阶段实现 + 上下文构建
辅助方法 (KB/KG/memory/PM/metacog block 拼装). 通过 self 访问 engine 状态.

设计原则 (ponytail):
- 方法体原样搬迁, 不改逻辑
- 对 engine.py 模块级符号 (helper 函数 / 常量) 用方法内 lazy import, 避免 circular
- Mixin 不持有自己的状态, 全部走 self
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


class EnginePerceiveMixin:
    """perceive 阶段方法族: 感知工作区 + 上下文块拼装. 通过 self 访问 engine 状态."""

    def _maybe_expire_inbox(self) -> None:
        """P2-5: 周期清理超时 pending inbox item, 防 unattended session 永久阻塞.

        跟 _maybe_save_engine_state 同节奏调 (复用 save_every_steps 的 N).
        inbox 不可用 (None / import 失败) 时静默跳过 — 不阻塞主循环.
        ponytail: 失败只 log debug, 不抛.
        """
        try:
            from huginn.interaction.inbox import get_inbox_store
            store = get_inbox_store()
            if store is None:
                return
            expired = store.expire_pending()
            if expired:
                logger.debug(
                    "inbox expire_pending dropped %d timed-out items (iter=%d)",
                    expired, self._iteration,
                )
        except Exception:
            logger.debug("_maybe_expire_inbox failed (non-fatal)", exc_info=True)

    def _get_perception(self):
        """懒加载 PerceptionLayer — 长生命周期, start 后持续监听文件/日志事件.

        一旦创建就在后台持续运行, _perceive() 只取 snapshot 不再 start/stop.
        析构时由 GC 或显式 stop() 回收线程.
        """
        if self._perception is None:
            try:
                from huginn.perception import PerceptionLayer

                self._perception = PerceptionLayer(self.workspace)
                self._perception.start()
            except Exception:
                return None
        return self._perception

    def _get_persona_manager(self):
        """懒加载 PersonaManager, 实例化时才扫描 persona 文件."""
        if self._persona_manager is None:
            from huginn.personas import PersonaManager

            self._persona_manager = PersonaManager(workspace=self.workspace)
        return self._persona_manager

    def _get_kb(self):
        """懒加载领域知识库. ChromaDB 或 seed 文件不可用时返回 None,
        调用方需自行判空."""
        if self._kb is None:
            try:
                from huginn.knowledge.store import get_knowledge_base

                self._kb = get_knowledge_base(str(self.workspace))
            except Exception:
                return None
        return self._kb

    def _extract_search_query(self, context: dict[str, Any]) -> str:
        """从 context 提取有意义的检索词, 不直接 dump JSON.

        之前用 json.dumps(context)[:500] 做 KB/KG/memory 三路检索的 query,
        但 JSON 语法噪声 (引号/括号/key名/timestamp) 会淹没语义锚点,
        embedding 质量差. 只取 objective + 文件名 + error 关键词.
        """
        parts: list[str] = []
        # objective 是最重要的语义锚点
        obj = context.get("objective") or context.get("goal") or ""
        if not obj:
            obj = getattr(self, "_objective", "") or ""
        if obj:
            parts.append(obj[:200])
        # 文件名暗示领域 (diffusion_analysis.py → 扩散)
        for f in (context.get("changed_files") or [])[:3]:
            name = str(f).split()[-1] if " " in str(f) else str(f)
            parts.append(name)
        # error 关键词
        for e in (context.get("error_patterns") or [])[:2]:
            parts.append(str(e)[:100])
        if not parts:
            # 兜底: 没有可提取字段时回退到 JSON (总比空 query 好)
            return json.dumps(context, ensure_ascii=False)[:200]
        return " ".join(parts)[:400]

    def _build_kb_text(self, query: str) -> str:
        """检索领域知识库, 把命中 chunk 拼成 prompt 上下文块. KB 没装、
        空、查询失败都返回空串, 不影响 loop. 用 query_with_dedup 去重,
        避免分块重叠导致的近似重复段落浪费 token."""
        if not query:
            return ""
        # 按需注入: 非知识类 query 跳过 KB 检索, 省 token 和注意力
        from huginn.context_builder import should_inject_kb
        if not should_inject_kb(query):
            return ""
        kb = self._get_kb()
        if kb is None:
            return ""
        try:
            if kb.count() == 0:
                return ""
            # 优先用带去重的检索
            if hasattr(kb, "query_with_dedup"):
                chunks = kb.query_with_dedup(query, top_k=5)
            else:
                chunks = kb.query(query, top_k=5)
            if not chunks:
                return ""
            # C1: 走共享 format_kb_chunks — 跟 ContextBuilder 同一路径, 含
            # image_ref + KB→memory cross-ref. 之前 engine 版没这俩, 是双路径漂移.
            from huginn.context_builder import format_kb_chunks
            recall_fn = None
            mem = getattr(self, "memory", None)
            if mem is not None:
                recall_fn = mem.recall_for_prompt
            body = format_kb_chunks(
                chunks,
                memory_recall_fn=recall_fn,
                with_image_ref=True,
                cross_ref_top_k=2,
            )
            if not body:
                return ""
            return (
                "### Domain Knowledge Context\n"
                "The following first-principles reference chunks may ground your "
                "hypothesis and plan. Cite source numbers when relevant.\n"
                f"{body}\n"
                "### End Domain Knowledge Context"
            )
        except Exception:
            return ""

    def _build_kg_text(self, query: str) -> str:
        """检索知识图谱, 把相关实体+关系拼成 prompt 上下文块.
        KG 没建、空、查询失败都返回空串. 这是 KG 读回闭环的关键 —
        _learn 写入的实体, _hypothesize/_plan 要能检索到."""
        if not query:
            return ""
        kg = getattr(self, "kg", None)
        if kg is None:
            return ""
        try:
            result = kg.query(query, depth=1, top_k=8)
            nodes = result.get("nodes") or []
            if not nodes:
                return ""
            lines = []
            for node in nodes[:8]:
                data = node.get("data", node)
                label = data.get("label", node.get("id", ""))
                etype = data.get("type", "")
                conf = data.get("confidence", 0)
                lines.append(f"- [{etype}] {label} (conf={conf:.2f})")
                # 把出边也带上
                for edge in (node.get("edges") or [])[:3]:
                    rel = edge.get("relation", "→")
                    dst = edge.get("dst_label", edge.get("dst", ""))
                    lines.append(f"  {rel} → {dst}")
            if not lines:
                return ""
            body = "\n".join(lines)
            # KG 缺口检测: 找 A-B 有边、B-C 有边、但 A-C 无边的三元组
            # 建议假设 "A 是否也和 C 有关?" — 这是 KG 主动驱动探索的关键.
            gap_hints = self._detect_kg_gaps(kg, nodes)
            gap_block = ""
            if gap_hints:
                gap_block = (
                    "\n\n### KG Gap Detection (potential research directions)\n"
                    + "\n".join(gap_hints)
                    + "\n"
                )
            return (
                "### Knowledge Graph Context\n"
                "Previously discovered entities and relations from prior runs:\n"
                f"{body}\n"
                "### End Knowledge Graph Context"
                f"{gap_block}"
            )
        except Exception:
            return ""

    def _detect_kg_gaps(self, kg: Any, nodes: list[dict]) -> list[str]:
        """检测 KG 中的知识缺口: A-B 有边, B-C 有边, 但 A-C 无边.
        返回 "Consider whether {A} also relates to {C}" 格式的提示.
        用 NetworkX 的 common_neighbors, 零依赖."""
        try:
            graph = getattr(kg, "_graph", None)
            if graph is None or graph.number_of_nodes() < 3:
                return []
            import networkx as nx

            hints: list[str] = []
            # 只检查高置信度节点 (conf > 0.5)
            high_conf_nodes = []
            for nid, data in graph.nodes(data=True):
                if data.get("confidence", 0) > 0.5:
                    high_conf_nodes.append(nid)
            # 对每对高置信度节点, 检查是否有共同邻居但彼此无边
            checked = 0
            for i, a in enumerate(high_conf_nodes[:10]):
                for b in high_conf_nodes[i + 1 : 10]:
                    if graph.has_edge(a, b) or graph.has_edge(b, a):
                        continue  # 已有边, 不是缺口
                    common = (
                        set(nx.common_neighbors(graph, a, b))
                        if graph.has_node(a) and graph.has_node(b)
                        else set()
                    )
                    if common:
                        a_label = graph.nodes[a].get("label", a)[:40]
                        b_label = graph.nodes[b].get("label", b)[:40]
                        hints.append(
                            f"- {a_label} and {b_label} share connections but no direct link — consider whether they relate"
                        )
                        checked += 1
                    if checked >= 3:
                        break
                if checked >= 3:
                    break
            return hints
        except Exception:
            return []

    def _build_memory_text(self, query: str, since: str | None = None) -> str:
        """检索长期记忆, 把跨会话的教训/发现拼成 prompt 上下文块.
        Memory 之前只写不读 — _learn 写入的迭代记录和失败教训,
        下轮 hypothesize/plan 完全看不到. 这个函数闭合了 memory 读回环.
        查询失败/空结果返回空串, 不影响 prompt.
        since: 可选 ISO 8601 时间戳, 只召回该时间之后写入的记忆.
            autoresearch 场景用于限定到本轮/本会话的知识窗口."""
        if not query:
            return ""
        parts: list[str] = []
        mem = getattr(self, "memory", None)
        if mem is not None:
            try:
                text = mem.recall_for_prompt(query, max_entries=3, since=since)
                if text and isinstance(text, str):
                    parts.append(text)
            except Exception:
                logger.debug("memory recall_for_prompt skipped", exc_info=True)

        # C4: meta_trace 注入 — engine 每轮 _distill_meta_trace 写 jsonl,
        # 这里读回来让 LLM 看到上轮蒸馏的结构化历史 (attempted/found/evidence/
        # limitations/next_hint). toggle off 时不注入, 避免 prompt 膨胀.
        try:
            from huginn.autoloop.engine import _autoloop_meta_trace_inject_enabled
        except ImportError:
            def _autoloop_meta_trace_inject_enabled():
                return False  # type: ignore[return-value]
        if _autoloop_meta_trace_inject_enabled():
            try:
                from huginn.context_builder import load_meta_trace_text
                ws = getattr(self, "workspace", None)
                if ws is not None:
                    trace_text = load_meta_trace_text(str(ws), last_n=5)
                    if trace_text:
                        parts.append(trace_text)
            except Exception:
                logger.debug("meta_trace inject failed", exc_info=True)

        return "\n\n".join(parts) if parts else ""

    def _build_pm_text(self) -> str:
        """C2: PM 层 trajectory_match 召回 — 当前 phase 序列是否是某历史轨迹的 prefix.

        命中 → 注入 next_step 建议给 LLM 参考. 同时记下命中 doc_id,
        供 _learn 调 update_pattern_confidence 做 ±ε 反馈 (C3 闭环).
        返回空串时不影响 prompt.

        启用条件: HUGINN_EXTREME_DISPATCH=1, 或长程任务 (max_iterations >= 20).
        与 engine_reflect 的 cycle/trajectory 检测共用同一触发语义 — 避免重复犯错
        对任何长程任务都有价值, 不该只属于 extreme. 短程任务默认关省计算.
        """
        import os
        # 长程任务 (max_iterations >= 20) 默认开 trajectory 召回,
        # 短程任务仍需 HUGINN_EXTREME_DISPATCH=1 才开 (省计算).
        _max_iter = getattr(self, "_max_iterations", 10)
        _extreme = os.environ.get("HUGINN_EXTREME_DISPATCH", "0").lower() in ("1", "true")
        if not (_extreme or _max_iter >= 20):
            return ""
        try:
            from huginn.knowledge.trajectory_pattern import trajectory_match
            current = getattr(self, "_current_run_phases", [])
            history = getattr(self, "_traj_history", None)
            if history is None:
                # 懒加载历史轨迹 (跟 _check_stuck 共用)
                try:
                    history = self._load_trajectory_action_history(limit=20)
                    self._traj_history = history
                except Exception:
                    return ""
            if len(current) < 2 or not history:
                return ""
            match = trajectory_match(current, history, min_similarity=0.4)
            if match is None:
                self._last_traj_match_doc_id = None
                self._last_traj_match_run_id = None
                return ""
            # C3 闭环: 记下 run_id 供 _learn 反查 KB doc_id 调 update_pattern_confidence.
            # _traj_run_ids 是 _load_trajectory_action_history 填充的平行数组,
            # history_id 是它的索引.
            hid = match.get("history_id")
            self._last_traj_match_doc_id = hid
            run_ids = getattr(self, "_traj_run_ids", [])
            self._last_traj_match_run_id = (
                run_ids[hid] if hid is not None and hid < len(run_ids) else None
            )
            advice = (
                f"### Trajectory Match (PM layer)\n"
                f"Current phase sequence matches history[{match['history_id']}] "
                f"(similarity={match['similarity']:.2f}). "
                f"Suggested next step: {match.get('next_step', '?')}. "
                f"This is advisory — ignore if context differs.\n"
                f"### End Trajectory Match"
            )
            return advice
        except Exception:
            self._last_traj_match_doc_id = None
            self._last_traj_match_run_id = None
            return ""

    def _ensure_target_chains(self) -> list:
        """C2: lazy build target_chains from self._objective.

        首次调用时把 objective 当单条 Mode-A item 调 build_target_chains,
        LLM 推导 required_results/methods/data/verification. 失败容错返回空 list.
        之后只读 self._target_chains. 跟 rcb_runner 的 _target_chains 路径同源.

        ponytail: objective 当单条 Mode-A item — RCBench 是 checklist 多条,
          autoloop 通常单 objective. 升级路径: 让 run_cognitive 接收 checklist.
        """
        # getattr 防 __new__ 测试场景 (跟 D1 的 _speculator_hint 同模式)
        if getattr(self, "_target_chains_built", False):
            return getattr(self, "_target_chains", [])
        self._target_chains_built = True  # 防重入, 失败也不重试
        objective = getattr(self, "_objective", "") or ""
        if not objective.strip():
            self._target_chains = []
            return self._target_chains
        try:
            from huginn.metacog.target_chain import build_target_chains
            kb = self._get_kb()
            model = getattr(self, "model", None)
            if model is None:
                self._target_chains = []
                return self._target_chains
            checklist = [{"mode": "A", "item": objective[:2000]}]
            self._target_chains = build_target_chains(checklist, kb, model, "") or []
        except Exception:
            logger.debug("build_target_chains failed in autoloop", exc_info=True)
            self._target_chains = []
        return self._target_chains

    def _build_metacog_block(self, *, include_prospective: bool = True) -> str:
        """C2: target_chain + prospective 注入 prompt.

        target_chain: 当前 objective 在目标分解树里的位置 (required_results /
        missing / progress), LLM 看了能避免偏题.
        prospective: 已触发的待执行前瞻意图, LLM 看了能避免遗漏计划.
        两者都空时返回空串, 不污染 prompt.

        ponytail: target_chain 首次调用触发 build_target_chains (1 次 LLM),
          之后只读. prospective 每次调 recall_prospective, 但 PM 层内部有
          scan_and_fire 缓存, 代价可控. 升级路径: 把两者合并到 metacog signal.
        """
        parts: list[str] = []
        tc = self._ensure_target_chains()
        if tc:
            try:
                from huginn.metacog.target_chain import format_target_chain_text
                step = getattr(self, "_iteration", 0) or 0
                tc_text = format_target_chain_text(tc, step)
                if tc_text:
                    parts.append(tc_text)
            except Exception:
                logger.debug("format_target_chain_text failed", exc_info=True)

        if include_prospective:
            mem = getattr(self, "memory", None)
            if mem is not None and hasattr(mem, "recall_prospective"):
                try:
                    step = getattr(self, "_iteration", 0) or 0
                    fired = mem.recall_prospective({"current_step": step})
                    if fired:
                        from huginn.context_builder import ContextBuilder
                        # 用 ContextBuilder 的格式化逻辑, 跟 rcb_runner 同源.
                        # ponytail: __new__ 绕过 __init__, 只用 build_prospective_text.
                        _ctx = ContextBuilder.__new__(ContextBuilder)
                        pro_text = _ctx.build_prospective_text(fired)
                        if pro_text:
                            parts.append(pro_text)
                except Exception:
                    logger.debug("prospective inject failed", exc_info=True)

        return "\n\n".join(p for p in parts if p)

    def _perceive(self) -> dict[str, Any] | None:
        """Perceive the workspace using the multi-modal perception layer.

        The PerceptionLayer is now a long-lived member started in __init__,
        so background watchers and log tailers actually accumulate events
        between iterations. Previously we started+stopped it here, which
        killed the watcher threads before they could collect anything.
        """
        perception = self._get_perception()
        if perception is None:
            return self._perceive_legacy()
        try:
            snapshot = perception.get_snapshot()
        except Exception:
            return self._perceive_legacy()
        context = snapshot.to_context()
        if not snapshot.has_activity():
            return None
        # L3/L4: 语义对齐 + 认知整合, 把冲突和推荐动作塞进 context
        try:
            cog = perception.get_cognitive_state()
            if cog.conflicts:
                context["semantic_conflicts"] = [
                    {"sources": [c.source_a, c.source_b], "description": c.description}
                    for c in cog.conflicts
                ]
            if cog.recommended_actions:
                context["recommended_actions"] = cog.recommended_actions
            if cog.recommended_tools:
                context["recommended_tools"] = cog.recommended_tools
            if cog.simulation_converged is not None:
                context["simulation_converged"] = cog.simulation_converged
            # G10/F14: perception 信号经 SignalHub 路由成 TransitionSignal.
            # ponytail: engine 无 csm 引用，走 _pending_signals 解耦；升级路径是 engine 注入 csm 直接 transition
            try:
                from huginn.metacog.signal_hub import SignalHub
                hub = SignalHub.shared()
                if getattr(cog, "errors_present", False):
                    sig = hub.route("perception_error", {"errors_present": True})
                    if sig is not None:
                        self._pending_signals.append(sig)
                if getattr(cog, "conflicts", None):
                    sig = hub.route("perception_conflict", {
                        "conflicts": [
                            {"sources": [c.source_a, c.source_b], "description": c.description}
                            for c in cog.conflicts
                        ],
                    })
                    if sig is not None:
                        self._pending_signals.append(sig)
                if getattr(cog, "simulation_converged", None) is True:
                    sig = hub.route("perception_converged", {"converged": True})
                    if sig is not None:
                        self._pending_signals.append(sig)
            except Exception:
                logger.debug("perception 信号构造失败, 不阻断 _perceive", exc_info=True)
        except Exception:
            logger.debug("L3/L4 cognitive integration 失败", exc_info=True)
        return context

    def _perceive_legacy(self) -> dict[str, Any] | None:
        """Legacy perceive (fallback)."""
        changed_files = []
        git_diff = ""
        try:
            import subprocess

            result = subprocess.run(
                ["git", "status", "--short"],
                cwd=self.workspace,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                changed_files = [
                    line.strip() for line in result.stdout.strip().split("\n")
                ]
                git_diff = subprocess.run(
                    ["git", "diff", "--stat"],
                    cwd=self.workspace,
                    capture_output=True,
                    text=True,
                    timeout=10,
                ).stdout
        except Exception:
            logger.warning(
                "error in _perceive_legacy: git diff collection failed", exc_info=True
            )
        error_patterns = []
        for log_file in self.workspace.rglob("*.log"):
            if log_file.stat().st_mtime > time.time() - 3600:
                try:
                    content = log_file.read_text(errors="ignore")
                    if "ERROR" in content or "FAIL" in content:
                        error_patterns.append(f"{log_file.name}: {content[:200]}")
                except Exception:
                    logger.warning(
                        "error in _perceive_legacy: log file error-pattern scan failed",
                        exc_info=True,
                    )
        if not changed_files and not error_patterns:
            return None
        return {
            "changed_files": changed_files,
            "git_diff": git_diff,
            "error_patterns": error_patterns,
            "timestamp": datetime.now().isoformat(),
        }
