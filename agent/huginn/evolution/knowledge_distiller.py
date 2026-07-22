"""Knowledge Distiller — converts execution experience into RAG knowledge.

Unlike LLM fine-tuning which changes model weights,
knowledge distillation adds facts to the Agent's retrievable memory.

Pipeline:
  Execution Logs → Pattern Extraction → Knowledge Facts → RAG Ingestion
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DistilledKnowledge:
    """A single piece of distilled knowledge ready for RAG."""

    knowledge_id: str
    content: str  # The actual text to store in RAG
    source_type: str  # "error_lesson", "success_pattern", "tool_tip", "domain_fact", "feynman_note"
    source_evidence: list[str]  # IDs of source logs that support this
    confidence: float = 0.5
    category: str = "general"  # For routing in hierarchical RAG
    tags: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    usage_count: int = 0
    verification_status: str = "unverified"  # "unverified", "confirmed", "rejected"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DistilledKnowledge:
        return cls(**data)


class KnowledgeDistiller:
    """Distills execution logs into structured knowledge for RAG."""

    def __init__(
        self,
        output_dir: str | None = None,
        sobkso_db_path: str | None = None,
        kb: Any = None,
    ):
        self.output_dir = (
            Path(output_dir) if output_dir else Path.home() / ".huginn" / "distilled"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sobkso_db_path = Path(sobkso_db_path) if sobkso_db_path else None
        self.knowledge_base: list[DistilledKnowledge] = []
        # F6: 蒸馏知识回写 KB — 之前蒸馏写 JSON 不进 KB, agent 检索不到结构化知识.
        # 现在每次 _save 后把新条目 add_text 到 KB, 闭合蒸馏→检索环.
        # kb=None 时 _save 里 lazy-load get_knowledge_base(), 失败只 warn.
        self._kb = kb
        self._kb_synced: set[str] = set()  # 已回写 KB 的 knowledge_id
        self._load_existing()
        # 已存在的条目视为已同步 (避免重启后重复写入)
        self._kb_synced.update(k.knowledge_id for k in self.knowledge_base)

    def _load_existing(self) -> None:
        kb_file = self.output_dir / "distilled_knowledge.json"
        if kb_file.exists():
            with kb_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
                self.knowledge_base = [
                    DistilledKnowledge.from_dict(item) for item in data
                ]

    def _save(self) -> None:
        kb_file = self.output_dir / "distilled_knowledge.json"
        with kb_file.open("w", encoding="utf-8") as f:
            json.dump(
                [k.to_dict() for k in self.knowledge_base],
                f,
                ensure_ascii=False,
                indent=2,
            )
        # F6: 回写 KB — 把新增的蒸馏知识 add_text 进主 KB, 让 agent 检索得到.
        # ponytail: lazy-load KB, 失败只 warn 不阻塞 _save 主路径.
        #   升级路径: KnowledgeDistiller 构造时注入 workspace 路径, 避免 lazy-load.
        try:
            self._writeback_to_kb()
        except Exception as exc:
            logger.warning("F6 KB writeback failed: %s", exc)

    def _writeback_to_kb(self) -> None:
        """把 self.knowledge_base 中未同步的条目写入主 KB."""
        if self._kb is None:
            try:
                from huginn.knowledge.store import get_knowledge_base
                # 没有 workspace 上下文时用默认 cache 路径 (~/.huginn)
                self._kb = get_knowledge_base()
            except Exception:
                return  # 无 KB 环境时静默跳过 (e.g. 离线 distill 命令)
        new_entries = [
            k for k in self.knowledge_base
            if k.knowledge_id not in self._kb_synced
        ]
        if not new_entries:
            return
        for k in new_entries:
            # filename 带 source_type + id 方便后续 cleanup_old_documents 按 source 追溯
            self._kb.add_text(
                text=k.content,
                filename=f"distilled_{k.source_type}_{k.knowledge_id}.txt",
                metadata={
                    "source_type": k.source_type,
                    "confidence": k.confidence,
                    "category": k.category,
                    "tags": ",".join(k.tags),
                    "distilled": "1",  # 标记蒸馏来源, 区分 agent _learn 直接写入
                },
            )
            self._kb_synced.add(k.knowledge_id)

    # ------------------------------------------------------------------
    # Distillation Methods
    # ------------------------------------------------------------------

    def _is_semantically_duplicate(self, content: str, threshold: float = 0.65) -> bool:
        """检查是否已有语义相似的蒸馏知识 (Jaccard 词集重叠).

        现有 knowledge_id (md5) 只防完全相同的 error, 不防措辞不同但
        内容相同的 lesson. 这里用 Jaccard 做轻量语义去重.
        ponytail: word-level Jaccard, ok for <500 entries;
        switch to embedding similarity if knowledge_base grows past 5K.
        """
        import re
        # 去标点再分词, 否则 "convergence:" 和 "convergence" 算不同词
        words_new = set(re.sub(r'[^\w\s]', '', content.lower()).split())
        if not words_new or len(words_new) < 4:
            return False
        for k in self.knowledge_base:
            words_old = set(re.sub(r'[^\w\s]', '', k.content.lower()).split())
            if not words_old:
                continue
            overlap = len(words_new & words_old)
            union = len(words_new | words_old)
            if union > 0 and overlap / union >= threshold:
                return True
        return False

    def distill_error_lessons(
        self, failure_logs: list[dict[str, Any]]
    ) -> list[DistilledKnowledge]:
        """Extract 'lessons learned' from failures."""
        new_knowledge = []
        for log in failure_logs:
            tool = log.get("tool_name", "unknown")
            error = log.get("error_message", "")
            software = log.get("software", "general")
            calc_type = log.get("calculation_type", "general")

            if not error:
                continue

            # Generate lesson text
            lesson = self._generate_error_lesson(tool, error, software, calc_type)
            if lesson:
                kid = f"err_{tool}_{hashlib.md5(error.encode()).hexdigest()[:8]}"
                if any(k.knowledge_id == kid for k in self.knowledge_base):
                    continue
                # 语义去重: 不同 error message 可能生成相同 lesson
                if self._is_semantically_duplicate(lesson):
                    continue

                dk = DistilledKnowledge(
                    knowledge_id=kid,
                    content=lesson,
                    source_type="error_lesson",
                    source_evidence=[log.get("session_id", "unknown")],
                    confidence=0.6,
                    category=f"troubleshooting_{software}",
                    tags=["error", "lesson", tool, software, calc_type],
                )
                self.knowledge_base.append(dk)
                new_knowledge.append(dk)

        if new_knowledge:
            self._save()
        return new_knowledge

    def distill_success_patterns(
        self, success_logs: list[dict[str, Any]]
    ) -> list[DistilledKnowledge]:
        """Extract successful parameter combinations as knowledge."""
        new_knowledge = []
        from collections import defaultdict

        grouped = defaultdict(list)

        for log in success_logs:
            key = f"{log.get('software', 'general')}_{log.get('calculation_type', 'general')}"
            grouped[key].append(log)

        for key, logs in grouped.items():
            if len(logs) < 2:
                continue

            software, calc_type = key.split("_", 1)
            # Find common successful parameters
            common_params = self._find_common_params(
                [log.get("tool_input", {}) for log in logs]
            )

            if common_params:
                content = f"Successful {calc_type} calculation with {software} uses these parameters: {json.dumps(common_params, ensure_ascii=False)}"
                kid = f"succ_{key}_{hashlib.md5(content.encode()).hexdigest()[:8]}"
                if any(k.knowledge_id == kid for k in self.knowledge_base):
                    continue

                dk = DistilledKnowledge(
                    knowledge_id=kid,
                    content=content,
                    source_type="success_pattern",
                    source_evidence=[
                        log.get("session_id", "unknown") for log in logs[:5]
                    ],
                    confidence=min(0.5 + len(logs) * 0.05, 0.95),
                    category=f"best_practices_{software}",
                    tags=["success", "pattern", software, calc_type],
                )
                self.knowledge_base.append(dk)
                new_knowledge.append(dk)

        if new_knowledge:
            self._save()
        return new_knowledge

    def distill_tool_tips(
        self, tool_logs: list[dict[str, Any]]
    ) -> list[DistilledKnowledge]:
        """Extract tool-specific tips and tricks."""
        new_knowledge = []
        from collections import defaultdict

        by_tool = defaultdict(list)

        for log in tool_logs:
            by_tool[log.get("tool_name", "unknown")].append(log)

        for tool, logs in by_tool.items():
            if len(logs) < 3:
                continue

            failures = [log for log in logs if not log.get("success")]
            successes = [log for log in logs if log.get("success")]

            # Compare failures vs successes to extract tips
            if failures and successes:
                tip = self._compare_to_tip(tool, failures, successes)
                if tip:
                    kid = f"tip_{tool}_{int(time.time() * 1000)}"
                    dk = DistilledKnowledge(
                        knowledge_id=kid,
                        content=tip,
                        source_type="tool_tip",
                        source_evidence=[
                            log.get("session_id", "unknown") for log in logs[:5]
                        ],
                        confidence=0.7,
                        category=f"tips_{tool}",
                        tags=["tip", tool],
                    )
                    self.knowledge_base.append(dk)
                    new_knowledge.append(dk)

        if new_knowledge:
            self._save()
        return new_knowledge

    def distill_domain_facts(
        self, conversations: list[dict[str, Any]]
    ) -> list[DistilledKnowledge]:
        """Extract domain facts from successful conversations."""
        new_knowledge = []
        for conv in conversations:
            user_msg = conv.get("user_message", "")
            agent_resp = conv.get("agent_response", "")
            topic_tags = conv.get("topic_tags", [])

            # Extract factual statements
            facts = self._extract_facts(user_msg, agent_resp)
            for fact in facts:
                kid = f"fact_{hashlib.md5(fact.encode()).hexdigest()[:8]}"
                if any(k.knowledge_id == kid for k in self.knowledge_base):
                    continue

                dk = DistilledKnowledge(
                    knowledge_id=kid,
                    content=fact,
                    source_type="domain_fact",
                    source_evidence=[conv.get("session_id", "unknown")],
                    confidence=0.5,
                    category=topic_tags[0] if topic_tags else "general",
                    tags=topic_tags,
                )
                self.knowledge_base.append(dk)
                new_knowledge.append(dk)

        if new_knowledge:
            self._save()
        return new_knowledge

    # ------------------------------------------------------------------
    # Feynman learning: 用通俗语言重组知识, 暴露理解缺口
    # ------------------------------------------------------------------

    def store_feynman_note(
        self,
        explanation: str,
        gaps: list[str],
        *,
        iteration: int = 0,
        hypothesis: str = "",
        tags: list[str] | None = None,
        confidence: float = 0.7,
    ) -> str:
        """存储 Feynman 式教学笔记到蒸馏知识库.

        Feynman 学习法: 如果不能用简单语言解释清楚, 就没真正理解.
        explanation 是 agent 用通俗语言写的解释, gaps 是解释不出来的部分.
        feynman_note 在 KB 检索时获得优先级 (基础概念 > 细节技巧).
        """
        import hashlib

        content_parts = [f"# Feynman Note (iter {iteration})"]
        if hypothesis:
            content_parts.append(f"## Hypothesis\n{hypothesis[:200]}")
        content_parts.append(f"## Simple Explanation\n{explanation}")
        if gaps:
            content_parts.append("## Knowledge Gaps\n" + "\n".join(f"- {g}" for g in gaps))
        content = "\n\n".join(content_parts)

        kid = f"feynman_{iteration}_{hashlib.md5(content.encode()).hexdigest()[:8]}"
        dk = DistilledKnowledge(
            knowledge_id=kid,
            content=content,
            source_type="feynman_note",
            source_evidence=[f"autoloop_iter_{iteration}"],
            confidence=confidence,
            category="feynman",
            tags=tags or ["feynman", "autoloop"],
        )
        # 去重: 同 iteration 不重复存
        existing = {k.knowledge_id for k in self.knowledge_base}
        if kid not in existing:
            self.knowledge_base.append(dk)
            self._save()
        return kid

    # ------------------------------------------------------------------
    # G8: Visual lessons distillation
    # ------------------------------------------------------------------

    def distill_visual_lessons(
        self,
        visual_entries: list[dict],
        *,
        min_support: int = 2,
    ) -> list[str]:
        """G8: 从历史 visual_primitives 蒸馏可复用的视觉经验.

        visual_entries 是 [{"tool_name", "primitives", "ts", "context"}] 列表.
        蒸馏逻辑:
          1. 按 tool_name 分组 (band_structure / EDS / phase_field / ...)
          2. 每组 ≥ min_support 条 → 抽一个 visual_lesson
          3. visual_lesson 内容: 该类图的常见特征 (高频关键词) + 出现次数

        ponytail: 关键词频率统计, 不上 LLM 抽取. 升级路径: 调 LLM 做语义聚类.
          ceiling: 不识别语义相似 (peak/max/min 会被当 3 个不同词).
        """
        if not visual_entries or len(visual_entries) < min_support:
            return []

        # 按 tool_name 分组
        groups: dict[str, list[dict]] = {}
        for e in visual_entries:
            tool = e.get("tool_name", "unknown")
            groups.setdefault(tool, []).append(e)

        kids: list[str] = []
        for tool, entries in groups.items():
            if len(entries) < min_support:
                continue

            # 抽 primitives 里的关键词频率
            from collections import Counter
            import re
            word_freq: Counter = Counter()
            for e in entries:
                prim = e.get("primitives", "") or ""
                # 提取 <point>/<box>/<band> 等 visual primitive 标签
                tags = re.findall(r"\[(\w+)\]|<(\w+)>", prim)
                for t in tags:
                    for g in t:
                        if g:
                            word_freq[g] += 1
                # 提取峰/异常/趋势等关键词
                for kw in ("peak", "min", "max", "anomal", "trend", "increasing", "decreasing"):
                    if kw in prim.lower():
                        word_freq[kw] += 1

            if not word_freq:
                continue

            top_features = word_freq.most_common(5)
            features_str = ", ".join(f"{k}({v})" for k, v in top_features)

            content_parts = [
                f"# Visual Lesson ({tool})",
                f"## Observed Pattern (n={len(entries)})",
                f"Common visual features in {len(entries)} {tool} results:",
                f"- Top features: {features_str}",
            ]
            # 附 2-3 个原始 primitives 作例子
            examples = entries[:3]
            for i, e in enumerate(examples):
                prim = (e.get("primitives", "") or "")[:200]
                content_parts.append(f"\nExample {i+1}:\n{prim}")
            content = "\n".join(content_parts)

            kid = f"visual_lesson_{tool}_{hashlib.md5(content.encode()).hexdigest()[:8]}"
            if any(k.knowledge_id == kid for k in self.knowledge_base):
                continue

            dk = DistilledKnowledge(
                knowledge_id=kid,
                content=content,
                source_type="visual_lesson",
                source_evidence=[e.get("ts", "unknown") for e in entries],
                confidence=min(0.8, 0.4 + 0.1 * len(entries)),
                category=f"visual_{tool}",
                tags=["visual", "lesson", tool],
            )
            self.knowledge_base.append(dk)
            kids.append(kid)

        if kids:
            self._save()
        return kids

    # ------------------------------------------------------------------
    # Export to RAG
    # ------------------------------------------------------------------

    def export_to_rag_format(self, output_path: str | None = None) -> str:
        """Export distilled knowledge in a format ready for ChromaDB ingestion."""
        if output_path is None:
            output_path = self.output_dir / "rag_ready_chunks.jsonl"

        with open(output_path, "w", encoding="utf-8") as f:
            for dk in self.knowledge_base:
                chunk = {
                    "id": dk.knowledge_id,
                    "text": dk.content,
                    "metadata": {
                        "source_type": dk.source_type,
                        "category": dk.category,
                        "confidence": dk.confidence,
                        "tags": dk.tags,
                        "created_at": dk.created_at,
                    },
                }
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

        return str(output_path)

    def merge_with_sobko_db(self, sobkso_chunks_path: str | None = None) -> str:
        """Merge distilled knowledge with existing Sobko database."""
        if sobkso_chunks_path is None:
            # Try to find Sobko chunks
            candidates = [
                os.environ.get("HUGINN_KB_CHUNKS_PATH", ""),
                os.path.join(os.environ.get("HUGINN_CACHE_DIR", ".huginn"), "kb_chunks.jsonl"),
            ]
            candidates = [c for c in candidates if c]  # filter empty
            for c in candidates:
                if Path(c).exists():
                    sobkso_chunks_path = c
                    break

        merged_path = self.output_dir / "merged_with_sobko.jsonl"

        # Write distilled knowledge first
        with merged_path.open("w", encoding="utf-8") as out:
            for dk in self.knowledge_base:
                chunk = {
                    "id": f"evolved_{dk.knowledge_id}",
                    "text": dk.content,
                    "metadata": {
                        "source": "agent_evolution",
                        "source_type": dk.source_type,
                        "category": dk.category,
                        "confidence": dk.confidence,
                        "tags": dk.tags,
                    },
                }
                out.write(json.dumps(chunk, ensure_ascii=False) + "\n")

            # Append existing Sobko chunks if available
            if sobkso_chunks_path and Path(sobkso_chunks_path).exists():
                with open(sobkso_chunks_path, encoding="utf-8") as sobkso:
                    for line in sobkso:
                        line = line.strip()
                        if line:
                            out.write(line + "\n")

        return str(merged_path)

    def auto_ingest_to_kb(self, kb=None) -> int:
        """Auto-ingest confirmed distilled knowledge into the KnowledgeBase.

        This bridges the Memory→KB gap: distilled experience (error lessons,
        success patterns, tool tips, domain facts) becomes RAG-retrievable,
        so future queries can benefit from past experience.

        Only ingests knowledge with verification_status == "confirmed" or
        confidence >= 0.7 to avoid polluting the KB with low-quality entries.

        Returns the number of newly ingested chunks.
        """
        if kb is None:
            try:
                from huginn.knowledge.store import get_knowledge_base

                kb = get_knowledge_base()
            except Exception:
                return 0
        if kb is None:
            return 0

        ingested = 0
        for dk in self.knowledge_base:
            # Only ingest high-quality knowledge
            if dk.verification_status == "rejected":
                continue
            if (
                dk.verification_status != "confirmed"
                and dk.confidence < 0.7
            ):
                continue
            try:
                kb.add_text(
                    text=dk.content,
                    metadata={
                        "source": "distilled_knowledge",
                        "source_type": dk.source_type,
                        "category": dk.category,
                        "confidence": dk.confidence,
                        "knowledge_id": dk.knowledge_id,
                        "tags": dk.tags,
                    },
                )
                ingested += 1
            except Exception as e:
                logger.debug("auto_ingest_to_kb chunk failed: %s", e)
                continue
        return ingested

    def verify_knowledge(
        self, knowledge_id: str, status: str = "confirmed"
    ) -> bool:
        """Update verification status of a distilled knowledge entry.

        Enables the self-correction loop: knowledge that gets recalled
        and leads to successful outcomes is promoted to "confirmed";
        knowledge that gets contradicted is marked "rejected".
        """
        for dk in self.knowledge_base:
            if dk.knowledge_id == knowledge_id:
                dk.verification_status = status
                if status == "confirmed":
                    dk.usage_count += 1
                    dk.confidence = min(1.0, dk.confidence + 0.1)
                elif status == "rejected":
                    dk.confidence = max(0.0, dk.confidence - 0.3)
                self._save()
                return True
        return False

    def _generate_error_lesson(
        self, tool: str, error: str, software: str, calc_type: str
    ) -> str | None:
        """Generate a human-readable lesson from an error."""
        error_lower = error.lower()

        templates = {
            "scf": f"When performing {calc_type} with {software}, SCF convergence failures often indicate: (1) poor initial geometry, (2) inappropriate basis set, or (3) problematic electronic structure. Try: switching algorithms, using convergence aids, or checking the initial guess.",
            "basis": f"{software} basis set errors in {calc_type} usually mean missing basis functions for certain elements. Verify all elements have appropriate basis sets, especially for heavy atoms or transition metals.",
            "memory": f"{software} memory errors during {calc_type} can be resolved by: increasing parallel cores (NCORE/KPAR for VASP), reducing system size, or using mixed-precision modes.",
            "timestep": f"In {software} MD simulations, 'lost atoms' or integration errors typically mean the timestep is too large. Reduce timestep by half and increase neighbor list skin distance.",
            "license": f"{software} license errors prevent execution. Check license server status and ensure your institution has valid licenses.",
            "timeout": f"{software} {calc_type} calculations may timeout for large systems. Consider: reducing k-points, using cheaper methods, or breaking into smaller subproblems.",
        }

        for key, template in templates.items():
            if key in error_lower:
                return template

        # Generic lesson
        return f"Error encountered in {software} {calc_type}: {error[:200]}. Review input parameters and consider consulting documentation for this specific error type."

    def _find_common_params(self, inputs: list[dict[str, Any]]) -> dict[str, Any]:
        """Find parameter values that appear in multiple successful runs."""
        if not inputs:
            return {}

        from collections import Counter

        param_values = {}
        for inp in inputs:
            for key, val in inp.items():
                if key.startswith("_"):
                    continue
                param_values.setdefault(key, []).append(str(val))

        common = {}
        for key, vals in param_values.items():
            if len(vals) >= 2:
                most_common = Counter(vals).most_common(1)[0]
                if most_common[1] >= len(inputs) * 0.5:  # Appears in >50% of cases
                    common[key] = most_common[0]

        return common

    def _compare_to_tip(
        self, tool: str, failures: list[dict], successes: list[dict]
    ) -> str | None:
        """Compare failure vs success patterns to generate a tip."""
        fail_params = self._find_common_params(
            [f.get("tool_input", {}) for f in failures]
        )
        succ_params = self._find_common_params(
            [s.get("tool_input", {}) for s in successes]
        )

        differences = []
        for key in set(fail_params.keys()) | set(succ_params.keys()):
            f_val = fail_params.get(key)
            s_val = succ_params.get(key)
            if f_val and s_val and f_val != s_val:
                differences.append(
                    f"{key}: failures use '{f_val}', successes use '{s_val}'"
                )

        if differences:
            return f"Tip for {tool}: " + "; ".join(differences)
        return None

    def _extract_facts(self, user_msg: str, agent_resp: str) -> list[str]:
        """Extract factual statements from conversation."""
        facts = []
        # Simple heuristic: extract sentences that look like definitions or explanations
        for text in [user_msg, agent_resp]:
            sentences = text.split("。")
            for sent in sentences:
                sent = sent.strip()
                if len(sent) > 20 and any(
                    kw in sent
                    for kw in [
                        "is a",
                        "are used",
                        "requires",
                        "depends on",
                        "calculated",
                        "is defined",
                        "refers to",
                        "consists of",
                        "based on",
                    ]
                ):
                    facts.append(sent)
        return facts[:3]  # Limit to top 3 per conversation
