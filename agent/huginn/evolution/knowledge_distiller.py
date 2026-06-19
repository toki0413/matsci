"""Knowledge Distiller — converts execution experience into RAG knowledge.

Unlike LLM fine-tuning which changes model weights,
knowledge distillation adds facts to the Agent's retrievable memory.

Pipeline:
  Execution Logs → Pattern Extraction → Knowledge Facts → RAG Ingestion
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class DistilledKnowledge:
    """A single piece of distilled knowledge ready for RAG."""

    knowledge_id: str
    content: str  # The actual text to store in RAG
    source_type: str  # "error_lesson", "success_pattern", "tool_tip", "domain_fact"
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
    ):
        self.output_dir = (
            Path(output_dir) if output_dir else Path.home() / ".huginn" / "distilled"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sobkso_db_path = Path(sobkso_db_path) if sobkso_db_path else None
        self.knowledge_base: list[DistilledKnowledge] = []
        self._load_existing()

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

    # ------------------------------------------------------------------
    # Distillation Methods
    # ------------------------------------------------------------------

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
                "C:/Users/wanzh/Sobko_MCP_project/chroma_db/chunks.jsonl",
                "C:/Users/wanzh/Desktop/Sobko_MCP_project/chroma_db/chunks.jsonl",
            ]
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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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
