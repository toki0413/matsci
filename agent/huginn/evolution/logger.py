"""Execution Logger — records every tool call, success, failure, and outcome
for later evolutionary analysis."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ToolCallRecord:
    """Record of a single tool invocation."""
    session_id: str
    tool_name: str
    tool_input: Dict[str, Any]
    result_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    success: bool = False
    walltime_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    tags: List[str] = field(default_factory=list)
    software: Optional[str] = None
    calculation_type: Optional[str] = None


@dataclass
class ConversationRecord:
    """Record of a user-agent conversation turn."""
    session_id: str
    user_message: str
    agent_response: str
    tools_used: List[str] = field(default_factory=list)
    satisfaction: Optional[float] = None  # 0-1, inferred from follow-up
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    topic_tags: List[str] = field(default_factory=list)


class ExecutionLogger:
    """Persistent execution logger for evolutionary feedback.

    Usage:
        logger = ExecutionLogger()
        logger.log_tool_call(tool_name="vasp_tool", tool_input={...}, result=..., success=True)
        logger.log_conversation(user="...", agent="...", tools_used=[...])
    """

    def __init__(self, persist_dir: Optional[str] = None):
        self.persist_dir = Path(persist_dir) if persist_dir else Path.home() / ".huginn" / "logs"
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._tool_calls: List[ToolCallRecord] = []
        self._conversations: List[ConversationRecord] = []
        self._session_stats: Dict[str, Dict[str, Any]] = {}
        self._load_existing()

    def _load_existing(self) -> None:
        """Load historical logs from disk."""
        tool_log = self.persist_dir / "tool_calls.jsonl"
        if tool_log.exists():
            with tool_log.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        self._tool_calls.append(ToolCallRecord(**data))

        conv_log = self.persist_dir / "conversations.jsonl"
        if conv_log.exists():
            with conv_log.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        self._conversations.append(ConversationRecord(**data))

    def log_tool_call(
        self,
        session_id: str,
        tool_name: str,
        tool_input: Dict[str, Any],
        result: Optional[Any] = None,
        error: Optional[str] = None,
        walltime_ms: float = 0.0,
        software: Optional[str] = None,
        calculation_type: Optional[str] = None,
    ) -> None:
        """Log a tool invocation."""
        success = error is None and result is not None
        if isinstance(result, dict):
            result_data = result
        elif hasattr(result, "model_dump"):
            result_data = result.model_dump()
        else:
            result_data = {"raw": str(result)}

        record = ToolCallRecord(
            session_id=session_id,
            tool_name=tool_name,
            tool_input=tool_input,
            result_data=result_data if success else None,
            error_message=error,
            success=success,
            walltime_ms=walltime_ms,
            software=software,
            calculation_type=calculation_type,
            tags=[],
        )
        self._tool_calls.append(record)
        self._flush_tool_record(record)

        # Update session stats
        stats = self._session_stats.setdefault(session_id, {
            "total_calls": 0,
            "success_calls": 0,
            "failed_calls": 0,
            "tools_used": set(),
        })
        stats["total_calls"] += 1
        if success:
            stats["success_calls"] += 1
        else:
            stats["failed_calls"] += 1
        stats["tools_used"].add(tool_name)

    def log_conversation(
        self,
        session_id: str,
        user_message: str,
        agent_response: str,
        tools_used: List[str] = None,
        topic_tags: List[str] = None,
    ) -> None:
        """Log a conversation turn."""
        record = ConversationRecord(
            session_id=session_id,
            user_message=user_message,
            agent_response=agent_response,
            tools_used=tools_used or [],
            topic_tags=topic_tags or [],
        )
        self._conversations.append(record)
        self._flush_conv_record(record)

    def _flush_tool_record(self, record: ToolCallRecord) -> None:
        """Append a single tool record to disk."""
        tool_log = self.persist_dir / "tool_calls.jsonl"
        with tool_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    def _flush_conv_record(self, record: ConversationRecord) -> None:
        """Append a single conversation record to disk."""
        conv_log = self.persist_dir / "conversations.jsonl"
        with conv_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    # Analytics API for Evolution
    # ------------------------------------------------------------------

    def get_failure_patterns(self, min_count: int = 2) -> List[Dict[str, Any]]:
        """Extract recurring failure patterns."""
        from collections import Counter
        failures = [r for r in self._tool_calls if not r.success]
        patterns = Counter()
        for f in failures:
            key = f"{f.tool_name}|{f.error_message[:80]}"
            patterns[key] += 1
        return [
            {"pattern": p, "tool": p.split("|")[0], "error": p.split("|")[1], "count": c}
            for p, c in patterns.most_common()
            if c >= min_count
        ]

    def get_tool_success_rate(self) -> Dict[str, float]:
        """Compute per-tool success rate."""
        from collections import defaultdict
        stats = defaultdict(lambda: {"success": 0, "total": 0})
        for r in self._tool_calls:
            stats[r.tool_name]["total"] += 1
            if r.success:
                stats[r.tool_name]["success"] += 1
        return {
            tool: s["success"] / s["total"] if s["total"] > 0 else 0.0
            for tool, s in stats.items()
        }

    def get_software_failure_rates(self) -> Dict[str, float]:
        """Compute per-software failure rate."""
        from collections import defaultdict
        stats = defaultdict(lambda: {"success": 0, "total": 0})
        for r in self._tool_calls:
            if r.software:
                stats[r.software]["total"] += 1
                if r.success:
                    stats[r.software]["success"] += 1
        return {
            sw: 1.0 - (s["success"] / s["total"] if s["total"] > 0 else 0.0)
            for sw, s in stats.items()
        }

    def get_recent_errors(self, n: int = 20) -> List[Dict[str, Any]]:
        """Get the N most recent errors."""
        failures = [r for r in self._tool_calls if not r.success]
        failures.sort(key=lambda x: x.timestamp, reverse=True)
        return [
            {
                "tool": r.tool_name,
                "error": r.error_message,
                "software": r.software,
                "timestamp": r.timestamp,
                "input": r.tool_input,
            }
            for r in failures[:n]
        ]

    def export_for_evolution(self, output_path: Optional[str] = None) -> str:
        """Export a summary dataset for the EvolutionEngine."""
        summary = {
            "total_tool_calls": len(self._tool_calls),
            "total_conversations": len(self._conversations),
            "success_rate": sum(1 for r in self._tool_calls if r.success) / max(len(self._tool_calls), 1),
            "tool_success_rates": self.get_tool_success_rate(),
            "software_failure_rates": self.get_software_failure_rates(),
            "failure_patterns": self.get_failure_patterns(min_count=2),
            "recent_errors": self.get_recent_errors(n=20),
            "session_stats": {
                sid: {
                    "total": s["total_calls"],
                    "success": s["success_calls"],
                    "failure": s["failed_calls"],
                    "tools": list(s["tools_used"]),
                }
                for sid, s in self._session_stats.items()
            },
        }
        if output_path is None:
            output_path = self.persist_dir / "evolution_summary.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        return str(output_path)
