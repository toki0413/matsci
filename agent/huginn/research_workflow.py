"""Research workflow: bridges Deli AutoResearch pipeline to the WS event stream.

Wraps DeliAutoResearch.run_full_pipeline() behind an async generator that
yields progress/hypothesis/experiment/result/draft events for the WebSocket
layer. Also keeps the agent's PhaseManager in sync with the current Deli stage.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

from huginn.academic.deli_research import (
    DeliAutoResearch,
    ResearchStage,
    ResearchState,
)
from huginn.personas import BUILT_IN_PERSONAS, Persona
from huginn.phases import (
    PHASE_STAGE_MAP,
    ResearchPhase,
    stage_to_phase,
)
from huginn.prompts import MATH_DEPTH_GUIDE
from huginn.types import progress_cb

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────

@dataclass
class ResearchWorkflowConfig:
    max_concurrent_branches: int = 3
    enable_hypothesis_generation: bool = True
    target_journal: str | None = None


# ──────────────────────────────────────────────────────────────────────
# Persona — 从 personas.py 取, 不再本地定义
# ──────────────────────────────────────────────────────────────────────

def _get_research_persona() -> Persona:
    """从 BUILT_IN_PERSONAS 查找 research persona."""
    for p in BUILT_IN_PERSONAS:
        if p.name == "research":
            return p
    # fallback: 内联构造 (不应发生, 但防御性处理)
    return Persona(name="research", system_prompt=MATH_DEPTH_GUIDE)

RESEARCH_PERSONA = _get_research_persona()


# ──────────────────────────────────────────────────────────────────────
# Workflow
# ──────────────────────────────────────────────────────────────────────

class ResearchWorkflow:
    """Runs the Deli pipeline and streams events to the WS layer.

    Usage::

        wf = ResearchWorkflow(agent, ResearchWorkflowConfig())
        async for event in wf.run("Li-ion battery cathode degradation"):
            ws.send_json(event)
    """

    def __init__(self, agent: Any, config: ResearchWorkflowConfig) -> None:
        self.agent = agent
        self.config = config
        self.deli = DeliAutoResearch(
            persona_system_prompt=RESEARCH_PERSONA.system_prompt
        )

    async def run(
        self, topic: str, thread_id: str = ""
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Async generator yielding research events.

        Event types:
        - {"type": "status", "stage": "...", "message": "..."}
        - {"type": "hypothesis", "hypothesis": "..."}
        - {"type": "experiment", "description": "..."}
        - {"type": "result", "summary": "..."}
        - {"type": "draft", "content": "..."}
        - {"type": "error", "message": "..."}
        """
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

        async def _on_progress(event: dict[str, Any]) -> None:
            # 透传完整事件, 保留 stage_index/total_stages/progress_pct 等字段
            event["type"] = "status"
            event["message"] = event.pop("detail", "")
            await queue.put(event)
            # keep PhaseManager in sync with the current Deli stage
            stage = event.get("stage", "")
            if stage:
                phase = stage_to_phase(stage)
                if phase != ResearchPhase.OPEN:
                    mgr = getattr(self.agent, "_phase_manager", None)
                    if mgr is not None:
                        mgr.transition(phase)

        token = progress_cb.set(_on_progress)

        async def _run_pipeline() -> ResearchState | None:
            try:
                return await self.deli.run_full_pipeline(
                    topic=topic,
                    target_journal=self.config.target_journal,
                    rag_search_fn=self._rag_search,
                )
            except Exception as exc:
                await queue.put({"type": "error", "message": str(exc)})
                return None
            finally:
                await queue.put(None)  # sentinel: pipeline finished

        pipeline_task = asyncio.create_task(_run_pipeline())

        try:
            # drain progress events until the pipeline signals completion
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield event

            state = await pipeline_task
            if state is None:
                return

            # gap 分析输出即假设
            if self.config.enable_hypothesis_generation:
                for gap in state.gaps:
                    text = gap.get("gap", str(gap)) if isinstance(gap, dict) else str(gap)
                    yield {"type": "hypothesis", "hypothesis": text}

            # 计算循环填充的 gap
            for desc in state.gaps_filled:
                yield {"type": "experiment", "description": desc}

            yield {
                "type": "result",
                "summary": (
                    f"Pipeline complete: {len(state.literature)} papers, "
                    f"{len(state.gaps)} gaps, {len(state.citations)} citations, "
                    f"{len(state.integrated_draft)} chars in draft"
                ),
            }

            if state.integrated_draft:
                yield {"type": "draft", "content": state.integrated_draft}

            # final phase sync
            final_phase = stage_to_phase(state.stage.value)
            if final_phase != ResearchPhase.OPEN:
                mgr = getattr(self.agent, "_phase_manager", None)
                if mgr is not None:
                    mgr.transition(final_phase)
        finally:
            progress_cb.reset(token)
            if not pipeline_task.done():
                pipeline_task.cancel()

    # ── RAG bridge ───────────────────────────────────────────────

    def _rag_search(self, query: str) -> list[dict[str, Any]] | None:
        """Query the agent's KnowledgeBase. Return None if unavailable."""
        kb = self._get_kb()
        if kb is None:
            return None
        try:
            if hasattr(kb, "count") and kb.count() == 0:
                return None
            chunks = kb.query(query, top_k=5)
            return chunks if chunks else None
        except Exception:
            logger.debug("KB query failed for: %s", query[:60], exc_info=True)
            return None

    def _get_kb(self) -> Any | None:
        """Resolve the KnowledgeBase from the agent (direct or via ContextBuilder)."""
        kb = getattr(self.agent, "_kb", None)
        if kb is not None:
            return kb
        ctx_builder = getattr(self.agent, "_ctx_builder", None)
        if ctx_builder is not None:
            return getattr(ctx_builder, "_kb", None)
        return None


# ──────────────────────────────────────────────────────────────────────
# Self-check
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = ResearchWorkflowConfig()
    assert cfg.max_concurrent_branches == 3
    assert cfg.enable_hypothesis_generation is True
    assert cfg.target_journal is None

    assert RESEARCH_PERSONA.name == "research"
    assert "Deli" in RESEARCH_PERSONA.system_prompt
    assert "gap" in RESEARCH_PERSONA.system_prompt.lower()

    # PHASE_STAGE_MAP imported from phases -- StrEnum values match strings
    assert PHASE_STAGE_MAP[ResearchPhase.LITERATURE] == [
        "topic_analysis", "literature_search", "gap_analysis"
    ]
    assert PHASE_STAGE_MAP[ResearchPhase.HYPOTHESIS] == []
    assert stage_to_phase("drafting") == ResearchPhase.EXECUTION
    assert stage_to_phase("literature_search") == ResearchPhase.LITERATURE
    assert stage_to_phase("peer_review") == ResearchPhase.REPORTING
    assert stage_to_phase("unknown") == ResearchPhase.OPEN

    # StrEnum equivalence: ResearchStage values are strings
    assert ResearchStage.TOPIC_ANALYSIS == "topic_analysis"
    assert ResearchStage.GAP_ANALYSIS in PHASE_STAGE_MAP[ResearchPhase.LITERATURE]

    print("research_workflow self-check OK")
