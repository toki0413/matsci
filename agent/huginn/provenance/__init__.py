"""计算溯源注册表 + FAIR provenance 快照.

合并了两个系统:
  1. _legacy.py (原 huginn/provenance.py): M4 FAIR provenance 快照
     — ProvenanceRecord / ProvenanceLogger / capture / export_crate
  2. registry.py: 实时溯源注册表 — 工具产出文件自动注册
  3. pipeline.py: 事件驱动仿真管线 — 根据工具调用建议下一步
"""

# ── 从 _legacy.py 重新导出原有 API ──────────────────────────────
from huginn.provenance._legacy import (
    ProvenanceLogger,
    ProvenanceRecord,
    ProvenanceSnapshot,
    capture,
    capture_run_inputs,
    export_crate,
    list_snapshots,
    save,
)

# ── 新增: 实时溯源注册表 ────────────────────────────────────────
from huginn.provenance.registry import (
    ProvenanceEntry,
    ProvenanceRegistry,
    VersionConflict,
    register_tool_output,
)

# ── 新增: 事件驱动仿真管线 ──────────────────────────────────────
from huginn.provenance.pipeline import (
    PIPELINE_RULES,
    PipelineRule,
    PipelineStage,
    PipelineSuggestion,
    SimulationPipeline,
    get_pipeline,
    pipeline_hook,
)

# ── 新增: Sim-to-Real 校正因子表 ────────────────────────────────
from huginn.provenance.correction import CorrectionEntry, CorrectionTable

__all__ = [
    # Legacy FAIR provenance
    "ProvenanceLogger",
    "ProvenanceRecord",
    "ProvenanceSnapshot",
    "capture",
    "capture_run_inputs",
    "export_crate",
    "list_snapshots",
    "save",
    # New: real-time registry
    "ProvenanceEntry",
    "ProvenanceRegistry",
    "VersionConflict",
    "register_tool_output",
    # New: event-driven pipeline
    "PIPELINE_RULES",
    "PipelineRule",
    "PipelineStage",
    "PipelineSuggestion",
    "SimulationPipeline",
    "get_pipeline",
    "pipeline_hook",
    # New: sim-to-real correction
    "CorrectionEntry",
    "CorrectionTable",
]
