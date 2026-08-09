"""向后兼容 re-export — 实现已迁移到 huginn.metacog.critique.

adversarial_critique / critique_decision / format_critique_for_agent
是通用元认知能力, 不再绑定 RCB 框架.
"""
from huginn.metacog.critique import (  # noqa: F401
    CritiqueResult,
    Decision,
    _strip_code_fences,
    _template_critique_decision,
    adversarial_critique,
    critique_decision,
    format_critique_for_agent,
)
