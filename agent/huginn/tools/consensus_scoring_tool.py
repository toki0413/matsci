"""shim: 文件位于 huginn.tools.sci.consensus_scoring_tool."""
from huginn.tools.sci.consensus_scoring_tool import (  # noqa: F401
    ConsensusScoringInput,
    ConsensusScoringTool,
)

__all__ = ["ConsensusScoringTool", "ConsensusScoringInput"]
