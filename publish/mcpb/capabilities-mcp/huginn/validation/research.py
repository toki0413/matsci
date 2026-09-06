"""Research validation re-export from autoloop layer.

Thin shim for red-team review. Callers get a stable import path even
though the implementation lives in autoloop/red_team.
"""

from huginn.autoloop.red_team import (
    RedTeamFinding,
    RedTeamReport,
    RedTeamReviewer,
)

__all__ = [
    "RedTeamFinding",
    "RedTeamReport",
    "RedTeamReviewer",
]
