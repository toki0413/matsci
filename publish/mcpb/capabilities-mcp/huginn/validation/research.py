"""Research validation re-export from autoloop layer.

Thin shim for red-team / blue-team review. Callers get a stable import path
even though the implementations live in autoloop/red_team + autoloop/blue_team.
"""

from huginn.autoloop.blue_team import (
    BlueTeamFinding,
    BlueTeamReport,
    BlueTeamReviewer,
)
from huginn.autoloop.red_team import (
    RedTeamFinding,
    RedTeamReport,
    RedTeamReviewer,
)

__all__ = [
    "RedTeamFinding",
    "RedTeamReport",
    "RedTeamReviewer",
    "BlueTeamFinding",
    "BlueTeamReport",
    "BlueTeamReviewer",
]
