"""Physical validation package — unified access to all validators.

Re-exports validators from execution/, autoloop/, and tools/ layers so
callers can `from huginn.validation import DimensionalValidator` etc.
Original modules stay in place; this package is a convenience facade.

Import order matters: physics must load before tool_output because
validate_tool.py does `from huginn.validation.physics import PhysicsValidator`.
"""

# physics also re-exports PhysicsAuditor/AuditReport/PhysicsFinding
from huginn.validation.physics import (
    AuditReport,
    PhysicsAuditor,
    PhysicsFinding,
    PhysicsValidator,
    ValidationCheck,
)
from huginn.validation.handle_validator import HandleValidator
from huginn.validation.dimensional import (
    DimensionalCheckResult,
    DimensionalValidator,
    PhysicalQuantity,
    Unit,
    UnitRegistry,
    registry,
)
from huginn.validation.research import (
    RedTeamFinding,
    RedTeamReport,
    RedTeamReviewer,
)

# tool_output pulls in the full tool stack (pydantic, tools.base, ...).
# Guard it so the rest of the package stays usable in minimal envs.
_extra: list[str] = []
try:
    from huginn.validation.tool_output import (
        ValidateTool,
        ValidateToolInput,
    )
    _extra = ["ValidateTool", "ValidateToolInput"]
except ImportError:  # pragma: no cover
    pass

__all__ = [
    # physics
    "PhysicsValidator",
    "ValidationCheck",
    "PhysicsAuditor",
    "AuditReport",
    "PhysicsFinding",
    # handle
    "HandleValidator",
    # dimensional
    "DimensionalValidator",
    "DimensionalCheckResult",
    "PhysicalQuantity",
    "Unit",
    "UnitRegistry",
    "registry",
    # research / red team
    "RedTeamReviewer",
    "RedTeamReport",
    "RedTeamFinding",
    # tool output (may be unavailable in minimal envs)
    *_extra,
]
