"""Tunable parameters — runtime configuration that can be changed without restart."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class ParamType(str, Enum):
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    STRING = "string"
    CHOICE = "choice"


@dataclass
class Parameter:
    name: str
    param_type: ParamType
    default: Any
    current: Any = None
    description: str = ""
    min_value: Any = None
    max_value: Any = None
    choices: list[str] = field(default_factory=list)
    category: str = "general"
    on_change: Callable[[Any], None] | None = None

    def __post_init__(self) -> None:
        if self.current is None:
            self.current = self.default

    def validate(self, value: Any) -> tuple[bool, str]:
        """Validate a proposed value."""
        if self.param_type == ParamType.INT:
            try:
                v = int(value)
            except (ValueError, TypeError):
                return False, f"Expected integer, got {type(value).__name__}"
            if self.min_value is not None and v < self.min_value:
                return False, f"Value {v} below minimum {self.min_value}"
            if self.max_value is not None and v > self.max_value:
                return False, f"Value {v} above maximum {self.max_value}"
            return True, ""
        elif self.param_type == ParamType.FLOAT:
            try:
                v = float(value)
            except (ValueError, TypeError):
                return False, f"Expected float, got {type(value).__name__}"
            if self.min_value is not None and v < self.min_value:
                return False, f"Value {v} below minimum {self.min_value}"
            if self.max_value is not None and v > self.max_value:
                return False, f"Value {v} above maximum {self.max_value}"
            return True, ""
        elif self.param_type == ParamType.BOOL:
            if not isinstance(value, bool):
                return False, f"Expected bool, got {type(value).__name__}"
            return True, ""
        elif self.param_type == ParamType.STRING:
            if not isinstance(value, str):
                return False, f"Expected string, got {type(value).__name__}"
            return True, ""
        elif self.param_type == ParamType.CHOICE:
            if value not in self.choices:
                return False, f"Value must be one of {self.choices}"
            return True, ""
        return True, ""


class ParameterRegistry:
    """Registry of tunable parameters."""

    _params: dict[str, Parameter] = {}

    @classmethod
    def register(cls, param: Parameter) -> None:
        cls._params[param.name] = param

    @classmethod
    def get(cls, name: str) -> Parameter | None:
        return cls._params.get(name)

    @classmethod
    def set_value(cls, name: str, value: Any) -> tuple[bool, str]:
        param = cls._params.get(name)
        if not param:
            return False, f"Unknown parameter: {name}"
        valid, msg = param.validate(value)
        if not valid:
            return False, msg
        old = param.current
        param.current = value
        if param.on_change and old != value:
            param.on_change(value)
        return True, ""

    @classmethod
    def get_value(cls, name: str, default: Any = None) -> Any:
        param = cls._params.get(name)
        if not param:
            return default
        return param.current

    @classmethod
    def list_params(cls, category: str | None = None) -> list[dict]:
        result = []
        for p in cls._params.values():
            if category and p.category != category:
                continue
            result.append(
                {
                    "name": p.name,
                    "type": p.param_type.value,
                    "current": p.current,
                    "default": p.default,
                    "description": p.description,
                    "category": p.category,
                    "min": p.min_value,
                    "max": p.max_value,
                    "choices": p.choices if p.choices else None,
                }
            )
        return result

    @classmethod
    def reset(cls, name: str | None = None) -> None:
        if name:
            p = cls._params.get(name)
            if p:
                p.current = p.default
        else:
            for p in cls._params.values():
                p.current = p.default

    @classmethod
    def categories(cls) -> list[str]:
        return sorted({p.category for p in cls._params.values()})


# Register built-in parameters
def _register_defaults() -> None:
    ParameterRegistry.register(
        Parameter(
            name="max_walltime_hours",
            param_type=ParamType.INT,
            default=168,
            min_value=1,
            max_value=720,
            description="Maximum walltime for HPC jobs (hours)",
            category="hpc",
        )
    )
    ParameterRegistry.register(
        Parameter(
            name="max_parallel_jobs",
            param_type=ParamType.INT,
            default=5,
            min_value=1,
            max_value=50,
            description="Maximum number of parallel HPC jobs",
            category="hpc",
        )
    )
    ParameterRegistry.register(
        Parameter(
            name="default_queue",
            param_type=ParamType.CHOICE,
            default="normal",
            choices=["debug", "normal", "gpu", "fat"],
            description="Default HPC queue",
            category="hpc",
        )
    )
    ParameterRegistry.register(
        Parameter(
            name="auto_validate_physics",
            param_type=ParamType.BOOL,
            default=True,
            description="Auto-validate physics constraints before submission",
            category="validation",
        )
    )
    ParameterRegistry.register(
        Parameter(
            name="mock_mode",
            param_type=ParamType.BOOL,
            default=True,
            description="Use mock results when real software unavailable",
            category="execution",
        )
    )
    ParameterRegistry.register(
        Parameter(
            name="cache_ttl_seconds",
            param_type=ParamType.INT,
            default=3600,
            min_value=60,
            max_value=86400,
            description="Tool result cache TTL in seconds",
            category="performance",
        )
    )
    ParameterRegistry.register(
        Parameter(
            name="max_script_timeout",
            param_type=ParamType.FLOAT,
            default=30.0,
            min_value=1.0,
            max_value=300.0,
            description="Maximum live script timeout (seconds)",
            category="execution",
        )
    )
    ParameterRegistry.register(
        Parameter(
            name="llm_temperature",
            param_type=ParamType.FLOAT,
            default=0.7,
            min_value=0.0,
            max_value=2.0,
            description="LLM sampling temperature",
            category="llm",
        )
    )
    ParameterRegistry.register(
        Parameter(
            name="llm_max_tokens",
            param_type=ParamType.INT,
            default=4096,
            min_value=256,
            max_value=32768,
            description="Maximum tokens per LLM response",
            category="llm",
        )
    )


_register_defaults()
