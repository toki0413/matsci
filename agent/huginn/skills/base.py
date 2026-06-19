"""Skill base class for material science workflows.

A Skill is a reusable, parameterized workflow template that the agent
can invoke by name. Skills combine tools, prompts, and validation rules
into declarative units.
"""

from __future__ import annotations

import ast
import inspect
import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, get_origin

from huginn.security import SafeEvalError, safe_eval


def _wants_dict(tool: Any) -> bool:
    """Return True if a tool's ``call`` method expects a plain dict for ``args``."""
    try:
        hints = typing.get_type_hints(tool.call)
    except Exception:
        return False
    ann = hints.get("args")
    if ann is None:
        return False
    origin = get_origin(ann)
    return origin is dict or ann is dict


@dataclass
class SkillStep:
    """A single step in a skill workflow."""

    name: str
    tool: str  # tool name
    input_mapping: dict[str, str]  # maps skill params → tool args
    output_key: str  # where to store result in context
    validation: str | None = None  # optional validation expression
    on_failure: str = "abort"  # "abort", "skip", "retry"
    retries: int = 0


@dataclass
class SkillParameter:
    """Declared parameter for a skill."""

    name: str
    type: str
    description: str
    default: Any = None
    required: bool = True


@dataclass
class SkillDefinition:
    """Declarative skill definition."""

    name: str
    description: str
    category: str  # "computation", "analysis", "diagnostics", "reporting"
    parameters: list[SkillParameter] = field(default_factory=list)
    steps: list[SkillStep] = field(default_factory=list)
    required_tools: list[str] = field(default_factory=list)
    estimated_cost: dict[str, float] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    def to_prompt(self) -> str:
        """Generate a natural language description for the LLM."""
        lines = [
            f"Skill: {self.name}",
            f"Description: {self.description}",
            f"Category: {self.category}",
            "Parameters:",
        ]
        for p in self.parameters:
            req = "(required)" if p.required else f"(default: {p.default})"
            lines.append(f"  - {p.name}: {p.type} — {p.description} {req}")
        lines.append("Steps:")
        for i, s in enumerate(self.steps, 1):
            lines.append(f"  {i}. {s.name} ({s.tool})")
        return "\n".join(lines)


class SkillExecutor(ABC):
    """Base class for skill execution engines."""

    @abstractmethod
    async def execute(
        self,
        skill: SkillDefinition,
        params: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a skill with given parameters. Returns execution results."""
        ...


class DeclarativeSkillExecutor(SkillExecutor):
    """Execute skills defined by SkillDefinition + SkillStep."""

    def __init__(self, tool_registry: Any):
        self.tool_registry = tool_registry

    async def execute(
        self,
        skill: SkillDefinition,
        params: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        from huginn.types import ToolContext

        results = {"skill": skill.name, "steps": [], "success": True}
        working_context = {**context, **params}

        # Apply parameter defaults when not supplied
        for param in skill.parameters:
            if param.name not in working_context and param.default is not None:
                working_context[param.name] = param.default

        for step in skill.steps:
            step_result = {
                "step": step.name,
                "success": False,
                "output": None,
                "error": None,
            }

            # Resolve tool inputs from context + params
            tool_input = {}
            for key, mapping in step.input_mapping.items():
                tool_input[key] = self._resolve_value(mapping, working_context)

            # Get tool
            tool = self.tool_registry.get(step.tool)
            if tool is None:
                step_result["error"] = f"Tool '{step.tool}' not found"
                if step.on_failure == "abort":
                    results["success"] = False
                    results["steps"].append(step_result)
                    break
                continue

            # Execute
            try:
                # Construct pydantic input
                input_model = tool.input_schema
                parsed = input_model(**tool_input)
                tool_ctx = ToolContext(session_id="skill", workspace=".")
                payload = parsed.model_dump() if _wants_dict(tool) else parsed
                if inspect.iscoroutinefunction(tool.call):
                    output = await tool.call(payload, tool_ctx)
                else:
                    output = tool.call(payload, tool_ctx)
                step_result["success"] = output.success
                step_result["output"] = output.data
                if output.error:
                    step_result["error"] = output.error

                # Store output in working context
                working_context[step.output_key] = output.data

                # Validation
                if step.validation and output.success:
                    # Simple eval-based validation (caution: only trusted skills)
                    try:
                        valid = safe_eval(step.validation, working_context)
                        if not valid:
                            step_result["success"] = False
                            step_result["error"] = "Validation failed"
                    except SafeEvalError as e:
                        step_result["error"] = f"Validation error (safe): {e}"
                    except Exception as e:
                        step_result["error"] = f"Validation error: {e}"

            except Exception as e:
                step_result["error"] = str(e)
                if step.on_failure == "abort":
                    results["success"] = False
                    results["steps"].append(step_result)
                    break

            results["steps"].append(step_result)
            if not step_result["success"] and step.on_failure == "abort":
                results["success"] = False
                break

        results["context"] = {
            k: v for k, v in working_context.items() if not k.startswith("_")
        }
        return results

    @staticmethod
    def _resolve_value(mapping: str, context: dict[str, Any]) -> Any:
        """Resolve a step input mapping against the working context.

        Mappings starting with ``$`` are treated as context lookups and support
        dotted paths such as ``$relax_result.relaxed_structure``. Other values
        are parsed with ``ast.literal_eval`` so that ``'relax'`` becomes the
        string ``relax`` and ``3`` becomes the integer ``3``.
        """
        mapping = mapping.strip()
        if mapping.startswith("$"):
            path = mapping[1:]
            value: Any = context
            for part in path.split("."):
                if isinstance(value, dict):
                    value = value.get(part)
                elif hasattr(value, part):
                    value = getattr(value, part)
                else:
                    return None
            return value
        try:
            return ast.literal_eval(mapping)
        except (ValueError, SyntaxError):
            return mapping
