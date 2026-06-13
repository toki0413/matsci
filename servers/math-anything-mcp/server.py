"""Math-Anything MCP Server — Mathematical semantics and diff server.

Provides MCP tools for:
- Semantic extraction of equations and variables from text
- Mathematical diff (compare two expressions/results for equivalence)
- Dimensional analysis
- Numerical precision tracking

Usage:
    python server.py
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

app = Server("math-anything-mcp")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

LATEX_PATTERN = re.compile(r"\\\((.+?)\\\)|\\\[(.+?)\\\]|\$\$(.+?)\$\$|\$(.+?)\$")
FLOAT_PATTERN = re.compile(r"(-?\d+\.\d+(?:[eE][+-]?\d+)?)")
UNIT_PATTERN = re.compile(
    r"(\d+\.?\d*)\s*(eV|GPa|MPa|kPa|Pa|J|kJ|eV/atom|Angstrom|Å|nm|pm|K|°C|g/cm\^3|kg/m\^3)"
)

DIMENSIONS: dict[str, dict[str, str]] = {
    "eV": {"energy": "1", "length": "0", "mass": "0", "time": "0", "temperature": "0"},
    "GPa": {"energy": "1", "length": "-3", "mass": "0", "time": "0", "temperature": "0"},
    "MPa": {"energy": "1", "length": "-3", "mass": "0", "time": "0", "temperature": "0"},
    "Pa": {"energy": "1", "length": "-3", "mass": "0", "time": "0", "temperature": "0"},
    "J": {"energy": "1", "length": "0", "mass": "0", "time": "0", "temperature": "0"},
    "kJ": {"energy": "1", "length": "0", "mass": "0", "time": "0", "temperature": "0"},
    "eV/atom": {"energy": "1", "length": "0", "mass": "0", "time": "0", "temperature": "0"},
    "Angstrom": {"energy": "0", "length": "1", "mass": "0", "time": "0", "temperature": "0"},
    "Å": {"energy": "0", "length": "1", "mass": "0", "time": "0", "temperature": "0"},
    "nm": {"energy": "0", "length": "1", "mass": "0", "time": "0", "temperature": "0"},
    "pm": {"energy": "0", "length": "1", "mass": "0", "time": "0", "temperature": "0"},
    "K": {"energy": "0", "length": "0", "mass": "0", "time": "0", "temperature": "1"},
    "°C": {"energy": "0", "length": "0", "mass": "0", "time": "0", "temperature": "1"},
}


def _extract_equations(text: str) -> list[dict[str, str]]:
    """Extract LaTeX-style equations from text."""
    equations = []
    for match in LATEX_PATTERN.finditer(text):
        eq = next(g for g in match.groups() if g is not None)
        equations.append({
            "raw": match.group(0),
            "latex": eq,
            "position": match.start(),
        })
    return equations


def _extract_variables(text: str) -> list[dict[str, Any]]:
    """Extract likely variable assignments from text."""
    variables = []
    # Pattern: "X = 123.45" or "X: 123.45" or "X is 123.45"
    var_patterns = [
        re.compile(r"([A-Za-z][A-Za-z0-9_\s]*)\s*=\s*(-?\d+\.?\d*(?:[eE][+-]?\d+)?)"),
        re.compile(r"([A-Za-z][A-Za-z0-9_\s]*)\s*:\s*(-?\d+\.?\d*(?:[eE][+-]?\d+)?)"),
    ]
    for pat in var_patterns:
        for m in pat.finditer(text):
            name = m.group(1).strip()
            value = float(m.group(2))
            variables.append({"name": name, "value": value, "position": m.start()})
    return variables


def _extract_numbers(text: str) -> list[dict[str, Any]]:
    """Extract all floating-point numbers with context."""
    numbers = []
    for m in FLOAT_PATTERN.finditer(text):
        # Get surrounding context (20 chars before/after)
        start = max(0, m.start() - 20)
        end = min(len(text), m.end() + 20)
        context = text[start:end].replace("\n", " ")
        numbers.append({
            "value": float(m.group(1)),
            "context": context,
            "position": m.start(),
        })
    return numbers


def _parse_expression(expr: str) -> dict[str, Any]:
    """Parse a mathematical expression string into an AST-like structure."""
    try:
        import sympy
        s = sympy.sympify(expr)
        return {
            "type": "sympy",
            "expression": str(s),
            "symbols": [str(x) for x in s.free_symbols],
            "is_constant": len(s.free_symbols) == 0,
        }
    except Exception:
        return {"type": "raw", "expression": expr, "symbols": [], "is_constant": False}


def _compare_numeric(a: float, b: float, rtol: float = 1e-5, atol: float = 1e-8) -> dict[str, Any]:
    diff = abs(a - b)
    rel_diff = diff / max(abs(a), abs(b), 1e-300)
    return {
        "equal": diff <= atol + rtol * max(abs(a), abs(b)),
        "absolute_difference": diff,
        "relative_difference": rel_diff,
        "within_tolerance": rel_diff <= rtol,
    }


def _dimensional_analysis(quantity: str) -> dict[str, Any]:
    m = UNIT_PATTERN.search(quantity)
    if not m:
        return {"error": "No recognized unit found", "input": quantity}
    value = float(m.group(1))
    unit = m.group(2)
    dims = DIMENSIONS.get(unit, {})
    return {
        "value": value,
        "unit": unit,
        "dimensions": dims,
        "si_convertible": unit in {"GPa", "MPa", "kPa", "Pa", "nm", "pm", "Å", "Angstrom"},
    }


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

TOOLS: list[Tool] = [
    Tool(
        name="extract_math",
        description="Extract equations, variables, and numerical values from scientific text",
        inputSchema={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Scientific text to analyze"},
                "extract_equations": {"type": "boolean", "default": True},
                "extract_variables": {"type": "boolean", "default": True},
                "extract_numbers": {"type": "boolean", "default": True},
            },
            "required": ["text"],
        },
    ),
    Tool(
        name="math_diff",
        description="Compare two mathematical expressions or numerical results for semantic equivalence",
        inputSchema={
            "type": "object",
            "properties": {
                "a": {"type": "string", "description": "First expression or value"},
                "b": {"type": "string", "description": "Second expression or value"},
                "mode": {"type": "string", "enum": ["numeric", "symbolic", "text"], "default": "numeric"},
                "rtol": {"type": "number", "default": 1e-5},
                "variables": {"type": "object", "description": "Variable values for numeric comparison"},
            },
            "required": ["a", "b"],
        },
    ),
    Tool(
        name="dimensional_analysis",
        description="Analyze physical dimensions of a quantity with units",
        inputSchema={
            "type": "object",
            "properties": {
                "quantity": {"type": "string", "description": "e.g. '5.43 GPa' or '300 K'"},
            },
            "required": ["quantity"],
        },
    ),
    Tool(
        name="track_precision",
        description="Track numerical precision and significant figures",
        inputSchema={
            "type": "object",
            "properties": {
                "values": {"type": "array", "items": {"type": "string"}, "description": "List of numerical strings"},
            },
            "required": ["values"],
        },
    ),
    Tool(
        name="normalize_expression",
        description="Normalize and simplify a mathematical expression",
        inputSchema={
            "type": "object",
            "properties": {
                "expression": {"type": "string"},
            },
            "required": ["expression"],
        },
    ),
]


@app.list_tools()
async def list_tools() -> list[Tool]:
    return TOOLS


@app.call_tool()
async def call_tool(name: str, arguments: dict | None) -> list[TextContent]:
    arguments = arguments or {}

    if name == "extract_math":
        return await _extract_math(arguments)
    elif name == "math_diff":
        return await _math_diff(arguments)
    elif name == "dimensional_analysis":
        return await _dimensional_analysis_tool(arguments)
    elif name == "track_precision":
        return await _track_precision(arguments)
    elif name == "normalize_expression":
        return await _normalize_expression(arguments)
    else:
        raise ValueError(f"Unknown tool: {name}")


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

async def _extract_math(args: dict) -> list[TextContent]:
    text = args.get("text", "")
    result: dict[str, Any] = {}

    if args.get("extract_equations", True):
        result["equations"] = _extract_equations(text)
    if args.get("extract_variables", True):
        result["variables"] = _extract_variables(text)
    if args.get("extract_numbers", True):
        result["numbers"] = _extract_numbers(text)

    return [TextContent(type="text", text=json.dumps(result, indent=2, ensure_ascii=False))]


async def _math_diff(args: dict) -> list[TextContent]:
    a_str = str(args.get("a", ""))
    b_str = str(args.get("b", ""))
    mode = args.get("mode", "numeric")
    rtol = args.get("rtol", 1e-5)
    variables = args.get("variables", {})

    result: dict[str, Any] = {"mode": mode}

    if mode == "numeric":
        try:
            a_val = float(a_str)
            b_val = float(b_str)
            result["comparison"] = _compare_numeric(a_val, b_val, rtol=rtol)
        except ValueError:
            result["error"] = "Cannot parse as numeric. Try symbolic or text mode."

    elif mode == "symbolic":
        try:
            import sympy
            a_expr = sympy.sympify(a_str)
            b_expr = sympy.sympify(b_str)
            diff = sympy.simplify(a_expr - b_expr)
            result["equivalent"] = diff == 0
            result["difference"] = str(diff)
            result["a_normalized"] = str(sympy.simplify(a_expr))
            result["b_normalized"] = str(sympy.simplify(b_expr))
        except ImportError:
            result["error"] = "sympy not installed. Install with: pip install sympy"
        except Exception as e:
            result["error"] = f"Symbolic comparison failed: {e}"

    else:  # text mode
        result["a_tokens"] = a_str.lower().split()
        result["b_tokens"] = b_str.lower().split()
        a_set = set(result["a_tokens"])
        b_set = set(result["b_tokens"])
        result["jaccard_similarity"] = len(a_set & b_set) / len(a_set | b_set) if (a_set | b_set) else 0.0

    return [TextContent(type="text", text=json.dumps(result, indent=2, ensure_ascii=False))]


async def _dimensional_analysis_tool(args: dict) -> list[TextContent]:
    quantity = args.get("quantity", "")
    result = _dimensional_analysis(quantity)
    return [TextContent(type="text", text=json.dumps(result, indent=2, ensure_ascii=False))]


async def _track_precision(args: dict) -> list[TextContent]:
    values = args.get("values", [])
    tracked = []
    for v in values:
        s = str(v)
        # Count significant figures (simplified)
        stripped = s.lstrip("-0.")
        sig_figs = len(stripped.replace(".", "").replace("e", "").replace("+", "").replace("-", ""))
        tracked.append({
            "original": s,
            "significant_figures": max(1, sig_figs),
            "parsed": float(s) if _is_float(s) else None,
        })
    return [TextContent(type="text", text=json.dumps(tracked, indent=2, ensure_ascii=False))]


async def _normalize_expression(args: dict) -> list[TextContent]:
    expr = args.get("expression", "")
    result = _parse_expression(expr)
    return [TextContent(type="text", text=json.dumps(result, indent=2, ensure_ascii=False))]


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="math-anything-mcp",
                server_version="0.1.0",
                capabilities=app.get_capabilities(),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())
