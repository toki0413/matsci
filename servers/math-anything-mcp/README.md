# Math-Anything MCP Server

Mathematical semantics and diff MCP server. Extracts equations and variables from text, checks whether two expressions/results are mathematically equivalent, carries out dimensional analysis, and tracks numerical precision. Pure Python, MCP `stdio` transport.

Part of the **Huginn** scientific research agent harness.

## Install

```bash
pip install matsci-math-anything-mcp
```

Or run directly:

```bash
uvx --from matsci-math-anything-mcp math-anything-mcp
```

## Usage

```bash
math-anything-mcp    # stdio (default; for Claude Desktop / Cursor / any MCP host)
```

Register in your MCP client config:

```json
{
  "mcpServers": {
    "math-anything": { "command": "math-anything-mcp" }
  }
}
```

## Tools

| Tool | Description |
|---|---|
| `extract_math` | Extract LaTeX equations and variable assignments from text |
| `math_diff` | Compare two expressions / results for mathematical equivalence |
| `dimensional_analysis` | Validate physical dimensions / verify computed units |
| `track_precision` | Track significant figures and numerical precision of a value |
| `normalize_expression` | Canonicalize an expression for comparison |

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT — see `LICENSE` at the repo root.