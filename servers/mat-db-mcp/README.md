# Mat-DB MCP Server

Materials database query MCP server — query **Materials Project (MP)**, **AFLOW**, **NOMAD**, **OQMD**, and **NIST interatomic potentials**. Pure Python, MCP `stdio`/`sse` transports. Falls back to built-in mock data when an API is unavailable, so it works out of the box.

Part of the **Huginn** scientific research agent harness.

## Install

```bash
pip install matsci-mat-db-mcp      # real MP queries:   pip install "matsci-mat-db-mcp[db]"
```

Or run directly without installing:

```bash
uvx --from matsci-mat-db-mcp mat-db-mcp
```

## Usage

```bash
mat-db-mcp                        # stdio (default; for Claude Desktop / Cursor / any MCP host)
mat-db-mcp --transport sse        # SSE (remote)
```

Register in your MCP client config:

```json
{
  "mcpServers": {
    "mat-db": {
      "command": "mat-db-mcp",
      "env": { "MP_API_KEY": "..." }
    }
  }
}
```

## Tools

| Tool | Description |
|---|---|
| `query_materials_project` | Last-mile MP / NOMAD / OQMD lookups with mock fallback |
| `search_by_property` | Filter materials by a property (band gap, energy, ...) |
| `get_structure` | Fetch structure / spacegroup / lattice for a formula |
| `query_interatomic_potentials` | NIST interatomic potential catalog |
| `compare_materials` | Side-by-side comparison of multiple materials |

No `MP_API_KEY`? The server transparently serves curated mock entries (`Si`, `GaAs`, `TiO2`, ...) so MCP integration and demos work immediately.

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT — see `LICENSE` at the repo root.