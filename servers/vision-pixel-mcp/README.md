# Vision-Pixel MCP Server

Generic pixel-level vision tools MCP server — crop/enlarge, dominant-color extraction, pixel diff, foreground extraction, simple SVG vectorization, and image summary. Pure Python (`Pillow` + `numpy`); no Node / sharp / tesseract / chrome. MCP `stdio`/`sse` transports.

Part of the **Huginn** scientific research agent harness. Complements Huginn's material-specific `image_analysis_tool` (SEM/TEM/EDS) with general-purpose pixel operations.

## Install

```bash
pip install matsci-vision-pixel-mcp
```

Or run directly:

```bash
uvx --from matsci-vision-pixel-mcp vision-pixel-mcp
```

## Usage

```bash
vision-pixel-mcp                       # stdio (default)
vision-pixel-mcp --transport sse       # SSE (remote)
```

Register in your MCP client config:

```json
{
  "mcpServers": {
    "vision-pixel": { "command": "vision-pixel-mcp" }
  }
}
```

## Tools

| Tool | Description |
|---|---|
| `vision_crop` | Crop and enlarge by pixel box `x1,y1,x2,y2` |
| `vision_colors` | Dominant colors (hex + share) |
| `vision_pixel_diff` | Per-pixel diff of two images (diff rate + worst 8×8 region) |
| `vision_extract_foreground` | Flood-fill foreground extraction (solid background → transparent PNG) |
| `vision_trace` | Simple SVG vectorization (color-block outlines) |
| `vision_describe` | Basic image summary (size / mode / dominant colors) |

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT — see `LICENSE` at the repo root.