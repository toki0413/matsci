# MCP 服务器（`servers/`）

一组用 Python + `mcp` SDK 实现的 **MCP（Model Context Protocol）服务器**，
供 Huginn agent 通过 MCP 调用外部数据库与通用能力。均为纯 Python、无 Node 依赖。
每个服务器用 `python server.py [--transport stdio|sse]` 启动（默认 stdio）。

| 目录 | 服务器名 | 能力 |
|---|---|---|
| `mat-db-mcp/` | `mat-db-mcp` | 材料数据库查询：Materials Project (MP)、AFLOW、NOMAD、OQMD、NIST 势。API 不可用时回退内置 mock 数据 |
| `math-anything-mcp/` | `math-anything-mcp` | 数学语义：公式/变量抽取（LaTeX）、数学 diff（两个表达式等价性）、量纲分析、数值精度追踪 |
| `vision-pixel-mcp/` | `vision-pixel-mcp` | 通用像素视觉（PIL/numpy，无 Node/tesseract）：裁剪、主色提取、逐像素 diff、洪泛抠图、SVG 矢量化、看图问答 |

> 与 `huginn` 内置的 `image_analysis_tool`（材料 SEM/TEM/EDS 分析）互补；
> `vision-pixel-mcp` 提供通用像素操作，`image_analysis_tool` 提供材料专用分析。

## 快速使用

```bash
# stdio 传输（默认，agent 本地子进程调用）
python mat-db-mcp/server.py

# SSE 传输（远程调用）
python mat-db-mcp/server.py --transport sse
```

每个 server 也可独立从 PyPI 安装（即可被 `uvx`/`pipx` 拉起，也满足 MCP 生态目录收录）：

```bash
# mat-db:   pip install matsci-mat-db-mcp          # 接真实 MP 数据需 [db] extra
#           uvx --from matsci-mat-db-mcp mat-db-mcp
# math:     pip install matsci-math-anything-mcp
#           uvx --from matsci-math-anything-mcp math-anything-mcp
# vision:   pip install matsci-vision-pixel-mcp
#           uvx --from matsci-vision-pixel-mcp vision-pixel-mcp
```

各服务器依赖（`mcp` 客户端库、`PIL`/`numpy` 等）由 `agent` 的 Python 环境提供；
如需独立部署，请在对应目录内 `pip install mcp pillow numpy`。

## MCP 配置

在 `agent/huginn.toml` 的 `[mcp_servers]` 表（对应 `HuginnConfig.mcp_servers`，
见 `agent/huginn/config.py`）注册服务器后，agent 即可把它们作为工具调用。
具体配置字段见 `agent/docs/plugins-contract.md` 与 `agent/docs/tools-contract.md`。