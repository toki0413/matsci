# huginn-mcp

npx 启动器，覆盖 Huginn 的 **5 个 MCP server**。每个 bin 只是一个薄包装：
找到对应 Python 命令并 `spawn`，stdio（MCP 传输）原样透传。

## 5 个 bin

| bin | 对应 backend | 前置依赖 |
|---|---|---|
| `npx mat-db-mcp`        | 材料数据库 (MP/AFLOW/NOMAD/OQMD/NIST) | `pip install matsci-mat-db-mcp` |
| `npx math-anything-mcp` | 数学语义/量纲/精度 | `pip install matsci-math-anything-mcp` |
| `npx vision-pixel-mcp`  | 像素取色/裁剪 | `pip install matsci-vision-pixel-mcp` |
| `npx capabilities-mcp`  | 能力集装箱 (只读) | `pip install huginn-agent` |
| `npx workflows-mcp`     | 工作流蓝图 (只读) | `pip install huginn-agent` |

> 依赖在运行时（Python 侧）解析：先把对应 pip 包装好，再 `npx <bin>` 即可。
> 若 PATH 中 `mat-db-mcp` 之类与已有 Python 命令同名冲突，可只保留一个。

## 用法（Claude Desktop / Cursor 等 MCP host）

```json
{
  "mcpServers": {
    "workflows-mcp":  { "command": "npx", "args": ["workflows-mcp"] },
    "mat-db-mcp":     { "command": "npx", "args": ["mat-db-mcp"] }
  }
}
```

## 自测

```bash
npm run test:list   # 打印 5 个 bin 名
```

源码：<https://github.com/toki0413/matsci>