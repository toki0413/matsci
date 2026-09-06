# Huginn MCP 生态目录提交手册

Huginn 的 **5 个 MCP server** 已具备发布形态（PyPI 可装 / 内置 CLI 可跑），
此手册给出投到主流 MCP 目录所需的一手资料。三处 README 均已含 mcp.so/Glama
爬取所需的 `## Tools` 表与安装块，提交后约 24h~1 周内被索引。

## 5 个 server 一览

| server | 安装 | 启动命令 | 工具 |
|---|---|---|---|
| **mat-db-mcp** | `pip install matsci-mat-db-mcp` | `mat-db-mcp` | 5 (MP/AFLOW/NOMAD/OQMD/NIST) |
| **math-anything-mcp** | `pip install matsci-math-anything-mcp` | `math-anything-mcp` | 数学语义/量纲/精度 |
| **vision-pixel-mcp** | `pip install matsci-vision-pixel-mcp` | `vision-pixel-mcp` | 取色/裁剪 |
| **capabilities-mcp** | `pip install huginn-agent` | `huginn capabilities-mcp` | 能力集装箱 (157) 只读 |
| **workflows-mcp** | `pip install huginn-agent` | `huginn workflows-mcp` | 工作流 manifest/view/render |

## 1) mcp.so — 表单提交（推荐，流量最大 ~20k servers）

登录 GitHub → 打开 **https://mcp.so/submit** → 逐个提交，字段如下：

> `Type`=MCP Server · `Name`=下文的 slug · `URL`=`https://github.com/toki0413/matsci`
> · Category 按工具性质选（`Data & Tooling` / `Developer Tools`）

| name (slug) | 一句话描述（可直接粘贴） | tags |
|---|---|---|
| `mat-db-mcp` | 查询 Materials Project / AFLOW / NOMAD / OQMD / NIST 势函数的材料科学数据库 MCP server | `materials`, `database`, `ml`, `df` |
| `math-anything-mcp` | 数学语义提取 / 表达式等价 diff / 量纲分析 / 数值精度跟踪的数学 MCP server | `math`, `latex`, `symbolic`, `units` |
| `vision-pixel-mcp` | 按像素精确取色（hex+rgb+占比）与区域裁剪的视觉 MCP server | `vision`, `image`, `color`, `pixel` |
| `capabilities-mcp` | 把 Huginn 157 个能力集装箱暴露成 MCP（原子+复合，只读） | `agent`, `tools`, `llm` |
| `workflows-mcp` | 把 Huginn 命名工作流（DFT/MD/CFD/证明管线）以蓝图 MCP 暴露 | `workflow`, `science`, `computation` |

mcp.so 会自动从 README 拉取描述、首个 ```` ```bash ```` 安装块、`## Tools` 表。
不要再重复提交，它每周重拉一次 README。

## 2) Smithery — CLI 发布（需 API key）

```bash
npm install -g @smithery/cli
export SMITHERY_API_KEY=sk_你的key   # https://smithery.ai/account/api-keys 生成
# 3 个 pip 版 server（stdio，经 PyPI 安装镜像）：
smithery mcp publish "https://pypi.org/project/matsci-mat-db-mcp"   -n toki0413/mat-db-mcp
smithery mcp publish "https://pypi.org/project/matsci-math-anything-mcp" -n toki0413/math-anything-mcp
smithery mcp publish "https://pypi.org/project/matsci-vision-pixel-mcp" -n toki0413/vision-pixel-mcp
# capabilities / workflows 依赖 huginn-agent，用命令入口：
smithery mcp publish "https://pypi.org/project/huginn-agent" -n toki0413/capabilities-mcp
smithery mcp publish "https://pypi.org/project/huginn-agent" -n toki0413/workflows-mcp
```

## 3) 其它目录（低成本白捡）

- **Glama** `glama.ai/mcp/servers`：给 GitHub 仓库加 topics `mcp`、`model-context-protocol` 后
  会自动索引；也可网页表单加 repo URL。
- **PulseMCP** `pulsemcp.com`：会从官方 Registry 周同步，把 server 挂到官方 Registry 即自动扩散。
- **awesome-mcp-servers**（`github.com/punkpeye/awesome-mcp-servers`）：开一个 PR 加一行
  `- [mat-db-mcp](https://github.com/toki0413/matsci), 查询 Materials Project 等材料数据库的 MCP server`。

## 4) 真实 MCP host 调用演示（stdio，已实测）

用官方 `mcp` 客户端 `ClientSession` 通过 stdio 起 `workflows-mcp`，走标准 MCP 协议真实调用：

```
--- tools/list (真实 MCP host 视角) ---
  workflow_manifest: 罗列全部命名工作流(含类型/规模/参数 schema)
  view_workflow:     导出某工作流的单文件定义, 可搬运/再导入
  render_workflow:   渲染某模板的 stage 骨架(纯描述, 不执行)
--- tools/call workflow_manifest ---
  success= True  count= 15
  前3: ['aimd_workflow', 'defect_workflow', 'dft_verify_workflow']
--- tools/call view_workflow ---
  spec= huginn/workflow 1  kind= template
```

连接代码（任何装了 `mcp` 的客户端都可）：

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

params = StdioServerParameters(command='python',
    args=['-m','huginn.workflows.mcp_export','--transport','stdio'])
async with stdio_client(params) as (read, write), ClientSession(read, write) as s:
    await s.initialize()
    tools = await s.list_tools()                 # → 3 个工具
    res = await s.call_tool('workflow_manifest', {})  # → 15 个工作流
```

## 5) 用户侧接入示例（贴进 README 供复制）

```json
{
  "mcpServers": {
    "mat-db-mcp": { "command": "mat-db-mcp" },
    "workflows-mcp": { "command": "huginn", "args": ["workflows-mcp"] }
  }
}
```