# Huginn · Hugging Face Space 演示

一个 Space 内四个标签页承载项目核心能力，零 LLM、零凭据、确定性、可离线复现：

1. **缺度追问取证** — 把互相矛盾的文献报道值按真实物理自由度拆开归因 → 暴露缺度
   → 对象级取证（fid + sha256 快照）→ 门禁不变量（不假装一致、不捏造自由度）
2. **判别对比** — 未分组 `mixed` vs 分组后的诚实判定差异
3. **MCP 能力** — mat-db / math-anything / vision-pixel 三个可独立发布的
   MCP server 工具清单 + 取主色交互（预置示例图，可上传自己的）
4. **系统概览** — 版本 / 能力集装箱数量 / 组件组成

## 本地运行

```bash
pip install -r demo/requirements.txt
cd demo
python app.py          # http://127.0.0.1:7860
```

## 部署到 Hugging Face Spaces

1. 新建 Space（适合 `Gradio`，硬件选免费 CPU 档）。
2. 把本目录的 `app.py`、`requirements.txt`、`assets/` 推到 Space 仓库。
3. Space 依赖 `huginn-agent @ git+…matsci`,仓库需为 **public** 可 clone。

> `demo_core.py` 是全部演示逻辑（无 Gradio 依赖、可单元测试），`app.py` 只做 UI 壳。