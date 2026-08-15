---
name: qwen-mm-plugins
description: Use when 需要读取/分析图像、图表、显微照片、文档截图、视频或生成可视化时。这是阿里 QwenLM 官方 Qwen-MM-Plugins core 的真实 MCP 工具集（通过 .mcp.json 接入 huginn），提供 read_image / visualize / read_video / crop / draw_bbox / media_info / save_view。触发场景：用户上传或引用本地图片/视频文件、需要 OCR 或看图提取信息、需要把数据/绘图表可视化、需要裁剪或标注图像、需要查询媒体元信息。Also use when the user asks to read an image, analyze a figure/micrograph/screenshot, extract info from an image, visualize data, crop or annotate an image, query media metadata, or read a video.
---

# qwen-mm-plugins

本 Skill 让 agent 使用**官方 Qwen-MM-Plugins core**（阿里 QwenLM 开源，经根目录 `.mcp.json` 的 `qwen-mm-plugins-core` stdio MCP server 真实接入 huginn）提供的多模态工具。这是官方插件，不是自研模块。

## 可用工具（由 MCP server 暴露，调用前先确认已注册）

| 工具 | 作用 |
|---|---|
| `read_image` | 读取本地图片并返回结构化描述/OCR 结果（支持路径或 base64） |
| `visualize` | 把数据/图表/代码/文档渲染成可视化（matplotlib 兜底；装 Blender 后走最佳质量） |
| `read_video` | 读取本地视频，抽取帧/信息 |
| `crop` | 按区域裁剪图像 |
| `draw_bbox` | 在图像上绘制边界框标注 |
| `media_info` | 查询图片/视频的元信息（尺寸、时长、码率等） |
| `save_view` | 保存当前渲染/视图 |

## 使用规则

1. 用户给的是**本地文件路径**时，直接把这个绝对路径传给工具（`read_image` 等）。不要先转 base64。
2. 需要"看图提取信息/OCR/识别图表数值" → 用 `read_image`。
3. 需要"把表格/数组/数据画成图"，或"把代码/文档渲染出来看效果" → 用 `visualize`。
4. 需要裁剪/标注 → 用 `crop` / `draw_bbox`。
5. 需要区分图/视频类型或拿元数据 → 用 `media_info`。
6. 工具返回的机器可读结果（如 OCR 文本、图表数据）优先直接引用，不要臆测图里没有的内容。

## 注意

- 官方 core 的**本地文件读取（read_image / visualize / read_video / crop / draw_bbox / media_info / save_view）不需要 API key**。
- 依赖模型能力：若当前模型不支持图像输入，`read_image` 返回的文本化结果仍可用来做后续分析——这是文本通道的用法，不要假装"看到了"原图。
- 云媒体 API（视频生成、DASHSCOPE 相关）需要 `DASHSCOPE_API_KEY`；未配置时不要调用会 401 的云能力，改用本地读取。
- 系统工具缺失（Blender / LibreOffice / pdflatex / playwright）时 `visualize` 会自动降级到 matplotlib，仍可用。
- 若该 MCP server 未连接（进程未起来），提示用户先启动 huginn 后端或检查 `.mcp.json` 配置，不要凭空捏造工具名。