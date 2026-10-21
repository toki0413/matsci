# 侧边任务使用文档

> 适用版本：侧边任务（fork + 转正）落地后。
> 代码入口：`huginn/side_tasks.py`、`huginn/routes/side.py`、`desktop/src/components/panels/SidePanel.tsx`
> 回归门：`tests/test_side_tasks.py`（CI: `Side tasks contract guard`）

## 1. 它解决什么问题

主任务跑到一半时，用户或 agent 产生一个次要需求。传统做法要么打断主任务，
要么新开会话重新交代背景。侧边任务让这个需求：

- **从当前对话的某条消息 fork 出去**，自动继承该点之前的上下文；
- 在独立 thread 里运行，**不占用、也不打断主任务**；
- 处理完可以**一键丢弃**（零残留），或者**转正**为正式任务（任务卡片 + 历史 + 产物）。

一句话：侧边任务是主对话的“草稿纸分支”，用完即弃；聊出价值时原地转正。

## 2. 核心概念

| 概念 | 说明 |
|---|---|
| 侧边任务 | 一个普通 thread，id 形如 `side_ab12cd34`；额外元数据存在任务表里 |
| fork 点 | 父对话里某条 user/assistant 消息；只继承该点及之前的上下文 |
| 种子 | fork 时快照下来的继承消息，只在侧边 thread 第一次对话时注入一次 |
| 转正 | 状态从 `side` 翻到 `promoted`，生成任务卡片；**thread_id 不变** |
| 产物 | 侧边对话中写类工具的目标路径（`path` / `file_path` / `output` / `output_path`） |

关键设计：侧边任务复用现有 checkpointer / WS / REST 链路，不是第二套对话系统。
转正不搬家，历史消息与产物自然延续。

## 3. 界面使用

### 3.1 从某条消息 fork

1. 打开一个已有对话（主 thread 必须已在后端注册）。
2. 鼠标悬停在任意 user 或 assistant 消息上，点击 **GitFork** 图标。
3. 主对话右侧会打开 **Side Tasks** 临时面板（与主对话同屏，不打断主流程），
   并聚焦到新建的侧边任务。
4. 顶部可展开 `Inherited context (N messages from <parent thread>)` 查看继承内容。

顶部栏的 **PanelRight** 按钮可随时开合该侧边面板；左侧导航的 **Side Tasks**
页签则提供全页管理视图（列表 + 详情双栏）。

### 3.2 继续侧边对话

在底部输入框输入问题，`Enter` 发送（`Shift+Enter` 换行）。
侧边任务使用独立 thread，主对话不会被插入任何消息。
窄栏（dock）模式下，点任务进入详情；左上角返回箭头回到任务列表。

### 3.3 两种收尾

- **Discard**：删除任务、thread 注册与 SQLite checkpointer 行，界面和磁盘都不留任务痕迹。
- **Promote**：转为正式任务，生成任务卡片，自动切回主对话并刷新线程列表。
  转正后的任务出现在任务列表/Threads 面板中，历史与产物保留。

也可以通过面板顶部的 **New side task** 直接以当前对话末尾为 fork 点新建。

## 4. HTTP API

所有端点在 `huginn/routes/side.py`。旧的 `/side` 轻量问答队列仍然保留；
下面的 `/side/tasks` 是侧边任务链路。

### 4.1 创建侧边任务（fork）

```http
POST /side/tasks
Content-Type: application/json

{
  "parent_thread_id": "main_thread_id",
  "from_message_index": 3,
  "title": "可选标题",
  "inherit_context": true
}
```

- `from_message_index`：父对话 **user/assistant 过滤后**的下标（含该条），从 0 开始。
  省略表示继承父对话全部 user/assistant 消息。
- `inherit_context=false`：不继承上下文（仍创建独立侧边任务）。
- `title` 省略时取种子里第一条 user 消息（截断 80 字符）；仍为空则用 `Side task`。

成功响应：

```json
{
  "success": true,
  "task": {
    "id": "side_ab12cd34",
    "thread_id": "side_ab12cd34",
    "title": "GaN k 点核查",
    "parent_thread_id": "main_thread_id",
    "fork_message_index": 3,
    "seed_count": 4,
    "seed_consumed": false,
    "message_count": 0,
    "artifacts": [],
    "status": "side",
    "is_promoted": false
  },
  "inherited_messages": 4
}
```

失败是**如实失败**，不会包装成“继承到空上下文”：

```json
{"success": false, "error": "parent thread 'x' not found"}
{"success": false, "error": "from_message_index 99 out of range (0..3)"}
```

### 4.2 列表 / 详情

```http
GET /side/tasks                # 全部（顺带执行 TTL 清理）
GET /side/tasks?status=side    # 只看侧边任务
GET /side/tasks?status=promoted # 只看已转正任务
GET /side/tasks/{task_id}      # 详情，额外返回 seed_messages / messages
```

列表响应包含 `expired` 字段，列出本次清理掉的过期侧边任务 id。

### 4.3 继续对话

```http
POST /side/tasks/{task_id}/messages
Content-Type: application/json

{"content": "4x4 k 点对 GaN 够吗？"}
```

响应（侧边任务走非流式，直接返回完整回答）：

```json
{
  "success": true,
  "answer": "对 GaN 的常规计算够用，正式能带计算建议加密到 6x6。",
  "artifacts": [{"path": "out/kpoints.png", "tool": "file_write_tool"}],
  "task": {"message_count": 2, "...": "..."}
}
```

失败语义：

- `content is required`：空消息；
- `no LLM model configured`：后端没有可用模型；
- `side agent returned an empty answer`：模型没有产出有效回答。
  此类失败**不会**把 user/assistant 轮次写入任务历史。

### 4.4 转正

```http
POST /side/tasks/{task_id}/promote
Content-Type: application/json

{"title": "可选新标题"}
```

响应：

```json
{
  "success": true,
  "thread_id": "side_ab12cd34",
  "task_card": {
    "task_id": "side_ab12cd34",
    "title": "GaN k 点核查",
    "goal": "主任务: 计算 GaN 带隙",
    "status": "active",
    "source_thread_id": "main_thread_id",
    "fork_message_index": 3,
    "history_count": 6,
    "artifacts": [{"path": "out/kpoints.png", "tool": "file_write_tool"}],
    "created_at": "2026-10-21T08:00:00+00:00",
    "promoted_at": "2026-10-21T08:30:00+00:00"
  }
}
```

重复转正返回：

```json
{"success": false, "error": "side task 'side_x' not found or already promoted"}
```

### 4.5 丢弃（零残留）

```http
DELETE /side/tasks/{task_id}
```

依次删除：任务表条目 → thread 注册 → SQLite checkpointer 的 `checkpoints` / `writes` 行。
checkpointer 不是 SQLite 或数据库不存在时，清理为 best-effort，不会阻断请求。

## 5. 生命周期与清理规则

| 规则 | 值 | 可配置项 |
|---|---|---|
| 侧边任务空闲 TTL | 1 小时 | `HUGINN_SIDE_TASK_TTL_S` |
| 任务表落盘路径 | `<runtime_home>/side_tasks.json` | `HUGINN_SIDE_TASKS_PATH` |
| checkpointer 路径 | 配置的 `checkpointer_path`，否则 `HUGINN_CHECKPOINTER_PATH` / `<runtime_home>/checkpoints.sqlite` | 见 `huginn/checkpointer.py` |
| TTL 清理触发点 | 每次 `GET /side/tasks` | — |
| 已转正任务 | 不参与 TTL | — |

“零残留”的范围：任务条目、thread 注册、SQLite checkpointer 行。审计日志/事件总线若
在其他链路记录过侧边事件，不在本功能的删除范围内。

## 6. 行为细节与诚实边界

- **fork 下标口径**：只统计 user/assistant，tool 与 system 消息不参与计数。
  前端按钮与后端使用同一口径换算。
- **种子只注入一次**：侧边 thread 第一次 `agent.chat()` 时把种子写入
  ConversationTree，随后 `seed_consumed=true`，由 checkpointer 接管后续历史，
  不会每轮重复放大上下文。
- **继承内容**：只继承 user/assistant 文本；工具消息不进入种子，避免重放
  dangling tool calls 导致 provider 400。
- **助手回复采集**：侧边消息接口取流式状态中最后一条非空 AI 文本作为回答。
- **产物识别**：只认写/执行工具（`huginn/permissions.py::WRITE_EXEC_TOOLS`）的
  `path` / `file_path` / `output` / `output_path` 参数，按路径去重；读类工具不产生产物，
  工具输出正文里的路径不会被反推。
- **转正的“卡片”**：任务卡片保存在任务表和 thread 元数据里；主对话/Threads 面板
  使用同一个 thread，历史与产物不需要迁移。
- **种子消费时机**：在 chat 开始时置位。若该轮在 LLM 返回前异常，种子不会自动重放；
  已写入 checkpointer 的部分仍保留。极端情况下重新 fork 即可。
- **旧问答队列**：`POST /side`、`GET /side/pending` 等接口保留兼容；桌面端“侧边”
  面板已替换为侧边任务面板。

## 7. 故障排查

| 现象 | 原因 / 处理 |
|---|---|
| `parent thread 'x' not found` | 父 thread 未注册；先在主对话发过消息或确认 thread id |
| `from_message_index ... out of range` | 下标按 user/assistant 过滤后计算；确认消息条数 |
| `inherited_messages: 0` | 父对话没有 checkpointer 历史，或只有 tool/system 消息 |
| `no LLM model configured` | 后端未配置模型；侧边任务需要真实模型才能回答 |
| `side agent returned an empty answer` | 模型无有效输出；任务历史未写入，可重试 |
| 侧边任务 1 小时后消失 | TTL 清理；调大 `HUGINN_SIDE_TASK_TTL_S` |
| 转正后找不到任务 | 到 Threads / 任务列表查看；thread_id 与侧边任务相同 |
| 丢弃后 checkpointer 仍有行 | 非 SQLite 后端或数据库路径不匹配；检查 `checkpointer_path` |

## 8. 开发与回归

```bash
# 契约门（红蓝双轨）
python -m pytest tests/test_side_tasks.py -q -o addopts=""

# 治理台账与契约审计
python -m huginn.cli.governance_audit --out docs/governance-state.md
python -m huginn.cli.contract_audit --check --baseline tests/golden/mece_findings_baseline.txt

# 前端
cd desktop && npx tsc --noEmit && npx vitest run
```

新增行为请同时补：

1. `tests/test_side_tasks.py` 的红/蓝用例（成功路径 + 拒绝路径）；
2. `governance_audit.CAPABILITY_SPECS` 的 `side_tasks` 锚点（模块存在 + 测试存在）；
3. `.github/workflows/ci.yml` 的 `Side tasks contract guard`（通常已存在，确认未悬空）。
