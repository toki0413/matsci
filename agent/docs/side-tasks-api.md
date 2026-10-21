# 侧边任务 API 参考

> 实现：`huginn/routes/side.py`、`huginn/side_tasks.py`
> 使用说明：[side-tasks.md](side-tasks.md) · 设计：[side-tasks-design.md](side-tasks-design.md)

## 0. 约定

- 所有请求/响应均为 JSON；请求头 `Content-Type: application/json`。
- **HTTP 状态码恒为 200**（沿用项目既有风格），业务结果以响应体 `success` 为准。
- 鉴权与其它路由一致：由全局 API key / bearer token 中间件处理；
  `request` 仅用于提取 `user_id` 写入 thread 元数据。
- 时间字段为 ISO 8601（UTC，例如 `2026-10-21T08:30:00+00:00`）。
- 侧边任务 id 与 thread id **相同**，形如 `side_ab12cd34`。

## 1. 端点总览

| 方法 | 路径 | 作用 |
|---|---|---|
| POST | `/side/tasks` | 从父对话某条消息 fork 出侧边任务 |
| GET | `/side/tasks` | 列出任务（`status` 过滤 + TTL 清理） |
| GET | `/side/tasks/{task_id}` | 任务详情（含种子与对话正文） |
| POST | `/side/tasks/{task_id}/messages` | 在侧边 thread 继续对话 |
| POST | `/side/tasks/{task_id}/promote` | 转正为正式任务 |
| DELETE | `/side/tasks/{task_id}` | 丢弃并清理 |
| POST | `/side` | 旧轻量问答队列：提交问题（兼容保留） |
| GET | `/side` / `/side/pending` / `/side/answered` | 旧队列列表 |
| GET | `/side/{question_id}` | 旧队列单问题 |
| DELETE | `/side` | 清空旧队列 |

> 路由顺序约束：`/side/tasks` 必须声明在 `/side/{question_id}` 之前，
> 否则 `tasks` 会被动态段吞掉。改动顺序时请跑 `test_side_tasks.py`。

## 2. 创建侧边任务（fork）

```http
POST /side/tasks
```

### 请求体

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `parent_thread_id` | string | 是 | 父对话 thread id，必须已注册 |
| `from_message_index` | int | 否 | fork 下标（见下） |
| `title` | string | 否 | 标题；缺省从种子里取第一条 user 消息 |
| `inherit_context` | bool | 否 | 默认 `true`；`false` 时不继承上下文 |

`from_message_index` 口径：

- 对父对话消息按 `user` / `assistant` 过滤后计数；
- 下标从 `0` 开始，**包含**该条消息；
- 省略表示继承全部 user/assistant 消息；
- 越界返回失败，不静默截断。

### 成功响应

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
    "is_promoted": false,
    "created_at": "2026-10-21T08:00:00+00:00",
    "updated_at": "2026-10-21T08:00:00+00:00",
    "promoted_at": null,
    "task_card": null
  },
  "inherited_messages": 4
}
```

### 失败响应

```json
{"success": false, "error": "parent_thread_id is required"}
{"success": false, "error": "parent thread 'main_x' not found"}
{"success": false, "error": "from_message_index must be an integer"}
{"success": false, "error": "from_message_index 99 out of range (0..3)"}
```

### 副作用

- 任务写入 `side_tasks.json`；
- 立即注册 side thread（`kind=side`、`parent_thread_id`、`fork_message_index`）。

### curl

```bash
curl -s -X POST http://127.0.0.1:8000/side/tasks \
  -H 'Content-Type: application/json' \
  -d '{"parent_thread_id":"main_thread_id","from_message_index":3,"title":"GaN k 点核查"}'
```

## 3. 列出任务

```http
GET /side/tasks?status=side
```

| 查询参数 | 取值 | 说明 |
|---|---|---|
| `status` | `side` / `promoted` / 省略 | 省略返回全部 |

响应：

```json
{
  "success": true,
  "count": 2,
  "expired": ["side_deadbeef"],
  "tasks": [ { "id": "side_ab12cd34", "seed_count": 4, "message_count": 2, "...": "..." } ]
}
```

- 每次调用都会先执行 TTL 清理；被清理的 id 出现在 `expired`。
- 列表项使用摘要投影：不返回 `seed_messages` / `messages` 正文。
- 排序：`updated_at` 倒序。

## 4. 任务详情

```http
GET /side/tasks/{task_id}
```

响应在列表字段基础上追加：

```json
{
  "success": true,
  "task": {
    "...": "摘要字段同上",
    "seed_messages": [
      {"role": "user", "content": "主任务: 计算 GaN 带隙"},
      {"role": "assistant", "content": "先做收敛测试"}
    ],
    "messages": [
      {"role": "user", "content": "4x4 够吗?", "at": "2026-10-21T08:10:00+00:00"},
      {"role": "assistant", "content": "够用", "at": "2026-10-21T08:10:05+00:00"}
    ]
  }
}
```

不存在：

```json
{"success": false, "error": "side task 'side_x' not found"}
```

## 5. 继续对话

```http
POST /side/tasks/{task_id}/messages
```

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `content` | string | 是 | 用户消息，去空白后不能为空 |

响应：

```json
{
  "success": true,
  "answer": "对 GaN 的常规计算够用。",
  "artifacts": [{"path": "out/kpoints.png", "tool": "file_write_tool"}],
  "task": { "...": "详情投影（含全部历史）" }
}
```

失败：

```json
{"success": false, "error": "content is required"}
{"success": false, "error": "side task 'side_x' not found"}
{"success": false, "error": "cannot create agent: ..."}
{"success": false, "error": "no LLM model configured"}
{"success": false, "error": "side agent failed: ..."}
{"success": false, "error": "side agent returned an empty answer"}
```

语义：

- 非流式：等待 agent 完成本轮后一次性返回最终 AI 文本；
- 只有非空回答才写入 user/assistant 轮次与产物；
- 失败时任务历史保持原样（不会留下半截轮次）；
- 首次调用会把 fork 种子注入该 thread，之后由 checkpointer 接管。

## 6. 转正

```http
POST /side/tasks/{task_id}/promote
```

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `title` | string | 否 | 新标题；缺省沿用原标题 |

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
  },
  "task": { "...": "详情投影（status=promoted）" }
}
```

失败：

```json
{"success": false, "error": "side task 'side_x' not found or already promoted"}
```

副作用：

- 任务状态 `side → promoted`；
- thread 元数据更新：`kind=task`、`promoted_from`、`task_card`；
- `thread_id` 不变，历史与产物不迁移；
- 转正后不再参与 TTL。

## 7. 丢弃

```http
DELETE /side/tasks/{task_id}
```

响应：

```json
{"success": true, "discarded": "side_ab12cd34"}
```

失败：

```json
{"success": false, "error": "side task 'side_x' not found"}
```

清理范围：

1. 任务表条目；
2. thread 注册（`_threads.pop`）；
3. SQLite checkpointer 的 `checkpoints` / `writes` 行。

非 SQLite checkpointer、库或表不存在时清理为 best-effort，不影响响应。

## 8. 旧轻量问答队列（兼容）

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/side` | body `{question, metadata?}`，返回 `{success, question}` |
| GET | `/side` | 全部问题 + `n_pending` / `n_answered` |
| GET | `/side/pending` | 待答问题 |
| GET | `/side/answered` | 已答问题 |
| GET | `/side/{question_id}` | 单问题状态/答案 |
| DELETE | `/side` | 清空 |

这些端点由 `huginn/side_conversation.py::SideChannel` 承载，与侧边任务相互独立；
新前端已改用 `/side/tasks`。

## 9. 数据字典

### 9.1 任务摘要（列表项）

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` / `thread_id` | string | 同一个值 |
| `title` | string | 标题 |
| `parent_thread_id` | string | 来源 thread |
| `fork_message_index` | int \| null | fork 下标 |
| `seed_count` | int | 继承消息数 |
| `seed_consumed` | bool | 种子是否已注入 |
| `message_count` | int | 侧边轮次数（user+assistant） |
| `artifacts` | array | `{path, tool}` |
| `status` | string | `side` / `promoted` |
| `is_promoted` | bool | 状态派生 |
| `created_at` / `updated_at` / `promoted_at` | string \| null | ISO 时间 |
| `task_card` | object \| null | 转正卡片 |

### 9.2 详情追加

| 字段 | 类型 | 说明 |
|---|---|---|
| `seed_messages` | array | `{role, content}`，fork 快照 |
| `messages` | array | `{role, content, at}`，侧边对话 |

## 10. 配置项

| 环境变量 / 配置 | 默认值 | 作用 |
|---|---|---|
| `HUGINN_SIDE_TASKS_PATH` | `<runtime_home>/side_tasks.json` | 任务表路径 |
| `HUGINN_SIDE_TASK_TTL_S` | `3600` | 侧边任务空闲 TTL（秒） |
| `HUGINN_CHECKPOINTER_PATH` | `<runtime_home>/checkpoints.sqlite` | 未配置 `checkpointer_path` 时的清理目标 |
| `checkpointer_path`（huginn.toml） | 空 | 优先级高于环境变量 |

## 11. 并发与幂等

- 任务表读写由进程内 `RLock` 串行化；单进程安全。
- 多进程同时写同一 JSON 文件会互相覆盖（已知边界，与 `todos.json` 相同）。
- `promote` 幂等性：第二次调用返回失败，不会重复生成卡片。
- `DELETE` 幂等性：第二次调用返回 `success=false`，不抛异常。
- `messages` 无内置并发控制；同一侧边任务不建议并行发多条消息。
