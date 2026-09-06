# SPEC: OpenWorker 代码采纳

OpenWorker (andrewyng/openworker, MIT License, Copyright (c) 2024 Andrew Ng)
源码本地副本: `_external/openworker/openworker_ref/`

6 条改造, 每条标明复用源 + 适配点 + 估算行数. 优先级 P0 → P2.

---

## P0-1: Path scoping 默认开

**改动点**: [security/sandbox.py:104](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/security/sandbox.py) `strict_work_dir: bool = False` → `True`

**复用源**: 无 (改默认值)

**适配**:
- 现有用户若依赖"任意 cwd", 加 env var `HUGINN_SANDBOX_RELAX=1` 兜底
- `allowed_work_dirs` 默认空 set 时, 启动日志 warn "strict_work_dir=True but allowed_work_dirs empty, falling back to cwd"

**估算**: ~5 行 (改默认值 + env var + warn)

---

## P0-2: Audit secret 脱敏

**改动点**: [events/audit_log.py](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/events/audit_log.py) 在 `_BufferedAuditWriter` 写入前插入 `_sanitize_args`

**复用源** (直接拷贝, ~30 行):
- [openworker/audit.py:123-137](file:///c:/Users/wanzh/Desktop/matsci-agent/_external/openworker/openworker_ref/coworker/audit.py) `_SECRET_KEYS` + `_BODY_KEYS` + `_summarize` 三层 redact
- [openworker/audit.py:152-169](file:///c:/Users/wanzh/Desktop/matsci-agent/_external/openworker/openworker_ref/coworker/audit.py) `_resource` 自动提 url/owner/repo/issue_key/page_id/ticket_id/calendar_id/message_id

**适配**:
- openworker 用 SQLite 13 列, huginn 用 JSONL + hash chain
- 脱敏插在 `_compute_hash` 之前 (保证 hash 是脱敏后的, 防 hash 泄露 secret)
- `_resolve_audit_path` 路径不变, 只改 `_BufferedAuditWriter._flush_one`

**估算**: ~35 行 (拷贝 _sanitize_args + _resource + 在 _flush_one 调用)

---

## P1-3: Standing rules 升级到 tool+target 维度

**改动点**: [agent/code_act_loop.py:113](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/agent/code_act_loop.py)
- `_auto_approved: dict[str, set[str]]` 从 `{session_id: {risk_level}}` 改成 `{session_id: {(tool_name, target_key)}}`

**复用源** (直接拷贝 + 改适配, ~80 行):
- [openworker/permissions.py:62-80](file:///c:/Users/wanzh/Desktop/matsci-agent/_external/openworker/openworker_ref/coworker/permissions.py) `task_rules: dict[str, set[str]]` 数据结构
- [openworker/permissions.py:204-214](file:///c:/Users/wanzh/Desktop/matsci-agent/_external/openworker/openworker_ref/coworker/permissions.py) `_under_writable_root` path 校验
- [openworker/audit.py:152-169](file:///c:/Users/wanzh/Desktop/matsci-agent/_external/openworker/openworker_ref/coworker/audit.py) `_resource` 提取 target (复用 P0-2)

**适配**:
- openworker 只对 EXTERNAL risk 生效 (exec/write-local 永远问); huginn 保留对 medium 风险也生效, 但 high 风险永远问
- target 提取按工具类型分发:
  - `file_*` / `bash_tool`: target = path (resolve 到 allowed_work_dirs 下)
  - `code_tool`: target = file path in code
  - `web_search_tool` / `literature_tool`: target = query hash (前 8 位)
  - 其他: target = "*"
- 老的 `_is_auto_approved(session, risk_level)` 改成 `_is_auto_approved(session, tool_name, target)`, 内部先查 (tool, target) 精确匹配, 再查 (tool, "*") 通配

**估算**: ~80 行 (拷贝 _resource + 改 _auto_approved 数据结构 + 适配 6 种工具的 target 提取)

---

## P1-4: Durable resume 单 turn

**改动点**: 在 [runtime/checkpoint.py](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/runtime/checkpoint.py) 加 `resume_unanswered_tool_calls()`, 在 [agent/core.py](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/agent/core.py) 主循环入口接续跑

**复用源** (直接拷贝, ~40 行):
- [openworker/engine.py:268-292](file:///c:/Users/wanzh/Desktop/matsci-agent/_external/openworker/openworker_ref/coworker/engine.py) `_unanswered_trailing_tool_calls` 重建逻辑

**适配** (~60 行):
- openworker 用 `self.messages: list[dict]`, huginn 用 langgraph checkpointer (SqliteSaver)
- 从 checkpointer 读最近 thread_state, 找 trailing AIMessage 的 tool_calls, 跟已有 ToolMessage 的 tool_call_id 比对, 未应答的重建为 ToolCall 对象
- 接入点: `HuginnAgent.chat()` 入口, 检测到 `resume_from_step=N` 参数时, 先调 `_replay_unanswered_tool_calls()`, 再进 normal loop
- ponytail: 不改 langgraph 内部, 只在 chat 入口加 prelude

**估算**: ~100 行 (拷贝 40 + 适配 langgraph state 60)

---

## P2-5: Inbox 跨会话审批队列

**改动点**: 新建 `interaction/inbox.py` + 接入 [interaction/interrupt.py](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/interaction/interrupt.py) + 接入 [agent/code_act_loop.py](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/agent/code_act_loop.py) 的 ASK 路径

**复用源** (直接拷贝整个文件, ~250 行):
- [openworker/inbox.py:1-334](file:///c:/Users/wanzh/Desktop/matsci-agent/_external/openworker/openworker_ref/coworker/inbox.py) 整个 InboxItem + state machine + 持久化
- 5 KIND (APPROVAL/QUESTION/NOTIFICATION/DIRECTORY/PLAN)
- `pending → resolved` 幂等 + first-responder-wins
- JSON 文件持久化 (跨 surface 共享)

**适配** (~50 行):
- openworker 的 `inbox_approver` callback 接 Slack/mobile; huginn 接 EventBus + InterruptManager
- huginn 的 ClarificationManager 迁移到 Inbox 的 QUESTION kind
- huginn 的 code_act_loop ASK 路径改成创建 APPROVAL item + await resolution
- 跨 surface: huginn 暂时只有 desktop, 但保留 Inbox 接口供未来接 IM
- ponytail: 不引入 Slack SDK, 只接 EventBus. 升级路径: 加 surface adapter

**估算**: ~300 行 (拷贝 250 + 适配 EventBus/InterruptManager 50)

---

## P2-6: SelfWake timer + 轻量 Scheduler

**改动点**: 新建 `runtime/selfwake.py` + `runtime/scheduler.py` (轻量版)

**复用源** (直接拷贝两个文件, ~280 行):
- [openworker/selfwake.py:1-185](file:///c:/Users/wanzh/Desktop/matsci-agent/_external/openworker/openworker_ref/coworker/selfwake.py) WakeStore + 3 trigger (timer/completion/event) + 4 工具 (sleep_for/sleep_until/wake_on/wake_on_event)
- [openworker/automation/scheduler.py:1-113](file:///c:/Users/wanzh/Desktop/matsci-agent/_external/openworker/openworker_ref/coworker/automation/scheduler.py) run-once-catch-up + skip-on-overlap + `asyncio.create_task` spawn

**适配** (~30 行):
- openworker 的 `runner` callback 跑 TurnEngine; huginn 跑 `AutoloopEngine.run_once()`
- huginn 接入 `AutoloopEngine.__init__` 时 spawn scheduler task, 退出时 cancel
- SelfWake 4 工具注册到 ToolRegistry, agent 可主动调
- ponytail: 不引入 cron 表达式 (openworker 也没有), 只支持 interval + event. 升级路径: 接 APScheduler

**估算**: ~310 行 (拷贝 280 + 适配 AutoloopEngine 30)

---

## 总览

| # | 改造 | 复用源 | 拷贝行数 | 适配行数 | 总行数 | 优先级 |
|---|---|---|---|---|---|---|
| P0-1 | Path scoping 默认开 | 无 | 0 | 5 | 5 | P0 |
| P0-2 | Audit secret 脱敏 | audit.py:123-169 | 35 | 0 | 35 | P0 |
| P1-3 | Standing rules tool+target | permissions.py:62-80 + audit.py:152-169 | 30 | 50 | 80 | P1 |
| P1-4 | Durable resume 单 turn | engine.py:268-292 | 40 | 60 | 100 | P1 |
| P2-5 | Inbox 跨会话审批 | inbox.py:1-334 | 250 | 50 | 300 | P2 |
| P2-6 | SelfWake + Scheduler | selfwake.py + scheduler.py | 280 | 30 | 310 | P2 |
| **合计** | | | **635** | **195** | **830** | |

**复用率**: 635/830 = 76.5% 直接拷贝 openworker 代码

---

## 执行顺序

1. P0-1 + P0-2 (安全补丁, 40 行, 1 个 PR)
2. P1-3 + P1-4 (核心改造, 180 行, 1 个 PR)
3. P2-5 + P2-6 (新模块, 610 行, 1 个 PR)

每个 PR 带 assert-based self-check, 不引入测试框架.

## License 合规

OpenWorker MIT License, Copyright (c) 2024 Andrew Ng.
复用代码在每个文件头部加:
```
# Portions derived from OpenWorker (https://github.com/andrewyng/openworker)
# MIT License, Copyright (c) 2024 Andrew Ng
```
