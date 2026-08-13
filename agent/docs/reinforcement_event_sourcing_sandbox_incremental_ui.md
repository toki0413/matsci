# Huginn 补强设计：事件溯源 · 沙箱机制 · 增量前端会话引擎

> 状态：**设计阶段（拟稿）** · 日期：2026-08-13
> 依据：DeepSeek Harness（`/tmp/dsh-repo`）与 Oh-my-pi（`/tmp/omp-repo`）三方对照调研。
> 原则：不推翻现有体系，做增量嫁接；复用 Huginn 已有的
> `EventBus`/`audit_log`（防篡改哈希链）、`ToolMetadata`（fail-closed）、
> `UnifiedSessionState`；把"事件日志"升格为会话 source of truth，把"前端
> 渲染"改为块级增量。

---

## 0. 三方对照（dsh / Oh-my-pi / Huginn）

| 维度 | DeepSeek Harness (dsh) | Oh-my-pi | Huginn 现状 |
|---|---|---|---|
| 事件溯源 | 投影层 `ProjectionDefinition{init/apply/view/stateVersion}` + 弱引用 cell 缓存 | 纯追加 JSONL 事件日志 + 树/leaf 指针 + `buildSessionContext` 重放 | 可变状态快照 + 事件总线仅做审计/可观测，事件与状态脱钩 |
| 会话分支 | 投影依赖图重放 / 一致读切面 | 事件级分支（leaf 指针移动，历史不丢）+ compaction 可见分界 | LangGraph checkpoint 快照分支，非事件级 |
| 沙箱 | **Landlock ABI 协商硬隔离**（`landlock-run`，`fs_mask_for_abi`+`NO_NEW_PRIVS`）+ spill 溢出 | 工作区 CoW/overlay 隔离（`pi-iso`）+ 审批 + 非交互 env，无 syscall 隔离 | 宿主 subprocess 白名单 + 可选容器，无 Landlock/seccomp |
| 工具结果溢出 | `spill-policy`：`maxInlineBytes` + 预览替换 + `artifact://` 定位器 | `OutputSink`：50KB 尾窗 + `artifact://` 引用 | 一次缓存整串 `tool_result` |
| 工具权限优先级 | hooks merge：**deny > ask > allow** | `bash.patterns` allow/deny/prompt + 强制 exec 审批 | 正则拦截 + 白名单平铺 |
| 增量前端 | 冻结块 + 溢出 spill + 输入状态机（draft/claim/paste/undo/redo） | commit 账本 + live-region seam + frozen-token 前缀 + 差分发射 | 整串字符串追加 + Virtuoso 虚拟滚动，无冻结块/块复用 |

**结论**：事件溯源与增量渲染以 Oh-my-pi 为主参照（最完整）；沙箱以 dsh 的
`landlock-run` 为具体 C 实现模板（Oh-my-pi 无 syscall 隔离）；工具溢出与
权限优先级复用 dsh/Oh-my-pi 现成模式。

---

## 1. 事件溯源补强

**目标**：把会话状态从"可变快照"改为"事件日志 + 投影"，可完整重放/分支/
回滚，且与现有 LangGraph checkpoint 兼容。

### A1 — 事件日志升格为会话 source of truth（`SessionEventLog`）

新增 `huginn/events/session_log.py`：

- 事件类型（复用 `event_types.py`，补结构化）：`message`、`reasoning`、
  `tool_call/result`、`model_change`、`phase_change`、`cognitive_mode_change`、
  `compaction`、`branch_summary`、`reset_boundary`、`custom(namespaced)`。
- 每条带 `seq`（单调）、`id`、`parent_id`、`ts`、`payload`。**追加式 JSONL**
  （复用 `audit_log.py` 的分片 + 哈希链能力，但服务会话 replay 而非审计）。
- 接口：`append()` / `read_after(seq)` / `build_state(leaf_id)`。

**来源**：Oh-my-pi `docs/session.md`（纯追加事件日志 + 树/leaf）。

### A2 — 投影层（`Projection`）

接口签名直接对齐 dsh `session-projection`：

```python
class ProjectionDefinition(Generic[K, S]):
    key: K
    def init(self) -> S: ...
    def apply(self, state: S, event: SessionEvent) -> S: ...   # 纯函数
    def view(self, state: S) -> SessionProjectionMap[K]: ...
    stateVersion: int                                           # 投影契约版本，变更则整体重建
```

- 投影清单：`SessionProjection`（喂给 LangGraph 输入）、`RuntimeStateProjection`
  （phase/cognitive_mode/model map）、`UiProjection`（喂给前端增量引擎）。
- **惰性重建 + 弱引用缓存**：`cell.state` 按 session 缓存，能重放
  `events.slice(0, seq)`，支持依赖图重放与一致读切面。
- **`stateVersion`**：投影契约变更时整体重建，避免跨版本状态漂移。

**来源**：dsh `packages/session/session-projection/src/index.ts`。

### A3 — 事件级分支/回滚

```python
class SessionLog:
    def branch(self, target_seq: int) -> BranchHandle: ...   # 移动 leaf，历史不变
    def branch_with_summary(self, target_seq, summary) -> None: ...
    async def rollback_to(self, seq: int) -> Snapshot: ...    # 从事件重放重建
    def snapshot(self) -> Snapshot: ...                       # 物化当前投影作快速起点
```

- 落地：`snapshot.take/revert` 事件类型已有，补对应执行器（`take` → 物化投影存
  `state_store`；`revert` → 从最新快照 + `read_after(snapshot_seq)` 重放到目标 seq）。
- **与 LangGraph 兼容**：`SessionProjection.apply()` 的产物直接喂给现有
  `graph.update_state()`；事件日志成为 checkpointer 的"事实来源"，checkpoint
  退化为性能缓存。

**来源**：Oh-my-pi 树/leaf 指针 + dsh 投影重放。

### A4 — compaction 语义对齐 Oh-my-pi

- `EventLog` 加 `CompactionEntry{ boundary_seq, summary, first_kept_seq }`；
  `build_state()` 只把 boundary 后的消息送入 LLM，**全历史保留**。
- 改进点：现有 `memory/session.py` 滑动窗口"丢旧消息" → 压缩不再丢数据，
  前端可见 `── compacted ──` 分隔（见 C2）。

**来源**：Oh-my-pi `docs/compaction.md`。

---

## 2. 沙箱机制补强

**目标**：默认路径从"宿主 subprocess 软沙箱"升级为**分层强制隔离**，复用
现有 `ToolMetadata` 权限体系。

### B1 — 进程级强制隔离（Landlock + seccomp）

参照 dsh `landlock-run`，新增 `huginn/security/landlock.py`：

- 探测 `landlock_create_ruleset` ABI（`<0` → 不支持，`<MAX_ABI` → 降级掩码），
  `prctl(NO_NEW_PRIVS)` + `restrict_self`。
- 规则：`ro_paths` 只读（`EXECUTE|READ_FILE|READ_DIR`），`rw_dirs`
  （`allowed_work_dirs`）全量，其余拒绝。规则集跨 execve 继承。
- 与现有 `SandboxExecutor` 集成：`Landlock.create(ro=..., rw=...)` 在
  `subprocess.run(preexec_fn=...)` 前应用；**不支持时降级回现有白名单**。
- 可选 seccomp：禁 `clone(多个)/mount/ptrace/process_vm_readv` 等。

**B1 增强 — 审批优先级（deny > ask > allow）**：把 dsh `hooks merge` 定级套到
`_validate_command()`：

```python
# 显式三态，优先级 deny > ask > allow
def evaluate_command(command):  # -> DENY | ASK | ALLOW
    if match_deny(command): return DENY          # 命中 deny 直接拒绝
    if match_ask(command):   return ASK          # requires_confirmation / is_destructive
    if match_allow(command): return ALLOW
    return ASK                                   # 默认 fail-closed
```

deny 永远压过 allow，避免"先 allow 后 deny"被绕过。替换现有 `check_command_safety`
的平铺正则拦截。

**来源**：dsh `landlock-run/src/main.c`（ABI 协商模板）+ `hook-protocol/merge.ts`
（deny>ask>allow）。

### B2 — 工具级执行后端绑定

- 新增 `ToolMetadata.sandbox_hint: Literal["host","container","paranoid","any"]`
  （默认 `any`）。
- `build_executor(tool)` 按 hint 选择：`paranoid` 强制容器（`require_digest=True`+
  `no_new_privileges`+`drop_all_capabilities`+`network_none`，缺容器则**拒绝而非
  静默回退**）；`host` 走 Landlock/seccomp。
- `is_destructive` 工具默认 `paranoid`。

### B3 — in-process exec 逃逸加固

- `restricted_python.py` AST 预扫描补一道：禁
  `object.__getattribute__`/`type.__subclasses__`/`().__class__.__bases__` 链、
  `gc.get_objects`、`ctypes`、`sys.modules` 遍历、`__import__` 间接。
- 内存监控从 `tracemalloc` 峰值升级为 `resource.setrlimit(RLIMIT_AS)` 软上限。
- 长期：为 `eval` 提供容器后端；`paranoid` hint 时 in-process exec 直接禁用。

### B4 — 漏洞面管理

- `ContainerSecurityConfig.require_digest`：`paranoid` 工具强制 digest 固定，
  无 digest 拒绝启动。
- HPC 远程执行补：作业脚本内 `ulimit` + 可选容器包裹；禁
  `--privileged`/root 常见提权 pattern。

---

## 3. 增量前端会话引擎补强

**目标**：把"整条字符串追加"改为"冻结块 + 增量 diff"，复用后端事件投影，
端到端增量。

### C1 — 块级消息模型（冻结块 + 增量追加）

- 后端 `UiProjection` 为每条消息维护 `blocks: [{kind, text, frozen, rev}]`；
  流式 `text_delta` 只更新最后一块，`frozen` 定稿后不可变。
- 前端新增 `useIncrementalMessages` hook：消息内容按块存储，`delta` 到达时只
  追加/替换尾部块；`frozen` 块跳过重渲染（`memo` 到块粒度）。
- 移植 dsh `ui-conversation` 冻结语义：`renderStablePrefix`——消息数组引用 +
  offset 不变则整段跳过。

**来源**：Oh-my-pi commit 账本 + frozen-token 前缀；dsh 冻结块。

### C2 — 增量 diff 同步 + compaction 分隔

- 会话切换/重连拉 `GET /threads/{id}/events?after=<seq>` 增量事件，前端
  `apply(event)` 块级更新（**替代全量 `GET /messages` 重建**）。
- `context_compacted` 升级：后端发 `compaction` 事件，前端在压缩点插入
  `── compacted ──` 块，**旧消息块保留**（与 A4 打通）。

**C2 细分 — 工具结果 spill**：沿用现有 `max_output_bytes`，对超限结果落地
`blob://<sha256>` + 生成预览 + 定位器（preview + `省略 N bytes` + `Read
artifact://`），前端懒加载 blob。**来源**：dsh `spill-policy` + Oh-my-pi
`OutputSink`。

### C3 — 多通道去重合并

- `messageStore` 以 `(threadId, messageId)` 为 key 去重；WS `text_delta`、REST
  历史、SSE 事件统一归并，避免切换线程/重连时重复恢复或覆盖。
- 用 `seq` 单调号做乱序收敛（与后端 `SessionEventLog` 对齐）。

### C4 — 派生数组缓存 + 懒加载

- `displayMessages` 改为 `useMemo` 按 `(visibleRange, phase)` 缓存；tool_group/
  回合摘要折叠用块级标记避免重建。
- 超长 `tool_result` 懒加载：`blob://` 引用按需 fetch。

### C5 — 渲染冷热分离

- 用 Oh-my-pi `NativeScrollbackLiveRegion` 思路：仅视口附近（live-region）做
  细粒度 diff，历史区冻结块只做一次渲染；`requestAnimationFrame` 批处理已有，
  补"块级 dirty 标记"。

### C6 — 事务化输入状态机（复用 dsh 模板）

- 移植 dsh `ui-conversation` `InputMachine`（draft/claim/paste/undo/redo、事务
  日志 + redo 栈）到 `useChatAndConnection` 乐观更新。
- 以 `seq` 对齐后端 `SessionEventLog` 做一致提交（替代现有 5s undo 窗口）。

**来源**：dsh `packages/client/ui-conversation/src/client/input/machine.ts`、
`ui-input-trigger`、`schema-form`。

---

## 4. 缺口对照（落地后关闭项）

| 子系统 | 现状缺口 | 本设计关闭项 |
|---|---|---|
| 事件溯源 | 无事件溯源内核 / 无投影 / 无事件级分支 / 内存 history 有界 | A1 A2 A3 |
| 事件溯源 | 压缩丢旧消息 | A4 |
| 沙箱 | 无强制进程级隔离 / require_digest 默认关 / in-process 逃逸 / 工具权限与沙箱脱钩 | B1 B2 B3 B4 |
| 沙箱 | 审批平铺无优先级 | B1 增强 |
| 前端 | 无冻结块 / 无增量 diff / 三通道重复 / displayMessages 线性派生 / 无懒加载 | C1 C2 C3 C4 C5 |
| 前端 | 乐观更新仅 5s undo 窗口 | C6 |

---

## 5. 落地优先级与实现任务拆分

> 命名：`BCSE`（Backend-Core Session/Sandbox/UI-Engine）。任务按模块拆分，
> 每个任务含验收标准（Acceptance）。

### P0 — 事件溯源地基（后端前提）

- **T-BCSE-01** 新建 `huginn/events/session_log.py`：`SessionEventLog`（追加式
  JSONL，含 `seq/id/parent_id/ts/payload`）。
  - ACC：可 append / read_after(seq) / 按 leaf 读路径；单测覆盖分支与乱序。
- **T-BCSE-02** 新建 `huginn/events/projection.py`：`ProjectionDefinition` +
  `ProjectionEngine`（弱引用 cell 缓存 + `stateVersion`）。
  - ACC：`init/apply/view` 纯函数；`apply` 幂等；`stateVersion` 变更触发重建。
- **T-BCSE-03** 事件类型扩展：在 `event_types.py` 补 `compaction`/`branch_summary`/
  `reset_boundary` 结构化载荷。
  - ACC：新建事件可被 A1/A2 消费，不破坏现有审计写入。

### P0 — 前端增量块级渲染

- **T-BCSE-04** 后端 `UiProjection` 输出 `blocks` 结构（`{kind,text,frozen,rev}`）。
  - ACC：`text_delta` 只更新最后一块；frozen 块不可变。
- **T-BCSE-05** 前端 `useIncrementalMessages` hook + 块级渲染改造
  `MessageContent.tsx`。
  - ACC：长会话下尾部块增量更新，frozen 块不重渲染（`memo` 到块）。
- **T-BCSE-06** 增量 diff 同步：`GET /threads/{id}/events?after=<seq>` + 前端
  `apply(event)`；compaction 事件插入 `── compacted ──` 分隔块。
  - ACC：切换/重连走增量而非全量重建；压缩不丢旧块。

### P1 — 沙箱强制隔离

- **T-BCSE-07** 新增 `huginn/security/landlock.py`（ABI 协商 + restrict_self +
  seccomp），集成进 `SandboxExecutor`（探测降级）。
  - ACC：不支持时降级回白名单；支持时 `allowed_work_dirs` 外读写被拒。
- **T-BCSE-08** 审批优先级改造 `_validate_command()`（deny > ask > allow）。
  - ACC：deny 压过 allow；默认 fail-closed。
- **T-BCSE-09** `ToolMetadata.sandbox_hint` + `build_executor(tool)` 工具级绑定。
  - ACC：`paranoid` 缺容器时拒绝而非静默回退。

### P1 — compaction / 分支语义

- **T-BCSE-10** `CompactionEntry` + `build_state()` 边界语义（A4）。
  - ACC：LLM 只见 boundary 后；全历史可导出。
- **T-BCSE-11** `SessionLog.branch/branch_with_summary/rollback_to/snapshot`（A3）
  + snapshot.take/revert 执行器。
  - ACC：分支历史不丢；rollback 从快照+重放重建。

### P2 — 加固与收尾

- **T-BCSE-12** `restricted_python.py` 逃逸加固 + `RLIMIT_AS`（B3）。
- **T-BCSE-13** `require_digest` 强制 + HPC 提权 pattern 拦截（B4）。
- **T-BCSE-14** 前端 C3 多通道去重 / C4 派生缓存+blob 懒加载 / C5 冷热分离 /
  C6 输入状态机。
- **T-BCSE-15** 跨模块集成测试：事件日志 → 投影 → WS/delta → 前端块渲染端到端。

---

## 6. 关键文件映射

- 新：`huginn/events/session_log.py`、`huginn/events/projection.py`
- 新：`huginn/security/landlock.py`
- 改：`huginn/events/event_types.py`、`huginn/security/sandbox.py`、
  `huginn/security/restricted_python.py`、`huginn/security/container_executor.py`
- 改：`huginn/tools/{base,defaults,registry}.py`、`huginn/execution/remote_executor.py`
- 改：`huginn/routes/{threads,ws}.py`、`huginn/routes/ws_helpers.py`
- 改：`desktop/src/hooks/useChatAndConnection.ts`、
  `desktop/src/components/MessageContent.tsx`、
  `desktop/src/components/panels/ChatPanel.tsx`