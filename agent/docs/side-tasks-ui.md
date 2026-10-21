# 侧边任务前端 UI 文档

> 组件：`desktop/src/components/panels/SidePanel.tsx`
> 接线：`desktop/src/App.tsx`、`desktop/src/components/panels/ChatPanel.tsx`
> 测试：`desktop/src/components/panels/SidePanel.spec.tsx`

## 1. 组件结构

```
App
├─ header
│  └─ PanelRight 按钮              # 开合 dock 面板
├─ ChatPanel
│  └─ 消息操作栏
│     ├─ user 消息: GitFork        # fork 到侧边任务
│     └─ assistant 消息: GitFork
└─ SidePanel
   ├─ variant="dock"               # 与主对话同屏，右侧 380/420/480px
   └─ variant="page"               # Side Tasks 页签的全页双栏
      ├─ PanelHeader: New / Refresh / Close(dock)
      ├─ 任务列表
      └─ 任务详情
         ├─ 状态徽章 / Promote / Discard
         ├─ 继承上下文（折叠）
         ├─ 对话历史
         ├─ 产物列表
         └─ 输入框 + 发送
```

## 2. `SidePanel` Props

```ts
interface SidePanelProps {
  activeThread: string;                 // 当前主对话 thread id；新建侧边任务时作为父线程
  focusTaskId?: string | null;          // 外部指定要打开的任务（fork 后自动聚焦）
  variant?: 'page' | 'dock';            // 默认 page
  onOpenThread: (threadId: string) => void;  // 转正后切到正式任务 thread
  onThreadsChanged?: () => void;        // 转正后刷新线程列表
  onClose?: () => void;                 // dock 模式关闭按钮
}
```

说明：

- `focusTaskId` 变化时自动 `GET /side/tasks/{id}` 并打开详情；
- `page`：列表固定 288px + 详情自适应；
- `dock`：列表与详情互斥显示（窄栏），详情左上角出现返回按钮；
- `onClose` 只在 `dock` 渲染。

## 3. 内部状态

| 状态 | 作用 |
|---|---|
| `tasks` | `GET /side/tasks` 的列表结果（含已转正） |
| `selectedId` | 列表选中项 |
| `detail` | `GET /side/tasks/{id}` 的详情投影 |
| `input` | 侧边消息输入框 |
| `loading` | 列表加载中（Refresh 图标旋转） |
| `sending` | 消息发送中（发送按钮旋转） |
| `error` | 列表加载错误文案 |
| `showSeed` | 继承上下文折叠开关 |

## 4. 交互流程

### 4.1 从消息 fork

1. 消息操作栏点击 GitFork。
2. `ChatPanel` 在 `messages` 上按 `role in (user, assistant)` 过滤，取当前消息下标。
3. 调用 `onForkToSideTask(index)`。
4. `App.forkToSideTask()` 调 `POST /side/tasks`，成功后：
   - `setSideFocusTaskId(task.id)`；
   - `setSideDockOpen(true)`；
   - 保持 `activeTab="chat"`（主对话不消失）。
5. `SidePanel` 的 `focusTaskId` effect 打开详情。

> 下标口径必须与后端一致：只统计 user/assistant，tool/system 不参与。

### 4.2 新建（无指定消息）

面板顶部 **New side task**：
`POST /side/tasks {parent_thread_id: activeThread}`，继承父对话全部上下文。

### 4.3 继续对话

- 输入框 `Enter` 发送，`Shift+Enter` 换行；
- 调 `POST /side/tasks/{id}/messages`；
- 成功后用响应里的 `task` 覆盖详情，并刷新列表；
- 失败通过 toast 展示 `error`，详情与历史不变。

### 4.4 Promote

- 仅 `!detail.is_promoted` 时显示；
- 调 `POST /side/tasks/{id}/promote`；
- 成功后：toast → `onThreadsChanged()` → `onOpenThread(thread_id)`；
- `App` 关闭 dock、`switchThread(thread_id)`、刷新线程列表。

### 4.5 Discard

- 仅 `!detail.is_promoted` 时显示（正式任务不允许从这里删除）；
- 调 `DELETE /side/tasks/{id}`；
- 成功后清空详情与选中项，刷新列表。

## 5. 视觉与布局

| 区域 | 样式 |
|---|---|
| dock 宽度 | `w-[380px]` → `lg:w-[420px]` → `xl:w-[480px]`，`shrink-0`，左边框 |
| page 列表 | `w-72 shrink-0`，右边框 |
| 任务项 | 卡片式 `rounded-lg border p-3`，选中态 `border-accent bg-accent/10` |
| 徽章 | 10px 文本 + `border px-1.5 py-0.5 rounded`，三色：muted / accent / success |
| 用户消息 | 右偏 `ml-8`，accent 边框 |
| 助手消息 | 左偏 `mr-8`，普通边框 |
| 产物 | `FileText` + 路径 + 工具徽章 |
| 继承上下文 | 按钮控制的折叠块（Chevron 切换，默认收起） |

设计 token 复用项目现有 Tailwind 语义色：`text-text-primary/secondary/muted`、
`bg-bg-primary/secondary/tertiary`、`border-border`、`accent`、`success`、`error`。

## 6. 可访问性

- 开合按钮：`aria-label="Toggle side tasks panel"`、`aria-pressed`；
- 关闭按钮：`aria-label="Close side tasks panel"`；
- 返回按钮：`aria-label="Back to side task list"`；
- 输入框：`aria-label="Side task message"`；
- 刷新按钮：`aria-label="Refresh"`；
- 任务列表项是原生 `<button>`，键盘可聚焦；
- 动画使用 `motion-reduce:animate-none`，尊重减少动态偏好；
- 颜色不是唯一信息载体：状态同时有文字徽章（side task / formal task）。

## 7. 国际化

`desktop/src/i18n.ts` 中 `side.*` 键已更新为侧边任务语义：

| key | en | zh |
|---|---|---|
| `side.title` | Side Tasks | 侧边任务 |
| `side.placeholder` | Ask a side question, then discard or promote… | 提出侧边问题, 之后可丢弃或转正… |
| `side.empty` | No side tasks. Fork a message from the main chat. | 暂无侧边任务。从主对话的消息 fork 一条即可。 |
| `side.answerPlaceholder` | Type the side task message… | 输入侧边任务消息… |

当前 `SidePanel` 内部按钮/徽章文案仍以英文为主（与 `ThreadsPanel` 等面板一致）；
如需完整双语，可逐步把这些文案替换为 `t('side.*')` 键。

## 8. 接线示例

### 8.1 全页模式

```tsx
{activeTab === "side" && (
  <SidePanel
    activeThread={activeThread}
    focusTaskId={sideFocusTaskId}
    onOpenThread={openPromotedTask}
    onThreadsChanged={loadThreads}
  />
)}
```

### 8.2 Dock 模式（与主对话同屏）

```tsx
{sideDockOpen && (
  <aside className="flex w-[380px] shrink-0 flex-col border-l border-border bg-bg-primary lg:w-[420px] xl:w-[480px]">
    <SidePanel
      variant="dock"
      activeThread={activeThread}
      focusTaskId={sideFocusTaskId}
      onOpenThread={openPromotedTask}
      onThreadsChanged={loadThreads}
      onClose={() => setSideDockOpen(false)}
    />
  </aside>
)}
```

### 8.3 消息操作栏

```tsx
<button
  onClick={() => {
    const ordered = messages.filter(
      (m) => m.role === "user" || m.role === "assistant",
    );
    const forkIdx = ordered.indexOf(msg);
    if (forkIdx >= 0) onForkToSideTask?.(forkIdx);
  }}
  aria-label="Fork into side task"
  title="Fork from here into a side task"
>
  <GitFork size={13} aria-hidden="true" />
</button>
```

## 9. 测试

```bash
cd desktop
npx tsc --noEmit
npx vitest run src/components/panels/SidePanel.spec.tsx
npx vitest run            # 全量
```

现有 5 条组件测试：

1. 列表渲染（继承数 / 产物数）；
2. 详情打开（继承上下文 / 对话 / 产物）；
3. Promote 调用并 `onOpenThread(thread_id)`；
4. Discard 调用 `DELETE /side/tasks/{id}`；
5. dock 模式的返回与关闭按钮。

新增交互时优先补 mock API 的组件测试，不要在测试里发真实请求。

## 10. 已知边界

- 侧边消息是非流式调用；长回答会等 agent 完整跑完，不显示逐字流。
- 列表加载失败时只有 `error` 文案，没有自动重试；点 Refresh 重试。
- 大量任务时未做虚拟滚动（侧边任务通常生命周期短、数量小）。
- i18n 未覆盖全部按钮文案（见 §7）。
- 同一侧边任务不要并行发送多条消息（后端无并发合并）。
