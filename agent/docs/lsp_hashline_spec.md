# P0 补强 Spec — LSP 符号级编辑工具 + Hashline 锚定编辑

> 目标：对标 oh-my-pi 的两项"IDE 级"能力，补齐 Huginn 代码 agent 最弱的一环。
> 1. **LSP 符号级编辑**：编辑工具从"纯文本替换"升级为"符号感知"（重命名/引用/诊断/代码动作），避免盲改。
> 2. **Hashline 锚定编辑**：编辑前校验内容 hash，防止并发 agent / 外部修改互相覆盖（直接呼应本次"代码提交失败，未保存将丢失"的教训）。
>
> 本 spec 只做设计，不做实现。改动量、接入点、安全边界、测试、失败模式全部基于现有源码调研（见"现状基线与接入点"）。

## 一、现状基线（源码调研事实）

### 1.1 编辑工具链
- [file_edit_tool.py](file:///workspace/agent/huginn/tools/file_edit_tool.py) — `FileEditTool`，单文件字符串替换。已有 `_make_diff`（unified diff）、`_content_hash`（sha256[:16]）、`_semantic_diff`（结构/参数文件语义 diff）。
- [multi_edit_tool.py](file:///workspace/agent/huginn/tools/multi_edit_tool.py) — `MultiEditTool`，跨文件原子替换，复用 `_content_hash`/`_make_diff`。**已返回 `snapshot_hash`（编辑前 hash）**，但**只作为审计字段，不参与写入前校验**。
- 均继承 `HuginnTool`，走 `PermissionChecker`（ASK/DENY），`_resolve_path` 强制 work_dir 边界。

### 1.2 工具注册机制
- [tools/__init__.py](file:///workspace/agent/huginn/tools/__init__.py#L28-L81) — `_CORE_MODULES` 列表（`(module, class)` 元组），`register_core_tools()` 启动时同步注册。文件编辑工具都在 core 里。
- [tools/registry.py](file:///workspace/agent/huginn/tools/registry.py) — `ToolRegistry.register/get/get_all_schemas`，`register_tool` 装饰器。
- 新工具接入 = 新建模块 + 在 `_CORE_MODULES` 加一行（或 optional 列表，若依赖重）。

### 1.3 会话上下文
- [core_types.py](file:///workspace/agent/huginn/core_types.py#L162-L177) — `ToolContext` 携带 `session_id / workspace / config / audit_logger`。**Hashline 锚定需要 `session_id` 做并发判据，已具备。**

### 1.4 事件溯源（可复用）
- [session_log.py](file:///workspace/agent/huginn/events/session_log.py) — `SessionEventLog`，`EVENT_*` 常量 + `append()`。已有 `EVENT_TOOL_CALL/EVENT_TOOL_RESULT` 等。文件编辑事件可走 `EVENT_CUSTOM` 或新增 `EVENT_FILE_EDIT`（见 §4.3）。

## 二、设计目标与原则

1. **不破坏现有文本编辑**：LSP 是"增强"而非"替代"。`file_edit_tool`/`multi_edit_tool` 保持可用，LSP 作为独立工具 + 编辑路径的符号校验层。
2. **优雅降级**：无 LSP 服务器 / 无对应语言时，LSP 工具返回明确错误，编辑工具照常工作（同 Landlock 的降级哲学）。
3. **hash 锚定默认开启但可关**：并发防护是安全默认，`dry_run`/`preview` 不受影响。
4. **符号级 = 多个工具**：一个"LSP 工具"承载多种操作（rename/refs/hover/diagnostics/codeaction），按 action 分发，避免工具爆炸。
5. **复用现有权限/审计/事件基建**：所有新工具走 `PermissionChecker`，写操作进 `audit_logger`，事件进 `SessionEventLog`。

## 三、P0-1：LSP 符号级编辑工具（`lsp_tool.py`）

### 3.1 依赖
- **首选 `pygls`**（任选）：Python 官方 LSP 实现，`pygls.lsp.client` 能连任意 stdio langserver。
- **降级 `jedi`**：纯 Python 静态分析，无需外部进程，覆盖 Python 的 rename/refs/hover/signature。无网络、无外部依赖，沙箱内最稳。
- 语言服务器发现：`PATH` 中找 `pyright-langserver`/`pyright`/`typescript-language-server`/`clangd` 等；找不到则回退 `jedi`（限 Python）或报"无 LSP 支持"。

> 工程取舍：起步**只做 Python（jedi）+ 可选外部 langserver**。材料科学 agent 的代码工具链以 Python 为主，jedi 覆盖大多数场景且零外部依赖，符合 ponytail 原则。多语言（TS/clangd）留作升级路径。

### 3.2 工具接口

```python
class LspToolInput(BaseModel):
    action: Literal[
        "rename",          # 符号重命名（符号级，含所有引用）
        "references",      # 查找引用
        "hover",           # 悬停文档/签名
        "diagnostics",     # 文件诊断（静态检查）
        "code_action",     # 可用的代码动作（格式化/quickfix）
        "definition",      # 跳转定义
    ] = Field(..., description="LSP 操作类型")
    file_path: str = Field(..., description="目标文件")
    line: int | None = Field(default=None, description="行号（0-based）")
    character: int | None = Field(default=None, description="列号（0-based）")
    name: str | None = Field(default=None, description="rename action 的新名称")
    working_dir: str | None = Field(default=None)
    provider: Literal["auto", "jedi", "external"] = Field(
        default="auto", description="auto=优先外部 langserver，无则回退 jedi"
    )
```

- `name = "lsp_tool"`，`category = "core"`，`destructive = True`（仅 rename 会改文件）。
- `is_read_only()`：除 `rename` 外均为只读。

### 3.3 实现要点

**`LspClient`（外部 langserver 封装，懒启动）**
- `connect(workspace)` → 启动 `pyright-langserver --stdio` 等，发 `initialize`。
- 请求：`textDocument/rename`、`references`、`hover`、`diagnostics`、`codeAction`、`definition`。
- 进程级缓存：同一 workspace 复用连接，避免每次建进程；超时/崩溃自动降级 jedi。
- **安全**：只允许 `ro` 读取 actions；`rename` 的 `textDocument/rename` 返回 edits 后，落盘前过 `_resolve_path` 边界检查 + 权限检查。

**`JediProvider`（纯 Python 降级）**
- 用 `jedi.Script(code, path)` 做 `rename`（`jedi.refactoring.rename`）、`infer`（定义）、`references`、`get_signatures`、`hover`。
- 无外部进程，天然贴合沙箱（§5 提到 Landlock 只放行 work dirs，jedi 读源码即可）。

**`is_read_only` / 权限**
- `rename` 走 `PermissionChecker.check("lsp_tool", is_destructive=True, args=...)`，ASK 给 diff 预览。
- 只读 actions 直接返回结果。

**诊断集成（可选增强）**
- `diagnostics` 结果可投递到 `code_act_selfcheck` / `lint_hook`，让代码循环在调用结果里看到编译错误。→ 作为 v2 增强，不进 P0 主路径（避免范围膨胀）。

### 3.4 注册
- 新建 `huginn/tools/lsp_tool.py`，[tools/__init__.py](file:///workspace/agent/huginn/tools/__init__.py#L28-L81) `_CORE_MODULES` 加一行 `("huginn.tools.lsp_tool", "LspTool")`。
- jedi 为轻依赖，走 core；外部 langserver 链接懒启动，不影响启动速度。

### 3.5 改动量
- 新文件 `huginn/tools/lsp_tool.py` ~250-320 行（`LspTool` + `LspClient` + `JediProvider` + actions）。
- [tools/__init__.py](file:///workspace/agent/huginn/tools/__init__.py) +1 行。
- 测试 `tests/test_lsp_tool.py` ~120 行（jedi 路径可测，外部 langserver mock）。

## 四、P0-2：Hashline 锚定编辑

### 4.1 问题定义
当前 `file_edit_tool`/`multi_edit_tool` 的 `snapshot_hash` **只记录不校验**。若文件在 agent 写出后被外部进程 / 另一 agent / 手动修改，下一次编辑会基于过期内容做文本替换，`old_string` 可能仍命中但覆盖别人的改动 → **未保存将丢失**。

### 4.2 接口修改（兼容）

给 `FileEditToolInput` 和 `MultiEditToolInput` 各加两个可选字段：

```python
expected_hash: str | None = Field(
    default=None,
    description="期望的当前文件内容 hash（sha256[:16]，来自上次 read/edit 返回的 snapshot_hash）。"
    "若提供且与磁盘实际不符，编辑被拒绝，防止覆盖并发修改。",
)
hash_policy: Literal["strict", "warn", "off"] = Field(
    default="strict",
    description="strict=hash 不匹配则拒绝；warn=警告但继续；off=不校验（默认行为）。",
)
```

- **默认 `strict`**（安全优先），但 `expected_hash=None` 时跳过校验（首次编辑 / 调用方未追踪 hash），行为与现状一致 → **向后兼容**。

### 4.3 校验逻辑（写入前，唯一插入点）

在 `file_edit_tool.call` 和 `multi_edit_tool.call` 的**写入阶段前**（`file_edit_tool` 在 `_content_hash(content)` 已算出的位置；`multi_edit_tool` 在 `validated` 构建完成后）：

```python
if input_data.expected_hash and input_data.hash_policy != "off":
    current_hash = _content_hash(content)  # 磁盘当前内容 hash
    if current_hash != input_data.expected_hash:
        msg = (
            f"File changed since snapshot (expected {input_data.expected_hash}, "
            f"got {current_hash}). Refusing to overwrite concurrent modification."
        )
        if input_data.hash_policy == "strict":
            return ToolResult(success=False, error=msg, data={"current_hash": current_hash})
        # warn: 记审计 + 继续
        logger.warning(msg)
```

- 对 `multi_edit_tool`：**每个文件单独校验**（每个 `SingleEdit` 可带 `expected_hash`），任一 strict 不匹配则整批拒绝（保持 atomic 语义）。
- 返回给调用方的 `snapshot_hash` 保持现状，供下一轮编辑作为 `expected_hash` 回传（形成闭环）。

### 4.4 事件溯源（可选，建议）
- 新增 `EVENT_FILE_HASH_MISMATCH = "file_hash_mismatch"` 到 [session_log.py](file:///workspace/agent/huginn/events/session_log.py#L37-L47) 的 `SESSION_EVENT_KINDS`。
- 在 mismatch 触发时 `append(EVENT_FILE_HASH_MISMATCH, {"file": path, "expected": ..., "got": ...})`，让前端增量引擎能感知"编辑被并发拒绝"并提示用户。
- **不进 P0 阻塞**：事件写入是增强，编辑工具核心逻辑不依赖它。

### 4.5 改动量
- [file_edit_tool.py](file:///workspace/agent/huginn/tools/file_edit_tool.py)：输入 schema +2 字段，写入前 +10 行校验逻辑。
- [multi_edit_tool.py](file:///workspace/agent/huginn/tools/multi_edit_tool.py)：`SingleEdit` +2 字段，写入前 +15 行校验。
- [session_log.py](file:///workspace/agent/huginn/events/session_log.py)：+2 行（常量 + kind 集合）。
- 测试 `tests/test_hashline.py` ~80 行（strict 拒绝 / warn 继续 / off 放行 / multi_atomic）。

## 五、安全边界

1. **LSP 只读 actions 不改文件**；`rename` 落盘前过 `_resolve_path` 边界 + 权限检查，与现有编辑工具一致。
2. **外部 langserver 是子进程**：建议复用现有 sandbox 的 `preexec_fn`（Landlock，§[sandbox.py](file:///workspace/agent/huginn/security/sandbox.py#L322-L339)）限制其只读 work dirs；jedi 无外部进程，天然安全。
3. **hash 校验只读不盲写**：`strict` 拒绝时零副作用，不改文件、不部分写入。
4. **向后兼容**：`expected_hash=None` 或 `hash_policy="off"` 时行为与现状完全一致。
5. **LSP 工具不可被 agent 用于任意命令执行**：code_action 只返回可用动作，不自动执行；外部 langserver 白名单（仅已知二进制）。

## 六、测试计划

- `test_lsp_tool.py`：
  - jedi 路径：rename 改名后引用同步更新；references 返回位置；hover 返回签名；无语言时返回明确错误。
  - 外部 langserver：mock 一个假 stdio server，验证 connect/initialize/请求协议正确。
  - 权限：rename 走 ASK 给预览，DENY 拒绝。
- `test_hashline.py`：
  - strict：`expected_hash` 与磁盘不符 → 拒绝，文件未变。
  - warn：不匹配 → 警告但继续写入。
  - off / hash=None：行为与旧版一致。
  - multi_atomic：同批中一个文件 hash 不匹配 → 整批拒绝（原子性）。
  - 闭环：先 read 拿 hash → edit（含 hash）→ 再 edit 用上次返回的 hash，第二次成功。

## 七、失败模式（诚实声明）

1. **LSP 服务器缺失/崩溃**：降级 jedi；jedi 也不支持该语言 → 返回"无 LSP 支持"，编辑工具不受影响。
2. **jedi 对大型文件慢**：限制文件大小（复用 `_MAX_FILE_BYTES`），超限跳过 LSP 增强。
3. **hash 误报**（文件未变但 hash 不同，如换行/编码差异）：`_content_hash` 用 utf-8 稳定编码，与 `file_edit_tool` 现有实现一致；`strict` 误报时调用方可改 `warn` 或重读最新 hash。
4. **rename 引用越界**（LSP 返回 work_dir 外的 edit）：`rename` 落盘前过滤掉目标不在 work_dir 的 edits。
5. **跨语言不可用**：初期仅 Python（jedi），TS/clangd 留作升级路径，不因缺失阻塞 agent。

## 八、执行顺序

1. **P0-2 Hashline**（独立，改动小，立即解决"未保存即丢"）：改两个编辑工具 + session_log 常量 → 测试 → commit。
2. **P0-1 LSP**（依赖关系独立，可并行）：新建 `lsp_tool.py`（先 jedi）+ 注册 + 测试 → commit。

两者互不阻塞，可并行。每项完成后跑 `pytest tests/test_hashline.py` / `tests/test_lsp_tool.py` + 相关回归（`test_file_read_tool_ext.py`、`test_multi_edit_tool` 若存在）。

## 九、升级路径（v2，不在本 spec 范围）

- LSP 多语言（TS/clangd/Rust-analyzer）+ `diagnostics` 接入 `code_act_selfcheck`。
- 编辑操作统一走 `dispatch_tool` 入口（H5-b，见 [harness_evolution_spec.md](file:///workspace/agent/docs/harness_evolution_spec.md) P9），让 hash 校验与 whitelist 集中。
- hash 校验失败时前端提示"文件已被外部修改"，引导用户二选一（重读 / 强制覆盖）。