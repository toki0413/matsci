# SPEC: 多协议适配层 (缺口① 落地设计)

**参考实现**: deepseek-recipe (https://github.com/deepseek-ai/deepseek-recipe),
MIT License, Copyright (c) 2026 DeepSeek. 本地源码副本: `/tmp/ext/deepseek-recipe/`.
已逐文件读完 crate: `protocol/openai/{chat_completion,responses}`、
`protocol/anthropic/messages`、`stream/{mod,output_parser,processor,decoder}.rs`、
`request/{mod,options}.rs`、`validation.rs`、`tool_call_id.rs`、`json_schema.rs`、
`image.rs`、`schema.rs`、`server-rs` / `server-py` 服务端样板 (约 22340 行核心代码).

**缺口**: huginn 目前**没有**任何 OpenAI / Anthropic 兼容端点 ——
[routes/compat.py](file:///workspace/agent/huginn/routes/compat.py) 是 math-anything
分析桩 (`/firewall/status`、`/sandbox/execute`), 与协议适配层语义无关;
[orchestration/protocol.py](file:///workspace/agent/huginn/orchestration/protocol.py) 等
`*protocol*.py` 是编排/实验协议, 不是对外协议适配. 因此一旦需要以
`/v1/chat/completions` / `/v1/responses` / `/v1/messages` 暴露 agent 能力, 每个协议
都要从头各写一遍请求转换与流式编码 —— 这正是 deepseek-recipe 用「共享 IR + 每协议一对
适配器」消除的重复。

---

## 一、从源码读出的架构判定 (可复用形状)

deepseek-recipe 的核心不是"三套独立代码", 而是四条互相咬合的契约:

1. **一份扁平共享 IR**: 三个协议先各自 `convert()` 成同一种对话中间表示 (messages +
   tools + 生成参数), 后端只认这一种。协议差异 (字段名、必填性、内容块形状) 在转换
   边界被吸收, 不进后端。
2. **每协议一对 `(convert, chunk_generator)` 收口**: `ProtocolRequest` trait 要求
   同一请求类型**同时**实现请求方向 `convert()` 与响应方向 `chunk_generator()`。请求
   与响应绑定在同一个类型上 —— 加一个协议只加一个类型 + 一对方法, 不可能只加半个
   (只转请求不转响应), 这是"新增协议必然成对"的编译期保证。
3. **统一 `OutputChunk` 顺序契约**: `stream/mod.rs` 把所有后端差异 (token 流 vs
   logprob、是否带 reasoning) 归一到一条有序 `OutputChunk` 流; 每协议的
   `ChunkGenerator` 只做 **chunk → 协议事件** 的映射, 不做解析。
4. **手写 KMP 字节级输出解析状态机**: `output_parser.rs` 才是把裸 token 流切成
   `Content / Json / ToolCalls / Reasoning / Finished` 的唯一处。KMP 失败表同时携带
   字节与"是否特殊 token"标志 —— **普通 token 拼出的 `</think>` 或 DSML 标签不触发
   标记**, 只有特殊 token 的拼写才触发。`decoder.rs` 对多 token UTF-8 字符做增量
   缓冲 (用 `\uFFFD` 作不完整标记), `processor.rs` 用 `PendingOutput` 合并相邻同动作
   段并跨 chunk 匹配停止序列。

一句话: **协议适配层 = 共享 IR × 每协议 (convert, chunk_generator) 对 × 单一解析
状态机 × 有序 chunk 契约**。huginn 直接照抄这个形状, 但**只抄形状, 不抄 DeepSeek 专属
实现**。

---

## 二、可复用 vs 必须排除

| 类别 | 处理 | 说明 |
|---|---|---|
| 扁平 IR + `(convert, chunk_generator)` 成对 trait | **照抄形状** | 语言无关的结构性设计, 与 DeepSeek 无关 |
| `OutputChunk` 顺序契约 | **照抄形状** | 后端/handler 解耦的关键接缝 |
| KMP 状态机 (Content/Json/ToolCalls/Reasoning/Finished) | **照抄形状**, 重写 | 逻辑可移植, 但 token 字典不同 |
| `responses` 请求转换: 按 `call_id` 分组工具调用/输出, 待齐后批量追加 | **照抄** | 与模型无关的协议语义 |
| 命名空间工具名扁平化 `{namespace}::{name}` 及反向映射 | **照抄** | 协议层语义 |
| 自定义工具参数内置 JSON `{"input": "..."}` 前缀解析 | **照抄** | 协议层语义 |
| 停止序列跨 chunk KMP 回溯 / 多 token UTF-8 缓冲 | **照抄** | 通用流式正确性 |
| prompt 渲染 / chat template | **排除** | DeepSeek 专属; huginn 走自己的 prompt 体系 |
| tokenizer 绑定 / token id 字典 | **排除** | DeepSeek 专属词表 |
| DSML 标签 (`<｜DSML｜tool_calls>`) 识别 | **排除** | DeepSeek 专属工具调用标记, 换成协议原生 tool_calls |
| 内联 `</think>` reasoning 标记 | **排除** (可配置) | DeepSeek 专属; OpenAI 用 `reasoning` content part |
| KV cache / PD 分离 / DP 副本均衡 | **排除** | 不在 deepseek-recipe 发布面, 也不在 huginn 目标内 |
| XGrammar / 结构化输出约束 | **排除** (本缺口不涉及) | DeepInfer 侧能力, 见附录 |

**版权合规**: 照抄的形状是 MIT/Apache 双许可下的结构性设计; 落地文件若含逐字移植的
算法 (KMP 状态机), 在文件头标注:
```
# Portions derived from deepseek-recipe (https://github.com/deepseek-ai/deepseek-recipe)
# MIT License, Copyright (c) 2026 DeepSeek
```

---

## 三、目标文件布局 (Phase 1 骨架)

```
agent/huginn/protocols/
  __init__.py          # 导出 IR / stream 契约 / 各协议适配器
  ir.py                # 扁平共享 IR (唯一对话中间表示)
  primitives.py        # 跨协议共享校验原语
  stream.py            # OutputChunk 顺序契约 + ChunkGenerator ABC + OutputParser 状态机
  openai_responses.py  # Responses 适配器 (convert + chunk_generator 成对)
agent/huginn/routes/
  protocols_api.py     # POST /v1/responses handler (后续 chat/completions, messages)
agent/tests/
  test_protocol_contract.py  # 契约门: 成对性 / IR 往返 / chunk 顺序
```

### 3.1 `ir.py` — 扁平共享 IR

```python
@dataclass(frozen=True)
class ContentPart:            # text | image | refusal
    kind: str
    text: str | None = None
    image: "ImageRef | None" = None

@dataclass(frozen=True)
class ToolDef:                # 三协议工具面归一
    name: str
    description: str
    parameters: dict          # JSON Schema
    kind: str = "function"    # function | custom | web_search

@dataclass(frozen=True)
class ToolCall:
    call_id: str
    name: str
    arguments: dict | str      # function→dict; custom→str; 解析在 primitives 收口
    namespace: str | None = None

@dataclass(frozen=True)
class ToolResult:
    call_id: str
    output: str | list[ContentPart]

@dataclass(frozen=True)
class Message:
    role: str                  # system | user | assistant | tool
    parts: tuple[ContentPart, ...]

@dataclass(frozen=True)
class ChatRequest:            # 后端唯一认得的形状
    messages: tuple[Message, ...]
    tools: tuple[ToolDef, ...] = ()
    tool_choice: str | dict | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    stop: tuple[str, ...] = ()
    stream: bool = False
    response_format: dict | None = None
    metadata: dict = field(default_factory=dict)
```

不变量: IR 里**不出现**任何协议名、任何 DeepSeek 字段、任何 tokenizer 概念。

### 3.2 `primitives.py` — 跨协议共享校验原语

收口三协议里重复出现的同语义逻辑 (deepseek-recipe 侧对应 `validation.rs` /
`tool_call_id.rs` / `json_schema.rs` / `image.rs`):

- `resolve_tool_call_id(raw, *, fallback_seed) -> str` — 缺失/重复 call_id 的确定化生成
  (responses 允许省略 id, chat 要求必有; 归一后后端只见唯一形式);
- `validate_json_schema(schema) -> schema` — 工具参数 schema 的浅校验 + 规范化;
- `normalize_image(part) -> ImageRef` — URL vs base64 归一 (排除 deepseek 的
  "图像预处理" 细节, 只保留形状);
- `flatten_namespace(ns, name) -> str` / `unflatten_namespace(flat) -> tuple` —
  `{namespace}::{name}` 双向映射 (responses 命名空间工具)。

### 3.3 `stream.py` — chunk 顺序契约 + 成对收口 + 解析状态机

```python
class ChunkKind(StrEnum):
    TEXT = "text"; REASONING = "reasoning"
    TOOL_CALL_DELTA = "tool_call_delta"; TOOL_CALL_DONE = "tool_call_done"
    FINISH = "finish"

@dataclass(frozen=True)
class OutputChunk:
    kind: ChunkKind
    seq: int                   # 全局单调递增, 顺序契约的承载
    text: str = ""
    tool_call: ToolCall | None = None
    finish_reason: str | None = None

class ChunkGenerator(ABC):
    """每协议唯一实现点: 只做 OutputChunk → 协议事件 映射, 不做解析."""
    @abstractmethod
    def generate(self, chunks: Iterable[OutputChunk]) -> Iterator["ProtocolEvent"]: ...

class ProtocolRequest(ABC):
    """每协议唯一实现点: 请求转换 + 响应生成成对绑定在同一类型上."""
    @abstractmethod
    def convert(self) -> ChatRequest: ...
    @abstractmethod
    def chunk_generator(self) -> ChunkGenerator: ...

class OutputParser:
    """KMP 字节级状态机: 裸 token 流 → 有序 OutputChunk.
    状态: Content → Json → ToolCalls → Reasoning → Finished.
    失败表携带 (byte, is_special_token); 仅特殊 token 拼写触发 \</think\> 类标记。
    """
```

`OutputChunk.seq` 由 parser 全局分配, 顺序契约即"seq 严格递增 + FINISH 唯一且居尾",
契约门硬判 (见 §五)。

### 3.4 `openai_responses.py` — Responses 适配器 (首批)

请求方向 `ResponsesRequest.convert()` (对齐 deepseek-recipe
`protocol/openai/responses/request/convert.rs`):

- 输入项按类型分路: 消息 / 推理项 / 函数调用 / 函数调用输出 / 自定义工具调用 /
  工具搜索调用;
- 工具调用与其输出按 `call_id` 分组, **待全部输出到达后批量追加**成 messages
  (避免半截 tool 轮次进 IR);
- 命名空间工具按 `{namespace}::{name}` 扁平化 (经 `primitives.flatten_namespace`);
- `apply_patch` 型自定义工具**硬编码为内置函数** shape;
- `web_search` 按配置**丢弃或拒绝** (huginn 无内建 web_search 工具面, 默认拒绝并回
  明确错误);
- 生成参数 (max_output_tokens/temperature/stop 等) 归一进 `ChatRequest`。

响应方向 `ResponsesChunkGenerator(ChunkGenerator)` (对齐
`responses/response/chunk_generator.rs`):

- 输出项按类型分路 (输出文本 / 推理文本 / 函数调用 / 自定义工具调用 / 工具搜索调用);
- 事件序列号全局递增 (与 `OutputChunk.seq` 同源);
- 命名空间工具名**反向映射**回原 namespace 与本地名 (经 `primitives.unflatten_namespace`);
- 自定义工具调用内置 JSON `{"input": "..."}` 前缀解析器;
- 仅在 `OutputChunk(kind=FINISH)` 时发终态事件 (response.completed / failed)。

### 3.5 `routes/protocols_api.py` — 端点挂载

- 新增 `router = APIRouter(tags=["protocols"])`, 挂 `POST /responses`。
- 注册进 [routes/__init__.py](file:///workspace/agent/huginn/routes/__init__.py)
  的 `ALL_ROUTERS`, 由 `include_v1_routes` 自动落到 `/v1` 前缀 → 对外即
  `POST /v1/responses` (与既有 `/v1` 版本化 + root 兼容策略一致)。
- handler 三段式 (对齐 deepseek-recipe `server-py/main.py`):
  1. `prepare`: 解析 body → `ResponsesRequest.convert()` → `ChatRequest`;
  2. `run`: 调 huginn agent (非流式直接取终态; 流式从 agent 产出 token 流);
  3. `respond`: token 流喂 `OutputParser` → `OutputChunk` 流 →
     `ResponsesChunkGenerator.generate()` → SSE。
- **不新增**传输层 / 不引第三方 SDK; 复用 huginn 既有 SSE 与 auth 中间件。
- 后续协议 (`chat_completions.py` / `anthropic_messages.py`) 各加一个
  `ProtocolRequest` 子类 + 一个路由, 共享 IR / parser / chunk 契约不变。

---

## 四、治理门 (接入既有棘轮体系)

按 [governance_audit.py](file:///workspace/agent/huginn/cli/governance_audit.py) +
[test_arch_gate_registry.py](file:///workspace/agent/tests/test_arch_gate_registry.py)
的登记纪律, 新增模块必须**同步登记**, 否则孤儿门 / 文档漂移即红:

1. **契约门** `tests/test_protocol_contract.py`:
   - 在 `ci.yml` 加显式 pytest guard step (否则被判孤儿门), 或显式登记进
     `FULL_SUITE_ONLY` (可见决策); 推荐前者 (快速失败)。
2. **能力锚点** 在 `governance_audit.CAPABILITY_SPECS` 增一条 `always_on` 型:
   `承载模块 = protocols/__init__.py` + `不变量测试 = tests/test_protocol_contract.py`。
   `always_on` 型无开关可翻转, 只锚"模块存在 + 回归网存在" —— 正好匹配本缺口
   (协议面靠 IR 契约而非 feature flag 治理)。
3. 重生成 [governance-state.md](file:///workspace/agent/docs/governance-state.md):
   `python -m huginn.cli.governance_audit --out docs/governance-state.md`。
4. 棘轮方向核对: `capability_floor` / `guard_step_floor` / `arch_gate_floor` 均为
   **只增不减**下限 —— 本缺口是新增能力, 不会触碰任何"放宽"红线; 若走 `FULL_SUITE_ONLY`
   则需在冻结 `_EXPECTED_BUDGET_IDS` 与登记表同步 (属可见决策, 非静默)。

---

## 五、契约测试内容 (`test_protocol_contract.py`)

红蓝双轨 (沿用从 awesome-cyber-security-university 提炼的"成对"原则):

**蓝队 (断言控制生效)**
- `test_every_protocol_request_is_paired`: 遍历 `protocols` 下所有
  `ProtocolRequest` 子类, 断言**同时**有可调用的 `convert()` 与 `chunk_generator()`
  —— 防"只加半个协议"静默漏响应方向;
- `test_output_chunk_seq_is_monotonic`: 由 `OutputParser` 产出的 chunk 流 `seq`
  严格递增, 且 `FINISH` 唯一且居尾;
- `test_ir_free_of_protocol_tokens`: `ChatRequest` 序列化后不含任何协议名 / DeepSeek
  专属字段 (镜像 `SPEC` §3.1 的不变量);
- `test_namespace_roundtrip`: `flatten_namespace` / `unflatten_namespace` 往返一致。

**红队 (主动尝试绕过)**
- `test_generic_token_cannot_forge_marker`: 构造"普通 token 拼出 `</think>` 形状"
  的输入, 断言**不**触发标记 (复刻 deepseek-recipe 解析器的核心防伪);
- `test_half_tool_turn_is_rejected`: 构造只给工具调用不给输出的 responses 请求,
  断言 `convert()` **拒绝**而非产出半截 tool 轮次;
- `test_orphan_protocol_class_is_caught`: 合成树上放一个只有 `convert()` 的类,
  断言成对性门判红 (前向非空转, 防门自身失效)。

---

## 六、分期执行 + 行数估算

| # | 交付 | 内容 | 估算行数 | 优先级 |
|---|---|---|---|---|
| P1-1 | IR + primitives | `ir.py` + `primitives.py` (IR 形状 + 4 原语) | ~180 | P1 |
| P1-2 | stream 契约 | `stream.py` (OutputChunk + 两个 ABC + OutputParser KMP 状态机) | ~260 | P1 |
| P1-3 | Responses 适配器 | `openai_responses.py` (convert + chunk_generator 成对) | ~320 | P1 |
| P1-4 | 端点 + 治理 | `routes/protocols_api.py` + 注册 + 治理门 + 治理文档重生成 | ~80 | P1 |
| P1-5 | 契约门 | `test_protocol_contract.py` (蓝队 4 + 红队 3) | ~140 | P1 |
| **合计** | | | **~980** | |

**执行顺序**: P1-1 → P1-2 → P1-3 (三者需连跑, IR/契约是编译面) →
P1-4 + P1-5 (一个 PR, 与治理门同步落地, 避免中途孤儿门/文档漂移)。
每个 PR 带 §五 的 assert 型自检, 不引新测试框架。

---

## 七、非目标 (显式排除, 防范围蔓延)

- **不做** DeepSeek prompt 渲染 / chat template / tokenizer 绑定;
- **不做** DSML 标签与内联 `</think>` 识别 (换成协议原生 tool_calls / reasoning part);
- **不做** KV cache、PD 分离、DP 副本负载均衡 (deepseek-recipe 发布面之外);
- **不做** 推理后端 (huginn 已有自己的 LLM 层; 本层只做协议↔IR 适配);
- **不做** 第三方 SDK 传输 (复用既有 SSE + auth);
- **不重排** 现有 `*protocol*.py` (编排/实验协议), 仅新增 `protocols/` 命名空间。

---

## 八、附录: 缺口② (环境变量值域 / 跨变量约束) — 独立设计, 本 SPEC 不含

DeepInfer `deepinfer/environ.py` 用泛型非数据描述符 `EnvVar[T]`: 首次访问解析并缓存,
支持**值域约束** (choices / min-max) 与**跨变量派生** (如
`MAX_BACKTRACKING_TOKENS = max(_int(v), USE_NEXTN_DECODE + 1)`)。

huginn 现状 ([env_schema.py](file:///workspace/agent/huginn/env_schema.py) 的 `EnvSpec` +
[env_access.py](file:///workspace/agent/huginn/env_access.py)) 已很强 (单一权威源 / 类型化
读端 / 裸读棘轮), 但 `EnvSpec` **只有注册面不变量** (type/default/scope/status),
**没有**值域校验与跨变量派生。缺口② = 给 `EnvSpec` 增两个可选字段
(`choices` / `range` / `derived_from`) 并在 `env_access` 解析时校验, 使
"非法值静默回落"变成"显式报错或显式回落"。此事与协议层正交, 应单开 SPEC, 不并入本缺口。

---

## License 合规

deepseek-recipe MIT License, Copyright (c) 2026 DeepSeek.
落地文件若含逐字移植算法 (KMP 状态机等), 在文件头加:
```
# Portions derived from deepseek-recipe (https://github.com/deepseek-ai/deepseek-recipe)
# MIT License, Copyright (c) 2026 DeepSeek
```