# 分层 Memory 结构化 Spec — 长周期任务上下文维护

> 治 PMK 循环断点 + 长周期任务 (1000+ iter) context 维护.
> 数学动机: 4 层 memory 是信息论分层, 每层 entropy 有界.
> 跟极限模式 `HUGINN_EXTREME_DISPATCH` 共生 — 平常不开 sliding window summarize.

## 动机

异步委派 4 天花板已解, 但 140,450 步长程任务有新天花板: **LLM context window 装不下完整轨迹**.

硬塞会塌方 (证据: [engine.py:760-811](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/autoloop/engine.py#L760-L811) `_trim_to_budget` 12KB 硬上限, 3 pass 截断到 "前 2 行各 100 字符"). 真正的瓶颈不是 window 大小, 是 **PMK 循环没闭合** — 写入了 memory/KB/KG 但召回面窄, 无法替 context window 减负.

## 数学结构

### 4 层 memory 不是工程分层, 是信息论分层

每层 entropy 有界, 层间转换是有损压缩 (rate-distortion theory):

| 层 | 数学定义 | 访问复杂度 | 熵界 | 持久化 |
|---|---|---|---|---|
| **Working** (WM) | 当前 iter active set, M_t ⊂ messages | O(1) 进程内 | H(WM) ≤ log(V_w) | 不持久 |
| **Episodic** (EM) | 时序事件流 E = {(t, persona, r_phys, action)}_t | SQLite + FTS5 + embedding | H(EM) ≤ V_e · H_max, 靠 decay 限界 | SQLite + vector |
| **Semantic** (SM) | 蒸馏 fact/concept, KB chunks + KG entities | RAG top_k | H(SM) ≤ cleanup 阈值 | ChromaDB + JSON |
| **Procedural** (PM) | stable_principles, 规则 (X→Y) + confidence c ∈ [0,1] | 关键词召回 | H(PM) 由 c 衰减限界 | SQLite |

### 层间转换 = 有损压缩

- **WM → EM**: sliding window 边界触发 LLM summarize. 数学上 n-gram 滑窗 + LLM 抽象, rate-distortion R(D) ≤ log(V_w / D).
- **EM → SM**: clustering + abstraction. 已有 `distill_episodic_to_procedural` (`memory/manager.py:1148-1212`) 雏形, 升级为显式 EM→SM 转换.
- **SM → PM**: Bayesian confidence update. 写入 c=0.5, 复用成功 +ε, 失败 -ε. c < c_min 删除.

### Context window = query-dependent attention

不再 12KB 硬上限 + 3 pass 截断, 而是**每层有 token budget, 按 phase 决定拉多少**:

```
phase:        perceive  hypothesize  plan  execute  validate  learn
WM budget:    4K        3K           2K    4K       2K        1K
EM budget:    1K        2K           2K    1K       3K        2K
SM budget:    1K        2K           3K    1K       2K        2K
PM budget:    0K        1K           1K    1K       1K        1K
─────────────────────────────────────────────────────────────────
total:        6K        8K           8K    7K       8K        6K
```

query-dependent: `_build_*_prompt` 按 phase 从每层取对应 budget, 拼成 context.

## 数学映射 — 5 个 ceiling 对应 5 个数学结构

### C1. Context window 装不下完整轨迹 → WM sliding window + LLM summarize

**治**: 12KB 硬上限 + 3 pass 截断 → 信息密度塌方.

**数学**: n-gram sliding window 容量 V_w (8K tokens). 边界触发时, 把 window 内容喂 LLM 做 rate-distortion 最优压缩 (summarize), 输出固定 V_e_per_summary token 写入 EM.

**接入**: `SessionContext` 加 `token_budget` 字段 (替代 `max_messages`), `_compact_if_needed` 改调 LLM summarize 推到 EM, 而非简单"保留 system + recent".

### C2. trajectory_match 召回面窄 → hypothesize/plan 也接 PM 层召回

**治**: trajectory_match 只在 `_check_stuck` 调 (G2 极限模式才开), hypothesize/plan 不用历史轨迹.

**数学**: 当前 tool 序列 = path graph G_t (M2 已实现), 历史 trajectory = path graph 集合 {G_h}. VF2 子图同构 G_t ⊆ G_h. 召回 = argmax_h sim(G_t, G_h).

**接入**: `_build_hypothesis_prompt` / `_build_plan_prompt` 加 PM block, 调 `trajectory_match(current_actions, traj_history)` → 命中则注入 `next_step` 建议.

### C3. pattern 无反馈闭环 → Bayesian confidence ±ε

**治**: trajectory_pattern 写入 KB 后无 ±ε, 无法演化. 写了就死.

**数学**: pattern confidence c ∈ [0,1]. 写入 c_0 = 0.5. 复用成功 c ← (c·α + 1·β)/(α+β), 失败 c ← (c·α + 0·β)/(α+β). α=经验权重 (默认 5), β=新证据权重 (默认 1). c < c_min (0.2) 删除.

**接入**: `extract_and_store_pattern` 写入前查同 task_pattern 是否已存在, 存在则更新 confidence; `_build_kb_text` 命中 pattern 时按 confidence 排序; KB metadata 加 `confidence` 字段.

### C4. typed memory 默认 off → 默认 on + lazy migrate

**治**: `HUGINN_USE_MEMORY_TYPING=1` 才开, 旧行 NULL. `_pick_hypothesis_persona` fallback 走正则 grep memory tags, 精度差.

**数学**: typed memory 是 schema on write, 把 `(memory_type, run_id, persona_id, status)` 从 unstructured tags JSON 升级为 typed columns. 类型系统 T = {episode, fact, principle, persona_history, calculation, distilled}. 查询 O(|T|) 而非 O(|tags|).

**接入**: `LongTermMemory` 启动时默认开 typed memory (env 仍可关), 旧行 lazy migrate — recall 时遇 NULL 用 regex 从 tags 补.

### C5. knowledge → persona 断 → EM 层 persona_history 显式召回

**治**: KB/KG 注入 prompt 但不影响 persona 选择. `_pick_hypothesis_persona` 查 `memory.recall_typed(memory_type="persona_history", persona_id="reviewer")` 取 r_phys 平均, 但 typed memory 默认 off.

**数学**: persona 选择是 EM 层查询. 给当前 context c, persona p, 选择 p* = argmax_p E[r_phys | c, p, history]. history = recall_typed(persona_history). 期望由历史 mean 估.

**接入**: C4 修后, `_pick_hypothesis_persona` 不再 fallback grep, 直接走 typed memory. 加 KG 层: KG entity_type="persona_use" 记录 (persona, context_hash, r_phys), query 时按 context 相似度召回.

## 改动清单 (精确到文件 + 方法)

### WM 层 — sliding window

- `memory/session.py`:
  - `SessionContext` 加 `token_budget: int = 8192` 字段 (替代 max_messages=100)
  - `_compact_if_needed` 改: token 超预算时调 `summarize_window(messages, llm_chat_fn)` 推到 EM, 而非简单 pop
  - 新增 `summarize_window(messages, llm_chat_fn) -> str`: LLM summarize, 返回固定 V_e_per_summary token 文本

### EM 层 — FTS5 + embedding 双路

- `memory/longterm.py`:
  - `_ensure_fts5` 启动时已跑, 不动
  - 新增 `_ensure_embedding_index`: `embeddings` 表 (id, embedding BLOB), 用 sentence-transformers 后台 batch 计算
  - `store(category, content, ...)` 触发 embedding 计算 (异步, 不阻塞)
  - `recall_embedding(query, top_k)` 新方法, 跟 `recall_fts` RRF 混合
- `memory/manager.py`:
  - `recall_for_prompt(query, max_entries)` 改: 调 `longterm.recall_rrf(query, top_k)` (FTS5 + embedding RRF)
  - 新增 `recall_persona_history(persona_id, limit)`: 显式查 typed memory persona_history, 不再走 grep

### SM 层 — KB/KG 不动, 加 trajectory_match 接入

- `autoloop/engine.py`:
  - `_build_hypothesis_prompt` 加 PM block: 调 `trajectory_match(action_history, traj_history)` → 注入 `next_step` 建议
  - `_build_plan_prompt` 同上
  - `_build_kb_text(query)` 命中 trajectory_pattern 时按 confidence 排序 (C3 修后)

### PM 层 — Bayesian confidence 闭环

- `knowledge/trajectory_pattern.py`:
  - `extract_and_store_pattern` 写入前查 `kb.query(task_pattern=X)` 去重, 已存在则 `confidence += ε`
  - KB metadata 加 `confidence: float` 字段
  - 新增 `update_pattern_confidence(kb, pattern_id, success: bool)`: 复用后调, ±ε
- `autoloop/engine.py`:
  - `_learn` 加: 如果本轮匹配过 trajectory_match 且本轮成功 → `update_pattern_confidence(+ε)`, 失败 → `update_pattern_confidence(-ε)`

### typed memory 默认 on

- `memory/longterm.py`:
  - `_init_schema` 不看 env, 默认建 typed columns
  - `recall_typed(memory_type, ...)` 不再 env 判断, 直接查
  - 旧行 (memory_type IS NULL) lazy migrate: recall 时遇 NULL 用 regex 从 tags 补 memory_type
- `memory/manager.py`:
  - `remember_typed` 默认走, 不看 `HUGINN_USE_MEMORY_TYPING`

### persona 选择走 EM 显式召回

- `autoloop/engine.py`:
  - `_pick_hypothesis_persona` 改: 删 grep fallback, 直接 `memory.recall_persona_history(persona_id, limit=10)` 取 r_phys 平均
  - `_learn` 加: 写 KG entity_type="persona_use" (persona, context_hash, r_phys), 给 C5 KG 层用

### context window 分层 budget

- `autoloop/engine.py`:
  - `_PROMPT_BUDGET = 12000` 改为 `_PROMPT_BUDGET_BY_PHASE: dict[str, dict[str, int]]` (按上面的表)
  - `_trim_to_budget(blocks, phase)` 按 phase 取每层 budget, 不再 3 pass 全压
  - `_build_*_prompt` 都接 phase 参数

### 极限模式 gating

跟异步委派极限模式一致, 平常默认关闭 sliding window summarize + trajectory_match 召回:

- `HUGINN_EXTREME_DISPATCH=1` 时:
  - WM sliding window 用 rule-based summarize (平常简单 pop)
  - hypothesize/plan 调 trajectory_match (平常不调)
  - pattern confidence 闭环开 (平常只写不更新)
- 不 gate:
  - typed memory 默认 on (安全, 性能好)
  - persona 走 EM 显式召回 (替代 grep, 无副作用)
  - context 分层 budget (替代 12KB 硬上限, 无副作用)

### WM summarize 策略 (不依赖专用小模型)

专用小模型不可得, C1 WM sliding window 默认走 **rule-based summarize**, 复用 [_summarize_trajectory](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/knowledge/trajectory_pattern.py#L74-L101) 模式:

- 抽 tool_calls 名字序列 (跟 _summarize_trajectory 一致)
- 抽 assistant 文本首句 (per message, 截 100 字符)
- 抽 tool result 关键字段 (r_phys / energy / success, 截 200 字符)
- 拼成 ~500 token 文本写入 EM

策略梯度 (按成本从低到高, env `HUGINN_WM_SUMMARIZE` 切换):

| 策略 | env 值 | LLM 调用 | 输出质量 | 默认 |
|---|---|---|---|---|
| rule-based | `rule` (默认) | 0 | 中 (规则抽 key facts) | ✓ |
| ngram | `ngram` | 0 | 中 (高频短语 top-N) | |
| llm | `llm` | 1/边界 | 高 (deepseek-chat) | |
| hybrid | `hybrid` | 0.5/边界 | 中高 (rule + 偶尔 LLM) | |

ponytail: 默认 rule, 不增加 LLM 调用. 升级路径: env 切 llm/hybrid.

## 前端接入

极限模式 + 分层 memory 设置需在前端 UI 暴露, 用户可切换:

### Settings 页面新增 "极限模式" 区块

| 设置项 | env | 类型 | 默认 | 说明 |
|---|---|---|---|---|
| 极限模式总开关 | `HUGINN_EXTREME_DISPATCH` | toggle | off | 开启异步委派 DAG + cycle 检测 + trajectory 召回 + pattern confidence |
| WM summarize 策略 | `HUGINN_WM_SUMMARIZE` | select | `rule` | rule/ngram/llm/hybrid (极限模式才生效) |
| WM token 预算 | `HUGINN_WM_TOKEN_BUDGET` | number | 8192 | sliding window 容量 |
| EM 召回 top_k | `HUGINN_EM_RECALL_TOP_K` | number | 5 | FTS5+embedding RRF 召回数 |
| PM confidence 阈值 | `HUGINN_PM_C_MIN` | number | 0.2 | 低于此值删除 pattern |
| Summarize 触发周期 | `HUGINN_WM_SUMMARIZE_EVERY_N` | number | 5 | 每 N iter 触发一次 (极限模式) |

### Session/Trajectory 视图新增 "Memory 层级" 面板

显示当前 run 的 4 层 memory 状态:

- **WM**: 当前 token 占用 / budget, 最近一次 summarize 时间
- **EM**: 记录数, 最近 10 条 episode 摘要 (点击展开)
- **SM**: KB chunks 数, KG 节点数, 最近写入的 pattern
- **PM**: stable_principles 数, top-5 by confidence

### 极限模式开关的 UX

- 默认隐藏在 "Advanced Settings" 折叠区, 避免误触
- 开启时弹确认: "极限模式会增加 LLM 调用和计算开销, 确认开启?"
- 开启后状态栏显示 `EXTREME` 标记 (跟 subagent dispatch 标记一致)
- 关闭时已写入的 pattern/episode 保留, 不清除

## 执行顺序

按依赖关系 + 风险最小:

1. **C4 typed memory 默认 on + lazy migrate** — 基础设施, 其他改动依赖. ~80 行.
2. **C5 persona 走 EM 显式召回** — 依赖 C4, 独立小改. ~30 行.
3. **C3 PM Bayesian confidence 闭环** — 独立, 治 pattern 写了就死. ~60 行.
4. **C2 trajectory_match 接入 hypothesize/plan** — 依赖 C3 (排序按 confidence). ~50 行.
5. **C1 WM sliding window + LLM summarize** — 最大改, ~150 行. 最后做.
6. **context 分层 budget** — 收尾, ~80 行.

每个 C 完成后立即 selfcheck. C 之间独立, 不互相阻塞 (除 C5 依赖 C4, C2 排序依赖 C3).

## 不做的事 (YAGNI)

- ❌ 换 KG.json 全量 save 为 Graphiti 性质 DB — 是单点瓶颈但 1000 iter 才显著, 先看分层 memory 能不能延后崩坏点
- ❌ 跨 workspace KB 共享 — 不是长周期任务瓶颈
- ❌ KG 实体合并 (同义实体) — 独立问题, 不在循环闭合范围
- ❌ pattern 跨 task type 通用化 — PM 层只做 confidence, 不做泛化
- ❌ 4 层 memory 的可视化 dashboard — 工程层, 不在数学结构范围

## 升级路径 (不在本 spec 范围)

- WM sliding window 的 summarize LLM 换专用小模型 (deepseek-chat 默认)
- EM embedding 换 domain-specific (materials science fine-tuned)
- PM confidence 加 Bayesian prior (而非固定 0.5)
- KG 存储 SQLite 后端 (替代 JSON 全量)
- 加第 5 层 **meta-memory** (memory about memory, 跟 prospective memory 区分)

## 数学边界

- 4 层 entropy 限界证明: H(total) = H(WM) + H(EM) + H(SM) + H(PM) ≤ log(V_w) + V_e·H_max + cleanup + c_decay. 给定 V_w, V_e, cleanup, c_decay, 总熵有界.
- sliding window summarize 的 rate-distortion: R(D) ≤ log(V_w / D), D = distortion (信息损失). LLM summarize 是 R(D) 的近似.
- Bayesian confidence 收敛: c_n → true success rate (Bernoulli 估计), α+β 越大收敛越慢但越稳.
