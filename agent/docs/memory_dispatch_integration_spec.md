# Memory / Context / Dispatch 接通 Spec

> 治 "写了但没接通" 三块断层: typed memory 写入但主检索不读 / LLM decider 信息盲区 / context 双路径漂移.
> 跟 [layered_memory_spec.md](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/docs/layered_memory_spec.md) 不重叠 — layered_memory_spec 治 4 层物理分层 + WM sliding window + Bayesian confidence, 本 spec 治 "已有字段没进主路径".
> 跟 [harness_evolution_spec.md](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/docs/harness_evolution_spec.md) 不重叠 — harness_evolution 治 prompt/workflow/harness 可演化, 本 spec 治 memory/context/dispatch 可观测.

## 问题画像 (基于代码实际状态, 非 spec)

### 三块断层 + 根因

| 断层 | 表面症状 | 根因 | 影响 |
|---|---|---|---|
| **M: typed memory 主检索不读** | `recall_for_prompt` 只调 `recall()` → FTS5+category, 4 列结构化字段 (`memory_type/run_id/persona_id/status`) 写了不读 | [manager.py:172-214](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/memory/manager.py#L172-L214) 主检索入口绕过 `recall_typed` | EM/SM 分层在检索层空操作; H3 learn 写 typed memory 但下一轮 decider 看不到 |
| **D: LLM decider 信息盲区** | learn/report 是哑 action, 端到端测试 LLM 几乎不选 learn | [engine.py:2282-2308](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/autoloop/engine.py#L2282-L2308) decider prompt 只给 LLM 看 5 个字段, learn 不更新 cog | 调度失控: LLM 选 learn 没正反馈, 选 report 浪费一轮 |
| **C: context 双路径漂移** | metacog 5 个信号 (target_chain/step_eval/PMK/prospective/meta_agent) 算了不注入; engine 跟 ContextBuilder 各一套 KB/KG build 逻辑漂移 | [context_builder.py](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/context_builder.py) 5 个方法是死代码, engine 不复用 | autoloop 完全不读 meta_trace/evolution_rules; metacog 信号进不到 prompt |

### 断层依赖图

```
M (typed memory 不读)  ──┬──> D (decider 看不到 learn 产出)
                        │
                        └──> C (context 双路径不统一, memory 检索不统一)

D (decider 信息盲区)    ──┬──> learn 哑 action (不更新 cog)
                        └──> report 哑 action (execute_fn no-op)

C (context 双路径漂移)  ──> metacog 信号死代码 (5 个方法无人调)
```

**依赖关系**: M 是 D 和 C 的前置 — typed memory 不接通, learn 写了也白写, decider 看不到; context 双路径合并前要先把 memory 检索统一. **优先级 M > D > C**.

## 三块 Spec

### M. typed memory 接通主路径

**治**: 4 列结构化字段 (`memory_type/run_id/persona_id/status`) 写入但 `recall_for_prompt` 按 category 字符串过滤, 等于白写.

**现状 (基于代码实测)**:
- typed API 10 值 enum, 实际写入路径只 4 值 (`iteration_result`/`failed_direction`/`cross_domain_transfer` + `stable_principle` 走 JSONL)
- `recall_for_prompt` ([manager.py:172](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/memory/manager.py#L172)) 主检索入口只调 `recall()` → FTS5 + category, 不调 `recall_typed()`
- `_recall_typed` ([manager.py:1007](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/memory/manager.py#L1007)) 严格匹配返空后扫全表 NULL 行反推 (write-on-read), 性能差
- `persona_history` typed memory 已实现写入 ([engine.py:5770](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/autoloop/engine.py#L5770)), `_pick_hypothesis_persona` 已接 ([engine.py:3869](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/autoloop/engine.py#L3869))

**改动**:

#### M1. `recall_for_prompt` 加 typed 路径

[manager.py:172-214](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/memory/manager.py#L172-L214) 改:

```python
def recall_for_prompt(self, query: str, max_entries: int = 3, ...):
    # 原有 FTS5+category 路径保留 (向后兼容)
    results = self.recall(query, top_k=max_entries)
    
    # M1 新增: typed memory 显式拉结构化记录, 按 memory_type 优先级排序
    # ponytail: 不改原有 FTS5 路径, 在其结果上叠加 typed 记录去重.
    # 升级路径: FTS5 和 typed 合并到同一 SQL, 避免两次查.
    typed_results = self._recall_typed_for_prompt(query, max_entries)
    # 去重 (按 content hash), typed 优先
    seen = {hash(r.get("content", "")) for r in results}
    for tr in typed_results:
        if hash(tr.get("content", "")) not in seen:
            results.append(tr)
            seen.add(hash(tr.get("content", "")))
    return self._format_recall_results(results)
```

#### M2. `_recall_typed_for_prompt` 新方法

按 `memory_type` 优先级拉结构化记录:

```python
_TYPE_PRIORITY = {
    "failed_direction": 0,    # 最优先: 失败方向避免重蹈覆辙
    "iteration_result": 1,    # 上轮结果
    "cross_domain_transfer": 2,
    "persona_history": 3,     # persona 历史效果 (已有专门路径, 这里冗余兜底)
    "stable_principle": 4,    # 已有 JSONL 路径, 这里冗余兜底
    # user/feedback/project/reference/calculation 5 个老类型不进 typed 路径
}

def _recall_typed_for_prompt(self, query: str, max_entries: int) -> list[dict]:
    # 按 memory_type 优先级拉, 每个 type 最多 1 条
    results = []
    for mtype, priority in sorted(_TYPE_PRIORITY.items(), key=lambda x: x[1]):
        typed = self.recall_typed(memory_type=mtype, limit=1)
        if typed:
            results.extend(typed)
        if len(results) >= max_entries:
            break
    return results[:max_entries]
```

#### M3. `retrieve` ORDER BY 加 `memory_type` 优先级

[longterm.py:612-619](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/memory/longterm.py#L612-L619) ORDER BY 改:

```sql
-- 原: ORDER BY _TIER_ORDER, importance DESC, access_count DESC
-- 新: ORDER BY CASE WHEN memory_type IS NOT NULL THEN 0 ELSE 1 END,
--                _TYPE_PRIORITY_ORDER,  -- memory_type 优先级
--                _TIER_ORDER, importance DESC, access_count DESC
```

ponytail: 不改 SQLite schema, 用 CASE WHEN 在 SQL 里算优先级. 升级路径: 加 `type_priority` 列 + 索引.

#### M4. selfcheck

- `recall_for_prompt` 返回结果含 typed 记录 (memory_type 非空)
- typed 记录按 priority 排序 (failed_direction 在 iteration_result 前)
- FTS5 路径保留, 不破坏现有检索
- 性能: typed 路径不扫全表 (用 SQL WHERE memory_type = ?)

**改动量**: ~80-120 行 (manager.py 60 + longterm.py 20 + selfcheck 30)

**风险**: 低. typed 路径叠加在 FTS5 之上, 不替换. toggle off 时 `_use_typing()=False` 走原路径.

---

### D. LLM decider 可观测性

**治**: decider prompt 只给 LLM 看 5 个字段, learn 不更新 cog, report 是哑 action.

**现状 (基于代码实测)**:
- [engine.py:2282-2308](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/autoloop/engine.py#L2282-L2308) decider prompt 字段: iteration / last_action / hypothesis(120 字符) / plan_mode / execution(DONE|NONE) / validation(PASSED|FAILED|NONE)
- LLM 看不到: validation 具体字段 / `_consecutive_failures` / `_pivot_count` / `_speculator_hint` / `action_history` 完整序列 / `_next_phase_hint`
- `learn` action 不更新 cog ([engine.py:1871](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/autoloop/engine.py#L1871)), 下一轮 decider 看不到正反馈
- `report` action 在 execute_fn 是 no-op ([engine.py:1914](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/autoloop/engine.py#L1914)), LLM 选 report 浪费一轮
- `reflect`/`explore` 不是 action (reflect 是 CognitiveLoop 钩子, explore 是 plan mode), 文档跟代码不符

**改动**:

#### D1. 扩 decider prompt 字段

[engine.py:2282-2308](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/autoloop/engine.py#L2282-L2308) `_build_decider_prompt` 加:

```
- Validation details: {validation 具体字段 (thinking_collapse/physics_validation_error/benchmarks/effort_floor_deficits), 截断 300 字符}
- Consecutive failures: {_consecutive_failures}/20 (硬停阈值)
- Pivot count: {_pivot_count}/10
- Refine count: {_refine_count}
- Action history (last 10): {action_history[-10:]}
- Speculator hints (accumulated): {_speculator_hint 截断 300 字符}
- Last learn summary: {cog.get("last_learn_summary", "none")}
```

#### D2. learn 更新 cog

[engine.py:1871](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/autoloop/engine.py#L1871) learn 分支改:

```python
if action == "learn":
    # D2: learn 写 cog, 让 decider 下一轮看到正反馈
    # ponytail: 不暴露完整 _learn 内部状态, 只塞 1 行摘要.
    # 升级路径: _learn 返回结构化 summary, 不靠 string.
    try:
        result = await self._learn(...)
        cog["last_learn_summary"] = (
            f"learned: persona={result.get('persona','?')} "
            f"r_phys={result.get('r_phys','?')} "
            f"principles_added={result.get('principles_added',0)}"
        )
    except Exception as e:
        cog["last_learn_summary"] = f"learn failed: {type(e).__name__}"
```

`_learn` 方法签名加返回值 (当前返回 None):

```python
async def _learn(self, ...) -> dict[str, Any]:
    # 原有逻辑保留, 末尾加 return summary
    return {
        "persona": persona_id,
        "r_phys": r_phys,
        "principles_added": n_added,
    }
```

#### D3. report action 改: 不让 LLM 选

[engine.py:2310-2324](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/autoloop/engine.py#L2310-L2324) `_is_action_legal` 改:

```python
def _is_action_legal(self, action: str, cog: dict) -> bool:
    if action in ("observe", "hypothesize", "skip", "stop"):
        return True
    # D3: report 不让 LLM 选, 由 _finalize_run 跑
    # ponytail: report 是 finalize 阶段, 不是决策阶段. 升级路径:
    # 如果要 LLM 主动触发 report, 改成 action="stop" + rationale="report ready".
    if action == "report":
        return False
    if action == "plan":
        return bool(cog.get("hypothesis"))
    # ... 其余不变
```

同步从 [cognitive_loop.py:65-68](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/autoloop/cognitive_loop.py#L65-L68) `VALID_ACTIONS` 删 `report` (保留在文档里说明 "report 由 _finalize_run 自动跑").

#### D4. 文档修正: reflect/explore 不是 action

在 spec 注释里说明:
- `reflect` 是 CognitiveLoop 内部钩子 (每轮自动调), 不是 LLM 可选 action
- `explore` 是 plan mode 值 (`mode="explore"`), 不是 action

#### D5. selfcheck

- decider prompt 含新增字段 (validation details / consecutive_failures / action_history / last_learn_summary)
- learn 后 `cog["last_learn_summary"]` 非空
- LLM 选 report 时 `_is_action_legal` 返回 False
- `VALID_ACTIONS` 不含 `report`

**改动量**: ~100-150 行 (engine.py decider 50 + learn 30 + _is_action_legal 10 + cognitive_loop 5 + selfcheck 40)

**风险**: 中. decider prompt 变长可能影响 LLM 决策质量 (需 A/B 测). report 从 VALID_ACTIONS 删除是 breaking change (如果有外部代码依赖).

---

### C. context 双路径合并 + 死代码清理

**治**: engine 和 ContextBuilder 各一套 KB/KG build 逻辑漂移; metacog 5 个信号是死代码.

**现状 (基于代码实测)**:
- [context_builder.py:317](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/context_builder.py#L317) `build_kb_text` 有 KB→memory cross-reference, [engine.py:567](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/autoloop/engine.py#L567) 版没有
- ContextBuilder 5 个死方法: `build_target_chain_text`/`build_step_eval_text`/`build_meta_agent_text`/`build_pmk_text`/`build_prospective_text`
- `SessionContext.working_memory` dict 是死字段 ([session.py:41](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/memory/session.py#L41))
- autoloop 不读 `meta_trace.jsonl` 和 `evolution_rules.json` (只有 chat agent 读)

**改动**:

#### C1. KB/KG build 合并到共享模块

新文件 `huginn/context/builders.py` (或复用 `context_builder.py` 里的方法):

```python
# ponytail: 不重写, 把 ContextBuilder 版的方法提取为独立函数,
# engine 和 ContextBuilder 都调. 升级路径: 完全替换 engine 版.
def build_kb_text(memory, kb_store, query: str, include_cross_ref: bool = True) -> str:
    # 从 ContextBuilder.build_kb_text 提取, engine 版调这个
    ...

def build_kg_text(kg, query: str) -> str:
    # 同上
    ...
```

[engine.py:567](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/autoloop/engine.py#L567) `_build_kb_text` 改成调共享函数, `_build_kg_text` 同上.

#### C2. metacog 信号注入 autoloop prompt

5 个死方法里, 优先注入 2 个最有价值的 (其他 3 个先删):

- `build_target_chain_text` → 注入 `_build_hypothesis_prompt` 和 `_build_plan_prompt` 的 `ctx_hint` block
  - target_chain 是当前 hypothesis 在目标分解树里的位置, LLM 看了能避免偏题
- `build_prospective_text` → 注入 `_build_plan_prompt` 的 `ctx_hint` block
  - prospective 是前瞻意图 (pending 的下一步), LLM 看了能避免遗漏计划

删 3 个: `build_step_eval_text`/`build_meta_agent_text`/`build_pmk_text` (价值低, 死代码太久没人调, 删比接更省事).

#### C3. 删死代码

- `SessionContext.working_memory` dict ([session.py:41](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/memory/session.py#L41)) — 从未注入任何 prompt
- `ContextBuilder.build_step_eval_text` / `build_meta_agent_text` / `build_pmk_text` — 3 个死方法

#### C4. autoloop 读 meta_trace / evolution_rules

[engine.py:698](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/autoloop/engine.py#L698) `_build_memory_text` 加 meta_trace block (可选, toggle on 才注入):

```python
def _build_memory_text(self, query: str) -> str:
    # 原有 recall_for_prompt 路径保留
    ...
    # C4 新增: meta_trace 注入 (toggle on)
    if _harness_enabled("harness_meta_trace_inject"):
        meta_trace = self._load_meta_trace()  # 读 .huginn/meta_trace.jsonl 最近 5 条
        if meta_trace:
            parts.append(f"## Meta trace:\n{meta_trace}")
    return "\n\n".join(parts)
```

#### C5. selfcheck

- engine `_build_kb_text` 跟 ContextBuilder `build_kb_text` 输出一致 (cross-reference 都有)
- target_chain/prospective 注入 hypothesis/plan prompt
- `SessionContext.working_memory` 字段已删
- 3 个死方法已删
- toggle on 时 meta_trace 注入 memory_text

**改动量**: ~150-200 行 (新共享模块 60 + engine 接入 40 + 删死代码 20 + metacog 注入 50 + selfcheck 30)

**风险**: 中. 删死代码是 breaking change (如果有外部代码依赖). metacog 注入可能增加 prompt 长度.

---

## 优先级排序 + 改动量估算

| 块 | 优先级 | 改动量 | 风险 | 前置依赖 | 收益 |
|---|---|---|---|---|---|
| **M (typed memory 接通)** | **P0** | ~100 行 | 低 | 无 | EM/SM 分层在检索层生效; H3 learn 产出可见 |
| **D (decider 可观测性)** | **P1** | ~125 行 | 中 | M (learn 写 cog 要先有 typed memory 产出) | learn/report 不再是哑 action; 调度可控 |
| **C (context 双路径合并)** | **P2** | ~175 行 | 中 | M (memory 检索统一后再合并路径) | metacog 信号进 prompt; 死代码清理 |

**总改动量**: ~400 行 (M 100 + D 125 + C 175)

**执行顺序**:
1. M 先做 (P0, 无依赖, 收益最大)
2. D 跟上 (P1, 依赖 M 的 learn 产出)
3. C 最后 (P2, 依赖 M 的检索统一)

**跟已有 spec 的关系**:
- 跟 [layered_memory_spec.md](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/docs/layered_memory_spec.md) 不重叠: layered_memory_spec 治 4 层物理分层 (WM sliding window / EM embedding / PM Bayesian confidence), 本 spec 治已有字段接通主路径. C4/C5 (typed memory 默认 on + persona_history) 已在 layered_memory_spec 实现完成.
- 跟 [harness_evolution_spec.md](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/docs/harness_evolution_spec.md) 不重叠: harness_evolution 治 prompt/workflow/harness 可演化 (H0-H4), 本 spec 治 memory/context/dispatch 可观测. H4 phase 分批已完成, 本 spec 的 D 块可以复用 PhaseRegistry 加 phase 特定 decider prompt.

## 数学结构 (选读)

### M 块: typed memory 检索是 schema on read

FTS5 + category 是 unstructured retrieval (O(|tags|) 查询). typed memory 是 schema on read, 把 `(memory_type, run_id, persona_id, status)` 从 unstructured tags JSON 升级为 typed columns, 查询 O(|T|) (T = type enum, |T|=10).

类型优先级 `failed_direction > iteration_result > cross_domain_transfer > persona_history > stable_principle` 反映信息密度: 失败方向是负信号 (避免重蹈覆辙), iteration_result 是正信号 (复用成功), cross_domain_transfer 是弱正信号 (类比迁移).

### D 块: decider 可观测性是 POMDP belief state update

LLM decider 是 POMDP 里的 policy π(a|b), b 是 belief state. 当前 decider 看到的 5 个字段是 b 的粗投影, 信息损失大. 扩字段 = 扩 b 的维度, 让 π 更接近 π*.

learn 不更新 cog = belief state 不随 action 更新, LLM 无法区分 "learn 了" 和 "没 learn". 加 `last_learn_summary` = 给 b 加一维, 让 π 能感知 learn 产出.

### C 块: context 双路径合并是 DRY 原则

两套 KB/KG build 逻辑是 code duplication, 漂移是 entropy 增加的必然结果. 合并到共享模块 = 减少状态空间, 降低维护成本.

metacog 信号死代码 = 计算了但不注入 = 信息熵浪费. 注入 prompt = 让信息参与决策.

## 验证策略

### M 块验证

1. selfcheck: `recall_for_prompt` 返回结果含 typed 记录 (memory_type 非空), 按 priority 排序
2. 端到端: 3 轮 autoloop, 验证 typed memory 写入后被 recall_for_prompt 拉回 (iteration_result 在第 N+1 轮 prompt 里出现)
3. 性能: typed 路径不扫全表 (SQL WHERE memory_type = ? 走索引)

### D 块验证

1. selfcheck: decider prompt 含新增字段, learn 后 cog["last_learn_summary"] 非空, report 在 _is_action_legal 返回 False
2. 端到端: 3 轮 autoloop, 验证 LLM 选 learn 后下一轮 decider prompt 有 last_learn_summary, LLM 更可能选 hypothesize (而不是重复 learn)
3. A/B: 对比扩字段前后 LLM 决策质量 (learn 被选频率 / pivot 频率 / stop 频率)

### C 块验证

1. selfcheck: engine `_build_kb_text` 跟 ContextBuilder 输出一致, target_chain/prospective 注入 prompt, 死代码已删
2. 端到端: 3 轮 autoloop, 验证 target_chain 在 hypothesis/plan prompt 里出现, prospective 在 plan prompt 里出现
3. 死代码清理: grep 确认 3 个死方法 + working_memory 字段已删, 无外部引用

## 升级路径 (ponytail)

- M: typed 路径叠加在 FTS5 之上, 不替换. 未来可合并到同一 SQL (FTS5 + typed columns JOIN).
- D: decider prompt 字段可配置化 (PhaseRegistry extra 字段, 跟 H4 phase 分批复用).
- C: 共享模块可扩展为完整 context builder (替代 engine 和 ContextBuilder 两套路径).
