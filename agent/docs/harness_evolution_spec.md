# Agentic Engineering 升级 Spec — 拆剩余 hardcode 骨架 + harness 可演化

> 治当前 huginn 工作流的"行为体骨架天花板" + "自我改进机制断链": CognitiveLoop 只在 autoloop 路径落地 (rcb_runner 没接), 7 个自我改进机制只有 3 个真闭环, phase 方法体 / prompt block body / subagent spec / VALID_ACTIONS 仍是 hardcode 字面量, agent 不能改自己的行为.
> 数学动机: harness engineering (Weng 2026) 的核心论点 — "代码是通用语言, harness 应成为优化目标". CognitiveLoop 拆了控制流, 本 spec 拆行为体 + 修断链, 让 harness 真正可演化.
> 跟现有 [layered_memory_spec.md](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/docs/layered_memory_spec.md) 共生 — 4 层 memory 是基础设施, 本 spec 在其上建 harness 演化层.

## 源码调研暴露的真问题 (spec 修正点)

三轮源码调研共发现 10 个 spec 没暴露的真问题, 必须先修 (P1-P7 第二轮, P8-P10 第三轮):

### P1. CognitiveLoop 不是"两条路径共享内核"

`cognitive_loop.py:1-25` docstring 承诺的 "rcb_runner 和 autoloop 的共享控制流内核" 是**愿景而非现状**:

- ✅ autoloop 真在用: [engine.py:2150](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/autoloop/engine.py#L2150) 实例化, 4 钩子是 700 行闭包真实现 (engine.py:1448-2148)
- ❌ rcb_runner 没接: [rcb_runner.py:424](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/cli/rcb_runner.py#L424) 自己写 `for _iter_n in range(...)` 手写循环, 只复用 4 个无状态工具函数 (`update_heat_engine_after_step` / `update_drift_and_metrics` / `build_pmk_state` / `check_pause_decision`)
- rcb_runner.py:241-243 注释自承: "ponytail: 不接 AutoloopEngine (它用 CoderRunner/WorkflowEngine, 不写 report/report.md, 会破坏 RCBench 评分). 用 mini-loop + 手写 trace"
- `OutputWriter` 抽象基类 (cognitive_loop.py:233) **0 生产实现**, autoloop 实例化时传 `output_writer=None`, 只有 selfcheck 的 MockWriter

**影响**: H4 的 PhaseSpec 不能假设两条路径都走 CognitiveLoop, 只能在 autoloop 路径做. rcb_runner 的 hardcode 留给后续.

### P2. 7 个自我改进机制只有 3 个真闭环

| 机制 | 真在用? | 证据 |
|---|---|---|
| EvolutionEngine patches/skills | ✅ 真闭环 | `_build_plan_prompt` 注入 patch_hints + skill_hints |
| `_try_evolved_fix` | ✅ 真闭环 | 仅 workflow mode, 硬编码规则匹配 (非 LLM 生成) |
| `FailedDirectionStore` → `forget_then_generate` | ✅ 真闭环 | 仅 imaginate 模式触发 |
| `EvolutionManager.recommend` | ❌ 死代码 | 只在 selfcheck 调, 生产 0 调用 |
| `SkillEvolutionLayer.get_skill_context` | ❌ 死代码 | 0 生产调用, Beta 信念维护了但不进 prompt |
| `distill_episodic_to_procedural` | ❌ 死代码 | 0 生产调用, 只在 selfcheck 跑 |
| `_distill_meta_trace` | ❌ 只写不读 | 写 `.huginn/meta_trace.jsonl`, 不回流 prompt |

**影响**: spec 表格原把 SkillEvolutionLayer / EvolutionManager 列为"已拆"是不准的. H1/H3 复用 Beta 更新可以, 但**注入路径必须新写**, 不能指望现有死代码.

### P3. `stable_principles` 根本不进 autoloop prompt

- `_build_hypothesis_prompt` (engine.py:7123-7297) 拼 13 个 block: body/git_log/fail/imagination/exec/math/kg/visual/kb/mem/pm/cluster/hint — **没有 stable_principles block**
- `_build_plan_prompt` (engine.py:7345-7402) 也没有
- stable_principles 只进 chat agent 的 system prompt ([prompt_builder.py:104](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/agent/prompt_builder.py#L104)), autoloop engine 完全跳过它
- `recall_procedural` (memory/manager.py:1244) 0 调用

**影响**: spec 原说"H1 patch store 是 PM 层扩展" — 但 PM 层在 autoloop 里根本不通. H1 必须自建注入路径, 不能搭现有 PM 层的车.

### P4. RSI directive 的 recall 不可靠

- `_generate_next_loop_directive` (engine.py:5915) 写 directive 进 memory (category=self_directive, tier=mid)
- 下轮经 `_build_memory_text` → `recall_for_prompt(top_k=3)` — directive 跟其他 memory **竞争 top_k=3 slot**
- directive 语义相关性不够就被挤掉, **不是定向 recall**
- directive 纯文本, 截断 300 字符, 无结构化字段

**影响**: H1 必须把 directive 从"memory 软检索"改成"patch store 定向读", 否则 patch 生成器拿不到上轮反思.

### P5. 两套 evolution 系统并存, 不通信

- `EvolutionEngine` (evolution/engine.py, 老, 真在用, 有 Beta 更新 + heuristic_fix + prompt patches)
- `EvolutionManager` (evolution/manager.py, 新, P14 统一 API, 默认 off, recommend 死代码)
- 两者没有相互调用, EvolutionManager.record_outcome 内部调 FailedDirectionStore + SkillEvolutionLayer, 但 EvolutionEngine 不走这条

**影响**: H1/H2 接入时必须明确选哪套. 建议 H1 **直接绕过两者新建** `harness/prompt_patch.py`, 复用 `ToolBelief` 的 Beta 数学但独立存储, 避免两套 evolution 系统的耦合.

### P6. H2 接入是"加一层"不是"重写" (第三轮调研修正)

- `_execute_dynamic_workflow` (engine.py:4012) 是单点入口, **WorkflowScript 在方法内部构造** (L4036 `WorkflowScript.from_dict`), 不是外部传入
- `WorkflowScript` 已经是 declarative + 可序列化, **但 `id` 每次 from_dict 重新生成** (L89 `f"wf_{uuid.uuid4().hex[:8]}"`), bandit 必须自己存 (script_dict, variant_id) 映射, 或在 script 里加 `variant_id` 字段
- **ToolBelief key schema = (tool_name, param_key, param_value)** (skills/evolution.py:179, 223), 跟 workflow variant 完全不匹配. 方案 A 新建 WorkflowBelief ~80 行; 方案 B hack key schema (`tool_name="workflow_variant"`) ~10 行但污染 prompt context. **推荐方案 A**
- **没有现成 bandit selector**: `recommend_params` (skills/evolution.py:362-386) 是 UCB **ranking** 返回排好序的 list, 不是 sampling. 要 Thompson sampling 得新写 ~30 行 `random.betavariate(1+ws, 1+wf)` selector
- **没有现成 variant 生成器**: subtask 是纯 LLM JSON 生成 (dynamic_workflow.py:60-93), 没有 mutator. 要做 N 变体必须新写 variant 生成器 (LLM 一次写 N 个, 或参数扰动)
- novelty **不能用 `_compute_semantic_overlap`** — 它在 `context_builder.py:72` (不是 `agent/context_builder.py`, spec 路径错), 只是 BoW TF-IDF cosine, 参数级 diff 不敏感 (encut 520 vs 540 cosine≈1.0). **改用参数级 Jaccard diff** (~15 行, 比 TF-IDF 准)
- fitness 收集时机: r_phys 在 `_validate` (engine.py:4062) 才算出, `_execute_dynamic_workflow` 返回 dict **没有 r_phys 字段**, bandit 更新点必须从 `_validate` 的 `results["r_phys"]` 或 `results["grader_reward"]` 取 (L4100-4130, L4257)
- **_try_evolved_fix leak** (engine.py:3988-4010): 它调 `_execute_workflow` 不是 `_execute_dynamic_workflow`, 跟 H2 是两条独立路径不直接冲突, **但 variant 失败可能触发 `_try_evolved_fix` 走 `_execute_workflow`, bandit 收不到 outcome**. H2 必须在 `_try_evolved_fix` 入口加 guard: `if execution_result.get("_variant_id"): return None`

**影响**: spec 原 H2 估算 (~220 行 + 1 接入) **严重低估**. 真实改动量 ~300-400 行新文件 + 3-4 处接入 (见 H2 section 重新估算).

### P7. phase 方法体的可演化点密度差异巨大 (第三轮调研修正)

| Phase | 真实行数 | 字面量占比 (真实) | 抽象难度 | spec 原估偏差 |
|---|---|---|---|---|
| `_report` | 74 | ~50-60% (模板 + try/except + 5 个 `_last_*` 实例耦合 + persona "tutor" 两处硬编码) | 低 (试点次选) | 字面量高估 (原 80%), 隐藏耦合点没提 |
| `_execute` | 42 | ~60% (if/elif 6 mode 分支, 不是 dispatch table) | 低 | "dispatch table" 错, mode 白名单原说两处实际 8+ 处 |
| `_perceive` | 58+49 | ~10-15% (3 SignalHub route + 4 context key, 不是关键词表) | 低 | 占比高估一倍 (原 30%) |
| `_hypothesize` | 92 | ~10-15% (env 字符串 + "SELECTED:" 标记 + f-string) | 低 | 占比高估 (原 40%), spec 说的 5 个常量 (_MATH_SIGNALS / _IMAGINATION_PROMPT_BLOCK / _MATH_DEPTH_PROMPT_BLOCK / persona 阈值 / dimension 白名单) **全不在 _hypothesize 内**, 在 `_build_hypothesis_prompt` 或 `_pick_hypothesis_persona` |
| `_plan` | 84 | ~20% (PlanStep/PlanStore dict + confirm/reject 字符串白名单) | 低 | 占比高估 (原 55%), spec 说的 4 项 (_override_plan_mode / _plan_check_tier / _DOMAIN_TEMPLATE_NAMES / _workflow_signals) **全不在 _plan 内**, 在 helper 方法 |
| `_learn` | 473 | ~30-40% (f-string summary + dict/tags 字面量) | 中 | 基本准确, 8 触发阈值 ✓, Feynman prompt 在 L6062 ✓, importance 公式在 `_generate_next_loop_directive` (L5972) 不在 _learn proper |
| `_validate` | **303** (不是 1380) | ~15-20% (dict key + f-string + 3 category 白名单) | **低** (第三适合做试点) | **行数严重错误** (原 1380 把 _validate main 和整个 validation 子系统 ~1334 行混淆), "散布在大块控制流难以抽象" 不准确 — 实际是 9 个独立 sub-block 各包 helper, 正好是 validator pipeline 理想形态 |

**关键修正**: spec 原把 `_validate` 列为"最难缓做"是基于错误行数 (1380 vs 真实 303) 和对结构的误判. 真实 _validate main 是 9 个独立 sub-block 的串行编排, 唯一共享状态是 `results` dict. `_build_reviewer_prompt` (L5409-5439) 已经是 `@staticmethod` 独立抽好了. **试点首选顺序应改: BUILTIN_SPECS → _report → _validate → _execute → 其他**.

**关键耦合点**: `mode` 白名单 **8+ 处独立硬编码** (不是 spec 原说的两处), PhaseSpec 抽象时必须做成单一 source of truth. 8 处分布:
- `_execute` if/elif 链 (engine.py:3929-3953, 6 modes)
- `_override_plan_mode` 硬编码 `"coder"/"workflow"/"explore"` (engine.py:7583-7622)
- `_parse_plan` 默认 `mode="coder"` (engine.py:7652)
- `_build_plan_prompt` body 描述 4 个 mode + MODE 输出格式漏 2 个 (engine.py:7489-7493, 7509)
- `_plan_check_scene_tag` 用 `"workflow"/"skill"` (engine.py:7948)
- `_plan_check_complexity` 用 `{"workflow":0.4, "skill":0.3, ...}` 漏 2 个 (engine.py:8042)
- `_get_refine_model` 的 `expensive_modes` 完全不同维度 (engine.py:1263)
- `_execute_*` helper 各自硬编码返回 `"mode": "..."` (engine.py:6466/6475/6564/6580/6595/6601/6624/6634/6653)

### P8. LLM client 不是 unified (H3 model 维度联合优化受阻)

- `_llm_chat` (engine.py:7054-7121) 是 autoloop 路径统一入口, **只覆盖 autoloop engine 自己**
- `_hypothesize` 的 hot_model 双路采样 (engine.py:2995) 不走 router, 直接 `self.model.bind(temperature=1.0)`, **是同一个模型实例 bind 不同 temperature, 不是真多模型并发**
- `subagent._summarize` (subagent.py:447) 直接 `await asyncio.to_thread(model.invoke, messages)`, 完全绕过 `_llm_chat`
- `ModelRouter.select(task, prefer_cheap)` 是 task-based 路由 (reasoning→强 / planning→中 / summarize→便宜), **不是 agent-choice 路由**
- `_llm_chat` 的 `model: Any = None` 参数由 caller 显式传入 (engine.py:7058), 没有接口让 LLM 自己输出 "用 alias X 跑这次"

**影响**: H3 joint optimization 只能联合 (block, stage), 加不进 model 维度. model 选择留作 H5 候选 (不在本 spec 范围).

### P9. Tool 调用 9+ 处直接 `tool.call()` 无统一 dispatch (H4 tool_whitelist 失效)

- engine.py 有 9 处直接 `await tool.call()` (L3706 / L4509 / L4533 / L5054 / L5093 / L5119 / L5274 / L5302 / L5339)
- 外加 2 处 `ToolRegistry.get("image_analysis_tool")` 直接取实例 (L6690, L6903)
- subagent 路径有 `agent.tool_filter` 在注册时过滤 (subagent.py:264-267), 但 phase 方法体内直接 `tool.call()` **不走 whitelist 检查**
- 没有统一 `dispatch_tool(name, args, ctx)` 入口

**影响**: H4 的 `tool_whitelist` 字段如果只在 PhaseRegistry 里存, 不在 `tool.call()` 调用前检查, 等于白存. ponytail 取舍: phase 内部 tool_whitelist 标记为 **advisory 不强制** (避免改 9 处 call site), 只在 subagent dispatch 路径强制.

### P10. BUILTIN_SPECS 应纳入 H4 而不是留给后续

- `BUILTIN_SPECS` 4 个 subagent spec (subagent.py:115-183) 是类属性硬编码
- `SubagentSpec` 结构 (subagent.py:41-71): `name / description / system_prompt / allowed_tools / max_tool_calls / max_iterations / summarize_result / summary_format / max_depth`
- **SubagentSpec = PhaseSpec − postcondition − fallback**, 是 PhaseSpec 最简单的子集
- 已有弱动态加载机制: `register_spec()` (subagent.py:316-318) 实例级注册, 但无文件加载 / yaml / LLM 提案
- 4 个 spec 共 ~60 行字面量, 100% 字面量占比, **比 _report (50-60%) 更易抽象**
- 已有 `_PHASE_PERSONAS` (engine.py:111) 跟 BUILTIN_SPECS 是同一层硬编码

**影响**: spec 原 L365 "BUILTIN_SPECS 可扩展为可演化 spec (本 spec 不做, 留给后续)" 判断错误. SubagentSpec 是 PhaseSpec 试点首选, 比任何 phase 方法体都简单. 应纳入 H4 scope.

## 动机

翁荔博客把 harness engineering 分 5 个优化对象: prompt → structured context → workflow → harness code → optimizer code. 当前 huginn 的覆盖:

| 优化对象 | 当前状态 | 文件证据 |
|---|---|---|
| prompt (template body) | ❌ 字面量字符串, agent 不能改 | [engine.py:7256](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/autoloop/engine.py#L7256) `_build_hypothesis_prompt` body 是 f-string |
| structured context | ✅ 4 路检索 + 分层 budget | [engine.py:554-700](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/autoloop/engine.py#L554) `_build_*_text` |
| workflow | 🟡 模板化 + dynamic_workflow 并行, 但不搜索演化 | [workflows/templates.py](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/workflows/templates.py), [dynamic_workflow.py:50](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/autoloop/dynamic_workflow.py#L50) |
| harness code | 🟡 CognitiveLoop 拆了控制流, phase 方法体仍 hardcode | [cognitive_loop.py:84](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/autoloop/cognitive_loop.py#L84) `CognitiveLoop` + [engine.py:2775+](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/autoloop/engine.py#L2775) `_perceive` 等 7 个 phase 方法 |
| optimizer code | ❌ 无元学习, 无 STOP 式递归改进器 | 无 |

调研出来的 3 个最大空缺 (对标文章 5 个模式):

1. **Self-Improving Harness 完全缺**: 现有 RSI directive ([engine.py:5915](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/autoloop/engine.py#L5915) `_generate_next_loop_directive`) 只写文本 hint 到 memory, 下轮通过 `_build_memory_text` 检索注入. **agent 不能改 prompt template, 不能改 phase 方法体, 不能改 tool 实现**. 这是"软 RSI", 不是文章说的"代码作为可执行搜索空间".

2. **Evolutionary Search 完全缺**: 对同一 objective, 当前只跑一个 workflow. 没有 ADAS / AFlow 式的"生成 N 个 workflow 变体 → 跑 → 用 r_phys 选最优 → 归档". [dynamic_workflow.py](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/autoloop/dynamic_workflow.py) 能 LLM 生成 subtask, 但不进 dependency graph, 不做适应度比较.

3. **Joint Optimization 完全缺**: prompt block 选择 ([engine.py:7123](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/autoloop/engine.py#L7123) `_build_hypothesis_prompt` 拼 13 个 block) 和 workflow stage 选择 ([workflows/templates.py](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/workflows/templates.py)) 各自独立. 没有"prompt block + workflow stage 统一成可搜索空间, 用 Beta 信念做 bandit".

**真正已成熟不动** (YAGNI, 不在本 spec 范围):
- ✅ CognitiveLoop 4 控制流钩子 observe/decide/execute/reflect (仅 autoloop 路径, rcb_runner 没接 — 见 P1)
- ✅ LoopState / ActionDecision / ReflectionResult dataclass
- ✅ 死循环防护 (3x redirect / 6x stop)
- ✅ 4 层 memory (PMK 闭环, commit `9204648` + `4a73f1f`)
- ✅ Sub-agent DAG 并行 + H¹ 一致性 (G1, [subagent_tool.py](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/tools/subagent_tool.py))
- ✅ EvolutionEngine patches/skills → `_build_plan_prompt` 真闭环 ([evolution/engine.py:64](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/evolution/engine.py#L64))
- ✅ `_try_evolved_fix` workflow 失败修复真闭环 ([engine.py:3988](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/autoloop/engine.py#L3988))
- ✅ FailedDirectionStore → forget_then_generate 真闭环 ([engine.py:5787](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/autoloop/engine.py#L5787))
- ✅ BenchmarkSuite 离线评估 ([self_improvement/core.py:183](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/self_improvement/core.py#L183))

**断链需修复 (H1/H2 接入时一并修)**:
- ❌ `SkillEvolutionLayer.get_skill_context` 0 生产调用 → H3 复用其 Beta 数学, 但注入路径新写
- ❌ `EvolutionManager.recommend` 0 生产调用 → H1 不走它, 新建独立 patch store
- ❌ `distill_episodic_to_procedural` 0 生产调用 → H0 (新增, 见下) 先把 stable_principles 接进 autoloop prompt
- ❌ `recall_procedural` 0 调用 → 同上
- ❌ `_distill_meta_trace` 只写不读 → 不修, 留作审计日志
- ❌ RSI directive 走 memory 软检索不可靠 → H1 改成 patch store 定向读
- ❌ `OutputWriter` 0 生产实现 → 不修, autoloop 用 `_record_provenance` 兜底

**真正还没拆的 hardcode** (本 spec 治这些):
1. `VALID_ACTIONS = frozenset({...})` ([cognitive_loop.py:65](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/autoloop/cognitive_loop.py#L65)) — 10 个 action 写死, agent 不能新增
2. `_perceive/_hypothesize/_plan/_execute/_validate/_learn/_report` 7 个 phase 方法体 ([engine.py:2775+](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/autoloop/engine.py#L2775)) — hardcode 逻辑
3. `_build_*_prompt` 的 block 顺序和默认 body ([engine.py:7123+](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/autoloop/engine.py#L7123)) — 字面量字符串
4. `BUILTIN_SPECS` 4 个 subagent spec ([agents/subagent.py:115](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/agents/subagent.py#L115)) — 类属性
5. `AUTOLOOP_PHASES = ("perceive", "hypothesize", ...)` ([engine.py:70](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/autoloop/engine.py#L70)) — frozen tuple

## 设计原则 (跟文章对齐)

1. **代码作为通用语言** — harness 改进不靠"文本 hint 塞 memory", 靠"agent 输出可执行 patch (JSON / Python 源码片段)". 文章原话: "code is the universal language for defining programs and systems".
2. **双层优化** — 内层 (base level) 跑任务, 外层 (meta level) 改 harness. 对应文章 MCE (Ye et al. 2026) 的 $\text{Inner: } c^*=\arg\max_c J_{\text{train}}(c;s)$, $\text{Outer: } s^*=\arg\max_s J_{\text{val}}(c^*)$. 我们用 r_phys 做适应度, 不用 train/val split (单任务场景没意义).
3. **Pareto 前沿而非单点** — 文章 Meta-Harness (Lee et al. 2026) 输出 Pareto 前沿上的候选集. 我们保留 top-K harness 变体, 不强求 single best.
4. **File system as memory** — harness 变体存文件 (`.huginn/harness_variants/<id>.json`), 不塞 context window. 对应文章 Pattern 2.
5. **拆骨架不破坏现有控制流** — CognitiveLoop 4 钩子不动, 在钩子内部把 phase 方法体 / prompt body / subagent spec 抽成可演化. 升级路径明确, 回退成本 = 删一个目录.

## 数学结构 (信息论分层, 不做证明)

### harness 变体 = 可执行 patch

harness 变体 $v$ 是一个 patch 字典:

$$v = \{\text{prompt\_patches}: \text{dict}, \text{workflow\_patches}: \text{dict}, \text{action\_extensions}: \text{list}, \text{phase\_patches}: \text{dict}\}$$

- `prompt_patches`: `{block_name: new_text}` — 替换 `_build_*_prompt` 里的特定 block
- `workflow_patches`: `{stage_name: {params: {...}}}` — 调 workflow stage 参数
- `action_extensions`: `[{name, condition, effect}]` — 扩展 VALID_ACTIONS (受控, 见 H3)
- `phase_patches`: `{phase_name: {hook_override: callable_spec}}` — 替换 phase 方法体的可序列化描述 (见 H4)

### 适应度 = 多目标 Pareto

每次 rollout 收集 3 个指标:
- $r_{\text{phys}}$: 物理奖励 ([engine.py:4062](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/autoloop/engine.py#L4062) `_validate`)
- $\text{efficiency} = \text{tool\_calls}^{-1} \cdot r_{\text{phys}}$: token/调用效率
- $\text{novelty} = 1 - \text{cosine}(\text{trajectory embedding}, \text{archive mean})$: 跟归档库的差异度

Pareto 前沿 = ${v \in V: \neg \exists v' \text{ dominates } v}$. 保留 top-K=5.

### Beta 信念 bandit

每个 harness 变体 $v$ 维护 Beta($\alpha_v, \beta_v$), rollout 成功 $\alpha += 1$, 失败 $\beta += 1$. UCB 选择:

$$v^* = \arg\max_v \left[ \frac{\alpha_v}{\alpha_v + \beta_v} + c \sqrt{\frac{\ln N}{n_v}} \right]$$

复用 [SkillEvolutionLayer](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/skills/evolution.py#L164) 的 Beta 更新逻辑, 不重写.

## 5 个改造方向 (H0/H1/H2/H3/H4)

### H0. stable_principles 接进 autoloop prompt (修 P3 断链, H1 的最小前置)

**治**: `_build_hypothesis_prompt` / `_build_plan_prompt` 没有 stable_principles block, autoloop 完全跳过 PM 层.

**接入**: 改 [engine.py:7123](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/autoloop/engine.py#L7123) `_build_hypothesis_prompt` 和 [engine.py:7345](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/autoloop/engine.py#L7345) `_build_plan_prompt`, 在 block list 里加一个:
```python
("principles", "\n".join(f"- {p}" for p in load_stable_principles()[:5]))
```
放在 `mem` block 之前 (优先级略高于 mem, 因为是结构化规则).

同时改 [engine.py:5441](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/autoloop/engine.py#L5441) `_learn`, 在合适位置调 `memory.distill_episodic_to_procedural(...)` (当前死代码), 让 STABLE_PRINCIPLE 真有产出.

**改动量**: ~20 行 (2 处 prompt 加 block + 1 处 _learn 调 distill + import).

**安全边界**:
- stable_principles 取 top-5, 避免塞爆 prompt
- distill 触发条件不变 (同 attempted 3 次成功), 不降低阈值
- ponytail: 不做 LLM 改写 principle, 模板化字符串够用. 升级路径: LLM 抽象.

**为什么先做**: H1 的 patch store 需要"PM 层在 autoloop 通"这个地基. 不先做 H0, H1 的 patch 跟 stable_principles 是两套独立 store, 不能统一管理.

### H1. Prompt Template Patch 闭环 (Self-Improving Harness 的最小工程实现) (第三轮调研修正)

**治**: RSI directive 只写文本 hint, 不改 prompt template body; directive 走 memory 软检索不可靠 (P4). **但 plan prompt 已部分接入** (engine.py:7377-7402 `evolution.get_prompt_patches()` 返回 `list[str]` 拼进 `skill` block), 只是 string-hint 不是 block 级 patch. H1 是**升级现有 string-hint 为 block 级 patch**, 不是全新接入.

**数学**: harness 变体 $v$ 的 `prompt_patches` 是 $\{b: \text{new\_text}\}$, 应用时 `_build_*_prompt` 的 block $b$ 用 $\text{new\_text}$ 替换默认值. 双层优化: 内层 = 跑任务拿 r_phys, 外层 = LLM 看 r_phys 生成新 patch.

**接入**: 新增 `huginn/harness/prompt_patch.py` (独立 store, 不走 EvolutionEngine / EvolutionManager, 避免 P5 两套系统耦合). **store 用懒加载单例**, 跟 `_get_evolution` (engine.py:460-467) 同模式, 保证跨 iter 状态持久:
- `PromptPatchStore` — 存取 `.huginn/prompt_patches/<id>.json`, schema `{id, phase, block_name, new_text, alpha, beta, created_at, last_used, directive_in}`. `directive_in` 字段保存生成此 patch 时用的 RSI directive (修 P4: directive 从 memory 软检索改成 patch store 定向读)
- `BlockPatch` dataclass — `{name, op=replace|prepend|append, body, priority}`, 替代 spec 原说的纯 `{block_name: new_text}` (支持 prepend/append 避免破坏 body 块的 `{context}` 占位符语义)
- `apply_patches(blocks, phase)` — 在 `_build_*_prompt` 拼完 `list[tuple[str, str]]` blocks 后, 调此函数按 phase 查 patch store, 替换/前置/后置对应 block. **必须在 `_trim_to_budget` 调用前注入**, 因为 `_scan_block_conflicts` 已在 `_trim_to_budget` 内自动跑 (engine.py:842), 不需要额外接入 conflict 检查
- `generate_patch(phase, blocks, r_phys, directive)` — LLM 看 r_phys 和上轮 directive, 输出 `{block_name: new_text}` JSON. 失败 (JSON parse 错误 / block 不存在) 静默丢弃, 不阻塞主循环
- `update_alpha_beta(patch_id, success)` — Beta 更新, 复用 `ToolBelief` (skills/evolution.py:62-161) 的数学但独立存储. **注意: spec 原说的 `update_fitness` 方法不存在**, 真实接入点是 `_learn` 的 `evolve_from_rewards()` 调用 (engine.py:5549-5571, evolution/engine.py:288), 根据 `tests_passed` 分支调 `update_alpha_beta`

**改动量**: ~150-210 行 (跟 spec 原 ~180 行基本吻合, 但接入点数量和位置需修正):
- 新文件 `huginn/harness/prompt_patch.py` ~120-180 行 (PromptPatchStore + BlockPatch + apply_patches + generate_patch + update_alpha_beta)
- engine.py 接入 **4 处** (spec 原说 3 处偏少, 每处比 spec 设想的小):
  1. `_build_hypothesis_prompt` (engine.py:7252-7295) — 在 `_trim_to_budget` 调用前插 `blocks = self._apply_block_patches(blocks, phase="hypothesize")`, ~3-5 行 (+ 新 helper `_apply_block_patches` ~15-20 行)
  2. `_build_plan_prompt` (engine.py:7479-7528) — 同上 ~3-5 行. **注意: 现有 L7377-7402 string-hint 逻辑迁移到 block patch 机制, 不重复注入**
  3. `_learn` (engine.py:5549-5571) — 在 `evolve_from_rewards()` 调用前后挂 Beta 更新, 根据 `tests_passed` 分支, ~5-10 行
  4. `_generate_next_loop_directive` (engine.py:5980 之后) — 加 `evolution.store_directive_patch(directive, importance=importance)` 调用, ~5-8 行. **注意: directive 召回路径已闭合** (`_build_memory_text` → `recall_for_prompt(top_k=3)`, engine.py:685-699), 不需要新写召回逻辑, 只需新增"同时写 patch store"的定向写

**安全边界**:
- patch 只能改 block 内容, 不能改 block 顺序 / 优先级 (避免 prompt 结构崩坏)
- patch store 最多 20 条 (LRU), 避免无限增长
- **conflict 检查不需要额外接入** — `_scan_block_conflicts` (engine.py:779-811) 已在 `_trim_to_budget` (engine.py:842) 自动跑, 只要 patch 在 `_trim_to_budget` 之前注入就自动覆盖
- `body` block 的 patch 必须保留 `{context}` 占位符语义 (BlockPatch.op=replace 时检查), 否则 hypothesis prompt 丢失上下文
- patch 生成失败静默丢弃 + $\beta += 1$, 不阻塞主循环
- ponytail: 不做 patch diff / version control, JSON 文件够用. 升级路径: 接 git.

### H2. Workflow Evolutionary Search (Evolutionary Search 模式) (第三轮调研修正)

**治**: 对同一 objective 只跑一个 workflow, 无搜索演化.

**数学**: workflow 变体 $w \in W$, 适应度 $f(w) = (r_{\text{phys}}, \text{efficiency}, \text{novelty})$. ADAS 式归档: $W_{t+1} = \text{top-K}(\text{ Pareto}(W_t \cup \{w_{\text{new}}\}))$. $w_{\text{new}}$ 由 LLM crossover 现有归档生成.

**接入** (新增 2 个文件 + 4 处 engine.py 接入):
- 新增 `huginn/autoloop/bandit.py` (~180-220 行):
  - `WorkflowBelief` dataclass — 独立于 `ToolBelief` (P6: key schema 不匹配), 字段 `variant_id / objective_hash / alpha / beta / last_updated / weighted_alpha / weighted_beta` (复用 ToolBelief 的 ANCCR 时间加权 + decay 数学, 但 key = (objective_hash, variant_id))
  - `WorkflowBandit` 类 — `record_variant_outcome(vid, reward, ts)` + `select_variant(candidates, exploration=0.3)` (**Thompson sampling** via `random.betavariate(1+ws, 1+wf)`, 不是 UCB ranking — P6: `recommend_params` 是 ranking 不是 sampling, 不能复用)
  - `VariantArchive` 类 — 存 `.huginn/workflow_archive/<objective_hash>.json`, schema `{variants: [{variant_id, script_dict, fitness, alpha, beta, created_at}]}`. **注意: WorkflowScript.id 每次 from_dict 重新生成, 必须用独立 variant_id 字段追踪**
  - `compute_novelty(new_script, archive)` — **参数级 Jaccard diff** (~15 行, 不用 `_compute_semantic_overlap` — P6: 它在 `context_builder.py:72` 不是 `agent/context_builder.py`, 且 BoW TF-IDF cosine 对参数级 diff 不敏感)
- 新增 `huginn/autoloop/variant_gen.py` (~80-120 行):
  - `generate_variants(objective, n=3, base_script=None) -> list[WorkflowScript]` — LLM prompt 让 agent 一次写 N 个 script dict (~30 行 prompt + parse), 或对 base_script 的 subtasks.args 做参数扰动 (encut ±50, kpoints 2x2→3x3, 需工具 schema 支撑, ~60 行)
- 扩 `WorkflowScript.from_dict` (dynamic_workflow.py:60-93, +5 行) — 加 `variant_id` 字段识别
- 扩 `engine._execute_dynamic_workflow` (engine.py:4012-4060, +30-50 行) — 如果 plan 带 `n_variants` 字段走 bandit loop: 调 `generate_variants` → `bandit.select_variant` → 跑选中 variant → 在 _validate 后调 `bandit.record_variant_outcome`. 或新写 `_execute_dynamic_workflow_bandit` 不动原方法
- 扩 `engine._validate` (engine.py:4100 附近, +10-15 行) — bandit 更新 hook: `if execution_result.get("_variant_id"): self._bandit.record_variant_outcome(vid, results.get("r_phys") or results.get("grader_reward", 0.5))`. **必须在 r_phys (L4100-4130) 和 grader_reward (L4257) 都算出之后**
- 扩 `engine._try_evolved_fix` (engine.py:3988-4010, +5-10 行) — **入口加 guard** (修 P6 leak): `if execution_result.get("_variant_id"): return None` (variant 失败不走 evolved_fix, 直接回 bandit loop 记录结果)

**改动量**: ~300-400 行新文件 (2 个) + 3-4 处 engine.py 接入. **spec 原 ~220 行 + 1 接入严重低估** (主要因为: 没有现成 bandit selector +30 行, ToolBelief key schema 不匹配需新建 WorkflowBelief +80 行, 没有现成 variant 生成器 +80-120 行).

**安全边界**:
- bandit 只在 `HUGINN_WORKFLOW_EVOLUTION=1` 时开 (默认 off, 跟极限模式一致) — **toggle 走 cfg 文件路径不走 env** (config.py 的 `get_config` 自动 reload, env 改了不生效需重启进程)
- variant 跑失败 (timeout / exception) 记 $\beta += 1$, 不污染主循环
- novelty 用**参数级 Jaccard diff** (~15 行), **不用 `_compute_semantic_overlap`** (P6: BoW TF-IDF cosine 对参数级 diff 不敏感, encut 520 vs 540 cosine≈1.0)
- fitness 收集时机: r_phys 在 `_validate` 才算出, `_execute_dynamic_workflow` 返回 dict **没有 r_phys 字段**, bandit 更新点必须在 `_validate` 的 `results["r_phys"]` 或 `results["grader_reward"]` 算出之后
- `_try_evolved_fix` 入口加 guard 隔离 variant 路径 (修 P6 leak)
- ponytail: 不做 AFlow 式 MCTS, ADAS 式随机归档 + Thompson sampling 够用. 升级路径: MCTS + dependency graph.

### H3. Joint Prompt + Workflow Optimization (Joint Optimization 模式) (第三轮调研修正)

**治**: prompt block 选择和 workflow stage 选择各自独立, 无联合优化.

**数学**: 统一搜索空间 $S = \{(b_1, ..., b_k; s_1, ..., s_m)\}$, 其中 $b_i$ 是 prompt block 选择 (on/off), $s_j$ 是 workflow stage 参数. 每个 $s \in S$ 维护 Beta 信念, UCB 选. **P8 限制: 不包含 model 维度** — LLM client 不是 unified (`_llm_chat` 只覆盖 autoloop engine, subagent._summarize 和 hot_model 双路都绕过), agent 选 model 的接口不存在. model 维度联合优化留作 H5 候选 (不在本 spec 范围).

**接入**: 新增 `huginn/harness/joint_optimizer.py`:
- `JointBandit` — 管理组合 $(b, s)$ 的 Beta 信念, 复用 [SkillEvolutionLayer](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/skills/evolution.py#L164) 的 Beta 更新
- `select_block_subset(phase, full_blocks)` — 按 Beta 信念选 block 子集 (低信念 block 概率性 off)
- `select_workflow_params(stage_name, defaults)` — 按 Beta 信念调 stage 参数 (如 `encut` 在 [450, 550] 区间选)
- H1 的 `apply_patches` 调 `select_block_subset`, H2 的 `generate_workflow_variants` 调 `select_workflow_params`

**改动量**: ~140 行新文件 + 2 处接入 (H1 和 H2 各一行).

**安全边界**:
- block 子集至少保留 `body / fail / exec` 3 个核心 block (避免 prompt 崩坏)
- workflow 参数有上下界 (复用 [config.py](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/config.py) 的 `HUGINN_*` env 约束)
- **不包含 model 维度** (P8 限制), 升级路径: unified LLM client + agent-choice 路由
- ponytail: 不做笛卡尔积全搜索, bandit + UCB 够用. 升级路径: 树搜索 + dependency graph.

### H4. Phase 行为体 + BUILTIN_SPECS 可演化 (拆剩余 hardcode 骨架) (第三轮调研修正)

**治**: CognitiveLoop 拆了控制流 (仅 autoloop 路径, 见 P1), 但 7 个 phase 方法体 + VALID_ACTIONS + AUTOLOOP_PHASES + BUILTIN_SPECS + _PHASE_PERSONAS 仍 hardcode. agent 不能新增 action, 不能改 phase 内部逻辑, 不能改 subagent spec. **P10 修正: BUILTIN_SPECS 纳入 H4 scope** (原 spec 错误地"留给后续"), 它是 PhaseSpec 最简单的子集, 是试点首选.

**数学**: 把 phase 方法体 + subagent spec 抽成可序列化的 `PhaseSpec`:
$$\text{PhaseSpec} = \{\text{name}, \text{prompt\_template}, \text{tool\_whitelist}, \text{postcondition}, \text{fallback}\}$$

`SubagentSpec = PhaseSpec − postcondition − fallback` (subagent.py:41-71), 是 PhaseSpec 的子集.

`VALID_ACTIONS` 从 frozenset 变成可 append 的 set, 但 append 受 `action_extensions` 控制 (LLM 提案 + 用户确认 / 规则过滤).

**接入** (按 P7 修正后的可演化点密度, **试点顺序改为 BUILTIN_SPECS → _report → _validate → _execute → 其他**):
- 新增 `huginn/harness/phase_spec.py`:
  - `PhaseSpec` dataclass — 描述一个 phase 的可演化字段 (prompt_template / tool_whitelist / postcondition / fallback)
  - `PhaseRegistry` — 存 `.huginn/phase_registry.json`, 默认载入 7 个 phase + 4 个 BUILTIN_SPECS 的当前 hardcode 版本作为 baseline. **用懒加载单例**, 跟 `_get_evolution` 同模式
  - `register_action_extension(name, condition, effect)` — 受控扩 VALID_ACTIONS, 条件 + 效果都是可序列化描述
  - `get_phase_spec(phase_name)` / `get_subagent_spec(name)` — 返回当前 spec (可能被 patch 覆盖)
- 扩 [engine.py](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/autoloop/engine.py) — 按修正后 P7 顺序分批:
  1. **BUILTIN_SPECS (试点首选, 100% 字面量)**: 把 `SubagentSpec` 4 个 spec (subagent.py:115-183) + `_PHASE_PERSONAS` (engine.py:111) 抽进 PhaseRegistry. **比 _report 更易** — SubagentSpec 已是 dataclass, 只需加文件加载 + register_spec() 已有
  2. **_report (74 行, 50-60% 字面量, 试点次选)**: 把 5 个 `_last_*` 实例状态读取收敛到 `_collect_report_evidence()` 方法, 主方法剩 30 行编排. `_build_science_report_prompt` (L6395-6443) 已是 `@staticmethod` 可零成本抽出. persona `"tutor"` 两处硬编码 (L6041 + L111) 统一到 PhaseRegistry
  3. **_validate (303 行不是 1380, 15-20% 字面量, 第三适合)**: 抽成 `list[Validator]`, 每个 validator `__call__(execution_result, results) -> dict`. 9 个 sub-block 各抽一个 validator. `_build_reviewer_prompt` (L5409-5439) 已是 `@staticmethod` 独立. reviewer 触发阈值 (L4184 `score < 0.5`) + MatWorldBench category 白名单 (L4287 `("structure","thermo","electronic")`) 抽成常量. **spec 原说"最难缓做"是基于错误行数和误判, 实际第三适合做试点**
  4. **_execute (42 行, 60%, if/elif 不是 dispatch table)**: 把 mode → 执行器映射表做成 spec.dispatch_table. **关键: 解决 mode 白名单 8+ 处硬编码** (P7 关键耦合点) — 做成单一 source of truth, 8 处都读它
  5. **_perceive (58+49 行, 10-15%)**: 3 SignalHub route + 4 context key 抽进 spec
  6. **_hypothesize (92 行, 10-15%)**: env 字符串 + "SELECTED:" 标记抽进 spec. **spec 原说的 5 个常量 (_MATH_SIGNALS / _IMAGINATION_PROMPT_BLOCK / _MATH_DEPTH_PROMPT_BLOCK / persona 阈值 / dimension 白名单) 全不在 _hypothesize 内**, 在 `_build_hypothesis_prompt` (H1 已处理) 或 `_pick_hypothesis_persona` (单独抽)
  7. **_plan (84 行, 20%)**: PlanStep/PlanStore dict + confirm/reject 字符串白名单抽进 spec. **spec 原说的 4 项 (_override_plan_mode / _plan_check_tier / _DOMAIN_TEMPLATE_NAMES / _workflow_signals) 全不在 _plan 内**, 在 helper 方法
  8. **_learn (473 行, 30-40%, 中等难度)**: 8 个触发阈值 + Feynman prompt (L6062) + importance 公式 (在 _generate_next_loop_directive L5972) 抽进 spec
- 扩 [cognitive_loop.py:65](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/autoloop/cognitive_loop.py#L65) `VALID_ACTIONS` — 改为 property, 返回 `base_actions | registered_extensions`

**改动量**: ~280 行新文件 (PhaseSpec + PhaseRegistry, 含 subagent spec 支持) + 8 处分批改造 (BUILTIN_SPECS + 7 phase) + 1 处 cognitive_loop.py. 总计 ~700 行 (比 spec 原 ~600 行略增, 主要因为纳入 BUILTIN_SPECS + 修正 _validate 实际可做). 但 **BUILTIN_SPECS 试点只需 ~80 行** (4 个 spec 抽进 registry + register_spec 已有), 验证 PhaseRegistry 设计的成本极低.

**安全边界**:
- phase spec patch 只能改 `prompt_template / tool_whitelist / postcondition`, 不能改 `name` 和 `fallback` (保证回退路径)
- **tool_whitelist 是 advisory 不强制** (P9 修正): phase 方法体内 9+ 处直接 `tool.call()` 不改, 只在 subagent dispatch 路径强制 (已有 `agent.tool_filter` 机制). 避免改 9 处 call site. 升级路径: 加 unified `dispatch_tool(name, args, ctx)` 入口
- action_extension 注册需过白名单 (允许的动作类型: `branch / parallel / retry`), 不允许任意 Python 代码执行 (除非用户显式开 `HUGINN_ALLOW_CODE_GEN=1`)
- PhaseRegistry baseline 是当前 hardcode 的镜像, 任何 patch 失败回退到 baseline
- PhaseRegistry 载入时做拓扑排序检查, 检测到 phase 调用关系成环回退到 baseline
- **P1 限制**: 只在 autoloop 路径做, rcb_runner 不动 (它没接 CognitiveLoop)
- ponytail: 不让 agent 直接改 engine.py 源码, 用 spec 覆盖 + fallback. 升级路径: 接 git diff 让 agent 改源码, 但需用户每次确认.

**跟 H1 的关系**: H1 改 prompt block body (字面量级), H4 改 phase 整体 spec (结构级, 含 prompt template + tool_whitelist + postcondition). H4 是 H1 的超集, 但 H1 是最小改, H4 是大改. 先做 H1 验证可行性, H4 分批做 (**从 BUILTIN_SPECS 试点, 比 _report 更易**).

## 前端接入

新增 "Harness Evolution" 面板 (跟 Memory Layers 面板同档, 复用 [MemoryPanel.tsx](file:///c:/Users/wanzh/Desktop/matsci-agent/desktop/src/components/panels/MemoryPanel.tsx) 的 layers 视图模式):

### Settings 新增 "Harness Evolution" 区块

**注意 (P 盲点修正)**: toggle **走 cfg 文件路径不走 env**. config.py 的 `get_config` 有 read-through cache + 磁盘 mtime 自动 reload (config.py:1262), env 变量在 `from_env()` 调用瞬间读, 运行时改 env 不生效需重启进程. 前端 toggle 写 cfg 文件让 `get_config` 自动 reload.

| 设置项 | cfg key | 类型 | 默认 | 说明 |
|---|---|---|---|---|
| Prompt patch 总开关 | `harness.prompt_patch` | toggle | off | 开启 H1 prompt template patch 闭环 |
| Workflow evolution 总开关 | `harness.workflow_evolution` | toggle | off | 开启 H2 workflow 变体搜索 |
| Joint optimization 总开关 | `harness.joint_opt` | toggle | off | 开启 H3 联合优化 (依赖 H1+H2) |
| Phase spec 演化 | `harness.phase_evolve` | toggle | off | 开启 H4 phase + subagent spec 演化 |
| Patch store 容量 | `harness.patch_store_max` | number | 20 | LRU 上限 |
| Workflow archive top-K | `harness.workflow_top_k` | number | 5 | Pareto 前沿保留数 |
| Bandit 探索常数 c | `harness.bandit_c` | number | 1.0 | Thompson sampling 探索项权重 |
| 允许代码生成 | `harness.allow_code_gen` | toggle | off | H4 action_extension 允许任意 Python (高风险) |

### Session/Trajectory 视图新增 "Harness Variants" 面板

显示当前 run 的 harness 变体状态:
- **Prompt patches**: 活跃 patch 数, top-5 by Beta mean, 最近应用时间
- **Workflow variants**: 归档库大小, Pareto 前沿 top-K, 最近 rollout 适应度
- **Joint bandit**: 当前选中的 (block subset, workflow params), Beta 信念分布
- **Phase registry**: 当前 7 phase 的 spec 版本, 是否被 patch 覆盖, 最近回退次数

### 开关 UX

- 默认隐藏在 "Advanced Settings" 折叠区 (跟极限模式一致)
- 开启 H3/H4 时弹确认: "Joint optimization / Phase evolution 会增加 LLM 调用和 rollout 次数, 确认开启?"
- H4 的 `HUGINN_ALLOW_CODE_GEN` 开启时二次确认: "允许 agent 生成任意 Python 代码, 风险极高, 确认?"
- 开启后状态栏显示 `EVOLVE` 标记
- 关闭时已生成的 patch/variant 保留, 不清除 (跟 PMK pattern 一致)

## 执行顺序 (第三轮调研修正)

按依赖关系 + 风险最小 + 杠杆最大:

1. **H0 stable_principles 接进 autoloop prompt** ✅ — 修 P3 断链, H1 最小前置. ~20 行. 独立可跑. (selfcheck 6/6, 13 principles loaded, 3 轮 autoloop 不崩)
2. **H1 Prompt Template Patch 闭环** ✅ (代码完成, 端到端 patch store 写入待 H2) — 最小改, 最直接验证"agent 改自己 prompt"可行性. ~150-210 行新文件 + 4 接入. 独立可跑. (selfcheck 5/5 + engine selfcheck 8/8; 3 轮 reasoning-only autoloop 不崩, 但 r_phys=None 未触发 generate_patch, patch store 0 patches — 端到端写入需 compute-heavy objective 走 validate phase, 留到 H2 workflow 测试)
3. **H2 Workflow Evolutionary Search** ✅ — 依赖 dynamic_workflow 已有基础. ~300-400 行新文件 (2 个) + 3-4 接入. 独立可跑. **改动量比 spec 原估翻倍**, 主要因为没有现成 bandit selector + ToolBelief key schema 不匹配 + 没有现成 variant 生成器. (bandit.py selfcheck 6/6 + variant_gen.py selfcheck 4/4 + engine selfcheck 12/12; 端到端冒烟 3 轮 bandit loop: archive 3 variants + Pareto 前沿更新 + Thompson sampling 信念分化 + _try_evolved_fix guard 生效; 修复 variant_id 跨轮冲突 bug, 加时间戳前缀)
4. **H3 Joint Optimization** ✅ — 依赖 H1 + H2. ~280 行新文件 + 2 接入. **P8 限制: 不包含 model 维度**. (joint_optimizer.py selfcheck 5/5 + engine selfcheck 13/13 + variant_gen selfcheck 5/5 含 H3 接入验证; H1 apply_patches 入口接 select_block_subset_for_phase + H2 _perturb_script 入口接 select_workflow_params_for_stage; UCB 冷启动=inf 优先探索 + Beta(α,β) 信念分化 + 核心 block 必保留 + 持久化 reload OK; H3↔H1 集成路径验证 OK; **端到端验证**: 8 轮真实 autoloop (KB disabled 绕过 chromadb crash) trace 显示 select_block_subset_for_phase 被调 4 次 (hypothesize+plan phase), 冷启动全选符合 UCB=inf 预期; 数学逻辑 3 轮模拟验证 Beta 分化 (A: α=2 β=0 mean=0.750 > B: α=0 β=1 mean=0.333) + UCB 分化 (A=0.954 > B=0.583) + 持久化 reload OK; **已知限制**: cognitive_loop LLM decide 在 validate 后选 pivot 不选 learn, 导致 _learn 没被调, joint_beliefs 0 files — 这是调度问题不是 H3 bug, 修复需调 decider prompt 或强制 learn phase, 留到 H4 phase 演化)
5. **H4 Phase + BUILTIN_SPECS 可演化** ✅ (BUILTIN_SPECS 试点完成) — 大改, 但**试点首选 BUILTIN_SPECS (~80 行) 验证 PhaseRegistry 设计**, 不必等 H1-H3. 试点后分批做 7 phase (~700 行总). (selfcheck 4/4 + engine selfcheck 8/8; 3 轮 autoloop toggle on 走 registry 合成路径, 4 subagent specs baseline 正常)

每个 H 完成后立即 selfcheck:
- H0: 跑 3 轮 autoloop, 验证 stable_principles 出现在 hypothesis/plan prompt, _learn 有 distill_episodic_to_procedural 调用
- H1: 跑 3 轮 autoloop, 验证 patch store 有写入, Beta 有更新 (via evolve_from_rewards), prompt block 有替换
- H2: 跑 1 个 objective, 验证 archive 有 ≥3 variant, Pareto 前沿有更新, _try_evolved_fix guard 生效
- H3: 跑 3 轮, 验证 bandit 选择有变化, Beta 信念有收敛趋势
- H4: 跑 3 轮, 验证 PhaseRegistry 有 patch 覆盖 (BUILTIN_SPECS 试点), 回退路径正常, VALID_ACTIONS 可扩

H 之间独立, 不互相阻塞 (除 H3 依赖 H1+H2, H4 试点 BUILTIN_SPECS 不依赖 H1). **建议 H4 BUILTIN_SPECS 试点跟 H1 并行做**, 两者验证不同抽象层 (H1 验证 prompt patch, H4 BUILTIN_SPECS 验证 PhaseRegistry dataclass).

## 不做的事 (YAGNI)

- ❌ 让 agent 直接改 [engine.py](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/autoloop/engine.py) 源码 — 风险过大, H4 的 spec 覆盖 + fallback 已能覆盖 80% 收益
- ❌ 实现完整 STOP 式递归改进器 ($I_t = I_{t-1}(I_{t-1})$) — 需要模型 capacity 足够高, 当前 GLM-5.2 不一定过 $M_c$ 阈值
- ❌ 实现 AFlow 式 MCTS workflow 搜索 — ADAS 式随机归档 + UCB 够用, MCTS 升级路径明确但暂不需要
- ❌ 实现 Meta-Harness 式 "harness for optimizing harnesses" — 三层递归, 当前一层都还没跑通
- ❌ 跑 benchmark 驱动的 prompt 调优 (如 DSPy) — 离线 benchmark 跟在线 autoloop 评估指标不一致, 闭环不干净
- ❌ 改 CognitiveLoop 4 钩子签名 — 已稳定, 在其内演化够用
- ❌ 改 LoopState / ActionDecision / ReflectionResult dataclass — 已稳定

## 与现有工作的共生关系

| 现有工作 | 本 spec 关系 |
|---|---|
| [cognitive_loop.py:84](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/autoloop/cognitive_loop.py#L84) CognitiveLoop 4 钩子 | H4 在钩子内部把 phase 方法体抽成 PhaseSpec, 钩子签名不动 |
| [cognitive_loop.py:65](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/autoloop/cognitive_loop.py#L65) VALID_ACTIONS frozenset | H4 改为 property, 返回 `base | registered_extensions` |
| [layered_memory_spec.md](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/docs/layered_memory_spec.md) PMK 4 层 | 本 spec 的 patch store 是 PM 层的扩展 (procedural memory 存 patch 而非存 pattern) |
| [engine.py:5915](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/autoloop/engine.py#L5915) RSI directive | H1 把 directive 从"文本 hint"升级到"可执行 patch" |
| [skills/evolution.py:164](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/skills/evolution.py#L164) SkillEvolutionLayer | H3 复用其 Beta 更新逻辑 |
| [dynamic_workflow.py:50](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/autoloop/dynamic_workflow.py#L50) WorkflowScript | H2 扩其为可演化 variant |
| [evolution/manager.py:40](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/evolution/manager.py#L40) EvolutionManager | H1/H2 的 patch/variant 失败时复用其 `record_failed_direction` |
| [routes/memory.py:164](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/routes/memory.py#L164) GET /memory/layers | 前端 Harness Variants 面板复用此端点模式 |
| [MemoryPanel.tsx](file:///c:/Users/wanzh/Desktop/matsci-agent/desktop/src/components/panels/MemoryPanel.tsx) Layers 视图 | Harness Variants 面板复用 LayerCard / LayerRow 组件 |
| [agents/subagent.py:115](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/agents/subagent.py#L115) BUILTIN_SPECS | **H4 纳入 scope (P10 修正)**, 作为 PhaseSpec 试点首选 (比 _report 更易), 已有 register_spec() 实例级注册机制 |
| [engine.py:460](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/autoloop/engine.py#L460) `_get_evolution` 懒加载单例 | H1/H2/H4 的 store (PromptPatchStore / WorkflowArchive / PhaseRegistry) 都用同模式, 保证跨 iter 状态持久 |
| [config.py:1262](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/config.py#L1262) `get_config` read-through cache | H1-H4 的 toggle 走 cfg 文件路径 (mtime 自动 reload), 不走 env (env 改了需重启进程) |
| [engine.py:7054](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/autoloop/engine.py#L7054) `_llm_chat` | H3 不包含 model 维度 (P8 限制), `_llm_chat` 只覆盖 autoloop engine, subagent._summarize 和 hot_model 双路都绕过. model 选择留作 H5 候选 |
| [engine.py:9 处 tool.call()](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/autoloop/engine.py#L3706) 直接 tool 调用 | H4 tool_whitelist 是 advisory 不强制 (P9), 只在 subagent dispatch 路径强制. 升级路径: unified `dispatch_tool` 入口 |

## 失败模式 (诚实声明)

1. **LLM 生成的 patch 可能垃圾**: JSON parse 错误 / block 名不存在 / new_text 跟原 block 语义冲突. 处理: 静默丢弃 + $\beta += 1$, 不阻塞主循环.
2. **Workflow variant 可能跑炸**: timeout / OOM / 工具失败. 处理: 复用 [subagent_tool.py](file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/tools/subagent_tool.py) 的 timeout 机制, 失败记 $\beta += 1$.
3. **Bandit 可能陷入局部最优**: Thompson sampling 探索常数 $c$ 太小, 一直选同一个 variant. 处理: 默认 $c=1.0$, 可调; novelty 项会拉回探索.
4. **Pareto 前沿可能退化**: 所有 variant 都被一个 dominate, 前沿缩到 1 个. 处理: 强制保留 novelty 最高的 1 个 (即使被 dominate), 保证多样性.
5. **H3 联合优化可能不收敛**: 搜索空间太大, Beta 信念更新慢. 处理: 先跑 H1+H2 验证单独收敛, H3 默认 off.
6. **可能跟极限模式冲突**: 极限模式开 DAG subagent + trajectory match, H2 也跑多 variant. 处理: 极限模式 + H2 同时开时, variant 跑在 subagent 里, 主循环不阻塞.
7. **H4 phase spec 可能引入死循环**: patch 后的 phase 调用关系成环. 处理: PhaseRegistry 载入时做拓扑排序检查, 检测到环回退到 baseline.
8. **H4 action_extension 可能被滥用**: LLM 注册恶意 action (如 `rm -rf`). 处理: 白名单 (branch/parallel/retry) + `harness.allow_code_gen` 双确认.
9. **H1 patch 替换 body 块丢 `{context}` 占位符** (第三轮新增): patch 替换 body 块时如果新文本没有 `{context}`, hypothesis prompt 丢失上下文. 处理: BlockPatch.op=replace 时强制检查 body 块的 `{context}` 占位符存在, 不存在则降级为 prepend.
10. **H2 _try_evolved_fix leak** (第三轮新增, P6): variant 失败可能触发 `_try_evolved_fix` 走 `_execute_workflow`, bandit 收不到 outcome. 处理: `_try_evolved_fix` 入口加 guard `if execution_result.get("_variant_id"): return None`.
11. **H4 tool_whitelist advisory 失效** (第三轮新增, P9): phase 方法体 9+ 处直接 `tool.call()` 不走 whitelist 检查, agent 改 tool_whitelist 但实际不生效. 处理: 文档明确 advisory 不强制, 只在 subagent dispatch 路径强制; 升级路径加 unified dispatch_tool.
12. **H3 model 维度缺位** (第三轮新增, P8): H3 不能联合优化 model 选择, 因为 LLM client 不 unified. 处理: H3 明确不包含 model 维度, 留作 H5 候选.

## 终止条件

本 spec 不要求一次性完成所有 H. 优先级:

1. **H0 优先** (修 P3 断链, ~20 行, H1 最小前置)
2. **H1 次之** (最小改, 最直接验证"agent 改自己 prompt"可行性, ~150-210 行 + 4 接入)
3. **H4 BUILTIN_SPECS 试点并行** (~80 行, 验证 PhaseRegistry dataclass 设计, 不依赖 H1)
4. **H2 然后** (~300-400 行 + 3-4 接入, 改动量比原估翻倍)
5. **H3 接着** (依赖 H1+H2, ~140 行 + 2 接入, 不含 model 维度)
6. **H4 剩余 phase 分批** (从 _report → _validate → _execute 顺序, ~700 行总)

每个 H 完成后:
- selfcheck (stable_principles 注入 / patch store 写入 / archive 更新 / bandit 选择变化 / PhaseRegistry 覆盖)
- 跑 3 轮 autoloop 验证不崩
- commit + push
- 更新本 spec 的 H 状态

## 升级路径 (H5 候选, 不在本 spec 范围)

第三轮调研发现的两个隐藏天花板, H1-H4 不修, 留作 H5 候选:

1. **H5-a unified LLM client** (P8): 让 H3 联合优化能加 model 维度. 改 `_llm_chat` 覆盖 subagent._summarize 和 hot_model 双路, 加 `select_model(alias_or_constraint)` 接口让 LLM 自己输出 "用 alias X 跑这次". 工程量大, 风险高.
2. **H5-b unified tool dispatch** (P9): 让 H4 tool_whitelist 真正强制. 新建 `dispatch_tool(name, args, ctx)` 入口, 改 engine.py 9+ 处直接 `tool.call()` 调用. 工程量中, 风险中.

只有当 H3 跑通后需要 model 维度, 或 H4 跑通后 tool_whitelist advisory 不够用时, 才启动 H5.

## 数学动机补充 (不做证明, 只标结构)

本 spec 不做严格数学证明, 但标出数学结构以便后续论文化:

- **H1 prompt patch** = 文章里 STOP (Zelikman et al. 2023) 的最小工程实现. STOP 的 $I_t = I_{t-1}(\hat u, I_{t-1}; M)$ 对应我们的 `generate_patch(phase, blocks, r_phys, directive)` — 改进器改进自己. 但我们不做递归 (不递归改 patch 生成器), 只做单层.
- **H2 workflow search** = 文章里 ADAS (Hu et al. 2025) 的简化版. ADAS 的"元代理编程新 agent"对应我们的 `generate_workflow_variants`. 我们不做代码生成 (风险大), 只做 `WorkflowScript` 参数搜索.
- **H3 joint optimization** = 文章里 MCE (Ye et al. 2026) 双层优化的退化版. MCE 的 $\text{Inner: } c^*=\arg\max_c J_{\text{train}}(c;s)$, $\text{Outer: } s^*=\arg\max_s J_{\text{val}}(c^*)$ 对应我们的"内层跑任务拿 r_phys, 外层 bandit 选 (block, stage) 组合". 我们不做 train/val split (单任务场景没意义), 用 r_phys 直接做适应度.
- **H4 phase 演化** = 文章里 "harness code as optimization target" 的工程化. 我们不让 agent 改源码 (风险), 用 spec 覆盖 + fallback 实现"行为体可演化但不失控".

**跟 4 层 memory 的关系**: H1 的 patch store 是 PM 层的扩展 — PM 现在存 `stable_principles` (文本规则), H1 让 PM 也存 `prompt_patches` (可执行规则). 数学上, stable_principles 是"声明式不变量", prompt_patches 是"命令式变换", 两者都是 RSI 不动点的近似.

**跟 CognitiveLoop 的关系**: H4 是 CognitiveLoop 抽象的延伸 — CognitiveLoop 抽控制流, H4 抽行为体. 控制流 + 行为体都可演化后, harness 才真正成为优化目标.

## 参考文献

- Weng, L. (2026). Harness Engineering for Self-Improvement. Lil'Log.
- Zelikman et al. (2023). STOP: Self-Taught Optimizer. arXiv:2310.02304.
- Hu et al. (2025). ADAS: Automated Design of Agentic Systems. arXiv:2408.08435.
- Zhang et al. (2025). AFlow. arXiv:2410.10762.
- Ye et al. (2026). MCE: Meta Context Engineering. arXiv:2601.21557.
- Lee et al. (2026). Meta-Harness. arXiv:2603.28052.
- Zhang et al. (2025). ACE: Agentic Context Engineering. arXiv:2510.04618.
