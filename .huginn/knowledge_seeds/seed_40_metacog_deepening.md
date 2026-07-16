# Seed 40: 元认知架构深化 + mode/phase aware prompt + context 分层外置

## 核心教训
- L1 元认知层加 SignalHub 子组件统一路由信号（不加新层，避免过度抽象）
- build_prompt(mode, phase, metacog_state) 三元 aware 构造器取代散落 PHASE_PROMPTS/mode/STABLE_PRINCIPLES
- context 分层：核心段（常驻）+ 按需段（mode/phase/metacog 触发）+ 外置段（agent 主动 recall）
- 打通 10 断层 F11-F20: skills↔memory, perception↔CSM, tools↔metacog, fusion↔CSM, provenance↔metacog, autoloop↔memory, model↔CSM, bench↔memory
- 删死代码: mark_seed_reflected (F12) + forest_orchestrator (F17)
- benchmark/ → self_improvement/ 改名消歧

## 关键决策
- SignalHub 不加 L1.5 层（ponytail: 信号路由是 L1 内部职责，单独加层只会让 audit 链路多一跳）
- context 外置复用 memory.longterm + RAG（不新建 context_store，多一个 store 多一份同步债）
- F12/F17 死代码删除而非实现（ponytail: 删比实现便宜，留着只会让后人误以为要补）
- build_prompt 保留 fallback（ponytail: 委托但不一刀切，mode/phase 缺失时退化到原 PHASE_PROMPTS 行为）

## 断层清单
- F11: skills↔memory (G9) — skill_tool recall/remember skill_invocation
- F12: mark_seed_reflected 死代码 (R10) — 删除
- F13: provenance↔metacog (G13) — audit 加 provenance_snapshot 参数
- F14: perception↔CSM (G10) — _perceive 发 SignalHub 信号
- F15: tools↔metacog (G11) — FailureModeRegistry observed_counts
- F16: /fusion /team↔CSM (G12) — 走 set_mode("research")
- F17: forest_orchestrator 死代码 (R11) — 删除
- F18: autoloop↔memory (G14) — _learn 落 autoloop_summary
- F19: model↔CSM (G15) — STATE_TO_MODEL_TASK 映射
- F20: bench↔memory (G16) — BenchmarkRunner memory_manager 参数

## ponytail 简化
- tools_segment 占位符（不做复杂过滤，调用方自己决定信不信任工具列表）
- persona/safety 最小核心段（不把整套 persona 抄进 prompt，留给 mode 段拼接）
- observed_counts 无 LRU 上限（ponytail: 失败模式种类有限，O(1) 查表，真涨爆了再加 LRU）
- memory 调用全 try/except 失败静默（memory 挂了不该让 agent 整体崩，audit 里留痕就够）

## 升级路径
- v5: L2 critique 扩展到 mode/phase 决策（现在 L2 只看 block 级，决策级留给 v5）
- v5: agent 主动 recall 外置段的策略学习（当前靠人写触发条件，v5 让 agent 学什么时候 recall）
- v5: multi-agent metacog（单 agent 的 SignalHub 已经够绕，多 agent 留给 v5）
