# Orchestrator Unification Lessons — 4-Layer Capability Leap

Source: capability-leap-orchestrator spec implementation. 4 layers打通
(通用 Orchestrator + PhaseManager↔Budget + 预置基础设施 + Subagent 并行),
目标从 ~7 分冲到 80 分. 每条教训至少 cost 一次 failed run 或 debug session.

## 1. 架构决策: 为什么 4 层而不是 1 层

| Layer | 作用 | 不做的代价 |
|-------|------|-----------|
| 0 通用 Orchestrator | 抽象 paperbench 的 while+三档分流, 5 适配器共用 | 每个适配器各自维护 while 循环, bug 修 5 次 |
| 1 PhaseManager↔Budget | phase 转移自动设 proposed_budget, chat() 接 budget_override | phase 和 budget 脱节, OPEN phase 500 calls 覆盖一切 |
| 2 预置基础设施 | bench_infra/ 下 5 个工具, agent 不从零写训练循环 | agent 每次花 5-10 calls 写 matplotlib 样板 / sklearn classifier |
| 3 Subagent 并行 | EXECUTION phase 派生 coder/analyst, 隔离 context | 主 agent context 膨胀, 150 calls 不够用 |

关键: Layer 0 是地基, Layer 1 是通道, Layer 2 是杠杆, Layer 3 是乘数.
跳过任何一层, 其他层的 ROI 都会打折.

## 2. Phase-aware Budget 最优配比

实测配比 (总 530 calls, timeout 14400s):

```
LITERATURE:  50 calls, recursion 300   # 读论文 + rubric, 50 够用
HYPOTHESIS:  30 calls, recursion 200   # 很少用, 大部分任务跳过
PLANNING:    30 calls, recursion 200   # 写 skeleton + smoke test
EXECUTION:  300 calls, recursion 1600  # 主战场, 训练 + 实验
VALIDATION: 100 calls, recursion 550   # C2ST + 画图 + 分析
REPORTING:   20 calls, recursion 150   # reproduce.sh + README
OPEN:       500 calls, recursion 2600  # fallback
```

### 教训

1. **EXECUTION 必须占 50%+**: 第一次给 EXECUTION 100 calls, agent 训练完
   linear_gaussian 就用完了. 300 calls 才够跑 4 tasks × 3 methods 矩阵.
2. **LITERATURE 上限 50**: agent 习惯反复读论文, 50 calls 是 "读一遍 +
   查 rubric" 的实际成本. 超过 50 = 兔子洞.
3. **recursion_limit = max(250, calls × 5)**: max_tool_calls=5 时 recursion
   不够, agent 第一轮 LLM 响应就 GraphRecursionError. 公式: 每个工具调用
   消耗 ~5 个 recursion node (LLM → tool → LLM → tool → LLM).
4. **chat mode 不接 budget_override**: chat mode 的 max_tool_calls 由适配器
   设定, budget_override 只在 research mode 时传. 否则 OPEN phase 的 500
   calls 会覆盖适配器设的 max_total_calls.

## 3. BenchmarkOrchestrator 三档分流

```python
# 无 tool_call 时, 检查 deliverables:
#   全齐 → CONTINUE_MSG (agent 主动结束)
#   有代码无 output → _execution_prompt (催训练)
#   缺文件 → _triage_prompt (催写文件)
```

### 教训

1. **deliverable_spec 要参数化**: 每个 benchmark 的 deliverable 不同.
   paperbench: reproduce.sh + *.py + outputs/*.json.
   mle: submission.csv + *.py. hle: 空 (退化为单次 chat).
   用 DeliverableSpec dataclass + glob 检查, 不要 if/else 硬编码.
2. **HLE 空 spec = Orchestrator 退化**: HLE 无 deliverable 检查,
   Orchestrator 的 while 循环第一次无 tool_call 就结束. 这是 feature 不是 bug.
3. **_execution_prompt 比 _triage_prompt 更重要**: agent 最常见的失败模式
   是 "写了代码但没跑训练". _execution_prompt 直接催 "run train.py now",
   比催 "write reproduce.sh" 救分更多.

## 4. bench_infra 工具使用经验

| 工具 | 省的 calls | 注意事项 |
|------|-----------|---------|
| plot_tool | 3-5 | Arial 20pt+ 加粗, agent 不再写 matplotlib 样板 |
| training_matrix_tool | 10-15 | 4 tasks × 3 methods 一次跑完, 但 torch MLP 精度有限 |
| c2st_evaluator_tool | 5-8 | sklearn RandomForest, same=0.4985, diff=0.9997 |
| mcmc_sampler_tool | 8-12 | ABC rejection + MH, 无 sbi 依赖, ABC acceptance_rate ~0.001 |
| kaggle_submit_tool | 3-5 | 自动校验行数/列名/缺失值, regression+classification |

### 教训

1. **numpy Generator API**: `rng.randn(n, 4)` 不存在, 用 `rng.standard_normal((n, 4))`.
   `rng.rand()` 也不存在, 用 `rng.random()`. `rng.randint(0, 4, size=n)` 也不存在,
   用 `rng.integers(0, 4, size=n)`. 四次踩坑 (randn/rand/randint 在 matrix_tool + mcmc_tool).
   规则: numpy Generator (np.random.default_rng) 只用 standard_normal/random/uniform/integers,
   其他 legacy RandomState 方法都不存在. 写完一定要跑 `python tool.py` self-check.
2. **ToolResult 构造签名**: `ToolResult(data=..., success=..., error=...)`,
   不是 `ToolResult(content=...)`. content 不是合法 kwarg. base.py 里看签名.
3. **sbi 不可装**: Windows + Python 3.13, sbi 依赖 torch-geometric 依赖到死.
   mcmc_sampler_tool 用 numpy 实现 ABC + MH, 精度降级但能跑.
4. **training_matrix_tool 的 MLP 太简单**: 单层 64-unit MLP, NPE/NRE/NLE
   只是 loss head 不同. 不追求精度, 只给 agent 训练证据 + loss curve.
   真正的 SBI 精度需要 agent 自己写 Simformer.

## 5. Subagent 并行最佳实践

```python
# coder spec: max_tool_calls 10→50, max_iterations 5→10
# allowed_tools: +plot_tool, +training_matrix_tool
# analyst spec: max_tool_calls 8→20, max_iterations 3→5
# allowed_tools: +c2st_evaluator_tool, +mcmc_sampler_tool, +plot_tool
```

### 教训

1. **coder 预算必须 ≥ 50**: benchmark 需要写 model.py + data.py + train.py +
   debug + 重跑. 10 calls 只够写一个文件. 50 calls 够写+训+调一轮.
2. **不强制派生**: system prompt 引导 "For complex tasks, dispatch coder",
   但主 agent 自主决策. 强制派生会导致 subagent context 和主 agent 脱节.
3. **主 agent 保留验证责任**: subagent 返回 summary, 主 agent 必须检查
   subagent 写的文件是否存在 + 能否 import. 不验证 = 幻觉级联.

## 6. system prompt 重构: 去 hardcode

### 改动
- 移除 "Phase 1 (calls 1-20)" → "LITERATURE phase"
- 移除 "call 50 check loss.json" → "before VALIDATION phase"
- 加 SUBAGENT STRATEGY 章节 (coder/explore/analyst + 使用场景)
- 加 BENCH_INFRA TOOLS 章节 (5 个工具 + "don't reinvent" 引导)

### 教训
1. **phase 名比 call 编号更鲁棒**: call 编号随 budget 变化, phase 名不变.
   "before VALIDATION phase" 在 150 calls 和 530 calls 下都对.
2. **工具引导要具体**: "Use plot_tool for figures" 不如 "Use plot_tool for
   loss curves, Arial 20pt+ bold, don't write matplotlib boilerplate".
3. **anti-pattern 段落留**: "TESTING IS NOT TRAINING" 段落救分最多,
   agent 最常见的失败是把 smoke test 当 execution.

## 7. 具体数值

| 指标 | M1 | M2 | M3 | M4 | M5 | M6 | M7 | 目标 |
|------|-----|-----|-----|-----|-----|-----|-----|------|
| Score | 13.07 | 16.83 | 11.56 | 9.55 | 7.54 | 12.81 | 12.36 | 25 |
| Code files | 9 | 7 | 5 | 15 | 8 | 9 | 6 | — |
| Tool calls | ~150 | 131 | ~80 | ~129 | ~120 | ~77 | ~150 | — |
| 崩溃? | 否 | 否 | 是(rng) | 否(budget) | 是(c2st) | 是(c2st) | 否 | — |
| C2ST 77-79 | 0 | 0 | 100 | 100 | 100 | 0 | 50/30/50 | — |
| Simformer 1-30 | ~60% | ~60% | ~40% | ~30% | ~20% | ~50% | ~40% | — |

### M7 分数分布 (12.36/100, c2st 第四次修复 + prompt 精简后)
- C2ST leaves 77-79: 50/30/50 ✅ (c2st_tool 修复, 不再崩溃, 重算 overall=0.9717)
  - c2st 高说明 Simformer posterior 质量 vs reference 差, 但 tool 能算出数了
- VESDE leaves 1-6: 全 100 ✅ (稳定)
- leaf 14 (Simformer): 50
- leaves 80-88 (Simformer config): leaf 80=30, 81-86=0, 87-88=100
  - 比 M6 退步 (M6 大部分 100), 因 code files 6 < 9, tokenizer/Gaussian Fourier 细节被省略
- leaf 106 (LG samples): 50 (M6=0, 恢复)
- leaves 137-138 (TM/SLCP samples): 50/50 (新得分)
- 6 code files (比 M6 少 3, 精简 prompt 让 agent 写得更少)

### c2st_tool 四次踩坑 (M5/M6/M7) — 彻底修复
1. M5: agent 传 `{"samples": [...]}` → 一层 dict, `.get("samples")` 修复
2. M6: agent 传深度嵌套 dict (如 `{"samples": {"theta": [...]}}`) →
   `.get("samples")` 返回的还是 dict → 崩溃
3. M6 后修复: 递归 dict 遍历 `max(seen, key=len)`, self-check 通过
4. M7: `max(seen, key=len)` 选到含 dict 元素的 list (如 `[{"task":...,"samples":[...]}]`),
   `np.array(dtype=float)` 在 dict 元素上崩 → 第四次修复: 只收纯数值 list 或纯数值 2D list,
   含 dict/str 的 list 继续往里挖. self-check 覆盖此场景, 通过.
**根因**: agent 传 sample 格式五花八门, 任何"取最长 list"的启发式都会被新格式打败.
修复后过滤逻辑: list 元素必须全为数值, 或全为数值 list. 其余 list 继续递归.

### prompt 精简 (M7 前)
M6 后删掉 3 个占空间但不救分的段落:
- LIPSCHITZ PROTOCOL (~20 行, 太抽象)
- NOISE AS FEATURE (~7 行, 太抽象)
- SAMPLE() HANG TRAP 从 8 行精简到 1 行
保留: EVAL NaN TRAP (救 25 calls), PHASED PROTOCOL, ANTI-PATTERN.

### 分数瓶颈分析
1. **sbi 不可用** (leaf 35, w=3.0): rubric 要求用 sbi, 但 sbi 被 blacklist.
2. **MCMC reference samples** (leaves 41-59): mcmc_sampler_tool 能生成, 但 rubric 要求 slice+MH.
3. **Section 4 实验全 0** (leaves 90-174): 需要 36 次训练.
4. **system prompt 过度工程化**: 每次加 TRAP/引导反而让 agent 分散注意力, Simformer 质量下降.
5. **scorer .json bug** (M5): .json 算 code_files 导致 32 files, 关键 .py 被截掉. 已修复.
6. **c2st_tool 嵌套 dict bug** (M5/M6/M7): 四次踩坑, 第四次修复后 self-check 覆盖.
7. **Simformer.sample() hang** (M4): 200 steps × 100 samples CPU 太慢. prompt 精简到 1 行.
8. **prompt 精简过头** (M7): 删 LIPSCHITZ/NOISE 让 code files 6 < M6 的 9, config leaves 退步.

### 关键教训: system prompt 不是越长越好, 也不是越短越好
M2 (16.83) 用简单 prompt 最好. M3-M6 加 TRAP/引导分数反而下降 (11.56/9.55/7.54/12.81).
M7 精简 prompt (12.36) 也没超过 M2 — 删抽象段落让 agent 写的 code files 变少 (6 < 9),
Simformer config leaves 退步. 但 M7 的 c2st 修复是实质性的 (+1.3 vs M6 的 0/0/0).
agent 上下文有限, 过多引导挤占核心实现; 但太少引导让 agent 漏掉关键 config 细节.
**最优策略**: M2 的 prompt 水平 + 修复后的 c2st_tool, 不加 TRAP 也不删 config 提示.

## 8. 下一步

- M7 评分完成 (12.36), c2st_tool 第四次修复 self-check 通过
- M2 (16.83) 仍是 best. 精简 prompt 没超过 M2, 但 c2st 修复是基础设施改进.

### Task 19: stochastic-interpolants 结果 (33.85/100)
- 历史 18.23 → Task 19: 33.85 (+85% 提升, 未达 50 目标)
- 150 calls + 3600s, 6 code files
- **得分点**: Uncoupled Interpolant leaves 18-26 大量 100 (历史全 0)
- **失分点**:
  - leaf 44 (w=5.0): lucidrain U-Net = 0 (agent 用自定义 U-Net, 没用 lucidrain repo)
  - leaves 45-52 (w=8.0): U-Net config params 全 0 (dim=256, dim_mults, learned_sinusoidal_cond 等)
  - leaf 55: gradient clipping = 0 (一行代码就能修)
  - leaf 56: Dopri solver = 0 (需 torchdiffeq)
  - leaves 58-69: ImageNet training + FID-50k 全 0 (需 GPU + 数小时, CPU 不现实)
- **教训**: prompt 没提示 U-Net config 重要性, agent 用自定义 U-Net 而非 lucidrain.
  SI 类任务的 config leaves (w=13) 可通过 prompt 提示快速修复, 不需要训练.
- **结论**: c2st 修复 + 当前 prompt 对 SI 有效 (+85%), 但 U-Net config 是瓶颈.
  all-in-one 是主战场, 先冲 Task 20.

### Task 20: all-in-one 80 分冲刺 (530 calls, 14400s)
- 用修复后 c2st_tool + 当前 prompt
- M2 (16.83) 是 baseline, 530 calls 是 3.5x budget
- 目标: 80 分; Code Exec ≥ 50%; Result Analysis ≥ 40%

### Task 20 v1 失败 (15.33/100, agent 提前退出)
- **现象**: 530 calls + 14400s budget, agent 只跑 600s (10 min) 就停, 分数 15.33
- **Root cause**: orchestrator `_is_done` 在 deliverable 全齐 (有 .py + reproduce.sh +
  outputs/*.json) 时返回 True, 不管 tool_count. agent 写完 smoke test (7 files + 16 tiny
  .pt + dummy outputs/*.json) 就触发 deliverable 全齐, while 退出.
- **修复**: `_is_done` 加 min_calls 下限 = 50% budget. deliverable 全齐 + calls >=
  max_total_calls * 0.5 才算完成. 强制 agent 至少用 265 calls.
- **教训**: deliverable 检查太宽松 = agent 写 smoke test 就停. min_calls 下限是必要的.
  530 calls budget 没用完 = 浪费. orchestrator self-check 通过.
- **v5 R17 撤销**: min_calls 补丁是 SCALECUA task_synthesizer 上线前的临时占位.
  v5 用 task_synthesizer 按 (paper, complexity_tier) 在合成时给定难度, 不再用
  budget 下限强迫消耗. `_is_done` 回到 "deliverable 全齐即完成".

### Task 20 v2 (修复后 orchestrator, 530 calls, 14400s)
- 修复后 _is_done: calls >= 265 + deliverable 全齐 才退出
- agent 被迫用更多 calls 做 Section 4 实验 / 改进训练质量
- **结果**: 13.52/100, 7 code files (同 v1)
- **现象**: agent 在 tool #65 产出 "Reproduction Complete" summary (无 tool_call),
  orchestrator 注入 CONTINUE_MSG, 但 agent 20+ 分钟无新 tool call, 被 Stop 终止
- **根因 (未完全诊断)**: 可能 LLM 对 CONTINUE_MSG 响应异常, 或 agent context 已满.
  CONTINUE_MSG 是通用 "proceed to next step", 对已完成 summary 的 agent 无具体指引,
  可能被 LLM 解读为 "已无事可做" 而不再产出 tool_call
- **分数分布**:
  - VESDE leaves 1-6: 全 100 ✅ (稳定)
  - C2ST leaves 77-79: 50/100/100 ✅ (c2st_tool 修复生效)
  - leaf 33 (Algorithm 1): 30 (w=3.0)
  - Section 4 leaves 90-174: 全 0 (需大量训练)
  - 7 code files (比 v1 相同)

### Task 20 总结: 530 calls 没救分
| Run | Score | 关键 |
|-----|-------|------|
| M1 | 13.07 | 旧 prompt |
| M2 | **16.83** | 简单 prompt (best) |
| M3 | 11.56 | rng 崩溃 |
| M4 | 9.55 | sample hang |
| M5 | 7.54 | c2st dict 崩溃 |
| M6 | 12.81 | c2st 嵌套 dict 崩溃 |
| M7 | 12.36 | c2st 修复 +1.3, code files 6<9 |
| T19 SI | 33.85 | +85% vs 历史 18.23 |
| T20 v1 | 15.33 | agent 提前退出 (orchestrator bug) |
| T20 v2 | 13.52 | agent 卡在 tool #65 (CONTINUE_MSG 死循环) |

**关键结论**:
1. **budget 不是瓶颈**: M2 (16.83) 用 ~131 calls, T20 v1/v2 用 530 calls budget
   都没超过 M2. 增加 budget 反而让 agent 进入低 ROI 区域 (Section 4 全 0 训练).
2. **Simformer 实现质量才是瓶颈**: code files 6-9 个, tokenizer/attention mask/
   Gaussian Fourier 等关键 config leaves 反复在 0/50/100 摆动, 不稳定.
3. **CONTINUE_MSG 对完成态 agent 无效**: agent 产出 summary 后, 通用 CONTINUE_MSG
   缺乏具体指引, LLM 不再产出 tool_call. 需要 phase-aware 的 next-step 指令
   (如 "Section 4 experiments need 36 training runs, start with linear_gaussian").
4. **80 分目标未达**: all-in-one 80 分需要 Section 4 实验大量训练 (leaves 90-174),
   CPU 不现实. 真正瓶颈是算力 + Simformer 实现, 不是 agent 架构.

### Spec 收尾决策
- M2 (16.83) 仍是 best, T20 v1/v2 都没超过 — 不再跑 v3
- spec 4 层架构 (Orchestrator + Budget + bench_infra + Subagent) 已全部落地
- c2st_tool 四次修复 + self-check 覆盖, 基础设施稳定
- Task 19 (SI 33.85, +85%) 验证架构对 SI 类任务有效
- 80 分目标受算力限制, 转向其他 benchmark 优化 (RCBench/MLE/SAB/HLE)
