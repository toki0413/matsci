# 长程科研闭环 (autoloop) 调试与验证 — 经验教训

> 环境提示: `/data/user/skills/` 是易失的(会被重置), 经验必须落进 git 仓库才持久。

## 一句话原则

**机器跑通 ≠ 科研产出。** 分三层, 工程修复只解决第一层:

| 层 | 内容 | 现状 |
|---|---|---|
| 通电(正确性) | control_trace / 输出提取 / 消息归一化 / 预算档位 | 已通 |
| 验证(自洽性) | 假设图多维评分 + 观察者(证伪, 做减法) | 已通 |
| **创新(新且重要)** | novelty + 显著性 + 决定性结论 | **缺** |

## 观测面
- 新增控制面机制**必须**配 `control_trace(name, iteration, evidence, action)`。没有 trace 就无法区分"未运行 / 运行失败 / 运行成功"。
- 结果读三处: `run.log`(control_trace) + `.huginn/hypothesis_graph_*.json`(节点/边) + `.huginn/engine_state/*.json`(评分/状态)。

## 长程 run 流程 (慢 endpoint)
1. 先跑目标单测 (`PYTHONPATH=/workspace/agent python -m pytest ...`; coverage 门会因只跑子集失败, 看 `N passed` 即可)。
2. 用 Grep 确认 env 名与档位 label(`open/medium/light`)存在, 再启动。
3. 复制 `_scratch/runNN.sh`; 目录 `research_outputs/shusheng_rsi_runNN/`; API key 在脚本里, 不在环境变量。
4. **减负**: 只保留本次验证相关的控制面, 关掉树搜索/Gramian/重放/通信层, 缩短单轮 (`HUGINN_MAX_TOOL_CALLS=4` / `HUGINN_CODELAB_TIMEOUT_S=60` / `HUGINN_FALLBACK_STREAM_IDLE=30`)。
5. 后台跑, 周期性 `wc -l run.log` + grep marker。

## 错题本 (Pitfalls)
- **子智能体只取末条消息** → 工具预算耗尽时末条为空 → `summary_len=0` 静默空转。必须取最后一条**非空**消息。
- **internlm 对 SystemMessage 数量/位置敏感** → 多条或错位触发 `in prompt processing error`。在模型边界合并为单一条置于消息流首。
- **假设图三维钉死** → darwin 恒定 2.50。四件套缺一不可: ①存 `testable_prediction` ②回写 `node.status` ③建派生边 `derive` ④并入 `task_perf`。
- **把观察者(blind_reconstruct)当 reward** → 自指。它是**差分传感器**(只看与执行判据的分歧, 调制探索强度), 不并入 darwin。
- **缺省判断用 `or 0.5`** → 0.0 是合法观测会被吞。用 `x is None`。
- **门控解析失败** → 必须 fail-closed(关), 不赌预算。
- **默认关闭的机制**: 不设 `HUGINN_ENABLE_AGENT_COLLAB=1` / `HUGINN_BLIND_RECONSTRUCTION=auto` 就是零行为变化。
- **报告 ≠ 结果 (最隐蔽的坑)**: 闭环能产出格式完整、带表格和结论段的 report, 但数值互相矛盾、rigid 与 fat 协议给出**相同**结果(探针根本不判别)、报告自己承认收敛失败、surprise≈0(无新信息)、甚至漂移到被命题**明令禁止**的领域。它"看起来完成", 实则既无创新点也未解决问题。
- **目标函数奖励"结构良好"而非"新"**: `graph_diversity` 用字符串唯一性, 重述能骗过; `supported/testable` 是自洽性; 观察者是证伪(做减法), **不生成**。必须显式并入 novelty(与已有低重叠=高新颖), 否则棘轮只优化"整洁度"。
- **novelty 等权平均 = 奖励"新而无用"**: 把 novelty 当第 N 维和结构分平均, 重述/换词也能抬分。创新必须**与问题进展挂钩**(门控/乘子), 否则棘轮奖励"新"本身, 而非"解决了问题的新"。

## 实测证据 (run78 / run79, 同一命题)
- run78: darwin 恒定 **2.50**(三维钉死), 观察者关。
- run79: darwin best **5.88**; 观察者 `control_trace` 覆盖 refute/support/skip, `summary_len` 960–1573。
- run79 报告: 13+ 实验全部 `reproducible=true`, **但** [ev4] rigid 与 fat 结果完全相同(探针不判别)、N_c 出现 `[-1,-1,4]` / `[Infinity,4,4,4]` 等矛盾值、`surprise score 0.07`、结论"partial support", 结尾漂移到 materials science(违反命题禁令)。

## 已补 (创新 / 解决问题层) — 一律"信号 + 诚实标注", 不做硬控制流
1. **判别性门槛**: 报告生成期加 `DISCRIMINATION RULE`; 事后 `_discrimination_gap` 审 Results —— 有足量数值却无对照/分离描述 → 落 `report_discrimination` trace (annotate) + 附 `Discrimination Audit` 告警。只标注, 不改结论、不终止。
2. **决定性闭环**: 报告生成期加 `DECISIVE CLOSURE RULE`; 事后 `_decisive_gap` 审全文 —— 有模糊措辞 (partial support / preliminary / suggests) 却无二元判定 → 落 `report_decisive` trace (annotate) + 附 `Decisive Closure Audit` 告警。只标注。
3. **novelty 并入评分 (进展门控)**: 开关 `HUGINN_NOVELTY_EVAL=1` 触发评估, novelty 落节点; darwin 里 novelty **不独立加分** —— 只在同时有真实进展 (`_last_task_perf`) 时按进展幅度计入 (`novelty × task_perf`)。"新而无用"不进棘轮, 只作探索整形信号。