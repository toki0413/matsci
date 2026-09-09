---
name: long-horizon-research-loop
description: 运行/续跑「书生长程自主科研循环」(huginn×书生), 让书生自主观察-推理-行动-成文-迭代不打断。
  触发: 要启动或继续多轮自主科研、让书生自己决策推进、长周期任务时调用。
---

# 书生长程自主科研循环(Long-Horizon Research Loop)

## When to Use
- 要"让书生自己观察、推理、行动, 智能迭代, 不要做一点停一点";
- 追长程科研任务(数十到数百轮连续自主推进);
- 用户明确"全流程让书生来干 / 让书生决策"时。

## 主入口
`/workspace/examples/shusheng_huginn_workflow.py`

## 启动命令(续跑一批)
```bash
INTERNLM_API_KEY=<key> python examples/shusheng_huginn_workflow.py \
  --start-cycle <N+1> --cycles <批数> --max-iters 12 --min-iters 4
```
- `--start-cycle N+1`: 从已有 cycleN 报告续跑(自动喂上一轮报告给书生观察)。
- `--cycles K`: 连续 K 轮(观察→推理→行动→报告→下一轮), 一次进程内不打断。
- `--max-iters 12 --min-iters 4`: Pareto 搜索迭代预算。
- `--dry`: 确定性运行, 不调模型, 用于验证整条管线。
- 续跑日志建议 `| tee /tmp/cycleN_M_run.log` 便于回溯。

## 循环架构(书生驱动 Huginn 工作流)
每轮 = 书生[观察]上轮报告 → [规划]提出开放问题与扫描配置 → [行动]真实实验
(Pareto 剪枝) → CriticAgent[对立审稿] → 门禁(claim_grounding 核对数值) → 成文报告 → 下一轮。
- 书生亲手写实验代码通过 Code Lab 沙箱校验后, 成为 author_c{cycle} 真实实验分支。
- 写码失败自动回退白名单扫描(不伪造, 数值仍真实执行)。

## Pitfalls(经修复的已知故障)
1. **写码失败回退白名单** —— 若日志反复出现 "CodeLab 校验失败":
   - 提取 bug: 代码块带尾部 ``` 栅栏 / 裸 def 兜底优先 → 已修 `code_lab.extract_code`
     (干净块优先, 兜底截断栅栏), 一般勿再动;
   - 模型写 `a,b=cfg` 解包 dict(解出键名 str) / 用不存在的 `np.*` → prompt 已明令禁止
     cfg['key'] 逐键读、只用真实 numpy API; 仍失败靠 ≤2 次修复回路兜底。
   - cfg 键名别名可用: `cfg['ti']/t_i/t = positions[0]`, `cfg['n_seeds']=seeds`,
     `cfg['basis_size']=basis`。
2. **兜底报告被门禁误杀**(unsubstantiated=[250.0,...]) —— 实验名 `author_c{cycle}` 里
   的 cycle 大整数被当数值主张; 已修 `claim_grounding._strip_ordinal_markers` 剔除
   轮次号/cycle/author_cNNN/S\d+_scan。若新命名模式再现, 在该函数补正则。
3. **书生抄历史报告数值** —— 成文 prompt 已明令"只引用本轮真实结果, 禁止引历史数值";
   若门禁仍报旧值未落地, 是正确拦截, 需让书生删/补跑, 勿放松门禁。
4. **报告头"门禁 pass" + 聚合 gate_blocked 并存** —— harness/CI 层固定伪影
   (audit.score_usage / governance.external_verify), 勿把 gate_blocked 当结论被否。
5. **最高卡点是数据缺口非代码** —— 补数据统一走确定性探针, 不造假。

## Verification
- 每轮日志出现 `[gate] pass unsubstantiated=[] source=agent` = 书生本体成文通过门禁;
- `source=fallback_assembly` = 模型未交付, 走确定性兜底(数值仍真实, 只是非模型行文);
- 报告落 `examples/out/shusheng_huginn_workflow/cycleNNN_report.md`;
- 一批结束打印 `===== 长程任务结束: 连续完成 K 轮 =====`。

## 复用既有结论
`agent/skills/science-research/SKILL.md` 已固化 264 轮领域结论(值约束位置定律/
异常点/约束类型相变/统计稳健性)与「closed」清单, 查询时优先引用, 勿重复重开
已收敛问题。