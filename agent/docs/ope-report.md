# OPE 报告（准入策略）

> 数据来源：`HUGINN_NORMATIVE_TRAJECTORY=1` 下 12 次真实 `file_read_tool` 调用
> （只读、真实执行、outcome 已回填）。这是**引导用真实轨迹**，不是生产规模语料。

| 指标 | 值 |
|---|---|
| tenant | bench |
| total samples | 12 |
| train / eval | 8 / 4 |
| IPS | 1.0 |
| SNIPS | 1.0 |
| DR | 1.0 |
| behaviour value | 1.0 |
| unauthorized rate | 0.0 |
| false pause rate | 0.0 |
| shadow unauthorized executions | 0 |

说明：offline OPE only; shadow before any rollout。策略只在 train 段更新案例统计，eval 段做 IPS/SNIPS/DR；
影子模式按 action mask 评估，未授权执行必须为 0。

完整机器可读结果见 `docs/ope-report.json`。
