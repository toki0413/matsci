# 模型档位契约 (ModelTier)

自动生成: `python -m huginn.cli.config_audit --model-tier --out docs/model-tier-contract.md`.
登记极简模式的三档 profile (`model_tier.py::_TIERS`)。每档聚合一组认知编排开关:是否启用 phase 机 / plan 门控 / 认知纪律形式 / compaction 力度 / 外部思考。安全层 (命令校验 / 物理 sanity check / 资源预算) 在所有档位都保留。与**思考强度** (ThinkingIntensity) 是两条正交轴, 互不影响, 可独立设置。

| 档位 | phase 机 | plan 门控 | 认知纪律 | compaction | 外部思考 | 语义 |
|---|---|---|---|---|---|---|
| `full` | ✅ 开 | ✅ 开 | `always` | `heavy` | ✅ 开 | 本地弱模型, 保留全部认知编排 (常驻纪律 + phase 门控) |
| `balanced` | ✅ 开 | ✅ 开 | `event` | `medium` | ✅ 开 | 中等模型, 认知纪律降级为事件驱动 (仅偏离才注入) |
| `minimal` | ✗ 关 | ✗ 关 | `event` | `light` | ✗ 关 | 顶尖大模型, 跳过 phase/plan 门控, 事件驱动守护, 轻 compaction |

运行时切换: `HUGINN_MODEL_TIER` 环境变量设默认档位, `set_tier()` 运行时切换(minimal 档 `external_thinking` 默认关, 切换时联动 `FeatureFlags.external_thinking`)。
