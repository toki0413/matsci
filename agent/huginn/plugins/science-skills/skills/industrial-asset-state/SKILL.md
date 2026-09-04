---
name: industrial-asset-state
description: Use when 需要把产线/设备状态整合成结构化回传(温度/压力/转速/运行时间/告警位)供入库存档或比对；触发场景包括采集设备状态、汇总 OT 传感读数、判断是否超阈值告警、把多源现场数据规整成一致 JSON 摘要。对应 chain-of-thought：输入设备状态数据源 → 输出"设备清单+各项读数+阈值判定+告警"的结构化摘要。
category: diagnostics
allowed-tools: [file_read_tool, numerical_tool]
tags: [industrial, ot-data-collection, asset-state]
---

# industrial-asset-state

把产线设备的状态信息（温度 / 压力 / 转速 / 运行时间 / 告警位）整合成**结构化回传 JSON**。
这是一个**指令式工业 Skill 示例**：它把"现场数据采集 → 规整 → 阈值判定 → 告警"这段
工程 Know-how 封装成可由 LLM 按正文执行的规范；无外部依赖、无脚本，可直接被
`SkillImporter` / `eco_tool.skill_install` 解析注册，作为接入工业 OT 后端的轻量封装层。

## 用法

- 输入：设备级状态数据（来自文件摘要、MCP 采集结果或直接给定）。
- 输出：统一 JSON，供 `memory`/`materials_database` 等下游入库或比对。

## 输出 schema

```json
{
  "source": "asset_state",
  "collected_at": "ISO8601",
  "assets": [
    {
      "id": "PUMP-01",
      "readings": {"temperature_c": 62.0, "pressure_bar": 1.8, "rpm": 1450},
      "runtime_h": 3120.5,
      "alarms": [],
      "threshold_hit": false
    }
  ]
}
```

## 方法规则（chain-of-thought 必须序贯执行）

1. 列出每个设备的 id 与各项读数，缺读数标 `null`，不要臆造。
2. 统一单位（温度 `°C`、压力 `bar`、转速 `rpm`），单位不明时在 `readings` 里附 `_unit_spec`。
3. 阈值判定：与默认阈值（或外部传入）比较，命中写进 `alarms`，并把 `threshold_hit` 置 `true`。
4. 输出必须是上表的紧凑 JSON；判定逻辑要可复核，不要隐藏缺失数据。

> 本 skill 是"采集-规整-判定"封装层的演示；需要真实探测 OT/PLC 时，把数据源换成
> `eco_tool.mcp_connect` 接的采集能力，正文规则保持不变。