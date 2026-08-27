---
name: instrument-ingest
description: Use when 需要把仪器导出的测量文件(XRD .xy / 两列文本 / 简单 CSV)解析成结构化峰位/强度摘要 JSON 供入库/比对；触发场景包括提峰、XRD 数据标准化、信号峰位提取、判断是否可能含新相、把裸两列测量数据转成可检索的测量摘要。对应 chain-of-thought: 输入仪器导出文件 → 输出"峰位+强度+FWHM+新相提示"的 JSON。
---

# instrument-ingest

把仪器导出文件（XRD `.xy` / 两列文本 / 简单 CSV）解析成**结构化测量摘要 JSON**。Huginn 与仪器之间没有实时的 OPC-UA/MQTT 驱动，桥接层就是文件：读取仪器导出的裸两列数据，提峰，产出一个紧凑、确定性的摘要，并给出"可能含新相"的提示供下游 `materials_database` 交叉验证。

## 用法

```
uv run scripts/read_xrd.py --query /path/to/xx.xy
uv run scripts/read_xrd.py --query /path/to/xx.csv --output /path/to/summary.json
```

- `--query`：仪器导出文件路径（`.xy` / `.csv` / 两列文本）。
- `--output`：可选；写入 JSON 文件路径，缺省打到 stdout。

## 输出 schema

```json
{
  "source": "instrument_export",
  "file": "xx.xy",
  "instrument": "xrd",
  "points": 1200,
  "x_min": 10.0,
  "x_max": 80.0,
  "intensity_max": 34567.0,
  "peak_count": 5,
  "peaks": [
    {"position": 27.4, "intensity": 34567.0, "approx_fwhm": 0.21}
  ],
  "new_phase_hint": true
}
```

## 方法规则（chain-of-thought 必须序贯执行）

1. 读文件时跳过 `#` / `//` / `;` 开头的注释行。
2. 分隔符支持空白 / 逗号 / 分号；取前两列作为 (x=角度, y=强度)。
3. 数据点少于 3 个 → 报告 `ERROR`，退出码 2，不产摘要。
4. 提峰用移动窗口局部极大，阈值为 `0.08 × max(intensity)`；不依赖 scipy（纯 numpy）。
5. 相邻窗口内的重复峰只保留强度更高的那个。
6. `new_phase_hint = peak_count >= 3`：这是启发式标记，供下游数据库交叉验证，不承诺真实新相。
7. 只做解析与特征提取，不做任何相鉴定或 Rietveld。

## 输出约定

- JSON 以 UTF-8 输出，`ensure_ascii=False`。
- 文件不存在 / 无法识别为两列数值表 / 数据点不足 → stderr 报 `ERROR: ...`，退出码 2。
- 作为 bridge 工具时走泛化参数骨架（`--query + --output`），单次调用即完成，**不需要交互确认**。