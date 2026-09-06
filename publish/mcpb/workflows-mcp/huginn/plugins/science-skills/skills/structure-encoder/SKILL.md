---
name: structure-encoder
description: Use when 用户提供晶体/材料结构文件(CIF/POSCAR/xyz)或化学式(如 LiFePO4)需要归一化成统一的材料描述子用于检索/入库/比对；触发场景包括结构文件标准化、化学式解析、材料指纹提取、结构 vs 化学式统一表示、把异构结构输入转成可检索的 JSON 描述子。对应 chain-of-thought: 输入结构 → 输出统一的"化学式+空间群+晶格+元素比例" JSON。
---

# structure-encoder

把异构的材料结构输入（CIF / POSCAR / xyz / 化学式）归一化成**统一描述子 JSON**，供 Huginn 的文本/知识库检索管道消费。这是"检索级异构材料编码"：不训练联合向量，而是产出稳定、确定性的结构指纹。

## 用法

```
uv run scripts/structure_embed.py --query LiFePO4
uv run scripts/structure_embed.py --query /path/to/xx.cif --output /path/to/desc.json
```

- `--query`：结构文件路径，或化学式（如 `LiFePO4`、`Si`）。
- `--output`：可选；写入 JSON 文件路径，缺省打到 stdout。
- 结构文件用 `pymatgen.Structure.from_file` 自动识别格式。

## 输出 schema

```json
{
  "source": "structure_file|formula",
  "formula": "reduced_formula",
  "n_atoms": 24,
  "elements": ["Li", "Fe", "P", "O"],
  "species_shares": {"Li": 0.111111, "Fe": 0.111111, "P": 0.111111, "O": 0.666667},
  "density": 3.5,
  "volume_ang3": 291.4,
  "space_group": "Pnma",
  "lattice_abc": [10.3, 6.0, 4.7],
  "lattice_angles": [90.0, 90.0, 90.0]
}
```

## 方法规则（chain-of-thought 必须序贯执行）

1. 判断 `--query` 是文件路径还是化学式。
   - 字符串能匹配一个存在的文件 → 按结构文件解析。
   - 否则 → 按化学式解析。
2. 结构文件：用 `pymatgen` 读入；输出化学式、原子数、元素比例、密度、体积、空间群、晶格参数。
3. 化学式：用 `Composition` 解析为约化式 + 元素比例。
4. `elements` 必须是字符串列表；比例四舍五入到 6 位小数。
5. 空间群判定可能因非周期结构失败 → 此时 `space_group` 为 `null`，不中断整个描述子。
6. 只做归一化，不做任何预测或仿真。

## 输出约定

- JSON 以 UTF-8 输出，`ensure_ascii=False`（保留化学式里的希腊字符等）。
- 结构文件不存在或公式非法 → stderr 报 `ERROR: ...`，退出码 2。
- 作为 bridge 工具时走泛化参数骨架（`--query + --output`），单次调用即完成，**不需要交互确认**。