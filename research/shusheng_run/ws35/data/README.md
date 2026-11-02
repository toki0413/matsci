# data/ea_dataset.csv — 文献抽取的 MgH2 表观活化能数据集

## 来源与校验

- 48 行，来自 20 篇一手研究论文（DOI 见 `doi` 列）。
- 每一行的 `quote` 都已**逐字回源校验**：在 Europe PMC 全文 XML、Crossref deposit abstract 或出版 PDF（`mdpi-res.com`）中找到原文。校验脚本与结果在策展目录，此处不随包提供。
- 原始策展 49 行，剔除 1 行（见下）。

## 列

| 列 | 含义 |
|---|---|
| `material` | 样品名（原文措辞，未统一命名） |
| `material_class` | `pure_MgH2` / `catalyzed_MgH2` / `alloy` / `complex_hydride` |
| `catalyst_or_additive` | 催化剂或添加物原文措辞 |
| `catalyst_family` | 已归并的 5 类：`mxene_carbon` / `oxide` / `metal_nanoparticle` / `other` / `none`。**这是数据集给定的，不要自行重分类** |
| `Ea_kJ_per_mol` | 表观活化能，kJ/mol（字符串，可能是 `109 ± 1` 这类带误差的写法） |
| `direction` | `desorption`（脱氢）或 `absorption`（吸氢） |
| `T_des_C` | 脱氢起始/峰值温度，°C。**可能为空、是区间（`300-350`）或上界（`below 280`）** |
| `method` | `Kissinger` / `isothermal_Arrhenius` / `DSC` |
| `delta_H_kJ_per_mol_H2` | 作者明确称为焓变的值，多为空 |
| `doi` | 论文 DOI，用来做整群 |
| `locator` / `quote` | 原文位置与原文引文（用于复核） |
| `notes` | 策展备注，含置信度问题 |
| `verify_status` / `verify_source` | 回源校验状态与命中的载体 |

## 使用前必读

1. **行不独立。** 同一 `doi` 的多行共享样品、仪器、升温速率；同一体系可能被同一作者用不同算法算过多次。任何推断都按 `doi` 整群，禁止按行独立重采样。
2. **必须剔除**：`material == "MgH2 + 5 wt% V"`（doi `10.3390/ma16155480`）。该行句内把 78 kJ/mol 与 392 °C 配对，但同文 Table 3 里该样品的高温阶段是 396.12 °C / 90 kJ/mol，阶段归属不明。
3. **同一体系的重复算法，不可当独立样本**：
   - `Mg-5Ni` 两行（`10.3390/mi13111809`，210 与 206 kJ/mol）是同一实验数据用不同 Tm 定义重算的结果。
   - `MgH2-10 wt% H-V2O5` 两行（`10.1007/s40820-024-01375-8`，113.8 与 84.55 kJ/mol）是同一论文里 Kissinger 与动力学模型+Arrhenius 两条路线。**这两行正是 H3 要用的"同样品跨方法"对子**。
4. **`direction == "absorption"` 的 6 行**不得混入脱氢分析。
5. **`T_des_C` 不是全都能当数值**：`300-350`、`below 280` 是区间/上界；空的就留空。不要做单位或区间换算。
6. **近似值**：`MgH2@Nb2C` / `MgH2@Ti2C` / `MgH2@graphene` 三行的 Ea 原文带 `∼`，是近似值。
7. **跨论文条件未统一**：升温速率、样品量、仪器、氢压都不同，这是本数据集最大的外部效度限制，结论里必须写明。

## 建议的解析约定

- `Ea_kJ_per_mol` 与 `delta_H_kJ_per_mol_H2` 用正则取第一个浮点数（`109 ± 1` → `109`）。
- `catalyst_family == "none"` 同时覆盖 `pure_MgH2`、`alloy`、`complex_hydride` 与一个无添加剂的合金样品——H2 只在 `catalyzed_MgH2` 上做，不受此影响。
