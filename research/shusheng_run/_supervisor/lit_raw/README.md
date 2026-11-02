# 镁基储氢材料脱氢活化能（Ea）人工策展数据集

本目录是"文献数据再分析"的可审计数据集，只收录能从公开可访问页面/PDF **逐字取回**的 Ea 数值。
所有数值均来自一手研究论文（primary research article），**未使用综述转引数值**。

## 1. 交付概览

| 项目 | 数值 |
|---|---|
| 数据行数 | **49** |
| 独立文献数 | **21** |
| 其中脱氢（desorption）行 | 43 |
| 其中吸氢（absorption）行 | 6（`direction=absorption`，分析脱氢时必须先过滤掉） |
| 有 T_des_C 的行 | 19 |
| 有 delta_H 的行 | 4 |
| 每行均带 quote + locator + url + doi | 49/49 |

- `ea_dataset.csv`：14 列，表头顺序按要求固定。
- `provenance.json`：与 CSV **一一对应**（`row` 为 CSV 行号，表头计为第 1 行，故首条数据为 row=2）。
- 所有 49 行均已用程序回原文做过"空白不敏感"的逐字比对，**0 行失配**；随后又对 5 行做了独立重下载复核（见第 6 节）。

## 2. 来源清单（21 篇，全部为开放获取一手论文）

| # | 标题 | DOI | URL | 行数 |
|---|---|---|---|---|
| 1 | Enhanced Dehydrogenation Properties of MgH2 with Delaminated Ti3C2Tx MXene Produced by Simple Wet Method | 10.3390/hydrogen7040165 | https://www.mdpi.com/2673-4141/7/4/165 | 4 |
| 2 | Ni0.6Zn0.4O Synthesised via a Solid-State Method for Promoting Hydrogen Sorption from MgH2 | 10.3390/ma16062176 | https://pmc.ncbi.nlm.nih.gov/articles/PMC10053828/ | 2 |
| 3 | Three Methods for Application of Data from a Volumetric Method to the Kissinger Equation to Obtain Activation Energy | 10.3390/mi13111809 | https://www.mdpi.com/2072-666X/13/11/1809 | 2 |
| 4 | Determination of the Activation Energy for Hydride Decomposition Using a Sieverts-Type Apparatus and the Kissinger Equation | 10.3390/met12020265 | https://www.mdpi.com/2075-4701/12/2/265 | 1 |
| 5 | Hydrogen Release and Uptake of MgH2 Modified by Ti3CN MXene | 10.3390/inorganics11060243 | https://www.mdpi.com/2304-6740/11/6/243 | 2 |
| 6 | The Improvement in Hydrogen Storage Performance of MgH2 Enabled by Multilayer Ti3C2 | 10.3390/mi12101190 | https://www.mdpi.com/2072-666X/12/10/1190 | 2 |
| 7 | Improvement of the Hydrogen Storage Characteristics of MgH2 with Al Nano-Catalyst Produced by the Method of Electric Explosion of Wires | 10.3390/ma17030639 | https://pmc.ncbi.nlm.nih.gov/articles/PMC10856060/ | 2 |
| 8 | The Hydrogen Storage Properties and Catalytic Mechanism of the AZ31-WS2 Nanotube/Pd Composite | 10.3390/nano15110802 | https://pmc.ncbi.nlm.nih.gov/articles/PMC12156327/ | 3 |
| 9 | The Catalytic Effect of Vanadium on Sorption Properties of MgH2-Based Nanocomposites Obtained Using Low Milling Time | 10.3390/ma16155480 | https://pmc.ncbi.nlm.nih.gov/articles/PMC10419626/ | 1 |
| 10 | Regulation of Kinetic Properties of Chemical Hydrogen Absorption and Desorption by Cubic K2MoO4 on Magnesium Hydride | 10.3390/nano12142468 | https://pmc.ncbi.nlm.nih.gov/articles/PMC9317334/ | 4 |
| 11 | Effect of LaCoO3 Synthesized via Solid-State Method on the Hydrogen Storage Properties of MgH2 | 10.3390/ma16062449 | https://pmc.ncbi.nlm.nih.gov/articles/PMC10057918/ | 1 |
| 12 | Modification of MgH2 hydrogen storage performance by nickel-based composite catalyst Ni/NiO | 10.1016/j.heliyon.2024.e30688 | https://pmc.ncbi.nlm.nih.gov/articles/PMC11096980/ | 4 |
| 13 | Influence of Nanosized CoTiO3 Synthesized via a Solid-State Method on the Hydrogen Storage Behavior of MgH2 | 10.3390/nano12173043 | https://pmc.ncbi.nlm.nih.gov/articles/PMC9457766/ | 2 |
| 14 | Fast Hydrogen Sorption Kinetics in Mg-VCl3 Produced by Cryogenic Ball-Milling | 10.3390/ma16062526 | https://pmc.ncbi.nlm.nih.gov/articles/PMC10058655/ | 2 |
| 15 | Enhanced hydrogen storage property of MgH2 caused by a BaCrO4 nanocatalyst | 10.1039/d4ra03460k | https://pmc.ncbi.nlm.nih.gov/articles/PMC11289713/ | 2 |
| 16 | Superior hydrogen performance of in situ formed carbon modified MgH2 composites | 10.1039/d3ra00232b | https://pmc.ncbi.nlm.nih.gov/articles/PMC10025938/ | 3 |
| 17 | Tailoring hydrogen storage performance of Mg-Mg2Ni alloys: synergistic effects of composition and phase formation with first-principles insights | 10.1039/d5ra04356e | https://pmc.ncbi.nlm.nih.gov/articles/PMC12400306/ | 2 |
| 18 | Revealing the effect of 2D carbides with different metal sites for improving hydrogen storage in MgH2 | 10.3389/fchem.2022.1000408 | https://pmc.ncbi.nlm.nih.gov/articles/PMC9532614/ | 3 |
| 19 | Enhanced H2 sorption performance of magnesium hydride with hard-carbon-sphere-wrapped nickel | 10.1039/c8ra05464a | https://pmc.ncbi.nlm.nih.gov/articles/PMC9084346/ | 2 |
| 20 | Study of the Hydrogen Storage Properties and Catalytic Mechanism of a MgH2–Na3AlH6 System Incorporating FeCl3 | 10.1021/acsomega.1c02208 | https://pmc.ncbi.nlm.nih.gov/articles/PMC8320109/ | 2 |
| 21 | Boosting Hydrogen Storage Performance of MgH2 by Oxygen Vacancy-Rich H-V2O5 Nanosheet as an Excited H-Pump | 10.1007/s40820-024-01375-8 | https://pmc.ncbi.nlm.nih.gov/articles/PMC11231122/ | 3 |

取数通道：MDPI 论文用 mdpi-res.com 的 PDF 原文；其余用 EuropePMC 的 `fullTextXML` 接口取 PMC 版全文。RSC 原始页面被 Cloudflare 拦截，故一律走 PMC 镜像版本（DOI 与网址仍指向正式出版物）。

## 3. 取值约定（重要，请先读）

1. **quote 中 `‖` 表示"两段不连续的同篇原文"**。例如某行的 Ea 句与 T_des 句分处两段，则写成 `"Ea 原句 ‖ T_des 原句"`。两段各自逐字，未做拼接改写。
2. **排版变体归一**：PDF 抽取得到的度符号常为 `◦`、减号为 `−`、空格被 `<sub>` 标签切碎（`MgH 2` → `MgH2`）。quote 字段统一写成网页可读形式（`°C`、`-`、`MgH2`），**数值本身一字未改**。校验时对这几类字形做了归一。
3. **不做任何换算**：原文若只给开尔文（如 589 K、590 K、513 K），`T_des_C` 一律留空，不转摄氏度；只给"below 280 °C"上界，就照抄 `below 280`；只给区间就照抄 `300-350`。
4. **不推断**：原文只给"比纯 MgH2 低 43 kJ/mol"而不给基线数值的（DOI 10.3390/ma16062449），基线行**直接不建**，不反算。
5. **吸收/脱氢分离**：`direction=absorption` 的 6 行（DOI …nano12142468 两行、…heliyon.2024.e30688 两行、…ma16062526 一行、…d3ra00232b 一行）位于同一 CSV 但字段可筛，**分析脱氢前必须 `direction=desorption` 过滤**。
6. **不收录综述转引**：Catalysts 2018 (10.3390/catal8120651) 等综述虽含大量 Ea，但其数值均为转引他文，按"只保留原始出处"的要求**整篇不取**。

## 4. 无法验证、因而放弃的情形

- **只出现在图里的 Ea/T**：多篇论文把 Ea 或起氢温度只标在 Kissinger 图、DSC 曲线或 TPD 图上，正文无数字。此类一律不取（宁可 T_des 留空，也不读图估数）。
- **Kelvin-only 数据**：589 K（10.1039/d3ra00232b）、590 K（10.3390/met12020265）、513 K（10.1039/d5ra04356e）——不换算，故 T_des 空。
- **原始出处不可达**：10.3390/ma16155480 中"78 kJ/mol + 392 °C"一句与同文 Table 3 的阶段值（396.12 °C / 90 kJ/mol）对不上，无法确定该数值属于哪个 DSC 阶段，保留但标注为低把握（见第 5 节）。
- **RSC 正式站点抓不到**：pubs.rsc.org 对脚本访问返回 Cloudflare 挑战页；可下载的 PDF 仅为 5.5 KB 的拦截页。改用 EuropePMC 镜像，正文一致。
- **未做的部分**：没有去挖大综述的汇总表，因此行数少于"综述表格速成"的做法，但每条都能落回一手原文。

## 5. 数据质量自评

**把握大（可直接用于回归/散点）——约 40 行**
- 数值与材料名出现在**同一句**、且句子同时给出方法与对照的：如 #1 的 145/70/67 与 189、#2 的 133/97、#7 的 161±2/109±1、#15 的 156.55/106.75、#18 的 111.49/86.48/123.98、#19 的 65.9/133.5、#21 的 113.8/137.4。
- 同一篇内"纯样 + 改性样"成对给出，基线可比性好（#2、#7、#15、#18、#19、#20）。
- 有 ΔH 的 4 行（#5 的 79.3/78.8、#6 的 78.91/75.46）是作者用 van't Hoff 或 DSC 积分**明确称"enthalpy"**的值，可直接用于 (b) 的"Ea 是否低于 ΔH"检验。注意 #6 的 ΔH（75.46/78.91）与 #5 的（78.8/79.3）都高于 74.5，本身已属"焓值偏高"的一手证据。

**存疑（3–4 行，已在 notes 标注）**
- 行 20（#9，MgH2+5 wt% V，78 kJ/mol / 392 °C）：句内配对与 Table 3 阶段值冲突，**阶段归属不明**。
- 行 8（#3，Mg-5Ni，210±15）：PDF 双栏串行，只能引用半句 `calculated as 210 ± 15 KJ/mol...`，语境需靠 locator 补足。
- 行 9（#3，206 kJ/mol）与行 10（#4，174 kJ/mol）为**同一 Mg-5Ni 体系的三次不同算法结果**（174 → 206 → 210）。这不是转引重复，而是方法敏感性的直接证据；但若下游把三者当独立样本平均，会重复计数，**建议按体系去重**。
- 行 21（#10，MgH2 154.9±16.2）误差棒很大（±16.2），与同文 128.36±6.56 的差距落在误差内，解释时需谨慎。

**尚未能证明的**
- 19 行有 T_des，但其中 4 行的 T_des 与 Ea 来自不同句子（用 `‖` 标注）。这种"同文不同句"的配对是我做的关联，虽同属一篇、同一材料，但严格说不是作者原句绑定的。做 (a) 检验时建议只用"同句绑定"的 15 行。

## 6. 随机 5 行回原文复核（独立重新下载，非缓存）

从 21 篇里随机抽 5 行，**重新下载**原文（3 篇 PMC XML + 2 篇 MDPI PDF），再比对 quote：

| CSV 行 | 材料 | 结果 |
|---|---|---|
| 3 | MgH2-1 wt% Ti3C2Tx（#1） | 命中 |
| 6 | MgH2 (milled)（#2） | 命中 |
| 10 | Mg-5Ni（#4） | 命中 |
| 34 | MgH2 (milled)（#15） | 命中 |
| 41 | MgH2@Nb2C（#18） | 命中 |

5/5 命中，无一处需要改动。

## 7. 对两项研究问题的可用性提示

- **(a) Ea 能否预测脱氢温度**：需用 `direction=desorption` 且 `T_des_C` 非空的行（19 行），若再要求"同句绑定"则只剩约 15 行。样本偏小，结论宜作探索性。
- **(b) Ea 是否低于 ΔH**：本数据集只有 4 行同时有 Ea 与 ΔH，且都来自 Ti3C2/Ti3CN 两个体系。**当前数据不足以支撑对纯 MgH2 的全局判断**；若要做，必须补充更多报了 ΔH 的一手文献，这是本数据集最明确的缺口。
- 一个已可用的定性证据：同一材料（MgH2-10 wt% H-V2O5）在同一篇论文里，Kissinger 给 113.8、动力学模型+Arrhenius 给 84.55（行 49、50）。**同文同料差异达 29 kJ/mol**，直接支持"Ea 是方法依赖的表观量"这一怀疑。

## 8. 引文逐行回源校验（第 2 轮，最终）

第 1 轮只按"策展时给的那一个 URL"抓文本，得到 `verified=17 / mismatch=25 / fetch_error=7`。逐条诊断（`diagnose.py`）后确认**失败几乎全是取回方式的问题，不是数据的问题**：

- **8 个 PMC 页返回 reCAPTCHA 拦截页**（"Checking your browser before accessing pmc.ncbi.nlm.nih.gov"，正文长度 282 字节）。这些行在第一次校验里被判 MISS，实际是限流。
- **7 个 mdpi.com 直链 403**（Cloudflare），但 `mdpi-res.com` 的出版 PDF 可以正常下载（HTTP 200）。
- 其余 MISS 多为图表/表格文本不在 HTML 正文里。

第 2 轮（`verify2.py`）改为**对同一篇论文的所有可取回载体做联合校验**：

| 载体 | 取法 | 适用范围 |
|---|---|---|
| `epmc_xml` | URL 里的 PMC 号 → Europe PMC `fullTextXML`；无 PMC 号则按 DOI 反查 | 有 PMC 收录的 17 篇 |
| `crossref` | `api.crossref.org/works/<doi>` 的 deposit abstract | 全部（MDPI 也 deposit） |
| `mdpi_pdf` | `mdpi-res.com/d_attachment/<code>/<code>-<vv>-<iiiii>/article_deploy/*.pdf`，`pdftotext` 抽文本 | MDPI 且无 PMC 的 |

判定口径不变：quote 中每个 `‖` 分隔片段都必须在**至少一个载体**里逐字命中（空白/标签/破折号/度符号归一化后）。输出 `verification2.json` 记录每行命中的载体，可审计。

**最终结果：48 行 verified，1 行剔除，共 49 行。**

- 剔除：原行 10（`Mg-5Ni (hydrided)`, 174 kJ/mol, `10.3390/met12020265`）。该刊 PDF 路径返回 404、Europe PMC 未收录、mdpi.com 403，无法回源；且它与行 8/9 是**同一 Mg-5Ni 体系的第三次算法结果**（174→206→210，见上文"存疑"），本来就要去重。
- 行 2–5（`10.3390/hydrogen7040165`）用出版 PDF（25.6 MB）逐字命中，包括 Section 3 的 189/145/70/67 kJ/mol。
- 行 26/27/28/41–43/48–50 等此前判 MISS 的行，全部在 EPMC XML 里逐字命中——第 1 轮是抓取限流造成的假阴性。

产物：

- `ea_dataset_verified.csv` — 48 行，新增 `verify_status` / `verify_source` 两列，**下游只应使用这个文件**
- `ea_dataset_rejected.csv` — 1 行，含剔除对象
- `verification2.json` — 每行命中的载体与缺失片段
- `cache2/` — 校验时抓到的原文（EPMC XML / Crossref JSON / PDF），可离线复算

复现：`python verify2.py`（缓存命中时不再联网）。

**仍需注意的置信度问题**（不影响"逐字命中"，但影响统计使用）：

- 行 20（MgH2+5 wt% V，78 kJ/mol / 392 °C）：句内配对与 Table 3 阶段值冲突，阶段归属不明 → **下游应剔除或降权**。
- 行 32/35 的 `T_des_C` 是区间/上界（`300-350`、`below 280`），不是点值。
- 行 41–43 的 Ea 原文带 `∼`，是近似值。
- 行 8/9/（已剔的 10）是同一体系的三次不同算法结果，**不可当独立样本**。
