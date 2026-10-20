# 科学发现能力对标缺口 — 补缺 Spec

> Status: ACTIVE (as-built, 见 §7)
> Source: 对标「自主科学发现闭环」十项标准的全仓实证审计 (2026-10-16)
> Mode: 审计 → 缺口 → 定向补齐; 不动已实装能力

## 1. 背景

对标标准: 文献与多模态科学数据 (图表/谱图/分子结构/实验曲线) 的精准检索解析、
假设生成与逻辑推导、实验/仿真方案设计、计算工具自主调用、可复现验证与迭代优化、
长文本理解、多步推理校验、多智能体协作。

审计结论: 十项中七项实装 (含双路图表反解析 `perception/data_extractor.py` +
`vision/symbol_encoder.py`, 此前曾误判为"仅渲染", 特此更正); **三项有真缺口**
—— 分子结构输入面 / NMR·质谱解析 / 复现重跑闭环. 本 spec 只补这三项.

### 缺口证据

| 编号 | 缺口 | 证据 (file:line) |
|---|---|---|
| G-A | **分子结构输入面缺失**: `rdkit_tool` 无 MOL/SDF 文件加载 action (仅 `smiles_to_sdf` 输出方向, `huginn/tools/sci/rdkit_tool.py:25-99`); PDB 蛋白解析全仓 0 处 (仅 `workflows/templates_qc.py:27` 作为输入文件名提及); 2D 结构图像→SMILES (结构 OCR) 0 处 | 见左 |
| G-B | **NMR/质谱解析缺失**: mzML/JCAMP-DX/JDX 解析器全仓 0 处; `huginn/academic/journal_db.py:529-547` 仅将 "1H/13C NMR, IR, MS" 作为**发表要求文本**, 无解析能力承接 | 见左 |
| G-C | **复现缺"重跑闭环"**: provenance 只记录不重放 (`autoloop/engine_act.py:769-790`); `autoloop/replay_audit.py` 是确定性**轨迹审计** (判空转), 非执行重放; 无「同 seed/同输入重跑 → 数值 diff」的 reproduce 机制. 账本可信已有, 重跑一致缺最后一环 | 见左 |

### 已有可复用底座 (不重造)

- 谱图解析范式: `characterization_tool.py` 的 action 分派 + `_read_xy` + scipy 峰检测;
- 分子解析: `rdkit_tool` 已有 `MolFromSmiles` 全链路; `metacog/structure_cognitive_map.py:137` 已示范 `MolFromMolBlock`;
- 复现: provenance 快照 (输入/输出/错误) + Rust 沙箱确定性 (docs/p0-state-store-and-rust.md)
  + `replay_audit` 的 shard 加载范式;
- 治理: 新工具落地须同步 `capabilities/runtime_probe` 清单 (L0 探针) 与
  `governance_audit` 能力锚点 — 缺同步即红, 不再静默.

## 2. 目标

1. **G-A**: 分子结构输入支持 MOL/SDF/PDB 文件 → RDKit 分子对象 (蛋白只到
   `Chem.MolFromPDBBlock` 级别, 不做全蛋白组学); 结构图像→SMILES 明确**不做**
   (见非目标).
2. **G-B**: NMR (JCAMP-DX .jdx) 与质谱 (mzML) 数值解析 → 结构化峰表, 承接
   journal_db 已声明的表征要求.
3. **G-C**: `reproduce` 机制: 给定 provenance 记录 → 同输入重跑 → 数值容差 diff →
   报告一致/漂移. 与 `replay_audit` 分工: 它审计**轨迹质量**, 本项验证**结果可复现**.

## 3. 非目标 (YAGNI)

- 不做 2D 结构图像→SMILES 的结构 OCR (误差率不可控, VLM `read_image` 已可应急;
  需要时再立项).
- 不做蛋白组学/生物信息全栈 (PDB 仅作结构输入之一).
- 不做连续参数贝叶斯优化 (joint_optimizer 的 UCB bandit 已可用; 需要时升级).
- 不改已实装的七项能力.

## 4. 改动面

| 单元 | 类型 | 职责 |
|---|---|---|
| `huginn/tools/sci/rdkit_tool.py` | 改 | 新 action `mol_file_to_mol` (MOL/SDF/PDB block → 分子; SDF 多记录逐条) |
| `huginn/tools/characterization_tool.py` 或新 `spectra_nmr_tool.py` | 新增 | `jdx_parse` (JCAMP-DX → 峰表) + `mzml_parse` (mzML → 质谱峰表); stdlib/自写解析, 不引重依赖 |
| `huginn/autoloop/reproduce.py` + CLI `huginn/cli/reproduce.py` | 新增 | 读 provenance 快照 → 重放工具调用 (同参同 seed) → 数值 diff (相对容差) → 报告 |
| `huginn/capabilities/runtime_probe.py` | 改 | 新工具入 L0 探针清单 (缺 rdkit 降级可见, 注入 capability_note) |
| `tests/test_arch_capability_anchors.py` 等 | 改 | 能力锚点 + 探针 floor 棘轮同步 |
| `docs/INDEX.md` | 改 | 本 spec 状态跟踪 |

## 5. 验收标准 (AC)

- **AC-1** `mol_file_to_mol` 能加载 MOL/SDF/PDB 三格式; SDF 多记录全返; 坏文件返回
  `success=False` + 可定位错误 (不 crash).
- **AC-2** `jdx_parse`/`mzml_parse` 对真实样例文件 (tests/fixtures) 解析出结构化峰表;
  样例文件小、可入库.
- **AC-3** `reproduce` 对一条已记录 provenance 的工具调用: 确定性工具 → 一致;
  人为改动输入 → 报告漂移. 报告含逐字段 diff, 不报"有漂移"空话.
- **AC-4** 三个新能力全部入 L0 探针 + 能力锚点 + 治理台账; 缺库时 capability_note
  注入可见 (复用既有 fail-open 范式).
- **AC-5** 镜像 (publish/mcpb) 同步, 漂移 0; ruff clean; 新测试 weightless 优先.

## 6. 风险与诚实边界

- JCAMP-DX 变体多 (核磁厂商方言), 解析器按"覆盖常见 NMR 变体 + 显式报不支持的
  变体"裁剪, 不承诺全兼容.
- `reproduce` 只覆盖**确定性工具**; LLM 调用类步骤标记 `not_reproducible` 显式跳过
  (可见决策, 不混入 diff).
- PDB 支持是"分子结构输入"级别, 不是"蛋白科学能力"——宣传口径须守住此边界.

## 7. 落地记录 (as-built, 2026-10-16)

三条 P0 缺口全部落地, 治理面同步收口:

| 缺口 | 实现 | 不变量测试 | 治理锚点 |
|---|---|---|---|
| G-A | `rdkit_tool.mol_file_to_mol` (MOL/SDF/PDB; SDF 多记录逐条, 单条失败不拖垮整批; `.mdl`/`.ent` 别名; 未知后缀显式报错列支持集) | `tests/test_rdkit_mol_file.py` | `mol_structure_input` (always_on) |
| G-B | `huginn/tools/sci/spectra_parsers.py` 纯 stdlib 解析器 (JCAMP-DX 峰表+SQZ/DIF/DUP 压缩轨迹; mzML 64/32-bit float+zlib), `characterization_tool` 薄适配 `jdx_parse`/`mzml_parse`; 不支持形态显式 ValueError, 不静默 | `tests/test_spectra_parsers.py` | `spectra_parse` (always_on) |
| G-C | `huginn/autoloop/reproduce.py` 逐字段数值 diff (相对/绝对容差; bool≠int; hash 口径复用 provenance.capture) + LLM 步显式 `not_reproducible`; CLI `huginn/cli/reproduce.py` (--snapshot/--record/JSONL; 空文件防假绿; 退出码按全量 reproduced) | `tests/test_reproduce.py` | `reproduce_closed_loop` (always_on) |

治理台账同步:
- 能力锚点 `capability_floor` 棘轮 16→19 (三条 always_on 型: 无开关能力不硬凑
  开关, 锚点退化为「承载模块存在 + 不变量测试存在」双判据, flag_ok 恒 True);
- L1 精选 smoke `runtime_probe_floor` 棘轮 4→7 (`jdx_parse`/`mol_file_to_mol`/
  `reproduce_diff` 三条端到端);
- `governance-state.md` 确定性再生成; 镜像 (publish/mcpb) 同步漂移 0; ruff clean;
  相关测试 129 passed.

偏离 spec 的实现决策 (与 §4 对照):
- 谱图解析落在新文件 `huginn/tools/sci/spectra_parsers.py` (spec 允许二选一),
  `characterization_tool` 只做薄适配——判据单一来源, 测试/工具/CLI 共用一份解析器;
- 探针清单实际落在 `huginn/capabilities/smoke_specs.py` (L1 精选 smoke, 权威
  floor 所在), L0 依赖探针面由 rdkit 缺库时的既有 fail-open 机制另守.
