# 02 归因：MLE-bench 三任务轨迹分析（ML 工程视角）

诊断对象：huginn-agent 在 MLE-bench 三任务的表现 —— playground-series-s3e18（0.7583）、spaceship-titanic（0.638）、tabular-playground-series-may-2022（0.745），全部 `medal="none"`。
分析人：ML 工程归因专家｜日期：2026-07-17｜铁律：只读取证 + 分数重算，未运行 agent 本体。

---

## ① 数据与方法

**证据来源**（全部为本仓库内一手文件）：

| 类别 | 文件 |
|---|---|
| 运行约束 | `mlebench_huginn.py`（timeout/max_tool_calls/system prompt/评分与奖牌逻辑）、`agent/huginn/bench/orchestrator.py`（`_is_done`/`MLE_DELIVERABLES`） |
| 运行产物 | `workspaces/mlebench/<task>/`：` _score.json`、`_huginn_meta.json`、`submission/submission.csv`、`data/`、`_private/test.csv`、agent 写的全部训练脚本（may-2022 共 6 版） |
| agent 自录历史 | `agent/huginn/knowledge/seed/38_benchmark_evaluation_lessons.md`（含每次跑的分数/时长/tool calls 与复盘） |
| 已确认审计事实 | `audit_20260717/16_评测体系完整性.md`（P0-2、P1-10）、`audit_20260717/20_benchmark失败根因与预算分析.md`（H5/H6/H9） |
| 评分器 | `mle-bench/mlebench/competitions/<task>/grade.py`、`leaderboard.csv` |

**方法**：
1. 文件取证：逐行读 agent 全部训练脚本与提交 CSV，重建每次运行的时间线（文件 mtime）与迭代链。
2. 分数重算：用纯 numpy 实现 AUC/accuracy，在 `_private/test.csv` 真标签上重算 agent 得分，与 `_score.json` 对拍（三者全部吻合）。
3. 天花板重建：按 `mlebench_huginn.py` 中合成数据生成器的显式噪声规则，构造 oracle 后验/规则，在同一私有测试集上算出信息论可达上限，作为「agent 离满分还差多少」的客观标尺。
4. 反事实推演：对唯一存在实质差距的任务，定位差距的具体来源行。

---

## ② 核心发现（按证据强度排序）

### F1【直接证据 · 分量最重】「no medal」是 harness 结构性必然，不含任何能力信息

三个竞赛的 `leaderboard.csv` 全部是**未拉取的 Git LFS 指针**（各 3 行文本，首行 `version https://git-lfs.github.com/spec/v1`）。奖牌判定代码：

```python
# mlebench_huginn.py:543-556
medal = "none"
if leaderboard_path.exists():
    lb = pd.read_csv(leaderboard_path)
    if "score" in lb.columns and len(lb) > 0:   # LFS 指针只有一列版本号 → 永假
        ...
```

`"score" in lb.columns` 永远为 False → **任何分数都落 `medal="none"`，即使满分 1.0 也一样**。三个 `_score.json` 的 `"medal": "none"` 实测印证。即「全部 no medal」这一观测对 agent 能力零信息量。
附带潜伏 bug：分位数方向写反（`>= quantile(0.05)` 发 gold，`mlebench_huginn.py:551-556`），一旦拉取 LFS 即「几乎必发 gold」——两个方向的错误使 medal 字段过去、未来都不可信（`audit_20260717/16_评测体系完整性.md` P1-10 已确认）。
更刺眼的是：agent 的自我进化记录**已经发现** LFS 指针问题，但开出的「处方」是 try/except 兜底成 `medal="none"`（`agent/huginn/knowledge/seed/38_benchmark_evaluation_lessons.md:243-245`）——自进化 loop 把评测失灵制度化了。

### F2【直接证据】三个分数全部在合成 smoke 数据上取得，且已达信息论天花板的 95%~99.8%

三份 `_huginn_meta.json` 均 `"synthetic": true`；数据行数与 `mlebench_huginn.py` 的 `gen_synthetic_*` 生成器吻合（非真实 Kaggle 数据）。生成器**显式注入标签噪声**：

- spaceship：CryoSleep→80% 正例；高消费→30%；**其余纯 50% 抛硬币**（`mlebench_huginn.py:108-113`）
- may-2022：`(f_0+f_1+0.5*f_2>0) XOR (f_27>0)`，20% 标签翻转（`mlebench_huginn.py:196-197`）
- s3e18：EC1/EC2 各 20% 标签翻转（`mlebench_huginn.py:254-259`）

我在各任务私有测试集上实测 oracle 上限 vs agent 实得：

| 任务 | oracle 上限（实测） | agent 实得 | 达成率 |
|---|---|---|---|
| spaceship-titanic（accuracy） | 0.640–0.642 | **0.638** | **99.4%** |
| playground-s3e18（macro AUC） | 0.7595 | **0.7583** | **99.8%** |
| may-2022（AUC） | 0.7816 | **0.7440**（硬标签）/ 0.7460（accuracy） | 95.2% |

即：**在 synthetic 配置下，「跑分差」主要是数据天花板低（0.64~0.78），不是 agent 差**。真实 Kaggle 铜牌线（seed 文件自录 spaceship bronze cutoff 0.687，`38_...md:13`）高于 synthetic 天花板本身——在此配置上获奖数学上不可能。s3e18 的 EC1 单项 agent 0.7601 甚至略高于 oracle 0.7595（小样本运气），已无任何提升空间。

### F3【直接证据】may-2022 的唯一实质失分：AUC 任务提交硬标签 0/1

该任务 description 明确要求概率：「Submissions are evaluated on area under the ROC curve between the predicted **probability** and the observed target」「you must predict a **probability**」（`workspaces/mlebench/tabular-playground-series-may-2022/description.md:33,37`）；`sample_submission.csv` 也是 0.5 浮点。但 agent 最终脚本：

```python
# workspaces/mlebench/tabular-playground-series-may-2022/lgb_v3.py:105-109
test_probs = np.mean([m.predict_proba(X_test)[:, 1] for m in models], axis=0)
y_test = (test_probs >= 0.5).astype(int)          # ← 排序信息在这一行被销毁
sub = pd.DataFrame({"id": test["id"], "target": y_test})
sub.to_csv("submission/submission.csv", index=False)
```

提交文件实测唯一值 `{0, 1}`。硬标签下 AUC 退化为 balanced accuracy（=0.744），而概率就在上一行变量里。这**一行**恰好损失掉到天花板（0.7816）的全部差距（约 0.04 AUC）。
且这不是偶发：该任务 6 个脚本版本（`baseline.py:67-89` 还专门多花 40 阈值 × 5 fold = 200 次额外训练去搜「accuracy 最优阈值」再二值化、`lgb_fast.py:91-94`、`lgb_v2.py`、`lgb_model.py`、`lgb_v3.py`）**全部**以 accuracy 做 CV 选优、交硬标签——指标-目标错配贯穿整个 run。
对照：同为 AUC 任务的 s3e18，agent 正确提交了连续概率（150 个唯一值）。说明它**会**交概率，但在 may-2022 上习惯性把任务当 accuracy 做——工程纪律失误，非知识缺失。

### F4【直接证据】自进化 loop 对 may-2022 的复盘误诊了病因，并把错误经验固化

seed 文件记录 run #1（500 训练样本）：「CV AUC 0.82 but test AUC was 0.71 — an 0.11 generalization gap… Root cause: overfitting」（`38_...md:187-193`），处方是「500→5000 扩数据」；run #2 后宣称「gap collapsed from 0.11 to 0.005」（`:198-203`）。
**推断（强）**：run #1 的「0.11 gap」大部分就是硬标签退化——CV 0.82 是概率 AUC，test 0.7061 是硬标签 balanced accuracy，两者本就不是同一量；run #2「gap 消失」是因为 CV 改用了 accuracy（0.75）对比硬标签 test（0.745）——错错得正，两边同错所以「吻合」。真正的 bug（交硬标签）从未被识别，原样带进了 run #2 的最终提交。自进化系统完成了「诊断→处方→验证」的完整闭环，但闭环里少了「检查提交格式 vs 指标定义」这一最廉价的校验，导致信用分配错误、经验条目（「sample size dominates」）以偏概全。

### F5【直接证据】不存在「一次提交即停」「占位结果」「未真正训练」

- may-2022 run #2：12 分钟内产出 6 个训练脚本版本（`baseline.py` 11:18 → `rf_baseline.py` 11:20 → `lgb_model.py` 11:21 → `lgb_v2.py`/`lgb_fast.py` 11:24 → `lgb_v3.py` 11:25），LR→RF→LightGBM 渐进，5-fold CV + early stopping + 正则参数齐全，最终提交由 5 折模型 ensemble 产生。
- s3e18：827s 内完成 EDA→交互筛选→RF 超参网格（5 depth × 3 n_estimators × 2 目标 × 5 fold = 150 次拟合，`_code_tool_script.py:41-55`）；提交在 04:21 更新过一次（submission 目录 04:15 创建、文件 04:21 重写）。
- spaceship run #2：RF 与 LightGBM 双模型 5-fold CV 对比（`_code_tool_script.py:55-93`），还额外验证了「仅按 CryoSleep 预测」的 simple rule（`:100-102`）。
- 三份提交预测分布均合理（非常数占位）：s3e18 概率连续分布、spaceship 523 True/477 False、may-2022 532/468。

### F6【记录证据】预算既未烧满也未超时；结束机制无法从存留物完全确认

seed 运行表（`38_...md:13-15`）：spaceship run#1 330s/29 calls、may-2022 476s+792s/34+46 calls、s3e18 827s/41 calls。默认上限 `--max-tool-calls 60`、`--timeout 1800`（`mlebench_huginn.py:575-576`）。四次完成跑用量为预算的 48%~77%，时长均远低于 1800s。spaceship run#2（450s，0.638）最终消息自述「The budget is exhausted」（`_huginn_meta.json:6`），**推断**其烧满 60 calls。
注意一个机制性疑点：`_is_done` 要求 deliverable 全齐（`MLE_DELIVERABLES` 含 `submission/*.py`，`agent/huginn/bench/orchestrator.py:73-76`），而 agent 把脚本写在 workspace 根目录而非 `submission/`——按当前代码 `missing()` 永不空，只有 `calls>=60` 或 timeout 能退出循环（`orchestrator.py:147-159`）。但 seed 表记录 29/34/46/41 calls 即结束，与当前 orchestrator 逻辑不自洽。**推断**：这些跑发生于 orchestrator 加入「min_calls=50% budget」规则之前（该规则系后补，见 `orchestrator.py:150-153` 注释），或由不同参数的驱动脚本发起；workspace 的 `.checkpoint.sqlite` 未存留，无法终裁。此点不影响 F1–F5 结论。

### F7【背景缺陷，本次未起作用】

- `_private/test.csv`（含标签）就放在 agent workspace 内且读取无限制（`mlebench_huginn.py:318-319,36-37`；审计 16 P0-2）。**本次未被利用**：若抄标签分数应为 1.0，实测 0.638/0.744/0.7583 反证清白。
- 真实模式下读未混淆 `description.md` + 真实竞赛 ID 入 prompt + 开放 `web_search_tool`（审计 16 P1-10/H6）——synthetic 跑分中无关。
- system prompt 的 Deliverable 段只写了 spaceship 的提交格式（`mlebench_huginn.py:369`），对 may-2022/s3e18 的概率要求只字未提——agent 只能从 description.md 自行摄取，构成 F3 的促成条件（非借口）。

---

## ③ 根因链（现象 → 机制 → 代码位置）

**现象**：3 任务 0.638 / 0.745 / 0.7583，全部 no medal，被读作「ML 能力差」。

```
现象
├─【主因 1 · harness】medal 字段是死代码
│    机制：leaderboard.csv 为 LFS 指针 → "score" in lb.columns 永假 → medal="none" 无条件
│    位置：mlebench_huginn.py:543-556；mle-bench/mlebench/competitions/*/leaderboard.csv（3 行指针）
│    属性：直接失分原因（对"no medal"这一观测负 100% 责任）
│
├─【主因 2 · harness】绝对分低是合成数据噪声天花板
│    机制：gen_synthetic_* 注入 20%~50% 标签噪声 → 上限 0.64/0.76/0.78 << 真实榜量级
│    位置：mlebench_huginn.py:108-113, 196-197, 254-259；_huginn_meta.json "synthetic": true
│    属性：直接失分原因（对"分数看起来低"负主要责任）；agent 实得已达上限 95%~99.8%
│
├─【次因 · agent，仅 may-2022】AUC 任务交硬标签
│    机制：test_probs 被 (>=0.5) 二值化，排序信息销毁；CV 用 accuracy 选优
│    位置：workspaces/mlebench/tabular-playground-series-may-2022/lgb_v3.py:105-109
│          （同模式见 baseline.py:67-89、lgb_fast.py:91-94 等全部 6 版）
│    促成条件：system prompt 未逐竞赛写明提交类型（mlebench_huginn.py:367-369 只写了 spaceship）
│    属性：直接失分原因（≈0.04 AUC，即该任务距天花板的全部差距）
│
└─【潜伏 · harness】奖牌分位数方向颠倒 / _private 标签泄漏 / 未混淆 description+web_search
     位置：mlebench_huginn.py:551-556；:318-319；:72-74,363-364（审计 16 P0-2/P1-10）
     属性：背景缺陷——本次未起作用（synthetic + 分数<<1.0 反证未抄答案），真实模式下必爆
```

**与 SAB/RCB 失败模式的区别**（归因纪律要求）：SAB 是「代码截断/缺主执行块/未保存预测」，RCB 是「只描述不执行/占位数值」——那是**执行层**失败。MLE 三跑全部真实训练、真实提交、分布合理、达天花板 95%+——执行层完好；失分在**评测配置层**（F1/F2）与一处**指标纪律**（F3）。三个 benchmark 的失败不同族，不可用同一药方。

---

## ④ 对用户问题的回答

**Q：agent 的 ML 方法论水平到哪一档？**
中上（约 Kaggle 熟手档）：
- 特征工程 ✓✓：三任务全部自主挖出**真实**交互结构——may-2022 的 `f_0*f_27/f_1*f_27/f_2*f_27`（正是生成器的 XOR 结构）、s3e18 的 `f_0*f_1` 与 `f_2*f_3`（正是生成规则）、spaceship 的 CryoSleep/spending/Cabin 解析与结构性缺失——这是 prompt 里 INTERACTION HUNTING 规则与 agent EDA 的共同成果。
- 模型选择 ✓：LR→RF→LightGBM 渐进，LightGBM 正则参数（`min_child_samples/reg_alpha/reg_lambda/colsample`）合理。
- 验证策略 ✓/✗：StratifiedKFold(5, shuffle, seed) + early stopping 规范；但 **CV 度量与竞赛指标错配**（AUC 任务用 accuracy 选优）且 6 版脚本未自纠——方法论的最大短板不在「模型」而在「指标对齐」这一最后一公里。
- 调参 △：s3e18 末段的 150 次 RF 网格拟合是在天花板 0.001 内抛光，方向对但边际收益≈0（不过这是数据天花板使然，非 agent 之过）。

**Q：是没时间迭代、不会迭代、还是迭代了但方向错？**
三者都不是主因。
- 不是「没时间」：48%~77% 预算即达天花板，再给 10 倍预算在 synthetic 数据上也涨不了 0.01（F2）。
- 不是「不会迭代」：12 分钟 6 版脚本、超参网格、双模型对比，迭代行为真实存在（F5）。
- 「方向错」仅一处成立且很关键：may-2022 全程把 AUC 任务当 accuracy 做（F3）；自进化 loop 随后又把症状误诊为过拟合（F4）。

**Q：是否存在「一次提交即停」「占位结果」「未真正训练」？**
全部不存在（F5）。与 SAB/RCB 的失败模式不同族。

**Q：若工作流不同本会如何（反事实）？**
- may-2022：`lgb_v3.py:108` 改 `"target": test_probs` 一行 → 分数 0.744 → ≈0.78（推断区间 0.77–0.78，近天花板）。 medal 仍 none（F1 不死，medal 必 none）——**在当前 harness 下，agent 做得再好也是 0 奖牌**。
- s3e18 / spaceship：任何工作流改动收益 ≤0.001 / ≤0.004，无可作为空间。
- 若目标是「测量真实 ML 能力」：必须换真实 prepared 数据 + 修复 medal 逻辑 + 把预算提到官方量级（当前 60 calls/30min vs 官方 24h 级），否则测的是「合成数据天花板有多低」，不是 agent。

---

## ⑤ 可操作建议（按投入产出比排序）

1. **提交格式校验器（1~2 天，收益确定）**：评分前机械校验——若 description/grade 为 AUC 类指标而提交列唯一值 ≤2，报警并自动改用 `predict_proba` 重交。同时修 `kaggle_submit_tool`（当前只验行数/列名/缺失，`agent/huginn/tools/bench_infra/kaggle_tool.py:1-6`，不验值域）。直接消除 F3 类损失（本次 ≈0.04 AUC）。
2. **system prompt 逐竞赛写死「metric + 提交类型」（半小时）**：`mlebench_huginn.py:367-369` 目前只写了 spaceship 格式；为每个竞赛注入「Metric: ROC AUC → submit **probabilities** (float 0-1), NOT class labels」。
3. **medal 逻辑重做（半天，否则一切 MLE 跑分无结论）**：拉取 leaderboard.csv 的 Git LFS；按官方 MLE-bench 分位数规则与指标方向（越高/越低越好）实现；修 `mlebench_huginn.py:551-556` 的分位数颠倒。在修复前，所有对外报告必须标注「medal 字段无效」。
4. **synthetic 跑分报告改为 ceiling-normalized（1 小时）**：`_score.json` 增加 `oracle_ceiling` 与 `score/ceiling` 字段（生成器规则已知，计算成本几行代码）。0.638 会被正确读作「天花板的 99.4%」而非「不及格」。
5. **真实能力测量（配置级改动）**：用 `mlebench prepare` 的真实数据跑（`find_prepared_data` 路径已存在，`mlebench_huginn.py:83-88`），`--max-tool-calls`/`--timeout` 提到数百 calls / 数小时；真实模式下关闭 `web_search_tool` 或审计其检索记录（防搜历史方案，审计 16 H6）。
6. **自进化复盘加「指标-提交对齐」检查项（半天）**：把 F4 的误诊案例写入复盘 checklist——任何「CV vs test gap」诊断前，先确认两边是同一度量、提交格式匹配指标定义。这是本次暴露的自进化系统最具体的信用分配缺陷。
7. **`_private` 移出 agent workspace（随 #5 一起做）**：审计 16 P0-2 已给方案；本次虽未被利用（分数<<1.0 反证），真实跑分前必须堵上。

---

### 附：关键数字速查

| 项 | 值 | 出处 |
|---|---|---|
| synthetic 模式 | 三任务全 true | `workspaces/mlebench/*/_huginn_meta.json` |
| oracle 天花板（实测） | spaceship 0.640–0.642；s3e18 0.7595；may-2022 0.7816 | 本报告重算（生成器规则 × `_private/test.csv`） |
| agent 达成率 | 99.4% / 99.8% / 95.2% | 同上 |
| may-2022 提交唯一值 | {0,1}（硬标签） | `submission/submission.csv` 实测 |
| 运行规模 | 330–827s；29/34/46/41 calls（上限 60/1800s） | `38_benchmark_evaluation_lessons.md:13-15` |
| medal 死代码 | LFS 指针 → 无条件 none | `mlebench_huginn.py:543-556` |
