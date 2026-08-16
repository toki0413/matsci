# Research Note: 吸引子几何决定系统发现的辨识上限对 Huginn 的启发

> 状态: **report**（研究参考，非实现契约；据此驱动的设计变更需另立 spec/plan）
> 日期: 2026-08-16
> 关联: `dynamics_discovery_tool`（SINDy）/ `symbolic_regression_tool` / `validation.identifiability`
> 论文: Gallo, Anselmi & Lazzari, "Attractor Geometry Determines the Identifiability
> Limits of System Discovery" (arXiv:2607.18490, 2026) — preprint。

本备忘记录该论文对 Huginn Agent 的启发，供后续架构决策参考。**不杜撰、不承诺实现**，
每条启发标注"可直接落地 / 需设计 / 仅认知参考"。

---

## 1. 论文核心主张

> 符号发现/系统辨识（SINDy 稀疏回归、进化符号回归 PySR）的天花板不由**算法设计**或
> **数据量**决定，而由**吸引子的几何**决定——长期运行动力学允许恢复什么。
>
> 一个数字，**λ_min(M)**（不变测度矩矩阵的最小特征值），在运行任何算法之前，用一段
> **短参考轨迹**就能设定辨识上限：
>
> - **λ_min(M) → 0**：吸引子没有覆盖函数空间，恢复对任何算法（稀疏/组合）都不可行；
>   此轨迹上看似完美的拟合是**非唯一/几何病态**的，不代表真方程。
> - **λ_min(M) 增长**：稀疏回归与进化符号回归都随之改善。

方法论上用 **Lorenz-84** 做 within-system 设计：一个强迫参数驱动固定点/极限环/混沌，
控制方程与候选库固定，于是性能差异只能归因于吸引子几何。核心结论：

- **混沌抬高 λ_min(M)**（铺开吸引子 → 覆盖函数空间更多方向），但混沌也放大轨迹、
  放大噪声 → "更深的混沌并不均匀更好"。
- **噪声的算法依赖差异**：噪声进 SINDy 的回归瓶颈是**线性**的，进 PySR 的判别通道是
  **超线性**的 → 同一个混沌过渡可以让两种方法朝相反方向变化。
- **跨系统迁移**：机制分数（mechanistic score）无需重拟合就能迁移到 held-out
  Lorenz-96，说明是机制而非曲线拟合。
- 从方程直接读出**判据**可预测"何时再加混沌不会改善可辨识条件"。

第一问不是"用哪个算法"，而是"**吸引子允许恢复什么**"。

---

## 2. 与我们 agent 的对应

| 论文概念 | Huginn 现状 | 启发 | 可落地性 |
|---|---|---|---|
| λ_min(M) 预飞辨识天花板 | `dynamics_discovery_tool` 只返回 R²/系数，不知"这条轨迹几何上允不允许唯一恢复" | 系统发现前先算吸引子覆盖 → 给 R² 一个**几何可信度**背书 | **已落地**（见 §3） |
| 恒定/极限环吸引子 → 难辨识 | 阻尼振荡/弛豫类数据是工具常见输入，恰好是低覆盖区 | 高 R² 不背书唯一性；工具应明示"这是非唯一拟合" | **已落地**（deficient 分级） |
| 混沌抬高 λ_min 但放大噪声 | 无噪声适应度建模 | 噪声鉴别的算法依赖差异 | 需设计 |
| 机制分数跨系统迁移 | — | λ_min 可作为**无需重拟合**的域适配诊断 | 需设计 |
| "何时加混沌不再改善"判据 | — | 从方程读出的条件可预测加数据/加混沌是否值得 | 仅认知参考 |

---

## 3. 落地进展

- **`huginn/validation/identifiability.py`（2026-08-16）**：新增系统辨识预飞天花板模块。
  - `moment_matrix(Theta)`：不变测度矩矩阵 `M = Thetaᵀ Theta / N`（候选库在轨迹上的
    样本 Gram 矩阵），对应遍历平均 `∫φᵢφⱼ dμ`。
  - `identifiability_ceiling(Theta)`：取 M 最小/最大特征值，归一化 `λ_min_rel = λ_min/λ_max`
    分级 `adequate / limited / deficient`，另给 `coverage_ratio`（吸引子覆盖的库函数方向
    占比）。λ_min≈0 → `deficient` → 几何不可辨识。
  - `assess_trajectory(t, X, ...)`：与 `dynamics_discovery_tool._build_library` 一致的
    候选库构造 → 高层预飞入口。纯 numpy，无 scipy/sklearn 依赖，确定性可测。
  - **honest boundary**：λ_min 是**必要条件非充分**——低于它必然恢复不了，高于它也不保证
    恢复成功；归一化阈值是启发式（数量级分离经验值），非定理。
- **`dynamics_discovery_tool`（2026-08-16）**：
  - `discover` 结果新增 `identifiability` 块（λ_min / λ_min_rel / level / coverage_ratio /
    note）——agent 拿到 R² 的同时能判断"这个 R² 是否几何可信、方程是否唯一"。
  - 新增 `action="identifiability"`**预飞动作**：`metadata.preflight=True`、
    `no_regression_run=True`,不跑任何回归, 只返回天花板 + `recommended_action`
    （deficient 建议先补更广覆盖轨迹，再跑符号回归）。附诚实边界说明。
  - 失败静默降级（`ceiling=None`），不阻塞主流程。
- **概率校准拟合分 NLL/BIC 已落地（2026-08-16）**：`discover` 结果新增 `fit_metrics`
  块 —— 高斯独立加性噪声假设下的负对数似然 `NLL=(N/2)log(2πσ²)+N/2`（σ² 用残差 MLE）、
  `BIC=2·NLL+k·log(N)`（对非零项数 k 做稀疏惩罚）、`noise_std`、`active_terms`。
  回应论文"不同方法/噪声尺度不可比（SINDy 线性 vs 进化超线性）"的关切：
  R² 高不等于模型好，比较模型应看 BIC 而非单看 R²。`_fit_metrics` 代实现，
  失败不影响主流程。
- **测试（2026-08-16）**：`tests/test_identifiability.py`（9 passed）+ 
  `test_dynamics_discovery.py` 新增 NLL/BIC 用例（共 34 passed among discovery suite）——
  - 矩矩阵数值正确性；
  - 恒定轨迹 → `deficient`（λ_min_rel < 1e-6，覆盖比 ≈ 1/6）；
  - **Lorenz 混沌投影 λ_min_rel 显著高于恒定情形**（验证"混沌铺开吸引子抬高 λ_min"），
    且覆盖度更高；
  - `discover` 输出带 identifiability 块；
  - 预飞 action 不跑回归、带 recommended_action + honest_boundary；
  - NLL/noise_std 随数据噪声单调上升；active_terms 正确统计非零项。

---

## 4. 对 agent 认知/编排的启发（超出当前落地）

- **R² 的几何背书**：当前 `validation`/`grader` 常以预测 R² 判方程好坏。本论文提示应把
  **λ_min(M) 覆盖度**与 R² 并列——高 R² + low λ_min = "拟合好但不可辨识"，应降级为
  **认知不确定性**（agent 不应把这类方程当确证模型，>仅当启发）。
- **数据采集决策**：`action="identifiability"` 的 recommended_action 事实上给出一条
  agent 可执行的**主动数据策略**：不足时补更非线性/混沌/多初值轨迹，而不是反复调算法。
  这与已有 `active_learning_tool` / `enhanced_sampling_tool` 的"计算哪个实验最有信息量"
  叙事互补。
- **未来闭环**（需设计）：把 λ_min 或 coverage_ratio 接进 `failed_direction` 归因
  （环境缺口：域模型采集到了几何不可辨识的数据）与 `EvolutionManager` 认知更新通道，
  让 agent 在"反复 SINDy 拟合不佳"时意识到是数据/域问题而非算法调参问题。

---

## 5. 诚实边界与未落地项

- **阈值是启发式**：`deficient=1e-6 / limited=1e-3` 是数量级分离的经验定值，未经
  SINDy/PySR benchmark 标定。并列为 `level_thresholds` 参数，可随域调节。
- **未建模"何时加混沌不再改善"判据**——从方程读条件，需方程级输入，当前工具只吃
  轨迹数据。
- **仅覆盖多项式+sin/cos/exp 库**，与 `dynamics_discovery_tool` 一致；不覆盖任意符号库。
- **NLL 假设高斯 i.i.d. 噪声**——对重尾/异方差噪声不成立；未建模论文的"进化解超线性
  噪声演化"（已落地的是 SINDy 原点的高斯 NLL/BIC，非进化解判别通道的完整建模）。