# AI4Science Model Toolkit — OneScience

Distilled from OneScience (gitee.com/onescience-ai/onescience, Apache 2.0, 2025).
OneScience 是超算互联网 (scnet.cn) 生态下的 AI4Science 模型工具包, 覆盖 5 个领域,
支持 GPU 和海光 DCU 双平台. 装包按领域分: `bash install.sh {earth|cfd|bio|matchem}`,
不指定领域则全装.

对 huginn 的相关性排序: 材料化学 (主战场) > CFD ≈ 结构力学 > 地球科学 > 生物信息.

## 0. 平台与安装

### 双平台支持

- GPU: 标准 CUDA + PyTorch 路径
- 海光 DCU: 通过 DTK (DTK 26.3+) 工具链, 走 `module load sghpc-mpi-gcc/26.3` 激活
- 代码层统一: 同一份 `onescience.models.*` 在两种后端上跑, 模型代码不分支

### SCNET 集群安装流程

```bash
module load sghpcdas/25.6        # 激活 conda
conda init bash && source ~/.bashrc
module load sghpc-mpi-gcc/26.3   # 激活 DTK
conda create -n onescience311 python=3.11 -y
conda activate onescience311
cd onescience/
bash install.sh matchem          # 只装材料化学, huginn 主战场
```

### 安装自检 (UNet 烟雾测试)

```python
import torch
from onescience.models.unet import UNet
inputs = torch.randn(1, 1, 96, 96, 96).cuda()
model = UNet(in_channels=1, out_channels=1, model_depth=5,
             feature_map_channels=[64,64,128,128,256,256,512,512,1024,1024],
             num_conv_blocks=2).cuda()
x = model(inputs)  # 输出 shape 应与 inputs 相同
```

### 在线试用入口

- 应用入口: https://www.scnet.cn/ui/mall/app (大多数模型可在线试用)
- 数据集入口: https://www.scnet.cn/ui/mall/search/goods?common1=DATA&common2=DATA-330
- 私有模型托管: 平台提供共有/私有模型托管服务

### 第三方依赖

- NVIDIA NeMo (Apache 2.0) — 用于部分生物信息与语言模型路径
- 详见仓库 NOTICE 文件

## 1. 材料化学 (AI for Materials Chemistry) — huginn 主战场

### 模型与数据集

| 问题类型 | 案例 | 架构 | 数据集 | 备注 |
|---|---|---|---|---|
| 通用原子尺度模拟 | UMA | 等变 GNN | OC20 + OMat24 + OMol25 + ODAC23 + OMC25 聚合 | 单模型多任务, 替代专项势函数 |
| 原子间势拟合 / 原子尺度模拟 | MACE | E(3)-等变 GNN | MPTrj, SPICE, OMat24 | 高精度可迁移, 已在 seed/11 记录 |

### 数据集详情 (材料化学)

| 数据集 | 规模 | 内容 | 用途 |
|---|---|---|---|
| OC20 | 百万级 DFT | 催化剂吸附与反应 | 表面催化反应能垒预测 |
| OMat24 | 百万级 (Meta 2024) | 无机材料 DFT | 性质预测, 对标 MPTrj |
| OMol25 | 大规模 (2025) | 分子 DFT | 有机分子构型空间 |
| ODAC23 | — | 催化反应路径与过渡态 | 反应动力学, 过渡态数据稀缺 |
| OMC25 | — | 多组分材料 DFT | 合金/掺杂体系 |
| MPTrj | Materials Project MD | MD 轨迹 | MACE 默认训练集 |
| SPICE | 高精度 | 分子与蛋白量子计算 | 双精度 DFT+DLPNO |

### 数据集基类 (代码层)

- `AseAtomsDataset` / `AseDBDataset` — ASE 原子对象封装, 兼容 `ase.Atoms` + `ase.db`
- `FairChemDataset` — FairChem 统一接口, OC20/OMat 系列共用, huginn 接入时优先用这个接口

### UMA 的数学基础

等变 GNN 的核心是 E(3) 群作用下的等变性:
- 输入原子坐标 $\mathbf{x}_i \in \mathbb{R}^3$, 经过 SE(3) 等变消息传递
- 输出能量 $E$ 是标量 (不变), 力 $\mathbf{F}_i = -\nabla_{\mathbf{x}_i} E$ 是向量 (等变)
- 单模型多任务: 同一个 backbone 输出多个 head (能量/力/应力/磁矩), 替代训练多个专项 MLP

### 对 huginn agent 的启示

1. **UMA 路线** — 单模型替代多个专项 MLP. 用户跨体系 (合金→分子→表面) 时,
   优先推荐 UMA. 但 UMA 推理成本高, 小体系仍用 MACE.
2. **数据集体量** — OC20/OMat24 都是百万级. agent 不能直接全量训练, 应走 fine-tune
   预训练 checkpoint. huginn 的 `train_pot` 工具应支持 `--from_pretrained uma_sm`.
3. **ODAC23 的特殊价值** — 过渡态数据稀缺, 用户做反应动力学时, ODAC23 是少数可直接
   用的开源过渡态集. agent 在 plan 阶段应主动检索这个数据集.
4. **FairChemDataset 接口** — huginn 的数据摄入 hook 应支持 FairChemDataset 格式,
   一次接入覆盖 OC20/OMat 全家.
5. **首性原理一致性** — 等变 GNN 严格满足旋转/平移对称性, 符合 user_profile 的
   "符合第一性原理的 ML" 偏好. 优先于 GAN/AE 等非对称架构.

## 2. 计算流体 (AI for CFD) — 扩展相关

### 模型与数据集 (12 个案例)

| 问题类型 | 案例 | 架构 | 数据集 |
|---|---|---|---|
| 汽车气动设计 | Transolver-Car-Design | Transformer | Shape-Net Car |
| 翼型设计 | Transolver-Airfoil-Design | Transformer | AirfRANS |
| 圆柱绕流 | MeshGraphNets | GNN | DeepMind 旋涡脱落 |
| 任意 2D 几何绕流 | DeepCFD | U-Net | DeepCFD |
| PDE 求解模型集 | PDENNEval | 多模型 | PDEBench + 自生成 |
| 物理驱动 PDE | PINNsformer | PINN (Transformer) | — |
| 不可压流体 | CFDBench | 多模型 | CFDBench (顶盖驱动/管道/坝/圆柱) |
| 复杂边界椭圆 PDE | BENO | Transformer + GNN | BENO |
| 拉格朗日网格 | lagrangian_mgn | GNN | DeepMind 拉格朗日 |
| CFD Benchmark | CFD_Benchmark | 多模型 | 多种数据集 |
| 湍流 | EagleMeshTransformer | Transformer | Eagle 无人机 (110万网格, 600场景) |
| 拓扑优化 | GP_for_TO | Gaussian Processes | — |

### CFD 数据集详情

| 数据集 | 简要描述 |
|---|---|
| Shape-Net Car | 汽车几何数据, 空气动力学优化 |
| AirfRANS | 翼型流场仿真, 翼型设计优化 |
| DeepMind 旋涡脱落 | 圆柱绕流时序, GNN 训练 |
| DeepCFD | 管道流 CFD 解, 含速度和压力场 |
| PDEBench | 多种 PDE 方程的大规模基准 |
| CFDBench | 顶盖驱动方腔流/管道流/坝流/圆柱扰流 |
| Eagle 无人机 | 大规模湍流, 110 万二维网格, 600 场景 |
| BENO | 复杂边界椭圆 PDE |

### 关键架构的数学基础

- **Transolver** — 通用几何 Transformer, 不依赖网格拓扑. Attention 在 latent space
  上做, 输入几何通过点云编码. 跨几何迁移能力强, 适合参数化设计.
- **PINNsformer** — PINN 从 MLP 升级到 Transformer. 损失函数仍是
  $\mathcal{L} = \mathcal{L}_{PDE} + \lambda_{BC}\mathcal{L}_{BC} + \lambda_{IC}\mathcal{L}_{IC}$,
  但用 self-attention 替代 MLP 做函数逼近, 长时序 PDE 更稳定.
- **MeshGraphNets** — 在网格图上做消息传递, 节点是网格点, 边是网格连接.
  输出下一时刻的速度/压力增量, 走 roll-out 推理.
- **GP_for_TO** — 拓扑优化用高斯过程, 完全可解释, 符合 user_profile 的可解释 ML 偏好.

### 对 huginn 的启示

1. **Transolver 系列** — 通用几何 Transformer, 跨几何迁移能力强. huginn 用户做
   参数化设计 (不同翼型/车身对比) 时, Transolver 比 OpenFOAM 快几个数量级.
2. **PINNsformer** — 把 PINN 从 MLP 升级到 Transformer, 长时序 PDE 更稳定.
   seed/07 (OpenFOAM) 的传统 CFD 用户可考虑 PINN 替代.
3. **GP_for_TO** — 拓扑优化用高斯过程, 符合 user_profile "可解释 ML 优先" 偏好,
   优先于神经网络的拓扑优化方法.
4. **CFD_Benchmark** — huginn bench 层可考虑接入 CFD_Benchmark 作为 agent CFD 能力评测集,
   补充当前以材料为主的 bench 覆盖.

## 3. 结构力学 (AI for Structural) — 扩展相关

### 模型

| 问题类型 | 案例 | 架构 | 数据集 |
|---|---|---|---|
| 经典弹塑性力学 | DEM_for_plasticity | PINN | — |
| 2D 平面应力 | Plane_Stress | PINN | — |

### 结构力学数据集 (README 列出, 案例未直接引用)

| 数据集 | 描述 |
|---|---|
| VortexShedding | 涡旋脱落流动数据 |
| VortexSheddingRe300-1000 | 指定雷诺数范围的涡旋脱落 |
| Stokes | Stokes 流动数据 |
| AhmedBody | 车身气动流动数据 |
| DrivAerNet | 汽车空气动力学数据 |
| Lagrangian | 粒子追踪数据 |
| BistrideMultiLayerGraph | 多层图结构数据 |

### PINN 在结构力学中的数学结构

损失函数:
$$\mathcal{L} = \frac{1}{N_{PDE}}\sum_i |r_\theta(\mathbf{x}_i)|^2 + \lambda_{BC}\mathcal{L}_{BC} + \lambda_{IC}\mathcal{L}_{IC}$$

其中 $r_\theta = \mathcal{N}[u_\theta](\mathbf{x})$ 是 PDE 残差, $u_\theta$ 是神经网络逼近的解.

弹塑性问题难点: 屈服面是不光滑的, 导致 PINN 收敛困难. 常用域分解 (XPINN) 或
光滑化屈服面 (regularized yield surface) 缓解.

### 对 huginn 的启示

- 结构力学 AI4Science 目前只有 PINN 路线, 没有数据驱动大模型.
  seed/06 (Abaqus) 用户如果要做 AI 增强仿真, PINN 是唯一选项.
- PINN 在弹塑性问题上收敛困难, 需要域分解 (XPINN). huginn 应在 plan_check 阶段
  检测"弹塑性 + PINN"组合并给出 XPINN 建议.
- DrivAerNet / AhmedBody 是开源汽车气动数据, huginn 用户做车身优化时可优先推荐.

## 4. 地球科学 (AI for Earth Science) — 弱相关, 架构可参考

### 模型与数据集 (9 个案例)

| 问题类型 | 案例 | 架构 | 数据集 |
|---|---|---|---|
| 降尺度 | CorrDiff | U-Net + Diffusion | ERA5, HRRR |
| 中期天气预报 | FourCastNet | AFNO | ERA5, TJWeather |
| 中期天气预报 | GraphCast | GNN | ERA5, TJWeather |
| 中期天气预报 | Pangu | 3D Transformer | ERA5, TJWeather |
| 短临降雨 | NowCastNet | GAN | MRMS |
| 中期天气预报 | FengWu | 3D Transformer | ERA5, TJWeather |
| 中长期天气预报 | Fuxi | 3D Transformer | ERA5, TJWeather |
| 中短期天气预报 | Xihe | 3D Transformer | CMEMS, TJWeather |
| 短中期海洋预报 | Oceancast | AFNO | EMCMS, CMEMS |

### 地球科学数据集

| 数据集 | 描述 |
|---|---|
| ERA5 再分析 | 全球高分辨率再分析, 温度/风/湿度, 多气压层 |
| TJWeather 中科天机 | 全球/中国区域高分辨率模拟, 160 种气象要素 |
| HRRR | 美国 3km 分辨率快速更新数据, 降尺度训练 |
| CWB 台湾 | 台湾地区雷达和卫星数据, 短临预报 |
| MRMS 多雷达 | 高时空分辨率降水估计 |
| EMCMS 海洋 | 海浪、风速等海洋气象 |
| CMEMS | 海温、海流、盐度等海洋环境 |
| SyntheticWeather | 合成天气数据, 测试用 |

### 关键架构的数学基础

- **AFNO (Adaptive Fourier Neural Operator)** — 频域 global attention, 复杂度
  $O(N\log N)$ 替代标准 attention 的 $O(N^2)$. 在频域做可学习的滤波, 等价于
  全局卷积. 适合处理大网格场数据.
- **GraphCast 的 GNN 路线** — 球面网格上的消息传递, 节点是网格点, 边是多跳邻居.
  思路可迁移到晶格振动 (phonon) 传播.
- **3D Transformer (Pangu/FengWu/Fuxi/Xihe)** — 把大气层当作 3D 体素, 时空
  attention 联合建模. Pangu 是华为, FengWu 是上海 AI Lab, Fuxi 是复旦, Xihe 是
  中山大学. 架构相似, 数据预处理和训练策略不同.
- **CorrDiff** — U-Net 做低分辨率→高分辨率映射, Diffusion 在 U-Net 输出上做细化.
  两阶段生成, 比纯 Diffusion 快.

### 对 huginn 的启示

- **AFNO 可迁移性** — 频域 global attention, 材料科学里的声子场/应力场可用.
  huginn 用户做大网格场数据时, AFNO 是比标准 Transformer 更高效的选择.
- **GraphCast 的 GNN 路线** — 球面网格的消息传递, 思路可迁移到晶格振动传播.
  对声子色散关系预测有参考价值.
- **3D Transformer 同质化** — Pangu/FengWu/Fuxi/Xihe 架构相似, 数据策略不同.
  huginn 不需要全部接入, 选一个 (推荐 Pangu, 开源最完整) 作为气象预测代表.

## 5. 生物信息 (AI for Biology) — 弱相关, 仅记录

### 模型与数据集 (8 个案例)

| 问题类型 | 案例 | 架构 | 数据集 |
|---|---|---|---|
| 蛋白质结构预测及设计 | AlphaFold3 | Pairformer + Diffusion | mmseqsDB, AF3 官方, PDB, UniRef, BFD, MGnify |
| 蛋白质结构预测及设计 | Protenix | Transformer + Diffusion | Protenix 官方 |
| 蛋白质语言模型/结构预测/反向折叠 | ESM | Transformer + ESMFold + GVP | UniRef, PDB, ESM Atlas |
| 蛋白质骨架设计 | RFdiffusion | Diffusion | — |
| 蛋白质骨架到序列设计 | ProteinMPNN | MPNN | — |
| 蛋白质设计及优化 | PT-DiT | Diffusion + Transformer | — |
| 突变预测/外显子分类/基因必要性 | Evo2 | StripedHyena2 | OpenGenome2 (2.5TB) |
| 药物设计 | MolSculptor | Autoencoder + Latent Diffusion | — |

### 生物信息数据集

| 数据集 | 描述 |
|---|---|
| AlphaFold3 数据集 | 蛋白质及生物大分子结构 |
| OpenGenome2 (2.5TB) | 大规模基因组序列 |
| Protenix 数据集 | 蛋白质-配体结构 |
| RFdiffusion 数据集 | 蛋白质骨架生成 |
| ProteinMPNN 数据集 | 蛋白质序列设计 |
| ProteinDataset | 蛋白质数据集基类 |
| GenomeDataset | 基因组数据集基类 |
| MultimerDataset | 蛋白质多聚体数据 |
| UnifiedDataset | 跨模态统一数据处理管道 |

### 关键架构说明

- **AlphaFold3 Pairformer** — 比 AlphaFold2 的 Evoformer 多了 pair-specific attention,
  能建模多链和配体. Diffusion 头替代了结构模块的迭代精修.
- **ESM (Evolutionary Scale Modeling)** — 蛋白质语言模型, 用 UniRef 训练.
  ESMFold 比 AlphaFold2 快 60 倍, 但精度略低.
- **Evo2 (StripedHyena2)** — 基因组基础模型, StripedHyena 架构 = Transformer +
  Hyena (长卷积) 混合. 2.5TB OpenGenome2 训练, 处理超长 DNA 序列.
- **MolSculptor** — 药物设计, Autoencoder 学分子 latent space, Latent Diffusion
  在 latent space 生成. 思路可迁移到材料结构生成.

### 对 huginn 的启示

- AlphaFold3 的 Pairformer + Diffusion 架构对材料原子结构生成有参考价值,
  但 huginn 当前不覆盖生物分子. 如未来扩展到生物材料 (seed/31), 可参考.
- MolSculptor 的 Latent Diffusion 思路可迁移到材料逆向设计: 用 VAE 学材料
  结构 latent, 在 latent space 做 Diffusion 生成新结构. 但 VAE 不符合
  user_profile 的可解释偏好, 应谨慎.
- Evo2 的 StripedHyena2 (Transformer + Hyena 混合) 处理超长序列, 思路可迁移
  到长周期晶体结构的建模.

## 6. 跨领域架构总结

| 架构 | 适用问题 | 代表模型 | 数学基础 | 对 huginn 的可迁移性 |
|---|---|---|---|---|
| **等变 GNN (E(3)-equivariant)** | 原子势函数, 分子 | MACE, UMA | SE(3) 群等变消息传递 | 直接用 (材料主战场) |
| **Transformer** | 通用几何, PDE | Transolver, Pangu, FengWu | Self-attention, 位置编码 | 翼型/车身设计可迁移 |
| **Diffusion** | 生成任务 | CorrDiff, RFdiffusion, MolSculptor | Score matching, 反向 SDE | 材料结构生成可探索 |
| **PINN** | 物理驱动 PDE | PINNsformer, DEM_for_plasticity | PDE 残差损失 + BC/IC | 结构力学首选 |
| **GNN (消息传递)** | 网格数据 | MeshGraphNets, GraphCast | 图卷积, 邻居聚合 | CFD 网格可迁移 |
| **AFNO (傅里叶神经算子)** | 大规模场数据 | FourCastNet, Oceancast | 频域全局卷积, $O(N\log N)$ | 声子/应力场可探索 |
| **GAN** | 短临预测 | NowCastNet | 对抗训练 | 不推荐 (不可解释, 违背 user_profile) |
| **GP** | 拓扑优化 | GP_for_TO | 高斯过程回归, 后验方差 | 可解释, 符合偏好 |
| **Autoencoder + Latent Diffusion** | 药物/分子生成 | MolSculptor | VAE + Diffusion | 材料结构生成可探索, VAE 谨慎 |
| **StripedHyena2** | 超长序列 | Evo2 | Transformer + Hyena 长卷积 | 长周期晶体可探索 |

## 7. 对 huginn 工作流的集成建议

### 7.1 材料化学 (优先级最高)

1. **UMA/MACE 工具封装** — 当前 seed/11 只记录了 MLP 框架, 没有具体模型.
   huginn 应在 `tools/` 下新增 `uma_inference` 和 `mace_inference` 工具,
   支持 `--from_pretrained` 加载 OneScience checkpoint.
2. **FairChemDataset 数据摄入** — huginn 的 `auto_ingest` 应支持 FairChemDataset
   格式, 一次接入覆盖 OC20/OMat 全家.
3. **ODAC23 过渡态检索** — 在 plan 阶段, 当用户做反应动力学时, agent 应主动检索
   ODAC23 数据集, 而非让用户手动指定.
4. **first-principles 一致性** — 等变 GNN 严格满足旋转/平移对称性, 优先于 GAN/AE.
   huginn 的 ML 推荐器应把"是否满足对称性"作为筛选条件.

### 7.2 CFD (优先级中)

1. **Transolver 作为 OpenFOAM 加速替代** — 小数据/快速设计阶段用 Transolver,
   最终验证用 OpenFOAM (seed/07). huginn 的 plan 阶段应根据精度需求自动选择.
2. **PINNsformer 替代 PINN-MLP** — 长时序 PDE 时, PINNsformer 比 PINN-MLP 稳定.
   huginn 的 `pinn_solver` 工具应支持 `--backbone {mlp,transformer}`.
3. **GP_for_TO 接入** — 拓扑优化任务优先推荐 GP 路线, 符合 user_profile 可解释偏好.
4. **CFD_Benchmark 接入 bench** — huginn bench 层可考虑接入 CFD_Benchmark 作为
   agent CFD 能力评测集, 补充当前以材料为主的 bench 覆盖.

### 7.3 结构力学 (优先级中低)

1. **XPINN 域分解建议** — 弹塑性 + PINN 组合在 plan_check 阶段应给出 XPINN 建议,
   避免单网络收敛失败.
2. **DrivAerNet/AhmedBody 推荐** — 车身气动优化任务时, 优先推荐这些开源数据集.

### 7.4 地球/生物 (优先级低, 仅架构参考)

1. **AFNO 探索性接入** — 大网格场数据 (声子场/应力场) 可探索 AFNO, 但不是 huginn
   主线.
2. **Pairformer + Diffusion 架构参考** — 如未来扩展到生物材料 (seed/31), 可参考
   AlphaFold3 架构.
3. **StripedHyena2 长序列建模** — 长周期晶体结构可探索, 但目前 huginn 不覆盖.

### 7.5 平台层

1. **SCNET 远程提交** — OneScience 跑在 SCNET 超算上, huginn 的 HPC client 应支持
   SCNET 集群提交 (module load + conda activate + bash install.sh 一次性脚本).
2. **DCU 后端支持** — 用户在 DCU 平台时, huginn 的容器执行器应能识别 DTK 环境,
   不强依赖 CUDA.
3. **在线试用 fallback** — 当本地资源不足时, huginn 可调用 scnet.cn 在线试用入口
   作为 fallback, 但要走 user_profile 的"本地数据隐私"优先策略.

## 8. 来源与许可证

- 仓库: https://gitee.com/onescience-ai/onescience (Apache 2.0, 2025)
- 平台: 超算互联网 https://www.scnet.cn
- 双平台: GPU (CUDA) + 海光 DCU (DTK)
- 第三方: NVIDIA NeMo (Apache 2.0), 详见 NOTICE
- 使用手册: https://download2.sourcefind.cn/65024/9/main/onesicence
