# Huginn 升级路径 ROADMAP

本文件集中记录代码库中散落的"升级路径"注释 (ponytail/ceiling/升级),
把"现在怎么干、天花板在哪、下一步怎么走"汇成单一文档, 便于排期与避免
重复踩坑。每项标注: **状态** (计划中 / 进行中 / 已完成)、**优先级**
(P0 紧急 / P1 重要 / P2 锦上添花)、**相关文件**。

状态判定规则:
- 已完成: 代码注释里明确写"已升级"或功能已落地可用
- 进行中: 功能已实现但默认关闭 / 部分场景覆盖
- 计划中: 仅有注释指向未来方向, 未实现

---

## autoloop/

### 1. 关键词匹配 → LLM 语义判定
- **状态**: 计划中
- **优先级**: P1
- **现状**: 假设维度 (dimension) 抽取、phase 语义分类、失败模式分类均用
  中英文关键词表 + 字符串 `in` 匹配, 非语义判定。
- **升级方向**: 接 LLM 判定 dimension / phase / 失败语义分类 (v8+)。
- **相关文件**:
  - `huginn/autoloop/hypothesis_loop.py` (line 63, 126, 143, 281, 2046, 2108, 2341)
- **注意**: 与 `_metacog_classify_family` (engine.py) 同范式, 不引入 embedding。

### 2. 结构耦合 → embedding 语义 (假设耦合温度 Tᵢⱼ)
- **状态**: 计划中
- **优先级**: P2
- **现状**: 假设间耦合温度用 dimension 共享 + sibling 关系做结构耦合,
  不捕捉语义矛盾; belief space 体积用 0 维代理。
- **升级方向**: LLM/embedding 算 Tᵢⱼ; hypothesis graph 节点/边数变化或
  embedding 空间 PCA 体积变化做真 belief space 体积。
- **相关文件**:
  - `huginn/autoloop/hypothesis_loop.py` (line 38, 322-323, 1089, 1982, 2595-2596, 2799-2800)

### 3. 单进程 in-memory / markdown append → SQLite 持久化
- **状态**: 计划中
- **优先级**: P1
- **现状**: 假设事件 log 在内存; 假设库用 markdown append, 不上 SQLite / index。
- **升级方向**: 写入 session event log 持久化; 加 FTS5 全文索引。
- **相关文件**:
  - `huginn/autoloop/hypothesis_loop.py` (line 181, 556)

### 4. plan 持久化到文件 (Context Management 2026)
- **状态**: 计划中
- **优先级**: P2
- **现状**: plan JSON 持久化语义固定, 展示层裁剪不改持久化。
- **升级方向**: plan 持久化到文件, chat 上下文只引用; 按 step 状态动态裁剪。
- **相关文件**:
  - `huginn/autoloop/plan_store.py` (line 333, 335)

### 5. SSIM 视觉检查升级
- **状态**: 计划中
- **优先级**: P2
- **现状**: visual_inspect 用 numpy 直方图 + corrcoef, 不引 scikit-image。
- **升级方向**: 真正的 SSIM 接受度测试。
- **相关文件**:
  - `huginn/autoloop/visual_inspect.py` (line 23, 448, 491)

---

## agent/

### 1. streaming root 标记
- **状态**: 已完成
- **优先级**: P1
- **现状**: root 标记已升级为按消息 metadata (`additional_kwargs`/`metadata`
  的 `is_root`) 标记, 无 metadata 标记时回退到按位置 (前 `keep_root_n` 条)。
- **相关文件**:
  - `huginn/agent/streaming.py` (line 204, 735)

### 2. compact_messages 真修剪 checkpointer 持久化 (G34)
- **状态**: 已完成
- **优先级**: P0
- **现状**: 之前 `compact_messages` 只修 `inputs["messages"]` 临时 list,
  checkpointer 历史未真删。现已用 LangGraph 官方 `RemoveMessage` +
  `update_state` 批量删, 与 `compact_messages` 同款 drop-oldest 逻辑。
- **相关文件**:
  - `huginn/agent/streaming.py` (line 560-561, 615, 727-738)

### 3. prefix_merging 跨 turn 前缀去重
- **状态**: 计划中
- **优先级**: P2
- **现状**: completion records 落盘 jsonl 给 red_team + RL 训练消费,
  跨 turn 前缀重复未去重。
- **升级方向**: 加 prefix_merging (跨 turn 前缀去重)。
- **相关文件**:
  - `huginn/agent/streaming.py` (line 250, 1965)

### 4. session 持久化升级
- **状态**: 计划中
- **优先级**: P2
- **现状**: snapshot 只读最新一条, 走 `memory.save_session_snapshot` + JSON。
- **升级方向**: 按 `session_id` 精确读 + 版本化; 增量 diff + 独立 store。
- **相关文件**:
  - `huginn/agent/session.py` (line 88, 131)

---

## tools/

### 1. VASP 硬编码参数 → schema 驱动
- **状态**: 计划中
- **优先级**: P1
- **现状**: workflow variant 生成器的参数扰动规则硬编码 VASP 常见参数
  (ENCUT, KPOINTS, ISMEAR 等), 适用工具集写死。
- **升级方向**: 从 ToolRegistry schema 动态读参数名与扰动规则, 跨仿真器复用。
- **相关文件**:
  - `huginn/autoloop/variant_gen.py` (line 24)
  - `huginn/tools/sim/vasp_tool.py` (参考)
  - `huginn/tools/sim/convergence_strategies.py` (参考, 已有 VASP/QE 双格式识别)

### 2. Rust fast path 重启用
- **状态**: 进行中 (默认关闭; sandbox 崩溃报告 + import 已修复)
- **优先级**: P0
- **现状**: Rust sandbox runner 在 RDKit+sklearn GPR 等场景静默崩溃, 返回空 stderr
  导致 "Unknown error" (audit 08: 8 个出分单元 62.5% 有工具层直接背书)。当前显式
  `HUGINN_USE_RUST_SANDBOX=1` 才启用。VASP OUTCAR parser 的 Rust 加速器在
  scf/band/dos 场景的 converged 字段不可信, 仅 relax/md/phonon 用 Rust。
  已修复 (pyext/src/sandbox.rs): ① 子进程被 signal 杀死时不再丢 signal 号
  (code().unwrap_or(-1) 改 signal 捕获), 崩溃命令返回 rc=-11 + message=
  "killed by signal 11" 而非空 stderr + 无意义 -1; ② 子模块注册进 sys.modules,
  `from huginn_ext.sandbox import run_sandboxed` 不再 ModuleNotFoundError。
  发布链路已接入 (release.yml build-rust-ext job + pyproject [rust] extra)。
- **升级方向**: RDKit+sklearn GPR 的 native 冲突本身是环境/依赖问题, sandbox 现已
  如实报告 signal; 接入发布链路后, 在验证环境安装 RDKit+sklearn 复现确认不再
  "Unknown error" 后, 再评估默认启用。Rust parser 的 converged 字段补 action-aware
  校验后覆盖 scf/band/dos。
- **相关文件**:
  - `pyext/src/sandbox.rs`
  - `huginn/tools/bash_tool.py` (line 176-224)
  - `huginn/tools/sim/vasp_tool.py` (line 705, 718-720)
  - `.github/workflows/release.yml` (build-rust-ext)

---

## knowledge/

### 1. BM25 + 向量 RRF 混合检索 (已落地)
- **状态**: 进行中 (无 domain 时混合, 有 domain 时退回纯向量)
- **优先级**: P1
- **现状**: 手写倒排索引 (k1=1.5/b=0.75 Robertson-Sparck Jones), 与
  ChromaDB 向量检索做 RRF 融合; 零依赖 (不引 rank_bm25)。BM25 索引
  lazy 重建, dirty 时从 ChromaDB 全量重建。
- **相关文件**:
  - `huginn/knowledge/store.py` (line 101-122, 585-622, 1049-1062)

### 2. BM25 按 domain 分片
- **状态**: 计划中
- **优先级**: P2
- **现状**: BM25 索引未按 domain 分片, 有 domain 过滤时跳过 (退回纯向量),
  丢失材料术语精确匹配能力。
- **升级方向**: BM25 按 domain 分片后, 有 domain 时也能混合检索。
- **相关文件**:
  - `huginn/knowledge/store.py` (line 1050-1051)

### 3. embedding 检索增强 (语义去重 / 相似 context 召回)
- **状态**: 计划中
- **优先级**: P2
- **现状**: 假设库去重为纯文本去重; context 召回按 persona 聚合 r_phys,
  不用 embedding 相似度。
- **升级方向**: embedding 相似度过滤重复方向; context_hash 距离或 embedding
  召回相似 context。
- **相关文件**:
  - `huginn/autoloop/hypothesis_loop.py` (line 1982, 2799-2800)
  - `huginn/knowledge/store.py` (参考, 当前 BM25+向量 RRF)

---

## bandit_controller

### 1. 稀疏表格 Q-learning → tile coding / 网络逼近
- **状态**: 计划中
- **优先级**: P1
- **现状**: tabular UCB1 bandit, 4 维 state space (item_idx / time_bucket /
  calls_bucket / progress_bucket) 稀疏, 跨任务 JSON 持久化 Q table 缓解
  cold-start。pattern 聚合丢失 state 细节。
- **升级方向**: tile coding + pattern 混合; TD(λ)/eligibility trace 做在线
  credit assignment; 或网络逼近。
- **相关文件**:
  - `huginn/agent/bandit_controller.py` (line 6-7, 254, 389)
  - `huginn/cli/rcb_runner.py` (line 1259, 同一 ceiling 表述)

---

## 维护说明

- 新增"升级路径"注释时, 在对应模块段补一行, 标注状态/优先级/文件:line。
- 完成某项升级后, 把状态改为"已完成", 保留条目作为演进记录 (不删)。
- 优先级定义:
  - P0: 影响正确性 / 已知 silent failure (如 Rust sandbox 崩溃)
  - P1: 影响能力上限 / 可观测的质量瓶颈 (如语义判定、Q-learning 稀疏)
  - P2: 锦上添花 / 长 tail 优化
