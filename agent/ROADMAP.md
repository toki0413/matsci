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
- **状态**: 已落地 (v25, flag `hypothesis_llm_semantic` 默认关, 优雅降级)
- **优先级**: P1
- **现状**: 假设维度 (dimension) 抽取、phase 语义分类、失败模式分类均用
  中英文关键词表 + 字符串 `in` 匹配, 非语义判定。
- **升级方向**: 接 LLM 判定 dimension / phase / 失败语义分类 (v8+)。
- **落地**: `huginn/autoloop/hypothesis_semantic.py` — 三处分类器 (classify_dimension /
  classify_family / classify_failure) 接 LLM 语义判定, 无 LLM / flag 关 / 异常 /
  标签非法时降级回关键词匹配 (行为向后兼容). 引擎 init 注入 model provider.
- **相关文件**:
  - `huginn/autoloop/hypothesis_semantic.py` (新增)
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
- **状态**: 已完成
- **优先级**: P2
- **现状**: plan JSON 持久化语义固定, 展示层裁剪不改持久化。
- **升级方向**: plan 持久化到文件, chat 上下文只引用; 按 step 状态动态裁剪。
- **落地**: `_render(plan, active_step_id)` 动态裁剪 — 已完成步骤折成一行计数,
  高亮 active step (▶), 只列剩余 pending step; `export_markdown` 落盘到
  `$HUGINN_CACHE_DIR/plans_md/<plan_id>.md`, `build_context_hint` 返回不落盘的
  裁剪引用. `context_builder.build_plan_text` 接 `self._build_plan_context_hint`
  在 chat 上下文只引用 active step + 后续 pending step, 降低 token 占用.
- **相关文件**:
  - `huginn/autoloop/plan_store.py` (line 333, 335)
  - `huginn/context_builder.py` (build_plan_text / _build_plan_context_hint)

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
- **状态**: 已完成
- **优先级**: P2
- **现状**: completion records 落盘 jsonl 给 red_team + RL 训练消费,
  跨 turn 前缀重复未去重。
- **升级方向**: 加 prefix_merging (跨 turn 前缀去重)。
- **落地**: `_dump_completion_records` 写盘前找同 thread 最新前一 turn 文件,
  `_compute_common_prefix` (按 type+content, 忽略 ts/噪声) 算公共前缀,
  前缀 >=2 时只写 `prefix_ref` (prev_file + prefix_len) + 后缀增量; 消费端用
  `_reconstruct_completion_records` 递归重建完整序列。测试 `tests/test_streaming.py
  ::TestPrefixMerging`.
- **相关文件**:
  - `huginn/agent/streaming.py` (line 250, 1965)

### 4. session 持久化升级
- **状态**: 已完成
- **优先级**: P2
- **现状**: snapshot 只读最新一条, 走 `memory.save_session_snapshot` + JSON。
- **升级方向**: 按 `session_id` 精确读 + 版本化; 增量 diff + 独立 store。
- **落地**: `longterm.list_by_source(source="session:{sid}")` 精确读指定 session 的
  快照 (不再"任意最新一条"); `save_session_snapshot` 写 `version` 字段自动递增
  (基于已有最大版本 +1); `load_session_snapshot(session_id)` 非空时精确读该
  session 最新一代, 为空回退任意最新 (向后兼容); `session._init_session_continuity`
  改为按本 session_id 精确恢复。测试 `tests/test_session_snapshot_versioning.py`.
- **相关文件**:
  - `huginn/agent/session.py` (line 88, 131)
  - `huginn/memory/manager.py` (save/load_session_snapshot)
  - `huginn/memory/longterm.py` (list_by_source)

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
- **状态**: 进行中 (默认关闭; 环境验证: 本沙箱未构建 huginn_ext, Python 回退路径全绿)
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
  **环境验证 (2026-08)**: 本沙箱 `import huginn_ext` 失败 (扩展未构建属预期),
  `HUGINN_USE_RUST_SANDBOX` 未设 → 走 Python 回退; `tests/test_bash_rust_sandbox.py`
  7 passed / 1 skipped (skip 即 Rust 路径, 因无扩展), 回退路径无回归。
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
- **状态**: 已落地 (P2#2)
- **优先级**: P2
- **现状**: BM25 索引未按 domain 分片, 有 domain 过滤时跳过 (退回纯向量),
  丢失材料术语精确匹配能力。
- **升级方向**: BM25 按 domain 分片后, 有 domain 时也能混合检索。
- **落地**: `_BM25Index.add` 记录每片 domain, `search(..., domain=)` 按 domain
  分片过滤; `_ensure_bm25_index` 拉 metadatas 取 domain; `query()` 有 domain
  时也走 RRF 混合 (不再退回纯向量).
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

## 测试覆盖率进度 (D4 攻 Rust 桥接 / 集成路径)

> 注: 环境自动同步曾把未提交的 `*_ext.py` 测试清空, 已重建并提交。今后改动需及时 commit 防同步重置。

### security 包 (6 模块, 121 用例全绿)
- `huginn/security/cumulative_audit.py` → **100%** (`tests/test_cumulative_audit_ext.py` → 15): 空/缺失目录快照、递归 rglob、文件计数字节、blocked 扩展名优先拦截、repro 任务专用上限、OSError stat 降级、audit_step 历史记录、history 浅拷贝
- `huginn/security/prompt_security.py` → **100%** (`tests/test_prompt_security_ext.py` → 12): 空内容直返、有/无 source 标记、`wrap_rag_chunks` 原地改写 + `_raw_document` 保留、非字符串/空/missing document 不动
- `huginn/security/code_act_sandbox.py` → **100%** (`tests/test_code_act_sandbox_ext.py` → 11): 工具过滤(名 list/二元组/全拦/空)、安全 builtins(移 exec/eval/compile/open/globals/locals)、safe_import 白名单/子模块 root/三 opt-in 模块 env flag、check_degrade 边界、exec_with_mem_cap(0 不监控/超阈值/恢复 tracing)
- `huginn/security/script_runner.py` → **96%** (`tests/test_script_runner_ext.py` → 16): 安全 globals + math、白名单 import 放行/拒绝、blocked 子模块、变量注入、stdout 截断、timeout、语法/运行错误
- `huginn/security/math_eval.py` → **89%** (`tests/test_math_eval_ext.py` → 26): 算术/比较/布尔/条件/容器/下标、`np.<func>` 白名单、未定义名拦截、非白名单调用/属性/导入/推导式拒绝、SyntaxError 包装
- `huginn/security/rate_limiter.py` → **97%** (`tests/test_rate_limiter_ext.py` → 28): 单轮/秒级/总成本三道闸门、disabled 放行、per-session 隔离、窗口剪枝、record_usage 记账、reset_turn/reset_all、预警、usage 提取辅助、单例/env

### Rust 桥接 (5 模块)
- `huginn/tools/file_read_tool.py` → **94%** (`tests/test_file_read_tool_ext.py` + `tests/test_file_read_tool_integration_ext.py` → 7+16): Rust `tail_lines` fast path(命中/start 计算/import 缺失回退/Rust 异常回退/文件缺失) + `call()` 端到端(基础读/line_offset+n_lines/相对路径/路径穿越拒绝/非绝对路径、unrestricted env、文件不存在/非文件/过大、token 截断/env max_size、PDF 成功·失败·空文本、stat 异常回退、`_apply_token_cap` 直接测)
- `huginn/rag/vector_store.py` → **95%** (`tests/test_vector_store_rust_ext.py` + `tests/test_vector_store_search_ext.py` + `tests/test_vector_store_crud_ext.py` + `tests/test_vector_store_client_ext.py` → 4+7+27+14): Rust `top_k` 桥接 + `search()` 集成(原生 query/截断 top_k/HNSW 失败降级 Rust fallback/带 metadata filter/unsets 空/无 embedding 关键字/空集合) + CRUD(ingest 空·带 id·自动 id、delete/list_documents/count/get·missing/update 各分支) + ingest_file/_parse_file(txt/json/csv/PDF 文本·需 pymupdf·OCR 回退)/_chunk_text 边界 + EncryptedVectorStore 全链路(加密 ingest/解密 get/锁定不加密/异常回退/元数据开关/delete/count/list/update/ingest_file) + 客户端初始化(fake chromadb: 单次初始化/embedding 可用·不可用·失败/collection 带·不带 ef/embedding 缓存·异常)
- `huginn/routes/health.py` → **100%** (`tests/test_health_rust_ext.py` + `tests/test_health_integration_ext.py` → 2+25): `/health/rust` 可用/不可用 + `_is_configured` 各判定(ollama provider/models ollama/内置 key/env key/禁用无 key/resolved key/default 无 key)、`/health/ready` 三路检查(sqlite ok/异常、llm 配置/未配置/异常、mcp 未配置/无 server/全连/断连/异常 + 503 汇聚)、legacy `/health`(configured/unconfigured/model_pool/mcp_servers)、`/health/guidance`(key 检测/ollama 可用与否/recommendation 三态/已配置无建议)
- `huginn/tools/bash_tool.py` → **92%** (`tests/test_bash_rust_sandbox_ext.py` + `tests/test_bash_tool_integration_ext.py` → 7+43): Rust `run_sandboxed` 成功/失败(有/无 stderr)/异常回退/import 缺失回退/默认关闭不触发 + command+args 拆分与 allowed_base_dirs 传递 + 空命令分支、ContainerExecutor 路径(成功/失败)、SandboxExecutor 路径(成功/失败/SandboxError/异常)、`get_executor` 抛 SandboxError、重活识别 `_is_heavy_bash`(jupyter/notebook/python+train|fit|epoch 等/非重活)、`_suggest_fix` 全错误模式、`_extract_progress`(空/过滤/截断)、重活 dispatch(persistent terminal 成功/start 失败降级、support subagent 成功/Čech H¹ rejection 路径/无 agent_factory 降级/subagent 异常降级、protocol 关闭直跑)
- `huginn/tools/sim/vasp_tool.py` → **85%** (`tests/test_vasp_rust_ext.py` + `tests/test_vasp_tool_integration_ext.py` + `tests/test_vasp_run_ext.py` + `tests/test_vasp_outcar_actions_ext.py` → 6+29+7+11): Rust `parse_outcar` relax 命中/error/converged=False/异常回退 + scf 不信任 Rust 落 Python + `_HAS_HUGINN_EXT=False` 走 Python + `_find_vasp`(env 命中·不存在·PATH·which 异常)/estimate_cost(poll·wait None/compute)/validate_input(poll·wait/eos 缺目录/缺 workdir/缺 POSCAR/成功)/call(缺目录/缺 POSCAR/mock 含 incar override)/poll·wait_job(未知/已知/task None/done·failed)/_parse_vasprun_quick(pymatgen/ElementTree/坏 xml)/_read_incar_params/_modify_incar/_uq_hint/_structure_file_hint/_is_float/`_run_vasp`(成功/硬失败/SCF 未收敛软失败重试/物理审计报错/超时/异常/call 走真运行)/`_parse_outcar_python` action-aware(scf EDIFF/band·dos 输出/dos 无输出/relax 离子标记/字段解析/pymatgen 路径)/_eos(拟合/不足 4 点/目录缺失)
- `huginn/tools/sim/lammps_tool.py` → **85%** (`tests/test_lammps_ext.py` + `tests/test_lammps_tool_integration_ext.py` → 6+54): Rust `parse_lammps_dump` 桥接(可用/不可用/python baseline) + `_find_lammps`(env 命中·不存在·PATH·which 异常)/estimate_cost(poll·wait None)/validate_input(poll·wait/analyze 缺轨迹·轨迹缺失·成功/缺 structure/缺 input script/缺 potential/成功)/call(analyze_trajectory 成功·缺文件/no executable/成功·硬失败·超时·异常/带 structure+fixes+potentials+mpiexec/物理审计软失败+autofix 重试/硬失败兜底审计异常/轨迹自动解析+provenance 异常吞掉)/`_handle_submit_async`(schema 缺 compute_action·input_script·job_id/后台跑完)/poll·wait_job(未知/task None/已 done/超时/任务异常)/`_parse_log`(缺失/热力学+警告+错误/读取异常)/`_run_equilibrium_check`(无 log/无 thermo/无 temp/平衡)/`_build_equilibrium_recommendation`(drift+temp/pressure/equilibrated/no reasons)/`_apply_script_fixes`/`_read_script_params`/`_try_autofix`/`_is_float`/`_to_float_or_str`/`_uq_hint` + DEM packing(无可执行·生成脚本多分散·单分散+e=1 弹性·成功·失败·超时·异常)
- `huginn/tools/sim/qe_tool.py` → **98%** (`tests/test_qe_tool.py` + `tests/test_qe_tool_integration_ext.py` → 4+30): `_find_qe`(env 命中/不存在/PATH which/无)、`call`(generate/run/parse/异常)、`_generate_input`(relax/vc-relax/md 块、未知元素 pseudo+mass 默认)、`_run_qe`(成功/硬失败/SCF 未收敛软失败/autofix 重试/物理审计报错软失败/兜底审计异常吞掉/无可执行回退)、`_read_output_tail`(有内容/异常空)、`_read_input_params`(数值转换)、`_apply_input_fixes`(存在覆盖/新增插入/无 &ELECTRONS 不改)、`_try_autofix`(命中/未命中/异常)、`_parse_output_file`(缺失)、`_parse_output`(energy/converged/forces 以 Total force·空行·结尾终止/stress/坏 energy)、`_parse_results`(多文件)
- `huginn/tools/sim/cp2k_tool.py` → **100%** (`tests/test_cp2k_tool.py` + `tests/test_cp2k_tool_integration_ext.py` → 4+9): `_find_cp2k`(env 命中/不存在/PATH which/无)、call(parse/generate/run/异常)、`_run_cp2k`(无可执行回退/成功/硬失败/审计异常吞掉)、`_parse_output_file`(energy/converged/forces/stress)
- `huginn/tools/sim/gaussian_tool.py` → **98%** (`tests/test_gaussian_tool_integration_ext.py` → 40): `_find_gaussian`(env/which/无)、`validate_input`(缺/成功)、`call`(缺 workdir/缺 gjf/route overrides/parse/缺 executable/resolved/未解决)、`_run_gaussian`(成功/硬失败/SCF 未收敛/opt 未收敛/物理审计报错/审计异常/autofix 重试/sandbox 异常)、`_read_route_params`(正常/异常)、`_apply_route_overrides`(替换+追加)、`_format_keyword`、`_find_gjf`(命名/自动检测 .com)、`_parse_and_return`(无 log/正常/termination false)、`_mock_result`、`_get_returncode`/`_get_stderr`、`_parse_log`(全字段/读取异常/Convergence failure)、`estimate_cost`、`_try_autofix`(命中/无 route fix/未命中/异常)
- `huginn/tools/sim/openmm_tool.py` → **99%** (`tests/test_openmm_tool_integration_ext.py` → 38, 需 fake openmm + fake numpy plugins): `is_read_only`/`is_destructive`、call 分派、`_energy_minimize`(pdb 缺失/import 缺失·skipped/成功·minimized.pdb/收敛判定/max_iterations=0/implicit solvent/审计异常吞掉/抛异常)、`_md_run`(pdb 缺失/import 缺失·skipped/成功·final.pdb+trajectory.dcd/nvt 无平衡步/审计异常吞掉/抛异常)、`_analyze`(traj 缺失/import 缺失·skipped/pdb 缺失/rmsd·energy(md_log.csv/缺失)·temperature(mean/std)·radius_gyration·未知类型/异常)、`_analyze_rmsd`/`_analyze_rg`/`_analyze_temperature`(无 log)、`_resolve_file`、`_nonbonded_method`(explicit·implicit·vacuum)、`_constraints`(vacuum·其余)、`_make_simulation`、`_parse_md_log`(正常·缺失·坏行·短行·读异常容忍)
- `huginn/tools/sim/gromacs_tool.py` → **99%** (`tests/test_gromacs_tool_integration_ext.py` → 31, 需 fake numpy plugin; gmx 未装 → `_gmx_available`=False 分支真实命中): `is_read_only`/`is_destructive`、call 分派、`_resolve_file`(None/缺失/绝对/相对)、`_md_run`(tpr 缺失/gmx 缺失·skipped/成功·日志解析+审计/失败/sandbox 拦截/超时/审计异常吞掉)、`_energy_minimize`(tpr 缺失/gmx 缺失·skipped/成功/失败/sandbox 拦截/超时/审计异常吞掉)、`_analyze_traj`(traj 缺失/gmx 缺失·skipped/rms·rmsd·rdf·gyrate 成功+自动选组 input/失败/sandbox 拦截/超时)、`_parse_md_log`(缺失/读异常/温度·压力·能量·LINCS·SHAKE·neighbor-list 统计·NaN 检测)、`_gmx_available` 直测; 注: 未知 action/analysis_type 分支被 pydantic Literal 守护, 不可达(死代码)
- `huginn/tools/sim/orca_tool.py` → **99%** (`tests/test_orca_tool_integration_ext.py` → 45, 需 fake numpy plugin; orca 未装 → 无 executable 分支真实命中): `_find_orca`(env 命中/不存在/which)、`estimate_cost`、`validate_input`(缺 workdir 404/成功)、call(缺 workdir/缺 inp/命名 inp 缺失/parse 成功/带 executable/带 input_overrides/无可执行·resolve str/无可执行·needs_resolution)、`_run_orca`(成功/硬失败/opt 未收敛软失败/SCF 未收敛软失败/物理审计报错/审计异常吞掉/autofix 重试/autofix 无修复 break/重试耗尽/sandbox 异常)、`_try_autofix`(无 fix/可应用/无可应用键/异常)、`_read_input_params`(正常/异常空)、`_apply_input_overrides`(替换+追加/非 ! 行跳过/None token 跳过/读异常)、`_format_orca_token`(scf_conv·grid·maxiter·maxcore·默认)、`_parse_out`(全字段·Total Energy 回退·读异常)、`_find_inp`(命名/glob)、`_parse_and_return`(无 out/成功/failed 无能量)、`_mock_result`、`_get_returncode`/`_get_stderr`(对象/dict/plain); 注: `_check_action_fields`(working_dir 空) 被 pydantic required 守护, 不可达(死代码)
- 注: `lammps_tool.py` 的 Rust fast-path 是**有意的性能债** (parse_trajectory 恒走 Python, 见源码 1136-1145 注释), 非实际桥接点, 不测。

### 可视化 (需 `pip install matplotlib`, 无头跑用 `MPLBACKEND=Agg`)
- `huginn/tools/visualize_gate.py` → **100%**, `visualize_qa.py` → **100%**, `visualize_tool.py` → **99%** (已有 tests/test_visualize_tool.py + 新增 `tests/test_visualize_tool_ext.py` → 44, 需 fake numpy plugin; ext 测通过 mock huginn.visualize/figure_ir/run_figure_gate 避开 matplotlib+numpy 冲突): `is_read_only`/`read_only`、call 的 materials actions (band_structure/dos/phonon/structure_3d) 成功·绘图异常吞掉·gate error 吞掉、report-based (benchmark/evolution/exploration) 成功·绘制异常、report 加载失败、`_run_gate`(normal/异常吞掉)、`_figure_ir_numeric`(单点/多点/非 dict 元素/缺 label/空 data/非数值/混合/series 非 list)、`_load_report`(report_data/report_path/两者皆缺抛错)、`_build_figure_ir`(4 种 materials 映射/异常吞掉)、`_build_report_figure_ir`(benchmark scores dict·list·递归扫描/evolution timeline dict·标量 list·递归/exploration candidates·递归/异常吞掉)、`_scan_numeric_fields`(数值/bool 过滤/嵌套/深度上限/数量上限)、`_scan_numeric_list`(最长 list/嵌套/bool 过滤/深度上限/标量输入/max_items 嵌套提前返回)、`_extract_timeline_values`/`_extract_candidate_values`(dict/标量/跳过); 注: unknown action 分支被 pydantic Literal 守护, `_scan_numeric_list` dict 分支 max_items 提前返回为防御性, 均不可达(死代码)
- `huginn/tools/visualize_check.py` → **97%** (`tests/test_visualize_check_ext.py` → 25): 图↔数据一致性校验全分支 — extract_figure_numeric(标量/列表/error/非 dict/异常)、check_figure_vs_expected(标量/列表匹配/漂移/长度不匹配/近零跳过)、check_figure_duplicate(无 index/搜索失败/空/排除自身/超阈值)、consistency_verdict(pass/fix/fail/error)
- `huginn/autoloop/visual_inspect.py` → **99%** (`tests/test_visual_inspect_ext.py` + `tests/test_visual_inspect_gate.py` → 28+5): `_histogram_correlation`(相同图>0.9/损坏字节回退0)、`_execute_visual_inspect` 全动作(zoom 含一致性二次 crop/low_confidence 强判/consistency 关无 score、measure/annotate/compare/inspect 默认、无视觉数据 error、PIL 缺失回退、crop 异常、registry 导入失败/工具成功、enrich 异常吞掉/设 hint)、`_attach_gate_note`(blank 附 gate/正常不附/无图 noop/qa 异常跳过)、`_measure_nearest_primitive`(5 种原语变体 + 最近点 + 上下文行 + 空)、`_annotate_visual_features`(无图无 ctx/有图无工具/工具异常/defect·phase·sem 场景/text 结构特征/tool_output 补充)、`_extract_text_visual_features`(段落/趋势/峰谷/均值/异常)、`_compare_visual_data`(peak/min/anomaly/compare 目标/空/无量化数据)、`_selfcheck`
- `huginn/tools/visual_hook.py` → **98%** (`tests/test_visual_hook_ext.py` → 91): `should_visualize`(空/非 dict/工具名模式/result 数值 key/code·bash 数值 stdout)、`_extract_metric_pairs`(去重/非指标 key 过滤)、`render_tool_output`(line/单 list/energies bar/标量/stress-strain/scores/metrics/未绘图 None/非 dict/matplotlib 缺失/过大返回 None)、`extract_visual_primitives`(1D 趋势/嵌套 bands/常量/异常坐标化/scores/metrics/导数·FWHM/单点 n==1/短列表/scores 异常值/嵌套·扁平无数值)、`_extract_2d_primitives`(EDS 质心·覆盖率·hotspot·IoU·非 dict 元素·非 dict hotspot·质心异常、phase_field 体积分数·interface·morphology·非 dict 相·centroid 异常/空)、`extract_comparative_primitives`(1D peak/valley shift/new_anomalies/嵌套跳过/空/全非数值/2D EDS 位移·lost·new·非 dict·centroid 异常/phase_field diff·异常)、`_estimate_data_confidence`(1D 少点·mid 档·+5 点/嵌套跳过/low_snr/高异常率/全非数值/EDS 低覆盖·no_elements/phase_field 无域)、`enrich_with_visual`、`extract_box_primitives`(区域/全白空/import 缺失/面积过滤/RGB 转灰度/零维图/max_boxes 截断/坏字节异常)、`parse_box_primitive`、`<point3d>` 原语收发(含 import 缺失/空 label)、`_selfcheck`; 注: 剩余 15 行(111-113/253-255/269-271/578-580/782-784) 为 `float()` 转换 except, 因列表推导已 isinstance 预过滤数字, 恒不触发(死代码)
- `huginn/tools/report_tool.py` → **99%** (`tests/test_report_tool_ext.py` + `tests/test_report_compile.py` → 43+4): ReportGenerator(`add_section`/`generate` 四格式 markdown·latex·json·html+未知名回退、`_build_sections` 全 8 段+可选段缺省、`_render_methods` brief/full、`_render_structure` 数值 change 计算·非数值 N/A、`_render_convergence` 空/满、`_render_results` 嵌套 dict·标量、`_render_validation` ✅/❌、`_render_literature`/`_render_resources`/`_render_reproducibility`)、ReportComparator(四段比较表/header markdown·json 空·latex)、ReportTool(`is_read_only` 各 action、generate 无结果·带 output_path·带 calculation_dir、export、compare 缺 datasets·从 workflow_results 提取·<2 数据集·带 output_path、`_scan_directory` input/output 文件收集·OUTCAR ENCUT 提取·无 ENCUT 边界、compile_pdf 无 tex_source·engine 缺失·timeout·二次失败·无 pdf 生成·成功); 注: 剩余 1 行(414) call 的 unknown action 分支被 pydantic Literal 守护, 不可达(死代码)
- 顺带修复 `visual_hook.py` 一个 bug: anomalies 坐标 `_normalize_coord` 返回字符串被 `ax, ay =` 解包导致 ValueError, 改为直接用字符串。

### 核心工具
- `huginn/tools/structure_tool.py` → **100%** (`tests/test_structure_tool_ext.py` → 30): `is_read_only`、input model_validator(batch_validate 缺 files/其它 action 缺 file_path/合法)、`_file_mtime`(存在/缺失)、`_local_to_output`(全字段/缺 formula·num_sites 回退)、`validate_input`(batch_validate 放行/文件不存在命中本地库·不在库 404/文件存在/坏 reference 404/好 reference 放行)、`call`(本地库命中/文件路径转发 _call_cached/batch_validate 单独分派)、`_handle_batch_validate`(空 files early return·文件去重保序·混合 valid/invalid·单文件异常吞掉)、`_call_cached`(文件不存在/pymatgen 主路径+SpacegroupAnalyzer 成功/analyzer 异常且无 get_space_group_info→None/analyzer 异常但有 get_space_group_info→fallback/ImportError 基本解析·POSCAR 原子数·原子数非法·非 POSCAR 不解析/解析异常包装); 注: 空 files early return 用 `model_construct` 直击, 生产路径被 model_validator 守护。

#### 架构修复 (2026-08, brooks-lint 审计) — 工具注册死引用与契约接入
- **状态**: 已完成
- **优先级**: P0 (silent failure)
- **修复 1 — 工具注册"魔法字符串"死引用**: `_OPTIONAL_MODULES` 用全限定字符串 `("huginn.<mod>", "Cls")` 注册工具, 拼错路径会被 `_do_register` 的 `except ImportError` **静默跳过**, 工具永久消失无告警。新增 `huginn/tools/__init__.py::validate_tool_specs()`: 逐条 importlib 解析所有 spec, huginn 内部路径错误立即 raise, 第三方缺依赖正常 skip。**一次性暴露并修复 21 个死引用** (lammps/comsol/qe/cp2k/openfoam/packing/abaqus/plasma/symbolic_regression/autodiff/unit/symmetry/tda/uq/evidence_fusion/high_throughput/multi_fidelity/xrd_sim/gap_analysis/doe/debugger/nudge/generative_design 均移到 `sim/`/`sci/`/`design/` 子包)。新增 `tests/test_tool_registration.py` (3 用例)。
- **修复 2 — 无消费者的协议层**: `huginn/orchestration/OrchestratorProtocol` 定义统一 orchestrator 契约, 但生产无 import, 契约无运行时强制力。在 `huginn/routes/agents.py` `/orchestrate` 入口加 `isinstance(orch, OrchestratorProtocol)` 断言, 让契约进入主链路, 未来 orchestrator 违约立即报错。
- **修复 3 — 自进化逻辑重复**: `huginn/self_improvement/` 的 `SelfImprovementLoop` 与生产的 `huginn/bench.runner.BenchmarkRunner` + `huginn/evolution.engine.EvolutionEngine` 并存且无生产消费者。在 `self_improvement/__init__.py` 标注"参考/测试实现, 生产走 bench+evolution", 避免误当主链路。
- **修复 4 — WebSocket 消息分发长链 → 注册表映射**: `huginn/routes/ws.py` 原用 15 个 `if/elif` 分支分发消息, 部分 handler 同时依赖 `msg` (WSMessage) 和 `data` (原始 dict), schema 定义与使用双通道不一致。重构为:
  - 新增 `_MESSAGE_HANDLERS: dict[str, Callable]` 注册表, 每个消息类型一行映射。
  - 统一 handler 签名为 `(websocket, msg: WSMessage, ctx: WSCtx)`, 所有参数从 `msg` 读取, 消除 `data.get(...)` 原始 dict 依赖。
  - 新增 `WSCtx` dataclass 打包连接级状态（approval/plan/auto_approve 等）, 避免每个 handler 各自传 5-6 个参数。
  - 新增 `action`/`edited_code`/`config` 字段到 `WSMessage` schema, 保证 handler 需要的字段都经过 pydantic 校验。
  - 将 `clarification_response`/`set_auto_approve`/`set_suggest_mode`/`suggest_response`/`ping` 5 个内联 handler 移到 `ws_helpers.py` 统一管理。
  - 17 个 WS 集成测试全部通过 (0 failures)。

#### 架构修复 (2026-08, 继承核查) — rcb_runner 拆分 / 死代码 / 依赖探测
- **状态**: 已完成
- **优先级**: P1 (主链路复杂度 / 可观测性)
- **修复 5 — `rcb_runner.py` God File 拆分**: 5700 行聚合入口 (跑分遗留物) 拆为 7 个模块, 保持聚合入口 `run()`/`main()` 向后兼容:
  - `rcb_utils.py` (纯工具, 无依赖) → `rcb_cognition.py` / `rcb_audit.py` (依赖 utils) → `rcb_step2.py` / `rcb_step3.py` / `rcb_mcmc.py` (依赖前两者) → `rcb_runner.py` (聚合)。依赖层级单向, 无循环 import。
  - 39 个 RCB 相关测试全部通过。
- **修复 6 — 真死代码删除**: 全库 AST+引用扫描确认 `huginn/autoloop/engine_selfcheck.py` (2860 行) 为唯一真死代码 — 全库无任何 `import` 消费, 仅 `engine.py` 注释提及 (其 `_extract_tests_passed` 已由 `cognitive_loop.py` 下沉并 re-export)。已删除, autoloop 50 测试通过。
- **修复 7 — optional 工具"有名无实"依赖探测**: 大量 optional 工具在模块顶层/函数体内 `try: import torch/pymatgen/rdkit...` 包裹, 缺依赖时模块仍能 import、工具照常注册, 但能力静默退化。新增 `probe_tool_dependencies()`: 静态扫描每个 optional 工具模块的 import, 过滤 hugging 内部 + stdlib, 用 `importlib.util.find_spec` 判定第三方依赖缺失并 `logger.warning` 显式告警。在 `register_optional_tools()` 末尾接入, 运维可即时看到能力降级清单。**评测发现两处覆盖短板并已修复**: ① 原只扫模块级 import, 漏掉函数体内懒导入 (sklearn/rdkit 各工具); ② 不穿透 shim (如 `tools/rdkit_tool.py` → `sci/rdkit_tool.py`), 导致 rdkit/pymatgen/openmm 等漏报。现 `_collect_top_level_imports` 遍历函数/类体 import, 且仅对"纯 shim"(顶层无逻辑)跟随其 re-export 目标; 对普通模块不跟随 `huginn.*` 门面 (避免 `huginn.llm` 的可选依赖污染到引用它的工具)。修复后探测从仅 5 个缺失模块提升到 33 个真实缺失 (pymatgen/rdkit/openmm/sklearn/vasp/vina 等全捕获)。

#### 架构修复 (2026-08, 继承核查) — 多入口收口 + 进化门控接入主链路
- **状态**: 已完成
- **优先级**: P1 (主链路复杂度 / 进化质量)
- **修复 8 — 多入口 MemoryManager 初始化重复 → 共享 factory**: FastAPI server / RCB harness / CLI 各自手写 MemoryManager 装配, 改一个能力要同步多处。新增 `huginn/memory/factory.py::build_memory_manager()` 作为唯一构造入口: 传 `memory_dir`(RCB 主题记忆)或 `memory_md_path`(server MEMORY.md)+ 可选 `longterm`/`llm`, 内部统一 `MemoryConfig` 装配; 任何单点失败保留空 longterm/llm 不抛 (memory 是增强, 不阻塞主流程)。`server_core.get_memory_manager()` 与 `rcb_runner.py` 均改为调用该 helper, 消除重复装配。新增 `memory/factory.py` (独立单测), 相关 memory/rcb 测试通过。
- **修复 9 — significance/OOD/adoption 门控接入进化采纳路径**: 此前 harness 门控 (significance Wilcoxon + OOD holdout + adoption 三级) 只服务评测, 未接入 `EvolutionEngine` 的自进化采纳路径 — 失败/奖励派生的规则与技能可绕过显著性与 OOD 检查直接进活动池, 存在过拟合与噪声吸收风险。新增 `EvolutionEngine._adoption_allowed()` 软门控 (fail-open):
  - **advisory (默认)**: 门控只评分/记录 `H5/H6` 配对决策, 永远采纳 — 保护 agent 能力, 最坏只是"少自动采纳一次"。
  - **strict (`harness_adoption_gate=1`)**: 仅 GREEN 采纳, 黄/红不入活动池; 数据保留在 gate store 不删除, 可手动复活。
  - 门控任何异常都 fail-open 返回 True, 门控崩溃绝不阻断进化。
  - 接入点: `evolve_from_failures` / `evolve_from_successes` / `evolve_prompt_patches` / `evolve_from_rewards` 的规则/技能添加点, 以及 `sync_to_registry` 技能注册点。`tests/test_evolution_engine.py` 通过。

#### 架构修复 (2026-08) — orchestrator 无效工具注册告警噪音
- **状态**: 已完成
- **优先级**: P2 (日志可观测性)
- **修复 10 — `register_tool` 在 ToolRegistry 类模式下是 no-op 却反复告警**: `ExecutionOrchestrator` 默认接全局 `ToolRegistry` 类 (classmethod `.get`), 此时 `register_tool(name, fn)` 只对 dict 模式有意义, 对类模式是 no-op。但 [`routes/execution.py`](agent/huginn/routes/execution.py) 与 [`cli/commands/execute.py`](agent/huginn/cli/commands/execute.py) 仍循环调用它对每个工具注册 → 每个都打 `register_tool ignored: tool_registry is type, not a dict` warning, 日志刷屏且误导调用方以为出错。
  - `huginn/execution/orchestrator.py::register_tool`: 对非 dict registry 的 warning 降级为 debug 日志 (类模式有自己的 `.register`, 属正常路径非错误)。
  - 删除 `routes/execution.py` / `cli/commands/execute.py` 中冗余的 `orch.register_tool(tool_name, ...)` 循环及未使用的 `_wrap_tool` 函数、`ToolContext`/`AuditLogger`/`ToolRegistry` 导入。`run()` 内部用 `ToolRegistry.get()` 取工具, 由 `_invoke_tool` 桥接。
  - 验证: ruff 通过, `tests/test_execution_orchestrator.py` + `tests/test_execution_backend.py` 16 用例全绿。

#### 新功能 (2026-08) — externalThinking: deep_think 外部草稿纸
- **状态**: 已完成
- **优先级**: P2 (能力增强, 默认关)
- **动机**: 厂商隐藏原生链式推理后, 模型"愿意"暴露的 `reasoning_content` 并非总有。oh-my-pi 的 `externalThinking` 思路: 提供一个普通工具, 让模型在动手前把分析写进工具参数, 因工具参数经 API 返回, 开发者可直接读取保存。落地为 Huginn 内可一键开启的正式功能。
- **实现**:
  - 新增核心工具 `huginn/tools/deep_think_tool.py::DeepThinkTool` (输入 `analysis: str`, `read_only=True`), 注册进 `_CORE_MODULES`。执行时经 `context.memory_manager.add_reasoning()` 写入 `session.reasoning_trace` — 与 [streaming.py](agent/huginn/agent/streaming.py) 的 `reasoning_content` 捕获**共用同一蒸馏/进化通道**。`memory_manager=None` 时 fail-open (成功占位, 不阻塞)。
  - 新增 `external_thinking` feature flag (默认 False, 三层覆盖: config / `HUGINN_FEATURE_EXTERNAL_THINKING` env / 运行时)。
  - 开启时在 `huginn/agent/context.py::_effective_system_prompt()` 注入指令: 要求模型在回答/改代码/调其他工具前先 `deep_think` 写分析。关闭时不注入, 默认行为不变。
  - **补充通道策略** (非 oh-my-pi 的强制替换): 不对接"强制关原生推理" (`forceReasoningOff`), 因 Huginn 对接 provider 多样。`deep_think` 拿显式草稿, `reasoning_content` 拿原生推理, 两路都汇入同一条 `reasoning_trace`。
- **测试**: `tests/test_deep_think_tool.py` (6 用例) — 注册/read_only、落 trace、fail-open、空分析拒绝、flag 开关注入/不注入。全绿。

### 方法论
- 本项目 conftest 接 pytest-cov, `python -m coverage run` 与其冲突会 "No data collected"; 用 coverage Python API (`coverage.Coverage().start()` + `pytest.main()`) 包裹采集。
- free-threaded/3.14 下 coverage 行追踪会破坏 pytest traceback 格式化 (linecache 被污染) → 复用 3.12 环境测覆盖率; 无完整依赖栈时逐模块跑避免 INTERNALERROR 中断收集。

---

## E2E 现状核查 (2026-08, 生产就绪度)

### 沙箱覆盖 E2E 套件 — 全绿
- **结果**: 57 通过 / 1 跳过 / 0 失败 (约 25s 构建 + 串行跑)。
- **套件** (`tests/`, 均用 TestClient + FakeLLM, 无需真实 LLM/HPC):
  - `e2e_user_journeys.py` (27): 认证生命周期 / 知识库 / Memory / 工作流 / RBAC / 沙箱 / 健康检查。
  - `e2e_agent_loop.py` (7): FakeLLM agent loop — 单轮/多轮/容错/Memory/Telemetry。
  - `pentest_api_security.py` (12): SSRF / JWT / RBAC / SQL 注入 / 命令注入。
  - `pentest_archive_safety.py` (5+1 skip): zip/tar slip / symlink / archive bomb。
  - `fuzz_api.py` (6): 畸形输入不崩溃。
- **依赖**: fuzz 的 `given` 与 agent_loop 的 `@pytest.mark.asyncio` 需要 `hypothesis` + `pytest-asyncio`, 二者已由 `pyproject.toml` 的 `dev` extra 覆盖 (第 67 行), CI 用 `pip install -e "./agent[dev]"` 天然具备。**requirements.lock 是运行时 lock, 按设计不含 dev 依赖**; 不要手工往 lock 加 pytest 系, 否则 `deps-and-secrets` 的 lock-drift 门禁会红。本地单跑 fuzz 前需 `pip install -e "./agent[dev]"` 或单独补装这两包。
- **结论**: 沙箱可覆盖范围证明**工程底盘稳定** (认证/RBAC/注入防护/归档安全/agent loop 均跑通)。真实 LLM / HPC / 仿真软件 / 多用户并发只能在部署侧验证 (见 `tests/e2e_deployment_checklist.md`)。

---

## 极简模式 / 模型档位 (2026-08, 前端可切换)

设计背景: 我们的工作流 (phase 机 / plan gating / 认知纪律 / 重 compaction) 本质是
对弱本地模型的补偿; 对顶尖大模型这些补偿的边际收益骤降、代价真实 (token / 摩擦)。
极简模式不是砍功能, 而是把"认知编排"从常驻改为事件驱动 + 按模型档位聚合。
安全层 (命令校验 / 物理 sanity / 预算警告) 在所有档位保留。

> **与"思考强度"是两条正交轴**: 极简模式 (ModelTier) 决定"认知编排开销"
> (phase/plan/纪律/compaction); 思考强度 (ThinkingIntensity, 低/中/高/**MAX**)
> 决定"模型推理深度" (provider reasoning budget)。两者互不影响, 可独立设置。
> 最高档 MAX 只对支持方生效 (Anthropic budget 64000 / OpenAI reasoning_effort=max)。

### 1. 模型档位 profile 注册表 (M1) — 已完成
- **优先级**: P1
- **落地**: `huginn/plugins/model_tier.py` — `ModelTier` (full/balanced/minimal) +
  `TierProfile` (use_phase_machine / use_plan_gating / cognitive_discipline /
  compaction_tier / external_thinking) + `TierProfileStore` 单例运行时切换。
- **相关文件**: `huginn/plugins/model_tier.py`

### 2. 档位落地到 phase / plan / compaction (M3) — 已完成
- **优先级**: P1
- **落地**: `phases._effective_phase()` — minimal 档 phase 折成 OPEN (跳过 phase
  前缀注入与工具门控); `context_builder.build_plan_text()` — minimal 档跳过 plan
  上下文注入; `model_tier.compaction_knobs()` — minimal 档 compaction 力度放宽
  2 倍, 接入 `streaming.py` 的两条 compaction 路径 (乘子作用于 keep_last_n /
  keep_root_n, adaptive/CSM/trace 逻辑保持原样)。
- **相关文件**: `huginn/phases.py` / `huginn/context_builder.py` /
  `huginn/plugins/model_tier.py` / `huginn/agent/streaming.py`

### 3. external thinking 显式开关 (M4) — 已完成
- **优先级**: P1
- **落地**: `model_tier.set_tier()` 切换档位时联动 `external_thinking` FeatureFlag,
  保证 prompt 注入点读到的 flag 与档位 profile 一致。
- **相关文件**: `huginn/plugins/model_tier.py`

### 4. 认知纪律事件驱动守护 (M2) — 已完成 (v1)
- **优先级**: P1
- **落地**: `huginn/cognitive_discipline.py` — `discipline_mode()` 读档位的
  always/event; `deviation_kind()` 检测偏离 (v1 识别工具失败); `event_reminder()`
  生成即时提醒; `inject_discipline_reminder()` 在 event 档位、检测到偏离时于
  streaming 发送前注入一条紧凑 `HumanMessage` 提醒。非 event 档 / 无偏离 / 异常
  均原样返回, 不影响发送路径。
- **升级方向**: 偏离检测从"工具失败"扩到"物理不合理值 / 未确认覆盖数据 / 伪造结果"
  等更多信号 (接 ErrorKind 结构化错误分类)。
- **相关文件**: `huginn/cognitive_discipline.py` / `huginn/agent/streaming.py` /
  `huginn/plugins/model_tier.py`

### 5. 前端设置项 — 已完成
- **优先级**: P1
- **落地**: 后端 `routes/config.py` 新增 `GET/POST /config/model-tier` + `/config`
  返回 `model_tier`; 前端 `useConfig.ts` 新增 `modelTier` 状态与 `switchModelTier`,
  `SettingsPanel.tsx` Advanced 标签新增「模型档位 (极简模式)」下拉。
- **相关文件**: `huginn/routes/config.py` / `desktop/src/hooks/useConfig.ts` /
  `desktop/src/components/panels/SettingsPanel.tsx`

---

## 维护说明

- 新增"升级路径"注释时, 在对应模块段补一行, 标注状态/优先级/文件:line。
- 完成某项升级后, 把状态改为"已完成", 保留条目作为演进记录 (不删)。
- 优先级定义:
  - P0: 影响正确性 / 已知 silent failure (如 Rust sandbox 崩溃)
  - P1: 影响能力上限 / 可观测的质量瓶颈 (如语义判定、Q-learning 稀疏)
  - P2: 锦上添花 / 长 tail 优化
