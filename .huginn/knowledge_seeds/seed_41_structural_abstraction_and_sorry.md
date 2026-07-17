# Seed 41: 结构主义抽象 + sorry 驱动研究 + 状态机合并 (v6 spec)

## 核心教训
- sorry 是地图, 不是坟墓 — placeholder 进 gaps 表待填, 不淘汰 (G45/G48)
- 结构先于对象 — 材料功能源于结构关系, 锁定结构允许替换实现者 (G46/G47)
- 同构即等价 — validate_structure_preservation 是结构主义的形式化判据 (G46/G58)
- "未被选择" ≠ "不可实现" — sorry_impossible 需反例, 无反例保留 placeholder (G48)
- CSM 是 canonical 状态机, 其他 phase 枚举通过 listener observer 同步 (R22)
- 物理结构跟数学结构并行存在 — math_structures 是文字分类 (advisory), physical_structures 是结构化形式化 (G53)

## 加法清单 (G45-G58, 14 项)

### sorry 一等公民 (G45/G48)
- Conjecture 加 sorry_status 字段: none/placeholder/filled/impossible
- verify_conjecture: 含 sorry → (False, "placeholder"), 不 reject
- fill_sorry_gaps: 遍历 placeholder, LLM 填充 → filled; 反例 → impossible; 无反例 → 保留

### 物理结构形式化 (G46/G47/G58)
- PhysicalStructure dataclass: relation_type / relation_expr / implementor_slots / constraints / relative_anchors
- 5 类预定义结构: catalytic_geometry / interface_binding / percolation_topology / band_symmetry / defect_chemistry
- enumerate_implementors: 解耦搜索, 锁定结构允许不同实现者填充
- validate_structure_preservation: StructureMapping → bool, sympy 符号化 + 物理约束
- relative_anchors (G58 Moschella ICLR 2023): 槽位 → anchor 实现者名列表, cosine similarity 向量

### 结构主义 pipeline 语义 (G49/G50/G51/G52)
- G49 Moonshine 三步结构主义重构: extract PhysicalStructure → transfer (锁结构换实现者) → generate + validate_structure_preservation
- G50 SignalHub 加 structure_violation / sorry_filled / sorry_impossible 三个结构信号源
- G51 _PHASE_GUIDE 的 hypothesize/validate 加结构关系语义: 先识别结构, 再断言同构保持
- G52 metacog_segment("s7_self_modify") 加结构主义反思: 失败携带 structure_relation_type 时调 enumerate_implementors

### Deli 衔接 + 假设置信度 + bench 配对 (G53/G54/G55)
- G53 ResearchState 加 physical_structures 字段; DeliAutoResearch 加 _MATH_TO_PHYSICAL_STRUCTURE 映射表 (pde→band_symmetry 等); _extract_physical_structures / _fill_sorry_gaps 方法
- G54 AutoloopEngine 加 _last_hypothesis_confidence / _last_hypothesis_evidence_strength 字段; _darwin_ratchet_check 末尾赋值 (score/10 + supported_ratio)
- G55 run_no_agent_baseline (LLM zero-shot, 控制组) + generate_evidence_manifest (outputs/ 扫 sha256 → manifest.json)

### SafetyMode + 量纲 plan_check (G56/G57)
- G56 SafetyMode StrEnum: safe/guided/autonomous/yolo; mode_allows_force_proceed 映射; _resolve_safety_mode + _is_force_proceed 改造
- G57 _dimensional_pre_check: regex 提取方程两边量纲, DimensionalValidator 验证; _plan_check 调用它, 不一致 → warnings 注入 context + risks

## 减法清单 (R22-R24, 3 项)
- R22 4 套状态机合并到 CSM — 小改路径: STATE_TO_PHASE 映射 + CSMListener Protocol + transition() 末尾广播. 不删旧 phase 枚举 (向后兼容)
- R23 历史 16.5MB 死重防回潮 — .gitignore 加 *.bin / *.dat / *.h5 / *.hdf5 / *.npy / *.npz / *.pickle / *.pkl / *.parquet. filter-repo 一次性清理需用户单独批准
- R24 砍 v5 sorry reject — 验证 conjecture_library.py 的 verify_conjecture: sorry 标记 placeholder 而非 reject. G45 已隐式完成, 无需代码改动

## 关键决策
- sorry 不 reject 进 gaps 表 (ponytail: sorry 是研究地图的标记, 不是坟墓; reject 丢信息)
- 5 类预定义结构覆盖大部分材料功能场景 (ponytail: 不做万能结构形式化, 5 类够用; 升级路径是 LLM 抽取自定义结构)
- PhysicalStructure 用 dataclass 不用 ORM (ponytail: 状态对象, 不持久化; 持久化交给 memory/audit)
- CSMListener 用 Protocol 不用 ABC (ponytail: 鸭子类型, 老代码不破; 不强制继承)
- STATE_TO_PHASE 映射不强制 1:1 — S3/S6/S7 是元状态, 映射到最接近的 phase (ponytail: 元状态归一化是损失y, 升级路径是 listener 自定义映射)
- SafetyMode enum 替换 force_proceed bool (ponytail: 4 mode 比 bool 表达力强; 旧 force_proceed=True 仍兼容, 走 guided 路径)
- _dimensional_pre_check 用 regex 提取量纲 (ponytail: 不做完整 LaTeX 解析; 升级路径是接 sympy parsing)
- no-agent baseline 不真正跑 (ponytail: 占位 BaselineScore, score=-1.0; 升级路径是接 LLM runner)
- evidence manifest 扫 outputs/ 算 sha256 (ponytail: 不做 incremental, 全量扫; 升级路径是 watch 文件变更)

## ponytail 简化
- PhysicalStructure.constraints 是 list[str] 不是 dict (_constraints 是 sympy 表达式列表, key 没意义)
- _MATH_TO_PHYSICAL_STRUCTURE 5 条映射 (pde/variational/conservation/geometric/statistical), exploratory/none 跳过 (ponytail: 结构未定不强行映射)
- _extract_physical_structures 失败静默降级 — physical_structures 留空 (ponytail: import 错/无匹配不该让 Deli pipeline 崩)
- fill_sorry_gaps 用 max_fill=5 限流 (ponytail: LLM 调用慢, 5 个够 demo; 升级路径是并行 LLM)
- CSMListener._notify_listeners try/except 全包 (ponytail: 一个 listener 失败不该影响其他 listener 和主流程)
- _dimensional_pre_check 逐行扫描, "=" 不在行里跳过 (ponytail: 不做多行方程; 升级路径是 AST 解析)
- generate_evidence_manifest 对空 workspace 返回空 files 列表 (ponytail: 不抛, 让调用方决定怎么处理)
- SafetyMode.GUIDED 是默认 (ponytail: 安全默认, 用户不指定就问)
- no-agent baseline score=-1.0 (ponytail: 占位值, 不混淆真实分数)

## 升级路径
- v7: Darwin ratchet crossover/mutation (AlphaEvolve 风格) — v6 只用 Conjecture sorry 进化 (G48 是 AlphaEvolve 的子集)
- v7: 多 LLM 协作填 sorry — v6 单 LLM best-effort
- v7: lean_tool 当 verifier service — v6 用 sympy + AST + sorry 扫描替代 lean compiler
- v7: CSMListener 升级为强协议 — v6 是 Protocol (鸭子类型), 升级路径是各 phase manager 真正注册为 listener
- v7: relative_anchors cosine similarity 完整实现 — v6 只存 anchor 名列表, cosine 计算留给下游
- v7: filter-repo 一次性清理 16.5MB 死重 — v6 只防新增, 历史死重需用户批准 filter-repo
- v7: dimensional_analysis 接 sympy parsing — v6 用 regex 提取量纲, 升级路径是完整表达式解析
- v7: no-agent baseline 真正跑 LLM — v6 占位, 升级路径是接 LLM runner 跑 zero-shot
- v7: PhysicalStructure 持久化 — v6 是状态对象, 升级路径是接 memory/audit 持久化
- v7: enumerate_implementors 并行搜索 — v6 串行, 升级路径是并行 LLM + RAG recall
