# Security Hardening + Bench Activation + System Slimming (v5)

Source: v5 spec `security-hardening-and-bench-activation` implementation.
覆盖 25 加法 (G20-G44) + 8 减法 (R14-R21) + 5 里程碑 (M1-M5).
集成 self-check 33/33 全过. Commit `feat(v5): security hardening + bench activation + system slimming`.

## 1. M1 — 7 项 P0 安全修复 (G20-G26)

每条 cost 一次 audit 报告 + 一次修复, 不修不能上线.

| ID | 修复 | 教训 |
|----|------|------|
| G20 | `is_dev = true` 硬编码 → `env::var("HUGINN_DEV_MODE")` | 桌面默认 dev 模式 = 生产环境 RCE, 编译期常量不能当开关 |
| G21 | Tauri webview shell allowlist `*` → `["open", "python"]` + 参数白名单 | shell allowlist 通配符 = 任意命令执行, webview 加固必须配 origin 白名单 |
| G22 | `restricted_python.py` 加 `visit_ImportFrom` 拦 `from builtins import *` | AST 只查 `Import` 节点漏 `ImportFrom`, import 逃逸是 sandbox 头号漏洞 |
| G23 | `getattr(getattr(obj, '__class__'), '__bases__')` 链拦截 | `__class__.__bases__.__subclasses__()` 是 Python sandbox escape 经典路径, 单层 attr 黑名单不够 |
| G24 | `security/secrets.py` 集中管理 + 7 副本清理 | `grep -r "sk-"` 全库明文密钥 = git push 一次就泄露, secrets 必须从环境变量读 |
| G25 | CI 加 `cargo test --workspace` + `pnpm test` 门禁 | 13 个已知失败测试 = CI 长期红, 没人看; 门禁必须覆盖所有语言 |
| G26 | `rubric.json` 不写 agent workspace, 改写 `rubric_hash.txt` | 评测泄漏 = agent 直接读答案拿满分, hash 比对 + `_execute_training_fallback` 不代跑 |

## 2. M2 — 5 个跑分生效件 (G27-G31)

每条单行/单函数 diff, 直击 19/20/21 号诊断报告暴露的能力短板.

| ID | 修复 | 跑分影响 |
|----|------|----------|
| G27 | RCB `tool_filter` 加 `symbolic_math_tool` / `lean_tool` / `validate_tool` | 数学工具实现完成度 7/10 但未注册 → 0 次调用; 注册后可被 agent 选用 |
| G28 | `_recompute_report_metrics` 解析 MAE/R²/accuracy/loss + >10% 偏差 flag | critique 只读 text 给 LLM 判断 = 循环论证, 重算打破闭环 |
| G29 | checklist 永驻 `system_prompt` + `STABLE_PRINCIPLES` 重载 | checklist 塞对话历史 = compaction 压缩丢, 永驻才不丢 |
| G30 | `HUGINN_RCB_INHERIT_PRINCIPLES` + 全局 `~/.huginn/stable_principles.jsonl` | 27 次 Material_003 重跑每次从零 = S7 不闭环; 跨任务复用才闭环 |
| G31 | `_perceive` empty 时 bypass 进 hypothesize + trajectory 0 tool_calls warning | 18/18 轨迹 perceive+report 两阶段 = 装置空转, 强制进 hypothesize 才激活 |

## 3. M3 — 8 个系统性主题生效件 (G32-G39)

每主题打 1-2 个最小生效件, 不引入新抽象.

| ID | 主题 | 生效件 |
|----|------|--------|
| G32 | fail-secure 文化 | `require_capability` 在 routes 入口 + `policy_engine` default `ask` (不 auto-allow) |
| G33 | 事件循环阻塞 | `asyncio.to_thread` 包装 14 个同步工具, 不再卡 event loop |
| G34 | 压缩管线空转 | `context_builder.py` `include_history = True` 默认, 历史真正进 graph |
| G35 | FTS5 索引腐坏 | `_rebuild_fts_index` + 启动校验 + bulk DELETE 路径触发重建 |
| G36 | E2E 测试 raise 化 | `report()` 从 print 改 `raise AssertionError`, 关键路径断言 |
| G37 | 修复回潮检测 | `audit-regression.yml` cron weekly 跑 7 项历史修复断言 (Born/BM3/DEM/MSD/SCF/RBAC/FTS5) |
| G38 | 文档真实性 | `MONITORING.md` `/metrics` 描述校准 (`_PUBLIC_PATHS` 公开), 不再瞎写 "requires admin API key" |
| G39 | KB 单例忽略 workspace | `get_knowledge_base(workspace)` factory, 按 workspace 路径缓存实例 |

## 4. M4 — 2 个外部参考接入 (G40-G41)

| ID | 参考 | 实现 |
|----|------|------|
| G40 | SCALECUA 任务合成 | `bench/task_synthesizer.py` — `synthesize_task(domain, difficulty)` + `sanity_check_judge(judge_script, known_answer)` 硬门禁 |
| G41 | Conjecture Machines 进化环 | `lean/conjecture_library.py` — `evolve_conjectures(seed, n_variants, max_gen)` + `ConjectureLibrary` SQLite 累积验证命题 |

### G41 关键 bug 修复: SQLite Windows 文件锁
- `sqlite3.Connection.__exit__` 只 commit 不 close, Windows 上文件锁残留阻碍 `TemporaryDirectory` 清理
- 修复: `_connect` 返回 `contextlib.closing(sqlite3.connect(...))` 包装
- 教训: `with sqlite3.connect(path) as conn:` 的 `__exit__` 行为跨平台不一致, Windows 文件锁是隐性 bug 源

## 5. M5 — 3 个 v4 预留接口实现 (G42-G44)

| ID | 接口 | 实现 |
|----|------|------|
| G42 | `critique_decision` | `cli/rcb_runner.py` — `Decision` + `CritiqueResult` dataclass + 模板规则三层检查 + LLM 路径复用 adversarial_critique 独立调用 |
| G43 | recall 策略学习 | `metacog/__init__.py` — `_DEFAULT_RECALL_STRATEGIES` 7 条 + `load_recall_strategy` / `match_recall_strategy` / `update_recall_strategy` (S7 写入端, 原子写 .tmp + rename) |
| G44 | multi-agent metacog | `memory/longterm.py` — `HUGINN_MULTI_AGENT=True` opt-in + `_stable_principles_lock` 双平台 (msvcrt/fcntl) + best-effort 失败不阻断 |

### G42 设计要点
- 模板规则三层: kind 合法 (`mode_switch`/`phase_transition`/`tool_select`) + frm/to 在合法集合 + rationale 非空
- `_VALID_MODES = {chat, plan, research}`, `_VALID_PHASES = 7-phase pipeline`
- `report → execute` 回退需 `context["force"]=True`, 防止意外回退
- LLM 路径: 模板 reject 直接返回, 模板 accept 才调 LLM 做深度 critique

### G43 策略表设计
- 默认 7 条策略: 每个 phase × 关键 metacog_state (S1_DISCOVER / S2_HYPOTHESIZE / S3_PLAN / S4_ACT / S5_VERIFY / S6_RESOLVE)
- `match_recall_strategy`: 精确 (phase, metacog_state) 匹配 → phase-only 退让
- `update_recall_strategy`: S7 闭环写入端, 原子写防并发损坏

### G44 文件锁设计
- `_LOCK_PLATFORM`: windows (msvcrt) / posix (fcntl) / none (no-op)
- `_stable_principles_lock(path, exclusive=True/False)`: 排他锁 (写) / 共享锁 (读)
- 锁文件用 `.lock` 后缀, 不污染数据文件
- best-effort: 锁失败不阻断 (跨 agent 竞态是低频事件, 阻断反而破坏单 agent 可用性)

## 6. 减法 (R14-R21)

| ID | 减法 | 决策 |
|----|------|------|
| R14 | 砍 `_DOMAIN_KNOWLEDGE` 8 领域硬编码表 (107 行) | 改走 RAG recall (`knowledge_seed` category); RAG 无数据降级到抽象概念, 结构仍正确 |
| R15 | `_CAUSAL_TRIGGERS` / `_ACTION_ABSTRACTIONS` 评估后保留 | SignalHub 是系统信号路由不做 NLP 关键词匹配, 替代不适用; 保留模板表供无 LLM 场景 |
| R16 | 5 个静态 benchmark 不做更细 deliverable | 保留 sanity baseline, 更细由 task_synthesizer 替代 |
| R17 | 砍 `min_calls=50% budget` 补丁 | SCALECUA task_synthesizer 按 (paper, complexity_tier) 给定难度, 不再用 budget 下限强迫消耗 |
| R18 | 历史 16.5MB 死重清理 | 评估后跳过: `git filter-repo` 重写所有 commit hash, 破坏 clone/fork; 留作独立维护任务 |
| R19 | 合并 4 套状态机 → CSM | 评估后推迟到 v6: 涉及 autoloop/engine + cognitive_engine + workflow 三处, 改动面太大 |
| R20 | 砍 dead code | `mark_seed_reflected` / `forest_orchestrator` v4 R10/R11 已清干净, grep 零命中 |
| R21 | 文档校准 | `MONITORING.md` `/metrics` 描述校准; `MATERIAL_SCIENCE_TOOL_WHITELIST` 实际存在 (spec 描述错误) |

## 7. 关键教训

1. **评测泄漏封堵必须用 hash, 不能用文件名遮挡** — agent 会 `ls` + `cat` 任何文件, 只有 hash 比对才安全
2. **AST 沙箱必须查 `ImportFrom`** — `from builtins import *` 是 `Import` 节点漏的逃逸路径
3. **`sqlite3.Connection.__exit__` 跨平台不一致** — Windows 上只 commit 不 close, 文件锁残留; 用 `closing()` 包装
4. **checklist 永驻 system_prompt, 不进对话历史** — 对话历史是 process 可压缩, checklist 是 task contract 不能丢
5. **跨任务 stable_principles 复用是 S7 闭环关键** — 否则 27 次重跑每次从零, S7 等于没闭环
6. **autoloop 装置激活要强制 bypass perceive** — 否则首轮 perceive empty → continue → 18/18 轨迹空转
7. **policy_engine default `ask` 是 fail-secure** — 不是 auto-allow, 用户必须确认
8. **多 agent 文件锁要 opt-in** — 默认开锁会破坏单 agent 可用性, `HUGINN_MULTI_AGENT=True` 才开
9. **`min_calls` 补丁是占位, SCALECUA 才是解决方案** — 占位补丁会浪费 budget, 真解决方案是任务合成给定难度
10. **RAG 替代硬编码领域表** — 硬编码 `_DOMAIN_KNOWLEDGE` 是经验主义, RAG 才对; RAG 无数据时降级到抽象概念, 结构仍正确

## 8. v6 预留接口

- **R19 状态机合并**: 4 套状态机 (autoloop engine + CSM + cognitive_engine + workflow state) → CSM 唯一状态机, 其他改 CSM 观察者. 大重构, 需独立测试周期
- **R18 历史 16.5MB 死重清理**: `git filter-repo --strip-blobs-with-ids large-blobs.txt`, 需备份 + 用户显式批准
- **L2 critique 扩展**: 当前 `critique_decision` 覆盖 mode/phase/tool, v6 扩展到 plan/strategy 层
- **active recall 策略学习**: 当前 G43 是手动策略表, v6 加 S7 自动学习
- **multi-agent metacog**: 当前 G44 是同 workspace 内共享, v6 扩展到跨 workspace 通信

## See Also
- `39_orchestrator_unification_lessons.md` — Orchestrator 4 层架构 + min_calls 补丁历史 (v5 R17 撤销)
- `.trae/specs/security-hardening-and-bench-activation/spec.md` — v5 spec 完整加法/减法清单
