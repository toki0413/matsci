# 结构完整性修复（四轨）

日期：2026-10-31
milestone: M-INT1..M-INT4（见 `docs/ROADMAP.md` 新增轨道）
worktree: 否 —— 四轨共享契约/门代码，并行会互相改写;单线程执行

## 背景（一句话）

验收靠「事后读产物 + 运行时解析散文契约」,约束靠「默认允许」,度量靠「设计覆盖率」。本 spec 只做四件反转默认的事,不新增机制族。

## A. 契约进数据流

contract: `contract.json`（工作区根,只读,由任务下发方提供）,schema 版本 `huginn.task-contract/v1`
- 字段:`schema, required_files[], required_keys[{key,type}], hypotheses{labels[],prediction_only[]}, loop_log{path,lines,fields[]}, static{forbidden_imports[],seed}`
- 优先级:contract 覆盖 objective 散文;**objective 不再被正则解析**
- 启动:driver 起跑前加载+校验;缺失且未设 `HUGINN_ALLOW_UNCONTRACTED=1` → 拒绝启动
- 绑定:交付物 `results.json` 必须写入 `contract_sha256`;门用同一份字节校验,不一致即 FAIL
- invariant:同一契约字节 → 同一判定结果,与题面措辞无关
- test:ws35 fixture 通过;ws34 fixture 因 loop_log 字段缺失 FAIL;改一个字 → sha 变化 → FAIL

## B. 默认拒绝（安全基线）

contract:
- `allow_local_bash()` = `HUGINN_ALLOW_LOCAL_BASH==1` **且** 本机 landlock 可用;否则 `SandboxError`
- `autoloop_auto_approve()` 默认 `workspace`:仅放行工作区内读写与只读命令;`all` 必须显式设 `HUGINN_APPROVE_MODE=all`
- 子进程环境剥离:匹配 `(API_KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)` 的变量不传给 bash 子进程(LLM 凭据仅进程内注入)
- invariant:无显式开关时,任何「工作区外写 / 任意环境变量外泄 / 未隔离执行」都必须失败而非降级
- test:默认环境跑 bash 越界写 → 拒绝;`env` 输出不含密钥;未装容器且未开 local bash → 拒跑

## C. 文档被断言或删除

contract:`docs/*.md` 首行三类标记之一 —— `<!-- generated: <cmd> -->` / `<!-- manual: true -->` / 无标记
- generated:测试重跑生成命令,字节比对
- manual:必须通过 env 键交叉检查(出现的 `HUGINN_*` 必须存在于 `env_schema`,新增键必须进文档)
- 无标记:测试 FAIL,要么补标记要么删文件
- 范围:先覆盖 `docs/env-contract.md`、`docs/feature-flags-contract.md`、`docs/mece-audit.md`、`docs/INDEX.md`;其余按批次扩展
- test:`tests/test_doc_assertions.py` 全绿;故意改 env-contract 一个键 → FAIL

## D. 执行覆盖度量

contract:`HUGINN_TRACE` 默认开,`=0` 显式关闭;`sys.setprofile` 采样,落 `<workspace>/trace.json`:每模块调用计数 + 未执行清单
- 输入对照:`research/huginn_module_census.csv`(123 顶层,97 KEEP)
- invariant:一次真实运行后,「KEEP 但零执行」的模块必须可被列出;不产生 trace 时不得声称已度量
- test:用 offline provider 跑 2 轮微型任务 → trace.json 含 engine/autoloop,counter>0;不含 `pet`;`HUGINN_TRACE=0` 时不落文件

## E. 结构登记与流程门

contract:
- 模块普查双向门:census 与真实目录/注册表对齐,漏登记与幽灵行都 FAIL
- 注释卫生:注释只写 why/约束;禁止 changelog(`以前/旧实现/早期`)与任务号考古(`P4-2:`);`comment-ok` 显式豁免;基线只减不增
- 流程门:每个 plan 必须在头部声明 `spec:` 指向 `docs/staging/specs/*.md`;每个任务必须带 `spec:` 字段,内容为 `#section` 锚点且该节真实存在;缺任一即不得开工
- 边界:spec 必须包含边界节(`非目标` / `deferred` / `failure modes` / `out of scope`)
- test:`tests/test_comment_hygiene.py`、`tests/test_silent_exceptions.py`、`tests/test_workflow_gate.py` 全绿;故意写一条 `P4-9:` 注释、或任务漏写 spec 锚点 → FAIL

## F. 模型面度量

contract:用真实 prompt 段渲染 + 真实 `ToolRegistry` 工具定义度量字符预算(与 LLM 收到的字节一致);空描述/占位标记硬判
- 冻结前只测不判;观察期后把实测上限写入 `BUDGET`,之后只减不增
- test:`tests/test_model_surface.py`;实测 main 9765 / s7 11491 / autoloop 7236 字符,工具定义 217685 字符(125 可见工具)

## G. 执行面: 多执行模式

contract: 开工方式本身是 plan 里的显式选择, 不是默认默契.
- 模式:`inline`(主上下文顺序执行)/ `subagent`(逐任务派发给隔离子代理, 只回收摘要)/ `mixed`(逐任务声明)
- plan 头部声明 `execution: <mode>`;任务可用 `exec: inline|subagent` 覆盖;`mixed` 下每个任务必须显式声明 `exec:`
- 选择规则:需要跨文件推理、与用户交互、依赖前序结果的任务走 `inline`;文件集互不重叠且验收可独立执行的走 `subagent`
- 派发前提:`[parallel]` 任务的文件集必须两两不交;`exec: subagent` 任务必须带 `files` 与 `acceptance` —— 否则子代理无法独立完成与自验
- invariant:未声明执行模式的 plan 不得开工;共享文件的任务不得并行派发
- test:`tests/test_workflow_gate.py` 的执行面校验全绿;合成仓验证未知模式 / `mixed` 缺声明 / 并行共享文件 / 子代理缺 `acceptance` 分别 FAIL

## failure modes

- contract.json 手改与 objective 冲突 → 拒绝启动(不猜)
- 无 landlock 的容器/CI → 显式 `HUGINN_UNSAFE_ENV=1` 才可跑,并在 results 里记 `unsafe: true`
- trace 采样开销过大 → 仅按文件计数,不记录参数

## 非目标

- 不做 git hook / CI 强制;门只做仓内可执行断言,流程靠 spec → plan → 开工的顺序文件留下证据
- 不一次清空存量:151 处注释考古与 427 处静默异常先冻结基线,分批收敛
- 不复制 dsh 代码,只移植其门的原则
- 不替人挑执行模式:门只校验声明合法与派发前提,不在 `inline` / `subagent` 之间自动裁决

## deferred

- 57 个零引用文件的删除:单独批次,需逐条确认
- 17 个 MOVE 模块的迁移:不动
- 契约 schema 的 JSON Schema 正式校验:先用 dataclass + 手写校验

## 测试总口径

`cd /workspace/agent && /workspace/venv/bin/python -m pytest tests/ -o addopts="" -q` 全绿;四轨各自新增守护测试;不可引入新 md 文档(本 spec/plan 除外)。
