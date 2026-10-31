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

contract:`HUGINN_TRACE=1` 时 `sys.setprofile` 采样,落 `<workspace>/trace.json`:每模块调用计数 + 未执行清单
- 输入对照:`research/huginn_module_census.csv`(123 顶层,97 KEEP)
- invariant:一次真实运行后,「KEEP 但零执行」的模块必须可被列出;不产生 trace 时不得声称已度量
- test:用 offline provider 跑 2 轮微型任务 → trace.json 含 engine/autoloop,counter>0;不含 `pet`

## failure modes

- contract.json 手改与 objective 冲突 → 拒绝启动(不猜)
- 无 landlock 的容器/CI → 显式 `HUGINN_UNSAFE_ENV=1` 才可跑,并在 results 里记 `unsafe: true`
- trace 采样开销过大 → 仅按文件计数,不记录参数

## deferred

- 57 个零引用文件的删除:单独批次,需逐条确认
- 17 个 MOVE 模块的迁移:不动
- 契约 schema 的 JSON Schema 正式校验:先用 dataclass + 手写校验

## 测试总口径

`cd /workspace/agent && /workspace/venv/bin/python -m pytest tests/ -o addopts="" -q` 全绿;四轨各自新增守护测试;不可引入新 md 文档(本 spec/plan 除外)。
