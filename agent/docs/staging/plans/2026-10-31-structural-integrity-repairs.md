# 计划：结构完整性修复（四轨）

spec: `docs/staging/specs/2026-10-31-structural-integrity-repairs.md`
milestone: M-INT1（A轨）、M-INT2（B轨）、M-INT3（C轨）、M-INT4（D轨）

## A. 契约进数据流

- [x] T1: contract 加载器与 sha
  goal: `contract.json` 解析成强类型对象并算 sha256
  files: `huginn/task_contract.py`、`tests/test_task_contract.py`
  acceptance: pytest tests/test_task_contract.py -q；缺字段/坏 schema 抛 ContractError
  spec: `#a-契约进数据流`

- [x] T2: 门切到 contract
  goal: 交付门只读 contract，删掉 objective 散文正则解析路径
  files: `huginn/autoloop/deliverables.py`、`tests/loops/test_structural_guards.py`
  acceptance: 原 ws35 fixture PASS；ws34 fixture 因 loop_log 字段缺失 FAIL
  spec: `#a-契约进数据流`

- [x] T3: 启动校验与产物绑定
  goal: driver 起跑前校验 contract；results.json 写 `contract_sha256`，门复核
  files: `huginn/run_cognitive_driver.py`、`huginn/autoloop/deliverables.py`
  acceptance: 无 contract 且未设 HUGINN_ALLOW_UNCONTRACTED → 拒跑；sha 不一致 → FAIL
  spec: `#a-契约进数据流`

- [x] T4: 双任务夹具回归
  goal: ws35 与 ws34 两个真实题面/产物做成只读夹具，锁住「同门不同题」
  files: `tests/fixtures/contracts/`、`tests/test_contract_fixtures.py`
  acceptance: 两夹具判定与人工评估一致（35 过、34 挂 loop_log）
  spec: `#a-契约进数据流`

## B. 默认拒绝

- [x] [parallel] T5: local bash 默认拒绝
  goal: 未显式开启或 landlock 不可用时抛 SandboxError
  files: `huginn/security/execution.py`、`huginn/security/landlock.py`
  acceptance: 默认环境 get_executor() 抛错；`HUGINN_UNSAFE_ENV=1` 才放行且标 unsafe
  spec: `#b-默认拒绝安全基线`

- [x] [parallel] T6: 审批三档
  goal: `HUGINN_APPROVE_MODE` = deny|workspace|all，默认 workspace
  files: `huginn/permissions.py`、`huginn/autoloop/engine.py`
  acceptance: 默认下工作区外写被拒；`all` 需显式设置
  spec: `#b-默认拒绝安全基线`

- [x] [parallel] T7: 子进程环境剥离
  goal: 敏感键不进 bash 子进程 env
  files: `huginn/security/sandbox.py`、`huginn/security/container_executor.py`
  acceptance: 子进程 `env` 看不到 INTERNLM_API_KEY/TOKEN 类变量
  spec: `#b-默认拒绝安全基线`

- [x] T8: 安全回归
  goal: 三条默认拒绝都有守护测试
  files: `tests/test_security_defaults.py`
  acceptance: pytest tests/test_security_defaults.py -q 全绿
  spec: `#b-默认拒绝安全基线`

## C. 文档断言

- [x] [parallel] T9: 文档标记与生成比对
  goal: generated/manual/无标记三类，无标记即 FAIL
  files: `scripts/doc_assertions.py`、`tests/test_doc_assertions.py`
  acceptance: 四个目标文档全绿；故意改 mece-audit → FAIL
  spec: `#c-文档被断言或删除`

- [x] T10: env 契约交叉检查
  goal: docs 中出现的 `HUGINN_*` 必须存在于 env_schema
  files: `scripts/doc_assertions.py`、`docs/env-contract.md`、`docs/feature-flags-contract.md`
  acceptance: 伪造一个不存在的键 → FAIL
  spec: `#c-文档被断言或删除`

## D. 执行覆盖

- [x] T11: trace 采样器
  goal: `HUGINN_TRACE=1` 时按文件计数并落 trace.json
  files: `huginn/trace.py`、`tests/test_trace.py`
  acceptance: 造两次假调用 → trace.json 计数正确
  spec: `#d-执行覆盖度量`

- [x] T12: 驱动接线与覆盖报告
  goal: driver 结束时写 trace 并对照 census 列「KEEP 但零执行」
  files: `huginn/run_cognitive_driver.py`、`huginn/trace.py`
  acceptance: offline 跑 2 轮 → trace.json 含 engine/autoloop，pet 不在
  spec: `#d-执行覆盖度量`

## 收口

- [ ] T13: 全量回归与治理重生成
  goal: 一次全绿 + 镜像/mece 重生成
  files: `docs/mece-audit.md`、镜像产物
  acceptance: `pytest tests/ -o addopts="" -q` 全绿；contract_audit 与 mirror_drift 通过
  spec: `#测试总口径`
  status: 未闭合。已修两类阻断（pytest 9 与 pytest-benchmark 不兼容 → 锁 `pytest<9`；chromadb ONNX 启动期联网下载把 TestClient 卡死 → 未缓存即降级 + 有界探测）。剩余硬限制是内存：cgroup 4GB 无 swap，单进程 19%、70 文件分块 83% 均被 OOM 杀（EXIT=137；CI 专门加 swap）。需 swap 或继续细分块跑完后再勾。
