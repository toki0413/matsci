# 部署方 E2E 验收 Checklist

> 部署到生产环境后, 必须手动执行以下验收项.
> 沙箱环境无法覆盖真实 LLM / HPC / 模拟软件 / 多用户并发, 这些只能在部署侧验证.

---

## 一、环境就绪检查

### 1.1 基础环境

- [ ] Python 3.11+ 已安装, `python --version` 输出正确
- [ ] 依赖已安装, `pip install -e ".[all]"` 无错误
- [ ] `HUGINN_API_KEY` 已设置为强随机值 (`python -c "import secrets; print(secrets.token_urlsafe(48))"`)
- [ ] `HUGINN_ADMIN_API_KEY` 已设置为不同的强随机值
- [ ] `HUGINN_JWT_SECRET` 已设置 (或复用 `HUGINN_API_KEY`)
- [ ] `HUGINN_DEV_MODE` **未设置** (生产环境必须关闭)
- [ ] `HUGINN_ENFORCE_WRITE_CAPABILITY=1` (默认开, 确认未被关闭)

### 1.2 服务启动

- [ ] `python -m huginn.server` 能正常启动, 无 import 错误
- [ ] `curl http://localhost:8000/health` 返回 200
- [ ] `curl http://localhost:8000/ready` 返回 200
- [ ] `curl http://localhost:8000/metrics` 返回 Prometheus 格式数据
- [ ] 日志无 ERROR 级别输出

---

## 二、真实 LLM 验证

> 沙箱用 FakeLLM 跑过 agent loop, 但真实 LLM 的工具调用格式、token 限制、
> 超时行为只能在生产验证.

### 2.1 OpenAI / Anthropic

- [ ] `HUGINN_PROVIDER` 设置正确 (openai/anthropic/google/ollama)
- [ ] `HUGINN_API_KEY` 是有效的 provider API key
- [ ] `HUGINN_MODEL` 设置为支持的模型 (gpt-4o / claude-3-opus / gemini-pro)
- [ ] 用 curl 测试 chat 端点:
  ```bash
  curl -X POST http://localhost:8000/v1/agents/default/chat \
    -H "Authorization: Bearer <jwt>" \
    -H "Content-Type: application/json" \
    -d '{"message": "Hello, what can you do?"}'
  ```
  返回非空, 且不是错误信息
- [ ] 测试工具调用: 问一个需要计算的问题 (如 "what is 2+2"), 确认 agent 调用了工具
- [ ] 测试多轮对话: 连续问 3 个相关问题, 确认上下文保持

### 2.2 Ollama (本地模型)

- [ ] Ollama 服务已启动, `ollama list` 显示模型
- [ ] `HUGINN_PROVIDER=ollama`, `HUGINN_MODEL=<model-name>`
- [ ] chat 端点能正常响应 (本地模型延迟较高, 设超时 60s+)

---

## 三、真实 HPC 集群验证

> 沙箱用 mock 测过 Paramiko/Slurm 代码路径, 真实集群验证必须跑
> [verify_hpc_environment.py](verify_hpc_environment.py).

### 3.1 SSH 连接

- [ ] `HUGINN_HPC_HOST`, `HUGINN_HPC_USER`, `HUGINN_HPC_KEY_FILE` 已设置
- [ ] `HUGINN_STRICT_HOST_KEY_CHECKING=true` (生产必须开)
- [ ] 集群指纹已在 `~/.ssh/known_hosts` 中
- [ ] `python -m tests.verify_hpc_environment` 8 项全部 PASS

### 3.2 Slurm 作业

- [ ] `sbatch --test-only` dry run 通过
- [ ] 提交一个 1 分钟的测试作业, 确认 `squeue` 能看到
- [ ] 作业完成后, 输出文件能通过 SFTP 取回
- [ ] 测试作业取消: 提交后立即 `scancel`, 确认状态变更

### 3.3 模拟软件

- [ ] `VASP_EXECUTABLE` 指向真实 vasp_std, 提交测试作业能跑
- [ ] `LAMMPS_EXECUTABLE` 指向真实 lmp, 提交测试作业能跑
- [ ] `QE_EXECUTABLE` 指向真实 pw.x, 提交测试作业能跑
- [ ] (按需) CP2K / OpenFOAM / COMSOL / ABAQUS 同上

---

## 四、真实用户场景验证

### 4.1 单用户完整工作流

- [ ] 用户登录 → 拿 JWT → 创建会话 → 调工具 → 写 memory → 登出
- [ ] 登出后, 同一个 JWT 调受保护端点返回 401 (吊销生效)
- [ ] 知识库: 上传 PDF → 检索 → 删除 → 验证清理
- [ ] 工作流: 选模板 → 执行 → 检查结果 (不依赖外部软件的模板)

### 4.2 多用户并发

- [ ] 2+ 用户同时调 chat 端点, 互不干扰 (不同 thread_id)
- [ ] 2+ 用户同时写 memory, 各自的 memory 不串
- [ ] 并发提交 5+ 个沙箱作业, 资源限制生效, 不 OOM

### 4.3 RBAC 权限

- [ ] VIEWER 用户能读 (GET), 不能写 (POST/DELETE 返回 403)
- [ ] OPERATOR 用户能写 memory / knowledge
- [ ] ADMIN 用户能调 `/admin/*` 端点
- [ ] 非 admin 用户调 admin 端点返回 403

---

## 五、安全验证

### 5.1 鉴权

- [ ] 无 API key / JWT 调受保护端点返回 401
- [ ] 错误的 API key 返回 401
- [ ] 过期 JWT 返回 401
- [ ] 篡改签名的 JWT 返回 401
- [ ] 吊销后的 JWT 返回 401

### 5.2 注入防护

- [ ] SQL 注入: memory search query 含 `' OR 1=1 --`, 不崩溃, 不返回全部数据
- [ ] 命令注入: bash_tool 参数含 `; rm -rf /`, 被沙箱拒绝
- [ ] 路径遍历: 静态文件请求 `../../etc/passwd` 返回 404
- [ ] SSRF: `/knowledge/ingest-url` 访问 `169.254.169.254` 被拦截

### 5.3 数据安全

- [ ] HTTPS 已启用 (nginx/Caddy 反代 + Let's Encrypt)
- [ ] API key 不出现在日志中 (检查 `/metrics` 和日志文件)
- [ ] memory.db / workspace 目录权限正确 (不 world-readable)
- [ ] 备份策略已配置 (memory.db + workspace 定期备份)

---

## 六、可观测性验证

### 6.1 日志

- [ ] 结构化 JSON 日志已启用 (`HUGINN_LOG_FORMAT=json`)
- [ ] 日志包含 request_id, 方便链路追踪
- [ ] ERROR 日志有告警通知 (邮件 / Slack / PagerDuty)

### 6.2 监控

- [ ] Prometheus 已配置抓取 `http://localhost:8000/metrics`
- [ ] Grafana dashboard 已导入, 显示 QPS / 延迟 / 错误率
- [ ] 告警规则已配置:
  - `HuginnHighErrorRate` (5xx > 0.1/s 持续 5min)
  - `HuginnSandboxFailure` (沙箱失败率 > 0.5/s 持续 2min)
  - `HuginnHighLatency` (P95 > 10s 持续 5min)

### 6.3 健康检查

- [ ] 负载均衡器健康检查指向 `/health` 或 `/ready`
- [ ] 服务重启后, 健康检查在 30s 内恢复
- [ ] `/health/rust` (如果用了 Rust 扩展) 返回 200

---

## 七、灾难恢复

- [ ] 服务进程崩溃后, systemd / supervisor 自动重启
- [ ] memory.db 损坏时, 服务能降级启动 (不 fatal)
- [ ] HPC 集群不可达时, 相关端点返回明确错误 (不 500 崩溃)
- [ ] LLM API 不可达时, chat 端点返回明确错误 (不 hang)

---

## 八、验收签字

- [ ] 以上所有项目已验证 (或已标记为不适用并说明原因)
- [ ] 验收人: _______________  日期: _______________
- [ ] 复核人: _______________  日期: _______________

---

## 沙箱已覆盖的 E2E (参考)

以下在沙箱已通过, 部署方无需重复:

| 套件 | 文件 | 测试数 | 覆盖范围 |
|------|------|--------|---------|
| 本地 E2E (用户旅程) | [tests/e2e_user_journeys.py](e2e_user_journeys.py) | 27 | 认证/知识库/Memory/工作流/RBAC/沙箱/健康检查 |
| LLM Mock E2E (agent loop) | [tests/e2e_agent_loop.py](e2e_agent_loop.py) | 7 | 单轮/多轮/容错/Memory/Telemetry |
| API 渗透测试 | [tests/pentest_api_security.py](pentest_api_security.py) | 12 | SSRF/JWT/RBAC/SQL注入/命令注入 |
| API 模糊测试 | [tests/fuzz_api.py](fuzz_api.py) | 6 类 | 畸形输入不崩溃 |
| 归档渗透测试 | [tests/pentest_archive_safety.py](pentest_archive_safety.py) | 6 | zip/tar slip / symlink / bomb |
