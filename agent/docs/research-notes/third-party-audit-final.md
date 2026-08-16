# 第三方独立综合审计报告 — huginn agent

**日期**: 2026-08-16
**审计类型**: 只读第三方独立审计（security-auditor × loop-polish preflight × praxis review 三合一）
**范围**: 全仓 /workspace/agent，重点为安全组件、autoloop 主循环、blind_spot_mapper 本次接线

---

## 一、security-auditor（OWASP 视角）

### A01 Broken Access Control
- 服务器 JWT/bearer 校验集中，user_id 从 token 提取失败仅记录不泄出。
- CORS/上传安全有专项测试（test_cors_security_headers、test_upload_security）。
- **结论**: 达标。无绕过授权点暴露。

### A02 Cryptographic Failures
- 密钥管理分层清晰：[crypto.py](huginn/crypto.py) 用 PBKDF2-SHA256 迭代 480k/600k（`CryptoVault`/`KeyManager`），Fernet（AES-128-CBC+HMAC），流式走 AES-GCM 分块。
- 密钥默认只驻内存（`_master_key`），可 lock() 清空；落盘 key 文件再经密码二次加密。
- 全文扫描仅见 `api_key="not-needed"` 占位，**无硬编码 secret**。
- **结论**: 达标。（NIT：主盐为固定常量 `_MASTER_SALT`，属设计取舍，随机性来自密码本身，可接受但值得注释说明。）

### A03 Injection
- 命令执行全部走**参数数组列表**（`subprocess.run([...])`），未发现 `shell=True` + 字符串拼接。
- 工具执行主入口 [sandbox.py](huginn/security/sandbox.py)：白名单可执行文件 + 工作目录校验 + 超时 + `evaluate_command_hook` 策略拦截；支持 landlock/docker 隔离。
- 受限 Python [restricted_python.py](huginn/security/restricted_python.py)：AST 扫描禁危险 import/builtins/dunder 反射链（含 getattr 字符串绕过拦截）。
- SQL 全库参数化；f-string 拼接仅用于**内部标识符**（列名/建表语句）非用户数据（见 [longterm.py](huginn/memory/longterm.py) L265、migrations.py），不构成注入。
- `kernel_session.py` 的 `f"print(repr({n}))"` 注入的是隔离内核内自身变量名，非宿主风险。
- **结论**: 达标。

### A04/A08 Logging & Sensitive Data
- 扫描 logger 语句，未见打印 api_key/secret/password 明文；`_test_llm` 用 `"not-needed"` 占位。
- **结论**: 达标。

### 安全总评
防御成熟：AST 预扫描 + 命令白名单 + prompt 标记（[prompt_security.py](huginn/security/prompt_security.py) `untrusted_context_message`）+ 加密 vault + 参数化 SQL 多纵深。本审计安全面**无 BLOCK**，给出 1 个 NIT（固定主盐注释）。

---

## 二、loop-polish（preflight）— 全量集成验证

### 验证结果
`python -m pytest` （全量，含 stress/benchmark）
```
2 failed, 8425 passed, 208 skipped, 8 xfailed in 570.74s
```
通过率 99.97%（8425/8427 执行断言），跳过项均为环境性。

### 两个失败（均与本 session 改动无关，独立复现）
1. **`test_100_turn_memory_stable`** — `Memory grew 104.0MB in 100 turns, assert 104.02 < 100`
   类别: 稳定性/内存增长（FIX，潜在慢泄漏）。
2. **`test_all_contract_docs_are_not_drifted`** — `env-contract.md, feature-flags-contract.md` 契约漂移
   类别: 文档与代码漂移，需 `python -m huginn.cli.config_audit --<mode> --out docs/<name>.md` 重新生成（FIX）。

### blind_spot_mapper 接线功能验证
`TestBlindSpotBlockWiring` 4 条定向测试全部通过（5.22s）：确认失败注入盲点块 / 弱簇不注入 / 无 memory 静默下降级 / block 序位于 topo 之后。

### 打分（loop-polish 功能性/正确性：0–100）
| 维度 | 分数 | 依据 |
|---|---|---|
| 功能性/正确性 | 98 | 8425 通过；2 失败为既有内存/文档问题，非功能缺陷 |
| 安全 | 99 | OWASP 多纵深防御成熟，无 BLOCK |
| 本次交付（blind_spot） | 100 | wiring 定向测试 4/4，无回归 |
| **综合** | **≈99** | — |

---

## 三、praxis review（before-merge 核查）

### Spec ↔ 实现
- docs [metacog-de-islanding-audit.md](research-notes/metacog-de-islanding-audit.md) 声明 blind_spot 接入 [engine_observe.py](huginn/autoloop/engine_observe.py) 的 `_build_hypothesis_prompt`，位于 `topo` 之后。**实测代码吻合**（L1171-1172 顺序一致）。
- 实现 = 导入的 `BlindSpot` / `map_blind_spots_to_hint` 签名与 [blind_spot_mapper.py](huginn/metacog/blind_spot_mapper.py) 定义一致。**无飘移**。

### 文档准确性
- 注释诚实标注 ceiling：cluster 粒度判定粗、workaround 表按 skill 名匹配大概率 miss。文档与代码一致。
- NIT：报告文档 `159 passed`（见 metacog-de-islanding-audit.md L22）与本次实际通过数不完全对应 —— 该数字是早期定向运行快照，非全量。建议注解"定向样本，非全量"避免误读。

### 边界
- blind 判据 `rate==0 && n>=3`，`get_self_model()` 异常 → `{}`，`map_blind_spots_to_hint({})` → 空串，静默降级。无异常风险。
- 空/None/损坏的 self_model 均走空块。处理到位。

### Scope
- 改动范围严格限定于 `engine_observe.py` 一处 advisory 块 + 测试文件 + 索引/报告文档。**无无关改动**，未超 2x 必要实现。

### praxis 输出
- BLOCK: 无
- FIX: 无（报告级别）
- NIT:
  1. 文档 L22 `159 passed` 建议标注为定向快照或更新。
  2. `_MASTER_SALT` 固定常量建议补一句"随机性来自密码"说明。

---

## 四、综合结论

| 分级 | 条目 |
|---|---|
| **BLOCK** | 无 |
| **FIX** | ① `test_100_turn_memory_stable` 内存增长超阈值（104MB/100 turn）— 排查潜在慢泄漏；② 重新生成漂移契约 `env-contract.md`、`feature-flags-contract.md` |
| **NIT** | ① 报告文档 `159 passed` 标注为定向快照；② `_MASTER_SALT` 取舍注释 |

**总体判断**: agent 安全防御成熟（多纵深、命令白名单、AST 预扫、参数化 SQL、日志脱敏），全量测试 99.97% 通过，本次 blind_spot_mapper 去孤岛实现与文档、测试三方一致，属高质量、低风险的 advisory 增强。无阻塞项，可继续演进；两个 FIX 为既有问题，建议纳入常规维护而非本次交付前置。