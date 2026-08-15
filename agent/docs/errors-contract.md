# 错误语义契约 (ErrorKind)

自动生成: `python -m huginn.cli.config_audit --errors --out docs/errors-contract.md`.
登记 `ToolResult.error_kind` 的分类语义 (core_types.py::ErrorKind)。下游 (debugging / trace / auto-retry) 据此区分失败类别; 默认 `NONE` 保持既有调用方行为不变。`points` 是产出该分类的静态扫描位置。映射入口: `tools/bash_tool.py::_result_error_kind` (returncode==0→NONE, timed_out→TIMEOUT, blocked→DENIED, 其余→FATAL)。

### ErrorKind 类别

| 类别 | 语义 | 产出点 |
|---|---|---|
| NONE | 正常或模型可见的业务失败 | core_types.py:77, tools/bash_tool.py:146 |
| TIMEOUT | 沙箱/命令超时 | tools/bash_tool.py:148 |
| DENIED | 沙箱策略拒绝 (SandboxError / result.blocked) | tools/bash_tool.py:150, tools/bash_tool.py:323, tools/code_tool.py:200 |
| SIGNAL | 被信号终止 |  |
| TRANSIENT | 瞬时错误, 可安全重试 |  |
| FATAL | 不可重试 | tools/bash_tool.py:151, tools/bash_tool.py:329, tools/code_tool.py:153 |
