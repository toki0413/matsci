# 插件契约 (Everything is a Plugin)

自动生成: `python -m huginn.cli.config_audit --plugins --out docs/plugins-contract.md`.
登记项目内所有可插拔注册面, 分两种形态: **形态 B** (策略注册表, 同步选策略, `register_*_policy` / `register_prompt_segment`), **形态 A** (事件钩子, `@filter.on_xxx`, 异步分发 + 可阻断)。静态扫描只报注册位置与显式 priority, 不判生命周期; `priority —` 表示注册点未显式传参 (用内置默认)。

### Prompt 段 (形态 B)

| 段名 | priority | 注册位置 |
|---|---|---|
| persona | 10 | agent/prompt_builder.py:276 |
| mode | 20 | agent/prompt_builder.py:277 |
| phase | 30 | agent/prompt_builder.py:278 |
| metacog | 40 | agent/prompt_builder.py:279 |
| tools | 50 | agent/prompt_builder.py:280 |
| thinking | 100 | agent/prompt_builder.py:282 |
| safety | 200 | agent/prompt_builder.py:283 |

### Compaction 策略 (形态 B)

| 策略名 | priority | 注册位置 |
|---|---|---|
| — |

### 记忆整理策略 (形态 B)

| 策略名 | priority | 注册位置 |
|---|---|---|
| — |

### 事件钩子 (形态 A)

| 事件 | handler | 注册位置 |
|---|---|---|
| on_llm_request | inject_rules | plugins/ponytail/main.py:175 |
