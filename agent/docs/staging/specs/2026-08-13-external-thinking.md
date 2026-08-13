# Spec: externalThinking → deep_think 外部草稿纸工具

> 背景：厂商隐藏原生推理（chain-of-thought）后，模型"愿意"暴露的
> `reasoning_content` 并非总有。oh-my-pi 的 `externalThinking` + `think` 工具
> 思路：提供一个普通工具，让模型在动手前把分析写进工具参数，经 API 返回给
> 开发者直接读取保存。本 spec 把该机制做成 Huginn 内可一键开启的正式功能，
> 与现有 `reasoning_content` 捕获（`session.reasoning_trace`）共用同一蒸馏通道。

## Decisions

- **contract**: 新增核心工具 `deep_think`（`huginn.tools.deep_think_tool.DeepThinkTool`），
  注册进 `_CORE_MODULES`。输入 `analysis: str`（必填）。无副作用，`read_only=True`。
- **data shape**: 工具执行时经 `context.memory_manager.add_reasoning(analysis)` 写入
  `session.reasoning_trace`（与 [streaming.py](file:///workspace/agent/huginn/agent/streaming.py#L400-L410)
  的 `reasoning_content` 捕获同通道）。返回 `success=True` + 简短确认，不把分析内容
  回显给 LLM（避免重复占用上下文）。
- **feature flag**: 新增 `external_thinking` 开关（默认 False），纳入
  [feature_flags.py](file:///workspace/agent/huginn/feature_flags.py#L30) 的 `_DEFAULTS` /
  `_DESCRIPTIONS`。开启方式复用现有三层：config `feature_flags` / 环境变量
  `HUGINN_FEATURE_EXTERNAL_THINKING=true` / 运行时 `FeatureFlags.enable()`。
- **prompt injection**: 开启时在 [context.py](file:///workspace/agent/huginn/agent/context.py#L36)
  `_effective_system_prompt()` 末尾追加一段指令：要求模型在回答问题/改代码/调其他工具前，
  先调用 `deep_think` 把分析过程写进工具。关闭时不注入（保持默认行为不变）。
- **failure mode**: `external_thinking` 开启但 `memory_manager` 为 None（如某些 bench 场景）
  时，工具仍返回成功占位但不记录（fail-open，不阻塞主流程）。工具注册缺依赖不可能
  （纯 stdlib/pydantic，无第三方依赖）。
- **out of scope**: 不实现"强制关闭原生 reasoning"（`forceReasoningOff`）。Huginn 现状
  依赖模型暴露 `reasoning_content`，deep_think 是**补充**通道而非替代。原生+外部双通道
  并存是接受的行为（见 Working notes）。
- **test**: 新增 `tests/test_deep_think_tool.py`：
  1. 工具注册成功且 `read_only=True`；
  2. 调用后 `analysis` 落入 `memory_manager.session.reasoning_trace`；
  3. `memory_manager=None` 时 fail-open（success=True，不抛）；
  4. `external_thinking` 开启时系统提示包含 deep_think 指令，关闭时不包含。

## Working notes

- oh-my-pi 的 `think` 工具限定 GPT/Claude/Gemini（支持原生推理替换的 transport），并用
  `forceReasoningOff` 关闭原生思考，避免双通道并存。Huginn 当前对接 provider 多样
  （OpenAI-compatible/Anthropic/Ollama/国产模型），统一强制关原生不现实，故本 spec
  采用"补充通道"策略：deep_think 拿到的是显式草稿，`reasoning_content` 拿到的是原生
  推理，两路都汇入 `reasoning_trace`，蒸馏闭环消费。若未来某 provider 确定不支持原生
  且可强制关，可加 provider 级开关（deferred）。
- 潜在权衡：模型可能敷衍调用 deep_think（写空/套话）。本 spec 不强制校验内容质量，
  由蒸馏闭环的 `verification_status` 机制自然过滤（confirmed 才进知识库）。
- 系统提示注入点选 `_effective_system_prompt()` 而非 personas.py，因为该函数是唯一
  统一入口（research/extreme 模式 + 环境上下文都经过它），且可按 `_mode` 条件注入。

## Deferred

- provider 级 `forceReasoningOff` 开关（需按 provider 能力逐一定制）。
- deep_think 内容质量校验/阈值（交给蒸馏闭环过滤）。