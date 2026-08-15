# 模型路由契约 (ModelRouter)

自动生成: `python -m huginn.cli.config_audit --routes --out docs/routes-contract.md`.
登记 ModelRouter 的 task → 偏好 tag 映射 (`_TASK_TAGS`)。`select(task)` 按列表顺序找第一个有匹配模型的 tag; 无匹配回落默认。任务也可经 `HUGINN_MODEL_<TASK>` 环境变量装配 (见 models/router.py::from_env)。

| task | 偏好 tag (按序) |
|---|---|
| `default` | default, agent |
| `agent` | agent, default |
| `coding` | coding, agent, default |
| `science` | science, reasoning, agent, default |
| `reasoning` | reasoning, science, agent, default |
| `summarize` | summarize, cheap, default |
| `format` | format, cheap, default |
| `cheap` | cheap, summarize, default |
| `local` | local, default |
| `verification` | verification, reasoning, science, default |
| `archival` | archival, cheap, summarize, default |

可选 task 全集 (11): `default`, `agent`, `coding`, `science`, `reasoning`, `summarize`, `format`, `cheap`, `local`, `verification`, `archival`
