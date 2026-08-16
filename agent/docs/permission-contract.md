# 权限契约 (Permission)

自动生成: `python -m huginn.cli.config_audit --permission --out docs/permission-contract.md`.
登记权限面: **PermissionMode** (二元决策) 与 **RiskLevel** (五档风险, 与 `ontology/actions.RiskLevel` 同粒度) 互补。`PermissionConfig` 提供细粒度判定维度: 多阶段叠加 (危险命令 → 路径规则 → 工具基础规则 → 成本分级 → 信任自适应), 每命中一个维度记入 `matched_rules` 供可观测。安全层 (危险命令 / 沙箱硬底线 / 成本预算) 即使 `auto_approve_all` 也保留。

### PermissionMode (二元决策)

| 模式 | 语义 |
|---|---|
| auto | 只读/安全工具直接放行 |
| ask | 潜在昂贵/破坏工具需确认 |
| deny | 显式拦截, 不可执行 |
| plan | 只读模式, 所有写工具强制 ASK |

### RiskLevel (五档风险)

| 等级 | 语义 |
|---|---|
| none | 纯只读/查询, 直接放行 |
| low | 本地只读/可逆变更, 默认放行 |
| medium | 外部 IO/网络/非破坏状态变更, 默认需确认 |
| high | 破坏性/危险, 必须确认 |
| critical | 不可逆/系统级/极高成本, 强制拦截或最高级确认 |

### PermissionConfig 细粒度维度

| 配置字段 |
|---|
| auto_approve_all |
| plan_mode |
| path_rules |
| sandbox_mode |
| cost_budget_hours |
| trust_adaptive |

### 工具默认规则

| 工具 | 模式 |
|---|---|
| `abaqus_tool` | `ask` |
| `agentic_search_tool` | `auto` |
| `bash_tool` | `ask` |
| `code_tool` | `ask` |
| `comsol_tool` | `ask` |
| `cp2k_tool` | `ask` |
| `database_tool` | `auto` |
| `debugger_tool` | `auto` |
| `descriptor_tool` | `auto` |
| `design_atom_tool` | `auto` |
| `design_plan_tool` | `auto` |
| `diff_tool` | `auto` |
| `doe_tool` | `auto` |
| `elmer_tool` | `ask` |
| `eval_tool` | `auto` |
| `experimental_data_tool` | `auto` |
| `extract_tool` | `auto` |
| `fem_tool` | `auto` |
| `fenics_tool` | `ask` |
| `file_delete_tool` | `deny` |
| `file_edit_tool` | `ask` |
| `file_read_tool` | `auto` |
| `file_write_tool` | `ask` |
| `gap_analysis_tool` | `auto` |
| `generative_design_tool` | `auto` |
| `git_tool` | `auto` |
| `github_tool` | `ask` |
| `glob` | `auto` |
| `grep` | `auto` |
| `gromacs_tool` | `ask` |
| `image_analysis_tool` | `auto` |
| `image_design_tool` | `auto` |
| `job_tool` | `ask` |
| `lammps_tool` | `ask` |
| `materials_database_tool` | `auto` |
| `nudge_tool` | `auto` |
| `onboarding_tool` | `auto` |
| `openfoam_tool` | `ask` |
| `packing_tool` | `ask` |
| `phase_tool` | `auto` |
| `qe_tool` | `ask` |
| `specialty_analysis_tool` | `auto` |
| `structural_analytical_tool` | `auto` |
| `structure_tool` | `auto` |
| `system_shell_tool` | `deny` |
| `validate_tool` | `auto` |
| `vasp_tool` | `ask` |
| `visualize_tool` | `auto` |
| `web_search_tool` | `auto` |

危险命令模式 (27 条):

  - `rm\s+-rf\s+/`
  - `rm\s+-rf\s+~`
  - `rm\s+-rf\s+\*`
  - `rm\s+-fr\s+/`
  - `mkfs\.\w+\s+/dev/`
  - `dd\s+if=.*of=/dev/`
  - `shutdown\b`
  - `reboot\b`
  - `\bsudo\b`
  - `:\(\)\s*\{\s*:\|:\&\s*\}\s*;`
  - `kill\s+-9\s+1\b`
  - `killall\b`
  - `>\s*/dev/sda`
  - `chmod\s+-R\s+777\s+/`
  - `chown\s+-R\s+.*\s+/`
  - `\bformat\s+[a-z]:`
  - `\bdel\s+/[fsq]\b`
  - `\brmdir\s+/s\b`
  - `\brd\s+/s\b`
  - `powershell\s+-enc\b`
  - `\bnc\s+-[elp]`
  - `netcat\b`
  - `crontab\s+-[er]`
  - `git\s+push\s+.*--force`
  - `git\s+push\s+.*-f\b`
  - `git\s+reset\s+--hard`
  - `git\s+clean\s+-fd`

沙箱硬底线路径 (只能收紧不能放宽):

| 路径 | 模式 |
|---|---|
| `INSTRUCTIONS.md` | `deny` |
| `score.py` | `deny` |
| `evaluation/*.py` | `deny` |
| `rubric.json` | `deny` |
| `.huginn/checkpoints*` | `deny` |
| `.huginn/engine_state*.json` | `deny` |

运行时配置: `PermissionConfig` 字段由前端设置面板 / `HUGINN_PERM_*` 环境变量注入; `path_rules` 支持 \(tool, glob, mode\) 工具×路径矩阵。
