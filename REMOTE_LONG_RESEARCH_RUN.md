# 强机（8vCPU / 32GB）长期自主科研闭环 —— 启动说明

> 目标机规格匹配：8 核 CPU / 32GB 内存（与弱机沙箱的 3 核 / 5.8GB 不同，足够跑长期闭环）
> 推理通道：**DeepSeek API**（弱机塞不下本地大模型，走云端推理即可）
> 第一个闭环目标：**方法级机理**（维恩位移 / 斯涅耳折射 / 托里拆利出口速度）

---

## 1. 你需要准备（3 件事）

1. **一台 8核/32GB 的 Ubuntu 机器**（ModelScope 实例 / 云主机均可），装好 `python3.11`。
2. **DeepSeek API key**（你已有的 deepseek key）。
3. **一台能访问这台机器的终端**（SSH 或网页终端）。

> ⚠️ 敏感凭据纪律：key **只用 export 注入环境**，不写进任何文件/脚本/日志，不进 git。

---

## 2. 文件清单（本目录已随仓库交付）

| 文件 | 用途 |
|---|---|
| `deploy_and_run_long_research.sh` | 一键：clone→venv→依赖→持久目录→冒烟→跑闭环 |
| `run_cognitive_driver.py` | 闭环驱动入口（已改为读 `WORKSPACE_DIR` 持久目录，不再写 /tmp） |

这两份在**弱机已完成语法/装配校验**（`bash -n` 通过、AutoloopEngine 装配链走到"仅差 API key"这一步），可直接在强机用。

---

## 3. 一键启动（强机上）

```bash
# 1. 设 key（只进环境）
export DEEPSEEK_API_KEY="你的deepseek_key"

# 2. 跑一键脚本（默认目标=方法级机理闭环，12 轮迭代）
bash deploy_and_run_long_research.sh
```

脚本会依次做：
```
[0/5] 校验环境（python3.11）
[1/5] clone/更新仓库 → /opt/matsci
[2/5] 建 .venv + 装依赖（pyproject [dev] + 兜底核心）
[3/5] 建持久目录 /data/huginn_runtime/{memory,workspace}
[4/5] 冒烟：AutoloopEngine 能实例化
[5/5] 跑长闭环：run_cognitive_driver "<objective>" 12
```

---

## 4. 可选：换目标 / 加迭代 / 换 model

```bash
# 换目标（例如只想跑某一个机理）
export DEFAULT_OBJECTIVE="Verify Snell's law: n1=1.0, incidence 60deg, n2=1.5 -> predict refraction angle and give PASS/FAIL vs arcsin(1/1.5*sin60)."
export DEFAULT_ITER=20
bash deploy_and_run_long_research.sh

# 换模型名（切换 deepseek 哪个模型）
HUGINN_MODEL=deepseek-reasoner  bash deploy_and_run_long_research.sh
```

---

## 5. 结果落盘位置

- 闭环摘要：`/data/huginn_runtime/workspace/run_result.json`
- 记忆库：`/data/huginn_runtime/memory/memory.db`
- 引擎 workspace：`/data/huginn_runtime/workspace/`

> 选 `/data` 而非 `/tmp` 的原因：弱机实测 `/tmp` 会被清空，`/data` 是持久盘，长期闭环的记忆/checkpoint 才不丢。

---

## 6. 诚实边界（重要）

- **弱机已验证**：代码/依赖/装配链就绪、AutoloopEngine 实例化链走到"仅缺 DEEPSEEK_API_KEY"。这是"能上强机"的证明。
- **弱机未跑**：真实 LLM 闭环（没 key 没法真跑）。所以"方法级机理闭环实际跑出 PASS/FAIL"这条结果，**要强机才能产出**——这是你要在 8核/32GB 上做的第一件事，跑完把 `run_result.json` 里 `goal_achieved` 和 `phase_names` 给我，我帮你解读。
- 方法级 sympy/scipy 负载低，8核32GB 完全够；**若未来目标开始跑 VASP/LAMMPS 仿真，资源需求另算**，别用这份说明的基准。

---

## 7. 下一步（跑完看这些）

跑完 `run_result.json` 有三个关键字段：
- `success` / `goal_achieved`：闭环是否达成
- `phase_names`：实际走过的 phase 序列（perceive→hypothesize→plan→execute→validate→learn→report）
- `tests_passed` / `constraints_satisfied`：数值结论与约束校验

把这三项贴回来，我帮你判断"方法级机理闭环是否真的自主走通"，并据此决定要不要放大迭代跑更复杂的断裂力学开放问题。