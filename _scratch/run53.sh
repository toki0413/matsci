#!/usr/bin/env bash
# run53 launcher — 长程确认 (long-horizon), 书生自主推进, 人只监控.
#
# 背景: run52 已跑完但**失败**于两个 harness bug (非书生问题):
#   ① 迭代档位 (ProgressiveBudget) 在 step 31-50 只放 coder, 而书生的 plan
#      mode 是 explore → _check_budget 连续拒绝 → execute 被静默跳过;
#   ② execute 一被跳过, 循环就拿**同一个旧 execution_result** 反复 validate,
#      reflect 每轮重算同一指纹压进 6 窗口 → 假 "exec convergence" → 904s/3600s
#      就 conclude+stop (39%→实际 25% 预算), pivot 硬约束根本没机会生效
#      (它只注入 code_lab 作者提示, 而 execute 没跑).
#
# 本轮修复 (3 处, 见 commit):
#   1) cognitive_loop execute_fn: 长程模式 (_long_horizon_keep_going) 下 budget
#      让位给挂钟 — 不再禁 explore;
#   2) engine_reflect: 仅当 execute **真产出新对象**时才做 重复/收敛 记账
#      (_fp_result_ref 身份判据), 复用旧结果的 validate 不再制造假收敛;
#   3) execute 被 budget/gate 跳过时打 WARNING 日志 (可审计).
#
# 观测口径 (与 run50/52 对齐):
#   ① run 是否跑到挂钟/目标, 而非 ~900s 假收敛早停;
#   ② repeat streak 越阈值后, 下一轮实验族/参数是否真变 (指纹改变);
#   ③ 换名债务是否累积到触发重定向; ④ 是否出现"判别实验"(换 family/扫描).
#
# 继承 (RSI bootstrap): .huginn/evolved_skills.json 自 run52 拷贝 (与
#   run49→run50/51/52 同一约定: 只带演化技能前行, 不带旧循环状态/goals).
set -u
export HUGINN_PROVIDER=internlm
export HUGINN_MODEL=intern-s2-preview
export INTERNLM_API_KEY=sk-pGqiBxoW3z84Pk80jF3bZB1UK4awwOVVpHtE68hwrQ0hQlba
export HUGINN_COGNITIVE_LLM_DECIDER=0
export HUGINN_CODELAB_TIMEOUT_S=900
export HUGINN_EXEC_ROUTE_DEBUG=1
export HUGINN_PERSISTENT_GOAL_MODE=1
export HUGINN_REPEAT_EXEC_HARD_STREAK=2
export PYTHONPATH=/workspace/agent
export HUGINN_CODELAB_SCAFFOLD=/workspace/examples/codelab_scaffolds/network_rigidity.py

cd /workspace/research_outputs/shusheng_rsi_run53
exec /workspace/.venv/bin/python -m huginn.cli.main autoloop "$(cat objective.txt)" \
  -i 60 --wall-clock-budget 3600 > run.log 2>&1