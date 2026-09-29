#!/usr/bin/env bash
# run50 launcher — 长程自主探索 (long-horizon), 书生自主推进, 人只监控.
#
# 相对 run49 的两处源头修复 (run47-49 实测空转: 每轮 nobj 恒定、换名归约闭环、
# prompt 逐字节冻结):
#   源头A · 输入冻结 (已修): goal 恒定时作者提示逐字节相同 → 同一问题反复问、
#      执行输出恒同、零新证据。现 code_lab.build_author_prompt 增 ``focus`` 参数,
#      engine_act._build_codelab_focus 把**随迭代演进**的当前假设 + 本轮实验步骤
#      注入提示 → prompt 随轮次解冻 (run47 实测 prompt_len 恒 5379)。
#   源头B · 脚手架托管管线 (已修): network_rigidity.py 原把"训练/扫描/判据/N_c
#      汇总"下沉为 capacity_scan 原语, 科学决策被架空 (每轮返回同 36 个 objectives)。
#      现**彻底解绑** —— 只注入基础数值工具 (mlp_fit/mlp_predict/poly_basis/_as_2d),
#      扫描/判据/聚合全部由书生自己实现。
#
# 保留: v11 进展不变量 (单调换名债务 _rename_debt → 无进展轮必被逼到终止出口)、
#   执行收敛指纹硬终止、提交闸 (harness 未过测试不进提交)。RSI bootstrap 继承
#   run49 产物。
set -u
export HUGINN_PROVIDER=internlm
export HUGINN_MODEL=intern-s2-preview
export INTERNLM_API_KEY=sk-pGqiBxoW3z84Pk80jF3bZB1UK4awwOVVpHtE68hwrQ0hQlba
export HUGINN_COGNITIVE_LLM_DECIDER=0
export HUGINN_CODELAB_TIMEOUT_S=900
export HUGINN_EXEC_ROUTE_DEBUG=1
export HUGINN_PERSISTENT_GOAL_MODE=1
export PYTHONPATH=/workspace/agent
export HUGINN_CODELAB_SCAFFOLD=/workspace/examples/codelab_scaffolds/network_rigidity.py

cd /workspace/research_outputs/shusheng_rsi_run50
exec python3 -m huginn.cli.main autoloop "$(cat objective.txt)" \
  -i 30 --wall-clock-budget 3600 > run.log 2>&1