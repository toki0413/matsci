#!/usr/bin/env bash
# run49 launcher — 长程自主探索 (long-horizon), 书生自主推进, 人只监控.
#
# 相对 run48 的两处根因修复 (run48 实测仍在空转: 每轮 nobj=36、换名归约、
# repeat-execution streak):
#   ① 命题被脚手架提前解掉: network_rigidity.py 的 TEMPLATE 原本内嵌一份
#      **完整可跑的示例族** (rigid=sin(pi x), fat=20 项随机多项式) —— 书生照抄
#      模板即得正确答案, 每轮 capacity_scan 返回同 36 个 objectives, 科学决策
#      被架空。现 TEMPLATE 只留骨架/契约 + 一个标为"必须替换"的退化占位族;
#      并加"退化族守卫"(标签近常数即抬错回灌), 逼书生自己推导 rigid/fat 族.
#   ② 循环无收敛出口: 新增 v10-F5 —— engine_reflect 维护最近 6 个执行结果指纹,
#      窗口填满且去重后 <=2 种 → 判"已无新信息", cognitive_loop 经 completion
#      audit 不阻断后结题停止 (区别于 F4 surprise 启发式收敛, 执行指纹是硬证据).
#
# 保留: 平台内核修复 (plan_check while+强制重建、数学域锚定、重复执行指纹、
#   换名归约两级升级、实质内容守卫). RSI bootstrap 继承 run48 产物.
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

cd /workspace/research_outputs/shusheng_rsi_run49
exec python3 -m huginn.cli.main autoloop "$(cat objective.txt)" \
  -i 30 --wall-clock-budget 3600 > run.log 2>&1