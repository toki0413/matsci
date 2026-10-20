#!/usr/bin/env bash
# run48 launcher — 长程自主探索 (long-horizon), 书生自主推进, 人只监控.
#
# 相对 run47 的唯一改动 = objective 的"规格回收":
#   run47 把答案本身喂给了书生 (判别口径 + 零违规判据 + 尺度纪律 + 第0步 +
#   实验设计 + 脚本契约 + 伪结果诊断 全部写死) → 假设层无空间可生成, 90 次
#   执行全 nobj=36、13 次重复指纹、9+8 次换名归约升级, 结构编码全零、
#   surprise 恒 1.0, 全程打转 (见 run47/run.log). 现 objective 只留"方向"
#   (命题 + 二值判别口径 + 真实性纪律 + 反换名边界), 族/样本/扫描/脚本
#   的设计交还书生自主决定.
#
# 保留: 平台内核修复 (plan_check while+强制重建、数学域锚定、重复执行指纹、
#   换名归约两级升级、实质内容守卫) 不变. RSI bootstrap 继承 run47 产物.
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

cd /workspace/research_outputs/shusheng_rsi_run48
exec python3 -m huginn.cli.main autoloop "$(cat objective.txt)" \
  -i 30 --wall-clock-budget 3600 > run.log 2>&1