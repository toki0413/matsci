#!/usr/bin/env bash
# run47 launcher — 长程自主探索 (long-horizon), 书生自主推进, 人只监控.
# 相对 run46 的修复 (计划生成/校验逻辑, 真正的 misalign 根因):
#   ① plan_check 的 max_refines 会被自适应 ewma 放宽压到 0 (bucket 多数通过
#      时 baseline-1), 于是 `for attempt in range(0+1)` 只校验一次, 首次失败
#      即 `return plan` → 明知达不成 hypothesis 的 plan 被原样执行, 零次 rebuild
#      (run46 实测: refine=0, repeat streak 反复). 现改为 while 循环, 失败且预算
#      为 0 时强制给一次重建预算 (通过路径不受影响).
#   ② 失败但 conf<0.3 不再直接放行: 低置信只降级记 warning, 仍走 refine.
#   ③ 历史失败按假设指纹 (_hyp_key) 隔离 + reason 去重: 断掉 scene="other"
#      万能桶导致的 checker 自我污染复读.
#   ④ 计划生成加"对齐闸门": 数值扫描只测数值命题, 非数值须换符号方法或把假设
#      改写为可测形式; 假设变了不许复用上一轮 description.
#   ⑤ 身份锚定 = 数学 (prompts.py "Mathematics Is the Base Unit" + 执行/验证链去域绑定).
# 继承 RSI 产物: 复用 run46 的 evolved_skills.json 作为 bootstrap.
set -u
export HUGINN_PROVIDER=internlm
export HUGINN_MODEL=intern-s2-preview
export INTERNLM_API_KEY=sk-pGqiBxoW3z84Pk80jF3bZB1UK4awwOVVpHtE68hwrQ0hQlba
export HUGINN_COGNITIVE_LLM_DECIDER=0
export HUGINN_CODELAB_TIMEOUT_S=900
export HUGINN_EXEC_ROUTE_DEBUG=1
export PYTHONPATH=/workspace/agent
export HUGINN_CODELAB_SCAFFOLD=/workspace/examples/codelab_scaffolds/network_rigidity.py

cd /workspace/research_outputs/shusheng_rsi_run47
exec python3 -m huginn.cli.main autoloop "$(cat objective.txt)" \
  -i 30 --wall-clock-budget 3600