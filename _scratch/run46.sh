#!/usr/bin/env bash
# run46 launcher — 长程自主探索 (long-horizon), 书生自主推进, 人只监控.
# 相对 run45 的修复 (misalign 永不纠正 + plan_check 自污染):
#   ① plan_check 失败不再短路放行: 原 conf<0.3 直接 `return plan`, 把 checker
#      判"不达标"的 plan 原样接受 → run45 45/45 全失败、零次 refine.
#      现: 低置信只降级记 warning, 仍必须 refine 到 attempt 用尽.
#   ② 历史失败按假设指纹 (_hyp_key) 隔离 + reason 去重: 原按 scene_tag 抽, 而
#      scene="other" 是万能桶 (容量扫描/拓扑/上同调全落这里) → checker 把自己
#      上一轮对另一个假设的判定当"已知坑"回灌, 锁死结论.
#   ③ 计划生成加"对齐闸门": 数值扫描只能测数值命题, 非数值(拓扑/代数/存在性)
#      须换符号方法或把假设改写为可测形式; 假设变了不许复用上一轮 description.
#   ④ 身份锚定 = 数学 (prompts.py "Mathematics Is the Base Unit" + 执行/验证链
#      code_act_loop / step_verifier / context_builder / lean conjecture 全部去域绑定).
# 继承 RSI 产物: 复用 run45 的 evolved_skills.json 作为 bootstrap.
set -u
export HUGINN_PROVIDER=internlm
export HUGINN_MODEL=intern-s2-preview
export INTERNLM_API_KEY=sk-pGqiBxoW3z84Pk80jF3bZB1UK4awwOVVpHtE68hwrQ0hQlba
export HUGINN_COGNITIVE_LLM_DECIDER=0
export HUGINN_CODELAB_TIMEOUT_S=900
export HUGINN_EXEC_ROUTE_DEBUG=1
export PYTHONPATH=/workspace/agent
export HUGINN_CODELAB_SCAFFOLD=/workspace/examples/codelab_scaffolds/network_rigidity.py

cd /workspace/research_outputs/shusheng_rsi_run46
exec python3 -m huginn.cli.main autoloop "$(cat objective.txt)" \
  -i 30 --wall-clock-budget 3600