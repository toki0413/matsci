#!/usr/bin/env bash
# run45 launcher — 长程自主探索 (long-horizon), 书生自主推进.
# 相对 run44 的根因修复 (身份锚定 = 数学, 而非"以数学为语言的自然科学"):
#   plan_check 走 _llm_chat(persona_name="default"), 即 HUGINN_SYSTEM_PROMPT.
#   原 section 1 写 "You are a natural-science research agent whose native
#   language is mathematics" + 域清单只列 physics/chemistry/materials —— 于是
#   run44 里 plan_check 每轮低置信拒绝, 理由恒定:
#     "violates the system prompt's core identity: I am a natural-science
#      research agent ..., not a general-purpose mathematical or ML researcher"
#   纯数学/ML 命题被判为"越界" → plan 永远立不住, 循环空转.
#   修复: section 1 身份改为 "Mathematics Is the Base Unit", 明确纯数学与
#   ML 理论是一等研究对象, 无需归约到物理体系; section 2 表格列去掉
#   "Natural-science instances" 限定, 并新增 function-space/optimization/
#   information 三行数学内部结构.
#   run44 同链路(执行/验证 prompt)也已数学化: code_act_loop / step_verifier /
#   context_builder / lean.conjecture_library.
# 继承 RSI 产物: 复用 run44 的 evolved_skills.json 作为 bootstrap.
#   故意不继承 plan_check_patterns.json —— 它记的是上一版身份下的"越界"失败,
#   对新身份是陈旧噪声, 会误导 checker.
set -u
export HUGINN_PROVIDER=internlm
export HUGINN_MODEL=intern-s2-preview
export INTERNLM_API_KEY=sk-pGqiBxoW3z84Pk80jF3bZB1UK4awwOVVpHtE68hwrQ0hQlba
export HUGINN_COGNITIVE_LLM_DECIDER=0
export HUGINN_CODELAB_TIMEOUT_S=900
export HUGINN_EXEC_ROUTE_DEBUG=1
export PYTHONPATH=/workspace/agent
export HUGINN_CODELAB_SCAFFOLD=/workspace/examples/codelab_scaffolds/network_rigidity.py

cd /workspace/research_outputs/shusheng_rsi_run45
exec python3 -m huginn.cli.main autoloop "$(cat objective.txt)" \
  -i 30 --wall-clock-budget 3600