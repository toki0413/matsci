#!/usr/bin/env bash
# run54 launcher — 验证 v13 两处判据修正 (长程, 书生自主推进, 人只监控).
#
# 背景 (run53 已满预算跑完, 机制侧通过): 满 3606s、零假收敛、零 execute 跳过、
#   零旧结果复用、55 次真实 code_lab 实验. 但暴露两处判据问题:
#   ① repeat 指纹口径过窄: 对 objectives/summary **内容**做哈希 → 要求两轮数值逐字节
#      相同才判"重复". 浮点抖动/换注释重写都会漏判 → 硬约束整轮未点亮
#      (repeat execution detected = 0, 真正逼出换族的是换名债务升级链).
#   ② 命题口径"所有报告数值必须是有限数(禁止 inf/nan/None)"**执行侧零强制**:
#      code_lab schema 只校验"值是数值", float('inf') 照收; validate 又据 objectives
#      非空判 solved → 书生用 ∞ 当"未达标"哨兵, ∞ 一路进最终报告 (run53 实测).
#
# 本轮修复 (2 处):
#   1) engine_reflect._exec_fingerprint: 有 script 时改用**代码结构指纹**
#      (AST, 忽略注释/格式, **保留数字常量**) — 同代码重跑恒同, 与浮点无关;
#      参数扫描改常量→指纹变, 不误判. 无 script 时退回旧内容哈希.
#   2) engine_reflect._is_code_lab_solved / _non_finite_objective_keys: objectives
#      含 inf/nan 时**不算真实执行证据**, 并注入明确 hint (否则静默失败).
#
# 观测口径:
#   ① 是否跑到挂钟 (3600s) 而非提前结题;
#   ② `repeat execution detected` / `force experiment variation` 是否被点亮;
#   ③ `非有限 objectives 不计为证据` 是否出现, 最终报告是否仍有 inf/nan/∞;
#   ④ 换名债务/判别实验轨迹.
#
# 继承 (RSI bootstrap): .huginn/evolved_skills.json 自 run53 拷贝 (只带演化技能
#   前行, 不带旧循环状态/goals). 目标沿用 run53 objective (同一命题, 便于对照).
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

cd /workspace/research_outputs/shusheng_rsi_run54
exec /workspace/.venv/bin/python -m huginn.cli.main autoloop "$(cat objective.txt)" \
  -i 60 --wall-clock-budget 3600 > run.log 2>&1