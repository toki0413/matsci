#!/usr/bin/env bash
# run13 launcher — Huginn-native autoloop for the NN-generalization rigidity probe.
# Light guidance only: objective.txt carries the research question + acceptance
# criteria; 书生 (InternLM) decides and executes. Rule-based sequencer (LLM
# decider off) keeps phase order hypothesize→plan→execute→validate→learn.
set -u
export HUGINN_PROVIDER=internlm
export HUGINN_MODEL=intern-s2-preview
export INTERNLM_API_KEY=sk-pGqiBxoW3z84Pk80jF3bZB1UK4awwOVVpHtE68hwrQ0hQlba
export HUGINN_COGNITIVE_LLM_DECIDER=0
export HUGINN_CODELAB_TIMEOUT_S=600
export PYTHONPATH=/workspace/agent

cd /workspace/research_outputs/shusheng_rsi_run13
exec python3 -m huginn.cli.main autoloop "$(cat objective.txt)" -i 24 -s tests_passed