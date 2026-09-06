"""Harness evolution layer (H1-H4).

独立 store 子系统, 不走 EvolutionEngine (避免 P5 两套系统耦合).
- prompt_patch: H1 prompt template patch 闭环
- phase_spec: H4 phase + BUILTIN_SPECS 可演化

toggle 都走 cfg.feature_flags.<key>, 默认 off, mtime auto reload.
"""
