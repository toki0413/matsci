"""rcb 包: rcb_runner 拆分出的子模块集合.

prompt_builders: 旧 hint 拼接 prompt builder (iter 0 / iter>0).
self_checks:     各 task 代码层 self-check 验收函数.

rcb_runner.py 仍为入口 (env setup + 主循环 run/_step2_execute/_step3_adversarial
+ main + __main__ 派发). 强耦合的核心及 module-level env 副作用 (必须在 import huginn
之前执行) 保留在 rcb_runner, 避免破坏 import 时的副作用顺序与 __file__ 相对路径.

向后兼容: `from huginn.cli.rcb_runner import X` 仍可用 (rcb_runner re-export 本包).
本包两个子模块顶层均不依赖 rcb_runner, 可被任意顺序 import, 无循环依赖.
"""
from huginn.cli.rcb.prompt_builders import (
    _legacy_build_step2_prompt,
    _legacy_build_iter_prompt,
)
from huginn.cli.rcb.self_checks import (
    self_check_v14_task4,
    self_check_v14_task6,
    self_check_a3,
    self_check_a2,
    self_check_a4,
    self_check_v14_task1,
    self_check_v14_task2,
    self_check_v14_task3,
    self_check_v14_task8,
    self_check_v15_task3,
    self_check_v15_task4,
    self_check_v14_comprehensive,
    self_check_v14_p234,
    self_check_v14_all,
)

__all__ = [
    "_legacy_build_step2_prompt",
    "_legacy_build_iter_prompt",
    "self_check_v14_task4",
    "self_check_v14_task6",
    "self_check_a3",
    "self_check_a2",
    "self_check_a4",
    "self_check_v14_task1",
    "self_check_v14_task2",
    "self_check_v14_task3",
    "self_check_v14_task8",
    "self_check_v15_task3",
    "self_check_v15_task4",
    "self_check_v14_comprehensive",
    "self_check_v14_p234",
    "self_check_v14_all",
]
