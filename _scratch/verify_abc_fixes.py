"""验证 A(孤儿泄漏) / B(code_lab 证据链) / C(报告域漂移) 三处根治修复."""
import asyncio
import os
import sys
import time

sys.path.insert(0, "/workspace/agent")

import huginn.utils.process as proc_util


def check_A_persistent_terminal():
    from huginn.tools.persistent_terminal import PersistentTerminal

    term = PersistentTerminal(timeout_seconds=60)
    sid = term.start([sys.executable, "-c", "import time; time.sleep(60)"])
    pid = term._sessions[sid].handle.proc.pid
    assert pid in proc_util._LIVE_CHILDREN, "session child 未登记进在册表"
    print(f"[A] session child pid={pid} 已登记在册 OK")
    term.kill(sid)
    time.sleep(0.3)
    assert pid not in proc_util._LIVE_CHILDREN, "kill 后未注销登记"
    print("[A] kill 后已注销登记 OK")


def check_A_compute_adapter_untrack():
    from huginn.security.compute_adapter import JobSpec, run_job

    before = dict(proc_util._LIVE_CHILDREN)
    run_job(JobSpec(command=(sys.executable, "-c", "print('hi')"), timeout=10))
    after = dict(proc_util._LIVE_CHILDREN)
    assert after == before, f"run_job 未注销登记: before={before} after={after}"
    print("[A] compute_adapter.run_job 正常结束已注销登记 OK")


def check_A_reap_process_group():
    """模拟父进程退出兜底: 登记一个 sleep 子进程 → kill_tracked_children 应整组回收."""
    import subprocess

    kwargs, own_group = proc_util.new_group_popen_kwargs()
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], **kwargs)
    proc_util.track_live_child(p.pid, own_group=own_group)
    n = proc_util.kill_tracked_children()
    assert n >= 1, "kill_tracked_children 未回收在册进程"
    p.wait(timeout=5)
    assert p.returncode != 0, "进程未被杀死"
    print(f"[A] kill_tracked_children 整组回收 {n} 个 OK")


def check_B_error_line():
    from huginn.research.code_lab import _format_exec_error
    from huginn.security.code_act_sandbox import (
        exec_with_mem_cap,
        make_safe_builtins,
    )
    import numpy as np

    code = (
        "import numpy as np\n"
        "def run(cfg):\n"
        "    arr = np.array([1, 2, 3])\n"
        "    bad = int(arr)\n"  # 第 4 行: int() 收到数组
        "    return {'objectives': {'x': bad}, 'summary': {}}\n"
    )
    ns = {"__builtins__": make_safe_builtins(), "np": np}
    try:
        exec_with_mem_cap(code, ns, 0)
        ns["run"]({})
        raise AssertionError("应抛异常")
    except Exception as e:
        msg = _format_exec_error(e, code)
    assert "出错位置: 代码第 4 行" in msg, f"未给出出错行: {msg!r}"
    assert "bad = int(arr)" in msg, f"未附源码行: {msg!r}"
    print("[B] 异常提示带出错行号+源码行 OK")
    print("    " + msg.replace("\n", "\n    "))


def check_B_sandbox_run_e2e():
    from huginn.research.code_lab import sandbox_run

    code = (
        "import numpy as np\n"
        "def run(cfg):\n"
        "    a = [1, 2, 3]\n"
        "    s = int(a)\n"
        "    return {'objectives': {'s': s}, 'summary': {}}\n"
    )
    res, reason = sandbox_run(code, {"seed": 0}, timeout=30)
    assert res is None, "失败代码不应产出结果"
    assert reason and "出错位置: 代码第 4 行" in reason, f"隔离路径未带行号: {reason!r}"
    print("[B] sandbox_run 隔离路径带回出错行 OK")


def check_C_domain_classifier():
    from huginn.autoloop.engine_act import EngineAct

    eng = EngineAct(None)
    ml = "解空间刚性: 小型前馈网络的泛化行为作为探针, 扫描 N_c(w)"
    assert eng._classify_workflow_domain(ml) == "", "纯 ML 命题不应被认成任何计算域"
    print("[C] ML 命题 → 域分类 ''(不再默认 dft) OK")
    assert eng._classify_workflow_domain("run a DFT band structure") == "dft"
    assert eng._classify_workflow_domain("CFD turbulent flow") == "cfd"
    print("[C] 显式 DFT/CFD 关键词仍正确识别 OK")


def check_C_workflow_refuses():
    from huginn.autoloop.engine_act import EngineAct

    eng = EngineAct(None)
    res = asyncio.run(eng._execute_workflow("关于解空间刚性的数值实验", {}))
    assert res["success"] is False and res["domain"] is None, res
    assert "拒绝默认 DFT" in res["error"], res
    print("[C] workflow 对未知域拒绝 DFT 兜底 OK")


if __name__ == "__main__":
    check_A_persistent_terminal()
    check_A_compute_adapter_untrack()
    check_A_reap_process_group()
    check_B_error_line()
    check_B_sandbox_run_e2e()
    check_C_domain_classifier()
    check_C_workflow_refuses()
    print("\nALL ABC CHECKS PASSED")