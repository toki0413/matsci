"""KernelSession / KernelSessionManager 测试.

环境里没有 jupyter_client 时走 subprocess 降级路径, 测试覆盖:
- 生命周期 start -> execute -> stop
- 变量跨 execute 持久化
- 超时代码被终止
- matplotlib inline 图像捕获 (mock jupyter client, 不依赖真内核)
- 多会话管理
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from huginn.execution.kernel_session import (
    KernelExecResult,
    KernelSession,
    KernelSessionManager,
)


@pytest.fixture
def session():
    """每个测试一个独立 session, 用完即停."""
    s = KernelSession(timeout=10.0)
    s.start()
    yield s
    s.stop()


def test_session_lifecycle(session):
    """start -> alive; execute -> ok; stop -> not alive."""
    assert session.alive is True
    res = session.execute("print(1+1)")
    assert res.status == "ok"
    assert "2" in res.stdout
    session.stop()
    assert session.alive is False


def test_variable_persistence(session):
    """x=42 后, 下一次 execute 仍能读到 x."""
    r1 = session.execute("x = 42")
    assert r1.status == "ok"
    r2 = session.execute("print(x)")
    assert r2.status == "ok"
    assert "42" in r2.stdout


def test_variable_mutation_persistence(session):
    """修改变量后跨调用保留新值."""
    session.execute("acc = 0")
    session.execute("acc = acc + 10")
    session.execute("acc = acc + 5")
    r = session.execute("print(acc)")
    assert "15" in r.stdout


def test_timeout():
    """超时代码被终止, 返回 timeout 状态."""
    # subprocess 路径用 subprocess.run(timeout=...), 设一个很小的 timeout
    s = KernelSession(timeout=0.5)
    s.start()
    try:
        res = s.execute("import time; time.sleep(5)")
        assert res.status == "timeout"
        assert res.error is not None
    finally:
        s.stop()


def test_error_capture(session):
    """代码抛异常时返回 error 状态 + traceback, 不崩会话."""
    res = session.execute("raise ValueError('boom')")
    assert res.status == "error"
    assert res.error is not None
    assert "ValueError" in res.error or "boom" in res.error
    # 会话还活着, 能继续执行
    ok = session.execute("print('still alive')")
    assert ok.status == "ok"
    assert "still alive" in ok.stdout


def test_matplotlib_inline_mocked():
    """mock jupyter client, 验证 display_data 里的 image/png 被捕获.

    不依赖真实 ipykernel: 手动塞一个 fake client, 让 _jupyter_exec
    收到一条带 image/png 的 display_data 消息 + idle 收尾.
    """
    sess = KernelSession(timeout=5.0)
    sess._started = True
    sess._km = MagicMock()  # 让 execute 走 jupyter 路径
    fake_png = "iVBORw0KGgoAAAANSUhEUg=="

    class _FakeKC:
        def __init__(self):
            self._msgs = [
                {"parent_header": {"msg_id": "mid"}, "msg_type": "display_data",
                 "content": {"data": {"image/png": fake_png, "text/plain": "<img>"}}},
                {"parent_header": {"msg_id": "mid"}, "msg_type": "status",
                 "content": {"execution_state": "idle"}},
            ]
            self._i = 0

        def execute(self, code, silent=False):
            return "mid"

        def get_iopub_msg(self, timeout=None):
            msg = self._msgs[self._i]
            self._i += 1
            return msg

    sess._kc = _FakeKC()
    res = sess.execute("import matplotlib.pyplot as plt; plt.plot([1,2,3])")
    assert res.status == "ok"
    assert len(res.images) == 1
    assert res.images[0] == fake_png
    assert "<img>" in res.stdout


def test_get_state(session):
    """get_state 返回顶层非 dunder 变量."""
    session.execute("a = 1; b = 'hello'")
    state = session.get_state()
    assert "a" in state
    assert "b" in state
    assert "1" in state["a"]


def test_manager_create_get_close():
    """多会话管理: create -> get -> list -> close."""
    mgr = KernelSessionManager(idle_timeout=3600.0)
    try:
        s1 = mgr.create(timeout=10.0)
        s2 = mgr.create(timeout=10.0)
        assert s1.session_id != s2.session_id
        assert mgr.get(s1.session_id) is not None
        assert mgr.get("nope") is None
        sessions = mgr.list_sessions()
        assert len(sessions) == 2
        assert {x["session_id"] for x in sessions} == {s1.session_id, s2.session_id}
        # close 一个
        assert mgr.close(s1.session_id) is True
        assert mgr.close(s1.session_id) is False  # 已关
        assert len(mgr.list_sessions()) == 1
    finally:
        mgr.close_all()


def test_manager_cleanup_idle():
    """空闲超时的会话被 reaper 清理."""
    mgr = KernelSessionManager(idle_timeout=0.01, cleanup_interval=999.0)
    try:
        s = mgr.create(timeout=10.0)
        assert mgr.get(s.session_id) is not None
        time.sleep(0.05)
        n = mgr.cleanup_idle()
        assert n == 1
        assert mgr.get(s.session_id) is None
    finally:
        mgr.close_all()


def test_restart_clears_state(session):
    """restart 后变量清空."""
    session.execute("z = 99")
    assert "99" in session.execute("print(z)").stdout
    session.restart()
    # restart 后 z 应该没了 (NameError)
    res = session.execute("print(z)")
    assert res.status == "error"
