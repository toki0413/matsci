"""P4-2 自检: StandingRulesStore + extract_target_from_args.

最小可运行检查 (ponytail):
1. grant + is_granted 基本流程
2. fnmatch target pattern 匹配
3. reset 清空
4. extract_target_from_args 从常见字段提取
"""
from __future__ import annotations


def _test_standing_rules_basic():
    """grant + is_granted + reset."""
    from huginn.permissions import (
        get_standing_rules_store,
        reset_standing_rules_store,
    )
    reset_standing_rules_store()
    store = get_standing_rules_store()

    # 初始无 rule
    assert not store.is_granted("s1", "file_write_tool", "/tmp/foo.txt")

    # grant 后命中
    store.grant("s1", "file_write_tool", "/tmp/*")
    assert store.is_granted("s1", "file_write_tool", "/tmp/foo.txt")
    assert store.is_granted("s1", "file_write_tool", "/tmp/bar/baz.py")
    assert not store.is_granted("s1", "file_write_tool", "/etc/passwd")
    assert not store.is_granted("s1", "bash_tool", "/tmp/foo.txt")

    # "*" 通配
    store.grant("s1", "bash_tool", "*")
    assert store.is_granted("s1", "bash_tool", "anything")
    assert store.is_granted("s1", "bash_tool", "*")

    # session 隔离: s2 没 grant, 不应命中
    assert not store.is_granted("s2", "file_write_tool", "/tmp/foo.txt")

    # 给 s2 也 grant 一条, 验证 reset("s1") 不影响 s2
    store.grant("s2", "bash_tool", "*")
    assert store.is_granted("s2", "bash_tool", "anything")

    # reset 单 session
    store.reset("s1")
    assert not store.is_granted("s1", "file_write_tool", "/tmp/foo.txt")
    assert store.is_granted("s2", "bash_tool", "anything")  # s2 不受影响

    # reset 全部
    store.reset()
    assert not store.is_granted("s2", "bash_tool", "anything")

    reset_standing_rules_store()
    print("1. StandingRulesStore grant/is_granted/reset OK")


def _test_standing_rules_list():
    """list_rules 可观测性."""
    from huginn.permissions import (
        get_standing_rules_store,
        reset_standing_rules_store,
    )
    reset_standing_rules_store()
    store = get_standing_rules_store()
    store.grant("s1", "file_write_tool", "/tmp/*")
    store.grant("s1", "bash_tool", "*")
    store.grant("s2", "git_tool", "*")

    all_rules = store.list_rules()
    assert len(all_rules) == 3

    s1_rules = store.list_rules("s1")
    assert len(s1_rules) == 2
    tools = {r["tool"] for r in s1_rules}
    assert tools == {"file_write_tool", "bash_tool"}

    store.reset()
    reset_standing_rules_store()
    print("2. list_rules OK")


def _test_extract_target():
    """extract_target_from_args 从常见字段提取."""
    from huginn.permissions import extract_target_from_args

    assert extract_target_from_args({"file_path": "/tmp/foo.txt"}) == "/tmp/foo.txt"
    assert extract_target_from_args({"path": "/data/output"}) == "/data/output"
    assert extract_target_from_args({"working_dir": "/home/user"}) == "/home/user"
    assert extract_target_from_args({"output_path": "/out/result.json"}) == "/out/result.json"
    assert extract_target_from_args({"target": "some_target"}) == "some_target"

    # 多字段时 file_path 优先
    assert extract_target_from_args({
        "file_path": "/tmp/foo.txt", "path": "/data/output"
    }) == "/tmp/foo.txt"

    # 无字段 → "*"
    assert extract_target_from_args({}) == "*"
    assert extract_target_from_args(None) == "*"

    # 非字符串字段跳过
    assert extract_target_from_args({"file_path": 123}) == "*"

    print("3. extract_target_from_args OK")


def _test_standing_rules_singleton():
    """单例模式: get_standing_rules_store 返回同一实例."""
    from huginn.permissions import (
        get_standing_rules_store,
        reset_standing_rules_store,
    )
    reset_standing_rules_store()
    s1 = get_standing_rules_store()
    s2 = get_standing_rules_store()
    assert s1 is s2
    reset_standing_rules_store()
    print("4. singleton OK")


def _main():
    _test_standing_rules_basic()
    _test_standing_rules_list()
    _test_extract_target()
    _test_standing_rules_singleton()
    print("\nAll P4-2 self-checks passed.")


if __name__ == "__main__":
    _main()
