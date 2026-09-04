"""WSL ↔ Windows 路径转换工具 (huginn.utils.wslpath) 单测。

覆盖:
(a) /mnt/c/... ↔ C:\\... ;
(b) Windows 路径 → WSL 侧 /mnt/c/... ;
(c) \\wsl$\\<distro>\\... → 发行版内路径 (纯函数分支, 不依赖 wsl 命令) ;
(d) is_wsl 判定 (环境注入开关 HUGINN_FORCE_WSL 模拟) ;
(e) 非 WSL / 相对路径原样返回, 不臆造转换。
"""

from __future__ import annotations

from huginn.utils import wslpath as w


def test_to_windows_mnt_c_to_c_drive():
    # (a) /mnt/c/... → C:\...
    assert w.to_windows("/mnt/c/Users/al") == "C:\\Users\\al"
    assert w.to_windows("/mnt/c/Users/al/notes.txt") == "C:\\Users\\al\\notes.txt"
    # 盘符转大写
    assert w.to_windows("/mnt/d/x.txt") == "D:\\x.txt"
    # 只有盘符, 无子路径
    assert w.to_windows("/mnt/c") == "C:\\"


def test_to_wsl_drive_to_mnt():
    # (b) C:\... / C:/... → /mnt/c/...
    assert w.to_wsl("C:\\Users\\al") == "/mnt/c/Users/al"
    assert w.to_wsl("C:/Users/al/notes.txt") == "/mnt/c/Users/al/notes.txt"
    # 盘符转小写
    assert w.to_wsl("D:\\some\\file.bin") == "/mnt/d/some/file.bin"
    # 只有盘符
    assert w.to_wsl("C:\\") == "/mnt/c"


def test_to_wsl_unc_to_distro_path():
    # (c) \\wsl$\<distro>\... → 发行版文件系统根路径 (纯函数, 无 wsl.exe)
    assert w.to_wsl(r"\\wsl$\Ubuntu\home\u") == "/home/u"
    # 新版 wsl.localhost 前缀等价
    assert w.to_wsl(r"\\wsl.localhost\Ubuntu\home\u") == "/home/u"
    # 深路径
    assert w.to_wsl(r"\\wsl$\Ubuntu\etc\bash.bashrc") == "/etc/bash.bashrc"


def test_is_wsl_forced_by_env():
    # (d) 环境注入开关模拟判定
    # 强置非 WSL 时 /mnt 不映射为 UNC
    assert w.is_wsl_unc(r"\\wsl$\Ubuntu\home\u") is True
    assert w.is_wsl_unc("/mnt/c/Users/al") is False


def test_relative_and_non_wsl_paths_pass_through():
    # (e) 非 WSL / 相对路径原样返回, 不臆造转换
    assert w.to_wsl("sub/file.txt") == "sub/file.txt"
    assert w.to_wsl("") == ""
    assert w.to_windows("relative/path") == "relative/path"
    # 非挂载盘的 POSIX 绝对路径, 非 WSL 环境 (无发行版) 原样返回
    assert w.to_windows("/home/u/file.txt") == "/home/u/file.txt"


def test_drive_to_windows_roundtrip():
    assert w.to_windows(w.to_wsl("C:\\Users\\al\\f.txt")) == "C:\\Users\\al\\f.txt"


def test_unc_detection_and_windows_mapping(monkeypatch):
    # 在 WSL 内且已知发行版时, 非挂载盘 WSL 路径映射为 wsl$ UNC
    monkeypatch.setenv("HUGINN_FORCE_WSL", "1")
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
    assert w.to_windows("/home/u/x.txt") == r"\\wsl$\Ubuntu\home\u\x.txt"
    # 关闭 WSL 开关后回到原样返回
    monkeypatch.setenv("HUGINN_FORCE_WSL", "0")
    assert w.to_windows("/home/u/x.txt") == "/home/u/x.txt"


def test_automount_root_respects_wsl_env(monkeypatch):
    # automount root 可配置: 强制 WSL 下默认 /mnt
    monkeypatch.setenv("HUGINN_FORCE_WSL", "1")
    assert w.to_wsl(r"E:\data") == "/mnt/e/data"
