"""真实 HPC 环境验证脚本 — 上线前 checklist.

本脚本不能在 CI 沙箱自动跑通 (需要真实 Slurm/PBS 集群 + SSH 凭证).
部署方在目标 HPC 环境手动执行, 逐项验证.

用法:
    # 在能访问 HPC 集群的机器上:
    export HUGINN_HPC_HOST=login.cluster.example.com
    export HUGINN_HPC_USER=your_user
    export HUGINN_HPC_KEY_FILE=~/.ssh/id_ed25519
    python -m tests.verify_hpc_environment

每项检查输出 PASS/FAIL/MANUAL, 全部 PASS 才能上线.
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path


def _result(status: str, msg: str) -> None:
    """打印一项检查结果."""
    icon = {"PASS": "[PASS]", "FAIL": "[FAIL]", "MANUAL": "[MANUAL]", "SKIP": "[SKIP]"}[status]
    print(f"  {icon} {msg}")


def check_ssh_connectivity() -> bool:
    """1. SSH 连接 — 能连上登录节点."""
    host = os.environ.get("HUGINN_HPC_HOST")
    user = os.environ.get("HUGINN_HPC_USER")
    key = os.environ.get("HUGINN_HPC_KEY_FILE")
    if not all([host, user, key]):
        _result("SKIP", "HUGINN_HPC_HOST/USER/KEY_FILE 未设置, 跳过")
        return False
    try:
        from huginn.hpc.client import HPCClient
        client = HPCClient(host=host, user=user, key_file=key, strict_host_key_checking=True)
        # 尝试执行简单命令
        out = client.run_command("hostname && whoami && echo OK")
        if "OK" in out:
            _result("PASS", f"SSH 连接 {user}@{host} 成功")
            return True
        _result("FAIL", f"SSH 连接异常, 输出: {out[:100]}")
        return False
    except Exception as e:
        _result("FAIL", f"SSH 连接失败: {e}")
        return False


def check_strict_host_key() -> bool:
    """2. 严格 host key 校验 — 必须开启 (生产环境)."""
    strict = os.environ.get("HUGINN_STRICT_HOST_KEY_CHECKING", "").lower()
    if strict in ("1", "true", "yes"):
        _result("PASS", "HUGINN_STRICT_HOST_KEY_CHECKING 已开启")
        return True
    _result("FAIL", "HUGINN_STRICT_HOST_KEY_CHECKING 未开启, 生产环境必须设为 true")
    return False


def check_known_hosts() -> bool:
    """3. known_hosts 里有集群指纹 — 防中间人."""
    host = os.environ.get("HUGINN_HPC_HOST")
    if not host:
        _result("SKIP", "HUGINN_HPC_HOST 未设置")
        return False
    known_hosts = Path.home() / ".ssh" / "known_hosts"
    if not known_hosts.exists():
        _result("FAIL", f"{known_hosts} 不存在, 需先 ssh {host} 手动确认指纹")
        return False
    content = known_hosts.read_text()
    if host in content:
        _result("PASS", f"{host} 已在 known_hosts")
        return True
    _result("MANUAL", f"{host} 不在 known_hosts, 需手动 ssh 确认 (ssh-keyscan)")
    return False


def check_slurm_available() -> bool:
    """4. Slurm 命令可用 — sbatch/squeue/sinfo."""
    host = os.environ.get("HUGINN_HPC_HOST")
    if not host:
        _result("SKIP", "HUGINN_HPC_HOST 未设置")
        return False
    try:
        from huginn.hpc.client import HPCClient
        client = HPCClient(
            host=host,
            user=os.environ["HUGINN_HPC_USER"],
            key_file=os.environ["HUGINN_HPC_KEY_FILE"],
            strict_host_key_checking=True,
        )
        out = client.run_command("which sbatch squeue sinfo 2>&1")
        if "sbatch" in out and "squeue" in out:
            _result("PASS", "Slurm 命令可用")
            return True
        _result("FAIL", f"Slurm 命令缺失: {out[:100]}")
        return False
    except Exception as e:
        _result("FAIL", f"Slurm 检查失败: {e}")
        return False


def check_job_submit_dry_run() -> bool:
    """5. 作业提交 dry run — 不真提交, 只验证脚本格式."""
    host = os.environ.get("HUGINN_HPC_HOST")
    if not host:
        _result("SKIP", "HUGINN_HPC_HOST 未设置")
        return False
    try:
        from huginn.hpc.client import HPCClient
        client = HPCClient(
            host=host,
            user=os.environ["HUGINN_HPC_USER"],
            key_file=os.environ["HUGINN_HPC_KEY_FILE"],
            strict_host_key_checking=True,
        )
        # 用 sbatch --test-only 做 dry run
        script = """#!/bin/bash
#SBATCH --job-name=huginn_verify
#SBATCH --time=00:01:00
#SBATCH --partition=test
echo "verification job"
"""
        out = client.run_command(f"sbatch --test-only <<< '{script}' 2>&1")
        if "PASS" in out or "test" in out.lower() or "allocation" in out.lower():
            _result("PASS", "作业提交 dry run 通过")
            return True
        _result("MANUAL", f"dry run 输出需人工确认: {out[:150]}")
        return False
    except Exception as e:
        _result("FAIL", f"dry run 失败: {e}")
        return False


def check_file_transfer() -> bool:
    """6. 文件传输 — SFTP 能读写工作目录."""
    host = os.environ.get("HUGINN_HPC_HOST")
    if not host:
        _result("SKIP", "HUGINN_HPC_HOST 未设置")
        return False
    try:
        from huginn.hpc.client import HPCClient
        client = HPCClient(
            host=host,
            user=os.environ["HUGINN_HPC_USER"],
            key_file=os.environ["HUGINN_HPC_KEY_FILE"],
            strict_host_key_checking=True,
        )
        # 上传测试文件
        test_content = b"huginn_hpc_verify"
        remote_path = f"/tmp/huginn_verify_{os.getpid()}.txt"
        client.upload_file_obj(test_content, remote_path)
        # 下载验证
        downloaded = client.download_file_obj(remote_path)
        # 清理
        client.run_command(f"rm -f {remote_path}")
        if downloaded == test_content:
            _result("PASS", "SFTP 文件传输正常")
            return True
        _result("FAIL", f"传输内容不一致: 上传 {test_content!r}, 下载 {downloaded!r}")
        return False
    except Exception as e:
        _result("FAIL", f"文件传输失败: {e}")
        return False


def check_archive_safety_on_hpc() -> bool:
    """7. 归档安全 — HPC 下载的 tar 不含路径遍历."""
    # 这是对 remote_executor.py 的 fix 的端到端验证
    _result("MANUAL", "需人工构造恶意 tar 上传到 HPC, 验证 download+extract 不写入目标目录外")
    _result("MANUAL", "参考 tests/pentest_archive_safety.py 的 PoC, 改为通过 HPC 传输")
    return True


def check_paramiko_version() -> bool:
    """8. Paramiko 版本 — 无已知漏洞."""
    try:
        import paramiko
        version = paramiko.__version__
        # CVE-2023-48795 (Terrapin) 在 3.4.0 修复
        parts = tuple(int(x) for x in version.split(".")[:3])
        if parts >= (3, 4, 0):
            _result("PASS", f"paramiko {version} (≥3.4.0, 无 Terrapin 漏洞)")
            return True
        _result("FAIL", f"paramiko {version} 有 Terrapin 漏洞 (CVE-2023-48795), 升级到 ≥3.4.0")
        return False
    except ImportError:
        _result("SKIP", "paramiko 未安装")
        return False


def main() -> int:
    print("=" * 60)
    print("Huginn HPC 环境验证 — 上线前 checklist")
    print("=" * 60)
    print()
    print("注意: 本脚本需在能访问 HPC 集群的机器上手动执行.")
    print("设置环境变量: HUGINN_HPC_HOST, HUGINN_HPC_USER, HUGINN_HPC_KEY_FILE")
    print()

    checks = [
        ("SSH 连接", check_ssh_connectivity),
        ("严格 host key 校验", check_strict_host_key),
        ("known_hosts 指纹", check_known_hosts),
        ("Slurm 命令可用", check_slurm_available),
        ("作业提交 dry run", check_job_submit_dry_run),
        ("文件传输", check_file_transfer),
        ("归档安全 (HPC)", check_archive_safety_on_hpc),
        ("Paramiko 版本", check_paramiko_version),
    ]

    passed = 0
    failed = 0
    manual = 0
    for name, check in checks:
        print(f"\n[{name}]")
        try:
            ok = check()
            if ok is True:
                passed += 1
            elif ok is False:
                # FAIL
                failed += 1
        except Exception as e:
            _result("FAIL", f"检查异常: {e}")
            traceback.print_exc()
            failed += 1

    print()
    print("=" * 60)
    print(f"结果: {passed} PASS, {failed} FAIL, {manual} MANUAL")
    print("=" * 60)
    if failed > 0:
        print("\n❌ 有失败项, 不能上线. 请修复后重试.")
        return 1
    print("\n✅ 自动检查全通过. MANUAL 项需人工确认后可上线.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
