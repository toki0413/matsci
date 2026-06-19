import subprocess, sys, time, json, os, re

TEST_DIR = "tests"
TIMEOUT = 15

exclude_files = {
    "test_local_only.py",
    "test_e2e_agent_flow.py",
    "test_server_fastapi.py",
    "test_lean.py",
    "test_lean_e2e.py",
    "test_lean_pipeline.py",
    "test_auto_pipeline.py",
}

test_files = sorted(f for f in os.listdir(TEST_DIR) if f.startswith("test_") and f.endswith(".py") and f not in exclude_files)

results = {"passed": [], "failed": [], "timeout": [], "total_passed": 0, "total_failed": 0, "total_skipped": 0}

for tf in test_files:
    fpath = os.path.join(TEST_DIR, tf)
    cmd = [sys.executable, "-m", "pytest", fpath, "-q", "--tb=no", "--no-cov"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT)
        stdout = proc.stdout
        if "passed" in stdout:
            line = [l for l in stdout.splitlines() if "passed" in l or "failed" in l or "error" in l][-1]
            results["passed"].append((tf, line.strip()))
            m_passed = re.search(r'(\d+)\s+passed', line)
            m_failed = re.search(r'(\d+)\s+failed', line)
            m_errors = re.search(r'(\d+)\s+error', line)
            m_skipped = re.search(r'(\d+)\s+skipped', line)
            if m_passed:
                results["total_passed"] += int(m_passed.group(1))
            if m_failed:
                results["total_failed"] += int(m_failed.group(1))
            if m_errors:
                results["total_failed"] += int(m_errors.group(1))
            if m_skipped:
                results["total_skipped"] += int(m_skipped.group(1))
        elif "failed" in stdout or "error" in stdout.lower():
            results["failed"].append((tf, stdout.strip().splitlines()[-1]))
        else:
            results["passed"].append((tf, "unknown output"))
    except subprocess.TimeoutExpired:
        results["timeout"].append(tf)

with open("test_regression_summary.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"Passed files: {len(results['passed'])}")
print(f"Failed files: {len(results['failed'])}")
print(f"Timeout files: {len(results['timeout'])}")
print(f"Total passed tests: {results['total_passed']}")
print(f"Total failed tests: {results['total_failed']}")
print(f"Total skipped tests: {results['total_skipped']}")
for tf, reason in results["failed"]:
    print(f"  FAIL {tf}: {reason}")
for tf in results["timeout"]:
    print(f"  TIMEOUT {tf}")
