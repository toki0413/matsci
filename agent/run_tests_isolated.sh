#!/usr/bin/env bash
# 在内存受限环境下逐文件运行测试 (每个文件独立子进程, 内存不累积).
# 用法: bash run_tests_isolated.sh [--parallel N]
# 输出: test_results.jsonl + 汇总报告
set -uo pipefail

cd /workspace/agent
RESULTS_FILE="test_results.jsonl"
FAILED_LIST="test_failed.txt"
SKIPPED_LIST="test_skipped_env.txt"
PARALLEL=1
SKIP_DIRS="benchmark|stress|property_based|__pycache__"

# 解析参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --parallel) PARALLEL="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# 清理旧结果
: > "$RESULTS_FILE"
: > "$FAILED_LIST"
: > "$SKIPPED_LIST"

# 收集测试文件 (跳过重型目录)
mapfile -t TEST_FILES < <(
    find tests -maxdepth 2 -name "*.py" \
        \( -name "test_*.py" -o -name "*_test.py" \) \
        | grep -vE "$SKIP_DIRS" \
        | sort
)

TOTAL=${#TEST_FILES[@]}
echo ">>> 发现 $TOTAL 个测试文件, 并行度=$PARALLEL" >&2
echo ">>> 跳过目录: $SKIP_DIRS" >&2

run_one() {
    local f="$1"
    # 每个文件独立子进程, 内存隔离; -p no:cacheprovider 省内存
    # 超时 1200s (test_api_contract.py 需 ~16min)
    local output
    local exit_code
    output=$(timeout 1200 python -m pytest "$f" \
        -o addopts="" \
        --no-header -q --tb=no \
        -p no:warnings -p no:cacheprovider \
        2>&1)
    exit_code=$?

    # 解析 passed/failed/skipped/error
    local line
    line=$(echo "$output" | grep -E "passed|failed|error|skipped|no tests ran" | tail -1)

    # JSONL 记录
    local ts
    ts=$(date +%s)
    printf '{"file":"%s","exit":%d,"ts":%d,"summary":"%s"}\n' \
        "$f" "$exit_code" "$ts" "${line//\"/\\\"}" >> "$RESULTS_FILE"

    # 实时进度
    if [[ $exit_code -ne 0 ]]; then
        echo "FAIL  $f  [$line]" >&2
        echo "$f" >> "$FAILED_LIST"
    else
        echo "OK    $f  [$line]" >&2
    fi
}
export -f run_one
export RESULTS_FILE FAILED_LIST SKIPPED_LIST

# 并行或串行运行
if [[ "$PARALLEL" -gt 1 ]]; then
    # 用 xargs 并行 (注意: 内存够时才用, 一般 PARALLEL=1)
    printf '%s\n' "${TEST_FILES[@]}" | \
        xargs -P "$PARALLEL" -I {} bash -c 'run_one "$@"' _ {} 2>&1
else
    for f in "${TEST_FILES[@]}"; do
        run_one "$f"
    done
fi

# 汇总
echo "" >&2
echo "========================================" >&2
echo "  汇总报告" >&2
echo "========================================" >&2
TOTAL_OK=$(grep -c '"exit":0' "$RESULTS_FILE" 2>/dev/null || echo 0)
TOTAL_FAIL=$(grep -vc '"exit":0' "$RESULTS_FILE" 2>/dev/null || echo 0)
echo "总文件数: $TOTAL" >&2
echo "通过: $TOTAL_OK" >&2
echo "失败: $TOTAL_FAIL" >&2
echo "" >&2
echo "详细结果: $RESULTS_FILE" >&2
echo "失败列表: $FAILED_LIST" >&2

# 如果有失败, 列出失败文件
if [[ -s "$FAILED_LIST" ]]; then
    echo "" >&2
    echo "失败文件:" >&2
    cat "$FAILED_LIST" >&2
fi
