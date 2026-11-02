#!/bin/bash
# T13 分块全量回归: 每块独立 pytest 进程, 避免单进程内存累积触发 cgroup OOM.
cd /workspace/agent || exit 1
PY=/workspace/venv/bin/python
BASE=(-o addopts="" -q -m "not network" --timeout=120 --timeout-method=thread -n 0 --maxfail=50)
SUMMARY=/workspace/agent/.chunk_summary.txt
: > "$SUMMARY"

mapfile -t ALL < <(ls tests/test_*.py)
SIZE=30
N=${#ALL[@]}
IDX=0
for ((i = 0; i < N; i += SIZE)); do
  IDX=$((IDX + 1))
  FILES=("${ALL[@]:i:SIZE}")
  LOG="/workspace/agent/.chunk_${IDX}.log"
  echo "=== chunk $IDX: ${#FILES[@]} files (${FILES[0]} .. ${FILES[-1]})" > "$LOG"
  "$PY" -m pytest "${BASE[@]}" "${FILES[@]}" >> "$LOG" 2>&1
  RC=$?
  # OOM / 崩溃时 RC 可能 >128
  LINE=$(grep -E "^(FAILED|ERROR) " "$LOG" | head -50)
  LAST=$(grep -E "passed|failed|error" "$LOG" | tail -1)
  echo "chunk $IDX rc=$RC | $LAST" >> "$SUMMARY"
  if [ -n "$LINE" ]; then
    echo "$LINE" >> "$SUMMARY"
  fi
done

for D in tests/loops tests/security tests/cv tests/e2e tests/hpc tests/golden tests/unit; do
  [ -d "$D" ] || continue
  IDX=$((IDX + 1))
  LOG="/workspace/agent/.chunk_${IDX}.log"
  echo "=== chunk $IDX: $D" > "$LOG"
  "$PY" -m pytest "${BASE[@]}" "$D" >> "$LOG" 2>&1
  RC=$?
  LINE=$(grep -E "^(FAILED|ERROR) " "$LOG" | head -50)
  LAST=$(grep -E "passed|failed|error" "$LOG" | tail -1)
  echo "chunk $IDX ($D) rc=$RC | $LAST" >> "$SUMMARY"
  if [ -n "$LINE" ]; then
    echo "$LINE" >> "$SUMMARY"
  fi
done

echo "ALL DONE" >> "$SUMMARY"
