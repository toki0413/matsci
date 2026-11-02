"""Step (1) -- download UCI Cleveland heart disease data (dataset A).

Uses only the standard library (urllib) to fetch the raw artifact, verifies its
SHA-256 against the task-specified credential, and writes
``data/download_summary.json`` with the required fields:

    source_url, timestamp, bytes, sha256, rows=303, cols=14, error

The raw file has no header and exactly 303 data rows x 14 comma-separated
columns. Missing values are encoded as the literal ``?`` in columns 1-13.

Exit code is always 0; any failure (network, digest mismatch, shape mismatch)
is recorded in the ``error`` field so downstream orchestration is never aborted.
"""

import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(WORKSPACE, "data")
OUT_PATH = os.path.join(DATA_DIR, "processed.cleveland.data")
SUMMARY_PATH = os.path.join(DATA_DIR, "download_summary.json")

# Task-specified expected SHA-256 of the raw artifact.
EXPECTED_SHA256 = (
    "a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8"
)

# Primary UCI mirror. A GitHub mirror of the same artifact is kept as a
# fallback so the pipeline can still run if archive.ics.uci.edu is unreachable.
SOURCE_URLS = [
    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
    "https://raw.githubusercontent.com/plotly/datasets/master/heart.csv",
]

EXPECTED_ROWS = 303
EXPECTED_COLS = 14
TIMEOUT = 60
CHUNK = 1 << 20


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def count_rows_cols(path):
    """Count physical rows and comma-separated columns in the raw file."""
    rows = 0
    cols = None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows += 1
            n = len(line.split(","))
            if cols is None:
                cols = n
            elif n != cols:
                cols = None  # ragged file
    return rows, cols


def download(url, dest):
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "huginn-coder/1.0 (research pipeline; dataset A fetch)",
        },
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp, open(
        dest, "wb"
    ) as out:
        status = getattr(resp, "status", None) or getattr(resp, "code", None)
        while True:
            block = resp.read(CHUNK)
            if not block:
                break
            out.write(block)
    elapsed = time.time() - t0
    return status, elapsed


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    summary = {
        "task": "uci-cleveland-heart-disease-download",
        "dataset": "A (UCI Cleveland heart disease)",
        "expected_sha256": EXPECTED_SHA256,
        "attempts": [],
        "final": None,
    }

    error = None
    source_url = None
    status = None
    elapsed = None
    digest = None
    rows = None
    cols = None
    bytes_ = None

    for url in SOURCE_URLS:
        attempt = {"source_url": url, "status": None, "error": None}
        try:
            status, elapsed = download(url, OUT_PATH)
            attempt["http_status"] = status
            attempt["download_time"] = round(elapsed, 4)
            if status != 200:
                attempt["status"] = "http_error"
                attempt["error"] = f"HTTP {status}"
                summary["attempts"].append(attempt)
                continue
        except urllib.error.URLError as e:
            attempt["status"] = "network_error"
            attempt["error"] = f"URLError: {e.reason}"
            summary["attempts"].append(attempt)
            continue
        except Exception as e:  # noqa: BLE001 -- recorded, never crashes
            attempt["status"] = "download_error"
            attempt["error"] = f"{type(e).__name__}: {e}"
            summary["attempts"].append(attempt)
            continue

        bytes_ = os.path.getsize(OUT_PATH)
        digest = sha256_file(OUT_PATH)
        attempt["bytes"] = bytes_
        attempt["sha256"] = digest
        attempt["digest_match"] = digest == EXPECTED_SHA256

        try:
            rows, cols = count_rows_cols(OUT_PATH)
        except Exception as e:  # noqa: BLE001
            rows, cols = None, None
        attempt["rows"] = rows
        attempt["cols"] = cols

        summary["attempts"].append(attempt)

        if digest == EXPECTED_SHA256 and rows == EXPECTED_ROWS and cols == EXPECTED_COLS:
            source_url = url
            break
        else:
            # Keep the file but note the mismatch; try the next mirror.
            error = (
                f"integrity check failed: digest_match={digest == EXPECTED_SHA256} "
                f"(got {digest}), rows={rows} (expected {EXPECTED_ROWS}), "
                f"cols={cols} (expected {EXPECTED_COLS})"
            )

    if source_url is None and summary["attempts"]:
        # Fall back to the last attempt's file if we have one.
        last = summary["attempts"][-1]
        source_url = last["source_url"]
        status = last.get("http_status") or last.get("status")
        elapsed = last.get("download_time")
        bytes_ = last.get("bytes") or (os.path.getsize(OUT_PATH) if os.path.exists(OUT_PATH) else None)
        digest = last.get("sha256") or sha256_file(OUT_PATH) if os.path.exists(OUT_PATH) else None
        rows = last.get("rows")
        cols = last.get("cols")
        if error is None:
            error = last.get("error")

    if digest is None and os.path.exists(OUT_PATH):
        digest = sha256_file(OUT_PATH)
    if bytes_ is None and os.path.exists(OUT_PATH):
        bytes_ = os.path.getsize(OUT_PATH)
    if rows is None and os.path.exists(OUT_PATH):
        rows, cols = count_rows_cols(OUT_PATH)

    digest_match = digest == EXPECTED_SHA256 if digest else False
    shape_ok = rows == EXPECTED_ROWS and cols == EXPECTED_COLS

    if not digest_match:
        error = (error or "") + f"; digest mismatch (got {digest})" if error else f"digest mismatch (got {digest})"
    if not shape_ok:
        error = (error or "") + f"; shape {rows}x{cols} != {EXPECTED_ROWS}x{EXPECTED_COLS}" if error else f"shape {rows}x{cols} != {EXPECTED_ROWS}x{EXPECTED_COLS}"

    final = {
        "source_url": source_url,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "ok" if (digest_match and shape_ok and error is None) else "failed",
        "http_status": status,
        "error": error,
        "download_time": round(elapsed, 4) if elapsed else None,
        "bytes": bytes_,
        "sha256": digest,
        "rows": rows,
        "cols": cols,
        "digest_match": digest_match,
        "shape_match": shape_ok,
    }
    summary["final"] = final

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"[wrote] {SUMMARY_PATH}")
    print(f"  source_url : {source_url}")
    print(f"  bytes      : {bytes_}")
    print(f"  sha256     : {digest}")
    print(f"  digest_match: {digest_match}")
    print(f"  shape      : {rows}x{cols} (expected {EXPECTED_ROWS}x{EXPECTED_COLS})")
    print(f"  error      : {error}")

    # Exit 0 on any failure, per task spec.
    return 0


if __name__ == "__main__":
    sys.exit(main())