"""Download UCI Cleveland heart disease dataset (dataset A).

Fetches ``processed.cleveland.data`` from the UCI Machine Learning Repository
into ``data/``, computes its SHA-256, verifies it against the expected digest,
and writes ``data/download_summary.json``.

Exit code is 0 on *any* failure (including digest mismatch, network error or
missing fields) so that downstream orchestration never aborts; the failure is
recorded in the ``error`` field of the summary instead.
"""

import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request

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

TIMEOUT = 60
CHUNK = 1 << 20


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url):
    """Download *url* to OUT_PATH. Return (bytes_written, elapsed_seconds)."""
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "huginn-dataset-a-downloader/1.0 (integrity-checked)",
            "Accept": "*/*",
        },
    )
    start = time.time()
    total = 0
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp, open(
        OUT_PATH, "wb"
    ) as out:
        for chunk in iter(lambda: resp.read(CHUNK), b""):
            out.write(chunk)
            total += len(chunk)
    return total, time.time() - start


def count_rows_cols(path):
    """Count data rows and columns in the raw UCI file.

    The UCI file is comma-separated with no header, so the column count is
    taken from the first non-empty line and rows from all non-empty lines.
    """
    rows = 0
    cols = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows += 1
            cols = max(cols, len(line.split(",")))
    return rows, cols


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    summary = {
        "task": "uci-cleveland-heart-disease-download",
        "dataset": "A (UCI Cleveland heart disease)",
        "expected_sha256": EXPECTED_SHA256,
        "attempts": [],
        "final": None,
    }

    last_error = None
    for url in SOURCE_URLS:
        attempt = {
            "source_url": url,
            "status": "failed",
            "error": None,
            "download_time": None,
            "bytes": None,
            "sha256": None,
            "rows": None,
            "cols": None,
            "digest_match": None,
        }
        try:
            nbytes, elapsed = fetch(url)
            attempt["download_time"] = round(elapsed, 4)
            attempt["bytes"] = nbytes

            digest = sha256_file(OUT_PATH)
            attempt["sha256"] = digest
            attempt["digest_match"] = digest.lower() == EXPECTED_SHA256.lower()

            rows, cols = count_rows_cols(OUT_PATH)
            attempt["rows"] = rows
            attempt["cols"] = cols

            if attempt["digest_match"]:
                attempt["status"] = "ok"
                print(f"[ok] {url} -> {OUT_PATH} "
                      f"({nbytes} B, {rows} rows, {cols} cols, sha256 ok)")
            else:
                attempt["error"] = (
                    f"sha256 mismatch: got {digest}, expected "
                    f"{EXPECTED_SHA256}"
                )
                print(f"[warn] {url} downloaded but digest mismatch: "
                      f"{attempt['error']}")
        except urllib.error.HTTPError as e:
            attempt["error"] = f"HTTP {e.code} {e.reason} for {url}"
            print(f"[warn] {attempt['error']}")
        except urllib.error.URLError as e:
            attempt["error"] = f"URL error for {url}: {e.reason}"
            print(f"[warn] {attempt['error']}")
        except Exception as e:  # noqa: BLE001 - never abort the pipeline
            attempt["error"] = f"{type(e).__name__} for {url}: {e}"
            print(f"[warn] {attempt['error']}")

        summary["attempts"].append(attempt)
        if attempt["status"] == "ok":
            summary["final"] = attempt
            break

    if summary["final"] is None:
        # Keep the last attempt's details as the final record so the failure is
        # diagnosable even when every source failed.
        summary["final"] = dict(summary["attempts"][-1])
        summary["final"]["status"] = "failed"
        last_error = summary["final"]["error"]
        print(f"[result] all sources failed: {last_error}")

    summary["final"] = {
        "source_url": summary["final"]["source_url"],
        "status": summary["final"]["status"],
        "error": summary["final"].get("error"),
        "download_time": summary["final"].get("download_time"),
        "bytes": summary["final"].get("bytes"),
        "sha256": summary["final"].get("sha256"),
        "rows": summary["final"].get("rows"),
        "cols": summary["final"].get("cols"),
        "digest_match": summary["final"].get("digest_match"),
    }

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"[wrote] {SUMMARY_PATH}")

    # Exit 0 on any failure, per task spec.
    return 0


if __name__ == "__main__":
    sys.exit(main())