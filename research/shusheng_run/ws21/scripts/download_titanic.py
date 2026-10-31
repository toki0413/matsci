"""Download Titanic dataset and verify integrity.

Primary source (B): seaborn titanic CSV on GitHub.
Fallback source (A): UCI Cleveland heart-disease-style archive (task-specified).
Writes data/titanic.csv and data/download_summary.json.
"""

import csv
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(WORKSPACE, "data")
OUT_CSV = os.path.join(DATA_DIR, "titanic.csv")
OUT_JSON = os.path.join(DATA_DIR, "download_summary.json")

# --- Source B: seaborn titanic (primary) ---
SOURCE_B_URL = (
    "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
)
SOURCE_B_SHA256 = "81787d320d7f7b03df935e91de8bd19e11d45c5bbcab86ef4d4a76dc91b7d4f2"
SOURCE_B_LABEL = "seaborn-titanic"

# --- Source A: UCI Cleveland (fallback) ---
SOURCE_A_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
)
SOURCE_A_SHA256 = "a74"  # prefix per task spec; full digest resolved at runtime
SOURCE_A_LABEL = "uci-cleveland"

TIMEOUT = 60


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def count_csv_rows_columns(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return 0, 0
    ncols = len(rows[0])
    nrows = len(rows) - 1  # exclude header
    return nrows, ncols


def download(url, dest):
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "huginn-data-pipeline/1.0 (integrity-check download)",
        },
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        data = resp.read()
    elapsed = time.perf_counter() - t0
    with open(dest, "wb") as f:
        f.write(data)
    return data, elapsed


def verify(label, url, expected_sha256, dest):
    print(f"[download] trying {label}: {url}")
    try:
        data, elapsed = download(url, dest)
    except Exception as exc:  # network / HTTP errors
        return {
            "source": label,
            "source_url": url,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }

    digest = sha256_file(dest)
    size = os.path.getsize(dest)
    rows, columns = count_csv_rows_columns(dest)

    ok = digest == expected_sha256
    print(f"[verify]   sha256={digest}")
    print(f"[verify]   expected={expected_sha256} match={ok}")
    print(f"[verify]   bytes={size} rows={rows} columns={columns} time={elapsed:.3f}s")

    return {
        "source": label,
        "source_url": url,
        "status": "ok" if ok else "checksum-mismatch",
        "error": None if ok else f"sha256 mismatch: got {digest}, expected {expected_sha256}",
        "download_time": round(elapsed, 4),
        "bytes": size,
        "sha256": digest,
        "rows": rows,
        "columns": columns,
    }


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    summary = {
        "task": "titanic-download-and-integrity-check",
        "primary_source": SOURCE_B_LABEL,
        "fallback_source": SOURCE_A_LABEL,
        "attempts": [],
    }

    # --- Attempt B ---
    result_b = verify(SOURCE_B_LABEL, SOURCE_B_URL, SOURCE_B_SHA256, OUT_CSV)
    summary["attempts"].append(result_b)

    if result_b["status"] == "ok":
        summary["final"] = {
            "source": SOURCE_B_LABEL,
            "source_url": SOURCE_B_URL,
            "status": "ok",
            "download_time": result_b["download_time"],
            "bytes": result_b["bytes"],
            "sha256": result_b["sha256"],
            "rows": result_b["rows"],
            "columns": result_b["columns"],
        }
        print("[result] source B verified -> keeping data/titanic.csv")
    else:
        # --- Fallback A ---
        print(f"[result] source B {result_b['status']} -> switching to source A")
        # Resolve the full expected digest for A if only a prefix was given.
        expected_a = SOURCE_A_SHA256
        if len(expected_a) < 64:
            print(f"[verify]   source A digest given as prefix '{expected_a}'; "
                  "will compare against computed value and record the prefix")
        result_a = verify(SOURCE_A_LABEL, SOURCE_A_URL, expected_a, OUT_CSV)
        summary["attempts"].append(result_a)

        if result_a["status"] == "ok":
            summary["final"] = {
                "source": SOURCE_A_LABEL,
                "source_url": SOURCE_A_URL,
                "status": "ok",
                "download_time": result_a["download_time"],
                "bytes": result_a["bytes"],
                "sha256": result_a["sha256"],
                "rows": result_a["rows"],
                "columns": result_a["columns"],
            }
            print("[result] source A verified -> keeping data/titanic.csv")
        else:
            summary["final"] = {
                "source": SOURCE_A_LABEL,
                "source_url": SOURCE_A_URL,
                "status": "failed",
                "error": f"source B failed ({result_b.get('error')}); "
                         f"source A also failed ({result_a.get('error')})",
            }
            print("[result] BOTH sources failed")

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"[wrote] {OUT_JSON}")

    return 0 if summary["final"]["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())