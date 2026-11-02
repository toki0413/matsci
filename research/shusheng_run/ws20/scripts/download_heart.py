#!/usr/bin/env python3
"""Download heart.csv from raw.githubusercontent.com, validate it, and emit a summary.

On failure: write failure_log.txt and fall back to an alternative dataset
(seaborn's titanic, or a UCI CSV mirror).
"""

import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request

URL = "https://raw.githubusercontent.com/plotly/datasets/master/heart.csv"
DEST_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DEST = os.path.join(DEST_DIR, "heart.csv")
FAILURE_LOG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "failure_log.txt")
MAX_BYTES = 5 * 1024 * 1024  # 5 MB


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def log_failure(msg):
    os.makedirs(os.path.dirname(FAILURE_LOG), exist_ok=True)
    with open(FAILURE_LOG, "w") as f:
        f.write(msg + "\n")
    print("[FAILURE] " + msg, file=sys.stderr)


def fallback_dataset():
    """Try seaborn titanic; if unavailable, try a UCI CSV mirror."""
    print("[FALLBACK] Attempting seaborn titanic dataset ...")
    try:
        import pandas as pd
        import seaborn as sns

        df = sns.load_dataset("titanic")
        out = os.path.join(DEST_DIR, "titanic.csv")
        df.to_csv(out, index=False)
        print(f"[FALLBACK OK] seaborn titanic -> {out}  rows={len(df)} cols={len(df.columns)}")
        return out, df
    except Exception as exc:
        print(f"[FALLBACK] seaborn failed: {exc}", file=sys.stderr)

    print("[FALLBACK] Attempting UCI heart disease CSV mirror ...")
    uci_urls = [
        "https://archive.ics.uci.edu/static/public/45/heart+disease.csv",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
    ]
    for url in uci_urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "huginn-data-fetch/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
            if len(data) > MAX_BYTES:
                continue
            ext = "csv" if "csv" in url else "data"
            out = os.path.join(DEST_DIR, f"heart_uci.{ext}")
            with open(out, "wb") as f:
                f.write(data)
            print(f"[FALLBACK OK] UCI -> {out}  bytes={len(data)}")
            return out, None
        except Exception as exc:
            print(f"[FALLBACK] UCI {url} failed: {exc}", file=sys.stderr)
    return None, None


def main():
    os.makedirs(DEST_DIR, exist_ok=True)
    summary = {"url": URL, "dest": DEST, "success": False}

    print(f"[DOWNLOAD] {URL}")
    print(f"[DEST]     {DEST}")

    try:
        req = urllib.request.Request(URL, headers={"User-Agent": "huginn-data-fetch/1.0"})
        with urllib.request.urlopen(req, timeout=90) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        log_failure(f"HTTP {e.code} for {URL}: {e.reason}")
        summary["error"] = f"HTTP {e.code}"
    except urllib.error.URLError as e:
        log_failure(f"URL error for {URL}: {e.reason}")
        summary["error"] = f"URLError: {e.reason}"
    except Exception as e:
        log_failure(f"Download exception for {URL}: {type(e).__name__}: {e}")
        summary["error"] = f"{type(e).__name__}: {e}"
    else:
        size = len(raw)
        print(f"[RECEIVED] {size} bytes")
        if size > MAX_BYTES:
            msg = f"File too large: {size} bytes > {MAX_BYTES} bytes limit"
            log_failure(msg)
            summary["error"] = msg
        else:
            with open(DEST, "wb") as f:
                f.write(raw)
            digest = sha256_file(DEST)
            print(f"[SHA256]   {digest}")

            try:
                import pandas as pd

                df = pd.read_csv(DEST)
                rows, cols = len(df), len(df.columns)
                colnames = list(df.columns)
                summary.update(
                    {
                        "success": True,
                        "bytes": size,
                        "sha256": digest,
                        "rows": rows,
                        "cols": cols,
                        "columns": colnames,
                        "dtypes": {k: str(v) for k, v in df.dtypes.items()},
                    }
                )
                print(f"[PARSED]   rows={rows} cols={cols}")
                print(f"[COLUMNS]  {colnames}")
            except Exception as e:
                msg = f"Pandas parse failed: {type(e).__name__}: {e}"
                log_failure(msg)
                summary["error"] = msg

    if not summary.get("success"):
        fb_path, fb_df = fallback_dataset()
        if fb_path:
            summary["fallback"] = fb_path
            if fb_df is not None:
                summary["fallback_rows"] = len(fb_df)
                summary["fallback_cols"] = len(fb_df.columns)
                summary["fallback_columns"] = list(fb_df.columns)
        else:
            summary["fallback"] = None
            log_failure("No fallback dataset available.")

    out_json = os.path.join(DEST_DIR, "download_summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[SUMMARY]  written to {out_json}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary.get("success") else 1


if __name__ == "__main__":
    sys.exit(main())