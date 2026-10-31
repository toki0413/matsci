"""Download UCI Heart Disease (Cleveland) dataset to data/heart.csv.

Primary URL: https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data
Fallback: the heart-disease zip archive hosted on the UCI archive.
"""

import hashlib
import io
import os
import sys
import urllib.request
import zipfile

PRIMARY_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
)
FALLBACK_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.zip"
)
OUT_PATH = "data/heart.csv"
CHUNK = 1 << 20

# The raw Cleveland file has no header. Column names follow the UCI description.
COLUMNS = [
    "age",
    "sex",
    "chest_pain_type",
    "resting_bp",
    "cholesterol",
    "fasting_blood_sugar",
    "rest_ecg",
    "max_hr",
    "exercise_angina",
    "st_depression",
    "st_slope",
    "num",  # raw target as published (0 = no disease, 1-4 = severity)
]


def fetch(url, timeout=60):
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (compatible; research-data-fetcher/1.0; "
                "python-urllib)"
            )
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def write_csv(raw_bytes):
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "wb") as fh:
        fh.write(raw_bytes)
    return OUT_PATH


def main():
    data = None
    source = None

    print("[download] primary URL:", PRIMARY_URL)
    try:
        data = fetch(PRIMARY_URL)
        source = "primary"
        print("[download] primary fetch OK, bytes:", len(data))
    except Exception as exc:  # network / HTTP error -> fallback
        print("[download] primary FAILED:", repr(exc))
        print("[download] trying fallback zip:", FALLBACK_URL)
        try:
            raw = fetch(FALLBACK_URL)
            source = "fallback"
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                names = [n for n in zf.namelist() if "cleveland" in n.lower()]
                if not names:
                    raise RuntimeError(
                        "fallback zip contains no cleveland member: "
                        + str(zf.namelist())
                    )
                data = zf.read(names[0])
            print("[download] fallback zip OK, extracted bytes:", len(data))
        except Exception as exc2:
            print("[download] fallback FAILED:", repr(exc2))
            sys.exit(1)

    path = write_csv(data)

    sha = hashlib.sha256(data).hexdigest()
    size = len(data)
    # Row/column counts on the raw bytes (trailing newline -> one empty line).
    lines = [ln for ln in data.decode("latin-1").splitlines() if ln.strip()]
    rows = len(lines)
    cols = len(lines[0].split(","))

    print("[download] saved to:", path)
    print("[download] source:", source)
    print("[download] sha256:", sha)
    print("[download] bytes:", size)
    print("[download] rows:", rows)
    print("[download] cols:", cols)

    # Also emit a machine-readable summary for downstream scripts.
    summary = {
        "path": path,
        "source": source,
        "sha256": sha,
        "bytes": size,
        "rows": rows,
        "cols": cols,
        "columns": COLUMNS,
    }
    summary_path = "data/_download_summary.json"
    with open(summary_path, "w") as fh:
        import json

        json.dump(summary, fh, indent=2)
    print("[download] summary written to:", summary_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())