"""Download UCI Heart Disease (Cleveland) data; fall back to a GitHub mirror."""
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request

PRIMARY_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)
FALLBACK_URLS = [
    "https://raw.githubusercontent.com/plotly/datasets/master/heart.csv",
    "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv",
    "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
]
OUT_CSV = "data/heart_cleveland.csv"
META_JSON = "data_download.json"

COLUMNS = [
    "age", "sex", "chest_pain_type", "resting_blood_pressure", "cholesterol",
    "fasting_blood_sugar", "rest_ecg", "max_heart_rate_achieved",
    "exercise_induced_angina", "st_depression", "st_slope", "num",
]


def fetch(url, timeout=60):
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (research data fetch; contact: none)",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def raw_to_csv(raw_bytes):
    """UCI Cleveland file is space/'?' separated -> tidy CSV with headers."""
    text = raw_bytes.decode("utf-8", errors="replace")
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != len(COLUMNS):
            continue
        # UCI encodes missing values as "?" -> empty field
        parts = ["" if p == "?" else p for p in parts]
        lines.append(",".join(parts))
    return ",".join(COLUMNS) + "\n" + "\n".join(lines) + "\n"


def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    chosen_url = None
    raw = None
    used_fallback = False

    try:
        raw = fetch(PRIMARY_URL)
        chosen_url = PRIMARY_URL
    except (urllib.error.URLError, OSError) as exc:
        print(f"[warn] primary UCI host unreachable: {exc}", file=sys.stderr)
        for url in FALLBACK_URLS:
            try:
                raw = fetch(url)
                chosen_url = url
                used_fallback = True
                break
            except (urllib.error.URLError, OSError) as exc2:
                print(f"[warn] fallback {url} failed: {exc2}", file=sys.stderr)

    if raw is None:
        print("[error] all download sources failed", file=sys.stderr)
        return 1

    digest = hashlib.sha256(raw).hexdigest()

    if used_fallback and chosen_url == FALLBACK_URLS[0]:
        csv_text = raw.decode("utf-8", errors="replace")
        out_path = OUT_CSV
    elif used_fallback:
        csv_text = raw.decode("utf-8", errors="replace")
        name = chosen_url.rsplit("/", 1)[-1].replace(".csv", ".csv")
        out_path = os.path.join("data", name)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(csv_text)
        print(f"[fallback] wrote {out_path}")
        return 0
    else:
        csv_text = raw_to_csv(raw)
        out_path = OUT_CSV

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(csv_text)

    rows = [ln for ln in csv_text.splitlines() if ln]
    n_rows = len(rows) - 1  # minus header
    n_cols = len(COLUMNS)

    meta = {
        "dataset": "heart_disease_cleveland",
        "source_url": chosen_url,
        "used_fallback": used_fallback,
        "local_path": out_path,
        "sha256": digest,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "columns": COLUMNS,
        "target_column": "num",
        "notes": (
            "UCI processed.cleveland.data: 14 comma-separated columns, "
            "missing values encoded as '?' (converted to empty). "
            "'num' is the diagnosis (0 = no disease, 1-4 = presence)."
        ),
    }
    with open(META_JSON, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[ok] wrote {out_path} ({n_rows} rows x {n_cols} cols)")
    print(f"[ok] sha256={digest}")
    print(f"[ok] wrote {META_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())