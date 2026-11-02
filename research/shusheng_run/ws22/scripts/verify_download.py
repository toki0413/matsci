"""Independent verification of the downloaded artifact (fresh process)."""
import csv
import hashlib
import json
import os

P = "data/titanic.csv"
EXPECTED = "81787d320d7f7b03df935e91de8bd19e11d45c5bbcab86ef4d4a76dc91b7d4f2"

h = hashlib.sha256()
with open(P, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
        h.update(chunk)
digest = h.hexdigest()

size = os.path.getsize(P)
with open(P, encoding="utf-8", newline="") as f:
    rows = list(csv.reader(f))
nrows, ncols = len(rows) - 1, len(rows[0])

summary = json.load(open("data/download_summary.json"))
final = summary["final"]

checks = {
    "sha256_match": digest == EXPECTED,
    "sha256_vs_json": digest == final["sha256"],
    "bytes_vs_json": size == final["bytes"],
    "rows_vs_json": nrows == final["rows"],
    "columns_vs_json": ncols == final["columns"],
    "json_status_ok": final["status"] == "ok",
    "required_keys": all(k in final for k in
                         ("source_url", "download_time", "bytes", "sha256", "rows", "columns")),
}

print(f"file      : {P}")
print(f"sha256    : {digest}")
print(f"expected  : {EXPECTED}")
print(f"bytes     : {size}  rows={nrows}  columns={ncols}")
print(f"header    : {rows[0]}")
print(f"first row : {rows[1]}")
print(f"last row  : {rows[-1]}")
print()
for k, v in checks.items():
    print(f"  [{'PASS' if v else 'FAIL'}] {k}")
print()
print("ALL CHECKS PASSED" if all(checks.values()) else "SOME CHECKS FAILED")