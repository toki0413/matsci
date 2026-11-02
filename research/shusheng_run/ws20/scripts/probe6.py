import urllib.request
import json

req = urllib.request.Request(
    "https://api.github.com/repos/plotly/datasets/contents/",
    headers={"User-Agent": "huginn/1.0"},
)
try:
    r = urllib.request.urlopen(req, timeout=30)
    data = json.loads(r.read())
    names = sorted(d["name"] for d in data if "heart" in d["name"].lower())
    print("heart matches:", names)
    print("total files:", len(data))
except Exception as e:
    print("ERR", type(e).__name__, getattr(e, "code", ""), e)