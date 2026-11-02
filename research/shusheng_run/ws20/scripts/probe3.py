import urllib.request
import urllib.error
import json

# Plotly datasets repo listing via GitHub API (no auth needed, rate-limited but ok)
paths = [
    "https://api.github.com/repos/plotly/datasets/contents/",
    "https://api.github.com/repos/plotly/datasets/contents/data",
]
for u in paths:
    try:
        req = urllib.request.Request(u, headers={"User-Agent": "huginn/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        names = [d["name"] for d in data if "heart" in d["name"].lower()]
        print(u, "->", len(names), "matches:", names[:20])
    except urllib.error.HTTPError as e:
        print("HTTP", e.code, u)
    except Exception as e:
        print(type(e).__name__, e, u)