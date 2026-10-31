import urllib.request
import json

urls = [
    "https://api.github.com/repos/plotly/datasets/contents/",
    "https://api.github.com/repos/plotly/datasets/contents/data",
    "https://raw.githubusercontent.com/plotly/datasets/master/iris.csv",
    "https://raw.githubusercontent.com/plotly/datasets/master/tips.csv",
    "https://www.google.com",
]
for u in urls:
    try:
        req = urllib.request.Request(u, headers={"User-Agent": "huginn/1.0"})
        r = urllib.request.urlopen(req, timeout=25)
        b = r.read()
        if "github.com" in u and u.startswith("https://api"):
            names = [d["name"] for d in json.loads(b) if "heart" in d["name"].lower()]
            print("OK", r.status, len(b), "-> heart matches:", names[:20])
        else:
            print("OK", r.status, len(b), u)
    except Exception as e:
        print("ERR", type(e).__name__, getattr(e, "code", ""), u)