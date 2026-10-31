import urllib.request
import urllib.error

urls = [
    "https://raw.githubusercontent.com/plotly/datasets/master/heart.csv",
    "https://archive.ics.uci.edu/static/public/45/heart+disease.csv",
    "https://raw.githubusercontent.com/plotly/datasets/master/iris.csv",
    "https://www.google.com",
]
for u in urls:
    try:
        req = urllib.request.Request(u, headers={"User-Agent": "huginn/1.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            b = resp.read()
        print("OK", resp.status, len(b), u)
    except urllib.error.HTTPError as e:
        print("HTTP", e.code, e.reason, u)
    except Exception as e:
        print(type(e).__name__, e, u)