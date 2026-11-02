import urllib.request

urls = [
    "https://raw.githubusercontent.com/plotly/datasets/master/heart.csv",
    "https://raw.githubusercontent.com/plotly/datasets/master/data/heart.csv",
    "https://raw.githubusercontent.com/plotly/datasets/master/heart-disease-uci.csv",
    "https://archive.ics.uci.edu/static/public/45/heart+disease.csv",
    "https://raw.githubusercontent.com/scipy/scipy/main/doc/source/stats/data/heart.csv",
    "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/heart.csv",
]
for u in urls:
    try:
        r = urllib.request.urlopen(u, timeout=25)
        b = r.read()
        print("OK", r.status, len(b), u)
    except Exception as e:
        print("ERR", type(e).__name__, getattr(e, "code", ""), u)