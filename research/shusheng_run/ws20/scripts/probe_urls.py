import urllib.request

urls = [
    "https://raw.githubusercontent.com/plotly/datasets/master/heart.csv",
    "https://raw.githubusercontent.com/plotly/datasets/master/data/heart.csv",
    "https://raw.githubusercontent.com/scipy/scipy/main/doc/source/stats/data/heart.csv",
    "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/heart.csv",
    "https://raw.githubusercontent.com/jeffheaton/Machine-Learning-and-Python-course/main/heart.csv",
    "https://raw.githubusercontent.com/IBM/heart-disease-prediction/main/heart.csv",
    "https://raw.githubusercontent.com/plotly/datasets/master/heart-disease-uci.csv",
]
for u in urls:
    try:
        r = urllib.request.Request(u, headers={"User-Agent": "huginn/1.0"})
        with urllib.request.urlopen(r, timeout=30) as resp:
            body = resp.read()
            print(f"OK  {resp.status}  {len(body):>8}  {u}")
    except Exception as e:
        print(f"ERR {type(e).__name__:20} {u}")