import analysis

df = analysis.load_data()
df = analysis.build_target(df)
res = analysis.build_results(df)
print("built", len(res))
import json
print(json.dumps(res["welch_t_test"]["thalach"], indent=2))
print(json.dumps(res["point_biserial_correlation"]["thalach"], indent=2))
print(json.dumps(res["shapiro_summary"], indent=2))
print(json.dumps(res["bootstrap_robustness"]["thalach"], indent=2))
rep = analysis.render_report(res)
print("report chars:", len(rep))