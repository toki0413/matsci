import json
r = json.load(open('/workspace/research/shusheng_run/ws35/results/results.json'))
print('H1 keys:', list(r['h1'].keys()))
print('H2 keys:', list(r['h2'].keys()))
print('H3 keys:', list(r['h3'].keys()))
print('H4 keys:', list(r['h4'].keys()))
print('H5 keys:', list(r['h5'].keys()))
print()
print('H1:', {k: r['h1'][k] for k in ['n_pairs', 'n_papers', 'mean_delta_ea_fixed', 'mean_delta_ea_random', 'tau_sq', 'bootstrap_ci_lower', 'bootstrap_ci_upper']})
print('H2:', {k: r['h2'][k] for k in ['eta_squared', 'p_value', 'ss_between', 'ss_total']})
print('H3 n_pairs:', r['h3']['n_pairs'])
print('H3 pairs:', r['h3']['pairs'])
print('H4:', {k: r['h4'][k] for k in ['n_papers', 'rmse', 'mae', 'r2']})
h5 = r['h5']
print('H5:', {k: h5[k] for k in ['n_train', 'n_holdout', 'predicted_values']})
if 'rmse' in h5:
    print('  rmse:', h5['rmse'])