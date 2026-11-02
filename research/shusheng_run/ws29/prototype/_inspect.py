import numpy as np
from bed_model import BedModel, energy_balance

m = BedModel(N=41, L=0.02, T0=573.0, T_cool=573.0, p_supply=2.0)
times, Ts, Xs = m.simulate(t_end=3000.0, dt=10.0, record_every=10)
eb = energy_balance(m, times, Ts, Xs, m.p_supply)
print("keys:", {k: (np.asarray(v).shape, v.dtype if hasattr(v, "dtype") else type(v))
                for k, v in eb.items()})
for i in range(0, len(eb["times"]), 5):
    print(f"t={eb['times'][i]:7.1f} U={eb['U'][i]:12.3e} Qrel={eb['Q_rel'][i]:12.3e} "
          f"Qout={eb['Q_out'][i]:12.3e} imb={eb['imbalance'][i]:12.3e} "
          f"rel={eb['rel_error'][i]:.4f} Tmax={Ts[i].max():.1f}")