"""Ad-hoc diagnostic: CN spatial/temporal convergence and cost."""
import time
import numpy as np
from nlse import analytic as A
from nlse.operators import CrankNicolson, SSFM

print("=== CN spatial convergence (dt=1e-6 fixed, t=0.1) ===")
prev = None
for ell in range(4):
    Nf, dxf, dtt, xf, kf = A.level_params(ell)
    dtt = 1e-6
    cf = CrankNicolson(xf, dxf, dtt)
    pf = A.exact(xf, 0.0)
    nsteps = int(round(0.1 / dtt))
    t0 = time.perf_counter()
    for n in range(nsteps):
        pf = cf.step(pf)
    el = time.perf_counter() - t0
    e = A.l2_relative(pf, A.exact(xf, 0.1))
    order = np.log2(prev / e) if prev else float("nan")
    print("  ell=%d N=%6d dx=%.5f L2=%.6e order=%.2f  (%.1fs)"
          % (ell, Nf, dxf, e, order, el))
    prev = e

print()
print("=== CN temporal convergence (dx fixed at level 3, t=0.1) ===")
Nf, dxf, dtt0, xf, kf = A.level_params(3)
prev = None
for k in range(4):
    dtf = dtt0 / (1 << k)
    cf = CrankNicolson(xf, dxf, dtf)
    pf = A.exact(xf, 0.0)
    nsteps = int(round(0.1 / dtf))
    for n in range(nsteps):
        pf = cf.step(pf)
    e = A.l2_relative(pf, A.exact(xf, 0.1))
    order = np.log2(prev / e) if prev else float("nan")
    print("  k=%d dx=%.5f dt=%.2e L2=%.6e order=%.2f" % (k, dxf, dtf, e, order))
    prev = e

print()
print("=== SSFM convergence (t=1) ===")
prev = None
for ell in range(6):
    Nf, dxf, dtt, xf, kf = A.level_params(ell)
    s = SSFM(xf, kf, dxf, dtt)
    pf, _ = s.march(A.exact(xf, 0.0), int(round(1.0 / dtt)), record_every=0)
    e = A.l2_relative(pf, A.exact(xf, 1.0))
    order = np.log2(prev / e) if prev else float("nan")
    print("  ell=%d N=%6d dx=%.5f dt=%.2e L2=%.6e order=%.2f"
          % (ell, Nf, dxf, dtt, e, order))
    prev = e