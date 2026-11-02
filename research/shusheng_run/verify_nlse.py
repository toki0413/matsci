import numpy as np

# i psi_t + 0.5 psi_xx + |psi|^2 psi = 0
L, N = 80.0, 2048
x = np.linspace(-L/2, L/2, N, endpoint=False)
dx = L/N
k = 2*np.pi*np.fft.fftfreq(N, d=dx)

def analytic(x, t, v=0.5, phase_coef=None):
    # 候选相位系数 c: exp(i[v x + c t])
    c = (0.5 - 0.5*v*v) if phase_coef is None else phase_coef
    return (1.0/np.cosh(x - v*t)) * np.exp(1j*(v*x + c*t))

def residual(v=0.5):
    # 数值求导检查两种相位系数哪个满足 PDE (t=1, x=0 附近最大残差)
    out = {}
    for c in [0.5-0.5*v*v, 0.5-0.25*v*v, 1.0-0.5*v*v]:
        t = 1.0
        h = 1e-5
        psi = analytic(x, t, v, c)
        psi_t = (analytic(x, t+h, v, c) - analytic(x, t-h, v, c))/(2*h)
        psi_xx = (analytic(x+h, t, v, c) - 2*psi + analytic(x-h, t, v, c))/h**2
        res = 1j*psi_t + 0.5*psi_xx + np.abs(psi)**2*psi
        out[c] = float(np.max(np.abs(res)))
    return out

def ssfm(dt=1e-3, tmax=10.0, v=0.5, N=N, L=L):
    xx = np.linspace(-L/2, L/2, N, endpoint=False)
    kk = 2*np.pi*np.fft.fftfreq(N, d=L/N)
    psi = analytic(xx, 0.0, v)
    lin = np.exp(-0.5j*kk*kk*dt)
    nsteps = int(round(tmax/dt))
    out = {}
    snapshots = {}
    def drifts(p):
        m = np.sum(np.abs(p)**2)*dx
        pk = np.fft.fft(p)
        px = np.fft.ifft(1j*kk*pk)
        E = np.sum(0.5*np.abs(px)**2 - 0.5*np.abs(p)**4)*dx
        P = np.sum(np.imag(np.conj(p)*px))*dx
        return m, E, P
    m0,E0,P0 = drifts(psi)
    for n in range(1, nsteps+1):
        psi = np.fft.ifft(lin*np.fft.fft(psi))
        psi = psi*np.exp(1j*np.abs(psi)**2*dt)
        t = n*dt
        if n % int(round(2.5/dt)) == 0 or n == nsteps:
            tgt = analytic(xx, t, v)
            l2 = np.sqrt(np.sum(np.abs(psi-tgt)**2)*dx)
            m,E,P = drifts(psi)
            snapshots[round(t,6)] = dict(L2=l2, dM=abs(m-m0)/m0, dE=abs(E-E0)/abs(E0), dP=abs(P-P0)/abs(P0) if abs(P0)>0 else float('nan'))
    return snapshots

print("=== 解析解残差（判断相位系数）===")
for c,r in residual().items():
    print(f"c={c:.4f}  max|residual|={r:.3e}")

print("\n=== 我的 SSFM 参考实现 (dt=1e-3, N=2048, L=80, v=0.5) ===")
s = ssfm(1e-3)
for t,d in s.items():
    print(f"t={t:5.1f}  L2={d['L2']:.3e}  dM={d['dM']:.1e}  dE={d['dE']:.1e}  dP={d['dP']:.1e}")

print("\n=== 时间步收敛（t=10）===")
for dt in [1e-3, 5e-4, 2.5e-4]:
    s = ssfm(dt)
    d = s[max(s)]
    print(f"dt={dt:g}  L2(t=10)={d['L2']:.3e}  dE={d['dE']:.1e}")
