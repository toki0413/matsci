import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

t = 2.8
a = 2.46
pi = np.pi

Gamma = np.array([0.0, 0.0])
K = np.array([4*pi/(3*a), 0.0])
M = np.array([0.0, 2*pi/(np.sqrt(3)*a)])

hbar = 6.582119569e-16
e = 1.602176634e-19
h = 6.62607015e-34
eps0 = 8.854187817e-12
c = 299792458

def dispersion(kx, ky):
    f = 1 + np.exp(-1j*kx*a) + np.exp(-1j*(0.5*kx*a + np.sqrt(3)/2*ky*a))
    return t * np.abs(f)

def gen_kpath(points, n=200):
    kp = []
    for i in range(len(points)-1):
        p0, p1 = points[i], points[i+1]
        for j in range(n):
            kp.append(p0 + (j/n)*(p1-p0))
    return np.array(kp)

kp = gen_kpath([Gamma, K, M, Gamma], 300)
Ep = np.array([dispersion(k[0],k[1]) for k in kp])
Em = -Ep.copy()

dist = np.zeros(len(kp))
for i in range(1, len(kp)):
    dist[i] = dist[i-1] + np.linalg.norm(kp[i]-kp[i-1])

nps = 300
lp = [0, nps-1, 2*nps-1, 3*nps-1]
ln = ['G', 'K', 'M', 'G']

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

ax = axes[0]
ax.plot(dist, Ep, 'b-', lw=2, label='pi* (conduction)')
ax.plot(dist, Em, 'r-', lw=2, label='pi (valence)')
for p in lp:
    ax.axvline(dist[p], color='gray', ls='--', alpha=0.5)
ax.set_xticks([dist[p] for p in lp])
ax.set_xticklabels(ln)
ax.set_ylabel('Energy (eV)', fontsize=13)
ax.set_title('Graphene pi-band (Tight-Binding)', fontsize=13)
ax.legend(fontsize=11)
ax.set_xlim(0, dist[-1])
ax.set_ylim(-8, 8)
ax.grid(alpha=0.3)
ax.annotate('Dirac point\n(E=0, gap=0)', xy=(dist[nps], 0),
            xytext=(dist[nps]+0.5, 3),
            arrowprops=dict(arrowstyle='->', color='darkgreen'),
            fontsize=10, color='darkgreen')

ax2 = axes[1]
dkx = np.linspace(-0.3, 0.3, 60)
dky = np.linspace(-0.3, 0.3, 60)
KX, KY = np.meshgrid(dkx, dky)
vF = 1e6
dk_m = np.sqrt(KX**2 + KY**2) * 1e10
Ec = hbar * vF * dk_m
c1 = ax2.contourf(KX, KY, Ec, levels=20, cmap='viridis')
ax2.set_xlabel(r'Delta k_x (A^{-1})', fontsize=12)
ax2.set_ylabel(r'Delta k_y (A^{-1})', fontsize=12)
ax2.set_title('Dirac Cone: E = +/- hbar v_F |Delta k|', fontsize=13)
ax2.set_aspect('equal')
plt.colorbar(c1, ax=ax2, label='E (eV)')

ax3 = axes[2]
Er = np.linspace(-3, 3, 1000)
D = (2/pi)*np.abs(Er)/(hbar*vF*1e10)**2
D = D/np.max(D)
ax3.plot(Er, D, 'g-', lw=2)
ax3.set_xlabel('E (eV)', fontsize=12)
ax3.set_ylabel('DOS (arb. units)', fontsize=12)
ax3.set_title('Density of States: D(E) prop |E|', fontsize=13)
ax3.axvline(0, color='k', ls='--', alpha=0.3)
ax3.grid(alpha=0.3)
ax3.annotate('D(E) prop |E|\n(vanish at E_F)', xy=(0, 0.05),
             fontsize=10, ha='center', color='darkgreen')

plt.tight_layout()
fig.savefig('graphene_band_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print('Figure saved.')

vF_tb = (3*t*a)/(2*hbar)*1e-10
sigma_min = pi*e**2/(2*h)
mu = 1e5
n_cm2 = 1e12
sigma = n_cm2*1e4*e*mu*1e-4
l_mfp = h*sigma/(2*e**2)*1/np.sqrt(pi*n_cm2*1e4)*1e9
alpha_g = e**2/(4*pi*eps0*hbar*vF_tb)

print('='*60)
print('Graphene Band Structure & Transport Analysis')
print('='*60)
print()
print(f'1. Lattice: a={a}A, bond={a/np.sqrt(3):.3f}A, t={t}eV')
print()
print(f'2. Band: vF={vF_tb:.2e}m/s = {vF_tb/c*100:.1f}%c, m*=0, Eg=0')
print()
print(f'3. Transport:')
print(f'   sigma_min = {sigma_min:.2e}S')
print(f'   mu = {mu} cm2/Vs')
print(f'   sigma = {sigma:.2f}S')
print(f'   l_mfp = {l_mfp:.0f}nm')
print()
print(f'4. Coupling: alpha_g = {alpha_g:.2f} (QED: 1/137 = {1/137.036:.4f})')
print(f'   alpha_g/alpha_QED = {alpha_g/(1/137.036):.1f}x')
print()
print('5. Quantum Transport Phenomena:')
print('   - Anomalous half-integer QHE: sigma_xy = +/- (4e^2/h)(n+1/2)')
print('   - Klein tunneling: T->1 (chiral barrier penetration)')
print('   - Ballistic transport at sub-micron scale')
print('   - Suppressed backscattering (chirality protection)')
print()
print('6. Low-energy theory:')
print('   H = hbar v_F sigma.k')
print('   E(k) = +/- hbar v_F |k|')
print('   m* = 0 (massless Dirac fermions)')
