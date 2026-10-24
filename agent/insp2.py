import sympy as sp

A, B, C, v, Omega = sp.symbols("A B C v Omega", real=True)
x, t_ = sp.symbols("x t", real=True)
y = B * (x - v * t_)
u = A * sp.sech(y) * sp.exp(sp.I * (C * x - Omega * t_))

# residual / u
r = sp.I * sp.diff(u, t_) + sp.diff(u, x, 2) + 2 * u**2 * sp.conjugate(u)
r = sp.simplify(sp.expand_complex(r / u))
print("residual/u =", r)

# substitute sech^2 = 1 - tanh^2
T = sp.Symbol("T")
r2 = sp.expand(r.subs(sp.sech(y) ** 2, 1 - T ** 2))
print("collected:", sp.collect(r2, T))
print()
print("constant term (T=0):", sp.simplify(r2.subs(T, 0)))
print("coeff of T:", sp.simplify(sp.diff(r2, T).subs(T, 0)))
print()
print("=> solve:  v = 2C,  Omega = C^2 - B^2 + 2A^2?")
print("   constant term = Omega - B^2 - C^2 + 2A^2 = 0  -> Omega = C^2 + B^2 - 2A^2")
print("   with B=A: Omega = C^2 - A^2   (matches standard result)")