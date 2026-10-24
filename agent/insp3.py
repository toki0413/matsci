import numpy as np

# Correct cyclic tridiagonal via Sherman-Morrison.
#
# A = M + alpha * e_n e_1^T + beta * e_1 e_n^T
# where M is the NON-cyclic tridiagonal matrix with
#    M[0,0]   = b[0] - alpha      (removing the corner a[0]=alpha at row 0 col n-1)
#    M[n-1,n-1] = b[n-1] - beta   (removing the corner c[n-1]=beta at row n-1 col 0)
#
# Note: a[0] = M[n-1, 0] is the sub-diagonal corner (row n-1, col 0)
#       c[n-1] = M[0, n-1] is the super-diagonal corner (row 0, col n-1)
#
# So A = M + alpha * (e_{n-1} e_0^T) + beta * (e_0 e_{n-1}^T)
#        = M + u1 v1^T + u2 v2^T
# with u1 = alpha*e_{n-1}, v1 = e_0 ;  u2 = beta*e_0, v2 = e_{n-1}
#
# Sherman-Morrison-Woodbury: (M + U V^T)^{-1} = M^{-1} - M^{-1} U (I + V^T M^{-1} U)^{-1} V^T M^{-1}
# U = [u1, u2] (n x 2), V = [v1, v2] (n x 2)
# x = M^{-1} d - M^{-1} U (I + V^T M^{-1} U)^{-1} V^T M^{-1} d

def tri_solve(a, b, c, d):
    """Non-cyclic tridiagonal solve (Thomas). a[0] and c[n-1] ignored."""
    n = len(b)
    dd = np.array(d, dtype=complex, copy=True)
    gam = np.empty(n, dtype=complex)
    bet = b[0]
    gam[0] = c[0] / bet
    dd[0] = dd[0] / bet
    for j in range(1, n):
        bet = b[j] - a[j] * gam[j - 1]
        gam[j] = c[j] / bet
        dd[j] = (dd[j] - a[j] * dd[j - 1]) / bet
    x = np.empty(n, dtype=complex)
    x[n - 1] = dd[n - 1]
    for j in range(n - 2, -1, -1):
        x[j] = dd[j] - gam[j] * x[j + 1]
    return x


def cyclic_thomas(a, b, c, d):
    n = len(b)
    alpha = a[0]
    beta = c[n - 1]

    aM = a.copy()
    cM = c.copy()
    bM = b.copy()
    aM[0] = 0.0
    cM[n - 1] = 0.0
    bM[0] = b[0] - alpha
    bM[n - 1] = b[n - 1] - beta

    x = tri_solve(aM, bM, cM, d)  # M^{-1} d

    # M^{-1} U columns: solve M u1 = e_{n-1}, M u2 = e_0
    e_last = np.zeros(n, dtype=complex)
    e_last[n - 1] = 1.0
    u1 = tri_solve(aM, bM, cM, e_last)
    e_first = np.zeros(n, dtype=complex)
    e_first[0] = 1.0
    u2 = tri_solve(aM, bM, cM, e_first)

    # V^T M^{-1} d  -> [v1^T x, v2^T x] = [x[0], x[n-1]]
    w = np.array([x[0], x[n - 1]], dtype=complex)

    # V^T M^{-1} U  -> 2x2 matrix
    # v1^T u1 = u1[0],  v1^T u2 = u2[0]
    # v2^T u1 = u1[n-1], v2^T u2 = u2[n-1]
    K = np.array(
        [[u1[0], u2[0]], [u1[n - 1], u2[n - 1]]], dtype=complex
    )
    # I + V^T M^{-1} U, but scaled by alpha, beta:
    # U = [alpha*u1, beta*u2], so V^T M^{-1} U = [[alpha*u1[0], beta*u2[0]], [alpha*u1[n-1], beta*u2[n-1]]]
    S = np.array(
        [[1.0 + alpha * u1[0], beta * u2[0]], [alpha * u1[n - 1], 1.0 + beta * u2[n - 1]]],
        dtype=complex,
    )
    lam = np.linalg.solve(S, w)
    return x - lam[0] * alpha * u1 - lam[1] * beta * u2


rng = np.random.default_rng(0)
for n in (8, 16, 64, 256, 1024):
    a = rng.standard_normal(n) + 0j
    b = rng.standard_normal(n) + 5 + 0j
    c = rng.standard_normal(n) + 0j
    d = rng.standard_normal(n) + 0j
    M = np.diag(b) + np.diag(a[1:], -1) + np.diag(c[:-1], 1)
    M[0, n - 1] = c[n - 1]
    M[n - 1, 0] = a[0]
    x1 = cyclic_thomas(a, b, c, d)
    x2 = np.linalg.solve(M, d)
    print(f"n={n:5d}  max err = {np.max(np.abs(x1 - x2)):.3e}")