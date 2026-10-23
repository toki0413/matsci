import numpy as np

# Careful derivation.
#
# Cyclic tridiagonal A:
#   row 0:      b[0] x0 + c[0] x1 + ... + c[n-1] x_{n-1} = d0
#   row i:      a[i] x_{i-1} + b[i] x_i + c[i] x_{i+1} = d_i
#   row n-1:    a[n-1] x_{n-2} + b[n-1] x_{n-1} + a[0] x_0 = d_{n-1}
#
# So the corner entries are:  A[0, n-1] = c[n-1]  and  A[n-1, 0] = a[0].
#
# Define M = A with those two corners ZEROED and the diagonal corrected:
#   M[0,0]   = b[0]        (we will subtract the corner contribution differently)
#
# Standard approach (e.g. "cyclic reduction" / Press):
# Write A = M + u v^T  where M is tridiagonal with
#   M[0,0] = b[0] - c[n-1]   ... no.
#
# Let's use the textbook formulation directly:
#   Let M be the tridiagonal matrix with diag b, sub a (with a[0] replaced by 0),
#   super c (with c[n-1] replaced by 0).
#   Then A = M + c[n-1] * e_0 e_{n-1}^T + a[0] * e_{n-1} e_0^T.
#
# Check: (M + beta e_0 e_{n-1}^T)[0, n-1] = M[0,n-1] + beta = 0 + beta = beta = c[n-1]. OK.
#        (M + alpha e_{n-1} e_0^T)[n-1, 0] = M[n-1,0] + alpha = 0 + alpha = a[0]. OK.
#
# So A = M + U V^T with U = [beta e_0, alpha e_{n-1}], V = [e_{n-1}, e_0].
#   U = n x 2: col0 = beta*e_0, col1 = alpha*e_{n-1}
#   V = n x 2: col0 = e_{n-1}, col1 = e_0
#
# Sherman-Morrison-Woodbury:
#   A^{-1} = M^{-1} - M^{-1} U (I + V^T M^{-1} U)^{-1} V^T M^{-1}
#
# x = A^{-1} d = M^{-1} d - M^{-1} U (I + V^T M^{-1} U)^{-1} (V^T M^{-1} d)


def tri_solve(a, b, c, d):
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
    alpha = a[0]      # A[n-1, 0]
    beta = c[n - 1]   # A[0, n-1]

    aM = a.copy()
    cM = c.copy()
    bM = b.copy()
    aM[0] = 0.0
    cM[n - 1] = 0.0
    # M has the SAME diagonal as A (b), corners zeroed.
    # A = M + beta e_0 e_{n-1}^T + alpha e_{n-1} e_0^T

    x = tri_solve(aM, b, cM, d)  # M^{-1} d   (diag is b, not bM!)

    # M^{-1} U:  col0 = beta * M^{-1} e_0, col1 = alpha * M^{-1} e_{n-1}
    e0 = np.zeros(n, dtype=complex)
    e0[0] = 1.0
    en = np.zeros(n, dtype=complex)
    en[n - 1] = 1.0
    m_inv_e0 = tri_solve(aM, b, cM, e0)
    m_inv_en = tri_solve(aM, b, cM, en)

    Uinv = np.column_stack([beta * m_inv_e0, alpha * m_inv_en])

    # V^T M^{-1} d:  V = [e_{n-1}, e_0] -> [x[n-1], x[0]]
    w = np.array([x[n - 1], x[0]], dtype=complex)

    # V^T M^{-1} U  (2x2): row i of V^T times col j of M^{-1}U
    # V^T rows: e_{n-1}^T, e_0^T
    # col0 of M^{-1}U = beta*m_inv_e0, col1 = alpha*m_inv_en
    K = np.array(
        [
            [beta * m_inv_e0[n - 1], alpha * m_inv_en[n - 1]],
            [beta * m_inv_e0[0], alpha * m_inv_en[0]],
        ],
        dtype=complex,
    )
    S = np.eye(2, dtype=complex) + K
    lam = np.linalg.solve(S, w)
    return x - Uinv @ lam


rng = np.random.default_rng(0)
for n in (8, 16, 64, 256, 1024):
    a = rng.standard_normal(n) + 0j
    b = rng.standard_normal(n) + 5 + 0j
    c = rng.standard_normal(n) + 0j
    d = rng.standard_normal(n) + 0j
    Mfull = np.diag(b) + np.diag(a[1:], -1) + np.diag(c[:-1], 1)
    Mfull[0, n - 1] = c[n - 1]
    Mfull[n - 1, 0] = a[0]
    x1 = cyclic_thomas(a, b, c, d)
    x2 = np.linalg.solve(Mfull, d)
    print(f"n={n:5d}  max err = {np.max(np.abs(x1 - x2)):.3e}")