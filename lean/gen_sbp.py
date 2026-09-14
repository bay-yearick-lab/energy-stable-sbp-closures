"""Exact rational certificates for the clamped B-spline coefficient forms.

Everything here is re-derived from the manuscript's own constructions with
`fractions.Fraction`, so the Lean literals it emits are exact.

Constructions (paper/main.tex, Section 3):
  knots      t_0..t_p = 0, t_N..t_{N+p} = 1, t_{p+j} = j/(N-p)   (main.tex 485-500)
  Greville   x_i = (1/p) sum_{m=1..p} t_{i+m}                     (main.tex 517-530)
  samples    B_ij = B_j(x_i)                                      (main.tex 560-572)
  mass       M_ij = int_0^1 B_i B_j dx                            (main.tex 577-582)
  advection  K_ij = int_0^1 B_i B_j' dx                           (main.tex 577-582)
  endpoint   B_d = -e_0 e_0^T + e_{N-1} e_{N-1}^T                 (main.tex 592-597)
  Marsden    x^k = sum_j [e_k(t_{j+1},..,t_{j+p})/C(p,k)] B_j(x)

The B-spline basis itself is built by the Cox-de Boor recursion in exact
rational piecewise-monomial form; no floating point appears anywhere.
"""
from fractions import Fraction as F
from itertools import combinations
from math import comb

# ---------------------------------------------------------------- polynomials
# A polynomial is a list of Fractions: p[d] is the coefficient of x^d (global x).

def padd(a, b):
    n = max(len(a), len(b))
    out = [F(0)] * n
    for d, c in enumerate(a):
        out[d] += c
    for d, c in enumerate(b):
        out[d] += c
    return out

def pscale_linear(a, c0, c1):
    """Multiply the polynomial `a` by the linear factor c0 + c1 * x."""
    out = [F(0)] * (len(a) + 1)
    for d, c in enumerate(a):
        out[d] += c0 * c
        out[d + 1] += c1 * c
    return out

def pmul(a, b):
    out = [F(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b):
                out[i + j] += x * y
    return out

def pderiv(a):
    return [F(d) * c for d, c in enumerate(a)][1:] or [F(0)]

def peval(a, x):
    return sum(c * x ** d for d, c in enumerate(a))

def pintegral(a, lo, hi):
    """Exact integral of the polynomial over [lo, hi]."""
    return sum(c / F(d + 1) * (hi ** (d + 1) - lo ** (d + 1)) for d, c in enumerate(a))

# ------------------------------------------------------------------ knots etc.

def knots(p, N):
    """Clamped open-uniform degree-p knot vector with N basis functions."""
    t = [F(0)] * (p + 1)
    t += [F(j, N - p) for j in range(1, N - p)]
    t += [F(1)] * (p + 1)
    assert len(t) == N + p + 1
    return t

def greville(p, N):
    t = knots(p, N)
    return [sum(t[i + m] for m in range(1, p + 1)) / F(p) for i in range(N)]

def spans(p, N):
    """Indices s of the nondegenerate knot spans [t_s, t_{s+1}]."""
    return list(range(p, N))

# ------------------------------------------------------- Cox-de Boor in exact Q

def basis_pieces(p, N):
    """B_j as a dict span index -> polynomial in global x, for j = 0..N-1."""
    t = knots(p, N)
    sp = spans(p, N)
    # degree 0
    cur = []
    for i in range(N + p):
        cur.append({s: [F(1)] for s in sp if s == i})
    for k in range(1, p + 1):
        nxt = []
        for i in range(N + p - k):
            piece = {}
            den1 = t[i + k] - t[i]
            if den1 != 0:
                for s, poly in cur[i].items():
                    piece[s] = padd(piece.get(s, [F(0)]),
                                    pscale_linear(poly, -t[i] / den1, F(1) / den1))
            den2 = t[i + k + 1] - t[i + 1]
            if den2 != 0:
                for s, poly in cur[i + 1].items():
                    piece[s] = padd(piece.get(s, [F(0)]),
                                    pscale_linear(poly, t[i + k + 1] / den2, F(-1) / den2))
            nxt.append(piece)
        cur = nxt
    return cur[:N]

def eval_basis(p, N, pieces, j, x):
    t = knots(p, N)
    sp = spans(p, N)
    # pick the span containing x; the right endpoint belongs to the last span
    for s in sp:
        if (t[s] <= x < t[s + 1]) or (s == sp[-1] and x == t[s + 1]):
            return peval(pieces[j].get(s, [F(0)]), x)
    raise ValueError(f"x = {x} outside [0, 1]")

# --------------------------------------------------------------- the matrices

def sample_matrix(p, N):
    pieces = basis_pieces(p, N)
    xs = greville(p, N)
    return [[eval_basis(p, N, pieces, j, xs[i]) for j in range(N)] for i in range(N)]

def mass_matrix(p, N):
    pieces = basis_pieces(p, N)
    t = knots(p, N)
    M = [[F(0)] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            tot = F(0)
            for s in spans(p, N):
                a = pieces[i].get(s)
                b = pieces[j].get(s)
                if a and b:
                    tot += pintegral(pmul(a, b), t[s], t[s + 1])
            M[i][j] = tot
    return M

def advection_matrix(p, N):
    pieces = basis_pieces(p, N)
    t = knots(p, N)
    K = [[F(0)] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            tot = F(0)
            for s in spans(p, N):
                a = pieces[i].get(s)
                b = pieces[j].get(s)
                if a and b:
                    tot += pintegral(pmul(a, pderiv(b)), t[s], t[s + 1])
            K[i][j] = tot
    return K

def endpoint_matrix(N):
    Bd = [[F(0)] * N for _ in range(N)]
    Bd[0][0] = F(-1)
    Bd[N - 1][N - 1] = F(1)
    return Bd

def marsden_alpha(p, N, k):
    """x^k = sum_j alpha_j B_j(x): alpha_j = e_k(t_{j+1},...,t_{j+p}) / C(p,k)."""
    t = knots(p, N)
    out = []
    for j in range(N):
        nodes = [t[j + m] for m in range(1, p + 1)]
        ek = sum((F(1) if not c else __import__("math").prod(c, start=F(1)))
                 for c in combinations(nodes, k)) if k > 0 else F(1)
        out.append(ek / F(comb(p, k)))
    return out

# ------------------------------------------------------------- linear algebra

def matmul(A, B):
    n, m, r = len(A), len(B[0]), len(B)
    return [[sum(A[i][k] * B[k][j] for k in range(r)) for j in range(m)] for i in range(n)]

def transpose(A):
    return [list(row) for row in zip(*A)]

def matadd(A, B):
    return [[a + b for a, b in zip(ra, rb)] for ra, rb in zip(A, B)]

def matvec(A, v):
    return [sum(a * x for a, x in zip(row, v)) for row in A]

def identity(n):
    return [[F(int(i == j)) for j in range(n)] for i in range(n)]

def inverse(M):
    n = len(M)
    A = [row[:] + r for row, r in zip(M, identity(n))]
    for c in range(n):
        piv_row = next(r for r in range(c, n) if A[r][c] != 0)
        A[c], A[piv_row] = A[piv_row], A[c]
        piv = A[c][c]
        A[c] = [x / piv for x in A[c]]
        for r in range(n):
            if r != c and A[r][c] != 0:
                f = A[r][c]
                A[r] = [x - f * y for x, y in zip(A[r], A[c])]
    return [row[n:] for row in A]

def ldl(M):
    """Exact LDL^T of a symmetric matrix: unit lower L, diagonal D."""
    n = len(M)
    L = identity(n)
    D = [F(0)] * n
    for j in range(n):
        D[j] = M[j][j] - sum(L[j][k] * L[j][k] * D[k] for k in range(j))
        assert D[j] != 0, "zero pivot"
        for i in range(j + 1, n):
            L[i][j] = (M[i][j] - sum(L[i][k] * L[j][k] * D[k] for k in range(j))) / D[j]
    return L, D

# --------------------------------------------------------------- Lean emitter

def q(x):
    return f"({x.numerator} : ℚ)" if x.denominator == 1 \
        else f"({x.numerator}/{x.denominator} : ℚ)"

def lean_matrix(name, size, M):
    body = ";\n    ".join(", ".join(q(x) for x in row) for row in M)
    return f"def {name} : Matrix (Fin {size}) (Fin {size}) ℚ :=\n  !![{body}]\n"

def lean_vector(name, size, v):
    body = ", ".join(q(x) for x in v)
    return f"def {name} : Fin {size} → ℚ := ![{body}]\n"

def lean_diag(name, size, d):
    body = ", ".join(q(x) for x in d)
    return f"def {name} : Fin {size} → ℚ := ![{body}]\n"

# ------------------------------------------------------------------ self-check

def verify(p, N):
    """Every identity Lean will certify is first checked here in Python."""
    Bm = sample_matrix(p, N)
    M = mass_matrix(p, N)
    K = advection_matrix(p, N)
    Bd = endpoint_matrix(N)
    t = knots(p, N)
    xs = greville(p, N)

    assert xs[0] == 0 and xs[N - 1] == 1
    for i in range(p - 1, N - p):
        assert xs[i + 1] - xs[i] == F(1, N - p), (i, xs[i + 1] - xs[i])
    # endpoint traces of the clamped basis
    assert Bm[0] == [F(int(j == 0)) for j in range(N)]
    assert Bm[N - 1] == [F(int(j == N - 1)) for j in range(N)]
    # partition of unity at the nodes
    assert all(sum(row) == 1 for row in Bm)
    # coefficient-space Green identity, both printed forms
    assert matadd(K, transpose(K)) == Bd
    assert matmul(matmul(transpose(Bm), Bd), Bm) == Bd
    # K 1 = 0
    assert matvec(K, [F(1)] * N) == [F(0)] * N
    # mass symmetry and an exact LDL^T with positive pivots
    assert M == transpose(M)
    L, D = ldl(M)
    assert all(d > 0 for d in D)
    Dm = [[D[i] if i == j else F(0) for j in range(N)] for i in range(N)]
    assert matmul(matmul(L, Dm), transpose(L)) == M
    # nonsingular sample matrix
    Binv = inverse(Bm)
    assert matmul(Bm, Binv) == identity(N)
    # polynomial reproduction through the Marsden coefficients
    for k in range(p + 1):
        a = marsden_alpha(p, N, k)
        assert matvec(Bm, a) == [x ** k for x in xs], k
    return dict(B=Bm, M=M, K=K, Bd=Bd, L=L, D=D, Binv=Binv, t=t, x=xs)

if __name__ == "__main__":
    for p, N in ((2, 6), (3, 8)):
        d = verify(p, N)
        print(f"p = {p}, N = {N}: all exact identities verified in Python")
