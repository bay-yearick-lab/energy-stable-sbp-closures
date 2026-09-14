import Mathlib

/-!
# Endpoint traces of the clamped basis on the first knot span

`paper/main.tex` lines 505-516 records the rank-two endpoint trace of the clamped
open-uniform basis,
`B_0(0) = 1`, `B_{N-1}(1) = 1`, `B_j(0) = 0` for `j > 0`, `B_j(1) = 0` for `j < N - 1`,
and attributes it to the repeated endpoint knots, which make the clamped basis coincide
on the first knot span `[0, h]` with the degree-`p` Bernstein basis of that span; in
particular `B_0` restricted to `[0, h]` is `(1 - x/h)^p`.

**What is taken as a definition here.** The identification of the clamped degree-`p`
B-splines on the first knot span with the Bernstein polynomials of that span is the
classical consequence of the `(p+1)`-fold knot repetition at `0`; it is not re-derived
from the Cox-de Boor recursion in this file. It enters as the definition
`clampedFirstSpan`, exactly as the periodic project takes the cardinal B-spline by its
truncated-power closed form. Everything below the definition is proved.

The concrete finite cases in `SBP26.lean` and `SBP38.lean` do not rest on this file: there
the sample matrices are generated from the Cox-de Boor recursion in exact rational
arithmetic and the endpoint rows are read off the certified matrices themselves.
-/

namespace BsplineBoundary

open Finset

/-- The degree-`p` Bernstein basis on `[0,1]`: `b_{p,j}(x) = C(p,j) x^j (1-x)^{p-j}`. -/
def bern (p j : ℕ) (x : ℚ) : ℚ := (p.choose j : ℚ) * x ^ j * (1 - x) ^ (p - j)

/-- The clamped degree-`p` basis restricted to the first knot span `[0, h]`, taken in its
classical Bernstein form `B_j(x) = b_{p,j}(x/h)`. -/
def clampedFirstSpan (p : ℕ) (h : ℚ) (j : ℕ) (x : ℚ) : ℚ := bern p j (x / h)

/-! ### The Bernstein basis -/

/-- The first Bernstein polynomial is the pure power `(1-x)^p`. -/
theorem bern_zero (p : ℕ) (x : ℚ) : bern p 0 x = (1 - x) ^ p := by
  simp [bern]

/-- The last Bernstein polynomial is the pure power `x^p`. -/
theorem bern_top (p : ℕ) (x : ℚ) : bern p p x = x ^ p := by
  simp [bern]

theorem bern_at_zero_zero (p : ℕ) : bern p 0 0 = 1 := by
  simp [bern]

theorem bern_at_zero_pos {p j : ℕ} (hj : 0 < j) : bern p j 0 = 0 := by
  simp [bern, zero_pow hj.ne']

theorem bern_at_one_top (p : ℕ) : bern p p 1 = 1 := by
  simp [bern]

theorem bern_at_one_lt {p j : ℕ} (hj : j < p) : bern p j 1 = 0 := by
  have : p - j ≠ 0 := by omega
  simp [bern, zero_pow this]

/-- Partition of unity for the Bernstein basis, from the binomial theorem. -/
theorem bern_sum (p : ℕ) (x : ℚ) : ∑ j ∈ range (p + 1), bern p j x = 1 := by
  have hb := add_pow x (1 - x) p
  have h1 : x + (1 - x) = 1 := by ring
  rw [h1, one_pow] at hb
  rw [hb]
  exact Finset.sum_congr rfl fun j _ => by rw [bern]; ring

/-! ### Endpoint traces of the clamped basis on the first span -/

/-- `B_0` on the first span is `(1 - x/h)^p` (`paper/main.tex`, line 510-513). -/
theorem clampedFirstSpan_zero (p : ℕ) (h x : ℚ) :
    clampedFirstSpan p h 0 x = (1 - x / h) ^ p := by
  simp [clampedFirstSpan, bern_zero]

/-- `B_0(0) = 1`. -/
theorem clampedFirstSpan_at_wall_zero (p : ℕ) (h : ℚ) :
    clampedFirstSpan p h 0 0 = 1 := by
  simp [clampedFirstSpan, bern_at_zero_zero]

/-- `B_j(0) = 0` for `j > 0`: only the first clamped basis function is alive at the wall
(`paper/main.tex`, line 514-516). -/
theorem clampedFirstSpan_at_wall_pos {p j : ℕ} (hj : 0 < j) (h : ℚ) :
    clampedFirstSpan p h j 0 = 0 := by
  simp [clampedFirstSpan, bern_at_zero_pos hj]

/-- At the far end `x = h` of the first span only `B_p` survives, which is the mirror
statement that closes the span. -/
theorem clampedFirstSpan_at_span_end_top {p : ℕ} {h : ℚ} (hh : h ≠ 0) :
    clampedFirstSpan p h p h = 1 := by
  simp [clampedFirstSpan, div_self hh, bern_at_one_top]

theorem clampedFirstSpan_at_span_end_lt {p j : ℕ} {h : ℚ} (hh : h ≠ 0) (hj : j < p) :
    clampedFirstSpan p h j h = 0 := by
  simp [clampedFirstSpan, div_self hh, bern_at_one_lt hj]

/-- Partition of unity on the first span. -/
theorem clampedFirstSpan_sum (p : ℕ) (h x : ℚ) :
    ∑ j ∈ range (p + 1), clampedFirstSpan p h j x = 1 := by
  simpa [clampedFirstSpan] using bern_sum p (x / h)

/-- The rank-two endpoint trace at the left wall, in the form used by the SBP identity:
the wall row of the sampled basis is the first standard basis covector. -/
theorem clampedFirstSpan_wall_row (p j : ℕ) (h : ℚ) :
    clampedFirstSpan p h j 0 = if j = 0 then 1 else 0 := by
  rcases Nat.eq_zero_or_pos j with hj | hj
  · subst hj; simp [clampedFirstSpan_at_wall_zero]
  · simp [clampedFirstSpan_at_wall_pos hj, hj.ne']

end BsplineBoundary
