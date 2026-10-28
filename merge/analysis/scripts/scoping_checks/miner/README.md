# Checks of the random fatigue life in the Palmgren-Miner rule

Scope: `scoping.tex`, subsubsection "Random fatigue life in the Palmgren--Miner rule", and its
Remark `rem:miner_dichotomy` (checked 28 Oct 2026).
Each script prints its results; the file `<script>_output.txt` beside it holds the output of
the run on this date. All random generators have fixed seeds, so a new run gives the same output.

Run, from this folder, with `python` (numpy, scipy, mpmath):

```
python miner_a_inverse_moments.py > miner_a_inverse_moments_output.txt   # about 50 s
python miner_b_schedule_loop.py  > miner_b_schedule_loop_output.txt     # about 45 s
python miner_c_dichotomy.py      > miner_c_dichotomy_output.txt         # about 8 min
```

## `miner_a_inverse_moments.py`

| Claim in `scoping.tex` | Check | Result |
|---|---|---|
| Closed forms `eq:miner_invmom` of E[N_f^-r] for the log-normal, Weibull and Gumbel-minimum laws | quadrature at 30 digits (mpmath) for 16 parameter sets | agree; worst relative error 3.8e-16 |
| Existence: Weibull finite if and only if r < k; Gumbel-minimum if and only if r beta < 1 | truncated moments as the cut-off goes to the origin | converge inside the ranges; grow without bound at and beyond the limits (at k = r = 2 the truncated moment equals E1(delta^2)) |
| The three lines of Var[1/N_f] | E[N^-2] - E[N^-1]^2 by quadrature | agree; worst relative error 8.5e-16 |
| Remark `rem:miner_tails`: a Gumbel-minimum log-life is Wei(1/beta, e^mu) | survival functions on 41 points | agree to 2.2e-16 |
| Quantiles q_eps of A (log-normal, Weibull, Gumbel) | closed form against scipy's quantile function, and F_A(q_eps) = eps | agree to 8.5e-16 |
| Divergent sample variance of 1/N_f at Weibull k <= 2 | median over 10 seeds of the sample variance, n = 1e3 to 1e7 | k = 1.5: 51 to 1463; k = 2: 4.4 to 17.2 (no limit); k = 3 and k = 5 approach the closed forms 0.8453 and 0.1338 |

## `miner_b_schedule_loop.py`

Interpretation (II), unit-specific scatter, D_k = M_k / A. Three laws of A: log-normal
(sigma = 0.5), Weibull (k = 2.5), Gumbel-minimum (beta = 0.25). Three missions with one stress
level each (Basquin, C = 1e12, m = 3), ARD repair with rho = 0.5, replacement with a new A.
Two schedules of 24 days (transitory 12, operating 12), one with repair and replacement in the
operating phase, one with repair only, each followed by four repetitions of the operating phase.

| Claim in `scoping.tex` | Check | Result |
|---|---|---|
| `eq:miner_quantile`: P(D_k >= 1) = F_A(M_k), exact along a schedule with repair and replacement | Monte Carlo of the physical damage (1e6 units per law and schedule, 72 days each) against F_A(M_k) | max abs(z) = 2.61 over 432 day-tests; no failure on the days with M_k = 0 |
| F_A(M_k) <= eps if and only if M_k <= q_eps | every day, three laws, eps = 0.01, 0.05, 0.1 | 0 mismatches |
| Rewriting as sum_j n_ij n_j / (q_eps Nbar_j) <= 1, without repair since the last replacement | count sum against M_k | equal (relative difference 1.4e-16) on the 52 days without repair; larger than M_k on all 92 days after a repair, so the condition "without repair" is needed |
| q_eps Nbar_j is the eps-quantile of N_f,j | empirical quantile of 1e6 lives | agree within the sampling error |
| `eq:loop_prob` is equivalent to `eq:miner_loop_det`, M_T <= M_H1 | 20020 random pairs, with ties and zeros | 0 mismatches for each law |
| `eq:miner_loop_det` chains | 5000 random operating phases with M_T <= M_H1 and M_k <= q on the horizon, 200 repetitions each | 0 days above the horizon state, 0 end states above the previous one, 0 states above q |
| The condition is not void | 3802 random phases with M_k <= q on the horizon but M_T > M_H1 | 1716 of them exceed q in a later repetition |
| Reliability on every repetition (Monte Carlo) | frequency on the four repetitions against eps met on the horizon | below it on every day, for both schedules and all three laws |

## `miner_c_dichotomy.py`

Reading of the experiment of Remark `rem:miner_dichotomy` ("log-normal factor, sigma = 0.5,
fifty days, median Miner sum 0.6"): equal daily increments m_t = 0.6 / 50; the deterministic
Miner sum M = 0.6 is the median of D = M / A under (II); under (I) each day draws its own factor
with the same law, D = sum_t m_t / A_t. Failure means D >= 1.

P(I) is computed by a discretised convolution on N cells of [0, 1) (FFT). Rounding each daily
increment down, or up, to the grid gives a sum below, or above, the true one, so the two
versions bound P(I) from below and from above. A conditional Monte Carlo estimator (the largest
daily term integrated exactly, the others sampled with a mean shift) checks the result
independently.

| Claim in `scoping.tex` | Check | Result |
|---|---|---|
| The means of (I) and (II) coincide; Var_II / Var_I = (sum m_t)^2 / sum m_t^2 | Monte Carlo (4e5 samples) for four profiles, log-normal and Weibull factors | agree within the sampling error |
| The ratio is at most the number of days, with equality for equal increments | 100000 random profiles | max ratio / days = 0.536; equal profile gives exactly 50 |
| P(II) = 0.15 in the experiment | Phi(ln 0.6 / 0.5) | 0.1535 |
| P(I) ~ 1e-6 in the experiment | convolution bracket at N = 2^21; conditional Monte Carlo, 3 shifts, 4e6 samples each | P(I) in [2.4539e-7, 2.4576e-7]; the three 95 % intervals overlap the bracket. So P(I) = 2.46e-7, not about 1e-6 |
| (same) | other readings | median of D under (I) = 0.6: P(I) = 9.7e-11; mean of D = 0.6: P(II) = 0.10, P(I) = 8.1e-11; sigma on log10 N_f: P(II) = 0.33. Values near 1e-6 need fewer effective days: equal increments on 40 to 45 days (3.1e-6 to 8.7e-7), or m_t proportional to t^0.42 (45.9 effective days). Random profiles give 7e-5 to 1e-3. A plain Monte Carlo of 1e6 samples shows one failure (printed 1e-6) with probability 0.19 |
| (same) | importance sampling with one uniform mean shift of all fifty normals | 2.31e-7, relative standard error 5 %, an unreliable estimator here: the event is reached most cheaply through one large daily factor, which a uniform shift rarely samples |
| "At high accumulated damage the order reverses" | P(I) and P(II) as functions of M, equal increments, 50 days, sigma = 0.5 | single crossing at M = 0.8663, where P(I) = P(II) = 0.387; P(I) < P(II) for M in [0.1, 0.862], P(I) > P(II) for M in [0.872, 3] |

Numerical accuracy: the FFT and a direct convolution differ by at most 3.7e-15 in the
cumulative distribution (N = 2^12); the grid check of the crossing allows 1e-12.
