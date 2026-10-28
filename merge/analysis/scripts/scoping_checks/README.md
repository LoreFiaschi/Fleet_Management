# Checks of numerical claims in `scoping.tex`

These scripts check claims of `scoping.tex` (repository root) in the Route-I part:
the tail experiment, the loop instance, and the two refinements of the Route-I constraints.
Each script is deterministic (no random numbers, or a fixed seed) and writes its result to
standard output. The `*_output.txt` files hold the output of the last run (28 Oct 2026).

Run from this folder:

```bash
python tail_experiment.py         > tail_experiment_output.txt          # about 30 s
python loop_routeI_instance.py    > loop_routeI_instance_output.txt     # about 30 s
python refine_schedule_support.py > refine_schedule_support_output.txt  # about 10 s
python refine_pwl_tightening.py   > refine_pwl_tightening_output.txt    # about 10 s
```

Environment of the last run: Python 3.10.11, numpy 2.2.6, scipy 1.15.3, mpmath 1.3.0,
sympy 1.14.0. No Gurobi is needed.

The subfolder `miner/` holds separate checks that this file does not describe.

## `tail_experiment.py`

**Claims checked.** Subsubsection "Conservative surrogates ...", paragraph "The choice among
the Route~I bounds depends on how far into the tail the constraint lies": the sentence "A
representative numerical experiment ...", the bullets Chernoff, Cantelli and Bernstein, and
their footnote.

**Method.** Sums of k in {30, 100} independent increments on [0, 1], with five laws: Uniform(0,1),
Beta(2,2), Beta(0.5,5), and the two-point laws on {0, 1} with probability 0.5 and 0.05.
The threshold tau gives an exact tail Pr(D > tau) close to eps, for eps in {1e-2, 1e-4, 1e-6, 1e-8}.
The exact tail comes from the Irwin-Hall formula (uniform), the binomial law (two-point; event
D >= tau), or a rigorous lattice bracket computed by FFT convolution (Beta laws; relative
half-width at most 0.4 %). The bounds are Markov, Cantelli, Hoeffding (width 1), Bernstein (b = 1)
and Chernoff (exact cumulant generating function, best s), plus Chernoff with the
mean-and-support bound of the cumulant generating function.

**Result at eps about 1e-4 (bound divided by the exact tail).**

| Bound | Range over the 10 cases |
|---|---|
| Chernoff, exact CGF, best s | 4.3 to 11.4 |
| Chernoff, mean and support only | 4.3 to 6.6 (two-point laws), 1000 to 3200 (continuous laws) |
| Bernstein, b = 1 | 15 to 557 |
| Cantelli | 334 to 707 |
| Hoeffding, width 1 | 7.8 to 6500 |
| Markov | 2200 to 8600 |

- The Chernoff and Cantelli claims reproduce.
- The Bernstein claim reproduces in part. Bernstein is tighter than Cantelli in 9 of the 10
  cases; it is looser for Beta(0.5,5) with k = 30. Its ratio falls as k grows. Its advantage
  over Cantelli grows with the depth of the tail: at eps about 1e-2 it is looser than Cantelli in
  4 of the 10 cases, at 1e-6 and 1e-8 it is tighter in all cases.
- The method check passes: for the uniform law, the lattice bracket contains the Irwin-Hall value
  at every depth; for the two-point laws, scipy and mpmath agree to 5e-15.

## `loop_routeI_instance.py`

**Claims checked.** Subsubsection "The loop constraint under the Route-I bounds"
(`sec:loop_routeI`): the instance "A concrete instance has tau = 10, b_l = 3, rho = 0.5 ..." with
its bullets Cantelli, Bernstein and Loop constraint; the sentence "A fixed margin delta > 0 ...
does not fix this"; Appendix `drv:loopI_no_chain` (instance with sum of variances 4).

**Result.**

- All numbers of the instance reproduce when the repair day is an ARD_inf repair (the text does
  not name the repair law): Cantelli U = 0.2000, 0.1997, 0.2151, limit 0.2495; Bernstein
  U = 0.3012, 0.3006, 0.3247, limit 0.3770; E[D_T] = 5 > 4. The appendix instance gives
  U(0) = U(1) = 0.2000 and U(2) = 0.2155.
- The moments are realisable by terms whose range has width at most 3 (explicit two-point laws).
- Margin claim, Cantelli under ARD_inf: a failing instance with U(1) <= U(0) - delta exists exactly
  when delta < U0 (1 - U0) / (2 - U0) (derived in the script header; 0.0889 at U0 = 0.2;
  instances reach 99.8 % of it; a seeded random search of 200000 samples finds none above it).
  Every delta > 0 rejects the steady state, U(1) = U(0), which the moment loop accepts.

## `refine_schedule_support.py`

**Claims checked.** Subsubsection "Two refinements of the Route-I constraints ..."
(`sec:routeI_refinements`), paragraph "Schedule-dependent Bernstein support.", and Appendix
`drv:sched_support`: the empty-schedule condition 2 kappa b_l/3 <= tau; the worked example
(kappa = 4.605, 0.4605, 0.079, 7 days, 0.0307, 0.939, 90 days, factor 12.9, 1.075 > tau);
the derivative 1 + c/sqrt(c^2 + 2 kappa v) in [1, 2] and the implication between rows; the
equivalence of the conditional rows with the single row at kappa b_l(zeta)/3; the side effects
(largest admissible variance decreasing in c; a mission with 2 c_j >= tau); the big-M constant
2 tau (tau - c_0) (not named in the footnote, checked as well).

**Result.** 17 checks, all PASS. At 2 c_j = tau exactly, the row admits only mu = v = 0.

## `refine_pwl_tightening.py`

**Claims checked.** Same subsubsection, paragraph "Piecewise-linear tightening of the tangent
surrogates.", Appendix `drv:pwl_recovery`, and the sentence "Neither refinement changes the loop
argument": the Cantelli equivalence; the tangent formula and the gap (mu - mu_r)^2; the loss
mu^2 of the tangent at 0; intersection inside every single tangent region inside the union inside
the exact region; the equality of the two encodings (half-plane identity); the vertex
sqrt(v_r v_{r+1}) with value (sqrt v_r + sqrt v_{r+1})/2; the two shortcuts that fail; the worst
gap (Delta/2)^2 and the factor 4; the table of admissible variances and the 94 % recovery; the
Bernstein version; the down-set property used by the loop.

**Result.** 26 checks, all PASS. The recovered fraction is exactly 15/16 = 93.75 %. The gap
formula min_r (mu - mu_r)^2 holds where the union admits v >= 0; beyond the last tangent point
plus Delta/2 the union admits no variance and the gap is (tau - mu)^2, still at most (Delta/2)^2.
