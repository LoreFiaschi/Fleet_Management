# Project Assessment — Stochastic Degradation-Aware Fleet Management

*Date: 2026-07-07 · Refers to: `scoping.tex` (post full-review round) and `spec/spec.tex` v0.5*

## Verdict

The project is strong, and the spec is implementation-grade. The scoping is close to a
camera-ready core (two sections still stubbed: related applications, novelty
positioning); the spec is ready to hand to an implementer.

## Strengths

- **Sound structural insight.** Scheduling decisions enter the degradation state through
  count-weighted cumulative parameters, keeping a stochastic reliability problem inside
  MILP territory.
- **Original repeatability theory.** The loop constraint must be imposed on
  self-propagating monotone statistics (mean + variance), not on the scalar risk bound —
  backed by explicit counterexamples on both the bound side and the exact-probability
  side. Publishable in its own right.
- **Distribution-free rainflow route.** Decouples the optimizer from parametric
  commitments with honest, quantified conservatism (Cantelli / Bernstein), plus a clear
  tightening path: schedule-dependent supports, multi-tangent PWL, random-N quantile
  route.
- **Verification discipline.** Every bound proven *and* numerically spot-checked;
  scripts archived in `agentic/workspace/verification/rainflow_bounds/`.

## Risks (decreasing order of concern)

1. **Spec–code drift.** The shipped code implements ~40% of the spec under different
   names/semantics. `rainflow.py`, `wiener.py`, `gamma.py`, replacement, and
   per-component models remain to be built; the code's Gaussian reliability constraint
   has no traceable derivation. The spec is the source of truth — pay this debt before
   it grows.
2. **Unproven scalability.** The exact form is a nonconvex MIQCP; no measurements yet of
   where F × L × 2H stops being solvable.
3. **Open-loop model.** The schedule is fixed at day 0 with no recourse as condition
   data arrives.

## Recommended additions (priority order)

1. **Rolling-horizon deployment** — re-solve with observed states as condition
   monitoring updates arrive; the loop constraint becomes a terminal safety envelope.
   Machinery already supports it (initial conditions are inputs). Cheap, high value.
2. **Scenario/SAA chance constraints** — sample from the empirical rainflow
   distribution, allow failure on at most ⌊εS⌋ scenarios (indicator constraints). Still
   MILP, probabilistic guarantees, and quantifies what the Cantelli/Bernstein
   conservatism actually costs. Compelling experiment for the paper.
3. **Distributionally robust framing** — Cantelli *is* the tight bound over the
   two-moment ambiguity set (Bertsimas–Popescu); saying so reframes the rainflow model
   as moment-DRO and sets up a Wasserstein-ball extension addressing the
   sampling-error open question. Mostly free.
4. **Decomposition for scale** — components couple only through assignment variables:
   textbook Benders / Dantzig–Wolfe structure (assignment master, per-component
   reliability subproblems).
5. **Policy baseline** — compare against simple condition-based threshold policies;
   reviewers will ask what the optimization buys.

## Single most valuable next step

Not more theory: build `rainflow.py` against spec v0.5 and get the first scalability
numbers.
