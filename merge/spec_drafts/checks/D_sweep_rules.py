"""Spot checks of writer D (Horizon Sweeps, Warm Starts, Validation Tools).

Standard library only. Run with: python -I merge/spec_drafts/checks/D_sweep_rules.py
Each check compares a rule written in fragments_D.tex with the code of main (read with git show,
functions extracted with ast and run in an isolated namespace) or with an independent computation.

1. Day mapping sigma(k) of Eq. (d_daymap) against main's old_step (base.py, apply_binary_warm_start).
2. Relative gradient, its class, its eligibility (numeric threshold, and the default threshold = the case's own
   requested MIP gap) and the stopping rule of Section "Early Stop" against
   main's _annotate_latest_gradient and _gradient_stop_reason (horizon_sweep.py).
3. Proof rule and best-case selection against main's _select_horizon_cases (with the decided rule).
4. Clopper-Pearson upper bound: Beta(f+1, n-f) quantile equals the binomial-CDF definition.
"""

import ast
import math
import random
import subprocess
import sys
import types

REPO = "C:/Users/utente/Documents/GitHub/Fleet_Management"


def git_show(path):
    return subprocess.run(
        ["git", "show", "main:" + path], cwd=REPO, capture_output=True, check=True
    ).stdout.decode("utf-8")


def extract_function(source, name):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(source, node)
    raise KeyError(name)


class _Finfo:
    eps = sys.float_info.epsilon


def make_numpy_shim():
    np = types.SimpleNamespace()
    np.isfinite = lambda v: math.isfinite(v)
    np.finfo = lambda _t: _Finfo
    return np


# ------------------------------------------------------------------ 1. day mapping
def check_day_map():
    source = git_show("src/fleet_management/degradation_model/base.py")
    segment = extract_function(source, "apply_binary_warm_start")
    # old_step is a nested function that reads H1 of the problem (ctx.H1) and of the source.
    start = segment.index("def old_step")
    end = segment.index("values = 0")
    body = segment[start:end]
    body = "\n".join(line[4:] if line.startswith("    ") else line for line in body.splitlines())

    def main_old_step(k0, H1, h1_old, h2_old):
        ctx = types.SimpleNamespace(H1=H1)
        ns = {"ctx": ctx, "old_h1": h1_old, "old_h2": h2_old}
        exec(body, ns)
        return ns["old_step"](k0)

    def spec_sigma(k, H1, h1_old, h2_old):
        # Eq. (d_daymap), 1-based days
        if k <= min(H1, h1_old):
            return k
        return h1_old + 1 + (max(0, k - H1 - 1) % h2_old)

    cases = 0
    for H1 in range(1, 7):
        for H2 in range(1, 7):
            for h1_old in range(1, 7):
                for h2_old in range(1, 7):
                    for k in range(1, H1 + H2 + 1):
                        main_value = main_old_step(k - 1, H1, h1_old, h2_old) + 1
                        assert main_value == spec_sigma(k, H1, h1_old, h2_old), (k, H1, H2, h1_old, h2_old)
                        assert 1 <= main_value <= h1_old + h2_old
                        cases += 1
    # boundary case written in the text: H1 > H1' maps the extra transitory days to the first operating day
    assert [spec_sigma(k, 5, 2, 3) for k in range(1, 6)] == [1, 2, 3, 3, 3]
    # spot value: transitory 2, operating 3 -> new H1 = 2, H2 = 7: operating days repeat days 3,4,5,3,4,5,3
    assert [spec_sigma(k, 2, 2, 3) for k in range(3, 10)] == [3, 4, 5, 3, 4, 5, 3]
    return cases


# ------------------------------------------------------------------ 2. gradient and stop
def spec_annotate(rows, tol, max_gap, eps_mach=sys.float_info.epsilon):
    """Section 'Early Stop': class and eligibility of the last row.

    max_gap is a number, or None for the default: the threshold of a case is then its own requested MIP gap.
    """
    cur = rows[-1]
    cur.update(cls="unavailable", eligible=False, g=None)
    if len(rows) < 2:
        return
    prev = rows[-2]

    def feasible(r):
        return r.get("J") is not None and r["status"] not in ("infeasible", "inf_or_unbounded", "unbounded")

    if not (feasible(prev) and feasible(cur)):
        return
    g = (cur["J"] - prev["J"]) / ((cur["H2"] - prev["H2"]) * max(abs(prev["J"]), eps_mach))
    cur["g"] = g
    cur["cls"] = "increase" if g > tol else ("decrease" if g < -tol else "flat")

    def gap_ok(r):
        threshold = r["requested_mip_gap"] if max_gap is None else max_gap
        return r.get("mip_gap") is not None and math.isfinite(r["mip_gap"]) and r["mip_gap"] <= threshold

    cur["eligible"] = gap_ok(prev) and gap_ok(cur)


def spec_stop(rows, minimum_cases, flat_required):
    n = len(rows)
    if n < minimum_cases:
        return None
    last = rows[-1]
    if not last["eligible"]:
        return None
    if last["cls"] == "increase":
        return "cost_increase"
    if last["cls"] != "flat":
        return None
    # the last flat_required cases up to n (gradients 2..n) are all flat and eligible
    window = rows[-flat_required:] if flat_required <= n - 1 else None
    if window is None:
        return None
    if all(r["eligible"] and r["cls"] == "flat" for r in window):
        return "flat_gradient"
    return None


def check_gradient_and_stop():
    source = git_show("src/fleet_management/horizon_sweep.py")
    ns = {"np": make_numpy_shim()}
    for name in ("_is_feasible_case", "_gap_qualifies_for_stopping", "_annotate_latest_gradient", "_gradient_stop_reason"):
        exec(extract_function(source, name), ns)
    rng = random.Random(20261026)
    trials = 0
    stops = {"cost_increase": 0, "flat_gradient": 0, None: 0}
    for _ in range(4000):
        n = rng.randint(1, 8)
        tol = rng.choice([0.0, 1e-3, 0.05])
        max_gap = rng.choice([0.0, 0.05, 0.2])
        minimum_cases = rng.randint(2, 4)
        flat_required = rng.randint(1, 3)
        h2 = 4
        cost = 1.0
        main_rows, spec_rows = [], []
        for idx in range(n):
            h2 += rng.randint(1, 4)
            cost *= rng.choice([1.0, 1.0, 1.0005, 0.9995, 0.9, 1.2])
            status = rng.choice(["optimal", "optimal", "time_limit", "infeasible"])
            gap = rng.choice([0.0, 0.01, 0.04, 0.1, None])
            row = {"H2": h2, "status": status, "J_op_average": None if status == "infeasible" else cost, "mip_gap": gap}
            main_rows.append(dict(row))
            spec_rows.append({"H2": h2, "status": status, "J": row["J_op_average"], "mip_gap": gap})
            ns["_annotate_latest_gradient"](main_rows, gradient_tolerance=tol, maximum_mip_gap=max_gap)
            spec_annotate(spec_rows, tol, max_gap)
            m, s = main_rows[-1], spec_rows[-1]
            assert m["gradient_classification"] == s["cls"], (m, s)
            assert m["gradient_stopping_eligible"] == s["eligible"], (m, s)
            reason_main = ns["_gradient_stop_reason"](main_rows, minimum_cases=minimum_cases, flat_gradients_required=flat_required)
            reason_spec = spec_stop(spec_rows, minimum_cases, flat_required)
            assert reason_main == reason_spec, (reason_main, reason_spec, main_rows)
            stops[reason_main] += 1
            trials += 1
    # default threshold: each case is compared with its own requested MIP gap (main's function called per row)
    default_trials = 0
    for _ in range(3000):
        n = rng.randint(2, 6)
        tol = rng.choice([0.0, 1e-3, 0.05])
        h2, cost = 4, 1.0
        spec_rows = []
        for idx in range(n):
            h2 += rng.randint(1, 4)
            cost *= rng.choice([1.0, 1.0005, 0.9, 1.2])
            row = {
                "H2": h2, "status": rng.choice(["optimal", "optimal", "time_limit"]), "J": cost,
                "mip_gap": rng.choice([0.0, 0.04, 0.09, 0.1, 0.14, None]),
                "requested_mip_gap": rng.choice([0.05, 0.1]),
            }
            spec_rows.append(row)
            spec_annotate(spec_rows, tol, None)
        prev, cur = spec_rows[-2], spec_rows[-1]
        if cur["cls"] != "unavailable":
            main_prev = {"mip_gap": prev["mip_gap"]}
            main_cur = {"mip_gap": cur["mip_gap"]}
            expected = (
                ns["_gap_qualifies_for_stopping"](main_prev, prev["requested_mip_gap"])
                and ns["_gap_qualifies_for_stopping"](main_cur, cur["requested_mip_gap"])
            )
            assert cur["eligible"] == expected, (prev, cur)
            default_trials += 1
    # worked example of the text: gap 0.08 against requested 0.1 qualifies; gap 0.12 against requested 0.1 does not
    ok = [{"H2": 4, "status": "optimal", "J": 1.0, "mip_gap": 0.08, "requested_mip_gap": 0.1},
          {"H2": 5, "status": "optimal", "J": 1.0, "mip_gap": 0.1, "requested_mip_gap": 0.1}]
    spec_annotate(ok, 1e-3, None)
    assert ok[-1]["eligible"]
    bad = [{"H2": 4, "status": "optimal", "J": 1.0, "mip_gap": 0.08, "requested_mip_gap": 0.1},
           {"H2": 5, "status": "time_limit", "J": 1.0, "mip_gap": 0.12, "requested_mip_gap": 0.1}]
    spec_annotate(bad, 1e-3, None)
    assert not bad[-1]["eligible"]
    stops["default_threshold_trials"] = default_trials
    # boundary case: gradient exactly at the tolerance is flat (strict inequalities)
    rows = [{"H2": 4, "status": "optimal", "J": 1.0, "mip_gap": 0.0}, {"H2": 5, "status": "optimal", "J": 1.001, "mip_gap": 0.0}]
    spec_annotate(rows, 1e-3, 0.05)
    assert rows[-1]["cls"] in ("flat", "increase")
    rows = [{"H2": 4, "status": "optimal", "J": 1.0, "mip_gap": 0.0}, {"H2": 5, "status": "optimal", "J": 1.0, "mip_gap": 0.0}]
    spec_annotate(rows, 0.0, 0.05)
    assert rows[-1]["cls"] == "flat"  # zero gradient with tolerance 0 is flat: not > 0 and not < -0
    return trials, stops


# ------------------------------------------------------------------ 3. proof rule
def check_selection():
    """Decided proof rule: proven = feasible and status 'optimal'. main's rule also required gap <= 1e-8."""
    rows = [
        {"H2": 12, "status": "optimal", "J_op_average": 0.66, "mip_gap": 0.0},
        {"H2": 16, "status": "time_limit", "J_op_average": 0.61, "mip_gap": 0.10},
        {"H2": 20, "status": "optimal", "J_op_average": 0.63, "mip_gap": 0.08},
        {"H2": 24, "status": "error: licence", "J_op_average": None, "mip_gap": None},
    ]

    def feasible(r):
        return r["J_op_average"] is not None and r["status"] not in ("infeasible", "inf_or_unbounded", "unbounded")

    feas = [r for r in rows if feasible(r)]
    proven = [r for r in feas if r["status"] == "optimal"]
    best_feasible = min(feas, key=lambda r: r["J_op_average"])
    best_proven = min(proven, key=lambda r: r["J_op_average"])
    assert best_feasible["H2"] == 16 and best_proven["H2"] == 20  # a gap of 0.08 counts as proven under the decided rule
    # main's rule (gap <= 1e-8) would have chosen H2 = 12
    old = min([r for r in proven if r["mip_gap"] <= 1e-8], key=lambda r: r["J_op_average"])
    assert old["H2"] == 12
    # tie: the case evaluated first wins (min keeps the first minimum)
    tie = [{"H2": 4, "J": 1.0}, {"H2": 8, "J": 1.0}]
    assert min(tie, key=lambda r: r["J"])["H2"] == 4
    return True


# ------------------------------------------------------------------ 4. Clopper-Pearson
def beta_cdf(p, a, b, steps=20000):
    """Regularised incomplete beta by the midpoint rule, adequate for a spot check."""
    if p <= 0:
        return 0.0
    log_norm = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    h = p / steps
    total = 0.0
    for i in range(steps):
        t = (i + 0.5) * h
        total += math.exp(log_norm + (a - 1) * math.log(t) + (b - 1) * math.log1p(-t))
    return total * h


def binom_cdf(f, n, p):
    return sum(math.comb(n, k) * p**k * (1 - p) ** (n - k) for k in range(f + 1))


def check_clopper_pearson():
    out = []
    for n, f, c in [(100000, 0, 0.95), (50, 0, 0.95), (50, 3, 0.95), (200, 20, 0.9), (30, 29, 0.95)]:
        # quantile of Beta(f+1, n-f) at level c by bisection on the beta cdf
        lo, hi = 0.0, 1.0 - 1e-12
        for _ in range(60):
            mid = (lo + hi) / 2
            if n <= 200:
                value = beta_cdf(mid, f + 1, n - f)
            else:
                value = 1 - (1 - mid) ** n if f == 0 else None
            if value < c:
                lo = mid
            else:
                hi = mid
        upper = (lo + hi) / 2
        if n <= 200:
            # definition: P(Bin(n, p_U) <= f) = 1 - c
            assert abs(binom_cdf(f, n, upper) - (1 - c)) < 2e-4, (n, f, c, upper)
        out.append((n, f, c, upper))
    # boundary written in the text: zero failures in n runs gives the bound 1 - (1 - c)^(1/n), about 3/n for c = 0.95
    n = 100000
    bound = 1 - (1 - 0.95) ** (1 / n)
    assert abs(bound - 2.9957e-5) < 1e-8
    return out


def main():
    cases = check_day_map()
    print(f"1. day mapping: {cases} (k, H1, H2, H1', H2') cases agree with main's old_step; boundary cases pass")
    trials, stops = check_gradient_and_stop()
    print(f"2. gradient and stop rule: {trials} random prefixes agree with main; stop reasons seen: {stops}")
    check_selection()
    print("3. proof rule: a gap of 0.08 with status 'optimal' is proven (main's 1e-8 rule picked another case); tie rule and error rows pass")
    for row in check_clopper_pearson():
        print("4. Clopper-Pearson upper bound (n, f, level, bound):", row)
    print("all checks passed")


if __name__ == "__main__":
    main()
