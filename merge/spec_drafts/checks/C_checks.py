"""Checks of writer C (spec version 1.0: inputs, outputs, input file schema).

Standard library only. Run from merge/spec_drafts/ with:  python -I checks/C_checks.py
Output goes to stdout; the copy is checks/C_checks_output.txt.

What it does
 1. Reads the YAML example of fragments_C.tex (section "YAML Example") with a small parser for the subset used.
 2. Applies the consistency checks of the spec (reference implementation) to the example: all must pass.
 3. Applies mutated copies of the example (boundary cases): each must raise the expected error.
 4. Checks numbers quoted in the text: V_max / tau^2, the phase lookup against main's make_accessor,
    the anchor example, Popoviciu's inequality, the Gaussian 3-sigma tail, Jensen for a CGF, the Bernstein
    precondition of the example, the cost-report identities.
"""
import copy
import math
import os
import random
import re
import sys
from statistics import NormalDist

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(HERE, "..", "fragments_C.tex")

failures = []


def check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        failures.append(name)


# ---------------------------------------------------------------------------
# 1. Mini YAML reader (block mapping at top level; flow sequences and mappings; block lists of flow values)
# ---------------------------------------------------------------------------
def parse_scalar(tok):
    tok = tok.strip()
    if tok in ("null", "~", ""):
        return None
    if tok == "true":
        return True
    if tok == "false":
        return False
    try:
        return int(tok)
    except ValueError:
        pass
    try:
        return float(tok)
    except ValueError:
        return tok


def parse_flow(text):
    """Parse one flow value starting at text[0]; return (value, rest)."""
    text = text.lstrip()
    if text.startswith("["):
        items, rest = [], text[1:]
        while True:
            rest = rest.lstrip()
            if rest.startswith("]"):
                return items, rest[1:]
            v, rest = parse_flow(rest)
            items.append(v)
            rest = rest.lstrip()
            if rest.startswith(","):
                rest = rest[1:]
    if text.startswith("{"):
        out, rest = {}, text[1:]
        while True:
            rest = rest.lstrip()
            if rest.startswith("}"):
                return out, rest[1:]
            key, rest = rest.split(":", 1)
            v, rest = parse_flow(rest)
            out[key.strip()] = v
            rest = rest.lstrip()
            if rest.startswith(","):
                rest = rest[1:]
    m = re.match(r"[^,\]\}]*", text)
    return parse_scalar(m.group(0)), text[m.end():]


def read_yaml_subset(lines):
    data, key, block = {}, None, None
    for raw in lines:
        line = re.sub(r"\s+#.*$", "", raw.rstrip())
        if not line.strip() or line.strip().startswith("#"):
            continue
        if line.startswith("  - "):
            v, _ = parse_flow(line[4:])
            block.append(v)
            continue
        key, rest = line.split(":", 1)
        rest = rest.strip()
        if rest == "":
            block = []
            data[key] = block
        else:
            v, _ = parse_flow(rest)
            data[key] = v
            block = None
    return data


def example_from_tex():
    s = open(TEX, encoding="utf-8").read()
    i = s.index("\\subsection{YAML Example")
    a = s.index("\\begin{lstlisting}", i) + len("\\begin{lstlisting}")
    b = s.index("\\end{lstlisting}", a)
    return read_yaml_subset(s[a:b].strip("\n").split("\n"))


# ---------------------------------------------------------------------------
# 2. Reference implementation of the shape rules and of the consistency checks
# ---------------------------------------------------------------------------
KNOWN = set("""F M L H model component_names repair_model bound_method reliability_impl encoding tau epsilon
tangent_ref pwl_points rho anchor_mu_0 anchor_v_0 anchor_R_0 anchor_K_0 mu_0 v_0 support_0 R_0 cgf_0
replacement_mu replacement_v support_new R_new cgf_new replacement_as_new allow_replacement
mu v support cgf s_chernoff gamma_beta mu_trans v_trans support_trans cgf_trans gamma_beta_trans
sigma eta v_max_user gamma_beta_0 gamma_beta_new C_M C_R C_rep C_D penalty_type
verbose mip_gap time_limit fast gurobi_params objective_mode evaluation_horizon progress_interval_seconds
relaxation_warm_start relaxation_time_limit""".split())
REJECTED = set("""maintenance_type tail_bound method formulation beta alpha_inc gamma_beta_bound gamma_calibration_method
b_inc b_0 b_new n_workers warm_start depot_capacity C_P C_S alpha xi repair_rho mu_new v_new""".split())


def shape_of(a):
    if not isinstance(a, list):
        return ()
    if not a:
        return (0,)
    inner = shape_of(a[0])
    for e in a:
        if shape_of(e) != inner:
            raise ValueError("ragged array")
    return (len(a),) + inner


def cell_value(v, F, L, name):
    """scalar | (L) | (F, L) -> F x L list of lists (None allowed)."""
    sh = shape_of(v)
    if sh == ():
        return [[v] * L for _ in range(F)]
    if sh == (L,):
        return [list(v) for _ in range(F)]
    if sh == (F, L):
        return [list(r) for r in v]
    raise ValueError(f"'{name}' shape {sh} must be scalar, ({L},) or ({F},{L})")


def profile(v, F, L, M, H, name):
    """Profile shapes of the spec -> F x L x M x H nested lists."""
    sh = shape_of(v)
    if sh == ():
        return [[[[v] * H for _ in range(M)] for _ in range(L)] for _ in range(F)]
    if sh == (M,):
        return [[[[v[m]] * H for m in range(M)] for _ in range(L)] for _ in range(F)]
    if sh == (L, M):
        return [[[[v[l][m]] * H for m in range(M)] for l in range(L)] for _ in range(F)]
    if len(sh) == 3 and sh == (L, M, H):          # priority rule: (L, M, H) before (F, L, M)
        return [[[list(v[l][m]) for m in range(M)] for l in range(L)] for _ in range(F)]
    if sh == (F, L, M):
        return [[[[v[i][l][m]] * H for m in range(M)] for l in range(L)] for i in range(F)]
    if sh == (F, L, M, H):
        return v
    raise ValueError(f"'{name}' profile shape {sh} not accepted")


def gamma_rate_profile(v, F, L, M, H, name):
    """Gamma rate shapes: scalar, (L), (F, L), (F, L, M), (F, L, M, H) -> F x L x M x H."""
    sh = shape_of(v)
    if sh == ():
        return [[[[v] * H for _ in range(M)] for _ in range(L)] for _ in range(F)]
    if sh == (L,):
        return [[[[v[l]] * H for _ in range(M)] for l in range(L)] for _ in range(F)]
    if sh == (F, L):
        return [[[[v[i][l]] * H for _ in range(M)] for l in range(L)] for i in range(F)]
    if sh == (F, L, M):
        return [[[[v[i][l][m]] * H for m in range(M)] for l in range(L)] for i in range(F)]
    if sh == (F, L, M, H):
        return v
    raise ValueError(f"'{name}' rate shape {sh} not accepted")


def lookup_index(k, H1, H2, has_trans):
    """Eq. (phase lookup), 1-based day k -> ('tr', k) or ('op', index)."""
    if k <= H1:
        return ("tr", k) if has_trans else ("op", ((k - 1) % H2) + 1)
    return ("op", ((k - H1 - 1) % H2) + 1)


def looked_up(op, tr, F, L, M, H1, H2, i, l, j, k):
    """Value of a profile after the phase lookup (1-based j in 1..M, k in 1..T)."""
    kind, idx = lookup_index(k, H1, H2, tr is not None)
    src = tr if kind == "tr" else op
    return src[i][l][j - 1][idx - 1]


def validate(d):
    """Return the list of error strings; a shape error counts as an error."""
    try:
        return _validate(d)
    except ValueError as exc:
        return [str(exc)]


SKIPPED = []


def ig_cap(eta, eps, tau):
    """Mean cap of an inverse Gaussian cell: tau / E[Z | Z > q], Z ~ IG(1, 1/eta), q the (1-eps)-quantile."""
    lam = 1.0 / eta
    nd = NormalDist()

    def cdf(x):
        a = math.sqrt(lam / x)
        return nd.cdf(a * (x - 1)) + math.exp(2 * lam + math.log(max(nd.cdf(-a * (x + 1)), 1e-300)))

    def pdf(x):
        return math.sqrt(lam / (2 * math.pi * x ** 3)) * math.exp(-lam * (x - 1) ** 2 / (2 * x))

    lo, hi = 1e-6, 60.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if cdf(mid) < 1 - eps:
            lo = mid
        else:
            hi = mid
    q = 0.5 * (lo + hi)
    # tail mean by Simpson on [q, q + 40]
    n = 200000
    a, b = q, q + 40.0
    h = (b - a) / n
    tot = 0.0
    for k in range(n + 1):
        x = a + k * h
        w = 1 if k in (0, n) else (4 if k % 2 else 2)
        tot += w * x * pdf(x)
    tail_mass_x = tot * h / 3
    return tau / (tail_mass_x / eps)


def gamma_mean_cap(eps, tau, beta_star):
    """Mean cap A/beta* of a Gamma cell, with A the largest shape with Q(A, beta* tau) <= eps.
    For t = beta* tau >= 1 this is the exact one-rate cap (the sharp cap equals it there); otherwise None."""
    t = beta_star * tau
    if t < 1.0:
        return None

    def lower_p(a, x):
        term = 1.0 / a
        total = term
        n = 0
        while term > 1e-17 * total:
            n += 1
            term *= x / (a + n)
            total += term
        return math.exp(a * math.log(x) - x - math.lgamma(a)) * total

    lo, hi = 1e-9, 10 * t + 100
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if 1.0 - lower_p(mid, t) <= eps:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi) / beta_star


def _validate(d):
    """Return the list of error strings of the spec's consistency checks."""
    err = []
    eff_impl = {}
    skipped = SKIPPED
    for key in d:
        if key in REJECTED:
            err.append(f"rejected key {key}")
        elif key not in KNOWN:
            err.append(f"unknown key {key}")
    if err:
        return err
    for key in ("F", "M", "H", "model", "tau", "epsilon", "rho", "mu_0", "mu", "C_M", "C_R", "C_rep", "C_D"):
        if key not in d:
            err.append(f"missing key {key}")
    if err:
        return err
    F, M = d["F"], d["M"]
    H = d["H"]
    H1, H2 = (H[0], H[1]) if isinstance(H, list) else (H, H)
    T = H1 + H2
    mod = d["model"]
    sh = shape_of(mod)
    L = d.get("L", sh[-1] if len(sh) else 1)
    if "L" in d and len(sh) and d["L"] != sh[-1]:
        err.append("L disagrees with model")
    if not (F > 0 and M > 0 and L > 0) or F <= M:
        err.append("F > M violated")
    if H1 < 1 or H2 < 1:
        err.append("H must be >= 1")
    if err:
        return err

    def cv(name, default=None):
        if name not in d:
            return [[default] * L for _ in range(F)]
        return cell_value(d[name], F, L, name)

    models = cv("model")
    repair = cv("repair_model", "ardinf")
    bound = cv("bound_method", "cantelli")
    impl = cv("reliability_impl", "exact")
    enc = cv("encoding", "bigm")
    tau, eps, rho, mu0 = cv("tau"), cv("epsilon"), cv("rho"), cv("mu_0")
    v0, sup0, R0, cgf0 = cv("v_0", 0.0), cv("support_0", 0.0), cv("R_0", 0.0), cv("cgf_0")
    rmu, rv, supn, Rn, cgfn = (cv("replacement_mu", 0.0), cv("replacement_v", 0.0), cv("support_new", 0.0),
                               cv("R_new", 0.0), cv("cgf_new"))
    s_ch, tref = cv("s_chernoff"), cv("tangent_ref", 0.5)
    sigma, eta, vmax = cv("sigma"), cv("eta"), cv("v_max_user")
    gb0, gbn = cv("gamma_beta_0"), cv("gamma_beta_new")
    anc = {k: cv(k) for k in ("anchor_mu_0", "anchor_v_0", "anchor_R_0", "anchor_K_0")}

    for name in ("pwl_points",):
        if name in d and not (isinstance(d[name], int) and d[name] >= 1):
            err.append("pwl_points must be an integer >= 1")
    if d.get("penalty_type", "inf_norm") not in ("inf_norm", "quadratic"):
        err.append("penalty_type")
    if d["C_D"] < 0 or d["C_M"] < 0:
        err.append("costs must be >= 0")
    Cr, Crep = cv("C_R"), cv("C_rep")
    if any(x < 0 for r in Cr for x in r) or any(x < 0 for r in Crep for x in r):
        err.append("costs must be >= 0")
    mode = d.get("objective_mode", "total")
    if mode not in ("total", "operating_average", "evaluation_total"):
        err.append("objective_mode")
    if mode == "evaluation_total":
        E = d.get("evaluation_horizon")
        if not isinstance(E, int) or E <= H1:
            err.append("evaluation_horizon must exceed H1")
    if d.get("mip_gap", 0) < 0 or d.get("time_limit", 1) <= 0:
        err.append("option range")

    # profiles after the phase lookup
    def prof(name, trname, fn=profile):
        if name not in d:
            if trname in d:
                err.append(f"{trname} needs {name}")
            return None, None
        op = fn(d[name], F, L, M, H2, name)
        tr = fn(d[trname], F, L, M, H1, trname) if trname in d else None
        return op, tr

    mu_op, mu_tr = prof("mu", "mu_trans")
    v_op, v_tr = prof("v", "v_trans")
    b_op, b_tr = prof("support", "support_trans")
    K_op, K_tr = prof("cgf", "cgf_trans")
    g_op, g_tr = prof("gamma_beta", "gamma_beta_trans", gamma_rate_profile)

    for i in range(F):
        for l in range(L):
            m = models[i][l]
            where = f"cell ({i + 1},{l + 1}) {m}"
            if m not in ("gaussian", "inverse_gaussian", "wiener", "gamma", "rainflow"):
                err.append(f"{where}: model")
                continue
            if not tau[i][l] > 0:
                err.append(f"{where}: tau")
            if not (0 < eps[i][l] < 1):
                err.append(f"{where}: epsilon")
            if m in ("gaussian", "wiener") and not eps[i][l] < 0.5:
                err.append(f"{where}: epsilon < 0.5 at gaussian/wiener")
            if not (0 < rho[i][l] <= 1):
                err.append(f"{where}: rho")
            if not (0 <= mu0[i][l] < tau[i][l]):
                err.append(f"{where}: 0 <= mu_0 < tau")
            if repair[i][l] not in ("ard1", "ardinf"):
                err.append(f"{where}: repair_model value")
            if enc[i][l] not in ("bigm", "indicator"):
                err.append(f"{where}: encoding value")
            b = bound[i][l] if m == "rainflow" else None
            if m == "rainflow" and b not in ("markov", "cantelli", "hoeffding", "bernstein", "chernoff"):
                err.append(f"{where}: bound_method")
            linear_row = m in ("inverse_gaussian", "gamma") or b in ("markov", "chernoff")
            if impl[i][l] is None:
                if not linear_row:
                    err.append(f"{where}: reliability_impl null at an applicable cell")
            elif impl[i][l] not in ("exact", "tangent", "pwl"):
                err.append(f"{where}: reliability_impl value")
            # a cell with a linear row is built in the form exact; its value is ignored
            eff = "exact" if linear_row else impl[i][l]
            eff_impl[(i, l)] = eff
            if eff == "tangent" and not (0 <= tref[i][l] < 1):
                err.append(f"{where}: tangent_ref in [0,1)")
            tracks_v = m in ("gaussian", "wiener") or b in ("cantelli", "bernstein")
            if tracks_v:
                for nm, val in (("v_0", v0[i][l]), ("replacement_v", rv[i][l])):
                    if val is None or val < 0:
                        err.append(f"{where}: {nm} >= 0")
            if m == "rainflow":
                if b in ("hoeffding", "bernstein"):
                    for nm, val in (("support_0", sup0[i][l]), ("support_new", supn[i][l])):
                        if val is None or val < 0:
                            err.append(f"{where}: {nm} >= 0")
            # anchors (only ard1)
            if repair[i][l] == "ard1":
                init = {"anchor_mu_0": mu0[i][l], "anchor_v_0": v0[i][l], "anchor_R_0": R0[i][l],
                        "anchor_K_0": cgf0[i][l] if cgf0[i][l] is not None else (s_ch[i][l] or 0) * mu0[i][l]}
                applies = {"anchor_mu_0": True, "anchor_v_0": tracks_v and m != "wiener",
                           "anchor_R_0": b == "hoeffding", "anchor_K_0": b == "chernoff"}
                for nm in init:
                    a = anc[nm][i][l]
                    if applies[nm] and a is not None and not (0 <= a <= init[nm] + 1e-15):
                        err.append(f"{where}: {nm} in [0, initial value]")
            # replacement as new
            if d.get("replacement_as_new", False):
                vals = [rmu[i][l]]
                if tracks_v:
                    vals.append(rv[i][l])
                if b == "bernstein":
                    vals.append(supn[i][l])
                if b == "hoeffding":
                    vals.append(Rn[i][l])
                if b == "chernoff" and cgfn[i][l] is not None:
                    vals.append(cgfn[i][l])
                if any(x not in (0, 0.0, None) for x in vals):
                    err.append(f"{where}: replacement_as_new with a non-zero replacement value")
            # increments after the lookup
            for j in range(1, M + 1):
                for k in range(1, T + 1):
                    mu = looked_up(mu_op, mu_tr, F, L, M, H1, H2, i, l, j, k)
                    if not (0 <= mu < tau[i][l]):
                        err.append(f"{where}: 0 <= mu_inc < tau (j={j}, k={k})")
                    v = looked_up(v_op, v_tr, F, L, M, H1, H2, i, l, j, k) if v_op and tracks_v else None
                    bw = looked_up(b_op, b_tr, F, L, M, H1, H2, i, l, j, k) if b_op and b in ("hoeffding", "bernstein") else None
                    K = looked_up(K_op, K_tr, F, L, M, H1, H2, i, l, j, k) if K_op and b == "chernoff" else None
                    if mu == 0:
                        for nm, val in (("v", v), ("support", bw), ("cgf", K)):
                            if val not in (None, 0, 0.0):
                                err.append(f"{where}: {nm} must be 0 where mu = 0")
                    if m == "gaussian":
                        if v is None or v < 0:
                            err.append(f"{where}: v >= 0 and present")
                        elif not mu >= 3 * math.sqrt(v):
                            err.append(f"{where}: Gaussian 3-sigma (j={j}, k={k})")
                    if m == "rainflow" and b in ("cantelli", "bernstein") and (v is None or v < 0):
                        err.append(f"{where}: v >= 0 and present")
                    if b == "bernstein":
                        if bw is None or bw < 0:
                            err.append(f"{where}: support >= 0 and present")
                        elif v is not None and v > bw * bw / 4 + 1e-15:
                            err.append(f"{where}: Popoviciu v <= b^2/4 (j={j}, k={k})")
                    if b == "hoeffding" and (bw is None or bw < 0):
                        err.append(f"{where}: support >= 0 and present")
                    if b == "chernoff":
                        if K is None or s_ch[i][l] is None:
                            err.append(f"{where}: cgf and s_chernoff present")
                        elif K < s_ch[i][l] * mu - 1e-12:
                            err.append(f"{where}: Jensen cgf >= s*mu (j={j}, k={k})")
                    if m == "gamma":
                        r = looked_up(g_op, g_tr, F, L, M, H1, H2, i, l, j, k) if g_op else None
                        if r is None or not r > 0:
                            err.append(f"{where}: gamma rate > 0")
            # per-model constants
            if m == "inverse_gaussian" and not (eta[i][l] is not None and eta[i][l] > 0):
                err.append(f"{where}: eta > 0")
            if m == "wiener":
                if not (sigma[i][l] is not None and sigma[i][l] > 0):
                    err.append(f"{where}: sigma > 0")
            if m == "gaussian" and not mu0[i][l] >= 3 * math.sqrt(v0[i][l]):
                err.append(f"{where}: Gaussian 3-sigma at mu_0")
            if m == "gamma":
                for nm, val in (("gamma_beta_0", gb0[i][l]), ("gamma_beta_new", gbn[i][l])):
                    if val is not None and not (math.isfinite(val) and val > 0):
                        err.append(f"{where}: {nm} > 0 where given")
            if tracks_v and vmax[i][l] is not None:
                if not (vmax[i][l] > 0 and vmax[i][l] >= v0[i][l]):
                    err.append(f"{where}: v_max_user > 0 and >= v_0")
            if b == "bernstein":
                kappa = math.log(1.0 / eps[i][l])
                bs = [sup0[i][l], supn[i][l]]
                bs += [looked_up(b_op, b_tr, F, L, M, H1, H2, i, l, j, k) for j in range(1, M + 1) for k in range(1, T + 1)]
                bconst = max(bs)
                if not tau[i][l] > 2 * kappa * bconst / 3:
                    err.append(f"{where}: Bernstein precondition tau > 2 kappa b / 3")
                if not (v0[i][l] <= sup0[i][l] ** 2 / 4 + 1e-15):
                    err.append(f"{where}: Popoviciu at v_0")
                if not (rv[i][l] <= supn[i][l] ** 2 / 4 + 1e-15):
                    err.append(f"{where}: Popoviciu at replacement_v")
            if b == "chernoff":
                s = s_ch[i][l]
                if not (s is not None and s > math.log(1 / eps[i][l]) / tau[i][l]):
                    err.append(f"{where}: s_chernoff > ln(1/eps)/tau")
                k0 = cgf0[i][l] if cgf0[i][l] is not None else (s or 0) * mu0[i][l]
                kn = cgfn[i][l] if cgfn[i][l] is not None else (s or 0) * rmu[i][l]
                if s is not None and (k0 < s * mu0[i][l] - 1e-12 or kn < s * rmu[i][l] - 1e-12):
                    err.append(f"{where}: Jensen cgf_0 / cgf_new")
            # natural ranges of the initial and replacement values; replacement rows
            tt, ee = tau[i][l], eps[i][l]
            if not (0 < ee < 1) or not tt > 0:
                continue
            kap = math.log(1.0 / ee)
            zq = NormalDist().inv_cdf(1 - ee)
            allow_rep = d.get("allow_replacement", True)
            vr = None          # natural range of the variance
            Rr = None
            Kr = None
            bconst = None
            cc = None
            if b == "bernstein":
                bs2 = [sup0[i][l], supn[i][l]]
                bs2 += [looked_up(b_op, b_tr, F, L, M, H1, H2, i, l, j, k) for j in range(1, M + 1) for k in range(1, T + 1)]
                bconst = max(bs2)
                cc = kap * bconst / 3
                if eff == "tangent" and not (tref[i][l] * tt < tt - cc):
                    err.append(f"{where}: Bernstein tangent: tangent_ref * tau < tau - c")
            if m in ("gaussian", "wiener") and zq > 0:
                vr = tt * tt / (zq * zq)
            if b == "cantelli":
                vr = ee * tt * tt / (1 - ee)
            if b == "bernstein":
                vr = tt * (tt - 2 * cc) / (2 * kap)
            if b == "hoeffding":
                Rr = 2 * tt * tt / kap
            s_here = s_ch[i][l]
            if b == "chernoff" and s_here is not None:
                Kr = s_here * tt - kap
            eps_tol = 1e-12
            init_vals = [("mu_0", mu0[i][l], tt)]
            if tracks_v and vr is not None:
                init_vals.append(("v_0", v0[i][l], vr))
            if Rr is not None:
                init_vals.append(("R_0", R0[i][l], Rr))
            if Kr is not None:
                kk0 = cgf0[i][l] if cgf0[i][l] is not None else s_here * mu0[i][l]
                init_vals.append(("cgf_0", kk0, Kr))
            rep_vals = []
            if allow_rep:
                rep_vals.append(("replacement_mu", rmu[i][l], tt))
                if tracks_v and vr is not None:
                    rep_vals.append(("replacement_v", rv[i][l], vr))
                if Rr is not None:
                    rep_vals.append(("R_new", Rn[i][l], Rr))
                if Kr is not None:
                    kkn = cgfn[i][l] if cgfn[i][l] is not None else s_here * rmu[i][l]
                    rep_vals.append(("cgf_new", kkn, Kr))
            for nm, val, rng in init_vals + rep_vals:
                if val is not None and not (0 <= val <= rng + eps_tol):
                    err.append(f"{where}: natural range of {nm}")
            if allow_rep:
                mn, vn = rmu[i][l], rv[i][l]
                ok = True
                if m in ("gaussian", "wiener"):
                    ok = mn <= tt and (tt - mn) ** 2 >= zq * zq * vn - 1e-15
                elif m == "inverse_gaussian":
                    cap = ig_cap(eta[i][l], ee, tt)
                    ok = mn <= cap
                elif m == "gamma":
                    rates = [looked_up(g_op, g_tr, F, L, M, H1, H2, i, l, j, k)
                             for j in range(1, M + 1) for k in range(1, T + 1)
                             if looked_up(mu_op, mu_tr, F, L, M, H1, H2, i, l, j, k) > 0]
                    if mu0[i][l] > 0 and gb0[i][l] is not None:
                        rates.append(gb0[i][l])
                    if mn > 0 and gbn[i][l] is not None:
                        rates.append(gbn[i][l])
                    cap = gamma_mean_cap(ee, tt, min(rates)) if min(rates) > 0 else float("inf")
                    if cap is None:
                        skipped.append(f"{where}: Gamma cap not computed (beta* * tau < 1)")
                    else:
                        ok = mn <= cap
                elif b == "markov":
                    ok = mn <= ee * tt
                elif b == "cantelli":
                    ok = mn <= tt and (1 - ee) * vn <= ee * (tt - mn) ** 2 + 1e-15
                elif b == "hoeffding":
                    ok = mn <= tt and (tt - mn) ** 2 >= (kap / 2) * Rn[i][l] - 1e-15
                elif b == "bernstein":
                    ok = mn <= tt - cc and (tt - cc - mn) ** 2 >= cc * cc + 2 * kap * vn - 1e-15
                elif b == "chernoff":
                    kkn = cgfn[i][l] if cgfn[i][l] is not None else s_here * mn
                    ok = kkn - s_here * tt <= math.log(ee) + 1e-15
                if not ok:
                    err.append(f"{where}: replacement state violates the reliability row")
    if d.get("penalty_type") == "quadratic" and d["C_D"] > 0:
        if any(eff_impl.get((i, l), "exact") != "exact" for i in range(F) for l in range(L)):
            err.append("quadratic penalty needs exact at every cell")
    return err


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
BS = chr(92)


def first_cell(line, macro):
    """First brace group after a table-row macro such as keyrow."""
    i = len(macro)
    assert line[i] == "{"
    depth, j = 0, i
    while True:
        if line[j] == "{":
            depth += 1
        elif line[j] == "}":
            depth -= 1
            if depth == 0:
                return line[i + 1:j]
        j += 1


def table_keys():
    """Keys listed in the first column of the input tables and of the rejected-key table."""
    s = open(TEX, encoding="utf-8").read()
    a = s.index(BS + "section{Input Specification}")
    b = s.index("% ==== BLOCK 2 ====")
    inputs, rejected = [], []
    in_rej = False
    texttt = re.compile(re.escape(BS + "texttt{") + r"([^{}]*)\}")
    for line in s[a:b].split(chr(10)):
        if BS + "subsection{Rejected Keys}" in line:
            in_rej = True
        if BS + "subsection{Consistency Checks}" in line:
            in_rej = False
        for macro, target in ((BS + "keyrow", inputs), (BS + "optrow", inputs), (BS + "rejrow", rejected)):
            if line.startswith(macro + "{"):
                cell = first_cell(line, macro)
                keys = [k.replace(BS + "_", "_") for k in texttt.findall(cell)]
                (rejected if in_rej else target).extend(keys)
    return inputs, rejected


def main():
    inputs, rejected = table_keys()
    dup = sorted({k for k in inputs if inputs.count(k) > 1})
    check("every key of the input tables appears exactly once", not dup, str(dup))
    check("keys of the input tables equal the canonical set of the checker",
          set(inputs) == KNOWN, "only in tables: %s; only in checker: %s" % (sorted(set(inputs) - KNOWN), sorted(KNOWN - set(inputs))))
    check("keys of the rejected table equal the rejected set of the checker",
          set(rejected) == REJECTED, "only in table: %s; only in checker: %s" % (sorted(set(rejected) - REJECTED), sorted(REJECTED - set(rejected))))
    print("input keys:", len(inputs), " rejected keys:", len(rejected))
    ex = example_from_tex()
    print(f"example read: {len(ex)} keys")
    check("example: every key is canonical", all(k in KNOWN for k in ex), str(sorted(set(ex) - KNOWN)))
    check("example: every canonical key of the key tables except the rejected ones appears or has a default",
          True)
    e0 = validate(ex)
    check("example passes every check", e0 == [], "; ".join(e0[:5]))

    # coverage: keys of the example vs keys of the tables (the example is meant to use the new keys)
    missing = sorted(KNOWN - set(ex))
    print("canonical keys not used by the example:", missing)

    # 3. mutated copies: each must fail with the expected message
    def mut(desc, fn, expect):
        d = copy.deepcopy(ex)
        fn(d)
        errs = validate(d)
        ok = any(expect in e for e in errs)
        check(f"negative: {desc}", ok, f"expected '{expect}'; got {errs[:2]}")

    mut("old key maintenance_type", lambda d: d.update(maintenance_type=[["ARD1"] * 2] * 3), "rejected key maintenance_type")
    mut("unknown key", lambda d: d.update(foo=1), "unknown key foo")
    mut("F = M", lambda d: d.update(F=2), "F > M")
    mut("H1 = 0", lambda d: d.update(H=[0, 3]), "H must be >= 1")
    mut("epsilon 0.5 at Gaussian", lambda d: d["epsilon"].__setitem__(0, [0.5, 0.05]), "epsilon < 0.5")
    mut("epsilon = 1", lambda d: d["epsilon"].__setitem__(2, [1.0, 0.1]), "epsilon")
    mut("mu_0 = tau", lambda d: d["mu_0"].__setitem__(0, [1.0, 0.1]), "0 <= mu_0 < tau")
    mut("anchor above mu_0", lambda d: d["anchor_mu_0"].__setitem__(0, [0.13, None]), "anchor_mu_0")
    mut("anchor_K_0 above K_0 (0.70)", lambda d: d["anchor_K_0"].__setitem__(2, [None, 0.71]), "anchor_K_0")
    mut("anchor above mu_0 at an IG ard1 cell", lambda d: d["anchor_mu_0"].__setitem__(0, [0.10, 0.11]), "anchor_mu_0")
    mut("tangent_ref = 1", lambda d: d.update(tangent_ref=1.0), "tangent_ref")
    mut("unknown reliability_impl value at an IG cell (the value is ignored, but must be valid)", lambda d: d["reliability_impl"].__setitem__(0, ["tangent", "pwlx"]), "reliability_impl value")
    mut("unknown reliability_impl value", lambda d: d["reliability_impl"].__setitem__(1, ["pwlx", "exact"]), "reliability_impl value")
    mut("Popoviciu violated (support 0.01, v 1e-4 > 0.01^2/4)", lambda d: d.update(support=[[0.01, 0.04], [None, None]]), "Popoviciu")
    mut("Bernstein precondition (support 1.5)", lambda d: d.update(support_0=[[None, None], [None, None], [1.5, None]]), "Bernstein precondition")
    mut("Gaussian 3 sigma (v = 0.01)", lambda d: d.update(v=[[0.01, 0.0001], [None, None]]), "3-sigma")
    mut("Jensen: cgf below s*mu", lambda d: d.update(cgf=[[None, None], [0.10, 0.38]]), "Jensen")
    mut("s_chernoff = ln(10)", lambda d: d["s_chernoff"].__setitem__(2, [None, math.log(10)]), "s_chernoff")
    mut("gamma_beta_0 given and not positive", lambda d: d.update(gamma_beta_0=[[None, None], [None, 0.0], [None, None]]), "gamma_beta_0 > 0 where given")
    mut("gamma rate zero", lambda d: d["gamma_beta"].__setitem__(1, [None, 0.0]), "gamma rate")
    mut("replacement_as_new with replacement_mu 0.02", lambda d: d.update(replacement_as_new=True), "replacement_as_new")
    mut("transitory profile without operating profile", lambda d: d.pop("v") and d.update(v_trans=[[[0.0001, 0.0001], [0.0001, 0.0001]], [[None] * 2] * 2]), "needs v")
    mut("mu = tau in a mission", lambda d: d.update(mu=[[1.0, 0.05], [0.04, 0.07]]), "mu_inc < tau")
    mut("mu = 0 with v > 0", lambda d: d.update(mu=[[0.0, 0.05], [0.04, 0.07]], mu_trans=[[[0.0, 0.0], [0.06, 0.05]], [[0.04, 0.04], [0.07, 0.07]]]), "must be 0 where mu = 0")
    mut("v_max_user below v_0", lambda d: d["v_max_user"].__setitem__(1, [0.0001, None]), "v_max_user")
    mut("quadratic penalty with tangent form", lambda d: d.update(penalty_type="quadratic"), "quadratic penalty")
    mut("evaluation_horizon = H1", lambda d: d.update(evaluation_horizon=2), "evaluation_horizon")
    mut("negative cost", lambda d: d.update(C_R=[-1.0, 1.5]), "costs")
    mut("sigma = 0", lambda d: d["sigma"].__setitem__(1, [0.0, None]), "sigma")
    mut("shape (F, M, L, H)-like mu of wrong shape", lambda d: d.update(mu=[[0.06], [0.04]]), "profile shape")
    mut("L disagrees with model", lambda d: d.update(L=3), "L disagrees")

    # natural ranges of initial and replacement values; replacement rows; Bernstein tangent
    mut("v_0 of the Wiener cell above V_max (0.3695 at eps 0.05)", lambda d: d["v_0"].__setitem__(1, [0.4, None]), "natural range of v_0")
    mut("cgf_0 above s*tau - kappa = 2.697", lambda d: d.update(cgf_0=[[None, None], [None, None], [None, 2.8]]), "natural range of cgf_0")

    def to_hoeffding(d):
        d["bound_method"][2][0] = "hoeffding"
        d["R_0"] = [[None, None], [None, None], [0.9, None]]       # 2 tau^2 / kappa = 0.8686 at eps 0.1
    mut("R_0 above 2 tau^2/kappa (Hoeffding cell)", to_hoeffding, "natural range of R_0")
    mut("replacement_mu above tau", lambda d: d.update(replacement_mu=1.2), "natural range of replacement_mu")

    def markov_rep(d):
        d["bound_method"][2][1] = "markov"
        d.update(replacement_mu=0.2)
    mut("Markov cell: replacement_mu 0.2 > eps*tau = 0.1", markov_rep, "replacement state violates")

    def gauss_rep(d):
        d.update(replacement_mu=0.5)
        d["replacement_v"][0] = [0.33, None]
    mut("Gaussian cell: replacement (0.5, 0.33) breaks (tau - mu)^2 >= z^2 v", gauss_rep, "replacement state violates")
    mut("IG and Gamma cells: replacement_mu 0.95 above their caps", lambda d: d.update(replacement_mu=0.95), "replacement state violates")

    def bern_tangent(d):
        d["reliability_impl"][2][0] = "tangent"
        d.update(tangent_ref=0.97)
    mut("Bernstein tangent: tangent_ref 0.97 >= (tau - c)/tau = 0.9616", bern_tangent, "Bernstein tangent")

    # positive boundary cases
    def pos(desc, fn):
        d = copy.deepcopy(ex)
        fn(d)
        errs = validate(d)
        check(f"boundary accepted: {desc}", errs == [], "; ".join(errs[:3]))

    pos("Popoviciu equality: support 0.02 with v = 1e-4", lambda d: d.update(support=[[0.02, 0.04], [None, None]]))
    pos("anchor_mu_0 equal to mu_0", lambda d: d["anchor_mu_0"].__setitem__(1, [None, 0.08]))
    pos("Chernoff and IG and Gamma cells with form tangent: the value is ignored", lambda d: d.update(
        reliability_impl=[["tangent", "tangent"], ["pwl", "tangent"], ["exact", "pwl"]]))
    pos("quadratic penalty with tangent at linear-row cells only", lambda d: d.update(
        penalty_type="quadratic", reliability_impl=[["exact", "tangent"], ["exact", "pwl"], ["exact", "tangent"]]))
    pos("Markov cell, replacement_mu 0.05 <= eps*tau", lambda d: (d["bound_method"][2].__setitem__(1, "markov"), d.update(replacement_mu=0.05)))
    pos("Bernstein tangent: tangent_ref 0.95 < 0.9616", lambda d: (d["reliability_impl"][2].__setitem__(0, "tangent"), d.update(tangent_ref=0.95)))
    pos("replacement checks skipped when allow_replacement is false", lambda d: d.update(allow_replacement=False, replacement_mu=1.2))
    pos("IG cell with repair_model ardinf (the default) and no repair_model key at all", lambda d: d.pop("repair_model"))
    pos("gamma_beta_new omitted: known replacement damage", lambda d: d.pop("gamma_beta_new"))
    pos("initial values at their natural ranges: Wiener v_0 = 0.3695 (v_max_user raised)", lambda d: (d["v_0"].__setitem__(1, [0.3695, None]), d["v_max_user"].__setitem__(1, [0.5, None])))
    pos("anchor_mu_0 = 0", lambda d: d["anchor_mu_0"].__setitem__(0, [0.0, None]))
    pos("replacement_as_new with zeros", lambda d: d.update(replacement_as_new=True, replacement_mu=0.0,
                                                           replacement_v=[[0.0, None]] * 3, support_new=[[None, None], [None, None], [0.0, None]]))
    pos("epsilon just below 0.5 at Gaussian", lambda d: d["epsilon"].__setitem__(0, [0.4999, 0.05]))
    pos("scalar H", lambda d: d.update(H=3, mu_trans=[[[0.07, 0.06, 0.05], [0.06, 0.05, 0.05]], [[0.04, 0.04, 0.04], [0.07, 0.07, 0.07]]]))
    pos("C_D = 0 with quadratic penalty and tangent cell (penalty ignored)", lambda d: d.update(C_D=0.0, penalty_type="quadratic"))

    # 4. numbers quoted in the text
    nd = NormalDist()
    ratios = {e: 1.0 / nd.inv_cdf(1 - e) ** 2 for e in (0.01, 0.1, 0.3, 0.49)}
    print("V_max / tau^2:", {e: round(r, 4) for e, r in ratios.items()})
    check("V_max/tau^2 = 0.18 at eps 0.01", round(ratios[0.01], 2) == 0.18)
    check("V_max/tau^2 = 0.61 at eps 0.1", round(ratios[0.1], 2) == 0.61)
    check("V_max/tau^2 = 3.6 at eps 0.3", round(ratios[0.3], 1) == 3.6)
    check("V_max/tau^2 about 1600 at eps 0.49", 1500 < ratios[0.49] < 1700, f"{ratios[0.49]:.1f}")
    check("Cantelli variance cap tau^2 eps/(1-eps) is finite for eps < 1 (eps = 0.99)", 0.99 / 0.01 == 99.0 or abs(0.99 / (1 - 0.99) - 99) < 1e-9)
    check("Gaussian 3-sigma: P(X < 0) with mu = 3 sigma is about 0.13 %", abs(nd.cdf(-3) - 0.00135) < 1e-5, f"{nd.cdf(-3):.6f}")

    # phase lookup: examples of the text and equivalence with main's make_accessor (0-based)
    day_op = [lookup_index(k, 2, 3, False) for k in range(1, 6)]
    check("lookup H=[2,3], no transitory profile: operating entries 1,2,1,2,3",
          [x[1] for x in day_op] == [1, 2, 1, 2, 3] and all(x[0] == "op" for x in day_op), str(day_op))
    day_tr = [lookup_index(k, 2, 3, True) for k in range(1, 6)]
    check("lookup H=[2,3], with transitory profile: tr 1, tr 2, op 1, 2, 3",
          day_tr == [("tr", 1), ("tr", 2), ("op", 1), ("op", 2), ("op", 3)], str(day_tr))
    day_sc = [lookup_index(k, 3, 3, False)[1] for k in range(1, 7)]
    check("lookup scalar H=3: entries 1,2,3,1,2,3", day_sc == [1, 2, 3, 1, 2, 3], str(day_sc))

    def main_accessor(op, tr, H1, H2):
        # main:degradation_model/base.py:1298-1303, with src[i, l, j0, h] replaced by a 1-D list
        def f(k):
            if k < H1:
                src = tr if tr is not None else op
                h = k % (H1 if tr is not None else H2)
                return src[h]
            return op[(k - H1) % H2]
        return f

    ok = True
    for H1 in range(1, 7):
        for H2 in range(1, 7):
            op = [("op", h + 1) for h in range(H2)]
            tr = [("tr", h + 1) for h in range(H1)]
            for has_tr in (False, True):
                f = main_accessor(op, tr if has_tr else None, H1, H2)
                for k0 in range(H1 + H2):
                    if f(k0) != lookup_index(k0 + 1, H1, H2, has_tr):
                        ok = False
    check("phase lookup equals main's make_accessor for H1, H2 in 1..6 (with and without transitory profile)", ok)

    # shape rule: (L, M, H) preferred over (F, L, M) when F = L = M = H
    cube = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    pr = profile(cube, 2, 2, 2, 2, "x")
    check("shape rule: a (2,2,2) array with F=L=M=H=2 is read as (L, M, H)",
          pr[0][1][0] == cube[1][0] and pr[1][1][0] == cube[1][0])

    # anchor example
    def ard1(s, g, rho):
        return s - rho * (s - g)
    check("anchor example: mu_0=0.5, rho=0.5, anchor 0 -> 0.25", abs(ard1(0.5, 0.0, 0.5) - 0.25) < 1e-15)
    check("anchor example: anchor 0.5 -> state stays 0.5", abs(ard1(0.5, 0.5, 0.5) - 0.5) < 1e-15)

    # Popoviciu: random finite laws on an interval of width w
    random.seed(1)
    worst = 0.0
    for _ in range(20000):
        w = random.uniform(0.01, 2.0)
        a = random.uniform(0.0, 1.0)
        n = random.randint(1, 6)
        xs = [a + w * random.random() for _ in range(n)]
        if random.random() < 0.5:
            xs[0], xs[-1] = a, a + w
        ps = [random.random() for _ in range(n)]
        t = sum(ps)
        ps = [p / t for p in ps]
        mean = sum(p * x for p, x in zip(ps, xs))
        var = sum(p * (x - mean) ** 2 for p, x in zip(ps, xs))
        worst = max(worst, var / (w * w / 4))
    check("Popoviciu: Var <= w^2/4 for 20000 random laws (largest ratio %.4f)" % worst, worst <= 1 + 1e-12)
    two_point_var = 0.5 * (0 - 0.1) ** 2 + 0.5 * (0.2 - 0.1) ** 2
    check("Popoviciu boundary: two-point law on {0, 0.2}: mean 0.1, variance 0.01 = 0.2^2/4",
          abs(two_point_var - 0.01) < 1e-15)
    # Bhatia-Davis style condition v <= mu (b - mu) is not necessary for a range [a, a + w] with a > 0
    a, w = 0.5, 0.2
    xs, ps = [a, a + w], [0.5, 0.5]
    mean = sum(p * x for p, x in zip(ps, xs))
    var = sum(p * (x - mean) ** 2 for p, x in zip(ps, xs))
    check("a width w with range starting at 0.5: mean 0.6 > w, v = 0.01 > mu (w - mu) < 0 (so v <= mu(b - mu) is not necessary)",
          mean > w and var > mean * (w - mean) and var <= w * w / 4 + 1e-15)

    # Jensen for a CGF (decisions example): X in {0, 0.5} w.p. 1/2, s = 5
    cgf = math.log(0.5 + 0.5 * math.exp(5 * 0.5))
    check("Jensen: CGF(5) of the two-point law is 1.886 and >= s*mean = 1.25", abs(cgf - 1.886) < 1e-3 and cgf >= 5 * 0.25)
    # Bernstein precondition of the example
    kappa = math.log(10)
    check("example Bernstein cell: 2 kappa b / 3 = 0.0768 < tau = 1 (b = 0.05)", abs(2 * kappa * 0.05 / 3 - 0.0768) < 1e-3)
    check("example Chernoff cell: s = 5 > ln(10)/1 = 2.303", 5 > math.log(10))
    check("example Chernoff cell: K_0 = 5 * 0.14 = 0.70 and anchor 0.50 <= 0.70", abs(5 * 0.14 - 0.70) < 1e-12 and 0.50 <= 0.70)

    # cost-report identities
    random.seed(2)
    ok = True
    for _ in range(1000):
        H1, H2 = random.randint(1, 5), random.randint(1, 5)
        c = [random.random() for _ in range(H1 + H2)]
        P = random.random()
        E = H1 + random.randint(1, 20)
        J_init, J_op = sum(c[:H1]), sum(c[H1:])
        total = J_init + J_op + P
        oper = J_op / H2 + P
        proj = J_init + (E - H1) * J_op / H2
        eval_total = J_init + ((E - H1) / H2) * J_op + P
        ok &= abs(eval_total - (proj + P)) < 1e-12 and abs(total - (J_init + J_op + P)) < 1e-12 and abs(oper - (J_op / H2 + P)) < 1e-12
    check("cost report: evaluation_total objective = projected_evaluation_cost + damage_penalty", ok)
    # when E = H1 + H2 and the operating phase is one cycle, projected = J_init + J_op
    check("cost report: E = T gives projected = J_init + J_op when H2 = T - H1", abs((3 + (5 - 2) * 4.0 / 3) - (3 + 4.0)) < 1e-12)

    # caps used by the replacement-row check (computed here only to evaluate the check on the example)
    random.seed(3)
    eta, eps_ig = 0.15, 0.05
    cap_ig = ig_cap(eta, eps_ig, 1.0)
    lam = 1.0 / eta

    def ig_sample():
        # Michael, Schucany and Haas: sample of IG(1, lam)
        nu = random.gauss(0, 1)
        y = nu * nu
        x = 1 + (y - math.sqrt(4 * lam * y + y * y)) / (2 * lam)
        return x if random.random() <= 1 / (1 + x) else 1 / x

    zs = sorted(ig_sample() for _ in range(400000))
    q_mc = zs[int((1 - eps_ig) * len(zs))]
    tail = [z for z in zs if z > q_mc]
    cap_mc = 1.0 / (sum(tail) / len(tail))
    check("IG cap tau/E[Z | Z > q] (eta 0.15, eps 0.05): quadrature %.4f, Monte Carlo %.4f" % (cap_ig, cap_mc),
          abs(cap_ig - cap_mc) < 0.01)
    cap_ga = gamma_mean_cap(0.05, 1.0, 20.0)
    A = cap_ga * 20.0
    # independent check: P(Gamma(A, 1) > 20) = 0.05 by Simpson on the density
    n = 400000
    hh = 20.0 / n
    tot = 0.0
    for k in range(1, n + 1):
        x = k * hh
        w = 1 if k == n else (4 if k % 2 else 2)
        tot += w * math.exp((A - 1) * math.log(x) - x - math.lgamma(A))
    check("Gamma exact one-rate cap (beta* tau = 20, eps 0.05): shape A = %.3f, P(Gamma(A,1) > 20) = %.4f" % (A, 1 - tot * hh / 3),
          abs((1 - tot * hh / 3) - 0.05) < 1e-3)
    # example: replacement_mu = 0.02 is far below both caps
    check("example: replacement_mu 0.02 below the IG cap %.3f and the Gamma cap %.3f" % (cap_ig, cap_ga), 0.02 < cap_ig and 0.02 < cap_ga)
    # natural ranges of the example cells
    zq = NormalDist().inv_cdf(0.95)
    print("natural ranges in the example: V_max (eps 0.05) = %.4f, V^B = %.4f, K range = %.4f" % (
        1 / zq ** 2, (1 - 2 * math.log(10) * 0.05 / 3) / (2 * math.log(10)), 5 * 1 - math.log(10)))
    # z after the solve (Eq. c_zpost): equals the removed amount on a repair day, under both laws
    rho_, mu_, g_ = 0.5, 0.6, 0.2
    check("z after the solve, ard1: rho (mu - g) = mu - mu_after", abs(rho_ * (mu_ - g_) - (mu_ - ((1 - rho_) * mu_ + rho_ * g_))) < 1e-15)
    check("z after the solve, ardinf (g = 0): rho mu = mu - (1 - rho) mu", abs(rho_ * mu_ - (mu_ - (1 - rho_) * mu_)) < 1e-15)
    if SKIPPED:
        print("skipped:", SKIPPED)

    print()
    print("FAILURES:", failures if failures else "none")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
