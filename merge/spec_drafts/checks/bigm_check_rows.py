"""Check the explicit big-M rows of fragments_B.tex against the generic rows of fragments_A.tex.

Each labelled row of the model sections is parsed from the LaTeX source, turned into a Python
expression, and compared with the generic row (eq:a_bigm_carry, eq:a_bigm_repair, eq:a_bigm_repl,
eq:a_bigm_anchor) instantiated with the quantity, the factor a and the big-M constant of the model.
Two rows are equal when their slacks (greater side minus lesser side) agree at random real points,
binaries included; equality as affine functions implies equality at binary points.
The Wiener row eq:w_vrow is compared at the three binary points with the generic carry and repair
rows (a = 1), on the box [0, M] and with a zero increment on a depot day.
"""
import random
import re
import sys

PATH = sys.argv[1]
src = open(PATH, encoding="utf-8").read()

# ---- LaTeX to Python ------------------------------------------------------------
SUBS = [
    (r"\bigl(", "("), (r"\bigr)", ")"), (r"\Bigl(", "("), (r"\Bigr)", ")"),
    (r"\geq", " >= "), (r"\leq", " <= "), (r"\,", " "), (r"\;", " "),
    (r"\delta\mu_k", " d_mu "), (r"\delta v_k", " d_v "), (r"\delta R_k", " d_R "),
    (r"\delta K_k", " d_K "),
    (r"x^m_k", " xm "), (r"x^r_k", " xr "), (r"e_k", " e "),
    (r"V_{\max}", " Vmax "), (r"M_v", " Mv "), (r"M_y", " My "),
    (r"^{\new}", "_new "),
    (r"\rho", " rho "), (r"\tau", " tau "), (r"\kappa", " kap "),
]
for q, name in ((r"\mu", "mu"), ("v", "v"), ("R", "R"), ("K", "K"), ("y", "y")):
    SUBS += [
        ("g^{%s}_{k-1}" % q, " g_%s_prev " % name),
        ("g^{%s}_k" % q, " g_%s " % name),
    ]
for q, name in ((r"\mu", "mu"), ("v", "v"), ("R", "R"), ("K", "K"), ("y", "y")):
    SUBS += [
        ("%s_{k-1}" % q, " %s_prev " % name),
        ("%s_k" % q, " %s " % name),
        ("%s_new" % q, " %s_new " % name),
    ]


def to_python(tex):
    s = tex
    for a, b in SUBS:
        s = s.replace(a, b)
    s = s.replace("^", "**")
    s = re.sub(r"\bs\b", " s ", s)
    toks = re.findall(r"[A-Za-z_][A-Za-z_0-9]*|\d+\.?\d*|\*\*|>=|<=|[-+*/()]", s)
    rest = re.sub(r"[A-Za-z_][A-Za-z_0-9]*|\d+\.?\d*|\*\*|>=|<=|[-+*/()]|\s", "", s)
    if rest:
        raise ValueError("unparsed text %r in %r" % (rest, tex))
    out = []
    for t in toks:
        if out:
            p = out[-1]
            operand_end = re.match(r"[A-Za-z_0-9.)]", p[-1]) and p not in ("**",)
            operand_start = re.match(r"[A-Za-z_0-9(]", t[0])
            if operand_end and operand_start:
                out.append("*")
        out.append(t)
    return " ".join(out)


def slack(expr, env):
    if ">=" in expr:
        lhs, rhs = expr.split(">=")
        return eval(lhs, {}, env) - eval(rhs, {}, env)
    lhs, rhs = expr.split("<=")
    return eval(rhs, {}, env) - eval(lhs, {}, env)


# ---- rows of fragments_B.tex ---------------------------------------------------------
rows = []  # (label, [python inequalities])
for block in re.findall(r"\\begin\{align\}(.*?)\\end\{align\}", src, re.S):
    for row in block.split(r"\\"):
        m = re.search(r"\\label\{([^}]*)\}", row)
        if not m:
            continue
        label = m.group(1)
        if not re.match(r"eq:(g|ig|w|gm|rf)_((mu|v|R|K)_(carry|inf|one|repl)|anchor_(set|keep)|vrow)$",
                        label):
            continue
        body = re.sub(r"\\label\{[^}]*\}", "", row)
        body = re.sub(r"\\text\{[^}]*\}|\\ARDinf|\\ARDone", "", body)
        body = body.replace("&", " ").replace("\\,", " ").strip().rstrip(".,").strip()
        parts = [p.strip().rstrip(",").strip() for p in re.split(r",\s*\\quad", body)]
        # anchor rows hold two inequalities separated by the alignment '&' (now spaces)
        if "anchor" in label:
            ineqs = re.findall(r"[^,]*?(?:\\leq)[^,]*", body)
            parts = [p.strip() for p in ineqs]
        rows.append((label, [to_python(p) for p in parts if p]))

# ---- generic rows of fragments_A.tex --------------------------------------------------
FACTOR = {"mu": "(1-rho)", "v": "(1-rho)**2", "R": "(1-rho)**2", "K": "(1-rho)"}
BIGM = {
    "g": {"mu": "tau", "v": "Vmax"}, "ig": {"mu": "tau"}, "w": {"mu": "tau", "v": "Vmax"},
    "gm": {"mu": "tau"}, "rf": {"mu": "tau", "v": "Mv", "R": "(2*tau**2/kap)", "K": "(s*tau-kap)"},
}
ANCHORED = {"g": ["mu", "v"], "ig": ["mu"], "w": ["mu"], "gm": ["mu"], "rf": ["mu", "v", "R", "K"]}


def generic(kind, q, M):
    a = FACTOR[q]
    s, sp, d, g, gp, sn = q, q + "_prev", "d_" + q, "g_" + q, "g_" + q + "_prev", q + "_new"
    if kind == "carry":
        return ["%s >= %s + %s - %s*(xm + xr)" % (s, sp, d, M)]
    if kind == "inf":
        return ["%s >= %s*%s - %s*(1 - xm)" % (s, a, sp, M)]
    if kind == "one":
        return ["%s >= %s*%s + (1 - %s)*%s - %s*(1 - xm)" % (s, a, sp, a, gp, M)]
    if kind == "repl":
        return ["%s >= %s - %s*(1 - xr)" % (s, sn, M), "%s <= %s + %s*(1 - xr)" % (s, sn, M)]
    if kind == "set":
        return ["%s - %s <= %s*(1 - e)" % (g, s, M), "%s - %s <= %s*(1 - e)" % (s, g, M)]
    if kind == "keep":
        return ["%s - %s <= %s*e" % (g, gp, M), "%s - %s <= %s*e" % (gp, g, M)]
    raise ValueError(kind)


NAMES = ["tau", "rho", "kap", "s", "Vmax", "Mv", "xm", "xr", "e"]
for q in ["mu", "v", "R", "K"]:
    NAMES += [q, q + "_prev", "d_" + q, "g_" + q, "g_" + q + "_prev", q + "_new"]

random.seed(1)
fails = 0
checked = 0
for label, model_rows in rows:
    prefix, rest = label[3:].split("_", 1)
    if rest == "vrow":
        continue
    if rest.startswith("anchor"):
        kind = rest.split("_")[1]
        pairs = [(q, BIGM[prefix][q]) for q in ANCHORED[prefix]]
    else:
        q, kind = rest.split("_")
        pairs = [(q, BIGM[prefix][q])]
    for q, M in pairs:
        gen = generic(kind, q, M)
        if len(gen) != len(model_rows):
            print("FAIL %s: %d rows, generic has %d" % (label, len(model_rows), len(gen)))
            fails += 1
            continue
        for mr, gr in zip(model_rows, gen):
            ok = True
            for _ in range(200):
                env = {n: random.uniform(-3, 3) for n in NAMES}
                env.update({"y": env[q], "g_y": env["g_" + q], "g_y_prev": env["g_" + q + "_prev"]})
                env["My"] = eval(M, {}, env)
                if abs(slack(mr, env) - slack(gr, env)) > 1e-9 * (1 + abs(slack(gr, env))):
                    ok = False
                    break
            checked += 1
            status = "ok  " if ok else "FAIL"
            fails += 0 if ok else 1
            print("%s %-18s q=%-2s model: %s\n%s %-18s      generic: %s" % (status, label, q, mr, " " * 4, "", gr))

# ---- Wiener single variance row ---------------------------------------------------------
vrow = [r for r in rows if r[0] == "eq:w_vrow"][0][1][0]
print("w_vrow parsed:", vrow)
ok = True
for xm, xr in ((0, 0), (1, 0), (0, 1)):
    for _ in range(2000):
        Vmax = random.uniform(0.1, 3)
        env = {"Vmax": Vmax, "xm": xm, "xr": xr, "v_prev": random.uniform(0, Vmax),
               "d_v": 0.0 if (xm or xr) else random.uniform(0, 1)}
        # lower bound on v from the single row, and from the generic carry and repair rows (a = 1)
        lb_single = max(0.0, env["v_prev"] + env["d_v"] - Vmax * xr)
        lb_generic = max(0.0, env["v_prev"] + env["d_v"] - Vmax * (xm + xr), env["v_prev"] - Vmax * (1 - xm))
        # the parsed row must give lb_single
        v = lb_single
        env["v"] = v
        if abs(lb_single - lb_generic) > 1e-12 or slack(vrow, env) < -1e-12:
            ok = False
        env["v"] = lb_single - 1e-6
        if lb_single > 1e-6 and slack(vrow, env) >= 0:
            ok = False
print(("ok  " if ok else "FAIL") + " eq:w_vrow equals the generic carry and repair rows (a = 1) at binary points")
fails += 0 if ok else 1
labels = [l for l, _ in rows]
print("rows parsed:", len(rows), "inequality comparisons:", checked, "failures:", fails)
print("labels:", " ".join(labels))
