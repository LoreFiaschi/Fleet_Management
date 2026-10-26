"""Writer A: spot checks for sec:encoding and sec:loop (standard library only).

Checks:
 1. Domination (Proposition prop:a_domination) and perpetual safety
    (Theorem thm:a_loop): random schedules, random slack, both laws.
 2. Example "one quantity, accepted only by the one-sided rows".
 3. Example "two quantities that move in opposite directions, rejected".
 4. Remark "why the loop row covers the full state" (ARD1 example).
 5. Remark "no loop row on a scalar bound" (Cantelli example).
 6. Proposition prop:a_ard1_replacement: no replacement in the operating phase
    => loop rows fail under ARD1 with rho < 1; boundary case rho = 1.
 7. Validity of the big-M constants (relaxed rows cut no point).
"""
import random

random.seed(12345)


def day_map(state, day, law, a_of, s_new):
    """Exact day map on a dict state {q: value, 'g_'+q: anchor}."""
    out = dict(state)
    kind = day["kind"]
    for q, a in a_of.items():
        s = state[q]
        g = state.get("g_" + q, 0.0)
        if kind == "none":
            out[q] = s + day["inc"][q]
        elif kind == "repair":
            if a is None:            # quantity without a repair map
                out[q] = s
            elif law == "ardinf":
                out[q] = a * s
            else:
                out[q] = a * s + (1 - a) * g
                out["g_" + q] = out[q]
        elif kind == "replace":
            out[q] = s_new[q]
            if law == "ard1" and a is not None:
                out["g_" + q] = s_new[q]
    return out


def leq(x, y, tol=1e-12):
    return all(x[k] <= y[k] + tol for k in x)


def random_schedule(n_days, p_rep=0.25, p_repl=0.05):
    days = []
    for _ in range(n_days):
        u = random.random()
        if u < p_repl:
            days.append({"kind": "replace"})
        elif u < p_repl + p_rep:
            days.append({"kind": "repair"})
        else:
            days.append({"kind": "none",
                         "inc": {"mu": random.uniform(0, 0.1),
                                 "v": random.uniform(0, 0.01)}})
    return days


def check1(n_trials=3000):
    """Domination and chaining with random slack, both laws."""
    worst = 0.0
    n_loop_ok = 0
    for t in range(n_trials):
        law = random.choice(["ardinf", "ard1"])
        rho = random.uniform(0.05, 1.0)
        a_of = {"mu": 1 - rho, "v": (1 - rho) ** 2}
        s_new = {"mu": random.uniform(0, 0.05), "v": random.uniform(0, 0.001)}
        H1, H2 = random.randint(1, 6), random.randint(1, 6)
        sched = random_schedule(H1 + H2)
        x0 = {"mu": random.uniform(0, 0.3), "v": random.uniform(0, 0.01)}
        if law == "ard1":
            x0["g_mu"] = random.uniform(0, x0["mu"])
            x0["g_v"] = random.uniform(0, x0["v"])
        # tracked trajectory with slack: state coordinates >= exact image;
        # anchors follow the anchor rows (equal to the tracked state on an
        # intervention day, unchanged otherwise); replacement rows exact.
        xi = [x0]
        zeta = [x0]
        for d in sched:
            img = day_map(xi[-1], d, law, a_of, s_new)
            nxt = dict(img)
            if d["kind"] != "replace":
                for q in a_of:
                    nxt[q] = img[q] + (random.uniform(0, 0.05) if random.random() < 0.5 else 0.0)
            if law == "ard1" and d["kind"] in ("repair", "replace"):
                for q in a_of:
                    nxt["g_" + q] = nxt[q]
            xi.append(nxt)
            zeta.append(day_map(zeta[-1], d, law, a_of, s_new))
        for k in range(len(xi)):
            assert leq(zeta[k], xi[k]), ("domination fails", t, k)
        # loop row on the full state
        if not leq(xi[H1 + H2], xi[H1]):
            continue
        n_loop_ok += 1
        # endless schedule: reference trajectory over 30 repetitions
        z = zeta[H1]
        for n in range(30):
            for h in range(1, H2 + 1):
                z = day_map(z, sched[H1 + h - 1], law, a_of, s_new)
                assert leq(z, xi[H1 + h]), ("chaining fails", t, n, h)
                worst = max(worst, max(z[k] - xi[H1 + h][k] for k in z))
    return n_loop_ok, worst


def check2():
    phi = lambda y: 0.5 * (y + 0.2)
    ok = abs(phi(0.1) - 0.15) < 1e-12 and phi(0.2) <= 0.2 + 1e-12
    traj = [0.1]
    y = 0.1
    peaks = []
    for _ in range(40):
        peak = y + 0.2
        y = 0.5 * peak
        traj += [peak, y]
        peaks.append(peak)
    first5 = [round(v, 6) for v in traj[:5]]
    removals = [0.5 * p for p in peaks[:3]]
    return ok, first5, round(y, 6), round(peaks[-1], 6), removals


def check3():
    kc2, tau = 9.0, 1.0
    phi = lambda m, v: (0.5 * m + 0.1, 0.25 * v + 0.00225)
    m, v = 0.5, 0.0
    peaks = []
    for _ in range(60):
        pm, pv = m + 0.2, v + 0.009
        peaks.append((pm, pv))
        m, v = phi(m, v)
    safe = all(kc2 * pv <= (tau - pm) ** 2 + 1e-15 and pm <= tau for pm, pv in peaks)
    later = peaks[1:]
    max_m_later = max(p[0] for p in later)
    max_v_later = max(p[1] for p in later)
    # one-sided: y_v >= 0.003 needed; tracked peak variance >= 0.012
    yv_min = 0.00225 / 0.75
    tracked_peak_ok = kc2 * (yv_min + 0.009) <= (tau - 0.7) ** 2
    exact_img = phi(0.5, 0.0)
    return (safe, peaks[0], round(max_m_later, 6), round(max_v_later, 6),
            round(yv_min, 6), tracked_peak_ok, exact_img)


def check4():
    rho = 0.5
    mu, g = 0.4, 0.1
    out = []
    peaks = []
    after_repair = []
    for rep in range(4):
        mu += 0.1
        peaks.append(mu)
        mu = (1 - rho) * mu + rho * g
        g = mu
        after_repair.append(mu)
        mu += 0.05
        out.append((round(mu, 6), round(g, 6)))
    steps = [round(after_repair[i + 1] - after_repair[i], 6) for i in range(3)]
    return out[0], [round(p, 6) for p in peaks], steps


def check5():
    tau, rho = 10.0, 0.5
    U = lambda m, v: v / (v + (tau - m) ** 2)
    m, v = 4.0, 9.0
    us = [U(m, v)]
    for _ in range(200):
        m, v = (1 - rho) * m + 3.0, (1 - rho) ** 2 * v + 3.99
        us.append(U(m, v))
    return [round(u, 4) for u in us[:3]], round(us[-1], 4)


def check6(n_trials=3000):
    """ARD1, rho < 1, no replacement in the operating phase: some loop row fails
    for every random choice of slack; rho = 1 boundary admits a periodic cycle."""
    fails = 0
    for t in range(n_trials):
        rho = random.uniform(0.05, 0.95)
        H1, H2 = random.randint(1, 5), random.randint(2, 6)
        days = []
        for k in range(H1 + H2):
            if random.random() < 0.3:
                days.append("repair")
            else:
                days.append(random.uniform(0.0, 0.1))
        # ensure one positive increment in the operating phase
        days[H1 + random.randrange(H2)] = random.uniform(0.01, 0.1)
        mu0 = random.uniform(0, 0.3)
        mu, g = mu0, random.uniform(0, mu0)
        traj = [(mu, g)]
        for d in days:
            if d == "repair":
                mu = (1 - rho) * mu + rho * g + (random.uniform(0, 0.02) if random.random() < 0.5 else 0.0)
                g = mu
            else:
                mu = mu + d + (random.uniform(0, 0.02) if random.random() < 0.5 else 0.0)
            traj.append((mu, g))
        mT, gT = traj[H1 + H2]
        mH, gH = traj[H1]
        if not (mT <= mH + 1e-12 and gT <= gH + 1e-12):
            fails += 1
    # boundary rho = 1: cycle mission 0.1, repair -> back to the anchor
    mu, g = 0.2, 0.2
    start = (mu, g)
    mu += 0.1
    mu = 0.0 * mu + 1.0 * g
    g = mu
    periodic = (abs(mu - start[0]) < 1e-12 and abs(g - start[1]) < 1e-12)
    return fails, n_trials, periodic


def check7(n_trials=20000):
    """Relaxed big-M rows cut no point of [0, M]."""
    M = 1.0
    bad = 0
    for _ in range(n_trials):
        s_prev, g_prev, s, g, s_new = (random.uniform(0, M) for _ in range(5))
        a = random.uniform(0, 1)
        psi = a * s_prev + (1 - a) * g_prev
        # depot day: delta = 0 on intervention days
        for xm, xr in ((1, 0), (0, 1)):
            if not s >= s_prev + 0.0 - M * (xm + xr) - 1e-12:
                bad += 1
        if not s >= psi - M * 1 - 1e-12:      # repair row with xm = 0
            bad += 1
        if not (s_new - M <= s + 1e-12 and s <= s_new + M + 1e-12):  # xr = 0
            bad += 1
        if not (abs(g - s) <= M + 1e-12 and abs(g - g_prev) <= M + 1e-12):
            bad += 1
    return bad


def check8(n_trials=3000):
    """(M2) with <=: theta_0 <= xi_0 and theta_k <= phi_k(theta_{k-1}) (as for the
    Chernoff quantity K and a starting anchor above the true one) give
    theta_k <= zeta_k on every day of 30 repetitions."""
    worst = -1.0
    for t in range(n_trials):
        law = random.choice(["ardinf", "ard1"])
        rho = random.uniform(0.05, 1.0)
        a_of = {"mu": 1 - rho, "v": (1 - rho) ** 2}
        s_new = {"mu": random.uniform(0, 0.05), "v": random.uniform(0, 0.001)}
        H1, H2 = random.randint(1, 6), random.randint(1, 6)
        sched = random_schedule(H1 + H2)
        x0 = {"mu": random.uniform(0, 0.3), "v": random.uniform(0, 0.01)}
        if law == "ard1":
            x0["g_mu"] = random.uniform(0, x0["mu"])
            x0["g_v"] = random.uniform(0, x0["v"])
        theta = {k: v * random.uniform(0.5, 1.0) for k, v in x0.items()}
        zeta = dict(x0)
        days = sched[:H1] + sched[H1:] * 31
        for d in days:
            zeta = day_map(zeta, d, law, a_of, s_new)
            img = day_map(theta, d, law, a_of, s_new)
            theta = {k: v - (random.uniform(0, 0.3) * v if random.random() < 0.3 else 0.0)
                     for k, v in img.items()}
            worst = max(worst, max(theta[k] - zeta[k] for k in theta))
    return worst


if __name__ == "__main__":
    n_ok, worst = check1()
    print("1. domination + chaining: loop-feasible trials =", n_ok,
          "; max (reference - tracked) =", worst, "(must be <= 0)")
    print("2. scalar example:", check2())
    print("3. two-quantity example:", check3())
    print("4. full-state loop example (end of 1st phase, peaks, rise per repetition):", check4())
    print("5. scalar-bound example: U at H1, T, T+H2 =", *check5())
    print("6. ARD1 replacement: loop fails / trials, rho=1 periodic:", check6())
    print("7. big-M validity, violations:", check7())
    print("8. (M2) with <=: max (true - reference) over 31 repetitions =", check8(), "(must be <= 0)")
