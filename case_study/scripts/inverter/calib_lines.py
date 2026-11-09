"""Calibrated per-stop damage model and per-line increments.

Model: N_f from Semikron Danfoss AN 21-001 rev 02, Table 3 (sintered chips), k_thickness = 1 (IGBT).
Calibration (ASSUMPTIONS listed in the output file): Tj,min = 60 C, t_on = 10 s, n_c = 2 cycles per stop,
reference: design case reaches D = 1 at 60,000 operating hours (Semikron 2019) with 46 stops per hour.
"""
import math

KB = 1.38e-23
K0 = 273.15
A0, A1, T0, LAM, ALPHA, EA, C, G = 2.05e11, 60, 38, 17, -4.3, 2.54e-20, 1.44, -1.21
TJMIN, TON, NC = 60.0, 10.0, 2
GRAV, ACC = 9.81, 1.0


def nf(dtj, tjmin=TJMIN, ton=TON, kth=1.0):
    tjm = tjmin + dtj / 2 + K0
    beta = math.exp(-(dtj - T0) / LAM)
    return A0 * A1 ** beta * dtj ** (-beta) * dtj ** ALPHA * math.exp(EA / (KB * tjm)) \
        * (C + ton ** G) / (C + 2 ** G) * kth


# calibrate dTj_ref: NC * 46 / nf(dTj) = 1/60000
target = 1 / 60000
stops_h = 46.0
lo, hi = 10.0, 80.0
for _ in range(100):
    mid = (lo + hi) / 2
    if NC * stops_h / nf(mid) > target:
        hi = mid
    else:
        lo = mid
dref = (lo + hi) / 2
print(f"Calibrated dTj_ref = {dref:.2f} K, N_f = {nf(dref):.3g}, d_stop = {NC/nf(dref):.3g}")


def d_stop(r_mass, x_grade=0.0):
    """Mean damage per stop for mass ratio r and grade term x = g*gbar/a (half up, half down)."""
    up = NC / nf(dref * r_mass * (1 + x_grade))
    dn = NC / nf(dref * r_mass * max(1 - x_grade, 1e-3))
    return 0.5 * (up + dn)


LINES = {
    # stops_per_km, km/day wd, mean speed, ascent sum, length sum, stops d0, length d0
    "759": (1.99, 241.9, 21.8, 70 + 77, 14.08 + 13.56, 28, 14.08),
    "768": (2.31, 229.2, 21.1, 54 + 40, 5.66 + 5.62, 13, 5.66),
}
print("\nPer-line values; r = operating mass / 30 t; grade averaged over up/down halves")
print("line | r   | x_grade | d_stop   | d_km     | d_trip(d0) | d_weekday | life [y, 365 d]")
for name, (spk, kmd, v, asc, length, ns0, l0) in LINES.items():
    gbar = asc / (length * 1000 / 2)
    x = GRAV * gbar / ACC
    for r in (0.6, 0.7, 0.8, 0.9, 1.0):
        ds = d_stop(r, x)
        dkm = ds * spk
        dtrip = ds * ns0
        dday = dkm * kmd
        print(f" {name} | {r:.1f} | {x:.3f}   | {ds:.3g} | {dkm:.3g} | {dtrip:.3g}  | {dday:.3g} | {1/(dday*365):.3g}")

# Sensitivities at r = 1, flat
base = d_stop(1.0)
print("\nSensitivities of d_stop at r=1, flat (factor on damage):")
print(f"  Tj,min +10 K: {NC/nf(dref, tjmin=70)/base:.2f}; -10 K: {NC/nf(dref, tjmin=50)/base:.2f}; -30 K: {NC/nf(dref, tjmin=30)/base:.2f}")
print(f"  t_on 5 s: {NC/nf(dref, ton=5)/base:.2f}; 20 s: {NC/nf(dref, ton=20)/base:.2f}")
print(f"  dTj x1.1: {NC/nf(dref*1.1)/base:.2f}; x0.9: {NC/nf(dref*0.9)/base:.2f}")
print(f"  SiC k=0.33 same dTj: {NC/nf(dref, kth=0.33)/base:.2f}")

# Hoeffding check: daily increments over 15 years with support b = 3 * mean (ASSUMPTION)
for name, (spk, kmd, v, asc, length, ns0, l0) in LINES.items():
    gbar = asc / (length * 1000 / 2)
    x = GRAV * gbar / ACC
    mu_day = d_stop(0.8, x) * spk * kmd
    b = 3 * mu_day
    days = 0.8 / mu_day if mu_day > 0 else float("nan")
    for eps in (0.01, 0.001):
        kappa = math.log(1 / eps)
        # largest mean m with (tau - m)^2 >= kappa/2 * n b^2, n = m/mu_day, tau = 0.8
        tau = 0.8
        lo, hi = 0.0, tau
        for _ in range(200):
            m = (lo + hi) / 2
            n = m / mu_day
            if (tau - m) ** 2 >= kappa / 2 * n * b * b:
                lo = m
            else:
                hi = m
        print(f"Hoeffding line {name}, r=0.8, eps={eps}: mu_day={mu_day:.3g}, b={b:.3g}, "
              f"max mean before replacement = {lo:.4f} (tau={tau}); margin used = {(tau-lo)/tau:.1%}")
