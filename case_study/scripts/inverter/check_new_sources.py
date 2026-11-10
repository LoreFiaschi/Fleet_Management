"""Cross-checks for the inverter dossier, written 10 Nov 2026 (scratchpad copy; the project scripts are unchanged).

1. S5 sintered-chip model against the Semikron 2011 example (dTj = 51 K, Tjm = 85 C, 760,000 cycles).
2. The stop rate of the calibration: 46 stops per hour from the line data, against the 10 km per
   operating hour that Semikron (S2) assumes for a city bus.
The function nf() is a copy of nf() in case_study/scripts/inverter/calib_lines.py.
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


# 1. Semikron 2011 example: Tjm = 85 C, dTj = 51 K -> Tjmin = 59.5 C
print("1. S5 sintered model at dTj = 51 K, Tjm = 85 C (maker's example: 760,000 cycles)")
for ton in (2, 10, 35):
    n = nf(51, tjmin=85 - 25.5, ton=ton)
    print(f"   t_on = {ton:2d} s: N_f = {n:.3g}; ratio model / maker = {n / 760000:.2f}")
print(f"   check: 760,000 cycles x 90 s = {760000 * 90 / 3600:.0f} h = {760000 * 90 / 3600 / 8760:.2f} years")

# 2. Stop rate. Line data from zurich_line_features.csv (as in calib_lines.py)
LINES = {"759": (1.99, 21.8, 70 + 77, 14.08 + 13.56), "768": (2.31, 21.1, 54 + 40, 5.66 + 5.62)}
print("\n2. Stops per operating hour")
for name, (spk, v, asc, length) in LINES.items():
    print(f"   line {name}: stops/km x mean speed = {spk} x {v} = {spk * v:.1f} per h; "
          f"at 10 km/h (S2): {spk * 10:.1f} per h")
r46 = sum(spk * v for spk, v, *_ in LINES.values()) / 2
r10 = sum(spk * 10 for spk, *_ in LINES.values()) / 2
print(f"   mean of the two lines: {r46:.2f} per h (line speeds); {r10:.2f} per h (10 km/h)")


def calibrate(stops_h, life_h=60000.0):
    lo, hi = 10.0, 80.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if NC * stops_h / nf(mid) > 1 / life_h:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def d_stop(dref, x):
    up = NC / nf(dref * (1 + x))
    dn = NC / nf(dref * max(1 - x, 1e-3))
    return 0.5 * (up + dn)


print("\n   Calibration and damage per km (r = 1, grade as in the dossier)")
for label, rate in (("46 stops/h (dossier)", 46.0), (f"{r10:.1f} stops/h (10 km/h)", r10)):
    dref = calibrate(rate)
    out = [f"   {label}: dTj_ref = {dref:.2f} K, d_stop(flat) = {NC / nf(dref):.3g}"]
    for name, (spk, v, asc, length) in LINES.items():
        x = GRAV * asc / (length * 1000 / 2) / ACC
        out.append(f"line {name}: d_km = {d_stop(dref, x) * spk:.3g}")
    print("; ".join(out))
