"""Damage per stop, per km and per day for lines 759 and 768 under the Semikron Danfoss
power-cycle model (AN 21-001 rev 02, Tables 1-3), with comparison models.

All inputs that are assumptions are marked ASSUMPTION.
"""
import math

KB = 1.38e-23
K0 = 273.15

SETS = {
    # A0, A1, T0, lam, alpha, Ea, C, gamma
    "baseplate_solder": (2.9e9, 60, 40, 17, -4.3, 4.50e-20, 1.0, -0.75),
    "nobase_solder": (2.90e9, 60, 40, 17, -4.3, 4.50e-20, 0.38, -0.7),
    "nobase_sinter": (2.05e11, 60, 38, 17, -4.3, 2.54e-20, 1.44, -1.21),
}


def nf_semikron(dtj, tjmin_c, ton, setname, kth=1.0, low_ext=True):
    A0, A1, T0, lam, alpha, Ea, C, g = SETS[setname]
    tjm = tjmin_c + dtj / 2 + K0
    beta = math.exp(-(dtj - T0) / lam)
    ext = (A1 ** beta) * dtj ** (-beta) if low_ext else 1.0
    return A0 * ext * dtj ** alpha * math.exp(Ea / (KB * tjm)) * (C + ton ** g) / (C + 2 ** g) * kth


def nf_hp2(dtj, tjmin_c, ton):
    # Lu et al. 2023 Eq. (13)-(14), Table V; exponent -0.3 inferred from continuity.
    tjmax = tjmin_c + dtj + K0
    kb_ev = 86e-6
    if dtj <= 45:
        a, b = 5.4e14, 7.0
    else:
        a, b = 2e9, 3.7
    if ton < 0.1:
        f = 2.25
    elif ton < 60:
        f = (ton / 1.5) ** -0.3
    else:
        f = 0.33
    return a * dtj ** (-b) * math.exp(0.215 / (kb_ev * tjmax)) * f


print("Check S5 example: dTj=60, Tjmin=40C, ton=2s, baseplate:",
      f"{nf_semikron(60, 40, 2, 'baseplate_solder'):.3g}")
print("Check S5 example: dTj=60, Tjmax=150C, ton=2s, baseplate:",
      f"{nf_semikron(60, 90, 2, 'baseplate_solder'):.3g}")

TJMIN = 60.0  # ASSUMPTION: coolant/heatsink 60 C as in Lu et al. 2023 (S3)
TON = 10.0    # ASSUMPTION: heating time of one acceleration of a bus, about 10 s
NC = 2        # ASSUMPTION: two thermal cycles per stop (acceleration and braking), S3 Fig. 19

print("\nCycles to failure N_f (Tj,min = 60 C, t_on = 10 s)")
print(" dTj | sinter IGBT | sinter SiC(k=.33) | solder base | HP2 (S3) | ext factor")
for dtj in (10, 15, 20, 25, 30, 40, 50, 60):
    ns = nf_semikron(dtj, TJMIN, TON, "nobase_sinter")
    nsic = nf_semikron(dtj, TJMIN, TON, "nobase_sinter", kth=0.33)
    nb = nf_semikron(dtj, TJMIN, TON, "baseplate_solder")
    nh = nf_hp2(dtj, TJMIN, TON)
    A0, A1, T0, lam, *_ = SETS["nobase_sinter"]
    beta = math.exp(-(dtj - T0) / lam)
    ext = A1 ** beta * dtj ** (-beta)
    print(f"{dtj:4d} | {ns:11.3g} | {nsic:17.3g} | {nb:11.3g} | {nh:8.3g} | {ext:6.2f}")

# local exponent q = -dlnN/dlnDT (sinter set)
print("\nLocal exponent q = -dln(N_f)/dln(dTj), sinter set (includes T_jm change):")
for dtj in (15, 20, 30, 40):
    h = 1e-3
    n1 = nf_semikron(dtj * (1 - h), TJMIN, TON, "nobase_sinter")
    n2 = nf_semikron(dtj * (1 + h), TJMIN, TON, "nobase_sinter")
    q = -(math.log(n2) - math.log(n1)) / (math.log(1 + h) - math.log(1 - h))
    print(f"  dTj={dtj}: q={q:.2f}")

# Arrhenius sensitivity of Tj,min
print("\nFactor on damage for Tj,min +10 K (sinter set, dTj=30):",
      f"{nf_semikron(30, 60, TON, 'nobase_sinter') / nf_semikron(30, 70, TON, 'nobase_sinter'):.3f}")
print("Factor on damage for Tj,min +10 K (solder set, dTj=30):",
      f"{nf_semikron(30, 60, TON, 'baseplate_solder') / nf_semikron(30, 70, TON, 'baseplate_solder'):.3f}")
print("Factor on damage for t_on 5 s vs 10 s (sinter):",
      f"{nf_semikron(30, 60, 10, 'nobase_sinter') / nf_semikron(30, 60, 5, 'nobase_sinter'):.3f}")
print("Factor on damage for t_on 20 s vs 10 s (sinter):",
      f"{nf_semikron(30, 60, 10, 'nobase_sinter') / nf_semikron(30, 60, 20, 'nobase_sinter'):.3f}")

# Lines (from case_study/zurich_line_features.csv, direction 0 and weekday)
LINES = {
    # name: (length_km_d0, n_stops_d0, length_km_d1, n_stops_d1, stops_per_km, mean_speed_kmh, km_per_vehicle_wd, ascent_d0, ascent_d1)
    "759": (14.08, 28, 13.56, 27, 1.99, 21.8, 241.9, 70, 77),
    "768": (5.66, 13, 5.62, 13, 2.31, 21.1, 229.2, 54, 40),
}
print("\nDamage per stop, per km and per weekday (sinter IGBT, NC=2 cycles per stop)")
for dtj in (15, 20, 30, 40):
    d_stop = NC / nf_semikron(dtj, TJMIN, TON, "nobase_sinter")
    for name, (L0, s0, L1, s1, spk, v, kmday, a0, a1) in LINES.items():
        d_km = d_stop * spk
        d_day = d_km * kmday
        hours = kmday / v
        life_years = 1 / (d_day * 365)
        print(f"  dTj={dtj:2d} line {name}: d_stop={d_stop:.3g} d_km={d_km:.3g} d_day={d_day:.3g} "
              f"(oper. h/day={hours:.1f}) life={life_years:.3g} y")

# Design-target route (S2): 60000 operating hours
print("\nDesign-target route: D per operating hour = 1/60000 =", f"{1/60000:.3g}")
for name, (L0, s0, L1, s1, spk, v, kmday, a0, a1) in LINES.items():
    hours = kmday / v
    print(f"  line {name}: hours/day={hours:.2f}, D/day <= {hours/60000:.3g}, D/km <= {1/(60000*v):.3g}, "
          f"life >= {60000/hours/365:.1f} y; stops per hour = {spk*v:.1f}")

# Mean-grade factor (my construction)
print("\nMean uphill grade (ascent / half length) and force factor (1 + g*grade/a), a = 1.0 m/s2:")
for name, (L0, s0, L1, s1, spk, v, kmday, a0, a1) in LINES.items():
    gbar = (a0 + a1) / ((L0 + L1) * 1000 / 2)
    print(f"  line {name}: mean uphill grade = {100*gbar:.2f} %, factor = {1 + 9.81*gbar/1.0:.3f}")

# Threshold from Weibull life scatter (15 % curve), tau = (ln(1/(1-p))/ln(1/0.85))^(1/k)
print("\nThreshold tau on the 15 % curve for unit-failure probability p and Weibull shape k:")
for k in (3, 5, 8, 13.5, 24):
    row = []
    for p in (0.001, 0.01, 0.05, 0.10):
        tau = (math.log(1 / (1 - p)) / math.log(1 / 0.85)) ** (1 / k)
        row.append(f"p={p}: {tau:.3f}")
    print(f"  k={k}: " + ", ".join(row))
