"""Tyre wear increment per vehicle-day for the case-study lines.

Usage: python line_increments.py <zurich_line_features.csv>

Default formula:  mu = r_axle * f_beb * km                       (mm per vehicle-day)
Optional route factor g_j = (1 - phi) + phi * X_j / Xbar, X_j = s_j a v_j^2 + g^2 theta_j (A_j + D_j) / L_j
(longitudinal slip-energy index per metre; assumptions marked below).
"""
import sys
import csv

sys.stdout.reconfigure(encoding="utf-8")
path = sys.argv[1]

R_STEER = 0.15 / 1000.0   # mm per km, S1 Table 6, 275/70 R22.5 new front
R_DRIVE = 0.16 / 1000.0   # mm per km, S1 Table 6, 275/70 R22.5 new rear
F_BEB = 1.35              # R8 (+30 % to +38.8 % tyre cost), central
PHI = 0.30                # share of manoeuvre wear, derived from S1 Table 5 (18-33 %)
A_ACC = 1.0               # m/s2, ASSUMPTION: mean acceleration and deceleration at stops
G = 9.81
TAU_STEER, TAU_DRIVE = 17.5, 18.4   # mm, h0 - 2.0 mm (S8, S1)
C = 0.30                  # CV of the wear rate between tyres (S1 section 4)

rows = list(csv.DictReader(open(path, encoding="utf-8")))
lines = []
for r in rows:
    L = float(r["length_km_d0"]) + float(r["length_km_d1"])           # km, both directions
    AD = sum(float(r[k]) for k in ("ascent_m_d0", "descent_m_d0", "ascent_m_d1", "descent_m_d1"))
    s = float(r["stops_per_km"])
    v = float(r["mean_speed_kmh"]) / 3.6                                 # m/s, proxy for cruise speed
    theta = 0.5 * max(float(r["max_grade_pct_d0"]), float(r["max_grade_pct_d1"])) / 100.0  # ASSUMPTION
    x_stop = s / 1000.0 * A_ACC * v * v              # m2/s4 per metre: s stops/km, each a*v^2 of int F^2/m^2 ds
    x_grade = G * G * theta * AD / (L * 1000.0)       # m2/s4: g^2 theta * (A+D) / distance
    km_day = float(r["km_per_vehicle_wd"]) if r["km_per_vehicle_wd"] else 0.0
    veh_km = float(r["veh_km_wd"]) if r["veh_km_wd"] else 0.0
    lines.append(dict(line=r["line"], s=s, v=v * 3.6, ADkm=AD / L, theta=theta, x_stop=x_stop,
                      x_grade=x_grade, X=x_stop + x_grade, km_day=km_day, veh_km=veh_km))

xbar = sum(l["X"] * l["veh_km"] for l in lines) / sum(l["veh_km"] for l in lines)
print(f"Xbar (veh-km weighted) = {xbar:.4f} m2/s4")
print("line | stops/km | v km/h | (A+D)/L m/km | theta | X_stop | X_grade | g_j | km/veh-day | mu_steer mm/day (g=1) | mu_drive mm/day (g=1) | mu_drive with g_j | gamma beta steer/drive (1/mm) | v_inc drive mm2 (g=1)")
for l in lines:
    g = (1 - PHI) + PHI * l["X"] / xbar
    ms = R_STEER * F_BEB * l["km_day"]
    md = R_DRIVE * F_BEB * l["km_day"]
    beta_s = 1.0 / (C * C * TAU_STEER)
    beta_d = 1.0 / (C * C * TAU_DRIVE)
    vinc = C * C * TAU_DRIVE * md
    print(f"{l['line']} | {l['s']:.2f} | {l['v']:.1f} | {l['ADkm']:.1f} | {l['theta']:.3f} | {l['x_stop']:.4f} | "
          f"{l['x_grade']:.4f} | {g:.2f} | {l['km_day']:.1f} | {ms:.4f} | {md:.4f} | {md * g:.4f} | "
          f"{beta_s:.3f}/{beta_d:.3f} | {vinc:.5f}")
# life checks
for name, tau, rr in (("steer", TAU_STEER, R_STEER), ("drive", TAU_DRIVE, R_DRIVE)):
    print(f"{name}: rate {rr * F_BEB * 1000:.3f} mm/1000 km, life to tau {tau / (rr * F_BEB):.0f} km")
