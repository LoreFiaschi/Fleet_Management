"""Battery fade increments per line for the case study (my computation).
Inputs: case_study/zurich_line_features.csv (path as argument).
Formula and coefficients: see component_battery.md, section 2."""
import csv
import sys

sys.stdout.reconfigure(encoding="utf-8")

# energy per km, 12 m bus without electric heating, calibrated on VBZ standard bus (C3):
E0 = 0.66      # kWh/km at zero mean grade (intercept of the two-point fit)
KG = 0.065     # kWh/km per percentage point of mean absolute grade
P_HVAC_MEAN = 8.0 * 0.6    # kW: Göhlich average auxiliary 8 kW, minus 40 % heat pump (B6, B1)
P_HVAC_WINTER = 24.0 * 0.6  # kW: Göhlich heating at -10 C, minus 40 % heat pump
E_NOM = 264.0  # kWh, eCitaro 12 m of line 759 (case study)
K_EFC = 4.0e-5   # fade per equivalent full cycle (central)
C_CAL = 2.7e-5   # fade per day (central, about 1 % per year)
TAU = 0.20
CV = 0.29
BETA = 1.0 / (CV ** 2 * TAU)

rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8")))
print("beta (Gamma rate per unit fade) = %.1f" % BETA)
print("line | g % | v km/h | e mean | e winter | km/day wd | EFC/day | mu/day | mu per 1000 km | years to tau (wd every day)")
for r in rows:
    L = float(r["length_km_d0"]) + float(r["length_km_d1"])
    A = sum(float(r[k]) for k in ("ascent_m_d0", "descent_m_d0", "ascent_m_d1", "descent_m_d1"))
    g = A / (L * 1000.0) * 100.0
    v = float(r["mean_speed_kmh"])
    e_mean = E0 + KG * g + P_HVAC_MEAN / v
    e_win = E0 + KG * g + P_HVAC_WINTER / v
    km = float(r["km_per_vehicle_wd"])
    efc = km * e_mean / E_NOM
    mu_day = C_CAL + K_EFC * efc
    mu_km = mu_day / km * 1000.0
    yrs = TAU / mu_day / 365.0
    print("%s | %.2f | %.1f | %.2f | %.2f | %.1f | %.2f | %.2e | %.2e | %.1f" % (
        r["line"], g, v, e_mean, e_win, km, efc, mu_day, mu_km, yrs))
