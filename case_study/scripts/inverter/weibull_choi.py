"""Weibull shape from B1, B10 and B63.2 of Choi et al. 2017, Table IV (nominal values)."""
import math

rows = {
    "10 s": (107413, 127800, 150930, 145264),
    "5 s": (135025, 154767, 176372, 171021),
    "2 s": (157749, 173938, 190995, 186728),
    "1 s": (165927, 183120, 201252, 196717),
    "0.8 s": (174278, 192225, 211138, 206407),
    "0.59 s": (178710, 205884, 235770, 228378),
}
for k, (b1, b10, eta, mttf) in rows.items():
    beta10 = math.log(-math.log(0.9)) / math.log(b10 / eta)
    beta1 = math.log(-math.log(0.99)) / math.log(b1 / eta)
    m = eta * math.gamma(1 + 1 / beta10)
    sd = eta * math.sqrt(math.gamma(1 + 2 / beta10) - math.gamma(1 + 1 / beta10) ** 2)
    print(f"{k:7s} beta(B10)={beta10:5.1f} beta(B1)={beta1:5.1f} MTTF_model={m:9.0f} table={mttf} CV={sd/m:.3f} B10/B50={b10/(eta*math.log(2)**(1/beta10)):.3f}")

# CV of a Weibull for some shapes
for b in (3, 5, 8, 13.5, 24):
    g1 = math.gamma(1 + 1 / b)
    g2 = math.gamma(1 + 2 / b)
    print(f"shape {b}: CV = {math.sqrt(g2 - g1**2)/g1:.3f}, B10/B50 = {(math.log(1/0.9)/math.log(2))**(1/b):.3f}")
