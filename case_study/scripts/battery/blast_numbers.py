"""My own evaluation of the coefficients printed in NREL BLAST-Lite model files
(lfp_gr_250AhPrismatic_2019, nmc111_gr_Sanyo2Ah_2014). No downloaded code is executed:
the formulas and numbers are retyped from the files that were read as text."""
import math


def Ua(soc):
    Xa = 8.5e-3 + soc * (0.78 - 8.5e-3)
    return (0.6379 + 0.5416 * math.exp(-305.5309 * Xa)
            + 0.044 * math.tanh(-1 * (Xa - 0.1958) / 0.1088)
            - 0.1978 * math.tanh((Xa - 1.0571) / 0.0854)
            - 0.6875 * math.tanh((Xa + 0.0117) / 0.0529)
            - 0.0175 * math.tanh((Xa - 0.5692) / 0.0875))


# LFP 250 Ah prismatic
P = dict(p1=8.37e4, p2=-5.21e3, p3=-3.56e3, pcal=0.526, p4=4.38e-8, p5=1.55e-8, p6=1.68e-7,
         p7=2.19e3, p8=1.55e5, pcyc=0.828)


def kcal_lfp(T, soc):
    return abs(P['p1']) * math.exp(P['p2'] / T) * math.exp(P['p3'] * Ua(soc) / T)


def kcyc_lfp(T, dod, crate):
    return (P['p4'] + P['p5'] * dod + P['p6'] * crate) * (math.exp(P['p7'] / T) + math.exp(-P['p8'] / T))


print("Ua(soc):", {s: round(Ua(s), 4) for s in (0.2, 0.3, 0.5, 0.6, 0.7, 0.9)})
for T_C in (10, 20, 25, 35):
    T = T_C + 273.15
    for soc in (0.5, 0.7):
        k = kcal_lfp(T, soc)
        print("LFP cal T=%d soc=%.1f kcal=%.3e  loss 1y=%.4f 5y=%.4f 10y=%.4f" % (
            T_C, soc, k, k * 365 ** P['pcal'], k * (5 * 365) ** P['pcal'], k * 3650 ** P['pcal']))
for T_C in (10, 25, 35):
    T = T_C + 273.15
    for dod in (0.4, 0.8):
        for cr in (0.1, 0.3, 0.5):
            k = kcyc_lfp(T, dod, cr)
            print("LFP cyc T=%d dod=%.1f C=%.1f kcyc=%.3e loss@1000EFC=%.4f @3000=%.4f @4000=%.4f" % (
                T_C, dod, cr, k, k * 1000 ** P['pcyc'], k * 3000 ** P['pcyc'], k * 4000 ** P['pcyc']))
# marginal loss per EFC at 2000 EFC: d/dN k N^p = k p N^(p-1)
k = kcyc_lfp(298.15, 0.8, 0.3)
for N in (100, 1000, 2000, 3000):
    print("LFP marginal per EFC at N=%d: %.3e" % (N, k * P['pcyc'] * N ** (P['pcyc'] - 1)))

# NMC Sanyo (Schmalstieg 2014 as coded by NREL)
soc_idx = [0, 0.008637153, 0.026779514, 0.044921875, 0.063064236, 0.081206597, 0.099348958, 0.117491319,
           0.135633681, 0.153776042, 0.171918403, 0.190060764, 0.208203125, 0.226345486, 0.244487847,
           0.262630208, 0.280772569, 0.298914931, 0.317057292, 0.335199653, 0.353342014, 0.371484375,
           0.389626736, 0.407769097, 0.425911458, 0.444053819, 0.462196181, 0.480338542, 0.498480903,
           0.516623264, 0.534765625, 0.552907986, 0.571050347, 0.589192708, 0.607335069, 0.625477431,
           0.643619792, 0.661762153, 0.679904514, 0.698046875, 0.716189236, 0.734331597, 0.752473958,
           0.770616319, 0.788758681, 0.806901042, 0.825043403, 0.843185764, 0.861328125, 0.879470486,
           0.897612847, 0.915755208, 0.933897569, 0.952039931, 0.970182292, 0.988324653, 0.998220486, 1]
ocv = [3.331, 3.345014187, 3.37917149, 3.411603677, 3.440585632, 3.466289865, 3.490268982, 3.511315401,
       3.529946658, 3.547197821, 3.561688798, 3.574972194, 3.586357962, 3.597053683, 3.605506753,
       3.613442288, 3.620342753, 3.626380661, 3.632073544, 3.63690387, 3.642079219, 3.646909545,
       3.652084894, 3.657605266, 3.663470662, 3.670198615, 3.677444104, 3.686759732, 3.696420384,
       3.708323686, 3.720572012, 3.734372943, 3.749553967, 3.765252525, 3.781123596, 3.797857224,
       3.814590852, 3.832532062, 3.849783226, 3.866861877, 3.884458064, 3.900846669, 3.917752809,
       3.934831461, 3.953462717, 3.971921462, 3.991415276, 4.011771649, 4.03195551, 4.05196686,
       4.070770628, 4.087849279, 4.104237884, 4.120108955, 4.135980025, 4.152541142, 4.160649189, 4.162]


def V(s):
    for i in range(len(soc_idx) - 1):
        if soc_idx[i] <= s <= soc_idx[i + 1]:
            a = (s - soc_idx[i]) / (soc_idx[i + 1] - soc_idx[i])
            return ocv[i] + a * (ocv[i + 1] - ocv[i])
    return ocv[-1]


def vrms(lo, hi, n=200):
    vals = [V(lo + (hi - lo) * i / n) for i in range(n + 1)]
    return math.sqrt(sum(v * v for v in vals) / len(vals))


cap = 2.15
for T_C in (10, 25, 35):
    T = T_C + 273.15
    for soc in (0.5, 0.7, 0.9):
        a = (7.543 * V(soc) - 23.75) * 1e6 * math.exp(-6976 / T)
        print("NMC cal T=%d soc=%.1f V=%.3f alpha=%.3e loss 1y=%.4f 5y=%.4f 10y=%.4f" % (
            T_C, soc, V(soc), a, a * 365 ** 0.75, a * 1825 ** 0.75, a * 3650 ** 0.75))
for lo, hi in ((0.3, 0.7), (0.1, 0.9), (0.1, 0.95), (0.2, 0.8)):
    vr = vrms(lo, hi)
    dod = hi - lo
    b = 7.348e-3 * (vr - 3.667) ** 2 + 7.6e-4 + 4.081e-3 * dod
    for N in (500, 1000, 2000, 3000, 4000):
        Q = N * 2 * cap
        print("NMC cyc window %.2f-%.2f Vrms=%.3f beta=%.3e EFC=%d loss=%.4f" % (lo, hi, vr, b, N, b * math.sqrt(Q)))
