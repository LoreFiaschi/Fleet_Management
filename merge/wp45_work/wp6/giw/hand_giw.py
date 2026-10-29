"""Expected arrays of the GIW tests: the exact day maps of the spec on fixed schedules.

Usage: python hand_giw.py   (prints the arrays that the plan's tests use)

Day index k = 0..T-1 (spec day k + 1); index -1 holds the initial values and anchors.
Activities per train and day: "M" mission, "R" repair (depot), "X" replacement (depot),
"I" idle. Laws: ardinf, ard1 (spec Eqs. g_map_mu, g_map_v, ig_map, w_map_v, a_z).
"""
import numpy as np
from scipy.special import ndtri

from ig_cap import ig_mean_cap


def run(days, law, *, mu0, v0, mu_inc, v_inc, rho, mu_new, v_new, wiener=False, sigma=0.0):
    mu, v, g_mu, g_v, z = [], [], [], [], []
    m_prev, v_prev, gm_prev, gv_prev = mu0, v0, mu0, v0      # starting anchors = initial values
    a_mu, a_v = 1.0 - rho, (1.0 - rho) ** 2
    for act in days:
        zk = 0.0
        if act == "M":
            m, vv = m_prev + mu_inc, v_prev + (sigma ** 2 if wiener else v_inc)
            gm, gv = gm_prev, gv_prev
        elif act == "I":
            m, vv, gm, gv = m_prev, v_prev, gm_prev, gv_prev
        elif act == "R":
            anchor_mu = gm_prev if law == "ard1" else 0.0
            anchor_v = gv_prev if law == "ard1" else 0.0
            m = a_mu * m_prev + (1.0 - a_mu) * anchor_mu
            vv = v_prev if wiener else a_v * v_prev + (1.0 - a_v) * anchor_v
            zk = rho * (m_prev - anchor_mu)
            gm, gv = m, vv
        else:                                                 # "X"
            m, vv = mu_new, v_new
            gm, gv = m, vv
        mu.append(m); v.append(vv); g_mu.append(gm); g_v.append(gv); z.append(zk)
        m_prev, v_prev, gm_prev, gv_prev = m, vv, gm, gv
    return {"mu": mu, "v": v, "g_mu": g_mu, "g_v": g_v, "z": z}


def show(title, out):
    print(title)
    for key, values in out.items():
        print(f"  {key:5s}", np.round(values, 10).tolist())


if __name__ == "__main__":
    # Gaussian: train 0 M R M X, train 1 I M I M with zero increments
    for law in ("ard1", "ardinf"):
        common = dict(mu0=0.2, v0=0.0, rho=0.5, mu_new=0.1, v_new=0.0)
        show(f"gaussian {law}, train 0", run("MRMX", law, mu_inc=0.3, v_inc=0.01, **common))
        show(f"gaussian {law}, train 1", run("IMIM", law, mu_inc=0.0, v_inc=0.0, **common))
    # Wiener: sigma = 0.1 both trains; train 0 M R M X (mu_inc 0.3), train 1 I M X M (mu_inc 0)
    for law in ("ard1", "ardinf"):
        common = dict(mu0=0.2, v0=0.0, rho=0.5, mu_new=0.1, v_new=0.0, v_inc=None,
                      wiener=True, sigma=0.1)
        show(f"wiener {law}, train 0", run("MRMX", law, mu_inc=0.3, **common))
        show(f"wiener {law}, train 1", run("IMXM", law, mu_inc=0.0, **common))
    # Inverse Gaussian: train 0 M R M X (mu_inc 0.15), train 1 I M I M (mu_inc 0)
    for law in ("ard1", "ardinf"):
        common = dict(mu0=0.1, v0=0.0, rho=0.5, mu_new=0.05, v_new=0.0, v_inc=0.0)
        show(f"inverse_gaussian {law}, train 0", run("MRMX", law, mu_inc=0.15, **common))
        show(f"inverse_gaussian {law}, train 1", run("IMIM", law, mu_inc=0.0, **common))
    cap = ig_mean_cap(0.5, 1.0, 0.1)
    print(f"IG cap mu_bar(eta=0.5, tau=1, eps=0.1) = {cap:.10f}")
    print(f"  boundary increments (mu_0 = 0.1): 0.99 cap -> {0.99 * cap - 0.1:.10f}, "
          f"1.01 cap -> {1.01 * cap - 0.1:.10f}")
    kappa2 = ndtri(0.9) ** 2
    print(f"kappa_G^2(eps=0.1) = {kappa2:.10f}; V_max(tau=1) = {1 / kappa2:.10f}")
    t = 0.5
    print(f"tangent at 0.5 tau: cap(mu) = {t * t / kappa2:.10f} - {2 * t / kappa2:.10f} (mu - 0.5)")
    print(f"  cap at mu = 0.65: {t * t / kappa2 - 2 * t / kappa2 * 0.15:.10f}")
