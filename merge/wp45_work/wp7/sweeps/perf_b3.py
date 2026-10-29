import sys, time
sys.path.insert(0, r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge\src")
import numpy as np
from fleet_management.config import load_config
from fleet_management import greedy_warm_start as g

F, M, L = 8, 4, 5
rng = np.random.default_rng(1)
mu = rng.uniform(0.01, 0.05, size=(F, 1, M))
data = dict(F=F, M=M, H=[10, 30], model="gaussian", tau=1.0, epsilon=0.1, rho=0.5, mu_0=0.05, v_0=0.0,
            mu=mu.tolist(), v=(mu ** 2 / 16).tolist(), reliability_impl="tangent",
            C_M=1.0, C_R=1.0, C_rep=5.0, C_D=0.0, verbose=0)
cfg = load_config(data)
t = time.time()
source, report = g.build_greedy_warm_start(cfg)
print("seconds", round(time.time() - t, 2), report.as_dict() | {"assignment_order": "..."})
