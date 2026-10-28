import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gamma_lib import negbin_pmf, gammaQ
t0 = time.time(); w, left = negbin_pmf(80.0, 0.05); print(len(w), left, sum(w), time.time()-t0)
t0 = time.time(); q = gammaQ(180.0, 200.0); print(q, time.time()-t0)
