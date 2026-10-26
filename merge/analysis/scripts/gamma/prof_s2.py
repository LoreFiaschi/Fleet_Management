import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main_calibration import calibrate_cell
from gamma_lib import sum_tail_fast, sum_tail
t0 = time.time()
r = calibrate_cell([0.1, 0.1], [10.0, 200.0], 1.0, 0.01, 20)
print(r, time.time() - t0)
t0 = time.time()
print(sum_tail_fast([(20*4.0, 10.0), (5*20.0, 200.0)], 1.0), time.time() - t0)
t0 = time.time()
print(sum_tail([(20*4.0, 10.0), (5*20.0, 200.0)], 1.0), time.time() - t0)
print(sum_tail_fast([(78.0, 2000.0), (74.0, 100.0)], 1.0))
print(sum_tail([(78.0, 2000.0), (74.0, 100.0)], 1.0))
