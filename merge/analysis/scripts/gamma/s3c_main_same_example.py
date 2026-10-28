"""S3: main's calibration leaves the S3 example unchanged (one rate, so a' = a)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gamma_lib import shape_for_tail
from main_calibration import calibrate_cell
eps, rho, b, tau = 0.01, 0.5, 0.01, 1.0
A = shape_for_tail(eps, b, tau)
means = [A / b, rho * A / b]
ap, safe, bstar, Amax = calibrate_cell(means, [b, b], tau, eps, 20)
print(f"means per mission {means[0]:.6f}, {means[1]:.6f}; rate {b}; b*={bstar}")
print(f"a' = {ap[0]:.8f}, {ap[1]:.8f}; a = {A:.8f}, {rho*A:.8f}; m* = {safe}; A_max = {Amax:.8f} (A_hat = {A:.8f})")
