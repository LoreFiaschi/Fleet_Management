import math
from scipy.special import ndtri
def f(x, tau, c2, c1): d = tau - x; return c2*d*d + c1*d
def fp(x, tau, c2, c1): return -2*c2*(tau - x) - c1
def tangent(mu, r, tau, c2, c1): return f(r, tau, c2, c1) + fp(r, tau, c2, c1)*(mu - r)
def pwl(mu, P, tau, c2, c1):
    best = -math.inf
    for p in range(1, P+1):
        lo, hi, h = (p-1)*tau/P, p*tau/P, (p-0.5)*tau/P
        if lo <= mu <= hi: best = max(best, tangent(mu, h, tau, c2, c1))
    return best
tau, eps = 1.0, 0.1
kg = ndtri(1-eps); k = math.log(1/eps)
print("V_max eps .1", tau**2/kg**2)
print("G exact cap mu .9", f(0.9, tau, 1/kg**2, 0))
print("G tangent r .4 cap mu .9", tangent(0.9, 0.4, tau, 1/kg**2, 0))
print("W pwl P6 cap mu .95", pwl(0.95, 6, tau, 1/kg**2, 0))
print("Ca range", eps*tau**2/(1-eps), "cap mu .9", f(0.9, tau, eps/(1-eps), 0))
print("Ho range", 2*tau**2/k, "cap mu .9", f(0.9, tau, 2/k, 0))
b=0.05; c=k*b/3; print("Be b.05 tau-c", tau-c, "2kb/3", 2*k*b/3)
b=0.5; c=k*b/3; print("Be b.5 2kb/3", 2*k*b/3, "sqrt", math.sqrt(tau*(tau-2*c)),
  "tangent .4 cap at 0", tangent(0.0, 0.4, tau, 1/(2*k), -b/3), "pwl2 cap at 0", pwl(0.0, 2, tau, 1/(2*k), -b/3),
  "range VB", tau*(tau-2*c)/(2*k))
print("Ch range", 5*tau - k)
# example, eps .1 RF cells, support_new .08
k=math.log(10); b=0.08; c=k*b/3; print("ex b.08 side", 1-c, "cap", f(0.02, 1, 1/(2*k), -b/3), "2kb/3", 2*k*b/3)
print("5*0.14", 5*0.14, "3*sqrt(1e-4)", 3*math.sqrt(1e-4))
