# Regularised upper incomplete gamma Q(a, x), standard library only (reviewer's own code).
import math

def _gser(a, x):
    # lower regularised P(a, x) by series
    if x <= 0.0:
        return 0.0
    gln = math.lgamma(a)
    ap = a
    s = 1.0 / a
    d = s
    for _ in range(100000):
        ap += 1.0
        d *= x / ap
        s += d
        if abs(d) < abs(s) * 1e-16:
            break
    return s * math.exp(-x + a * math.log(x) - gln)

def _gcf(a, x):
    # upper regularised Q(a, x) by Lentz continued fraction
    gln = math.lgamma(a)
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 100000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < 1e-16:
            break
    return math.exp(-x + a * math.log(x) - gln) * h

def Q(a, x):
    if a <= 0.0:
        return 0.0
    if x <= 0.0:
        return 1.0
    if x < a + 1.0:
        return 1.0 - _gser(a, x)
    return _gcf(a, x)

def P(a, x):
    return 1.0 - Q(a, x)
