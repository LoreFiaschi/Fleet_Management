"""Where the time of sharp_cap goes, and the effect of fewer golden-section steps."""
import time

import cap_proto as cp

cases = [(t, eps) for eps in (0.001, 0.01, 0.1) for t in (0.01, 0.1, 0.3, 1, 10, 100)]

t0 = time.perf_counter()
for t, eps in cases:
    cp.one_rate_cap(t, eps)
print(f"one_rate_cap: {1000 * (time.perf_counter() - t0) / len(cases):.2f} ms")

t0 = time.perf_counter()
for t, eps in cases:
    cp.worst_one_term_tail(0.5 * t * eps, t)
print(f"worst_one_term_tail: {1000 * (time.perf_counter() - t0) / len(cases):.2f} ms")

calls = {"n": 0}
original = cp.worst_one_term_tail


def counted(A, t):
    calls["n"] += 1
    return original(A, t)


cp.worst_one_term_tail = counted
for t, eps in cases:
    cp.sharp_cap(t, eps)
print(f"w calls per cap: {calls['n'] / len(cases):.1f}")
cp.worst_one_term_tail = original

reference = {(t, eps): cp.sharp_cap(t, eps) for t, eps in cases}
for steps in (40, 60):
    cp._GOLDEN_STEPS = steps
    t0 = time.perf_counter()
    diff = max(abs(cp.sharp_cap(t, eps) - a) / a for (t, eps), a in reference.items())
    print(f"golden steps {steps}: {1000 * (time.perf_counter() - t0) / len(cases):.1f} ms per cap, "
          f"largest relative change {diff:.2e}")
