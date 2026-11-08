"""T12 and T4 (pwl count, linear bounds) of the plan through the emulator."""
import numpy as np
import fixture_check  # noqa: F401  (runs the fixture checks first)
import fixture_draft as fd

for changes, bound, match in (({"bound_method": np.full((2, 1), "weibull", dtype=object)}, "cantelli", "unknown bound_method"),
                              ({"support_const": np.full((2, 1), np.nan)}, "bernstein", "support_const is not finite")):
    try:
        fd._build(fd._cfg(bound, **changes))
        print("FAIL no error", bound)
    except ValueError as error:
        print("ok" if match in str(error) else "FAIL message", bound, "|", error)
for bound, impl, count in (("cantelli", "pwl", 64), ("hoeffding", "pwl", 64), ("bernstein", "pwl", 64),
                           ("markov", "pwl", 0), ("chernoff", "pwl", 0), ("cantelli", "tangent", 0)):
    ctx = fd._build(fd._cfg(bound, reliability_impl=impl))
    print("ok" if len(ctx.pwl_w) == count else "FAIL", "pwl_w", bound, impl, len(ctx.pwl_w),
          "NonConvex set" if hasattr(ctx.model.Params, "NonConvex") else "NonConvex not set")
