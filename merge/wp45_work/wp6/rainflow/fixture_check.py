"""Check the fixture draft of the plan against rf_rows.make_cfg, through the emulator.

Usage: python fixture_check.py
"""
import sys
import types

import numpy as np

import rf_rows  # loads fakegp, base and the prototype builder

pytest_stub = types.ModuleType("pytest")
pytest_stub.importorskip = lambda name: __import__(name)
sys.modules["pytest"] = pytest_stub
import fixture_draft  # noqa: E402

failures = 0
for bound in rf_rows.rainflow.BOUND_QUANTITY:
    for law in ("ard1", "ardinf"):
        for failing in (False, True):
            ours = vars(fixture_draft._cfg(bound, law, failing=failing))
            ref = vars(rf_rows.make_cfg(bound, law, eps_index=int(failing)))
            for key in sorted(set(ours) | set(ref)):
                a, b = ours.get(key, "MISSING"), ref.get(key, "MISSING")
                if key in ("tangent_ref", "options"):
                    continue
                if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
                    same = (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)
                            and a.shape == b.shape
                            and (np.array_equal(a.astype(str), b.astype(str)) if a.dtype == object
                                 else np.allclose(a, b, equal_nan=True, atol=1e-12)))
                else:
                    same = a == b
                if not same:
                    failures += 1
                    print(f"DIFF {bound} {law} failing={failing} {key}: {a!r} vs {b!r}")
        ctx, out = rf_rows.solve_fixed(fixture_draft._cfg(bound, law, encoding="indicator"))
        if out["status"] != "optimal":
            failures += 1
            print("not optimal", bound, law)
print(f"failures: {failures}")

# helpers of the draft: trajectories (indicator: fix n too) and the two-component fleet
fd = fixture_draft
for law in ("ard1", "ardinf"):
    for bound in rf_rows.rainflow.BOUND_QUANTITY:
        ctx, out = fd._solve_fixed(fd._cfg(bound, law))
        ok = out["status"] == "optimal"
        for q in fd._tracked(bound):
            key, gkey = fd._RESULT_KEYS[q]
            states, anchors = fd._EXPECTED[q][law]
            ok = ok and np.allclose(out[key][0, 0], states) and np.allclose(out[key][1, 0], fd._START[q])
            if anchors is not None:
                ok = ok and np.allclose(out[gkey][0, 0], anchors)
        ok = ok and np.allclose(out["z"][0, 0], fd._Z[law])
        if not ok:
            failures += 1
            print("draft helpers FAIL", bound, law)
    cfg = fd._stack(fd._cfg("cantelli", law), fd._cfg("chernoff", law))
    ctx, out = fd._solve_fixed(cfg, fd._build_schedule(L=2))
    ok = (out["status"] == "optimal"
          and np.allclose(out["v"][0, 0], fd._EXPECTED["v"][law][0])
          and np.allclose(out["K"][0, 1], fd._EXPECTED["K"][law][0])
          and np.isnan(out["v"][:, 1]).all() and np.isnan(out["K"][:, 0]).all()
          and np.allclose(out["mu"][0, 1], fd._EXPECTED["mu"][law][0]))
    if not ok:
        failures += 1
        print("stack FAIL", law)
print(f"failures after helpers: {failures}")
