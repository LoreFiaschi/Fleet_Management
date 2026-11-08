import sys
import types
import warnings

import numpy as np

package = types.ModuleType("fleet_management")
package.__path__ = [r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge\src\fleet_management"]
sys.modules["fleet_management"] = package
from fleet_management import config as c

F, L, M, Hs = 3, 2, 2, 3
allc = np.ones((F, L), dtype=bool)
some = np.zeros((F, L), dtype=bool)
some[0, 0] = True
some[2, 1] = True


def attempt(label, func, *args, **kw):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            out = func(*args, **kw)
            shape = getattr(out, "shape", None)
            print(f"{label}: ok shape={shape} warnings={[str(w.message) for w in caught]}")
            return out
        except Exception as exc:
            print(f"{label}: {type(exc).__name__}: {exc}")


print("--- _float_array")
attempt("scalar", c._float_array, 1.0, "x")
attempt("None", c._float_array, None, "x")
attempt("list with None", c._float_array, [1.0, None], "x")
attempt("string", c._float_array, "high", "x")
attempt("ragged", c._float_array, [[1], [1, 2]], "x")
attempt("dict", c._float_array, {}, "x")

print("--- _selector")
sel = lambda v, applies=allc, default=None, allowed=("a", "b"), fill="": c._selector(
    {"k": v}, "k", F, L, applies, default, allowed, fill)
attempt("str", sel, "a")
attempt("(L,)", sel, ["a", "b"])
attempt("(F,L)", sel, [["a", "b"]] * 3)
attempt("(F,)", sel, ["a", "b", "a"])
attempt("bad value", sel, "z")
attempt("bad value (F,L) not applies", sel, [["a", "z"]] * 3, some)
attempt("null at applies", sel, [["a", None]] * 3)
attempt("'' at applies", sel, [["a", ""]] * 3)
r = attempt("(F,L) values at non-applies -> warns", sel, [["a", "b"]] * 3, some)
print(r.tolist())
r = attempt("str broadcast not applies -> no warn", sel, "a", some)
print(r.tolist())
r = attempt("absent default", lambda: c._selector({}, "k", F, L, some, "a", ("a", "b"), "-"))
print(r.tolist())
attempt("absent no default", lambda: c._selector({}, "k", F, L, some, None, ("a", "b")))
r = attempt("absent no default, nowhere", lambda: c._selector({}, "k", F, L, ~allc, None, ("a", "b")))
print(r.tolist())

print("--- _masks")
model = np.array([["gaussian", "inverse_gaussian"], ["wiener", "gamma"], ["rainflow", "rainflow"]], dtype=object)
bound = np.array([["", ""], ["", ""], ["bernstein", "chernoff"]], dtype=object)
rep = np.array([["ard1", "ard1"], ["ardinf", "ard1"], ["ard1", "ard1"]], dtype=object)
rel = np.array([["tangent", "exact"], ["pwl", "exact"], ["exact", "exact"]], dtype=object)
m = c._masks(model, bound, rep, rel)
for k, v in m.items():
    print(k, v.astype(int).tolist())

print("--- _cell_value")
cv = lambda v, applies=allc, **kw: c._cell_value({"k": v}, "k", F, L, applies, **kw)
for label, v in (("scalar", 1.0), ("(L,)", [1.0, 2.0]), ("(F,L)", [[1.0, 2.0]] * 3), ("(F,)", [1.0] * 3),
                 ("null cell", [[1.0, None]] * 3), ("inf", [[1.0, float("inf")]] * 3), ("str", "abc")):
    attempt(label, cv, v)
r = attempt("(F,L) with values at non-applies", cv, [[1.0, 2.0]] * 3, some)
print(r.tolist())
r = attempt("scalar not applies no warning", cv, 2.0, some)
print(r.tolist())
r = attempt("optional null", cv, [[1.0, None]] * 3, optional=True)
print(r.tolist())
r = attempt("absent optional", lambda: c._cell_value({}, "k", F, L, allc, optional=True))
r = attempt("absent default scalar", lambda: c._cell_value({}, "k", F, L, some, 0.5))
print(r.tolist())
r = attempt("absent default array", lambda: c._cell_value({}, "k", F, L, some, np.arange(6.0).reshape(3, 2)))
print(r.tolist())
attempt("absent required", lambda: c._cell_value({}, "k", F, L, some))
r = attempt("absent required nowhere", lambda: c._cell_value({}, "k", F, L, ~allc))
print(np.isnan(r).all())

print("--- _profile")
pf = lambda v, applies=allc, **kw: c._profile({"k": v}, "k", F, L, M, Hs, applies, **kw)
ref = 0.04 + 0.0001 * np.arange(36).reshape(3, 2, 2, 3)
for label, v in (("()", 0.5), ("(M,)", ref[0, 0, :, 0]), ("(L,M)", ref[0, :, :, 0]), ("(L,M,Hs)", ref[0]),
                 ("(F,L,M)", ref[..., 0]), ("(F,L,M,Hs)", ref), ("(4,)", [1.0] * 4), ("(L,)", [1.0, 2.0]),
                 ("(F,L)", np.ones((F, L)))):
    out = attempt(f"profile {label}", pf, v)
exp = {"(M,)": np.broadcast_to(ref[0, 0, :, 0][None, None, :, None], (F, L, M, Hs)),
       "(L,M)": np.broadcast_to(ref[0, :, :, 0][None, :, :, None], (F, L, M, Hs)),
       "(L,M,Hs)": np.broadcast_to(ref[0][None], (F, L, M, Hs)),
       "(F,L,M)": np.broadcast_to(ref[..., 0][..., None], (F, L, M, Hs))}
for label, v in (("(M,)", ref[0, 0, :, 0]), ("(L,M)", ref[0, :, :, 0]), ("(L,M,Hs)", ref[0]), ("(F,L,M)", ref[..., 0])):
    print(label, "equal broadcast:", np.array_equal(pf(v), exp[label]))
for label, v in (("()", 0.5), ("(L,)", [1.0, 2.0]), ("(F,L)", np.ones((F, L))), ("(F,L,M)", ref[..., 0]), ("(F,L,M,Hs)", ref),
                 ("(M,)", [1.0, 2.0]), ("(L,M)", np.ones((L, M)))):
    attempt(f"rate {label}", pf, v, rates=True)
r = attempt("(F,L,M) values at non-applies", pf, ref[..., 0], some)
print(np.isnan(r).any(axis=(2, 3)).tolist())
attempt("(L,M) values at non-applies (no warn)", pf, ref[0, :, :, 0], some)
bad = ref.copy(); bad[1, 1, 0, 0] = np.nan
attempt("null entry", pf, bad)
bad = ref.copy(); bad[1, 1, 0, 0] = np.inf
attempt("inf entry", pf, bad)
attempt("absent", lambda: c._profile({}, "k", F, L, M, Hs, allc))
attempt("absent required", lambda: c._profile({}, "k", F, L, M, Hs, allc, required=True))
attempt("absent required nowhere", lambda: c._profile({}, "k", F, L, M, Hs, ~allc, required=True))
