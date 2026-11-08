"""Task C check: run validator_dashboard.main() on the recording stand-in for streamlit.

Scenarios: no Load; Load on the fixtures in both modes (and a rerun that keeps the session state);
a failed load; a mode change after a load.
"""
import importlib
import sys
import types
from pathlib import Path

S = Path(__file__).resolve().parent
sys.path.insert(0, str(S))
sys.path.insert(0, r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge\src")
import fake_streamlit as fake  # noqa: E402

FIX = Path(r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge\results\validator_tests")


class Stop(BaseException):
    """What st.stop() raises (a script-control exception, not an Exception)."""


class Box:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def install(mode, input_path, results_path, load, session):
    st = fake.install()
    seen = {}
    plain_selectbox = st.selectbox

    def selectbox(label, options, *a, **k):
        if label == "Mode":
            seen["mode_options"] = list(options)
            fake.calls.append(("selectbox", (label, list(options)), k))
            return mode
        return plain_selectbox(label, options, *a, **k)

    def text_input(label, value="", **k):
        fake.calls.append(("text_input", (label,), {"value": value, **k}))
        return {"Input file": input_path, "Result file": results_path}[label]

    def number_input(label, **k):
        fake.calls.append(("number_input", (label,), k))
        return k["value"]

    def button(label, **k):
        fake.calls.append(("button", (label,), k))
        return load

    def stop():
        fake.calls.append(("stop", (), {}))
        raise Stop

    def set_page_config(**k):
        fake.calls.append(("set_page_config", (), k))

    st.selectbox, st.text_input, st.number_input = selectbox, text_input, number_input
    st.button, st.stop, st.set_page_config = button, stop, set_page_config
    st.sidebar = Box()
    st.session_state = session
    return seen


def run(mode, input_path, results_path, load, session):
    seen = install(mode, input_path, results_path, load, session)
    sys.modules.pop("fleet_management.validator_dashboard", None)
    vd = importlib.import_module("fleet_management.validator_dashboard")
    try:
        vd.main()
        stopped = False
    except Stop:
        stopped = True
    names = [c[0] for c in fake.calls]
    return stopped, names, seen, [c for c in fake.calls if c[0] in ("error", "info", "tabs")]


# Import without streamlit
sys.modules.pop("streamlit", None)
vd = importlib.import_module("fleet_management.validator_dashboard")
print("import without streamlit: ok;", "streamlit" in sys.modules and "loaded" or "streamlit not loaded")
print("MODES:", list(vd.MODES))
print("main:", vd.main.__doc__)

inp, res = str(FIX / "input.yaml"), str(FIX / "result.yaml")
session = {}

print("\n1. no Load")
stopped, names, seen, msgs = run("Result inspector", inp, res, False, session)
print("   stopped", stopped, "| mode options", seen["mode_options"], "| msgs", msgs, "| session", list(session))
assert stopped and "tabs" not in names and "info" in names and "loaded" not in session

print("\n2. Load, Result inspector")
stopped, names, seen, msgs = run("Result inspector", inp, res, True, session)
print("   stopped", stopped, "| msgs", [m[1][0] if m[0] == "tabs" else m[0] for m in msgs], "| session", list(session))
assert not stopped and "loaded" in session and session["loaded"][0] == "Result inspector"
assert any(m[0] == "tabs" for m in msgs)
tabs = [m for m in msgs if m[0] == "tabs"][0][1][0]
print("   tabs", tabs)

print("\n2b. rerun without Load keeps the page")
stopped, names, seen, msgs = run("Result inspector", inp, res, False, session)
assert not stopped and any(m[0] == "tabs" for m in msgs)
print("   page shown again: ok")

print("\n3. mode change after a load, no Load")
stopped, names, seen, msgs = run("Gamma diagnostic", inp, res, False, session)
print("   stopped", stopped, "| msgs", [m[0] for m in msgs])
assert stopped and not any(m[0] == "tabs" for m in msgs) and any(m[0] == "info" for m in msgs)

print("\n4. failed load (missing result file): error, no page, session cleaned")
stopped, names, seen, msgs = run("Result inspector", inp, str(FIX / "nope.yaml"), True, session)
errs = [m for m in msgs if m[0] == "error"]
print("   stopped", stopped, "| error:", errs[0][1][0] if errs else None, "| session", list(session))
assert stopped and errs and "Could not load the files" in errs[0][1][0] and "loaded" not in session
assert not any(m[0] == "tabs" for m in msgs)

print("\n5. Load, Gamma diagnostic")
try:
    stopped, names, seen, msgs = run("Gamma diagnostic", inp, res, True, session)
    errs = [m for m in msgs if m[0] == "error"]
    if errs:
        print("   load failed:", errs[0][1][0])
    else:
        tabs = [m for m in msgs if m[0] == "tabs"][0][1][0]
        print("   stopped", stopped, "| tabs", tabs, "| session", list(session))
except Exception as exc:  # a failure of the Gamma validators (other plan)
    print("   EXC", type(exc).__name__, exc)

print("\nsidebar calls of the last run:")
for c in fake.calls:
    if c[0] in ("selectbox", "text_input", "number_input", "button", "set_page_config") and c[1][:1] in (("Mode",), ("Input file",), ("Result file",), ("Tolerance",), ("Load",), ()):
        print("  ", c)
