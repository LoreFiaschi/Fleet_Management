"""Review fix: drop width="stretch" from st.dataframe calls (the default of current Streamlit;
an error on releases before the string values of width). Keeps the bytes otherwise."""
from pathlib import Path

ROOT = Path(r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge\src\fleet_management\validation")
for name in ("dashboard_inspector.py", "dashboard_gamma.py"):
    path = ROOT / name
    data = path.read_bytes()
    count = data.count(b', width="stretch"')
    path.write_bytes(data.replace(b', width="stretch"', b""))
    print(name, "replaced", count)
