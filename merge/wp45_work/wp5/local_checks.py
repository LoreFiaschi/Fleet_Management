"""Local checks of the pure functions of base.py (no Gurobi needed).

Usage: python local_checks.py
Loads base.py through the gurobipy stub and checks the pure functions that the
WP5 plan defines (section 4 of merge/wp5_plan.md). A function that does not
exist yet is reported as MISSING, so the script can run after every step.
The last line says how many checks failed.
"""
import math

import numpy as np

from load_base import load

base = load()
failures = 0


def check(label, condition):
    global failures
    print(("PASS " if condition else "FAIL ") + label)
    if not condition:
        failures += 1


def have(name):
    if hasattr(base, name):
        return True
    print("MISSING " + name)
    return False


# --- phase_day: Eq. c_phase with the 0-based day index k = spec day - 1 -----
if have("phase_day"):
    expected_without = [(False, 0), (False, 1), (False, 0), (False, 1), (False, 2), (False, 0)]
    expected_with = [(True, 0), (True, 1), (False, 0), (False, 1), (False, 2), (False, 0)]
    got_without = [tuple(base.phase_day(k, 2, 3, False)) for k in range(6)]
    got_with = [tuple(base.phase_day(k, 2, 3, True)) for k in range(6)]
    check("phase_day without transitory profile, H1=2, H2=3", got_without == expected_without)
    check("phase_day with transitory profile, H1=2, H2=3", got_with == expected_with)
    check("phase_day H1=3, H2=2, k=2 without transitory", tuple(base.phase_day(2, 3, 2, False)) == (False, 0))

# --- cost_days -------------------------------------------------------------
if have("cost_days"):
    check("cost_days total", list(base.cost_days("total", 2, 5)) == [0, 1, 2, 3, 4])
    check("cost_days evaluation_total", list(base.cost_days("evaluation_total", 2, 5)) == [0, 1, 2, 3, 4])
    check("cost_days operating_average", list(base.cost_days("operating_average", 2, 5)) == [2, 3, 4])

# --- repair_factor ---------------------------------------------------------
if have("repair_factor"):
    check("repair_factor mu", math.isclose(base.repair_factor("mu", 0.5), 0.5))
    check("repair_factor K", math.isclose(base.repair_factor("K", 0.5), 0.5))
    check("repair_factor v", math.isclose(base.repair_factor("v", 0.5), 0.25))
    check("repair_factor R", math.isclose(base.repair_factor("R", 0.5), 0.25))

# --- collapse_selector -----------------------------------------------------
if have("collapse_selector"):
    same = np.array([["bigm", "bigm"], ["bigm", "bigm"]])
    mixed = np.array([["bigm", "indicator"], ["bigm", "bigm"]])
    blank = np.array([["", ""], ["", ""]])
    part = np.array([["cantelli", ""], ["", ""]])
    check("collapse_selector one value", base.collapse_selector(same) == "bigm")
    check("collapse_selector mixed", base.collapse_selector(mixed) == [["bigm", "indicator"], ["bigm", "bigm"]])
    check("collapse_selector all blank -> None", base.collapse_selector(blank, empty_as_none=True) is None)
    check(
        "collapse_selector partly blank",
        base.collapse_selector(part, empty_as_none=True) == [["cantelli", None], [None, None]],
    )

# --- hand-worked case of section 4 (F=2, L=1, M=1, H1=H2=2) ------------------
mu_ard1 = np.array([[[0.5, 0.35, 0.65, 0.1]], [[0.2, 0.2, 0.2, 0.2]]])
g_ard1 = np.array([[[0.2, 0.35, 0.35, 0.1]], [[0.2, 0.2, 0.2, 0.2]]])
mu_ardinf = np.array([[[0.5, 0.25, 0.55, 0.1]], [[0.2, 0.2, 0.2, 0.2]]])
m = np.zeros((2, 1, 4)); m[0, 0, 1] = 1.0
r = np.zeros((2, 1, 4)); r[0, 0, 3] = 1.0
x = np.zeros((2, 2, 4))
x[0, 1, 0] = x[0, 0, 1] = x[0, 1, 2] = x[0, 0, 3] = 1.0
x[1, 1, 1] = x[1, 1, 3] = 1.0
mu_0 = np.full((2, 1), 0.2)
rho = np.full((2, 1), 0.5)

if have("transitory_repair_cost"):
    z_ard1 = base.transitory_repair_cost(
        mu_ard1, g_ard1, m, mu_0, np.full((2, 1), 0.2), rho, np.full((2, 1), True), 2)
    z_ardinf = base.transitory_repair_cost(
        mu_ardinf, np.full((2, 1, 4), np.nan), m, mu_0, np.full((2, 1), np.nan), rho,
        np.full((2, 1), False), 2)
    check("transitory_repair_cost shape", z_ard1.shape == (2, 1, 2))
    check("transitory_repair_cost ard1", np.allclose(z_ard1[0, 0], [0.0, 0.15]) and np.allclose(z_ard1[1, 0], 0.0))
    check("transitory_repair_cost ardinf", np.allclose(z_ardinf[0, 0], [0.0, 0.25]) and np.allclose(z_ardinf[1, 0], 0.0))

if have("peak_damage"):
    check("peak_damage inf_norm", math.isclose(base.peak_damage(mu_ard1, "inf_norm"), 0.65))
    check("peak_damage quadratic", math.isclose(base.peak_damage(mu_ard1, "quadratic"), 0.65 ** 2))

if have("cost_report"):
    z = np.full((2, 1, 4), np.nan)
    z[0, 0, :] = [0.0, 0.15, 0.0, 0.0]
    report = base.cost_report(
        x, r, z, 1.0, np.array([[2.0], [0.0]]), np.full((2, 1), 3.0), 0.5, 0.85, 2, 2, 6)
    check("cost_report step_costs", np.allclose(report["step_costs"], [0.0, 1.3, 0.0, 4.0]))
    check("cost_report J_initialization", math.isclose(report["J_initialization"], 1.3))
    check("cost_report J_op", math.isclose(report["J_op"], 4.0))
    check("cost_report J_op_average", math.isclose(report["J_op_average"], 2.0))
    check("cost_report peak_damage", math.isclose(report["peak_damage"], 0.85))
    check("cost_report damage_penalty", math.isclose(report["damage_penalty"], 0.425))
    check("cost_report operating_objective", math.isclose(report["operating_objective"], 2.425))
    check("cost_report J_total", math.isclose(report["J_total"], 5.725))
    check("cost_report projected_evaluation_cost", math.isclose(report["projected_evaluation_cost"], 9.3))
    check("cost_report evaluation_objective", math.isclose(report["evaluation_objective"], 9.725))
    check(
        "cost_report component_costs",
        report["component_costs"]["C_M"] == 1.0 and report["component_costs"]["C_D"] == 0.5
        and np.asarray(report["component_costs"]["C_R"]).shape == (2, 1),
    )
    no_e = base.cost_report(
        x, r, z, 1.0, np.array([[2.0], [0.0]]), np.full((2, 1), 3.0), 0.5, 0.85, 2, 2, None)
    check("cost_report without E", no_e["projected_evaluation_cost"] is None and no_e["evaluation_objective"] is None)
    check(
        "cost_report key set",
        set(report) == {
            "step_costs", "J_initialization", "J_op", "J_op_average", "peak_damage",
            "damage_penalty", "operating_objective", "J_total", "projected_evaluation_cost",
            "evaluation_objective", "component_costs",
        },
    )

# --- names that must be gone after WP5 ---------------------------------------
for gone in ("add_maintenance_gating", "add_base_constraints", "resolve_costs", "pick",
             "cell_max", "make_accessor", "collapse_dict", "AccumulatedDegradationModel"):
    if hasattr(base, gone):
        print("PRESENT " + gone + " (Task A removes it)")

print(f"failures: {failures}")
