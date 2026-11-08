import sys
from pathlib import Path
sys.path.insert(0, r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge\src")
import yaml
from fleet_management import horizon_sweep as h


def make_row(H2, status, J, gap, H1=2, count=True):
    row = {k: None for k in h.ROW_KEYS}
    T = H1 + H2
    row.update({"H1": H1, "H2": H2, "T": T, "status": status, "J_op_average": J, "objective": J,
                "mip_gap": gap, "requested_mip_gap": 0.1})
    if count:
        row["formulation"] = {k: 10 * T for k in h.FORMULATION_KEYS}
        row["formulation"]["quadratic_constraints"] = None
    return row


rows = [make_row(2, "optimal", 1.0, 0.0), make_row(3, "optimal", 0.9, 0.0),
        make_row(5, "error: x", None, None, count=False), make_row(7, "optimal", 0.9, 0.0)]
growth = h._formulation_growth(rows)
assert growth["from_H2"] == 2 and growth["to_H2"] == 7 and growth["added_time_steps"] == 5
assert growth["added_variables"] == 50 and growth["variables_per_added_time_step"] == 10.0
assert growth["added_quadratic_constraints"] is None and growth["quadratic_constraints_per_added_time_step"] is None
assert h._formulation_growth(rows[:1]) is None and h._formulation_growth([rows[0], rows[2]]) is None

settings = {"candidates": [2, 3, 5, 7], "stop_on_gradient": False, "gradient_tolerance": 1e-3,
            "flat_gradients_required": 2, "minimum_cases": 3, "maximum_mip_gap_for_stopping": None,
            "warm_start": True, "warm_start_mode": "both", "n_workers": 1, "objective_mode": "operating_average"}
h._annotate_gradients(rows, gradient_tolerance=1e-3, maximum_mip_gap_for_stopping=None)
report = h._assemble_sweep_report(source=Path("in.yaml"), sizes={"F": 3, "M": 1, "L": 1}, H1=2, settings=settings,
                                  rows=rows, last_completed_H2=7, stopping_reason=None, complete=True)
assert list(report) == ["input", "complete", "last_completed_H2", "objective", "varied_parameter", "fixed_dimensions",
                        "H1", "planned_operating_horizons", "evaluated_operating_horizons", "warm_start", "n_workers",
                        "stopping_rule", "cases", "formulation_growth", "best_proven_H2", "best_feasible_H2",
                        "best_proven_J_op_average", "best_feasible_J_op_average", "best_feasible_status"], list(report)
assert len(report) == 19
assert report["stopping_rule"]["reason"] == "candidate_range_exhausted" and report["stopping_rule"]["stopped_early"] is False
assert report["warm_start"] == {"enabled": True, "mode": "both"}
assert list(report["stopping_rule"]) == ["enabled", "relative_gradient_tolerance_per_H2", "flat_gradients_required",
                                         "minimum_cases", "maximum_mip_gap_for_stopping", "reason", "stopped_early"]
part = h._assemble_sweep_report(source=Path("in.yaml"), sizes={"F": 3, "M": 1, "L": 1}, H1=2, settings=settings,
                                rows=rows, last_completed_H2=7, stopping_reason="cost_increase", complete=False)
assert part["stopping_rule"]["reason"] == "in_progress" and part["complete"] is False and part["stopping_rule"]["stopped_early"] is False
yaml.safe_dump(report, sort_keys=False)

grid = h._assemble_grid_report(source=Path("in.yaml"), sizes={"F": 3, "M": 1, "L": 1}, h1_values=[2], h2_values=[2, 3, 5, 7],
                               subreports=[report], warm_start=True, evaluation_horizon=None, complete=False)
assert list(grid) == ["input", "complete", "objective", "comparison_metric", "mip_certificate_metric", "evaluation_horizon",
                      "varied_parameters", "fixed_dimensions", "planned_transitory_horizons", "planned_operating_horizons",
                      "warm_start_enabled", "H1_sweeps", "cases", "best_proven", "best_feasible"]
assert list(grid["best_proven"]) == ["H1", "H2", "T", "status", "mip_gap", "objective", "J_initialization", "J_op_average",
                                     "peak_damage", "damage_penalty", "operating_objective", "projected_evaluation_cost",
                                     "evaluation_objective"], list(grid["best_proven"])
assert grid["best_proven"]["H2"] == 3 and grid["comparison_metric"] == "J_op_average"
assert grid["cases"][0] is not rows[0] and grid["cases"] == rows
dumped = yaml.safe_dump(grid, sort_keys=False)
assert "&id" not in dumped and "*id" not in dumped
assert yaml.safe_load(dumped) == grid
print("A4 check OK")
