from pathlib import Path
from fleet_management.solver import solve
from fleet_management.validator import validate_baseline_assignment_feasibility

input_path = "input/tiny_alpha05_presolver.yaml"
deg = "gaussian"
results_path = "results/output_tiny_alpha05_solver_generated.yaml"
log_path = "results/tiny_alpha05_solver_generated_feasibility.log"

print("Running solver...")

solve(
    input_path=input_path,
    degradation=deg,
    results_path=results_path,
)

if not Path(results_path).exists():
    raise FileNotFoundError(
        f"Solver finished, but result file was not created: {results_path}"
    )

print(f"Solver finished. Result written to {results_path}.")
print("Running validator...")

report = validate_baseline_assignment_feasibility(
    input_path=input_path,
    results_path=results_path,
    log_path=log_path,
)

print(f"Validator finished. Log written to: {log_path}")
print(f"Validation passed: {report['passed']}")
print(f"Infeasible assignments: {report['infeasible_assignments']}")