from fleet_management import solve, validate_gamma_result

solve(
    "input/tiny_gamma.yaml",
    degradation="gamma",
    results_path="results/output_tiny_gamma.yaml",
)

report = validate_gamma_result(
    "input/tiny_gamma.yaml",
    "results/output_tiny_gamma.yaml",
)

print("Validation passed:", report["passed"])
for check in report["checks"]:
    print(check["passed"], check["name"], check.get("error", check.get("max_error", 0.0)))