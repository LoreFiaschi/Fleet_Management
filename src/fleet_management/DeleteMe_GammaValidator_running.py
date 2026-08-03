from fleet_management import solve, validate_gamma_result

solve(
    "input/tiny_gamma.yaml",
    degradation="gamma",
    results_path="results/output_tiny_gamma.yaml",
)

report = validate_gamma_result(
    input_path="input/tiny_gamma.yaml",
    results_path="results/output_tiny_gamma.yaml",
    validation_path="results/validation_tiny_gamma.yaml",
)

print("Validation passed:", report["passed"])

for check in report["checks"]:
    print(
        check["passed"],
        check["name"],
        check["maximum_violation"],
    )