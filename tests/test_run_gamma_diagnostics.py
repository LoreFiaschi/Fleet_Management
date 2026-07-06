# DEPRECATED
from fleet_management.validator import (
    build_gamma_diagnostic_dataframe,
    validate_gamma_synthetic_diagnostic,
)

input_path = "input/tiny_gamma_synthetic_replacement.yaml"

df = build_gamma_diagnostic_dataframe(input_path=input_path)

print(df)
print(df.columns)

report = validate_gamma_synthetic_diagnostic(
    input_path=input_path,
    log_path="results/tiny_gamma_synthetic_replacement_diagnostic.log",
)

print(report)