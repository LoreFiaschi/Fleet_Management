from fleet_management import solve

# 1. Solve the optimization problem (model is now per-component, read from the
# YAML file's 'model' field, so there is no separate `degradation` argument).
solve("../input/data_example.yaml", results_path="../results/output.yaml")
