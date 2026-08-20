# Gamma horizon regression

`gamma_unequal_horizon.yaml` exercises a uniform Gamma fleet with `H1 = 2`,
`H2 = 3`, and therefore `T = 5`. It supplies distinct transitory and operating
damage profiles so incorrect phase indexing is observable in the saved states.

With the project dependencies and a valid Gurobi licence installed, run from
the repository root:

```bash
python regression/check_gamma_horizon.py
```

The checker compares the solver result against a unique optimum obtained by
exhaustive enumeration and then runs the independent Gamma result validator.
