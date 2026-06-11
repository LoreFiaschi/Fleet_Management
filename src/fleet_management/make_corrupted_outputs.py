# This script purposefully corrupts output files to test validator
from pathlib import Path
import copy
import yaml


BASE_OUTPUT = Path("results/output_baseline.yaml")
OUT_DIR = Path("results/validator_tests")


def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_yaml(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def make_case(name: str, mutate_fn) -> None:
    data = load_yaml(BASE_OUTPUT)
    corrupted = copy.deepcopy(data)
    mutate_fn(corrupted)
    save_yaml(corrupted, OUT_DIR / f"{name}.yaml")


# 1) Solver status should fail
make_case(
    "bad_status",
    lambda d: d.__setitem__("status", "infeasible"),
)


# 2) Binary x check should fail
def corrupt_binary_x(d):
    d["x"][0][0][0] = 0.5

make_case("bad_x_binary", corrupt_binary_x)


# 3) Assignment constraint sum_j x[i,j,k] <= 1 should fail
def corrupt_assignment(d):
    # Force one car to do maintenance and mission at the same time
    d["x"][0][0][0] = 1.0
    d["x"][0][1][0] = 1.0

make_case("bad_assignment", corrupt_assignment)


# 4) Demand constraint sum_i x[i,j,k] == 1 should fail
def corrupt_demand(d):
    # Remove all cars from mission/slot j=0 at time k=0
    for i in range(d["F"]):
        d["x"][i][0][0] = 0.0

make_case("bad_demand", corrupt_demand)


# 5) u >= mu should fail
def corrupt_u_ge_mu(d):
    # Set u too low compared to mu
    d["u"][0] = -1.0

make_case("bad_u_ge_mu", corrupt_u_ge_mu)


# 6) Capacity sum(mu) <= F - M should fail
def corrupt_capacity(d):
    # Make degradation sum too large at k=0
    for i in range(d["F"]):
        for ell in range(d["L"]):
            d["mu"][i][ell][0] = 10.0

make_case("bad_capacity", corrupt_capacity)


# 7) Periodic mu should fail: mu[:,:,2H-1] <= mu[:,:,H-1]
def corrupt_mu_periodic(d):
    H = d["H"]
    d["mu"][0][0][2 * H - 1] = d["mu"][0][0][H - 1] + 1.0

make_case("bad_mu_periodic", corrupt_mu_periodic)


# 8) Periodic v should fail for Gaussian
def corrupt_v_periodic(d):
    H = d["H"]
    d["v"][0][0][2 * H - 1] = d["v"][0][0][H - 1] + 1.0

make_case("bad_v_periodic", corrupt_v_periodic)


# 9) Objective recomputation should fail
make_case(
    "bad_objective",
    lambda d: d.__setitem__("objective", float(d["objective"]) + 123.0),
)


# 10) Dimension check should fail immediately
def corrupt_dimension(d):
    # Remove one time step from u
    d["u"] = d["u"][:-1]

make_case("bad_dimension", corrupt_dimension)


print(f"Corrupted test files written to {OUT_DIR}")