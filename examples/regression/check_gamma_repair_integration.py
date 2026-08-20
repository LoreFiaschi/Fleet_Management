"""Force one Gamma ARD-inf repair and verify its solver dynamics."""

from fleet_management.config import load_config
from fleet_management.degradation_model.base import (
    build_fleet,
    resolve_run_options,
)


def main() -> None:
    cfg = load_config(
        {
            "F": 2,
            "M": 1,
            "L": 1,
            "H": [2, 2],
            "model": "gamma",
            "repair_model": "ardinf",
            "tau": 0.6,
            "epsilon": 0.1,
            "rho": 0.5,
            "mu_0": 0.02,
            "replacement_mu": 0.01,
            "mu": 0.05,
            "gamma_beta": 10.0,
            "gamma_beta_trans": 10.0,
            "gamma_beta_bound": 10.0,
            "gamma_beta_0": 10.0,
            "gamma_beta_new": 10.0,
            "C_M": 1.0,
            "C_R": 0.5,
            "C_D": 2.0,
            "C_rep": 0.2,
            "allow_replacement": True,
            "depot_capacity": 1,
            "mip_gap": 0.0,
            "verbose": 0,
        }
    )

    opts = resolve_run_options(cfg)
    ctx = build_fleet(cfg, opts, model_name="gamma_forced_repair_test")

    # Force vehicle 0, component 0 to repair at the first step.
    ctx.model.addConstr(ctx.m_rep[0, 0, 0] == 1, name="force_gamma_repair")
    ctx.model.optimize()

    if ctx.model.SolCount == 0:
        raise AssertionError("forced-repair model has no feasible solution")

    data = ctx.extras["gamma"]
    shape_after = data["A_var"][0, 0, 0].X
    shape_before = float(data["initial_shape"][0, 0])
    mean_after = ctx.mu_var[0, 0, 0].X
    removed_mean = ctx.z_var[0, 0, 0].X

    expected_mean = (1.0 - cfg.rho[0, 0]) * cfg.mu_0[0, 0]
    expected_removed = cfg.rho[0, 0] * cfg.mu_0[0, 0]

    tol = 1e-8
    if abs(shape_after - shape_before) > tol:
        raise AssertionError("repair incorrectly reduced the Gamma bounding shape")
    if abs(mean_after - expected_mean) > tol:
        raise AssertionError("repair physical-mean transition is incorrect")
    if abs(removed_mean - expected_removed) > tol:
        raise AssertionError("repair removed-damage value is incorrect")

    print("PASS forced Gamma ARD-inf repair integration")
    print("shape before :", shape_before)
    print("shape after  :", shape_after)
    print("mean before  :", cfg.mu_0[0, 0])
    print("mean after   :", mean_after)
    print("removed mean :", removed_mean)
    print("objective    :", ctx.model.ObjVal)


if __name__ == "__main__":
    main()