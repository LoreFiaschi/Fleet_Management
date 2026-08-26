"""Force one Gamma ARD-inf repair and verify fixed-rate shape scaling."""

from fleet_management.config import load_config
from fleet_management.degradation_model.base import (
    build_fleet,
    resolve_run_options,
)


def assert_close(actual, expected, name, tolerance=1e-8):
    if abs(float(actual) - float(expected)) > tolerance:
        raise AssertionError(
            f"{name}: got {float(actual):.12g}, "
            f"expected {float(expected):.12g}"
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
    ctx = build_fleet(
        cfg,
        opts,
        model_name="gamma_forced_ardinf_repair_test",
    )

    # Force vehicle 0, component 0 to repair at the first step.
    ctx.model.addConstr(
        ctx.m_rep[0, 0, 0] == 1,
        name="force_gamma_ardinf_repair",
    )
    ctx.model.optimize()

    if ctx.model.SolCount == 0:
        raise AssertionError("forced ARD-inf repair model has no feasible solution")

    data = ctx.extras["gamma"]

    rho = float(cfg.rho[0, 0])
    remaining = 1.0 - rho
    common_rate = float(data["common_rate"][0, 0])

    shape_before = float(data["initial_shape"][0, 0])
    shape_after = float(data["A_var"][0, 0, 0].X)

    mean_before = float(cfg.mu_0[0, 0])
    mean_after = float(ctx.mu_var[0, 0, 0].X)
    removed_mean = float(ctx.z_var[0, 0, 0].X)

    expected_shape = remaining * shape_before
    expected_mean = remaining * mean_before
    expected_removed = rho * mean_before

    assert_close(
        shape_after,
        expected_shape,
        "ARD-inf repaired bounding shape",
    )
    assert_close(
        mean_after,
        expected_mean,
        "ARD-inf repaired physical mean",
    )
    assert_close(
        removed_mean,
        expected_removed,
        "ARD-inf removed physical mean",
    )

    # This case uses the same exact and bounding rate, so the shape-derived
    # mean must coincide with the separately stored physical mean.
    assert_close(
        shape_after / common_rate,
        mean_after,
        "fixed-rate shape-derived mean",
    )

    print("PASS fixed-rate Gamma ARD-inf repair integration")
    print("remaining fraction :", remaining)
    print("common rate       :", common_rate)
    print("shape before      :", shape_before)
    print("shape after       :", shape_after)
    print("expected shape    :", expected_shape)
    print("mean before       :", mean_before)
    print("mean after        :", mean_after)
    print("removed mean      :", removed_mean)
    print("objective         :", ctx.model.ObjVal)


if __name__ == "__main__":
    main()