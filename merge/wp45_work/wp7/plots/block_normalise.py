def _normalise_solution(data: dict[str, Any], presolved: dict[str, int]) -> dict[str, Any]:
    """Check a result mapping against the spec's shapes and build the view of the plot.

    ``presolved`` holds the counts parsed from ``runner.out``, or is empty.
    """
    missing = [key for key in REQUIRED_RESULT_KEYS if data.get(key) is None]
    if missing:
        raise ValueError(
            "The result has no plottable solution; missing keys: " + ", ".join(missing)
        )

    F, M, L = int(data["F"]), int(data["M"]), int(data["L"])
    H1, H2 = int(data["H1"]), int(data["H2"])
    tau = np.asarray(data["tau"], dtype=float)
    mu_0 = np.asarray(data["mu_0"], dtype=float)
    mu = np.asarray(data["mu"], dtype=float)
    x = np.asarray(data["x"], dtype=float)
    for name, value in (("tau", tau), ("mu_0", mu_0)):
        if value.shape != (F, L):
            raise ValueError(f"{name} has shape {value.shape}; expected {(F, L)}")
    if mu.ndim != 3 or mu.shape[:2] != (F, L):
        raise ValueError(f"mu has shape {mu.shape}; expected ({F}, {L}, T)")
    T = int(mu.shape[2])
    if data.get("T") is not None and int(data["T"]) != T:
        raise ValueError(f"mu has {T} days; the key T is {data['T']}")
    if H1 + H2 != T:
        raise ValueError(f"H1 + H2 must equal T; got {H1} + {H2} != {T}")
    if x.shape != (F, M + 1, T):
        raise ValueError(f"x has shape {x.shape}; expected {(F, M + 1, T)}")
    if np.any(tau <= 0.0):
        raise ValueError("All degradation thresholds must be positive.")

    m = _optional_array(data, "m", (F, L, T))
    r = _optional_array(data, "r", (F, L, T))

    # Counts of the summary panel: the presolved counts of runner.out if that log
    # exists, otherwise the counts of the original model in ``performance``.
    if presolved:
        counts, counts_source = presolved, "runner.out (presolved)"
    else:
        counts, counts_source = data.get("performance") or {}, "performance"

    grid = np.empty((F, L, T + 1), dtype=float)
    grid[:, :, 0], grid[:, :, 1:] = mu_0, mu
    return {
        "raw": data, "F": F, "M": M, "L": L, "T": T, "H1": H1, "H2": H2,
        "x": x, "m": m, "r": r, "grid": grid, "tau": tau,
        "models": _model_grid(data, F, L),
        "component_names": _component_names(data, L),
        "variables": counts.get("variables"),
        "continuous_variables": counts.get("continuous_variables"),
        "integer_variables": counts.get("integer_variables"),
        "linear_rows": counts.get("linear_constraints"),
        "counts_source": counts_source,
        # Count component actions, not train visits: two components repaired on
        # the same depot day are two repairs.
        "repairs": int(np.count_nonzero(m > 0.5)),
        "replacements": int(np.count_nonzero(r > 0.5)),
    }
