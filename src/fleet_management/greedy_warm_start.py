"""Deterministic greedy schedule construction for MIP starts.

The heuristic deliberately constructs only the discrete decisions.  Gurobi
reconstructs the continuous degradation states when the start is submitted to
the complete model.  It is therefore a warm-start heuristic, not a substitute
for validation by the original constraints.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GreedyWarmStartReport:
    feasible_assignment: bool
    artificial_assignments: int
    repeatability_feasible: bool
    repeatability_failures: int
    repairs: int
    replacements: int
    maximum_proxy_ratio: float
    maximum_repeatability_excess: float
    assignment_order: list[list[int]]

    def as_dict(self) -> dict:
        return {
            "feasible_assignment": self.feasible_assignment,
            "artificial_assignments": self.artificial_assignments,
            "repeatability_feasible": self.repeatability_feasible,
            "repeatability_failures": self.repeatability_failures,
            "repairs": self.repairs,
            "replacements": self.replacements,
            "maximum_proxy_ratio": self.maximum_proxy_ratio,
            "maximum_repeatability_excess": (
                self.maximum_repeatability_excess
            ),
            "assignment_order": self.assignment_order,
        }


def _mean_increment(cfg, i: int, l: int, mission: int, k: int) -> float:
    """Read a normalized phase-aware mean increment from ``FleetConfig``."""
    if k < cfg.H1 and cfg.mu_trans is not None:
        return float(cfg.mu_trans[i, l, mission, k])
    h = (k - cfg.H1) % cfg.H2 if k >= cfg.H1 else k % cfg.H2
    return float(cfg.mu[i, l, mission, h])


def _variance_increment(cfg, i: int, l: int, mission: int, k: int) -> float:
    """Return the phase-aware variance increment for a rainflow cell."""
    if cfg.v is None:
        return 0.0
    if k < cfg.H1 and cfg.v_trans is not None:
        return float(cfg.v_trans[i, l, mission, k])
    h = (k - cfg.H1) % cfg.H2 if k >= cfg.H1 else k % cfg.H2
    return float(cfg.v[i, l, mission, h])


def _uses_variance_repeatability(cfg, i: int, l: int) -> bool:
    """Whether the model compares the terminal variance state for this cell."""
    return (
        str(cfg.model[i, l]) == "rainflow"
        and str(cfg.bound_method[i, l]) in {"cantelli", "bernstein"}
        and cfg.v is not None
    )


def _gamma_shape_data(cfg, i: int, l: int) -> dict | None:
    """Reproduce the model's calibrated Gamma shape data for one cell."""
    if str(cfg.model[i, l]) != "gamma":
        return None

    from fleet_management.degradation_model.gamma_utils.gamma_repeated_calibration import (
        calibrate_gamma_cell_tail_bound,
        required_shape_for_tail,
    )

    def rate_profile(values, shape, name):
        if values is None:
            raise ValueError(f"Gamma warm start needs {name!r}.")
        array = np.asarray(values, dtype=float)
        if array.ndim == 2:
            return np.full(shape, float(array[i, l]), dtype=float)
        if array.ndim == 4:
            return np.broadcast_to(array[i, l], shape).astype(float, copy=True)
        raise ValueError(f"unsupported normalized {name} shape {array.shape}")

    operating_mu = np.asarray(cfg.mu[i, l], dtype=float)
    operating_beta = rate_profile(
        cfg.gamma_beta, operating_mu.shape, "gamma_beta"
    )
    beta_trans_cfg = getattr(cfg, "gamma_beta_trans", None)
    if cfg.mu_trans is None:
        indices = np.arange(cfg.H1) % cfg.H2
        transitory_mu = operating_mu[..., indices]
        transitory_beta = (
            operating_beta[..., indices]
            if beta_trans_cfg is None
            else rate_profile(
                beta_trans_cfg, transitory_mu.shape, "gamma_beta_trans"
            )
        )
    else:
        transitory_mu = np.asarray(cfg.mu_trans[i, l], dtype=float)
        transitory_beta = (
            operating_beta[..., np.arange(cfg.H1) % cfg.H2]
            if beta_trans_cfg is None
            else rate_profile(
                beta_trans_cfg, transitory_mu.shape, "gamma_beta_trans"
            )
        )

    combined_mu = np.concatenate((transitory_mu, operating_mu), axis=-1)
    combined_beta = np.concatenate(
        (transitory_beta, operating_beta), axis=-1
    )
    beta_bound = getattr(cfg, "gamma_beta_bound", None)
    selected_rate = (
        None if beta_bound is None
        else float(np.asarray(beta_bound, dtype=float)[i, l])
    )
    beta_0 = getattr(cfg, "gamma_beta_0", None)
    beta_new = getattr(cfg, "gamma_beta_new", None)
    kwargs = {
        "expected_damage": combined_mu,
        "rates": combined_beta,
        "threshold": float(cfg.tau[i, l]),
        "max_total_count": cfg.T,
        "initial_expected_damage": float(cfg.mu_0[i, l]),
        "initial_rate": None if beta_0 is None else float(beta_0[i, l]),
        "replacement_expected_damage": float(cfg.replacement_mu[i, l]),
        "replacement_rate": (
            None if beta_new is None else float(beta_new[i, l])
        ),
        "common_rate": selected_rate,
    }
    method = getattr(cfg, "gamma_calibration_method", "finite_count")
    if method == "repeated_increment":
        calibration = calibrate_gamma_cell_tail_bound(
            epsilon=float(cfg.epsilon[i, l]), **kwargs
        )
    elif method == "finite_count":
        from fleet_management.degradation_model.gamma_utils.gamma_tail_bound import (
            calculate_seeded_profile_tail_bound_parameters,
        )

        calibration = calculate_seeded_profile_tail_bound_parameters(**kwargs)
    else:
        raise ValueError(f"unsupported gamma_calibration_method={method!r}")

    split = cfg.H1
    return {
        "transitory": np.asarray(
            calibration.bounded_shapes[..., :split], dtype=float
        ),
        "operating": np.asarray(
            calibration.bounded_shapes[..., split:], dtype=float
        ),
        "initial": float(calibration.initial_bounded_shape),
        "replacement": float(calibration.replacement_bounded_shape),
        "maximum": float(required_shape_for_tail(
            float(cfg.epsilon[i, l]),
            float(calibration.common_rate),
            float(cfg.tau[i, l]),
        )),
    }


def _mission_index(x: np.ndarray, i: int, k: int) -> int | None:
    active = np.flatnonzero(x[i, 1:, k] > 0.5)
    return None if active.size == 0 else int(active[0])


def _simulate_cell(
    cfg,
    x: np.ndarray,
    m: np.ndarray,
    r: np.ndarray,
    i: int,
    l: int,
    gamma_data: dict | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Replay the implemented descriptors for one discrete schedule.

    The recursions mirror the solver's ARD-infinity and ARD1 dynamics. Gamma
    cells track both the physical mean and calibrated bounding-shape state.
    """
    mean = np.zeros(cfg.T, dtype=float)
    track_variance = _uses_variance_repeatability(cfg, i, l)
    variance = np.zeros(cfg.T, dtype=float) if track_variance else None
    shape = np.zeros(cfg.T, dtype=float) if gamma_data is not None else None
    mean_latch = 0.0
    variance_latch = 0.0
    shape_latch = 0.0
    rho = float(cfg.rho[i, l])
    remaining = 1.0 - rho
    repair_model = str(cfg.repair_model[i, l])

    for k in range(cfg.T):
        previous_mean = float(cfg.mu_0[i, l]) if k == 0 else mean[k - 1]
        previous_variance = (
            float(cfg.v_0[i, l]) if k == 0 else variance[k - 1]
        ) if track_variance else 0.0
        previous_shape = (
            gamma_data["initial"] if k == 0 else shape[k - 1]
        ) if gamma_data is not None else 0.0

        if r[i, l, k] > 0.5:
            mean[k] = float(cfg.replacement_mu[i, l])
            if track_variance:
                variance[k] = float(cfg.replacement_v[i, l])
            if gamma_data is not None:
                shape[k] = gamma_data["replacement"]
            if repair_model == "ard1":
                mean_latch = mean[k]
                if track_variance:
                    variance_latch = variance[k]
                if gamma_data is not None:
                    shape_latch = shape[k]
            continue

        if m[i, l, k] > 0.5:
            if repair_model == "ard1":
                mean[k] = remaining * previous_mean + rho * mean_latch
                if track_variance:
                    variance[k] = (
                        remaining * remaining * previous_variance
                        + (1.0 - remaining * remaining) * variance_latch
                    )
                mean_latch = mean[k]
                if track_variance:
                    variance_latch = variance[k]
                if gamma_data is not None:
                    shape[k] = (
                        remaining * previous_shape + rho * shape_latch
                    )
                    shape_latch = shape[k]
            else:
                mean[k] = remaining * previous_mean
                if track_variance:
                    variance[k] = remaining * remaining * previous_variance
                if gamma_data is not None:
                    shape[k] = remaining * previous_shape
            continue

        mission = _mission_index(x, i, k)
        mean[k] = previous_mean + (
            0.0 if mission is None
            else _mean_increment(cfg, i, l, mission, k)
        )
        if track_variance:
            variance[k] = previous_variance + (
                0.0 if mission is None
                else _variance_increment(cfg, i, l, mission, k)
            )
        if gamma_data is not None:
            if mission is None:
                shape_increment = 0.0
            elif k < cfg.H1:
                shape_increment = gamma_data["transitory"][mission, k]
            else:
                h = (k - cfg.H1) % cfg.H2
                shape_increment = gamma_data["operating"][mission, h]
            shape[k] = previous_shape + float(shape_increment)

    return mean, variance, shape


def _repeatability_score(
    cfg,
    mean: np.ndarray,
    variance: np.ndarray | None,
    shape: np.ndarray | None,
    gamma_data: dict | None,
    i: int,
    l: int,
) -> tuple[float, float]:
    """Return normalized terminal violation and maximum raw excess."""
    start, end = cfg.H1 - 1, cfg.T - 1
    mean_excess = max(0.0, float(mean[end] - mean[start]))
    mean_scale = max(float(cfg.tau[i, l]), 1e-12)
    normalized = [mean_excess / mean_scale]
    raw = [mean_excess]
    if variance is not None:
        variance_excess = max(
            0.0, float(variance[end] - variance[start])
        )
        normalized.append(
            variance_excess / max(mean_scale * mean_scale, 1e-12)
        )
        raw.append(variance_excess)
    if shape is not None and gamma_data is not None:
        shape_limit = max(float(gamma_data["maximum"]), 1e-12)
        shape_loop_excess = max(0.0, float(shape[end] - shape[start]))
        shape_reliability_excess = max(
            0.0, float(np.max(shape)) - shape_limit
        )
        normalized.extend([
            shape_loop_excess / shape_limit,
            shape_reliability_excess / shape_limit,
        ])
        raw.extend([shape_loop_excess, shape_reliability_excess])
    return max(normalized), max(raw)


def _enforce_repeatability(
    cfg,
    x: np.ndarray,
    m: np.ndarray,
    r: np.ndarray,
    *,
    tolerance: float = 1e-10,
) -> tuple[int, float, float]:
    """Insert operating-phase repairs until all proxy states close.

    Mission assignments are fixed at this point, so vehicle-component pairs
    are independent. Each iteration tries every unused depot slot in H2 and
    keeps the repair giving the smallest exact descriptor excess. For Gamma,
    this includes bounding-shape reliability and loop repeatability.
    """
    failures = 0
    maximum_excess = 0.0
    maximum_proxy_ratio = 0.0

    for i in range(cfg.F):
        for l in range(cfg.L):
            gamma_data = _gamma_shape_data(cfg, i, l)
            while True:
                mean, variance, shape = _simulate_cell(
                    cfg, x, m, r, i, l, gamma_data
                )
                maximum_proxy_ratio = max(
                    maximum_proxy_ratio,
                    float(np.max(mean)) / max(float(cfg.tau[i, l]), 1e-12),
                )
                if shape is not None:
                    maximum_proxy_ratio = max(
                        maximum_proxy_ratio,
                        float(np.max(shape)) / gamma_data["maximum"],
                    )
                score, raw_excess = _repeatability_score(
                    cfg, mean, variance, shape, gamma_data, i, l
                )
                if score <= tolerance:
                    break

                candidates = [
                    k for k in range(cfg.H1, cfg.T)
                    if x[i, 0, k] > 0.5
                    and m[i, l, k] < 0.5
                    and r[i, l, k] < 0.5
                ]
                best: tuple[tuple[float, float, int], int] | None = None
                for k in candidates:
                    m[i, l, k] = 1.0
                    candidate_mean, candidate_variance, candidate_shape = (
                        _simulate_cell(cfg, x, m, r, i, l, gamma_data)
                    )
                    candidate_score, candidate_raw = _repeatability_score(
                        cfg, candidate_mean, candidate_variance,
                        candidate_shape, gamma_data, i, l
                    )
                    m[i, l, k] = 0.0
                    key = (candidate_score, candidate_raw, -k)
                    if best is None or key < best[0]:
                        best = (key, k)

                if best is None or best[0][0] >= score - tolerance:
                    failures += 1
                    maximum_excess = max(maximum_excess, raw_excess)
                    break
                m[i, l, best[1]] = 1.0

            mean, variance, shape = _simulate_cell(
                cfg, x, m, r, i, l, gamma_data
            )
            _, raw_excess = _repeatability_score(
                cfg, mean, variance, shape, gamma_data, i, l
            )
            maximum_excess = max(maximum_excess, raw_excess)

    return failures, maximum_excess, maximum_proxy_ratio


def build_greedy_warm_start(
    cfg,
    *,
    repair_trigger: float = 0.70,
) -> tuple[dict, GreedyWarmStartReport]:
    """Construct a balanced, repair-aware discrete schedule.

    At each time step, missions are considered from largest to smallest
    normalized mean increment.  Each mission is assigned to the unused real
    vehicle with the smallest predicted maximum component-damage ratio.  A
    depot vehicle initially repairs a component when its proxy state exceeds
    ``repair_trigger * tau``. A second pass inserts additional repairs into
    operating-phase depot slots until the replayed mean, variance and Gamma
    bounding-shape descriptors satisfy reliability and repeatability.

    When ``F < M`` the missing mission assignments are counted as artificial
    assignments.  They are intentionally *not* represented as real vehicles
    in the returned MIP start.  Such a report is the trigger for the separate
    penalized-artificial-vehicle formulation; it must not be mistaken for a
    feasible start of the original model.
    """
    if not 0.0 <= repair_trigger <= 1.0:
        raise ValueError("repair_trigger must lie in [0, 1]")

    F, M, L, T = cfg.F, cfg.M, cfg.L, cfg.T
    x = np.zeros((F, M + 1, T), dtype=float)
    m = np.zeros((F, L, T), dtype=float)
    r = np.zeros((F, L, T), dtype=float)
    idle = np.zeros((F, L, T), dtype=float)
    state = np.asarray(cfg.mu_0, dtype=float).copy()

    artificial = 0
    maximum_ratio = 0.0
    assignment_order: list[list[int]] = []

    for k in range(T):
        # Largest normalized mission first makes the greedy choice stable and
        # reserves the healthiest real vehicle for the hardest duty.
        difficulty = []
        for j in range(M):
            value = max(
                _mean_increment(cfg, i, l, j, k)
                / max(float(cfg.tau[i, l]), 1e-12)
                for i in range(F)
                for l in range(L)
            )
            difficulty.append((value, j))
        missions = [j for _, j in sorted(difficulty, reverse=True)]
        assignment_order.append([j + 1 for j in missions])

        available = set(range(F))
        assigned: dict[int, int] = {}
        for j in missions:
            if not available:
                artificial += 1
                continue

            def score(i: int) -> tuple[float, float, int]:
                ratios = [
                    (state[i, l] + _mean_increment(cfg, i, l, j, k))
                    / max(float(cfg.tau[i, l]), 1e-12)
                    for l in range(L)
                ]
                # Secondary sum gives deterministic load balancing when the
                # maximum ratios tie; vehicle index is the final tie-breaker.
                return max(ratios), sum(ratios), i

            vehicle = min(available, key=score)
            available.remove(vehicle)
            assigned[vehicle] = j
            x[vehicle, j + 1, k] = 1.0

        for i in range(F):
            if i in assigned:
                j = assigned[i]
                for l in range(L):
                    state[i, l] += _mean_increment(cfg, i, l, j, k)
                    maximum_ratio = max(
                        maximum_ratio,
                        state[i, l] / max(float(cfg.tau[i, l]), 1e-12),
                    )
                continue

            # A real vehicle not serving a mission is placed in the depot.
            x[i, 0, k] = 1.0
            for l in range(L):
                threshold = repair_trigger * float(cfg.tau[i, l])
                if state[i, l] > threshold:
                    m[i, l, k] = 1.0
                    state[i, l] *= 1.0 - float(cfg.rho[i, l])
                else:
                    idle[i, l, k] = 1.0
                maximum_ratio = max(
                    maximum_ratio,
                    state[i, l] / max(float(cfg.tau[i, l]), 1e-12),
                )

    repeatability_failures, repeatability_excess, replay_ratio = (
        _enforce_repeatability(cfg, x, m, r)
    )
    maximum_ratio = max(maximum_ratio, replay_ratio)
    # Repairs inserted by the repeatability pass replace explicit idle actions.
    idle = np.maximum(0.0, x[:, 0, :][:, None, :] - m - r)

    warm_start = {
        "H1": cfg.H1,
        "H2": cfg.H2,
        "x": x,
        "m": m,
        "r": r,
        "idle": idle,
        # The heuristic determines the discrete schedule.  Ask the backend to
        # solve the corresponding fixed-binary continuous subproblem before
        # submitting the start to the original MIP.  This turns a partial
        # discrete start into a complete, solver-verified incumbent.
        "complete_binary_start": True,
        "completion_time_limit": 60.0,
    }
    report = GreedyWarmStartReport(
        feasible_assignment=artificial == 0,
        artificial_assignments=artificial,
        repeatability_feasible=repeatability_failures == 0,
        repeatability_failures=repeatability_failures,
        repairs=int(m.sum()),
        replacements=int(r.sum()),
        maximum_proxy_ratio=float(maximum_ratio),
        maximum_repeatability_excess=float(repeatability_excess),
        assignment_order=assignment_order,
    )
    return warm_start, report
