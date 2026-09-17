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
    repairs: int
    replacements: int
    maximum_proxy_ratio: float
    assignment_order: list[list[int]]

    def as_dict(self) -> dict:
        return {
            "feasible_assignment": self.feasible_assignment,
            "artificial_assignments": self.artificial_assignments,
            "repairs": self.repairs,
            "replacements": self.replacements,
            "maximum_proxy_ratio": self.maximum_proxy_ratio,
            "assignment_order": self.assignment_order,
        }


def _mean_increment(cfg, i: int, l: int, mission: int, k: int) -> float:
    """Read a normalized phase-aware mean increment from ``FleetConfig``."""
    if k < cfg.H1 and cfg.mu_trans is not None:
        return float(cfg.mu_trans[i, l, mission, k])
    h = (k - cfg.H1) % cfg.H2 if k >= cfg.H1 else k % cfg.H2
    return float(cfg.mu[i, l, mission, h])


def build_greedy_warm_start(
    cfg,
    *,
    repair_trigger: float = 0.70,
) -> tuple[dict, GreedyWarmStartReport]:
    """Construct a balanced, repair-aware discrete schedule.

    At each time step, missions are considered from largest to smallest
    normalized mean increment.  Each mission is assigned to the unused real
    vehicle with the smallest predicted maximum component-damage ratio.  A
    depot vehicle repairs a component when its proxy state exceeds
    ``repair_trigger * tau``; otherwise that component is explicitly idle.

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

    warm_start = {
        "H1": cfg.H1,
        "H2": cfg.H2,
        "x": x,
        "m": m,
        "r": r,
        "idle": idle,
    }
    report = GreedyWarmStartReport(
        feasible_assignment=artificial == 0,
        artificial_assignments=artificial,
        repairs=int(m.sum()),
        replacements=int(r.sum()),
        maximum_proxy_ratio=float(maximum_ratio),
        assignment_order=assignment_order,
    )
    return warm_start, report
