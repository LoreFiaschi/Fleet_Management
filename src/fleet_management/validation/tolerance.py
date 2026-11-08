"""Tolerances of the validators (the structural validator and the Gamma validators).

A tolerance is numerical: it says how large a gap between two numbers counts as zero.
Gurobi accepts a row violated by up to ``FeasibilityTol`` and a binary up to ``IntFeasTol``
away from 0 or 1, and a binary inside a big-M row moves a tracked state by up to
``IntFeasTol * M_s``. The tolerance of a check is resolved in this order:

1. the explicit argument, when it is a number: one value for every check;
2. else the key ``tolerance`` of the input block ``validation``: one value for every check;
3. else a default per check, from ``FeasibilityTol`` and ``IntFeasTol``:

   * a tracked quantity, an anchor, a start ``y_*``, ``z``, the cap of a Gamma cell:
     ``FeasibilityTol + IntFeasTol * M_s``, with ``M_s`` the natural range of the quantity
     at the cell (``cfg.range_mu``, ``range_v``, ``range_R``, ``range_K``);
   * a binary check: ``IntFeasTol``;
   * a probability or a cost: ``FeasibilityTol``.

``FeasibilityTol`` and ``IntFeasTol`` come from ``cfg.options["gurobi_params"]`` (the names
are matched without regard to case), else Gurobi's defaults.
"""

from __future__ import annotations

import math

import numpy as np

GUROBI_FEASIBILITY_TOL = 1e-6
GUROBI_INT_FEAS_TOL = 1e-5

# the natural range that a quantity of a result uses (an anchor and a start use that of the state)
_RANGE_OF = {"mu": "range_mu", "v": "range_v", "R": "range_R", "K": "range_K"}


def _positive_finite(value, name: str) -> float:
    """``value`` as a float; a ``ValueError`` naming ``name`` unless it is a finite number > 0."""
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not (math.isfinite(float(value)) and float(value) > 0.0)):
        raise ValueError(f"{name} must be a finite number > 0 (got {value!r}).")
    return float(value)


def gurobi_tolerances(cfg) -> tuple[float, float]:
    """``(FeasibilityTol, IntFeasTol)`` of the input: ``gurobi_params``, else Gurobi's defaults."""
    params = (getattr(cfg, "options", None) or {}).get("gurobi_params") or {}
    found = {str(key).lower(): value for key, value in params.items()}
    feasibility = found.get("feasibilitytol", GUROBI_FEASIBILITY_TOL)
    integrality = found.get("intfeastol", GUROBI_INT_FEAS_TOL)
    return (_positive_finite(feasibility, "gurobi_params FeasibilityTol"),
            _positive_finite(integrality, "gurobi_params IntFeasTol"))


def block_tolerance(cfg) -> float | None:
    """The key ``tolerance`` of the input block ``validation``, or ``None`` (absent or null)."""
    block = (getattr(cfg, "options", None) or {}).get("validation") or {}
    value = block.get("tolerance")
    return None if value is None else _positive_finite(value, "validation.tolerance")


def resolve_tolerance(cfg, argument) -> float | None:
    """Steps 1 and 2: the single tolerance for every check, or ``None`` (step 3 applies)."""
    if argument is not None:
        return float(argument)
    return block_tolerance(cfg)


def report_tolerance(cfg, single: float | None) -> float | dict:
    """The report key ``tolerance``: the single value, or the two Gurobi tolerances."""
    if single is not None:
        return float(single)
    feasibility, integrality = gurobi_tolerances(cfg)
    return {"FeasibilityTol": feasibility, "IntFeasTol": integrality}


def binary_tolerance(cfg, single: float | None) -> float:
    """Tolerance of a check on binaries: ``single``, else ``IntFeasTol``."""
    return float(single) if single is not None else gurobi_tolerances(cfg)[1]


def cost_tolerance(cfg, single: float | None) -> float:
    """Tolerance of a check on a probability or a cost: ``single``, else ``FeasibilityTol``."""
    return float(single) if single is not None else gurobi_tolerances(cfg)[0]


def state_tolerance(cfg, q: str, single: float | None = None, i=None, l=None):
    """Tolerance of a check on the quantity ``q`` (``mu``, ``v``, ``R``, ``K``, ``g_mu``, ``y_v``, ...).

    ``single`` if given, else ``FeasibilityTol + IntFeasTol * M_s`` with ``M_s`` the natural
    range of the quantity. Without ``i`` and ``l`` an ``F x L`` array; with them a float.
    Where the range is not defined (a cell that does not track the quantity) the tolerance is
    ``FeasibilityTol``. ``z`` and the cap of a Gamma cell are checked as ``q = "mu"``.
    """
    shape = (cfg.F, cfg.L)
    if single is not None:
        table = np.full(shape, float(single))
    else:
        feasibility, integrality = gurobi_tolerances(cfg)
        base = q.split("_", 1)[1] if q.startswith(("g_", "y_")) else q
        span = getattr(cfg, _RANGE_OF[base])
        if span is None and base == "mu":
            span = cfg.tau
        span = np.zeros(shape) if span is None else np.broadcast_to(
            np.asarray(span, dtype=float), shape)
        table = feasibility + integrality * np.nan_to_num(span, nan=0.0, posinf=0.0)
    return table if i is None else float(table[i, l])
