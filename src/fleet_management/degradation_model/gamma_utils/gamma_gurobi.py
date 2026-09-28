"""Compatibility import for the relocated legacy Gamma backend.

New code should import from ``degradation_model.legacy.gamma_gurobi`` or use
the public :func:`fleet_management.solve` entry point.
"""

from fleet_management.degradation_model.legacy.gamma_gurobi import (
    solve_fleet_management,
    validate_inputs,
)

__all__ = ["solve_fleet_management", "validate_inputs"]
