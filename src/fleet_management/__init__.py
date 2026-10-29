"""Public API of the package fleet_management (spec, "Public API").

``import fleet_management`` works without Gurobi: the modules below import ``gurobipy``,
or ``degradation_model.base``, only inside the functions that solve.
"""

from fleet_management.solver import solve
from fleet_management.horizon_sweep import sweep_horizon_grid, sweep_operating_horizons
from fleet_management.utils.plotter import plot_horizon_sweep, plot_management
from fleet_management.validation.validator import validate
from fleet_management.degradation_model.gamma_utils.gamma_replay_validator import (
    validate_gamma_replay_files,
    validate_gamma_replay_schedule,
)
from fleet_management.degradation_model.gamma_utils.gamma_stochastic_validator import (
    validate_gamma_schedule,
    validate_gamma_schedule_files,
    validate_gamma_stochastic_schedule,
)


__all__ = [
    "solve",
    "sweep_operating_horizons",
    "sweep_horizon_grid",
    "plot_management",
    "plot_horizon_sweep",
    "validate",
    "validate_gamma_replay_schedule",
    "validate_gamma_replay_files",
    "validate_gamma_stochastic_schedule",
    "validate_gamma_schedule",
    "validate_gamma_schedule_files",
]
