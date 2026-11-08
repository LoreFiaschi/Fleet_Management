"""Review of Task B: import the package with gurobipy and streamlit refused, then compare each
exported signature with the spec's (lore copy, lines 214-2651). Reads only."""
import inspect
import sys

sys.path.insert(0, r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge\src")
sys.modules["gurobipy"] = None
sys.modules["streamlit"] = None

import fleet_management  # noqa: E402

SPEC = {
    "solve": '(input_path, results_path="output.yaml", *, warm_start=None, warm_start_mode="start")',
    "sweep_operating_horizons": '(input_path, operating_horizons, *, output_path=None, stop_on_gradient=False, '
        'gradient_tolerance=0.001, flat_gradients_required=2, minimum_cases=3, maximum_mip_gap_for_stopping=None, '
        'warm_start=False, warm_start_mode="start", n_workers=1, objective_mode="operating_average", '
        'evaluation_horizon=None)',
    "sweep_horizon_grid": '(input_path, transitory_horizons, operating_horizons, *, output_path=None, '
        'warm_start=True, evaluation_horizon=None, **sweep_options)',
    "plot_management": '(input_file_path, plot_file_path=None)',
    "plot_horizon_sweep": '(sweep_report_path, plot_file_path=None)',
    "validate": '(input_path, results_path, validation_path="validation.yaml", tol=1e-06)',
    "validate_gamma_replay_schedule": '(cfg, result, *, tolerance=1e-08, raise_on_failure=False)',
    "validate_gamma_replay_files": '(input_path, result_path, report_path=None, **kwargs)',
    "validate_gamma_stochastic_schedule": '(cfg, result, *, repetitions=100000, random_seed=20260908, '
        'maximum_schedule_failure_rate=None, confidence_level=0.95, batch_size=20000, tolerance=1e-08, '
        'raise_on_failure=False)',
    "validate_gamma_schedule": '(cfg, result, *, mode=None, repetitions=None, random_seed=None, '
        'maximum_schedule_failure_rate=None, confidence_level=None, batch_size=None, tolerance=1e-08, '
        'raise_on_failure=False)',
    "validate_gamma_schedule_files": '(input_path, result_path, report_path=None, **kwargs)',
}


def plain(function) -> str:
    """The signature without annotations, with double quotes, as the spec writes it."""
    sig = inspect.signature(function)
    params = [p.replace(annotation=inspect.Parameter.empty) for p in sig.parameters.values()]
    text = str(sig.replace(parameters=params, return_annotation=inspect.Signature.empty))
    return text.replace("'", '"')


print("names in order:", list(fleet_management.__all__) == list(SPEC))
bad = 0
for name, expected in SPEC.items():
    got = plain(getattr(fleet_management, name))
    ok = got == expected
    bad += not ok
    ret = inspect.signature(getattr(fleet_management, name)).return_annotation
    print(("OK  " if ok else "DIFF"), name, "| return:", ret)
    if not ok:
        print("   spec:", expected)
        print("   code:", got)
print("loaded:", {m: m in sys.modules for m in ("fleet_management.degradation_model.base",
                                                "fleet_management.utils.mixed_plotter")})
print("differences:", bad)
