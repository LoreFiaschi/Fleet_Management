import sys
import tempfile
from pathlib import Path
sys.path.insert(0, r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge\src")
from fleet_management import horizon_sweep as h

DEFAULTS = dict(output_path=None, stop_on_gradient=False, gradient_tolerance=1e-3, flat_gradients_required=2,
                minimum_cases=3, maximum_mip_gap_for_stopping=None, warm_start=False, warm_start_mode="start",
                n_workers=1, objective_mode="operating_average", evaluation_horizon=None)

ok = h._check_sweep_arguments(2, [4, 2, 3], **DEFAULTS)
assert ok["candidates"] == [2, 3, 4] and ok["target"] is None and ok["maximum_mip_gap_for_stopping"] is None
assert ok["gradient_tolerance"] == 1e-3 and ok["n_workers"] == 1 and ok["evaluation_horizon"] is None

CASES = [
    (dict(ops=[]), "operating_horizons must contain at least one value."),
    (dict(ops=[2, 2]), "operating_horizons must not contain duplicates."),
    (dict(ops=[0]), "operating_horizons must contain positive integers (got 0)."),
    (dict(ops=[2.5]), "operating_horizons must contain positive integers (got 2.5)."),
    (dict(ops=[True]), "operating_horizons must contain positive integers (got True)."),
    (dict(gradient_tolerance=-1), "gradient_tolerance must be a finite number >= 0."),
    (dict(gradient_tolerance=float("inf")), "gradient_tolerance must be a finite number >= 0."),
    (dict(gradient_tolerance=True), "gradient_tolerance must be a finite number >= 0."),
    (dict(flat_gradients_required=0), "flat_gradients_required must be an integer >= 1."),
    (dict(minimum_cases=1), "minimum_cases must be an integer >= 2."),
    (dict(maximum_mip_gap_for_stopping=-0.1), "maximum_mip_gap_for_stopping must be None or a finite number >= 0."),
    (dict(warm_start_mode="hints"), "warm_start_mode must be one of ('start', 'hint', 'both') (got 'hints')."),
    (dict(n_workers=0), "n_workers must be an integer >= 1."),
    (dict(n_workers=2, warm_start=True), "warm_start=True needs n_workers=1: a warm start uses the preceding case."),
    (dict(n_workers=2, stop_on_gradient=True), "stop_on_gradient=True needs n_workers=1: the early stop needs the cases in order."),
    (dict(objective_mode="average"), "objective_mode must be one of ('total', 'operating_average', 'evaluation_total') (got 'average')."),
    (dict(objective_mode="evaluation_total"), "objective_mode 'evaluation_total' needs evaluation_horizon."),
    (dict(evaluation_horizon=2), "evaluation_horizon must be an integer > H1 = 2 (got 2)."),
    (dict(output_path="r.json"), "output_path must end in .yaml or .yml."),
]
for change, message in CASES:
    kwargs = dict(DEFAULTS)
    ops = change.pop("ops", [2, 3, 4])
    kwargs.update(change)
    try:
        h._check_sweep_arguments(2, ops, **kwargs)
    except ValueError as error:
        assert str(error) == message, (str(error), message)
    else:
        raise AssertionError(f"no error for {change}")

# no parent folder is created when another check fails
with tempfile.TemporaryDirectory() as d:
    target = Path(d) / "new" / "sub" / "r.yaml"
    kwargs = dict(DEFAULTS, output_path=target, minimum_cases=1)
    try:
        h._check_sweep_arguments(2, [2, 3], **kwargs)
    except ValueError:
        pass
    assert not (Path(d) / "new").exists()
    kwargs = dict(DEFAULTS, output_path=target)
    ok = h._check_sweep_arguments(2, [2, 3], **kwargs)
    assert ok["target"] == target and target.parent.is_dir()
print("A5 check OK", len(CASES), "messages")
