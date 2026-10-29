"""Mutation check of tests/test_gamma_validators.py against the scratch prototype (proto_src).

Each mutation edits one line of a copy of proto_src; the test module must then fail.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SP = HERE.parent.parent
RUNNER = SP / "wp6" / "gamma" / "run_tests_gamma.py"
TEST = r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge\tests\test_gamma_validators.py"
GAMMA = "fleet_management/degradation_model/gamma_utils"

MUTATIONS = [
    ("replay two-sided", "gamma_replay_validator", 'ref_mu, saved, max(0.0, ref_mu - saved))',
     'ref_mu, saved, abs(ref_mu - saved))'),
    ("replay anchor starts at 0", "gamma_replay_validator",
     "ref_g = float(cfg.anchor_mu_0[i, l]) if anchored else 0.0", "ref_g = 0.0"),
    ("replay cap = tau", "gamma_replay_validator",
     'check(i, l, k, event, "cap", caps[i, l], saved, max(0.0, saved - caps[i, l]))',
     'check(i, l, k, event, "cap", 1.0, saved, max(0.0, saved - 1.0))'),
    ("replay drops depot-day violation", "gamma_replay_validator",
     "if x[i, 0, k] == 0:", "if False:"),
    ("replay z NaN not skipped", "gamma_replay_validator",
     "if not math.isnan(z[i, l, k]):", "if True:"),
    ("replay horizon not checked", "gamma_replay_validator",
     "if T is None or int(T) != cfg.T:", "if False:"),
    ("exact: no anchor split", "gamma_tail_validator",
     'g_0 = float(cfg.anchor_mu_0[i, l])', 'g_0 = 0.0'),
    ("exact: repair keeps rate", "gamma_tail_validator",
     "[t.after_ardinf(remaining) for t in active]", "list(active)"),
    ("exact: rho = 1 keeps terms", "gamma_tail_validator",
     "repaired = [] if remaining == 0.0 else", "repaired = "),
    ("exact: reliability two-sided on saved", "gamma_tail_validator",
     'check(i, l, k, event, "reliability", eps, upper, max(0.0, upper - eps))',
     'check(i, l, k, event, "reliability", eps, upper, max(0.0, upper - eps - 0.01))'),
    ("mc: no anchor split", "gamma_stochastic_validator",
     "latch = _draw(rng, g_0, beta_0, size)", "latch = np.zeros(size)"),
    ("mc: rate 1 for a constant", "gamma_stochastic_validator",
     "if math.isnan(rate):\n        return np.full(size, float(mean))", "if math.isnan(rate):\n        rate = 1.0"),
    ("mc: int() on repetitions", "gamma_stochastic_validator",
     'repetitions = _integer(repetitions, "repetitions")', "repetitions = int(repetitions)"),
    ("dispatcher: mode lower-cased", "gamma_stochastic_validator",
     'if settings["mode"] not in VALIDATION_MODES:',
     'settings["mode"] = str(settings["mode"]).strip().lower()\n    if settings["mode"] not in VALIDATION_MODES:'),
    ("dispatcher: unknown key ignored", "gamma_stochastic_validator",
     "if key not in _SETTING_DEFAULTS:", "if False:"),
    ("dispatcher: both raises early", "gamma_stochastic_validator",
     "stochastic = validate_gamma_stochastic_schedule(cfg, result, tolerance=tolerance,\n"
     "                                                        raise_on_failure=False, **settings)",
     "stochastic = validate_gamma_stochastic_schedule(cfg, result, tolerance=tolerance,\n"
     "                                                        raise_on_failure=False, **settings)\n"
     "        assert deterministic['valid']"),
    ("file: report path checked late", "gamma_replay_validator",
     "    destination = _check_report_path(report_path)\n    raise_on_failure = kwargs.pop(\"raise_on_failure\", False)\n    cfg, result = _read_pair(input_path, result_path)\n",
     "    raise_on_failure = kwargs.pop(\"raise_on_failure\", False)\n    cfg, result = _read_pair(input_path, result_path)\n    destination = _check_report_path(report_path)\n"),
    ("file: raise before write", "gamma_replay_validator",
     "    if destination:\n        destination.parent.mkdir",
     "    if raise_on_failure and not report['valid']:\n        raise AssertionError('x')\n    if destination:\n        destination.parent.mkdir"),
    ("file: no folder created", "gamma_replay_validator",
     "destination.parent.mkdir(parents=True, exist_ok=True)", "pass"),
]


def run(index, name, module, old, new):
    work = HERE / "mutations" / f"m{index:02d}"       # one fresh copy per mutation; nothing is deleted
    if not work.exists():
        shutil.copytree(HERE / "proto_src", work)
    path = work / GAMMA / f"{module}.py"
    text = path.read_text(encoding="utf-8")
    if old not in text:
        return f"{name}: PATTERN NOT FOUND"
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    env = dict(os.environ, FM_SRC=str(work), PYTHONIOENCODING="utf-8")
    out = subprocess.run([sys.executable, str(RUNNER), TEST], capture_output=True, text=True, env=env)
    last = (out.stdout.strip().splitlines() or [out.stderr[-200:]])[-1]
    return f"{name}: {last[:260]}"


if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else ""
    for index, mutation in enumerate(MUTATIONS):
        if only in mutation[0]:
            print(run(index, *mutation), flush=True)
