Slide 8 validator rerun
=======================

Purpose
-------
Generate matched validation evidence for Gamma ARD-infinity and ARD1 with
replacement enabled. Each fixed schedule exercises one repair and three
replacements. The script runs deterministic replay, 100,000 fixed-seed
stochastic replays, and a deliberate corruption test.

Installation
------------
Extract this archive into the Fleet_Management repository root. The script
must end up at:

  examples/regression/report_gamma_validator_slide8.py

It relies on these existing repository files:

  examples/regression/gamma_replacement_truth_table_common.py
  src/fleet_management/degradation_model/gamma_utils/gamma_replay_validator.py
  src/fleet_management/degradation_model/gamma_utils/gamma_stochastic_validator.py

Run locally from PowerShell
---------------------------
  cd C:\path\to\Fleet_Management
  .\.venv\Scripts\Activate.ps1
  python .\examples\regression\report_gamma_validator_slide8.py

Optional quick smoke test
-------------------------
  python .\examples\regression\report_gamma_validator_slide8.py `
    --repetitions 10000 `
    --batch-size 5000 `
    --output-dir .\results\validator_slide8_smoke

Expected outputs
----------------
  results/validator_slide8/ardinf_validation.yaml
  results/validator_slide8/ard1_validation.yaml
  results/validator_slide8/validator_slide8_summary.yaml
  results/validator_slide8/validator_slide8_summary.csv

The CSV contains the values needed for Slide 8. "Failed stochastic replays"
means sampled trajectories in which at least one Gamma state exceeded its
failure threshold. It does not mean the validator program crashed.
