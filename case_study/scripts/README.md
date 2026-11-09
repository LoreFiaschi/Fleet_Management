# Computation scripts of the component dossiers

The research agents wrote these scripts in the session scratchpad on 9 Nov 2026. They were copied here so
that the numbers of the dossiers can be rebuilt. The dossiers cite them under their scratchpad paths
(`scripts_bat\`, `scripts_tyre\`, `scripts_inv\`).

| Script | Dossier | What it computes |
|---|---|---|
| `battery/blast_numbers.py` | `component_battery.md` | calendar and cycle fade coefficients from the NREL BLAST-Lite models |
| `battery/line_increments.py` | `component_battery.md` | battery fade per line and period |
| `tires/line_increments.py` | `component_tires.md` | tread wear per line and period |
| `tires/usable_fraction.py` | `component_tires.md` | the share of the tread usable at a given spread of the wear rate |
| `inverter/calib_lines.py` | `component_inverter.md` | the temperature swing per stop that meets the design life |
| `inverter/damage_lines.py` | `component_inverter.md` | Miner damage of the power modules per line and period |
| `inverter/weibull_choi.py` | `component_inverter.md` | the Weibull shape of power-cycling life from Choi et al. 2017 |
