from __future__ import annotations

from pathlib import Path

import yaml

from fleet_management.ev_instance import read_ev_instance
from fleet_management.ev_deterministic import solve_ev_deterministic


def solve_ev(
    input_path: str,
    results_path: str = "results/output_ev.yaml",
) -> dict:
    instance = read_ev_instance(input_path)

    result = solve_ev_deterministic(
        F=instance["F"],
        M=instance["M"],
        L=instance["L"],
        H=instance["H"],
        damage_increment=instance["damage_increment"],
        initial_damage=instance["initial_damage"],
        repair_fraction=instance["repair_fraction"],
        alpha=instance["alpha"],
        C_M=instance["C_M"],
        C_D=instance["C_D"],
        C_R=instance["C_R"],
        verbose=instance["verbose"],
        mip_gap=instance["mip_gap"],
    )

    output = {
        key: value
        for key, value in result.items()
        if key != "model"
    }

    # Convert numpy arrays to lists for yaml
    for key in ("x", "D", "z", "u"):
        if output[key] is not None:
            output[key] = output[key].tolist()
    
    output["vehicles"] = instance["vehicles"]
    output["missions"] = instance["missions"]
    output["components"] = instance["components"]
    output["intitial_damage"] = instance["initial_damage"].tolist()
    output["repair_fraction"] = instance["repair_fraction"].tolist()
    output["damage_increment"] = instance["damage_increment"].tolist()

    path = Path(results_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        yaml.safe_dump(output, f, sort_keys=False)
    
    return result
