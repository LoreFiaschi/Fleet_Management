# Read simple EV Yaml and convert to solver arrays

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from fleet_management.degradation_models import build_degradation_model


def read_ev_instance(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    with open(path, "r") as f:
        data = yaml.safe_load(f)
    
    return extract_ev_instance(data)


def extract_ev_instance(data: dict[str, Any]) -> dict[str, Any]:
    vehicles = data["vehicles"]
    missions = data["missions"]
    components = data["components"]

    F = len(vehicles)
    M = len(missions)
    L = len(components)
    H = int(data["H"])

    damage_increment = np.zeros((F, M, L, H), dtype=float)

    for ell, component in enumerate(components):
        model = build_degradation_model(component)

        for i, vehicle in enumerate(vehicles):
            for j, mission in enumerate(missions):
                for k in range(H):
                    damage_increment[i, j, ell, k] = model.damage_increment(
                        vehicle=vehicle,
                        mission=mission,
                        time_index=k
                    )
    
    initial_damage = np.asarray(data["initial_damage"], dtype=float)
    repair_fraction = np.asarray(data["repair_fraction"], dtype=float)

    if initial_damage.shape != (F, L):
        raise ValueError(
            f"initial_damage shape {initial_damage.shape} does not match "
            f"(F={F}, L={L})."
        )
    
    if repair_fraction.shape != (F, L):
        raise ValueError(
            f"repair_fraction shape {repair_fraction.shape} does not match "
            f"(F={F}, L={L})."
        )
    
    return {
        "F": F,
        "M": M,
        "L": L,
        "H": H,
        "vehicles": vehicles,
        "missions": missions,
        "components": components,
        "damage_increment": damage_increment,
        "initial_damage": initial_damage,
        "repair_fraction": repair_fraction,
        "alpha": float(data["alpha"]),
        "C_M": float(data["C_M"]),
        "C_D": float(data["C_D"]),
        "C_R": np.asarray(data["C_R"], dtype=float),
        "verbose": int(data.get("verbose", 1)),
        "mip_gap": (
            None
            if data.get("mip_gap", None) is None
            else float(data["mip_gap"])
        ),
    }
