from __future__ import annotations

from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class DegradationModel:
    component_id: str

    def damage_increment(
            self,
            vehicle: dict[str, Any],
            mission: dict[str, Any],
            time_index: int,
    ) -> float:
        raise NotImplementedError
    
@dataclass(frozen=True)
class TireWearModel(DegradationModel):
    """
    Simple deterministic tire-wear model.

    Damage is dimensionless and normalized with threshold = 1.0 denoting admissible wear limit.

    First version is intentionally simple:
        damage = base_rate_per_km * distance_km * mass_factor * road_factor * driving_style_factor
    """

    base_rate_per_km: float
    reference_mass_kg: float = 18000.0

    def damage_increment(self, vehicle:  dict[str, Any], mission: dict[str, Any], time_index: int) -> float:
        distance_km = float(mission["distance_km"])
        vehicle_mass_kg = float(vehicle.get("mass_kg", self.reference_mass_kg))

        road_factor = float(mission.get("road_factor", 1.0))
        driving_style_factor = float(mission.get("driving_style_factor", 1.0))

        mass_factor = vehicle_mass_kg / self.reference_mass_kg

        return (
            self.base_rate_per_km
            * distance_km
            * mass_factor
            * road_factor
            * driving_style_factor
        )
    
def build_degradation_model(component: dict[str, Any]) -> DegradationModel:
    model_type = component["model"]

    if model_type == "tire_wear_linear":
        return TireWearModel(
            component_id=component["id"],
            base_rate_per_km=float(component["base_rate_per_km"]),
            reference_mass_kg=float(component.get("reference_mass_kg", 18000.0)),
        )
    
    raise ValueError(f"Unsupported degradation model: {model_type}")
