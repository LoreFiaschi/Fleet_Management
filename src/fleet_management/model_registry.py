"""
Central degradation-model registry and parameter extraction.

This module is intended to be the single place where new degradation families
are registered. Solver, validator, and dashboard code should use this module
instead of duplicating degradation-specific parsing logic.
"""
# DEPRECATED
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


SUPPORTED_DEGRADATIONS = {
    "gaussian",
    "inverse_gaussian",
    "gamma",
}


COMMON_KEYS = {
    "F",
    "M",
    "H",
    "L",
    "mu_0",
    "mu",
}


GAUSSIAN_KEYS = {
    "alpha",
    "v_0",
    "v",
    "C_M",
    "C_R",
    "C_S",
    "C_P",
}


INVERSE_GAUSSIAN_KEYS = {
    "alpha",
    "c",
    "C_M",
    "C_R",
    "C_S",
    "C_P",
}


GAMMA_KEYS = {
    "tau",
    "epsilon",
    "gamma_beta",
}


REQUIRED_KEYS_BY_DEGRADATION = {
    "gaussian": COMMON_KEYS | GAUSSIAN_KEYS,
    "inverse_gaussian": COMMON_KEYS | INVERSE_GAUSSIAN_KEYS,
    "gamma": COMMON_KEYS | GAMMA_KEYS,
}


@dataclass(frozen=True)
class DegradationModelSpec:
    name: str
    required_keys: set[str]
    extractor: Callable[[dict], dict]


def validate_required_keys(data: dict, degradation: str) -> None:
    """Check that the input dictionary contains all required keys."""

    degradation_lower = degradation.lower()

    if degradation_lower not in REQUIRED_KEYS_BY_DEGRADATION:
        raise ValueError(
            f"Unsupported degradation type '{degradation}'. "
            f"Supported types: {sorted(REQUIRED_KEYS_BY_DEGRADATION)}"
        )

    required_keys = REQUIRED_KEYS_BY_DEGRADATION[degradation_lower]
    missing = required_keys - set(data.keys())

    if missing:
        raise KeyError(
            f"Missing required keys for degradation '{degradation_lower}': "
            f"{sorted(missing)}"
        )


def _as_2d_array(value, F: int, L: int, name: str) -> np.ndarray:
    """
    Convert value to shape (F, L).

    Accepts:
    - shape (F, L)
    - shape (F,) if L == 1
    """

    arr = np.asarray(value, dtype=float)

    if arr.shape == (F, L):
        return arr

    if L == 1 and arr.shape == (F,):
        return arr[:, np.newaxis]

    raise ValueError(
        f"'{name}' shape {arr.shape} does not match expected shape "
        f"(F={F}, L={L})."
    )


def broadcast_4d_param(
    value,
    F: int,
    M: int,
    L: int,
    H: int,
    name: str,
) -> np.ndarray:
    """
    Broadcast degradation parameter to shape (F, M, L, H).

    Accepted shapes:
    - (F, M, L, H)
    - (M, L, H)
    - (M, L)
    - (F, M, H) if L == 1
    - (F, M) if L == 1
    - scalar
    """

    arr = np.asarray(value, dtype=float)

    if arr.shape == (F, M, L, H):
        return arr

    if arr.shape == (M, L, H):
        return np.broadcast_to(arr[np.newaxis, :, :, :], (F, M, L, H)).copy()

    if arr.shape == (M, L):
        return np.broadcast_to(arr[np.newaxis, :, :, np.newaxis], (F, M, L, H)).copy()

    if L == 1 and arr.shape == (F, M, H):
        return arr[:, :, np.newaxis, :]

    if L == 1 and arr.shape == (F, M):
        return np.broadcast_to(arr[:, :, np.newaxis, np.newaxis], (F, M, L, H)).copy()

    if arr.shape == ():
        return np.full((F, M, L, H), float(arr))

    raise ValueError(
        f"'{name}' shape {arr.shape} cannot be broadcast to "
        f"(F={F}, M={M}, L={L}, H={H})."
    )


def extract_common_parameters(data: dict) -> dict:
    """Extract parameters shared by all degradation models."""

    F = int(data["F"])
    M = int(data["M"])
    H = int(data["H"])
    L = int(data.get("L", 1))

    mu_0 = _as_2d_array(data["mu_0"], F, L, "mu_0")
    mu_param = broadcast_4d_param(data["mu"], F, M, L, H, "mu")

    return {
        "F": F,
        "M": M,
        "H": H,
        "L": L,
        "mu_0": mu_0,
        "mu_param": mu_param,
    }


def extract_gaussian_parameters(data: dict) -> dict:
    """Extract Gaussian degradation parameters."""

    params = extract_common_parameters(data)

    F = params["F"]
    M = params["M"]
    H = params["H"]
    L = params["L"]

    v_0 = _as_2d_array(data["v_0"], F, L, "v_0")
    v_param = broadcast_4d_param(data["v"], F, M, L, H, "v")

    params.update(
        {
            "degradation": "gaussian",
            "alpha": float(data["alpha"]),
            "v_0": v_0,
            "v_param": v_param,
            "C_M": float(data["C_M"]),
            "C_R": float(data["C_R"]),
            "C_S": float(data["C_S"]),
            "C_P": float(data["C_P"]),
        }
    )

    return params


def extract_inverse_gaussian_parameters(data: dict) -> dict:
    """Extract inverse-Gaussian degradation parameters."""

    params = extract_common_parameters(data)

    params.update(
        {
            "degradation": "inverse_gaussian",
            "alpha": float(data["alpha"]),
            "c": np.asarray(data["c"], dtype=float),
            "C_M": float(data["C_M"]),
            "C_R": float(data["C_R"]),
            "C_S": float(data["C_S"]),
            "C_P": float(data["C_P"]),
        }
    )

    return params


def extract_gamma_parameters(data: dict) -> dict:
    """
    Extract Gamma-process degradation parameters.

    The project convention is shape-rate:

        D ~ Gamma(A, beta)
        E[D] = A / beta

    The input still stores expected damage increments mu[i,j,l,k].
    These are later converted to Gamma shape increments by:

        A_increment = beta_l * mu[i,j,l,k]

    beta is component-specific and must remain fixed for closure.
    """

    params = extract_common_parameters(data)

    L = params["L"]

    gamma_beta = np.asarray(data["gamma_beta"], dtype=float)

    if gamma_beta.shape == ():
        gamma_beta = np.full((L,), float(gamma_beta))

    if gamma_beta.shape != (L,):
        raise ValueError(
            f"'gamma_beta' shape {gamma_beta.shape} does not match expected "
            f"shape (L={L},)."
        )

    if np.any(gamma_beta <= 0.0):
        raise ValueError("'gamma_beta' values must be strictly positive.")

    tau = float(data["tau"])
    epsilon = float(data["epsilon"])

    if tau <= 0.0:
        raise ValueError(f"'tau' must be positive, got {tau}.")

    if epsilon < 0.0 or epsilon > 1.0:
        raise ValueError(f"'epsilon' must be in [0, 1], got {epsilon}.")

    params.update(
        {
            "degradation": "gamma",
            "tau": tau,
            "epsilon": epsilon,
            "gamma_beta": gamma_beta,
        }
    )

    return params


MODEL_REGISTRY = {
    "gaussian": DegradationModelSpec(
        name="gaussian",
        required_keys=REQUIRED_KEYS_BY_DEGRADATION["gaussian"],
        extractor=extract_gaussian_parameters,
    ),
    "inverse_gaussian": DegradationModelSpec(
        name="inverse_gaussian",
        required_keys=REQUIRED_KEYS_BY_DEGRADATION["inverse_gaussian"],
        extractor=extract_inverse_gaussian_parameters,
    ),
    "gamma": DegradationModelSpec(
        name="gamma",
        required_keys=REQUIRED_KEYS_BY_DEGRADATION["gamma"],
        extractor=extract_gamma_parameters,
    ),
}


def extract_degradation_parameters(data: dict, degradation: str | None = None) -> dict:
    """
    Extract degradation parameters through the central registry.

    Parameters
    ----------
    data:
        Parsed YAML/JSON input dictionary.

    degradation:
        Optional explicit degradation name. If None, uses data["degradation"].

    Returns
    -------
    dict
        Parsed and validated model parameters.
    """

    if degradation is None:
        if "degradation" not in data:
            raise KeyError(
                "Input file does not contain key 'degradation', and no "
                "degradation argument was provided."
            )
        degradation_lower = str(data["degradation"]).lower()
    else:
        degradation_lower = degradation.lower()

    if degradation_lower not in MODEL_REGISTRY:
        raise ValueError(
            f"Unsupported degradation type '{degradation_lower}'. "
            f"Supported types: {sorted(MODEL_REGISTRY)}"
        )

    validate_required_keys(data, degradation_lower)

    return MODEL_REGISTRY[degradation_lower].extractor(data)