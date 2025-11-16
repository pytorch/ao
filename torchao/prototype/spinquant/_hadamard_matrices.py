"""Utility helpers for lazily loading Hadamard matrices used by spinquant."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Optional

import torch

_DATA_FILENAME = "_hadamard_matrices.pkl"
_HADAMARD_CACHE: Optional[Dict[str, torch.Tensor]] = None


def _load_hadamard_matrices() -> Dict[str, torch.Tensor]:
    """Load and cache the Hadamard matrices from the serialized payload."""

    global _HADAMARD_CACHE

    if _HADAMARD_CACHE is None:
        data_path = Path(__file__).with_name(_DATA_FILENAME)
        if not data_path.exists():
            raise FileNotFoundError(
                f"Expected Hadamard data file '{_DATA_FILENAME}' next to {__file__}."
            )
        with data_path.open("rb") as source:
            raw_matrices = pickle.load(source)
        _HADAMARD_CACHE = {
            name: torch.tensor(values, dtype=torch.float32)
            for name, values in raw_matrices.items()
        }
    return _HADAMARD_CACHE


def _get_hadamard_matrix(name: str) -> torch.Tensor:
    matrices = _load_hadamard_matrices()
    try:
        return matrices[name].clone()  # return a fresh tensor like the previous implementation
    except KeyError as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"Unknown Hadamard matrix '{name}'.") from exc


def get_had12() -> torch.Tensor:
    return _get_hadamard_matrix("had12")


def get_had20() -> torch.Tensor:
    return _get_hadamard_matrix("had20")


def get_had28() -> torch.Tensor:
    return _get_hadamard_matrix("had28")


def get_had36() -> torch.Tensor:
    return _get_hadamard_matrix("had36")


def get_had40() -> torch.Tensor:
    return _get_hadamard_matrix("had40")


def get_had44() -> torch.Tensor:
    return _get_hadamard_matrix("had44")


def get_had52() -> torch.Tensor:
    return _get_hadamard_matrix("had52")


def get_had60() -> torch.Tensor:
    return _get_hadamard_matrix("had60")


def get_had108() -> torch.Tensor:
    return _get_hadamard_matrix("had108")


def get_had140() -> torch.Tensor:
    return _get_hadamard_matrix("had140")


def get_had156() -> torch.Tensor:
    return _get_hadamard_matrix("had156")


def get_had172() -> torch.Tensor:
    return _get_hadamard_matrix("had172")
