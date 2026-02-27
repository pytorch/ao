"""Utility helpers for lazily loading Hadamard matrices used by spinquant."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, Optional

import torch

_DATA_FILENAME = "_hadamard_matrices.pkl"
_JSON_FILENAME = "_hadamard_matrices.json"
_HADAMARD_CACHE: Optional[Dict[str, torch.Tensor]] = None


def _write_pickle(raw_matrices: Dict[str, list], pickle_path: Path) -> None:
    pickle_path.parent.mkdir(parents=True, exist_ok=True)
    with pickle_path.open("wb") as sink:
        pickle.dump(raw_matrices, sink, protocol=pickle.HIGHEST_PROTOCOL)


def _load_raw_matrices() -> Dict[str, list]:
    data_path = Path(__file__).with_name(_DATA_FILENAME)
    json_path = Path(__file__).with_name(_JSON_FILENAME)

    if data_path.exists():
        with data_path.open("rb") as source:
            return pickle.load(source)

    if not json_path.exists():
        raise FileNotFoundError(
            f"Expected Hadamard data file '{_DATA_FILENAME}' or '{_JSON_FILENAME}' "
            f"next to {__file__}."
        )

    with json_path.open("r") as source:
        raw_matrices = json.load(source)

    try:
        _write_pickle(raw_matrices, data_path)
    except OSError:
        # Best-effort cache priming; safe to ignore when running from read-only envs.
        pass

    return raw_matrices


def _load_hadamard_matrices() -> Dict[str, torch.Tensor]:
    """Load and cache the Hadamard matrices from the serialized payload."""

    global _HADAMARD_CACHE

    if _HADAMARD_CACHE is None:
        raw_matrices = _load_raw_matrices()
        _HADAMARD_CACHE = {
            name: torch.tensor(values, dtype=torch.float32)
            for name, values in raw_matrices.items()
        }
    return _HADAMARD_CACHE


def _get_hadamard_matrix(name: str) -> torch.Tensor:
    matrices = _load_hadamard_matrices()
    try:
        return matrices[
            name
        ].clone()  # return a fresh tensor like the previous implementation
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
