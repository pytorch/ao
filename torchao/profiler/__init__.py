import inspect

import torch

# Re-exports
from .device_spec import CUDADeviceSpec, DeviceSpec
from .performance_counter import (
    CUDAPerformanceTimer,
    PerformanceCounterMode,
    PerformanceStats,
    PerformanceTimer,
    TransformerPerformanceCounter,
)

__all__ = [
    "CUDAPerformanceTimer",
    "PerformanceCounterMode",
    "PerformanceStats",
    "PerformanceTimer",
    "TransformerPerformanceCounter",
    "CUDADeviceSpec",
    "DeviceSpec",
    "total_model_params",
]

_HUGGINGFACE_CAUSAL_LM_BASE_CLASSES = [
    "causallm",
    "pretrainedmodel",
    "generationmixin",
]


def get_all_base_classes(object):
    return [cls.__name__.lower() for cls in inspect.getmro(object.__class__)]


def total_model_params(
    model: torch.nn.Module,
    exclude_embeddings: bool = True,
    embedding_key: str = "tok_embeddings",
) -> int:
    num_params = sum(p.numel() for p in model.parameters())

    # Exclude embeddings when calculating FLOP since they don't contribute to FLOP count
    if exclude_embeddings:
        # Not the cleanest, but check if any base class of the model is in _HUGGINGFACE_CAUSAL_LM_BASE_CLASSES
        if (
            len(
                set(get_all_base_classes(model)).intersection(
                    _HUGGINGFACE_CAUSAL_LM_BASE_CLASSES
                )
            )
            > 0
        ):
            num_params -= model.model.embed_tokens.weight.numel()
        elif hasattr(model, embedding_key):
            num_params -= getattr(model, embedding_key).weight.numel()
        else:
            raise ValueError(
                f"Could not find embedding in model {model.__class__.__name__}, please specify embedding attribute key"
            )
    return num_params
