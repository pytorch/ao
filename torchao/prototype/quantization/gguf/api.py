import torch
from dataclasses import dataclass
from torchao.core.config import AOBaseConfig
from torchao.quantization.transform_module import register_quantize_module_handler
from .gguf_quantized_tensor import GGUFQuantizedTensor

__all__ = [
    "GGUFWeightOnlyConfig",
]

@dataclass
class GGUFWeightOnlyConfig(AOBaseConfig):
    dtype: torch.dtype = torch.uint4
    n_blocks_per_superblock: int = 8


@register_quantize_module_handler(GGUFWeightOnlyConfig)
def _gguf_weight_only_transform(
    module: torch.nn.Module,
    config: GGUFWeightOnlyConfig,
):
    """
    Applies gguf weight-only quantization to linear layers.

    Args:
        dtype: torch.uint1 to torch.uint8, torch.int32 supported.
        n_blocks_per_superblock: the number of super blocks in a 256 element block for gguf, e.g. when it is 8
            it means we have blocks of 32 and 8 blocks in a superblock of 256 elements.
    Returns:
        Callable for quantization transformation.
    """
    weight = module.weight
    if (weight.ndim != 2) or (weight.shape[-1] % 256 != 0):
        return module

    quantized_weight = GGUFQuantizedTensor.from_float(
        weight,
        n_blocks_per_superblock=config.n_blocks_per_superblock,
        target_dtype=config.dtype,
    )
    module.weight = torch.nn.Parameter(quantized_weight, requires_grad=False)
    return module
