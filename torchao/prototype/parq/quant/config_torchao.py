from dataclasses import dataclass
from typing import Optional

import torch

from torchao.core.config import AOBaseConfig
from torchao.dtypes import Int4CPULayout
from torchao.quantization import MappingType, PerAxis, PerGroup
from torchao.quantization.quant_api import (
    Int4WeightOnlyConfig,
    Int8DynamicActivationIntxWeightConfig,
    IntxWeightOnlyConfig,
)
from torchao.quantization.quantize_.common.packing_format import PackingFormat
from torchao.quantization.quantize_.workflows import IntxUnpackedToInt8Tensor
from torchao.quantization.transform_module import register_quantize_module_handler
from torchao.utils import check_cpu_version

from .quant_api import (
    choose_qparams_stretched_affine,
    quantize_stretched_affine,
    to_stretched_affine_quantized_intx,
)
from .uniform_torchao import (
    _BIT_WIDTH_TO_DTYPE,
    Int4UnifTorchaoQuantizer,
    StretchedUnifTorchaoQuantizer,
)


@dataclass
class StretchedIntxWeightOnlyConfig(IntxWeightOnlyConfig):
    b: Optional[int] = None
    quant_min: Optional[int] = None
    quant_max: Optional[int] = None
    activation_quantization: Optional[str] = "int8_asym_per_token"


@register_quantize_module_handler(StretchedIntxWeightOnlyConfig)
def _stretched_intx_weight_only_transform(
    module: torch.nn.Module, config: StretchedIntxWeightOnlyConfig
) -> torch.nn.Module:
    weight = module.weight
    granularity = config.granularity
    mapping_type = MappingType.ASYMMETRIC

    assert weight.dim() == 2, (
        f"StretchedIntxWeightOnlyConfig only works for 2-d Tensor, got: {weight.dim()}"
    )
    if isinstance(granularity, PerGroup):
        group_size = granularity.group_size
    elif isinstance(granularity, PerAxis):
        assert granularity.axis == 0, (
            f"axis must be 0 with PerAxis, but got {granularity.axis}"
        )
        group_size = weight.shape[-1]
    else:
        raise ValueError(f"granularity must be PerGroup or PerAxis, got {granularity}")

    block_size = (1, group_size)
    target_dtype = torch.int8
    q_args = (weight, mapping_type, block_size, target_dtype, config.b)
    if config.version == 2:
        scale, zero_point = choose_qparams_stretched_affine(
            *q_args,
            quant_min=config.quant_min,
            quant_max=config.quant_max,
        )
        qdata = quantize_stretched_affine(
            weight,
            block_size,
            scale,
            zero_point,
            target_dtype,
            quant_min=config.quant_min,
            quant_max=config.quant_max,
        )
        n_blocks = [qdata.shape[i] // block_size[i] for i in range(len(block_size))]
        scale = scale.reshape(*n_blocks)
        zero_point = zero_point.reshape(*n_blocks)

        weight = IntxUnpackedToInt8Tensor(
            qdata=qdata,
            scale=scale,
            zero_point=zero_point,
            target_dtype=getattr(torch, f"int{config.b}"),
            block_size=block_size,
            dtype=weight.dtype,
            activation_quantization=config.activation_quantization,
        )
    else:
        weight = to_stretched_affine_quantized_intx(
            *q_args,
            quant_min=config.quant_min,
            quant_max=config.quant_max,
            scale_dtype=config.scale_dtype,
            _layout=config.layout,
        )
    module.weight = torch.nn.Parameter(weight, requires_grad=False)
    return module


def get_config_from_quantizer(
    quantizer,
    is_embed: bool,
    device: torch.device,
    b: int,
    block_size: Optional[int],
    version: int = 2,
) -> AOBaseConfig:
    granularity = PerGroup(block_size) if block_size is not None else PerAxis(0)
    weight_dtype = _BIT_WIDTH_TO_DTYPE[b]
    if isinstance(quantizer, Int4UnifTorchaoQuantizer):
        kwargs = {"layout": Int4CPULayout()} if check_cpu_version(device) else {}
        config = Int4WeightOnlyConfig(group_size=block_size, **kwargs)
    elif isinstance(quantizer, StretchedUnifTorchaoQuantizer):
        config = StretchedIntxWeightOnlyConfig(
            b=b,
            quant_min=quantizer.quant_min,
            quant_max=quantizer.quant_max,
            granularity=granularity,
            version=version,
        )
    elif is_embed:
        config = IntxWeightOnlyConfig(
            weight_dtype=weight_dtype,
            granularity=granularity,
            mapping_type=quantizer.mapping_type,
            packing_format=PackingFormat.UNPACKED_TO_INT8,
            version=version,
        )
    else:
        config = Int8DynamicActivationIntxWeightConfig(
            weight_dtype=weight_dtype,
            weight_granularity=granularity,
            weight_mapping_type=quantizer.mapping_type,
            act_mapping_type=MappingType.ASYMMETRIC,
            packing_format=PackingFormat.UNPACKED_TO_INT8,
            version=version,
        )
    return config
