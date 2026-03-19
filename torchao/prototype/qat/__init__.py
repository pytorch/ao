# Temporary location for prototype QAT features that will
# eventually live in torchao/quantization/qat

from .mx import (
    MXFakeQuantizeConfig,
    MXFakeQuantizedLinear,
)
from .nvfp4 import (
    NVFP4FakeQuantizeConfig,
    NVFP4FakeQuantizedLinear,
)
from .nvfp4_moe import (
    NVFP4FakeQuantizedMoE,
    NVFP4FakeQuantizedQwen3MoeBlock,
    apply_nvfp4_moe_qat,
    remove_nvfp4_moe_qat,
)

__all__ = [
    "MXFakeQuantizeConfig",
    "MXFakeQuantizedLinear",
    "NVFP4FakeQuantizeConfig",
    "NVFP4FakeQuantizedLinear",
    "NVFP4FakeQuantizedMoE",
    "NVFP4FakeQuantizedQwen3MoeBlock",
    "apply_nvfp4_moe_qat",
    "remove_nvfp4_moe_qat",
]
