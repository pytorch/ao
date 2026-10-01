# Temporary location for prototype QAT features that will
# eventually live in torchao/quantization/qat

from .mx import (
    MXFakeQuantizeConfig,
    MXFakeQuantizedLinear,
    mx_fake_quantize,
    mx_fake_quantized_grouped_mm,
)
from .nvfp4 import (
    NVFP4FakeQuantizeConfig,
    NVFP4FakeQuantizedLinear,
)

__all__ = [
    "MXFakeQuantizeConfig",
    "MXFakeQuantizedLinear",
    "mx_fake_quantize",
    "mx_fake_quantized_grouped_mm",
    "NVFP4FakeQuantizeConfig",
    "NVFP4FakeQuantizedLinear",
]
