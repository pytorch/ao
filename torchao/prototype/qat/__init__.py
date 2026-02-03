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

__all__ = [
    "MXFakeQuantizeConfig",
    "MXFakeQuantizedLinear",
    "NVFP4FakeQuantizeConfig",
    "NVFP4FakeQuantizedLinear",
]
