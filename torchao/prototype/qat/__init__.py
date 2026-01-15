# Temporary location for prototype QAT features that will
# eventually live in torchao/quantization/qat

from .mxfp4 import (
    MXFP4FakeQuantizeConfig,
    MXFP4FakeQuantizedLinear,
)
from .nvfp4 import (
    NVFP4FakeQuantizeConfig,
    NVFP4FakeQuantizedLinear,
)

__all__ = [
    "MXFP4FakeQuantizeConfig",
    "MXFP4FakeQuantizedLinear",
    "NVFP4FakeQuantizeConfig",
    "NVFP4FakeQuantizedLinear",
]
